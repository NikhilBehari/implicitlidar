"""Evaluate synthesized distributed sensors and baselines for robot tracking.

For each held-out test trajectory, ray-traces against the end-effector
spheres along that trajectory, reconstructs the trajectory by interpolating
through the hits, and reports the (2D) shapely Fréchet distance to the
ground-truth pose sequence.

Rays for the synthesized sensors are deterministic in
azimuth/elevation (set to per-sensor mean) with origins evenly spaced inside
the per-sensor 95 % confidence ellipse; baselines are vertical downward rays
distributed over the top face of the trajectory bounding hull.

Usage::

    python -m implicitlidar.experiments.robot_tracking.evaluate \\
        --config implicitlidar/experiments/robot_tracking/configs/default.yaml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from scipy.stats import norm
from shapely.geometry import LineString
from tqdm import tqdm

from implicitlidar.core import (
    AZIMUTH_DIM,
    ELEVATION_DIM,
    TIME_DIM,
    SceneSDF,
    Sensor,
    rows_to_sensors,
)
from implicitlidar.eval import (
    baseline_bandwidth_mbps,
    hit_points,
    mean_pm_2sem,
    sensor_bandwidth_mbps,
    trace_rays_against_scene,
)
from implicitlidar.scenes.robot_arm import KukaTrajectoryGenerator, end_effector_scene
from implicitlidar.utils import ensure_dir, load_config, parse_overrides, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--components", type=str, default=None)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    config = load_config(args.config, parse_overrides(args.override))
    device = select_device()
    print(f"[evaluate] device = {device}")

    run_dir = Path(config["output"]["run_dir"])
    sensor_dir = run_dir / "sensors"
    if not sensor_dir.exists():
        raise FileNotFoundError(f"No sensors found at {sensor_dir}; run synthesize first.")

    components = (
        [int(c) for c in args.components.split(",")]
        if args.components else
        sorted(int(p.stem.split("_")[-1]) for p in sensor_dir.glob("sensors_*.csv"))
    )

    eval_cfg = config["evaluation"]
    print(f"[evaluate] sampling {eval_cfg['num_test_trajectories']} test trajectories")
    test_trajectories = _sample_test_trajectories(config, device)

    # Trajectory bounding hull for baseline ray origins.
    all_positions = np.concatenate([pos for pos, _ in test_trajectories], axis=0)
    baseline_hull = _compute_square_shell(all_positions[None])
    rng = np.random.default_rng(0)

    use_spline_fit = bool(config["target"].get("use_visibility", False))
    rows: list[dict] = []
    for n in components:
        sensors = rows_to_sensors(pd.read_csv(sensor_dir / f"sensors_{n}.csv").values.tolist())
        for budget in tqdm(eval_cfg["ray_budgets"], desc=f"{n} sensors"):
            rays_ours = _sample_sensor_rays(sensors, n_rays=budget)
            rows.extend(_eval_one(
                method=f"ours_{n}", rays=rays_ours,
                test_trajectories=test_trajectories, sensors=sensors,
                budget=budget, device=device, use_spline_fit=use_spline_fit,
            ))
            rays_even = _emit_baseline_rays(baseline_hull, budget, sampling="even")
            rows.extend(_eval_one(
                method="uniform", rays=rays_even,
                test_trajectories=test_trajectories, sensors=None,
                budget=budget, device=device, use_spline_fit=use_spline_fit,
            ))
            rays_random = _emit_baseline_rays(baseline_hull, budget, sampling="random", rng=rng)
            rows.extend(_eval_one(
                method="random", rays=rays_random,
                test_trajectories=test_trajectories, sensors=None,
                budget=budget, device=device, use_spline_fit=use_spline_fit,
            ))

    df = pd.DataFrame(rows)
    out_csv = ensure_dir(run_dir / "results") / "frechet.csv"
    df.to_csv(out_csv, index=False)
    print(f"[evaluate] results saved to {out_csv}")
    if not df.empty and "frechet_mean_m" in df.columns:
        print(df.pivot_table(values="frechet_mean_m", index="ray_budget", columns="method").to_string())


# Test trajectories + bounding hull


def _sample_test_trajectories(config: dict, device) -> list[tuple[np.ndarray, SceneSDF]]:
    """Sample held-out test trajectories and build their per-trajectory scenes."""
    eval_cfg = config["evaluation"]
    scene_cfg = config["scene"]
    tg = KukaTrajectoryGenerator()
    out: list[tuple[np.ndarray, SceneSDF]] = []
    for i in range(int(eval_cfg["num_test_trajectories"])):
        joint_traj, ee_traj = tg.sample(
            seed=int(eval_cfg["test_seed"]) + i,
            place_noise_xy=tuple(scene_cfg.get("trajectory_variance", (0.0, 0.0))),
            steps_per_segment=int(scene_cfg.get("steps_per_segment", 10)),
        )
        ee_positions = np.stack([pos for pos, _ in ee_traj], axis=0)
        scene = end_effector_scene(ee_traj, sphere_radius=float(scene_cfg.get("sphere_radius", 0.02)),
                                   device=device)
        out.append((ee_positions, scene))
    return out


def _compute_square_shell(traj_array: np.ndarray, *,
                          xy_margin: float = 0.1,
                          z_limits: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Rectangular prism around an array of trajectories, with margin in xy."""
    x_vals = traj_array[..., 0]
    y_vals = traj_array[..., 1]
    return np.array([
        [x_vals.min() - xy_margin, y_vals.min() - xy_margin, z_limits[0]],
        [x_vals.max() + xy_margin, y_vals.max() + xy_margin, z_limits[1]],
    ])


# Ray sampling


def _z_at_confidence(confidence: float) -> float:
    return float(norm.ppf((1 + confidence) / 2))


def _ellipse_interior_points(
    sensor: Sensor, *, confidence: float, n_pts: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Even-spacing inside the per-sensor 95 % origin ellipse, baseline."""
    rng = rng if rng is not None else np.random.default_rng(0)
    z = _z_at_confidence(confidence)
    cov = np.array([
        [sensor.origin_cov[0, 0], sensor.origin_cov[0, 1]],
        [sensor.origin_cov[0, 1], sensor.origin_cov[1, 1]],
    ])
    vals, vecs = np.linalg.eigh(cov)
    a, b = np.sqrt(np.clip(vals, 1e-12, None)) * z
    transform = vecs @ np.diag([a, b])
    center = np.array([sensor.origin_mean[0], sensor.origin_mean[1]])

    area = math.pi * a * b
    cell_area = max(area / max(n_pts, 1), 1e-12)
    d = math.sqrt(cell_area)

    xs = np.arange(-a, a + d, d)
    ys = np.arange(-b, b + d, d)
    xv, yv = np.meshgrid(xs, ys)
    uv = np.stack([xv.ravel(), yv.ravel()], axis=1)
    inside = uv[(uv[:, 0] / a) ** 2 + (uv[:, 1] / b) ** 2 <= 1.0]
    if inside.shape[0] < n_pts:
        n_extra = n_pts - inside.shape[0]
        extras = []
        while len(extras) < n_extra:
            pts = rng.uniform([-a, -b], [a, b], size=(n_extra * 2, 2))
            mask = (pts[:, 0] / a) ** 2 + (pts[:, 1] / b) ** 2 <= 1.0
            extras.extend(pts[mask].tolist())
        inside = np.vstack([inside, extras[:n_extra]])
    selected = inside[:n_pts]
    pts_xy = (transform @ (selected / np.array([a, b])).T).T + center
    return np.hstack([pts_xy, np.full((n_pts, 1), sensor.origin_mean[2])])


def _sample_sensor_rays(
    sensors: list[Sensor], n_rays: int, *,
    origin_conf: float = 0.95, time_conf: float = 0.95,
) -> np.ndarray:
    """Allocate ``n_rays`` across sensors by mixture weight; rays follow the
    deterministic-direction, ellipse-interior-origin convention."""
    weights = np.array([s.weight for s in sensors], dtype=float)
    counts = np.round(weights / weights.sum() * n_rays).astype(int)
    diff = n_rays - counts.sum()
    if diff:
        counts[int(np.argmax(weights))] += diff

    zt = _z_at_confidence(time_conf)
    out = np.zeros((n_rays, 8), dtype=float)  # x, y, z, az, el, t_min, t_max, sid
    cursor = 0
    for sid, (sensor, cnt) in enumerate(zip(sensors, counts)):
        if cnt <= 0:
            continue
        origins = _ellipse_interior_points(sensor, confidence=origin_conf, n_pts=cnt)
        out[cursor:cursor + cnt, 0:3] = origins
        out[cursor:cursor + cnt, 3] = sensor.mean[AZIMUTH_DIM]
        out[cursor:cursor + cnt, 4] = sensor.mean[ELEVATION_DIM]
        out[cursor:cursor + cnt, 5] = sensor.mean[TIME_DIM] - zt * sensor.sigma[TIME_DIM]
        out[cursor:cursor + cnt, 6] = sensor.mean[TIME_DIM] + zt * sensor.sigma[TIME_DIM]
        out[cursor:cursor + cnt, 7] = sid
        cursor += cnt
    return out


def _emit_baseline_rays(
    hull: np.ndarray, n_rays: int, *,
    sampling: str = "even", rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Vertical downward rays originating on the top of the trajectory hull."""
    (xmin, ymin, _), (xmax, ymax, zmax) = hull
    if sampling == "even":
        n_x = math.ceil(math.sqrt(n_rays))
        n_y = math.ceil(n_rays / n_x)
        xs = np.linspace(xmin, xmax, n_x)
        ys = np.linspace(ymin, ymax, n_y)
        grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)
        origins = grid[:n_rays]
        azs = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    elif sampling == "random":
        rng = rng if rng is not None else np.random.default_rng()
        origins = np.column_stack([
            rng.uniform(xmin, xmax, n_rays),
            rng.uniform(ymin, ymax, n_rays),
        ])
        azs = rng.uniform(0, 2 * np.pi, n_rays)
    else:
        raise ValueError(f"sampling must be 'even' or 'random', got {sampling!r}")

    out = np.zeros((n_rays, 8), dtype=float)
    out[:, 0] = origins[:, 0]
    out[:, 1] = origins[:, 1]
    out[:, 2] = zmax
    out[:, 3] = azs
    out[:, 4] = -np.pi / 2
    out[:, 5] = 0.0
    out[:, 6] = 1.0
    out[:, 7] = 0  # single virtual "sensor"
    return out


# Per-method evaluation


def _eval_one(
    *, method: str, rays: np.ndarray,
    test_trajectories: list[tuple[np.ndarray, SceneSDF]],
    sensors: list[Sensor] | None, budget: int, device,
    use_spline_fit: bool,
) -> list[dict]:
    """Hit-test each test trajectory's spheres, reconstruct, and Fréchet-score."""
    design = np.zeros((len(rays), 6), dtype=float)
    design[:, 0:3] = rays[:, 0:3]
    design[:, AZIMUTH_DIM] = rays[:, 3]
    design[:, ELEVATION_DIM] = rays[:, 4]
    design[:, TIME_DIM] = rays[:, 6]
    sids = rays[:, 7].astype(int)
    time_gates = rays[:, 5:7]  # (N, 2) -> [t_min, t_max]

    frechets = []
    for ee_positions, scene in test_trajectories:
        all_hits_per_scene = trace_rays_against_scene(
            design, scene, sensor_ids=sids, time_gates=time_gates, device=device,
        )
        hits_xyz = (np.concatenate([hit_points(h) for h in all_hits_per_scene], axis=0)
                    if all_hits_per_scene else np.empty((0, 3)))
        if len(hits_xyz) < 2:
            continue
        est = _compute_trajectory_line(
            hits_xyz, gt_points=ee_positions if use_spline_fit else None, num_samples=200,
        )
        frechets.append(_shapely_frechet(est, ee_positions))
    if not frechets:
        return []
    mean, two_sem = mean_pm_2sem(np.array(frechets))
    bw = _bandwidth_for_method(rays, sensors)
    return [{
        "method": method,
        "ray_budget": int(budget),
        "n_trajectories": len(frechets),
        "frechet_mean_m": round(mean, 4),
        "frechet_2sem_m": round(two_sem, 4),
        "bandwidth_mbps": round(bw, 4),
    }]


def _shapely_frechet(a: np.ndarray, b: np.ndarray) -> float:
    """2D Fréchet distance over the (x, y) projection."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(shapely.frechet_distance(LineString(a), LineString(b)))


def _compute_trajectory_line(
    points: np.ndarray, *, gt_points: np.ndarray | None, num_samples: int = 200,
    smoothing: float = 0.05,
) -> np.ndarray:
    """Reconstruct a polyline through ``points``.

    With ``gt_points=None`` (no-occlusion case): assume ``points`` is already
    in trajectory order, drop consecutive duplicates, and resample by linear
    interpolation along the polyline.

    With ``gt_points`` provided (occlusion case): fit a smoothed cubic spline
    through the hits, reparameterized by the arc length of ``gt_points``.
    """
    arr = np.asarray(points, float)
    if arr.shape[0] < 2:
        raise ValueError("Need at least 2 points")

    if gt_points is None:
        diffs = np.any(np.diff(arr, axis=0) != 0, axis=1)
        pts = arr[np.concatenate(([True], diffs))]
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cumlen = np.concatenate(([0.0], np.cumsum(seg)))
        if cumlen[-1] == 0:
            return np.tile(pts[0], (num_samples, 1))
        u = cumlen / cumlen[-1]
        u_fine = np.linspace(0.0, 1.0, num_samples)
        return np.column_stack([np.interp(u_fine, u, pts[:, k]) for k in range(3)])

    from scipy.interpolate import splev, splprep

    gt = np.asarray(gt_points, float)
    d_gt = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    u_gt = np.concatenate(([0.0], np.cumsum(d_gt)))
    u_gt = u_gt / u_gt[-1] if u_gt[-1] > 0 else np.linspace(0, 1, gt.shape[0])

    tree = cKDTree(gt)
    _, idx = tree.query(arr)
    order = np.argsort(idx)
    ordered = arr[order]
    u_pts = u_gt[idx[order]]
    if np.any(np.diff(u_pts) <= 0):
        u_pts = np.linspace(0.0, 1.0, ordered.shape[0])

    k = min(3, ordered.shape[0] - 1)
    tck, _ = splprep(ordered.T, u=u_pts, s=smoothing, k=k)
    u_fine = np.linspace(0.0, 1.0, num_samples)
    return np.vstack(splev(u_fine, tck)).T


def _bandwidth_for_method(rays: np.ndarray, sensors) -> float:
    if sensors is None:
        # Each baseline ray carries its own [t_min, t_max] gate (rays[:, 5:7]);
        # the histogram covers the union of those gates.
        gate_width = float(rays[:, 6].max() - rays[:, 5].min())
        return baseline_bandwidth_mbps(len(rays), gate_width)
    return sensor_bandwidth_mbps(sensors, len(rays))


if __name__ == "__main__":
    main()
