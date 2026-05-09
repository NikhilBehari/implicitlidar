"""Evaluate the motion-adaptive scanning configuration on warehouse test scenes.

Loads the per-position rays produced by :mod:`.synthesize`, plus matched
even/random baselines (per-position scanning), and tests them against held-out
front-plane test scenes. Per-scene hit rate = boxes hit /
boxes total, aggregated across scenes; miss rate = 1 − hit rate.

Usage::

    python -m implicitlidar.experiments.warehouse_detection.evaluate \\
        --config implicitlidar/experiments/warehouse_detection/configs/default.yaml
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from implicitlidar.core import (
    AZIMUTH_DIM,
    ELEVATION_DIM,
    TIME_DIM,
    Sensor,
    rows_to_sensors,
)
from implicitlidar.eval import baseline_bandwidth_mbps, mean_pm_2sem, sensor_bandwidth_mbps
from implicitlidar.scenes.warehouse import (
    DEFAULT_LAYOUT,
    FrontPlaneScene,
    generate_front_plane_test_scenes,
    ray_hits_front_plane,
)
from implicitlidar.utils import ensure_dir, load_config, parse_overrides, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    config = load_config(args.config, parse_overrides(args.override))
    device = select_device()
    print(f"[evaluate] device = {device}")

    run_dir = Path(config["output"]["run_dir"])
    sensor_dir = run_dir / "sensors"
    if not sensor_dir.exists():
        raise FileNotFoundError(f"No sensors found at {sensor_dir}; run synthesize first.")

    eval_cfg = config["evaluation"]
    n_test_scenes = int(eval_cfg.get("n_test_scenes", 40))
    test_seed = int(eval_cfg.get("test_seed", 84))  # = 2 × train_seed.
    test_scenes = generate_front_plane_test_scenes(
        n_test_scenes, seed=test_seed, layout=DEFAULT_LAYOUT,
        shelf_spacing_x=float(eval_cfg.get("shelf_spacing_x", 0.2)),
    )
    n_total_boxes = sum(len(s.planes) for s in test_scenes)
    print(f"[evaluate] {len(test_scenes)} held-out test scenes (seed={test_seed}); "
          f"{n_total_boxes} total target boxes")

    sensor_files = sorted(
        (p for p in sensor_dir.glob("sensors_motion_adaptive_r*.csv")),
        key=lambda p: int(re.search(r"_r(\d+)", p.stem).group(1)),
    )
    if not sensor_files:
        raise FileNotFoundError(f"No motion-adaptive sensor CSVs in {sensor_dir}")

    # Baseline ray spec: full elevation hemisphere, fixed azimuth (downward
    # toward the shelf row), full ToF range.
    az_baseline = float(eval_cfg.get("baseline_azimuth", -1.57))
    el_lo = float(eval_cfg.get("baseline_elevation_min", 0.0))
    el_hi = float(eval_cfg.get("baseline_elevation_max", np.pi / 2))
    t_max_baseline = float(eval_cfg.get("baseline_t_max", 2.0))

    rng = np.random.default_rng(test_seed)
    rows: list[dict] = []
    for sensor_csv in tqdm(sensor_files, desc="ray budgets"):
        rays_per_position = int(re.search(r"_r(\d+)", sensor_csv.stem).group(1))
        per_position = _load_per_position_sensors(sensor_csv)
        x_queries = [x for x, _ in per_position]
        n_total_rays = sum(len(s) for _, s in per_position)

        rays_ours = _materialize_motion_adaptive_rays(per_position)
        rows.extend(_evaluate_method(
            method=f"ours_r{rays_per_position}",
            rays=rays_ours, sensors=sum((s for _, s in per_position), []),
            test_scenes=test_scenes, ray_budget=n_total_rays,
            n_query_positions=len(x_queries),
        ))

        rays_even = _baseline_rays(
            x_queries, rays_per_position, sampling="even",
            az=az_baseline, el_min=el_lo, el_max=el_hi, t_max=t_max_baseline,
        )
        rows.extend(_evaluate_method(
            method=f"even_r{rays_per_position}",
            rays=rays_even, sensors=None,
            test_scenes=test_scenes, ray_budget=len(rays_even),
            n_query_positions=len(x_queries),
        ))

        rays_random = _baseline_rays(
            x_queries, rays_per_position, sampling="random",
            az=az_baseline, el_min=el_lo, el_max=el_hi, t_max=t_max_baseline,
            rng=rng,
        )
        rows.extend(_evaluate_method(
            method=f"random_r{rays_per_position}",
            rays=rays_random, sensors=None,
            test_scenes=test_scenes, ray_budget=len(rays_random),
            n_query_positions=len(x_queries),
        ))

    df = pd.DataFrame(rows)
    out_csv = ensure_dir(run_dir / "results") / "miss_rate.csv"
    df.to_csv(out_csv, index=False)
    print(f"[evaluate] results saved to {out_csv}")
    if not df.empty:
        print(df.to_string(index=False))


# Ray construction


def _load_per_position_sensors(csv_path: Path) -> list[tuple[float, list[Sensor]]]:
    """Parse a motion-adaptive sensor CSV into ``[(x_query, [Sensor, ...]), ...]``."""
    df = pd.read_csv(csv_path)
    out: list[tuple[float, list[Sensor]]] = []
    for x_query, group in df.groupby("x_query", sort=True):
        rows = group.drop(columns=["x_query"]).values.tolist()
        out.append((float(x_query), rows_to_sensors(rows)))
    return out


def _materialize_motion_adaptive_rays(
    per_position: list[tuple[float, list[Sensor]]],
) -> np.ndarray:
    """Stack per-position sensors into ``(N, 7)`` rays
    ``(origin_x, origin_y, origin_z, az, el, t_min, t_max)``.

    Each sensor's ``(az_mean, el_mean, t_mean)`` becomes one ray, with time
    gate ``[t_mean − 2σ_t, t_mean + 2σ_t]``.
    """
    rays = []
    for x_query, sensors in per_position:
        for s in sensors:
            t_mean, t_sigma = s.mean[TIME_DIM], s.sigma[TIME_DIM]
            rays.append([
                float(x_query), 0.0, 0.0,
                float(s.mean[AZIMUTH_DIM]),
                float(s.mean[ELEVATION_DIM]),
                float(t_mean - 2.0 * t_sigma),
                float(t_mean + 2.0 * t_sigma),
            ])
    return np.asarray(rays, dtype=float)


def _baseline_rays(
    x_queries: list[float], rays_per_position: int, *,
    sampling: str, az: float, el_min: float, el_max: float, t_max: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Even or random elevations from each ``x_query``, fixed azimuth, full ToF."""
    rays = []
    for x in x_queries:
        if sampling == "even":
            els = np.linspace(el_min, el_max, rays_per_position)
        elif sampling == "random":
            assert rng is not None
            els = rng.uniform(el_min, el_max, rays_per_position)
        else:
            raise ValueError(f"sampling must be 'even' or 'random', got {sampling!r}")
        for el in els:
            rays.append([float(x), 0.0, 0.0, az, float(el), 0.0, float(t_max)])
    return np.asarray(rays, dtype=float)


# Per-method evaluation


def _evaluate_method(
    *, method: str, rays: np.ndarray, sensors,
    test_scenes: list[FrontPlaneScene], ray_budget: int,
    n_query_positions: int,
) -> list[dict]:
    if len(rays) == 0:
        return []
    origins = rays[:, :3]
    azs = rays[:, 3]
    els = rays[:, 4]
    t_mins = rays[:, 5]
    t_maxs = rays[:, 6]
    dirs = np.stack([
        np.cos(els) * np.cos(azs),
        np.cos(els) * np.sin(azs),
        np.sin(els),
    ], axis=1)

    per_scene_hit_rates = []
    for scene in test_scenes:
        n_boxes = len(scene.planes)
        hit_flags = np.zeros(n_boxes, dtype=bool)
        for box_i, plane in enumerate(scene.planes):
            for r in range(len(rays)):
                if ray_hits_front_plane(origins[r], dirs[r], t_mins[r], t_maxs[r], plane):
                    hit_flags[box_i] = True
                    break
        per_scene_hit_rates.append(hit_flags.sum() / max(n_boxes, 1) * 100.0)

    miss_rates = 100.0 - np.asarray(per_scene_hit_rates)
    mean, two_sem = mean_pm_2sem(miss_rates)
    bw = _bandwidth_for_method(rays, sensors, n_query_positions=n_query_positions)
    return [{
        "method": method,
        "ray_budget": int(ray_budget),
        "n_test_scenes": len(per_scene_hit_rates),
        "miss_rate_mean_pct": round(float(mean), 3),
        "miss_rate_2sem_pct": round(float(two_sem), 3),
        "bandwidth_mbps": round(bw, 4),
    }]


def _bandwidth_for_method(rays: np.ndarray, sensors, *, n_query_positions: int) -> float:
    """Mbps reported per query position (the robot occupies one x at a time).

    Design-space ``τ`` is in scene units; the warehouse converts to physical
    meters via ``scaling_factor_m=6.0`` for the SPAD-bandwidth derivation.
    """
    if sensors is None:
        gate = float((rays[:, 6] - rays[:, 5]).mean())
        return baseline_bandwidth_mbps(
            len(rays) // n_query_positions, gate, scaling_factor_m=6.0,
        )
    # Warehouse synthesizes per-sensor time gates as [t_mean ± 2σ], so the
    # bandwidth helper uses gate_z=2.0 instead of the 95 % CI default.
    return sensor_bandwidth_mbps(
        sensors, len(rays) // n_query_positions,
        gate_z=2.0, scaling_factor_m=6.0,
    )


if __name__ == "__main__":
    main()
