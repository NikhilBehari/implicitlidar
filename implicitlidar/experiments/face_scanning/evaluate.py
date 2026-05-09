"""Evaluate synthesized sensors and baselines on the held-out face set.

Loads the sensor CSVs produced by :mod:`.synthesize`, draws rays from each
synthesized configuration plus from the uniform and random baselines at
matched ray budgets, traces them against every test face, reconstructs each
face by Delaunay triangulation of the hits, and reports Chamfer distance.

Outputs
-------

* ``<run_dir>/results/chamfer.csv`` — one row per ``(method, ray_budget)``
  with mean Chamfer distance (mm), 2·SEM, and required photon-stream
  bandwidth (Mbps).

Usage::

    python -m implicitlidar.experiments.face_scanning.evaluate \\
        --config implicitlidar/experiments/face_scanning/configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm

from implicitlidar.core import (
    AZIMUTH_DIM,
    ELEVATION_DIM,
    TIME_DIM,
    SceneSDF,
    rows_to_sensors,
    sample_rays,
)
from implicitlidar.eval import (
    baseline_bandwidth_mbps,
    hit_points,
    mean_pm_2sem,
    mesh_chamfer_squared,
    mesh_from_hits,
    random_rays,
    sensor_bandwidth_mbps,
    trace_rays_against_scene,
    uniform_rays,
)
from implicitlidar.scenes.faces import load_face_scene
from implicitlidar.utils import ensure_dir, load_config, parse_overrides, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--components", type=str, default=None,
                        help="Comma-separated component counts to evaluate. "
                             "Defaults to all sensor CSVs found in <run_dir>/sensors.")
    parser.add_argument("--ground-truth-samples", type=int, default=4000,
                        help="Number of surface samples per face for the Chamfer reference cloud.")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    config = load_config(args.config, parse_overrides(args.override))
    device = select_device()
    print(f"[evaluate] device = {device}")

    run_dir = Path(config["output"]["run_dir"])
    sensor_dir = run_dir / "sensors"
    if not sensor_dir.exists():
        raise FileNotFoundError(f"No sensors found at {sensor_dir}; run synthesize first.")

    test_scene = load_face_scene(config["evaluation"]["test_scene"]).to(device)
    test_paths = sorted(Path(config["evaluation"]["test_scene"]).glob("*.obj"))
    print(f"[evaluate] {len(test_paths)} test face mesh(es)")
    gt_meshes = [trimesh.load(str(p), force="mesh") for p in test_paths]

    # Load each sensor CSV.
    components = (
        [int(c) for c in args.components.split(",")]
        if args.components else
        sorted(int(p.stem.split("_")[-1]) for p in sensor_dir.glob("sensors_*.csv"))
    )
    if not components:
        raise FileNotFoundError(f"No sensor CSVs in {sensor_dir}")

    results_dir = ensure_dir(run_dir / "results")
    az_lo, az_hi = config["design_space"]["low"][AZIMUTH_DIM], config["design_space"]["high"][AZIMUTH_DIM]
    el_lo, el_hi = config["design_space"]["low"][ELEVATION_DIM], config["design_space"]["high"][ELEVATION_DIM]
    t_lo, t_hi = config["design_space"]["low"][TIME_DIM], config["design_space"]["high"][TIME_DIM]
    baseline_origin = tuple(float(v) for v in config["design_space"]["low"][:3])
    rng = np.random.default_rng(0)

    rows: list[dict] = []
    for n in components:
        sensors = _load_sensors(sensor_dir / f"sensors_{n}.csv")
        for budget in tqdm(config["evaluation"]["ray_budgets"], desc=f"{n} sensors"):
            rays, sids, gates = sample_rays(sensors, total_rays=budget, rng=rng)
            rows.extend(_evaluate_method(
                method=f"ours_{n}",
                rays=rays, sids=sids, time_gates=gates,
                scene=test_scene, gt_meshes=gt_meshes,
                budget=budget, sensors=sensors, device=device,
            ))
            uniform = uniform_rays(
                n_rays=budget, origin=baseline_origin,
                azimuth_range=(az_lo, az_hi),
                elevation_range=(el_lo, el_hi),
                t_min=t_lo, t_max=t_hi,
            )
            rows.extend(_evaluate_method(
                method="uniform",
                rays=uniform, sids=np.zeros(len(uniform), dtype=int),
                time_gates=np.tile([t_lo, t_hi], (len(uniform), 1)),
                scene=test_scene, gt_meshes=gt_meshes,
                budget=budget, sensors=None, device=device,
            ))
            rand = random_rays(
                n_rays=budget, origin=baseline_origin,
                azimuth_range=(az_lo, az_hi),
                elevation_range=(el_lo, el_hi),
                t_min=t_lo, t_max=t_hi,
                seed=int(rng.integers(0, 1 << 31)),
            )
            rows.extend(_evaluate_method(
                method="random",
                rays=rand, sids=np.zeros(len(rand), dtype=int),
                time_gates=np.tile([t_lo, t_hi], (len(rand), 1)),
                scene=test_scene, gt_meshes=gt_meshes,
                budget=budget, sensors=None, device=device,
            ))

    df = pd.DataFrame(rows)
    out_csv = results_dir / "chamfer.csv"
    df.to_csv(out_csv, index=False)
    print(f"[evaluate] results saved to {out_csv}")
    if not df.empty and "chamfer_mean_mm" in df.columns:
        print(df.pivot_table(values="chamfer_mean_mm", index="ray_budget", columns="method").to_string())


# Helpers


def _load_sensors(path: Path):
    df = pd.read_csv(path)
    rows = df.values.tolist()
    return rows_to_sensors(rows)


def _evaluate_method(
    *,
    method: str,
    rays: np.ndarray,
    sids: np.ndarray,
    time_gates: np.ndarray,
    scene: SceneSDF,
    gt_meshes: list[trimesh.Trimesh],
    budget: int,
    sensors,
    device,
) -> list[dict]:
    """Trace rays, Delaunay-reconstruct each face, mesh-vs-mesh Chamfer (mm)."""
    hits = trace_rays_against_scene(
        rays, scene, sensor_ids=sids, time_gates=time_gates, device=device,
    )
    chamfers_mm = []
    for rec, gt_mesh in zip(hits, gt_meshes):
        pts = hit_points(rec)
        if len(pts) < 4:
            continue
        rec_mesh = mesh_from_hits(pts)
        if rec_mesh is None:
            continue
        # Convert m^2 mean squared distance to RMS millimeters.
        chamfers_mm.append(1000.0 * float(np.sqrt(mesh_chamfer_squared(rec_mesh, gt_mesh))))
    if not chamfers_mm:
        return []
    mean, two_sem = mean_pm_2sem(np.array(chamfers_mm))
    bw = _bandwidth_for_method(rays, time_gates, sensors)
    return [{
        "method": method,
        "ray_budget": int(budget),
        "n_scenes_hit": len(chamfers_mm),
        "chamfer_mean_mm": round(mean, 3),
        "chamfer_2sem_mm": round(two_sem, 3),
        "bandwidth_mbps": round(bw, 4),
    }]


def _bandwidth_for_method(rays: np.ndarray, time_gates: np.ndarray, sensors) -> float:
    if sensors is None:
        # Baselines emit one big histogram covering the full τ range per ray.
        gate_width = float(time_gates[:, 1].max() - time_gates[:, 0].min())
        return baseline_bandwidth_mbps(len(rays), gate_width)
    return sensor_bandwidth_mbps(sensors, len(rays))


if __name__ == "__main__":
    main()
