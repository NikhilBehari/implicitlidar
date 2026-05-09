"""Reconstruct a face surface from a real point cloud and synthesized sensor rays.

Pipeline:

1. **Load the point cloud** — output of :mod:`.process_measurement`.
2. **Materialize the rays** — each row of a synthesized sensor CSV is one
   sensor; rays are drawn from each via
   :func:`~implicitlidar.core.sample_rays`, or from the uniform / random
   baselines for comparison.
3. **Match rays to cloud** — for each ray segment ``(origin, origin + τ·d̂)``
   find the cloud point with the smallest perpendicular distance to the
   segment, accepting it as a "hit" when within ``--tol`` meters.
4. **Mesh** the hits via 2-D Delaunay triangulation in the dominant plane.

The output is an OBJ mesh per (sensor design × ray budget) combination.

Usage::

    python -m implicitlidar.experiments.real_world.reconstruct \\
        --point-cloud outputs/data/real_world/point_cloud.npy \\
        --sensors outputs/runs/face_scanning/default/sensors/sensors_2.csv \\
        --rays 576 \\
        --out outputs/runs/real_world/sensors_2_576.obj
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from implicitlidar.core import rows_to_sensors, sample_rays
from implicitlidar.eval import random_rays, uniform_rays


def match_rays_to_cloud(
    rays: np.ndarray,
    cloud: np.ndarray,
    time_gates: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    """For each ray, return the cloud point closest to the ray's gated segment, or skip.

    Each input ray is a 6-D design point ``(x, y, z, az, el, τ)`` with a
    matching gate ``[t_min, t_max]``; the segment runs from
    ``origin + t_min·d̂`` to ``origin + t_max·d̂``. A cloud point is accepted
    as a hit if its perpendicular distance to the segment is at most
    ``tolerance`` *and* its projection onto the segment is within the
    segment span.

    Returns an ``(K, 3)`` array of accepted cloud points (one per ray that
    matched; missing rays are dropped).
    """
    hits: list[np.ndarray] = []
    for ray, (t_lo, t_hi) in zip(rays, time_gates):
        origin = ray[:3]
        az, el = ray[3], ray[4]
        direction = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ])
        seg_start = origin + direction * t_lo
        seg_end = origin + direction * t_hi
        seg_vec = seg_end - seg_start
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len <= 0:
            continue
        unit = seg_vec / seg_len
        diffs = cloud - seg_start
        proj = diffs @ unit
        in_segment = (proj >= 0) & (proj <= seg_len)
        if not in_segment.any():
            continue
        diffs_in = diffs[in_segment]
        proj_in = proj[in_segment]
        perpendicular = np.linalg.norm(
            diffs_in - np.outer(proj_in, unit), axis=1
        )
        idx = int(np.argmin(perpendicular))
        if perpendicular[idx] <= tolerance:
            hits.append(cloud[in_segment][idx])
    return np.array(hits) if hits else np.zeros((0, 3))


def reconstruct_mesh_obj(
    hits: np.ndarray,
    out_path: Path,
    *,
    plane_axes: tuple[int, int] = (1, 2),
) -> Path:
    """Write a triangulated OBJ mesh from ``hits`` via Delaunay reconstruction."""
    if len(hits) < 4:
        raise ValueError(f"Need at least 4 hits to triangulate; got {len(hits)}")
    coords_2d = hits[:, list(plane_axes)]
    tri = Delaunay(coords_2d)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for x, y, z in hits:
            f.write(f"v {x} {y} {z}\n")
        for i, j, k in tri.simplices:
            f.write(f"f {i + 1} {j + 1} {k + 1}\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-cloud", type=Path, required=True,
                        help="3D point cloud produced by process_measurement.py.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sensors", type=Path, default=None,
                     help="Path to a synthesized sensor CSV (e.g. sensors_4.csv).")
    src.add_argument("--baseline", choices=["uniform", "random"], default=None,
                     help="Use a baseline scanning pattern instead of synthesized sensors.")
    parser.add_argument("--rays", type=int, required=True,
                        help="Total number of rays to materialize.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output OBJ mesh path.")
    parser.add_argument("--tol", type=float, default=0.025,
                        help="Maximum perpendicular distance (meters) for a ray-to-point match.")
    parser.add_argument("--baseline-origin", type=float, nargs=3, default=(3.0, 0.0, 0.0),
                        help="Origin used for baselines (default matches the smartphone-LiDAR setup).")
    parser.add_argument("--baseline-azimuth-range", type=float, nargs=2, default=(2.356, 3.927))
    parser.add_argument("--baseline-elevation-range", type=float, nargs=2, default=(-0.785, 0.785))
    parser.add_argument("--baseline-t-range", type=float, nargs=2, default=(0.1, 5.0))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cloud = np.load(args.point_cloud)
    print(f"[reconstruct] loaded {len(cloud)} cloud points from {args.point_cloud}")

    rng = np.random.default_rng(args.seed)
    t_lo, t_hi = args.baseline_t_range
    if args.sensors is not None:
        sensors = rows_to_sensors(pd.read_csv(args.sensors).values.tolist())
        rays, _, time_gates = sample_rays(sensors, total_rays=args.rays, rng=rng)
        method = f"sensors:{args.sensors.stem}"
    elif args.baseline == "uniform":
        rays = uniform_rays(
            n_rays=args.rays, origin=tuple(args.baseline_origin),
            azimuth_range=tuple(args.baseline_azimuth_range),
            elevation_range=tuple(args.baseline_elevation_range),
            t_min=t_lo, t_max=t_hi,
        )
        time_gates = np.tile([t_lo, t_hi], (len(rays), 1))
        method = "uniform"
    else:
        rays = random_rays(
            n_rays=args.rays, origin=tuple(args.baseline_origin),
            azimuth_range=tuple(args.baseline_azimuth_range),
            elevation_range=tuple(args.baseline_elevation_range),
            t_min=t_lo, t_max=t_hi,
            seed=args.seed,
        )
        time_gates = np.tile([t_lo, t_hi], (len(rays), 1))
        method = "random"
    print(f"[reconstruct] materialized {len(rays)} rays from {method}")

    hits = match_rays_to_cloud(rays, cloud, time_gates, tolerance=args.tol)
    print(f"[reconstruct] {len(hits)}/{len(rays)} rays matched (tolerance {args.tol} m)")
    if len(hits) < 4:
        raise SystemExit(f"Too few hits ({len(hits)}) to triangulate; consider raising --tol.")

    out = reconstruct_mesh_obj(hits, args.out)
    print(f"[reconstruct] wrote OBJ mesh -> {out}")


if __name__ == "__main__":
    main()
