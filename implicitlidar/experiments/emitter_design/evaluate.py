"""Evaluate emitter designs against the co-located baseline.

Pipeline:

1. For each face mesh, render a transient image with Mitsuba 3 +
   ``mitransient`` using the synthesized emitters as projector light
   sources and a fixed perspective sensor as the detector.
2. For each detector ray (drawn from a face_scanning sensor design),
   look up the corresponding pixel in the transient image. The peak
   bin of the histogram for each emitter (encoded as one color channel)
   gives the optical path length ``L = ‖P − Oₑ‖ + ‖P − O_d‖``.
3. Solve the bistatic geometry equation for the surface point ``P``
   along the detector ray.
4. Reconstruct a mesh from the per-pixel hit points (Delaunay
   triangulation in the dominant plane) and compute the symmetric
   squared Chamfer distance against the ground-truth surface.

A single co-located emitter at the detector origin is included as a
baseline for head-to-head comparison.

Usage::

    python -m implicitlidar.experiments.emitter_design.evaluate \\
        --config implicitlidar/experiments/emitter_design/configs/default.yaml \\
        --components 1,2,4
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
    rows_to_sensors,
    sample_rays,
)
from implicitlidar.eval import mean_pm_2sem, mesh_chamfer_squared, mesh_from_hits
from implicitlidar.utils import ensure_dir, load_config, parse_overrides

from ._render import hit_points_from_transient, render_transient
from ._xml import LIDAR_TEMPLATE_XML, Frustum, frustums_from_sensors

# Inverse of the Mitsuba scene template's -90° rotation about X.
_SCENE_TO_FILE_ROT = np.array([
    [1.0,  0.0,  0.0],
    [0.0,  0.0, -1.0],
    [0.0,  1.0,  0.0],
])


def reconstructed_chamfer_squared(rec_pts: np.ndarray, gt_mesh: trimesh.Trimesh) -> float:
    """Delaunay-reconstruct ``rec_pts`` then mesh-vs-mesh squared Chamfer vs GT."""
    rec_pts_in_file_frame = rec_pts @ _SCENE_TO_FILE_ROT.T
    rec_mesh = mesh_from_hits(rec_pts_in_file_frame)
    if rec_mesh is None:
        return float("inf")
    return mesh_chamfer_squared(rec_mesh, gt_mesh)


def _enumerate_designs(
    sensor_dir: Path, components_arg: str | None,
    baseline_detector_origin: tuple[float, float, float],
) -> list[tuple[str, list[Frustum]]]:
    """Enumerate the (name, frustums) pairs to render: baseline + each EM fit."""
    designs: list[tuple[str, list[Frustum]]] = []
    designs.append(("baseline_colocated", [
        Frustum(origin=baseline_detector_origin, azimuth_deg=180.0, elevation_deg=0.0,
                fov_width_deg=60.0, fov_height_deg=50.0)
    ]))
    csv_paths = sorted(sensor_dir.glob("emitters_*.csv"))
    if components_arg:
        wanted = {int(c) for c in components_arg.split(",")}
        csv_paths = [p for p in csv_paths if int(p.stem.split("_")[-1]) in wanted]
    for p in csv_paths:
        n = int(p.stem.split("_")[-1])
        sensors = rows_to_sensors(pd.read_csv(p).values.tolist())
        designs.append((f"emitter_{n}", frustums_from_sensors(sensors)))
    return designs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--components", type=str, default=None,
                        help="Comma-separated emitter component counts. "
                             "Defaults to every CSV in <run_dir>/sensors.")
    parser.add_argument("--temporal-bins", type=int, default=20)
    parser.add_argument("--temporal-range", type=float, nargs=2, default=(0.0, 6.0))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--spp", type=int, default=2048)
    parser.add_argument("--detector-rays", type=int, default=576,
                        help="Total detector rays drawn from the detector sensor design.")
    parser.add_argument("--detector-sensors", type=Path,
                        default=Path("outputs/runs/face_scanning/default/sensors/sensors_10.csv"),
                        help="Sensor CSV from face_scanning whose 10-component mixture defines "
                             "the detector ray pattern; rays are laid out via even-grid sampling.")
    parser.add_argument("--ground-truth-samples", type=int, default=10_000)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    config = load_config(args.config, parse_overrides(args.override))

    run_dir = Path(config["output"]["run_dir"])
    sensor_dir = run_dir / "sensors"
    if not sensor_dir.exists():
        raise FileNotFoundError(f"No sensors found at {sensor_dir}; run synthesize first.")

    test_paths = sorted(Path(config["evaluation"]["test_scene"]).glob("*.obj"))
    if not test_paths:
        raise FileNotFoundError(f"No test meshes found in {config['evaluation']['test_scene']}")
    print(f"[evaluate] {len(test_paths)} test face mesh(es)")

    detector_origin = tuple(float(v) for v in config["scene"]["detector_origin"])
    if not args.detector_sensors.exists():
        raise FileNotFoundError(
            f"Detector sensor CSV not found at {args.detector_sensors}; "
            "run face_scanning.synthesize first or pass --detector-sensors."
        )
    detector_sensors = rows_to_sensors(pd.read_csv(args.detector_sensors).values.tolist())
    det_rays_arr, _, det_gates = sample_rays(
        detector_sensors, total_rays=args.detector_rays, rng=np.random.default_rng(0),
    )
    detector_rays = [
        {
            "origin_x": float(r[0]), "origin_y": float(r[1]), "origin_z": float(r[2]),
            "azimuth": float(r[AZIMUTH_DIM]), "elevation": float(r[ELEVATION_DIM]),
            "t_min": float(g[0]), "t_max": float(g[1]),
        }
        for r, g in zip(det_rays_arr, det_gates)
    ]
    print(f"[evaluate] generated {len(detector_rays)} detector rays from {args.detector_sensors.name}")

    designs = _enumerate_designs(sensor_dir, args.components, detector_origin)

    in_xml = LIDAR_TEMPLATE_XML
    if not in_xml.exists():
        raise FileNotFoundError(f"Missing bundled scene template: {in_xml}")

    work_xml = ensure_dir(run_dir / "results") / "scene_temp.xml"
    rows: list[dict] = []
    for design_name, frustums in designs:
        chamfers = []
        for mesh_path in tqdm(test_paths, desc=design_name):
            transient, cam_info = render_transient(
                in_xml, work_xml,
                frustums=frustums, mesh_path=str(mesh_path),
                resolution=args.resolution, temporal_bins=args.temporal_bins,
                temporal_range=tuple(args.temporal_range), spp=args.spp,
            )
            rec_pts = hit_points_from_transient(
                transient,
                detector_rays=detector_rays,
                frustums=frustums,
                cam_info=cam_info,
                temporal_bins=args.temporal_bins,
                temporal_range=tuple(args.temporal_range),
            )
            valid_pts = rec_pts[~np.isnan(rec_pts).any(axis=1)]
            if len(valid_pts) < 4:
                continue
            gt_mesh = trimesh.load(str(mesh_path), force="mesh")
            chamfers.append(reconstructed_chamfer_squared(valid_pts, gt_mesh))
        if not chamfers:
            continue
        mean, two_sem = mean_pm_2sem(np.array(chamfers))
        rows.append({
            "design": design_name,
            "n_emitters": len(frustums),
            "n_test_scenes": len(chamfers),
            "chamfer_sq_mean": round(mean, 6),
            "chamfer_sq_2sem": round(two_sem, 6),
            "temporal_bins": args.temporal_bins,
            "resolution": args.resolution,
            "spp": args.spp,
        })

    df = pd.DataFrame(rows)
    out_csv = ensure_dir(run_dir / "results") / "transient_chamfer.csv"
    df.to_csv(out_csv, index=False)
    print(f"[evaluate] results saved to {out_csv}")
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
