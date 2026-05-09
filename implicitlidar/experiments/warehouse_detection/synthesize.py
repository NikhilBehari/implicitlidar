"""Synthesize a motion-adaptive scanning configuration from a trained flow.

Walks a series of robot positions ``x_query`` along the warehouse aisle
and, at each, EM-fits a small Gaussian mixture to flow samples whose
origin falls within ``position_window`` of the query. Each mixture
component is one ray with a learned elevation and time gate, yielding a
motion-adaptive scanning LiDAR design. All per-position rays are written
to a single ``sensors_motion_adaptive_r<n>.csv`` file.

Usage::

    python -m implicitlidar.experiments.warehouse_detection.synthesize \\
        --config implicitlidar/experiments/warehouse_detection/configs/default.yaml \\
        --rays-per-position 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from implicitlidar.core import (
    AZIMUTH_DIM,
    CSV_COLUMNS,
    SensorMixture,
    build_inert_flow,
    sensors_to_rows,
)
from implicitlidar.scenes.warehouse import load_warehouse_scene
from implicitlidar.utils import (
    ensure_dir,
    load_checkpoint,
    load_config,
    parse_overrides,
    select_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--rays-per-position", type=int, default=None,
                        help="Override synthesis.rays_per_position.")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    config = load_config(args.config, parse_overrides(args.override))
    device = select_device()
    print(f"[synthesize] device = {device}")

    run_dir = Path(config["output"]["run_dir"])
    ckpt = args.checkpoint or run_dir / "checkpoints" / "flow_final.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Flow checkpoint not found: {ckpt}")

    scene = load_warehouse_scene(config["scene"]["source"]).to(device)
    flow = build_inert_flow(config, scene, device=device)
    load_checkpoint(flow, ckpt, device=device)
    print(f"[synthesize] loaded flow checkpoint from {ckpt}")

    flow.eval()
    with torch.no_grad():
        design, _ = flow.sample(int(config["synthesis"]["flow_samples"]))
    samples = design.detach().cpu().numpy()
    print(f"[synthesize] sampled {samples.shape[0]} design points from flow")

    synth_cfg = config["synthesis"]
    n_rays = int(args.rays_per_position or synth_cfg["rays_per_position"])
    window = float(synth_cfg["position_window"])
    query_positions = list(synth_cfg["query_positions"])

    out_dir = ensure_dir(run_dir / "sensors")
    csv_path = out_dir / f"sensors_motion_adaptive_r{n_rays}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_query", *CSV_COLUMNS])
        for x_query in query_positions:
            mask = np.abs(samples[:, 0] - x_query) <= window
            if mask.sum() < n_rays:
                print(f"[synthesize] skipping x={x_query}: only {mask.sum()} samples in window")
                continue
            local = samples[mask].copy()
            local[:, 0] = x_query
            mixture = SensorMixture(
                n_components=n_rays,
                origin_mode="fixed",
                fixed_origin=(x_query, 0.0, 0.0),
                circular_dims=(AZIMUTH_DIM,),
                max_iter=int(synth_cfg.get("em_max_iter", 200)),
                rng_seed=int(synth_cfg.get("em_seed", 0)),
            ).fit(local)
            for row in sensors_to_rows(mixture.sensors()):
                writer.writerow([x_query, *row])
            print(f"[synthesize]   x={x_query:.2f}: fit {n_rays} rays from {mask.sum()} samples")

    print(f"[synthesize] motion-adaptive sensors -> {csv_path}")


if __name__ == "__main__":
    main()
