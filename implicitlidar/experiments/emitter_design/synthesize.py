"""Synthesize emitter configurations from a trained emitter_design flow.

EM-fits ``n`` Gaussian components to samples drawn from the trained flow.
Each component is one projector emitter with a learned origin
covariance, azimuth/elevation, and time gate. Writes one
``emitters_<n>.csv`` per fit.

Usage::

    python -m implicitlidar.experiments.emitter_design.synthesize \\
        --config implicitlidar/experiments/emitter_design/configs/default.yaml \\
        --components 1,2,4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from implicitlidar.core import (
    build_inert_flow,
    fit_sensor_mixture,
    synthesize_from_flow,
)
from implicitlidar.scenes.faces import load_face_scene
from implicitlidar.utils import load_config, parse_overrides


def _build_flow(config: dict, device: torch.device):
    return build_inert_flow(
        config, load_face_scene(config["scene"]["source"]).to(device), device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--components", type=str, default=None,
                        help="Comma-separated component counts (e.g. '1,2,4').")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    config = load_config(args.config, parse_overrides(args.override))
    components = (
        [int(c) for c in args.components.split(",")] if args.components
        else [int(config["synthesis"]["n_components"])]
    )
    synthesize_from_flow(
        config,
        build_flow_for_loading=_build_flow,
        fit_components=fit_sensor_mixture,
        component_counts=components,
        csv_name=lambda n: f"emitters_{n}.csv",
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
