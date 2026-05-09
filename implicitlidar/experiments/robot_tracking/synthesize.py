"""Synthesize a distributed sensor configuration from a trained robot_tracking flow.

EM-fits a Gaussian mixture (with full 3×3 origin covariances clamped to a
thin slab) to samples drawn from the trained flow, yielding ``n``
distributed ceiling-mounted sensors covering distinct portions of the
robot's workspace.

Usage::

    python -m implicitlidar.experiments.robot_tracking.synthesize \\
        --config implicitlidar/experiments/robot_tracking/configs/default.yaml \\
        --components 2,4,8
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
from implicitlidar.utils import load_config, parse_overrides

from ._scene import build_robot_scenes


def _build_flow(config: dict, device: torch.device):
    target_scene, _, _ = build_robot_scenes(config, device=device)
    return build_inert_flow(config, target_scene, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--components", type=str, default=None,
                        help="Comma-separated component counts (e.g. '2,4,8').")
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
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
