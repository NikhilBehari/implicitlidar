"""Train the implicit density for robot_tracking.

Usage::

    python -m implicitlidar.experiments.robot_tracking.train \\
        --config implicitlidar/experiments/robot_tracking/configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from implicitlidar.core import TargetDensity, train_experiment
from implicitlidar.utils import load_config, parse_overrides

from ._scene import build_robot_scenes


def _build_target(config: dict, device: torch.device) -> TargetDensity:
    target_scene, occluder_scene, trajectories = build_robot_scenes(config, device=device)
    print(f"[train] generated {len(trajectories)} trajectory(ies); "
          f"target SDFs={len(target_scene)}, "
          f"occluder SDFs={len(occluder_scene) if occluder_scene else 0}")
    return TargetDensity.from_config(target_scene, config["target"], occluder_sdf=occluder_scene).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    config = load_config(args.config, parse_overrides(args.override))
    train_experiment(config, build_target=_build_target)


if __name__ == "__main__":
    main()
