"""Train the implicit density for face_scanning.

Usage::

    python -m implicitlidar.experiments.face_scanning.train \\
        --config implicitlidar/experiments/face_scanning/configs/default.yaml

Add ``--override training.iterations=200`` to point-edit any nested key.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from implicitlidar.core import TargetDensity, cache_kwargs_from_config, train_experiment
from implicitlidar.scenes.faces import load_face_scene
from implicitlidar.utils import load_config, parse_overrides


def _build_target(config: dict, device: torch.device) -> TargetDensity:
    scene_cfg = config["scene"]
    scene = load_face_scene(scene_cfg["source"], **cache_kwargs_from_config(scene_cfg, device)).to(device)
    print(f"[train] loaded {len(scene)} face mesh(es) from {scene_cfg['source']}")
    return TargetDensity.from_config(scene, config["target"]).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    config = load_config(args.config, parse_overrides(args.override))
    train_experiment(config, build_target=_build_target)


if __name__ == "__main__":
    main()
