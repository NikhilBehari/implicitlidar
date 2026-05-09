"""Train the implicit density for warehouse_detection.

Usage::

    python -m implicitlidar.experiments.warehouse_detection.train \\
        --config implicitlidar/experiments/warehouse_detection/configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from implicitlidar.core import TargetDensity, cache_kwargs_from_config, train_experiment
from implicitlidar.scenes.warehouse import load_warehouse_scene
from implicitlidar.utils import load_config, parse_overrides


def _build_target(config: dict, device: torch.device) -> TargetDensity:
    scene_cfg = config["scene"]
    cache_kwargs = cache_kwargs_from_config(scene_cfg, device)
    target_scene = load_warehouse_scene(scene_cfg["source"], **cache_kwargs).to(device)
    occluder_scene = (
        load_warehouse_scene(scene_cfg["occluder"], **cache_kwargs).to(device)
        if scene_cfg.get("occluder") else None
    )
    print(f"[train] loaded {len(target_scene)} target scene(s)"
          + (f" and {len(occluder_scene)} occluder(s)" if occluder_scene else ""))
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
