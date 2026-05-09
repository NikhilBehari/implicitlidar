"""Filesystem helpers for run directories, checkpoints, and config snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create the directory (and parents) if it does not exist; return as `Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(model: nn.Module, path: str | Path, tag: str | int = "final") -> Path:
    """Save `model.state_dict()` to `<path_stem>_<tag>.pth`.

    Example
    -------
    >>> save_checkpoint(flow, "runs/face/model.pth", tag=10000)
    PosixPath('runs/face/model_10000.pth')
    """
    p = Path(path)
    out = p.with_name(f"{p.stem}_{tag}{p.suffix}")
    ensure_dir(out.parent)
    torch.save(model.state_dict(), out)
    return out


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> nn.Module:
    """Load a checkpoint into `model` in-place and return it (in eval mode)."""
    state = torch.load(Path(path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def snapshot_config(config: dict[str, Any], run_dir: str | Path, name: str = "config.yaml") -> Path:
    """Dump a config dict to YAML inside `run_dir` so the run can be re-launched."""
    out = ensure_dir(run_dir) / name
    with out.open("w") as f:
        yaml.safe_dump(_jsonable(config), f, sort_keys=False)
    return out


def _jsonable(obj: Any) -> Any:
    """Recursively convert non-serializable values (e.g. types, devices) to strings."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, type):
        return obj.__name__
    if isinstance(obj, (torch.device, Path)):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)
