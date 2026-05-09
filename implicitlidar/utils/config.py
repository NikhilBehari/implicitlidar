"""YAML configuration loading.

A configuration is a nested dict. We provide:

* :func:`load_config` — read a YAML file, with an optional ``--override`` syntax for
  point edits (e.g. ``training.iterations=500``).
* :func:`resolve_activation` — convert string activation names (``"ReLU"``) to
  ``torch.nn`` classes, since YAML cannot represent Python types.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch.nn as nn
import yaml


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a YAML config and apply optional dotted-key overrides.

    Parameters
    ----------
    path
        Path to the YAML file.
    overrides
        Mapping ``{"training.iterations": 500, ...}``. Each key is a
        dot-separated path into the nested config.

    Returns
    -------
    The fully-loaded config dict, with any string activation names resolved to
    `torch.nn` classes.
    """
    with Path(path).open() as f:
        config = yaml.safe_load(f) or {}
    if overrides:
        for dotted_key, value in overrides.items():
            _set_nested(config, dotted_key.split("."), value)
    _resolve_activations(config)
    return config


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """Parse a list of ``"key.subkey=value"`` strings into an overrides dict.

    Values are YAML-decoded so that ``training.iterations=500`` becomes ``int``,
    ``training.lr=1e-4`` becomes ``float``, ``flow.spline.permute_mask=true``
    becomes ``bool``, and ``design_space.low=[-1,-1,1,0,-1.57,0]`` becomes a list.
    """
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must be of the form key=value, got {item!r}")
        key, raw = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(raw)
    return out


def merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``updates`` into a deep copy of ``base``."""
    out = copy.deepcopy(base)
    _deep_update(out, updates)
    return out


# Internal helpers


def _set_nested(d: dict, keys: list[str], value: Any) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def _resolve_activations(config: dict) -> None:
    """Walk the config tree and convert ``activation: "ReLU"`` to ``nn.ReLU``."""
    if not isinstance(config, dict):
        return
    for key, value in config.items():
        if key == "activation" and isinstance(value, str):
            config[key] = getattr(nn, value)
        elif isinstance(value, dict):
            _resolve_activations(value)
