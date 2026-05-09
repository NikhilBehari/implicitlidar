"""Shared training/synthesis runners used by every experiment's CLI.

Each experiment's ``train.py`` and ``synthesize.py`` is a thin wrapper that
provides one task-specific factory (the scene loader / mixture-fitter) and
delegates the rest of the pipeline to the helpers here.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch

from ..utils import (
    ensure_dir,
    load_checkpoint,
    save_checkpoint,
    select_device,
    snapshot_config,
)
from .design_space import Bounds
from .em_synthesis import CSV_COLUMNS, Sensor, sensors_to_rows
from .flow import FlowModel, build_flow
from .target_density import SceneSDF, TargetDensity
from .training import train_flow

# Training


def train_experiment(
    config: dict,
    *,
    build_target: Callable[[dict, torch.device], TargetDensity],
    device: torch.device | str | None = None,
) -> Path:
    """Train the implicit density end-to-end from a parsed YAML config.

    ``build_target`` is the only task-specific piece: it receives the parsed
    config and the resolved device, and returns a fully-constructed
    :class:`TargetDensity` (already on the device). The runner then handles
    bounds / flow construction, the training loop, periodic checkpointing,
    and the final checkpoint save. Returns the path to the final checkpoint.
    """
    device = torch.device(device) if device else select_device()
    print(f"[train] device = {device}")

    run_dir = ensure_dir(config["output"]["run_dir"])
    snapshot_config(config, run_dir)

    target = build_target(config, device)

    bounds = Bounds.from_config(config["design_space"], device=device)
    flow = build_flow(bounds, target, config["flow"], device=device)

    train_cfg = config["training"]
    print(f"[train] training flow: {train_cfg['iterations']} iters, "
          f"batch_size={train_cfg['batch_size']}")

    train_flow(
        flow,
        iterations=train_cfg["iterations"],
        batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["learning_rate"],
        entropy_reg=train_cfg["entropy_reg"],
        visibility_warmup_iters=train_cfg.get("visibility_warmup_iters"),
        callbacks=[periodic_checkpoint(flow, run_dir, config["output"]["save_every"])],
    )

    final = save_checkpoint(flow, run_dir / "checkpoints" / "flow.pth", tag="final")
    print(f"[train] saved final checkpoint to {final}")
    return final


def periodic_checkpoint(flow: FlowModel, run_dir: Path, save_every: int):
    """A `train_flow` callback that snapshots the flow every ``save_every`` iterations."""
    def cb(step: int, info: dict) -> None:
        if save_every > 0 and step > 0 and step % save_every == 0:
            save_checkpoint(flow, run_dir / "checkpoints" / "flow.pth", tag=step)
    return cb


def build_inert_flow(
    config: dict, scene_sdf: SceneSDF, *, device: torch.device,
) -> FlowModel:
    """Reconstruct the flow architecture for checkpoint loading.

    The target density is needed to instantiate the flow but is not used
    during synthesis (no gradient computation), so we attach an inert
    visibility-off :class:`TargetDensity`. The loaded checkpoint overwrites
    every learned weight.
    """
    bounds = Bounds.from_config(config["design_space"], device=device)
    target = TargetDensity(scene_sdf, sigma=config["target"]["sigma"], use_visibility=False).to(device)
    return build_flow(bounds, target, config["flow"], device=device)


# Synthesis


def synthesize_from_flow(
    config: dict,
    *,
    build_flow_for_loading: Callable[[dict, torch.device], FlowModel],
    fit_components: Callable[[Any, int, dict], list[Sensor]],
    component_counts: list[int],
    csv_name: Callable[[int], str] = lambda n: f"sensors_{n}.csv",
    extra_csv_columns: Iterable[str] = (),
    extra_row_prefix: Callable[[int], list] | None = None,
    checkpoint: Path | None = None,
    device: torch.device | str | None = None,
) -> list[Path]:
    """Sample the trained flow and fit a sensor mixture per component count.

    ``build_flow_for_loading`` reconstructs the flow architecture from the
    config so that ``load_checkpoint`` can repopulate the weights.
    ``fit_components(samples, n, synthesis_cfg)`` returns the synthesized
    sensors. The runner samples from the flow, dispatches to the fitter for
    every requested component count, and writes one CSV per fit. Returns the
    list of written CSV paths.

    For per-position synthesis (warehouse_detection), pass ``extra_csv_columns``
    and ``extra_row_prefix`` to prepend a column (e.g. ``x_query``) to every row.
    """
    device = torch.device(device) if device else select_device()
    print(f"[synthesize] device = {device}")

    run_dir = Path(config["output"]["run_dir"])
    ckpt = checkpoint or run_dir / "checkpoints" / "flow_final.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Flow checkpoint not found: {ckpt}")

    flow = build_flow_for_loading(config, device)
    load_checkpoint(flow, ckpt, device=device)
    print(f"[synthesize] loaded flow checkpoint from {ckpt}")

    flow.eval()
    with torch.no_grad():
        design, _ = flow.sample(int(config["synthesis"]["flow_samples"]))
    samples = design.detach().cpu().numpy()
    print(f"[synthesize] sampled {samples.shape[0]} design points from flow")

    out_dir = ensure_dir(run_dir / "sensors")
    written: list[Path] = []
    for n in component_counts:
        sensors = fit_components(samples, n, config["synthesis"])
        path = out_dir / csv_name(n)
        _write_sensor_csv(path, sensors, extra_columns=list(extra_csv_columns),
                          extra_prefix=extra_row_prefix(n) if extra_row_prefix else None)
        print(f"[synthesize] {n} sensors -> {path}")
        written.append(path)
    return written


def _write_sensor_csv(
    path: Path,
    sensors: list[Sensor],
    *,
    extra_columns: list[str] | None = None,
    extra_prefix: list | None = None,
) -> None:
    """Write a sensor CSV with the 16-column schema."""
    extra_columns = extra_columns or []
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*extra_columns, *CSV_COLUMNS])
        for row in sensors_to_rows(sensors):
            writer.writerow([*(extra_prefix or []), *row])
