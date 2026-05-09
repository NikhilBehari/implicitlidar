"""Reverse-KL training of the implicit density.

Minimizes ``E_q[(1 + η)·log q(d) - log p*(d)]`` over flow samples; ``η`` is
the entropy regularization that promotes diverse sampled designs. Supports
an optional linear warmup on the visibility weight so the model first learns
to place rays on surfaces before being penalized for occluded ones.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.optim as optim
from tqdm import tqdm

from .flow import FlowModel
from .target_density import TargetDensity

CallbackFn = Callable[[int, dict[str, Any]], None]
"""A callback ``cb(iteration, info)`` invoked once per training step.

``info`` always contains ``loss``, ``visibility_weight``, and the sampled
``design`` tensor. Return value is ignored.
"""


def linear_warmup(step: int, total_steps: int, warmup_iters: int | None, final_weight: float) -> float:
    """Linear ramp from 0 to ``final_weight`` over ``[warmup_iters, total_steps]``.

    Returns ``final_weight`` if ``warmup_iters`` is ``None`` (no warmup).
    """
    if warmup_iters is None:
        return float(final_weight)
    if step < warmup_iters:
        return 0.0
    denom = total_steps - warmup_iters
    progress = (step - warmup_iters) / denom if denom > 0 else 1.0
    return float(progress * final_weight)


def train_flow(
    flow: FlowModel,
    *,
    iterations: int,
    batch_size: int,
    learning_rate: float = 1e-3,
    entropy_reg: float = 0.0,
    visibility_warmup_iters: int | None = None,
    callbacks: list[CallbackFn] | None = None,
    progress: bool = True,
) -> FlowModel:
    """Train ``flow`` against the target density attached to it.

    The flow's ``target`` attribute must be a :class:`TargetDensity` whose
    ``visibility_weight`` will be modulated each step (when warmup is enabled).

    Parameters
    ----------
    flow
        A :class:`FlowModel` whose underlying `normflows.NormalizingFlow` was
        constructed with a target via :func:`~.flow.build_flow`.
    iterations
        Number of optimizer steps.
    batch_size
        Number of design samples drawn per step.
    learning_rate
        Adam learning rate.
    entropy_reg
        Entropy-regularization coefficient ``η``. Values around
        0.5–1.0 produce diverse samples without sacrificing density quality.
    visibility_warmup_iters
        If given, the visibility weight ramps linearly from ``0`` to its
        configured final value over ``[warmup_iters, iterations]``.
    callbacks
        Optional list of callbacks invoked once per step (see :data:`CallbackFn`).
    progress
        Show a `tqdm` progress bar.

    Returns
    -------
    The same ``flow``, trained in-place.
    """
    target = flow.target
    if not isinstance(target, TargetDensity):
        raise TypeError(
            f"flow.target must be a TargetDensity, got {type(target).__name__}. "
            "Use build_flow(bounds, target, ...) to attach one."
        )
    optimizer = optim.Adam(flow.flow.parameters(), lr=learning_rate)
    final_visibility_weight = float(target.visibility_weight)

    iterator = range(iterations)
    if progress:
        iterator = tqdm(iterator, desc="train_flow")
    for step in iterator:
        optimizer.zero_grad()
        design, log_q = flow.sample(batch_size)
        weight = linear_warmup(step, iterations, visibility_warmup_iters, final_visibility_weight)
        target.visibility_weight = weight
        loss = torch.mean((1 + entropy_reg) * log_q - target(design))
        loss.backward()
        optimizer.step()

        if callbacks:
            info = {"loss": loss.item(), "visibility_weight": weight, "design": design.detach()}
            for cb in callbacks:
                cb(step, info)
        if progress:
            iterator.set_postfix(loss=f"{loss.item():.3f}", vis_w=f"{weight:.2f}")

    # Restore the user's configured weight (even if warmup did not finish).
    target.visibility_weight = final_visibility_weight
    return flow


def average_visibility(flow: FlowModel, n_samples: int = 5000) -> float:
    """Mean ray visibility of samples drawn from the flow (sanity check)."""
    target = flow.target
    if not isinstance(target, TargetDensity) or not target.use_visibility:
        return float("nan")
    from .target_density import ray_visibility
    design, _ = flow.sample(n_samples)
    occluder = target.occluder_sdf if target.occluder_sdf is not None else target.scene_sdf
    vis = ray_visibility(
        design,
        occluder,
        num_samples=target.visibility_num_samples,
        beta=target.visibility_beta,
    )
    return float(vis.mean().item())
