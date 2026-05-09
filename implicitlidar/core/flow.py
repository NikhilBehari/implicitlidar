"""Normalizing-flow representation of the implicit density.

A stack of autoregressive rational-quadratic spline layers (via `normflows`)
transforms a 6D uniform base distribution into the task-driven implicit
density. Dimensions with ``low == high`` are held fixed at sample time and
not learned by the flow.
"""

from __future__ import annotations

import normflows as nf
import torch
import torch.nn as nn

from .design_space import Bounds

# Base distribution: uniform over the free subspace


class UniformDistribution(nn.Module):
    """A 6D-or-lower uniform base distribution over the free subspace.

    Implements both the ``forward`` (sample) and ``log_prob`` interfaces that
    `normflows.NormalizingFlow` expects.
    """

    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        super().__init__()
        self.register_buffer("low", low)
        self.register_buffer("high", high)
        self.dim = low.shape[0]
        # Uniform density is constant over the box; precompute its log-value.
        self.register_buffer(
            "log_prob_value",
            torch.tensor(-torch.sum(torch.log(high - low)).item()),
        )

    def forward(self, num_samples: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        eps = torch.rand((num_samples, self.dim), device=self.low.device)
        z = self.low + (self.high - self.low) * eps
        # Materialize a fresh (non-view) tensor: downstream `normflows` mutates
        # this in place when accumulating Jacobian log-determinants.
        log_p = torch.full((num_samples,), self.log_prob_value.item(), device=self.low.device)
        return z, log_p

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        log_p = torch.full((z.shape[0],), self.log_prob_value.item(), device=z.device)
        out_of_box = ((z < self.low) | (z > self.high)).any(dim=1)
        log_p[out_of_box] = float("-inf")
        return log_p


# A simple invertible shift layer (used to center the spline domain at 0)


class ShiftTransform(nn.Module):
    """``f(x) = x − c`` with unit Jacobian. Composes inside a normalizing flow."""

    def __init__(self, shift: torch.Tensor):
        super().__init__()
        self.register_buffer("shift", shift)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x - self.shift, torch.zeros(x.shape[0], device=x.device)

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return y + self.shift, torch.zeros(y.shape[0], device=y.device)


# Flow model: spline stack + bound-aware sampling


class FlowModel(nn.Module):
    """The implicit density ``p(d; Θ)`` as an autoregressive spline flow.

    Wraps a `normflows.NormalizingFlow` plus the :class:`Bounds` it was trained
    on, so that ``sample()`` always yields full-dimensional ``(N, 6)`` design
    points (with fixed dimensions reinjected as constants).

    Parameters
    ----------
    flow
        The underlying `normflows.NormalizingFlow` over the free subspace.
    bounds
        The :class:`Bounds` that defines which dimensions are learned vs. fixed.
    """

    def __init__(self, flow: nf.NormalizingFlow, bounds: Bounds):
        super().__init__()
        self.flow = flow
        self.bounds = bounds

    def sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``num_samples`` design points and their log-densities.

        Returns
        -------
        design : (N, 6) tensor of full design points.
        log_q  : (N,)   tensor of log densities under the flow.
        """
        free_z, log_q = self.flow.sample(num_samples)
        return self.bounds.expand(free_z), log_q

    def log_prob(self, free_z: torch.Tensor) -> torch.Tensor:
        """Log density under the flow for free-subspace points."""
        return self.flow.log_prob(free_z)

    @property
    def target(self):
        """The target density module attached to the underlying flow (for training)."""
        return self.flow.p

    @target.setter
    def target(self, value):
        self.flow.p = value


# Construction


def build_spline_stack(
    *,
    free_dims: int,
    num_layers: int,
    num_blocks: int,
    hidden_channels: int,
    num_bins: int,
    tail_bound: torch.Tensor,
    circular_indices: list[int],
    activation: type[nn.Module] = nn.ReLU,
    dropout: float = 0.0,
    permute_mask: bool = True,
    init_identity: bool = True,
) -> list[nn.Module]:
    """Construct ``num_layers`` autoregressive rational-quadratic spline layers.

    Each layer is a ``CircularAutoregressiveRationalQuadraticSpline`` from
    `normflows` that respects circular dimensions (e.g. the azimuth ``φ``).
    The Jacobian is strictly triangular and its log-determinant is an O(D)
    diagonal sum.
    """
    return [
        nf.flows.CircularAutoregressiveRationalQuadraticSpline(
            num_input_channels=free_dims,
            num_blocks=num_blocks,
            num_hidden_channels=hidden_channels,
            ind_circ=list(circular_indices),
            num_bins=num_bins,
            tail_bound=tail_bound,
            activation=activation,
            dropout_probability=dropout,
            permute_mask=permute_mask,
            init_identity=init_identity,
        )
        for _ in range(num_layers)
    ]


def build_flow(
    bounds: Bounds,
    target,  # `TargetDensity`
    flow_config: dict,
    device: torch.device | str = "cpu",
) -> FlowModel:
    """Instantiate a :class:`FlowModel` from a config block.

    The expected ``flow_config`` schema is

    .. code-block:: yaml

        num_layers: 6
        circular_dims: [3]            # azimuth is periodic by default
        spline:
          num_blocks: 2
          hidden_channels: 64
          num_bins: 16
          tail_bound_multiplier: 1.0  # tail_bound = multiplier * half_range
          activation: !!python/name:torch.nn.ReLU
          dropout: 0.0
          permute_mask: true
          init_identity: true
    """
    spline_cfg = flow_config["spline"]
    free_low = bounds.free_low().to(device)
    free_high = bounds.free_high().to(device)
    free_dims = bounds.free_dims
    centers = ((free_low + free_high) / 2).to(device)
    half_ranges = ((free_high - free_low) / 2).to(device)
    tail_bound = half_ranges * float(spline_cfg.get("tail_bound_multiplier", 1.0))

    circular_full = list(flow_config.get("circular_dims", [3]))
    circular_in_free = _remap_circular_indices(circular_full, bounds.free_mask)

    base = UniformDistribution(free_low, free_high)
    spline_stack = build_spline_stack(
        free_dims=free_dims,
        num_layers=int(flow_config["num_layers"]),
        num_blocks=int(spline_cfg["num_blocks"]),
        hidden_channels=int(spline_cfg["hidden_channels"]),
        num_bins=int(spline_cfg["num_bins"]),
        tail_bound=tail_bound,
        circular_indices=circular_in_free,
        activation=spline_cfg.get("activation", nn.ReLU),
        dropout=float(spline_cfg.get("dropout", 0.0)),
        permute_mask=bool(spline_cfg.get("permute_mask", True)),
        init_identity=bool(spline_cfg.get("init_identity", True)),
    )
    transforms = [ShiftTransform(centers), *spline_stack, ShiftTransform(-centers)]
    flow = nf.NormalizingFlow(base, transforms, target).to(device)
    return FlowModel(flow=flow, bounds=bounds)


def _remap_circular_indices(circular_full: list[int], free_mask: torch.Tensor) -> list[int]:
    """Map circular dim indices from the full (6D) space into the free subspace.

    For example, if ``z`` is held fixed (index 2) and azimuth (index 3) is
    circular, the spline operates over a 5-D free space and azimuth maps to
    free-space index 2.
    """
    out = []
    free_idx = 0
    for i in range(6):
        if not bool(free_mask[i].item()):
            continue
        if i in circular_full:
            out.append(free_idx)
        free_idx += 1
    return out
