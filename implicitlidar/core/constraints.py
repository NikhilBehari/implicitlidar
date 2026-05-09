"""Constraint application.

Constraints act on either :class:`Bounds` (support truncation, before flow
sampling) or :class:`SensorMixture` (sigma clamps, fixed origins, applied
post-EM). Compose them by applying in sequence; no flow retraining required.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

from .design_space import (
    AZIMUTH_DIM,
    ELEVATION_DIM,
    SPATIAL_DIMS,
    TIME_DIM,
    Bounds,
)
from .em_synthesis import SensorMixture

# Bounds-level constraints (applied before flow sampling)


@dataclass(frozen=True)
class SupportTruncation:
    """Restrict the flow's support to a sub-box of ``D``.

    Each entry of ``low``/``high`` is either a number (clipping bound) or
    ``None`` (leave the existing bound unchanged). Use this to enforce
    placement, angular, or temporal limits without retraining.

    Example
    -------
    >>> # Restrict origins to the ceiling plane (z = 1.0).
    >>> SupportTruncation(low=[None, None, 1.0, None, None, None],
    ...                   high=[None, None, 1.0, None, None, None])
    """

    low: list[float | None]
    high: list[float | None]

    def __post_init__(self):
        if len(self.low) != 6 or len(self.high) != 6:
            raise ValueError("SupportTruncation low/high must each have length 6.")

    def apply(self, bounds: Bounds) -> Bounds:
        new_low = bounds.low.clone()
        new_high = bounds.high.clone()
        for i in range(6):
            if self.low[i] is not None:
                new_low[i] = max(float(new_low[i].item()), float(self.low[i]))
            if self.high[i] is not None:
                new_high[i] = min(float(new_high[i].item()), float(self.high[i]))
        if (new_high < new_low).any():
            raise ValueError(
                f"Truncation produced an empty box: low={new_low.tolist()}, high={new_high.tolist()}"
            )
        return Bounds(low=new_low, high=new_high)


# Sensor-mixture constraints (applied before/after EM)


@dataclass(frozen=True)
class SigmaClamp:
    """Bound the per-component standard deviations of selected dimensions.

    Use this to cap field of view (clamp on ``φ``, ``ψ``) or shorten time
    gates (clamp on ``τ``).

    Example
    -------
    >>> SigmaClamp({AZIMUTH_DIM: (0.0, 0.3), TIME_DIM: (0.0, 0.1)})
    """

    bounds_per_dim: dict[int, tuple[float, float]]

    def apply(self, mixture: SensorMixture) -> SensorMixture:
        for dim, (lo, hi) in self.bounds_per_dim.items():
            mixture.sigmas[:, dim] = np.clip(mixture.sigmas[:, dim], lo, hi)
        return mixture


@dataclass(frozen=True)
class FixSensorOrigin:
    """Pin every component's spatial origin to a single point.

    Equivalent to constructing the mixture with ``origin_mode='fixed'``, but
    can be applied post-hoc to override origins learned by EM.
    """

    origin: tuple[float, float, float]

    def apply(self, mixture: SensorMixture) -> SensorMixture:
        mixture.means[:, list(SPATIAL_DIMS)] = np.asarray(self.origin, dtype=float)
        mixture.sigmas[:, list(SPATIAL_DIMS)] = 0.0
        mixture.origin_covariances[:] = 0.0
        return mixture


# Sample-level filtering (used when post-truncating samples drawn from a flow)


def filter_samples_by_support(
    samples: torch.Tensor,
    truncation: SupportTruncation,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean-mask samples to those satisfying ``truncation``.

    Returns ``(filtered_samples, mask)``. Useful when you want to sample
    densely from a learned flow then enforce a constraint at the sample level
    (rather than refitting the flow over the truncated bounds).
    """
    keep = torch.ones(samples.size(0), dtype=torch.bool, device=samples.device)
    for i in range(6):
        if truncation.low[i] is not None:
            keep &= samples[:, i] >= float(truncation.low[i])
        if truncation.high[i] is not None:
            keep &= samples[:, i] <= float(truncation.high[i])
    return samples[keep], keep


def apply_constraints(
    bounds: Bounds | None = None,
    mixture: SensorMixture | None = None,
    *,
    truncation: SupportTruncation | None = None,
    sensor_constraints: Iterable[SigmaClamp | FixSensorOrigin] = (),
) -> tuple[Bounds | None, SensorMixture | None]:
    """Convenience wrapper that applies any combination of constraints.

    Returns the (possibly modified) ``(bounds, mixture)`` pair.
    """
    if bounds is not None and truncation is not None:
        bounds = truncation.apply(bounds)
    if mixture is not None:
        for c in sensor_constraints:
            c.apply(mixture)
    return bounds, mixture


__all__ = [
    "SupportTruncation",
    "SigmaClamp",
    "FixSensorOrigin",
    "filter_samples_by_support",
    "apply_constraints",
    # Re-export dim indices for convenient construction.
    "AZIMUTH_DIM",
    "ELEVATION_DIM",
    "TIME_DIM",
    "SPATIAL_DIMS",
]
