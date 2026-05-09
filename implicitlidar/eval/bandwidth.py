"""Per-method bandwidth helpers, layered on top of :func:`bandwidth_mbps`.

* :func:`sensor_bandwidth_mbps` — a learned sensor mixture allocating its
  ray budget across components by mixture weight, with per-sensor time
  gates from the EM-fitted ``σ_τ``.
* :func:`baseline_bandwidth_mbps` — a single virtual "sensor" emitting all
  rays at one time-gate width.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..core import TIME_DIM, Sensor
from .metrics import bandwidth_mbps

# 95% confidence-interval half-width in standard-deviation units. Used as
# the default gate-width multiplier when a learned sensor's σ_τ defines its
# time gate (face_scanning, robot_tracking).
GATE_Z_95: float = 1.96


def sensor_bandwidth_mbps(
    sensors: Sequence[Sensor], n_rays: int, *,
    gate_z: float = GATE_Z_95,
    scaling_factor_m: float = 1.0,
) -> float:
    """Bandwidth (Mbps) for a sensor mixture allocating ``n_rays`` by weight.

    ``gate_z`` is the half-width of each per-sensor time gate in units of
    that sensor's σ_τ (1.96 for a 95 % CI; warehouse_detection passes 2.0).
    ``scaling_factor_m`` converts τ-units into physical meters when the
    design space is not already in meters.
    """
    weights = np.array([s.weight for s in sensors], dtype=float)
    weights = weights / weights.sum()
    rays_per_sensor = (weights * n_rays).astype(int)
    gate_widths = np.array([2 * gate_z * s.sigma[TIME_DIM] for s in sensors])
    return bandwidth_mbps(
        rays_per_sensor=rays_per_sensor,
        time_gate_per_sensor_m=gate_widths,
        scaling_factor_m=scaling_factor_m,
    )


def baseline_bandwidth_mbps(
    n_rays: int, gate_width_m: float, *, scaling_factor_m: float = 1.0,
) -> float:
    """Bandwidth (Mbps) for a baseline emitting ``n_rays`` rays with a single ``gate_width_m`` gate."""
    return bandwidth_mbps(
        rays_per_sensor=np.array([n_rays]),
        time_gate_per_sensor_m=np.array([gate_width_m]),
        scaling_factor_m=scaling_factor_m,
    )
