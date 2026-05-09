"""Baseline ray-emission strategies used as comparators in every experiment.

Each baseline returns rays in the 6D ``(x, y, z, az, el, τ)``
format, matching :func:`~implicitlidar.core.sampling.sample_rays`.

* :func:`uniform_rays` — even grid in ``(az, el)`` over a fixed FoV from a
  fixed origin (the standard scanning / flash-LiDAR layout).
* :func:`random_rays` — i.i.d. uniform samples in ``(az, el, τ)``.
"""

from __future__ import annotations

import numpy as np

from ..core import AZIMUTH_DIM, ELEVATION_DIM, TIME_DIM


def uniform_rays(
    *,
    n_rays: int,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    azimuth_range: tuple[float, float] = (-np.pi / 2, np.pi / 2),
    elevation_range: tuple[float, float] = (-np.pi / 2, np.pi / 2),
    t_max: float = 1.0,
    t_min: float = 0.0,
) -> np.ndarray:
    """Even ``(az, el)`` grid emitted from a single fixed origin.

    The grid is laid out so that its aspect ratio matches the FoV (the
    azimuth and elevation directions get cells in proportion to their
    angular spans). Each ray's ``τ`` is set to the midpoint of
    ``[t_min, t_max]``.
    """
    az_lo, az_hi = azimuth_range
    el_lo, el_hi = elevation_range
    az_span = az_hi - az_lo
    el_span = el_hi - el_lo
    aspect = az_span / el_span if el_span > 0 else 1.0
    m = max(1, int(np.ceil(np.sqrt(n_rays * aspect))))
    n = int(np.ceil(n_rays / m))
    az_grid = az_lo + (np.arange(1, m + 1) / (m + 1)) * az_span
    el_grid = el_lo + (np.arange(1, n + 1) / (n + 1)) * el_span
    grid = np.array([(a, e) for a in az_grid for e in el_grid][:n_rays])

    rays = np.zeros((len(grid), 6))
    rays[:, :3] = np.asarray(origin, dtype=float)
    rays[:, AZIMUTH_DIM] = grid[:, 0]
    rays[:, ELEVATION_DIM] = grid[:, 1]
    rays[:, TIME_DIM] = 0.5 * (t_min + t_max)
    return rays


def random_rays(
    *,
    n_rays: int,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    azimuth_range: tuple[float, float] = (-np.pi / 2, np.pi / 2),
    elevation_range: tuple[float, float] = (-np.pi / 2, np.pi / 2),
    t_min: float = 0.0,
    t_max: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """i.i.d. uniform samples in ``(az, el, τ)`` from a fixed origin."""
    rng = np.random.default_rng(seed)
    rays = np.zeros((n_rays, 6))
    rays[:, :3] = np.asarray(origin, dtype=float)
    rays[:, AZIMUTH_DIM] = rng.uniform(*azimuth_range, size=n_rays)
    rays[:, ELEVATION_DIM] = rng.uniform(*elevation_range, size=n_rays)
    rays[:, TIME_DIM] = rng.uniform(t_min, t_max, size=n_rays)
    return rays
