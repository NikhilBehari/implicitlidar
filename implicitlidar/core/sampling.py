"""Ray emission from synthesized sensors.

Materialize physical rays from a fitted :class:`~.em_synthesis.SensorMixture`:
the global ray budget is allocated across components by mixture weight, then
each ray is laid out on an even ``(az, el)`` grid inside that component's
95 % confidence ellipse, with the per-sensor 95 % CI on ``τ`` returned as
the ray's time gate ``[t_min, t_max]``.
"""

from __future__ import annotations

import numpy as np

from .design_space import (
    AZIMUTH_DIM,
    ELEVATION_DIM,
    SPATIAL_DIMS,
    TIME_DIM,
    direction_vector_np,
)
from .em_synthesis import Sensor, _ci_to_z

_DEFAULT_RNG = np.random.default_rng()


def allocate_ray_budget(
    weights: np.ndarray, total_rays: int, *, rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Distribute ``total_rays`` across components in proportion to ``weights``.

    Each ray is independently assigned to a component via weighted sampling;
    the returned ``(G,)`` integer counts always sum to ``total_rays``.
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    rng = rng or _DEFAULT_RNG
    picks = rng.choice(len(weights), size=total_rays, p=weights)
    counts = np.bincount(picks, minlength=len(weights))
    return counts.astype(int)


def sample_rays(
    sensors: list[Sensor],
    total_rays: int,
    *,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lay out an even ``(az, el)`` grid of rays inside each sensor's 95 % CI.

    For each sensor allocated ``k`` rays, picks an ``m × n`` grid in the
    ``(az, el)`` CI box (with the grid's aspect ratio matching the box) and
    keeps the first ``k`` cells; the time gate ``[t_min, t_max]`` is the
    per-sensor ``τ`` CI. Returns

    ``rays``        — ``(total_rays, 6)`` array of design points, with
                      ``rays[:, TIME_DIM]`` set to each sensor's ``τ`` mean.
    ``sensor_ids``  — ``(total_rays,)`` int array assigning each ray to a sensor.
    ``time_gates``  — ``(total_rays, 2)`` array of per-ray ``[t_min, t_max]``.
    """
    rng = rng or _DEFAULT_RNG
    z = _ci_to_z(ci)
    weights = np.array([s.weight for s in sensors])
    counts = allocate_ray_budget(weights, total_rays, rng=rng)

    rays_blocks: list[np.ndarray] = []
    ids_blocks: list[np.ndarray] = []
    gates_blocks: list[np.ndarray] = []
    for i, (sensor, k) in enumerate(zip(sensors, counts)):
        if k == 0:
            continue
        rays = np.zeros((k, 6))
        rays[:, list(SPATIAL_DIMS)] = sensor.origin_mean
        am, asg = sensor.mean[AZIMUTH_DIM], sensor.sigma[AZIMUTH_DIM]
        em, esg = sensor.mean[ELEVATION_DIM], sensor.sigma[ELEVATION_DIM]
        tm, tsg = sensor.mean[TIME_DIM], sensor.sigma[TIME_DIM]
        az_lo, az_hi = am - z * asg, am + z * asg
        el_lo, el_hi = em - z * esg, em + z * esg
        t_lo, t_hi = tm - z * tsg, tm + z * tsg

        az_span, el_span = az_hi - az_lo, el_hi - el_lo
        ratio = az_span / el_span if el_span > 0 else 1.0
        m = max(1, int(np.ceil(np.sqrt(k * ratio))))
        n = int(np.ceil(k / m))
        az_grid = az_lo + (np.arange(1, m + 1) / (m + 1)) * az_span
        el_grid = el_lo + (np.arange(1, n + 1) / (n + 1)) * el_span
        grid = np.array([(a, e) for a in az_grid for e in el_grid][:k])
        rays[:, AZIMUTH_DIM] = grid[:, 0]
        rays[:, ELEVATION_DIM] = grid[:, 1]
        rays[:, TIME_DIM] = tm

        rays_blocks.append(rays)
        ids_blocks.append(np.full(k, i, dtype=int))
        gates_blocks.append(np.tile([t_lo, t_hi], (k, 1)))

    return (
        np.concatenate(rays_blocks, axis=0),
        np.concatenate(ids_blocks, axis=0),
        np.concatenate(gates_blocks, axis=0),
    )


def design_to_segments(rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert design points into ``(origin, scene_point)`` segment endpoints.

    ``rays`` has shape ``(N, 6)``. Returns two ``(N, 3)`` arrays.
    """
    origin = rays[:, :3]
    scene = origin + rays[:, TIME_DIM:TIME_DIM + 1] * direction_vector_np(
        rays[:, AZIMUTH_DIM], rays[:, ELEVATION_DIM]
    )
    return origin, scene
