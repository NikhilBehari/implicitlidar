"""Ray–scene intersection via sphere tracing.

Given a batch of design points (rays) and a :class:`SceneSDF`, computes the
hit point of each ray with each scene by sphere tracing (ray-marching by SDF
distance, with a bisection refinement at sign changes).

A small HDF5-backed cache layer is provided so that the (expensive) ray-trace
results can be persisted across evaluation runs without re-tracing.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from ..core import (
    AZIMUTH_DIM,
    ELEVATION_DIM,
    TIME_DIM,
    SceneSDF,
    direction_vector_np,
)

# Sphere tracing (single SDF, single ray)


def sphere_trace(
    sdf: nn.Module,
    origin: torch.Tensor,
    direction: torch.Tensor,
    t_min: float,
    t_max: float,
    *,
    max_steps: int = 256,
    epsilon: float = 1e-4,
    refine_steps: int = 5,
) -> float | None:
    """Sphere-trace a single ray against a scalar SDF.

    Returns the parameter ``t ∈ [t_min, t_max]`` of the hit point, or ``None``
    if the ray misses (does not bracket a sign change in the search interval).

    Bisection on a sign change yields a refined intersection in a few extra
    SDF evaluations; the final point is returned only if it is within
    ``epsilon`` of the surface.
    """
    t = float(t_min)
    pt = origin + t * direction
    d_prev = _sdf_value(sdf, pt)
    if abs(d_prev) < epsilon:
        return t
    for _ in range(max_steps):
        t_next = t + d_prev
        if t_next < t_min or t_next > t_max:
            return None
        t_prev, d_prev_prev = t, d_prev
        t = t_next
        d_curr = _sdf_value(sdf, origin + t * direction)
        if abs(d_curr) < epsilon:
            return t
        if d_prev_prev * d_curr < 0:
            lo, hi = t_prev, t
            dlo = d_prev_prev
            for _ in range(refine_steps):
                tm = 0.5 * (lo + hi)
                dm = _sdf_value(sdf, origin + tm * direction)
                if dm * dlo > 0:
                    lo, dlo = tm, dm
                else:
                    hi = tm
            t_refined = 0.5 * (lo + hi)
            d_final = _sdf_value(sdf, origin + t_refined * direction)
            return t_refined if abs(d_final) < epsilon else None
        d_prev = d_curr
    return None


# Batch intersection


# Structured-array dtype used for serializing per-ray results to HDF5.
RAY_HIT_DTYPE = np.dtype([
    ("origin_x",  "f4"), ("origin_y",  "f4"), ("origin_z", "f4"),
    ("azimuth",   "f4"), ("elevation", "f4"),
    ("t_min",     "f4"), ("t_max",     "f4"),
    ("component", "i4"),
    ("hit_x",     "f4"), ("hit_y",     "f4"), ("hit_z",    "f4"),
])


def trace_rays_against_scene(
    rays: np.ndarray,
    scene: SceneSDF,
    *,
    sensor_ids: np.ndarray | None = None,
    time_gates: np.ndarray | None = None,
    device: torch.device | str = "cpu",
) -> list[np.ndarray]:
    """Trace every ray against every scene SDF in ``scene``.

    Parameters
    ----------
    rays
        ``(N, 6)`` array of design points ``(x, y, z, az, el, τ)``. The ``τ``
        coordinate is interpreted as the *expected* time-of-flight; the search
        interval ``[t_min, t_max]`` defaults to ``[0, τ]`` unless ``time_gates``
        is given.
    scene
        A :class:`SceneSDF`; this function returns one structured array per
        constituent SDF (i.e. one per scene).
    sensor_ids
        Optional ``(N,)`` integer array assigning each ray to a sensor. Stored
        in the structured-array record for downstream analysis.
    time_gates
        Optional ``(N, 2)`` array of ``[t_min, t_max]`` per ray. If absent,
        ``[0, τ]`` is used.

    Returns
    -------
    A list of ``(N,)`` structured arrays (one per scene), each with fields
    matching :data:`RAY_HIT_DTYPE`. Missed rays have NaN hit coordinates.
    """
    device = torch.device(device)
    n = rays.shape[0]
    if sensor_ids is None:
        sensor_ids = np.full(n, -1, dtype=np.int32)
    if time_gates is None:
        time_gates = np.stack([np.zeros(n), rays[:, TIME_DIM]], axis=1)

    az = rays[:, AZIMUTH_DIM]
    el = rays[:, ELEVATION_DIM]
    directions = direction_vector_np(az, el)

    out: list[np.ndarray] = []
    for sdf in scene:
        records = np.empty(n, dtype=RAY_HIT_DTYPE)
        for i in range(n):
            origin_t = torch.tensor(rays[i, :3], dtype=torch.float32, device=device)
            dir_t = torch.tensor(directions[i], dtype=torch.float32, device=device)
            t_hit = sphere_trace(sdf, origin_t, dir_t, float(time_gates[i, 0]), float(time_gates[i, 1]))
            if t_hit is None:
                hit = (np.nan, np.nan, np.nan)
            else:
                hit_pt = rays[i, :3] + t_hit * directions[i]
                hit = (float(hit_pt[0]), float(hit_pt[1]), float(hit_pt[2]))
            records[i] = (
                rays[i, 0], rays[i, 1], rays[i, 2],
                az[i], el[i],
                time_gates[i, 0], time_gates[i, 1],
                int(sensor_ids[i]),
                hit[0], hit[1], hit[2],
            )
        out.append(records)
    return out


# HDF5 caching


def save_ray_hits(path: str | Path, hits_per_scene: Sequence[np.ndarray], scene_keys: Sequence[str] | None = None) -> Path:
    """Persist per-scene ray-hit records to an HDF5 file.

    Each scene becomes a dataset under the root group, named ``scene_<i>`` by
    default or ``scene_keys[i]`` if provided.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for i, hits in enumerate(hits_per_scene):
            key = scene_keys[i] if scene_keys is not None else f"scene_{i:04d}"
            f.create_dataset(key, data=hits, compression="gzip")
    return path


def load_ray_hits(path: str | Path) -> dict[str, np.ndarray]:
    """Load per-scene ray-hit records from an HDF5 file written by :func:`save_ray_hits`."""
    out = {}
    with h5py.File(Path(path), "r") as f:
        for key in f.keys():
            out[key] = f[key][:]
    return out


# Helpers


def _sdf_value(sdf: nn.Module, pt: torch.Tensor) -> float:
    """Evaluate an SDF at a single point and return the scalar value."""
    out = sdf(pt.unsqueeze(0))
    val = out[0] if isinstance(out, tuple) else out
    if val.dim() > 1 and val.size(-1) == 1:
        val = val.squeeze(-1)
    return float(val.item())


def hit_mask(records: np.ndarray) -> np.ndarray:
    """Boolean mask of rays that hit (non-NaN ``hit_x``)."""
    return ~np.isnan(records["hit_x"])


def hit_points(records: np.ndarray) -> np.ndarray:
    """Extract a ``(K, 3)`` array of hit points (only rays that actually hit)."""
    mask = hit_mask(records)
    return np.stack([records["hit_x"][mask], records["hit_y"][mask], records["hit_z"][mask]], axis=1)
