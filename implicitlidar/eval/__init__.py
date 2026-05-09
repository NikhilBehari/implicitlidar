"""Evaluation: ray–scene intersection, reconstruction, metrics, baselines.

* :mod:`.intersections` — sphere-traced ray–scene intersection with HDF5
  caching of the per-scene hit records.
* :mod:`.reconstruction` — convert hit points into the structured estimate
  appropriate for each task (mesh, trajectory, detection mask).
* :mod:`.metrics` — Chamfer, Fréchet, miss rate, and raw bandwidth.
* :mod:`.bandwidth` — sensor-mixture and baseline bandwidth wrappers.
* :mod:`.baselines` — uniform and random ray emission strategies.
"""

from .bandwidth import GATE_Z_95, baseline_bandwidth_mbps, sensor_bandwidth_mbps
from .baselines import random_rays, uniform_rays
from .intersections import (
    RAY_HIT_DTYPE,
    hit_mask,
    hit_points,
    load_ray_hits,
    save_ray_hits,
    sphere_trace,
    trace_rays_against_scene,
)
from .metrics import (
    DEFAULT_BIN_WIDTH_PS,
    DEFAULT_BITS_PER_BIN,
    DEFAULT_SCAN_RATE_HZ,
    SPEED_OF_LIGHT,
    bandwidth_mbps,
    chamfer_distance,
    frechet_distance,
    mean_pm_2sem,
    miss_rate,
)
from .reconstruction import (
    box_detections_from_hits,
    mesh_chamfer_squared,
    mesh_from_hits,
    trajectory_from_hits,
)

__all__ = [
    # intersections
    "RAY_HIT_DTYPE",
    "sphere_trace",
    "trace_rays_against_scene",
    "save_ray_hits",
    "load_ray_hits",
    "hit_mask",
    "hit_points",
    # reconstruction
    "mesh_from_hits",
    "mesh_chamfer_squared",
    "trajectory_from_hits",
    "box_detections_from_hits",
    # metrics
    "chamfer_distance",
    "frechet_distance",
    "miss_rate",
    "bandwidth_mbps",
    "mean_pm_2sem",
    "DEFAULT_BIN_WIDTH_PS",
    "DEFAULT_BITS_PER_BIN",
    "DEFAULT_SCAN_RATE_HZ",
    "SPEED_OF_LIGHT",
    # bandwidth
    "sensor_bandwidth_mbps",
    "baseline_bandwidth_mbps",
    "GATE_Z_95",
    # baselines
    "uniform_rays",
    "random_rays",
]
