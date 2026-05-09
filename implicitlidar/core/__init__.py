"""Core methodology.

The entire learned-density pipeline:

* :mod:`.design_space` — 6D design space, forward map, :class:`Bounds`.
* :mod:`.target_density` — surface × visibility target over one or many SDFs.
* :mod:`.flow` — autoregressive rational-quadratic spline flow.
* :mod:`.training` — reverse-KL training with optional visibility warmup.
* :mod:`.em_synthesis` — EM Gaussian-mixture sensor synthesis.
* :mod:`.constraints` — support truncation, sigma clamps, origin fixing.
* :mod:`.sampling` — physical ray emission from synthesized sensors.
* :mod:`.runner` — shared train / synthesize CLI runners.
"""

from .constraints import (
    FixSensorOrigin,
    SigmaClamp,
    SupportTruncation,
    apply_constraints,
    filter_samples_by_support,
)
from .design_space import (
    ANGULAR_DIMS,
    AZIMUTH_DIM,
    DIM_NAMES,
    ELEVATION_DIM,
    SPATIAL_DIMS,
    TIME_DIM,
    Bounds,
    direction_vector,
    direction_vector_np,
    scene_point,
    scene_point_np,
)
from .em_synthesis import (
    CSV_COLUMNS,
    Sensor,
    SensorMixture,
    fit_sensor_mixture,
    rows_to_sensors,
    sensors_to_rows,
)
from .flow import FlowModel, ShiftTransform, UniformDistribution, build_flow
from .runner import (
    build_inert_flow,
    periodic_checkpoint,
    synthesize_from_flow,
    train_experiment,
)
from .sampling import allocate_ray_budget, sample_rays
from .target_density import SceneSDF, TargetDensity, cache_kwargs_from_config, ray_visibility
from .training import linear_warmup, train_flow

__all__ = [
    # design_space
    "Bounds",
    "DIM_NAMES",
    "SPATIAL_DIMS",
    "ANGULAR_DIMS",
    "AZIMUTH_DIM",
    "ELEVATION_DIM",
    "TIME_DIM",
    "direction_vector",
    "direction_vector_np",
    "scene_point",
    "scene_point_np",
    # target_density
    "SceneSDF",
    "TargetDensity",
    "ray_visibility",
    "cache_kwargs_from_config",
    # flow
    "FlowModel",
    "ShiftTransform",
    "UniformDistribution",
    "build_flow",
    # training
    "train_flow",
    "linear_warmup",
    # em_synthesis
    "Sensor",
    "SensorMixture",
    "fit_sensor_mixture",
    "CSV_COLUMNS",
    "sensors_to_rows",
    "rows_to_sensors",
    # constraints
    "SupportTruncation",
    "SigmaClamp",
    "FixSensorOrigin",
    "apply_constraints",
    "filter_samples_by_support",
    # sampling
    "sample_rays",
    "allocate_ray_budget",
    # runner
    "train_experiment",
    "synthesize_from_flow",
    "periodic_checkpoint",
    "build_inert_flow",
]
