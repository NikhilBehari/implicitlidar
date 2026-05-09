"""Detector ray geometry + bistatic surface-point recovery.

* :func:`ray_to_pixel` — project a detector ray into the perspective
  sensor's pixel coordinates (returns ``None`` when out of frustum).
* :func:`bistatic_hit_point` — solve the bistatic geometry equation for
  the surface point along a detector ray given its measured
  optical-path-length.
"""

from __future__ import annotations

import math

from ._vec import cross, normalize


def ray_to_pixel(
    ray_origin: tuple[float, float, float],
    azimuth: float, elevation: float,
    cam_origin: tuple[float, float, float],
    cam_target: tuple[float, float, float],
    cam_up: tuple[float, float, float],
    fov_deg: float, resolution: int,
) -> tuple[int, int] | None:
    """Project a detector ray into a sensor pixel coordinate.

    Returns ``None`` if the ray does not pass through the camera frustum
    (or starts at a different origin than the perspective sensor).
    """
    if any(abs(ray_origin[i] - cam_origin[i]) > 1e-6 for i in range(3)):
        return None
    d = normalize((math.cos(elevation) * math.cos(azimuth),
                    math.sin(elevation),
                    math.cos(elevation) * math.sin(azimuth)))
    f = normalize(tuple(cam_target[i] - cam_origin[i] for i in range(3)))
    r = normalize(cross(f, cam_up))
    u = cross(r, f)
    cx = sum(d[i] * r[i] for i in range(3))
    cy = sum(d[i] * u[i] for i in range(3))
    cz = sum(d[i] * f[i] for i in range(3))
    if cz <= 0:
        return None
    t = math.tan(math.radians(fov_deg / 2)) * cz
    ndc_x, ndc_y = cx / t, cy / t
    if abs(ndc_x) > 1 or abs(ndc_y) > 1:
        return None
    px = int((ndc_x * 0.5 + 0.5) * resolution)
    py = int((0.5 - ndc_y * 0.5) * resolution)
    if 0 <= px < resolution and 0 <= py < resolution:
        return px, py
    return None


def bistatic_hit_point(
    emitter_origin: tuple[float, float, float],
    detector_origin: tuple[float, float, float],
    detector_direction: tuple[float, float, float],
    optical_path_length: float,
) -> tuple[float, float, float] | None:
    """Solve for the surface point along a detector ray under bistatic geometry.

    Given the emitter origin ``Oₑ``, the detector origin ``O_d`` and ray
    direction ``d̂``, and the total optical path length
    ``L = ‖P − Oₑ‖ + ‖P − O_d‖`` measured by the transient sensor, returns
    ``P = O_d + t·d̂`` where

    .. math::

        t = \\frac{L^2 - \\|O_d - O_e\\|^2}{2\\,(L + (O_d - O_e) \\cdot d̂)}.
    """
    Dx = detector_origin[0] - emitter_origin[0]
    Dy = detector_origin[1] - emitter_origin[1]
    Dz = detector_origin[2] - emitter_origin[2]
    b = Dx * detector_direction[0] + Dy * detector_direction[1] + Dz * detector_direction[2]
    D2 = Dx * Dx + Dy * Dy + Dz * Dz
    den = 2.0 * (optical_path_length + b)
    if den <= 0:
        return None
    t = (optical_path_length ** 2 - D2) / den
    if t < 0:
        return None
    return (
        detector_origin[0] + t * detector_direction[0],
        detector_origin[1] + t * detector_direction[1],
        detector_origin[2] + t * detector_direction[2],
    )
