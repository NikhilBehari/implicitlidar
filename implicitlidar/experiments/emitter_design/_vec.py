"""3-vector math primitives shared by ``_xml`` and ``_geometry``."""

from __future__ import annotations

import math


def normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n, v[2] / n)


def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def sph_to_cart(az_deg: float, el_deg: float) -> tuple[float, float, float]:
    a, e = math.radians(az_deg), math.radians(el_deg)
    return math.cos(e) * math.cos(a), math.sin(e), math.cos(e) * math.sin(a)
