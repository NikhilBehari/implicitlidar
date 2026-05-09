"""Evaluation metrics for the three reconstruction tasks.

* :func:`chamfer_distance` — symmetric Chamfer distance between two point sets
  (face_scanning: face mesh reconstruction).
* :func:`frechet_distance` — discrete Fréchet distance between two ordered
  curves (robot_tracking: end-effector trajectory).
* :func:`miss_rate` — fraction of ground-truth boxes not detected by any ray
  (warehouse_detection: warehouse object detection).
* :func:`bandwidth_mbps` — data throughput in Mbps for a sensor configuration,
  given a per-sensor time gate.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# Reconstruction-quality metrics


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer distance between two point clouds.

    .. math::

        d_{\\mathrm{Chamfer}}(A, B) =
            \\frac{1}{|A|}\\sum_{x\\in A}\\min_{y\\in B}\\|x - y\\| +
            \\frac{1}{|B|}\\sum_{y\\in B}\\min_{x\\in A}\\|y - x\\|

    Inputs are ``(*, 3)`` arrays. Returns a scalar.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    d_ab, _ = tree_b.query(a, k=1)
    d_ba, _ = tree_a.query(b, k=1)
    return float(d_ab.mean() + d_ba.mean())


def frechet_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Discrete Fréchet distance between two ordered curves (Alt & Godau 1995).

    Inputs are ordered ``(N, D)`` and ``(M, D)`` arrays. Returns a scalar in
    the same units as the inputs.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    n, m = len(p), len(q)
    if n == 0 or m == 0:
        return float("inf")
    ca = np.full((n, m), -1.0)

    def _c(i: int, j: int) -> float:
        if ca[i, j] != -1:
            return ca[i, j]
        d = float(np.linalg.norm(p[i] - q[j]))
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(i - 1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(0, j - 1), d)
        else:
            ca[i, j] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), d)
        return ca[i, j]

    return _c(n - 1, m - 1)


# Detection


def miss_rate(detected: np.ndarray, n_ground_truth: int) -> float:
    """Fraction of ground-truth objects that were not detected.

    ``detected`` is a boolean array of length ``n_ground_truth`` (one entry per
    object). Returns a value in ``[0, 1]``.
    """
    detected = np.asarray(detected, dtype=bool)
    if n_ground_truth == 0:
        return 0.0
    return float(1.0 - detected.sum() / n_ground_truth)


# System-cost metrics


# Speed of light (m/s).
SPEED_OF_LIGHT = 3.0e8

# SPAD-system constants used in the bandwidth derivation.
DEFAULT_BIN_WIDTH_PS = 100.0
DEFAULT_BITS_PER_BIN = 40
DEFAULT_SCAN_RATE_HZ = 10.0


def bandwidth_mbps(
    rays_per_sensor: np.ndarray,
    time_gate_per_sensor_m: np.ndarray,
    *,
    scaling_factor_m: float = 1.0,
    bin_width_ps: float = DEFAULT_BIN_WIDTH_PS,
    bits_per_bin: int = DEFAULT_BITS_PER_BIN,
    scan_rate_hz: float = DEFAULT_SCAN_RATE_HZ,
) -> float:
    """Required photon-stream bandwidth for a sensor configuration, in Mbps.

    Each sensor records a per-ray histogram covering its time gate. The window
    width in meters is converted to a *two-way* time of flight in picoseconds,
    divided by the bin width to get the number of histogram bins per ray, and
    multiplied by ``bits_per_bin`` to get the per-ray data volume.

    Parameters
    ----------
    rays_per_sensor
        ``(G,)`` integer array of rays allocated to each sensor.
    time_gate_per_sensor_m
        ``(G,)`` array of per-sensor time-gate widths in meters of one-way
        optical-path-length (matching the ``τ`` coordinate of the design space).
    scaling_factor_m
        Meters per unit of ``τ``. Defaults to 1.0 (i.e. ``τ`` is already in
        meters); set to a different value if the design space rescales depth.
    bin_width_ps, bits_per_bin, scan_rate_hz
        SPAD-detector and acquisition parameters (default 100 ps bin width,
        40 bits per bin, 10 Hz scan rate).

    Returns
    -------
    Megabits per second.
    """
    rays_per_sensor = np.asarray(rays_per_sensor, dtype=float)
    delta_m = np.asarray(time_gate_per_sensor_m, dtype=float) * scaling_factor_m
    # Two-way ToF in picoseconds (the histogram covers the full round trip).
    tof_ps = 2.0 * delta_m * 1e12 / SPEED_OF_LIGHT
    bins_per_ray = np.ceil(tof_ps / bin_width_ps)
    bits_per_scan = float((rays_per_sensor * bins_per_ray).sum() * bits_per_bin)
    return bits_per_scan * scan_rate_hz / 1e6


def mean_pm_2sem(values: np.ndarray) -> tuple[float, float]:
    """Return ``(mean, 2·SEM)`` for a 1-D array of metric values across trials."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, 2 * sem
