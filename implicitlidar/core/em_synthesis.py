"""Sensor synthesis via expectation–maximization.

Fits a Gaussian mixture in the 6D design space to samples drawn from the
trained implicit density. Each component is a physical sensor: its origin
marginal is the placement region, its angular marginal is the field of view,
its ``τ`` marginal is the time gate, and its mixture weight is the per-sensor
ray-budget allocation. A single :class:`SensorMixture` class covers fixed,
diagonal-origin, and full-origin variants via constructor parameters.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.stats import norm

from .design_space import (
    AZIMUTH_DIM,
    ELEVATION_DIM,
    SPATIAL_DIMS,
    TIME_DIM,
    direction_vector_np,
)

OriginMode = Literal["fixed", "diagonal", "full"]
"""How the spatial origin ``(x, y, z)`` of each sensor is modeled.

* ``"fixed"`` — every component shares a single user-supplied origin (e.g. a
  smartphone face scanner).
* ``"diagonal"`` — each component has independent ``(σ_x, σ_y, σ_z)``
  variances; useful when sensors must lie on independent axis-aligned bands.
* ``"full"`` — each component has a full ``3 × 3`` covariance, optionally
  constrained to a thin region (line/plane) via :attr:`max_origin_thickness`.
  Used for distributed systems (e.g. ceiling-mounted robot trackers).
"""


# Helpers (circular azimuth handling)


def _wrap_to_pi(diff: np.ndarray) -> np.ndarray:
    """Wrap an angle difference into ``(-π, π]``."""
    return (diff + np.pi) % (2 * np.pi) - np.pi


def _gaussian_pdf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    return (1.0 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -0.5 * ((x - mean) / sigma) ** 2
    )


def _circular_gaussian_pdf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    diff = _wrap_to_pi(x - mean)
    return _gaussian_pdf(diff, 0.0, sigma)


# Sensor specification (one row of the saved CSV)


@dataclass
class Sensor:
    """A single physical sensor extracted from a fitted mixture component.

    Fields are stored in the 6D dimension order
    ``(x, y, z, az, el, τ)``.

    * :attr:`origin_mean` — best-guess sensor location.
    * :attr:`origin_cov` — covariance over the placement region (zero matrix
      when the origin is held fixed).
    * :attr:`mean` — full 6D mean ``(μ_x, μ_y, μ_z, μ_az, μ_el, μ_τ)``.
    * :attr:`sigma` — full 6D marginal standard deviations.
    * :attr:`weight` — mixture weight ``π_g``.
    """

    origin_mean: np.ndarray   # (3,)
    origin_cov: np.ndarray    # (3, 3)
    mean: np.ndarray          # (6,) [x, y, z, az, el, τ]
    sigma: np.ndarray         # (6,)
    weight: float

    def fov(self, ci: float = 0.95) -> tuple[tuple[float, float], tuple[float, float]]:
        """``(azimuth_interval, elevation_interval)`` at confidence level ``ci``."""
        z = _ci_to_z(ci)
        az_mu, el_mu = self.mean[AZIMUTH_DIM], self.mean[ELEVATION_DIM]
        az_s, el_s = self.sigma[AZIMUTH_DIM], self.sigma[ELEVATION_DIM]
        return ((az_mu - z * az_s, az_mu + z * az_s),
                (el_mu - z * el_s, el_mu + z * el_s))

    def time_gate(self, ci: float = 0.95) -> tuple[float, float]:
        """``(τ_min, τ_max)`` time gate at confidence level ``ci``."""
        z = _ci_to_z(ci)
        return (self.mean[TIME_DIM] - z * self.sigma[TIME_DIM],
                self.mean[TIME_DIM] + z * self.sigma[TIME_DIM])


def _ci_to_z(ci: float) -> float:
    return float(norm.ppf(0.5 + ci / 2))


# Gaussian mixture


@dataclass
class SensorMixture:
    """Gaussian mixture over the 6D LiDAR design space.

    Configurations:

    * **Fixed-origin** (face scanning): ``origin_mode='fixed'`` with
      ``fixed_origin=(x, y, z)``.
    * **Distributed**, full-covariance origin (robot tracking):
      ``origin_mode='full'`` with optional ``max_origin_thickness`` to constrain
      the region to a thin slab (e.g. a ceiling plane).
    * **Single-ray scanning** (warehouse): ``origin_mode='fixed'`` with
      ``n_components=ray_budget`` and downstream code drawing one ray per
      component.
    """

    n_components: int
    origin_mode: OriginMode = "diagonal"
    fixed_origin: np.ndarray | None = None
    max_origin_thickness: float | None = None  # eigenvalue cap for 'full'
    initial_origin_sigma: float | None = None
    initial_param_sigma: float = 1.0
    circular_dims: tuple[int, ...] = (AZIMUTH_DIM,)
    max_iter: int = 200
    tol: float = 1e-6
    rng_seed: int | None = None

    # Fitted parameters (filled in by :meth:`fit`).
    pi: np.ndarray = field(init=False)              # (G,)
    means: np.ndarray = field(init=False)           # (G, 6)
    sigmas: np.ndarray = field(init=False)          # (G, 6) — marginal stds
    origin_covariances: np.ndarray = field(init=False)  # (G, 3, 3)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.rng_seed)
        self.pi = np.full(self.n_components, 1.0 / self.n_components)
        self.means = np.zeros((self.n_components, 6))
        self.sigmas = np.full((self.n_components, 6), self.initial_param_sigma)
        if self.initial_origin_sigma is not None:
            self.sigmas[:, list(SPATIAL_DIMS)] = self.initial_origin_sigma
        self.origin_covariances = np.stack(
            [np.diag(self.sigmas[k, list(SPATIAL_DIMS)] ** 2) for k in range(self.n_components)]
        )
        if self.fixed_origin is not None:
            origin = np.asarray(self.fixed_origin, dtype=float)
            self.means[:, list(SPATIAL_DIMS)] = origin

    # E and M steps

    def _component_pdf(self, X: np.ndarray, k: int) -> np.ndarray:
        # Dimensions with sigma == 0 are degenerate (e.g. an origin held fixed by
        # configuration). Their data values are constant and equal to the mean, so
        # they contribute a constant Dirac factor that drops out of responsibilities;
        # we skip them to avoid 0/0 in the Gaussian PDF.
        pdf = np.ones(X.shape[0])
        for j in range(6):
            if self.sigmas[k, j] <= 1e-12:
                continue
            if j in self.circular_dims:
                pdf *= _circular_gaussian_pdf(X[:, j], self.means[k, j], self.sigmas[k, j])
            else:
                pdf *= _gaussian_pdf(X[:, j], self.means[k, j], self.sigmas[k, j])
        return pdf

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        resp = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            resp[:, k] = self.pi[k] * self._component_pdf(X, k)
        return resp / (resp.sum(axis=1, keepdims=True) + 1e-10)

    def _m_step(self, X: np.ndarray, resp: np.ndarray) -> None:
        N = X.shape[0]
        Nk = resp.sum(axis=0)
        self.pi = Nk / N
        self.pi /= self.pi.sum()
        for k in range(self.n_components):
            self._update_origin(X, resp, Nk, k)
            self._update_sensor_dims(X, resp, Nk, k)

    def _update_origin(self, X: np.ndarray, resp: np.ndarray, Nk: np.ndarray, k: int) -> None:
        if self.origin_mode == "fixed":
            self.sigmas[k, list(SPATIAL_DIMS)] = 0.0
            self.origin_covariances[k] = np.zeros((3, 3))
            return
        if Nk[k] < 1e-8:
            return
        # Update mean.
        self.means[k, list(SPATIAL_DIMS)] = (
            (resp[:, k][:, None] * X[:, list(SPATIAL_DIMS)]).sum(axis=0) / (Nk[k] + 1e-10)
        )
        if self.origin_mode == "diagonal":
            for j in SPATIAL_DIMS:
                var = (resp[:, k] * (X[:, j] - self.means[k, j]) ** 2).sum() / (Nk[k] + 1e-10)
                self.sigmas[k, j] = float(np.sqrt(var))
            self.origin_covariances[k] = np.diag(self.sigmas[k, list(SPATIAL_DIMS)] ** 2)
        else:  # 'full'
            diffs = X[:, list(SPATIAL_DIMS)] - self.means[k, list(SPATIAL_DIMS)]
            cov = (resp[:, k][:, None] * diffs).T @ diffs / Nk[k] + 1e-6 * np.eye(3)
            if self.max_origin_thickness is not None:
                cov = _clamp_thin(cov, self.max_origin_thickness)
            self.origin_covariances[k] = cov
            self.sigmas[k, list(SPATIAL_DIMS)] = np.sqrt(np.maximum(np.diag(cov), 0.0))

    def _update_sensor_dims(self, X: np.ndarray, resp: np.ndarray, Nk: np.ndarray, k: int) -> None:
        for j in (AZIMUTH_DIM, ELEVATION_DIM, TIME_DIM):
            if j in self.circular_dims:
                sin_sum = (resp[:, k] * np.sin(X[:, j])).sum()
                cos_sum = (resp[:, k] * np.cos(X[:, j])).sum()
                self.means[k, j] = np.arctan2(sin_sum, cos_sum)
                diff = _wrap_to_pi(X[:, j] - self.means[k, j])
                self.sigmas[k, j] = float(np.sqrt((resp[:, k] * diff ** 2).sum() / (Nk[k] + 1e-10)))
            else:
                m = (resp[:, k] * X[:, j]).sum() / (Nk[k] + 1e-10)
                self.means[k, j] = m
                self.sigmas[k, j] = float(np.sqrt((resp[:, k] * (X[:, j] - m) ** 2).sum() / (Nk[k] + 1e-10)))

    # Fit / log-likelihood

    def log_likelihood(self, X: np.ndarray) -> float:
        ll = 0.0
        for i in range(X.shape[0]):
            row = X[i:i + 1]
            prob = sum(self.pi[k] * self._component_pdf(row, k)[0] for k in range(self.n_components))
            ll += float(np.log(prob + 1e-10))
        return ll

    def fit(self, X: np.ndarray, *, verbose: bool = False) -> SensorMixture:
        """Fit the mixture to ``X`` (shape ``(N, 6)``) by EM.

        ``X`` should be a NumPy array of design points sampled from the trained
        flow. The first three columns are the sensor origins; columns 3–5 are
        ``(az, el, τ)``.
        """
        if X.ndim != 2 or X.shape[1] != 6:
            raise ValueError(f"Expected (N, 6) data; got {X.shape}.")
        self._initialize(X)
        prev_ll = -np.inf
        for it in range(1, self.max_iter + 1):
            resp = self._e_step(X)
            self._m_step(X, resp)
            ll = self.log_likelihood(X)
            if verbose:
                print(f"[EM] iter {it:3d}  ll={ll:.4f}")
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return self

    def _initialize(self, X: np.ndarray) -> None:
        if self.fixed_origin is not None and self.origin_mode != "fixed":
            raise ValueError("fixed_origin requires origin_mode='fixed'.")
        # Random init from data range, then patch in the fixed origin if any.
        lo, hi = X.min(axis=0), X.max(axis=0)
        self.means = self._rng.uniform(lo, hi, size=(self.n_components, 6))
        if self.fixed_origin is not None:
            self.means[:, list(SPATIAL_DIMS)] = np.asarray(self.fixed_origin, dtype=float)

    # Predictions / sampling

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard-assign each row of ``X`` to its most-probable component."""
        return self._e_step(X).argmax(axis=1)

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw ``n`` samples ``(samples, component_indices)`` from the mixture."""
        ks = self._rng.choice(self.n_components, size=n, p=self.pi / self.pi.sum())
        out = np.zeros((n, 6))
        for i, k in enumerate(ks):
            origin_mean = self.means[k, list(SPATIAL_DIMS)]
            cov = self.origin_covariances[k]
            if np.allclose(cov, 0.0):
                origin_sample = origin_mean
            else:
                origin_sample = self._rng.multivariate_normal(origin_mean, cov)
            sensor_sample = self._rng.normal(
                loc=self.means[k, [AZIMUTH_DIM, ELEVATION_DIM, TIME_DIM]],
                scale=np.maximum(self.sigmas[k, [AZIMUTH_DIM, ELEVATION_DIM, TIME_DIM]], 1e-12),
            )
            out[i, list(SPATIAL_DIMS)] = origin_sample
            out[i, AZIMUTH_DIM] = sensor_sample[0]
            out[i, ELEVATION_DIM] = sensor_sample[1]
            out[i, TIME_DIM] = sensor_sample[2]
        return out, ks

    # Conversion to physical sensors

    def sensors(self) -> list[Sensor]:
        """Extract one :class:`Sensor` per fitted mixture component."""
        return [
            Sensor(
                origin_mean=self.means[k, list(SPATIAL_DIMS)].copy(),
                origin_cov=self.origin_covariances[k].copy(),
                mean=self.means[k].copy(),
                sigma=self.sigmas[k].copy(),
                weight=float(self.pi[k]),
            )
            for k in range(self.n_components)
        ]


def fit_sensor_mixture(
    samples: np.ndarray, n_components: int, synthesis_cfg: dict,
) -> list[Sensor]:
    """Fit a :class:`SensorMixture` from a YAML ``synthesis`` config block.

    Reads ``origin_mode``, ``fixed_origin``, ``max_origin_thickness``,
    ``em_max_iter``, and ``em_seed``; returns the synthesized sensors.
    """
    origin_mode = synthesis_cfg.get("origin_mode", "fixed")
    fixed_origin = (
        tuple(float(v) for v in synthesis_cfg["fixed_origin"])
        if origin_mode == "fixed" else None
    )
    max_thickness = synthesis_cfg.get("max_origin_thickness")
    mixture = SensorMixture(
        n_components=n_components,
        origin_mode=origin_mode,
        fixed_origin=fixed_origin,
        max_origin_thickness=float(max_thickness) if max_thickness is not None else None,
        circular_dims=(AZIMUTH_DIM,),
        max_iter=int(synthesis_cfg.get("em_max_iter", 200)),
        rng_seed=int(synthesis_cfg.get("em_seed", 0)),
    ).fit(samples)
    return mixture.sensors()


# Sensor I/O helpers


CSV_COLUMNS: tuple[str, ...] = (
    "origin_x", "origin_y", "origin_z",
    "azimuth_mean", "elevation_mean", "time_mean",
    "orig_cov_xx", "orig_cov_xy", "orig_cov_xz",
    "orig_cov_yy", "orig_cov_yz", "orig_cov_zz",
    "azimuth_sigma", "elevation_sigma", "time_sigma",
    "weight",
)


def sensors_to_rows(sensors: Iterable[Sensor]) -> list[list[float]]:
    """Serialize sensors to rows matching :data:`CSV_COLUMNS`."""
    rows = []
    for s in sensors:
        C = s.origin_cov
        rows.append([
            float(s.origin_mean[0]), float(s.origin_mean[1]), float(s.origin_mean[2]),
            float(s.mean[AZIMUTH_DIM]), float(s.mean[ELEVATION_DIM]), float(s.mean[TIME_DIM]),
            float(C[0, 0]), float(C[0, 1]), float(C[0, 2]),
            float(C[1, 1]), float(C[1, 2]),
            float(C[2, 2]),
            float(s.sigma[AZIMUTH_DIM]), float(s.sigma[ELEVATION_DIM]), float(s.sigma[TIME_DIM]),
            float(s.weight),
        ])
    return rows


def rows_to_sensors(rows: Iterable[list[float]]) -> list[Sensor]:
    """Inverse of :func:`sensors_to_rows`."""
    out = []
    for r in rows:
        origin = np.array([r[0], r[1], r[2]])
        cov = np.array([
            [r[6],  r[7],  r[8]],
            [r[7],  r[9],  r[10]],
            [r[8],  r[10], r[11]],
        ])
        mean = np.zeros(6)
        mean[list(SPATIAL_DIMS)] = origin
        mean[AZIMUTH_DIM]   = r[3]
        mean[ELEVATION_DIM] = r[4]
        mean[TIME_DIM]      = r[5]
        sigma = np.zeros(6)
        sigma[list(SPATIAL_DIMS)] = np.sqrt(np.maximum(np.diag(cov), 0.0))
        sigma[AZIMUTH_DIM]   = r[12]
        sigma[ELEVATION_DIM] = r[13]
        sigma[TIME_DIM]      = r[14]
        out.append(
            Sensor(origin_mean=origin, origin_cov=cov, mean=mean, sigma=sigma, weight=r[15])
        )
    return out


# Sensor-frame geometry helpers


def sensor_scene_points(origin: np.ndarray, sensor_params: np.ndarray) -> np.ndarray:
    """Forward map for sensor-only parameters ``(az, el, τ)`` from a fixed origin.

    ``origin`` may be shape ``(3,)`` (one origin for all rays) or ``(N, 3)``
    (per-ray origins). ``sensor_params`` has shape ``(N, 3)``.
    """
    sensor_params = np.atleast_2d(sensor_params)
    az, el, t = sensor_params[:, 0], sensor_params[:, 1], sensor_params[:, 2]
    return origin + t[:, None] * direction_vector_np(az, el)


def _clamp_thin(cov: np.ndarray, max_thickness: float) -> np.ndarray:
    """Cap the two smallest eigenvalues of ``cov`` at ``max_thickness²``.

    This collapses the placement covariance toward a thin region (e.g. a line
    or plane). Used when sensors must lie on a constrained surface like a
    ceiling.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)
    threshold = max_thickness ** 2
    eigvals[order[0]] = min(eigvals[order[0]], threshold)
    eigvals[order[1]] = min(eigvals[order[1]], threshold)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T
