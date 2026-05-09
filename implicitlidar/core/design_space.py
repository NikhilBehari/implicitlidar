"""The 6D LiDAR design space.

A design point is ``d = (x, y, z, φ, ψ, τ)``: spatial origin, azimuth,
elevation, and one-way time-of-flight. This module exposes the ray direction
``v(φ, ψ)``, the forward map ``M(d) = x + τ·v(φ, ψ)`` to the observed scene
point, and a :class:`Bounds` dataclass that supports degenerate dimensions
(``low == high``) treated by the flow as fixed constants. Use this dimension
order everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Dimension order used everywhere a 6D design point is indexed.
# Reference these names (or the integer indices) when indexing.
DIM_NAMES: tuple[str, ...] = ("x", "y", "z", "az", "el", "t")
SPATIAL_DIMS: tuple[int, int, int] = (0, 1, 2)
ANGULAR_DIMS: tuple[int, int] = (3, 4)
AZIMUTH_DIM: int = 3
ELEVATION_DIM: int = 4
TIME_DIM: int = 5

# Azimuth is the only periodic dimension by default. Configs may override this
# (e.g. ``circular_dims: []`` for a fixed-azimuth scanning LiDAR).
DEFAULT_CIRCULAR_DIMS: tuple[int, ...] = (3,)


# Ray geometry


def direction_vector(azimuth: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
    """Unit ray direction ``v(φ, ψ) ∈ S²`` from azimuth and elevation.

    .. math::
        v(\\phi, \\psi) = (\\cos\\psi \\cos\\phi,\\ \\cos\\psi \\sin\\phi,\\ \\sin\\psi)^\\top

    Parameters
    ----------
    azimuth, elevation
        Tensors of identical shape. Azimuth in ``[0, 2π)``, elevation in
        ``[-π/2, π/2]``.

    Returns
    -------
    Tensor of shape ``(*input_shape, 3)`` of unit vectors.
    """
    cos_el = torch.cos(elevation)
    return torch.stack(
        [cos_el * torch.cos(azimuth), cos_el * torch.sin(azimuth), torch.sin(elevation)],
        dim=-1,
    )


def direction_vector_np(azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    """NumPy version of :func:`direction_vector` — used in EM and visualization."""
    cos_el = np.cos(elevation)
    return np.stack(
        [cos_el * np.cos(azimuth), cos_el * np.sin(azimuth), np.sin(elevation)],
        axis=-1,
    )


def scene_point(design: torch.Tensor) -> torch.Tensor:
    """Forward map ``M: D → S ⊂ ℝ³``.

    Returns the scene point an ideal co-located emitter–detector at design
    ``(x, az, el, τ)`` would observe: ``M(d) = x + τ · v(az, el)``.

    Parameters
    ----------
    design
        Tensor of shape ``(N, 6)`` in dimension order ``(x, y, z, az, el, τ)``.

    Returns
    -------
    Tensor of shape ``(N, 3)``.
    """
    origin = design[..., :3]
    az = design[..., AZIMUTH_DIM]
    el = design[..., ELEVATION_DIM]
    t = design[..., TIME_DIM]
    return origin + t.unsqueeze(-1) * direction_vector(az, el)


def scene_point_np(design: np.ndarray) -> np.ndarray:
    """NumPy version of :func:`scene_point`."""
    origin = design[..., :3]
    az = design[..., AZIMUTH_DIM]
    el = design[..., ELEVATION_DIM]
    t = design[..., TIME_DIM]
    return origin + t[..., None] * direction_vector_np(az, el)


# Bounds (the rectangular support of D)


@dataclass(frozen=True)
class Bounds:
    """Rectangular box ``[low_i, high_i]`` for each of the six dimensions.

    Dimensions where ``low == high`` are treated as *fixed constants* — they are
    not learned by the flow but reinjected during sampling.

    Use :meth:`from_config` to construct from a YAML config block.
    """

    low: torch.Tensor   # (6,)
    high: torch.Tensor  # (6,)

    @classmethod
    def from_config(
        cls,
        config: dict,
        device: torch.device | str = "cpu",
    ) -> Bounds:
        """Construct from a config block of the form ``{low: [...], high: [...]}``."""
        low = torch.as_tensor(config["low"], dtype=torch.float32, device=device)
        high = torch.as_tensor(config["high"], dtype=torch.float32, device=device)
        if low.shape != (6,) or high.shape != (6,):
            raise ValueError(
                f"Bounds must be 6-dimensional; got low={tuple(low.shape)}, high={tuple(high.shape)}"
            )
        if (high < low).any():
            raise ValueError(f"high must be ≥ low elementwise; got low={low.tolist()}, high={high.tolist()}")
        return cls(low=low, high=high)

    @property
    def free_mask(self) -> torch.Tensor:
        """Boolean (6,) tensor: ``True`` for dimensions the flow should learn."""
        return self.low != self.high

    @property
    def free_dims(self) -> int:
        return int(self.free_mask.sum().item())

    @property
    def fixed_indices(self) -> dict[int, float]:
        """Mapping ``{dim: constant_value}`` for the dimensions held fixed."""
        return {i: float(self.low[i].item()) for i in range(6) if not bool(self.free_mask[i].item())}

    def free_low(self) -> torch.Tensor:
        return self.low[self.free_mask]

    def free_high(self) -> torch.Tensor:
        return self.high[self.free_mask]

    def free_center(self) -> torch.Tensor:
        return (self.free_low() + self.free_high()) / 2

    def free_half_range(self) -> torch.Tensor:
        return (self.free_high() - self.free_low()) / 2

    def expand(self, free_z: torch.Tensor) -> torch.Tensor:
        """Reinject fixed-dimension constants into a sample of free dimensions.

        Given a tensor ``free_z`` of shape ``(N, free_dims)`` produced by the
        flow over the free subspace, returns a full ``(N, 6)`` design tensor
        with the constant values plugged into the fixed slots.
        """
        if free_z.dim() != 2 or free_z.size(1) != self.free_dims:
            raise ValueError(
                f"Expected (N, {self.free_dims}) free samples; got {tuple(free_z.shape)}"
            )
        n = free_z.size(0)
        full = torch.empty(n, 6, device=free_z.device, dtype=free_z.dtype)
        free_idx = 0
        for i in range(6):
            if bool(self.free_mask[i].item()):
                full[:, i] = free_z[:, free_idx]
                free_idx += 1
            else:
                full[:, i] = self.low[i]
        return full
