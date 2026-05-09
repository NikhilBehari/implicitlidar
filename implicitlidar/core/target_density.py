"""Task-driven target density ``p*(d)``.

The (unnormalized) target combines surface proximity and ray visibility over
a class of task scenes (signed distance functions). Supports a single mesh
or many, visibility on or off, and an optional separate occluder geometry.
Returns log-density.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import pytorch_volumetric as pv
import torch
import torch.nn as nn

from .design_space import (
    AZIMUTH_DIM,
    ELEVATION_DIM,
    TIME_DIM,
    direction_vector,
    scene_point,
)

# A scalar-valued SDF: takes (N, 3) points, returns (N,) signed distances.
# Some pytorch_volumetric SDFs return ``(values, gradients)`` — we wrap them
# uniformly via :class:`SceneSDF`.
SDFCallable = Callable[[torch.Tensor], torch.Tensor]


def cache_kwargs_from_config(scene_cfg: dict, device) -> dict:
    """Pull ``cache_*`` kwargs from a YAML ``scene`` block (sane defaults)."""
    return {
        "cache_resolution": scene_cfg.get("cache_resolution"),
        "cache_dir": scene_cfg.get("cache_dir"),
        "cache_padding": scene_cfg.get("cache_padding", 0.05),
        "cache_device": str(device),
    }


# SceneSDF — a uniform interface for one or many SDFs


class SceneSDF(nn.Module):
    """One or several SDFs behind a single iterable interface.

    Parameters
    ----------
    source
        A path to an OBJ file or a directory of OBJ files, or a sequence
        of pre-loaded SDF callables (used by procedural scenes).
    cache_resolution
        If set, wraps each mesh SDF in
        :class:`pytorch_volumetric.CachedSDF` — a precomputed voxel grid
        with GPU-side lookup. Faster on mesh-heavy scenes; should be
        meaningfully smaller than the surface-proximity ``sigma``
        (rule of thumb: ``< sigma / 5``). Default ``None`` uses the exact
        :class:`pytorch_volumetric.MeshSDF`.
    """

    def __init__(
        self,
        source: str | Iterable[SDFCallable | nn.Module],
        *,
        cache_resolution: float | None = None,
        cache_dir: str | None = None,
        cache_padding: float = 0.05,
        cache_device: str | torch.device = "cuda",
    ):
        super().__init__()
        self._cache_resolution = cache_resolution
        self._cache_dir = cache_dir
        self._cache_padding = float(cache_padding)
        self._cache_device = cache_device

        if isinstance(source, str):
            self.sdfs = nn.ModuleList(_load_sdfs_from_path(
                source,
                cache_resolution=cache_resolution,
                cache_dir=cache_dir,
                cache_padding=cache_padding,
                cache_device=cache_device,
            ))
        else:
            modules: list[nn.Module] = []
            for s in source:
                modules.append(s if isinstance(s, nn.Module) else _CallableSDF(s))
            self.sdfs = nn.ModuleList(modules)
        if len(self.sdfs) == 0:
            raise ValueError("SceneSDF requires at least one SDF.")

    # The following two methods make `SceneSDF` substitutable for a plain
    # callable SDF (returning (values, None) like pytorch_volumetric does).

    def forward(self, pts: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Return the pointwise mean SDF over all constituent SDFs."""
        return self.mean(pts), None

    def mean(self, pts: torch.Tensor) -> torch.Tensor:
        values = torch.stack([_sdf_values(sdf, pts) for sdf in self.sdfs], dim=0)
        return values.mean(dim=0)

    def __iter__(self):
        return iter(self.sdfs)

    def __len__(self) -> int:
        return len(self.sdfs)


# Ray visibility (the transmittance term)


def ray_visibility(
    design: torch.Tensor,
    occluder_sdf: SDFCallable | nn.Module,
    num_samples: int = 64,
    beta: float = 100.0,
) -> torch.Tensor:
    """Transmittance of each ray through an occluder geometry.

    Discretizes the visibility line integral with ``num_samples`` evenly spaced
    points between the ray origin and its scene point. The occlusion density
    at sample point ``p`` is ``β · sigmoid(-β · SDF(p))`` (high when ``p`` is
    inside the geometry), and the transmittance is the alpha-compositing
    product ``∏(1 − α_k)`` with ``α_k = 1 − exp(-β · Δ · occ_k)``.

    Returns a tensor of shape ``(N,)`` in ``(0, 1]``.
    """
    n = design.size(0)
    origin = design[:, :3]
    az = design[:, AZIMUTH_DIM:AZIMUTH_DIM + 1]
    el = design[:, ELEVATION_DIM:ELEVATION_DIM + 1]
    tau = design[:, TIME_DIM:TIME_DIM + 1]
    direction = direction_vector(az.squeeze(-1), el.squeeze(-1))  # (N, 3)

    ratios = torch.linspace(0.0, 1.0, steps=num_samples, device=design.device).view(1, num_samples, 1)
    distances = ratios * tau.unsqueeze(1)                          # (N, K, 1)
    pts = origin.unsqueeze(1) + distances * direction.unsqueeze(1) # (N, K, 3)
    sdf_vals = _sdf_values(occluder_sdf, pts.reshape(-1, 3)).view(n, num_samples)

    occupancy = torch.sigmoid(-beta * sdf_vals)
    delta = tau / num_samples                                      # (N, 1)
    alpha = 1.0 - torch.exp(-beta * delta * occupancy)             # (N, K)
    return torch.exp(torch.sum(torch.log1p(-alpha + 1e-10), dim=1))


# TargetDensity — the log-density consumed by flow training


class TargetDensity(nn.Module):
    """Log target density ``log p*(d)`` over the 6D design space.

    Parameters
    ----------
    scene_sdf
        A :class:`SceneSDF` of one or more target geometries. The surface
        proximity term is computed from each in turn and combined via
        ``logsumexp``.
    sigma
        Standard deviation of the surface-proximity Gaussian.
    use_visibility
        If ``False``, the visibility term ``T_i`` is dropped (occlusion-naive
        baseline used in the robot_tracking ``with_visibility.yaml`` ablation).
    visibility_num_samples, visibility_beta
        Discretization parameters of the visibility integral.
    visibility_weight
        Multiplier on the log-visibility term. Used together with
        :func:`~implicitlidar.core.training.linear_warmup` for curriculum
        schedules.
    occluder_sdf
        Optional separate :class:`SceneSDF` for the occluding geometry. If
        ``None``, each scene SDF occludes its own visibility term.
    eps
        Numerical stabilizer added inside ``log(visibility + eps)``.
    """

    @classmethod
    def from_config(
        cls,
        scene_sdf: SceneSDF,
        target_cfg: dict,
        *,
        occluder_sdf: SceneSDF | None = None,
    ) -> TargetDensity:
        """Build from a YAML ``target`` config block."""
        return cls(
            scene_sdf,
            sigma=target_cfg["sigma"],
            use_visibility=target_cfg["use_visibility"],
            visibility_num_samples=target_cfg["visibility_num_samples"],
            visibility_beta=target_cfg["visibility_beta"],
            visibility_weight=target_cfg["visibility_weight"],
            occluder_sdf=occluder_sdf,
        )

    def __init__(
        self,
        scene_sdf: SceneSDF,
        *,
        sigma: float = 0.1,
        use_visibility: bool = True,
        visibility_num_samples: int = 64,
        visibility_beta: float = 100.0,
        visibility_weight: float = 1.0,
        occluder_sdf: SceneSDF | None = None,
        eps: float = 1e-2,
    ):
        super().__init__()
        self.scene_sdf = scene_sdf
        self.occluder_sdf = occluder_sdf
        self.sigma = float(sigma)
        self.use_visibility = bool(use_visibility)
        self.visibility_num_samples = int(visibility_num_samples)
        self.visibility_beta = float(visibility_beta)
        # Mutable: training schedules tweak this in-place.
        self.visibility_weight = float(visibility_weight)
        self.eps = float(eps)

    def forward(self, design: torch.Tensor) -> torch.Tensor:
        """Log target density at each design point. Returns shape ``(N,)``."""
        s = scene_point(design)
        scene_iter: Sequence[nn.Module] = list(self.scene_sdf)
        occluder_iter: Sequence[nn.Module] | None
        if self.occluder_sdf is None:
            # Self-occlusion: each scene SDF occludes its own ray.
            occluder_iter = scene_iter
        elif len(self.occluder_sdf) == len(self.scene_sdf):
            occluder_iter = list(self.occluder_sdf)
        else:
            # Single shared occluder geometry.
            shared = self.occluder_sdf
            occluder_iter = [shared] * len(scene_iter)

        log_terms = []
        for scene_i, occ_i in zip(scene_iter, occluder_iter):
            sdf_at_s = _sdf_values(scene_i, s)
            log_proximity = -0.5 * (sdf_at_s ** 2) / (self.sigma ** 2)
            if self.use_visibility:
                vis = ray_visibility(
                    design,
                    occ_i,
                    num_samples=self.visibility_num_samples,
                    beta=self.visibility_beta,
                )
                log_proximity = log_proximity + self.visibility_weight * torch.log(vis + self.eps)
            log_terms.append(log_proximity)
        return torch.logsumexp(torch.stack(log_terms, dim=0), dim=0)


# Helpers


class _CallableSDF(nn.Module):
    """Adapt a plain callable into an `nn.Module` with the same interface."""

    def __init__(self, fn: SDFCallable):
        super().__init__()
        self._fn = fn

    def forward(self, pts: torch.Tensor):
        return self._fn(pts)


def _load_sdfs_from_path(
    source: str,
    *,
    cache_resolution: float | None = None,
    cache_dir: str | None = None,
    cache_padding: float = 0.05,
    cache_device: str | torch.device = "cuda",
) -> list[nn.Module]:
    import os
    if os.path.isdir(source):
        files = sorted(f for f in os.listdir(source) if f.endswith(".obj"))
        if not files:
            raise FileNotFoundError(f"No .obj files found in directory: {source}")
        paths = [os.path.join(source, f) for f in files]
    elif os.path.isfile(source):
        paths = [source]
    else:
        raise FileNotFoundError(f"Mesh path does not exist: {source}")
    return [
        _load_mesh_sdf(
            p,
            cache_resolution=cache_resolution,
            cache_dir=cache_dir,
            cache_padding=cache_padding,
            cache_device=cache_device,
        )
        for p in paths
    ]


def _load_mesh_sdf(
    filepath: str,
    *,
    cache_resolution: float | None = None,
    cache_dir: str | None = None,
    cache_padding: float = 0.05,
    cache_device: str | torch.device = "cuda",
) -> nn.Module:
    """Load an OBJ as a `pv.MeshSDF`, optionally voxel-cached on GPU."""
    return _MeshSDFWrapper(
        filepath,
        cache_resolution=cache_resolution,
        cache_dir=cache_dir,
        cache_padding=cache_padding,
        cache_device=cache_device,
    )


class _MeshSDFWrapper(nn.Module):
    """Thin nn.Module wrapper over `pv.MeshSDF`, optionally GPU-voxel-cached.

    With ``cache_resolution=None`` (default), the wrapper exposes
    :class:`pytorch_volumetric.MeshSDF` directly — exact closest-point queries
    via Open3D's CPU BVH.

    With ``cache_resolution`` set, the wrapper builds a
    :class:`pytorch_volumetric.CachedSDF` voxel grid covering the mesh's
    bounding box (inflated by ``cache_padding``) and serves all subsequent
    queries from the GPU-resident grid. The grid is persisted to ``cache_dir``
    under a deterministic name so subsequent runs reuse it without rebuilding.
    """

    def __init__(
        self,
        filepath: str,
        *,
        cache_resolution: float | None = None,
        cache_dir: str | None = None,
        cache_padding: float = 0.05,
        cache_device: str | torch.device = "cuda",
    ):
        super().__init__()
        import os
        mesh_obj = pv.MeshObjectFactory(filepath)
        gt_sdf = pv.MeshSDF(mesh_obj)

        if cache_resolution is None:
            self.mesh_sdf = gt_sdf
        else:
            cache_dir = cache_dir or os.path.join("outputs", "data", "sdf_cache")
            os.makedirs(cache_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(filepath))[0]
            cache_path = os.path.join(cache_dir, f"{stem}_r{cache_resolution}.pkl")
            self.mesh_sdf = pv.CachedSDF(
                object_name=stem,
                resolution=float(cache_resolution),
                range_per_dim=mesh_obj.bounding_box(padding=float(cache_padding)),
                gt_sdf=gt_sdf,
                device=str(cache_device),
                cache_path=cache_path,
            )

    def forward(self, pts: torch.Tensor):
        return self.mesh_sdf(pts)


def _sdf_values(sdf: SDFCallable | nn.Module, pts: torch.Tensor) -> torch.Tensor:
    """Call an SDF and return a 1-D tensor of values, ignoring any aux outputs."""
    out = sdf(pts)
    values = out[0] if isinstance(out, tuple) else out
    if values.dim() > 1 and values.size(-1) == 1:
        values = values.squeeze(-1)
    return values
