"""Unit tests for the core methodology.

Covers the design space, target density, normalizing-flow training,
EM sensor synthesis, constraints, and ray sampling.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from implicitlidar.core import (
    AZIMUTH_DIM,
    TIME_DIM,
    Bounds,
    SceneSDF,
    SensorMixture,
    SigmaClamp,
    SupportTruncation,
    TargetDensity,
    allocate_ray_budget,
    build_flow,
    direction_vector,
    ray_visibility,
    sample_rays,
    scene_point,
    train_flow,
)

# design_space


def test_direction_vector_unit_norm():
    az = torch.tensor([0.0, np.pi / 2, np.pi, -np.pi / 4])
    el = torch.tensor([0.0, 0.0, np.pi / 6, -np.pi / 3])
    v = direction_vector(az, el)
    norms = torch.linalg.norm(v, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_scene_point_at_origin_zero_tau():
    design = torch.zeros(3, 6)
    design[:, :3] = torch.tensor([[1.0, 2.0, 3.0]] * 3)
    s = scene_point(design)
    assert torch.allclose(s, design[:, :3])


def test_bounds_free_dims():
    bounds = Bounds.from_config({"low": [0, 0, 1, 0, -1, 0.1], "high": [1, 1, 1, 6.28, 0, 5]})
    assert bounds.free_dims == 5  # z is fixed
    assert bounds.fixed_indices == {2: 1.0}


def test_bounds_expand_reinjects_constants():
    bounds = Bounds.from_config({"low": [0, 0, 1, 0, -1, 0.1], "high": [1, 1, 1, 6.28, 0, 5]})
    free_z = torch.zeros(4, bounds.free_dims)
    full = bounds.expand(free_z)
    assert full.shape == (4, 6)
    assert torch.allclose(full[:, 2], torch.full((4,), 1.0))


# target_density


class _SphereSDF(nn.Module):
    def __init__(self, center, radius):
        super().__init__()
        self.register_buffer("c", torch.tensor(center, dtype=torch.float32))
        self.r = float(radius)

    def forward(self, pts):
        return torch.linalg.norm(pts - self.c, dim=-1) - self.r, None


def test_target_density_peaks_at_surface():
    """Without visibility the log-density is maximized when the ray endpoint is on the surface."""
    sphere = _SphereSDF([0.5, 0.0, 0.0], 0.2)
    target = TargetDensity(SceneSDF([sphere]), sigma=0.05, use_visibility=False)
    on_surface = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.3]])  # τ=0.3 → endpoint at (0.3,0,0), inside sphere edge
    far = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 5.0]])         # τ=5 → way past surface
    assert target(on_surface).item() > target(far).item()


def test_ray_visibility_in_unit_interval():
    sphere = _SphereSDF([0.5, 0.0, 0.0], 0.2)
    rays = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    vis = ray_visibility(rays, sphere, num_samples=32, beta=100.0)
    assert torch.all((vis >= 0) & (vis <= 1))


# flow + training (smallest possible end-to-end)


def test_flow_training_decreases_loss():
    sphere = _SphereSDF([0.5, 0.0, 0.0], 0.2)
    target = TargetDensity(SceneSDF([sphere]), sigma=0.05, use_visibility=False)
    bounds = Bounds.from_config({"low": [0, 0, 0, 0, -1, 0.1], "high": [0, 0, 0, 6.28, 1, 1.5]})
    flow_cfg = {
        "num_layers": 2, "circular_dims": [3],
        "spline": {"num_blocks": 1, "hidden_channels": 16, "num_bins": 8,
                   "tail_bound_multiplier": 1.0, "activation": nn.ReLU,
                   "dropout": 0.0, "permute_mask": True, "init_identity": True},
    }
    flow = build_flow(bounds, target, flow_cfg, device="cpu")

    initial_loss = []
    final_loss = []

    def cb(step, info):
        if step < 5:
            initial_loss.append(info["loss"])
        if step >= 95:
            final_loss.append(info["loss"])

    train_flow(flow, iterations=100, batch_size=128, learning_rate=1e-3,
               entropy_reg=0.5, callbacks=[cb], progress=False)
    assert np.mean(final_loss) < np.mean(initial_loss)


# em_synthesis


def test_em_recovers_clusters():
    rng = np.random.default_rng(0)
    centers = np.array([[0.5, -0.5, 1.0], [-0.5, 0.5, 0.5]])  # (az, el, τ)
    samples = np.zeros((400, 6))
    for k, c in enumerate(centers):
        n = 200
        slice_ = slice(k * n, (k + 1) * n)
        samples[slice_, AZIMUTH_DIM] = rng.normal(c[0], 0.05, size=n)
        samples[slice_, AZIMUTH_DIM + 1] = rng.normal(c[1], 0.05, size=n)
        samples[slice_, TIME_DIM] = rng.normal(c[2], 0.05, size=n)
    mix = SensorMixture(n_components=2, origin_mode="fixed", fixed_origin=(0.0, 0.0, 0.0),
                        circular_dims=(AZIMUTH_DIM,), max_iter=100, rng_seed=0).fit(samples)
    sensors = mix.sensors()
    recovered_az = sorted(s.mean[AZIMUTH_DIM] for s in sensors)
    expected_az = sorted(centers[:, 0].tolist())
    assert all(abs(r - e) < 0.1 for r, e in zip(recovered_az, expected_az))


def test_em_fixed_origin_keeps_origin_zero():
    rng = np.random.default_rng(0)
    samples = np.zeros((100, 6))
    samples[:, AZIMUTH_DIM] = rng.normal(0, 0.1, size=100)
    samples[:, AZIMUTH_DIM + 1] = rng.normal(0, 0.1, size=100)
    samples[:, TIME_DIM] = rng.normal(0.5, 0.05, size=100)
    mix = SensorMixture(n_components=2, origin_mode="fixed", fixed_origin=(1.0, 2.0, 3.0),
                        circular_dims=(AZIMUTH_DIM,), max_iter=20, rng_seed=0).fit(samples)
    for s in mix.sensors():
        assert np.allclose(s.origin_mean, [1.0, 2.0, 3.0])


# constraints


def test_support_truncation_clips_bounds():
    bounds = Bounds.from_config({"low": [0, 0, 0, 0, -1, 0.1], "high": [1, 1, 1, 6.28, 1, 5]})
    truncation = SupportTruncation(low=[None] * 6, high=[None, None, None, None, None, 2.0])
    new_bounds = truncation.apply(bounds)
    assert float(new_bounds.high[TIME_DIM].item()) == pytest.approx(2.0)


def test_sigma_clamp_caps_per_component_sigma():
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, size=(200, 6))
    samples[:, AZIMUTH_DIM] = rng.normal(0, 0.5, size=200)
    mix = SensorMixture(n_components=2, origin_mode="fixed", fixed_origin=(0, 0, 0),
                        circular_dims=(AZIMUTH_DIM,), max_iter=20, rng_seed=0).fit(samples)
    SigmaClamp({AZIMUTH_DIM: (0.0, 0.1)}).apply(mix)
    assert float(mix.sigmas[:, AZIMUTH_DIM].max()) <= 0.1 + 1e-9


# sampling


def test_allocate_ray_budget_sums_to_total():
    weights = np.array([0.1, 0.2, 0.7])
    counts = allocate_ray_budget(weights, total_rays=10_000, rng=np.random.default_rng(0))
    assert counts.sum() == 10_000
    # Largest weight should get the most rays (with high probability for n=10k).
    assert counts[2] > counts[0]


def test_sample_rays_returns_correct_total_and_gates():
    samples = np.zeros((100, 6))
    samples[:, AZIMUTH_DIM] = 0.1
    mix = SensorMixture(n_components=2, origin_mode="fixed", fixed_origin=(0, 0, 0),
                        circular_dims=(AZIMUTH_DIM,), max_iter=10, rng_seed=0).fit(samples)
    rays, sids, gates = sample_rays(mix.sensors(), total_rays=64, rng=np.random.default_rng(0))
    assert rays.shape == (64, 6)
    assert sids.shape == (64,)
    assert gates.shape == (64, 2)
    assert (gates[:, 1] >= gates[:, 0]).all()
