"""Mitsuba 3 transient render + per-ray surface-point recovery.

* :func:`render_transient` — patch the bundled scene template with the
  current emitter design and render one transient image (``(H, W, T, 3)``).
* :func:`hit_points_from_transient` — back-project each detector ray into
  pixel coordinates, find the temporal peak inside the per-emitter time
  window, and recover the surface point via :func:`bistatic_hit_point`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ._geometry import bistatic_hit_point, ray_to_pixel
from ._xml import Frustum, extract_camera_info, patch_lidar_xml


def render_transient(
    in_xml: Path,
    out_xml: Path,
    *,
    frustums: list[Frustum],
    mesh_path: str,
    resolution: int,
    temporal_bins: int,
    temporal_range: tuple[float, float],
    spp: int,
) -> tuple[np.ndarray, dict]:
    """Render one transient image and return it together with camera info.

    Returns ``(transient_array, camera_info)`` where ``transient_array`` has
    shape ``(H, W, T, 3)`` (one color channel per emitter, up to three) and
    ``camera_info`` is the perspective sensor's
    ``{origin, target, up, fov_deg, resolution}``.
    """
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant("llvm_ad_rgb")
    # Importing mitransient registers the transient_path integrator and
    # transient_hdr_film plugins used by the bundled scene template.
    import mitransient  # noqa: F401

    patched = patch_lidar_xml(
        in_xml, out_xml,
        frustums=frustums,
        object_path=mesh_path,
        resolution=resolution,
        temporal_bins=temporal_bins,
        temporal_range=temporal_range,
    )
    scene = mi.load_file(str(patched))
    _, transient = mi.render(scene, spp=spp)
    return np.asarray(transient), extract_camera_info(patched)


def hit_points_from_transient(
    transient_np: np.ndarray,
    *,
    detector_rays: list[dict],
    frustums: list[Frustum],
    cam_info: dict,
    temporal_bins: int,
    temporal_range: tuple[float, float],
    peak_threshold_frac: float = 0.05,
) -> np.ndarray:
    """Recover per-ray surface points from the transient image.

    Each detector ray is back-projected to a pixel; for every color channel
    (one per emitter) the temporal histogram is masked to the emitter's
    ``[t_mean ± 2σ]`` plus the detector gate, the peak bin gives the
    optical-path length, and :func:`bistatic_hit_point` solves for the
    surface point. Multiple emitters returning to the same pixel are
    averaged.
    """
    n_channels = min(len(frustums), 3)
    bin_width = (temporal_range[1] - temporal_range[0]) / temporal_bins
    centers = (np.arange(temporal_bins) + 0.5) * bin_width

    pixels = [
        ray_to_pixel((r["origin_x"], r["origin_y"], r["origin_z"]),
                     r["azimuth"], r["elevation"],
                     cam_info["origin"], cam_info["target"], cam_info["up"],
                     cam_info["fov_deg"], cam_info["resolution"])
        for r in detector_rays
    ]

    # Per-channel intensity threshold (5 % of the per-channel max across all rays).
    valid_pixels = [p for p in pixels if p is not None]
    thresholds = []
    for ch in range(n_channels):
        per_pixel_max = [transient_np[py, px, :, ch].max() for (px, py) in valid_pixels]
        thresholds.append(peak_threshold_frac * (max(per_pixel_max) if per_pixel_max else 0.0))

    out_pts = np.full((len(detector_rays), 3), np.nan)
    for i, (ray, pix) in enumerate(zip(detector_rays, pixels)):
        if pix is None:
            continue
        px, py = pix
        dd = (
            math.cos(ray["elevation"]) * math.cos(ray["azimuth"]),
            math.sin(ray["elevation"]),
            math.cos(ray["elevation"]) * math.sin(ray["azimuth"]),
        )
        per_channel_pts = []
        for ch in range(n_channels):
            hist = transient_np[py, px, :, ch]
            t_e_min = frustums[ch].time_mean - 2.0 * frustums[ch].time_sigma
            t_e_max = frustums[ch].time_mean + 2.0 * frustums[ch].time_sigma
            mask = (centers >= ray["t_min"] + t_e_min) & (centers <= ray["t_max"] + t_e_max)
            window = np.where(mask, hist, 0.0)
            if window.max() < thresholds[ch]:
                continue
            L = float(centers[int(window.argmax())])
            pt = bistatic_hit_point(frustums[ch].origin,
                                    (ray["origin_x"], ray["origin_y"], ray["origin_z"]),
                                    dd, L)
            if pt is not None:
                per_channel_pts.append(pt)
        if per_channel_pts:
            out_pts[i] = np.mean(np.array(per_channel_pts), axis=0)
    return out_pts
