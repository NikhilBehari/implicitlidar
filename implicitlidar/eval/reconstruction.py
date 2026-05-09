"""Per-task reconstruction from ray-hit point sets.

Each of the three quantitative experiments turns a set of ray–surface
intersection points into a structured estimate (mesh, trajectory, or
detection list). This module collects those reconstruction operators:

* :func:`mesh_from_hits` — Delaunay-triangulated surface mesh from hit points
  (used by face_scanning).
* :func:`trajectory_from_hits` — spline-interpolated curve through ordered
  hit points (used by robot_tracking).
* :func:`box_detections_from_hits` — boolean detection list given hit points
  and a set of axis-aligned bounding boxes (used by warehouse_detection).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import trimesh
from scipy.interpolate import splev, splprep
from scipy.spatial import Delaunay

# Mesh reconstruction (used by face_scanning)


def mesh_from_hits(hits: np.ndarray) -> trimesh.Trimesh | None:
    """Reconstruct a surface mesh from ray hits via 2D Delaunay triangulation.

    The hits are projected onto their dominant plane (the two-axis subspace
    with maximal variance), Delaunay-triangulated there, and lifted back into
    3D using the original points.

    Returns ``None`` if there are not enough hits (≤ 3) to triangulate.
    """
    hits = np.asarray(hits, dtype=float)
    if hits.shape[0] < 4:
        return None
    centered = hits - hits.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Project onto the two axes of maximum variance.
    plane_axes = eigvecs[:, np.argsort(eigvals)[-2:]]
    coords_2d = centered @ plane_axes
    try:
        tri = Delaunay(coords_2d)
    except Exception:
        return None
    return trimesh.Trimesh(vertices=hits, faces=tri.simplices, process=False)


def mesh_chamfer_squared(
    rec_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh, *,
    n_samples: int = 10000, front_x_threshold: float | None = 0.01,
) -> float:
    """Symmetric squared Chamfer between mesh-sampled point clouds.

    Used for face mesh reconstruction in face_scanning and emitter_design. ``front_x_threshold``
    filters the GT samples to ``x > threshold`` so only the front of the head
    (the only side a frontal scanner can see) contributes; pass ``None`` to
    disable. Returns ``mean(d_gt²) + mean(d_rec²)``.
    """
    from scipy.spatial import cKDTree

    pts_rec = np.asarray(rec_mesh.sample(n_samples), dtype=float)
    pts_gt = np.asarray(gt_mesh.sample(n_samples), dtype=float)
    if front_x_threshold is not None:
        pts_gt = pts_gt[pts_gt[:, 0] > front_x_threshold]
    if len(pts_gt) == 0 or len(pts_rec) == 0:
        return 0.0
    d_gt_to_rec, _ = cKDTree(pts_rec).query(pts_gt)
    d_rec_to_gt, _ = cKDTree(pts_gt).query(pts_rec)
    return float(np.mean(d_gt_to_rec ** 2) + np.mean(d_rec_to_gt ** 2))


# Trajectory reconstruction (used by robot_tracking)


def trajectory_from_hits(
    hits: np.ndarray,
    *,
    n_samples: int = 200,
    smoothing: float = 0.0,
    spline_order: int = 3,
) -> np.ndarray:
    """Spline-interpolate an ordered curve through 3D hit points.

    ``hits`` is an ``(N, 3)`` array (assumed already in temporal order). Returns
    an ``(n_samples, 3)`` array sampled uniformly along the spline.

    Falls back to the input points if there are fewer than ``spline_order + 1``
    points (the minimum needed for a B-spline of the requested order).
    """
    hits = np.asarray(hits, dtype=float)
    if len(hits) < spline_order + 1:
        return hits
    tck, _ = splprep([hits[:, 0], hits[:, 1], hits[:, 2]], s=smoothing, k=spline_order)
    u = np.linspace(0.0, 1.0, n_samples)
    x, y, z = splev(u, tck)
    return np.stack([x, y, z], axis=1)


# Detection (used by warehouse_detection)


def box_detections_from_hits(
    hits: np.ndarray,
    boxes: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Return a per-box boolean mask of detections.

    A box is detected if at least one hit point lies within its axis-aligned
    bounding box. ``boxes`` is a sequence of ``(min_corner, max_corner)``
    tuples, each a length-3 array.
    """
    hits = np.asarray(hits, dtype=float)
    detected = np.zeros(len(boxes), dtype=bool)
    if len(hits) == 0:
        return detected
    for i, (lo, hi) in enumerate(boxes):
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        in_box = np.all((hits >= lo) & (hits <= hi), axis=1)
        detected[i] = bool(in_box.any())
    return detected
