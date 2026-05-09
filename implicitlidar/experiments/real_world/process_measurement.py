"""Convert a real single-photon transient measurement into a 3D point cloud.

Pipeline:

1. **Load transient** — read the HDF5 ``.mat`` file produced by the SPAD
   capture system; the dataset is a 3-D array of shape ``(H, W, T)``
   (height × width × temporal-bin index).
2. **Peak-find depth** — for each pixel, identify the bin index of the
   transient peak after the laser-pulse onset ``t0``; this is the time of
   flight, gated to ``[t_min, t_max]`` to suppress spurious returns.
3. **Crop ROI** — restrict to the rectangular pixel region containing the
   subject (the rest is background or sensor edge).
4. **Build point cloud** — convert each ``(pixel, depth)`` triple into a 3D
   point in the SPAD's coordinate frame.

The resulting cloud is saved as an ``.npy`` file and is the input to
:mod:`.reconstruct`.

Usage::

    python -m implicitlidar.experiments.real_world.process_measurement \\
        --transient outputs/data/real_world/depth_transient.mat \\
        --out outputs/data/real_world/point_cloud.npy \\
        --t0 3925 --t-min 4000 --t-max 4350 \\
        --y0 60 --y1 210 --x0 78 --x1 178 \\
        --depth-scale 0.009
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def load_transient(path: str | Path, dataset_name: str = "transient") -> np.ndarray:
    """Load a transient measurement from an HDF5 ``.mat`` file.

    Falls back to the first 3-D dataset in the file if ``dataset_name`` is
    not present, matching the layout produced by the SPAD acquisition rig.
    Returns a ``(T, H, W)`` array; the row and column axes are reversed
    (``[:, ::-1, ::-1]``) to match the camera's image-coordinate convention.
    """
    with h5py.File(str(path), "r") as f:
        if dataset_name in f:
            data = f[dataset_name][()]
        else:
            for obj in f.values():
                if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                    data = obj[()]
                    break
            else:
                raise KeyError(f"No 3D transient dataset found in {path}")
    return data[:, ::-1, ::-1]


def compute_depth(
    transient: np.ndarray,
    *,
    t0: int,
    t_min: int | None = None,
    t_max: int | None = None,
) -> np.ndarray:
    """Per-pixel time-of-flight bin index, peak-found after ``t0``.

    Returns an ``(H, W)`` integer array; pixels whose peak falls outside
    ``[t_min, t_max]`` (when given) are zeroed.
    """
    sub = transient[t0:]
    peaks = sub.argmax(axis=0) + t0
    if t_min is not None:
        peaks = np.where(peaks < t_min, 0, peaks)
    if t_max is not None:
        peaks = np.where(peaks > t_max, 0, peaks)
    return peaks


def extract_roi(depth: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Crop the per-pixel depth map to the ``[y0:y1, x0:x1]`` window."""
    return depth[y0:y1, x0:x1]


def compute_point_cloud(roi: np.ndarray, *, depth_scale: float | None) -> np.ndarray:
    """Convert a ROI depth map into an ``(N, 3)`` point cloud.

    The pixel grid is normalized to span ``[-1, 1]`` along its longer side,
    and the depth axis is rescaled by ``depth_scale`` (meters per bin) and
    flipped so that closer points have larger ``x`` (matching the SPAD
    geometry the experiment uses).
    """
    h, w = roi.shape
    spatial_scale = 2.0 / h
    x_lin = (np.arange(w) - (w - 1) / 2) * spatial_scale
    y_lin = (np.arange(h) - (h - 1) / 2) * spatial_scale
    Xg, Yg = np.meshgrid(x_lin, y_lin)
    Zg = roi
    mask = Zg > 0
    raw = Zg[mask]
    if depth_scale is not None:
        x = (raw.max() - raw) * depth_scale
    else:
        x = raw.astype(float)
    y = Xg[mask]
    z = -Yg[mask]
    return np.column_stack((x, y, z))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transient", type=Path, required=True,
                        help="Path to the SPAD transient HDF5 .mat file.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output path for the processed point cloud (.npy).")
    parser.add_argument("--dataset-name", type=str, default="transient")
    parser.add_argument("--t0", type=int, required=True,
                        help="Laser-pulse onset bin index.")
    parser.add_argument("--t-min", type=int, default=None,
                        help="Inclusive lower bound on accepted peak bin (suppresses early returns).")
    parser.add_argument("--t-max", type=int, default=None,
                        help="Inclusive upper bound on accepted peak bin (suppresses late returns).")
    parser.add_argument("--y0", type=int, required=True)
    parser.add_argument("--y1", type=int, required=True)
    parser.add_argument("--x0", type=int, required=True)
    parser.add_argument("--x1", type=int, required=True)
    parser.add_argument("--depth-scale", type=float, default=None,
                        help="Meters per temporal bin. Omit to keep depth in raw bin units.")
    args = parser.parse_args()

    print(f"[process] loading transient from {args.transient}")
    trans = load_transient(args.transient, args.dataset_name)
    print(f"[process] transient shape: {trans.shape}")

    depth = compute_depth(trans, t0=args.t0, t_min=args.t_min, t_max=args.t_max)
    roi = extract_roi(depth, args.y0, args.y1, args.x0, args.x1)
    cloud = compute_point_cloud(roi, depth_scale=args.depth_scale)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, cloud)
    print(f"[process] {len(cloud)} points -> {args.out}")


if __name__ == "__main__":
    main()
