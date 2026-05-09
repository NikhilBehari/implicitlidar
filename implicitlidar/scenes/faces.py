"""Face-mesh scenes for face_scanning.

* :func:`load_face_scene` loads a directory of pre-prepared face meshes as a
  :class:`SceneSDF` ready for training the implicit density.
* :func:`generate_face_meshes` samples random head meshes from the
  Basel Face Model 2009 PCA basis and writes them to disk as watertight
  OBJ files.

The generated meshes have a flat back (faces with ``z < 0`` in BFM
coordinates are clipped, the front surface is re-sampled on a uniform
2D grid, and the resulting open mesh is closed by a duplicated flat
back and connecting side walls), are made watertight via
:mod:`mesh2sdf`, and are oriented so the nose points along ``+x``
(matching the smartphone-LiDAR sensor placement at ``(3, 0, 0)``
looking along azimuth ``π``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import Delaunay

from ..core import SceneSDF


def load_face_scene(
    path: str | Path,
    *,
    cache_resolution: float | None = None,
    cache_dir: str | None = None,
    cache_padding: float = 0.05,
    cache_device: str = "cuda",
) -> SceneSDF:
    """Load all OBJ meshes in ``path`` (file or directory) as a :class:`SceneSDF`.

    See :class:`~implicitlidar.core.SceneSDF` for the meaning of the ``cache_*``
    arguments. Setting ``cache_resolution`` to a positive value is a major
    speedup for visibility-integral training over many face meshes.
    """
    return SceneSDF(
        str(path),
        cache_resolution=cache_resolution,
        cache_dir=cache_dir,
        cache_padding=cache_padding,
        cache_device=cache_device,
    )


# Basel Face Model dataset preparation


def generate_face_meshes(
    bfm_h5_path: str | Path,
    out_dir: str | Path,
    *,
    n_meshes: int,
    n_shape_components: int = 199,
    decimation_ratio: float = 0.10,
    resample_resolution: int = 100,
    sdf_grid_size: int = 128,
    sdf_mesh_scale: float = 0.8,
    seed: int = 0,
) -> list[Path]:
    """Sample ``n_meshes`` watertight head meshes from the Basel Face Model.

    Parameters
    ----------
    bfm_h5_path
        Path to ``model2019_fullHead.h5`` (or compatible) downloaded from
        the Basel Face Model website.
    out_dir
        Directory to write the generated meshes to. Created if missing.
    n_meshes
        Number of meshes to sample.
    n_shape_components
        Number of PCA components to retain when sampling shape coefficients
        (default 199, the full BFM 2009 shape basis).
    decimation_ratio
        Quadric decimation ratio applied to the raw BFM mesh before further
        processing (default 0.10).
    resample_resolution
        Grid resolution for the front-surface re-sampling step.
    sdf_grid_size, sdf_mesh_scale
        Parameters forwarded to :func:`mesh2sdf.compute` for the watertight
        conversion of the open front surface.
    seed
        Random seed for the mesh sampler.

    Returns
    -------
    The list of OBJ paths written to ``out_dir``.
    """
    import h5py

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    with h5py.File(bfm_h5_path, "r") as f:
        shape_mean = f["shape/model/mean"][:]
        shape_pca = f["shape/model/pcaBasis"][:]
        shape_var = f["shape/model/pcaVariance"][:]
        triangles = f["shape/representer/cells"][:].T

    n_shape = min(n_shape_components, shape_pca.shape[1])
    written: list[Path] = []
    for i in range(n_meshes):
        coeffs = rng.standard_normal(n_shape) * np.sqrt(shape_var[:n_shape])
        verts = (shape_mean + shape_pca[:, :n_shape] @ coeffs).reshape(-1, 3)

        verts, faces = _decimate(verts, triangles, decimation_ratio)
        verts = _normalize_extent(verts, target_extent=2.0)

        front_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        watertight = _close_with_flat_back(front_mesh, resolution=resample_resolution)
        watertight = _refine_watertight(watertight, mesh_scale=sdf_mesh_scale, grid_size=sdf_grid_size)

        # BFM forward axis is +z; remap so the nose points along +x.
        watertight.vertices = watertight.vertices[:, [2, 0, 1]]
        watertight.fix_normals()

        path = out / f"head_{i}.obj"
        watertight.export(path)
        written.append(path)
    return written


# Helpers


def _decimate(verts: np.ndarray, faces: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Quadric-decimate a mesh to ``ratio`` of its original face count."""
    if ratio >= 1.0:
        return verts, faces
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    target = max(4, int(len(faces) * ratio))
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def _normalize_extent(verts: np.ndarray, *, target_extent: float) -> np.ndarray:
    """Center at the bbox midpoint and scale so the largest axis spans ``target_extent``."""
    bbmin, bbmax = verts.min(axis=0), verts.max(axis=0)
    center = 0.5 * (bbmin + bbmax)
    extent = float((bbmax - bbmin).max())
    if extent <= 0:
        return verts - center
    return (verts - center) * (target_extent / extent)


def _close_with_flat_back(mesh: trimesh.Trimesh, *, resolution: int) -> trimesh.Trimesh:
    """Clip the back of the head, re-sample the front, and close the mesh.

    The front surface (``z >= 0``) is re-sampled on a regular ``(x, y)``
    grid via downward ray casting. The closed mesh is the front surface
    plus a duplicated flat back at ``z = 0`` and a side wall connecting
    the boundary loops.
    """
    front_verts, front_faces = _filter_front(mesh)
    front_mesh = trimesh.Trimesh(vertices=front_verts, faces=front_faces, process=False)
    sampled, sampled_faces = _resample_front(front_mesh, resolution=resolution)
    sampled_faces = _orient_faces_outward(sampled, sampled_faces)

    n_front = len(sampled)
    back_faces = sampled_faces[:, [2, 1, 0]] + n_front
    boundary_loop = _boundary_loop(sampled_faces)
    side_faces = _side_wall(boundary_loop, n_front)

    back_verts = sampled.copy()
    back_verts[:, 2] = 0.0

    closed = trimesh.Trimesh(
        vertices=np.vstack([sampled, back_verts]),
        faces=np.vstack([sampled_faces, back_faces, side_faces]),
        process=False,
    )
    closed.fix_normals()
    return closed


def _filter_front(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Keep only vertices with ``z >= 0`` and the faces that survive."""
    verts = mesh.vertices
    keep = verts[:, 2] >= 0
    remap = -np.ones(len(verts), dtype=int)
    remap[keep] = np.arange(int(keep.sum()))
    surviving = [remap[face] for face in mesh.faces if keep[face].all()]
    return verts[keep], np.asarray(surviving, dtype=int)


def _resample_front(mesh: trimesh.Trimesh, *, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Cast downward rays on a regular grid and Delaunay-triangulate the topmost hits."""
    verts = mesh.vertices
    xy_min = verts[:, :2].min(axis=0)
    xy_max = verts[:, :2].max(axis=0)
    xs = np.linspace(xy_min[0], xy_max[0], resolution)
    ys = np.linspace(xy_min[1], xy_max[1], resolution)
    gx, gy = np.meshgrid(xs, ys)
    z_above = float(verts[:, 2].max()) + 1.0
    origins = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z_above)])
    directions = np.tile([0.0, 0.0, -1.0], (origins.shape[0], 1))

    locations, ray_idx, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=directions
    )
    sampled = []
    for r in np.unique(ray_idx):
        hits = locations[ray_idx == r]
        sampled.append(hits[np.argmax(hits[:, 2])])
    sampled = np.asarray(sampled)
    faces = Delaunay(sampled[:, :2]).simplices
    return sampled, faces


def _orient_faces_outward(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Flip any face whose normal points along ``-z`` so all front normals are ``+z``."""
    out = []
    for face in faces:
        v0, v1, v2 = verts[face]
        normal = np.cross(v1 - v0, v2 - v0)
        out.append(face[[0, 2, 1]] if normal[2] < 0 else face)
    return np.asarray(out, dtype=int)


def _boundary_loop(faces: np.ndarray) -> list[int]:
    """Walk the open boundary edges of a triangle soup and return them as a vertex loop."""
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    unique, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    graph: dict[int, list[int]] = {}
    for a, b in boundary:
        a, b = int(a), int(b)
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)
    start = next(iter(graph))
    loop = [start]
    current, prev = start, None
    while True:
        next_v = next((n for n in graph[current] if n != prev), None)
        if next_v is None or next_v == start:
            break
        loop.append(next_v)
        prev, current = current, next_v
    return loop


def _side_wall(boundary_loop: list[int], n_front: int) -> np.ndarray:
    """Two triangles per boundary edge connecting the front loop to its back duplicate."""
    faces = []
    m = len(boundary_loop)
    for i in range(m):
        a = boundary_loop[i]
        b = boundary_loop[(i + 1) % m]
        faces.append([a, b, b + n_front])
        faces.append([a, b + n_front, a + n_front])
    return np.asarray(faces, dtype=int)


def _refine_watertight(mesh: trimesh.Trimesh, *, mesh_scale: float, grid_size: int) -> trimesh.Trimesh:
    """Run :func:`mesh2sdf.compute` to obtain a strictly watertight mesh."""
    import mesh2sdf

    verts = np.asarray(mesh.vertices)
    bbmin, bbmax = verts.min(axis=0), verts.max(axis=0)
    center = 0.5 * (bbmin + bbmax)
    scale = 2.0 * mesh_scale / float((bbmax - bbmin).max())
    scaled = (verts - center) * scale
    _sdf, fixed = mesh2sdf.compute(
        scaled, np.asarray(mesh.faces), grid_size, fix=True, level=2.0 / grid_size, return_mesh=True
    )
    fixed.vertices = np.asarray(fixed.vertices) / scale + center
    return trimesh.Trimesh(vertices=fixed.vertices, faces=np.asarray(fixed.faces), process=False)
