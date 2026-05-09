"""Procedural warehouse scenes for warehouse_detection (object detection).

Generates rows of multi-shelf rack scenes with parametrized shelf heights and
box sizes. Each scene
is exported as a single OBJ mesh suitable for SDF queries.

Workflow
--------

* :class:`MultiShelfGenerator` — procedurally builds and exports ``n_scenes``
  warehouse meshes given a list of shelf-row parameters.
* :func:`load_warehouse_scene` — load all generated OBJs in a directory as a
  :class:`SceneSDF` for training.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from ..core import SceneSDF


def load_warehouse_scene(
    path: str | Path,
    *,
    cache_resolution: float | None = None,
    cache_dir: str | None = None,
    cache_padding: float = 0.05,
    cache_device: str = "cuda",
) -> SceneSDF:
    """Load all warehouse OBJ meshes in ``path`` as a :class:`SceneSDF`.

    See :class:`~implicitlidar.core.SceneSDF` for the ``cache_*`` arguments.
    """
    return SceneSDF(
        str(path),
        cache_resolution=cache_resolution,
        cache_dir=cache_dir,
        cache_padding=cache_padding,
        cache_device=cache_device,
    )


def load_box_annotations(path: str | Path) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Load per-scene box bounding boxes saved by :meth:`MultiShelfGenerator.generate_dataset`.

    Returns a list of ``n_scenes`` entries, each a list of ``(min_corner, max_corner)``
    tuples in world coordinates.
    """
    payload = json.loads(Path(path).read_text())
    return [
        [(np.asarray(box["min"]), np.asarray(box["max"])) for box in scene["boxes"]]
        for scene in payload["scenes"]
    ]


# Per-row shelf parameters


@dataclass
class ShelfParams:
    """Parameters describing one row of a warehouse rack.

    A row consists of ``len(spacing) + 1`` shelves stacked vertically with the
    given ``spacing`` between successive shelf surfaces. Each shelf carries one
    box whose height is sampled uniformly from the corresponding entry of
    ``box_height_ranges``.
    """

    spacing: list[float]
    shelf_dim: tuple[float, float]
    box_height_ranges: list[tuple[float, float]]
    bottom_offset: float = 0.2
    box_xy_dim: tuple[float, float] | None = None

    def __post_init__(self):
        if len(self.box_height_ranges) != len(self.spacing) + 1:
            raise ValueError(
                "box_height_ranges must have exactly one more entry than spacing "
                f"(got {len(self.box_height_ranges)} vs {len(self.spacing) + 1})"
            )


# Default warehouse layout used by warehouse_detection — four rows of varying
# height profiles.
DEFAULT_LAYOUT: list[ShelfParams] = [
    ShelfParams(
        spacing=[0.4, 0.4],
        shelf_dim=(1.0, 0.3),
        box_height_ranges=[(0.1, 0.375)] * 3,
        bottom_offset=0.0,
        box_xy_dim=(0.75, 0.27),
    ),
    ShelfParams(
        spacing=[],
        shelf_dim=(1.0, 0.3),
        box_height_ranges=[(0.1, 0.3)],
        bottom_offset=0.0,
        box_xy_dim=(0.75, 0.27),
    ),
    ShelfParams(
        spacing=[0.3],
        shelf_dim=(1.0, 0.3),
        box_height_ranges=[(0.1, 0.3), (0.1, 0.3)],
        bottom_offset=0.0,
        box_xy_dim=(0.75, 0.27),
    ),
    ShelfParams(
        spacing=[0.4, 0.4],
        shelf_dim=(1.0, 0.3),
        box_height_ranges=[(0.05, 0.15)] * 3,
        bottom_offset=0.0,
        box_xy_dim=(0.75, 0.27),
    ),
]


# Generator


@dataclass
class MultiShelfGenerator:
    """Procedural generator for multi-row warehouse rack scenes."""

    layout: Sequence[ShelfParams] = field(default_factory=lambda: list(DEFAULT_LAYOUT))
    shelf_spacing_x: float = 0.2
    pole_thickness: float = 0.02
    rod_thickness: float = 0.01
    rod_bar_z: float = -2.0  # z-height of the back-side horizontal connector bar
    sdf_grid_size: int = 128
    sdf_normalization_scale: float = 0.8

    def generate_dataset(
        self,
        out_dir: str | Path,
        *,
        n_scenes: int,
        seed: int = 0,
        split_targets_and_occluders: bool = True,
    ) -> dict[str, list[Path]]:
        """Generate ``n_scenes`` warehouse meshes and write them as OBJ files.

        When ``split_targets_and_occluders`` is true, two parallel files are
        written for every scene:

        * ``<out_dir>/target/scene_<i>.obj`` — full scene (shelves *with* boxes).
          The boxes are the detection targets the implicit density should focus on.
        * ``<out_dir>/occluder/scene_<i>.obj`` — same scene with boxes removed.
          Used as the visibility occluder (rays passing through shelf metal are
          penalized).

        With the flag off, a single ``<out_dir>/scene_<i>.obj`` is written.

        Mesh post-processing fills tiny topology holes via :mod:`mesh2sdf` so
        the resulting meshes have well-defined signed distances everywhere.

        Returns
        -------
        Mapping ``{key: [paths]}`` listing every generated file. Keys are
        ``"target"`` and ``"occluder"`` in split mode, ``"scene"`` otherwise.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        random.seed(seed)
        np.random.seed(seed)
        written: dict[str, list[Path]] = {"target": [], "occluder": [], "scene": []}

        if split_targets_and_occluders:
            target_dir = out / "target"
            occluder_dir = out / "occluder"
            target_dir.mkdir(parents=True, exist_ok=True)
            occluder_dir.mkdir(parents=True, exist_ok=True)

        # Per-scene list of box AABBs; written alongside the OBJs as `boxes.json`
        # for downstream detection metrics (the boxes are the detection targets).
        scenes_metadata: list[dict] = []

        for i in range(n_scenes):
            target_mesh, occluder_mesh, box_aabbs = self._build_scene(
                produce_occluder=split_targets_and_occluders,
            )
            if split_targets_and_occluders:
                written["target"].append(self._export_watertight(target_mesh, out / "target" / f"scene_{i}.obj"))
                written["occluder"].append(
                    self._export_watertight(occluder_mesh, out / "occluder" / f"scene_{i}.obj")
                )
            else:
                written["scene"].append(self._export_watertight(target_mesh, out / f"scene_{i}.obj"))
            scenes_metadata.append({
                "scene_id": i,
                "boxes": [{"min": list(map(float, lo)), "max": list(map(float, hi))}
                          for lo, hi in box_aabbs],
            })

        (out / "boxes.json").write_text(json.dumps({"scenes": scenes_metadata}, indent=2))
        return {k: v for k, v in written.items() if v}

    # Geometry construction

    def _build_scene(
        self, *, produce_occluder: bool,
    ) -> tuple[trimesh.Trimesh, trimesh.Trimesh | None, list[tuple[np.ndarray, np.ndarray]]]:
        """Return ``(target_mesh, occluder_mesh_or_None, box_aabbs)``.

        Both meshes share the shelf and pole geometry; the occluder mesh
        differs only in that the boxes (the detection targets) are absent.
        ``box_aabbs`` lists each box's world-frame ``(min, max)`` corners.
        """
        target_parts: list[trimesh.Trimesh] = []
        occluder_parts: list[trimesh.Trimesh] = []
        box_aabbs_world: list[tuple[np.ndarray, np.ndarray]] = []
        x_cursor = 0.0
        shelf_metadata: list[tuple[ShelfParams, float]] = []

        for params in self.layout:
            translation = np.array([x_cursor, -1.0, 0.0])
            full, boxes_local = self._build_shelf(params, include_boxes=True)
            full.apply_translation(translation)
            target_parts.append(full)
            for lo, hi in boxes_local:
                box_aabbs_world.append((lo + translation, hi + translation))
            if produce_occluder:
                empty, _ = self._build_shelf(params, include_boxes=False)
                empty.apply_translation(translation)
                occluder_parts.append(empty)
            shelf_metadata.append((params, x_cursor))
            x_cursor += params.shelf_dim[0] + self.shelf_spacing_x

        rods = list(self._build_connector_rods(shelf_metadata))
        target_mesh = trimesh.util.concatenate(target_parts + rods)
        if not produce_occluder:
            return target_mesh, None, box_aabbs_world
        occluder_mesh = trimesh.util.concatenate(occluder_parts + rods)
        return target_mesh, occluder_mesh, box_aabbs_world

    def _build_shelf(
        self, params: ShelfParams, *, include_boxes: bool,
    ) -> tuple[trimesh.Trimesh, list[tuple[np.ndarray, np.ndarray]]]:
        thickness = self.pole_thickness
        width, depth = params.shelf_dim
        box_w, box_d = params.box_xy_dim if params.box_xy_dim else (0.5 * width, 0.9 * depth)

        parts: list[trimesh.Trimesh] = []
        box_aabbs: list[tuple[np.ndarray, np.ndarray]] = []
        z_cursor = params.bottom_offset
        for z_gap, (min_h, max_h) in zip(params.spacing + [None], params.box_height_ranges):
            shelf = trimesh.creation.box(extents=[width, depth, thickness])
            shelf.apply_translation([0, 0, z_cursor + thickness / 2])
            parts.append(shelf)

            if include_boxes:
                box_height = random.uniform(min_h, max_h)
                box = trimesh.creation.box(extents=[box_w, box_d, box_height])
                box_center_z = z_cursor + thickness + box_height / 2
                box.apply_translation([0, 0, box_center_z])
                parts.append(box)
                box_aabbs.append((
                    np.array([-box_w / 2, -box_d / 2, box_center_z - box_height / 2]),
                    np.array([ box_w / 2,  box_d / 2, box_center_z + box_height / 2]),
                ))

            z_cursor += thickness
            if z_gap is not None:
                z_cursor += z_gap

        ceiling = z_cursor
        for cx, cy in [(-width / 2, -depth / 2), (-width / 2, depth / 2),
                       (width / 2, -depth / 2), (width / 2, depth / 2)]:
            pole = trimesh.creation.box(extents=[thickness, thickness, ceiling])
            tx = cx + np.sign(cx) * (thickness / 2)
            ty = cy + np.sign(cy) * (thickness / 2)
            pole.apply_translation([tx, ty, ceiling / 2])
            parts.append(pole)
        return trimesh.util.concatenate(parts), box_aabbs

    def _build_connector_rods(self, shelf_metadata):
        """Connecting rods between adjacent shelf rows.

        For each adjacent pair: two horizontal rods on the back side of the
        rack at ``rod_bar_z``, plus four vertical rods linking each rod
        endpoint up to ground level (z = 0). Six rods per shelf gap.
        """
        pole_offset = self.pole_thickness / 2
        for (p0, x0), (p1, x1) in zip(shelf_metadata, shelf_metadata[1:]):
            w0, d0 = p0.shelf_dim
            w1, d1 = p1.shelf_dim
            back_y0 = -1.0 - d0 / 2 - pole_offset
            back_y1 = -1.0 - d1 / 2 - pole_offset
            start_left  = [x0 - w0 / 2 - pole_offset, back_y0, self.rod_bar_z]
            end_left    = [x1 - w1 / 2 - pole_offset, back_y1, self.rod_bar_z]
            start_right = [x0 + w0 / 2 + pole_offset, back_y0, self.rod_bar_z]
            end_right   = [x1 + w1 / 2 + pole_offset, back_y1, self.rod_bar_z]

            yield self._connector_rod(start_left,  end_left,  self.rod_thickness)
            yield self._connector_rod(start_right, end_right, self.rod_thickness)
            yield self._connector_rod(start_left,  [start_left[0],  start_left[1],  0.0], self.rod_thickness)
            yield self._connector_rod(end_left,    [end_left[0],    end_left[1],    0.0], self.rod_thickness)
            yield self._connector_rod(start_right, [start_right[0], start_right[1], 0.0], self.rod_thickness)
            yield self._connector_rod(end_right,   [end_right[0],   end_right[1],   0.0], self.rod_thickness)

    def _export_watertight(self, mesh: trimesh.Trimesh, final_path: Path) -> Path:
        raw_path = final_path.with_suffix(".raw.obj")
        mesh.export(raw_path)
        fixed = self._make_watertight(raw_path)
        raw_path.unlink()
        fixed.export(final_path)
        return final_path

    def _connector_rod(self, start, end, thickness: float) -> trimesh.Trimesh:
        vec = np.array(end) - np.array(start)
        length = float(np.linalg.norm(vec))
        rod = trimesh.creation.box(extents=[length, thickness, thickness])
        if length > 0:
            xform = trimesh.geometry.align_vectors([1, 0, 0], vec / length)
            xform[:3, 3] = (np.array(start) + np.array(end)) / 2
            rod.apply_transform(xform)
        return rod

    # Watertight post-processing via mesh2sdf

    def _make_watertight(self, filepath: str | Path) -> trimesh.Trimesh:
        import mesh2sdf

        mesh = trimesh.load(filepath, force="mesh")
        bbmin, bbmax = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
        center = 0.5 * (bbmin + bbmax)
        scale = 2.0 * self.sdf_normalization_scale / (bbmax - bbmin).max()
        mesh.vertices = (mesh.vertices - center) * scale

        level = 2 / self.sdf_grid_size
        _, fixed = mesh2sdf.compute(
            mesh.vertices, mesh.faces, self.sdf_grid_size,
            fix=True, level=level, return_mesh=True,
        )
        fixed.vertices = fixed.vertices / scale + center
        return fixed


# Held-out front-plane test scenes (warehouse_detection evaluation)


@dataclass
class FrontPlaneQuad:
    """A constant-y quad on the front face of one box."""

    y: float
    x_min: float
    x_max: float
    z_min: float
    z_max: float


@dataclass
class FrontPlaneScene:
    """Held-out test scene for the box-detection metric.

    Each plane is a single quad on the front face of one box. Box detection
    = ray-plane intersection within the quad's ``(x, z)`` extent and within
    the ray's time gate.
    """

    planes: list[FrontPlaneQuad]


def generate_front_plane_test_scenes(
    n_scenes: int, *, seed: int,
    layout: Sequence[ShelfParams] = DEFAULT_LAYOUT,
    shelf_spacing_x: float = 0.2,
    pole_thickness: float = 0.02,
) -> list[FrontPlaneScene]:
    """Generate held-out test scenes consisting only of box front faces.

    Boxes are sampled with the same per-row params as
    :class:`MultiShelfGenerator` so the test distribution is consistent with
    training, but only their front-face quads are kept (the metric does not
    need the rest of the geometry).
    """
    rng = random.Random(seed)
    scenes: list[FrontPlaneScene] = []
    for _ in range(n_scenes):
        planes: list[FrontPlaneQuad] = []
        x_cursor = 0.0
        for params in layout:
            box_w, box_d = params.box_xy_dim or (0.5 * params.shelf_dim[0], 0.9 * params.shelf_dim[1])
            front_y = -1.0 + box_d / 2  # consistent with MultiShelfGenerator's y-translation of -1.0
            z_cursor = params.bottom_offset
            for gap, (h_min, h_max) in zip(list(params.spacing) + [None], params.box_height_ranges):
                h = rng.uniform(h_min, h_max)
                z0 = z_cursor + pole_thickness
                z1 = z0 + h
                planes.append(FrontPlaneQuad(
                    y=front_y,
                    x_min=x_cursor - box_w / 2, x_max=x_cursor + box_w / 2,
                    z_min=z0, z_max=z1,
                ))
                z_cursor += pole_thickness
                if gap is not None:
                    z_cursor += gap
            x_cursor += params.shelf_dim[0] + shelf_spacing_x
        scenes.append(FrontPlaneScene(planes=planes))
    return scenes


def ray_hits_front_plane(
    origin: np.ndarray, direction: np.ndarray, t_min: float, t_max: float,
    plane: FrontPlaneQuad,
) -> bool:
    """Return whether the segment ``origin + t·direction``, ``t ∈ [t_min, t_max]``,
    intersects the front-plane quad."""
    if abs(direction[1]) < 1e-12:
        return False
    t_hit = (plane.y - origin[1]) / direction[1]
    if not (t_min <= t_hit <= t_max):
        return False
    pt = origin + t_hit * direction
    return (plane.x_min <= pt[0] <= plane.x_max) and (plane.z_min <= pt[2] <= plane.z_max)
