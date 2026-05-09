"""Mitsuba scene-template patching for emitter_design.

The bundled :data:`LIDAR_TEMPLATE_XML` carries the perspective sensor and
transient HDR film already wired up; the helpers here insert one
projector emitter per :class:`Frustum`, set the temporal-bin range, and
point the scene's shape slot at a target mesh path.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from ._vec import sph_to_cart

# Bundled Mitsuba LiDAR scene template (under assets/).
LIDAR_TEMPLATE_XML: Path = (
    Path(__file__).resolve().parents[3] / "assets" / "scenes" / "lidar_template.xml"
)


@dataclass
class Frustum:
    """One projector emitter, parameterized in the units of the XML scene.

    ``time_mean`` and ``time_sigma`` carry the per-emitter time gate (in
    optical-path-length meters); the transient peak finder masks each
    emitter's histogram to ``[t_mean − 2σ, t_mean + 2σ]`` plus the
    detector gate.
    """

    origin: tuple[float, float, float]
    azimuth_deg: float
    elevation_deg: float
    fov_width_deg: float
    fov_height_deg: float
    time_mean: float = 0.0
    time_sigma: float = 0.0
    scale: float = 1.0


def frustums_from_sensors(sensors) -> list[Frustum]:
    """Convert :class:`Sensor` objects into projector frustums.

    The projector field of view is taken as 4σ in azimuth and elevation
    (≈ 95 % CI of the EM-fitted Gaussian).
    """
    return [
        Frustum(
            origin=(float(s.origin_mean[0]), float(s.origin_mean[1]), float(s.origin_mean[2])),
            azimuth_deg=math.degrees(float(s.mean[3])),
            elevation_deg=math.degrees(float(s.mean[4])),
            fov_width_deg=4 * math.degrees(float(s.sigma[3])),
            fov_height_deg=4 * math.degrees(float(s.sigma[4])),
            time_mean=float(s.mean[5]),
            time_sigma=float(s.sigma[5]),
        )
        for s in sensors
    ]


def patch_lidar_xml(
    in_xml: Path,
    out_xml: Path,
    *,
    frustums: list[Frustum],
    object_path: str,
    resolution: int,
    temporal_bins: int,
    temporal_range: tuple[float, float],
) -> Path:
    """Patch the bundled LiDAR scene template with one render's parameters."""
    tree = ET.parse(str(in_xml))
    root = tree.getroot()
    _patch_projectors(root, frustums)
    _patch_temporal(root, temporal_range[0], temporal_range[1], temporal_bins)
    _patch_object(root, object_path)
    _patch_resolution(root, resolution)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, xml_declaration=True, encoding="utf-8")
    return out_xml


def extract_camera_info(xml_path: Path) -> dict:
    """Read the perspective sensor's pose and intrinsics from a scene XML."""
    root = ET.parse(str(xml_path)).getroot()
    sensor = root.find(".//sensor[@type='perspective']")
    fov = float(sensor.find("float[@name='fov']").get("value"))
    res = int(sensor.find("film/integer[@name='width']").get("value"))
    la = sensor.find("transform/lookat")
    return {
        "origin": tuple(map(float, la.get("origin").split(","))),
        "target": tuple(map(float, la.get("target").split(","))),
        "up": tuple(map(float, la.get("up").split(","))),
        "fov_deg": fov,
        "resolution": res,
    }


def _patch_projectors(root: ET.Element, frustums: list[Frustum]) -> None:
    """Replace existing projector emitters with one per frustum (cyclic RGB color)."""
    for old in root.findall("emitter[@type='projector']"):
        root.remove(old)
    palette = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    for idx, fr in enumerate(frustums):
        r, g, b = palette[idx % 3]
        ux, uy, uz = sph_to_cart(fr.azimuth_deg, fr.elevation_deg)
        ox, oy, oz = fr.origin
        fov_max = max(fr.fov_width_deg, fr.fov_height_deg)
        sx = math.tan(math.radians(fr.fov_width_deg * 0.5)) / math.tan(math.radians(fov_max * 0.5))
        sy = math.tan(math.radians(fr.fov_height_deg * 0.5)) / math.tan(math.radians(fov_max * 0.5))

        emitter = ET.SubElement(root, "emitter", {"type": "projector", "id": f"projector{idx}"})
        tex = ET.SubElement(emitter, "texture", {"name": "irradiance", "type": "checkerboard"})
        ET.SubElement(tex, "rgb", {"name": "color0", "value": f"{r:.6f}, {g:.6f}, {b:.6f}"})
        ET.SubElement(tex, "rgb", {"name": "color1", "value": f"{r:.6f}, {g:.6f}, {b:.6f}"})
        ET.SubElement(emitter, "float", {"name": "scale", "value": f"{fr.scale:.6f}"})
        ET.SubElement(emitter, "float", {"name": "fov", "value": f"{fov_max:.6f}"})
        ET.SubElement(emitter, "string", {"name": "fov_axis", "value": "smaller"})

        xform = ET.SubElement(emitter, "transform", {"name": "to_world"})
        ET.SubElement(xform, "scale", {"x": f"{sx:.6f}", "y": f"{sy:.6f}", "z": "1"})
        ET.SubElement(xform, "lookat", {
            "origin": f"{ox}, {oy}, {oz}",
            "target": f"{ox + ux}, {oy + uy}, {oz + uz}",
            "up": "0, 1, 0",
        })


def _patch_temporal(root: ET.Element, start_opl: float, end_opl: float, num_bins: int) -> None:
    root.find("default[@name='temporal_bins']").set("value", str(num_bins))
    film = root.find(".//film[@type='transient_hdr_film']")
    film.find("integer[@name='temporal_bins']").set("value", str(num_bins))
    bin_width = (end_opl - start_opl) / num_bins
    film.find("float[@name='start_opl']").set("value", str(start_opl))
    film.find("float[@name='bin_width_opl']").set("value", str(bin_width))


def _patch_object(root: ET.Element, object_path: str) -> None:
    shape = root.find(".//shape[@id='user_object']")
    shape.find("string[@name='filename']").set("value", object_path)


def _patch_resolution(root: ET.Element, res: int) -> None:
    root.find("default[@name='res']").set("value", str(res))
    film = root.find(".//film[@type='transient_hdr_film']")
    film.find("integer[@name='width']").set("value", str(res))
    film.find("integer[@name='height']").set("value", str(res))

