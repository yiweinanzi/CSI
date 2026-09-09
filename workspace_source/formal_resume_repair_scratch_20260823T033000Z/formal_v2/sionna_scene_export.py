from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .external_adapters.wigatr_protocol import grid_to_triangular_mesh
from .formal_dataset import FormalDataset
from .formal_io import sha256_file, write_json
from .formal_protocol import PatchSpec


SIONNA_REVISION = "04ddb9312116b408093b9d3ad363a3df355093a6"
SIONNA_RT_VERSION = "1.2.1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export formal external-validation worlds as authenticated Sionna scenes")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--license-id", required=True)
    parser.add_argument("--carrier-frequency-hz", type=float, required=True)
    parser.add_argument("--subcarrier-spacing-hz", type=float, required=True)
    parser.add_argument("--receiver-z-m", type=float, default=1.5)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--refraction", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = export_sionna_scenes(
        args.dataset,
        args.output,
        license_id=args.license_id,
        carrier_frequency_hz=args.carrier_frequency_hz,
        subcarrier_spacing_hz=args.subcarrier_spacing_hz,
        receiver_z_m=args.receiver_z_m,
        max_depth=args.max_depth,
        refraction=bool(args.refraction),
    )
    print(manifest)
    return 0


def export_sionna_scenes(
    dataset_path,
    output_root,
    *,
    license_id,
    carrier_frequency_hz,
    subcarrier_spacing_hz,
    receiver_z_m,
    max_depth,
    refraction,
) -> Path:
    dataset = FormalDataset.load(dataset_path)
    if not str(license_id).strip():
        raise ValueError("Sionna exported assets require a nonempty license identifier")
    registered_licenses = {str(value) for value in dataset.metadata["assets"]["license_ids"]}
    if str(license_id) not in registered_licenses:
        raise ValueError("Sionna export license must match a license registered by the dataset")
    for name, value in (
        ("carrier_frequency_hz", carrier_frequency_hz),
        ("subcarrier_spacing_hz", subcarrier_spacing_hz),
        ("receiver_z_m", receiver_z_m),
    ):
        if not np.isfinite(value) or float(value) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth <= 0:
        raise ValueError("Sionna max_depth must be a positive integer")
    representation = dataset.metadata["representation"]
    resolution = float(representation["map_resolution_m"])
    origin = tuple(float(value) for value in representation["map_origin_xy_m"])
    channels = tuple(str(value) for value in dataset.map_channel_names.tolist())
    worlds = []
    scenes = dataset.indices_for_role("external_validation")
    if scenes.size == 0:
        raise RuntimeError("Sionna export requires at least one external_validation scene")
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for scene_value in scenes:
        scene = int(scene_value)
        for world in range(dataset.world_count):
            world_root = output / str(dataset.scene_ids[scene]) / f"world-{world}"
            mesh_root = world_root / "mesh"
            mesh_root.mkdir(parents=True, exist_ok=False)
            mesh, materials = grid_to_triangular_mesh(
                dataset.maps[scene, world],
                channels,
                resolution_m=resolution,
                origin_xy_m=origin,
                occupancy_threshold=0.5,
                minimum_height_m=0.001,
            )
            asset_rows = []
            material_meshes = []
            for material in sorted(set(int(value) for value in materials.tolist())):
                path = mesh_root / f"material-{material}.ply"
                _write_triangle_ply(path, mesh[materials == material])
                asset_rows.append({"path": str(path.relative_to(world_root)), "sha256": sha256_file(path), "license_id": str(license_id)})
                material_meshes.append((material, path))
            ground = mesh_root / "ground.ply"
            _write_triangle_ply(ground, _ground_triangles(dataset.maps.shape[-2:], origin, resolution))
            asset_rows.append({"path": str(ground.relative_to(world_root)), "sha256": sha256_file(ground), "license_id": str(license_id)})
            scene_xml = world_root / "scene.xml"
            _write_scene_xml(scene_xml, ground, material_meshes)
            assets = world_root / "assets.json"
            write_json(assets, {"schema_version": "csi-pairs-v6-sionna-assets-v1", "assets": asset_rows})
            worlds.append(
                {
                    "scene_id": str(dataset.scene_ids[scene]),
                    "world": world,
                    "scene_xml": str(scene_xml),
                    "scene_xml_sha256": sha256_file(scene_xml),
                    "asset_manifest": str(assets),
                    "asset_manifest_sha256": sha256_file(assets),
                    "canonical_map_sha256": str(dataset.canonical_map_sha256[scene, world]),
                }
            )
    spec = PatchSpec.from_metadata(dataset.metadata)
    manifest = output / "sionna_scene_manifest.json"
    write_json(
        manifest,
        {
            "schema_version": "csi-pairs-v6-sionna-scene-manifest-v1",
            "dataset_sha256": sha256_file(dataset.source_path),
            "engine_config_sha256": dataset.metadata["engine"]["config_sha256"],
            "sionna_revision": SIONNA_REVISION,
            "sionna_rt_version": SIONNA_RT_VERSION,
            "license_id": "Apache-2.0",
            "carrier_frequency_hz": float(carrier_frequency_hz),
            "bandwidth_hz": float(subcarrier_spacing_hz) * spec.subcarriers,
            "subcarrier_spacing_hz": float(subcarrier_spacing_hz),
            "max_depth": int(max_depth),
            "refraction": bool(refraction),
            "receiver_z_m": float(receiver_z_m),
            "tx_array": _array(1, 1),
            "rx_array": _array(spec.antennas, 1),
            "worlds": worlds,
        },
    )
    return manifest


def _array(rows, columns):
    return {
        "num_rows": int(rows),
        "num_cols": int(columns),
        "vertical_spacing": 0.5,
        "horizontal_spacing": 0.5,
        "pattern": "iso",
        "polarization": "V",
    }


def _ground_triangles(shape, origin, resolution):
    rows, columns = shape
    x0, y0 = origin
    x1 = x0 + columns * resolution
    y1 = y0 + rows * resolution
    corners = np.asarray(((x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)), dtype=np.float32)
    return np.asarray(((corners[0], corners[1], corners[2]), (corners[0], corners[2], corners[3])), dtype=np.float32)


def _write_triangle_ply(path: Path, triangles: np.ndarray) -> None:
    values = np.asarray(triangles, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or values.shape[0] == 0:
        raise ValueError("Sionna PLY export requires nonempty triangular geometry")
    vertices = values.reshape(-1, 3)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {vertices.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {values.shape[0]}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend("{:.9g} {:.9g} {:.9g}".format(*vertex) for vertex in vertices)
    lines.extend(f"3 {3 * index} {3 * index + 1} {3 * index + 2}" for index in range(values.shape[0]))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_scene_xml(path: Path, ground: Path, material_meshes) -> None:
    scene = ET.Element("scene", version="2.1.0")
    palette = {
        "mat-itu_concrete": (0.5, 0.5, 0.5),
        "mat-itu_marble": (0.7, 0.64, 0.49),
        "mat-itu_metal": (0.22, 0.22, 0.25),
        "mat-itu_wood": (0.04, 0.58, 0.18),
        "mat-itu_glass": (0.0, 0.0, 1.0),
        "mat-itu_wet_ground": (0.3, 0.25, 0.2),
    }
    material_names = tuple(name for name in palette if name != "mat-itu_wet_ground")
    used = {"mat-itu_wet_ground"}
    used.update(material_names[material % len(material_names)] for material, _ in material_meshes)
    materials = [(name, palette[name]) for name in sorted(used)]
    for material_id, color in materials:
        bsdf = ET.SubElement(scene, "bsdf", type="diffuse", id=material_id)
        ET.SubElement(bsdf, "rgb", name="reflectance", value=" ".join(str(value) for value in color))

    def add_shape(identifier, mesh_path, material_id):
        shape = ET.SubElement(scene, "shape", type="ply", id=identifier)
        ET.SubElement(shape, "string", name="filename", value=str(mesh_path.relative_to(path.parent)))
        ET.SubElement(shape, "ref", id=material_id, name="bsdf")
        ET.SubElement(shape, "boolean", name="face_normals", value="true")

    add_shape("mesh-ground", ground, "mat-itu_wet_ground")
    for material, mesh_path in material_meshes:
        add_shape(f"mesh-material-{material}", mesh_path, material_names[material % len(material_names)])
    ET.ElementTree(scene).write(path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    raise SystemExit(main())
