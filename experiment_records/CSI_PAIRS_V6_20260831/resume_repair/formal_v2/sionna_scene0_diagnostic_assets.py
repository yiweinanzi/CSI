"""Build and validate only scene 0 assets for local Sionna diagnostics.

This module deliberately does not emit ``asset_manifest.json``. Its output is a
single-bank diagnostic input and cannot be mistaken for the frozen 34-bank
formal asset package.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from formal_v2 import sionna_osm_candidate as candidate


DIAGNOSTIC_ASSET_SCHEMA = "csi-pairs-sionna-osm-scene0-diagnostic-assets-v1"
SCIENTIFIC_USE = "DIAGNOSTIC_NOT_FORMAL_EVIDENCE"
EXPECTED_SCENE = {
    "scene_index": 0,
    "scene_id": "osm-sionna-source-chicago-bank-00",
    "city_id": "source-chicago",
    "role": "source_encoder_train",
    "bank_index_within_city": 0,
}
ASSET_RELATIVE_PATHS = (
    "bank.json",
    "mesh/background.ply",
    "mesh/ground.ply",
    "mesh/primitive-0.ply",
    "mesh/primitive-1.ply",
    "scene.xml",
)
MESH_MATERIALS = {
    "ground": "mat-itu_wet_ground",
    "background": "mat-itu_concrete",
    "primitive-0": "mat-itu_brick",
    "primitive-1": "mat-itu_wood",
}


def _assert_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be an existing non-symlink file: {path}")
    return path


def _local_sim_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if parent.name == "CSI_M4_LOCAL_SIM":
            return parent
    raise RuntimeError("builder is not located below CSI_M4_LOCAL_SIM")


def _require_output_scope(output: Path) -> None:
    allowed = _local_sim_root()
    try:
        output.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"output must remain below {allowed}: {output}") from error


def _output_row(root: Path, path: Path, *, license_id: str) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": candidate.sha256_file(path),
        "license_id": license_id,
        "simulation_not_measurement": True,
        "scientific_use": SCIENTIFIC_USE,
    }


def _read_ascii_triangle_ply(path: Path) -> dict[str, object]:
    """Read enough of an ASCII PLY to prove its vertices and faces are usable."""

    lines = path.read_text(encoding="ascii", errors="strict").splitlines()
    if len(lines) < 10 or lines[0] != "ply" or "format ascii 1.0" not in lines[:4]:
        raise ValueError(f"unsupported or malformed ASCII PLY: {path}")
    try:
        header_end = lines.index("end_header")
    except ValueError as error:
        raise ValueError(f"PLY has no end_header: {path}") from error
    vertex_rows = [line for line in lines[:header_end] if line.startswith("element vertex ")]
    face_rows = [line for line in lines[:header_end] if line.startswith("element face ")]
    if len(vertex_rows) != 1 or len(face_rows) != 1:
        raise ValueError(f"PLY must declare exactly one vertex and face element: {path}")
    vertex_count = int(vertex_rows[0].split()[2])
    face_count = int(face_rows[0].split()[2])
    if vertex_count < 3 or face_count < 1:
        raise ValueError(f"PLY contains no usable triangles: {path}")
    data = lines[header_end + 1 :]
    if len(data) != vertex_count + face_count:
        raise ValueError(f"PLY row count does not match its header: {path}")
    vertices = np.asarray(
        [[float(value) for value in row.split()] for row in data[:vertex_count]],
        dtype=np.float64,
    )
    if vertices.shape != (vertex_count, 3) or not np.isfinite(vertices).all():
        raise ValueError(f"PLY vertices are not finite XYZ triples: {path}")
    for row in data[vertex_count:]:
        fields = row.split()
        if len(fields) != 4 or fields[0] != "3":
            raise ValueError(f"PLY contains a non-triangular face: {path}")
        indices = [int(value) for value in fields[1:]]
        if min(indices) < 0 or max(indices) >= vertex_count or len(set(indices)) != 3:
            raise ValueError(f"PLY face has invalid vertex indices: {path}")
    return {
        "path": path.name,
        "format": "ascii 1.0",
        "vertex_count": vertex_count,
        "face_count": face_count,
        "bounds_min_xyz_m": vertices.min(axis=0).tolist(),
        "bounds_max_xyz_m": vertices.max(axis=0).tolist(),
        "all_values_finite": True,
        "all_faces_triangular": True,
    }


def _validate_scene_xml(bank_dir: Path) -> dict[str, object]:
    xml_path = bank_dir / "scene.xml"
    root = ET.parse(xml_path).getroot()
    if root.tag != "scene" or root.get("version") != "2.1.0":
        raise ValueError("scene.xml must be a Mitsuba 2.1.0 scene")
    material_ids = {node.get("id") for node in root.findall("bsdf")}
    expected_materials = set(MESH_MATERIALS.values())
    if material_ids != expected_materials:
        raise ValueError(
            f"scene material ids differ: expected {sorted(expected_materials)}, "
            f"got {sorted(str(value) for value in material_ids)}"
        )
    shapes: dict[str, dict[str, str]] = {}
    references = []
    for shape in root.findall("shape"):
        shape_id = str(shape.get("id"))
        if shape.get("type") != "ply" or not shape_id.startswith("mesh-"):
            raise ValueError(f"unexpected scene shape: {shape_id}")
        mesh_name = shape_id.removeprefix("mesh-")
        filenames = [
            str(node.get("value"))
            for node in shape.findall("string")
            if node.get("name") == "filename"
        ]
        refs = [
            str(node.get("id"))
            for node in shape.findall("ref")
            if node.get("name") == "bsdf"
        ]
        if filenames != [f"mesh/{mesh_name}.ply"] or refs != [MESH_MATERIALS.get(mesh_name)]:
            raise ValueError(f"shape filename/material binding is invalid: {shape_id}")
        relative = Path(filenames[0])
        referenced = (bank_dir / relative).resolve()
        try:
            referenced.relative_to(bank_dir.resolve())
        except ValueError as error:
            raise ValueError(f"scene reference escapes the bank directory: {relative}") from error
        _assert_regular_file(referenced, "scene mesh reference")
        references.append(str(relative))
        shapes[mesh_name] = {"filename": str(relative), "material_id": refs[0]}
    if set(shapes) != set(MESH_MATERIALS):
        raise ValueError(f"scene mesh set differs: {sorted(shapes)}")
    return {
        "xml_readable": True,
        "scene_version": root.get("version"),
        "materials": sorted(expected_materials),
        "shapes": shapes,
        "all_references_resolve": True,
        "referenced_meshes": sorted(references),
    }


def validate_scene0_assets(
    bank_dir: str | Path,
    record: dict,
    config: dict,
) -> dict[str, object]:
    """Validate the generated geometry and deterministic 256-point receiver set."""

    from shapely.geometry import Point, Polygon

    root = Path(bank_dir).resolve()
    for relative in ASSET_RELATIVE_PATHS:
        _assert_regular_file(root / relative, "diagnostic scene asset")
    for key, expected in EXPECTED_SCENE.items():
        if record.get(key) != expected:
            raise ValueError(f"scene-0 identity mismatch for {key}: {record.get(key)!r}")
    if record.get("schema_version") != candidate.BANK_SCHEMA:
        raise ValueError("bank record schema mismatch")
    if int(config["positions_per_bank"]) != 256:
        raise ValueError("scene-0 diagnostic requires exactly 256 receiver positions")
    buildings = record.get("buildings")
    if not isinstance(buildings, list) or len(buildings) != int(record.get("building_count", -1)):
        raise ValueError("bank building_count does not match its building records")
    polygons = [Polygon(row["local_exterior_xy_m"]) for row in buildings]
    if not polygons or any(polygon.is_empty or not polygon.is_valid for polygon in polygons):
        raise ValueError("bank contains an empty or invalid building polygon")
    primitive_ids = [int(value) for value in record.get("primitive_osm_ids", [])]
    building_ids = {int(row["osm_id"]) for row in buildings}
    if len(primitive_ids) != 2 or len(set(primitive_ids)) != 2 or not set(primitive_ids) <= building_ids:
        raise ValueError("bank must bind two distinct primitive OSM ids")
    origin = Point(0.0, 0.0)
    selected_clearance = min(float(polygon.distance(origin)) for polygon in polygons)
    declared_clearance = float(record["bs_center_clearance_m"])
    if selected_clearance < 4.0 or declared_clearance < 4.0:
        raise ValueError("base station violates the frozen four-meter building clearance")

    positions = candidate._receiver_positions(record, config, 0)
    if positions.shape != (256, 2) or not np.isfinite(positions).all():
        raise ValueError("receiver generator did not produce 256 finite XY positions")
    if np.unique(positions, axis=0).shape[0] != 256:
        raise ValueError("receiver generator produced duplicate positions")
    map_config = config["map"]
    origin_xy = np.asarray(map_config["origin_xy_m"], dtype=np.float64)
    extent_xy = origin_xy + float(map_config["size"]) * float(map_config["resolution_m"])
    if np.any(positions < origin_xy) or np.any(positions > extent_xy):
        raise ValueError("receiver position lies outside the frozen map extent")
    receiver_clearances = np.asarray(
        [
            min(float(polygon.distance(Point(float(x), float(y)))) for polygon in polygons)
            for x, y in positions
        ],
        dtype=np.float64,
    )
    if float(receiver_clearances.min()) < 2.0:
        raise ValueError("receiver position violates the frozen two-meter building clearance")

    ply = {
        path.stem: _read_ascii_triangle_ply(path)
        for path in sorted((root / "mesh").glob("*.ply"))
    }
    if set(ply) != set(MESH_MATERIALS):
        raise ValueError(f"generated PLY set differs: {sorted(ply)}")
    xml = _validate_scene_xml(root)
    return {
        "status": "PASS",
        "scene_identity_exact": True,
        "bank_json_readable": True,
        "xml": xml,
        "ply": ply,
        "materials_exact": True,
        "base_station": {
            "xy_m": [0.0, 0.0],
            "required_minimum_clearance_m": 4.0,
            "declared_clearance_m": declared_clearance,
            "minimum_clearance_to_stored_buildings_m": selected_clearance,
            "clearance_pass": True,
        },
        "receivers": {
            "count": int(positions.shape[0]),
            "unique_count": int(np.unique(positions, axis=0).shape[0]),
            "all_finite": True,
            "all_inside_map": True,
            "required_minimum_building_clearance_m": 2.0,
            "minimum_building_clearance_m": float(receiver_clearances.min()),
            "positions_sha256": candidate._sha256_bytes(
                np.ascontiguousarray(positions).tobytes()
            ),
        },
    }


def _byte_comparison(generated: Path, frozen: Path) -> dict[str, object]:
    generated_bytes = generated.read_bytes()
    frozen_bytes = frozen.read_bytes()
    aligned = min(len(generated_bytes), len(frozen_bytes))
    first = next(
        (index for index in range(aligned) if generated_bytes[index] != frozen_bytes[index]),
        None,
    )
    if first is None and len(generated_bytes) != len(frozen_bytes):
        first = aligned
    mismatch_count = sum(
        generated_bytes[index] != frozen_bytes[index] for index in range(aligned)
    ) + abs(len(generated_bytes) - len(frozen_bytes))
    return {
        "generated_path": str(generated),
        "generated_bytes": len(generated_bytes),
        "generated_sha256": candidate.sha256_file(generated),
        "frozen_path": str(frozen),
        "frozen_bytes": len(frozen_bytes),
        "frozen_sha256": candidate.sha256_file(frozen),
        "byte_identical": generated_bytes == frozen_bytes,
        "differing_byte_count_with_size_delta": mismatch_count,
        "first_differing_byte_offset": first,
    }


def _load_frozen_scene0(frozen_asset_root: Path) -> tuple[Path, dict, Path]:
    manifest_path = _assert_regular_file(
        frozen_asset_root / "asset_manifest.json", "frozen formal asset manifest"
    )
    manifest = candidate._read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != candidate.ASSET_SCHEMA:
        raise ValueError("frozen reference asset manifest schema mismatch")
    matches = [row for row in manifest.get("banks", []) if row.get("scene_index") == 0]
    if len(matches) != 1:
        raise ValueError("frozen reference must contain exactly one scene_index=0 row")
    row = matches[0]
    for key, expected in EXPECTED_SCENE.items():
        if row.get(key) != expected:
            raise ValueError(f"frozen scene-0 identity mismatch for {key}")
    bank_dir = (frozen_asset_root / str(row["bank_record_path"])).parent.resolve()
    for relative in ASSET_RELATIVE_PATHS:
        _assert_regular_file(bank_dir / relative, "frozen scene-0 asset")
    return manifest_path, row, bank_dir


def load_diagnostic_asset_manifest(
    asset_root: str | Path,
) -> tuple[Path, dict, dict]:
    """Load this diagnostic package without weakening the formal manifest loader."""

    root = Path(asset_root).resolve()
    manifest_path = _assert_regular_file(
        root / "diagnostic_asset_manifest.json", "diagnostic asset manifest"
    )
    manifest = candidate._read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != DIAGNOSTIC_ASSET_SCHEMA:
        raise ValueError("diagnostic asset manifest schema mismatch")
    if manifest.get("simulation_not_measurement") is not True:
        raise ValueError("diagnostic manifest lacks simulation classification")
    if manifest.get("scientific_use") != SCIENTIFIC_USE:
        raise ValueError("diagnostic manifest scientific-use classification mismatch")
    config_input = root / str(manifest["config_path"])
    config_path = config_input.resolve()
    if config_input.is_symlink() or not config_path.is_relative_to(root):
        raise ValueError("diagnostic asset config escapes the asset root")
    config = candidate.load_config(config_path)
    if candidate.sha256_file(config_path) != manifest["config_sha256"]:
        raise ValueError("diagnostic asset config snapshot hash mismatch")
    banks = manifest.get("banks")
    if not isinstance(banks, list) or len(banks) != 1:
        raise ValueError("diagnostic asset manifest must contain exactly one bank")
    row = banks[0]
    for key, expected in EXPECTED_SCENE.items():
        if row.get(key) != expected:
            raise ValueError(f"diagnostic asset ledger mismatch for {key}")
    bank_record_path = Path(str(row.get("bank_record_path", "")))
    scene_xml_path = Path(str(row.get("scene_xml_path", "")))
    expected_root = Path("banks") / EXPECTED_SCENE["scene_id"]
    expected_paths = {str(expected_root / relative) for relative in ASSET_RELATIVE_PATHS}
    if bank_record_path != expected_root / "bank.json" or scene_xml_path != expected_root / "scene.xml":
        raise ValueError("diagnostic bank record and scene XML paths are inconsistent")
    files = row.get("files")
    if not isinstance(files, list) or len(files) != len(expected_paths):
        raise ValueError("diagnostic asset manifest must register the exact six scene files")
    actual_paths = [str(file_row.get("path", "")) for file_row in files if isinstance(file_row, dict)]
    if len(actual_paths) != len(files) or set(actual_paths) != expected_paths:
        raise ValueError("diagnostic asset manifest scene file set differs from the exact contract")
    if len(set(actual_paths)) != len(actual_paths):
        raise ValueError("diagnostic asset manifest contains duplicate scene files")
    for file_row in files:
        path = root / str(file_row["path"])
        if not path.resolve().is_relative_to(root):
            raise ValueError("diagnostic asset file escapes the asset root")
        _assert_regular_file(path, "diagnostic asset file")
        if candidate.sha256_file(path) != file_row["sha256"]:
            raise ValueError(f"diagnostic asset file changed: {path}")
    return root, manifest, config


def build_scene0_diagnostic_assets(
    config_path: str | Path,
    raw_cache: str | Path,
    frozen_asset_root: str | Path,
    output_root: str | Path,
) -> Path:
    config_file = _assert_regular_file(Path(config_path).resolve(), "source config")
    cache_root = Path(raw_cache).resolve()
    frozen_root = Path(frozen_asset_root).resolve()
    output = Path(output_root).resolve()
    _require_output_scope(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    raw_source = _assert_regular_file(
        cache_root / f"{EXPECTED_SCENE['city_id']}.json", "frozen Chicago raw OSM cache"
    )
    config = candidate.load_config(config_file)
    ledger_row = candidate.expected_scene_ledger(config)[0]
    for key, expected in EXPECTED_SCENE.items():
        if ledger_row.get(key) != expected:
            raise ValueError(f"expected_scene_ledger scene 0 changed for {key}")
    city = next(row for row in config["cities"] if row["city_id"] == EXPECTED_SCENE["city_id"])
    frozen_manifest_path, frozen_bank_row, frozen_bank_dir = _load_frozen_scene0(frozen_root)

    output.mkdir(parents=True, exist_ok=False)
    config_snapshot = candidate._write_json_exclusive(output / "generator_config.json", config)
    raw_root = output / "raw_osm"
    raw_root.mkdir()
    bank_root = output / "banks"
    bank_root.mkdir()
    raw_path, query, response, endpoint = candidate._download_city_osm(
        city, config, raw_root, cache_root
    )
    if endpoint != f"frozen-cache-sha256:{candidate.sha256_file(raw_source)}":
        raise RuntimeError("diagnostic builder unexpectedly did not use the frozen raw cache")
    buildings = candidate._parse_osm_buildings(response, city)
    city_ledger = sorted(
        (
            row
            for row in candidate.expected_scene_ledger(config)
            if row["city_id"] == city["city_id"]
        ),
        key=lambda row: int(row["bank_index_within_city"]),
    )
    selected, _selection_audit = candidate._select_city_banks(
        city, config, buildings, raw_path, query, city_ledger
    )
    record = selected[int(ledger_row["bank_index_within_city"])]
    bank_dir = bank_root / str(ledger_row["scene_id"])
    bank_dir.mkdir()
    bank_json = candidate._write_json_exclusive(bank_dir / "bank.json", record)
    candidate._write_scene_assets(record, bank_dir)
    validation = validate_scene0_assets(bank_dir, candidate._read_json(bank_json), config)

    comparisons = {
        relative: _byte_comparison(bank_dir / relative, frozen_bank_dir / relative)
        for relative in ASSET_RELATIVE_PATHS
    }
    identical_count = sum(row["byte_identical"] for row in comparisons.values())
    asset_files = [
        _output_row(
            output,
            bank_dir / relative,
            license_id="ODbL-1.0" if relative == "bank.json" else "ODbL-1.0-DERIVED",
        )
        for relative in ASSET_RELATIVE_PATHS
    ]
    produced_files = [
        _output_row(output, config_snapshot, license_id="PROJECT-CONFIG"),
        _output_row(output, raw_path, license_id="ODbL-1.0"),
        *asset_files,
    ]
    output_hash_map = {str(row["path"]): str(row["sha256"]) for row in produced_files}
    output_set_sha256 = candidate._sha256_bytes(
        candidate.canonical_json(output_hash_map).encode("ascii")
    )
    generator_path = Path(candidate.__file__).resolve()
    tool_path = Path(__file__).resolve()
    bank_row = {
        **ledger_row,
        "bank_record_path": str(bank_json.relative_to(output)),
        "bank_record_sha256": candidate.sha256_file(bank_json),
        "scene_xml_path": str((bank_dir / "scene.xml").relative_to(output)),
        "scene_xml_sha256": candidate.sha256_file(bank_dir / "scene.xml"),
        "files": asset_files,
    }
    status = "PASS" if validation["status"] == "PASS" and identical_count == 6 else "FAIL"
    manifest = {
        "schema_version": DIAGNOSTIC_ASSET_SCHEMA,
        "status": status,
        "simulation_not_measurement": True,
        "scientific_use": SCIENTIFIC_USE,
        "formal_data_ready": "NO",
        "post_audit_status": "POST_AUDIT_NO_GO",
        "formal_scientific_use": "FORBIDDEN",
        "purpose": "single-bank local LLVM asset and reproducibility diagnostic only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scene_count": 1,
        "config_path": str(config_snapshot.relative_to(output)),
        "config_sha256": candidate.sha256_file(config_snapshot),
        "source_config_path": str(config_file),
        "source_config_sha256": candidate.sha256_file(config_file),
        "source_config_canonical_sha256": candidate._sha256_bytes(
            candidate.canonical_json(config).encode("ascii")
        ),
        "raw_source": {
            "city_id": city["city_id"],
            "source_cache_path": str(raw_source),
            "source_cache_sha256": candidate.sha256_file(raw_source),
            "output_path": str(raw_path.relative_to(output)),
            "output_sha256": candidate.sha256_file(raw_path),
            "bytes": raw_path.stat().st_size,
            "query": query,
            "query_sha256": candidate._sha256_bytes(query.encode("ascii")),
            "endpoint": endpoint,
            "osm_base_timestamp": response.get("osm3s", {}).get("timestamp_osm_base", "unknown"),
            "parsed_closed_building_way_count": len(buildings),
            "simulation_not_measurement": True,
            "scientific_use": SCIENTIFIC_USE,
        },
        "generator": {
            "path": str(generator_path),
            "sha256": candidate.sha256_file(generator_path),
            "reused_functions": [
                "load_config",
                "expected_scene_ledger",
                "_download_city_osm",
                "_parse_osm_buildings",
                "_select_city_banks",
                "_write_scene_assets",
            ],
        },
        "builder": {"path": str(tool_path), "sha256": candidate.sha256_file(tool_path)},
        "banks": [bank_row],
        "validation": validation,
        "frozen_scene0_comparison": {
            "frozen_asset_root": str(frozen_root),
            "frozen_asset_manifest_path": str(frozen_manifest_path),
            "frozen_asset_manifest_sha256": candidate.sha256_file(frozen_manifest_path),
            "frozen_bank_record_sha256_from_manifest": frozen_bank_row["bank_record_sha256"],
            "compared_file_count": len(comparisons),
            "byte_identical_file_count": identical_count,
            "all_six_assets_byte_identical": identical_count == 6,
            "files": comparisons,
        },
        "produced_files": produced_files,
        "output_hashes": output_hash_map,
        "output_set_sha256": output_set_sha256,
    }
    manifest_path = candidate._write_json_exclusive(
        output / "diagnostic_asset_manifest.json", manifest
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build only frozen ledger scene 0 as a non-formal diagnostic asset package; "
            "all inputs are read-only and output is exclusive."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="unchanged frozen candidate config")
    parser.add_argument(
        "--raw-cache",
        type=Path,
        required=True,
        help="directory containing the frozen source-chicago.json Overpass response",
    )
    parser.add_argument(
        "--frozen-asset-root",
        type=Path,
        required=True,
        help="read-only 34-bank asset root used for the six-file scene-0 comparison",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new directory below CSI_M4_LOCAL_SIM; existing paths are never overwritten",
    )
    args = parser.parse_args(argv)
    manifest = build_scene0_diagnostic_assets(
        args.config, args.raw_cache, args.frozen_asset_root, args.output
    )
    payload = candidate._read_json(manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "manifest_sha256": candidate.sha256_file(manifest),
                "status": payload["status"],
                "simulation_not_measurement": True,
                "scientific_use": SCIENTIFIC_USE,
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
