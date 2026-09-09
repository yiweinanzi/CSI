from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from urllib import parse as urlparse
from urllib import request as urlrequest
import xml.etree.ElementTree as ET

import numpy as np


CONFIG_SCHEMA = "csi-pairs-sionna-osm-candidate-config-v1"
ASSET_SCHEMA = "csi-pairs-sionna-osm-asset-manifest-v1"
BANK_SCHEMA = "csi-pairs-sionna-osm-bank-v1"
SHARD_SCHEMA = "csi-pairs-sionna-osm-render-shard-v1"
DATASET_SCHEMA = "csi-pairs-formal-dataset-v2.1-v6"
SIONNA_REVISION = "04ddb9312116b408093b9d3ad363a3df355093a6"
SIONNA_VERSION = "2.0.1"
SIONNA_RT_VERSION = "1.2.1"
SIONNA_PYTHON_VERSION = "3.12.13"
MITSUBA_VERSION = "3.7.1"
DRJIT_VERSION = "1.2.0"
SIONNA_RUNTIME_RELATIVE = Path("formal_v2/external_adapters/.runtime-sionna/venv")
SIONNA_OPTIX_RELATIVE = Path(
    "formal_v2/external_adapters/.runtime-sionna/driver-libs/root/usr/lib/x86_64-linux-gnu"
)
SIONNA_CUDA_DRIVER = Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1")
SIONNA_LIBLLVM = Path("/lib/x86_64-linux-gnu/libLLVM-18.so")
SIONNA_MATERIAL_THICKNESS_M = 0.1
PATH_VERTEX_QUANTIZATION_M = 1e-5
# Current CUDA PathSolver output fails the authenticated zero-tolerance replay gate.
# Keep future candidates fail-closed until an exact regeneration protocol is verified.
SIONNA_EXACT_REGENERATION_VERIFIED = False
SOURCE_ROLES = (
    "source_encoder_train",
    "source_method_selection",
    "source_probe_train",
    "source_probe_selection",
    "source_calibration_fit",
    "source_calibration_selection",
    "source_final_unseen_bank",
)
MAP_CHANNEL_NAMES = np.asarray(("occupancy", "height", "material"), dtype="U16")
MATERIAL_CATEGORY = {
    "free_space": 0,
    "glass": 1,
    "wood": 2,
    "metal": 3,
    "concrete": 4,
}
STABLE_SURFACE_IDS = {
    "ground": 100,
    "background": 101,
    "primitive-0": 102,
    "primitive-1": 103,
}
REGENERATED_FIELDS = (
    "maps",
    "csi_clean",
    "csi_repeat",
    "free_space",
    "phase_reference_ids",
    "phase_reference_values",
    "phase_reference_source_sha256",
    "engine_config_json",
    "noop_maps",
    "path_ids",
    "path_power",
    "path_surface_ids",
    "noop_path_ids",
    "noop_path_power",
    "noop_path_surface_ids",
)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: str | Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_json_constant)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _write_json_exclusive(path: str | Path, value: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        handle.write("\n")
    target.chmod(0o644)
    return target


def _write_text_exclusive(path: str | Path, value: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        handle.write(value)
    target.chmod(0o644)
    return target


def _write_npz_exclusive(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite NPZ: {target}")
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.link(temporary_path, target)
        target.chmod(0o644)
    finally:
        temporary_path.unlink(missing_ok=True)
    return target


def load_config(path: str | Path) -> dict:
    config = _read_json(path)
    if not isinstance(config, dict) or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("Sionna OSM candidate config schema mismatch")
    required = {
        "schema_version",
        "seed",
        "overpass_endpoint",
        "map",
        "radio",
        "positions_per_bank",
        "world_bits",
        "cities",
    }
    if set(config) != required:
        raise ValueError("Sionna OSM candidate config fields must be exact")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("candidate seed must be a nonnegative integer")
    if type(config["positions_per_bank"]) is not int or config["positions_per_bank"] < 256:
        raise ValueError("formal candidate requires at least 256 positions per bank")
    world_bits = np.asarray(config["world_bits"], dtype=np.int64)
    if world_bits.shape != (4, 2) or set(map(tuple, world_bits.tolist())) != {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }:
        raise ValueError("candidate world_bits must be the complete two-bit hypercube")
    cities = config["cities"]
    if not isinstance(cities, list) or len(cities) != 6:
        raise ValueError("candidate config requires exactly six city records")
    city_ids = set()
    counts = {"source": 0, "target": 0, "external": 0}
    for city in cities:
        expected = {
            "city_id",
            "split_group",
            "center_lat",
            "center_lon",
            "query_delta_lat",
            "query_delta_lon",
            "bank_count",
        }
        if not isinstance(city, dict) or set(city) != expected:
            raise ValueError("city config fields must be exact")
        city_id = city["city_id"]
        if not isinstance(city_id, str) or not city_id or city_id in city_ids:
            raise ValueError("city ids must be unique nonempty strings")
        city_ids.add(city_id)
        group = city["split_group"]
        if group not in counts:
            raise ValueError("city split_group is invalid")
        counts[group] += 1
        required_banks = 7 if group == "source" else 8 if group == "target" else 2
        if city["bank_count"] != required_banks:
            raise ValueError(f"{city_id} must register {required_banks} banks")
    if counts != {"source": 2, "target": 2, "external": 2}:
        raise ValueError("candidate requires two source, target, and external cities")
    radio = config["radio"]
    if radio["repeats"] < 3 or radio["tx_antennas"] != 2 or radio["subcarriers"] != 4:
        raise ValueError("candidate radio shape must provide 3 repeats and a 2x4 complex grid")
    map_config = config["map"]
    if map_config["size"] != 256 or map_config["resolution_m"] != 1.0:
        raise ValueError("candidate map must use the frozen 256x256 one-meter grid")
    return config


def expected_scene_ledger(config: dict) -> list[dict[str, object]]:
    by_group: dict[str, list[dict]] = {"source": [], "target": [], "external": []}
    for city in config["cities"]:
        by_group[city["split_group"]].append(city)
    rows: list[dict[str, object]] = []
    for role_index, role in enumerate(SOURCE_ROLES):
        for city in by_group["source"]:
            rows.append(_ledger_row(city, role_index, role))
    for city in by_group["target"]:
        for bank_index in range(int(city["bank_count"])):
            rows.append(_ledger_row(city, bank_index, "target"))
    for city in by_group["external"]:
        for bank_index in range(int(city["bank_count"])):
            rows.append(_ledger_row(city, bank_index, "external_validation"))
    if len(rows) != 34:
        raise RuntimeError("formal candidate ledger must contain exactly 34 scene banks")
    for scene_index, row in enumerate(rows):
        row["scene_index"] = scene_index
    return rows


def _ledger_row(city: dict, bank_index: int, role: str) -> dict[str, object]:
    city_id = str(city["city_id"])
    scene_id = f"osm-sionna-{city_id}-bank-{bank_index:02d}"
    return {
        "city_id": city_id,
        "bank_index_within_city": int(bank_index),
        "role": role,
        "scene_id": scene_id,
        "bank_id": scene_id.replace("scene", "bank"),
        "base_map_cluster_id": f"osm-foundation-{city_id}-{bank_index:02d}",
    }


def prepare_assets(
    config_path: str | Path,
    output_root: str | Path,
    raw_cache: str | Path | None = None,
) -> Path:
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_snapshot = _write_json_exclusive(output / "generator_config.json", config)
    raw_root = output / "raw_osm"
    raw_root.mkdir()
    bank_root = output / "banks"
    bank_root.mkdir()
    city_banks: dict[str, list[dict]] = {}
    raw_sources = []
    for city in config["cities"]:
        raw_path, query, response, endpoint = _download_city_osm(
            city, config, raw_root, raw_cache
        )
        polygons = _parse_osm_buildings(response, city)
        selected = _select_city_banks(city, config, polygons, raw_path, query)
        city_banks[str(city["city_id"])] = selected
        raw_sources.append(
            {
                "city_id": city["city_id"],
                "path": str(raw_path.relative_to(output)),
                "bytes": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
                "overpass_query": query,
                "overpass_endpoint": endpoint,
                "osm_base_timestamp": response.get("osm3s", {}).get("timestamp_osm_base", "unknown"),
                "parsed_closed_building_way_count": len(polygons),
                "license_id": "ODbL-1.0",
            }
        )
    ledger = expected_scene_ledger(config)
    bank_rows = []
    for row in ledger:
        source = city_banks[str(row["city_id"])][int(row["bank_index_within_city"])]
        record = {**source, **row, "schema_version": BANK_SCHEMA}
        bank_dir = bank_root / str(row["scene_id"])
        bank_dir.mkdir()
        bank_json = _write_json_exclusive(bank_dir / "bank.json", record)
        _write_scene_assets(record, bank_dir)
        files = []
        for path in sorted(candidate for candidate in bank_dir.rglob("*") if candidate.is_file()):
            files.append(
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "license_id": "ODbL-1.0" if path == bank_json else "ODbL-1.0-DERIVED",
                }
            )
        bank_rows.append(
            {
                **row,
                "bank_record_path": str(bank_json.relative_to(output)),
                "bank_record_sha256": sha256_file(bank_json),
                "scene_xml_path": str((bank_dir / "scene.xml").relative_to(output)),
                "scene_xml_sha256": sha256_file(bank_dir / "scene.xml"),
                "files": files,
            }
        )
    attribution = _write_text_exclusive(
        output / "ATTRIBUTION.md",
        "# Asset attribution\n\n"
        "Building footprints and tags are from OpenStreetMap contributors and are used under "
        "the Open Data Commons Open Database License 1.0 (ODbL-1.0). "
        "https://www.openstreetmap.org/copyright\n\n"
        "Scene meshes are generated derivatives. No real RF measurements are included.\n",
    )
    manifest = {
        "schema_version": ASSET_SCHEMA,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_snapshot.relative_to(output)),
        "config_sha256": sha256_file(config_snapshot),
        "source_config_sha256": sha256_file(config_file),
        "licenses": [
            {
                "license_id": "ODbL-1.0",
                "url": "https://opendatacommons.org/licenses/odbl/1-0/",
                "attribution_url": "https://www.openstreetmap.org/copyright",
                "redistribution_allowed_with_terms": True,
            },
            {
                "license_id": "Apache-2.0",
                "url": "https://www.apache.org/licenses/LICENSE-2.0",
                "scope": "Sionna and Sionna RT engine source",
                "redistribution_allowed_with_terms": True,
            },
        ],
        "attribution_path": str(attribution.relative_to(output)),
        "attribution_sha256": sha256_file(attribution),
        "raw_sources": raw_sources,
        "banks": bank_rows,
    }
    manifest_path = _write_json_exclusive(output / "asset_manifest.json", manifest)
    return manifest_path


def _download_city_osm(
    city: dict,
    config: dict,
    raw_root: Path,
    raw_cache: str | Path | None = None,
) -> tuple[Path, str, dict, str]:
    south = float(city["center_lat"]) - float(city["query_delta_lat"])
    north = float(city["center_lat"]) + float(city["query_delta_lat"])
    west = float(city["center_lon"]) - float(city["query_delta_lon"])
    east = float(city["center_lon"]) + float(city["query_delta_lon"])
    query = (
        f'[out:json][timeout:120];way["building"]'
        f"({south:.7f},{west:.7f},{north:.7f},{east:.7f});out geom tags;"
    )
    cached = None if raw_cache is None else Path(raw_cache).resolve() / f"{city['city_id']}.json"
    if cached is not None and cached.is_file() and not cached.is_symlink():
        raw_path = raw_root / cached.name
        shutil.copy2(cached, raw_path)
        payload = raw_path.read_bytes()
        parsed = json.loads(payload.decode("utf-8"), parse_constant=_reject_json_constant)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("elements"), list):
            raise RuntimeError(f"cached Overpass response is malformed for {city['city_id']}")
        return raw_path, query, parsed, f"frozen-cache-sha256:{sha256_file(cached)}"
    body = urlparse.urlencode({"data": query}).encode("ascii")
    endpoints = []
    for endpoint in (
        str(config["overpass_endpoint"]),
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass.nchc.org.tw/api/interpreter",
    ):
        if endpoint not in endpoints:
            endpoints.append(endpoint)
    last_error: Exception | None = None
    payload = b""
    selected_endpoint = ""
    for endpoint in endpoints:
        request = urlrequest.Request(
            endpoint,
            data=body,
            headers={"User-Agent": "CSI-PAIRS-formal-audit/1.0 (research reproducibility)"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(request, timeout=180) as response:
                payload = response.read()
            selected_endpoint = endpoint
            break
        except Exception as error:  # The successful endpoint is frozen in the asset manifest.
            last_error = error
    if not payload:
        raise RuntimeError(f"Overpass query failed for {city['city_id']}: {last_error}")
    parsed = json.loads(payload.decode("utf-8"), parse_constant=_reject_json_constant)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("elements"), list):
        raise RuntimeError(f"Overpass response is malformed for {city['city_id']}")
    raw_path = raw_root / f"{city['city_id']}.json"
    with raw_path.open("xb") as handle:
        handle.write(payload)
    return raw_path, query, parsed, selected_endpoint


def _parse_osm_buildings(response: dict, city: dict) -> list[dict]:
    from pyproj import CRS, Transformer
    from shapely.geometry import Polygon

    longitude = float(city["center_lon"])
    latitude = float(city["center_lat"])
    zone = int(math.floor((longitude + 180.0) / 6.0) + 1)
    epsg = (32600 if latitude >= 0 else 32700) + zone
    transformer = Transformer.from_crs("EPSG:4326", CRS.from_epsg(epsg), always_xy=True)
    rows = []
    for element in response["elements"]:
        if element.get("type") != "way" or not isinstance(element.get("geometry"), list):
            continue
        geometry = element["geometry"]
        if len(geometry) < 4:
            continue
        lon_lat = [(float(point["lon"]), float(point["lat"])) for point in geometry]
        if lon_lat[0] != lon_lat[-1]:
            continue
        x, y = transformer.transform(
            [value[0] for value in lon_lat], [value[1] for value in lon_lat]
        )
        polygon = Polygon(zip(x, y))
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.geom_type != "Polygon" or polygon.is_empty:
            continue
        tags = element.get("tags") if isinstance(element.get("tags"), dict) else {}
        rows.append(
            {
                "osm_type": "way",
                "osm_id": int(element["id"]),
                "tags": {str(key): str(value) for key, value in sorted(tags.items())},
                "polygon": polygon,
                "height_m": _building_height(tags, int(element["id"])),
                "utm_epsg": epsg,
            }
        )
    return rows


def _building_height(tags: dict, osm_id: int) -> float:
    for key in ("height", "building:height"):
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(tags.get(key, "")))
        if match:
            return float(np.clip(float(match.group()), 3.0, 120.0))
    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(tags.get("building:levels", ""))
    )
    if match:
        return float(np.clip(float(match.group()) * 3.2, 3.0, 120.0))
    bucket = int.from_bytes(hashlib.sha256(str(osm_id).encode("ascii")).digest()[:2], "big") % 7
    return float(9.0 + 3.0 * bucket)


def _select_city_banks(
    city: dict,
    config: dict,
    buildings: list[dict],
    raw_path: Path,
    query: str,
) -> list[dict]:
    from pyproj import CRS, Transformer
    from shapely.geometry import Point, box

    if not buildings:
        raise RuntimeError(f"no closed OSM building ways were parsed for {city['city_id']}")
    epsg = int(buildings[0]["utm_epsg"])
    transformer = Transformer.from_crs("EPSG:4326", CRS.from_epsg(epsg), always_xy=True)
    center_x, center_y = transformer.transform(float(city["center_lon"]), float(city["center_lat"]))
    map_config = config["map"]
    radius = int(map_config["candidate_grid_radius"])
    spacing = float(map_config["candidate_grid_spacing_m"])
    half_extent = float(map_config["size"]) * float(map_config["resolution_m"]) / 2.0
    margin = 2.0
    candidates = []
    for grid_y in range(-radius, radius + 1):
        for grid_x in range(-radius, radius + 1):
            nominal_x = center_x + grid_x * spacing
            nominal_y = center_y + grid_y * spacing
            try:
                bs_x, bs_y = _nearest_free_bs(nominal_x, nominal_y, buildings)
            except RuntimeError:
                continue
            extent = box(
                bs_x - half_extent + margin,
                bs_y - half_extent + margin,
                bs_x + half_extent - margin,
                bs_y + half_extent - margin,
            )
            selected = [
                building
                for building in buildings
                if float(building["polygon"].area) >= float(map_config["minimum_building_area_m2"])
                and extent.contains(building["polygon"])
            ]
            if len(selected) < int(map_config["minimum_buildings_per_bank"]):
                continue
            selected.sort(key=lambda row: (-float(row["polygon"].area), int(row["osm_id"])))
            score = sum(min(float(value["polygon"].area), 2500.0) for value in selected)
            center_clearance = min(
                float(value["polygon"].distance(Point(bs_x, bs_y))) for value in selected
            )
            candidates.append((score + 20.0 * len(selected), center_clearance, bs_x, bs_y, selected))
    candidates.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))
    required = int(city["bank_count"])
    if len(candidates) < required:
        raise RuntimeError(
            f"{city['city_id']} has only {len(candidates)} qualifying non-overlap grid banks; "
            f"{required} are required"
        )
    chosen = candidates[:required]
    rows = []
    for bank_index, (_, clearance, bs_x, bs_y, selected) in enumerate(chosen):
        primitive_ids = {int(selected[0]["osm_id"]), int(selected[1]["osm_id"])}
        building_rows = []
        for building in selected:
            polygon = building["polygon"]
            coordinates = [
                [round(float(x - bs_x), 6), round(float(y - bs_y), 6)]
                for x, y in polygon.exterior.coords
            ]
            building_rows.append(
                {
                    "osm_type": building["osm_type"],
                    "osm_id": int(building["osm_id"]),
                    "height_m": round(float(building["height_m"]), 6),
                    "height_source": _height_source(building["tags"]),
                    "local_exterior_xy_m": coordinates,
                    "is_registered_primitive": int(building["osm_id"]) in primitive_ids,
                    "tags": building["tags"],
                }
            )
        rows.append(
            {
                "city_id": city["city_id"],
                "bank_index_within_city": bank_index,
                "utm_epsg": epsg,
                "bs_utm_xy_m": [round(float(bs_x), 6), round(float(bs_y), 6)],
                "bs_center_clearance_m": round(clearance, 6),
                "flat_ground_elevation_m": 0.0,
                "building_count": len(building_rows),
                "primitive_osm_ids": [int(selected[0]["osm_id"]), int(selected[1]["osm_id"])],
                "buildings": building_rows,
                "source": {
                    "kind": "OpenStreetMap Overpass building footprints",
                    "raw_path_name": raw_path.name,
                    "raw_sha256": sha256_file(raw_path),
                    "query_sha256": _sha256_bytes(query.encode("ascii")),
                    "license_id": "ODbL-1.0",
                    "attribution_url": "https://www.openstreetmap.org/copyright",
                },
            }
        )
    return rows


def _nearest_free_bs(x: float, y: float, buildings: list[dict]) -> tuple[float, float]:
    from shapely.geometry import Point

    offsets = [(0.0, 0.0)]
    for radius in (10.0, 20.0, 30.0, 40.0, 50.0):
        offsets.extend(
            (radius * math.cos(angle), radius * math.sin(angle))
            for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        )
    for dx, dy in offsets:
        point = Point(x + dx, y + dy)
        if all(float(row["polygon"].distance(point)) >= 4.0 for row in buildings):
            return float(x + dx), float(y + dy)
    raise RuntimeError("could not place a transmitter with four-meter building clearance")


def _height_source(tags: dict) -> str:
    if tags.get("height") or tags.get("building:height"):
        return "osm_height_tag_clamped_3_to_120_m"
    if tags.get("building:levels"):
        return "osm_levels_times_3.2_m_clamped_3_to_120_m"
    return "deterministic_osm_id_hash_fallback_9_to_27_m"


def _write_scene_assets(record: dict, bank_dir: Path) -> None:
    from shapely.geometry import Polygon

    mesh_root = bank_dir / "mesh"
    mesh_root.mkdir()
    primitive_ids = [int(value) for value in record["primitive_osm_ids"]]
    groups: dict[str, list[tuple[object, float]]] = {
        "background": [],
        "primitive-0": [],
        "primitive-1": [],
    }
    for building in record["buildings"]:
        polygon = Polygon(building["local_exterior_xy_m"])
        osm_id = int(building["osm_id"])
        if osm_id == primitive_ids[0]:
            group = "primitive-0"
        elif osm_id == primitive_ids[1]:
            group = "primitive-1"
        else:
            group = "background"
        groups[group].append((polygon, float(building["height_m"])))
    for group, values in groups.items():
        if not values:
            raise RuntimeError(f"scene group is empty: {group}")
        triangles = np.concatenate([_extruded_triangles(polygon, height) for polygon, height in values])
        _write_triangle_ply(mesh_root / f"{group}.ply", triangles)
    origin = (-128.0, -128.0)
    size = 256.0
    ground = np.asarray(
        (
            ((origin[0], origin[1], 0.0), (origin[0] + size, origin[1], 0.0), (origin[0] + size, origin[1] + size, 0.0)),
            ((origin[0], origin[1], 0.0), (origin[0] + size, origin[1] + size, 0.0), (origin[0], origin[1] + size, 0.0)),
        ),
        dtype=np.float32,
    )
    _write_triangle_ply(mesh_root / "ground.ply", ground)
    _write_scene_xml(bank_dir / "scene.xml")


def _extruded_triangles(polygon, height: float) -> np.ndarray:
    from shapely.ops import triangulate

    triangles = []
    for triangle in triangulate(polygon):
        if not polygon.covers(triangle.representative_point()):
            continue
        points = list(triangle.exterior.coords)[:3]
        triangles.append([[float(x), float(y), float(height)] for x, y in points])
    coordinates = list(polygon.exterior.coords)
    for first, second in zip(coordinates[:-1], coordinates[1:]):
        x0, y0 = first
        x1, y1 = second
        triangles.append(((x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, height)))
        triangles.append(((x0, y0, 0.0), (x1, y1, height), (x0, y0, height)))
    values = np.asarray(triangles, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or values.shape[0] == 0:
        raise RuntimeError("building extrusion produced no triangles")
    return values


def _write_triangle_ply(path: Path, triangles: np.ndarray) -> None:
    values = np.asarray(triangles, dtype=np.float32)
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
    lines.extend(
        f"3 {3 * index} {3 * index + 1} {3 * index + 2}"
        for index in range(values.shape[0])
    )
    _write_text_exclusive(path, "\n".join(lines) + "\n")


def _write_scene_xml(path: Path) -> None:
    scene = ET.Element("scene", version="2.1.0")
    colors = {
        "mat-itu_wet_ground": (0.18, 0.18, 0.16),
        "mat-itu_concrete": (0.54, 0.54, 0.54),
        "mat-itu_brick": (0.52, 0.24, 0.18),
        "mat-itu_wood": (0.31, 0.20, 0.10),
    }
    for material_id, color in colors.items():
        bsdf = ET.SubElement(scene, "bsdf", type="diffuse", id=material_id)
        ET.SubElement(
            bsdf,
            "rgb",
            name="reflectance",
            value=" ".join(str(value) for value in color),
        )
    for name, material in (
        ("ground", "mat-itu_wet_ground"),
        ("background", "mat-itu_concrete"),
        ("primitive-0", "mat-itu_brick"),
        ("primitive-1", "mat-itu_wood"),
    ):
        shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-{name}")
        ET.SubElement(shape, "string", name="filename", value=f"mesh/{name}.ply")
        ET.SubElement(shape, "ref", id=material, name="bsdf")
        ET.SubElement(shape, "boolean", name="face_normals", value="true")
    tree = ET.ElementTree(scene)
    with path.open("xb") as handle:
        tree.write(handle, encoding="utf-8", xml_declaration=True)


def _sionna_bootstrap_environment(
    project_root: str | Path,
    current: dict[str, str] | None = None,
    *,
    cuda_driver: str | Path = SIONNA_CUDA_DRIVER,
    libllvm: str | Path = SIONNA_LIBLLVM,
) -> tuple[Path, dict[str, str]]:
    root = Path(project_root).resolve()
    runtime = root / SIONNA_RUNTIME_RELATIVE
    python = runtime / "bin" / "python"
    optix = root / SIONNA_OPTIX_RELATIVE
    driver_path = Path(cuda_driver)
    llvm_path = Path(libllvm)
    required = {
        "Sionna runtime Python": python,
        "NVIDIA CUDA driver": driver_path,
        "Dr.Jit LLVM library": llvm_path,
        "OptiX driver directory": optix,
        "OptiX library": optix / "libnvoptix.so.1",
    }
    missing = [label for label, candidate in required.items() if not candidate.exists()]
    if missing:
        raise RuntimeError("Sionna runtime prerequisites are missing: " + ", ".join(missing))
    environment = dict(os.environ if current is None else current)
    preload = [value for value in environment.get("LD_PRELOAD", "").split(":") if value]
    driver = str(driver_path)
    environment["LD_PRELOAD"] = ":".join([driver, *[value for value in preload if value != driver]])
    library_path = [
        value for value in environment.get("LD_LIBRARY_PATH", "").split(":") if value
    ]
    optix_text = str(optix)
    environment["LD_LIBRARY_PATH"] = ":".join(
        [optix_text, *[value for value in library_path if value != optix_text]]
    )
    environment["DRJIT_LIBLLVM_PATH"] = str(llvm_path)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    project_text = str(root)
    python_path = [value for value in environment.get("PYTHONPATH", "").split(os.pathsep) if value]
    environment["PYTHONPATH"] = os.pathsep.join(
        [project_text, *[value for value in python_path if value != project_text]]
    )
    return python, environment


def ensure_sionna_runtime(reexec_arguments: list[str]) -> None:
    project_root = Path(__file__).resolve().parents[1]
    python, environment = _sionna_bootstrap_environment(project_root)
    runtime_prefix = python.parent.parent.resolve()
    preload_entries = os.environ.get("LD_PRELOAD", "").split(":")
    correct_python = Path(sys.prefix).resolve() == runtime_prefix
    driver_preloaded = str(SIONNA_CUDA_DRIVER) in preload_entries
    if correct_python and driver_preloaded:
        return
    if os.environ.get("CSI_PAIRS_SIONNA_BOOTSTRAPPED") == "1":
        raise RuntimeError(
            "Sionna runtime bootstrap did not activate the fixed Python and CUDA driver"
        )
    environment["CSI_PAIRS_SIONNA_BOOTSTRAPPED"] = "1"
    os.execve(str(python), [str(python), *reexec_arguments], environment)


def load_asset_manifest(asset_root: str | Path) -> tuple[Path, dict, dict]:
    root = Path(asset_root).resolve()
    manifest_path = root / "asset_manifest.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != ASSET_SCHEMA:
        raise ValueError("asset manifest schema mismatch")
    config_path = root / str(manifest["config_path"])
    config = load_config(config_path)
    if sha256_file(config_path) != manifest["config_sha256"]:
        raise ValueError("asset config snapshot hash mismatch")
    expected = expected_scene_ledger(config)
    if len(manifest.get("banks", [])) != len(expected):
        raise ValueError("asset manifest bank count mismatch")
    for row, ledger in zip(manifest["banks"], expected):
        for key in ("scene_index", "scene_id", "city_id", "role", "bank_id", "base_map_cluster_id"):
            if row.get(key) != ledger[key]:
                raise ValueError(f"asset manifest ledger mismatch for {key}")
        for file_row in row["files"]:
            path = root / str(file_row["path"])
            if path.is_symlink() or not path.is_file() or sha256_file(path) != file_row["sha256"]:
                raise ValueError(f"asset file is missing or changed: {path}")
    return root, manifest, config


def _gpu_snapshot(physical_index: int) -> dict[str, object]:
    fields = "index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu"
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
            "-i",
            str(physical_index),
        ],
        text=True,
    ).strip()
    values = [value.strip() for value in output.split(",")]
    if len(values) != 7:
        raise RuntimeError(f"unexpected nvidia-smi output: {output!r}")
    return {
        "physical_index": int(values[0]),
        "name": values[1],
        "uuid": values[2],
        "memory_total_mib": int(values[3]),
        "memory_used_mib": int(values[4]),
        "memory_free_mib": int(values[5]),
        "utilization_percent": int(values[6]),
    }


def render_shard(
    asset_root: str | Path,
    output_path: str | Path,
    scene_start: int,
    scene_end: int,
    physical_gpu_index: int,
) -> Path:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu_index):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must expose exactly the requested physical GPU; "
            f"expected {physical_gpu_index!r}, got {visible!r}"
        )
    root, manifest, config = load_asset_manifest(asset_root)
    if not (0 <= scene_start < scene_end <= len(manifest["banks"])):
        raise ValueError("render shard scene range is invalid")
    before = _gpu_snapshot(physical_gpu_index)
    started = time.monotonic()
    arrays_by_scene = []
    for scene_index in range(scene_start, scene_end):
        row = manifest["banks"][scene_index]
        bank_record = _read_json(root / str(row["bank_record_path"]))
        rendered = render_bank(bank_record, root, config, scene_index)
        arrays_by_scene.append(rendered)
        print(
            canonical_json(
                {
                    "event": "formal_candidate_bank_complete",
                    "physical_gpu_index": physical_gpu_index,
                    "scene_index": scene_index,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            ),
            flush=True,
        )
    arrays = {"scene_indices": np.arange(scene_start, scene_end, dtype=np.int64)}
    for name in arrays_by_scene[0]:
        arrays[name] = np.stack([row[name] for row in arrays_by_scene], axis=0)
    output = _write_npz_exclusive(output_path, arrays)
    after = _gpu_snapshot(physical_gpu_index)
    result = {
        "schema_version": SHARD_SCHEMA,
        "status": "PASS",
        "fixture": False,
        "scientific_use": "CANDIDATE",
        "simulation_not_measurement": True,
        "asset_manifest_sha256": sha256_file(root / "asset_manifest.json"),
        "scene_start_inclusive": scene_start,
        "scene_end_exclusive": scene_end,
        "scene_count": scene_end - scene_start,
        "duration_seconds": time.monotonic() - started,
        "output_path": str(output.resolve()),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "gpu": {
            "cuda_visible_devices": visible,
            "physical_gpu_index": physical_gpu_index,
            "uuid": before["uuid"],
            "before": before,
            "after": after,
        },
        "runtime": _sionna_runtime_record(),
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
    }
    _write_json_exclusive(output.with_suffix(".manifest.json"), result)
    return output


def _sionna_runtime_record() -> dict:
    import mitsuba

    return {
        "python": os.sys.version.split()[0],
        "sionna": importlib.metadata.version("sionna"),
        "sionna_rt": importlib.metadata.version("sionna-rt"),
        "mitsuba": importlib.metadata.version("mitsuba"),
        "drjit": importlib.metadata.version("drjit"),
        "mitsuba_variant": mitsuba.variant(),
        "sionna_revision": SIONNA_REVISION,
        "cuda_driver_preload": str(SIONNA_CUDA_DRIVER),
        "cuda_driver_sha256": sha256_file(SIONNA_CUDA_DRIVER.resolve()),
        "drjit_libllvm_path": str(SIONNA_LIBLLVM),
        "drjit_libllvm_sha256": sha256_file(SIONNA_LIBLLVM.resolve()),
    }


def _validate_shard_runtime(runtime: object) -> None:
    required = {
        "python",
        "sionna",
        "sionna_rt",
        "mitsuba",
        "drjit",
        "mitsuba_variant",
        "sionna_revision",
        "cuda_driver_preload",
        "cuda_driver_sha256",
        "drjit_libllvm_path",
        "drjit_libllvm_sha256",
    }
    if not isinstance(runtime, dict) or set(runtime) != required:
        raise ValueError("render shard runtime fields must be exact")
    expected = {
        "python": SIONNA_PYTHON_VERSION,
        "sionna": SIONNA_VERSION,
        "sionna_rt": SIONNA_RT_VERSION,
        "mitsuba": MITSUBA_VERSION,
        "drjit": DRJIT_VERSION,
        "mitsuba_variant": "cuda_ad_mono_polarized",
        "sionna_revision": SIONNA_REVISION,
        "cuda_driver_preload": str(SIONNA_CUDA_DRIVER),
        "cuda_driver_sha256": sha256_file(SIONNA_CUDA_DRIVER.resolve()),
        "drjit_libllvm_path": str(SIONNA_LIBLLVM),
        "drjit_libllvm_sha256": sha256_file(SIONNA_LIBLLVM.resolve()),
    }
    if runtime != expected:
        raise ValueError("render shard runtime differs from the frozen Sionna runtime")


def render_bank(bank: dict, asset_root: Path, config: dict, scene_index: int) -> dict[str, np.ndarray]:
    from sionna.rt import (
        ITURadioMaterial,
        PathSolver,
        PlanarArray,
        Receiver,
        Transmitter,
        load_scene,
        subcarrier_frequencies,
    )

    radio = config["radio"]
    worlds = np.asarray(config["world_bits"], dtype=np.int64)
    scene_xml = asset_root / "banks" / str(bank["scene_id"]) / "scene.xml"
    rt_scene = load_scene(scene_xml)
    expected_objects = set(STABLE_SURFACE_IDS)
    if set(rt_scene.objects) != expected_objects:
        raise RuntimeError(
            f"Sionna object catalog mismatch for {bank['scene_id']}: "
            f"expected={sorted(expected_objects)}, actual={sorted(rt_scene.objects)}"
        )
    runtime_to_stable = {
        int(rt_scene.objects[name].object_id): stable_id
        for name, stable_id in STABLE_SURFACE_IDS.items()
    }
    for material_type in ("glass", "metal"):
        name = f"itu_{material_type}"
        if name not in rt_scene.radio_materials:
            rt_scene.add(
                ITURadioMaterial(
                    name=name,
                    itu_type=material_type,
                    thickness=SIONNA_MATERIAL_THICKNESS_M,
                )
            )
    rt_scene.frequency = float(radio["carrier_frequency_hz"])
    bandwidth_hz = float(radio["subcarrier_spacing_hz"]) * int(radio["subcarriers"])
    rt_scene.bandwidth = bandwidth_hz
    rt_scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=int(radio["tx_antennas"]),
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    rt_scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    rt_scene.add(Transmitter("tx", position=[0.0, 0.0, float(radio["transmitter_z_m"])]))
    positions = _receiver_positions(bank, config, scene_index)
    for position_index, xy in enumerate(positions):
        rt_scene.add(
            Receiver(
                f"rx-{position_index}",
                position=[float(xy[0]), float(xy[1]), float(radio["receiver_z_m"])],
            )
        )
    specular = PathSolver()
    frequencies = subcarrier_frequencies(
        int(radio["subcarriers"]), float(radio["subcarrier_spacing_hz"])
    )
    world_count = worlds.shape[0]
    position_count = positions.shape[0]
    channels = 2 * int(radio["tx_antennas"]) * int(radio["subcarriers"])
    max_paths = int(radio["max_stored_paths"])
    max_depth = int(radio["max_depth"])
    clean = np.zeros((world_count, position_count, channels), dtype=np.float64)
    maps = np.zeros((world_count, 3, 256, 256), dtype=np.float64)
    noop_maps = np.zeros_like(maps)
    path_ids = np.full((world_count, position_count, max_paths), -1, dtype=np.int64)
    path_power = np.zeros_like(path_ids, dtype=np.float64)
    path_surfaces = np.full(
        (world_count, position_count, max_paths, max_depth), -1, dtype=np.int64
    )
    noop_ids = np.full_like(path_ids, -1)
    noop_power = np.zeros_like(path_power)
    noop_surfaces = np.full_like(path_surfaces, -1)
    phase_ids, phase_values, phase_sources = _phase_references(bank, positions, config)
    primitive_mapping = _primitive_mapping(scene_index)
    for world_index, bits in enumerate(worlds):
        physical_states = _physical_states(bits, primitive_mapping)
        _set_world_materials(rt_scene, physical_states)
        maps[world_index] = _rasterize_world(bank, physical_states)
        traced = specular(
            rt_scene,
            max_depth=max_depth,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=False,
            synthetic_array=True,
            seed=int(config["seed"]) + scene_index * 1000 + world_index,
        )
        raw_cfr = _flatten_cfr(traced, frequencies, bandwidth_hz, position_count, radio)
        gauge_fixed = raw_cfr * np.conjugate(phase_values)[:, None] / np.abs(phase_values)[:, None]
        clean[world_index] = np.concatenate((gauge_fixed.real, gauge_fixed.imag), axis=1)
        path_ids[world_index], path_power[world_index], path_surfaces[world_index] = _extract_paths(
            traced, runtime_to_stable, position_count, max_paths, max_depth
        )
        noop_maps[world_index] = _rasterize_world(bank, physical_states)
        noop = specular(
            rt_scene,
            max_depth=max_depth,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=False,
            synthetic_array=True,
            seed=int(config["seed"]) + 50_000_000 + scene_index * 1000 + world_index,
        )
        noop_ids[world_index], noop_power[world_index], noop_surfaces[world_index] = _extract_paths(
            noop, runtime_to_stable, position_count, max_paths, max_depth
        )
    repeat_seeds = _repeat_seeds(config, scene_index, world_count, position_count)
    repeated = _add_observation_noise(clean, repeat_seeds, float(radio["observation_noise_std"]))
    free_space = _validate_common_free_space(maps, positions)
    return {
        "csi_repeat": repeated,
        "csi_clean": clean,
        "maps": maps,
        "noop_maps": noop_maps,
        "positions": positions,
        "free_space": free_space,
        "radio_config": np.asarray(
            (
                float(radio["carrier_frequency_hz"]),
                int(radio["tx_antennas"]),
                int(radio["subcarriers"]),
                float(radio["subcarrier_spacing_hz"]),
            ),
            dtype=np.float64,
        ),
        "bs_pose": np.asarray(
            (0.0, 0.0, float(radio["transmitter_z_m"]), 1.0, 0.0, 0.0, 0.0),
            dtype=np.float64,
        ),
        "repeat_seeds": repeat_seeds,
        "phase_reference_ids": phase_ids,
        "phase_reference_values": phase_values,
        "phase_reference_source_sha256": phase_sources,
        "path_ids": path_ids,
        "path_power": path_power,
        "path_surface_ids": path_surfaces,
        "noop_path_ids": noop_ids,
        "noop_path_power": noop_power,
        "noop_path_surface_ids": noop_surfaces,
        "primitive_surface_ids": np.asarray(((102,), (103,)), dtype=np.int64)[primitive_mapping],
        "primitive_ids": primitive_mapping,
        "anchor_bits": _anchor_bits(scene_index),
        "natural_world_index": np.asarray(_natural_world_index(scene_index, worlds), dtype=np.int64),
    }


def _receiver_positions(bank: dict, config: dict, scene_index: int) -> np.ndarray:
    from shapely.geometry import Point, Polygon

    count = int(config["positions_per_bank"])
    polygons = [Polygon(row["local_exterior_xy_m"]) for row in bank["buildings"]]
    primitive_ids = [int(value) for value in bank["primitive_osm_ids"]]
    by_id = {int(row["osm_id"]): Polygon(row["local_exterior_xy_m"]) for row in bank["buildings"]}
    primitives = [by_id[value] for value in primitive_ids]
    offset_x = ((scene_index % 17) - 8) * 0.013
    offset_y = (((scene_index * 7) % 19) - 9) * 0.011
    values = np.linspace(-116.0, 116.0, 31)
    candidates = []
    for y in values + offset_y:
        for x in values + offset_x:
            point = Point(float(x), float(y))
            if all(float(polygon.distance(point)) >= 2.0 for polygon in polygons):
                candidates.append((float(x), float(y)))
    if len(candidates) < count:
        raise RuntimeError(
            f"bank {bank['scene_id']} has only {len(candidates)} receiver-clear grid points"
        )
    remaining = list(range(len(candidates)))
    selected: list[int] = []
    for primitive in primitives:
        ranked = sorted(
            remaining,
            key=lambda index: (
                primitive.distance(Point(*candidates[index])),
                candidates[index][0],
                candidates[index][1],
            ),
        )
        take = ranked[: min(64, count - len(selected))]
        selected.extend(take)
        used = set(take)
        remaining = [index for index in remaining if index not in used]
    rng = np.random.default_rng(int(config["seed"]) + scene_index * 1009)
    remainder = np.asarray(remaining, dtype=np.int64)
    rng.shuffle(remainder)
    selected.extend(int(value) for value in remainder[: count - len(selected)])
    output = np.asarray([candidates[index] for index in selected], dtype=np.float64)
    order = np.arange(count)
    rng.shuffle(order)
    return output[order]


def _primitive_mapping(scene_index: int) -> np.ndarray:
    return np.asarray((0, 1) if scene_index % 2 == 0 else (1, 0), dtype=np.int64)


def _anchor_bits(scene_index: int) -> np.ndarray:
    return np.asarray(((scene_index // 2) % 2, scene_index % 2), dtype=np.int64)


def _natural_world_index(scene_index: int, worlds: np.ndarray) -> int:
    anchor = _anchor_bits(scene_index)
    matches = np.flatnonzero(np.all(worlds == anchor, axis=1))
    if matches.size != 1:
        raise RuntimeError("natural anchor is absent from the world hypercube")
    return int(matches[0])


def _physical_states(bits: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    states = np.zeros(2, dtype=np.int64)
    for bit_index, value in enumerate(bits):
        states[int(mapping[bit_index])] = int(value)
    return states


def _set_world_materials(rt_scene, states: np.ndarray) -> None:
    rt_scene.objects["primitive-0"].radio_material = (
        "itu_concrete" if int(states[0]) == 0 else "itu_glass"
    )
    rt_scene.objects["primitive-1"].radio_material = (
        "itu_wood" if int(states[1]) == 0 else "itu_metal"
    )


def _rasterize_world(bank: dict, states: np.ndarray) -> np.ndarray:
    from shapely import contains_xy
    from shapely.geometry import Polygon

    cells = np.arange(256, dtype=np.float64) - 127.5
    x, y = np.meshgrid(cells, cells, indexing="xy")
    occupancy = np.zeros((256, 256), dtype=np.float64)
    height = np.zeros_like(occupancy)
    material = np.zeros_like(occupancy)
    primitive_ids = [int(value) for value in bank["primitive_osm_ids"]]
    ordered = sorted(
        bank["buildings"],
        key=lambda row: (int(row["osm_id"]) in set(primitive_ids), int(row["osm_id"])),
    )
    for building in ordered:
        polygon = Polygon(building["local_exterior_xy_m"])
        mask = contains_xy(polygon, x, y)
        if not np.any(mask):
            continue
        occupancy[mask] = 1.0
        height[mask] = float(building["height_m"])
        osm_id = int(building["osm_id"])
        if osm_id == primitive_ids[0]:
            category = MATERIAL_CATEGORY["concrete" if int(states[0]) == 0 else "glass"]
        elif osm_id == primitive_ids[1]:
            category = MATERIAL_CATEGORY["wood" if int(states[1]) == 0 else "metal"]
        else:
            category = MATERIAL_CATEGORY["concrete"]
        material[mask] = category
    return np.stack((occupancy, height, material), axis=0)


def _phase_references(bank: dict, positions: np.ndarray, config: dict):
    radio = config["radio"]
    frequency = float(radio["carrier_frequency_hz"])
    tx = np.asarray((0.0, 0.0, float(radio["transmitter_z_m"])), dtype=np.float64)
    identifiers = np.empty(positions.shape[0], dtype="U128")
    references = np.empty(positions.shape[0], dtype=np.complex128)
    source_sha = np.empty(positions.shape[0], dtype="U64")
    speed_of_light = 299_792_458.0
    for index, xy in enumerate(positions):
        receiver = np.asarray((xy[0], xy[1], float(radio["receiver_z_m"])), dtype=np.float64)
        distance = float(np.linalg.norm(receiver - tx))
        reference = np.exp(-1j * 2.0 * np.pi * frequency * distance / speed_of_light)
        record = {
            "schema_version": "csi-pairs-free-space-complex-reference-v1",
            "scene_id": bank["scene_id"],
            "position_index": index,
            "carrier_frequency_hz": frequency,
            "speed_of_light_m_s": speed_of_light,
            "tx_bs_centered_xyz_m": tx.tolist(),
            "receiver_bs_centered_xyz_m": receiver.tolist(),
            "distance_m": distance,
        }
        identifiers[index] = f"free-space-phase:{bank['scene_id']}:p{index:04d}"
        references[index] = reference
        source_sha[index] = _sha256_bytes(canonical_json(record).encode("ascii"))
    return identifiers, references, source_sha


def _flatten_cfr(paths, frequencies, bandwidth_hz, position_count: int, radio: dict) -> np.ndarray:
    values = np.asarray(
        paths.cfr(
            frequencies=frequencies,
            sampling_frequency=bandwidth_hz,
            num_time_steps=1,
            out_type="numpy",
        )
    )
    expected = (
        position_count,
        1,
        1,
        int(radio["tx_antennas"]),
        1,
        int(radio["subcarriers"]),
    )
    if values.shape != expected:
        raise RuntimeError(f"unexpected Sionna CFR shape: expected={expected}, actual={values.shape}")
    return values[:, :, 0, :, 0, :].reshape(position_count, -1)


def _stable_path_id(
    surface_ids: np.ndarray, delay_s: float, interaction_vertices_m: np.ndarray
) -> int:
    vertices = np.asarray(interaction_vertices_m, dtype=np.float64)
    if vertices.shape != (len(surface_ids), 3) or not np.all(np.isfinite(vertices)):
        raise RuntimeError("stable path vertices must be finite [interaction, xyz]")
    quantized_vertices = np.rint(vertices / PATH_VERTEX_QUANTIZATION_M).astype("<i8")
    record = np.concatenate(
        (
            np.asarray(surface_ids, dtype="<i8"),
            np.asarray((int(round(delay_s * 1e12)),), dtype="<i8"),
            quantized_vertices.reshape(-1),
        )
    )
    return int.from_bytes(hashlib.sha256(record.tobytes()).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _extract_paths(
    paths,
    runtime_to_stable: dict[int, int],
    position_count: int,
    max_paths: int,
    max_depth: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.asarray(paths.valid)
    delay = np.asarray(paths.tau)
    objects = np.asarray(paths.objects)
    vertices = np.asarray(paths.vertices)
    real = np.asarray(paths.a[0])
    imag = np.asarray(paths.a[1])
    expected_prefix = (max_depth, position_count, 1)
    if (
        valid.ndim != 3
        or delay.shape != valid.shape
        or objects.shape[:3] != expected_prefix
        or vertices.shape[:3] != expected_prefix
        or vertices.shape[-1] != 3
    ):
        raise RuntimeError(
            f"unexpected Sionna path tensors: valid={valid.shape}, delay={delay.shape}, "
            f"objects={objects.shape}, vertices={vertices.shape}"
        )
    power = np.sum(real * real + imag * imag, axis=(1, 2, 3))
    output_ids = np.full((position_count, max_paths), -1, dtype=np.int64)
    output_power = np.zeros((position_count, max_paths), dtype=np.float64)
    output_surfaces = np.full((position_count, max_paths, max_depth), -1, dtype=np.int64)
    invalid_shape = np.iinfo(np.uint32).max
    for receiver in range(position_count):
        entries = []
        for path_index in np.flatnonzero(valid[receiver, 0]):
            surfaces = np.full(max_depth, -1, dtype=np.int64)
            interaction_vertices = np.zeros((max_depth, 3), dtype=np.float64)
            for depth, raw_value in enumerate(objects[:, receiver, 0, path_index]):
                raw_id = int(raw_value)
                if raw_id == invalid_shape:
                    continue
                if raw_id not in runtime_to_stable:
                    raise RuntimeError(f"unregistered Sionna surface id: {raw_id}")
                surfaces[depth] = runtime_to_stable[raw_id]
                interaction_vertices[depth] = vertices[depth, receiver, 0, path_index]
            entries.append(
                (
                    _stable_path_id(
                        surfaces,
                        float(delay[receiver, 0, path_index]),
                        interaction_vertices,
                    ),
                    float(power[receiver, path_index]),
                    surfaces,
                )
            )
        entries = sorted(entries, key=lambda row: row[1], reverse=True)[:max_paths]
        entries = sorted(entries, key=lambda row: row[0])
        if len({row[0] for row in entries}) != len(entries):
            raise RuntimeError("stable Sionna path identifier collision")
        for output_index, (path_id, path_value, surfaces) in enumerate(entries):
            output_ids[receiver, output_index] = path_id
            output_power[receiver, output_index] = path_value
            output_surfaces[receiver, output_index] = surfaces
    return output_ids, output_power, output_surfaces


def _repeat_seeds(
    config: dict, scene_index: int, world_count: int, position_count: int
) -> np.ndarray:
    repeat_count = int(config["radio"]["repeats"])
    seeds = np.empty((world_count, position_count, repeat_count), dtype=np.int64)
    for world in range(world_count):
        for position in range(position_count):
            for repeat in range(repeat_count):
                seeds[world, position, repeat] = (
                    int(config["seed"])
                    + scene_index * 10_000_000
                    + world * 100_000
                    + position * 100
                    + repeat
                )
    return seeds


def _add_observation_noise(
    clean: np.ndarray, repeat_seeds: np.ndarray, noise_std: float
) -> np.ndarray:
    repeated = np.empty((*clean.shape[:-1], repeat_seeds.shape[-1], clean.shape[-1]))
    for world in range(clean.shape[0]):
        for position in range(clean.shape[1]):
            for repeat in range(repeat_seeds.shape[-1]):
                noise = np.random.default_rng(
                    int(repeat_seeds[world, position, repeat])
                ).normal(0.0, noise_std, size=clean.shape[-1])
                repeated[world, position, repeat] = clean[world, position] + noise
    return repeated


def _validate_common_free_space(maps: np.ndarray, positions: np.ndarray) -> np.ndarray:
    free = np.ones((maps.shape[0], positions.shape[0]), dtype=np.bool_)
    for position_index, xy in enumerate(positions):
        column = int(math.floor(float(xy[0]) + 128.0))
        row = int(math.floor(float(xy[1]) + 128.0))
        if not (0 <= row < 256 and 0 <= column < 256):
            raise RuntimeError("receiver position lies outside the map")
        if np.any(maps[:, 0, row, column] >= 0.5):
            raise RuntimeError("receiver position intersects an occupied map cell")
    return free


def build_engine_config(asset_root: Path, manifest: dict, config: dict) -> dict:
    return {
        "profile": "formal-candidate-sionna-rt-osm-urban-banks-v1",
        "simulation_not_measurement": True,
        "fixture": False,
        "scientific_use": "CANDIDATE",
        "seed": int(config["seed"]),
        "scene_count": len(manifest["banks"]),
        "world_bits": config["world_bits"],
        "positions_per_bank": int(config["positions_per_bank"]),
        "asset_manifest_sha256": sha256_file(asset_root / "asset_manifest.json"),
        "asset_config_sha256": manifest["config_sha256"],
        "source_config_sha256": manifest["source_config_sha256"],
        "raw_osm_sources": [
            {
                "city_id": row["city_id"],
                "sha256": row["sha256"],
                "query_sha256": _sha256_bytes(row["overpass_query"].encode("ascii")),
                "overpass_endpoint": row["overpass_endpoint"],
                "osm_base_timestamp": row["osm_base_timestamp"],
                "license_id": row["license_id"],
            }
            for row in manifest["raw_sources"]
        ],
        "engine": {
            "name": "NVIDIA Sionna RT PathSolver",
            "sionna_revision": SIONNA_REVISION,
            "sionna_rt_version": SIONNA_RT_VERSION,
            "license_id": "Apache-2.0",
        },
        "radio": config["radio"],
        "map": config["map"],
        "interventions": {
            "primitive_0": "OSM building material concrete-to-glass",
            "primitive_1": "OSM building material wood-to-metal",
            "world_rendering": "complete state render for every hypercube node",
            "itu_material_thickness_m": SIONNA_MATERIAL_THICKNESS_M,
        },
        "path_identity": {
            "surface_ids": "ordered stable registered surface identifiers",
            "delay_quantization_s": 1e-12,
            "interaction_vertex_quantization_m": PATH_VERTEX_QUANTIZATION_M,
            "digest": "sha256 truncated to nonnegative signed int64",
        },
        "geometry": {
            "source": "OpenStreetMap closed building ways",
            "license_id": "ODbL-1.0",
            "terrain": "flat_ground_no_terrain_tile_source",
            "building_height_rules": [
                "OSM height tag clamped to [3,120] m",
                "OSM building:levels times 3.2 m clamped to [3,120] m",
                "deterministic OSM-id hash fallback in [9,27] m",
            ],
        },
        "generator": {
            "module_path": "formal_v2/sionna_osm_candidate.py",
            "module_sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def merge_shards(
    config_path: str | Path,
    asset_root: str | Path,
    shard_paths: list[str | Path],
    output_path: str | Path,
) -> Path:
    config = load_config(config_path)
    root, manifest, asset_config = load_asset_manifest(asset_root)
    if canonical_json(config) != canonical_json(asset_config):
        raise ValueError("merge config differs from the frozen asset config")
    rows = []
    shard_manifests = []
    for path_value in shard_paths:
        path = Path(path_value).resolve()
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"render shard must be a regular non-symlink file: {path}")
        shard_manifest = _read_json(path.with_suffix(".manifest.json"))
        if shard_manifest.get("schema_version") != SHARD_SCHEMA:
            raise ValueError("render shard manifest schema mismatch")
        if (
            shard_manifest.get("status") != "PASS"
            or shard_manifest.get("fixture") is not False
            or shard_manifest.get("scientific_use") != "CANDIDATE"
            or shard_manifest.get("simulation_not_measurement") is not True
        ):
            raise ValueError("render shard classification or status is invalid")
        if Path(str(shard_manifest.get("output_path", ""))).resolve() != path:
            raise ValueError("render shard manifest output path mismatch")
        if int(shard_manifest.get("output_bytes", -1)) != path.stat().st_size:
            raise ValueError("render shard manifest byte count mismatch")
        if shard_manifest["output_sha256"] != sha256_file(path):
            raise ValueError("render shard hash mismatch")
        if shard_manifest["asset_manifest_sha256"] != sha256_file(root / "asset_manifest.json"):
            raise ValueError("render shard asset manifest mismatch")
        generator = Path(str(shard_manifest.get("generator_path", ""))).resolve()
        if generator != Path(__file__).resolve() or shard_manifest.get(
            "generator_sha256"
        ) != sha256_file(Path(__file__).resolve()):
            raise ValueError("render shard generator source path/hash mismatch")
        _validate_shard_runtime(shard_manifest.get("runtime"))
        with np.load(path, allow_pickle=False) as archive:
            indices = np.asarray(archive["scene_indices"], dtype=np.int64)
            start = int(shard_manifest.get("scene_start_inclusive", -1))
            end = int(shard_manifest.get("scene_end_exclusive", -1))
            if (
                indices.tolist() != list(range(start, end))
                or int(shard_manifest.get("scene_count", -1)) != indices.size
            ):
                raise ValueError("render shard scene range does not match its archive")
            for local_index, scene_index in enumerate(indices):
                rows.append(
                    (
                        int(scene_index),
                        {
                            name: np.array(archive[name][local_index], copy=True)
                            for name in archive.files
                            if name != "scene_indices"
                        },
                    )
                )
        shard_manifests.append(shard_manifest)
    rows.sort(key=lambda row: row[0])
    if [row[0] for row in rows] != list(range(34)):
        raise ValueError("render shards must cover every scene index 0..33 exactly once")
    gpu_uuids = {row["gpu"]["uuid"] for row in shard_manifests}
    physical_indices = {row["gpu"]["physical_gpu_index"] for row in shard_manifests}
    if len(shard_paths) != 2 or len(gpu_uuids) != 2 or physical_indices != {0, 1}:
        raise ValueError("formal candidate merge requires exactly two distinct GPUs 0 and 1")
    if any(
        row["gpu"].get("cuda_visible_devices") != str(row["gpu"]["physical_gpu_index"])
        for row in shard_manifests
    ):
        raise ValueError("render shard GPU visibility binding is invalid")
    if len({canonical_json(row["runtime"]) for row in shard_manifests}) != 1:
        raise ValueError("render shards used different Sionna runtimes")
    stack = lambda name: np.stack([row[name] for _, row in rows], axis=0)
    arrays = {
        name: stack(name)
        for name in (
            "csi_repeat",
            "csi_clean",
            "maps",
            "positions",
            "free_space",
            "radio_config",
            "bs_pose",
            "repeat_seeds",
            "phase_reference_ids",
            "phase_reference_values",
            "phase_reference_source_sha256",
            "noop_maps",
            "path_ids",
            "path_power",
            "path_surface_ids",
            "noop_path_ids",
            "noop_path_power",
            "noop_path_surface_ids",
            "primitive_surface_ids",
            "primitive_ids",
            "anchor_bits",
            "natural_world_index",
        )
    }
    ledger = expected_scene_ledger(config)
    arrays["map_channel_names"] = MAP_CHANNEL_NAMES
    arrays["world_bits"] = np.asarray(config["world_bits"], dtype=np.int64)
    arrays["scene_ids"] = np.asarray([row["scene_id"] for row in ledger], dtype="U96")
    arrays["city_ids"] = np.asarray([row["city_id"] for row in ledger], dtype="U64")
    arrays["bank_ids"] = np.asarray([row["bank_id"] for row in ledger], dtype="U96")
    arrays["base_map_cluster_ids"] = np.asarray(
        [row["base_map_cluster_id"] for row in ledger], dtype="U96"
    )
    arrays["scene_roles"] = np.asarray([row["role"] for row in ledger], dtype="U40")
    position_count = int(config["positions_per_bank"])
    arrays["position_roles"] = np.full((34, position_count), "standard", dtype="U16")
    arrays["position_ids"] = np.empty((34, position_count), dtype="U160")
    for scene, ledger_row in enumerate(ledger):
        if ledger_row["role"] == "target":
            arrays["position_roles"][scene, ::2] = "support_pool"
            arrays["position_roles"][scene, 1::2] = "query"
        for position, xy in enumerate(arrays["positions"][scene]):
            arrays["position_ids"][scene, position] = (
                f"{ledger_row['city_id']}:{ledger_row['scene_id']}:"
                f"x{float(xy[0]):+.6f}:y{float(xy[1]):+.6f}"
            )
    array_sha = lambda value: hashlib.sha256(
        np.ascontiguousarray(np.asarray(value, dtype="<f8")).tobytes()
    ).hexdigest()
    arrays["canonical_map_sha256"] = np.asarray(
        [[array_sha(arrays["maps"][scene, world]) for world in range(4)] for scene in range(34)],
        dtype="U64",
    )
    arrays["noop_map_sha256"] = np.asarray(
        [
            [array_sha(arrays["noop_maps"][scene, world]) for world in range(4)]
            for scene in range(34)
        ],
        dtype="U64",
    )
    engine_config = build_engine_config(root, manifest, config)
    engine_text = canonical_json(engine_config)
    arrays["engine_config_json"] = np.asarray(engine_text)
    metadata = {
        "schema_version": DATASET_SCHEMA,
        "dataset_id": "CSI-PAIRS-V6-SIONNA-OSM-FORMAL-CANDIDATE",
        "dataset_version": "2026-08-09-v1",
        "scientific_use": "CANDIDATE",
        "fixture": False,
        "engine": {
            "name": "NVIDIA-Sionna-RT-PathSolver",
            "version": SIONNA_RT_VERSION,
            "source_revision": SIONNA_REVISION,
            "license_id": "Apache-2.0",
            "config_sha256": _sha256_bytes(engine_text.encode("ascii")),
            "deterministic": SIONNA_EXACT_REGENERATION_VERIFIED,
        },
        "representation": {
            "csi_layout": "real_then_imag",
            "csi_units": "dimensionless_complex_baseband_channel_coefficient",
            "phase_gauge_rule": "shared_complex_reference",
            "coordinate_system": "bs_centered_right_handed_meters",
            "position_units": "m",
            "map_units": "m",
            "clean_target_definition": "Sionna RT CFR before independent observation noise; exact regeneration not verified",
            "antenna_count": int(config["radio"]["tx_antennas"]),
            "subcarrier_count": int(config["radio"]["subcarriers"]),
            "patch_complex_size": 2,
            "patch_antenna_size": 1,
            "patch_subcarrier_size": 2,
            "map_resolution_m": float(config["map"]["resolution_m"]),
            "map_origin_xy_m": [float(value) for value in config["map"]["origin_xy_m"]],
            "alignment_physical_representation": "complex_csi_plus_delay_angle_power",
        },
        "assets": {
            "license_ids": ["ODbL-1.0"],
            "provenance_uri": "https://www.openstreetmap.org/copyright",
            "material_library": "Sionna-RT-ITU-materials-plus-OSM-building-footprints-v1",
            "material_category_count": len(MATERIAL_CATEGORY),
            "redistribution_allowed": True,
        },
        "generation": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "generator_command": "python -m formal_v2.sionna_osm_candidate merge",
            "seed": int(config["seed"]),
        },
        "external_reference": {
            "available": True,
            "kind": "independent-engine-rerender-required-not-yet-supplied",
            "dataset_id": "PENDING-INDEPENDENT-ENGINE-EVIDENCE",
            "pairing_rule": "same registered external banks, worlds, positions, radio configuration, and edit primitives",
        },
    }
    arrays["metadata_json"] = np.asarray(canonical_json(metadata))
    output = _write_npz_exclusive(output_path, arrays)
    merge_manifest = {
        "schema_version": "csi-pairs-sionna-osm-candidate-generation-v1",
        "status": "PASS",
        "fixture": False,
        "scientific_use": "CANDIDATE",
        "simulation_not_measurement": True,
        "output_path": str(output.resolve()),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "asset_manifest_path": str((root / "asset_manifest.json").resolve()),
        "asset_manifest_sha256": sha256_file(root / "asset_manifest.json"),
        "shards": [
            {
                "path": row["output_path"],
                "sha256": row["output_sha256"],
                "gpu_uuid": row["gpu"]["uuid"],
                "physical_gpu_index": row["gpu"]["physical_gpu_index"],
                "duration_seconds": row["duration_seconds"],
            }
            for row in sorted(shard_manifests, key=lambda value: value["gpu"]["physical_gpu_index"])
        ],
        "engine_config_sha256": metadata["engine"]["config_sha256"],
        "generator_sha256": sha256_file(Path(__file__).resolve()),
    }
    _write_json_exclusive(output.with_suffix(".generation.json"), merge_manifest)
    return output


def regenerate_dataset(dataset_path: str | Path, output_dir: str | Path) -> Path:
    from .formal_dataset import FormalDataset

    dataset = FormalDataset.load(dataset_path)
    if dataset.is_fixture or dataset.metadata["scientific_use"] != "CANDIDATE":
        raise RuntimeError("Sionna OSM verifier only accepts the non-fixture CANDIDATE dataset")
    asset_root = Path(dataset.source_path).parent / "assets"
    root, manifest, config = load_asset_manifest(asset_root)
    expected_engine = build_engine_config(root, manifest, config)
    if expected_engine != dataset.engine_config:
        raise RuntimeError("dataset engine config does not reconstruct from immutable assets")
    rows = []
    for scene_index, bank_row in enumerate(manifest["banks"]):
        bank = _read_json(root / str(bank_row["bank_record_path"]))
        rows.append(render_bank(bank, root, config, scene_index))
        print(
            canonical_json(
                {"event": "independent_regeneration_bank_complete", "scene_index": scene_index}
            ),
            flush=True,
        )
    arrays = {
        "maps": np.stack([row["maps"] for row in rows]),
        "csi_clean": np.stack([row["csi_clean"] for row in rows]),
        "csi_repeat": np.stack([row["csi_repeat"] for row in rows]),
        "free_space": np.stack([row["free_space"] for row in rows]),
        "phase_reference_ids": np.stack([row["phase_reference_ids"] for row in rows]),
        "phase_reference_values": np.stack([row["phase_reference_values"] for row in rows]),
        "phase_reference_source_sha256": np.stack(
            [row["phase_reference_source_sha256"] for row in rows]
        ),
        "engine_config_json": np.asarray(canonical_json(expected_engine)),
        "noop_maps": np.stack([row["noop_maps"] for row in rows]),
        "path_ids": np.stack([row["path_ids"] for row in rows]),
        "path_power": np.stack([row["path_power"] for row in rows]),
        "path_surface_ids": np.stack([row["path_surface_ids"] for row in rows]),
        "noop_path_ids": np.stack([row["noop_path_ids"] for row in rows]),
        "noop_path_power": np.stack([row["noop_path_power"] for row in rows]),
        "noop_path_surface_ids": np.stack([row["noop_path_surface_ids"] for row in rows]),
    }
    if set(arrays) != set(REGENERATED_FIELDS):
        raise RuntimeError("regeneration output fields drifted from the V6 contract")
    return _write_npz_exclusive(Path(output_dir) / "regenerated.npz", arrays)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a non-fixture formal-candidate CSI-PAIRS dataset from frozen OSM assets and Sionna RT"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    assets = commands.add_parser("prepare-assets")
    assets.add_argument("--config", required=True)
    assets.add_argument("--output", required=True)
    assets.add_argument("--raw-cache")
    render = commands.add_parser("render-shard")
    render.add_argument("--asset-root", required=True)
    render.add_argument("--output", required=True)
    render.add_argument("--scene-start", required=True, type=int)
    render.add_argument("--scene-end", required=True, type=int)
    render.add_argument("--physical-gpu-index", required=True, type=int)
    merge = commands.add_parser("merge")
    merge.add_argument("--config", required=True)
    merge.add_argument("--asset-root", required=True)
    merge.add_argument("--shard", action="append", required=True)
    merge.add_argument("--output", required=True)
    regenerate = commands.add_parser("regenerate")
    regenerate.add_argument("--dataset", required=True)
    regenerate.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"prepare-assets", "render-shard", "regenerate"}:
        if argv is not None:
            raise RuntimeError("RT commands require process argv for authenticated re-execution")
        ensure_sionna_runtime(["-m", "formal_v2.sionna_osm_candidate", *sys.argv[1:]])
    if args.command == "prepare-assets":
        output = prepare_assets(args.config, args.output, args.raw_cache)
    elif args.command == "render-shard":
        output = render_shard(
            args.asset_root,
            args.output,
            args.scene_start,
            args.scene_end,
            args.physical_gpu_index,
        )
    elif args.command == "merge":
        output = merge_shards(args.config, args.asset_root, args.shard, args.output)
    else:
        output = regenerate_dataset(args.dataset, args.output)
    print(canonical_json({"status": "PASS", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
