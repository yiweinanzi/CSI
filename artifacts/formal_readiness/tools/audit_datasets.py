#!/usr/bin/env python3
"""Exhaustively inventory local CSI-PAIRS NPZ candidates without mutating them."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_ROOT = REPO_ROOT / "code" / "CSI-PAIRS-v2.0-server"
DATA_ROOT = REPO_ROOT / "generated_datasets"
sys.path.insert(0, str(SERVER_ROOT))

from formal_v2.formal_dataset import FormalDataset, REQUIRED_ARRAYS  # noqa: E402
from formal_v2.formal_io import sha256_file  # noqa: E402


ID_ARRAYS = (
    "scene_ids",
    "city_ids",
    "bank_ids",
    "base_map_cluster_ids",
    "position_ids",
    "phase_reference_ids",
    "phase_reference_source_sha256",
    "canonical_map_sha256",
    "noop_map_sha256",
)


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _numeric_stats(array: np.ndarray) -> dict[str, Any]:
    if np.issubdtype(array.dtype, np.complexfloating):
        finite = np.isfinite(array.real) & np.isfinite(array.imag)
        magnitude = np.abs(array)
        return {
            "finite_count": int(finite.sum()),
            "nonfinite_count": int(array.size - finite.sum()),
            "nan_count": int((np.isnan(array.real) | np.isnan(array.imag)).sum()),
            "inf_count": int((np.isinf(array.real) | np.isinf(array.imag)).sum()),
            "magnitude_min": float(magnitude.min()) if magnitude.size else None,
            "magnitude_max": float(magnitude.max()) if magnitude.size else None,
        }
    if np.issubdtype(array.dtype, np.number):
        finite = np.isfinite(array)
        return {
            "finite_count": int(finite.sum()),
            "nonfinite_count": int(array.size - finite.sum()),
            "nan_count": int(np.isnan(array).sum()) if np.issubdtype(array.dtype, np.inexact) else 0,
            "posinf_count": int(np.isposinf(array).sum()) if np.issubdtype(array.dtype, np.inexact) else 0,
            "neginf_count": int(np.isneginf(array).sum()) if np.issubdtype(array.dtype, np.inexact) else 0,
            "min": float(array.min()) if array.size else None,
            "max": float(array.max()) if array.size else None,
        }
    return {}


def _row_digest(value: np.ndarray, decimals: int | None = None) -> str:
    row = np.asarray(value, dtype="<f8")
    if decimals is not None:
        row = np.round(row, decimals=decimals)
    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()


def _group_for_role(role: str) -> str:
    if role.startswith("source_"):
        return "source"
    if role == "target":
        return "target"
    if role == "external_validation":
        return "external_validation"
    return f"unknown:{role}"


def _set_intersections(values: dict[str, set[str]]) -> dict[str, dict[str, Any]]:
    groups = sorted(values)
    result: dict[str, dict[str, Any]] = {}
    for left_index, left in enumerate(groups):
        for right in groups[left_index + 1 :]:
            overlap = values[left] & values[right]
            result[f"{left}__{right}"] = {
                "count": len(overlap),
                "examples": sorted(overlap)[:10],
            }
    return result


def _formal_analysis(path: Path) -> dict[str, Any]:
    dataset = FormalDataset.load(path)
    dataset.validate(
        require_clean_csi=True,
        minimum_repeats=3,
        minimum_target_cities=2,
        minimum_source_cities=2,
        minimum_banks_per_target_city=2,
        minimum_independent_base_map_clusters_per_target_city=8,
        minimum_banks_per_source_role=2,
        minimum_unique_support_positions_per_target_city=128,
    )

    roles = [str(value) for value in dataset.scene_roles]
    groups = [_group_for_role(role) for role in roles]
    indices_by_group: dict[str, list[int]] = defaultdict(list)
    for scene, group in enumerate(groups):
        indices_by_group[group].append(scene)

    identity_intersections: dict[str, Any] = {}
    identity_sources = {
        "scene_ids": dataset.scene_ids,
        "city_ids": dataset.city_ids,
        "bank_ids": dataset.bank_ids,
        "base_map_cluster_ids": dataset.base_map_cluster_ids,
        "canonical_foundation_sha256": dataset.canonical_base_map_digests,
    }
    for name, array in identity_sources.items():
        grouped = {
            group: {str(array[index]) for index in indices}
            for group, indices in indices_by_group.items()
        }
        identity_intersections[name] = _set_intersections(grouped)

    exact_csi: dict[str, set[str]] = defaultdict(set)
    rounded_csi: dict[str, set[str]] = defaultdict(set)
    exact_observations: dict[str, set[str]] = defaultdict(set)
    for scene, group in enumerate(groups):
        for value in dataset.csi_clean[scene].reshape(-1, dataset.channel_count):
            exact_csi[group].add(_row_digest(value))
            rounded_csi[group].add(_row_digest(value, decimals=10))
        for value in dataset.csi_repeat[scene].reshape(-1, dataset.channel_count):
            exact_observations[group].add(_row_digest(value))

    target_position_isolation: dict[str, Any] = {}
    target_cities = sorted(
        {str(dataset.city_ids[index]) for index in dataset.indices_for_role("target")}
    )
    for city in target_cities:
        support_ids: set[str] = set()
        query_ids: set[str] = set()
        support_coords: set[str] = set()
        query_coords: set[str] = set()
        for scene_value in dataset.indices_for_role("target"):
            scene = int(scene_value)
            if str(dataset.city_ids[scene]) != city:
                continue
            for position in range(dataset.position_count):
                role = str(dataset.position_roles[scene, position])
                identifier = str(dataset.position_ids[scene, position])
                coordinate = ",".join(f"{float(x):.9f}" for x in dataset.positions[scene, position])
                if role == "support_pool":
                    support_ids.add(identifier)
                    support_coords.add(coordinate)
                elif role == "query":
                    query_ids.add(identifier)
                    query_coords.add(coordinate)
        target_position_isolation[city] = {
            "unique_support_ids": len(support_ids),
            "unique_query_ids": len(query_ids),
            "unique_support_coordinates": len(support_coords),
            "unique_query_coordinates": len(query_coords),
            "support_query_id_overlap": len(support_ids & query_ids),
            "support_query_coordinate_overlap": len(support_coords & query_coords),
        }

    role_counts = Counter(roles)
    city_counts = Counter(str(value) for value in dataset.city_ids)
    metadata = dataset.metadata
    return {
        "formal_contract": "PASS",
        "formal_config_thresholds": "PASS",
        "classification": {
            "fixture": dataset.is_fixture,
            "scientific_use": metadata["scientific_use"],
            "data_type": (
                "sionna_rt_simulation_fixture"
                if dataset.is_fixture and "Sionna" in metadata["engine"]["name"]
                else "synthetic_fixture"
                if dataset.is_fixture
                else "sionna_rt_simulation_candidate"
                if "Sionna" in metadata["engine"]["name"]
                else "non_fixture_candidate"
            ),
        },
        "dataset_id": metadata["dataset_id"],
        "dataset_version": metadata["dataset_version"],
        "engine": metadata["engine"],
        "assets": metadata["assets"],
        "generation": metadata["generation"],
        "external_reference": metadata["external_reference"],
        "shape": {
            "scenes": dataset.scene_count,
            "worlds_per_scene": dataset.world_count,
            "positions_per_scene": dataset.position_count,
            "repeats": dataset.repeat_count,
            "channels": dataset.channel_count,
            "clean_scene_world_position_samples": int(
                dataset.scene_count * dataset.world_count * dataset.position_count
            ),
            "repeated_observation_samples": int(
                dataset.scene_count
                * dataset.world_count
                * dataset.position_count
                * dataset.repeat_count
            ),
        },
        "counts": {
            "scene_roles": dict(sorted(role_counts.items())),
            "cities": dict(sorted(city_counts.items())),
            "unique_cities": len(city_counts),
            "unique_banks": len(set(str(value) for value in dataset.bank_ids)),
            "unique_base_map_clusters": len(
                set(str(value) for value in dataset.base_map_cluster_ids)
            ),
            "unique_canonical_foundations": len(
                set(str(value) for value in dataset.canonical_base_map_digests)
            ),
        },
        "missing_labels": {
            name: int(np.sum(np.asarray(getattr(dataset, name)) == ""))
            for name in ID_ARRAYS
            if hasattr(dataset, name)
        },
        "split_identity_intersections": identity_intersections,
        "csi_duplicate_intersections": {
            "clean_exact": _set_intersections(exact_csi),
            "clean_rounded_10_decimals": _set_intersections(rounded_csi),
            "repeat_exact": _set_intersections(exact_observations),
        },
        "target_support_query_isolation": target_position_isolation,
        "range_checks": {
            "negative_path_power": int(np.sum(dataset.path_power < 0)),
            "negative_noop_path_power": int(np.sum(dataset.noop_path_power < 0)),
            "free_space_false": int(np.sum(~dataset.free_space)),
            "repeat_seed_duplicates": int(
                dataset.repeat_seeds.size - np.unique(dataset.repeat_seeds).size
            ),
            "phase_reference_zero": int(np.sum(np.abs(dataset.phase_reference_values) == 0)),
        },
        "contract_report": dataset.contract_report(),
    }


def audit_npz(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        corrupt_member = archive.testzip()
        zip_members = [
            {
                "name": item.filename,
                "compressed_bytes": item.compress_size,
                "uncompressed_bytes": item.file_size,
                "crc32": f"{item.CRC:08x}",
            }
            for item in archive.infolist()
        ]
    arrays: dict[str, Any] = {}
    required_present = False
    with np.load(path, allow_pickle=False) as archive:
        required_present = REQUIRED_ARRAYS.issubset(set(archive.files))
        for name in sorted(archive.files):
            array = np.asarray(archive[name])
            record: dict[str, Any] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "elements": int(array.size),
                "uncompressed_bytes": int(array.nbytes),
                "array_sha256": _array_sha256(array),
            }
            record.update(_numeric_stats(array))
            if array.dtype.kind in "US":
                record["empty_string_count"] = int(np.sum(array == ""))
                record["unique_count"] = len(set(str(value) for value in array.flat))
            arrays[name] = record
    result: dict[str, Any] = {
        "absolute_path": str(path.resolve()),
        "file_bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "zip_integrity": "PASS" if corrupt_member is None else "FAIL",
        "corrupt_zip_member": corrupt_member,
        "zip_member_count": len(zip_members),
        "zip_members": zip_members,
        "all_arrays_scanned": True,
        "required_formal_arrays_present": required_present,
        "arrays": arrays,
    }
    if required_present:
        try:
            result["formal_analysis"] = _formal_analysis(path)
        except Exception as error:
            result["formal_analysis"] = {
                "formal_contract": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
            }
    else:
        result["formal_analysis"] = {
            "formal_contract": "NOT_APPLICABLE",
            "reason": "intermediate shard does not contain every formal archive member",
        }
    return result


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: audit_datasets.py OUTPUT_JSON")
    output = Path(sys.argv[1]).resolve()
    candidates = sorted(DATA_ROOT.rglob("*.npz")) if DATA_ROOT.is_dir() else []
    files = sorted(path for path in DATA_ROOT.rglob("*") if path.is_file()) if DATA_ROOT.is_dir() else []
    report = {
        "schema_version": "csi-pairs-local-dataset-audit-v1",
        "repository_root": str(REPO_ROOT),
        "data_root": str(DATA_ROOT),
        "data_root_exists": DATA_ROOT.is_dir(),
        "total_files": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "npz_count": len(candidates),
        "datasets": [audit_npz(path) for path in candidates],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_path = output.parent / "SHA256SUMS_DATASETS"
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.resolve()}\n" for path in files),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "checksum_manifest": str(checksum_path),
        "files": len(files),
        "npz": len(candidates),
        "bytes": report["total_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
