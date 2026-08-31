#!/usr/bin/env python3
"""Strictly compare two independently rendered diagnostic Sionna banks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


REQUIRED_GROUPS = {
    "csi": {"csi_clean", "csi_repeat"},
    "visibility_sources": {
        "path_ids", "path_power", "noop_path_ids", "noop_path_power"
    },
    "path_identity": {
        "path_ids", "path_power", "path_surface_ids",
        "noop_path_ids", "noop_path_power", "noop_path_surface_ids",
    },
    "surface_material_intervention": {
        "maps", "noop_maps", "primitive_surface_ids", "primitive_ids",
        "anchor_bits", "natural_world_index",
    },
    "phase_reference": {
        "phase_reference_ids", "phase_reference_values",
        "phase_reference_source_sha256",
    },
    "scene_world_position_repeat_identity": {
        "scene_indices", "positions", "repeat_seeds", "bs_pose", "radio_config"
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _zip_metadata(path: Path) -> list[dict[str, object]]:
    with zipfile.ZipFile(path) as archive:
        return [
            {
                "filename": row.filename,
                "date_time": list(row.date_time),
                "compress_type": row.compress_type,
                "crc": row.CRC,
                "compressed_bytes": row.compress_size,
                "uncompressed_bytes": row.file_size,
            }
            for row in archive.infolist()
        ]


def _array_report(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    same_shape = left.shape == right.shape
    same_dtype = left.dtype == right.dtype
    left_finite = bool(np.all(np.isfinite(left))) if np.issubdtype(left.dtype, np.number) else None
    right_finite = bool(np.all(np.isfinite(right))) if np.issubdtype(right.dtype, np.number) else None
    exact_values = bool(np.array_equal(left, right, equal_nan=False)) if same_shape else False
    exact = same_shape and same_dtype and exact_values
    report: dict[str, object] = {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "shape_exact": same_shape,
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "dtype_exact": same_dtype,
        "left_finite": left_finite,
        "right_finite": right_finite,
        "values_exact": exact_values,
        "exact": exact,
    }
    if same_shape and np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
        difference = np.abs(left - right)
        report["max_abs_diff"] = (
            float(np.max(difference)) if difference.size and np.all(np.isfinite(difference)) else None
        )
        report["mismatch_count"] = int(np.count_nonzero(left != right))
    return report


def _visibility(archive: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    path_ids = archive[f"{prefix}path_ids"]
    path_power = archive[f"{prefix}path_power"]
    return np.any((path_ids >= 0) & (path_power > 0), axis=-1)


def compare(left_path: Path, right_path: Path) -> dict[str, object]:
    left = _load(left_path)
    right = _load(right_path)
    left_fields = set(left)
    right_fields = set(right)
    common = sorted(left_fields & right_fields)
    arrays = {name: _array_report(left[name], right[name]) for name in common}
    required = set().union(*REQUIRED_GROUPS.values())
    groups = {
        name: {
            "required_fields": sorted(fields),
            "present_a": fields <= left_fields,
            "present_b": fields <= right_fields,
            "exact": fields <= set(common) and all(arrays[field]["exact"] for field in fields),
        }
        for name, fields in REQUIRED_GROUPS.items()
    }
    derived_visibility = {}
    for prefix, label in (("", "registered"), ("noop_", "null")):
        a_visibility = _visibility(left, prefix)
        b_visibility = _visibility(right, prefix)
        derived_visibility[label] = _array_report(a_visibility, b_visibility)
    finite_ok = all(
        row["left_finite"] is not False and row["right_finite"] is not False
        for row in arrays.values()
    )
    fields_exact = left_fields == right_fields and all(row["exact"] for row in arrays.values())
    sha_a = _sha256(left_path)
    sha_b = _sha256(right_path)
    zip_a = _zip_metadata(left_path)
    zip_b = _zip_metadata(right_path)
    zip_payload_equal = [
        (row["filename"], row["crc"], row["uncompressed_bytes"]) for row in zip_a
    ] == [
        (row["filename"], row["crc"], row["uncompressed_bytes"]) for row in zip_b
    ]
    status = "PASS" if (
        required <= left_fields
        and required <= right_fields
        and fields_exact
        and finite_ok
        and all(row["exact"] for row in derived_visibility.values())
    ) else "FAIL"
    return {
        "schema_version": "csi-pairs-m4-single-bank-exact-comparison-v1",
        "status": status,
        "simulation_not_measurement": True,
        "scientific_use": "DIAGNOSTIC_NOT_FORMAL_EVIDENCE",
        "rtol": 0,
        "atol": 0,
        "equal_nan": False,
        "process_a": {
            "path": str(left_path.resolve()), "bytes": left_path.stat().st_size, "sha256": sha_a
        },
        "process_b": {
            "path": str(right_path.resolve()), "bytes": right_path.stat().st_size, "sha256": sha_b
        },
        "field_sets_exact": left_fields == right_fields,
        "fields_only_in_a": sorted(left_fields - right_fields),
        "fields_only_in_b": sorted(right_fields - left_fields),
        "required_fields": sorted(required),
        "all_required_fields_present": required <= left_fields and required <= right_fields,
        "all_required_fields_exact": all(groups[name]["exact"] for name in groups),
        "exact_field_count": sum(bool(row["exact"]) for row in arrays.values()),
        "total_field_count": len(left_fields | right_fields),
        "finite_all_numeric_fields": finite_ok,
        "max_csi_abs_diff": arrays.get("csi_clean", {}).get("max_abs_diff"),
        "output_sha_identical": sha_a == sha_b,
        "zip_payload_crc_and_size_identical": zip_payload_equal,
        "zip_container_metadata_identical": zip_a == zip_b,
        "container_difference": (
            "NONE" if sha_a == sha_b else
            "ZIP_METADATA_ONLY" if fields_exact and zip_payload_equal else
            "ARRAY_OR_PAYLOAD_DIFFERENCE"
        ),
        "groups": groups,
        "derived_visibility": derived_visibility,
        "arrays": arrays,
        "zip_metadata_a": zip_a,
        "zip_metadata_b": zip_b,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process-a", type=Path, required=True)
    parser.add_argument("--process-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = compare(args.process_a.resolve(), args.process_b.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="ascii") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"]}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
