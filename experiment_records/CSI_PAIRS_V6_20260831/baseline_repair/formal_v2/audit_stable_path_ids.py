#!/usr/bin/env python3
"""Audit captured frozen Sionna path signatures within and across processes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _signature_maps(payload: dict) -> tuple[dict[int, set[str]], dict[str, set[int]]]:
    by_id: dict[int, set[str]] = defaultdict(set)
    by_signature: dict[str, set[int]] = defaultdict(set)
    for row in payload["records"]:
        path_id = int(row["path_id"])
        signature = str(row["canonical_record_sha256"])
        by_id[path_id].add(signature)
        by_signature[signature].add(path_id)
    return by_id, by_signature


def _material_for_surface(surface: int, states: np.ndarray) -> str:
    if surface < 0:
        return "none"
    if surface == 100:
        return "itu_wet_ground"
    if surface == 101:
        return "itu_concrete"
    if surface == 102:
        return "itu_concrete" if int(states[0]) == 0 else "itu_glass"
    if surface == 103:
        return "itu_wood" if int(states[1]) == 0 else "itu_metal"
    return f"unregistered:{surface}"


def _output_associations(
    archive: dict[str, np.ndarray], world_bits: np.ndarray
) -> dict[str, object]:
    id_surfaces: dict[int, set[tuple[int, ...]]] = defaultdict(set)
    id_materials: dict[int, set[tuple[str, ...]]] = defaultdict(set)
    id_power_bits: dict[int, set[str]] = defaultdict(set)
    context_slots: dict[tuple[str, int, int, int], list[tuple[tuple[int, ...], str]]] = defaultdict(list)
    all_ids: list[int] = []
    mapping = np.asarray(archive["primitive_ids"])[0]
    for prefix in ("", "noop_"):
        ids = np.asarray(archive[f"{prefix}path_ids"])[0]
        powers = np.asarray(archive[f"{prefix}path_power"])[0]
        surfaces = np.asarray(archive[f"{prefix}path_surface_ids"])[0]
        for world in range(ids.shape[0]):
            states = np.zeros(2, dtype=np.int64)
            for bit_index, bit in enumerate(world_bits[world]):
                states[int(mapping[bit_index])] = int(bit)
            for position in range(ids.shape[1]):
                for slot in np.flatnonzero(ids[world, position] >= 0):
                    path_id = int(ids[world, position, slot])
                    surface_row = tuple(int(value) for value in surfaces[world, position, slot])
                    material_row = tuple(
                        _material_for_surface(value, states) for value in surface_row
                    )
                    power_hex = float(powers[world, position, slot]).hex()
                    all_ids.append(path_id)
                    id_surfaces[path_id].add(surface_row)
                    id_materials[path_id].add(material_row)
                    id_power_bits[path_id].add(power_hex)
                    context_slots[(prefix or "registered", world, position, path_id)].append(
                        (surface_row, power_hex)
                    )
    context_conflicts = {
        key: values for key, values in context_slots.items()
        if len(values) > 1 and len(set(values)) > 1
    }
    counts = Counter(all_ids)
    return {
        "valid_slot_count": len(all_ids),
        "unique_path_id_count": len(counts),
        "repeated_path_id_count": sum(count > 1 for count in counts.values()),
        "repeated_path_slot_count": sum(count - 1 for count in counts.values() if count > 1),
        "id_to_multiple_surface_sequences": sum(len(rows) > 1 for rows in id_surfaces.values()),
        "id_to_multiple_material_sequences": sum(len(rows) > 1 for rows in id_materials.values()),
        "id_to_multiple_power_records": sum(len(rows) > 1 for rows in id_power_bits.values()),
        "same_context_conflict_count": len(context_conflicts),
        "same_context_conflicts": [
            {"context": list(key), "records": [list(row) for row in values]}
            for key, values in list(sorted(context_conflicts.items()))[:10]
        ],
    }


def audit(
    process_a: Path,
    process_b: Path,
    signatures_a: Path,
    signatures_b: Path,
    generator_config: Path,
) -> dict[str, object]:
    archive_a = _load_npz(process_a)
    archive_b = _load_npz(process_b)
    payload_a = _read_json(signatures_a)
    payload_b = _read_json(signatures_b)
    config = _read_json(generator_config)
    world_bits = np.asarray(config["world_bits"], dtype=np.int64)
    id_to_sig_a, sig_to_id_a = _signature_maps(payload_a)
    id_to_sig_b, sig_to_id_b = _signature_maps(payload_b)
    id_collisions_a = {key: value for key, value in id_to_sig_a.items() if len(value) > 1}
    id_collisions_b = {key: value for key, value in id_to_sig_b.items() if len(value) > 1}
    signature_collisions_a = {key: value for key, value in sig_to_id_a.items() if len(value) > 1}
    signature_collisions_b = {key: value for key, value in sig_to_id_b.items() if len(value) > 1}
    shared_signatures = set(sig_to_id_a) & set(sig_to_id_b)
    cross_signature_mismatches = {
        signature: {"process_a_ids": sorted(sig_to_id_a[signature]), "process_b_ids": sorted(sig_to_id_b[signature])}
        for signature in shared_signatures
        if sig_to_id_a[signature] != sig_to_id_b[signature]
    }
    shared_ids = set(id_to_sig_a) & set(id_to_sig_b)
    cross_id_mismatches = {
        str(path_id): {
            "process_a_signatures": sorted(id_to_sig_a[path_id]),
            "process_b_signatures": sorted(id_to_sig_b[path_id]),
        }
        for path_id in shared_ids
        if id_to_sig_a[path_id] != id_to_sig_b[path_id]
    }
    association_a = _output_associations(archive_a, world_bits)
    association_b = _output_associations(archive_b, world_bits)
    first_conflicts = []
    for label, rows in (
        ("id_collision_a", id_collisions_a),
        ("id_collision_b", id_collisions_b),
        ("signature_collision_a", signature_collisions_a),
        ("signature_collision_b", signature_collisions_b),
        ("cross_signature_mismatch", cross_signature_mismatches),
        ("cross_id_mismatch", cross_id_mismatches),
    ):
        for key, value in sorted(rows.items(), key=lambda row: str(row[0])):
            first_conflicts.append({"kind": label, "key": str(key), "value": sorted(value) if isinstance(value, set) else value})
            if len(first_conflicts) == 10:
                break
        if len(first_conflicts) == 10:
            break
    conflict_count = (
        len(id_collisions_a) + len(id_collisions_b)
        + len(signature_collisions_a) + len(signature_collisions_b)
        + len(cross_signature_mismatches) + len(cross_id_mismatches)
        + int(association_a["same_context_conflict_count"])
        + int(association_b["same_context_conflict_count"])
    )
    return {
        "schema_version": "csi-pairs-m4-stable-path-id-audit-v1",
        "status": "PASS" if conflict_count == 0 else "FAIL",
        "simulation_not_measurement": True,
        "scientific_use": "DIAGNOSTIC_NOT_FORMAL_EVIDENCE",
        "canonical_definition_source_a": payload_a["canonical_definition_source"],
        "canonical_definition_sha256_a": payload_a["canonical_definition_sha256"],
        "canonical_definition_source_b": payload_b["canonical_definition_source"],
        "canonical_definition_sha256_b": payload_b["canonical_definition_sha256"],
        "PATH_ID_TOTAL_A": association_a["valid_slot_count"],
        "PATH_ID_TOTAL_B": association_b["valid_slot_count"],
        "PATH_ID_UNIQUE_A": association_a["unique_path_id_count"],
        "PATH_ID_UNIQUE_B": association_b["unique_path_id_count"],
        "PATH_ID_DUPLICATES_A": len(id_collisions_a),
        "PATH_ID_DUPLICATES_B": len(id_collisions_b),
        "ID_TO_MULTIPLE_SIGNATURES": len(set(id_collisions_a) | set(id_collisions_b)),
        "SIGNATURE_TO_MULTIPLE_IDS": len(set(signature_collisions_a) | set(signature_collisions_b)),
        "CROSS_PROCESS_ID_MISMATCHES": len(cross_signature_mismatches) + len(cross_id_mismatches),
        "FIRST_10_CONFLICTS": first_conflicts,
        "process_a_associations": association_a,
        "process_b_associations": association_b,
        "id_collisions_a": {str(key): sorted(value) for key, value in id_collisions_a.items()},
        "id_collisions_b": {str(key): sorted(value) for key, value in id_collisions_b.items()},
        "signature_collisions_a": {key: sorted(value) for key, value in signature_collisions_a.items()},
        "signature_collisions_b": {key: sorted(value) for key, value in signature_collisions_b.items()},
        "cross_signature_mismatches": cross_signature_mismatches,
        "cross_id_mismatches": cross_id_mismatches,
        "interpretation": {
            "canonical_conflict_rule": "only frozen signature-to-ID ambiguity or same-context descriptor ambiguity fails the stable-ID gate",
            "material_and_power_variants": "reported separately because frozen v1 identity is geometry-based and excludes material state and power",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process-a", type=Path, required=True)
    parser.add_argument("--process-b", type=Path, required=True)
    parser.add_argument("--signatures-a", type=Path, required=True)
    parser.add_argument("--signatures-b", type=Path, required=True)
    parser.add_argument("--generator-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = audit(
        args.process_a.resolve(), args.process_b.resolve(),
        args.signatures_a.resolve(), args.signatures_b.resolve(),
        args.generator_config.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="ascii") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"]}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
