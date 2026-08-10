#!/usr/bin/env python3
"""Verify the extracted CSI-PAIRS A100 dataset handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


PACKAGE_ID = "CSI-PAIRS-A100-DATASETS-v2"
EXPECTED_EXTERNAL_FILES = 133
EXPECTED_EXTERNAL_GROUPS = {
    "DeepMIMO": 2,
    "UrbanMIMOMap": 121,
    "RadioMapSeer": 5,
    "WWM": 5,
}
EXPECTED_ROLE_COUNTS = {
    "external_public_payload": 133,
    "primary_raw_inputs": 8,
    "cpu_llvm22_34bank_candidate": 266,
    "cpu_same_engine_verification_evidence": 12,
    "a100_sionna_fixture": 9,
    "workspace_metadata": 7,
    "package_control_plane": 20,
}
EXPECTED_A100_FIXTURE_SHA256 = "ae3445739fa3415f7676f544bab28b102459d9819c40028b5ed7e23f7cd5cead"
EXPECTED_OSM_FILES = {
    "external-denver.json",
    "external-miami.json",
    "source-austin.json",
    "source-chicago.json",
    "target-boston.json",
    "target-seattle.json",
}
FORBIDDEN_PATH_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "partial_downloads",
    "qualcomm_wireless_indoor",
}
SOURCE_ROLES = (
    "source_encoder_train",
    "source_method_selection",
    "source_probe_train",
    "source_probe_selection",
    "source_calibration_fit",
    "source_calibration_selection",
    "source_final_unseen_bank",
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> tuple[list[dict], int]:
    require(root.is_dir() and not root.is_symlink(), f"unsafe package root: {root}")
    members = list(root.rglob("*"))
    symlinks = [path.relative_to(root).as_posix() for path in members if path.is_symlink()]
    require(not symlinks, f"symlinks are forbidden in the package: {symlinks[:5]}")

    manifest_path = root / "MANIFEST.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "MANIFEST.json is missing or unsafe")
    manifest = load_json(manifest_path)
    require(manifest.get("schema_version") == "csi-pairs-a100-package-manifest-v1", "bad manifest schema")
    require(manifest.get("package_id") == PACKAGE_ID, "bad package_id")
    require(
        manifest.get("package_distribution") == "INTERNAL_RESEARCH_TEAM_TRANSFER_ONLY",
        "manifest package distribution changed",
    )
    require(
        manifest.get("statuses")
        == {
            "PACKAGE_INTEGRITY": "VERIFY_AFTER_EXTRACTION",
            "TRAINING_CORE_COMPLETE": True,
            "ALL_ORIGINAL_SOURCES_COMPLETE": False,
            "FORMAL_TRAINING_READY": "NO",
            "SCIENTIFIC_EVIDENCE": "NOT_ASSESSED",
        },
        "manifest status boundaries changed",
    )
    entries = manifest.get("entries")
    require(isinstance(entries, list) and entries, "manifest entries are missing")

    listed_paths: list[str] = []
    total_bytes = 0
    for entry in entries:
        require(isinstance(entry, dict), "manifest entry is not an object")
        rel = entry.get("path")
        expected_bytes = entry.get("bytes")
        expected_digest = entry.get("source_sha256")
        require(set(entry) == {"path", "bytes", "role", "source_sha256"}, f"manifest entry fields changed: {rel}")
        require(isinstance(rel, str) and rel, "manifest path is invalid")
        pure = PurePosixPath(rel)
        require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe manifest path: {rel}")
        require(isinstance(expected_bytes, int) and expected_bytes >= 0, f"bad byte count: {rel}")
        require(
            isinstance(expected_digest, str)
            and len(expected_digest) == 64
            and all(char in "0123456789abcdef" for char in expected_digest),
            f"missing or invalid source SHA-256: {rel}",
        )
        require(not (set(pure.parts) & FORBIDDEN_PATH_PARTS), f"forbidden path in package: {rel}")
        require(not rel.endswith((".part", ".partial")), f"partial payload in package: {rel}")
        require(PACKAGE_ID not in pure.parts, f"manifest path must be relative to package root: {rel}")

        path = root.joinpath(*pure.parts)
        require(path.is_file() and not path.is_symlink(), f"missing or unsafe file: {rel}")
        require(path.resolve(strict=True).is_relative_to(root.resolve(strict=True)), f"file escaped package root: {rel}")
        actual_bytes = path.stat().st_size
        require(actual_bytes == expected_bytes, f"size mismatch: {rel}: {actual_bytes} != {expected_bytes}")
        actual_digest = sha256(path)
        require(actual_digest == expected_digest, f"SHA-256 mismatch: {rel}")
        listed_paths.append(rel)
        total_bytes += actual_bytes

    duplicates = [path for path, count in Counter(listed_paths).items() if count != 1]
    require(not duplicates, f"duplicate manifest paths: {duplicates[:5]}")
    require(len(entries) == manifest.get("file_count"), "manifest file_count mismatch")
    require(total_bytes == manifest.get("payload_bytes"), "manifest payload_bytes mismatch")

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    expected_paths = set(listed_paths)
    require(actual_paths == expected_paths, f"inventory mismatch: missing={sorted(expected_paths - actual_paths)[:5]} unexpected={sorted(actual_paths - expected_paths)[:5]}")
    expected_dirs = {
        parent.as_posix()
        for relative in expected_paths | {"MANIFEST.json"}
        for parent in PurePosixPath(relative).parents
        if parent.as_posix() != "."
    }
    actual_dirs = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir()
    }
    require(actual_dirs == expected_dirs, f"directory inventory mismatch: missing={sorted(expected_dirs - actual_dirs)[:5]} unexpected={sorted(actual_dirs - expected_dirs)[:5]}")
    return entries, total_bytes


def verify_external_inventory(entries: list[dict]) -> None:
    prefix = "external_public_datasets/external_wireless/"
    paths = [entry["path"] for entry in entries if entry.get("role") == "external_public_payload"]
    require(len(paths) == EXPECTED_EXTERNAL_FILES, f"external file count is {len(paths)}, expected {EXPECTED_EXTERNAL_FILES}")
    groups = Counter(path.removeprefix(prefix).split("/", 1)[0] for path in paths)
    require(dict(groups) == EXPECTED_EXTERNAL_GROUPS, f"external group counts differ: {dict(groups)}")
    require(all(path.startswith(prefix) for path in paths), "external payload escaped its package root")
    roles = Counter(entry.get("role") for entry in entries)
    require(dict(roles) == EXPECTED_ROLE_COUNTS, f"package role counts differ: {dict(roles)}")


def _split_assignment(city_id: str, bank_index: int, role: str) -> dict[str, object]:
    scene_id = f"osm-sionna-{city_id}-bank-{bank_index:02d}"
    return {
        "scene_index": -1,
        "scene_id": scene_id,
        "city_id": city_id,
        "bank_index_within_city": bank_index,
        "role": role,
        "base_map_cluster_id": f"osm-foundation-{city_id}-{bank_index:02d}",
    }


def expected_split_assignments(ledger: dict) -> list[dict[str, object]]:
    require(
        ledger.get("schema_version") == "csi-pairs-formal-main-split-ledger-v1",
        "bad formal split ledger schema",
    )
    source = ledger.get("source", {})
    source_cities = source.get("cities_in_scene_order")
    source_rows = source.get("bank_index_to_role")
    require(source_cities == ["source-chicago", "source-austin"], "source city split changed")
    require(isinstance(source_rows, list) and len(source_rows) == 7, "seven source roles are required")
    require(tuple(row.get("role") for row in source_rows) == SOURCE_ROLES, "source role order changed")
    require(tuple(row.get("bank_index") for row in source_rows) == tuple(range(7)), "source bank split changed")

    rows: list[dict[str, object]] = []
    for source_row in source_rows:
        for city_id in source_cities:
            rows.append(_split_assignment(city_id, int(source_row["bank_index"]), str(source_row["role"])))

    target = ledger.get("target", {})
    target_cities = target.get("cities_in_scene_order")
    target_indices = target.get("bank_indices")
    require(target_cities == ["target-seattle", "target-boston"], "target city split changed")
    require(target_indices == list(range(8)), "target bank split changed")
    partition = target.get("position_partition", {})
    require(
        partition.get("support_pool", {}).get("index_rule") == "position_index_mod_2_eq_0"
        and partition.get("support_pool", {}).get("positions_per_bank") == 128,
        "target support split changed",
    )
    require(
        partition.get("query", {}).get("index_rule") == "position_index_mod_2_eq_1"
        and partition.get("query", {}).get("positions_per_bank") == 128,
        "target query split changed",
    )
    require(target.get("label_budgets_unique_positions") == [0, 8, 32, 128], "target budgets changed")
    for city_id in target_cities:
        for bank_index in target_indices:
            rows.append(_split_assignment(city_id, int(bank_index), "target"))

    external = ledger.get("external_validation", {})
    external_cities = external.get("cities_in_scene_order")
    external_indices = external.get("bank_indices")
    require(external_cities == ["external-denver", "external-miami"], "external city split changed")
    require(external_indices == [0, 1], "external bank split changed")
    for city_id in external_cities:
        for bank_index in external_indices:
            rows.append(_split_assignment(city_id, int(bank_index), "external_validation"))

    require(len(rows) == 34, "formal split must contain 34 banks")
    for scene_index, row in enumerate(rows):
        row["scene_index"] = scene_index
    return rows


def verify_formal_split_ledger(root: Path, candidate: dict) -> None:
    ledger = load_json(root / "docs/FORMAL_MAIN_SPLIT_LEDGER.json")
    require(
        ledger.get("storage_policy") == "ONE_DATASET_WITH_ROLE_INDEXES_NO_PAYLOAD_DUPLICATION",
        "formal split storage policy changed",
    )
    require(ledger.get("dataset_status") == "CANDIDATE_NOT_CLAIM", "formal split scientific boundary changed")
    require(ledger.get("dataset_sha256") == candidate.get("dataset_sha256"), "formal split dataset hash mismatch")
    expected = expected_split_assignments(ledger)

    asset_manifest = load_json(root / "cpu_llvm22_34bank_candidate/assets/asset_manifest.json")
    observed = [
        {
            "scene_index": row.get("scene_index"),
            "scene_id": row.get("scene_id"),
            "city_id": row.get("city_id"),
            "bank_index_within_city": row.get("bank_index_within_city"),
            "role": row.get("role"),
            "base_map_cluster_id": row.get("base_map_cluster_id"),
        }
        for row in asset_manifest.get("banks", [])
    ]
    require(observed == expected, "formal split ledger differs from candidate assets")
    counts = dict(sorted(Counter(str(row["role"]) for row in expected).items()))
    require(counts == dict(sorted(candidate.get("scene_role_counts", {}).items())), "formal split role counts differ")


def verify_status_boundaries(root: Path) -> None:
    policy = load_json(root / "PACKAGE_POLICY.json")
    readiness = policy.get("readiness", {})
    require(readiness.get("FORMAL_TRAINING_READY") == "NO", "formal training status must remain NO")
    require(readiness.get("SCIENTIFIC_EVIDENCE") == "NOT_ASSESSED", "scientific evidence must remain NOT_ASSESSED")

    registry = load_json(root / "external_dataset_registry/EXTERNAL_DATASET_REGISTRY.json")
    require(registry.get("training_core_complete") is True, "external training core is not complete")
    require(registry.get("all_original_sources_complete") is False, "original-source completeness must remain false")
    require(registry.get("silent_stage0_use") == "FORBIDDEN", "silent Stage-0 use must remain forbidden")

    licenses = load_json(root / "docs/LICENSE_STATUS.json")
    require(licenses.get("package_distribution") == "INTERNAL_RESEARCH_TEAM_TRANSFER_ONLY", "package distribution boundary changed")
    require(licenses.get("public_redistribution") == "FORBIDDEN_PENDING_DATASET_LEVEL_REVIEW", "license boundary changed")

    split_policy = load_json(root / "docs/SPLIT_POLICY.json")
    require(split_policy.get("assignment_status") == "BLOCKED_NOT_FROZEN", "external split status changed")
    require(split_policy.get("paper_experiment_use") == "FORBIDDEN_UNTIL_IMMUTABLE_ASSIGNMENT_LEDGER", "external paper-use split gate changed")

    candidate = load_json(root / "cpu_llvm22_34bank_candidate/inspect-v4/data_contract.json")
    require(candidate.get("status") == "PASS", "CPU candidate contract did not pass")
    require(candidate.get("fixture") is False, "CPU candidate incorrectly marked as fixture")
    require(candidate.get("scientific_use") == "CANDIDATE", "CPU source status changed")
    expected_shape = {"scenes": 34, "worlds": 4, "positions": 256, "repeats": 3, "channels": 16, "bits": 2}
    shape = candidate.get("shape", {})
    require(all(shape.get(key) == value for key, value in expected_shape.items()), f"CPU candidate shape changed: {shape}")
    candidate_dataset = root / "cpu_llvm22_34bank_candidate/dataset.npz"
    require(candidate_dataset.is_file(), "CPU candidate dataset.npz is missing")
    require(sha256(candidate_dataset) == candidate.get("dataset_sha256"), "CPU candidate dataset hash mismatch")
    verify_formal_split_ledger(root, candidate)

    gate = load_json(root / "cpu_same_engine_verification_evidence/live_regeneration/data_verification/gate.json")
    require(gate.get("status") == "PASS" and gate.get("passed") is True, "same-engine gate did not pass")
    require(gate.get("scientific_use") == "CANDIDATE_NOT_CLAIM", "same-engine evidence boundary changed")
    require(gate.get("rtol") == 0 and gate.get("atol") == 0, "same-engine tolerances are not zero")

    fixture = load_json(root / "a100_sionna_fixture/csi_pairs_v2_1_v6_sionna_rt_dual_a100/generation_manifest.json")
    fixture_dataset = fixture.get("dataset", {})
    require(fixture_dataset.get("fixture") is True, "A100 fixture marker is missing")
    require(fixture_dataset.get("scientific_use") == "FORBIDDEN", "A100 fixture scientific boundary changed")
    require(fixture.get("validation", {}).get("contract_status") == "PASS", "A100 fixture contract did not pass")
    fixture_path = root / "a100_sionna_fixture/csi_pairs_v2_1_v6_sionna_rt_dual_a100/csi_pairs_v2_1_v6_sionna_rt_dual_a100.npz"
    require(fixture_path.is_file(), "A100 fixture NPZ is missing")
    require(sha256(fixture_path) == EXPECTED_A100_FIXTURE_SHA256, "A100 fixture dataset hash mismatch")

    osm_root = root / "primary_raw_inputs/CSI-PAIRS-A100-input-v2/raw_osm"
    osm_files = {path.name for path in osm_root.glob("*.json") if path.is_file()}
    require(osm_files == EXPECTED_OSM_FILES, f"six-city OSM inputs differ: {sorted(osm_files)}")


def probe_zip_containers(root: Path, entries: list[dict]) -> tuple[int, int]:
    zip_count = 0
    npz_count = 0
    for entry in entries:
        rel = entry["path"]
        if not rel.endswith((".zip", ".npz")):
            continue
        path = root / rel
        try:
            with zipfile.ZipFile(path) as archive:
                require(bool(archive.infolist()), f"empty ZIP/NPZ container: {rel}")
        except (OSError, zipfile.BadZipFile) as error:
            raise VerificationError(f"unreadable ZIP/NPZ container {rel}: {error}") from error
        if rel.endswith(".npz"):
            npz_count += 1
        else:
            zip_count += 1
    return zip_count, npz_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep-hash", action="store_true", help="compatibility flag; full hashing is always mandatory")
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    try:
        entries, total_bytes = verify_manifest(root)
        verify_external_inventory(entries)
        verify_status_boundaries(root)
        zip_count, npz_count = probe_zip_containers(root, entries)
    except VerificationError as error:
        print(f"PACKAGE_VERIFY=FAIL reason={error}", file=sys.stderr)
        return 1

    print("PACKAGE_VERIFY=PASS")
    print(f"PACKAGE_ID={PACKAGE_ID}")
    print(f"MANIFEST_FILES={len(entries)}")
    print(f"PAYLOAD_BYTES={total_bytes}")
    print(f"EXTERNAL_REGISTERED_FILES={EXPECTED_EXTERNAL_FILES}")
    print(f"ZIP_CONTAINERS_OPENED={zip_count}")
    print(f"NPZ_CONTAINERS_OPENED={npz_count}")
    print("FORMAL_SPLIT_LEDGER=PASS")
    print("SOURCE_PERMISSION_ROLES=7")
    print("TARGET_BANKS=16")
    print("TARGET_SUPPORT_POSITIONS_PER_BANK=128")
    print("TARGET_QUERY_POSITIONS_PER_BANK=128")
    print(f"SHA256_FILES_CHECKED={len(entries)}")
    print("DEEP_HASH=PASS")
    print("TRAINING_CORE_COMPLETE=true")
    print("ALL_ORIGINAL_SOURCES_COMPLETE=false")
    print("A100_PACKAGE_ENGINEERING_READY=YES")
    print("FORMAL_TRAINING_READY=NO")
    print("SCIENTIFIC_EVIDENCE=NOT_ASSESSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
