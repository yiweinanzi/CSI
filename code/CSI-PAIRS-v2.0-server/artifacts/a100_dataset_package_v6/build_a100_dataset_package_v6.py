#!/usr/bin/env python3
"""Build the internal CSI-PAIRS Linux/A100 dataset handoff as a Zip64 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


BUFFER_BYTES = 16 * 1024 * 1024
ZIP_TIMESTAMP = (2026, 8, 10, 0, 0, 0)
PACKAGE_ID = "CSI-PAIRS-A100-DATASETS-v2"
SOURCE_ROLES = (
    "source_encoder_train",
    "source_method_selection",
    "source_probe_train",
    "source_probe_selection",
    "source_calibration_fit",
    "source_calibration_selection",
    "source_final_unseen_bank",
)


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class PackageEntry:
    source: Path
    archive_path: str
    role: str
    bytes: int
    mode: int
    source_sha256: str | None = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BuildError(f"cannot read JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def safe_relative(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    require(value and not path.is_absolute() and ".." not in path.parts, f"unsafe {label}: {value}")
    return path


def entry_for(
    source: Path,
    archive_path: str,
    role: str,
    *,
    executable: bool = False,
    source_sha256: str | None = None,
) -> PackageEntry:
    require(source.is_file(), f"missing source file: {source}")
    safe_relative(archive_path, "archive path")
    observed_sha256 = sha256(source)
    if source_sha256 is not None:
        require(
            observed_sha256 == source_sha256,
            f"source SHA-256 mismatch: {source}",
        )
    mode = 0o755 if executable else 0o644
    return PackageEntry(
        source=source,
        archive_path=archive_path,
        role=role,
        bytes=source.stat().st_size,
        mode=mode,
        source_sha256=observed_sha256,
    )


def parse_checksum_manifest(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, f"bad checksum line {line_number}: {path}")
        digest, relative = fields
        require(len(digest) == 64 and all(char in "0123456789abcdef" for char in digest), f"bad SHA-256 at line {line_number}: {path}")
        safe_relative(relative, "checksum path")
        records.append((digest, relative))
    paths = [relative for _, relative in records]
    require(len(paths) == len(set(paths)), f"duplicate checksum paths: {path}")
    return records


def excluded(relative: PurePosixPath, exact_exclusions: set[str], global_exclusions: set[str]) -> bool:
    value = relative.as_posix()
    if value in exact_exclusions:
        return True
    if set(relative.parts) & global_exclusions:
        return True
    return value.endswith((".part", ".partial"))


def add_tree(entries: list[PackageEntry], source_root: Path, destination: str, role: str, exclusions: set[str], global_exclusions: set[str]) -> None:
    require(source_root.is_dir(), f"missing source directory: {source_root}")
    for source in sorted(source_root.rglob("*")):
        if source.is_symlink():
            raise BuildError(f"symlink is not allowed in component {role}: {source}")
        if not source.is_file():
            continue
        relative = PurePosixPath(source.relative_to(source_root).as_posix())
        if excluded(relative, exclusions, global_exclusions):
            continue
        archive_path = f"{destination.rstrip('/')}/{relative.as_posix()}"
        executable = bool(source.stat().st_mode & 0o111)
        entries.append(entry_for(source, archive_path, role, executable=executable))


def validate_external_groups(config: dict, records: list[tuple[str, str]]) -> None:
    expected_count = config["external_payload"]["expected_file_count"]
    require(len(records) == expected_count, f"external manifest has {len(records)} files, expected {expected_count}")
    for group in config["external_payload"]["expected_groups"]:
        count = sum(relative.startswith(group["source_prefix"]) for _, relative in records)
        require(count == group["expected_file_count"], f"{group['group_id']} has {count} files, expected {group['expected_file_count']}")


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
        "unexpected formal split ledger schema",
    )
    source = ledger.get("source", {})
    source_cities = source.get("cities_in_scene_order")
    source_rows = source.get("bank_index_to_role")
    require(
        source_cities == ["source-chicago", "source-austin"],
        "formal split source-city order changed",
    )
    require(isinstance(source_rows, list) and len(source_rows) == 7, "formal split requires seven source roles")
    observed_roles = tuple(row.get("role") for row in source_rows if isinstance(row, dict))
    observed_indices = tuple(row.get("bank_index") for row in source_rows if isinstance(row, dict))
    require(observed_roles == SOURCE_ROLES, "formal source-role order changed")
    require(observed_indices == tuple(range(7)), "formal source bank indices changed")
    require(source.get("expected_banks_per_role") == 2, "each source role must contain two banks")

    rows: list[dict[str, object]] = []
    for source_row in source_rows:
        require(
            isinstance(source_row.get("allowed"), list)
            and bool(source_row["allowed"])
            and isinstance(source_row.get("forbidden"), list)
            and bool(source_row["forbidden"]),
            f"source permission record is incomplete: {source_row.get('role')}",
        )
        for city_id in source_cities:
            rows.append(_split_assignment(city_id, int(source_row["bank_index"]), str(source_row["role"])))

    target = ledger.get("target", {})
    target_cities = target.get("cities_in_scene_order")
    target_indices = target.get("bank_indices")
    require(
        target_cities == ["target-seattle", "target-boston"],
        "formal split target-city order changed",
    )
    require(target_indices == list(range(8)), "formal target bank indices changed")
    require(target.get("scene_role") == "target", "formal target role changed")
    require(target.get("positions_per_bank") == 256, "formal target position count changed")
    position_partition = target.get("position_partition", {})
    require(
        position_partition.get("support_pool", {}).get("index_rule") == "position_index_mod_2_eq_0"
        and position_partition.get("support_pool", {}).get("positions_per_bank") == 128,
        "formal target support_pool rule changed",
    )
    require(
        position_partition.get("query", {}).get("index_rule") == "position_index_mod_2_eq_1"
        and position_partition.get("query", {}).get("positions_per_bank") == 128,
        "formal target query rule changed",
    )
    require(
        target.get("label_budgets_unique_positions") == [0, 8, 32, 128],
        "formal target label budgets changed",
    )
    for city_id in target_cities:
        for bank_index in target_indices:
            rows.append(_split_assignment(city_id, int(bank_index), "target"))

    external = ledger.get("external_validation", {})
    external_cities = external.get("cities_in_scene_order")
    external_indices = external.get("bank_indices")
    require(
        external_cities == ["external-denver", "external-miami"],
        "formal split external-city order changed",
    )
    require(external_indices == [0, 1], "formal external bank indices changed")
    require(external.get("scene_role") == "external_validation", "external validation role changed")
    for city_id in external_cities:
        for bank_index in external_indices:
            rows.append(_split_assignment(city_id, int(bank_index), "external_validation"))

    require(len(rows) == 34, f"formal split expands to {len(rows)} banks, expected 34")
    for scene_index, row in enumerate(rows):
        row["scene_index"] = scene_index
    return rows


def validate_formal_split_ledger(workspace: Path, repo_root: Path, candidate_contract: dict) -> None:
    ledger = load_json(repo_root / "artifacts/a100_dataset_package_v6/FORMAL_MAIN_SPLIT_LEDGER.json")
    require(
        ledger.get("storage_policy") == "ONE_DATASET_WITH_ROLE_INDEXES_NO_PAYLOAD_DUPLICATION",
        "formal split storage policy changed",
    )
    require(
        ledger.get("dataset_sha256") == candidate_contract.get("dataset_sha256"),
        "formal split ledger is not bound to the CPU candidate",
    )
    require(ledger.get("dataset_status") == "CANDIDATE_NOT_CLAIM", "formal split candidate boundary changed")
    expected = expected_split_assignments(ledger)

    asset_manifest = load_json(
        workspace
        / "deliveries/CSI-PAIRS-2xA100-FORMAL-HANDOFF-c1b4419d/formal_candidate/assets/asset_manifest.json"
    )
    observed = []
    for row in asset_manifest.get("banks", []):
        observed.append(
            {
                "scene_index": row.get("scene_index"),
                "scene_id": row.get("scene_id"),
                "city_id": row.get("city_id"),
                "bank_index_within_city": row.get("bank_index_within_city"),
                "role": row.get("role"),
                "base_map_cluster_id": row.get("base_map_cluster_id"),
            }
        )
    require(observed == expected, "formal split ledger does not match the candidate asset manifest")

    expected_counts = dict(sorted(Counter(str(row["role"]) for row in expected).items()))
    contract_counts = dict(sorted(candidate_contract.get("scene_role_counts", {}).items()))
    require(expected_counts == contract_counts, "formal split role counts do not match inspected candidate")
    require(
        set(candidate_contract.get("target_cities", []))
        == set(ledger["target"]["cities_in_scene_order"]),
        "formal split target cities do not match inspected candidate",
    )


def validate_scientific_boundaries(workspace: Path, repo_root: Path) -> None:
    candidate_contract = load_json(
        workspace
        / "deliveries/CSI-PAIRS-2xA100-FORMAL-HANDOFF-c1b4419d/formal_candidate/inspect-v4/data_contract.json"
    )
    expected_shape = {"scenes": 34, "worlds": 4, "positions": 256, "repeats": 3, "channels": 16, "bits": 2}
    shape = candidate_contract.get("shape", {})
    require(candidate_contract.get("status") == "PASS", "CPU 34-bank candidate contract is not PASS")
    require(candidate_contract.get("fixture") is False, "CPU 34-bank candidate is marked as a fixture")
    require(candidate_contract.get("scientific_use") == "CANDIDATE", "CPU candidate source status changed")
    require(all(shape.get(key) == value for key, value in expected_shape.items()), f"CPU candidate shape changed: {shape}")
    validate_formal_split_ledger(workspace, repo_root, candidate_contract)

    verification_gate = load_json(
        workspace
        / "deliveries/CSI-PAIRS-2xA100-FORMAL-HANDOFF-c1b4419d/mac_live_verification/live_regeneration/data_verification/gate.json"
    )
    require(verification_gate.get("status") == "PASS", "same-engine candidate verification is not PASS")
    require(verification_gate.get("scientific_use") == "CANDIDATE_NOT_CLAIM", "same-engine verification boundary changed")
    require(verification_gate.get("rtol") == 0 and verification_gate.get("atol") == 0, "same-engine verification tolerances are not zero")

    fixture = load_json(
        workspace
        / "CSI_PAIRS_DATASET_SUITE_V6/04_FIXTURES_SOFTWARE_ONLY/csi_pairs_v2_1_v6_sionna_rt_dual_a100/generation_manifest.json"
    )
    fixture_dataset = fixture.get("dataset", {})
    require(fixture_dataset.get("fixture") is True, "dual-A100 dataset is not marked fixture=true")
    require(fixture_dataset.get("scientific_use") == "FORBIDDEN", "dual-A100 fixture scientific boundary changed")
    require(fixture.get("validation", {}).get("contract_status") == "PASS", "dual-A100 fixture contract is not PASS")

    registry = load_json(repo_root / "artifacts/a100_dataset_package_v6/external_dataset_registry/EXTERNAL_DATASET_REGISTRY.json")
    require(registry.get("training_core_complete") is True, "external training core is not complete")
    require(registry.get("all_original_sources_complete") is False, "all original sources must remain incomplete")
    require(registry.get("silent_stage0_use") == "FORBIDDEN", "silent Stage-0 use boundary changed")


def build_plan(config: dict, workspace: Path, repo_root: Path) -> list[PackageEntry]:
    entries: list[PackageEntry] = []
    global_exclusions = set(config.get("required_exclusions", []))

    external = config["external_payload"]
    checksum_path = repo_root / external["source_checksum_manifest"]
    records = parse_checksum_manifest(checksum_path)
    validate_external_groups(config, records)
    external_root = workspace / external["source_root"]
    destination_root = external["destination_root"].rstrip("/")
    for digest, relative in records:
        source = external_root.joinpath(*PurePosixPath(relative).parts)
        entries.append(
            entry_for(
                source,
                f"{destination_root}/{relative}",
                "external_public_payload",
                source_sha256=digest,
            )
        )

    for component in config["workspace_components"]:
        add_tree(
            entries,
            workspace / component["source"],
            component["destination"],
            component["component_id"],
            set(component.get("exclude", [])),
            global_exclusions,
        )

    for metadata in config["workspace_metadata"]:
        entries.append(
            entry_for(
                workspace / metadata["source"],
                metadata["destination"],
                "workspace_metadata",
            )
        )

    for document in config["repository_files"]:
        entries.append(
            entry_for(
                repo_root / document["source"],
                document["destination"],
                "package_control_plane",
                executable=document.get("executable", False),
            )
        )

    entries.sort(key=lambda item: item.archive_path)
    paths = [entry.archive_path for entry in entries]
    duplicates = [path for path, count in Counter(paths).items() if count != 1]
    require(not duplicates, f"duplicate package paths: {duplicates[:5]}")
    for entry in entries:
        pure = PurePosixPath(entry.archive_path)
        require(not (set(pure.parts) & global_exclusions), f"forbidden component in package path: {entry.archive_path}")
        require(not entry.archive_path.endswith((".part", ".partial")), f"partial file in package: {entry.archive_path}")
    return entries


def manifest_for(config: dict, entries: list[PackageEntry]) -> bytes:
    payload = {
        "schema_version": "csi-pairs-a100-package-manifest-v1",
        "package_id": config["package_id"],
        "archive_root": config["archive_root"],
        "target_platform": config["target_platform"],
        "build_timestamp_policy": config["fixed_zip_timestamp"],
        "compression": "ZIP_STORED_WITH_ZIP64",
        "file_count": len(entries),
        "payload_bytes": sum(entry.bytes for entry in entries),
        "statuses": {
            "PACKAGE_INTEGRITY": "VERIFY_AFTER_EXTRACTION",
            "TRAINING_CORE_COMPLETE": True,
            "ALL_ORIGINAL_SOURCES_COMPLETE": False,
            "FORMAL_TRAINING_READY": "NO",
            "SCIENTIFIC_EVIDENCE": "NOT_ASSESSED",
        },
        "entries": [
            {
                **{
                    "path": entry.archive_path,
                    "bytes": entry.bytes,
                    "role": entry.role,
                },
                **({"source_sha256": entry.source_sha256} if entry.source_sha256 else {}),
            }
            for entry in entries
        ],
    }
    return (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")


def zip_info(path: str, mode: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (mode & 0xFFFF) << 16
    return info


def write_file(archive: zipfile.ZipFile, package_root: str, entry: PackageEntry) -> None:
    info = zip_info(f"{package_root}/{entry.archive_path}", entry.mode)
    info.file_size = entry.bytes
    with entry.source.open("rb") as source, archive.open(info, "w", force_zip64=True) as destination:
        shutil.copyfileobj(source, destination, length=BUFFER_BYTES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_gib(value: int) -> str:
    return f"{value / (1024 ** 3):.3f} GiB"


def build_archive(config: dict, entries: list[PackageEntry], output: Path) -> tuple[str, float]:
    sidecar = output.with_name(output.name + ".sha256")
    require(not output.exists(), f"refusing to overwrite output: {output}")
    require(not sidecar.exists(), f"refusing to overwrite checksum: {sidecar}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial-{os.getpid()}")
    require(not partial.exists(), f"partial output already exists: {partial}")

    payload_bytes = sum(entry.bytes for entry in entries)
    free_bytes = shutil.disk_usage(output.parent).free
    require(free_bytes >= payload_bytes + (1024 ** 3), f"insufficient free space: need at least {format_gib(payload_bytes + (1024 ** 3))}, have {format_gib(free_bytes)}")

    started = time.monotonic()
    written = 0
    try:
        with zipfile.ZipFile(partial, mode="x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for index, entry in enumerate(entries, start=1):
                write_file(archive, config["archive_root"], entry)
                written += entry.bytes
                elapsed = max(time.monotonic() - started, 0.001)
                rate = written / elapsed / (1024 ** 2)
                print(
                    f"PACK {index}/{len(entries)} bytes={written}/{payload_bytes} rate={rate:.1f}MiB/s {entry.archive_path}",
                    flush=True,
                )
            manifest = manifest_for(config, entries)
            archive.writestr(zip_info(f"{config['archive_root']}/MANIFEST.json", 0o644), manifest)
        os.replace(partial, output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    print("Computing the single outer ZIP SHA-256...", flush=True)
    digest = sha256(output)
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    return digest, time.monotonic() - started


def main() -> int:
    repo_root_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=repo_root_default)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    repo_root = args.repo_root.resolve()
    config_path = (args.config or repo_root / "formal_v2/configs/a100_dataset_suite_v6.json").resolve()
    output = (args.output or workspace / f"deliveries/{PACKAGE_ID}.zip").resolve()

    try:
        require(workspace.is_dir(), f"workspace root does not exist: {workspace}")
        require(repo_root.is_dir(), f"repository root does not exist: {repo_root}")
        config = load_json(config_path)
        require(config.get("schema_version") == "csi-pairs-a100-dataset-suite-config-v1", "unexpected config schema")
        require(config.get("package_id") == PACKAGE_ID, "unexpected package_id")
        require(config.get("archive_root") == config.get("package_id"), "archive_root must match package_id")
        validate_scientific_boundaries(workspace, repo_root)
        entries = build_plan(config, workspace, repo_root)
        role_counts = Counter(entry.role for entry in entries)
        payload_bytes = sum(entry.bytes for entry in entries)
        print(f"PREFLIGHT=PASS files={len(entries)} bytes={payload_bytes} size={format_gib(payload_bytes)}")
        print(f"ROLE_COUNTS={json.dumps(dict(sorted(role_counts.items())), sort_keys=True)}")
        print("TRAINING_CORE_COMPLETE=true")
        print("ALL_ORIGINAL_SOURCES_COMPLETE=false")
        print("FORMAL_TRAINING_READY=NO")
        if args.preflight_only:
            return 0
        digest, duration = build_archive(config, entries, output)
    except (BuildError, OSError, UnicodeError, zipfile.BadZipFile) as error:
        print(f"BUILD=FAIL reason={error}", file=sys.stderr)
        return 1

    print("BUILD=PASS")
    print(f"OUTPUT={output}")
    print(f"OUTPUT_BYTES={output.stat().st_size}")
    print(f"OUTPUT_SHA256={digest}")
    print(f"DURATION_SECONDS={duration:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
