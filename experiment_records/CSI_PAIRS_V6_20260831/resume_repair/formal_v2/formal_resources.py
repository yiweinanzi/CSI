from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import urlparse

from .formal_io import artifact_manifest, read_strict_json, sha256_file, write_csv, write_json


RESOURCE_SCHEMA = "csi-pairs-v6-waibu-resources-v1"
REQUIRED_FILES = {
    "2406.14995v2.pdf",
    "2502.11965v2.pdf",
    "2505.09160v2.pdf",
    "2601.03789v1.pdf",
    "2603.25216v1.pdf",
    "2604.07086v1.pdf",
    "2606.04770v1.pdf",
    "Wi-GATr-main.zip",
    "PMNet-a0e0c592.zip",
    "sionna-main.zip",
    "sionna-large-radio-maps-main.zip",
}
IMPLEMENTATION_STATUSES = {
    "official-code-adaptation",
    "paper-spec-controlled-implementation",
    "style-controlled-implementation",
}
RESOURCE_VALIDATION_MODES = {"formal", "delivery"}


def verify_waibu_resources(registry_path: str | Path, waibu_root: str | Path, output_root: str | Path) -> dict:
    registry_file = Path(registry_path).resolve()
    resource_root = Path(waibu_root).resolve()
    output_dir = Path(output_root).resolve() / "waibu_resources"
    payload = read_strict_json(registry_file)
    rows = validate_resource_registry(payload, resource_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "resource_inventory.csv", rows)
    passed = all(row["status"] == "PASS" for row in rows)
    gate = {
        "schema_version": "csi-pairs-v6-waibu-resource-gate-v1",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "registry_path": str(registry_file),
        "registry_sha256": sha256_file(registry_file),
        "waibu_root": str(resource_root),
        "resource_count": len(rows),
        "verified_count": sum(row["status"] == "PASS" for row in rows),
        "rule": "A resource hash authenticates local bytes only; it does not authenticate scientific results.",
    }
    write_json(output_dir / "gate.json", gate)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "csi-pairs-waibu-resource-stage-manifest-v1",
            "files": artifact_manifest(output_dir),
        },
    )
    return gate


def validate_resource_registry_structure(payload: dict) -> list[dict]:
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "resources"}:
        raise ValueError("waibu resource registry fields must be exact")
    if payload["schema_version"] != RESOURCE_SCHEMA:
        raise ValueError("waibu resource registry schema mismatch")
    resources = payload["resources"]
    if not isinstance(resources, list):
        raise ValueError("waibu resources must be a list")
    expected_fields = {
        "file",
        "sha256",
        "resource_id",
        "title",
        "kind",
        "integration_role",
        "implementation_status",
        "allowed_name",
        "source_url",
        "license_url",
        "redistribution_allowed",
    }
    names = [row.get("file") for row in resources if isinstance(row, dict)]
    if len(names) != len(set(names)) or set(names) != REQUIRED_FILES:
        raise ValueError("waibu registry must contain each frozen resource exactly once")
    output = []
    for row in resources:
        if set(row) != expected_fields:
            raise ValueError("waibu resource fields must be exact")
        if row["kind"] not in {"paper", "official-source-archive"}:
            raise ValueError("waibu resource kind is invalid")
        if row["implementation_status"] not in IMPLEMENTATION_STATUSES:
            raise ValueError("waibu implementation status is invalid")
        string_fields = expected_fields.difference({"redistribution_allowed"})
        if not all(isinstance(row[key], str) and row[key].strip() for key in string_fields):
            raise ValueError("waibu resource string fields must be nonempty")
        if re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None:
            raise ValueError("waibu resource SHA-256 must be 64 lowercase hexadecimal characters")
        if type(row["redistribution_allowed"]) is not bool:
            raise ValueError("waibu redistribution flag must be boolean")
        for key in ("source_url", "license_url"):
            parsed = urlparse(row[key])
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError(f"waibu {key} must be a credential-free HTTPS URL")
        if row["kind"] == "paper":
            redistributable_license = row["license_url"] in {
                "https://creativecommons.org/licenses/by/4.0/",
                "https://creativecommons.org/licenses/by-sa/4.0/",
                "https://creativecommons.org/publicdomain/zero/1.0/",
            }
            if row["redistribution_allowed"] is not redistributable_license:
                raise ValueError("waibu paper redistribution flag contradicts its recorded license")
        if row["kind"] == "official-source-archive" and row["redistribution_allowed"] is not True:
            raise ValueError("waibu official source archives must have a redistributable license")
        output.append(dict(row))
    return output


def validate_resource_registry(
    payload: dict,
    waibu_root: str | Path,
    *,
    mode: str = "formal",
) -> list[dict]:
    if mode not in RESOURCE_VALIDATION_MODES:
        raise ValueError(f"unknown waibu resource validation mode: {mode}")
    resources = validate_resource_registry_structure(payload)
    supplied_root = Path(waibu_root)
    if not supplied_root.is_dir() or supplied_root.is_symlink():
        raise ValueError("waibu root must be an existing regular directory")
    root = supplied_root.resolve()
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    unexpected = actual_files - REQUIRED_FILES
    if unexpected:
        raise ValueError(f"waibu directory contains unexpected files: {sorted(unexpected)}")
    required_present = (
        REQUIRED_FILES
        if mode == "formal"
        else {
            row["file"]
            for row in resources
            if row["redistribution_allowed"] is True
        }
    )
    missing = required_present - actual_files
    if missing:
        scope = "formal" if mode == "formal" else "redistributable delivery"
        raise ValueError(f"waibu {scope} resources are missing: {sorted(missing)}")

    output = []
    for row in resources:
        path = root / row["file"]
        if row["file"] not in actual_files:
            output.append(
                {
                    **row,
                    "actual_sha256": "",
                    "status": "LOCAL_FETCH_REQUIRED",
                }
            )
            continue
        if path.is_symlink() or path.resolve().parent != root or not path.is_file():
            raise ValueError("waibu resource path escapes its frozen directory")
        actual_sha256 = sha256_file(path)
        output.append(
            {
                **row,
                "actual_sha256": actual_sha256,
                "status": "PASS" if actual_sha256 == row["sha256"] else "FAIL",
            }
        )
    return output


def stage_redistributable_resources(
    registry_path: str | Path,
    source_root: str | Path,
    destination_root: str | Path,
) -> list[dict]:
    registry_file = Path(registry_path).resolve()
    payload = read_strict_json(registry_file)
    rows = validate_resource_registry(payload, source_root, mode="delivery")
    invalid = [
        row["file"]
        for row in rows
        if row["redistribution_allowed"] and row["status"] != "PASS"
    ]
    if invalid:
        raise ValueError(f"redistributable resources failed authentication: {invalid}")

    destination = Path(destination_root)
    destination.mkdir(parents=True, exist_ok=False)
    source = Path(source_root).resolve()
    for row in rows:
        if row["redistribution_allowed"]:
            shutil.copy2(source / row["file"], destination / row["file"])
    staged = validate_resource_registry(payload, destination, mode="delivery")
    leaked = [
        row["file"]
        for row in staged
        if not row["redistribution_allowed"] and row["status"] != "LOCAL_FETCH_REQUIRED"
    ]
    if leaked:
        raise ValueError(f"nonredistributable resources entered delivery: {leaked}")
    return staged


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSI-PAIRS resource delivery helper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage = subparsers.add_parser(
        "stage-redistributable",
        help="copy only authenticated resources with downstream redistribution permission",
    )
    stage.add_argument("--registry", required=True)
    stage.add_argument("--source-root", required=True)
    stage.add_argument("--destination-root", required=True)
    return parser


def _main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        rows = stage_redistributable_resources(
            args.registry,
            args.source_root,
            args.destination_root,
        )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "PASS",
                "copied": sorted(
                    row["file"] for row in rows if row["redistribution_allowed"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
