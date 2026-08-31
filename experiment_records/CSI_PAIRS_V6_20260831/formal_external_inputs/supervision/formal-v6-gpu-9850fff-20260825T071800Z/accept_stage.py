from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path


EVALUATION_FILES = {
    "alignment_shortcut_baselines.csv",
    "cgs_active_effect_bins.csv",
    "cgs_per_bank.csv",
    "compatibility_pair_effects.csv",
    "compatibility_probe_contract.csv",
    "compatibility_route_distributions.csv",
    "gate.json",
    "response_pair_effects.csv",
    "response_per_bank.csv",
    "response_probe_contract.csv",
}

CSV_KEYS = {
    "compatibility_probe_contract.csv": ("seed", "arm"),
    "response_probe_contract.csv": ("seed", "arm", "probe"),
    "cgs_per_bank.csv": ("seed", "arm", "bank_id"),
    "alignment_shortcut_baselines.csv": ("seed", "arm", "bank_id", "baseline"),
    "compatibility_route_distributions.csv": ("seed", "arm", "bank_id", "route"),
    "cgs_active_effect_bins.csv": ("seed", "arm", "bank_id", "effect_bin"),
    "compatibility_pair_effects.csv": ("seed", "arm", "pair_id"),
    "response_pair_effects.csv": ("seed", "arm", "pair_id"),
    "response_per_bank.csv": ("seed", "arm", "bank_id"),
}

LARGE_ORDERED_CSVS = {
    "compatibility_pair_effects.csv",
    "response_pair_effects.csv",
}

NONFINITE_TOKENS = {
    "nan",
    "+nan",
    "-nan",
    "inf",
    "+inf",
    "-inf",
    "infinity",
    "+infinity",
    "-infinity",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    os.replace(temporary, path)


def git_state(project_root: Path) -> dict:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "status_short": run("status", "--short").splitlines(),
    }


def load_context(project_root: Path, config_path: Path, dataset_path: Path):
    sys.path.insert(0, str(project_root))
    from formal_v2.formal_config import load_formal_config
    from formal_v2.formal_dataset import FormalDataset

    config = load_formal_config(config_path)
    dataset = FormalDataset.load(
        dataset_path,
        require_clean_csi=bool(config["data"]["require_clean_csi"]),
    )
    dataset.validate(
        require_clean_csi=bool(config["data"]["require_clean_csi"]),
        minimum_repeats=int(config["data"]["minimum_repeats"]),
        minimum_target_cities=int(config["data"]["minimum_target_cities"]),
        minimum_source_cities=int(config["data"]["minimum_source_cities"]),
        minimum_banks_per_target_city=int(
            config["data"]["minimum_banks_per_target_city"]
        ),
        minimum_independent_base_map_clusters_per_target_city=int(
            config["data"]["minimum_independent_base_map_clusters_per_target_city"]
        ),
        minimum_banks_per_source_role=int(
            config["data"]["minimum_banks_per_source_role"]
        ),
        minimum_independent_source_final_unseen_clusters=int(
            config["data"]["minimum_independent_source_final_unseen_clusters"]
        ),
        minimum_independent_external_validation_clusters=int(
            config["data"]["minimum_independent_external_validation_clusters"]
        ),
        minimum_unique_support_positions_per_target_city=max(
            int(value) for value in config["localization"]["label_budgets"]
        ),
    )
    return config, dataset


def authenticate_stage(
    project_root: Path,
    stage_root: Path,
    artifact_path: Path,
    schema: str,
    config: dict,
    dataset,
) -> tuple[dict, dict]:
    from formal_v2.formal_evidence import (
        EVIDENCE_AUTH_KEYS,
        _authenticate_stage_inventory,
        evidence_context,
        require_stage_manifested_gate,
    )
    from formal_v2.formal_io import read_strict_json

    payload = read_strict_json(artifact_path)
    if artifact_path.name == "gate.json":
        require_stage_manifested_gate(
            artifact_path,
            payload,
            config,
            dataset,
            schema_version=schema,
        )
    else:
        if payload.get("schema_version") != schema:
            raise RuntimeError(f"artifact schema mismatch: {artifact_path}")
        expected = evidence_context(
            config, dataset, str(payload.get("scientific_use", ""))
        )
        for key in EVIDENCE_AUTH_KEYS:
            if payload.get(key) != expected[key]:
                raise RuntimeError(f"artifact {key} mismatch: {artifact_path}")
        manifest_path = stage_root / "manifest.json"
        manifest = read_strict_json(manifest_path)
        if manifest.get("schema_version") != "csi-pairs-formal-stage-manifest-v2.1-v6":
            raise RuntimeError(f"stage manifest schema mismatch: {manifest_path}")
        for key in EVIDENCE_AUTH_KEYS:
            if manifest.get(key) != expected[key]:
                raise RuntimeError(f"stage manifest {key} mismatch: {manifest_path}")
        _authenticate_stage_inventory(stage_root, manifest.get("files"))
        matches = [
            row
            for row in manifest.get("files", [])
            if isinstance(row, dict) and row.get("path") == artifact_path.name
        ]
        if len(matches) != 1 or matches[0].get("sha256") != sha256_file(artifact_path):
            raise RuntimeError(f"artifact is not authenticated by manifest: {artifact_path}")
    manifest = read_strict_json(stage_root / "manifest.json")
    return payload, manifest


def reject_temporary_or_symlink(stage_root: Path) -> None:
    for path in stage_root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"stage contains symlink: {path}")
        name = path.name.lower()
        if path.is_file() and (
            name.endswith((".tmp", ".partial", ".incomplete", "~"))
            or name.startswith(".tmp")
        ):
            raise RuntimeError(f"stage contains temporary artifact: {path}")


def _ordered_pair_key(row: dict[str, str], arms: list[str]) -> tuple[int, int, str]:
    try:
        seed = int(row["seed"])
        arm = arms.index(row["arm"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("pair-effect row has invalid seed/arm") from error
    return seed, arm, row["pair_id"]


def scan_csv(
    path: Path,
    expected_evidence: dict,
    arms: list[str],
) -> dict:
    key_fields = CSV_KEYS.get(path.name)
    row_count = 0
    seen: set[tuple[str, ...]] | None = (
        set() if key_fields and path.name not in LARGE_ORDERED_CSVS else None
    )
    previous_ordered_key: tuple[int, int, str] | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise RuntimeError(f"CSV has empty or duplicate header: {path}")
        for row_count, row in enumerate(reader, start=1):
            if None in row or any(value is None for value in row.values()):
                raise RuntimeError(f"CSV row width mismatch at {path}:{row_count + 1}")
            for value in row.values():
                if value.strip().lower() in NONFINITE_TOKENS:
                    raise RuntimeError(f"CSV contains non-finite value at {path}:{row_count + 1}")
            for field in ("dataset_sha256", "config_sha256", "scientific_use"):
                if field in row and row[field] != str(expected_evidence[field]):
                    raise RuntimeError(f"CSV evidence mismatch for {field} at {path}:{row_count + 1}")
            if "fixture" in row and row["fixture"].lower() != str(
                expected_evidence["fixture"]
            ).lower():
                raise RuntimeError(f"CSV evidence mismatch for fixture at {path}:{row_count + 1}")
            if key_fields:
                if any(field not in row or row[field] == "" for field in key_fields):
                    raise RuntimeError(f"CSV key is missing at {path}:{row_count + 1}")
                if path.name in LARGE_ORDERED_CSVS:
                    ordered_key = _ordered_pair_key(row, arms)
                    if previous_ordered_key is not None and ordered_key <= previous_ordered_key:
                        raise RuntimeError(
                            f"CSV pair key is duplicate or out of generator order at {path}:{row_count + 1}"
                        )
                    previous_ordered_key = ordered_key
                else:
                    key = tuple(row[field] for field in key_fields)
                    if key in seen:
                        raise RuntimeError(f"CSV duplicate key at {path}:{row_count + 1}: {key}")
                    seen.add(key)
    if row_count == 0:
        raise RuntimeError(f"CSV has no data rows: {path}")
    return {
        "path": path.name,
        "rows": row_count,
        "columns": reader.fieldnames,
        "bytes": path.stat().st_size,
    }


def validate_inventory(
    stage: str,
    stage_root: Path,
    manifest: dict,
    payload: dict,
    config: dict,
) -> tuple[list[dict], int]:
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("stage manifest has no files")
    paths = {row.get("path") for row in entries if isinstance(row, dict)}
    if len(paths) != len(entries) or None in paths:
        raise RuntimeError("stage manifest paths are missing or duplicated")
    if stage == "evaluation" and paths != EVALUATION_FILES:
        missing = sorted(EVALUATION_FILES - paths)
        extra = sorted(paths - EVALUATION_FILES)
        raise RuntimeError(f"evaluation inventory mismatch: missing={missing}, extra={extra}")
    csv_stats = []
    expected_evidence = {
        "dataset_sha256": payload["dataset_sha256"],
        "config_sha256": payload["config_sha256"],
        "fixture": payload["fixture"],
        "scientific_use": payload["scientific_use"],
    }
    for relative in sorted(paths):
        path = stage_root / relative
        if path.suffix.lower() == ".csv":
            csv_stats.append(
                scan_csv(path, expected_evidence, list(config["factorial"]["arms"]))
            )
    return csv_stats, sum(int(row["bytes"]) for row in entries)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--command-record")
    parser.add_argument("--timing-record")
    parser.add_argument("--run-log")
    parser.add_argument("--log-location")
    parser.add_argument("--exit-code", default="UNKNOWN")
    args = parser.parse_args()

    receipt_path = Path(args.receipt).resolve()
    started = dt.datetime.now(dt.timezone.utc)
    receipt = {
        "schema_version": "csi-pairs-fixed-run-stage-acceptance-v1",
        "stage": args.stage,
        "output_acceptance": "INCOMPLETE",
        "started_utc": started.isoformat(),
        "process_exit_code": args.exit_code,
    }
    try:
        project_root = Path(args.project_root).resolve()
        stage_root = Path(args.stage_root).resolve()
        artifact_path = Path(args.artifact).resolve()
        config_path = Path(args.config).resolve()
        dataset_path = Path(args.dataset).resolve()
        if not stage_root.is_dir() or stage_root.is_symlink():
            raise RuntimeError(f"stage root is missing or unsafe: {stage_root}")
        if artifact_path.parent != stage_root or not artifact_path.is_file():
            raise RuntimeError(f"stage artifact is missing: {artifact_path}")
        reject_temporary_or_symlink(stage_root)
        config, dataset = load_context(project_root, config_path, dataset_path)
        payload, manifest = authenticate_stage(
            project_root,
            stage_root,
            artifact_path,
            args.schema,
            config,
            dataset,
        )
        csv_stats, total_bytes = validate_inventory(
            args.stage, stage_root, manifest, payload, config
        )
        state = git_state(project_root)
        if state["head"] != "9850fffe0b34f16b45066973308f18b10555ca5d":
            raise RuntimeError(f"fixed Git revision changed: {state['head']}")
        receipt.update(
            {
                "output_acceptance": "PASS",
                "artifact": str(artifact_path),
                "artifact_sha256": sha256_file(artifact_path),
                "manifest": str(stage_root / "manifest.json"),
                "manifest_sha256": sha256_file(stage_root / "manifest.json"),
                "file_count": len(manifest["files"]),
                "total_manifested_bytes": total_bytes,
                "csv": csv_stats,
                "scientific_status": payload.get("status"),
                "scientific_passed": payload.get("passed"),
                "dataset_sha256": payload.get("dataset_sha256"),
                "config_sha256": payload.get("config_sha256"),
                "source_tree_sha256": payload.get("source_tree_sha256"),
                "runtime_provenance_sha256": payload.get("runtime_provenance_sha256"),
                "git": state,
            }
        )
        if args.command_record:
            record = Path(args.command_record).resolve()
            receipt["command_record"] = str(record)
            receipt["command_record_sha256"] = sha256_file(record)
        if args.timing_record:
            timing = Path(args.timing_record).resolve()
            receipt["timing_record"] = str(timing)
            receipt["timing_record_sha256"] = sha256_file(timing)
            receipt["timing"] = timing.read_text(encoding="utf-8").splitlines()
        if args.run_log:
            run_log = Path(args.run_log).resolve()
            receipt["run_log"] = str(run_log)
            receipt["run_log_sha256"] = sha256_file(run_log)
            receipt["run_log_bytes"] = run_log.stat().st_size
        if args.log_location:
            receipt["log_location"] = args.log_location
    except Exception as error:
        receipt.update(
            {
                "output_acceptance": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    receipt["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json_atomic(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=True))
    return 0 if receipt["output_acceptance"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
