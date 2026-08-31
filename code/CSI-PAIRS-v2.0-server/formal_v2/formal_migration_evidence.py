from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .formal_evaluation_compare import (
    CSV_CONTRACTS,
    REPORT_SCHEMA as EVALUATION_COMPARATOR_SCHEMA,
    REQUIRED_ARTIFACTS,
)
from .formal_evaluation_subset_compare import (
    CHECKPOINT_SELECTION_RULE as REAL_SUBSET_CHECKPOINT_SELECTION_RULE,
    EVALUATION_TABLE_FIELDS as REAL_SUBSET_TABLE_FIELDS,
    GATE_REQUIRED_TABLES as REAL_SUBSET_GATE_TABLES,
    IMPLEMENTATION_EVIDENCE_FIELDS as REAL_SUBSET_IMPLEMENTATION_FIELDS,
    REPORT_SCHEMA as REAL_SUBSET_REPORT_SCHEMA,
    SELECTION_RULE as REAL_SUBSET_SELECTION_RULE,
    TABLE_PRIMARY_KEYS as REAL_SUBSET_PRIMARY_KEYS,
    WORKER_FRAGMENT_SCHEMA as REAL_SUBSET_FRAGMENT_SCHEMA,
    WORKER_REQUEST_SCHEMA as REAL_SUBSET_REQUEST_SCHEMA,
    compare_evaluation_tables as _compare_real_subset_tables,
    gate_aggregate_report as _real_subset_gate_aggregates,
)
from .formal_io import parse_strict_json


MIGRATION_TECHNICAL_EVIDENCE_SCHEMAS = {
    "performance": "csi-pairs-v6-migration-performance-report-v1",
    "equivalence": "csi-pairs-v6-migration-equivalence-report-v1",
    "resume": "csi-pairs-v6-migration-resume-report-v1",
    "corrupt_checkpoint": "csi-pairs-v6-migration-corrupt-checkpoint-report-v1",
    "stale_checkpoint": "csi-pairs-v6-migration-stale-checkpoint-report-v1",
    "lock": "csi-pairs-v6-migration-lock-report-v1",
    "progress": "csi-pairs-v6-migration-progress-report-v1",
}
MIGRATION_DIFF_SCHEMA = "csi-pairs-v6-migration-base-to-new-diff-v1"
MIGRATION_FREEZE_SCHEMA = "csi-pairs-v6-legacy-evaluation-freeze-receipt-v1"
MIGRATION_LEGACY_INVENTORY_SCHEMA = (
    "csi-pairs-v6-legacy-evaluation-inventory-v1"
)
MIGRATION_EVIDENCE_KEYS = (
    *MIGRATION_TECHNICAL_EVIDENCE_SCHEMAS,
    "base_to_new_diff",
    "legacy_evaluation_inventory",
    "legacy_evaluation_freeze",
)

TECHNICAL_REPORT_FIELDS = {
    "schema_version",
    "status",
    "created_utc",
    "source_commit",
    "source_tree_sha256",
    "command",
    "inputs",
    "results",
}
DIFF_REPORT_FIELDS = {
    *TECHNICAL_REPORT_FIELDS,
    "base_commit",
    "base_source_tree_sha256",
    "new_commit",
    "new_source_tree_sha256",
    "diff_sha256",
}
FREEZE_REPORT_FIELDS = {
    *TECHNICAL_REPORT_FIELDS,
    "pid_identity",
    "start_ticks",
    "cmd",
    "cwd",
    "legacy_inventory_sha256",
    "lock_state",
}
EXPECTED_IDENTITY_FIELDS = {
    "legacy_run_root",
    "new_run_root",
    "legacy_commit",
    "legacy_source_tree_sha256",
    "new_commit",
    "new_source_tree_sha256",
    "requirements_lock_sha256",
    "dataset_sha256",
    "config_sha256",
    "protocol_sha256",
    "new_run_id",
    "new_run_nonce",
    "new_compute_plan_sha256",
}
FILE_BINDING_FIELDS = {"path", "bytes", "sha256"}
PERFORMANCE_SCALE_FIELDS = {
    "scale",
    "sample_count",
    "wall_seconds",
    "cpu_seconds",
    "peak_rss_bytes",
    "execution_device",
    "gpu_utilization_percent",
    "gpu_utilization_statistic",
    "gpu_sample_count",
    "peak_vram_bytes",
    "output",
    "oracle_equivalent",
}
PERFORMANCE_RESULT_FIELDS = {
    "scales",
    "empirical_time_exponent",
    "empirical_peak_rss_exponent",
    "time_exponent_limit",
    "no_quadratic_memory",
    "gpu_benchmark",
}
GPU_BENCHMARK_FIELDS = {
    "execution_device",
    "batch_size",
    "wall_seconds",
    "gpu_utilization_percent",
    "gpu_utilization_statistic",
    "gpu_sample_count",
    "gpu_sampling_interval_seconds",
    "peak_vram_bytes",
    "artifact",
}
EQUIVALENCE_RESULT_FIELDS = {
    "smoke_report",
    "smoke_oracle",
    "formal_subset_report",
}
SMOKE_ORACLE_FIELDS = {"manifest", "identity"}
COMPARATOR_IDENTITY_FIELDS = {
    "artifact_label",
    "dataset_sha256",
    "config_sha256",
    "fixture",
    "scientific_use",
    "source_tree_sha256",
    "requirements_lock_sha256",
    "runtime_provenance_sha256",
    "runtime_provenance",
}
RESUME_RESULT_FIELDS = {
    "interrupted",
    "resumed",
    "completed",
    "reused_shard_count",
    "recomputed_shard_count",
    "output_equivalent",
}
CORRUPT_RESULT_FIELDS = {
    "corruption_detected",
    "corrupt_shard_count",
    "quarantined_shard_count",
    "recomputed_shard_count",
    "unaffected_shard_count",
}
STALE_RESULT_FIELDS = {
    "stale_identity_detected",
    "rejected_shard_count",
    "reused_stale_shard_count",
}
LOCK_RESULT_FIELDS = {
    "second_writer_attempted",
    "second_writer_rejected",
    "first_writer_preserved",
    "write_count",
}
PROGRESS_RESULT_FIELDS = {
    "monotonic",
    "atomic",
    "final_complete",
    "update_count",
    "completed_units",
    "total_units",
}
DIFF_RESULT_FIELDS = {
    "diff",
    "reviewed",
    "applies_cleanly",
    "changed_file_count",
}
INVENTORY_RESULT_FIELDS = {
    "legacy_evaluation_root",
    "files",
    "files_sha256",
}
FREEZE_RESULT_FIELDS = {
    "inventory_authenticated",
    "process_state",
    "pid_identity_matched",
    "lock_state_observed",
}

TIME_EXPONENT_LIMIT = 1.5
QUADRATIC_MEMORY_EXPONENT = 2.0
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SCALE_NAMES = ("N", "2N", "4N")
_COMPARATOR_ARTIFACTS = tuple(REQUIRED_ARTIFACTS) + ("manifest.json",)
_COMPARATOR_REPORT_FIELDS = {
    "schema_version",
    "generated_utc",
    "reference_root",
    "candidate_root",
    "tolerance",
    "required_artifacts",
    "csv",
    "gate",
    "manifest",
    "summary",
    "equivalent",
    "exact_match",
    "status",
}
_COMPARATOR_TOLERANCE_RULE = (
    "abs(a-b) <= absolute + relative * max(abs(a), abs(b))"
)
_REAL_SUBSET_REPORT_FIELDS = {
    "schema_version",
    "status",
    "passed",
    "created_utc",
    "authoritative_layer",
    "contract",
    "workers",
    "shared_inputs",
    "selection",
    "layers",
    "read_only",
    "resources",
    "report_path",
}
_REAL_SUBSET_CONTRACT_FIELDS = {
    "formal_dataset_only",
    "full_legacy_evaluation_executed",
    "legacy_run_read_only",
    "frozen_legacy_run_body_executed",
    "cross_source_expected_implementation_fields",
    "scientific_float_requirement",
    "absolute_tolerance",
    "relative_tolerance",
}
_REAL_SUBSET_WORKER_FIELDS = {
    "role",
    "source_identity",
    "runtime_provenance",
    "request",
    "fragment",
    "resources",
}
_REAL_SUBSET_SOURCE_FIELDS = {
    "source_root",
    "git_root",
    "git_commit",
    "tracked_tree_clean",
    "tracked_status",
    "untracked_scientific_paths",
    "tracked_diff_sha256",
    "files",
    "source_tree_sha256",
    "runtime_provenance_sha256",
}
_REAL_SUBSET_SOURCE_FILE_FIELDS = {"path", "bytes", "sha256", "git_tracked"}
_REAL_SUBSET_ARTIFACT_FIELDS = {"path", "bytes", "mtime_ns", "sha256"}
_REAL_SUBSET_IDENTITY_ARTIFACT_FIELDS = {
    *_REAL_SUBSET_ARTIFACT_FIELDS,
    "identity_sha256",
}
_REAL_SUBSET_FRAGMENT_FIELDS = {
    "schema_version",
    "role",
    "created_utc",
    "request",
    "source_identity",
    "runtime_provenance",
    "input_bindings",
    "selection",
    "tables",
    "same_source_merged_tables",
    "read_only",
    "resources",
}
_REAL_SUBSET_FRAGMENT_REQUEST_FIELDS = {"path", "bytes", "sha256"}
_REAL_SUBSET_REQUEST_FIELDS = {
    "schema_version",
    "config_path",
    "dataset_path",
    "legacy_run_root",
    "device",
    "batch_size",
    "source_scenes",
    "target_scenes_per_city",
    "checkpoint_count",
    "checkpoint_arm",
    "role",
    "fragment_path",
    "expected_git_commit",
    "expected_evaluation_sha256",
}
_REAL_SUBSET_INPUT_FIELDS = {"config_sha256", "dataset_sha256", "files"}
_REAL_SUBSET_SELECTION_FIELDS = {
    "scene_rule",
    "checkpoint_rule",
    "scenes_in_execution_order",
    "checkpoints_in_execution_order",
    "sha256",
}
_REAL_SUBSET_SCENE_FIELDS = {
    "scene_index",
    "scene_id",
    "scene_role",
    "city_id",
    "bank_id",
    "base_map_cluster_id",
    "canonical_base_map_digest",
    "canonical_bank_digest",
}
_REAL_SUBSET_CHECKPOINT_FIELDS = {"seed", "arm", "sha256"}
_REAL_SUBSET_READ_ONLY_FIELDS = {"passed", "files", "legacy_evaluation"}
_REAL_SUBSET_READ_ONLY_RECORD_FIELDS = {"path", "before", "after", "unchanged"}
_REAL_SUBSET_EVALUATION_AUDIT_FIELDS = {"root", "before", "after", "unchanged"}
_REAL_SUBSET_DIRECTORY_SNAPSHOT_FIELDS = {
    "root",
    "root_mode",
    "root_mtime_ns",
    "entry_count",
    "total_file_bytes",
    "entries_sha256",
}
_REAL_SUBSET_RESOURCE_FIELDS = {
    "elapsed_seconds",
    "maximum_resident_set_bytes",
    "maximum_cuda_allocated_bytes",
}
_REAL_SUBSET_LAYER_FIELDS = {
    "cross_source_end_to_end",
    "same_source_decomposition",
}
_REAL_SUBSET_CROSS_LAYER_FIELDS = {
    "authoritative",
    "legacy_source_role",
    "candidate_source_role",
    "comparison",
    "gate_required_aggregates",
}
_REAL_SUBSET_SAME_LAYER_FIELDS = {
    "authoritative",
    "description",
    "comparison",
    "gate_required_aggregates",
}
_REAL_SUBSET_DESCRIPTION = (
    "candidate-source merged-scene decomposition versus candidate-source "
    "per-scene streaming; this layer is not legacy-source evidence"
)
_REAL_SUBSET_EMPTY_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()


def bind_file(
    path: str | Path,
    *,
    legacy_run_root: str | Path | None = None,
    new_run_root: str | Path | None = None,
    require_external: bool = False,
) -> dict[str, object]:
    """Return a live absolute path/size/SHA-256 binding for one regular file."""
    candidate = _regular_file(path, "bound migration input")
    if require_external:
        legacy_root, new_root = _binding_roots(legacy_run_root, new_run_root)
        _require_external(candidate, legacy_root, new_root, "bound migration input")
    size, digest = _file_size_sha256(candidate)
    return {"path": str(candidate), "bytes": size, "sha256": digest}


def write_json_exclusive_atomic(path: str | Path, payload: object) -> Path:
    """Durably create canonical JSON without ever replacing an existing path."""
    requested = Path(path)
    requested.parent.mkdir(parents=True, exist_ok=True)
    if requested.is_symlink() or requested.parent.is_symlink():
        raise RuntimeError("migration evidence path cannot be a symlink")
    target = requested.resolve()
    parent = target.parent
    if not parent.is_dir():
        raise RuntimeError("migration evidence parent must be a regular directory")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite migration evidence: {target}")
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    temporary = parent / f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target, follow_symlinks=False)
        directory_descriptor = os.open(
            parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def validate_evidence_results(
    name: str,
    payload: Mapping[str, object],
    *,
    expected_identity: Mapping[str, object],
    legacy_run_root: str | Path,
    new_run_root: str | Path,
) -> None:
    """Reauthenticate the exact nested contract for one migration report."""
    if name not in MIGRATION_EVIDENCE_KEYS:
        raise RuntimeError(f"unknown migration evidence report: {name}")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"migration {name} report must be an object")
    legacy_root = Path(legacy_run_root).resolve()
    new_root = Path(new_run_root).resolve()
    identity = _validated_identity(expected_identity, legacy_root, new_root)
    inputs = payload.get("inputs")
    _require_exact_fields(inputs, {"identity", "artifacts"}, f"migration {name} inputs")
    if inputs["identity"] != identity:
        raise RuntimeError(f"migration {name} input identity mismatch")
    artifacts = _validate_input_artifacts(
        name,
        inputs["artifacts"],
        legacy_root=legacy_root,
        new_root=new_root,
    )
    results = payload.get("results")
    if name == "performance":
        _validate_performance(results, artifacts, legacy_root, new_root)
    elif name == "equivalence":
        _validate_equivalence(results, artifacts, identity, legacy_root, new_root)
    elif name == "resume":
        _validate_resume(results, artifacts)
    elif name == "corrupt_checkpoint":
        _validate_corrupt(results, artifacts)
    elif name == "stale_checkpoint":
        _validate_stale(results, artifacts)
    elif name == "lock":
        _validate_lock(results, artifacts)
    elif name == "progress":
        _validate_progress(results, artifacts)
    elif name == "base_to_new_diff":
        _validate_diff(payload, results, artifacts, legacy_root, new_root)
    elif name == "legacy_evaluation_inventory":
        _validate_inventory(results, artifacts, legacy_root)
    else:
        _validate_freeze(
            payload,
            results,
            artifacts,
            identity,
            legacy_root,
            new_root,
        )


def validate_evidence_report(
    name: str,
    payload: Mapping[str, object],
    *,
    expected_identity: Mapping[str, object],
    legacy_run_root: str | Path,
    new_run_root: str | Path,
) -> None:
    """Validate a complete report, including its outer schema and source identity."""
    identity = _validated_identity(
        expected_identity, Path(legacy_run_root).resolve(), Path(new_run_root).resolve()
    )
    if name in MIGRATION_TECHNICAL_EVIDENCE_SCHEMAS:
        fields = TECHNICAL_REPORT_FIELDS
        schema = MIGRATION_TECHNICAL_EVIDENCE_SCHEMAS[name]
        status = "PASS"
        commit = identity["new_commit"]
        source_tree = identity["new_source_tree_sha256"]
    elif name == "base_to_new_diff":
        fields = DIFF_REPORT_FIELDS
        schema = MIGRATION_DIFF_SCHEMA
        status = "PASS"
        commit = identity["new_commit"]
        source_tree = identity["new_source_tree_sha256"]
    elif name == "legacy_evaluation_inventory":
        fields = TECHNICAL_REPORT_FIELDS
        schema = MIGRATION_LEGACY_INVENTORY_SCHEMA
        status = "PASS"
        commit = identity["legacy_commit"]
        source_tree = identity["legacy_source_tree_sha256"]
    elif name == "legacy_evaluation_freeze":
        fields = FREEZE_REPORT_FIELDS
        schema = MIGRATION_FREEZE_SCHEMA
        status = "FROZEN"
        commit = identity["legacy_commit"]
        source_tree = identity["legacy_source_tree_sha256"]
    else:
        raise RuntimeError(f"unknown migration evidence report: {name}")
    _require_exact_fields(payload, fields, f"migration {name} report")
    if (
        payload["schema_version"] != schema
        or payload["status"] != status
        or payload["source_commit"] != commit
        or payload["source_tree_sha256"] != source_tree
        or not isinstance(payload["command"], str)
        or not payload["command"].strip()
    ):
        raise RuntimeError(f"migration {name} report identity mismatch")
    _parse_utc(payload["created_utc"], f"migration {name} created_utc")
    validate_evidence_results(
        name,
        payload,
        expected_identity=identity,
        legacy_run_root=legacy_run_root,
        new_run_root=new_run_root,
    )


def write_performance_report(
    path: str | Path,
    *,
    expected_identity: Mapping[str, object],
    command: str,
    measurements: Sequence[Mapping[str, object]],
    measurement_artifact_paths: Sequence[str | Path],
    gpu_benchmark: Mapping[str, object],
    created_utc: str | None = None,
) -> dict[str, object]:
    """Write authenticated N/2N/4N timing, CPU, RSS, output, and oracle evidence."""
    identity, legacy_root, new_root = _writer_identity(expected_identity)
    if len(measurements) != 3:
        raise RuntimeError("performance evidence requires exactly three measurements")
    scales = []
    for measurement in measurements:
        _require_exact_fields(
            measurement,
            PERFORMANCE_SCALE_FIELDS - {"output"} | {"output_path"},
            "performance measurement",
        )
        scales.append(
            {
                "scale": measurement["scale"],
                "sample_count": measurement["sample_count"],
                "wall_seconds": measurement["wall_seconds"],
                "cpu_seconds": measurement["cpu_seconds"],
                "peak_rss_bytes": measurement["peak_rss_bytes"],
                "execution_device": measurement["execution_device"],
                "gpu_utilization_percent": measurement[
                    "gpu_utilization_percent"
                ],
                "gpu_utilization_statistic": measurement[
                    "gpu_utilization_statistic"
                ],
                "gpu_sample_count": measurement["gpu_sample_count"],
                "peak_vram_bytes": measurement["peak_vram_bytes"],
                "output": bind_file(
                    measurement["output_path"],
                    legacy_run_root=legacy_root,
                    new_run_root=new_root,
                    require_external=True,
                ),
                "oracle_equivalent": measurement["oracle_equivalent"],
            }
        )
    wall_exponent = _growth_exponent(
        scales[0]["wall_seconds"], scales[2]["wall_seconds"]
    )
    rss_exponent = _growth_exponent(
        scales[0]["peak_rss_bytes"], scales[2]["peak_rss_bytes"]
    )
    _require_exact_fields(
        gpu_benchmark,
        GPU_BENCHMARK_FIELDS - {"artifact"} | {"artifact_path"},
        "GPU benchmark measurement",
    )
    gpu_artifact = bind_file(
        gpu_benchmark["artifact_path"],
        legacy_run_root=legacy_root,
        new_run_root=new_root,
        require_external=True,
    )
    bound_gpu_benchmark = {
        key: gpu_benchmark[key]
        for key in GPU_BENCHMARK_FIELDS
        if key != "artifact"
    }
    bound_gpu_benchmark["artifact"] = gpu_artifact
    results = {
        "scales": scales,
        "empirical_time_exponent": wall_exponent,
        "empirical_peak_rss_exponent": rss_exponent,
        "time_exponent_limit": TIME_EXPONENT_LIMIT,
        "no_quadratic_memory": rss_exponent < QUADRATIC_MEMORY_EXPONENT,
        "gpu_benchmark": bound_gpu_benchmark,
    }
    payload = _technical_payload(
        "performance",
        identity,
        command,
        results,
        [*measurement_artifact_paths, gpu_benchmark["artifact_path"]],
        created_utc,
    )
    return _validate_and_write(path, "performance", payload, identity)


def write_equivalence_report(
    path: str | Path,
    *,
    expected_identity: Mapping[str, object],
    command: str,
    smoke_report_path: str | Path,
    formal_subset_report_path: str | Path,
    created_utc: str | None = None,
) -> dict[str, object]:
    """Write bindings to zero-tolerance smoke and nonfixture formal comparisons."""
    identity, legacy_root, new_root = _writer_identity(expected_identity)
    smoke = bind_file(
        smoke_report_path,
        legacy_run_root=legacy_root,
        new_run_root=new_root,
        require_external=True,
    )
    formal = bind_file(
        formal_subset_report_path,
        legacy_run_root=legacy_root,
        new_run_root=new_root,
        require_external=True,
    )
    smoke_payload = _read_bound_json(smoke, Path(str(smoke["path"])))
    try:
        oracle_manifest_path = smoke_payload["manifest"]["reference"]["path"]
        oracle_identity = smoke_payload["manifest"]["reference"]["identity"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("smoke comparator does not bind its oracle manifest") from error
    oracle_manifest = bind_file(
        oracle_manifest_path,
        legacy_run_root=legacy_root,
        new_run_root=new_root,
        require_external=True,
    )
    results = {
        "smoke_report": smoke,
        "smoke_oracle": {
            "manifest": oracle_manifest,
            "identity": dict(oracle_identity),
        },
        "formal_subset_report": formal,
    }
    payload = _technical_payload(
        "equivalence",
        identity,
        command,
        results,
        [smoke_report_path, oracle_manifest_path, formal_subset_report_path],
        created_utc,
    )
    return _validate_and_write(path, "equivalence", payload, identity)


def write_resume_report(
    path: str | Path,
    *,
    expected_identity: Mapping[str, object],
    command: str,
    artifact_paths: Sequence[str | Path],
    interrupted: bool,
    resumed: bool,
    completed: bool,
    reused_shard_count: int,
    recomputed_shard_count: int,
    output_equivalent: bool,
    created_utc: str | None = None,
) -> dict[str, object]:
    results = {
        "interrupted": interrupted,
        "resumed": resumed,
        "completed": completed,
        "reused_shard_count": reused_shard_count,
        "recomputed_shard_count": recomputed_shard_count,
        "output_equivalent": output_equivalent,
    }
    return _write_control_report(
        path, "resume", expected_identity, command, artifact_paths, results, created_utc
    )


def write_corrupt_checkpoint_report(
    path: str | Path,
    *,
    expected_identity: Mapping[str, object],
    command: str,
    artifact_paths: Sequence[str | Path],
    corruption_detected: bool,
    corrupt_shard_count: int,
    quarantined_shard_count: int,
    recomputed_shard_count: int,
    unaffected_shard_count: int,
    created_utc: str | None = None,
) -> dict[str, object]:
    results = {
        "corruption_detected": corruption_detected,
        "corrupt_shard_count": corrupt_shard_count,
        "quarantined_shard_count": quarantined_shard_count,
        "recomputed_shard_count": recomputed_shard_count,
        "unaffected_shard_count": unaffected_shard_count,
    }
    return _write_control_report(
        path,
        "corrupt_checkpoint",
        expected_identity,
        command,
        artifact_paths,
        results,
        created_utc,
    )


def write_stale_checkpoint_report(
    path: str | Path,
    *,
    expected_identity: Mapping[str, object],
    command: str,
    artifact_paths: Sequence[str | Path],
    stale_identity_detected: bool,
    rejected_shard_count: int,
    reused_stale_shard_count: int,
    created_utc: str | None = None,
) -> dict[str, object]:
    results = {
        "stale_identity_detected": stale_identity_detected,
        "rejected_shard_count": rejected_shard_count,
        "reused_stale_shard_count": reused_stale_shard_count,
    }
    return _write_control_report(
        path,
        "stale_checkpoint",
        expected_identity,
        command,
        artifact_paths,
        results,
        created_utc,
    )


def write_lock_report(
    path: str | Path,
    *,
    expected_identity: Mapping[str, object],
    command: str,
    artifact_paths: Sequence[str | Path],
    second_writer_attempted: bool,
    second_writer_rejected: bool,
    first_writer_preserved: bool,
    write_count: int,
    created_utc: str | None = None,
) -> dict[str, object]:
    results = {
        "second_writer_attempted": second_writer_attempted,
        "second_writer_rejected": second_writer_rejected,
        "first_writer_preserved": first_writer_preserved,
        "write_count": write_count,
    }
    return _write_control_report(
        path, "lock", expected_identity, command, artifact_paths, results, created_utc
    )


def write_progress_report(
    path: str | Path,
    *,
    expected_identity: Mapping[str, object],
    command: str,
    artifact_paths: Sequence[str | Path],
    monotonic: bool,
    atomic: bool,
    final_complete: bool,
    update_count: int,
    completed_units: int,
    total_units: int,
    created_utc: str | None = None,
) -> dict[str, object]:
    results = {
        "monotonic": monotonic,
        "atomic": atomic,
        "final_complete": final_complete,
        "update_count": update_count,
        "completed_units": completed_units,
        "total_units": total_units,
    }
    return _write_control_report(
        path,
        "progress",
        expected_identity,
        command,
        artifact_paths,
        results,
        created_utc,
    )


def write_base_to_new_diff_report(
    path: str | Path,
    *,
    expected_identity: Mapping[str, object],
    command: str,
    diff_path: str | Path,
    changed_file_count: int,
    reviewed: bool,
    applies_cleanly: bool,
    created_utc: str | None = None,
) -> dict[str, object]:
    identity, legacy_root, new_root = _writer_identity(expected_identity)
    diff = bind_file(
        diff_path,
        legacy_run_root=legacy_root,
        new_run_root=new_root,
        require_external=True,
    )
    payload = {
        "schema_version": MIGRATION_DIFF_SCHEMA,
        "status": "PASS",
        "created_utc": created_utc or _utc_now(),
        "source_commit": identity["new_commit"],
        "source_tree_sha256": identity["new_source_tree_sha256"],
        "command": _command(command),
        "inputs": {"identity": identity, "artifacts": [diff]},
        "results": {
            "diff": diff,
            "reviewed": reviewed,
            "applies_cleanly": applies_cleanly,
            "changed_file_count": changed_file_count,
        },
        "base_commit": identity["legacy_commit"],
        "base_source_tree_sha256": identity["legacy_source_tree_sha256"],
        "new_commit": identity["new_commit"],
        "new_source_tree_sha256": identity["new_source_tree_sha256"],
        "diff_sha256": diff["sha256"],
    }
    return _validate_and_write(path, "base_to_new_diff", payload, identity)


def write_legacy_evaluation_inventory_report(
    path: str | Path,
    *,
    expected_identity: Mapping[str, object],
    command: str,
    created_utc: str | None = None,
) -> dict[str, object]:
    identity, legacy_root, _ = _writer_identity(expected_identity)
    evaluation_root = _regular_directory(
        legacy_root / "evaluation", "legacy evaluation root"
    )
    files = []
    artifacts = []
    for artifact in sorted(item for item in evaluation_root.rglob("*") if item.is_file()):
        resolved = _regular_file(artifact, "legacy evaluation artifact")
        if not _is_within(resolved, evaluation_root):
            raise RuntimeError("legacy evaluation artifact escapes its root")
        binding = bind_file(resolved)
        files.append(
            {
                "path": resolved.relative_to(evaluation_root).as_posix(),
                "bytes": binding["bytes"],
                "sha256": binding["sha256"],
            }
        )
        artifacts.append(binding)
    payload = {
        "schema_version": MIGRATION_LEGACY_INVENTORY_SCHEMA,
        "status": "PASS",
        "created_utc": created_utc or _utc_now(),
        "source_commit": identity["legacy_commit"],
        "source_tree_sha256": identity["legacy_source_tree_sha256"],
        "command": _command(command),
        "inputs": {"identity": identity, "artifacts": artifacts},
        "results": {
            "legacy_evaluation_root": str(evaluation_root),
            "files": files,
            "files_sha256": _canonical_sha256(files),
        },
    }
    return _validate_and_write(
        path, "legacy_evaluation_inventory", payload, identity
    )


def write_legacy_evaluation_freeze_receipt(
    path: str | Path,
    *,
    expected_identity: Mapping[str, object],
    command: str,
    inventory_path: str | Path,
    pid_identity: str,
    start_ticks: int,
    cmd: str,
    cwd: str | Path,
    process_state: str,
    pid_identity_matched: bool,
    lock_state: Mapping[str, object],
    created_utc: str | None = None,
) -> dict[str, object]:
    identity, legacy_root, new_root = _writer_identity(expected_identity)
    if not isinstance(cwd, (str, Path)) or not Path(cwd).is_absolute():
        raise RuntimeError("legacy freeze observed cwd must be absolute")
    inventory = bind_file(
        inventory_path,
        legacy_run_root=legacy_root,
        new_run_root=new_root,
        require_external=True,
    )
    payload = {
        "schema_version": MIGRATION_FREEZE_SCHEMA,
        "status": "FROZEN",
        "created_utc": created_utc or _utc_now(),
        "source_commit": identity["legacy_commit"],
        "source_tree_sha256": identity["legacy_source_tree_sha256"],
        "command": _command(command),
        "inputs": {"identity": identity, "artifacts": [inventory]},
        "results": {
            "inventory_authenticated": True,
            "process_state": process_state,
            "pid_identity_matched": pid_identity_matched,
            "lock_state_observed": lock_state.get("observed") is True,
        },
        "pid_identity": pid_identity,
        "start_ticks": start_ticks,
        "cmd": cmd,
        "cwd": str(Path(cwd).resolve()),
        "legacy_inventory_sha256": inventory["sha256"],
        "lock_state": dict(lock_state),
    }
    return _validate_and_write(
        path, "legacy_evaluation_freeze", payload, identity
    )


def _write_control_report(
    path: str | Path,
    name: str,
    expected_identity: Mapping[str, object],
    command: str,
    artifact_paths: Sequence[str | Path],
    results: dict[str, object],
    created_utc: str | None,
) -> dict[str, object]:
    identity, _, _ = _writer_identity(expected_identity)
    payload = _technical_payload(
        name, identity, command, results, artifact_paths, created_utc
    )
    return _validate_and_write(path, name, payload, identity)


def _technical_payload(
    name: str,
    identity: dict[str, object],
    command: str,
    results: dict[str, object],
    artifact_paths: Sequence[str | Path],
    created_utc: str | None,
) -> dict[str, object]:
    legacy_root = Path(str(identity["legacy_run_root"]))
    new_root = Path(str(identity["new_run_root"]))
    artifacts = [
        bind_file(
            artifact,
            legacy_run_root=legacy_root,
            new_run_root=new_root,
            require_external=True,
        )
        for artifact in artifact_paths
    ]
    return {
        "schema_version": MIGRATION_TECHNICAL_EVIDENCE_SCHEMAS[name],
        "status": "PASS",
        "created_utc": created_utc or _utc_now(),
        "source_commit": identity["new_commit"],
        "source_tree_sha256": identity["new_source_tree_sha256"],
        "command": _command(command),
        "inputs": {"identity": identity, "artifacts": artifacts},
        "results": results,
    }


def _validate_and_write(
    path: str | Path,
    name: str,
    payload: dict[str, object],
    identity: Mapping[str, object],
) -> dict[str, object]:
    legacy_root = Path(str(identity["legacy_run_root"])).resolve()
    new_root = Path(str(identity["new_run_root"])).resolve()
    target = _external_target(path, legacy_root, new_root)
    validate_evidence_report(
        name,
        payload,
        expected_identity=identity,
        legacy_run_root=legacy_root,
        new_run_root=new_root,
    )
    write_json_exclusive_atomic(target, payload)
    return payload


def _validate_input_artifacts(
    name: str,
    value: object,
    *,
    legacy_root: Path,
    new_root: Path,
) -> list[tuple[dict[str, object], Path]]:
    if not isinstance(value, list):
        raise RuntimeError(f"migration {name} input artifacts must be a list")
    if not value and name != "legacy_evaluation_inventory":
        raise RuntimeError(f"migration {name} inputs must bind at least one artifact")
    authenticated = []
    observed = set()
    evaluation_root = (legacy_root / "evaluation").resolve()
    for binding in value:
        if name == "legacy_evaluation_inventory":
            path = _authenticate_file_binding(binding)
            if not _is_within(path, evaluation_root):
                raise RuntimeError("legacy inventory input escapes the evaluation root")
        else:
            path = _authenticate_file_binding(
                binding,
                legacy_root=legacy_root,
                new_root=new_root,
                require_external=True,
            )
        if path in observed:
            raise RuntimeError(f"migration {name} input artifacts must be unique")
        observed.add(path)
        authenticated.append((dict(binding), path))
    return authenticated


def _validate_performance(results, artifacts, legacy_root, new_root):
    _require_exact_fields(results, PERFORMANCE_RESULT_FIELDS, "performance results")
    scales = results["scales"]
    if not isinstance(scales, list) or len(scales) != 3:
        raise RuntimeError("performance scales must be exactly N, 2N, and 4N")
    counts = []
    walls = []
    rss = []
    outputs = set()
    execution_devices = []
    for index, row in enumerate(scales):
        _require_exact_fields(row, PERFORMANCE_SCALE_FIELDS, "performance scale")
        if row["scale"] != _SCALE_NAMES[index]:
            raise RuntimeError("performance scales must be ordered N, 2N, and 4N")
        counts.append(_positive_int(row["sample_count"], "performance sample count"))
        walls.append(_positive_number(row["wall_seconds"], "performance wall time"))
        _positive_number(row["cpu_seconds"], "performance CPU time")
        rss.append(_positive_int(row["peak_rss_bytes"], "performance peak RSS"))
        _validate_gpu_measurement(row, "performance scale", allow_cpu=True)
        execution_devices.append(row["execution_device"])
        output = _authenticate_file_binding(
            row["output"],
            legacy_root=legacy_root,
            new_root=new_root,
            require_external=True,
        )
        if output in outputs:
            raise RuntimeError("performance outputs must be distinct")
        outputs.add(output)
        if row["oracle_equivalent"] is not True:
            raise RuntimeError("every performance scale must pass oracle equivalence")
    if counts[1] != 2 * counts[0] or counts[2] != 4 * counts[0]:
        raise RuntimeError("performance sample counts must double exactly")
    if len(set(execution_devices)) != 1:
        raise RuntimeError("performance scales must use one execution device")
    measured_time = _growth_exponent(walls[0], walls[2])
    reported_time = _finite_number(
        results["empirical_time_exponent"], "empirical time exponent"
    )
    limit = _finite_number(results["time_exponent_limit"], "time exponent limit")
    if (
        not math.isclose(reported_time, measured_time, rel_tol=1e-12, abs_tol=1e-12)
        or limit != TIME_EXPONENT_LIMIT
        or reported_time >= limit
    ):
        raise RuntimeError("performance empirical time exponent is invalid")
    measured_rss = _growth_exponent(rss[0], rss[2])
    reported_rss = _finite_number(
        results["empirical_peak_rss_exponent"], "empirical peak RSS exponent"
    )
    if (
        not math.isclose(reported_rss, measured_rss, rel_tol=1e-12, abs_tol=1e-12)
        or reported_rss >= QUADRATIC_MEMORY_EXPONENT
        or results["no_quadratic_memory"] is not True
    ):
        raise RuntimeError("performance memory growth is not authenticated")
    gpu = results["gpu_benchmark"]
    _require_exact_fields(gpu, GPU_BENCHMARK_FIELDS, "GPU benchmark results")
    _validate_gpu_measurement(gpu, "GPU benchmark", allow_cpu=False)
    _positive_int(gpu["batch_size"], "GPU benchmark batch size")
    _positive_number(gpu["wall_seconds"], "GPU benchmark wall time")
    _positive_number(
        gpu["gpu_sampling_interval_seconds"], "GPU sampling interval"
    )
    gpu_artifact = _authenticate_file_binding(
        gpu["artifact"],
        legacy_root=legacy_root,
        new_root=new_root,
        require_external=True,
    )
    if not any(path == gpu_artifact and binding == gpu["artifact"] for binding, path in artifacts):
        raise RuntimeError("GPU benchmark artifact must be bound as a performance input")


def _validate_gpu_measurement(value, label, *, allow_cpu):
    device = value.get("execution_device")
    utilization = _finite_number(
        value.get("gpu_utilization_percent"), f"{label} GPU utilization"
    )
    statistic = value.get("gpu_utilization_statistic")
    samples = _nonnegative_int(value.get("gpu_sample_count"), f"{label} GPU samples")
    peak_vram = _nonnegative_int(value.get("peak_vram_bytes"), f"{label} peak VRAM")
    if not isinstance(device, str) or not device:
        raise RuntimeError(f"{label} execution device is invalid")
    if utilization < 0.0 or utilization > 100.0:
        raise RuntimeError(f"{label} GPU utilization is outside [0, 100]")
    if device == "cpu" and allow_cpu:
        if (
            utilization != 0.0
            or statistic != "NOT_APPLICABLE"
            or samples != 0
            or peak_vram != 0
        ):
            raise RuntimeError(f"{label} CPU execution has nonzero GPU measurements")
        return
    if (
        not device.startswith("cuda:")
        or statistic not in {"MEAN", "PEAK"}
        or samples <= 0
        or peak_vram <= 0
    ):
        raise RuntimeError(f"{label} GPU sampling evidence is incomplete")


def _validate_equivalence(results, artifacts, identity, legacy_root, new_root):
    _require_exact_fields(results, EQUIVALENCE_RESULT_FIELDS, "equivalence results")
    bound_inputs = {path: binding for binding, path in artifacts}
    smoke_binding = results["smoke_report"]
    smoke_path = _authenticate_file_binding(
        smoke_binding,
        legacy_root=legacy_root,
        new_root=new_root,
        require_external=True,
    )
    formal_binding = results["formal_subset_report"]
    formal_path = _authenticate_file_binding(
        formal_binding,
        legacy_root=legacy_root,
        new_root=new_root,
        require_external=True,
    )
    oracle = results["smoke_oracle"]
    _require_exact_fields(oracle, SMOKE_ORACLE_FIELDS, "smoke oracle binding")
    oracle_binding = oracle["manifest"]
    oracle_path = _authenticate_file_binding(
        oracle_binding,
        legacy_root=legacy_root,
        new_root=new_root,
        require_external=True,
    )
    expected_inputs = {
        smoke_path: smoke_binding,
        oracle_path: oracle_binding,
        formal_path: formal_binding,
    }
    if len(expected_inputs) != 3 or bound_inputs != expected_inputs:
        raise RuntimeError(
            "equivalence inputs must be its smoke, oracle-manifest, and subset reports"
        )
    smoke_report = _read_bound_json(smoke_binding, smoke_path)
    oracle_manifest = _read_bound_json(oracle_binding, oracle_path)
    _validate_smoke_comparator_report(
        smoke_report,
        oracle_manifest=oracle_manifest,
        oracle_binding=oracle,
        oracle_path=oracle_path,
        expected_identity=identity,
    )
    formal_report = _read_bound_json(formal_binding, formal_path)
    _validate_real_subset_report(
        formal_report,
        report_path=formal_path,
        expected_identity=identity,
        legacy_root=legacy_root,
        new_root=new_root,
    )


def _validate_smoke_comparator_report(
    report,
    *,
    oracle_manifest,
    oracle_binding,
    oracle_path,
    expected_identity,
):
    kind = "smoke"
    if not isinstance(report, dict):
        raise RuntimeError(f"{kind} comparator report must be an object")
    _require_exact_fields(report, _COMPARATOR_REPORT_FIELDS, f"{kind} comparator report")
    if (
        report.get("schema_version") != EVALUATION_COMPARATOR_SCHEMA
        or report.get("status") != "PASS"
        or report.get("equivalent") is not True
        or type(report.get("exact_match")) is not bool
    ):
        raise RuntimeError(f"{kind} comparator report did not pass")
    tolerance = report.get("tolerance")
    _require_exact_fields(
        tolerance, {"absolute", "relative", "rule"}, f"{kind} comparator tolerance"
    )
    if (
        tolerance.get("absolute") != 0.0
        or tolerance.get("relative") != 0.0
        or tolerance.get("rule") != _COMPARATOR_TOLERANCE_RULE
    ):
        raise RuntimeError(f"{kind} comparator must use zero tolerance")
    try:
        generated = datetime.fromisoformat(str(report["generated_utc"]))
    except ValueError as error:
        raise RuntimeError(f"{kind} comparator timestamp is invalid") from error
    reference_root = Path(str(report["reference_root"]))
    candidate_root = Path(str(report["candidate_root"]))
    if (
        generated.tzinfo is None
        or generated.utcoffset() is None
        or generated.utcoffset().total_seconds() != 0
        or not reference_root.is_absolute()
        or not candidate_root.is_absolute()
        or reference_root == candidate_root
    ):
        raise RuntimeError(f"{kind} comparator roots or timestamp are invalid")
    if report.get("required_artifacts") != list(_COMPARATOR_ARTIFACTS):
        raise RuntimeError(f"{kind} comparator artifact contract is incomplete")
    csv = report.get("csv")
    if not isinstance(csv, dict) or set(csv) != set(CSV_CONTRACTS):
        raise RuntimeError(f"{kind} comparator must cover exactly nine CSV artifacts")
    for item in csv.values():
        if (
            not isinstance(item, dict)
            or item.get("equivalent") is not True
            or type(item.get("reference_row_count")) is not int
            or item["reference_row_count"] <= 0
            or type(item.get("candidate_row_count")) is not int
            or item["candidate_row_count"] != item["reference_row_count"]
        ):
            raise RuntimeError(
                f"{kind} comparator contains an empty or failed CSV artifact"
            )
    summary = report.get("summary")
    _require_exact_fields(
        summary,
        {
            "csv_passed",
            "csv_total",
            "csv_exact",
            "gate_equivalent",
            "gate_exact",
            "manifest_comparable",
            "manifest_exact",
        },
        f"{kind} comparator summary",
    )
    if (
        not isinstance(summary, dict)
        or type(summary.get("csv_passed")) is not int
        or summary.get("csv_passed") != len(CSV_CONTRACTS)
        or type(summary.get("csv_total")) is not int
        or summary.get("csv_total") != len(CSV_CONTRACTS)
        or summary.get("gate_equivalent") is not True
        or summary.get("manifest_comparable") is not True
    ):
        raise RuntimeError(f"{kind} comparator summary is incomplete")
    gate = report.get("gate")
    if (
        not isinstance(gate, dict)
        or gate.get("equivalent") is not True
        or not isinstance(gate.get("comparison"), dict)
        or gate["comparison"].get("equivalent") is not True
        or not isinstance(gate.get("reference"), dict)
        or gate["reference"].get("valid") is not True
        or not isinstance(gate.get("candidate"), dict)
        or gate["candidate"].get("valid") is not True
    ):
        raise RuntimeError(f"{kind} comparator gate did not pass")
    bindings = gate.get("identity_binding")
    if (
        not isinstance(bindings, dict)
        or not isinstance(bindings.get("reference"), dict)
        or bindings["reference"].get("valid") is not True
        or not isinstance(bindings.get("candidate"), dict)
        or bindings["candidate"].get("valid") is not True
    ):
        raise RuntimeError(f"{kind} comparator gate identity is not bound")
    manifest = report.get("manifest")
    if not isinstance(manifest, dict) or manifest.get("comparable") is not True:
        raise RuntimeError(f"{kind} comparator manifests are not comparable")
    identities = {}
    for side in ("reference", "candidate"):
        manifest_side = manifest.get(side)
        if not isinstance(manifest_side, dict) or manifest_side.get("valid") is not True:
            raise RuntimeError(f"{kind} comparator {side} manifest is invalid")
        identities[side] = manifest_side.get("identity")
        _validate_smoke_identity(identities[side], side=side)
    reference = identities["reference"]
    candidate = identities["candidate"]
    frozen_fields = {
        "artifact_label",
        "dataset_sha256",
        "config_sha256",
        "fixture",
        "scientific_use",
        "requirements_lock_sha256",
    }
    if any(reference[field] != candidate[field] for field in frozen_fields):
        raise RuntimeError("smoke oracle and candidate frozen identities differ")
    if (
        reference["source_tree_sha256"] == candidate["source_tree_sha256"]
        or candidate["source_tree_sha256"]
        != expected_identity["new_source_tree_sha256"]
        or candidate["requirements_lock_sha256"]
        != expected_identity["requirements_lock_sha256"]
    ):
        raise RuntimeError("smoke candidate does not bind the new implementation")
    _require_exact_fields(
        oracle_binding["identity"],
        COMPARATOR_IDENTITY_FIELDS,
        "smoke oracle identity",
    )
    if oracle_binding["identity"] != reference:
        raise RuntimeError("smoke oracle identity does not bind the comparator reference")
    if (
        manifest["reference"].get("path") != str(oracle_path)
        or (Path(str(report["reference_root"])) / "manifest.json").resolve()
        != oracle_path
    ):
        raise RuntimeError("smoke comparator reference does not bind the oracle manifest")
    if (
        not isinstance(oracle_manifest, dict)
        or oracle_manifest.get("schema_version")
        != "csi-pairs-formal-stage-manifest-v2.1-v6"
        or not isinstance(oracle_manifest.get("files"), list)
    ):
        raise RuntimeError("smoke oracle manifest must be an object")
    if any(oracle_manifest.get(field) != reference[field] for field in reference):
        raise RuntimeError("smoke oracle manifest identity changed")


def _validate_smoke_identity(value, *, side):
    _require_exact_fields(
        value, COMPARATOR_IDENTITY_FIELDS, f"smoke comparator {side} identity"
    )
    if (
        value["fixture"] is not True
        or value["scientific_use"] != "FORBIDDEN"
        or not isinstance(value["artifact_label"], str)
        or not value["artifact_label"].strip()
        or not _is_hex(value["dataset_sha256"], 64)
        or not _is_hex(value["config_sha256"], 64)
        or not _is_hex(value["source_tree_sha256"], 64)
        or not _is_hex(value["requirements_lock_sha256"], 64)
    ):
        raise RuntimeError(f"smoke comparator {side} identity mismatch")
    runtime = value["runtime_provenance"]
    if not isinstance(runtime, dict):
        raise RuntimeError(f"smoke comparator {side} runtime provenance is missing")
    if (
        runtime.get("source_tree_sha256") != value["source_tree_sha256"]
        or runtime.get("requirements_lock_sha256")
        != value["requirements_lock_sha256"]
        or value["runtime_provenance_sha256"] != _canonical_sha256(runtime)
    ):
        raise RuntimeError(f"smoke comparator {side} runtime provenance mismatch")


def _validate_real_subset_report(
    report,
    *,
    report_path,
    expected_identity,
    legacy_root,
    new_root,
):
    kind = "formal real-subset"
    _require_exact_fields(report, _REAL_SUBSET_REPORT_FIELDS, f"{kind} report")
    if (
        report["schema_version"] != REAL_SUBSET_REPORT_SCHEMA
        or report["status"] != "PASS"
        or report["passed"] is not True
        or report["authoritative_layer"] != "layers.cross_source_end_to_end"
        or report["report_path"] != str(report_path)
        or not Path(str(report["report_path"])).is_absolute()
        or Path(str(report["report_path"])).resolve() != report_path
    ):
        raise RuntimeError(f"{kind} report identity is invalid")
    _parse_utc(report["created_utc"], f"{kind} created_utc")
    _validate_real_subset_contract(report["contract"])

    shared_inputs = _validate_real_subset_inputs(
        report["shared_inputs"], expected_identity=expected_identity
    )
    workers = report["workers"]
    _require_exact_fields(workers, {"legacy", "new"}, f"{kind} workers")
    authenticated = {}
    for side, role, commit_key, tree_key in (
        ("legacy", "frozen_legacy", "legacy_commit", "legacy_source_tree_sha256"),
        ("new", "streaming_candidate", "new_commit", "new_source_tree_sha256"),
    ):
        authenticated[side] = _validate_real_subset_worker(
            workers[side],
            side=side,
            role=role,
            expected_commit=expected_identity[commit_key],
            expected_source_tree=expected_identity[tree_key],
            expected_identity=expected_identity,
            shared_inputs=report["shared_inputs"],
            selection=report["selection"],
            legacy_root=legacy_root,
            new_root=new_root,
        )

    legacy_worker = authenticated["legacy"]
    new_worker = authenticated["new"]
    _validate_real_subset_worker_pair(
        legacy_worker,
        new_worker,
        expected_identity=expected_identity,
        shared_inputs=shared_inputs,
    )
    _validate_real_subset_selection(
        report["selection"],
        legacy_request=legacy_worker["request"],
        new_request=new_worker["request"],
    )
    _validate_real_subset_layers(
        report["layers"],
        legacy_fragment=legacy_worker["fragment"],
        new_fragment=new_worker["fragment"],
    )
    _validate_real_subset_report_read_only(
        report["read_only"],
        legacy_fragment=legacy_worker["fragment"],
        new_fragment=new_worker["fragment"],
    )
    _validate_real_subset_report_resources(
        report["resources"],
        legacy_fragment=legacy_worker["fragment"],
        new_fragment=new_worker["fragment"],
    )


def _validate_real_subset_contract(value):
    _require_exact_fields(value, _REAL_SUBSET_CONTRACT_FIELDS, "real-subset contract")
    if (
        value["formal_dataset_only"] is not True
        or value["full_legacy_evaluation_executed"] is not False
        or value["legacy_run_read_only"] is not True
        or value["frozen_legacy_run_body_executed"] is not True
        or value["cross_source_expected_implementation_fields"]
        != list(REAL_SUBSET_IMPLEMENTATION_FIELDS)
        or value["scientific_float_requirement"]
        != "IEEE-754 binary64 bitwise equality"
        or type(value["absolute_tolerance"]) is not float
        or value["absolute_tolerance"] != 0.0
        or type(value["relative_tolerance"]) is not float
        or value["relative_tolerance"] != 0.0
    ):
        raise RuntimeError("real-subset contract is not formal zero-tolerance evidence")


def _validate_real_subset_inputs(value, *, expected_identity):
    _require_exact_fields(value, _REAL_SUBSET_INPUT_FIELDS, "real-subset shared inputs")
    if (
        value["config_sha256"] != expected_identity["config_sha256"]
        or value["dataset_sha256"] != expected_identity["dataset_sha256"]
    ):
        raise RuntimeError("real-subset shared formal identity mismatch")
    files = value["files"]
    if not isinstance(files, list) or len(files) < 2:
        raise RuntimeError("real-subset shared inputs are incomplete")
    authenticated = {}
    ordered_paths = []
    for record in files:
        path = _authenticate_real_subset_file_record(
            record, label="real-subset shared input", allow_identity=True
        )
        if path in authenticated:
            raise RuntimeError("real-subset shared input paths must be unique")
        authenticated[path] = record
        ordered_paths.append(str(path))
    if ordered_paths != sorted(ordered_paths):
        raise RuntimeError("real-subset shared inputs must use canonical path order")
    return authenticated


def _validate_real_subset_worker(
    value,
    *,
    side,
    role,
    expected_commit,
    expected_source_tree,
    expected_identity,
    shared_inputs,
    selection,
    legacy_root,
    new_root,
):
    label = f"real-subset {side} worker"
    _require_exact_fields(value, _REAL_SUBSET_WORKER_FIELDS, label)
    if value["role"] != role:
        raise RuntimeError(f"{label} role is invalid")
    source = _validate_real_subset_source_identity(
        value["source_identity"],
        side=side,
        expected_commit=expected_commit,
        expected_source_tree=expected_source_tree,
    )
    runtime = value["runtime_provenance"]
    if not isinstance(runtime, Mapping) or not runtime:
        raise RuntimeError(f"{label} runtime provenance is missing")
    runtime_digest = _canonical_sha256(runtime)
    if (
        runtime.get("source_tree_sha256") != expected_source_tree
        or runtime.get("requirements_lock_sha256")
        != expected_identity["requirements_lock_sha256"]
        or source["runtime_provenance_sha256"] != runtime_digest
    ):
        raise RuntimeError(f"{label} runtime provenance identity mismatch")
    resources = _validate_real_subset_resources(value["resources"], label)

    request_path = _authenticate_real_subset_file_record(
        value["request"],
        label=f"{label} request",
        allow_identity=False,
        legacy_root=legacy_root,
        new_root=new_root,
        require_external=True,
    )
    fragment_path = _authenticate_real_subset_file_record(
        value["fragment"],
        label=f"{label} fragment",
        allow_identity=False,
        legacy_root=legacy_root,
        new_root=new_root,
        require_external=True,
    )
    if request_path == fragment_path:
        raise RuntimeError(f"{label} request and fragment must be distinct")
    request = _read_bound_json(value["request"], request_path)
    _validate_real_subset_request(
        request,
        label=label,
        role=role,
        expected_commit=expected_commit,
        source=source,
        request_path=request_path,
        fragment_path=fragment_path,
        expected_identity=expected_identity,
        shared_inputs=shared_inputs,
        legacy_root=legacy_root,
    )
    fragment = _read_bound_json(value["fragment"], fragment_path)
    _validate_real_subset_fragment(
        fragment,
        label=label,
        role=role,
        source=source,
        runtime=runtime,
        resources=resources,
        request_binding=value["request"],
        shared_inputs=shared_inputs,
        selection=selection,
        expected_identity=expected_identity,
        expected_source_tree=expected_source_tree,
    )
    return {
        "worker": value,
        "source": source,
        "runtime": runtime,
        "request": request,
        "request_path": request_path,
        "fragment": fragment,
        "fragment_path": fragment_path,
    }


def _validate_real_subset_source_identity(
    value, *, side, expected_commit, expected_source_tree
):
    label = f"real-subset {side} source identity"
    _require_exact_fields(value, _REAL_SUBSET_SOURCE_FIELDS, label)
    source_root = _regular_directory(value["source_root"], f"{label} source root")
    git_root = _regular_directory(value["git_root"], f"{label} git root")
    if (
        not Path(str(value["source_root"])).is_absolute()
        or str(source_root) != value["source_root"]
        or not Path(str(value["git_root"])).is_absolute()
        or str(git_root) != value["git_root"]
        or not _is_within(source_root, git_root)
        or value["git_commit"] != expected_commit
        or value["source_tree_sha256"] != expected_source_tree
        or value["tracked_tree_clean"] is not True
        or value["tracked_status"] != []
        or value["untracked_scientific_paths"] != []
        or value["tracked_diff_sha256"] != _REAL_SUBSET_EMPTY_DIFF_SHA256
        or not _is_hex(value["runtime_provenance_sha256"], 64)
    ):
        raise RuntimeError(f"{label} is malformed")
    if (
        _real_subset_git(source_root, "rev-parse", "--show-toplevel")
        != str(git_root)
        or _real_subset_git(source_root, "rev-parse", "HEAD") != expected_commit
        or _real_subset_git(
            source_root, "status", "--porcelain", "--untracked-files=no"
        )
        != ""
    ):
        raise RuntimeError(f"{label} no longer names its clean committed source")
    diff = subprocess.run(
        ("git", "-C", str(source_root), "diff", "--binary", "HEAD", "--"),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    if hashlib.sha256(diff).hexdigest() != value["tracked_diff_sha256"]:
        raise RuntimeError(f"{label} tracked diff digest changed")

    expected_files = {"formal_evaluation.py", "formal_evaluation_subset_compare.py"}
    if side == "new":
        expected_files.add("formal_evaluation_streaming.py")
    files = value["files"]
    _require_exact_fields(files, expected_files, f"{label} source files")
    authenticated = {}
    for name, record in files.items():
        _require_exact_fields(record, _REAL_SUBSET_SOURCE_FILE_FIELDS, f"{label} {name}")
        if record["git_tracked"] is not True:
            raise RuntimeError(f"{label} {name} is not committed source")
        path = _authenticate_real_subset_source_file(record, f"{label} {name}")
        if name != "formal_evaluation_subset_compare.py":
            expected_path = source_root / "formal_v2" / name
            if path != expected_path:
                raise RuntimeError(f"{label} {name} path mismatch")
        tracked_root = Path(
            _real_subset_git(path.parent, "rev-parse", "--show-toplevel")
        ).resolve()
        try:
            relative = path.relative_to(tracked_root).as_posix()
        except ValueError as error:
            raise RuntimeError(f"{label} {name} escapes its git root") from error
        completed = subprocess.run(
            (
                "git",
                "-C",
                str(tracked_root),
                "ls-files",
                "--error-unmatch",
                "--",
                relative,
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{label} {name} is not tracked live source")
        authenticated[name] = path
    return {**dict(value), "_paths": authenticated}


def _validate_real_subset_request(
    value,
    *,
    label,
    role,
    expected_commit,
    source,
    request_path,
    fragment_path,
    expected_identity,
    shared_inputs,
    legacy_root,
):
    _require_exact_fields(value, _REAL_SUBSET_REQUEST_FIELDS, f"{label} request body")
    if (
        value["schema_version"] != REAL_SUBSET_REQUEST_SCHEMA
        or value["role"] != role
        or value["legacy_run_root"] != str(legacy_root)
        or value["device"] != "cuda:0"
        or value["expected_git_commit"] != expected_commit
        or value["expected_evaluation_sha256"]
        != source["files"]["formal_evaluation.py"]["sha256"]
        or value["fragment_path"] != str(fragment_path)
    ):
        raise RuntimeError(f"{label} request identity mismatch")
    for field in (
        "batch_size",
        "source_scenes",
        "target_scenes_per_city",
        "checkpoint_count",
    ):
        _positive_int(value[field], f"{label} request {field}")
    if value["checkpoint_arm"] is not None and (
        not isinstance(value["checkpoint_arm"], str)
        or not value["checkpoint_arm"].strip()
    ):
        raise RuntimeError(f"{label} request checkpoint arm is invalid")
    inputs_by_path = {
        Path(str(record["path"])): record for record in shared_inputs["files"]
    }
    for field, digest_key in (
        ("config_path", "config_sha256"),
        ("dataset_path", "dataset_sha256"),
    ):
        raw_path = value[field]
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise RuntimeError(f"{label} request {field} is not absolute")
        path = Path(raw_path).resolve()
        record = inputs_by_path.get(path)
        if (
            record is None
            or raw_path != str(path)
            or record.get("identity_sha256") != expected_identity[digest_key]
        ):
            raise RuntimeError(f"{label} request {field} is not a shared formal input")
        if field == "dataset_path" and record["sha256"] != expected_identity[digest_key]:
            raise RuntimeError(f"{label} dataset bytes do not match formal identity")


def _validate_real_subset_fragment(
    value,
    *,
    label,
    role,
    source,
    runtime,
    resources,
    request_binding,
    shared_inputs,
    selection,
    expected_identity,
    expected_source_tree,
):
    _require_exact_fields(value, _REAL_SUBSET_FRAGMENT_FIELDS, f"{label} fragment")
    if (
        value["schema_version"] != REAL_SUBSET_FRAGMENT_SCHEMA
        or value["role"] != role
        or value["source_identity"] != {key: source[key] for key in _REAL_SUBSET_SOURCE_FIELDS}
        or value["runtime_provenance"] != runtime
        or value["resources"] != resources
        or value["input_bindings"] != shared_inputs
        or value["selection"] != selection
    ):
        raise RuntimeError(f"{label} fragment binding mismatch")
    _parse_utc(value["created_utc"], f"{label} fragment created_utc")
    fragment_request = value["request"]
    _require_exact_fields(
        fragment_request,
        _REAL_SUBSET_FRAGMENT_REQUEST_FIELDS,
        f"{label} fragment request binding",
    )
    if fragment_request != {
        key: request_binding[key] for key in _REAL_SUBSET_FRAGMENT_REQUEST_FIELDS
    }:
        raise RuntimeError(f"{label} fragment does not bind its exact request")
    _validate_real_subset_tables(
        value["tables"],
        label=f"{label} tables",
        expected_identity=expected_identity,
        expected_source_tree=expected_source_tree,
        runtime_sha256=source["runtime_provenance_sha256"],
    )
    if role == "frozen_legacy":
        if value["same_source_merged_tables"] is not None:
            raise RuntimeError("frozen legacy fragment fabricated same-source rows")
    else:
        _validate_real_subset_tables(
            value["same_source_merged_tables"],
            label=f"{label} same-source merged tables",
            expected_identity=expected_identity,
            expected_source_tree=expected_source_tree,
            runtime_sha256=source["runtime_provenance_sha256"],
        )
    _validate_real_subset_read_only(
        value["read_only"],
        shared_inputs,
        label,
        role=role,
        legacy_root=Path(str(expected_identity["legacy_run_root"])),
    )


def _validate_real_subset_tables(
    value,
    *,
    label,
    expected_identity,
    expected_source_tree,
    runtime_sha256,
):
    _require_exact_fields(value, set(REAL_SUBSET_TABLE_FIELDS), label)
    for table, fields in REAL_SUBSET_TABLE_FIELDS.items():
        contract = CSV_CONTRACTS.get(table)
        if contract is None or tuple(contract.fields) != tuple(fields):
            raise RuntimeError(f"{label} {table} has no frozen field contract")
        rows = value[table]
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"{label} {table} is empty")
        keys = []
        for row in rows:
            if not isinstance(row, dict) or tuple(row) != tuple(fields):
                raise RuntimeError(f"{label} {table} field set/order mismatch")
            if (
                not isinstance(row["artifact_label"], str)
                or not row["artifact_label"].strip()
                or row["dataset_sha256"] != expected_identity["dataset_sha256"]
                or row["config_sha256"] != expected_identity["config_sha256"]
                or row["fixture"] is not False
                or row["scientific_use"] != "CANDIDATE_NOT_CLAIM"
                or row["source_tree_sha256"] != expected_source_tree
                or row["requirements_lock_sha256"]
                != expected_identity["requirements_lock_sha256"]
                or row["runtime_provenance_sha256"] != runtime_sha256
            ):
                raise RuntimeError(f"{label} {table} row evidence mismatch")
            for field in contract.float_fields:
                if type(row[field]) is not float or not math.isfinite(row[field]):
                    raise RuntimeError(f"{label} {table}/{field} is not a finite float")
            for field in contract.structured_fields:
                if not isinstance(row[field], (Mapping, list)):
                    raise RuntimeError(f"{label} {table}/{field} is not structured")
            key = tuple(row[field] for field in REAL_SUBSET_PRIMARY_KEYS[table])
            try:
                hash(key)
            except TypeError as error:
                raise RuntimeError(f"{label} {table} primary key is unhashable") from error
            keys.append(key)
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"{label} {table} primary keys are not unique")


def _validate_real_subset_read_only(
    value, shared_inputs, label, *, role, legacy_root
):
    _require_exact_fields(value, _REAL_SUBSET_READ_ONLY_FIELDS, f"{label} read-only")
    files = value["files"]
    if value["passed"] is not True or not isinstance(files, list):
        raise RuntimeError(f"{label} did not prove read-only formal inputs")
    shared_by_path = {
        Path(str(record["path"])): record for record in shared_inputs["files"]
    }
    observed = set()
    for audit in files:
        _require_exact_fields(
            audit, _REAL_SUBSET_READ_ONLY_RECORD_FIELDS, f"{label} read-only record"
        )
        before = audit["before"]
        after = audit["after"]
        path = _authenticate_real_subset_file_record(
            before, label=f"{label} read-only before", allow_identity=True
        )
        after_path = _authenticate_real_subset_file_record(
            after, label=f"{label} read-only after", allow_identity=False
        )
        before_core = {key: before[key] for key in _REAL_SUBSET_ARTIFACT_FIELDS}
        if (
            path in observed
            or audit["path"] != str(path)
            or after_path != path
            or before_core != after
            or shared_by_path.get(path) != before
            or audit["unchanged"] is not True
        ):
            raise RuntimeError(f"{label} read-only audit mismatch")
        observed.add(path)
    if observed != set(shared_by_path):
        raise RuntimeError(f"{label} read-only audit does not cover every shared input")
    evaluation_audit = value["legacy_evaluation"]
    if role == "streaming_candidate":
        if evaluation_audit is not None:
            raise RuntimeError("candidate real-subset worker fabricated legacy snapshot")
        return
    _require_exact_fields(
        evaluation_audit,
        _REAL_SUBSET_EVALUATION_AUDIT_FIELDS,
        "frozen real-subset legacy evaluation audit",
    )
    requested_root = legacy_root / "evaluation"
    expected_root = requested_root.resolve()
    before = evaluation_audit["before"]
    after = evaluation_audit["after"]
    _require_exact_fields(
        before,
        _REAL_SUBSET_DIRECTORY_SNAPSHOT_FIELDS,
        "frozen real-subset legacy evaluation before snapshot",
    )
    _require_exact_fields(
        after,
        _REAL_SUBSET_DIRECTORY_SNAPSHOT_FIELDS,
        "frozen real-subset legacy evaluation after snapshot",
    )
    live = _real_subset_directory_snapshot(requested_root)
    if (
        evaluation_audit["root"] != str(expected_root)
        or evaluation_audit["unchanged"] is not True
        or before != after
        or before != live
    ):
        raise RuntimeError("frozen real-subset legacy evaluation snapshot changed")


def _validate_real_subset_worker_pair(
    legacy_worker, new_worker, *, expected_identity, shared_inputs
):
    legacy_source = legacy_worker["source"]
    new_source = new_worker["source"]
    if (
        legacy_source["source_root"] == new_source["source_root"]
        or legacy_source["git_commit"] == new_source["git_commit"]
        or legacy_source["source_tree_sha256"] == new_source["source_tree_sha256"]
        or legacy_worker["request_path"] == new_worker["request_path"]
        or legacy_worker["fragment_path"] == new_worker["fragment_path"]
    ):
        raise RuntimeError("real-subset workers are not independent cross-source executions")
    legacy_harness = legacy_source["_paths"]["formal_evaluation_subset_compare.py"]
    new_harness = new_source["_paths"]["formal_evaluation_subset_compare.py"]
    if (
        legacy_harness != new_harness
        or new_harness
        != Path(new_source["source_root"]) / "formal_v2" / new_harness.name
    ):
        raise RuntimeError("real-subset workers did not bind the candidate comparator harness")
    common_request_fields = _REAL_SUBSET_REQUEST_FIELDS - {
        "role",
        "fragment_path",
        "expected_git_commit",
        "expected_evaluation_sha256",
    }
    if any(
        legacy_worker["request"][field] != new_worker["request"][field]
        for field in common_request_fields
    ):
        raise RuntimeError("real-subset workers did not execute the same request")
    if (
        legacy_worker["fragment"]["input_bindings"]
        != new_worker["fragment"]["input_bindings"]
        or legacy_worker["fragment"]["selection"]
        != new_worker["fragment"]["selection"]
    ):
        raise RuntimeError("real-subset workers did not bind identical inputs and selection")
    if set(shared_inputs) != {
        Path(str(record["path"]))
        for record in legacy_worker["fragment"]["input_bindings"]["files"]
    }:
        raise RuntimeError("real-subset worker inputs differ from live shared inputs")
    if (
        legacy_source["git_commit"] != expected_identity["legacy_commit"]
        or new_source["git_commit"] != expected_identity["new_commit"]
    ):
        raise RuntimeError("real-subset worker commits changed")


def _validate_real_subset_selection(value, *, legacy_request, new_request):
    _require_exact_fields(value, _REAL_SUBSET_SELECTION_FIELDS, "real-subset selection")
    if (
        value["scene_rule"] != REAL_SUBSET_SELECTION_RULE
        or value["checkpoint_rule"] != REAL_SUBSET_CHECKPOINT_SELECTION_RULE
    ):
        raise RuntimeError("real-subset selection rule mismatch")
    scenes = value["scenes_in_execution_order"]
    checkpoints = value["checkpoints_in_execution_order"]
    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError("real-subset scene selection is empty")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise RuntimeError("real-subset checkpoint selection is empty")
    bank_ids = []
    scene_ids = []
    scene_indices = []
    source_count = 0
    target_city_counts = {}
    for scene in scenes:
        _require_exact_fields(scene, _REAL_SUBSET_SCENE_FIELDS, "real-subset scene")
        if (
            type(scene["scene_index"]) is not int
            or scene["scene_index"] < 0
            or not all(
                isinstance(scene[field], str) and scene[field]
                for field in (
                    "scene_id",
                    "scene_role",
                    "city_id",
                    "bank_id",
                    "base_map_cluster_id",
                )
            )
            or not _is_hex(scene["canonical_base_map_digest"], 64)
            or not _is_hex(scene["canonical_bank_digest"], 64)
            or scene["scene_role"]
            not in {"source_final_unseen_bank", "target"}
        ):
            raise RuntimeError("real-subset scene identity is malformed")
        bank_ids.append(scene["bank_id"])
        scene_ids.append(scene["scene_id"])
        scene_indices.append(scene["scene_index"])
        if scene["scene_role"] == "source_final_unseen_bank":
            source_count += 1
        else:
            target_city_counts[scene["city_id"]] = (
                target_city_counts.get(scene["city_id"], 0) + 1
            )
    if (
        len(bank_ids) != len(set(bank_ids))
        or len(scene_ids) != len(set(scene_ids))
        or len(scene_indices) != len(set(scene_indices))
        or list(zip(bank_ids, scene_indices)) != sorted(zip(bank_ids, scene_indices))
        or source_count != legacy_request["source_scenes"]
        or not target_city_counts
        or any(
            count != legacy_request["target_scenes_per_city"]
            for count in target_city_counts.values()
        )
    ):
        raise RuntimeError("real-subset scene coverage/order is invalid")

    try:
        config = parse_strict_json(
            Path(str(legacy_request["config_path"])).read_text(encoding="utf-8")
        )
        arm_order = config["factorial"]["arms"]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError("real-subset bound config has no checkpoint arm order") from error
    if (
        not isinstance(arm_order, list)
        or not arm_order
        or any(not isinstance(arm, str) or not arm for arm in arm_order)
        or len(arm_order) != len(set(arm_order))
    ):
        raise RuntimeError("real-subset checkpoint arm order is malformed")
    arm_rank = {arm: index for index, arm in enumerate(arm_order)}
    checkpoint_keys = []
    for checkpoint in checkpoints:
        _require_exact_fields(
            checkpoint, _REAL_SUBSET_CHECKPOINT_FIELDS, "real-subset checkpoint"
        )
        if (
            type(checkpoint["seed"]) is not int
            or not isinstance(checkpoint["arm"], str)
            or not checkpoint["arm"]
            or checkpoint["arm"] not in arm_rank
            or not _is_hex(checkpoint["sha256"], 64)
        ):
            raise RuntimeError("real-subset checkpoint identity is malformed")
        checkpoint_keys.append(
            (checkpoint["seed"], checkpoint["arm"], checkpoint["sha256"])
        )
    if (
        len(checkpoint_keys) != len(set(checkpoint_keys))
        or checkpoint_keys
        != sorted(
            checkpoint_keys,
            key=lambda item: (item[0], arm_rank[item[1]], item[2]),
        )
        or len(checkpoints) != legacy_request["checkpoint_count"]
        or len(checkpoints) != new_request["checkpoint_count"]
        or (
            legacy_request["checkpoint_arm"] is not None
            and any(
                checkpoint["arm"] != legacy_request["checkpoint_arm"]
                for checkpoint in checkpoints
            )
        )
    ):
        raise RuntimeError("real-subset checkpoint coverage is invalid")
    if value["sha256"] != _real_subset_selection_sha256(scenes, checkpoints):
        raise RuntimeError("real-subset selection digest is invalid")


def _validate_real_subset_layers(value, *, legacy_fragment, new_fragment):
    _require_exact_fields(value, _REAL_SUBSET_LAYER_FIELDS, "real-subset layers")
    cross = value["cross_source_end_to_end"]
    same = value["same_source_decomposition"]
    _require_exact_fields(cross, _REAL_SUBSET_CROSS_LAYER_FIELDS, "cross-source layer")
    _require_exact_fields(same, _REAL_SUBSET_SAME_LAYER_FIELDS, "same-source layer")
    cross_comparison = _compare_real_subset_tables(
        legacy_fragment["tables"],
        new_fragment["tables"],
        implementation_fields=REAL_SUBSET_IMPLEMENTATION_FIELDS,
    )
    cross_gate = _real_subset_gate_aggregates(
        legacy_fragment["tables"],
        new_fragment["tables"],
        implementation_fields=REAL_SUBSET_IMPLEMENTATION_FIELDS,
    )
    same_comparison = _compare_real_subset_tables(
        new_fragment["same_source_merged_tables"], new_fragment["tables"]
    )
    same_gate = _real_subset_gate_aggregates(
        new_fragment["same_source_merged_tables"], new_fragment["tables"]
    )
    expected_cross = {
        "authoritative": True,
        "legacy_source_role": "frozen_legacy_oracle",
        "candidate_source_role": "streaming_candidate",
        "comparison": cross_comparison,
        "gate_required_aggregates": cross_gate,
    }
    expected_same = {
        "authoritative": False,
        "description": _REAL_SUBSET_DESCRIPTION,
        "comparison": same_comparison,
        "gate_required_aggregates": same_gate,
    }
    if cross != expected_cross or same != expected_same:
        raise RuntimeError("real-subset comparison layers do not match bound worker rows")
    for label, comparison, gate in (
        ("cross-source", cross_comparison, cross_gate),
        ("same-source", same_comparison, same_gate),
    ):
        if (
            comparison["exact"] is not True
            or comparison["bitwise_float_equal"] is not True
            or comparison["field_contract_equal"] is not True
            or comparison["row_order_equal"] is not True
            or comparison["primary_keys_unique"] is not True
            or comparison["table_count"] != len(REAL_SUBSET_TABLE_FIELDS)
            or comparison["legacy_total_row_count"] < len(REAL_SUBSET_TABLE_FIELDS)
            or comparison["float_comparisons"] <= 0
            or comparison["bitwise_equal_float_count"]
            != comparison["float_comparisons"]
            or comparison["max_absolute_difference"] != 0.0
            or comparison["max_relative_difference"] != 0.0
            or gate["exact"] is not True
            or gate["table_count"] != len(REAL_SUBSET_GATE_TABLES)
            or set(gate["tables"]) != set(REAL_SUBSET_GATE_TABLES)
            or gate["formal_gate_executed"] is not False
        ):
            raise RuntimeError(f"{label} real-subset comparison is not exact")


def _validate_real_subset_report_read_only(value, *, legacy_fragment, new_fragment):
    _require_exact_fields(
        value, {"passed", "legacy", "new"}, "real-subset report read-only"
    )
    if (
        value["passed"] is not True
        or value["legacy"] != legacy_fragment["read_only"]
        or value["new"] != new_fragment["read_only"]
    ):
        raise RuntimeError("real-subset report read-only summary mismatch")


def _validate_real_subset_report_resources(value, *, legacy_fragment, new_fragment):
    _require_exact_fields(
        value, {"elapsed_seconds", "legacy", "new"}, "real-subset report resources"
    )
    _positive_number(value["elapsed_seconds"], "real-subset report elapsed time")
    if (
        value["legacy"] != legacy_fragment["resources"]
        or value["new"] != new_fragment["resources"]
    ):
        raise RuntimeError("real-subset report resource summary mismatch")


def _validate_real_subset_resources(value, label):
    _require_exact_fields(value, _REAL_SUBSET_RESOURCE_FIELDS, f"{label} resources")
    _positive_number(value["elapsed_seconds"], f"{label} elapsed time")
    _positive_int(value["maximum_resident_set_bytes"], f"{label} maximum RSS")
    _positive_int(value["maximum_cuda_allocated_bytes"], f"{label} maximum VRAM")
    return dict(value)


def _authenticate_real_subset_source_file(value, label):
    binding = {key: value[key] for key in FILE_BINDING_FIELDS}
    return _authenticate_file_binding(binding)


def _authenticate_real_subset_file_record(
    value,
    *,
    label,
    allow_identity,
    legacy_root=None,
    new_root=None,
    require_external=False,
):
    allowed = (
        (_REAL_SUBSET_ARTIFACT_FIELDS, _REAL_SUBSET_IDENTITY_ARTIFACT_FIELDS)
        if allow_identity
        else (_REAL_SUBSET_ARTIFACT_FIELDS,)
    )
    if not isinstance(value, Mapping) or set(value) not in allowed:
        raise RuntimeError(f"{label} fields are invalid")
    binding = {key: value[key] for key in FILE_BINDING_FIELDS}
    path = _authenticate_file_binding(
        binding,
        legacy_root=legacy_root,
        new_root=new_root,
        require_external=require_external,
    )
    if type(value["mtime_ns"]) is not int or path.stat().st_mtime_ns != value["mtime_ns"]:
        raise RuntimeError(f"{label} mtime changed")
    if "identity_sha256" in value and not _is_hex(value["identity_sha256"], 64):
        raise RuntimeError(f"{label} identity SHA-256 is malformed")
    return path


def _real_subset_selection_sha256(scenes, checkpoints):
    payload = {
        "scene_rule": REAL_SUBSET_SELECTION_RULE,
        "checkpoint_rule": REAL_SUBSET_CHECKPOINT_SELECTION_RULE,
        "scenes": list(scenes),
        "checkpoints": [
            {
                "seed": checkpoint["seed"],
                "arm": checkpoint["arm"],
                "sha256": checkpoint["sha256"],
            }
            for checkpoint in checkpoints
        ],
    }
    return _canonical_sha256(payload)


def _real_subset_directory_snapshot(path):
    root = _regular_directory(path, "real-subset legacy evaluation root")
    root_stat = root.stat()
    entries = []
    total_file_bytes = 0
    for entry in sorted(
        root.rglob("*"), key=lambda candidate: candidate.relative_to(root).as_posix()
    ):
        if entry.is_symlink():
            raise RuntimeError("real-subset legacy evaluation snapshot contains a symlink")
        entry_stat = entry.stat()
        relative = entry.relative_to(root).as_posix()
        if entry.is_dir():
            record = {
                "path": relative,
                "kind": "directory",
                "mode": entry_stat.st_mode,
                "mtime_ns": entry_stat.st_mtime_ns,
            }
        elif entry.is_file():
            total_file_bytes += entry_stat.st_size
            record = {
                "path": relative,
                "kind": "file",
                "mode": entry_stat.st_mode,
                "bytes": entry_stat.st_size,
                "mtime_ns": entry_stat.st_mtime_ns,
                "sha256": _file_size_sha256(entry)[1],
            }
        else:
            raise RuntimeError("real-subset legacy evaluation snapshot contains a special file")
        entries.append(record)
    return {
        "root": str(root),
        "root_mode": root_stat.st_mode,
        "root_mtime_ns": root_stat.st_mtime_ns,
        "entry_count": len(entries),
        "total_file_bytes": total_file_bytes,
        "entries_sha256": _canonical_sha256(entries),
    }


def _real_subset_git(root, *arguments):
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"real-subset git identity check failed: {' '.join(arguments)}"
        )
    return completed.stdout.strip()


def _validate_resume(results, artifacts):
    _require_observation_artifacts(artifacts, "resume")
    _require_exact_fields(results, RESUME_RESULT_FIELDS, "resume results")
    _positive_int(results["reused_shard_count"], "reused shard count")
    _nonnegative_int(results["recomputed_shard_count"], "recomputed shard count")
    if (
        results["interrupted"] is not True
        or results["resumed"] is not True
        or results["completed"] is not True
        or results["output_equivalent"] is not True
    ):
        raise RuntimeError("resume evidence did not authenticate successful reuse")


def _validate_corrupt(results, artifacts):
    _require_observation_artifacts(artifacts, "corrupt-checkpoint")
    _require_exact_fields(results, CORRUPT_RESULT_FIELDS, "corrupt-checkpoint results")
    corrupt = _positive_int(results["corrupt_shard_count"], "corrupt shard count")
    quarantined = _positive_int(
        results["quarantined_shard_count"], "quarantined shard count"
    )
    recomputed = _positive_int(
        results["recomputed_shard_count"], "recomputed shard count"
    )
    _nonnegative_int(
        results["unaffected_shard_count"], "unaffected shard count"
    )
    if (
        results["corruption_detected"] is not True
        or quarantined != corrupt
        or recomputed != corrupt
    ):
        raise RuntimeError("corrupt-checkpoint recovery evidence is invalid")


def _validate_stale(results, artifacts):
    _require_observation_artifacts(artifacts, "stale-checkpoint")
    _require_exact_fields(results, STALE_RESULT_FIELDS, "stale-checkpoint results")
    if (
        results["stale_identity_detected"] is not True
        or _positive_int(results["rejected_shard_count"], "rejected shard count") <= 0
        or _nonnegative_int(
            results["reused_stale_shard_count"], "reused stale shard count"
        )
        != 0
    ):
        raise RuntimeError("stale-checkpoint rejection evidence is invalid")


def _validate_lock(results, artifacts):
    _require_observation_artifacts(artifacts, "lock")
    _require_exact_fields(results, LOCK_RESULT_FIELDS, "lock results")
    if (
        results["second_writer_attempted"] is not True
        or results["second_writer_rejected"] is not True
        or results["first_writer_preserved"] is not True
        or _positive_int(results["write_count"], "lock write count") != 1
    ):
        raise RuntimeError("lock exclusion evidence is invalid")


def _validate_progress(results, artifacts):
    _require_observation_artifacts(artifacts, "progress")
    _require_exact_fields(results, PROGRESS_RESULT_FIELDS, "progress results")
    _positive_int(results["update_count"], "progress update count")
    completed = _positive_int(results["completed_units"], "completed units")
    total = _positive_int(results["total_units"], "total units")
    if (
        results["monotonic"] is not True
        or results["atomic"] is not True
        or results["final_complete"] is not True
        or completed != total
    ):
        raise RuntimeError("progress evidence is invalid")


def _validate_diff(payload, results, artifacts, legacy_root, new_root):
    _require_exact_fields(results, DIFF_RESULT_FIELDS, "base-to-new diff results")
    diff_path = _authenticate_file_binding(
        results["diff"],
        legacy_root=legacy_root,
        new_root=new_root,
        require_external=True,
    )
    if (
        len(artifacts) != 1
        or artifacts[0][1] != diff_path
        or artifacts[0][0] != results["diff"]
        or payload.get("diff_sha256") != results["diff"].get("sha256")
        or results["reviewed"] is not True
        or results["applies_cleanly"] is not True
        or _positive_int(results["changed_file_count"], "changed file count") <= 0
    ):
        raise RuntimeError("base-to-new diff evidence is invalid")


def _validate_inventory(results, artifacts, legacy_root):
    _require_exact_fields(results, INVENTORY_RESULT_FIELDS, "legacy inventory results")
    evaluation_root = _regular_directory(
        legacy_root / "evaluation", "legacy evaluation root"
    )
    if results["legacy_evaluation_root"] != str(evaluation_root):
        raise RuntimeError("legacy inventory root mismatch")
    rows = results["files"]
    if not isinstance(rows, list):
        raise RuntimeError("legacy evaluation inventory files must be a list")
    if results["files_sha256"] != _canonical_sha256(rows):
        raise RuntimeError("legacy evaluation inventory digest mismatch")
    bound_by_path = {path: binding for binding, path in artifacts}
    expected_paths = set()
    prior_relative = None
    for row in rows:
        _require_exact_fields(row, FILE_BINDING_FIELDS, "legacy inventory row")
        relative = row["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or (prior_relative is not None and relative <= prior_relative)
        ):
            raise RuntimeError("legacy inventory paths must be sorted safe relative paths")
        artifact = _regular_file(evaluation_root / relative, "legacy inventory artifact")
        if not _is_within(artifact, evaluation_root) or artifact in expected_paths:
            raise RuntimeError("legacy inventory artifact escapes or is duplicated")
        expected_binding = {
            "path": str(artifact),
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        _authenticate_file_binding(expected_binding)
        if bound_by_path.get(artifact) != expected_binding:
            raise RuntimeError("legacy inventory absolute input binding mismatch")
        expected_paths.add(artifact)
        prior_relative = relative
    actual_paths = {
        _regular_file(item, "legacy evaluation artifact")
        for item in evaluation_root.rglob("*")
        if item.is_file()
    }
    if expected_paths != actual_paths or set(bound_by_path) != actual_paths:
        raise RuntimeError("legacy evaluation inventory is incomplete")


def _validate_freeze(payload, results, artifacts, identity, legacy_root, new_root):
    _require_exact_fields(results, FREEZE_RESULT_FIELDS, "legacy freeze results")
    if len(artifacts) != 1:
        raise RuntimeError("legacy freeze must bind exactly one inventory report")
    inventory_binding, inventory_path = artifacts[0]
    if payload.get("legacy_inventory_sha256") != inventory_binding["sha256"]:
        raise RuntimeError("legacy freeze inventory digest mismatch")
    inventory = _read_bound_json(inventory_binding, inventory_path)
    if (
        not isinstance(inventory, dict)
        or inventory.get("schema_version") != MIGRATION_LEGACY_INVENTORY_SCHEMA
        or inventory.get("status") != "PASS"
        or inventory.get("source_commit") != identity["legacy_commit"]
        or inventory.get("source_tree_sha256")
        != identity["legacy_source_tree_sha256"]
    ):
        raise RuntimeError("legacy freeze references the wrong inventory report")
    validate_evidence_report(
        "legacy_evaluation_inventory",
        inventory,
        expected_identity=identity,
        legacy_run_root=legacy_root,
        new_run_root=new_root,
    )
    lock_state = payload.get("lock_state")
    _require_exact_fields(
        lock_state,
        {"observed", "status", "owner_pid_identity"},
        "legacy freeze lock state",
    )
    if (
        results["inventory_authenticated"] is not True
        or results["process_state"] != "RUNNING"
        or results["pid_identity_matched"] is not True
        or results["lock_state_observed"] is not True
        or lock_state["observed"] is not True
        or lock_state["status"] != "HELD"
        or not isinstance(payload.get("pid_identity"), str)
        or not payload["pid_identity"].strip()
        or lock_state["owner_pid_identity"] != payload["pid_identity"]
        or type(payload.get("start_ticks")) is not int
        or payload["start_ticks"] <= 0
        or not isinstance(payload.get("cmd"), str)
        or not payload["cmd"].strip()
        or not isinstance(payload.get("cwd"), str)
        or not Path(payload["cwd"]).is_absolute()
    ):
        raise RuntimeError("legacy freeze process or lock observation is invalid")


def _require_observation_artifacts(artifacts, label):
    if not artifacts:
        raise RuntimeError(f"{label} evidence must bind an observation artifact")


def _writer_identity(expected_identity):
    if not isinstance(expected_identity, Mapping):
        raise RuntimeError("migration evidence identity must be an object")
    legacy_root = Path(str(expected_identity.get("legacy_run_root", ""))).resolve()
    new_root = Path(str(expected_identity.get("new_run_root", ""))).resolve()
    identity = _validated_identity(expected_identity, legacy_root, new_root)
    return identity, legacy_root, new_root


def _validated_identity(value, legacy_root, new_root):
    _require_exact_fields(value, EXPECTED_IDENTITY_FIELDS, "migration evidence identity")
    identity = dict(value)
    if (
        identity["legacy_run_root"] != str(legacy_root)
        or identity["new_run_root"] != str(new_root)
        or not Path(identity["legacy_run_root"]).is_absolute()
        or not Path(identity["new_run_root"]).is_absolute()
        or legacy_root == new_root
        or _is_within(legacy_root, new_root)
        or _is_within(new_root, legacy_root)
        or not _is_hex(identity["legacy_commit"], 40)
        or not _is_hex(identity["new_commit"], 40)
        or identity["legacy_commit"] == identity["new_commit"]
    ):
        raise RuntimeError("migration evidence run/source identity is malformed")
    for key in (
        "legacy_source_tree_sha256",
        "new_source_tree_sha256",
        "requirements_lock_sha256",
        "dataset_sha256",
        "config_sha256",
        "protocol_sha256",
        "new_run_nonce",
        "new_compute_plan_sha256",
    ):
        if not _is_hex(identity[key], 64):
            raise RuntimeError(f"migration evidence identity {key} is malformed")
    if (
        identity["legacy_source_tree_sha256"] == identity["new_source_tree_sha256"]
        or identity["new_run_id"]
        != f"csi-pairs-{str(identity['new_run_nonce'])[:16]}"
    ):
        raise RuntimeError("migration evidence optimized identity is malformed")
    return identity


def _authenticate_file_binding(
    value,
    *,
    legacy_root=None,
    new_root=None,
    require_external=False,
):
    _require_exact_fields(value, FILE_BINDING_FIELDS, "migration file binding")
    raw_path = value["path"]
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise RuntimeError("migration file binding path must be absolute")
    path = _regular_file(raw_path, "bound migration artifact")
    if str(path) != raw_path:
        raise RuntimeError("migration file binding path must be canonical")
    if require_external:
        _require_external(path, legacy_root, new_root, "bound migration artifact")
    if (
        type(value["bytes"]) is not int
        or value["bytes"] < 0
        or not _is_hex(value["sha256"], 64)
    ):
        raise RuntimeError("migration file binding size or SHA-256 is malformed")
    size, digest = _file_size_sha256(path)
    if size != value["bytes"] or digest != value["sha256"]:
        raise RuntimeError("bound migration artifact changed")
    return path


def _read_bound_json(binding, path):
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"bound JSON report cannot be read: {error}") from error
    if (
        len(encoded) != binding["bytes"]
        or hashlib.sha256(encoded).hexdigest() != binding["sha256"]
    ):
        raise RuntimeError("bound JSON report changed while authenticating")
    try:
        return parse_strict_json(encoded.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise RuntimeError(f"bound JSON report is invalid: {error}") from error


def _file_size_sha256(path):
    digest = hashlib.sha256()
    size = 0
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("bound migration input is not a regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("bound migration input changed while hashing")
    return size, digest.hexdigest()


def _regular_file(path, label):
    candidate = Path(path)
    if candidate.is_symlink():
        raise RuntimeError(f"{label} cannot be a symlink")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"{label} is missing or not a regular file")
    return resolved


def _regular_directory(path, label):
    candidate = Path(path)
    if candidate.is_symlink():
        raise RuntimeError(f"{label} cannot be a symlink")
    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"{label} is missing or not a regular directory")
    return resolved


def _external_target(path, legacy_root, new_root):
    requested = Path(path)
    if requested.is_symlink():
        raise RuntimeError("migration evidence path cannot be a symlink")
    target = requested.resolve()
    _require_external(target, legacy_root, new_root, "migration evidence")
    return target


def _binding_roots(legacy_run_root, new_run_root):
    if legacy_run_root is None or new_run_root is None:
        raise ValueError("both run roots are required for an external file binding")
    return Path(legacy_run_root).resolve(), Path(new_run_root).resolve()


def _require_external(path, legacy_root, new_root, label):
    if legacy_root is None or new_root is None:
        raise ValueError("both run roots are required for external-path validation")
    if _is_within(path, legacy_root) or _is_within(path, new_root):
        raise RuntimeError(f"{label} must be external to both run roots")


def _is_within(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    return path == root or root in path.parents


def _require_exact_fields(value, fields, label):
    if not isinstance(value, Mapping) or set(value) != set(fields):
        observed = set(value) if isinstance(value, Mapping) else set()
        raise RuntimeError(
            f"{label} fields must be exact; "
            f"missing={sorted(set(fields) - observed)}, "
            f"unexpected={sorted(observed - set(fields))}"
        )


def _positive_number(value, label):
    result = _finite_number(value, label)
    if result <= 0:
        raise RuntimeError(f"{label} must be positive")
    return result


def _finite_number(value, label):
    if type(value) not in {int, float} or not math.isfinite(value):
        raise RuntimeError(f"{label} must be a finite number")
    return float(value)


def _positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value, label):
    if type(value) is not int or value < 0:
        raise RuntimeError(f"{label} must be a nonnegative integer")
    return value


def _growth_exponent(first, fourth):
    first_value = _positive_number(first, "growth baseline")
    fourth_value = _positive_number(fourth, "growth endpoint")
    return math.log(fourth_value / first_value, 4.0)


def _canonical_sha256(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _is_hex(value, width):
    if not isinstance(value, str):
        return False
    pattern = _HEX40 if width == 40 else _HEX64
    return pattern.fullmatch(value) is not None


def _command(value):
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("migration evidence command must be nonempty")
    return value


def _parse_utc(value, label):
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError(f"{label} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise RuntimeError(f"{label} is invalid") from error
    if parsed.tzinfo != timezone.utc:
        raise RuntimeError(f"{label} must use UTC")
    return parsed


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
