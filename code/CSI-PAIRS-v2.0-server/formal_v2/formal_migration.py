from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .formal_config import ARMS
from .formal_evidence import (
    EVIDENCE_AUTH_KEYS,
    FACTORIAL_SCHEMA,
    QUALIFICATION_SCHEMA,
    RUNTIME_PROVENANCE_FIELDS,
)
from .formal_io import read_strict_json, sha256_file
from .formal_llm_judge import (
    APPROVAL_ATTESTATION,
    LLM_JUDGE_APPROVAL_SCHEMA,
    LLM_JUDGE_REQUIRED,
    parse_llm_judge,
)
from .formal_migration_evidence import (
    DIFF_REPORT_FIELDS as _DIFF_REPORT_FIELDS,
    FREEZE_REPORT_FIELDS as _FREEZE_REPORT_FIELDS,
    MIGRATION_DIFF_SCHEMA,
    MIGRATION_EVIDENCE_KEYS,
    MIGRATION_FREEZE_SCHEMA,
    MIGRATION_LEGACY_INVENTORY_SCHEMA,
    MIGRATION_TECHNICAL_EVIDENCE_SCHEMAS,
    TECHNICAL_REPORT_FIELDS as _TECHNICAL_REPORT_FIELDS,
    validate_evidence_results,
)


MIGRATION_REQUEST_SCHEMA = "csi-pairs-v6-legacy-upstream-migration-request-v2"
MIGRATION_APPROVAL_SCHEMA = "csi-pairs-v6-legacy-upstream-migration-approval-v2"
MIGRATION_ACCEPTED_SCHEMA = "csi-pairs-v6-legacy-upstream-migration-accepted-v2"
MIGRATION_COMPUTE_PLAN_SCHEMA = "csi-pairs-v6-migration-compute-plan-v1"
MIGRATION_AWAITING_LLM_JUDGE = "AWAITING_LLM_JUDGE"
MIGRATION_APPROVAL_ATTESTATION = (
    "This allowed LLM-as-judge reviewed the exact legacy qualification, factorial, "
    "checkpoint, approval, source, protocol, dataset, config, frozen legacy evaluation, "
    "base-to-new diff, performance, equivalence, progress, resume, corruption, staleness, "
    "locking, compute-plan, run-nonce, and destination bindings. It authorizes only "
    "immutable upstream reuse; it does not rewrite provenance, promote a scientific "
    "gate, or authorize legacy evaluation output."
)
MAX_APPROVAL_CLOCK_SKEW = timedelta(minutes=5)
MAX_APPROVAL_LIFETIME = timedelta(hours=24)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_RUNNING_SOURCE_ROOT = Path(__file__).resolve().parent

_STAGE_MANIFEST_SCHEMA = "csi-pairs-formal-stage-manifest-v2.1-v6"
_CHECKPOINT_INDEX_SCHEMA = "csi-pairs-formal-checkpoint-index-v2.1-v6"
_CHECKPOINT_SCHEMA = "csi-pairs-formal-checkpoint-v2.1-v6"
_RESUME_INDEX_SCHEMA = "csi-pairs-v6-factorial-resume-index-v1"
_RESUME_POINTER_SCHEMA = "csi-pairs-v6-factorial-resume-pointer-v1"
_RESUME_CHECKPOINT_SCHEMA = "csi-pairs-v6-factorial-resume-checkpoint-v1"
_FROZEN_PILOT_SCHEMA = "csi-pairs-v6-frozen-pilot-v1"
_FORMAL_FACTORIAL_STEPS = 20_000

_MIGRATION_COMPUTE_PLAN_FIELDS = {
    "schema_version",
    "status",
    "created_utc",
    "run_id",
    "run_nonce",
    "source_commit",
    "source_tree_sha256",
    "output_root",
    "dataset_sha256",
    "config_sha256",
    "gpu_mapping",
    "batch_size",
    "shard_size",
    "estimated_output_bytes",
    "minimum_free_disk_bytes",
    "free_disk_bytes_at_plan",
    "estimated_wall_time_seconds",
    "authorized_wall_time_seconds",
    "estimated_gpu_hours",
    "authorized_gpu_hours",
}
_GPU_MAPPING_FIELDS = {
    "logical_device",
    "physical_index",
    "uuid",
    "name",
    "total_memory_bytes",
}
_MIGRATION_EVIDENCE_KEYS = MIGRATION_EVIDENCE_KEYS

_SCIENTIFIC_CONTRACT = {
    "generated_under_new_source": [
        "evaluation",
        "risk",
        "path",
        "baselines",
        "controls",
        "claims",
    ],
    "legacy_artifacts_read_only": True,
    "legacy_scientific_status_preserved": True,
    "no_provenance_rewrite": True,
    "reuse_scope": ["qualification", "factorial"],
    "scientific_pass_not_inferred": True,
}

_REQUEST_FIELDS = {
    "schema_version",
    "status",
    "migration_id",
    "migration_nonce",
    "new_run_id",
    "new_run_nonce",
    "created_utc",
    "legacy_run_root",
    "new_run_root",
    "dataset_path",
    "dataset_sha256",
    "config_sha256",
    "expected_seeds",
    "expected_arms",
    "protocol_path",
    "protocol_sha256",
    "requirements_lock_sha256",
    "new_compute_plan",
    "migration_evidence",
    "migration_evidence_sha256",
    "legacy_source",
    "new_source",
    "legacy_approval",
    "qualification",
    "factorial",
    "checkpoint_inventory",
    "checkpoint_inventory_sha256",
    "legacy_scientific_state",
    "scientific_contract",
}
_SOURCE_FIELDS = {
    "source_root",
    "git_root",
    "git_commit",
    "source_tree_sha256",
    "requirements_lock_path",
    "requirements_lock_sha256",
}
_LEGACY_APPROVAL_FIELDS = {
    "run_id",
    "run_nonce",
    "preflight_path",
    "preflight_sha256",
    "prepared_path",
    "prepared_sha256",
    "request_path",
    "request_sha256",
    "accepted_path",
    "accepted_sha256",
    "external_approval_path",
    "external_approval_sha256",
    "judge",
}
_QUALIFICATION_BINDING_FIELDS = {
    "gate_path",
    "gate_sha256",
    "manifest_path",
    "manifest_sha256",
    "teacher_checkpoint_path",
    "teacher_checkpoint_sha256",
    "evidence_sha256",
}
_FACTORIAL_BINDING_FIELDS = {
    "root",
    "gate_path",
    "gate_sha256",
    "manifest_path",
    "manifest_sha256",
    "checkpoint_index_path",
    "checkpoint_index_sha256",
    "resume_index_path",
    "resume_index_sha256",
    "normalization_path",
    "normalization_sha256",
    "frozen_pilot_path",
    "frozen_pilot_sha256",
    "factorial_statistics_path",
    "factorial_statistics_sha256",
    "training_summary_path",
    "training_summary_sha256",
    "localization_per_bank_path",
    "localization_per_bank_sha256",
    "localization_per_sample_path",
    "localization_per_sample_sha256",
    "checkpoint_context_sha256",
    "training_steps",
    "evidence_sha256",
}
_CHECKPOINT_BINDING_FIELDS = {
    "seed",
    "arm",
    "path",
    "bytes",
    "sha256",
    "teacher_checkpoint_sha256",
    "source_tree_sha256",
}
_SCIENTIFIC_STATE_FIELDS = {
    "qualification_status",
    "qualification_passed",
    "factorial_status",
    "factorial_passed",
    "factorial_gate_vector",
}
_FILE_BINDING_FIELDS = {
    "path",
    "bytes",
    "sha256",
    "schema_version",
    "status",
}
_COMPUTE_PLAN_BINDING_FIELDS = {
    *_FILE_BINDING_FIELDS,
    "run_id",
    "run_nonce",
    "gpu_mapping",
    "batch_size",
    "shard_size",
    "minimum_free_disk_bytes",
    "authorized_wall_time_seconds",
    "authorized_gpu_hours",
}
_MIGRATION_APPROVAL_FIELDS = {
    "schema_version",
    "decision",
    "migration_id",
    "migration_nonce",
    "new_run_id",
    "new_run_nonce",
    "request_sha256",
    "legacy_source_tree_sha256",
    "new_source_tree_sha256",
    "checkpoint_inventory_sha256",
    "new_compute_plan_sha256",
    "migration_evidence_sha256",
    "judge",
    "approved_utc",
    "expires_utc",
    "attestation",
}
_ACCEPTED_FIELDS = {
    "schema_version",
    "status",
    "migration_id",
    "migration_nonce",
    "new_run_id",
    "new_run_nonce",
    "request_path",
    "request_sha256",
    "approval_manifest_path",
    "approval_manifest_sha256",
    "legacy_run_root",
    "new_run_root",
    "legacy_source_tree_sha256",
    "new_source_tree_sha256",
    "checkpoint_inventory_sha256",
    "new_compute_plan_sha256",
    "migration_evidence_sha256",
    "accepted_utc",
}

_LEGACY_APPROVAL_REQUEST_FIELDS = {
    "schema_version",
    "run_id",
    "run_nonce",
    "prepared_utc",
    "prepared_root",
    "config_sha256",
    "dataset_sha256",
    "fixture",
    "source_tree_sha256",
    "requirements_lock_sha256",
    "runtime_provenance_sha256",
    "runtime_provenance",
    "external_runtime_provenance",
    "gpu_inventory",
    "required_gpu_count",
    "execution_devices",
    "required_environment_values",
    "compute_plan",
    "input_bindings",
    "preflight_path",
    "preflight_sha256",
    "gate_bindings",
    "teacher_checkpoint",
    "teacher_checkpoint_sha256",
    "decision_required",
    "scientific_use",
    "review_scope",
}
_LEGACY_PREPARED_FIELDS = {
    "schema_version",
    "run_id",
    "run_nonce",
    "prepared_root",
    "request_path",
    "request_sha256",
    "status",
}
_LEGACY_ACCEPTED_FIELDS = {
    "schema_version",
    "status",
    "run_id",
    "run_nonce",
    "request_sha256",
    "approval_manifest_path",
    "approval_manifest_sha256",
    "compute_plan_sha256",
    "accepted_utc",
}
_LEGACY_EXTERNAL_APPROVAL_FIELDS = {
    "schema_version",
    "decision",
    "run_id",
    "run_nonce",
    "request_sha256",
    "compute_plan_sha256",
    "approved_gate_sha256s",
    "judge",
    "approved_utc",
    "expires_utc",
    "attestation",
}

_FACTORIAL_REQUIRED_FILES = (
    "checkpoint_index.json",
    "factorial_statistics.json",
    "frozen_pilot.json",
    "gate.json",
    "localization_per_bank.csv",
    "localization_per_sample.csv",
    "localization_summary.csv",
    "normalization.json",
    "resume_index.json",
    "training_summary.csv",
)


@dataclass(frozen=True)
class AuthenticatedLegacyUpstream:
    legacy_run_root: Path
    new_run_root: Path
    qualification_gate: Path
    factorial_root: Path
    factorial_gate: Path
    checkpoint_index: Path
    checkpoint_rows: tuple[dict[str, object], ...]
    checkpoint_inventory_sha256: str
    legacy_source_tree_sha256: str
    new_source_git_commit: str
    new_source_tree_sha256: str
    new_run_id: str
    new_run_nonce: str
    new_compute_plan: dict[str, object]
    new_compute_plan_sha256: str
    migration_evidence_sha256: str
    request_sha256: str
    accepted_sha256: str
    qualification_evidence: dict[str, object]
    factorial_evidence: dict[str, object]
    legacy_scientific_state: dict[str, object]


def write_migration_request(
    *,
    legacy_run_root: str | Path,
    legacy_source_root: str | Path,
    new_run_root: str | Path,
    new_source_root: str | Path,
    protocol_path: str | Path,
    dataset_path: str | Path,
    config_sha256: str,
    expected_seeds: Sequence[int],
    migration_nonce: str,
    new_run_nonce: str,
    new_compute_plan_path: str | Path,
    migration_evidence_paths: Mapping[str, str | Path],
    created_utc: str | None = None,
) -> dict[str, object]:
    """Write one non-replayable request for immutable legacy upstream reuse."""
    legacy_root = _regular_directory(legacy_run_root, "legacy run root")
    destination = _regular_directory(new_run_root, "new run root")
    _require_disjoint_roots(legacy_root, destination)
    source_root = _regular_directory(new_source_root, "new source root")
    if source_root != _RUNNING_SOURCE_ROOT:
        raise RuntimeError("new source root is not the source tree executing migration")
    nonce = _require_hex(migration_nonce, 64, "migration nonce")
    run_nonce = _require_hex(new_run_nonce, 64, "new run nonce")
    if nonce == run_nonce:
        raise ValueError("migration nonce and new run nonce must be distinct")
    seeds = _validated_seeds(expected_seeds)
    config_digest = _require_hex(config_sha256, 64, "config sha256")
    timestamp = created_utc or _utc_now()
    _parse_utc(timestamp, "created_utc")

    request = _build_request(
        legacy_run_root=legacy_root,
        legacy_source_root=_regular_directory(
            legacy_source_root, "legacy source root"
        ),
        new_run_root=destination,
        new_source_root=source_root,
        protocol_path=_regular_file(protocol_path, "frozen protocol"),
        dataset_path=_regular_file(dataset_path, "formal dataset"),
        config_sha256=config_digest,
        expected_seeds=seeds,
        migration_nonce=nonce,
        new_run_nonce=run_nonce,
        new_compute_plan_path=_regular_file(
            new_compute_plan_path, "new migration compute plan"
        ),
        migration_evidence_paths=migration_evidence_paths,
        created_utc=timestamp,
    )
    request_path = destination / "migration" / "request.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    if request_path.parent.is_symlink():
        raise RuntimeError("migration request directory cannot be a symlink")
    _write_json_exclusive_atomic(request_path, request)
    return {
        **request,
        "request_path": str(request_path),
        "request_sha256": sha256_file(request_path),
    }


def accept_migration_request(
    request_path: str | Path,
    approval_manifest_path: str | Path,
    *,
    new_run_root: str | Path,
    new_source_root: str | Path,
    protocol_path: str | Path,
    dataset_path: str | Path,
    config_sha256: str,
    expected_seeds: Sequence[int],
    accepted_utc: str | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Accept a request only after a separately stored allowed-LLM approval."""
    request_file, request = _validate_current_request(
        request_path,
        new_run_root=new_run_root,
        new_source_root=new_source_root,
        protocol_path=protocol_path,
        dataset_path=dataset_path,
        config_sha256=config_sha256,
        expected_seeds=expected_seeds,
    )
    approval_path = _regular_file(
        approval_manifest_path, "external migration approval manifest"
    )
    legacy_root = Path(str(request["legacy_run_root"])).resolve()
    destination = Path(str(request["new_run_root"])).resolve()
    if _is_within(approval_path, legacy_root) or _is_within(approval_path, destination):
        raise RuntimeError("migration approval manifest must be external to both run roots")
    approval = read_strict_json(approval_path)
    current = now or datetime.now(timezone.utc)
    _validate_migration_approval(
        approval, request, sha256_file(request_file), now=current
    )
    timestamp = accepted_utc or _utc_now()
    accepted_time = _parse_utc(timestamp, "accepted_utc")
    if accepted_time > current + MAX_APPROVAL_CLOCK_SKEW:
        raise RuntimeError("migration accepted timestamp is unacceptably far in the future")
    approved_time = _parse_utc(str(approval["approved_utc"]), "approved_utc")
    expires_time = _parse_utc(str(approval["expires_utc"]), "expires_utc")
    if accepted_time < approved_time or accepted_time > expires_time:
        raise RuntimeError("migration acceptance is outside the approval validity window")

    accepted_path = destination / "migration" / "accepted.json"
    accepted = _accepted_record(
        request_file, request, approval_path, timestamp
    )
    _write_json_exclusive_atomic(accepted_path, accepted)
    return {
        **accepted,
        "accepted_path": str(accepted_path),
        "accepted_sha256": sha256_file(accepted_path),
    }


def authenticate_migration_receipt(
    accepted_path: str | Path,
    *,
    new_run_root: str | Path,
    new_source_root: str | Path,
    protocol_path: str | Path,
    dataset_path: str | Path,
    config_sha256: str,
    expected_seeds: Sequence[int],
) -> AuthenticatedLegacyUpstream:
    """Reauthenticate every mutable input; accepted approval expiry is not replayed."""
    destination = _regular_directory(new_run_root, "new run root")
    expected_accepted = destination / "migration" / "accepted.json"
    accepted_file = _regular_file(accepted_path, "migration accepted receipt")
    if accepted_file != expected_accepted:
        raise RuntimeError("migration accepted receipt is not at its canonical run path")
    accepted = read_strict_json(accepted_file)
    _require_exact_fields(accepted, _ACCEPTED_FIELDS, "migration accepted receipt")
    if (
        accepted["schema_version"] != MIGRATION_ACCEPTED_SCHEMA
        or accepted["status"] != "ACCEPTED"
        or accepted["new_run_root"] != str(destination)
    ):
        raise RuntimeError("migration accepted receipt identity mismatch")

    request_file, request = _validate_current_request(
        accepted["request_path"],
        new_run_root=destination,
        new_source_root=new_source_root,
        protocol_path=protocol_path,
        dataset_path=dataset_path,
        config_sha256=config_sha256,
        expected_seeds=expected_seeds,
    )
    if accepted["request_sha256"] != sha256_file(request_file):
        raise RuntimeError("migration accepted receipt request hash mismatch")
    approval_path = _regular_file(
        accepted["approval_manifest_path"], "external migration approval manifest"
    )
    if accepted["approval_manifest_sha256"] != sha256_file(approval_path):
        raise RuntimeError("migration accepted receipt approval hash mismatch")
    legacy_root = Path(str(request["legacy_run_root"])).resolve()
    if _is_within(approval_path, legacy_root) or _is_within(approval_path, destination):
        raise RuntimeError("migration approval manifest is no longer external")
    approval = read_strict_json(approval_path)
    accepted_time = _parse_utc(str(accepted["accepted_utc"]), "accepted_utc")
    _validate_migration_approval(
        approval,
        request,
        sha256_file(request_file),
        now=accepted_time,
    )
    expected = _accepted_record(
        request_file,
        request,
        approval_path,
        str(accepted["accepted_utc"]),
    )
    if accepted != expected:
        raise RuntimeError("migration accepted receipt differs from authenticated bindings")

    checkpoints = tuple(dict(row) for row in request["checkpoint_inventory"])
    qualification_manifest = read_strict_json(
        request["qualification"]["manifest_path"]
    )
    factorial_manifest = read_strict_json(request["factorial"]["manifest_path"])
    qualification_evidence = _authenticated_manifest_evidence(
        qualification_manifest, "qualification"
    )
    factorial_evidence = _authenticated_manifest_evidence(
        factorial_manifest, "factorial"
    )
    return AuthenticatedLegacyUpstream(
        legacy_run_root=legacy_root,
        new_run_root=destination,
        qualification_gate=Path(str(request["qualification"]["gate_path"])),
        factorial_root=Path(str(request["factorial"]["root"])),
        factorial_gate=Path(str(request["factorial"]["gate_path"])),
        checkpoint_index=Path(
            str(request["factorial"]["checkpoint_index_path"])
        ),
        checkpoint_rows=checkpoints,
        checkpoint_inventory_sha256=str(request["checkpoint_inventory_sha256"]),
        legacy_source_tree_sha256=str(
            request["legacy_source"]["source_tree_sha256"]
        ),
        new_source_git_commit=str(request["new_source"]["git_commit"]),
        new_source_tree_sha256=str(request["new_source"]["source_tree_sha256"]),
        new_run_id=str(request["new_run_id"]),
        new_run_nonce=str(request["new_run_nonce"]),
        new_compute_plan=dict(request["new_compute_plan"]),
        new_compute_plan_sha256=str(request["new_compute_plan"]["sha256"]),
        migration_evidence_sha256=str(request["migration_evidence_sha256"]),
        request_sha256=sha256_file(request_file),
        accepted_sha256=sha256_file(accepted_file),
        qualification_evidence=qualification_evidence,
        factorial_evidence=factorial_evidence,
        legacy_scientific_state=dict(request["legacy_scientific_state"]),
    )


def _authenticated_manifest_evidence(
    manifest: object, stage: str
) -> dict[str, object]:
    if not isinstance(manifest, dict):
        raise RuntimeError(f"authenticated legacy {stage} manifest is malformed")
    keys = (*EVIDENCE_AUTH_KEYS, "scientific_use")
    if any(key not in manifest for key in keys):
        raise RuntimeError(
            f"authenticated legacy {stage} manifest lacks evidence identity"
        )
    return {key: manifest[key] for key in keys}


def _build_request(
    *,
    legacy_run_root: Path,
    legacy_source_root: Path,
    new_run_root: Path,
    new_source_root: Path,
    protocol_path: Path,
    dataset_path: Path,
    config_sha256: str,
    expected_seeds: tuple[int, ...],
    migration_nonce: str,
    new_run_nonce: str,
    new_compute_plan_path: Path,
    migration_evidence_paths: Mapping[str, str | Path],
    created_utc: str,
) -> dict[str, object]:
    legacy_source = _source_identity(legacy_source_root)
    new_source = _source_identity(new_source_root)
    if legacy_source["git_commit"] == new_source["git_commit"]:
        raise RuntimeError("migration requires a distinct optimized git commit")
    if legacy_source["source_tree_sha256"] == new_source["source_tree_sha256"]:
        raise RuntimeError("migration requires a distinct optimized source tree")
    if (
        legacy_source["requirements_lock_sha256"]
        != new_source["requirements_lock_sha256"]
    ):
        raise RuntimeError("migration cannot change the locked main runtime")

    dataset_digest = sha256_file(dataset_path)
    protocol_digest = sha256_file(protocol_path)
    new_run_id = f"csi-pairs-{new_run_nonce[:16]}"
    compute_plan = _authenticate_new_compute_plan(
        new_compute_plan_path,
        new_run_root=new_run_root,
        new_run_id=new_run_id,
        new_run_nonce=new_run_nonce,
        new_source=new_source,
        dataset_sha256=dataset_digest,
        config_sha256=config_sha256,
        legacy_run_root=legacy_run_root,
    )
    migration_evidence = _authenticate_migration_evidence(
        migration_evidence_paths,
        legacy_run_root=legacy_run_root,
        new_run_root=new_run_root,
        legacy_source=legacy_source,
        new_source=new_source,
        dataset_sha256=dataset_digest,
        config_sha256=config_sha256,
        protocol_sha256=protocol_digest,
        new_run_id=new_run_id,
        new_run_nonce=new_run_nonce,
        new_compute_plan_sha256=str(compute_plan["sha256"]),
    )
    migration_evidence_sha256 = _canonical_sha256(migration_evidence)
    inspected = _inspect_legacy_upstream(
        legacy_run_root,
        legacy_source,
        dataset_sha256=dataset_digest,
        config_sha256=config_sha256,
        expected_seeds=expected_seeds,
    )
    checkpoints = inspected["checkpoint_inventory"]
    checkpoint_digest = _canonical_sha256(checkpoints)
    return {
        "schema_version": MIGRATION_REQUEST_SCHEMA,
        "status": MIGRATION_AWAITING_LLM_JUDGE,
        "migration_id": f"csi-pairs-migration-{migration_nonce[:16]}",
        "migration_nonce": migration_nonce,
        "new_run_id": new_run_id,
        "new_run_nonce": new_run_nonce,
        "created_utc": created_utc,
        "legacy_run_root": str(legacy_run_root),
        "new_run_root": str(new_run_root),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_digest,
        "config_sha256": config_sha256,
        "expected_seeds": list(expected_seeds),
        "expected_arms": list(ARMS),
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_digest,
        "requirements_lock_sha256": legacy_source[
            "requirements_lock_sha256"
        ],
        "new_compute_plan": compute_plan,
        "migration_evidence": migration_evidence,
        "migration_evidence_sha256": migration_evidence_sha256,
        "legacy_source": legacy_source,
        "new_source": new_source,
        "legacy_approval": inspected["legacy_approval"],
        "qualification": inspected["qualification"],
        "factorial": inspected["factorial"],
        "checkpoint_inventory": checkpoints,
        "checkpoint_inventory_sha256": checkpoint_digest,
        "legacy_scientific_state": inspected["legacy_scientific_state"],
        "scientific_contract": dict(_SCIENTIFIC_CONTRACT),
    }


def _authenticate_new_compute_plan(
    path: Path,
    *,
    new_run_root: Path,
    new_run_id: str,
    new_run_nonce: str,
    new_source: dict[str, object],
    dataset_sha256: str,
    config_sha256: str,
    legacy_run_root: Path,
) -> dict[str, object]:
    plan_path = _regular_file(path, "new migration compute plan")
    if _is_within(plan_path, legacy_run_root) or _is_within(plan_path, new_run_root):
        raise RuntimeError("migration compute plan must be external to both run roots")
    plan = read_strict_json(plan_path)
    _require_exact_fields(plan, _MIGRATION_COMPUTE_PLAN_FIELDS, "migration compute plan")
    if (
        plan["schema_version"] != MIGRATION_COMPUTE_PLAN_SCHEMA
        or plan["status"] != "FROZEN"
        or plan["run_id"] != new_run_id
        or plan["run_nonce"] != new_run_nonce
        or plan["source_commit"] != new_source["git_commit"]
        or plan["source_tree_sha256"] != new_source["source_tree_sha256"]
        or plan["output_root"] != str(new_run_root)
        or plan["dataset_sha256"] != dataset_sha256
        or plan["config_sha256"] != config_sha256
    ):
        raise RuntimeError("migration compute plan identity mismatch")
    _parse_utc(str(plan["created_utc"]), "migration compute-plan created_utc")
    integer_fields = (
        "batch_size",
        "shard_size",
        "estimated_output_bytes",
        "minimum_free_disk_bytes",
        "free_disk_bytes_at_plan",
        "estimated_wall_time_seconds",
        "authorized_wall_time_seconds",
    )
    if any(type(plan[key]) is not int or plan[key] <= 0 for key in integer_fields):
        raise RuntimeError("migration compute-plan integer budgets must be positive")
    if (
        plan["minimum_free_disk_bytes"] < plan["estimated_output_bytes"]
        or plan["free_disk_bytes_at_plan"] < plan["minimum_free_disk_bytes"]
        or plan["authorized_wall_time_seconds"] < plan["estimated_wall_time_seconds"]
    ):
        raise RuntimeError("migration compute-plan disk or wall-time budget is insufficient")
    for key in ("estimated_gpu_hours", "authorized_gpu_hours"):
        value = plan[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise RuntimeError("migration compute-plan GPU budgets must be positive")
    if plan["authorized_gpu_hours"] < plan["estimated_gpu_hours"]:
        raise RuntimeError("migration compute-plan GPU authorization is insufficient")
    gpu_mapping = plan["gpu_mapping"]
    if not isinstance(gpu_mapping, list) or len(gpu_mapping) != 2:
        raise RuntimeError("migration compute plan must bind exactly two GPUs")
    logical_devices: set[str] = set()
    physical_indices: set[int] = set()
    uuids: set[str] = set()
    for row in gpu_mapping:
        _require_exact_fields(row, _GPU_MAPPING_FIELDS, "migration GPU mapping")
        if (
            row["logical_device"] not in {"cuda:0", "cuda:1"}
            or type(row["physical_index"]) is not int
            or row["physical_index"] < 0
            or not isinstance(row["uuid"], str)
            or not row["uuid"].strip()
            or not isinstance(row["name"], str)
            or not row["name"].strip()
            or type(row["total_memory_bytes"]) is not int
            or row["total_memory_bytes"] <= 0
        ):
            raise RuntimeError("migration GPU mapping is malformed")
        logical_devices.add(str(row["logical_device"]))
        physical_indices.add(int(row["physical_index"]))
        uuids.add(str(row["uuid"]))
    if (
        logical_devices != {"cuda:0", "cuda:1"}
        or len(physical_indices) != 2
        or len(uuids) != 2
    ):
        raise RuntimeError("migration GPU mapping is duplicate or incomplete")
    binding = {
        "path": str(plan_path),
        "bytes": plan_path.stat().st_size,
        "sha256": sha256_file(plan_path),
        "schema_version": plan["schema_version"],
        "status": plan["status"],
        "run_id": plan["run_id"],
        "run_nonce": plan["run_nonce"],
        "gpu_mapping": gpu_mapping,
        "batch_size": plan["batch_size"],
        "shard_size": plan["shard_size"],
        "minimum_free_disk_bytes": plan["minimum_free_disk_bytes"],
        "authorized_wall_time_seconds": plan["authorized_wall_time_seconds"],
        "authorized_gpu_hours": plan["authorized_gpu_hours"],
    }
    _require_exact_fields(
        binding, _COMPUTE_PLAN_BINDING_FIELDS, "migration compute-plan binding"
    )
    return binding


def _authenticate_migration_evidence(
    paths: Mapping[str, str | Path],
    *,
    legacy_run_root: Path,
    new_run_root: Path,
    legacy_source: dict[str, object],
    new_source: dict[str, object],
    dataset_sha256: str,
    config_sha256: str,
    protocol_sha256: str,
    new_run_id: str,
    new_run_nonce: str,
    new_compute_plan_sha256: str,
) -> dict[str, dict[str, object]]:
    if not isinstance(paths, Mapping) or set(paths) != set(_MIGRATION_EVIDENCE_KEYS):
        observed = set(paths) if isinstance(paths, Mapping) else set()
        raise RuntimeError(
            "migration technical evidence is incomplete; "
            f"missing={sorted(set(_MIGRATION_EVIDENCE_KEYS) - observed)}, "
            f"unexpected={sorted(observed - set(_MIGRATION_EVIDENCE_KEYS))}"
        )
    expected_identity = {
        "legacy_run_root": str(legacy_run_root),
        "new_run_root": str(new_run_root),
        "legacy_commit": legacy_source["git_commit"],
        "legacy_source_tree_sha256": legacy_source["source_tree_sha256"],
        "new_commit": new_source["git_commit"],
        "new_source_tree_sha256": new_source["source_tree_sha256"],
        "requirements_lock_sha256": legacy_source["requirements_lock_sha256"],
        "dataset_sha256": dataset_sha256,
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
        "new_run_id": new_run_id,
        "new_run_nonce": new_run_nonce,
        "new_compute_plan_sha256": new_compute_plan_sha256,
    }
    payloads: dict[str, dict[str, object]] = {}
    bindings: dict[str, dict[str, object]] = {}
    observed_paths: set[Path] = set()
    for name in _MIGRATION_EVIDENCE_KEYS:
        evidence_path = _regular_file(paths[name], f"migration {name} evidence")
        if (
            _is_within(evidence_path, legacy_run_root)
            or _is_within(evidence_path, new_run_root)
            or evidence_path in observed_paths
        ):
            raise RuntimeError(
                "migration evidence must be unique and external to both run roots"
            )
        observed_paths.add(evidence_path)
        payload = read_strict_json(evidence_path)
        payloads[name] = payload
        if name in MIGRATION_TECHNICAL_EVIDENCE_SCHEMAS:
            fields = _TECHNICAL_REPORT_FIELDS
            schema = MIGRATION_TECHNICAL_EVIDENCE_SCHEMAS[name]
            status = "PASS"
            source = new_source
        elif name == "base_to_new_diff":
            fields = _DIFF_REPORT_FIELDS
            schema = MIGRATION_DIFF_SCHEMA
            status = "PASS"
            source = new_source
        elif name == "legacy_evaluation_inventory":
            fields = _TECHNICAL_REPORT_FIELDS
            schema = MIGRATION_LEGACY_INVENTORY_SCHEMA
            status = "PASS"
            source = legacy_source
        else:
            fields = _FREEZE_REPORT_FIELDS
            schema = MIGRATION_FREEZE_SCHEMA
            status = "FROZEN"
            source = legacy_source
        _require_exact_fields(payload, fields, f"migration {name} report")
        if (
            payload["schema_version"] != schema
            or payload["status"] != status
            or payload["source_commit"] != source["git_commit"]
            or payload["source_tree_sha256"] != source["source_tree_sha256"]
            or not isinstance(payload["command"], str)
            or not payload["command"].strip()
            or not isinstance(payload["inputs"], dict)
            or payload["inputs"].get("identity") != expected_identity
            or not isinstance(payload["results"], dict)
            or not payload["results"]
        ):
            raise RuntimeError(f"migration {name} report identity mismatch")
        _parse_utc(str(payload["created_utc"]), f"migration {name} created_utc")
        validate_evidence_results(
            name,
            payload,
            expected_identity=expected_identity,
            legacy_run_root=legacy_run_root,
            new_run_root=new_run_root,
        )
        binding = {
            "path": str(evidence_path),
            "bytes": evidence_path.stat().st_size,
            "sha256": sha256_file(evidence_path),
            "schema_version": schema,
            "status": status,
        }
        _require_exact_fields(
            binding, _FILE_BINDING_FIELDS, f"migration {name} evidence binding"
        )
        bindings[name] = binding
    diff = payloads["base_to_new_diff"]
    if (
        diff["source_commit"] != diff["new_commit"]
        or diff["source_tree_sha256"] != diff["new_source_tree_sha256"]
        or diff["base_commit"] != legacy_source["git_commit"]
        or diff["base_source_tree_sha256"] != legacy_source["source_tree_sha256"]
        or diff["new_commit"] != new_source["git_commit"]
        or diff["new_source_tree_sha256"] != new_source["source_tree_sha256"]
        or not _is_hex(diff["diff_sha256"], 64)
    ):
        raise RuntimeError("migration base-to-new diff identity mismatch")
    inventory = payloads["legacy_evaluation_inventory"]
    inventory_results = inventory["results"]
    if (
        inventory_results.get("legacy_evaluation_root")
        != str((legacy_run_root / "evaluation").resolve())
        or not isinstance(inventory_results.get("files"), list)
        or inventory_results.get("files_sha256")
        != _canonical_sha256(inventory_results.get("files"))
    ):
        raise RuntimeError("legacy evaluation inventory identity mismatch")
    _authenticate_frozen_inventory(
        legacy_run_root / "evaluation", inventory_results["files"]
    )
    freeze = payloads["legacy_evaluation_freeze"]
    if (
        freeze["legacy_inventory_sha256"]
        != bindings["legacy_evaluation_inventory"]["sha256"]
        or not isinstance(freeze["pid_identity"], str)
        or not freeze["pid_identity"].strip()
        or type(freeze["start_ticks"]) is not int
        or freeze["start_ticks"] <= 0
        or not isinstance(freeze["cmd"], str)
        or not freeze["cmd"].strip()
        or not isinstance(freeze["cwd"], str)
        or not Path(freeze["cwd"]).is_absolute()
        or not isinstance(freeze["lock_state"], dict)
        or not freeze["lock_state"]
    ):
        raise RuntimeError("legacy evaluation freeze receipt identity mismatch")
    return bindings


def _authenticate_frozen_inventory(root: Path, rows: object) -> None:
    evaluation_root = _regular_directory(root, "legacy evaluation root")
    if not isinstance(rows, list):
        raise RuntimeError("legacy evaluation inventory files are malformed")
    expected: set[str] = set()
    for row in rows:
        _require_exact_fields(
            row, {"path", "bytes", "sha256"}, "legacy evaluation inventory row"
        )
        relative = row["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or relative in expected
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or type(row["bytes"]) is not int
            or row["bytes"] < 0
            or not _is_hex(row["sha256"], 64)
        ):
            raise RuntimeError("legacy evaluation inventory row identity is malformed")
        artifact = _regular_file(
            evaluation_root / relative, "legacy evaluation frozen artifact"
        )
        if (
            not _is_within(artifact, evaluation_root)
            or artifact.stat().st_size != row["bytes"]
            or sha256_file(artifact) != row["sha256"]
        ):
            raise RuntimeError("legacy evaluation frozen artifact changed")
        expected.add(relative)
    actual = {
        path.relative_to(evaluation_root).as_posix()
        for path in evaluation_root.rglob("*")
        if path.is_file()
    }
    if expected != actual:
        raise RuntimeError("legacy evaluation inventory is incomplete")


def _validate_current_request(
    request_path: str | Path,
    *,
    new_run_root: str | Path,
    new_source_root: str | Path,
    protocol_path: str | Path,
    dataset_path: str | Path,
    config_sha256: str,
    expected_seeds: Sequence[int],
) -> tuple[Path, dict[str, object]]:
    destination = _regular_directory(new_run_root, "new run root")
    request_file = _regular_file(request_path, "migration request")
    if request_file != destination / "migration" / "request.json":
        raise RuntimeError("migration request is not at its canonical run path")
    request = read_strict_json(request_file)
    _require_exact_fields(request, _REQUEST_FIELDS, "migration request")
    if (
        request["schema_version"] != MIGRATION_REQUEST_SCHEMA
        or request["status"] != MIGRATION_AWAITING_LLM_JUDGE
        or request["new_run_root"] != str(destination)
    ):
        raise RuntimeError("migration request identity mismatch")
    nonce = _require_hex(str(request["migration_nonce"]), 64, "migration nonce")
    if request["migration_id"] != f"csi-pairs-migration-{nonce[:16]}":
        raise RuntimeError("migration request ID does not match its nonce")
    run_nonce = _require_hex(str(request["new_run_nonce"]), 64, "new run nonce")
    if nonce == run_nonce or request["new_run_id"] != f"csi-pairs-{run_nonce[:16]}":
        raise RuntimeError("migration request new-run identity is invalid")
    _require_exact_fields(
        request["new_compute_plan"],
        _COMPUTE_PLAN_BINDING_FIELDS,
        "migration compute-plan binding",
    )
    evidence_bindings = request["migration_evidence"]
    if not isinstance(evidence_bindings, dict) or set(evidence_bindings) != set(
        _MIGRATION_EVIDENCE_KEYS
    ):
        raise RuntimeError("migration request technical-evidence keys are invalid")
    for name, binding in evidence_bindings.items():
        _require_exact_fields(
            binding, _FILE_BINDING_FIELDS, f"migration {name} evidence binding"
        )
    created = str(request["created_utc"])
    _parse_utc(created, "created_utc")

    source_root = _regular_directory(new_source_root, "new source root")
    if source_root != _RUNNING_SOURCE_ROOT:
        raise RuntimeError("new source root is not the source tree executing migration")
    protocol = _regular_file(protocol_path, "frozen protocol")
    dataset = _regular_file(dataset_path, "formal dataset")
    config_digest = _require_hex(config_sha256, 64, "config sha256")
    seeds = _validated_seeds(expected_seeds)
    legacy_root = _regular_directory(request["legacy_run_root"], "legacy run root")
    legacy_source_root = _regular_directory(
        request["legacy_source"]["source_root"], "legacy source root"
    )
    _require_disjoint_roots(legacy_root, destination)
    expected = _build_request(
        legacy_run_root=legacy_root,
        legacy_source_root=legacy_source_root,
        new_run_root=destination,
        new_source_root=source_root,
        protocol_path=protocol,
        dataset_path=dataset,
        config_sha256=config_digest,
        expected_seeds=seeds,
        migration_nonce=nonce,
        new_run_nonce=run_nonce,
        new_compute_plan_path=_regular_file(
            request["new_compute_plan"]["path"], "new migration compute plan"
        ),
        migration_evidence_paths={
            name: binding["path"] for name, binding in evidence_bindings.items()
        },
        created_utc=created,
    )
    if request != expected:
        raise RuntimeError("migration request differs from current authenticated inputs")
    return request_file, request


def _inspect_legacy_upstream(
    root: Path,
    source: dict[str, object],
    *,
    dataset_sha256: str,
    config_sha256: str,
    expected_seeds: tuple[int, ...],
) -> dict[str, object]:
    qualification_root = root / "qualification"
    factorial_root = root / "factorial"
    qualification_gate_path = _regular_file(
        qualification_root / "gate.json", "legacy qualification gate"
    )
    qualification_manifest_path = _regular_file(
        qualification_root / "manifest.json", "legacy qualification manifest"
    )
    qualification_gate = read_strict_json(qualification_gate_path)
    qualification_evidence = _authenticate_stage(
        qualification_root,
        qualification_gate,
        qualification_manifest_path,
        expected_schema=QUALIFICATION_SCHEMA,
        expected_scientific_use="FORMAL_EXPERIMENT_ALLOWED",
        source=source,
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
    )
    if (
        qualification_gate.get("passed") is not True
        or qualification_gate.get("status") != "QUALIFICATION_PASS"
        or qualification_gate.get("upstream_gates", {}).get("G1") != "PASS"
        or qualification_gate.get("upstream_gates", {}).get("G2") != "PASS"
    ):
        raise RuntimeError("legacy qualification did not preserve G1/G2 PASS")
    teacher_path = _regular_file(
        qualification_gate.get("teacher_checkpoint", ""),
        "legacy teacher checkpoint",
    )
    if not _is_within(teacher_path, qualification_root):
        raise RuntimeError("legacy teacher checkpoint escapes qualification")
    teacher_digest = _require_hex(
        str(qualification_gate.get("teacher_checkpoint_sha256", "")),
        64,
        "legacy teacher checkpoint sha256",
    )
    if sha256_file(teacher_path) != teacher_digest:
        raise RuntimeError("legacy teacher checkpoint hash mismatch")

    legacy_approval = _authenticate_legacy_approval(
        root,
        qualification_gate_path,
        qualification_manifest_path,
        teacher_path,
        teacher_digest,
        qualification_evidence,
    )

    factorial_gate_path = _regular_file(
        factorial_root / "gate.json", "legacy factorial gate"
    )
    factorial_manifest_path = _regular_file(
        factorial_root / "manifest.json", "legacy factorial manifest"
    )
    factorial_gate = read_strict_json(factorial_gate_path)
    factorial_evidence = _authenticate_stage(
        factorial_root,
        factorial_gate,
        factorial_manifest_path,
        expected_schema=FACTORIAL_SCHEMA,
        expected_scientific_use="CANDIDATE_NOT_CLAIM",
        source=source,
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
    )
    if factorial_gate.get("qualification_gate_sha256") != sha256_file(
        qualification_gate_path
    ):
        raise RuntimeError("legacy factorial gate does not bind qualification")
    status = factorial_gate.get("status")
    passed = factorial_gate.get("passed")
    if status not in {"PASS", "FAIL"} or passed is not (status == "PASS"):
        raise RuntimeError("legacy factorial scientific status is malformed")
    gate_vector = factorial_gate.get("gate_vector")
    expected_gate_ids = {f"G{index}" for index in range(9)}
    if (
        not isinstance(gate_vector, dict)
        or set(gate_vector) != expected_gate_ids
        or any(
            value not in {"PASS", "FAIL", "BLOCKED", "NOT_ASSESSED"}
            for value in gate_vector.values()
        )
    ):
        raise RuntimeError("legacy factorial gate vector is malformed")

    for relative in _FACTORIAL_REQUIRED_FILES:
        _regular_file(factorial_root / relative, f"legacy factorial {relative}")
    normalization_path = factorial_root / "normalization.json"
    normalization = read_strict_json(normalization_path)
    if not isinstance(normalization, dict) or not normalization:
        raise RuntimeError("legacy factorial normalization is malformed")
    frozen_pilot_path = factorial_root / "frozen_pilot.json"
    _validate_frozen_pilot(read_strict_json(frozen_pilot_path))

    checkpoint_index_path = factorial_root / "checkpoint_index.json"
    checkpoint_index = read_strict_json(checkpoint_index_path)
    checkpoints = _authenticate_checkpoint_index(
        checkpoint_index,
        factorial_root,
        factorial_evidence,
        teacher_digest,
        normalization,
        expected_seeds,
    )
    resume_index_path = factorial_root / "resume_index.json"
    context_sha, training_steps = _authenticate_resume_index(
        read_strict_json(resume_index_path),
        factorial_root,
        factorial_evidence,
        expected_seeds,
    )

    qualification_binding = {
        "gate_path": str(qualification_gate_path),
        "gate_sha256": sha256_file(qualification_gate_path),
        "manifest_path": str(qualification_manifest_path),
        "manifest_sha256": sha256_file(qualification_manifest_path),
        "teacher_checkpoint_path": str(teacher_path),
        "teacher_checkpoint_sha256": teacher_digest,
        "evidence_sha256": _canonical_sha256(qualification_evidence),
    }
    factorial_binding = {
        "root": str(factorial_root.resolve()),
        "gate_path": str(factorial_gate_path),
        "gate_sha256": sha256_file(factorial_gate_path),
        "manifest_path": str(factorial_manifest_path),
        "manifest_sha256": sha256_file(factorial_manifest_path),
        "checkpoint_index_path": str(checkpoint_index_path),
        "checkpoint_index_sha256": sha256_file(checkpoint_index_path),
        "resume_index_path": str(resume_index_path),
        "resume_index_sha256": sha256_file(resume_index_path),
        "normalization_path": str(normalization_path),
        "normalization_sha256": sha256_file(normalization_path),
        "frozen_pilot_path": str(frozen_pilot_path),
        "frozen_pilot_sha256": sha256_file(frozen_pilot_path),
        "factorial_statistics_path": str(
            factorial_root / "factorial_statistics.json"
        ),
        "factorial_statistics_sha256": sha256_file(
            factorial_root / "factorial_statistics.json"
        ),
        "training_summary_path": str(factorial_root / "training_summary.csv"),
        "training_summary_sha256": sha256_file(
            factorial_root / "training_summary.csv"
        ),
        "localization_per_bank_path": str(
            factorial_root / "localization_per_bank.csv"
        ),
        "localization_per_bank_sha256": sha256_file(
            factorial_root / "localization_per_bank.csv"
        ),
        "localization_per_sample_path": str(
            factorial_root / "localization_per_sample.csv"
        ),
        "localization_per_sample_sha256": sha256_file(
            factorial_root / "localization_per_sample.csv"
        ),
        "checkpoint_context_sha256": context_sha,
        "training_steps": training_steps,
        "evidence_sha256": _canonical_sha256(factorial_evidence),
    }
    _require_exact_fields(
        qualification_binding,
        _QUALIFICATION_BINDING_FIELDS,
        "qualification migration binding",
    )
    _require_exact_fields(
        factorial_binding,
        _FACTORIAL_BINDING_FIELDS,
        "factorial migration binding",
    )
    scientific_state = {
        "qualification_status": qualification_gate["status"],
        "qualification_passed": qualification_gate["passed"],
        "factorial_status": status,
        "factorial_passed": passed,
        "factorial_gate_vector": gate_vector,
    }
    return {
        "legacy_approval": legacy_approval,
        "qualification": qualification_binding,
        "factorial": factorial_binding,
        "checkpoint_inventory": checkpoints,
        "legacy_scientific_state": scientific_state,
    }


def _authenticate_stage(
    stage_root: Path,
    gate: object,
    manifest_path: Path,
    *,
    expected_schema: str,
    expected_scientific_use: str,
    source: dict[str, object],
    dataset_sha256: str,
    config_sha256: str,
) -> dict[str, object]:
    if not isinstance(gate, dict) or gate.get("schema_version") != expected_schema:
        raise RuntimeError(f"legacy gate schema mismatch: {stage_root}")
    manifest = read_strict_json(manifest_path)
    expected_manifest_fields = {
        "schema_version",
        "scientific_use",
        "files",
        *EVIDENCE_AUTH_KEYS,
    }
    _require_exact_fields(manifest, expected_manifest_fields, "legacy stage manifest")
    if manifest["schema_version"] != _STAGE_MANIFEST_SCHEMA:
        raise RuntimeError("legacy stage manifest schema mismatch")
    evidence = {
        key: manifest[key]
        for key in (*EVIDENCE_AUTH_KEYS, "scientific_use")
    }
    if (
        evidence["dataset_sha256"] != dataset_sha256
        or evidence["config_sha256"] != config_sha256
        or evidence["fixture"] is not False
        or evidence["scientific_use"] != expected_scientific_use
        or evidence["source_tree_sha256"] != source["source_tree_sha256"]
        or evidence["requirements_lock_sha256"]
        != source["requirements_lock_sha256"]
    ):
        raise RuntimeError("legacy stage evidence identity mismatch")
    _validate_recorded_runtime(evidence)
    for key, value in evidence.items():
        if gate.get(key) != value:
            raise RuntimeError(f"legacy gate evidence mismatch: {key}")
    _authenticate_stage_inventory(stage_root, manifest["files"], evidence)
    gate_path = stage_root / "gate.json"
    matches = [
        row
        for row in manifest["files"]
        if isinstance(row, dict) and row.get("path") == "gate.json"
    ]
    if len(matches) != 1 or matches[0].get("sha256") != sha256_file(gate_path):
        raise RuntimeError("legacy gate is absent from its stage manifest")
    return evidence


def _authenticate_stage_inventory(
    stage_root: Path, entries: object, evidence: dict[str, object]
) -> None:
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("legacy stage manifest has no inventory")
    expected: dict[str, dict[str, object]] = {}
    entry_fields = {"path", "bytes", "sha256", *evidence}
    for row in entries:
        _require_exact_fields(row, entry_fields, "legacy stage inventory row")
        relative = row["path"]
        digest = row["sha256"]
        size = row["bytes"]
        if (
            not isinstance(relative, str)
            or not relative
            or relative in expected
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not _is_hex(digest, 64)
            or type(size) is not int
            or size < 0
        ):
            raise RuntimeError("legacy stage inventory row identity is malformed")
        for key, value in evidence.items():
            if row[key] != value:
                raise RuntimeError(f"legacy inventory evidence mismatch: {key}")
        path = _regular_file(stage_root / relative, "legacy stage artifact")
        if not _is_within(path, stage_root):
            raise RuntimeError("legacy stage artifact escapes its stage root")
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise RuntimeError(f"legacy stage artifact changed: {relative}")
        expected[relative] = row
    actual = {
        path.relative_to(stage_root).as_posix()
        for path in stage_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(expected) != actual:
        raise RuntimeError(
            "legacy stage inventory is incomplete: "
            f"missing={sorted(actual - set(expected))[:5]}, "
            f"unexpected={sorted(set(expected) - actual)[:5]}"
        )


def _authenticate_checkpoint_index(
    index: object,
    factorial_root: Path,
    evidence: dict[str, object],
    teacher_sha256: str,
    normalization: dict[str, object],
    expected_seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    expected_index_fields = {
        "schema_version",
        "scientific_use",
        "checkpoints",
        *EVIDENCE_AUTH_KEYS,
    }
    _require_exact_fields(index, expected_index_fields, "legacy checkpoint index")
    if index["schema_version"] != _CHECKPOINT_INDEX_SCHEMA:
        raise RuntimeError("legacy checkpoint index schema mismatch")
    for key, value in evidence.items():
        if index[key] != value:
            raise RuntimeError(f"legacy checkpoint index evidence mismatch: {key}")
    rows = index["checkpoints"]
    expected_cells = [(seed, arm) for seed in expected_seeds for arm in ARMS]
    if not isinstance(rows, list) or len(rows) != len(expected_cells):
        raise RuntimeError("legacy checkpoint index does not contain 3 x 4 cells")
    scalar_evidence = {
        key: value
        for key, value in evidence.items()
        if not isinstance(value, (dict, list))
    }
    row_fields = {
        "seed",
        "arm",
        "path",
        "sha256",
        "parameters",
        "measured_flops_per_step",
        "execution_device",
        "teacher_checkpoint_sha256",
        *scalar_evidence,
    }
    inventory = []
    for row, (seed, arm) in zip(rows, expected_cells):
        _require_exact_fields(row, row_fields, "legacy checkpoint index row")
        if type(row["seed"]) is not int or row["seed"] != seed or row["arm"] != arm:
            raise RuntimeError("legacy checkpoint index order or cell identity changed")
        for key, value in scalar_evidence.items():
            if row[key] != value:
                raise RuntimeError(f"legacy checkpoint row evidence mismatch: {key}")
        if (
            row["teacher_checkpoint_sha256"] != teacher_sha256
            or type(row["parameters"]) is not int
            or row["parameters"] <= 0
            or type(row["measured_flops_per_step"]) is not int
            or row["measured_flops_per_step"] <= 0
            or not isinstance(row["execution_device"], str)
            or not row["execution_device"]
        ):
            raise RuntimeError("legacy checkpoint index row metadata is invalid")
        path = _bound_relative_file(
            factorial_root, row["path"], row["sha256"], "legacy factorial checkpoint"
        )
        payload = _validate_checkpoint_payload(
            path,
            row,
            evidence,
            teacher_sha256,
            normalization,
        )
        binding = {
            "seed": seed,
            "arm": arm,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": str(row["sha256"]),
            "teacher_checkpoint_sha256": teacher_sha256,
            "source_tree_sha256": str(payload["source_tree_sha256"]),
        }
        _require_exact_fields(
            binding, _CHECKPOINT_BINDING_FIELDS, "checkpoint migration binding"
        )
        inventory.append(binding)
    return inventory


def _validate_checkpoint_payload(
    path: Path,
    row: dict[str, object],
    evidence: dict[str, object],
    teacher_sha256: str,
    normalization: dict[str, object],
) -> dict[str, object]:
    try:
        from .formal_model import CSIPairsFormalModel, torch

        if torch is None:
            raise RuntimeError("PyTorch is unavailable")
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError(f"legacy checkpoint is unreadable: {path}") from error
    required = {
        "schema_version",
        "arm",
        "seed",
        "model_spec",
        "normalization",
        "teacher_checkpoint_sha256",
        "checkpoint_rule",
        "state_dict",
        "scientific_use",
        *EVIDENCE_AUTH_KEYS,
    }
    _require_exact_fields(payload, required, "legacy checkpoint payload")
    if (
        payload["schema_version"] != _CHECKPOINT_SCHEMA
        or payload["arm"] != row["arm"]
        or type(payload["seed"]) is not int
        or payload["seed"] != row["seed"]
        or payload["checkpoint_rule"] != "fixed_final_step_no_target_selection"
        or payload["teacher_checkpoint_sha256"] != teacher_sha256
        or payload["normalization"] != normalization
    ):
        raise RuntimeError("legacy checkpoint payload identity mismatch")
    for key, value in evidence.items():
        if payload[key] != value:
            raise RuntimeError(f"legacy checkpoint payload evidence mismatch: {key}")
    if not isinstance(payload["model_spec"], dict) or not isinstance(
        payload["state_dict"], dict
    ):
        raise RuntimeError("legacy checkpoint model contract is malformed")
    try:
        model = CSIPairsFormalModel(**payload["model_spec"])
        model.load_state_dict(payload["state_dict"], strict=True)
        if set(model.state_dict()) != set(payload["state_dict"]):
            raise RuntimeError("checkpoint state keys differ from the model")
    except Exception as error:
        raise RuntimeError("legacy checkpoint state does not load strictly") from error
    return payload


def _authenticate_resume_index(
    index: object,
    factorial_root: Path,
    evidence: dict[str, object],
    expected_seeds: tuple[int, ...],
) -> tuple[str, int]:
    required = {
        "schema_version",
        "scientific_use",
        "checkpoint_interval_steps",
        "jobs",
        *EVIDENCE_AUTH_KEYS,
    }
    _require_exact_fields(index, required, "legacy factorial resume index")
    if index["schema_version"] != _RESUME_INDEX_SCHEMA:
        raise RuntimeError("legacy factorial resume index schema mismatch")
    for key, value in evidence.items():
        if index[key] != value:
            raise RuntimeError(f"legacy resume index evidence mismatch: {key}")
    if (
        type(index["checkpoint_interval_steps"]) is not int
        or index["checkpoint_interval_steps"] <= 0
    ):
        raise RuntimeError("legacy resume checkpoint interval is invalid")
    jobs = index["jobs"]
    expected_cells = [(seed, arm) for seed in expected_seeds for arm in ARMS]
    if not isinstance(jobs, list) or len(jobs) != len(expected_cells):
        raise RuntimeError("legacy resume index does not contain 3 x 4 jobs")
    job_fields = {
        "seed",
        "arm",
        "pointer_path",
        "pointer_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "completed_steps",
    }
    pointer_fields = {
        "schema_version",
        "status",
        "seed",
        "arm",
        "completed_steps",
        "total_steps",
        "batch_size",
        "generation",
        "context_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_bytes",
        "checkpoint_schema_version",
    }
    contexts: set[str] = set()
    training_steps: set[int] = set()
    for job, (seed, arm) in zip(jobs, expected_cells):
        _require_exact_fields(job, job_fields, "legacy factorial resume job")
        if type(job["seed"]) is not int or job["seed"] != seed or job["arm"] != arm:
            raise RuntimeError("legacy resume job order or identity changed")
        pointer_path = _bound_relative_file(
            factorial_root,
            job["pointer_path"],
            job["pointer_sha256"],
            "legacy factorial resume pointer",
        )
        pointer = read_strict_json(pointer_path)
        _require_exact_fields(pointer, pointer_fields, "legacy factorial resume pointer")
        if (
            pointer["schema_version"] != _RESUME_POINTER_SCHEMA
            or pointer["checkpoint_schema_version"] != _RESUME_CHECKPOINT_SCHEMA
            or pointer["status"] != "COMPLETE"
            or type(pointer["seed"]) is not int
            or pointer["seed"] != seed
            or pointer["arm"] != arm
            or pointer["completed_steps"] != pointer["total_steps"]
            or pointer["completed_steps"] != job["completed_steps"]
            or type(pointer["completed_steps"]) is not int
            or type(pointer["total_steps"]) is not int
            or pointer["completed_steps"] <= 0
            or type(job["completed_steps"]) is not int
            or type(pointer["batch_size"]) is not int
            or pointer["batch_size"] <= 0
            or type(pointer["generation"]) is not int
            or pointer["generation"] < 0
        ):
            raise RuntimeError("legacy factorial resume pointer is not complete")
        checkpoint = _bound_relative_file(
            pointer_path.parent,
            pointer["checkpoint_path"],
            pointer["checkpoint_sha256"],
            "legacy factorial resume checkpoint",
        )
        indexed_checkpoint = _bound_relative_file(
            factorial_root,
            job["checkpoint_path"],
            job["checkpoint_sha256"],
            "legacy indexed resume checkpoint",
        )
        if (
            checkpoint != indexed_checkpoint
            or checkpoint.stat().st_size != pointer["checkpoint_bytes"]
            or job["checkpoint_sha256"] != pointer["checkpoint_sha256"]
        ):
            raise RuntimeError("legacy resume checkpoint binding mismatch")
        contexts.add(_require_hex(pointer["context_sha256"], 64, "resume context sha256"))
        training_steps.add(int(pointer["completed_steps"]))
    if len(contexts) != 1 or training_steps != {_FORMAL_FACTORIAL_STEPS}:
        raise RuntimeError("legacy resume jobs do not share one frozen context and step count")
    return next(iter(contexts)), next(iter(training_steps))


def _authenticate_legacy_approval(
    root: Path,
    qualification_gate_path: Path,
    qualification_manifest_path: Path,
    teacher_path: Path,
    teacher_sha256: str,
    evidence: dict[str, object],
) -> dict[str, object]:
    approval_root = root / "approval"
    preflight_path = _regular_file(
        approval_root / "preflight.json", "legacy approval preflight"
    )
    prepared_path = _regular_file(
        approval_root / "prepared.json", "legacy prepared record"
    )
    request_path = _regular_file(
        approval_root / "request.json", "legacy approval request"
    )
    accepted_path = _regular_file(
        approval_root / "accepted.json", "legacy accepted approval record"
    )
    preflight = read_strict_json(preflight_path)
    prepared = read_strict_json(prepared_path)
    request = read_strict_json(request_path)
    accepted = read_strict_json(accepted_path)
    _require_exact_fields(prepared, _LEGACY_PREPARED_FIELDS, "legacy prepared record")
    _require_exact_fields(request, _LEGACY_APPROVAL_REQUEST_FIELDS, "legacy approval request")
    _require_exact_fields(accepted, _LEGACY_ACCEPTED_FIELDS, "legacy accepted approval")
    if (
        request["schema_version"] != "csi-pairs-full-run-approval-request-v2"
        or prepared["schema_version"] != "csi-pairs-full-run-prepared-v2"
        or accepted["schema_version"] != "csi-pairs-full-run-approval-accepted-v2"
        or prepared["status"] != "AWAITING_LLM_JUDGE"
        or accepted["status"] != "ACCEPTED"
        or request["decision_required"] != LLM_JUDGE_REQUIRED
        or request["scientific_use"] != "FORMAL_EXPERIMENT_ALLOWED"
        or request["fixture"] is not False
        or request["prepared_root"] != str(root)
        or prepared["prepared_root"] != str(root)
        or prepared["request_path"] != "approval/request.json"
        or prepared["request_sha256"] != sha256_file(request_path)
        or request["preflight_path"] != "approval/preflight.json"
        or request["preflight_sha256"] != sha256_file(preflight_path)
    ):
        raise RuntimeError("legacy approval request/prepared identity mismatch")
    for key in ("run_id", "run_nonce"):
        if prepared[key] != request[key] or accepted[key] != request[key]:
            raise RuntimeError(f"legacy approval {key} binding mismatch")
    if (
        request["run_id"] != f"csi-pairs-{str(request['run_nonce'])[:16]}"
        or not _is_hex(request["run_nonce"], 64)
        or accepted["request_sha256"] != sha256_file(request_path)
    ):
        raise RuntimeError("legacy approval run identity is invalid")
    # The v2 full-run approval request predates per-artifact labels.  Bind the
    # shared evidence identity exactly, but do not require the stage-only
    # ``artifact_label`` field that is absent from the locked request schema.
    for key in (
        "dataset_sha256",
        "config_sha256",
        "fixture",
        "source_tree_sha256",
        "requirements_lock_sha256",
        "runtime_provenance_sha256",
        "runtime_provenance",
    ):
        if request[key] != evidence[key]:
            raise RuntimeError(f"legacy approval request evidence mismatch: {key}")
    for key in (
        "config_sha256",
        "dataset_sha256",
        "fixture",
        "source_tree_sha256",
        "requirements_lock_sha256",
        "runtime_provenance_sha256",
        "runtime_provenance",
    ):
        if preflight.get(key) != request[key]:
            raise RuntimeError(f"legacy preflight binding mismatch: {key}")
    teacher_binding = _bound_relative_file(
        root,
        request["teacher_checkpoint"],
        request["teacher_checkpoint_sha256"],
        "legacy approved teacher checkpoint",
    )
    if teacher_binding != teacher_path or request["teacher_checkpoint_sha256"] != teacher_sha256:
        raise RuntimeError("legacy approval teacher binding mismatch")
    gates = request["gate_bindings"]
    if not isinstance(gates, dict) or not gates or "G1_G2" not in gates:
        raise RuntimeError("legacy approval omitted the G1/G2 binding")
    bound_gate_paths: set[Path] = set()
    for name, binding in gates.items():
        if not isinstance(name, str) or not name:
            raise RuntimeError("legacy approval gate name is malformed")
        _require_exact_fields(
            binding,
            {"gate_path", "gate_sha256", "manifest_path", "manifest_sha256"},
            f"legacy approval {name} gate binding",
        )
        gate_file = _bound_relative_file(
            root,
            binding["gate_path"],
            binding["gate_sha256"],
            f"legacy approval {name} gate",
        )
        manifest_file = _bound_relative_file(
            root,
            binding["manifest_path"],
            binding["manifest_sha256"],
            f"legacy approval {name} manifest",
        )
        if gate_file.parent != manifest_file.parent or gate_file in bound_gate_paths:
            raise RuntimeError("legacy approval gate binding paths are inconsistent")
        bound_gate_paths.add(gate_file)
    qualification_binding = gates["G1_G2"]
    if qualification_binding != {
        "gate_path": "qualification/gate.json",
        "gate_sha256": sha256_file(qualification_gate_path),
        "manifest_path": "qualification/manifest.json",
        "manifest_sha256": sha256_file(qualification_manifest_path),
    }:
        raise RuntimeError("legacy approval G1/G2 binding mismatch")
    if (
        not isinstance(request.get("compute_plan"), dict)
        or not _is_hex(request["compute_plan"].get("sha256"), 64)
        or accepted["compute_plan_sha256"] != request["compute_plan"]["sha256"]
    ):
        raise RuntimeError("legacy approval compute-plan binding mismatch")

    external_path = _regular_file(
        accepted["approval_manifest_path"], "legacy external LLM approval"
    )
    if _is_within(external_path, root):
        raise RuntimeError("legacy LLM approval was stored inside its run root")
    if accepted["approval_manifest_sha256"] != sha256_file(external_path):
        raise RuntimeError("legacy external approval hash mismatch")
    external = read_strict_json(external_path)
    _require_exact_fields(
        external, _LEGACY_EXTERNAL_APPROVAL_FIELDS, "legacy external LLM approval"
    )
    if (
        external["schema_version"] != LLM_JUDGE_APPROVAL_SCHEMA
        or external["decision"] != "APPROVE"
        or external["run_id"] != request["run_id"]
        or external["run_nonce"] != request["run_nonce"]
        or external["request_sha256"] != sha256_file(request_path)
        or external["compute_plan_sha256"] != request["compute_plan"]["sha256"]
        or external["attestation"] != APPROVAL_ATTESTATION
        or external["approved_gate_sha256s"]
        != {name: value["gate_sha256"] for name, value in gates.items()}
    ):
        raise RuntimeError("legacy external approval does not bind the prepared run")
    parse_llm_judge(external["judge"])
    prepared_time = _parse_utc(str(request["prepared_utc"]), "prepared_utc")
    approved_time = _parse_utc(str(external["approved_utc"]), "approved_utc")
    expires_time = _parse_utc(str(external["expires_utc"]), "expires_utc")
    accepted_time = _parse_utc(str(accepted["accepted_utc"]), "accepted_utc")
    if (
        approved_time < prepared_time
        or expires_time <= approved_time
        or expires_time - approved_time > MAX_APPROVAL_LIFETIME
        or accepted_time < approved_time
        or accepted_time > expires_time
    ):
        raise RuntimeError("legacy approval chronology is invalid")
    result = {
        "run_id": request["run_id"],
        "run_nonce": request["run_nonce"],
        "preflight_path": str(preflight_path),
        "preflight_sha256": sha256_file(preflight_path),
        "prepared_path": str(prepared_path),
        "prepared_sha256": sha256_file(prepared_path),
        "request_path": str(request_path),
        "request_sha256": sha256_file(request_path),
        "accepted_path": str(accepted_path),
        "accepted_sha256": sha256_file(accepted_path),
        "external_approval_path": str(external_path),
        "external_approval_sha256": sha256_file(external_path),
        "judge": external["judge"],
    }
    _require_exact_fields(result, _LEGACY_APPROVAL_FIELDS, "legacy approval binding")
    return result


def _validate_recorded_runtime(evidence: dict[str, object]) -> None:
    runtime = evidence.get("runtime_provenance")
    _require_exact_fields(runtime, RUNTIME_PROVENANCE_FIELDS, "legacy runtime provenance")
    if (
        runtime.get("schema_version") != "csi-pairs-runtime-provenance-v3"
        or runtime.get("source_tree_sha256") != evidence["source_tree_sha256"]
        or runtime.get("requirements_lock_sha256")
        != evidence["requirements_lock_sha256"]
        or runtime.get("python_implementation") != "CPython"
        or runtime.get("python_dont_write_bytecode") is not True
    ):
        raise RuntimeError("legacy recorded runtime identity mismatch")
    if evidence.get("runtime_provenance_sha256") != _canonical_sha256(runtime):
        raise RuntimeError("legacy runtime provenance digest mismatch")


def _validate_frozen_pilot(payload: object) -> None:
    required = {
        "schema_version",
        "source_roles",
        "role_permissions",
        "pilot_contract",
        "pilot_seed",
        "pilot_steps",
        "alignment_scale",
        "response_scale",
        "alignment_null_tolerance",
        "checkpoint_reused_for_final_training",
    }
    _require_exact_fields(payload, required, "legacy frozen pilot")
    if (
        payload["schema_version"] != _FROZEN_PILOT_SCHEMA
        or payload["checkpoint_reused_for_final_training"] is not False
        or type(payload["pilot_seed"]) is not int
        or type(payload["pilot_steps"]) is not int
        or payload["pilot_steps"] <= 0
        or not isinstance(payload["source_roles"], list)
        or not isinstance(payload["role_permissions"], dict)
    ):
        raise RuntimeError("legacy frozen pilot identity is invalid")
    for key in (
        "alignment_scale",
        "response_scale",
        "alignment_null_tolerance",
    ):
        value = payload[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise RuntimeError(f"legacy frozen pilot {key} is invalid")
    if payload["alignment_scale"] <= 0 or payload["response_scale"] <= 0:
        raise RuntimeError("legacy frozen loss scales must be positive")


def _source_identity(source_root: Path) -> dict[str, object]:
    source_root = _regular_directory(source_root, "formal source root")
    symlinks = [
        path
        for path in source_root.rglob("*")
        if path.is_symlink()
        and not any(
            part.startswith(".venv-") or part.startswith(".runtime-")
            for part in path.relative_to(source_root).parts
        )
    ]
    if symlinks:
        raise RuntimeError("formal source root contains symbolic links")
    forbidden = [
        path
        for path in source_root.rglob("*.pyc")
        if path.is_file()
        and not any(
            part.startswith(".venv-") or part.startswith(".runtime-")
            for part in path.relative_to(source_root).parts
        )
    ]
    if forbidden:
        raise RuntimeError("formal source root contains forbidden bytecode")
    git_root = Path(
        _git(source_root, "rev-parse", "--show-toplevel").strip()
    ).resolve()
    if not _is_within(source_root, git_root):
        raise RuntimeError("formal source root is outside its git worktree")
    commit = _git(source_root, "rev-parse", "HEAD").strip()
    if _HEX40.fullmatch(commit) is None:
        raise RuntimeError("formal source git commit is invalid")
    tracked_status = _git(
        source_root, "status", "--porcelain", "--untracked-files=no"
    )
    if tracked_status.strip():
        raise RuntimeError("formal source git worktree has tracked modifications")
    included_paths = _source_tree_files(source_root)
    relative_root = source_root.relative_to(git_root).as_posix() or "."
    tracked_paths = {
        value
        for value in _git(
            git_root,
            "ls-files",
            "--cached",
            "-z",
            "--",
            relative_root,
        ).split("\0")
        if value
    }
    included_repository_paths = {
        path.relative_to(git_root).as_posix() for path in included_paths
    }
    untracked_included = sorted(included_repository_paths - tracked_paths)
    if untracked_included:
        raise RuntimeError(
            "formal source tree contains digest-included untracked files: "
            + ", ".join(untracked_included)
        )
    source_digest = _source_tree_sha256(source_root)
    requirements_path = _requirements_lock_path(source_root)
    return {
        "source_root": str(source_root),
        "git_root": str(git_root),
        "git_commit": commit,
        "source_tree_sha256": source_digest,
        "requirements_lock_path": str(requirements_path),
        "requirements_lock_sha256": sha256_file(requirements_path),
    }


def _source_tree_sha256(source_root: Path) -> str:
    paths = _source_tree_files(source_root)
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(source_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _source_tree_files(source_root: Path) -> tuple[Path, ...]:
    included_suffixes = {
        ".py",
        ".json",
        ".sh",
        ".txt",
        ".toml",
        ".lock",
        ".yaml",
        ".yml",
    }
    return tuple(
        sorted(
            path
            for path in source_root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.suffix in included_suffixes
            and not any(
                part.startswith(".venv-") or part.startswith(".runtime-")
                for part in path.relative_to(source_root).parts
            )
            and "__pycache__" not in path.parts
        )
    )


def _requirements_lock_path(source_root: Path) -> Path:
    # A migration does not re-resolve dependencies. It binds the exact checked-in
    # lock for the host platform used by the formal pipeline.
    candidates = [
        source_root / "requirements-lock-linux-x86_64-cu121.txt",
        source_root / "requirements-lock.txt",
    ]
    present = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if not present:
        raise RuntimeError("formal source root has no supported requirements lock")
    if os.uname().sysname == "Linux" and os.uname().machine == "x86_64":
        expected = candidates[0]
    else:
        expected = candidates[1]
    return _regular_file(expected, "formal requirements lock")


def _validate_migration_approval(
    approval: object,
    request: dict[str, object],
    request_sha256: str,
    *,
    now: datetime,
) -> None:
    _require_exact_fields(
        approval, _MIGRATION_APPROVAL_FIELDS, "migration LLM approval"
    )
    if (
        approval["schema_version"] != MIGRATION_APPROVAL_SCHEMA
        or approval["decision"] != "APPROVE"
        or approval["migration_id"] != request["migration_id"]
        or approval["migration_nonce"] != request["migration_nonce"]
        or approval["new_run_id"] != request["new_run_id"]
        or approval["new_run_nonce"] != request["new_run_nonce"]
        or approval["request_sha256"] != request_sha256
        or approval["legacy_source_tree_sha256"]
        != request["legacy_source"]["source_tree_sha256"]
        or approval["new_source_tree_sha256"]
        != request["new_source"]["source_tree_sha256"]
        or approval["checkpoint_inventory_sha256"]
        != request["checkpoint_inventory_sha256"]
        or approval["new_compute_plan_sha256"]
        != request["new_compute_plan"]["sha256"]
        or approval["migration_evidence_sha256"]
        != request["migration_evidence_sha256"]
        or approval["attestation"] != MIGRATION_APPROVAL_ATTESTATION
    ):
        raise RuntimeError("migration LLM approval binding mismatch")
    parse_llm_judge(approval["judge"])
    created = _parse_utc(str(request["created_utc"]), "created_utc")
    approved = _parse_utc(str(approval["approved_utc"]), "approved_utc")
    expires = _parse_utc(str(approval["expires_utc"]), "expires_utc")
    if (
        approved < created
        or approved > now + MAX_APPROVAL_CLOCK_SKEW
        or expires <= approved
        or expires - approved > MAX_APPROVAL_LIFETIME
        or now > expires
    ):
        raise RuntimeError("migration LLM approval chronology is invalid")


def _accepted_record(
    request_path: Path,
    request: dict[str, object],
    approval_path: Path,
    accepted_utc: str,
) -> dict[str, object]:
    return {
        "schema_version": MIGRATION_ACCEPTED_SCHEMA,
        "status": "ACCEPTED",
        "migration_id": request["migration_id"],
        "migration_nonce": request["migration_nonce"],
        "new_run_id": request["new_run_id"],
        "new_run_nonce": request["new_run_nonce"],
        "request_path": str(request_path),
        "request_sha256": sha256_file(request_path),
        "approval_manifest_path": str(approval_path),
        "approval_manifest_sha256": sha256_file(approval_path),
        "legacy_run_root": request["legacy_run_root"],
        "new_run_root": request["new_run_root"],
        "legacy_source_tree_sha256": request["legacy_source"][
            "source_tree_sha256"
        ],
        "new_source_tree_sha256": request["new_source"]["source_tree_sha256"],
        "checkpoint_inventory_sha256": request[
            "checkpoint_inventory_sha256"
        ],
        "new_compute_plan_sha256": request["new_compute_plan"]["sha256"],
        "migration_evidence_sha256": request["migration_evidence_sha256"],
        "accepted_utc": accepted_utc,
    }


def _bound_relative_file(
    root: Path, relative: object, digest: object, label: str
) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not _is_hex(digest, 64)
    ):
        raise RuntimeError(f"{label} binding is malformed")
    path = _regular_file(root / relative, label)
    if not _is_within(path, root) or sha256_file(path) != digest:
        raise RuntimeError(f"{label} escapes its root or has a mismatched hash")
    return path


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise RuntimeError(f"{label} cannot be a symlink")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"{label} is missing or not a regular file")
    return resolved


def _regular_directory(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise RuntimeError(f"{label} cannot be a symlink")
    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"{label} is missing or not a regular directory")
    return resolved


def _require_disjoint_roots(first: Path, second: Path) -> None:
    if first == second or _is_within(first, second) or _is_within(second, first):
        raise RuntimeError("legacy and new run roots must be disjoint")


def _is_within(path: Path, root: Path) -> bool:
    path = Path(path).resolve()
    root = Path(root).resolve()
    return path == root or root in path.parents


def _validated_seeds(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("expected seeds must be a sequence of integers")
    seeds = tuple(values)
    if (
        len(seeds) != 3
        or len(set(seeds)) != 3
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("migration requires exactly three distinct nonnegative seeds")
    return seeds


def _require_exact_fields(value: object, fields: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != set(fields):
        observed = set(value) if isinstance(value, dict) else set()
        raise RuntimeError(
            f"{label} fields must be exact; "
            f"missing={sorted(set(fields) - observed)}, "
            f"unexpected={sorted(observed - set(fields))}"
        )


def _require_hex(value: object, width: int, label: str) -> str:
    if not _is_hex(value, width):
        raise ValueError(f"{label} must be {width} lowercase hexadecimal characters")
    return str(value)


def _is_hex(value: object, width: int) -> bool:
    if not isinstance(value, str):
        return False
    pattern = _HEX64 if width == 64 else _HEX40
    return pattern.fullmatch(value) is not None


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_utc(value: str, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError(f"{label} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise RuntimeError(f"{label} is invalid") from error
    if parsed.tzinfo != timezone.utc:
        raise RuntimeError(f"{label} must use UTC")
    return parsed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(source_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git identity check failed: {' '.join(arguments)}: {completed.stderr.strip()}"
        )
    return completed.stdout


def _write_json_exclusive_atomic(path: Path, payload: object) -> None:
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError("migration receipt parent must be a regular directory")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite migration evidence: {path}")
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
    temporary = parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
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
        os.link(temporary, path, follow_symlinks=False)
        directory_descriptor = os.open(
            parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        raise
    finally:
        temporary.unlink(missing_ok=True)
