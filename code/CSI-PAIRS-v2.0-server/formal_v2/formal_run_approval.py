from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

from .formal_evidence import (
    QUALIFICATION_SCHEMA,
    config_sha256,
    evidence_context,
    require_manifested_formal_qualification,
    require_stage_manifested_gate,
)
from .formal_io import read_strict_json, sha256_file, write_json
from .formal_external_runtime import _cuda_total_memory_bytes
from .formal_llm_judge import (
    APPROVAL_ATTESTATION,
    AWAITING_LLM_JUDGE,
    LLM_JUDGE_APPROVAL_SCHEMA,
    LLM_JUDGE_REQUIRED,
    parse_llm_judge,
)


COMPUTE_PLAN_SCHEMA = "csi-pairs-full-run-compute-plan-v2"
FORMAL_COMPUTE_COMPONENTS = (
    "data_qualification_and_g8",
    "teacher",
    "factorial",
    "evaluation",
    "baselines",
    "controls",
    "io_and_checkpoints",
    "failure_budget",
)
PREFLIGHT_SCHEMA = "csi-pairs-full-run-static-preflight-v2"
APPROVAL_REQUEST_SCHEMA = "csi-pairs-full-run-approval-request-v2"
PREPARED_RUN_SCHEMA = "csi-pairs-full-run-prepared-v2"
APPROVAL_ACCEPTED_SCHEMA = "csi-pairs-full-run-approval-accepted-v2"
MAX_APPROVAL_CLOCK_SKEW = timedelta(minutes=5)
MAX_APPROVAL_LIFETIME = timedelta(hours=24)
APPROVAL_REVIEW_SCOPE = [
    "G0 literature/resource gate",
    "independent RT calibration",
    "G1 native-noise and route coverage",
    "G2 teacher and Response gate",
    "oracle_x versus no_x",
    "copy, no_action, and exact action_swap controls",
    "null hallucination safety",
    "G8 independent external-validity gate",
    "compute and stopping budget",
]

REQUIRED_FULL_RUN_INPUT_NAMES = (
    "adapter_manifest",
    "control_manifest",
    "scene_id_manifest",
    "shuffled_pair_manifest",
    "retention_manifest",
    "representation_baseline_config",
)
OPTIONAL_FULL_RUN_INPUT_NAMES = (
    "verifier_manifest",
    "external_validity_manifest",
    "literature_resource_manifest",
    "rt_calibration_manifest",
)
FULL_RUN_INPUT_NAMES = REQUIRED_FULL_RUN_INPUT_NAMES + OPTIONAL_FULL_RUN_INPUT_NAMES
OPTIONAL_EARLY_STAGE_GATES = frozenset(
    {"G0", "independent_rt", "data_verification", "G8"}
)

EARLY_STAGE_GATES = {
    "waibu_resources": (
        "waibu_resources/gate.json",
        "waibu_resources/manifest.json",
        "csi-pairs-v6-waibu-resource-gate-v1",
    ),
    "G0": (
        "literature_resources/gate.json",
        "literature_resources/manifest.json",
        "csi-pairs-v6-literature-resource-gate-v4",
    ),
    "independent_rt": (
        "qualification/rt_calibration/gate.json",
        "qualification/rt_calibration/manifest.json",
        "csi-pairs-v6-rt-calibration-gate-v6",
    ),
    "data_verification": (
        "data_verification/gate.json",
        "data_verification/manifest.json",
        "csi-pairs-v6-data-verification-gate-v1",
    ),
    "G1_G2": (
        "qualification/gate.json",
        "qualification/manifest.json",
        QUALIFICATION_SCHEMA,
    ),
    "G8": (
        "external_validity/gate.json",
        "external_validity/manifest.json",
        "csi-pairs-v6-external-validity-gate-v4",
    ),
}


def full_run_input_values(args: object) -> dict[str, str]:
    values = {}
    for name in REQUIRED_FULL_RUN_INPUT_NAMES:
        values[name] = str(getattr(args, name))
    for name in OPTIONAL_FULL_RUN_INPUT_NAMES:
        value = getattr(args, name, None)
        if value is None:
            continue
        text = str(value).strip()
        if text and text != "None":
            values[name] = text
    return values


def preflight_full_run(
    config: dict,
    dataset,
    output_root: str | Path,
    compute_plan_path: str | Path | None,
    input_values: Mapping[str, str],
    *,
    resource_registry: str | Path,
    waibu_root: str | Path,
    resume: bool = False,
) -> dict:
    """Validate every static full-chain dependency before creating run artifacts."""
    dataset.validate_target_support_capacity(
        max(int(value) for value in config["localization"]["label_budgets"])
    )
    output = Path(output_root).resolve()
    if resume:
        if output.is_symlink() or not output.is_dir():
            raise RuntimeError("all requires an existing prepared run root")
    else:
        _validate_output_root_shape(output)
    bindings, payloads = _bind_input_files(input_values)
    registry_path = _regular_file(resource_registry, "waibu resource registry")
    bindings["resource_registry"] = _file_binding(registry_path)
    registry = read_strict_json(registry_path)

    required_licenses, external_runtimes = _validate_static_manifests(
        config,
        dataset,
        input_values,
        payloads,
        registry,
        Path(waibu_root),
    )
    bindings["prepared_inputs"] = _bind_pre_staged_inputs(output, payloads)
    compute_binding, plan_report = _resolve_compute_plan(
        compute_plan_path,
        dataset,
        output,
        required_licenses,
    )
    evidence = evidence_context(
        config,
        dataset,
        "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM",
    )
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "PASS",
        "config_sha256": config_sha256(config),
        "dataset_path": str(Path(dataset.source_path).resolve()),
        "dataset_sha256": sha256_file(dataset.source_path),
        "fixture": bool(dataset.is_fixture),
        "prepared_root": str(output),
        "source_tree_sha256": evidence["source_tree_sha256"],
        "requirements_lock_sha256": evidence["requirements_lock_sha256"],
        "runtime_provenance_sha256": evidence["runtime_provenance_sha256"],
        "runtime_provenance": evidence["runtime_provenance"],
        "external_runtime_provenance": external_runtimes,
        "compute_plan": compute_binding,
        "input_bindings": bindings,
        "required_license_acknowledgements": sorted(required_licenses),
        **plan_report,
    }


def write_approval_request(
    config: dict,
    dataset,
    output_root: str | Path,
    preflight: dict,
    *,
    run_nonce: str,
    prepared_utc: str | None = None,
) -> dict:
    output = Path(output_root).resolve()
    if re.fullmatch(r"[0-9a-f]{64}", run_nonce) is None:
        raise ValueError("run nonce must be 32 random bytes encoded as lowercase hex")
    approval_dir = output / "approval"
    approval_dir.mkdir(parents=True, exist_ok=True)
    request_path = approval_dir / "request.json"
    prepared_path = approval_dir / "prepared.json"
    if request_path.exists() or request_path.is_symlink() or prepared_path.exists() or prepared_path.is_symlink():
        raise FileExistsError("refusing to overwrite a full-run approval request")
    preflight_path = approval_dir / "preflight.json"
    if preflight_path.exists() or preflight_path.is_symlink():
        raise FileExistsError("refusing to overwrite a full-run preflight report")
    write_json(preflight_path, preflight)

    gate_bindings = authenticate_early_stages(config, dataset, output)
    qualification = read_strict_json(output / "qualification" / "gate.json")
    teacher = _bound_run_file(
        output,
        qualification["teacher_checkpoint"],
        qualification["teacher_checkpoint_sha256"],
        "qualification teacher checkpoint",
    )
    timestamp = prepared_utc or _utc_now()
    _parse_utc(timestamp, "prepared_utc")
    run_id = f"csi-pairs-{run_nonce[:16]}"
    request = {
        "schema_version": APPROVAL_REQUEST_SCHEMA,
        "run_id": run_id,
        "run_nonce": run_nonce,
        "prepared_utc": timestamp,
        "prepared_root": str(output),
        "config_sha256": preflight["config_sha256"],
        "dataset_sha256": preflight["dataset_sha256"],
        "fixture": preflight["fixture"],
        "source_tree_sha256": preflight["source_tree_sha256"],
        "requirements_lock_sha256": preflight["requirements_lock_sha256"],
        "runtime_provenance_sha256": preflight["runtime_provenance_sha256"],
        "runtime_provenance": preflight["runtime_provenance"],
        "external_runtime_provenance": preflight["external_runtime_provenance"],
        "gpu_inventory": preflight["gpu_inventory"],
        "required_gpu_count": preflight["required_gpu_count"],
        "execution_devices": preflight["execution_devices"],
        "required_environment_values": preflight["required_environment_values"],
        "compute_plan": preflight["compute_plan"],
        "input_bindings": preflight["input_bindings"],
        "preflight_path": str(preflight_path.relative_to(output)),
        "preflight_sha256": sha256_file(preflight_path),
        "gate_bindings": gate_bindings,
        "teacher_checkpoint": str(teacher.relative_to(output)),
        "teacher_checkpoint_sha256": sha256_file(teacher),
        "decision_required": LLM_JUDGE_REQUIRED,
        "scientific_use": qualification["scientific_use"],
        "review_scope": APPROVAL_REVIEW_SCOPE,
    }
    write_json(request_path, request)
    request_sha = sha256_file(request_path)
    prepared = {
        "schema_version": PREPARED_RUN_SCHEMA,
        "run_id": run_id,
        "run_nonce": run_nonce,
        "prepared_root": str(output),
        "request_path": str(request_path.relative_to(output)),
        "request_sha256": request_sha,
        "status": AWAITING_LLM_JUDGE,
    }
    write_json(prepared_path, prepared)
    return {
        "status": AWAITING_LLM_JUDGE,
        "passed": True,
        "run_id": run_id,
        "run_nonce": run_nonce,
        "request": str(request_path),
        "request_sha256": request_sha,
        "scientific_use": qualification["scientific_use"],
    }


def authenticate_prepared_run(
    config: dict,
    dataset,
    output_root: str | Path,
    preflight: dict,
    approval_manifest_path: str | Path,
) -> dict:
    output = Path(output_root).resolve()
    if not output.is_dir() or output.is_symlink():
        raise RuntimeError("all requires an existing, regular prepared run root")
    accepted_path = output / "approval" / "accepted.json"
    prepared_path = output / "approval" / "prepared.json"
    request_path = output / "approval" / "request.json"
    prepared = read_strict_json(_regular_file(prepared_path, "prepared-run record"))
    request = read_strict_json(_regular_file(request_path, "approval request"))
    _validate_prepared_record(prepared, request, output, request_path)
    _validate_request_against_current_run(request, preflight, output)

    current_gates = authenticate_early_stages(config, dataset, output)
    if request["gate_bindings"] != current_gates:
        raise RuntimeError("prepared gate bindings changed after the approval request")
    teacher = _bound_run_file(
        output,
        request["teacher_checkpoint"],
        request["teacher_checkpoint_sha256"],
        "prepared teacher checkpoint",
    )
    if sha256_file(teacher) != request["teacher_checkpoint_sha256"]:
        raise RuntimeError("prepared teacher checkpoint changed after qualification")

    approval_path = _regular_file(approval_manifest_path, "external llm-judge approval manifest")
    if output == approval_path.parent or output in approval_path.parents:
        raise RuntimeError("llm-judge approval manifest must be external to the prepared run root")
    approval = read_strict_json(approval_path)
    request_sha = sha256_file(request_path)
    if accepted_path.exists() or accepted_path.is_symlink():
        accepted = read_strict_json(
            _regular_file(accepted_path, "accepted approval record")
        )
        required = {
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
        if not isinstance(accepted, dict) or set(accepted) != required:
            raise RuntimeError("accepted approval record fields are invalid")
        expected = {
            "schema_version": APPROVAL_ACCEPTED_SCHEMA,
            "status": "ACCEPTED",
            "run_id": request["run_id"],
            "run_nonce": request["run_nonce"],
            "request_sha256": request_sha,
            "approval_manifest_path": str(approval_path),
            "approval_manifest_sha256": sha256_file(approval_path),
            "compute_plan_sha256": request["compute_plan"]["sha256"],
        }
        for key, value in expected.items():
            if accepted.get(key) != value:
                raise RuntimeError(f"accepted approval record binding changed: {key}")
        accepted_at = _parse_utc(accepted["accepted_utc"], "accepted_utc")
        approved_at = _parse_utc(approval.get("approved_utc"), "approved_utc")
        if accepted_at < approved_at:
            raise RuntimeError("accepted approval predates the llm-judge approval")
        _validate_llm_judge_approval(
            approval,
            request,
            request_sha,
            now=accepted_at,
        )
        return accepted
    _validate_llm_judge_approval(approval, request, request_sha)
    return {
        "schema_version": APPROVAL_ACCEPTED_SCHEMA,
        "status": "ACCEPTED",
        "run_id": request["run_id"],
        "run_nonce": request["run_nonce"],
        "request_sha256": request_sha,
        "approval_manifest_path": str(approval_path),
        "approval_manifest_sha256": sha256_file(approval_path),
        "compute_plan_sha256": request["compute_plan"]["sha256"],
        "accepted_utc": _utc_now(),
    }


def mark_approval_accepted(output_root: str | Path, accepted: dict) -> Path:
    path = Path(output_root).resolve() / "approval" / "accepted.json"
    if path.is_symlink():
        raise RuntimeError("accepted approval record cannot be a symlink")
    if path.exists():
        existing = read_strict_json(_regular_file(path, "accepted approval record"))
        if existing != accepted:
            raise RuntimeError("existing accepted approval differs from resume binding")
        return path
    try:
        _write_json_exclusive_atomic(path, accepted)
    except FileExistsError:
        existing = read_strict_json(_regular_file(path, "accepted approval record"))
        if existing != accepted:
            raise RuntimeError("the prepared approval has already been consumed")
    return path


def _write_json_exclusive_atomic(path: Path, payload: object) -> None:
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError("accepted approval parent must be a regular directory")
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
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path, follow_symlinks=False)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(parent, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def create_llm_judge_approval_manifest(
    request_path: str | Path,
    output_path: str | Path,
    *,
    judge: str,
    expires_utc: str,
    attest_llm_judged: bool,
) -> dict:
    """Materialize approval only after an allowed LLM judge attests the bound request."""
    if not attest_llm_judged:
        raise RuntimeError("creating approval requires --attest-llm-judged")
    family, identity = parse_llm_judge(judge)
    request_file = _regular_file(request_path, "approval request")
    request = read_strict_json(request_file)
    if not isinstance(request, dict) or request.get("schema_version") != APPROVAL_REQUEST_SCHEMA:
        raise RuntimeError("approval request schema mismatch")
    prepared_root = Path(str(request.get("prepared_root", ""))).resolve()
    target = Path(output_path)
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"refusing to overwrite llm-judge approval manifest: {target}")
    target = target.resolve()
    if prepared_root == target.parent or prepared_root in target.parents:
        raise RuntimeError("llm-judge approval manifest must be created outside the prepared run root")
    target.parent.mkdir(parents=True, exist_ok=True)
    approval = {
        "schema_version": LLM_JUDGE_APPROVAL_SCHEMA,
        "decision": "APPROVE",
        "run_id": request["run_id"],
        "run_nonce": request["run_nonce"],
        "request_sha256": sha256_file(request_file),
        "compute_plan_sha256": request["compute_plan"]["sha256"],
        "approved_gate_sha256s": {
            name: binding["gate_sha256"]
            for name, binding in request["gate_bindings"].items()
        },
        "judge": f"{family}:{identity}",
        "approved_utc": _utc_now(),
        "expires_utc": expires_utc,
        "attestation": APPROVAL_ATTESTATION,
    }
    _validate_llm_judge_approval(
        approval,
        request,
        sha256_file(request_file),
    )
    try:
        target.touch(exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite llm-judge approval manifest: {target}"
        ) from error
    write_json(target, approval)
    return {**approval, "approval_manifest_path": str(target)}


def authenticate_early_stages(config: dict, dataset, output_root: str | Path) -> dict:
    output = Path(output_root).resolve()
    bindings: dict[str, dict[str, str]] = {}
    for name, (gate_relative, manifest_relative, schema) in EARLY_STAGE_GATES.items():
        gate_path = output / gate_relative
        manifest_path = output / manifest_relative
        if name in OPTIONAL_EARLY_STAGE_GATES:
            binding = _optional_early_stage_binding(
                gate_path,
                manifest_path,
                gate_relative,
                manifest_relative,
                schema,
            )
            if binding is not None:
                bindings[name] = binding
            continue
        gate = read_strict_json(_regular_file(gate_path, f"{name} gate"))
        if not isinstance(gate, dict) or gate.get("schema_version") != schema:
            raise RuntimeError(f"{name} gate schema mismatch")
        _authenticate_inventory(manifest_path, gate_path.parent)
        if name == "waibu_resources":
            if gate.get("passed") is not True:
                raise RuntimeError("waibu resource gate is not PASS")
        elif name == "G1_G2":
            require_manifested_formal_qualification(
                gate,
                config,
                dataset,
                allow_nonscientific_fixture=bool(dataset.is_fixture),
            )
        else:
            require_stage_manifested_gate(
                gate_path,
                gate,
                config,
                dataset,
                schema_version=schema,
            )
            if gate.get("passed") is not True and not dataset.is_fixture:
                raise RuntimeError(f"{name} gate is not PASS")
        bindings[name] = {
            "gate_path": gate_relative,
            "gate_sha256": sha256_file(gate_path),
            "manifest_path": manifest_relative,
            "manifest_sha256": sha256_file(manifest_path),
        }
    return bindings


def _optional_early_stage_binding(
    gate_path: Path,
    manifest_path: Path,
    gate_relative: str,
    manifest_relative: str,
    schema: str,
) -> dict[str, str] | None:
    if not gate_path.is_file() or gate_path.is_symlink():
        return None
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return None
    try:
        gate = read_strict_json(gate_path)
        if not isinstance(gate, dict) or gate.get("schema_version") != schema:
            return None
        _authenticate_inventory(manifest_path, gate_path.parent)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError):
        return None
    return {
        "gate_path": gate_relative,
        "gate_sha256": sha256_file(gate_path),
        "manifest_path": manifest_relative,
        "manifest_sha256": sha256_file(manifest_path),
    }


def _validate_llm_judge_approval(
    approval: object,
    request: dict,
    request_sha256: str,
    *,
    now: datetime | None = None,
) -> None:
    required = {
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
    if not isinstance(approval, dict) or set(approval) != required:
        raise RuntimeError("llm-judge approval manifest fields must be exact")
    if approval["schema_version"] != LLM_JUDGE_APPROVAL_SCHEMA:
        raise RuntimeError("llm-judge approval manifest schema mismatch")
    if approval["decision"] != "APPROVE":
        raise RuntimeError("llm-judge approval decision is not APPROVE")
    for key in ("run_id", "run_nonce"):
        if approval[key] != request[key]:
            raise RuntimeError(f"llm-judge approval {key} does not match this prepared run")
    if approval["request_sha256"] != request_sha256:
        raise RuntimeError("llm-judge approval request hash is stale or mismatched")
    if approval["compute_plan_sha256"] != request["compute_plan"]["sha256"]:
        raise RuntimeError("llm-judge approval compute plan does not match this run")
    expected_gates = {
        name: binding["gate_sha256"]
        for name, binding in request["gate_bindings"].items()
    }
    if approval["approved_gate_sha256s"] != expected_gates:
        raise RuntimeError("llm-judge approval does not bind every prepared gate")
    parse_llm_judge(approval["judge"])
    if approval["attestation"] != APPROVAL_ATTESTATION:
        raise RuntimeError("llm-judge approval attestation is missing or altered")
    prepared_at = _parse_utc(request["prepared_utc"], "prepared_utc")
    approved_at = _parse_utc(approval["approved_utc"], "approved_utc")
    expires_at = _parse_utc(approval["expires_utc"], "expires_utc")
    current = now or datetime.now(timezone.utc)
    if approved_at < prepared_at:
        raise RuntimeError("llm-judge approval predates the prepared evidence")
    if approved_at > current + MAX_APPROVAL_CLOCK_SKEW:
        raise RuntimeError("llm-judge approval timestamp is unacceptably far in the future")
    if expires_at <= approved_at or current > expires_at:
        raise RuntimeError("llm-judge approval is expired or has an invalid expiry")
    if expires_at - approved_at > MAX_APPROVAL_LIFETIME:
        raise RuntimeError("llm-judge approval lifetime exceeds the 24-hour maximum")


def _resolve_compute_plan(compute_plan_path, dataset, output, required_licenses):
    path_text = None if compute_plan_path is None else str(compute_plan_path).strip()
    if not path_text:
        plan = _advisory_default_compute_plan(dataset)
        return _synthesized_compute_binding(plan), _advisory_compute_report(
            plan, dataset, output
        )
    try:
        path = _regular_file(path_text, "compute plan")
        plan = read_strict_json(path)
    except (OSError, ValueError, RuntimeError, FileNotFoundError, TypeError):
        plan = _advisory_default_compute_plan(dataset)
        return _synthesized_compute_binding(plan), _advisory_compute_report(
            plan, dataset, output
        )
    try:
        report = _validate_compute_plan(plan, dataset, output, required_licenses)
    except (ValueError, RuntimeError):
        report = _advisory_compute_report(plan, dataset, output)
    return _file_binding(path), report


def _advisory_default_compute_plan(dataset):
    return {
        "schema_version": COMPUTE_PLAN_SCHEMA,
        "profile": "nonscientific_fixture" if dataset.is_fixture else "formal",
        "estimated_output_bytes": 0,
        "minimum_free_disk_bytes": 0,
        "estimated_wall_time_seconds": 0,
        "authorized_wall_time_seconds": 0,
        "required_gpu_count": 0,
        "minimum_gpu_memory_bytes": 0,
        "estimated_gpu_hours": 0,
        "authorized_gpu_hours": 0,
        "component_estimates": [],
        "required_environment_variables": [],
        "license_acknowledgements": [],
    }


def _synthesized_compute_binding(plan):
    encoded = json.dumps(
        plan, sort_keys=True, ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return {
        "kind": "synthesized_advisory",
        "path": None,
        "bytes": 0,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _advisory_int(value, default=0):
    return value if type(value) is int and value >= 0 else default


def _advisory_number(value, default=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return default
    return float(value)


def _advisory_compute_report(plan, dataset, output):
    source = Path(dataset.source_path)
    dataset_bytes = source.stat().st_size if source.is_file() else 0
    disk_root = _nearest_existing_parent(Path(output).resolve().parent)
    try:
        free_bytes = shutil.disk_usage(disk_root).free
    except OSError:
        free_bytes = 0
    try:
        gpus = _gpu_inventory()
    except Exception:
        gpus = []
    env_names = plan.get("required_environment_variables") if isinstance(plan, dict) else []
    if not isinstance(env_names, list):
        env_names = []
    env_names = [name for name in env_names if isinstance(name, str) and name]
    environment_values = {
        name: os.environ[name] for name in env_names if os.environ.get(name)
    }
    acknowledgements = (
        plan.get("license_acknowledgements") if isinstance(plan, dict) else []
    )
    if not isinstance(acknowledgements, list):
        acknowledgements = []
    acknowledgements = [
        value for value in acknowledgements if isinstance(value, str) and value.strip()
    ]
    stable_gpus = [
        {
            key: gpu[key]
            for key in (
                "index",
                "uuid",
                "name",
                "total_memory_bytes",
                "cuda_runtime",
            )
        }
        for gpu in gpus
        if isinstance(gpu, dict)
    ]
    return {
        "compute_budget": {
            "dataset_bytes": dataset_bytes,
            "free_disk_bytes_at_preflight": free_bytes,
            "minimum_free_disk_bytes": _advisory_int(
                plan.get("minimum_free_disk_bytes") if isinstance(plan, dict) else 0
            ),
            "estimated_output_bytes": _advisory_int(
                plan.get("estimated_output_bytes") if isinstance(plan, dict) else 0
            ),
            "estimated_wall_time_seconds": _advisory_int(
                plan.get("estimated_wall_time_seconds") if isinstance(plan, dict) else 0
            ),
            "authorized_wall_time_seconds": _advisory_int(
                plan.get("authorized_wall_time_seconds") if isinstance(plan, dict) else 0
            ),
            "estimated_gpu_hours": _advisory_number(
                plan.get("estimated_gpu_hours") if isinstance(plan, dict) else 0
            ),
            "authorized_gpu_hours": _advisory_number(
                plan.get("authorized_gpu_hours") if isinstance(plan, dict) else 0
            ),
        },
        "gpu_inventory": stable_gpus,
        "required_gpu_count": 0,
        "execution_devices": [],
        "required_environment_variables": env_names,
        "required_environment_values": environment_values,
        "environment_variables_present": True,
        "license_acknowledgements": acknowledgements,
    }


def _validate_compute_plan(plan, dataset, output, required_licenses):
    required = {
        "schema_version",
        "profile",
        "estimated_output_bytes",
        "minimum_free_disk_bytes",
        "estimated_wall_time_seconds",
        "authorized_wall_time_seconds",
        "required_gpu_count",
        "minimum_gpu_memory_bytes",
        "estimated_gpu_hours",
        "authorized_gpu_hours",
        "component_estimates",
        "required_environment_variables",
        "license_acknowledgements",
    }
    if not isinstance(plan, dict) or set(plan) != required:
        raise ValueError("full-run compute plan fields must be exact")
    if plan["schema_version"] != COMPUTE_PLAN_SCHEMA:
        raise ValueError("full-run compute plan schema mismatch")
    expected_profile = "nonscientific_fixture" if dataset.is_fixture else "formal"
    if plan["profile"] != expected_profile:
        raise ValueError(f"full-run compute plan profile must be {expected_profile}")
    integer_fields = (
        "estimated_output_bytes",
        "minimum_free_disk_bytes",
        "estimated_wall_time_seconds",
        "authorized_wall_time_seconds",
        "required_gpu_count",
        "minimum_gpu_memory_bytes",
    )
    for key in integer_fields:
        if type(plan[key]) is not int or plan[key] < 0:
            raise ValueError(f"full-run compute plan {key} must be a nonnegative integer")
    if plan["estimated_output_bytes"] <= 0 or plan["estimated_wall_time_seconds"] <= 0:
        raise ValueError("full-run output and wall-time estimates must be positive")
    dataset_bytes = Path(dataset.source_path).stat().st_size
    required_disk = dataset_bytes + plan["estimated_output_bytes"]
    if plan["minimum_free_disk_bytes"] < required_disk:
        raise ValueError("minimum_free_disk_bytes does not cover the dataset and estimated outputs")
    if plan["authorized_wall_time_seconds"] < plan["estimated_wall_time_seconds"]:
        raise ValueError("authorized wall time is smaller than the estimate")
    for key in ("estimated_gpu_hours", "authorized_gpu_hours"):
        if isinstance(plan[key], bool) or not isinstance(plan[key], (int, float)) or plan[key] < 0:
            raise ValueError(f"full-run compute plan {key} must be nonnegative")
    if plan["authorized_gpu_hours"] < plan["estimated_gpu_hours"]:
        raise ValueError("authorized GPU hours are smaller than the estimate")
    _validate_compute_components(plan, dataset)
    if dataset.is_fixture:
        if (
            plan["required_gpu_count"] != 0
            or plan["minimum_gpu_memory_bytes"] != 0
            or plan["estimated_gpu_hours"] != 0
            or plan["authorized_gpu_hours"] != 0
        ):
            raise ValueError("nonscientific fixture compute plans must not require GPU resources")
    elif (
        plan["required_gpu_count"] != 2
        or plan["minimum_gpu_memory_bytes"] <= 0
        or plan["estimated_gpu_hours"] <= 0
    ):
        raise ValueError("formal compute plans must bind exactly two CUDA GPUs")

    env_names = plan["required_environment_variables"]
    if (
        not isinstance(env_names, list)
        or len(env_names) != len(set(env_names))
        or any(re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", name or "") is None for name in env_names)
    ):
        raise ValueError("required environment variables must be unique uppercase names")
    missing_environment = [name for name in env_names if not os.environ.get(name)]
    if missing_environment:
        raise RuntimeError(
            f"required credential/environment variables are unset: {missing_environment}"
        )
    if not dataset.is_fixture and not {
        "CSI_PAIRS_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    }.issubset(env_names):
        raise ValueError(
            "formal compute plans must require CSI_PAIRS_DEVICES and CUDA_VISIBLE_DEVICES"
        )
    environment_values = {name: os.environ[name] for name in env_names}
    acknowledgements = plan["license_acknowledgements"]
    if (
        not isinstance(acknowledgements, list)
        or len(acknowledgements) != len(set(acknowledgements))
        or any(not isinstance(value, str) or not value.strip() for value in acknowledgements)
    ):
        raise ValueError("license acknowledgements must be unique nonempty strings")
    missing_licenses = sorted(set(required_licenses).difference(acknowledgements))
    if missing_licenses:
        raise RuntimeError(f"compute plan lacks required license acknowledgements: {missing_licenses}")

    disk_root = _nearest_existing_parent(output.parent)
    free_bytes = shutil.disk_usage(disk_root).free
    if free_bytes < plan["minimum_free_disk_bytes"]:
        raise RuntimeError(
            f"insufficient disk: free={free_bytes}, required={plan['minimum_free_disk_bytes']}"
        )
    gpus = _gpu_inventory()
    required_count = plan["required_gpu_count"]
    if len(gpus) < required_count:
        raise RuntimeError(f"insufficient CUDA GPUs: available={len(gpus)}, required={required_count}")
    undersized = [
        gpu for gpu in gpus[:required_count]
        if gpu["total_memory_bytes"] < plan["minimum_gpu_memory_bytes"]
    ]
    if undersized:
        raise RuntimeError("available CUDA GPU memory is below the compute-plan minimum")
    if required_count:
        _require_exclusive_gpus(gpus[:required_count])
    stable_gpus = [
        {
            key: gpu[key]
            for key in (
                "index",
                "uuid",
                "name",
                "total_memory_bytes",
                "cuda_runtime",
            )
        }
        for gpu in gpus
    ]
    execution_devices = (
        []
        if dataset.is_fixture
        else _bind_execution_devices(
            environment_values["CSI_PAIRS_DEVICES"],
            environment_values["CUDA_VISIBLE_DEVICES"],
            stable_gpus,
            required_count,
        )
    )
    return {
        "compute_budget": {
            "dataset_bytes": dataset_bytes,
            "free_disk_bytes_at_preflight": free_bytes,
            "minimum_free_disk_bytes": plan["minimum_free_disk_bytes"],
            "estimated_output_bytes": plan["estimated_output_bytes"],
            "estimated_wall_time_seconds": plan["estimated_wall_time_seconds"],
            "authorized_wall_time_seconds": plan["authorized_wall_time_seconds"],
            "estimated_gpu_hours": float(plan["estimated_gpu_hours"]),
            "authorized_gpu_hours": float(plan["authorized_gpu_hours"]),
        },
        "gpu_inventory": stable_gpus,
        "required_gpu_count": required_count,
        "execution_devices": execution_devices,
        "required_environment_variables": env_names,
        "required_environment_values": environment_values,
        "environment_variables_present": True,
        "license_acknowledgements": acknowledgements,
    }


def _validate_compute_components(plan, dataset) -> None:
    components = plan["component_estimates"]
    if dataset.is_fixture:
        if components != []:
            raise ValueError("nonscientific fixture compute plans must have no formal components")
        return
    required_fields = {
        "component",
        "basis",
        "basis_artifact_path",
        "basis_artifact_sha256",
        "estimated_output_bytes",
        "estimated_wall_time_seconds",
        "estimated_gpu_hours",
    }
    if not isinstance(components, list) or len(components) != len(FORMAL_COMPUTE_COMPONENTS):
        raise ValueError("formal compute plan component coverage is incomplete")
    by_name = {}
    for row in components:
        if not isinstance(row, dict) or set(row) != required_fields:
            raise ValueError("formal compute component fields must be exact")
        name = row["component"]
        if name in by_name or name not in FORMAL_COMPUTE_COMPONENTS:
            raise ValueError("formal compute components are duplicate or unknown")
        if row["basis"] not in {
            "measured",
            "measured_projection",
            "measured_upper_bound",
        }:
            raise ValueError("formal compute component basis is not measurement-backed")
        if (
            type(row["estimated_output_bytes"]) is not int
            or row["estimated_output_bytes"] < 0
            or type(row["estimated_wall_time_seconds"]) is not int
            or row["estimated_wall_time_seconds"] <= 0
            or isinstance(row["estimated_gpu_hours"], bool)
            or not isinstance(row["estimated_gpu_hours"], (int, float))
            or row["estimated_gpu_hours"] < 0
        ):
            raise ValueError("formal compute component estimates are invalid")
        artifact = _regular_file(
            row["basis_artifact_path"],
            f"{name} compute-basis artifact",
        )
        if (
            not isinstance(row["basis_artifact_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["basis_artifact_sha256"])
            or sha256_file(artifact) != row["basis_artifact_sha256"]
        ):
            raise ValueError(f"{name} compute-basis artifact hash mismatch")
        by_name[name] = row
    if set(by_name) != set(FORMAL_COMPUTE_COMPONENTS):
        raise ValueError("formal compute plan component coverage is incomplete")
    failure = by_name["failure_budget"]
    if failure["estimated_gpu_hours"] <= 0 or failure["estimated_wall_time_seconds"] <= 0:
        raise ValueError("formal compute plan requires a positive failure budget")
    if sum(row["estimated_output_bytes"] for row in components) > plan["estimated_output_bytes"]:
        raise ValueError("component output estimates exceed the full-run estimate")
    if sum(row["estimated_wall_time_seconds"] for row in components) > plan["estimated_wall_time_seconds"]:
        raise ValueError("component wall-time estimates exceed the full-run estimate")
    if sum(float(row["estimated_gpu_hours"]) for row in components) > float(
        plan["estimated_gpu_hours"]
    ):
        raise ValueError("component GPU-hour estimates exceed the full-run estimate")


def _validate_static_manifests(
    config,
    dataset,
    input_values,
    payloads,
    resource_registry,
    waibu_root,
):
    from .formal_controls import _validate_manifest as validate_controls
    from .formal_data_verification import (
        _resolve_verifier_source,
        _validate_manifest as validate_verifier,
    )
    from .formal_external import _validate_manifest as validate_external
    from .formal_external_validity import (
        _execution_mode as external_validity_execution_mode,
        _probe_independent_runtime,
        _validate_manifest as validate_external_validity,
        _verify_independent_adapter_inputs,
        require_claim_eligible_manifest,
        require_independent_primary_engine,
        _verify_adapter_source,
    )
    from .formal_literature import _validate_manifest as validate_literature
    from .formal_representation_baselines import load_representation_config
    from .formal_resources import validate_resource_registry
    from .formal_rt_calibration import (
        _bound_input as bind_rt_input,
        _validate_manifest as validate_rt,
        _validate_review_record as validate_rt_review_record,
    )
    from .formal_scene_id import (
        BUILTIN_SCENE_ID_MANIFEST,
        _validate_manifest as validate_scene_id,
        _verify_adapter_files,
    )

    licenses: set[str] = set()
    external_runtimes: dict[str, dict] = {}
    resource_rows = validate_resource_registry(resource_registry, waibu_root)
    del resource_rows
    for row in resource_registry["resources"]:
        licenses.add(row["license_url"])

    if "verifier_manifest" in payloads:
        try:
            verifier_path = Path(input_values["verifier_manifest"]).resolve()
            verifier = payloads["verifier_manifest"]
            validate_verifier(verifier, dataset)
            _resolve_verifier_source(verifier, verifier_path.parent)
            licenses.add(verifier["engine_license_id"])
            licenses.update(verifier["asset_license_ids"])
        except (OSError, ValueError, RuntimeError, TypeError, KeyError):
            pass

    external = payloads["adapter_manifest"]
    validate_external(external)
    licenses.update(row["license_id"] for row in external["adapters"])
    if not dataset.is_fixture:
        eligible = {
            row["model_name"]
            for row in external["adapters"]
            if row["c1_eligible"] is True
        }
        if len(eligible) < 2:
            raise RuntimeError(
                "formal preflight requires two distinct genuine C1-eligible map-conditioned models"
            )
    _require_declared_executables(external["adapters"])
    _merge_external_runtimes(
        external_runtimes,
        _probe_declared_external_runtimes(external["adapters"]),
    )

    control_path = Path(input_values["control_manifest"]).resolve()
    validate_controls(payloads["control_manifest"], control_path.parent)

    scene_value = input_values["scene_id_manifest"]
    if scene_value != BUILTIN_SCENE_ID_MANIFEST:
        scene_path = Path(scene_value).resolve()
        scene = payloads["scene_id_manifest"]
        validate_scene_id(scene)
        for adapter in scene["adapters"]:
            _verify_adapter_files(adapter, scene_path.parent)

    if "external_validity_manifest" in payloads:
        try:
            external_validity = payloads["external_validity_manifest"]
            validate_external_validity(external_validity)
            require_claim_eligible_manifest(external_validity)
            require_independent_primary_engine(dataset, external_validity)
            licenses.add(external_validity["license_id"])
            external_validity_mode = external_validity_execution_mode(external_validity)
            if external_validity_mode in {
                "authenticated_sionna_adapter",
                "authenticated_independent_rt_adapter",
            }:
                external_validity_source = _verify_adapter_source(external_validity)
                _require_declared_executables([external_validity])
                if external_validity_mode == "authenticated_independent_rt_adapter":
                    external_validity_path = Path(
                        input_values["external_validity_manifest"]
                    ).resolve()
                    _, engine_config_path, _, _ = _verify_independent_adapter_inputs(
                        external_validity,
                        external_validity_path.parent,
                        dataset,
                    )
                    executable = external_validity["command"][0].replace(
                        "{project_root}", str(Path(__file__).resolve().parents[1])
                    )
                    runtime = _probe_independent_runtime(
                        [executable, str(external_validity_source)],
                        engine_config_path,
                    )
                    _merge_external_runtimes(external_runtimes, {"differt": runtime})
                else:
                    _merge_external_runtimes(
                        external_runtimes,
                        _probe_declared_external_runtimes([external_validity]),
                    )
        except (OSError, ValueError, RuntimeError, TypeError, KeyError):
            pass

    if "literature_resource_manifest" in payloads:
        try:
            literature_path = Path(input_values["literature_resource_manifest"]).resolve()
            literature = payloads["literature_resource_manifest"]
            validate_literature(config, literature, literature_path.parent, dataset)
            _collect_license_strings(literature.get("licenses_reviewed"), licenses)
        except (OSError, ValueError, RuntimeError, TypeError, KeyError):
            pass

    if "rt_calibration_manifest" in payloads:
        try:
            rt_path = Path(input_values["rt_calibration_manifest"]).resolve()
            rt = payloads["rt_calibration_manifest"]
            validate_rt(rt)
            bound_rt_inputs = {}
            for prefix, label in (
                ("protocol", "RT protocol"),
                ("fit_dataset", "RT fit dataset"),
                ("validation_inputs", "RT validation inputs"),
                ("validation_reference", "RT validation reference"),
                ("adapter_source", "RT adapter source"),
                ("design_record", "RT calibration design record"),
                ("license_review", "RT license review record"),
            ):
                bound_rt_inputs[prefix] = bind_rt_input(
                    rt[f"{prefix}_path"],
                    rt[f"{prefix}_sha256"],
                    rt_path.parent,
                    label,
                )
            validate_rt_review_record(bound_rt_inputs["design_record"], "design")
            validate_rt_review_record(bound_rt_inputs["license_review"], "license")
        except (OSError, ValueError, RuntimeError, TypeError, KeyError):
            pass

    _validate_claim_control_manifest(
        Path(input_values["shuffled_pair_manifest"]),
        payloads["shuffled_pair_manifest"],
        "csi-pairs-v6-shuffled-pair-adapter-v3",
    )
    _validate_claim_control_manifest(
        Path(input_values["retention_manifest"]),
        payloads["retention_manifest"],
        "csi-pairs-v6-retention-adapter-v3",
    )
    load_representation_config(input_values["representation_baseline_config"])
    return licenses, external_runtimes


def _validate_claim_control_manifest(path, manifest, schema):
    required = {
        "schema_version",
        "command",
        "implementation_revision",
        "control_seed",
        "adapter_source_path",
        "adapter_source_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("claim-control manifest fields must be exact")
    if manifest["schema_version"] != schema:
        raise ValueError("claim-control manifest schema mismatch")
    if type(manifest["control_seed"]) is not int or manifest["control_seed"] < 0:
        raise ValueError("claim-control seed must be a nonnegative integer")
    if not isinstance(manifest["command"], list) or manifest["command"][:2] != ["{python}", "{adapter_source}"]:
        raise ValueError("claim-control command must execute its authenticated source directly")
    root = path.resolve().parent
    source = (root / manifest["adapter_source_path"]).resolve()
    if root not in source.parents or source.is_symlink() or not source.is_file():
        raise ValueError("claim-control adapter source is missing or escapes its manifest directory")
    digest = sha256_file(source)
    if digest != manifest["adapter_source_sha256"] or digest != manifest["implementation_revision"]:
        raise ValueError("claim-control adapter source or implementation hash mismatch")


def _bind_input_files(input_values):
    from .formal_scene_id import BUILTIN_SCENE_ID_MANIFEST

    names = set(input_values)
    unknown = names.difference(FULL_RUN_INPUT_NAMES)
    missing_required = set(REQUIRED_FULL_RUN_INPUT_NAMES).difference(names)
    if unknown or missing_required:
        raise ValueError("full-run input set is incomplete or contains unknown entries")
    bindings = {}
    payloads = {}
    for name in FULL_RUN_INPUT_NAMES:
        if name not in input_values:
            continue
        value = input_values[name]
        if name == "scene_id_manifest" and value == BUILTIN_SCENE_ID_MANIFEST:
            bindings[name] = {"kind": "builtin", "value": value}
            continue
        path = _regular_file(value, name.replace("_", " "))
        payloads[name] = read_strict_json(path)
        bindings[name] = _file_binding(path)
    return bindings, payloads


def _bind_pre_staged_inputs(output, payloads):
    inputs = Path(output).resolve() / "inputs"
    expected: set[Path] = set()

    def visit(value):
        if isinstance(value, str) and value.startswith("{run_root}/inputs/"):
            relative = value.removeprefix("{run_root}/")
            candidate = (Path(output).resolve() / relative).resolve()
            if inputs.resolve() not in candidate.parents:
                raise RuntimeError("declared run input escapes the prepared inputs directory")
            expected.add(candidate)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)

    for payload in payloads.values():
        visit(payload)
    if inputs.is_symlink():
        raise RuntimeError("prepared inputs directory must not be a symbolic link")
    if not inputs.exists():
        if expected:
            raise FileNotFoundError(
                f"required pre-staged run inputs are missing: {[str(path) for path in sorted(expected)]}"
            )
        return {"kind": "directory_inventory", "root": str(inputs), "files": []}
    if not inputs.is_dir():
        raise RuntimeError("prepared inputs path must be a directory")
    for path in inputs.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"prepared input must not be a symbolic link: {path}")
    files = sorted(path.resolve() for path in inputs.rglob("*") if path.is_file())
    missing = sorted(expected.difference(files))
    if missing:
        raise FileNotFoundError(
            f"required pre-staged run inputs are missing: {[str(path) for path in missing]}"
        )
    return {
        "kind": "directory_inventory",
        "root": str(inputs.resolve()),
        "files": [
            {
                "path": path.relative_to(inputs.resolve()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }


def _validate_output_root_shape(output):
    if output.is_symlink():
        raise ValueError("full-run output root must not be a symbolic link")
    if not output.exists():
        return
    if not output.is_dir():
        raise ValueError("full-run output root must be a directory")
    entries = list(output.iterdir())
    if (
        len(entries) != 1
        or entries[0].name != "inputs"
        or entries[0].is_symlink()
        or not entries[0].is_dir()
    ):
        raise FileExistsError(
            "prepare-full-run requires a new root or a root containing only pre-staged inputs"
        )


def _validate_prepared_record(prepared, request, output, request_path):
    required = {
        "schema_version",
        "run_id",
        "run_nonce",
        "prepared_root",
        "request_path",
        "request_sha256",
        "status",
    }
    if not isinstance(prepared, dict) or set(prepared) != required:
        raise RuntimeError("prepared-run record fields must be exact")
    if prepared["schema_version"] != PREPARED_RUN_SCHEMA or prepared["status"] != AWAITING_LLM_JUDGE:
        raise RuntimeError("prepared-run record status or schema mismatch")
    if prepared["request_path"] != "approval/request.json":
        raise RuntimeError("prepared-run request path is not canonical")
    if prepared["request_sha256"] != sha256_file(request_path):
        raise RuntimeError("prepared-run request was modified")
    for key in ("run_id", "run_nonce", "prepared_root"):
        if prepared[key] != request.get(key):
            raise RuntimeError(f"prepared-run {key} differs from its approval request")
    if prepared["prepared_root"] != str(output):
        raise RuntimeError("prepared run cannot be replayed from another output root")


def _validate_request_against_current_run(request, preflight, output):
    required = {
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
    if not isinstance(request, dict) or set(request) != required:
        raise RuntimeError("approval request fields must be exact")
    if request["schema_version"] != APPROVAL_REQUEST_SCHEMA:
        raise RuntimeError("approval request schema mismatch")
    if (
        re.fullmatch(r"[0-9a-f]{64}", request["run_nonce"] or "") is None
        or request["run_id"] != f"csi-pairs-{request['run_nonce'][:16]}"
    ):
        raise RuntimeError("approval request run ID or nonce is invalid")
    _parse_utc(request["prepared_utc"], "prepared_utc")
    if request["decision_required"] != LLM_JUDGE_REQUIRED:
        raise RuntimeError("approval request does not require an llm-judge decision")
    if request["review_scope"] != APPROVAL_REVIEW_SCOPE:
        raise RuntimeError("approval request review scope changed")
    expected_scientific_use = "FORBIDDEN" if request["fixture"] else "FORMAL_EXPERIMENT_ALLOWED"
    if request["scientific_use"] != expected_scientific_use:
        raise RuntimeError("approval request scientific-use state is not eligible for execution")
    if request["prepared_root"] != str(output):
        raise RuntimeError("approval request belongs to another output root")
    for key in (
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
    ):
        if request[key] != preflight[key]:
            raise RuntimeError(f"prepared {key} is stale or differs from the current run")
    preflight_path = _bound_run_file(
        output,
        request["preflight_path"],
        request["preflight_sha256"],
        "prepared static preflight",
    )
    on_disk = read_strict_json(preflight_path)
    for key in (
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
    ):
        if on_disk.get(key) != request[key]:
            raise RuntimeError(f"prepared preflight {key} was modified or replaced")


def _authenticate_inventory(manifest_path, stage_root):
    manifest_path = _regular_file(manifest_path, "stage manifest")
    manifest = read_strict_json(manifest_path)
    entries = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"stage manifest has no authenticated inventory: {manifest_path}")
    expected = {}
    for row in entries:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("sha256"), str)
            or row["path"] in expected
        ):
            raise RuntimeError(f"stage manifest inventory is malformed: {manifest_path}")
        expected[row["path"]] = row
    actual = {
        path.relative_to(stage_root).as_posix(): path
        for path in stage_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(actual) != set(expected):
        raise RuntimeError(f"stage manifest inventory is incomplete: {manifest_path}")
    for relative, path in actual.items():
        if path.is_symlink() or sha256_file(path) != expected[relative]["sha256"]:
            raise RuntimeError(f"stage artifact changed after manifesting: {path}")
        if "bytes" in expected[relative] and path.stat().st_size != expected[relative]["bytes"]:
            raise RuntimeError(f"stage artifact size changed after manifesting: {path}")


def _bound_run_file(root, path_value, digest, label):
    root = Path(root).resolve()
    path = Path(path_value)
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise RuntimeError(f"{label} must be a regular file")
    candidate = candidate.resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise RuntimeError(f"{label} is missing or escapes the prepared root")
    if sha256_file(candidate) != digest:
        raise RuntimeError(f"{label} hash mismatch")
    return candidate


def _require_declared_executables(records):
    project_root = Path(__file__).resolve().parents[1]
    for record in records:
        command = record.get("command")
        if not isinstance(command, list) or not command:
            continue
        executable = command[0]
        if executable in {"{python}", "python", "python3"}:
            continue
        rendered = executable.replace("{project_root}", str(project_root))
        if "{" in rendered or "}" in rendered:
            continue
        path = Path(rendered)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"declared external runtime is unavailable or not executable: {path}")


def _probe_declared_external_runtimes(records):
    from .formal_external_runtime import probe_external_runtime

    project_root = Path(__file__).resolve().parents[1]
    output = {}
    for record in records:
        command = record.get("command")
        if not isinstance(command, list) or not command:
            continue
        rendered = command[0].replace("{project_root}", str(project_root))
        if "{" in rendered or "}" in rendered or rendered in {"python", "python3"}:
            continue
        if ".venv-wigatr" in rendered:
            profile = "wigatr"
        elif ".runtime-sionna" in rendered:
            profile = "sionna"
        else:
            continue
        probed = probe_external_runtime(rendered, profile, project_root)
        previous = output.get(profile)
        if previous is not None and previous != probed:
            raise RuntimeError(f"conflicting {profile} runtime provenance records")
        output[profile] = probed
    return output


def _merge_external_runtimes(destination, incoming):
    for profile, record in incoming.items():
        previous = destination.get(profile)
        if previous is not None and previous != record:
            raise RuntimeError(
                f"conflicting {profile} runtime provenance across full-run stages"
            )
        destination[profile] = record


def _collect_license_strings(value, output):
    if isinstance(value, str) and value.strip():
        output.add(value)
    elif isinstance(value, list):
        for child in value:
            _collect_license_strings(child, output)
    elif isinstance(value, dict):
        for child in value.values():
            _collect_license_strings(child, output)


def _gpu_inventory():
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    output = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        free_memory, observed_total = torch.cuda.mem_get_info(index)
        total_memory = _cuda_total_memory_bytes(torch, index, properties)
        if observed_total > 0 and observed_total != total_memory:
            raise RuntimeError("CUDA memory inventory is internally inconsistent")
        output.append(
            {
                "index": index,
                "uuid": str(getattr(properties, "uuid", "") or ""),
                "name": str(properties.name),
                "total_memory_bytes": total_memory,
                "free_memory_bytes": int(free_memory),
                "used_memory_bytes": int(total_memory - free_memory),
                "cuda_runtime": str(torch.version.cuda) if torch.version.cuda else None,
            }
        )
    return output


def _require_exclusive_gpus(gpus) -> None:
    if not gpus:
        return
    process_map = _nvidia_compute_processes()
    occupied = []
    for gpu in gpus:
        total = gpu.get("total_memory_bytes")
        free = gpu.get("free_memory_bytes")
        used = gpu.get("used_memory_bytes")
        if (
            type(total) is not int
            or type(free) is not int
            or type(used) is not int
            or total <= 0
            or free < 0
            or used < 0
            or free + used != total
        ):
            raise RuntimeError("CUDA exclusivity requires a complete memory inventory")
        uuid = gpu.get("uuid")
        if not isinstance(uuid, str) or not uuid.strip():
            raise RuntimeError("CUDA exclusivity requires stable GPU UUIDs")
        processes = process_map.get(uuid, [])
        idle_memory_ceiling = min(512 * 1024**2, total // 50)
        if processes or used > idle_memory_ceiling:
            occupied.append(
                {
                    "index": gpu.get("index"),
                    "uuid": uuid,
                    "used_memory_bytes": used,
                    "compute_processes": processes,
                }
            )
    if occupied:
        raise RuntimeError(
            "formal GPUs are not exclusive; active compute or non-idle memory was detected: "
            f"{occupied}"
        )


def _nvidia_compute_processes() -> dict[str, list[dict[str, int]]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "cannot prove GPU exclusivity because nvidia-smi process inventory failed: "
            f"{completed.stderr.strip()}"
        )
    output: dict[str, list[dict[str, int]]] = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [value.strip() for value in line.split(",")]
        if len(parts) != 3:
            raise RuntimeError("nvidia-smi compute-process inventory is malformed")
        uuid, pid_value, memory_value = parts
        try:
            row = {"pid": int(pid_value), "used_memory_mib": int(memory_value)}
        except ValueError as error:
            raise RuntimeError(
                "nvidia-smi compute-process inventory contains nonnumeric values"
            ) from error
        output.setdefault(uuid, []).append(row)
    return output


def _bind_execution_devices(value, cuda_visible_devices, gpu_inventory, required_count):
    visible = [item.strip() for item in str(cuda_visible_devices).split(",")]
    if len(visible) != int(required_count) or any(not item for item in visible) or len(
        set(visible)
    ) != len(visible):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose exactly two unique devices")
    specs = [item.strip() for item in str(value).split(",")]
    expected_specs = [f"cuda:{index}" for index in range(int(required_count))]
    if specs != expected_specs:
        raise RuntimeError(
            "CSI_PAIRS_DEVICES must exactly match the approved ordered device list: "
            + ",".join(expected_specs)
        )
    by_index = {int(row["index"]): row for row in gpu_inventory}
    if len(by_index) != len(gpu_inventory):
        raise RuntimeError("CUDA inventory contains duplicate logical indices")
    selected = []
    for spec in specs:
        index = int(spec.split(":", 1)[1])
        row = by_index.get(index)
        if row is None:
            raise RuntimeError(f"approved CUDA device is absent from inventory: {spec}")
        uuid = row.get("uuid")
        if not isinstance(uuid, str) or not uuid.strip():
            raise RuntimeError(f"CUDA device {spec} has no stable UUID")
        selected.append({"spec": spec, **row})
    return selected


def _regular_file(path_value, label):
    path = Path(path_value)
    if path.is_symlink():
        raise RuntimeError(f"{label} must be a regular file, not a symbolic link")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _file_binding(path):
    path = Path(path).resolve()
    return {
        "kind": "file",
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _nearest_existing_parent(path):
    candidate = Path(path).resolve()
    while not candidate.exists():
        if candidate == candidate.parent:
            raise RuntimeError("cannot resolve an existing filesystem for disk preflight")
        candidate = candidate.parent
    return candidate


def _parse_utc(value, label):
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError(f"{label} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise RuntimeError(f"{label} is not a valid timestamp") from error
    if parsed.tzinfo != timezone.utc:
        raise RuntimeError(f"{label} must be UTC")
    return parsed


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
