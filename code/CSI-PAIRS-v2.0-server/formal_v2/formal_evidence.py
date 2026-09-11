"""Dataset/checkpoint integrity and scientific checks for retained experiment stages.

Runtime records contain versions and source digests, not installation certification.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable
from .formal_config import public_formal_config
from .formal_dataset import FormalDataset
from .formal_io import sha256_file
from .experiment_runtime import (configure_reproducible_runtime, runtime_provenance,
    validate_runtime_provenance, source_digest as _source_tree_sha256,
    RUNTIME_PROVENANCE_FIELDS, TORCH_RUNTIME_FIELDS, SCHEMA as RUNTIME_PROVENANCE_SCHEMA)


QUALIFICATION_SCHEMA = "csi-pairs-formal-qualification-gate-v4-v6"

FACTORIAL_SCHEMA = "csi-pairs-formal-factorial-gate-v2.1-v6"

GATE_IDS = tuple(f"G{index}" for index in range(9))

CLAIM_IDS = tuple(f"C{index}" for index in range(1, 14))

ASSESSMENT_STATES = {"PASS", "FAIL", "BLOCKED", "NOT_ASSESSED"}

EVIDENCE_AUTH_KEYS = (
    "artifact_label",
    "dataset_sha256",
    "config_sha256",
    "fixture",
    "source_tree_sha256",
    "requirements_lock_sha256",
    "runtime_provenance_sha256",
    "runtime_provenance",
)

def config_sha256(config: dict) -> str:
    payload = json.dumps(
        public_formal_config(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def evidence_context_from_recorded_runtime(
    config: dict,
    dataset: FormalDataset,
    scientific_use: str,
    recorded_runtime: object,
) -> dict[str, object]:
    """Authenticate main-runtime evidence from an isolated adapter interpreter."""
    configure_reproducible_runtime()
    runtime = validate_runtime_provenance(recorded_runtime)
    runtime_payload = json.dumps(
        runtime,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "artifact_label": config["artifact_label"],
        "dataset_sha256": sha256_file(dataset.source_path),
        "config_sha256": config_sha256(config),
        "fixture": dataset.is_fixture,
        "scientific_use": "FORBIDDEN" if dataset.is_fixture else scientific_use,
        "source_tree_sha256": runtime["source_tree_sha256"],
        "requirements_lock_sha256": runtime["requirements_lock_sha256"],
        "runtime_provenance_sha256": hashlib.sha256(runtime_payload).hexdigest(),
        "runtime_provenance": runtime,
    }

def bind_rows(rows: Iterable[dict], evidence: dict[str, object]) -> list[dict]:
    output = []
    for source in rows:
        row = dict(source)
        for key, value in evidence.items():
            if isinstance(value, (dict, list)):
                continue
            if key in row and row[key] != value:
                raise ValueError(f"row attempts to override evidence field {key!r}")
            row[key] = value
        output.append(row)
    return output

def require_formal_qualification(
    gate: object,
    config: dict,
    dataset: FormalDataset,
    *,
    allow_nonscientific_fixture: bool,
    evidence_runtime: object | None = None,
) -> dict:
    if not isinstance(gate, dict):
        raise RuntimeError("qualification gate must be an object")
    if (
        not dataset.is_fixture
        and gate.get("scientific_use") != "FORMAL_EXPERIMENT_ALLOWED"
    ):
        raise RuntimeError("factorial requires scientific_use=FORMAL_EXPERIMENT_ALLOWED")
    required = {
        "schema_version",
        "passed",
        "scientific_use",
        "fixture",
        "dataset_sha256",
        "config_sha256",
        "upstream_gates",
        "teacher_checkpoint",
        "teacher_checkpoint_sha256",
        "artifact_label",
        "source_tree_sha256",
        "requirements_lock_sha256",
        "runtime_provenance_sha256",
        "runtime_provenance",
        "primary_route_contract",
        "physical_response",
    }
    missing = required.difference(gate)
    if missing:
        raise RuntimeError(f"qualification gate is missing authenticated fields: {sorted(missing)}")
    if gate["schema_version"] != QUALIFICATION_SCHEMA:
        raise RuntimeError("qualification gate schema is not V6-compatible")
    from .formal_routing import PRIMARY_ROUTE_CONTRACT

    if gate["primary_route_contract"] != PRIMARY_ROUTE_CONTRACT:
        raise RuntimeError("qualification gate primary route contract is not V6-compatible")
    expected = (
        evidence_context(config, dataset, str(gate["scientific_use"]))
        if evidence_runtime is None
        else evidence_context_from_recorded_runtime(
            config,
            dataset,
            str(gate["scientific_use"]),
            evidence_runtime,
        )
    )
    for key in EVIDENCE_AUTH_KEYS:
        if gate[key] != expected[key]:
            raise RuntimeError(f"qualification gate {key} does not match the current run")
    software_fixture_execution = bool(
        dataset.is_fixture
        and allow_nonscientific_fixture
        and gate["scientific_use"] == "FORBIDDEN"
    )
    if not bool(gate["passed"]) and not software_fixture_execution:
        raise RuntimeError("formal factorial is blocked because the upstream qualification gate failed")
    if dataset.is_fixture:
        if not allow_nonscientific_fixture:
            raise RuntimeError("fixture training requires explicit software-test permission")
        if gate["scientific_use"] != "FORBIDDEN":
            raise RuntimeError("fixture qualification gate must remain FORBIDDEN")
    elif gate["scientific_use"] != "FORMAL_EXPERIMENT_ALLOWED":
        raise RuntimeError("factorial requires scientific_use=FORMAL_EXPERIMENT_ALLOWED")
    for gate_id, status in gate["upstream_gates"].items():
        if status not in ASSESSMENT_STATES:
            raise RuntimeError(f"invalid upstream assessment state for {gate_id}")
        if (
            gate_id in {"G1", "G2"}
            and status != "PASS"
            and not software_fixture_execution
        ):
            raise RuntimeError(f"required upstream gate {gate_id} is not PASS")
    teacher_checkpoint = Path(str(gate["teacher_checkpoint"]))
    if not teacher_checkpoint.is_file():
        raise RuntimeError("qualification teacher checkpoint is missing")
    if sha256_file(teacher_checkpoint) != gate["teacher_checkpoint_sha256"]:
        raise RuntimeError("qualification teacher checkpoint hash mismatch")
    physical = gate["physical_response"]
    if not isinstance(physical, dict):
        raise RuntimeError("qualification physical_response must be an object")
    if dataset.is_fixture:
        if (
            physical.get("status") != "NOT_ASSESSED_FIXTURE_FORBIDDEN"
            or physical.get("formal_physical_response_required_for_nonfixture") is not True
        ):
            raise RuntimeError("fixture qualification weakened physical-response exclusion")
    else:
        from .formal_action_inverse_response import (
            PHYSICAL_RESPONSE_CONTRACT,
            load_physical_response_checkpoint,
            load_physical_response_config,
        )

        checkpoint = Path(str(physical.get("checkpoint", "")))
        bindings = physical.get("bindings")
        physical_config = physical.get("physical_config")
        physical_config_path = (
            Path(__file__).resolve().parent / "configs" / "physical_response_v1.json"
        )
        if (
            physical.get("status") != "FORMAL_QUALIFICATION_MODEL_FIT"
            or physical.get("contract") != PHYSICAL_RESPONSE_CONTRACT
            or physical.get("checkpoint_round_trip_valid") is not True
            or checkpoint.is_symlink()
            or not checkpoint.is_file()
            or not isinstance(bindings, dict)
            or physical_config != load_physical_response_config()
            or physical.get("physical_config_sha256")
            != sha256_file(physical_config_path)
            or int(physical_config.get("selection_position_stride", -1)) != 1
            or physical_config.get("world_scope") != "all"
        ):
            raise RuntimeError("qualification physical-response contract is invalid")
        if sha256_file(checkpoint) != physical.get("checkpoint_sha256"):
            raise RuntimeError("qualification physical-response checkpoint hash mismatch")
        model_source = Path(str(bindings.get("model_source", "")))
        candidate_config = Path(str(bindings.get("candidate_config", "")))
        qualification_root = teacher_checkpoint.resolve().parent.parent
        physical_checkpoint_root = (
            qualification_root / "checkpoints" / "qualification_probes"
        ).resolve()
        if (
            bindings.get("dataset_sha256") != sha256_file(dataset.source_path)
            or bindings.get("formal_config_sha256") != expected["config_sha256"]
            or bindings.get("physical_config_sha256")
            != sha256_file(physical_config_path)
            or candidate_config.is_symlink()
            or not candidate_config.is_file()
            or sha256_file(candidate_config)
            != bindings.get("candidate_config_sha256")
            or model_source.is_symlink()
            or not model_source.is_file()
            or sha256_file(model_source) != bindings.get("model_source_sha256")
            or not checkpoint.resolve().is_relative_to(physical_checkpoint_root)
        ):
            raise RuntimeError("qualification physical-response binding mismatch")
        _model, checkpoint_bindings = load_physical_response_checkpoint(checkpoint)
        if checkpoint_bindings != bindings:
            raise RuntimeError("qualification physical-response payload binding mismatch")
    return gate

def complete_gate_vector(overrides: dict[str, str] | None = None) -> dict[str, str]:
    result = {gate_id: "NOT_ASSESSED" for gate_id in GATE_IDS}
    if overrides:
        for gate_id, status in overrides.items():
            if gate_id not in result:
                raise ValueError(f"unknown V6 gate: {gate_id}")
            if status not in ASSESSMENT_STATES:
                raise ValueError(f"invalid gate status: {status}")
            result[gate_id] = status
    return result

def blocked_claim_vector() -> dict[str, str]:
    return {claim_id: "BLOCKED" for claim_id in CLAIM_IDS}

def require_stage_manifested_gate(
    path: str | Path,
    payload: object,
    config: dict,
    dataset: FormalDataset,
    *,
    schema_version: str,
    evidence_runtime: object | None = None,
) -> dict:
    """Authenticate a gate against its stage manifest and current evidence context."""
    from .formal_io import read_strict_json

    gate_path = Path(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != schema_version:
        raise RuntimeError(f"gate schema mismatch for {gate_path}")
    if not gate_path.is_file():
        raise RuntimeError(f"gate file is missing: {gate_path}")
    on_disk = read_strict_json(gate_path)
    if on_disk != payload:
        raise RuntimeError(f"supplied gate payload differs from its manifested file: {gate_path}")
    expected = (
        evidence_context(config, dataset, str(payload.get("scientific_use", "")))
        if evidence_runtime is None
        else evidence_context_from_recorded_runtime(
            config,
            dataset,
            str(payload.get("scientific_use", "")),
            evidence_runtime,
        )
    )
    for key in EVIDENCE_AUTH_KEYS:
        if payload.get(key) != expected[key]:
            raise RuntimeError(f"gate {key} mismatch for {gate_path}")
    manifest_path = gate_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"gate has no stage manifest: {gate_path}")
    manifest = read_strict_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "csi-pairs-formal-stage-manifest-v2.1-v6"
    ):
        raise RuntimeError(f"stage manifest schema mismatch: {manifest_path}")
    for key in EVIDENCE_AUTH_KEYS:
        if manifest.get(key) != expected[key]:
            raise RuntimeError(f"stage manifest {key} mismatch: {manifest_path}")
    entries = manifest.get("files")
    _authenticate_stage_inventory(gate_path.parent, entries)
    relative = gate_path.name
    matches = [
        row
        for row in entries
        if isinstance(row, dict) and row.get("path") == relative
    ] if isinstance(entries, list) else []
    if len(matches) != 1 or matches[0].get("sha256") != sha256_file(gate_path):
        raise RuntimeError(f"gate is absent from or mismatched with its stage manifest: {gate_path}")
    return payload

def _authenticate_stage_inventory(stage_root: Path, entries: object) -> None:
    """Reauthenticate the completed stage while allowing later nested stages."""
    root = stage_root.resolve()
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"stage manifest has no authenticated inventory: {root}")
    expected: dict[str, Path] = {}
    for row in entries:
        if not isinstance(row, dict):
            raise RuntimeError(f"stage manifest inventory is malformed: {root}")
        relative = row.get("path")
        digest = row.get("sha256")
        size = row.get("bytes")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in expected
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or type(size) is not int
            or size < 0
        ):
            raise RuntimeError(f"stage manifest inventory is malformed: {root}")
        path = root / relative
        if path.is_symlink():
            raise RuntimeError(f"stage artifact is a symbolic link: {path}")
        resolved = path.resolve()
        if root not in resolved.parents or not resolved.is_file():
            raise RuntimeError(f"stage artifact is missing or escapes its root: {path}")
        if resolved.stat().st_size != size or sha256_file(resolved) != digest:
            raise RuntimeError(f"stage artifact changed after manifesting: {path}")
        expected[relative] = resolved

    # Evaluation/controls gain separately manifested retention/shuffle children
    # later in the full chain. They are not part of the parent stage inventory.
    later_nested_roots = {
        manifest_path.parent.resolve()
        for manifest_path in root.rglob("manifest.json")
        if manifest_path.parent.resolve() != root
        and not any(
            relative == manifest_path.parent.relative_to(root).as_posix()
            or relative.startswith(
                manifest_path.parent.relative_to(root).as_posix() + "/"
            )
            for relative in expected
        )
    }
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and not any(nested == path.resolve() or nested in path.resolve().parents for nested in later_nested_roots)
    }
    if set(expected) != actual:
        raise RuntimeError(
            "stage manifest inventory is incomplete: "
            f"missing={sorted(actual - set(expected))[:5]}, "
            f"unexpected={sorted(set(expected) - actual)[:5]}"
        )

def require_manifested_formal_qualification(
    gate: object,
    config: dict,
    dataset: FormalDataset,
    *,
    allow_nonscientific_fixture: bool,
    evidence_runtime: object | None = None,
) -> dict:
    """Validate qualification semantics and authenticate its complete stage output."""
    validated = require_formal_qualification(
        gate,
        config,
        dataset,
        allow_nonscientific_fixture=allow_nonscientific_fixture,
        evidence_runtime=evidence_runtime,
    )
    teacher_parent = Path(str(validated["teacher_checkpoint"])).parent
    stage_dir = teacher_parent.parent if teacher_parent.name == "checkpoints" else teacher_parent
    gate_path = stage_dir / "gate.json"
    return require_stage_manifested_gate(
        gate_path,
        validated,
        config,
        dataset,
        schema_version=QUALIFICATION_SCHEMA,
        evidence_runtime=evidence_runtime,
    )

def evidence_context(config: dict, dataset: FormalDataset, scientific_use: str) -> dict:
    configure_reproducible_runtime()
    return evidence_context_from_recorded_runtime(config, dataset, scientific_use, runtime_provenance())
