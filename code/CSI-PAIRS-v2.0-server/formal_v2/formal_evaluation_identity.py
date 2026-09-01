from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .formal_evaluation_resume import (
    EvaluationExecutionProfile,
    EvaluationRunIdentity,
    NO_MIGRATION_SHA256,
    require_safe_directory_prefix,
)
from .formal_evaluation_streaming import (
    STREAMING_EVALUATION_SCHEMA,
    evaluation_output_schema_sha256,
    run_streaming_formal_evaluation,
)
from .formal_evidence import FACTORIAL_SCHEMA, QUALIFICATION_SCHEMA, evidence_context
from .formal_io import read_strict_json, sha256_file
from .formal_model import resolve_execution_devices
from .formal_upstream import AuthenticatedUpstream


FIXTURE_EVALUATION_PLAN_SCHEMA = "csi-pairs-v6-fixture-evaluation-plan-v1"
FIXTURE_CHECKPOINT_INDEX_SCHEMA = "csi-pairs-formal-checkpoint-index-v2.1-v6"
FIXTURE_EVALUATION_DEVICES = ("cuda:0", "cuda:1")
DEFAULT_PRODUCTION_BATCH_SIZE = 256


def _is_cuda_device(value: object) -> bool:
    if type(value) is not str or not value.startswith("cuda:"):
        return False
    suffix = value[5:]
    return suffix.isdigit() and str(int(suffix)) == suffix


def resolve_production_batch_size(config: dict, plan: dict | None = None) -> int:
    """Production / default streaming batch size. Fixture path stays frozen at 1."""

    evaluation = config.get("evaluation") if isinstance(config, dict) else None
    if isinstance(evaluation, dict):
        value = evaluation.get("batch_size")
        if type(value) is int and value >= 1:
            return value
    if isinstance(plan, dict):
        value = plan.get("batch_size")
        if type(value) is int and value >= 1:
            return value
    return DEFAULT_PRODUCTION_BATCH_SIZE


def resolve_production_devices(dataset, plan: dict | None = None) -> tuple[str, ...]:
    """Accept 1 or 2 CUDA devices from CSI_PAIRS_DEVICES / compute plan."""

    planned: tuple[str, ...] | None = None
    if isinstance(plan, dict):
        mapping = plan.get("gpu_mapping")
        if isinstance(mapping, list) and 1 <= len(mapping) <= 2:
            candidates = tuple(
                str(row.get("logical_device"))
                for row in mapping
                if isinstance(row, dict)
            )
            if (
                len(candidates) == len(mapping)
                and all(_is_cuda_device(device) for device in candidates)
                and len(set(candidates)) == len(candidates)
            ):
                planned = candidates
    resolved = tuple(str(device) for device in resolve_execution_devices(dataset))
    if planned is not None:
        if resolved != planned:
            raise RuntimeError(
                "CSI_PAIRS_DEVICES must match the compute-plan GPU mapping: "
                + ",".join(planned)
            )
        return planned
    if (
        not resolved
        or not all(_is_cuda_device(device) for device in resolved)
        or not 1 <= len(resolved) <= 2
        or len(set(resolved)) != len(resolved)
    ):
        raise RuntimeError("production evaluation requires 1 or 2 unique CUDA devices")
    return resolved


@dataclass(frozen=True)
class MigratedEvaluationExecution:
    identity: EvaluationRunIdentity
    devices: tuple[str, ...]
    batch_size: int


@dataclass(frozen=True)
class FixtureEvaluationExecution:
    identity: EvaluationRunIdentity
    devices: tuple[str, ...]
    batch_size: int
    output_root: Path
    upstream_root: Path
    qualification_gate_path: Path
    factorial_gate_path: Path
    checkpoint_index_path: Path
    checkpoint_inventory_sha256: str


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise RuntimeError(f"{label} cannot be a symbolic link")
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"{label} is missing or not a regular file")
    return resolved


def _require_fixture_binding(
    payload: object,
    evidence: dict[str, object],
    label: str,
    *,
    schema: str | None = None,
) -> dict:
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    if schema is not None and payload.get("schema_version") != schema:
        raise RuntimeError(f"{label} schema mismatch")
    expected = {
        "artifact_label": evidence["artifact_label"],
        "dataset_sha256": evidence["dataset_sha256"],
        "config_sha256": evidence["config_sha256"],
        "fixture": True,
        "scientific_use": "FORBIDDEN",
        "requirements_lock_sha256": evidence["requirements_lock_sha256"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"{label} {key} mismatch")
    return payload


def _running_source_identity() -> dict[str, object]:
    from .formal_migration import _source_identity

    return _source_identity(Path(__file__).resolve().parent)


def _fixture_checkpoint_inventory(
    checkpoint_index: dict,
    factorial_root: Path,
    config: dict,
    evidence: dict[str, object],
) -> list[dict[str, object]]:
    rows = checkpoint_index.get("checkpoints")
    if not isinstance(rows, list):
        raise RuntimeError("fixture checkpoint index checkpoints must be a list")
    expected_cells = {
        (int(seed), str(arm))
        for seed in config["seeds"]
        for arm in config["factorial"]["arms"]
    }
    cells: set[tuple[int, str]] = set()
    inventory: list[dict[str, object]] = []
    for row in rows:
        bound = _require_fixture_binding(row, evidence, "fixture checkpoint row")
        seed = bound.get("seed")
        arm = bound.get("arm")
        relative = bound.get("path")
        expected_sha256 = bound.get("sha256")
        if type(seed) is not int or type(arm) is not str or not arm:
            raise RuntimeError("fixture checkpoint row seed/arm is invalid")
        if type(relative) is not str or not relative or Path(relative).is_absolute():
            raise RuntimeError("fixture checkpoint row path is invalid")
        if type(expected_sha256) is not str or len(expected_sha256) != 64:
            raise RuntimeError("fixture checkpoint row SHA-256 is invalid")
        require_safe_directory_prefix((factorial_root / relative).parent)
        checkpoint = _regular_file(
            factorial_root / relative,
            "fixture checkpoint",
        )
        if not _is_within(checkpoint, factorial_root):
            raise RuntimeError("fixture checkpoint escapes the factorial root")
        observed_sha256 = sha256_file(checkpoint)
        if observed_sha256 != expected_sha256:
            raise RuntimeError("fixture checkpoint SHA-256 mismatch")
        cell = (seed, arm)
        if cell in cells:
            raise RuntimeError("fixture checkpoint index contains a duplicate seed/arm")
        cells.add(cell)
        inventory.append(
            {
                "seed": seed,
                "arm": arm,
                "path": str(checkpoint),
                "sha256": observed_sha256,
            }
        )
    if cells != expected_cells or len(rows) != len(expected_cells):
        raise RuntimeError("fixture checkpoint index cell inventory mismatch")
    return inventory


def build_fixture_evaluation_execution(
    config: dict,
    dataset,
    *,
    output_root: str | Path,
    upstream_root: str | Path,
    run_nonce: str,
    execution_devices: tuple[str, ...] = FIXTURE_EVALUATION_DEVICES,
    batch_size: int = 1,
) -> FixtureEvaluationExecution:
    """Bind a permanently forbidden fixture run without claiming a migration."""

    if getattr(dataset, "is_fixture", None) is not True:
        raise RuntimeError("fixture streaming evaluation requires a fixture dataset")
    if execution_devices != FIXTURE_EVALUATION_DEVICES or batch_size != 1:
        raise RuntimeError(
            "fixture equivalence evaluation requires cuda:0,cuda:1 and batch_size=1"
        )
    upstream_candidate = Path(upstream_root)
    if upstream_candidate.is_symlink():
        raise RuntimeError("fixture upstream root cannot be a symbolic link")
    require_safe_directory_prefix(upstream_candidate)
    upstream = upstream_candidate.resolve()
    if not upstream.is_dir():
        raise RuntimeError("fixture upstream root must be a regular directory")
    output_candidate = Path(output_root)
    if not output_candidate.is_absolute():
        raise RuntimeError("fixture evaluation output root must be absolute")
    require_safe_directory_prefix(output_candidate)
    if output_candidate.is_symlink():
        raise RuntimeError("fixture evaluation output root cannot be a symbolic link")
    output = output_candidate.resolve()
    if _is_within(output, upstream) or _is_within(upstream, output):
        raise RuntimeError("fixture upstream and output roots must be disjoint")

    evidence = evidence_context(config, dataset, "FORBIDDEN")
    if evidence.get("fixture") is not True or evidence.get("scientific_use") != "FORBIDDEN":
        raise RuntimeError("fixture evidence is not permanently FORBIDDEN")
    source = _running_source_identity()
    if (
        source.get("source_tree_sha256") != evidence["source_tree_sha256"]
        or source.get("requirements_lock_sha256")
        != evidence["requirements_lock_sha256"]
    ):
        raise RuntimeError("fixture running source identity differs from runtime evidence")

    require_safe_directory_prefix(upstream / "qualification")
    require_safe_directory_prefix(upstream / "factorial")
    qualification_path = _regular_file(
        upstream / "qualification" / "gate.json",
        "fixture qualification gate",
    )
    factorial_path = _regular_file(
        upstream / "factorial" / "gate.json",
        "fixture factorial gate",
    )
    checkpoint_path = _regular_file(
        upstream / "factorial" / "checkpoint_index.json",
        "fixture checkpoint index",
    )
    qualification_sha256 = sha256_file(qualification_path)
    factorial_sha256 = sha256_file(factorial_path)
    qualification = _require_fixture_binding(
        read_strict_json(qualification_path),
        evidence,
        "fixture qualification gate",
        schema=QUALIFICATION_SCHEMA,
    )
    factorial = _require_fixture_binding(
        read_strict_json(factorial_path),
        evidence,
        "fixture factorial gate",
        schema=FACTORIAL_SCHEMA,
    )
    if factorial.get("qualification_gate_sha256") != qualification_sha256:
        raise RuntimeError("fixture factorial gate does not bind the qualification gate")
    checkpoint_index = _require_fixture_binding(
        read_strict_json(checkpoint_path),
        evidence,
        "fixture checkpoint index",
        schema=FIXTURE_CHECKPOINT_INDEX_SCHEMA,
    )
    inventory = _fixture_checkpoint_inventory(
        checkpoint_index,
        checkpoint_path.parent,
        config,
        evidence,
    )
    if any(
        row.get("teacher_checkpoint_sha256")
        != qualification.get("teacher_checkpoint_sha256")
        for row in checkpoint_index["checkpoints"]
    ):
        raise RuntimeError("fixture checkpoints do not bind the qualified teacher")

    profile = EvaluationExecutionProfile(
        execution_devices=execution_devices,
        batch_size=batch_size,
    )
    output_schema_sha256 = evaluation_output_schema_sha256()
    plan = {
        "schema_version": FIXTURE_EVALUATION_PLAN_SCHEMA,
        "scientific_use": "FORBIDDEN",
        "source_git_commit": source["git_commit"],
        "source_tree_sha256": evidence["source_tree_sha256"],
        "requirements_lock_sha256": evidence["requirements_lock_sha256"],
        "dataset_sha256": evidence["dataset_sha256"],
        "config_sha256": evidence["config_sha256"],
        "upstream_root": str(upstream),
        "output_root": str(output),
        "qualification_gate_sha256": qualification_sha256,
        "factorial_gate_sha256": factorial_sha256,
        "checkpoint_index_sha256": sha256_file(checkpoint_path),
        "fixture_checkpoint_inventory_sha256": _canonical_sha256(inventory),
        "migration_accepted_sha256": NO_MIGRATION_SHA256,
        "legacy_checkpoint_inventory_sha256": NO_MIGRATION_SHA256,
        "run_nonce": run_nonce,
        "execution_profile": profile.as_dict(),
        "output_schema_id": STREAMING_EVALUATION_SCHEMA,
        "output_schema_sha256": output_schema_sha256,
    }
    identity = EvaluationRunIdentity(
        code_revision=str(source["git_commit"]),
        source_tree_sha256=str(evidence["source_tree_sha256"]),
        config_sha256=str(evidence["config_sha256"]),
        dataset_sha256=str(evidence["dataset_sha256"]),
        migration_accepted_sha256=NO_MIGRATION_SHA256,
        legacy_checkpoint_inventory_sha256=NO_MIGRATION_SHA256,
        qualification_gate_sha256=qualification_sha256,
        factorial_gate_sha256=factorial_sha256,
        runtime_provenance_sha256=str(evidence["runtime_provenance_sha256"]),
        run_nonce=run_nonce,
        compute_plan_sha256=_canonical_sha256(plan),
        execution_profile=profile,
        output_schema_id=STREAMING_EVALUATION_SCHEMA,
        output_schema_sha256=output_schema_sha256,
    )
    return FixtureEvaluationExecution(
        identity=identity,
        devices=execution_devices,
        batch_size=batch_size,
        output_root=output,
        upstream_root=upstream,
        qualification_gate_path=qualification_path,
        factorial_gate_path=factorial_path,
        checkpoint_index_path=checkpoint_path,
        checkpoint_inventory_sha256=NO_MIGRATION_SHA256,
    )


def run_fixture_streaming_evaluation(
    config: dict,
    dataset,
    *,
    output_root: str | Path,
    upstream_root: str | Path,
    run_nonce: str,
    execution_devices: tuple[str, ...] = FIXTURE_EVALUATION_DEVICES,
    batch_size: int = 1,
) -> dict:
    execution = build_fixture_evaluation_execution(
        config,
        dataset,
        output_root=output_root,
        upstream_root=upstream_root,
        run_nonce=run_nonce,
        execution_devices=execution_devices,
        batch_size=batch_size,
    )
    return run_streaming_formal_evaluation(
        config,
        dataset,
        execution.output_root,
        upstream_root=execution.upstream_root,
        qualification_gate_path=execution.qualification_gate_path,
        factorial_gate_path=execution.factorial_gate_path,
        checkpoint_index_path=execution.checkpoint_index_path,
        checkpoint_inventory_sha256=execution.checkpoint_inventory_sha256,
        run_identity=execution.identity,
        execution_devices=execution.devices,
        batch_size=execution.batch_size,
    )


def build_migrated_evaluation_execution(
    config: dict, dataset, upstream: AuthenticatedUpstream
) -> MigratedEvaluationExecution:
    """Bind streaming execution to one authenticated migration and compute plan."""

    migration = upstream.migration
    if migration is None:
        raise RuntimeError("streaming migration execution requires an accepted migration")
    evidence = evidence_context(
        config,
        dataset,
        "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM",
    )
    if bool(dataset.is_fixture):
        raise RuntimeError("formal migration execution cannot use a fixture")
    if evidence["source_tree_sha256"] != migration.new_source_tree_sha256:
        raise RuntimeError("running source tree differs from the accepted migration")
    for key in (
        "dataset_sha256",
        "config_sha256",
        "requirements_lock_sha256",
    ):
        if evidence[key] != migration.factorial_evidence[key]:
            raise RuntimeError(f"migrated evaluation {key} differs from legacy upstream")

    plan = migration.new_compute_plan
    # Fixture / subset-compare keep batch_size==1 and the two-GPU pin in
    # build_fixture_evaluation_execution. Production / migrated / default
    # streaming takes batch_size from config (default 256) and accepts 1 or
    # 2 CUDA devices from CSI_PAIRS_DEVICES / the compute plan.
    batch_size = resolve_production_batch_size(config, plan)
    devices = resolve_production_devices(dataset, plan)
    execution_profile = EvaluationExecutionProfile(
        execution_devices=devices,
        batch_size=batch_size,
    )

    qualification_sha256 = sha256_file(upstream.qualification_gate)
    factorial_sha256 = sha256_file(upstream.factorial_gate)
    identity = EvaluationRunIdentity(
        code_revision=migration.new_source_git_commit,
        source_tree_sha256=str(evidence["source_tree_sha256"]),
        config_sha256=str(evidence["config_sha256"]),
        dataset_sha256=str(evidence["dataset_sha256"]),
        migration_accepted_sha256=migration.accepted_sha256,
        legacy_checkpoint_inventory_sha256=(
            migration.checkpoint_inventory_sha256
        ),
        qualification_gate_sha256=qualification_sha256,
        factorial_gate_sha256=factorial_sha256,
        runtime_provenance_sha256=str(evidence["runtime_provenance_sha256"]),
        run_nonce=migration.new_run_nonce,
        compute_plan_sha256=migration.new_compute_plan_sha256,
        execution_profile=execution_profile,
        output_schema_id=STREAMING_EVALUATION_SCHEMA,
        output_schema_sha256=evaluation_output_schema_sha256(),
    )
    return MigratedEvaluationExecution(
        identity=identity,
        devices=devices,
        batch_size=batch_size,
    )


def build_local_evaluation_execution(
    config: dict,
    dataset,
    output_root: str | Path,
    upstream: AuthenticatedUpstream,
) -> MigratedEvaluationExecution:
    """Bind default streaming evaluation without an accepted migration."""

    evidence = evidence_context(
        config,
        dataset,
        "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM",
    )
    source = _running_source_identity()
    if bool(getattr(dataset, "is_fixture", False)):
        devices = FIXTURE_EVALUATION_DEVICES
        batch_size = 1
        resolved = tuple(str(device) for device in resolve_execution_devices(dataset))
        if resolved != devices:
            raise RuntimeError(
                "fixture equivalence evaluation requires cuda:0,cuda:1 and batch_size=1"
            )
    else:
        devices = resolve_production_devices(dataset)
        batch_size = resolve_production_batch_size(config)
    execution_profile = EvaluationExecutionProfile(
        execution_devices=devices,
        batch_size=batch_size,
    )
    qualification_sha256 = sha256_file(upstream.qualification_gate)
    factorial_sha256 = sha256_file(upstream.factorial_gate)
    output = Path(output_root).resolve()
    local_plan = {
        "kind": "local_streaming_execution",
        "output_root": str(output),
        "devices": list(devices),
        "batch_size": batch_size,
        "qualification_gate_sha256": qualification_sha256,
        "factorial_gate_sha256": factorial_sha256,
    }
    identity = EvaluationRunIdentity(
        code_revision=str(source["git_commit"]),
        source_tree_sha256=str(evidence["source_tree_sha256"]),
        config_sha256=str(evidence["config_sha256"]),
        dataset_sha256=str(evidence["dataset_sha256"]),
        migration_accepted_sha256=NO_MIGRATION_SHA256,
        legacy_checkpoint_inventory_sha256=NO_MIGRATION_SHA256,
        qualification_gate_sha256=qualification_sha256,
        factorial_gate_sha256=factorial_sha256,
        runtime_provenance_sha256=str(evidence["runtime_provenance_sha256"]),
        run_nonce=_canonical_sha256(
            {
                "output_root": str(output),
                "dataset_sha256": evidence["dataset_sha256"],
                "config_sha256": evidence["config_sha256"],
                "qualification_gate_sha256": qualification_sha256,
                "factorial_gate_sha256": factorial_sha256,
            }
        ),
        compute_plan_sha256=_canonical_sha256(local_plan),
        execution_profile=execution_profile,
        output_schema_id=STREAMING_EVALUATION_SCHEMA,
        output_schema_sha256=evaluation_output_schema_sha256(),
    )
    return MigratedEvaluationExecution(
        identity=identity,
        devices=devices,
        batch_size=batch_size,
    )
