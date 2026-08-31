from __future__ import annotations

from dataclasses import dataclass

from .formal_evaluation_resume import (
    EvaluationExecutionProfile,
    EvaluationRunIdentity,
)
from .formal_evaluation_streaming import (
    STREAMING_EVALUATION_SCHEMA,
    evaluation_output_schema_sha256,
)
from .formal_evidence import evidence_context
from .formal_io import sha256_file
from .formal_model import resolve_execution_devices
from .formal_upstream import AuthenticatedUpstream


@dataclass(frozen=True)
class MigratedEvaluationExecution:
    identity: EvaluationRunIdentity
    devices: tuple[str, ...]
    batch_size: int


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
    batch_size = plan.get("batch_size")
    if type(batch_size) is not int or batch_size != 1:
        raise RuntimeError(
            "formal optimized evaluation requires frozen batch_size=1 for exact parity"
        )
    mapping = plan.get("gpu_mapping")
    if not isinstance(mapping, list) or len(mapping) != 2:
        raise RuntimeError("migration compute plan does not bind exactly two GPUs")
    planned_devices = tuple(str(row.get("logical_device")) for row in mapping)
    if planned_devices != ("cuda:0", "cuda:1"):
        raise RuntimeError("migration GPU order must be exactly cuda:0,cuda:1")
    devices = tuple(str(device) for device in resolve_execution_devices(dataset))
    if devices != planned_devices:
        raise RuntimeError(
            "CSI_PAIRS_DEVICES must exactly match the accepted migration GPU order"
        )
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
