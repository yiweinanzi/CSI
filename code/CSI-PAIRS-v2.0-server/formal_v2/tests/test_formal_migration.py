from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from formal_v2 import formal_migration as migration
from formal_v2 import formal_migration_evidence as migration_evidence
from formal_v2.formal_config import ARMS
from formal_v2.formal_evaluation_compare import CSV_CONTRACTS, REQUIRED_ARTIFACTS
from formal_v2.formal_evaluation_subset_compare import (
    CHECKPOINT_SELECTION_RULE,
    EVALUATION_TABLE_FIELDS,
    IMPLEMENTATION_EVIDENCE_FIELDS,
    REPORT_SCHEMA as REAL_SUBSET_REPORT_SCHEMA,
    SELECTION_RULE,
    WORKER_FRAGMENT_SCHEMA,
    WORKER_REQUEST_SCHEMA,
    compare_evaluation_tables,
    gate_aggregate_report,
)
from formal_v2.formal_evidence import (
    EVIDENCE_AUTH_KEYS,
    FACTORIAL_SCHEMA,
    QUALIFICATION_SCHEMA,
    RUNTIME_PROVENANCE_FIELDS,
)
from formal_v2.formal_io import artifact_manifest, sha256_file, write_json
from formal_v2.formal_llm_judge import (
    APPROVAL_ATTESTATION,
    LLM_JUDGE_APPROVAL_SCHEMA,
    LLM_JUDGE_REQUIRED,
)
from formal_v2.formal_model import CSIPairsFormalModel, torch


SEEDS = (20270001, 20270002, 20270003)
CONFIG_SHA256 = hashlib.sha256(b"frozen-v6-config").hexdigest()
MIGRATION_NONCE = hashlib.sha256(b"migration-nonce").hexdigest()
NEW_RUN_NONCE = hashlib.sha256(b"new-run-nonce").hexdigest()
RUN_NONCE = hashlib.sha256(b"legacy-run-nonce").hexdigest()
CREATED_UTC = "2026-08-31T00:00:00Z"
APPROVED_UTC = "2026-08-31T00:01:00Z"
ACCEPTED_UTC = "2026-08-31T00:02:00Z"
EXPIRES_UTC = "2026-08-31T01:01:00Z"
NOW = datetime(2026, 8, 31, 0, 2, tzinfo=timezone.utc)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_ordered_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


class FormalMigrationFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.legacy_run = root / "legacy-run"
        self.new_run = root / "new-run"
        self.external_root = root / "external-approvals"
        self.legacy_run.mkdir()
        self.new_run.mkdir()
        self.external_root.mkdir()

        self.dataset = root / "dataset.npz"
        self.dataset.write_bytes(b"formal-dataset\x00")
        self.protocol = root / "frozen-protocol-v6.md"
        self.protocol.write_text("frozen V6 protocol\n", encoding="ascii")

        self.legacy_source = self._make_source("legacy-source", "legacy")
        self.new_source = self._make_source("new-source", "linear-resume")
        self.legacy_source_identity = migration._source_identity(self.legacy_source)
        self.new_source_identity = migration._source_identity(self.new_source)
        self.runtime = self._runtime_record()
        self.qualification_evidence = self._evidence(
            "qualification", "FORMAL_EXPERIMENT_ALLOWED"
        )
        self.factorial_evidence = self._evidence(
            "factorial", "CANDIDATE_NOT_CLAIM"
        )

        self._make_qualification()
        self._make_factorial()
        self._make_legacy_approval()
        self._make_migration_evidence()

    def _make_source(self, name: str, marker: str) -> Path:
        repository = self.root / name
        source = repository / "formal_v2"
        source.mkdir(parents=True)
        (source / "identity.py").write_text(
            f'SOURCE_VARIANT = "{marker}"\n', encoding="ascii"
        )
        (source / "formal_evaluation.py").write_text(
            f'EVALUATION_VARIANT = "{marker}"\n', encoding="ascii"
        )
        if marker != "legacy":
            (source / "formal_evaluation_streaming.py").write_text(
                'STREAMING_IMPLEMENTATION = "linear-resume"\n', encoding="ascii"
            )
            (source / "formal_evaluation_subset_compare.py").write_text(
                'COMPARATOR_HARNESS = "dual-source-v1"\n', encoding="ascii"
            )
        (source / "requirements-lock-linux-x86_64-cu121.txt").write_text(
            "torch==2.4.0 --hash=sha256:" + "1" * 64 + "\n",
            encoding="ascii",
        )
        _git(repository, "init", "--quiet")
        _git(repository, "config", "user.name", "migration-test")
        _git(repository, "config", "user.email", "migration@example.invalid")
        _git(repository, "add", ".")
        _git(repository, "commit", "--quiet", "-m", f"{marker} source")
        return source.resolve()

    def _runtime_record(self) -> dict[str, object]:
        runtime = {key: None for key in RUNTIME_PROVENANCE_FIELDS}
        runtime.update(
            {
                "schema_version": "csi-pairs-runtime-provenance-v3",
                "source_tree_sha256": self.legacy_source_identity[
                    "source_tree_sha256"
                ],
                "requirements_lock_sha256": self.legacy_source_identity[
                    "requirements_lock_sha256"
                ],
                "python_implementation": "CPython",
                "python_dont_write_bytecode": True,
            }
        )
        return runtime

    def _evidence(self, label: str, scientific_use: str) -> dict[str, object]:
        return {
            "artifact_label": label,
            "dataset_sha256": sha256_file(self.dataset),
            "config_sha256": CONFIG_SHA256,
            "fixture": False,
            "source_tree_sha256": self.legacy_source_identity[
                "source_tree_sha256"
            ],
            "requirements_lock_sha256": self.legacy_source_identity[
                "requirements_lock_sha256"
            ],
            "runtime_provenance_sha256": _canonical_sha256(self.runtime),
            "runtime_provenance": self.runtime,
            "scientific_use": scientific_use,
        }

    @staticmethod
    def _write_manifest(stage: Path, evidence: dict[str, object]) -> None:
        write_json(
            stage / "manifest.json",
            {
                "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
                **evidence,
                "files": artifact_manifest(stage, evidence=evidence),
            },
        )

    def _make_qualification(self) -> None:
        stage = self.legacy_run / "qualification"
        stage.mkdir()
        self.teacher = stage / "checkpoints" / "teacher.pt"
        self.teacher.parent.mkdir()
        self.teacher.write_bytes(b"authenticated-teacher-checkpoint")
        self.qualification_gate = stage / "gate.json"
        write_json(
            self.qualification_gate,
            {
                "schema_version": QUALIFICATION_SCHEMA,
                "status": "QUALIFICATION_PASS",
                "passed": True,
                "upstream_gates": {"G1": "PASS", "G2": "PASS"},
                "teacher_checkpoint": str(self.teacher.resolve()),
                "teacher_checkpoint_sha256": sha256_file(self.teacher),
                **self.qualification_evidence,
            },
        )
        self._write_manifest(stage, self.qualification_evidence)

    def _make_factorial(self) -> None:
        if torch is None:
            raise unittest.SkipTest("formal migration checkpoint tests require PyTorch")
        stage = self.legacy_run / "factorial"
        stage.mkdir()
        self.factorial_root = stage
        self.normalization = {
            "schema_version": "migration-test-normalization-v1",
            "channel_mean": [0.0],
            "channel_scale": [1.0],
        }
        write_json(stage / "normalization.json", self.normalization)
        write_json(
            stage / "frozen_pilot.json",
            {
                "schema_version": "csi-pairs-v6-frozen-pilot-v1",
                "source_roles": [
                    "source_encoder_train",
                    "source_method_selection",
                ],
                "role_permissions": {
                    "source_encoder_train": "endpoint-only pilot optimization",
                    "source_method_selection": "no-op selection",
                },
                "pilot_contract": "frozen pilot test contract",
                "pilot_seed": 20276002,
                "pilot_steps": 128,
                "alignment_scale": 0.1,
                "response_scale": 0.49,
                "alignment_null_tolerance": 0.0,
                "checkpoint_reused_for_final_training": False,
            },
        )

        model_spec = {
            "patch_count": 1,
            "patch_rows": 1,
            "patch_columns": 1,
            "patch_dim": 2,
            "map_channels": 1,
            "action_channels": 1,
            "radio_dim": 1,
            "latent_dim": 4,
            "state_dim": 4,
            "map_dim": 4,
            "hidden_dim": 4,
            "attention_heads": 1,
            "csi_encoder_layers": 1,
        }
        model = CSIPairsFormalModel(**model_spec)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        scalar_evidence = {
            key: value
            for key, value in self.factorial_evidence.items()
            if not isinstance(value, (dict, list))
        }
        checkpoint_rows: list[dict[str, object]] = []
        resume_rows: list[dict[str, object]] = []
        context_sha256 = hashlib.sha256(b"frozen-resume-context").hexdigest()
        for seed in SEEDS:
            for arm in ARMS:
                checkpoint = stage / "checkpoints" / f"seed_{seed}_{arm}.pt"
                checkpoint.parent.mkdir(exist_ok=True)
                torch.save(
                    {
                        "schema_version": "csi-pairs-formal-checkpoint-v2.1-v6",
                        "arm": arm,
                        "seed": seed,
                        "model_spec": model_spec,
                        "normalization": self.normalization,
                        "teacher_checkpoint_sha256": sha256_file(self.teacher),
                        "checkpoint_rule": "fixed_final_step_no_target_selection",
                        "state_dict": model.state_dict(),
                        **self.factorial_evidence,
                    },
                    checkpoint,
                )
                checkpoint_rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "path": checkpoint.relative_to(stage).as_posix(),
                        "sha256": sha256_file(checkpoint),
                        "parameters": parameters,
                        "measured_flops_per_step": 1,
                        "execution_device": "cpu",
                        "teacher_checkpoint_sha256": sha256_file(self.teacher),
                        **scalar_evidence,
                    }
                )

                pointer = stage / "resume" / f"seed_{seed}" / f"{arm}.latest.json"
                pointer.parent.mkdir(parents=True, exist_ok=True)
                resume_checkpoint = pointer.parent / f"{arm}.generation_1.pt"
                resume_checkpoint.write_bytes(f"{seed}:{arm}:20000".encode("ascii"))
                write_json(
                    pointer,
                    {
                        "schema_version": "csi-pairs-v6-factorial-resume-pointer-v1",
                        "status": "COMPLETE",
                        "seed": seed,
                        "arm": arm,
                        "completed_steps": 20_000,
                        "total_steps": 20_000,
                        "batch_size": 1,
                        "generation": 1,
                        "context_sha256": context_sha256,
                        "checkpoint_path": resume_checkpoint.name,
                        "checkpoint_sha256": sha256_file(resume_checkpoint),
                        "checkpoint_bytes": resume_checkpoint.stat().st_size,
                        "checkpoint_schema_version": (
                            "csi-pairs-v6-factorial-resume-checkpoint-v1"
                        ),
                    },
                )
                resume_rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "pointer_path": pointer.relative_to(stage).as_posix(),
                        "pointer_sha256": sha256_file(pointer),
                        "checkpoint_path": resume_checkpoint.relative_to(stage).as_posix(),
                        "checkpoint_sha256": sha256_file(resume_checkpoint),
                        "completed_steps": 20_000,
                    }
                )

        self.checkpoint_index = stage / "checkpoint_index.json"
        write_json(
            self.checkpoint_index,
            {
                "schema_version": "csi-pairs-formal-checkpoint-index-v2.1-v6",
                **self.factorial_evidence,
                "checkpoints": checkpoint_rows,
            },
        )
        write_json(
            stage / "resume_index.json",
            {
                "schema_version": "csi-pairs-v6-factorial-resume-index-v1",
                **self.factorial_evidence,
                "checkpoint_interval_steps": 200,
                "jobs": resume_rows,
            },
        )
        write_json(stage / "factorial_statistics.json", {"effect": 0.0})
        (stage / "training_summary.csv").write_text(
            "seed,arm,steps\n20270001,endpoint,20000\n", encoding="ascii"
        )
        (stage / "localization_per_bank.csv").write_text(
            "bank,metric\n0,0.0\n", encoding="ascii"
        )
        (stage / "localization_per_sample.csv").write_text(
            "sample,metric\n0,0.0\n", encoding="ascii"
        )
        (stage / "localization_summary.csv").write_text(
            "metric,value\nmedian,0.0\n", encoding="ascii"
        )
        self.factorial_gate = stage / "gate.json"
        gate_vector = {f"G{index}": "NOT_ASSESSED" for index in range(9)}
        gate_vector["G5"] = "FAIL"
        write_json(
            self.factorial_gate,
            {
                "schema_version": FACTORIAL_SCHEMA,
                "status": "FAIL",
                "passed": False,
                "gate_vector": gate_vector,
                "qualification_gate_sha256": sha256_file(
                    self.qualification_gate
                ),
                **self.factorial_evidence,
            },
        )
        self.refresh_factorial_manifest()

    def refresh_factorial_manifest(self) -> None:
        self._write_manifest(self.factorial_root, self.factorial_evidence)

    def _make_legacy_approval(self) -> None:
        approval_root = self.legacy_run / "approval"
        approval_root.mkdir()
        preflight = {
            "schema_version": "csi-pairs-full-run-static-preflight-v2",
            "config_sha256": CONFIG_SHA256,
            "dataset_sha256": sha256_file(self.dataset),
            "fixture": False,
            "source_tree_sha256": self.legacy_source_identity[
                "source_tree_sha256"
            ],
            "requirements_lock_sha256": self.legacy_source_identity[
                "requirements_lock_sha256"
            ],
            "runtime_provenance_sha256": _canonical_sha256(self.runtime),
            "runtime_provenance": self.runtime,
        }
        preflight_path = approval_root / "preflight.json"
        write_json(preflight_path, preflight)
        run_id = f"csi-pairs-{RUN_NONCE[:16]}"
        compute_sha256 = hashlib.sha256(b"formal-compute-plan").hexdigest()
        waibu_root = self.legacy_run / "waibu_resources"
        waibu_root.mkdir()
        self.waibu_gate = waibu_root / "gate.json"
        write_json(self.waibu_gate, {"schema_version": "test-waibu-v1", "status": "PASS"})
        waibu_manifest = waibu_root / "manifest.json"
        write_json(
            waibu_manifest,
            {
                "schema_version": "test-waibu-manifest-v1",
                "files": artifact_manifest(waibu_root),
            },
        )
        gate_bindings = {
            "G1_G2": {
                "gate_path": "qualification/gate.json",
                "gate_sha256": sha256_file(self.qualification_gate),
                "manifest_path": "qualification/manifest.json",
                "manifest_sha256": sha256_file(
                    self.legacy_run / "qualification" / "manifest.json"
                ),
            },
            "waibu_resources": {
                "gate_path": "waibu_resources/gate.json",
                "gate_sha256": sha256_file(self.waibu_gate),
                "manifest_path": "waibu_resources/manifest.json",
                "manifest_sha256": sha256_file(waibu_manifest),
            },
        }
        request = {
            "schema_version": "csi-pairs-full-run-approval-request-v2",
            "run_id": run_id,
            "run_nonce": RUN_NONCE,
            "prepared_utc": "2026-08-30T23:55:00Z",
            "prepared_root": str(self.legacy_run.resolve()),
            "config_sha256": CONFIG_SHA256,
            "dataset_sha256": sha256_file(self.dataset),
            "fixture": False,
            "source_tree_sha256": self.legacy_source_identity[
                "source_tree_sha256"
            ],
            "requirements_lock_sha256": self.legacy_source_identity[
                "requirements_lock_sha256"
            ],
            "runtime_provenance_sha256": _canonical_sha256(self.runtime),
            "runtime_provenance": self.runtime,
            "external_runtime_provenance": {},
            "gpu_inventory": ["fixture-gpu-0", "fixture-gpu-1"],
            "required_gpu_count": 2,
            "execution_devices": ["cuda:0", "cuda:1"],
            "required_environment_values": {},
            "compute_plan": {"sha256": compute_sha256},
            "input_bindings": {},
            "preflight_path": "approval/preflight.json",
            "preflight_sha256": sha256_file(preflight_path),
            "gate_bindings": gate_bindings,
            "teacher_checkpoint": self.teacher.relative_to(
                self.legacy_run
            ).as_posix(),
            "teacher_checkpoint_sha256": sha256_file(self.teacher),
            "decision_required": LLM_JUDGE_REQUIRED,
            "scientific_use": "FORMAL_EXPERIMENT_ALLOWED",
            "review_scope": ["bound formal upstream"],
        }
        request_path = approval_root / "request.json"
        write_json(request_path, request)
        write_json(
            approval_root / "prepared.json",
            {
                "schema_version": "csi-pairs-full-run-prepared-v2",
                "run_id": run_id,
                "run_nonce": RUN_NONCE,
                "prepared_root": str(self.legacy_run.resolve()),
                "request_path": "approval/request.json",
                "request_sha256": sha256_file(request_path),
                "status": "AWAITING_LLM_JUDGE",
            },
        )
        external_path = self.external_root / "legacy-approval.json"
        write_json(
            external_path,
            {
                "schema_version": LLM_JUDGE_APPROVAL_SCHEMA,
                "decision": "APPROVE",
                "run_id": run_id,
                "run_nonce": RUN_NONCE,
                "request_sha256": sha256_file(request_path),
                "compute_plan_sha256": compute_sha256,
                "approved_gate_sha256s": {
                    name: value["gate_sha256"]
                    for name, value in gate_bindings.items()
                },
                "judge": "codex:migration-test-legacy-review",
                "approved_utc": "2026-08-30T23:56:00Z",
                "expires_utc": "2026-08-31T00:56:00Z",
                "attestation": APPROVAL_ATTESTATION,
            },
        )
        write_json(
            approval_root / "accepted.json",
            {
                "schema_version": "csi-pairs-full-run-approval-accepted-v2",
                "status": "ACCEPTED",
                "run_id": run_id,
                "run_nonce": RUN_NONCE,
                "request_sha256": sha256_file(request_path),
                "approval_manifest_path": str(external_path.resolve()),
                "approval_manifest_sha256": sha256_file(external_path),
                "compute_plan_sha256": compute_sha256,
                "accepted_utc": "2026-08-30T23:57:00Z",
            },
        )

    def _make_migration_evidence(self) -> None:
        evaluation = self.legacy_run / "evaluation"
        evaluation.mkdir()
        (evaluation / "legacy.log").write_text(
            "legacy evaluation frozen\n", encoding="ascii"
        )
        run_id = f"csi-pairs-{NEW_RUN_NONCE[:16]}"
        self.compute_plan = self.external_root / "new-compute-plan.json"
        write_json(
            self.compute_plan,
            {
                "schema_version": migration.MIGRATION_COMPUTE_PLAN_SCHEMA,
                "status": "FROZEN",
                "created_utc": CREATED_UTC,
                "run_id": run_id,
                "run_nonce": NEW_RUN_NONCE,
                "source_commit": self.new_source_identity["git_commit"],
                "source_tree_sha256": self.new_source_identity[
                    "source_tree_sha256"
                ],
                "output_root": str(self.new_run.resolve()),
                "dataset_sha256": sha256_file(self.dataset),
                "config_sha256": CONFIG_SHA256,
                "gpu_mapping": [
                    {
                        "logical_device": f"cuda:{index}",
                        "physical_index": index,
                        "uuid": f"GPU-test-{index}",
                        "name": "A100",
                        "total_memory_bytes": 40 * 1024**3,
                    }
                    for index in range(2)
                ],
                "batch_size": 16,
                "shard_size": 128,
                "estimated_output_bytes": 1_000,
                "minimum_free_disk_bytes": 2_000,
                "free_disk_bytes_at_plan": 3_000,
                "estimated_wall_time_seconds": 100,
                "authorized_wall_time_seconds": 200,
                "estimated_gpu_hours": 1.0,
                "authorized_gpu_hours": 2.0,
            },
        )
        identity = {
            "legacy_run_root": str(self.legacy_run.resolve()),
            "new_run_root": str(self.new_run.resolve()),
            "legacy_commit": self.legacy_source_identity["git_commit"],
            "legacy_source_tree_sha256": self.legacy_source_identity[
                "source_tree_sha256"
            ],
            "new_commit": self.new_source_identity["git_commit"],
            "new_source_tree_sha256": self.new_source_identity[
                "source_tree_sha256"
            ],
            "requirements_lock_sha256": self.legacy_source_identity[
                "requirements_lock_sha256"
            ],
            "dataset_sha256": sha256_file(self.dataset),
            "config_sha256": CONFIG_SHA256,
            "protocol_sha256": sha256_file(self.protocol),
            "new_run_id": run_id,
            "new_run_nonce": NEW_RUN_NONCE,
            "new_compute_plan_sha256": sha256_file(self.compute_plan),
        }
        self.migration_evidence: dict[str, Path] = {}
        command = "python -B migration-validation"

        benchmark_trace = self.external_root / "performance-observations.json"
        write_json(benchmark_trace, {"measurement_source": "synthetic-test-harness"})
        gpu_trace = self.external_root / "gpu-benchmark-observations.json"
        write_json(gpu_trace, {"sampling_source": "synthetic-test-harness"})
        measurements = []
        for index, (scale, count, wall, cpu, rss) in enumerate(
            (
                ("N", 100, 1.0, 0.8, 1_000),
                ("2N", 200, 2.0, 1.6, 1_500),
                ("4N", 400, 4.0, 3.2, 2_000),
            )
        ):
            output = self.external_root / f"performance-output-{index}.bin"
            output.write_bytes(f"output-{scale}".encode("ascii"))
            measurements.append(
                {
                    "scale": scale,
                    "sample_count": count,
                    "wall_seconds": wall,
                    "cpu_seconds": cpu,
                    "peak_rss_bytes": rss,
                    "execution_device": "cuda:0",
                    "gpu_utilization_percent": 75.0 + index,
                    "gpu_utilization_statistic": "MEAN",
                    "gpu_sample_count": 10 + index,
                    "peak_vram_bytes": 2_000 + index * 100,
                    "output_path": output,
                    "oracle_equivalent": True,
                }
            )
        performance = self.external_root / "performance.json"
        migration_evidence.write_performance_report(
            performance,
            expected_identity=identity,
            command=command,
            measurements=measurements,
            measurement_artifact_paths=[benchmark_trace],
            gpu_benchmark={
                "execution_device": "cuda:0",
                "batch_size": 16,
                "wall_seconds": 0.5,
                "gpu_utilization_percent": 80.0,
                "gpu_utilization_statistic": "MEAN",
                "gpu_sample_count": 20,
                "gpu_sampling_interval_seconds": 0.1,
                "peak_vram_bytes": 4_000,
                "artifact_path": gpu_trace,
            },
            created_utc=CREATED_UTC,
        )
        self.migration_evidence["performance"] = performance

        smoke_report = self.external_root / "smoke-comparator.json"
        formal_report = self.external_root / "formal-subset-comparator.json"
        smoke = self._comparator_report(
            identity, fixture=True, reference_source_tree="e" * 64
        )
        for side in ("reference", "candidate"):
            smoke["manifest"][side]["identity"]["dataset_sha256"] = "7" * 64
            smoke["manifest"][side]["identity"]["config_sha256"] = "8" * 64
        oracle_root = self.external_root / "smoke-oracle" / "evaluation"
        candidate_root = self.external_root / "smoke-candidate" / "evaluation"
        oracle_root.mkdir(parents=True)
        candidate_root.mkdir(parents=True)
        oracle_manifest = oracle_root / "manifest.json"
        candidate_manifest = candidate_root / "manifest.json"
        write_json(
            oracle_manifest,
            {
                "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
                **smoke["manifest"]["reference"]["identity"],
                "files": [],
            },
        )
        write_json(
            candidate_manifest,
            {
                "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
                **smoke["manifest"]["candidate"]["identity"],
                "files": [],
            },
        )
        smoke["reference_root"] = str(oracle_root)
        smoke["candidate_root"] = str(candidate_root)
        smoke["manifest"]["reference"]["path"] = str(oracle_manifest)
        smoke["manifest"]["candidate"]["path"] = str(candidate_manifest)
        write_json(smoke_report, smoke)
        self.write_real_subset_report(
            formal_report, identity=identity, label="migration-formal-subset"
        )
        equivalence = self.external_root / "equivalence.json"
        migration_evidence.write_equivalence_report(
            equivalence,
            expected_identity=identity,
            command=command,
            smoke_report_path=smoke_report,
            formal_subset_report_path=formal_report,
            created_utc=CREATED_UTC,
        )
        self.migration_evidence["equivalence"] = equivalence

        control_trace = self.external_root / "control-observations.log"
        control_trace.write_text("authenticated control observations\n", encoding="ascii")
        resume = self.external_root / "resume.json"
        migration_evidence.write_resume_report(
            resume,
            expected_identity=identity,
            command=command,
            artifact_paths=[control_trace],
            interrupted=True,
            resumed=True,
            completed=True,
            reused_shard_count=3,
            recomputed_shard_count=1,
            output_equivalent=True,
            created_utc=CREATED_UTC,
        )
        self.migration_evidence["resume"] = resume
        corrupt = self.external_root / "corrupt_checkpoint.json"
        migration_evidence.write_corrupt_checkpoint_report(
            corrupt,
            expected_identity=identity,
            command=command,
            artifact_paths=[control_trace],
            corruption_detected=True,
            corrupt_shard_count=1,
            quarantined_shard_count=1,
            recomputed_shard_count=1,
            unaffected_shard_count=3,
            created_utc=CREATED_UTC,
        )
        self.migration_evidence["corrupt_checkpoint"] = corrupt
        stale = self.external_root / "stale_checkpoint.json"
        migration_evidence.write_stale_checkpoint_report(
            stale,
            expected_identity=identity,
            command=command,
            artifact_paths=[control_trace],
            stale_identity_detected=True,
            rejected_shard_count=2,
            reused_stale_shard_count=0,
            created_utc=CREATED_UTC,
        )
        self.migration_evidence["stale_checkpoint"] = stale
        lock = self.external_root / "lock.json"
        migration_evidence.write_lock_report(
            lock,
            expected_identity=identity,
            command=command,
            artifact_paths=[control_trace],
            second_writer_attempted=True,
            second_writer_rejected=True,
            first_writer_preserved=True,
            write_count=1,
            created_utc=CREATED_UTC,
        )
        self.migration_evidence["lock"] = lock
        progress = self.external_root / "progress.json"
        migration_evidence.write_progress_report(
            progress,
            expected_identity=identity,
            command=command,
            artifact_paths=[control_trace],
            monotonic=True,
            atomic=True,
            final_complete=True,
            update_count=4,
            completed_units=4,
            total_units=4,
            created_utc=CREATED_UTC,
        )
        self.migration_evidence["progress"] = progress

        patch = self.external_root / "base-to-new.patch"
        patch.write_text("diff --git a/formal.py b/formal.py\n", encoding="ascii")
        diff = self.external_root / "base-to-new-diff.json"
        migration_evidence.write_base_to_new_diff_report(
            diff,
            expected_identity=identity,
            command=command,
            diff_path=patch,
            changed_file_count=1,
            reviewed=True,
            applies_cleanly=True,
            created_utc=CREATED_UTC,
        )
        self.migration_evidence["base_to_new_diff"] = diff

        inventory = self.external_root / "legacy-evaluation-inventory.json"
        migration_evidence.write_legacy_evaluation_inventory_report(
            inventory,
            expected_identity=identity,
            command=command,
            created_utc=CREATED_UTC,
        )
        self.migration_evidence["legacy_evaluation_inventory"] = inventory
        freeze = self.external_root / "legacy-evaluation-freeze.json"
        migration_evidence.write_legacy_evaluation_freeze_receipt(
            freeze,
            expected_identity=identity,
            command=command,
            inventory_path=inventory,
            pid_identity="66858:legacy-evaluation",
            start_ticks=12345,
            cmd="python -B -m formal_v2.formal_cli run-evaluation",
            cwd=self.legacy_source.parent,
            process_state="RUNNING",
            pid_identity_matched=True,
            lock_state={
                "observed": True,
                "status": "HELD",
                "owner_pid_identity": "66858:legacy-evaluation",
            },
            created_utc=CREATED_UTC,
        )
        self.migration_evidence["legacy_evaluation_freeze"] = freeze

    def write_real_subset_report(
        self,
        path: Path,
        *,
        identity: dict[str, object] | None = None,
        label: str = "formal-subset",
    ) -> dict[str, object]:
        bound_identity = identity or json.loads(
            self.migration_evidence["performance"].read_text(encoding="utf-8")
        )["inputs"]["identity"]
        config_path = self.external_root / f"{label}-config.json"
        write_json(
            config_path,
            {
                "schema_version": "formal-subset-config-fixture-v1",
                "factorial": {"arms": ["endpoint"]},
            },
        )

        def input_record(
            source: Path, *, identity_sha256: str | None = None
        ) -> dict[str, object]:
            resolved = source.resolve()
            stat_result = resolved.stat()
            record = {
                "path": str(resolved),
                "bytes": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns,
                "sha256": sha256_file(resolved),
            }
            if identity_sha256 is not None:
                record["identity_sha256"] = identity_sha256
            return record

        shared_files = sorted(
            (
                input_record(
                    config_path,
                    identity_sha256=str(bound_identity["config_sha256"]),
                ),
                input_record(
                    self.dataset,
                    identity_sha256=str(bound_identity["dataset_sha256"]),
                ),
            ),
            key=lambda record: str(record["path"]),
        )
        shared_inputs = {
            "config_sha256": bound_identity["config_sha256"],
            "dataset_sha256": bound_identity["dataset_sha256"],
            "files": shared_files,
        }
        scenes = [
            {
                "scene_index": 10,
                "scene_id": "source-unseen-scene",
                "scene_role": "source_final_unseen_bank",
                "city_id": "source-city",
                "bank_id": "bank-a",
                "base_map_cluster_id": "cluster-a",
                "canonical_base_map_digest": "a" * 64,
                "canonical_bank_digest": "b" * 64,
            },
            {
                "scene_index": 20,
                "scene_id": "target-scene",
                "scene_role": "target",
                "city_id": "target-city",
                "bank_id": "bank-b",
                "base_map_cluster_id": "cluster-b",
                "canonical_base_map_digest": "c" * 64,
                "canonical_bank_digest": "d" * 64,
            },
        ]
        checkpoints = [
            {"seed": 20270001, "arm": "endpoint", "sha256": "9" * 64}
        ]
        selection = {
            "scene_rule": SELECTION_RULE,
            "checkpoint_rule": CHECKPOINT_SELECTION_RULE,
            "scenes_in_execution_order": scenes,
            "checkpoints_in_execution_order": checkpoints,
            "sha256": _canonical_sha256(
                {
                    "scene_rule": SELECTION_RULE,
                    "checkpoint_rule": CHECKPOINT_SELECTION_RULE,
                    "scenes": scenes,
                    "checkpoints": checkpoints,
                }
            ),
        }

        def source_file_record(source: Path) -> dict[str, object]:
            resolved = source.resolve()
            return {
                "path": str(resolved),
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
                "git_tracked": True,
            }

        harness = self.new_source / "formal_evaluation_subset_compare.py"

        def worker_source(
            source: Path,
            *,
            commit: str,
            source_tree: str,
            require_streaming: bool,
            runtime_sha256: str,
        ) -> dict[str, object]:
            files = {
                "formal_evaluation.py": source_file_record(
                    source / "formal_evaluation.py"
                ),
                "formal_evaluation_subset_compare.py": source_file_record(harness),
            }
            if require_streaming:
                files["formal_evaluation_streaming.py"] = source_file_record(
                    source / "formal_evaluation_streaming.py"
                )
            repository = source.parent.resolve()
            return {
                "source_root": str(repository),
                "git_root": str(repository),
                "git_commit": commit,
                "tracked_tree_clean": True,
                "tracked_status": [],
                "untracked_scientific_paths": [],
                "tracked_diff_sha256": hashlib.sha256(b"").hexdigest(),
                "files": files,
                "source_tree_sha256": source_tree,
                "runtime_provenance_sha256": runtime_sha256,
            }

        def table_rows(
            *, source_tree: str, runtime_sha256: str
        ) -> dict[str, list[dict[str, object]]]:
            tables = {}
            for table, fields in EVALUATION_TABLE_FIELDS.items():
                contract = CSV_CONTRACTS[table]
                row = {}
                for field in fields:
                    if field in contract.float_fields:
                        row[field] = 0.0
                    elif field in contract.structured_fields:
                        row[field] = {"fixture_metric": 0.0}
                    elif field == "seed":
                        row[field] = 20270001
                    elif field == "artifact_label":
                        row[field] = label
                    elif field == "dataset_sha256":
                        row[field] = bound_identity["dataset_sha256"]
                    elif field == "config_sha256":
                        row[field] = bound_identity["config_sha256"]
                    elif field == "fixture":
                        row[field] = False
                    elif field == "scientific_use":
                        row[field] = "CANDIDATE_NOT_CLAIM"
                    elif field == "source_tree_sha256":
                        row[field] = source_tree
                    elif field == "requirements_lock_sha256":
                        row[field] = bound_identity["requirements_lock_sha256"]
                    elif field == "runtime_provenance_sha256":
                        row[field] = runtime_sha256
                    else:
                        row[field] = f"fixture-{field}"
                tables[table] = [row]
            return tables

        evaluation_snapshot = migration_evidence._real_subset_directory_snapshot(
            self.legacy_run / "evaluation"
        )
        worker_state = {}
        for side, role, source, commit_key, tree_key in (
            (
                "legacy",
                "frozen_legacy",
                self.legacy_source,
                "legacy_commit",
                "legacy_source_tree_sha256",
            ),
            (
                "new",
                "streaming_candidate",
                self.new_source,
                "new_commit",
                "new_source_tree_sha256",
            ),
        ):
            source_tree = str(bound_identity[tree_key])
            runtime = {
                "schema_version": "real-subset-runtime-fixture-v1",
                "source_tree_sha256": source_tree,
                "requirements_lock_sha256": bound_identity[
                    "requirements_lock_sha256"
                ],
            }
            runtime_sha256 = _canonical_sha256(runtime)
            source_identity = worker_source(
                source,
                commit=str(bound_identity[commit_key]),
                source_tree=source_tree,
                require_streaming=side == "new",
                runtime_sha256=runtime_sha256,
            )
            request_path = path.with_name(f"{path.name}.{role}.request.json")
            fragment_path = path.with_name(f"{path.name}.{role}.fragment.json")
            request = {
                "schema_version": WORKER_REQUEST_SCHEMA,
                "config_path": str(config_path.resolve()),
                "dataset_path": str(self.dataset.resolve()),
                "legacy_run_root": str(self.legacy_run.resolve()),
                "device": "cuda:0",
                "batch_size": 1,
                "source_scenes": 1,
                "target_scenes_per_city": 1,
                "checkpoint_count": 1,
                "checkpoint_arm": "endpoint",
                "role": role,
                "fragment_path": str(fragment_path.resolve()),
                "expected_git_commit": bound_identity[commit_key],
                "expected_evaluation_sha256": source_identity["files"][
                    "formal_evaluation.py"
                ]["sha256"],
            }
            write_json(request_path, request)
            request_binding = input_record(request_path)
            resources = {
                "elapsed_seconds": 1.0,
                "maximum_resident_set_bytes": 1_000_000,
                "maximum_cuda_allocated_bytes": 2_000_000,
            }
            read_only_files = [
                {
                    "path": record["path"],
                    "before": record,
                    "after": {
                        key: record[key]
                        for key in ("path", "bytes", "mtime_ns", "sha256")
                    },
                    "unchanged": True,
                }
                for record in shared_files
            ]
            tables = table_rows(
                source_tree=source_tree, runtime_sha256=runtime_sha256
            )
            same_source = (
                table_rows(source_tree=source_tree, runtime_sha256=runtime_sha256)
                if side == "new"
                else None
            )
            fragment = {
                "schema_version": WORKER_FRAGMENT_SCHEMA,
                "role": role,
                "created_utc": CREATED_UTC,
                "request": {
                    "path": request_binding["path"],
                    "bytes": request_binding["bytes"],
                    "sha256": request_binding["sha256"],
                },
                "source_identity": source_identity,
                "runtime_provenance": runtime,
                "input_bindings": shared_inputs,
                "selection": selection,
                "tables": tables,
                "same_source_merged_tables": same_source,
                "read_only": {
                    "passed": True,
                    "files": read_only_files,
                    "legacy_evaluation": (
                        {
                            "root": str((self.legacy_run / "evaluation").resolve()),
                            "before": evaluation_snapshot,
                            "after": evaluation_snapshot,
                            "unchanged": True,
                        }
                        if side == "legacy"
                        else None
                    ),
                },
                "resources": resources,
            }
            _write_ordered_json(fragment_path, fragment)
            worker_state[side] = {
                "role": role,
                "source_identity": source_identity,
                "runtime_provenance": runtime,
                "request": request_binding,
                "fragment": input_record(fragment_path),
                "resources": resources,
                "fragment_body": fragment,
            }

        legacy_fragment = worker_state["legacy"]["fragment_body"]
        new_fragment = worker_state["new"]["fragment_body"]
        cross_comparison = compare_evaluation_tables(
            legacy_fragment["tables"],
            new_fragment["tables"],
            implementation_fields=IMPLEMENTATION_EVIDENCE_FIELDS,
        )
        cross_gate = gate_aggregate_report(
            legacy_fragment["tables"],
            new_fragment["tables"],
            implementation_fields=IMPLEMENTATION_EVIDENCE_FIELDS,
        )
        same_comparison = compare_evaluation_tables(
            new_fragment["same_source_merged_tables"], new_fragment["tables"]
        )
        same_gate = gate_aggregate_report(
            new_fragment["same_source_merged_tables"], new_fragment["tables"]
        )
        workers = {
            side: {
                key: value
                for key, value in state.items()
                if key != "fragment_body"
            }
            for side, state in worker_state.items()
        }
        report = {
            "schema_version": REAL_SUBSET_REPORT_SCHEMA,
            "status": "PASS",
            "passed": True,
            "created_utc": CREATED_UTC,
            "authoritative_layer": "layers.cross_source_end_to_end",
            "contract": {
                "formal_dataset_only": True,
                "full_legacy_evaluation_executed": False,
                "legacy_run_read_only": True,
                "frozen_legacy_run_body_executed": True,
                "cross_source_expected_implementation_fields": list(
                    IMPLEMENTATION_EVIDENCE_FIELDS
                ),
                "scientific_float_requirement": (
                    "IEEE-754 binary64 bitwise equality"
                ),
                "absolute_tolerance": 0.0,
                "relative_tolerance": 0.0,
            },
            "workers": workers,
            "shared_inputs": shared_inputs,
            "selection": selection,
            "layers": {
                "cross_source_end_to_end": {
                    "authoritative": True,
                    "legacy_source_role": "frozen_legacy_oracle",
                    "candidate_source_role": "streaming_candidate",
                    "comparison": cross_comparison,
                    "gate_required_aggregates": cross_gate,
                },
                "same_source_decomposition": {
                    "authoritative": False,
                    "description": (
                        "candidate-source merged-scene decomposition versus "
                        "candidate-source per-scene streaming; this layer is not "
                        "legacy-source evidence"
                    ),
                    "comparison": same_comparison,
                    "gate_required_aggregates": same_gate,
                },
            },
            "read_only": {
                "passed": True,
                "legacy": legacy_fragment["read_only"],
                "new": new_fragment["read_only"],
            },
            "resources": {
                "elapsed_seconds": 3.0,
                "legacy": legacy_fragment["resources"],
                "new": new_fragment["resources"],
            },
            "report_path": str(path.resolve()),
        }
        _write_ordered_json(path, report)
        return report

    @staticmethod
    def _comparator_report(
        identity: dict[str, object],
        *,
        fixture: bool,
        reference_source_tree: str | None = None,
    ) -> dict:
        def comparator_identity(source_tree: str) -> dict[str, object]:
            runtime = {
                "schema_version": "runtime-v1",
                "source_tree_sha256": source_tree,
                "requirements_lock_sha256": identity["requirements_lock_sha256"],
            }
            return {
                "artifact_label": "migration-smoke" if fixture else "migration-formal-subset",
                "dataset_sha256": identity["dataset_sha256"],
                "config_sha256": identity["config_sha256"],
                "fixture": fixture,
                "scientific_use": (
                    "FORBIDDEN" if fixture else "FORMAL_EXPERIMENT_ALLOWED"
                ),
                "source_tree_sha256": source_tree,
                "requirements_lock_sha256": identity[
                    "requirements_lock_sha256"
                ],
                "runtime_provenance_sha256": _canonical_sha256(runtime),
                "runtime_provenance": runtime,
            }

        reference_identity = comparator_identity(
            reference_source_tree or str(identity["legacy_source_tree_sha256"])
        )
        candidate_identity = comparator_identity(
            str(identity["new_source_tree_sha256"])
        )
        return {
            "schema_version": "csi-pairs-v6-evaluation-equivalence-report-v1",
            "generated_utc": "2026-08-31T00:00:00+00:00",
            "reference_root": str(Path(identity["legacy_run_root"]) / "comparison"),
            "candidate_root": str(Path(identity["new_run_root"]) / "comparison"),
            "status": "PASS",
            "equivalent": True,
            "exact_match": False,
            "tolerance": {
                "absolute": 0.0,
                "relative": 0.0,
                "rule": "abs(a-b) <= absolute + relative * max(abs(a), abs(b))",
            },
            "required_artifacts": list(REQUIRED_ARTIFACTS) + ["manifest.json"],
            "csv": {
                name: {
                    "equivalent": True,
                    "reference_row_count": 1,
                    "candidate_row_count": 1,
                }
                for name in CSV_CONTRACTS
            },
            "gate": {
                "reference": {"valid": True},
                "candidate": {"valid": True},
                "comparison": {"equivalent": True},
                "identity_binding": {
                    "reference": {"valid": True},
                    "candidate": {"valid": True},
                },
                "equivalent": True,
            },
            "manifest": {
                "reference": {"valid": True, "identity": reference_identity},
                "candidate": {"valid": True, "identity": candidate_identity},
                "comparable": True,
            },
            "summary": {
                "csv_passed": len(CSV_CONTRACTS),
                "csv_total": len(CSV_CONTRACTS),
                "csv_exact": 0,
                "gate_equivalent": True,
                "gate_exact": False,
                "manifest_comparable": True,
                "manifest_exact": False,
            },
        }

    def request_arguments(self) -> dict[str, object]:
        return {
            "legacy_run_root": self.legacy_run,
            "legacy_source_root": self.legacy_source,
            "new_run_root": self.new_run,
            "new_source_root": self.new_source,
            "protocol_path": self.protocol,
            "dataset_path": self.dataset,
            "config_sha256": CONFIG_SHA256,
            "expected_seeds": SEEDS,
            "migration_nonce": MIGRATION_NONCE,
            "new_run_nonce": NEW_RUN_NONCE,
            "new_compute_plan_path": self.compute_plan,
            "migration_evidence_paths": self.migration_evidence,
            "created_utc": CREATED_UTC,
        }

    def acceptance_arguments(self) -> dict[str, object]:
        return {
            "new_run_root": self.new_run,
            "new_source_root": self.new_source,
            "protocol_path": self.protocol,
            "dataset_path": self.dataset,
            "config_sha256": CONFIG_SHA256,
            "expected_seeds": SEEDS,
        }

    def write_migration_approval(
        self,
        request: dict[str, object],
        *,
        path: Path | None = None,
        judge: str = "codex:migration-test-review",
        request_sha256: str | None = None,
    ) -> Path:
        destination = path or self.external_root / "migration-approval.json"
        write_json(
            destination,
            {
                "schema_version": migration.MIGRATION_APPROVAL_SCHEMA,
                "decision": "APPROVE",
                "migration_id": request["migration_id"],
                "migration_nonce": request["migration_nonce"],
                "new_run_id": request["new_run_id"],
                "new_run_nonce": request["new_run_nonce"],
                "request_sha256": request_sha256 or request["request_sha256"],
                "legacy_source_tree_sha256": request["legacy_source"][
                    "source_tree_sha256"
                ],
                "new_source_tree_sha256": request["new_source"][
                    "source_tree_sha256"
                ],
                "checkpoint_inventory_sha256": request[
                    "checkpoint_inventory_sha256"
                ],
                "new_compute_plan_sha256": request["new_compute_plan"]["sha256"],
                "migration_evidence_sha256": request[
                    "migration_evidence_sha256"
                ],
                "judge": judge,
                "approved_utc": APPROVED_UTC,
                "expires_utc": EXPIRES_UTC,
                "attestation": migration.MIGRATION_APPROVAL_ATTESTATION,
            },
        )
        return destination


class FormalMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = FormalMigrationFixture(Path(self.temporary.name))
        self.source_patch = mock.patch.object(
            migration, "_RUNNING_SOURCE_ROOT", self.fixture.new_source
        )
        self.source_patch.start()

    def tearDown(self) -> None:
        self.source_patch.stop()
        self.temporary.cleanup()

    def _request_and_approval(self) -> tuple[dict[str, object], Path]:
        request = migration.write_migration_request(
            **self.fixture.request_arguments()
        )
        approval = self.fixture.write_migration_approval(request)
        return request, approval

    def _accept(
        self, request: dict[str, object], approval: Path
    ) -> dict[str, object]:
        return migration.accept_migration_request(
            request["request_path"],
            approval,
            **self.fixture.acceptance_arguments(),
            accepted_utc=ACCEPTED_UTC,
            now=NOW,
        )

    def test_round_trip_preserves_fail_state_and_refuses_replay(self) -> None:
        request, approval = self._request_and_approval()
        self.assertEqual(len(request["checkpoint_inventory"]), 12)
        self.assertEqual(
            [(row["seed"], row["arm"]) for row in request["checkpoint_inventory"]],
            [(seed, arm) for seed in SEEDS for arm in ARMS],
        )
        accepted = self._accept(request, approval)
        authenticated = migration.authenticate_migration_receipt(
            accepted["accepted_path"], **self.fixture.acceptance_arguments()
        )

        self.assertEqual(authenticated.legacy_run_root, self.fixture.legacy_run)
        self.assertEqual(len(authenticated.checkpoint_rows), 12)
        self.assertEqual(
            authenticated.checkpoint_inventory_sha256,
            request["checkpoint_inventory_sha256"],
        )
        self.assertEqual(
            authenticated.new_source_git_commit,
            request["new_source"]["git_commit"],
        )
        self.assertEqual(
            authenticated.new_compute_plan,
            request["new_compute_plan"],
        )
        self.assertEqual(
            authenticated.qualification_evidence["source_tree_sha256"],
            request["legacy_source"]["source_tree_sha256"],
        )
        self.assertEqual(
            authenticated.factorial_evidence["source_tree_sha256"],
            request["legacy_source"]["source_tree_sha256"],
        )
        self.assertEqual(
            authenticated.legacy_scientific_state["factorial_status"], "FAIL"
        )
        self.assertIs(
            authenticated.legacy_scientific_state["factorial_passed"], False
        )
        self.assertEqual(
            authenticated.legacy_scientific_state["factorial_gate_vector"]["G5"],
            "FAIL",
        )
        self.assertTrue(
            request["scientific_contract"]["scientific_pass_not_inferred"]
        )
        with self.assertRaises(FileExistsError):
            migration.write_migration_request(**self.fixture.request_arguments())
        with self.assertRaises(FileExistsError):
            self._accept(request, approval)

        other_run = Path(self.temporary.name) / "other-new-run"
        other_run.mkdir()
        other_arguments = self.fixture.acceptance_arguments()
        other_arguments["new_run_root"] = other_run
        with self.assertRaises(RuntimeError):
            migration.authenticate_migration_receipt(
                accepted["accepted_path"], **other_arguments
            )

    def test_acceptance_reauthenticates_stage_artifacts(self) -> None:
        request, approval = self._request_and_approval()
        training_summary = self.fixture.factorial_root / "training_summary.csv"
        training_summary.write_text("tampered\n", encoding="ascii")
        with self.assertRaises(RuntimeError):
            self._accept(request, approval)

    def test_resume_reauthenticates_checkpoint_bytes(self) -> None:
        request, approval = self._request_and_approval()
        accepted = self._accept(request, approval)
        first_checkpoint = Path(request["checkpoint_inventory"][0]["path"])
        with first_checkpoint.open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaises(RuntimeError):
            migration.authenticate_migration_receipt(
                accepted["accepted_path"], **self.fixture.acceptance_arguments()
            )

    def test_request_requires_frozen_twenty_thousand_steps(self) -> None:
        resume_index_path = self.fixture.factorial_root / "resume_index.json"
        resume_index = json.loads(resume_index_path.read_text(encoding="utf-8"))
        first_job = resume_index["jobs"][0]
        pointer_path = self.fixture.factorial_root / first_job["pointer_path"]
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        pointer["completed_steps"] = 19_999
        pointer["total_steps"] = 19_999
        write_json(pointer_path, pointer)
        first_job["completed_steps"] = 19_999
        first_job["pointer_sha256"] = sha256_file(pointer_path)
        write_json(resume_index_path, resume_index)
        self.fixture.refresh_factorial_manifest()
        with self.assertRaises(RuntimeError):
            migration.write_migration_request(**self.fixture.request_arguments())

    def test_request_reauthenticates_every_legacy_approval_gate(self) -> None:
        request, approval = self._request_and_approval()
        write_json(
            self.fixture.waibu_gate,
            {"schema_version": "test-waibu-v1", "status": "TAMPERED"},
        )
        with self.assertRaises(RuntimeError):
            self._accept(request, approval)

    def test_checkpoint_index_requires_all_cells_in_frozen_order(self) -> None:
        original = json.loads(
            self.fixture.checkpoint_index.read_text(encoding="utf-8")
        )
        for mutation in ("missing", "reordered"):
            with self.subTest(mutation=mutation):
                payload = json.loads(json.dumps(original))
                if mutation == "reordered":
                    payload["checkpoints"][0], payload["checkpoints"][1] = (
                        payload["checkpoints"][1],
                        payload["checkpoints"][0],
                    )
                else:
                    payload["checkpoints"].pop()
                write_json(self.fixture.checkpoint_index, payload)
                self.fixture.refresh_factorial_manifest()
                with self.assertRaises(RuntimeError):
                    migration.write_migration_request(
                        **self.fixture.request_arguments()
                    )

    def test_dirty_legacy_or_new_source_is_rejected(self) -> None:
        (self.fixture.legacy_source / "identity.py").write_text(
            'SOURCE_VARIANT = "dirty-legacy"\n', encoding="ascii"
        )
        with self.assertRaises(RuntimeError):
            migration.write_migration_request(**self.fixture.request_arguments())

    def test_approval_must_be_external_exact_and_from_allowed_llm(self) -> None:
        request = migration.write_migration_request(
            **self.fixture.request_arguments()
        )
        approval_path = self.fixture.write_migration_approval(
            request, judge="human:reviewer"
        )
        with self.assertRaises(ValueError):
            self._accept(request, approval_path)

        approval_path = self.fixture.write_migration_approval(
            request, request_sha256="0" * 64
        )
        with self.assertRaises(RuntimeError):
            self._accept(request, approval_path)

        inside_run = self.fixture.new_run / "migration" / "in-run-approval.json"
        self.fixture.write_migration_approval(request, path=inside_run)
        with self.assertRaises(RuntimeError):
            self._accept(request, inside_run)

    def test_request_binds_all_required_identity_layers(self) -> None:
        request, _approval = self._request_and_approval()
        self.assertEqual(request["legacy_run_root"], str(self.fixture.legacy_run))
        self.assertEqual(request["new_run_root"], str(self.fixture.new_run))
        self.assertEqual(request["dataset_sha256"], sha256_file(self.fixture.dataset))
        self.assertEqual(request["config_sha256"], CONFIG_SHA256)
        self.assertEqual(request["protocol_sha256"], sha256_file(self.fixture.protocol))
        self.assertEqual(
            request["legacy_source"]["git_commit"],
            _git(self.fixture.legacy_source, "rev-parse", "HEAD"),
        )
        self.assertEqual(
            request["new_source"]["git_commit"],
            _git(self.fixture.new_source, "rev-parse", "HEAD"),
        )
        self.assertEqual(
            request["qualification"]["manifest_sha256"],
            sha256_file(self.fixture.legacy_run / "qualification" / "manifest.json"),
        )
        self.assertEqual(
            request["factorial"]["manifest_sha256"],
            sha256_file(self.fixture.factorial_root / "manifest.json"),
        )
        self.assertEqual(
            request["legacy_approval"]["judge"],
            "codex:migration-test-legacy-review",
        )
        self.assertEqual(set(request["expected_arms"]), set(ARMS))
        self.assertEqual(
            set(request["qualification"]), migration._QUALIFICATION_BINDING_FIELDS
        )
        self.assertEqual(set(request["factorial"]), migration._FACTORIAL_BINDING_FIELDS)
        self.assertEqual(
            set(request["legacy_approval"]), migration._LEGACY_APPROVAL_FIELDS
        )
        for key in EVIDENCE_AUTH_KEYS:
            self.assertIn(key, self.fixture.qualification_evidence)

    def test_missing_or_tampered_technical_evidence_fails_closed(self) -> None:
        arguments = self.fixture.request_arguments()
        incomplete = dict(self.fixture.migration_evidence)
        incomplete.pop("equivalence")
        arguments["migration_evidence_paths"] = incomplete
        with self.assertRaises(RuntimeError):
            migration.write_migration_request(**arguments)

        request, approval = self._request_and_approval()
        progress = self.fixture.migration_evidence["progress"]
        payload = json.loads(progress.read_text(encoding="utf-8"))
        payload["results"]["validated"] = False
        write_json(progress, payload)
        with self.assertRaises(RuntimeError):
            self._accept(request, approval)

    def test_compute_plan_and_freeze_inventory_are_reauthenticated(self) -> None:
        request, approval = self._request_and_approval()
        plan = json.loads(self.fixture.compute_plan.read_text(encoding="utf-8"))
        plan["batch_size"] += 1
        write_json(self.fixture.compute_plan, plan)
        with self.assertRaises(RuntimeError):
            self._accept(request, approval)

    def test_source_identity_rejects_digest_included_untracked_code(self) -> None:
        (self.fixture.new_source / "injected.py").write_text(
            "UNTRACKED_EXECUTION = True\n", encoding="ascii"
        )
        with self.assertRaisesRegex(RuntimeError, "digest-included untracked"):
            migration._source_identity(self.fixture.new_source)

    def test_source_identity_rejects_dirty_tracked_code(self) -> None:
        tracked = self.fixture.new_source / "identity.py"
        tracked.write_text(
            tracked.read_text(encoding="ascii") + "DIRTY_EXECUTION = True\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(RuntimeError, "tracked modifications"):
            migration._source_identity(self.fixture.new_source)


if __name__ == "__main__":
    unittest.main()
