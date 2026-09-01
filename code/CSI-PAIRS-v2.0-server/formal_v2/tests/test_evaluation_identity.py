from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

if "fcntl" not in sys.modules:
    try:
        import fcntl as _fcntl
    except ImportError:
        _fcntl = types.ModuleType("fcntl")
        _fcntl.LOCK_EX = 2
        _fcntl.LOCK_SH = 1
        _fcntl.LOCK_UN = 8
        _fcntl.LOCK_NB = 4
        _fcntl.flock = lambda *_args, **_kwargs: None
        sys.modules["fcntl"] = _fcntl

from formal_v2.formal_evaluation_identity import (
    build_fixture_evaluation_execution,
    build_migrated_evaluation_execution,
    resolve_production_batch_size,
    run_fixture_streaming_evaluation,
)
from formal_v2.formal_evaluation_resume import (
    EvaluationExecutionProfile,
    NO_MIGRATION_SHA256,
)


class EvaluationIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.qualification = root / "qualification.json"
        self.factorial = root / "factorial.json"
        self.qualification.write_bytes(b"qualification\n")
        self.factorial.write_bytes(b"factorial\n")
        self.evidence = {
            "dataset_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "requirements_lock_sha256": "3" * 64,
            "source_tree_sha256": "4" * 64,
            "runtime_provenance_sha256": "5" * 64,
        }
        self.migration = SimpleNamespace(
            new_source_tree_sha256="4" * 64,
            factorial_evidence={
                "dataset_sha256": "1" * 64,
                "config_sha256": "2" * 64,
                "requirements_lock_sha256": "3" * 64,
            },
            new_compute_plan={
                "batch_size": 1,
                "gpu_mapping": [
                    {"logical_device": "cuda:0"},
                    {"logical_device": "cuda:1"},
                ],
            },
            new_source_git_commit="6" * 40,
            accepted_sha256="7" * 64,
            checkpoint_inventory_sha256="8" * 64,
            new_run_nonce="9" * 64,
            new_compute_plan_sha256="a" * 64,
        )
        self.upstream = SimpleNamespace(
            migration=self.migration,
            qualification_gate=self.qualification,
            factorial_gate=self.factorial,
        )
        self.dataset = SimpleNamespace(is_fixture=False)

    def _build(self):
        with (
            mock.patch(
                "formal_v2.formal_evaluation_identity.evidence_context",
                return_value=dict(self.evidence),
            ),
            mock.patch(
                "formal_v2.formal_evaluation_identity.resolve_execution_devices",
                return_value=("cuda:0", "cuda:1"),
            ),
        ):
            return build_migrated_evaluation_execution(
                {"artifact_label": "test"}, self.dataset, self.upstream
            )

    def test_identity_binds_accepted_migration_and_exact_execution(self) -> None:
        execution = self._build()
        self.assertEqual(execution.devices, ("cuda:0", "cuda:1"))
        self.assertEqual(execution.batch_size, 1)
        self.assertEqual(
            execution.identity.execution_profile,
            EvaluationExecutionProfile(
                execution_devices=("cuda:0", "cuda:1"),
                batch_size=1,
            ),
        )
        self.assertEqual(execution.identity.code_revision, "6" * 40)
        self.assertEqual(execution.identity.migration_accepted_sha256, "7" * 64)
        self.assertEqual(
            execution.identity.legacy_checkpoint_inventory_sha256, "8" * 64
        )

    def test_production_plan_batch_size_256_is_accepted(self) -> None:
        self.migration.new_compute_plan["batch_size"] = 256
        execution = self._build()
        self.assertEqual(execution.batch_size, 256)

    def test_production_config_batch_size_overrides_plan(self) -> None:
        self.migration.new_compute_plan["batch_size"] = 1
        with (
            mock.patch(
                "formal_v2.formal_evaluation_identity.evidence_context",
                return_value=dict(self.evidence),
            ),
            mock.patch(
                "formal_v2.formal_evaluation_identity.resolve_execution_devices",
                return_value=("cuda:0", "cuda:1"),
            ),
        ):
            execution = build_migrated_evaluation_execution(
                {"artifact_label": "test", "evaluation": {"batch_size": 256}},
                self.dataset,
                self.upstream,
            )
        self.assertEqual(execution.batch_size, 256)
        self.assertEqual(resolve_production_batch_size({}), 256)

    def test_production_accepts_one_gpu(self) -> None:
        self.migration.new_compute_plan["gpu_mapping"] = [
            {"logical_device": "cuda:0"}
        ]
        with (
            mock.patch(
                "formal_v2.formal_evaluation_identity.evidence_context",
                return_value=dict(self.evidence),
            ),
            mock.patch(
                "formal_v2.formal_evaluation_identity.resolve_execution_devices",
                return_value=("cuda:0",),
            ),
        ):
            execution = build_migrated_evaluation_execution(
                {"artifact_label": "test"}, self.dataset, self.upstream
            )
        self.assertEqual(execution.devices, ("cuda:0",))

    def test_wrong_device_order_is_rejected(self) -> None:
        self.migration.new_compute_plan["gpu_mapping"].reverse()
        with self.assertRaisesRegex(RuntimeError, "GPU mapping"):
            self._build()

    def test_changed_source_tree_is_rejected(self) -> None:
        self.migration.new_source_tree_sha256 = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "source tree"):
            self._build()

    def test_fixture_is_rejected(self) -> None:
        self.dataset.is_fixture = True
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            self._build()


class FixtureEvaluationIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.upstream = root / "old-smoke"
        qualification_root = self.upstream / "qualification"
        factorial_root = self.upstream / "factorial"
        qualification_root.mkdir(parents=True)
        factorial_root.mkdir(parents=True)
        self.output = root / "external" / "NEW_SMOKE_EVALUATION"
        self.config = {
            "artifact_label": "fixture",
            "seeds": [17],
            "factorial": {"arms": ["endpoint"]},
        }
        self.dataset = SimpleNamespace(is_fixture=True)
        self.evidence = {
            "artifact_label": "fixture",
            "dataset_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "fixture": True,
            "scientific_use": "FORBIDDEN",
            "requirements_lock_sha256": "3" * 64,
            "source_tree_sha256": "4" * 64,
            "runtime_provenance_sha256": "5" * 64,
        }
        self.source = {
            "git_commit": "6" * 40,
            "source_tree_sha256": "4" * 64,
            "requirements_lock_sha256": "3" * 64,
        }
        self.qualification = qualification_root / "gate.json"
        qualification = {
            **self._binding(),
            "schema_version": "csi-pairs-formal-qualification-gate-v4-v6",
            "teacher_checkpoint_sha256": "7" * 64,
        }
        self.qualification.write_text(
            json.dumps(qualification, sort_keys=True), encoding="ascii"
        )
        checkpoint = factorial_root / "checkpoints" / "seed_17" / "endpoint.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fixture checkpoint\n")
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        checkpoint_row = {
            **self._binding(),
            "seed": 17,
            "arm": "endpoint",
            "path": "checkpoints/seed_17/endpoint.pt",
            "sha256": checkpoint_sha256,
            "teacher_checkpoint_sha256": "7" * 64,
        }
        self.checkpoint_index = factorial_root / "checkpoint_index.json"
        self.checkpoint_index.write_text(
            json.dumps(
                {
                    **self._binding(),
                    "schema_version": "csi-pairs-formal-checkpoint-index-v2.1-v6",
                    "checkpoints": [checkpoint_row],
                },
                sort_keys=True,
            ),
            encoding="ascii",
        )
        self.factorial = factorial_root / "gate.json"
        self.factorial.write_text(
            json.dumps(
                {
                    **self._binding(),
                    "schema_version": "csi-pairs-formal-factorial-gate-v2.1-v6",
                    "qualification_gate_sha256": hashlib.sha256(
                        self.qualification.read_bytes()
                    ).hexdigest(),
                },
                sort_keys=True,
            ),
            encoding="ascii",
        )
        self.run_nonce = "8" * 64

    def _binding(self):
        return {
            "artifact_label": self.evidence["artifact_label"],
            "dataset_sha256": self.evidence["dataset_sha256"],
            "config_sha256": self.evidence["config_sha256"],
            "fixture": True,
            "scientific_use": "FORBIDDEN",
            "requirements_lock_sha256": self.evidence[
                "requirements_lock_sha256"
            ],
        }

    def _build(self, **changes):
        arguments = {
            "output_root": self.output,
            "upstream_root": self.upstream,
            "run_nonce": self.run_nonce,
        }
        arguments.update(changes)
        with (
            mock.patch(
                "formal_v2.formal_evaluation_identity.evidence_context",
                return_value=dict(self.evidence),
            ),
            mock.patch(
                "formal_v2.formal_evaluation_identity._running_source_identity",
                return_value=dict(self.source),
            ),
        ):
            return build_fixture_evaluation_execution(
                self.config,
                self.dataset,
                **arguments,
            )

    def test_fixture_identity_is_no_migration_and_deterministic(self) -> None:
        first = self._build()
        second = self._build()
        self.assertEqual(first.identity, second.identity)
        self.assertEqual(
            first.identity.migration_accepted_sha256, NO_MIGRATION_SHA256
        )
        self.assertEqual(
            first.identity.legacy_checkpoint_inventory_sha256,
            NO_MIGRATION_SHA256,
        )
        self.assertEqual(first.identity.code_revision, "6" * 40)
        self.assertEqual(first.devices, ("cuda:0", "cuda:1"))
        self.assertEqual(first.batch_size, 1)
        self.assertEqual(first.output_root, self.output.resolve())
        self.assertEqual(first.upstream_root, self.upstream.resolve())

    def test_nonfixture_wrong_plan_and_overlapping_output_are_rejected(self) -> None:
        self.dataset.is_fixture = False
        with self.assertRaisesRegex(RuntimeError, "fixture dataset"):
            self._build()
        self.dataset.is_fixture = True
        with self.assertRaisesRegex(RuntimeError, "cuda:0,cuda:1"):
            self._build(execution_devices=("cuda:0",))
        with self.assertRaisesRegex(RuntimeError, "batch_size=1"):
            self._build(batch_size=2)
        with self.assertRaisesRegex(RuntimeError, "disjoint"):
            self._build(output_root=self.upstream / "candidate")

    def test_source_and_upstream_identity_mismatches_are_rejected(self) -> None:
        self.source["source_tree_sha256"] = "9" * 64
        with self.assertRaisesRegex(RuntimeError, "source identity"):
            self._build()
        self.source["source_tree_sha256"] = "4" * 64
        factorial = json.loads(self.factorial.read_text(encoding="ascii"))
        factorial["config_sha256"] = "9" * 64
        self.factorial.write_text(
            json.dumps(factorial, sort_keys=True), encoding="ascii"
        )
        with self.assertRaisesRegex(RuntimeError, "config_sha256"):
            self._build()

    def test_checkpoint_hash_and_teacher_binding_are_fail_closed(self) -> None:
        checkpoint = self.upstream / "factorial/checkpoints/seed_17/endpoint.pt"
        checkpoint.write_bytes(b"tampered\n")
        with self.assertRaisesRegex(RuntimeError, "checkpoint SHA-256"):
            self._build()

    def test_symlinked_upstream_component_is_rejected(self) -> None:
        real_factorial = self.upstream / "factorial"
        relocated = self.upstream.parent / "relocated-factorial"
        real_factorial.rename(relocated)
        real_factorial.symlink_to(relocated, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "regular directory"):
            self._build()

    def test_fixture_runner_uses_only_builder_bound_arguments(self) -> None:
        execution = self._build()
        expected = {"status": "FAIL", "scientific_use": "FORBIDDEN"}
        with (
            mock.patch(
                "formal_v2.formal_evaluation_identity."
                "build_fixture_evaluation_execution",
                return_value=execution,
            ) as build,
            mock.patch(
                "formal_v2.formal_evaluation_identity."
                "run_streaming_formal_evaluation",
                return_value=expected,
            ) as run,
        ):
            result = run_fixture_streaming_evaluation(
                self.config,
                self.dataset,
                output_root=self.output,
                upstream_root=self.upstream,
                run_nonce=self.run_nonce,
            )
        self.assertIs(result, expected)
        build.assert_called_once_with(
            self.config,
            self.dataset,
            output_root=self.output,
            upstream_root=self.upstream,
            run_nonce=self.run_nonce,
            execution_devices=("cuda:0", "cuda:1"),
            batch_size=1,
        )
        run.assert_called_once_with(
            self.config,
            self.dataset,
            execution.output_root,
            upstream_root=execution.upstream_root,
            qualification_gate_path=execution.qualification_gate_path,
            factorial_gate_path=execution.factorial_gate_path,
            checkpoint_index_path=execution.checkpoint_index_path,
            checkpoint_inventory_sha256=NO_MIGRATION_SHA256,
            run_identity=execution.identity,
            execution_devices=("cuda:0", "cuda:1"),
            batch_size=1,
        )


if __name__ == "__main__":
    unittest.main()
