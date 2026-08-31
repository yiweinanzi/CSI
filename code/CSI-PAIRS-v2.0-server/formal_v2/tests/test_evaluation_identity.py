from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from formal_v2.formal_evaluation_identity import (
    build_migrated_evaluation_execution,
)
from formal_v2.formal_evaluation_resume import EvaluationExecutionProfile


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

    def test_nonexact_batch_size_is_rejected(self) -> None:
        self.migration.new_compute_plan["batch_size"] = 256
        with self.assertRaisesRegex(RuntimeError, "batch_size=1"):
            self._build()

    def test_wrong_device_order_is_rejected(self) -> None:
        self.migration.new_compute_plan["gpu_mapping"].reverse()
        with self.assertRaisesRegex(RuntimeError, "GPU order"):
            self._build()

    def test_changed_source_tree_is_rejected(self) -> None:
        self.migration.new_source_tree_sha256 = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "source tree"):
            self._build()

    def test_fixture_is_rejected(self) -> None:
        self.dataset.is_fixture = True
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            self._build()


if __name__ == "__main__":
    unittest.main()
