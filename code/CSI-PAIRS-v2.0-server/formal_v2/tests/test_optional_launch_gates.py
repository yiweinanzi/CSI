from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from formal_v2.formal_claims import _claim_package_status
from formal_v2.formal_data_verification import require_verified_roles_from_root
from formal_v2.formal_io import write_json
from formal_v2.formal_run_approval import (
    REQUIRED_FULL_RUN_INPUT_NAMES,
    _bind_input_files,
    _resolve_compute_plan,
    full_run_input_values,
)


class OptionalLaunchGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_path = self.root / "dataset.npz"
        self.dataset_path.write_bytes(b"dataset")

    def tearDown(self):
        self.temporary.cleanup()

    def test_full_run_input_values_omit_absent_optional_manifests(self):
        args = SimpleNamespace(
            adapter_manifest="adapters.json",
            control_manifest="controls.json",
            scene_id_manifest="builtin:sigmap-scene-id-v1",
            shuffled_pair_manifest="shuffled.json",
            retention_manifest="retention.json",
            representation_baseline_config="repr.json",
            verifier_manifest=None,
            literature_resource_manifest=None,
            rt_calibration_manifest=None,
            external_validity_manifest=None,
        )
        values = full_run_input_values(args)
        self.assertEqual(set(values), set(REQUIRED_FULL_RUN_INPUT_NAMES))
        self.assertNotIn("literature_resource_manifest", values)
        self.assertNotIn("verifier_manifest", values)

    def test_bind_input_files_accepts_required_inputs_only(self):
        adapter = self.root / "adapters.json"
        write_json(adapter, {"schema_version": "test", "adapters": []})
        control = self.root / "controls.json"
        write_json(
            control,
            {
                "schema_version": "test-control",
                "rows": [],
            },
        )
        shuffled = self.root / "shuffled.json"
        write_json(
            shuffled,
            {
                "schema_version": "csi-pairs-v6-shuffled-pair-adapter-v3",
                "command": ["{python}", "{adapter_source}"],
                "implementation_revision": "a" * 64,
                "control_seed": 0,
                "adapter_source_path": "shuffled.py",
                "adapter_source_sha256": "a" * 64,
            },
        )
        (self.root / "shuffled.py").write_bytes(b"x")
        values = {
            "adapter_manifest": str(adapter),
            "control_manifest": str(control),
            "scene_id_manifest": "builtin:sigmap-scene-id-v1",
            "shuffled_pair_manifest": str(shuffled),
            "retention_manifest": str(shuffled),
            "representation_baseline_config": str(adapter),
        }
        # shuffled/retention/control/adapter JSON are not fully valid for later
        # static validation; binding only requires regular JSON files.
        bindings, payloads = _bind_input_files(values)
        self.assertIn("adapter_manifest", bindings)
        self.assertNotIn("literature_resource_manifest", bindings)
        self.assertNotIn("external_validity_manifest", payloads)

    def test_missing_compute_plan_is_synthesized_advisory(self):
        dataset = SimpleNamespace(is_fixture=False, source_path=self.dataset_path)
        binding, report = _resolve_compute_plan(
            None, dataset, self.root / "run", {"unused-license"}
        )
        self.assertEqual(binding["kind"], "synthesized_advisory")
        self.assertEqual(len(binding["sha256"]), 64)
        self.assertEqual(report["required_gpu_count"], 0)
        self.assertEqual(report["execution_devices"], [])

    def test_placeholder_compute_plan_does_not_raise(self):
        dataset = SimpleNamespace(is_fixture=False, source_path=self.dataset_path)
        plan_path = self.root / "compute_plan_formal.template.json"
        write_json(
            plan_path,
            {
                "schema_version": "csi-pairs-full-run-compute-plan-v2",
                "profile": "formal",
                "estimated_output_bytes": 0,
                "minimum_free_disk_bytes": 0,
                "estimated_wall_time_seconds": 0,
                "authorized_wall_time_seconds": 0,
                "required_gpu_count": 2,
                "minimum_gpu_memory_bytes": 0,
                "estimated_gpu_hours": 0,
                "authorized_gpu_hours": 0,
                "component_estimates": [],
                "required_environment_variables": [
                    "CSI_PAIRS_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                ],
                "license_acknowledgements": [],
            },
        )
        binding, report = _resolve_compute_plan(
            plan_path, dataset, self.root / "run", {"unused-license"}
        )
        self.assertEqual(binding["kind"], "file")
        self.assertEqual(report["required_gpu_count"], 0)
        self.assertTrue(report["environment_variables_present"])

    def test_missing_data_verification_is_a_hard_stop_for_non_fixture(self):
        dataset = SimpleNamespace(is_fixture=False, source_path=self.dataset_path)
        with self.assertRaisesRegex(RuntimeError, "live independent data-verification"):
            require_verified_roles_from_root(
                self.root / "empty-run",
                {},
                dataset,
                ("target",),
            )
        fixture = SimpleNamespace(is_fixture=True, source_path=self.dataset_path)
        gate = require_verified_roles_from_root(
            self.root / "empty-run",
            {},
            fixture,
            ("target",),
        )
        self.assertTrue(gate.get("skipped"))
        self.assertEqual(gate["status"], "NOT_ASSESSED")

    def test_claim_package_complete_without_g0_or_g8(self):
        gates = {
            "G0": "NOT_ASSESSED",
            "G1": "PASS",
            "G2": "PASS",
            "G3": "PASS",
            "G4": "PASS",
            "G5": "PASS",
            "G6": "PASS",
            "G7": "PASS",
            "G8": "NOT_ASSESSED",
        }
        self.assertEqual(_claim_package_status(gates, {"G0": "missing"}), "COMPLETE")
        self.assertEqual(
            _claim_package_status(gates, {"G3": "broken"}),
            "INCOMPLETE_FAIL_CLOSED",
        )
        self.assertEqual(
            _claim_package_status(
                gates,
                {"wrong_map": "missing"},
                assessments={"wrong_map": "INCOMPLETE"},
            ),
            "COMPLETE",
        )
        self.assertEqual(
            _claim_package_status(
                gates,
                {},
                stage_failures=[{"stage": "formal evaluation", "error": "boom"}],
            ),
            "INCOMPLETE_FAIL_CLOSED",
        )

    def test_streaming_evaluation_requires_live_verification(self):
        from formal_v2.formal_evaluation_streaming import (
            run_streaming_formal_evaluation,
        )

        with self.assertRaisesRegex(RuntimeError, "live independent data-verification"):
            run_streaming_formal_evaluation(
                {},
                SimpleNamespace(is_fixture=False),
                self.root / "eval-run",
                upstream_root=self.root / "eval-run",
                qualification_gate_path=self.root / "q.json",
                factorial_gate_path=self.root / "f.json",
                checkpoint_index_path=self.root / "c.json",
                checkpoint_inventory_sha256="0" * 64,
                run_identity=object(),
                execution_devices=("cpu",),
                batch_size=1,
            )


if __name__ == "__main__":
    unittest.main()
