from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from formal_v2.formal_cli import (
    _prepare_full_run,
    _run_authorized_full_chain,
    build_parser,
    main as formal_cli_main,
)
from formal_v2.formal_run_approval import (
    APPROVAL_ATTESTATION,
    APPROVAL_REQUEST_SCHEMA,
    APPROVAL_REVIEW_SCOPE,
    COMPUTE_PLAN_SCHEMA,
    HUMAN_APPROVAL_SCHEMA,
    create_human_approval_manifest,
    _bind_pre_staged_inputs,
    _merge_external_runtimes,
    _validate_prepared_record,
    _validate_request_against_current_run,
    _validate_compute_plan,
    _validate_human_approval,
    mark_approval_accepted,
)
from formal_v2.formal_io import sha256_file, write_json


class FullRunApprovalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_path = self.root / "dataset.npz"
        self.dataset_path.write_bytes(b"dataset")

    def tearDown(self):
        self.temporary.cleanup()

    def test_external_runtime_profiles_cannot_be_silently_overwritten(self):
        runtimes = {"sionna": {"environment_sha256": "a" * 64}}
        _merge_external_runtimes(
            runtimes, {"sionna": {"environment_sha256": "a" * 64}}
        )
        with self.assertRaisesRegex(RuntimeError, "conflicting sionna runtime"):
            _merge_external_runtimes(
                runtimes, {"sionna": {"environment_sha256": "b" * 64}}
            )

    def _request(self):
        return {
            "schema_version": APPROVAL_REQUEST_SCHEMA,
            "run_id": "csi-pairs-0123456789abcdef",
            "run_nonce": "0123456789abcdef" * 4,
            "prepared_utc": "2026-08-08T00:00:00Z",
            "compute_plan": {"sha256": "a" * 64},
            "gate_bindings": {
                "G0": {"gate_sha256": "b" * 64},
                "G1_G2": {"gate_sha256": "c" * 64},
            },
        }

    def _approval(self):
        request = self._request()
        return {
            "schema_version": HUMAN_APPROVAL_SCHEMA,
            "decision": "APPROVE",
            "run_id": request["run_id"],
            "run_nonce": request["run_nonce"],
            "request_sha256": "d" * 64,
            "compute_plan_sha256": "a" * 64,
            "approved_gate_sha256s": {"G0": "b" * 64, "G1_G2": "c" * 64},
            "approver": "authorized-human",
            "approved_utc": "2026-08-08T00:01:00Z",
            "expires_utc": "2026-08-09T00:00:00Z",
            "attestation": APPROVAL_ATTESTATION,
        }

    def test_human_approval_binds_request_nonce_compute_plan_and_every_gate(self):
        _validate_human_approval(
            self._approval(),
            self._request(),
            "d" * 64,
            now=datetime(2026, 8, 8, 1, tzinfo=timezone.utc),
        )

    def test_human_approval_rejects_cross_run_replay_and_stale_request(self):
        approval = self._approval()
        approval["run_nonce"] = "f" * 64
        with self.assertRaisesRegex(RuntimeError, "run_nonce"):
            _validate_human_approval(approval, self._request(), "d" * 64)
        approval = self._approval()
        with self.assertRaisesRegex(RuntimeError, "stale"):
            _validate_human_approval(approval, self._request(), "e" * 64)

    def test_human_approval_rejects_expiry_and_partial_gate_binding(self):
        approval = self._approval()
        approval["approved_gate_sha256s"].pop("G0")
        with self.assertRaisesRegex(RuntimeError, "every prepared gate"):
            _validate_human_approval(approval, self._request(), "d" * 64)
        approval = self._approval()
        with self.assertRaisesRegex(RuntimeError, "expired"):
            _validate_human_approval(
                approval,
                self._request(),
                "d" * 64,
                now=datetime(2026, 8, 10, tzinfo=timezone.utc),
            )

    def test_human_approval_rejects_future_timestamp_and_excessive_lifetime(self):
        approval = self._approval()
        with self.assertRaisesRegex(RuntimeError, "future"):
            _validate_human_approval(
                approval,
                self._request(),
                "d" * 64,
                now=datetime(2026, 8, 7, 23, 50, tzinfo=timezone.utc),
            )
        approval = self._approval()
        approval["expires_utc"] = "2026-08-09T00:01:01Z"
        with self.assertRaisesRegex(RuntimeError, "24-hour"):
            _validate_human_approval(
                approval,
                self._request(),
                "d" * 64,
                now=datetime(2026, 8, 8, 1, tzinfo=timezone.utc),
            )

    def test_compute_plan_rejects_insufficient_authorized_budget(self):
        dataset = SimpleNamespace(is_fixture=True, source_path=self.dataset_path)
        plan = self._fixture_plan()
        plan["authorized_wall_time_seconds"] = 9
        with self.assertRaisesRegex(ValueError, "authorized wall time"):
            _validate_compute_plan(plan, dataset, self.root / "run", {"fixture-license"})

    def test_formal_compute_plan_requires_real_cuda_capacity(self):
        dataset = SimpleNamespace(is_fixture=False, source_path=self.dataset_path)
        plan = self._fixture_plan()
        plan.update(
            {
                "profile": "formal",
                "required_gpu_count": 1,
                "minimum_gpu_memory_bytes": 1024,
                "estimated_gpu_hours": 2.0,
                "authorized_gpu_hours": 2.0,
            }
        )
        with patch("formal_v2.formal_run_approval._gpu_inventory", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "insufficient CUDA GPUs"):
                _validate_compute_plan(plan, dataset, self.root / "run", {"fixture-license"})

    def test_consumed_approval_marker_is_exclusive(self):
        approval_dir = self.root / "run" / "approval"
        approval_dir.mkdir(parents=True)
        record = {"schema_version": "accepted", "status": "ACCEPTED"}
        path = mark_approval_accepted(self.root / "run", record)
        self.assertTrue(path.is_file())
        with self.assertRaisesRegex(RuntimeError, "already been consumed"):
            mark_approval_accepted(self.root / "run", record)

    def test_human_approval_creation_requires_explicit_attestation_and_external_path(self):
        run = self.root / "run"
        request_path = run / "approval" / "request.json"
        request_path.parent.mkdir(parents=True)
        request = self._request()
        request["prepared_root"] = str(run.resolve())
        request["prepared_utc"] = "2026-01-01T00:00:00Z"
        write_json(request_path, request)
        expires_utc = (
            datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat().replace("+00:00", "Z")
        with self.assertRaisesRegex(RuntimeError, "attest-reviewed"):
            create_human_approval_manifest(
                request_path,
                self.root / "approval.json",
                approver="reviewer",
                expires_utc=expires_utc,
                attest_reviewed=False,
            )
        with self.assertRaisesRegex(RuntimeError, "outside"):
            create_human_approval_manifest(
                request_path,
                run / "approval.json",
                approver="reviewer",
                expires_utc=expires_utc,
                attest_reviewed=True,
            )
        result = create_human_approval_manifest(
            request_path,
            self.root / "approval.json",
            approver="reviewer",
            expires_utc=expires_utc,
            attest_reviewed=True,
        )
        self.assertEqual(result["decision"], "APPROVE")
        self.assertTrue((self.root / "approval.json").is_file())

    def test_prepared_input_mutation_changes_authenticated_inventory(self):
        run = self.root / "run"
        inputs = run / "inputs"
        inputs.mkdir(parents=True)
        sample = inputs / "formal-scenes.npz"
        sample.write_bytes(b"original")
        first = _bind_pre_staged_inputs(run, {})
        sample.write_bytes(b"mutated")
        second = _bind_pre_staged_inputs(run, {})
        self.assertNotEqual(first, second)

    def test_request_rejects_changed_external_runtime_and_input_inventory(self):
        run, preflight, request = self._bound_request()
        _validate_request_against_current_run(request, preflight, run)

        changed_runtime = dict(preflight)
        changed_runtime["external_runtime_provenance"] = {
            "wigatr": {"environment_sha256": "f" * 64}
        }
        with self.assertRaisesRegex(RuntimeError, "external_runtime_provenance"):
            _validate_request_against_current_run(request, changed_runtime, run)

        changed_inputs = dict(preflight)
        changed_inputs["input_bindings"] = {
            "prepared_inputs": {"files": [{"sha256": "f" * 64}]}
        }
        with self.assertRaisesRegex(RuntimeError, "input_bindings"):
            _validate_request_against_current_run(request, changed_inputs, run)

    def test_prepared_record_is_authenticated_to_its_exact_root(self):
        run, _preflight, request = self._bound_request()
        request_path = run / "approval" / "request.json"
        write_json(request_path, request)
        prepared = {
            "schema_version": "csi-pairs-full-run-prepared-v1",
            "run_id": request["run_id"],
            "run_nonce": request["run_nonce"],
            "prepared_root": str(run.resolve()),
            "request_path": "approval/request.json",
            "request_sha256": sha256_file(request_path),
            "status": "AWAITING_HUMAN_APPROVAL",
        }
        _validate_prepared_record(prepared, request, run.resolve(), request_path)
        with self.assertRaisesRegex(RuntimeError, "another output root"):
            _validate_prepared_record(
                prepared,
                request,
                self.root / "relocated-run",
                request_path,
            )

    def test_boolean_only_all_has_no_authorization_power(self):
        status = formal_cli_main(
            [
                "all",
                "--config",
                "config.json",
                "--output",
                "output",
                "--approve-full-experiment",
                "--verifier-manifest",
                "verifier.json",
                "--adapter-manifest",
                "adapters.json",
                "--external-validity-manifest",
                "external.json",
                "--literature-resource-manifest",
                "literature.json",
                "--rt-calibration-manifest",
                "rt.json",
            ]
        )
        self.assertEqual(status, 2)

    def test_prepare_runs_g0_and_rt_before_data_and_qualification(self):
        calls = []

        def stage(name):
            def invoke(*_args, **_kwargs):
                calls.append(name)
                return {"status": "PASS", "passed": True}

            return invoke

        args = SimpleNamespace(
            literature_resource_manifest="literature.json",
            rt_calibration_manifest="rt.json",
            verifier_manifest="verifier.json",
        )
        dataset = SimpleNamespace(is_fixture=False)
        with (
            patch("formal_v2.formal_resources.verify_waibu_resources", stage("resources")),
            patch("formal_v2.formal_literature.run_literature_resource_gate", stage("G0")),
            patch("formal_v2.formal_rt_calibration.run_rt_calibration_gate", stage("RT")),
            patch("formal_v2.formal_data_verification.run_data_verification", stage("data")),
            patch("formal_v2.formal_qualification.run_formal_qualification", stage("G1_G2")),
            patch(
                "formal_v2.formal_run_approval.write_approval_request",
                side_effect=lambda *_args, **_kwargs: calls.append("request") or {"status": "AWAITING_HUMAN_APPROVAL"},
            ),
        ):
            _prepare_full_run({}, dataset, self.root / "run", args, {})
        self.assertEqual(calls, ["resources", "G0", "RT", "data", "G1_G2", "request"])

    def test_authorized_chain_order_and_failure_short_circuit(self):
        run = self.root / "authorized-run"
        qualification = run / "qualification" / "gate.json"
        qualification.parent.mkdir(parents=True)
        write_json(qualification, {"status": "PASS", "passed": True})
        args = SimpleNamespace(
            approval_manifest="approval.json",
            allow_nonscientific_fixture=False,
            adapter_manifest="adapters.json",
            representation_baseline_config="representations.json",
            control_manifest="controls.json",
            scene_id_manifest="scene-id.json",
            external_validity_manifest="external-validity.json",
            shuffled_pair_manifest="shuffled.json",
            retention_manifest="retention.json",
        )
        dataset = SimpleNamespace(is_fixture=False)

        expected = [
            "authenticate",
            "consume",
            "wrong_map",
            "factorial",
            "evaluation",
            "risk",
            "path",
            "external",
            "representations",
            "resources",
            "scene_id",
            "external_validity",
            "shuffled",
            "retention",
            "claims",
        ]

        def execute(failing_stage=None):
            calls = []

            def stage(name):
                def invoke(*_args, **_kwargs):
                    calls.append(name)
                    if name == failing_stage:
                        return {"status": "FAIL", "passed": False}
                    return {"status": "PASS", "passed": True}

                return invoke

            with (
                patch(
                    "formal_v2.formal_run_approval.authenticate_prepared_run",
                    side_effect=stage("authenticate"),
                ),
                patch(
                    "formal_v2.formal_run_approval.mark_approval_accepted",
                    side_effect=stage("consume"),
                ),
                patch("formal_v2.formal_wrong_map.run_formal_wrong_map", side_effect=stage("wrong_map")),
                patch("formal_v2.formal_factorial.run_formal_factorial", side_effect=stage("factorial")),
                patch("formal_v2.formal_evaluation.run_formal_evaluation", side_effect=stage("evaluation")),
                patch("formal_v2.formal_risk.run_risk_contract", side_effect=stage("risk")),
                patch("formal_v2.formal_path.run_path_audit", side_effect=stage("path")),
                patch("formal_v2.formal_external.run_external_baselines", side_effect=stage("external")),
                patch(
                    "formal_v2.formal_representation_baselines.run_representation_baselines",
                    side_effect=stage("representations"),
                ),
                patch("formal_v2.formal_controls.run_resource_controls", side_effect=stage("resources")),
                patch("formal_v2.formal_scene_id.run_scene_id_audit", side_effect=stage("scene_id")),
                patch(
                    "formal_v2.formal_external_validity.run_external_validity",
                    side_effect=stage("external_validity"),
                ),
                patch(
                    "formal_v2.formal_claim_controls.run_shuffled_pair_control",
                    side_effect=stage("shuffled"),
                ),
                patch(
                    "formal_v2.formal_claim_controls.run_retention_audit",
                    side_effect=stage("retention"),
                ),
                patch("formal_v2.formal_claims.assemble_claim_evidence", side_effect=stage("claims")),
            ):
                if failing_stage is None:
                    result = _run_authorized_full_chain(
                        {}, dataset, run, args, {"status": "PASS"}
                    )
                    self.assertTrue(result["passed"])
                else:
                    with self.assertRaisesRegex(RuntimeError, "authorized chain stopped"):
                        _run_authorized_full_chain(
                            {}, dataset, run, args, {"status": "PASS"}
                        )
            return calls

        self.assertEqual(execute(), expected)
        self.assertEqual(
            execute(failing_stage="factorial"),
            expected[: expected.index("factorial") + 1],
        )

    def test_parser_exposes_prepare_and_external_approval(self):
        help_text = build_parser().format_help()
        self.assertIn("prepare-full-run", help_text)
        self.assertIn("all", help_text)

    def _fixture_plan(self):
        return {
            "schema_version": COMPUTE_PLAN_SCHEMA,
            "profile": "nonscientific_fixture",
            "estimated_output_bytes": 1024,
            "minimum_free_disk_bytes": 2048,
            "estimated_wall_time_seconds": 10,
            "authorized_wall_time_seconds": 10,
            "required_gpu_count": 0,
            "minimum_gpu_memory_bytes": 0,
            "estimated_gpu_hours": 0.0,
            "authorized_gpu_hours": 0.0,
            "required_environment_variables": [],
            "license_acknowledgements": ["fixture-license"],
        }

    def _bound_request(self):
        run = (self.root / "bound-run").resolve()
        approval_dir = run / "approval"
        approval_dir.mkdir(parents=True)
        runtime = {"schema_version": "runtime", "python_version": "3.12.11"}
        external_runtime = {
            "wigatr": {"environment_sha256": "1" * 64},
            "sionna": {"environment_sha256": "2" * 64},
        }
        compute_plan = {
            "kind": "file",
            "path": str(self.root / "compute.json"),
            "bytes": 1,
            "sha256": "3" * 64,
        }
        input_bindings = {
            "prepared_inputs": {
                "kind": "directory_inventory",
                "root": str(run / "inputs"),
                "files": [],
            }
        }
        shared = {
            "config_sha256": "4" * 64,
            "dataset_sha256": "5" * 64,
            "fixture": True,
            "source_tree_sha256": "6" * 64,
            "requirements_lock_sha256": "7" * 64,
            "runtime_provenance_sha256": hashlib.sha256(
                json.dumps(runtime, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "runtime_provenance": runtime,
            "external_runtime_provenance": external_runtime,
            "gpu_inventory": [],
            "compute_plan": compute_plan,
            "input_bindings": input_bindings,
        }
        preflight = {
            "schema_version": "csi-pairs-full-run-static-preflight-v1",
            "status": "PASS",
            "dataset_path": str(self.dataset_path),
            "prepared_root": str(run),
            "required_license_acknowledgements": [],
            "free_disk_bytes_at_preflight": 100,
            "required_environment_variables": [],
            **shared,
        }
        preflight_path = approval_dir / "preflight.json"
        write_json(preflight_path, preflight)
        request = {
            "schema_version": APPROVAL_REQUEST_SCHEMA,
            "run_id": "csi-pairs-0123456789abcdef",
            "run_nonce": "0123456789abcdef" * 4,
            "prepared_utc": "2026-08-08T00:00:00Z",
            "prepared_root": str(run),
            **shared,
            "preflight_path": "approval/preflight.json",
            "preflight_sha256": sha256_file(preflight_path),
            "gate_bindings": {"G0": {"gate_sha256": "8" * 64}},
            "teacher_checkpoint": "qualification/checkpoints/teacher.pt",
            "teacher_checkpoint_sha256": "9" * 64,
            "decision_required": "HUMAN_REVIEW_REQUIRED",
            "scientific_use": "FORBIDDEN",
            "review_scope": APPROVAL_REVIEW_SCOPE,
        }
        return run, preflight, request


class ExternalRuntimeProvenanceTests(unittest.TestCase):
    def _record(self, profile):
        from formal_v2.formal_external_runtime import (
            PROFILE_DISTRIBUTIONS,
            SCHEMA,
        )

        distributions = {
            name: {
                "name": name,
                "version": version,
                "record_sha256": "a" * 64,
            }
            for name, version in PROFILE_DISTRIBUTIONS[profile].items()
        }
        base = {
            "schema_version": SCHEMA,
            "profile": profile,
            "python_executable": "/runtime/bin/python",
            "python_prefix": "/runtime",
            "python_version": "3.10.15" if profile == "wigatr" else "3.12.11",
            "python_implementation": "CPython",
            "platform_system": "Linux",
            "platform_release": "6.8",
            "platform_machine": "x86_64",
            "cuda_visible_devices": "0" if profile == "wigatr" else None,
            "cublas_workspace_config": ":4096:8",
            "lock_files": {"lock": "b" * 64},
            "installed_distributions": distributions,
            "torch": {
                "version": "2.0.1+cu117" if profile == "wigatr" else "2.9.1+cpu",
                "cuda_version": "11.7" if profile == "wigatr" else None,
                "cudnn_version": 8500 if profile == "wigatr" else None,
                "cuda_available": profile == "wigatr",
                "deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cudnn_allow_tf32": False,
                "cuda_matmul_allow_tf32": False,
                "float32_matmul_precision": "highest",
                "nvidia_driver_versions": ["535.104"] if profile == "wigatr" else [],
                "devices": (
                    [{
                        "index": 0,
                        "name": "GPU",
                        "total_memory_bytes": 1024,
                        "compute_capability": [8, 0],
                    }]
                    if profile == "wigatr"
                    else []
                ),
            },
        }
        payload = json.dumps(
            base,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return {**base, "environment_sha256": hashlib.sha256(payload).hexdigest()}

    def test_external_runtime_digest_and_locked_versions_are_authenticated(self):
        from formal_v2.formal_external_runtime import validate_external_runtime

        record = self._record("wigatr")
        validate_external_runtime(
            record,
            profile="wigatr",
            require_execution_ready=True,
        )
        record["installed_distributions"]["torch"]["version"] = "2.0.2"
        with self.assertRaisesRegex(RuntimeError, "must be exactly"):
            validate_external_runtime(record, profile="wigatr")

    def test_wigatr_execution_readiness_requires_driver_and_cuda_state(self):
        from formal_v2.formal_external_runtime import validate_external_runtime

        record = self._record("wigatr")
        record["torch"]["nvidia_driver_versions"] = []
        without_hash = {
            key: value for key, value in record.items() if key != "environment_sha256"
        }
        record["environment_sha256"] = hashlib.sha256(
            json.dumps(
                without_hash,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        validate_external_runtime(record, profile="wigatr")
        with self.assertRaisesRegex(RuntimeError, "CUDA-ready"):
            validate_external_runtime(
                record,
                profile="wigatr",
                require_execution_ready=True,
            )


if __name__ == "__main__":
    unittest.main()
