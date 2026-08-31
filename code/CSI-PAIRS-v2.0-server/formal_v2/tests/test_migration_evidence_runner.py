from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import os
import signal
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from formal_v2 import formal_migration_evidence_runner as runner
from formal_v2.formal_evaluation_resume import (
    NO_MIGRATION_SHA256,
    EvaluationExecutionProfile,
    EvaluationRunIdentity,
    EvaluationShardIdentity,
)
from formal_v2.formal_evaluation_streaming import (
    STREAMING_EVALUATION_SCHEMA,
    evaluation_output_schema_sha256,
)
from formal_v2.formal_io import sha256_file
from formal_v2.formal_migration_evidence import validate_evidence_report
from formal_v2.tests.test_formal_migration import FormalMigrationFixture


def _failed_resume_worker(*_args: object) -> None:
    os._exit(runner.CONTROL_CHILD_FAILURE_EXIT_CODE)


def _lock_worker_never_ready(*_args: object) -> None:
    while True:
        time.sleep(1.0)


def _lock_worker_exit_before_ready(*_args: object) -> None:
    os._exit(runner.CONTROL_CHILD_FAILURE_EXIT_CODE)


def _ignore_sigterm_worker(ready_sender) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready_sender.send_bytes(b"1")
    ready_sender.close()
    while True:
        time.sleep(1.0)


class MigrationEvidenceRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.legacy = self.root / "legacy-run"
        self.new = self.root / "new-run"
        self.external = self.root / "external"
        for path in (
            self.legacy / "qualification",
            self.legacy / "factorial",
            self.new,
            self.external,
        ):
            path.mkdir(parents=True, exist_ok=True)
        (self.legacy / "qualification" / "gate.json").write_text(
            '{"status":"PASS"}\n', encoding="ascii"
        )
        (self.legacy / "factorial" / "gate.json").write_text(
            '{"status":"FAIL"}\n', encoding="ascii"
        )
        self.config = self.root / "config.json"
        self.dataset = self.root / "dataset.npz"
        self.config.write_text("{}\n", encoding="ascii")
        self.dataset.write_bytes(b"formal-dataset-test-binding")
        self.identity = {
            "legacy_run_root": str(self.legacy.resolve()),
            "new_run_root": str(self.new.resolve()),
            "legacy_commit": "1" * 40,
            "legacy_source_tree_sha256": "2" * 64,
            "new_commit": "3" * 40,
            "new_source_tree_sha256": "4" * 64,
            "requirements_lock_sha256": "5" * 64,
            "dataset_sha256": "6" * 64,
            "config_sha256": "7" * 64,
            "protocol_sha256": "8" * 64,
            "new_run_id": "csi-pairs-" + "9" * 16,
            "new_run_nonce": "9" * 64,
            "new_compute_plan_sha256": "a" * 64,
        }
        self.identity_path = self.external / "identity.json"
        self.identity_path.write_text(
            json.dumps(self.identity, sort_keys=True) + "\n", encoding="ascii"
        )

    def _run_identity(self) -> EvaluationRunIdentity:
        return EvaluationRunIdentity(
            code_revision=self.identity["new_commit"],
            source_tree_sha256=self.identity["new_source_tree_sha256"],
            config_sha256=self.identity["config_sha256"],
            dataset_sha256=self.identity["dataset_sha256"],
            migration_accepted_sha256=NO_MIGRATION_SHA256,
            legacy_checkpoint_inventory_sha256=NO_MIGRATION_SHA256,
            qualification_gate_sha256=sha256_file(
                self.legacy / "qualification" / "gate.json"
            ),
            factorial_gate_sha256=sha256_file(
                self.legacy / "factorial" / "gate.json"
            ),
            runtime_provenance_sha256="b" * 64,
            run_nonce=self.identity["new_run_nonce"],
            compute_plan_sha256=self.identity["new_compute_plan_sha256"],
            execution_profile=EvaluationExecutionProfile(
                execution_devices=("cuda:0", "cuda:1"), batch_size=1
            ),
            output_schema_id=STREAMING_EVALUATION_SCHEMA,
            output_schema_sha256=evaluation_output_schema_sha256(),
        )

    def _control_material(
        self,
    ) -> tuple[
        bytes,
        tuple[tuple[int, int, bytes], ...],
        EvaluationRunIdentity,
        tuple[EvaluationShardIdentity, ...],
    ]:
        source_payload = (
            b'{"schema_version":"real-formal-subset","rows":['
            + b'"authenticated"' * 32
            + b"]}\n"
        )
        source_sha = hashlib.sha256(source_payload).hexdigest()
        parts = runner._split_payload(source_payload)
        run = self._run_identity()
        identities = runner._shard_identities(run, source_sha, parts)
        return source_payload, parts, run, identities

    def _assert_no_new_multiprocessing_children(self, before: set[int]) -> None:
        after = {
            int(process.pid)
            for process in multiprocessing.active_children()
            if process.pid is not None
        }
        self.assertEqual(after - before, set())

    def _subset_output(self, path: Path, checkpoint_count: int) -> dict[str, object]:
        return {
            "schema_version": runner.SUBSET_REPORT_SCHEMA,
            "status": "PASS",
            "passed": True,
            "created_utc": "2026-08-31T00:00:00Z",
            "contract": {
                "dataset_kind": "formal_nonfixture_real_banks",
                "legacy_source_equivalence_assessed": False,
                "scientific_float_requirement": "IEEE-754 binary64 bitwise equality",
            },
            "inputs": {
                "dataset": {
                    "fixture": False,
                    "identity_sha256": self.identity["dataset_sha256"],
                },
                "config": {"identity_sha256": self.identity["config_sha256"]},
                "implementation": {
                    "comparator_source_tree_sha256": self.identity[
                        "new_source_tree_sha256"
                    ]
                },
            },
            "selection": {
                "checkpoint_count": checkpoint_count,
                "requested_checkpoint_count": checkpoint_count,
            },
            "comparison": {"exact": True},
            "gate_required_aggregates": {"exact": True},
            "execution": {
                "device": "cuda:0",
                "report_path": str(path.resolve()),
                "maximum_cuda_allocated_bytes": 4096,
            },
        }

    def test_expected_identity_rejects_extra_fields_and_existing_output(self) -> None:
        self.assertEqual(runner.load_expected_identity(self.identity_path), self.identity)
        mutated = {**self.identity, "unexpected": True}
        bad = self.external / "bad-identity.json"
        bad.write_text(json.dumps(mutated), encoding="ascii")
        with self.assertRaisesRegex(RuntimeError, "fields"):
            runner.load_expected_identity(bad)
        existing = self.external / "existing"
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            runner._validate_external_root(existing, self.identity)
        with self.assertRaises(RuntimeError):
            runner._validate_external_root(self.new / "inside", self.identity)

    def test_performance_runner_writes_strict_authenticated_report(self) -> None:
        runtimes = {
            "runtime_provenance_sha256": "b" * 64,
        }
        calls = []

        def fake_scale_worker(**kwargs):
            checkpoint_count = kwargs["checkpoint_count"]
            output_path = kwargs["output_path"]
            stdout_path = kwargs["stdout_path"]
            stderr_path = kwargs["stderr_path"]
            output_path.write_text(
                json.dumps(self._subset_output(output_path, checkpoint_count)) + "\n",
                encoding="ascii",
            )
            stdout_path.write_text("worker complete\n", encoding="ascii")
            stderr_path.write_text("", encoding="ascii")
            index = len(calls)
            calls.append(checkpoint_count)
            sample = {
                "observed_utc": "2026-08-31T00:00:00Z",
                "evaluator_namespace_pid": 100 + index,
                "nvidia_host_pid": 1000 + index,
                "gpu_index": 0,
                "gpu_uuid": "GPU-test",
                "gpu_utilization_percent": 70.0 + index,
                "process_vram_bytes": 4096 + index,
                "compute_process_count": 1,
            }
            return {
                "command": ["python", "-I", "-B", "real-worker"],
                "pid": 100 + index,
                "started_utc": "2026-08-31T00:00:00Z",
                "completed_utc": "2026-08-31T00:00:01Z",
                "exit_code": 0,
                "wall_seconds": float(2**index),
                "cpu_seconds": float(2**index),
                "peak_rss_bytes": 1000 * (index + 1),
                "gpu_utilization_percent": 70.0 + index,
                "gpu_utilization_statistic": "MEAN",
                "gpu_sample_count": 1,
                "gpu_sample_interval_seconds": 0.5,
                "peak_vram_bytes": 4096 + index,
                "gpu_process_binding": {
                    "method": runner.GPU_PROCESS_BINDING_METHOD,
                    "evaluator_namespace_pid": 100 + index,
                    "nvidia_host_pid": 1000 + index,
                    "gpu_index": 0,
                    "gpu_uuid": "GPU-test",
                    "baseline_compute_process_count": 0,
                    "target_gpu_empty_before_launch": True,
                    "all_observed_samples_exclusive": True,
                },
                "gpu_samples": [sample],
                "output": self._subset_output(output_path, checkpoint_count),
                "stdout": runner._binding(stdout_path),
                "stderr": runner._binding(stderr_path),
            }

        args = Namespace(
            identity=str(self.identity_path),
            config=str(self.config),
            dataset=str(self.dataset),
            output_dir=str(self.external / "performance-run"),
            device="cuda:0",
            batch_size=8,
            base_checkpoint_count=1,
            source_scenes=1,
            target_scenes_per_city=1,
            checkpoint_arm=None,
            gpu_sample_interval_seconds=0.5,
        )
        with (
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}),
            mock.patch.object(
                runner, "_production_context", return_value=({}, object(), runtimes)
            ),
            mock.patch.object(runner, "_run_scale_worker", side_effect=fake_scale_worker),
        ):
            report = runner.run_performance_evidence(args)
            with self.assertRaises(FileExistsError):
                runner.run_performance_evidence(args)
        self.assertEqual(calls, [1, 2, 4])
        self.assertEqual(report["status"], "PASS")
        validate_evidence_report(
            "performance",
            report,
            expected_identity=self.identity,
            legacy_run_root=self.legacy,
            new_run_root=self.new,
        )
        receipt = json.loads(
            (Path(args.output_dir) / "runner_receipt.json").read_text(encoding="ascii")
        )
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(len(receipt["scales"]), 3)
        control_args = Namespace(
            identity=str(self.identity_path),
            config=str(self.config),
            dataset=str(self.dataset),
            output_dir=str(self.external / "control-run"),
            performance_report=str(Path(args.output_dir) / "performance.json"),
        )
        with mock.patch.object(
            runner, "_production_context", return_value=({}, object(), runtimes)
        ):
            control_reports = runner.run_control_evidence(control_args)
        self.assertEqual(set(control_reports), set(runner.CONTROL_NAMES))
        for name, control_report in control_reports.items():
            with self.subTest(control=name):
                validate_evidence_report(
                    name,
                    control_report,
                    expected_identity=self.identity,
                    legacy_run_root=self.legacy,
                    new_run_root=self.new,
                )

    def test_performance_observation_rejects_fabricated_oracle(self) -> None:
        output = self.external / "subset.json"
        output.write_text(
            json.dumps(self._subset_output(output, 1)) + "\n", encoding="ascii"
        )
        stdout = self.external / "stdout.log"
        stderr = self.external / "stderr.log"
        stdout.write_text("ok\n", encoding="ascii")
        stderr.write_text("", encoding="ascii")
        sample = {
            "observed_utc": "2026-08-31T00:00:00Z",
            "evaluator_namespace_pid": 1,
            "nvidia_host_pid": 101,
            "gpu_index": 0,
            "gpu_uuid": "GPU-test",
            "gpu_utilization_percent": 50.0,
            "process_vram_bytes": 1024,
            "compute_process_count": 1,
        }
        observation = runner._performance_observation(
            scale="N",
            checkpoint_count=1,
            identity=self.identity,
            output_path=output,
            device="cuda:0",
            batch_size=1,
            execution={
                "command": ["python", "-B", "real-worker"],
                "pid": 1,
                "started_utc": "2026-08-31T00:00:00Z",
                "completed_utc": "2026-08-31T00:00:01Z",
                "exit_code": 0,
                "wall_seconds": 1.0,
                "cpu_seconds": 0.5,
                "peak_rss_bytes": 1000,
                "gpu_utilization_percent": 50.0,
                "gpu_utilization_statistic": "MEAN",
                "gpu_sample_count": 1,
                "gpu_sample_interval_seconds": 0.5,
                "peak_vram_bytes": 1024,
                "gpu_process_binding": {
                    "method": runner.GPU_PROCESS_BINDING_METHOD,
                    "evaluator_namespace_pid": 1,
                    "nvidia_host_pid": 101,
                    "gpu_index": 0,
                    "gpu_uuid": "GPU-test",
                    "baseline_compute_process_count": 0,
                    "target_gpu_empty_before_launch": True,
                    "all_observed_samples_exclusive": True,
                },
                "gpu_samples": [sample],
                "stdout": runner._binding(stdout),
                "stderr": runner._binding(stderr),
                "output": self._subset_output(output, 1),
            },
        )
        fabricated = copy.deepcopy(observation)
        fabricated["oracle"]["equivalent"] = False
        with self.assertRaisesRegex(RuntimeError, "oracle"):
            runner.validate_performance_observation(
                fabricated, identity=self.identity
            )
        invalid_rows = []
        extra = copy.deepcopy(observation)
        extra["gpu_samples"][0]["unexpected"] = True
        invalid_rows.append(("fields", extra))
        wrong_host_pid = copy.deepcopy(observation)
        wrong_host_pid["gpu_samples"][0]["nvidia_host_pid"] = 102
        invalid_rows.append(("sample", wrong_host_pid))
        shared_gpu = copy.deepcopy(observation)
        shared_gpu["gpu_samples"][0]["compute_process_count"] = 2
        invalid_rows.append(("sample", shared_gpu))
        nonfinite = copy.deepcopy(observation)
        nonfinite["gpu_samples"][0]["gpu_utilization_percent"] = float("nan")
        invalid_rows.append(("sample", nonfinite))
        bad_binding = copy.deepcopy(observation)
        bad_binding["gpu_process_binding"]["target_gpu_empty_before_launch"] = False
        invalid_rows.append(("binding", bad_binding))
        for message, invalid in invalid_rows:
            with self.subTest(gpu_validation=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    runner.validate_performance_observation(
                        invalid, identity=self.identity
                    )

    def test_gpu_sampling_uses_exclusive_host_process_delta(self) -> None:
        gpu_rows = [
            {
                "index": "0",
                "uuid": "GPU-test",
                "utilization_percent": "75",
                "memory_mib": "2048",
            }
        ]
        process = {
            "pid": "4242",
            "uuid": "GPU-test",
            "used_memory_mib": "1024",
        }
        with mock.patch.object(runner, "_nvidia_rows", return_value=(gpu_rows, [process])):
            sample = runner._sample_target_gpu(
                123,
                0,
                expected_gpu_uuid="GPU-test",
                expected_nvidia_host_pid=None,
            )
        self.assertEqual(sample["evaluator_namespace_pid"], 123)
        self.assertEqual(sample["nvidia_host_pid"], 4242)
        self.assertEqual(sample["compute_process_count"], 1)
        contender = {**process, "pid": "4343", "used_memory_mib": "512"}
        with mock.patch.object(
            runner, "_nvidia_rows", return_value=(gpu_rows, [process, contender])
        ):
            with self.assertRaisesRegex(RuntimeError, "not exclusive"):
                runner._sample_target_gpu(
                    123,
                    0,
                    expected_gpu_uuid="GPU-test",
                    expected_nvidia_host_pid=4242,
                )
        changed = {**process, "pid": "9999"}
        with mock.patch.object(
            runner, "_nvidia_rows", return_value=(gpu_rows, [changed])
        ):
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                runner._sample_target_gpu(
                    123,
                    0,
                    expected_gpu_uuid="GPU-test",
                    expected_nvidia_host_pid=4242,
                )

    def test_scale_worker_reaps_exact_child_when_sampling_fails(self) -> None:
        process = mock.Mock()
        process.pid = 4321
        process.returncode = None

        def wait(*, timeout):
            self.assertEqual(timeout, 10.0)
            process.returncode = -15
            return -15

        process.wait.side_effect = wait
        baseline = {
            "gpu_index": 0,
            "gpu_uuid": "GPU-test",
            "gpu_utilization_percent": 0.0,
            "device_memory_bytes": 0,
            "compute_processes": [],
        }
        with (
            mock.patch.object(runner, "_target_gpu_snapshot", return_value=baseline),
            mock.patch.object(runner.subprocess, "Popen", return_value=process),
            mock.patch.object(runner.os, "wait4", return_value=(0, 0, None)),
            mock.patch.object(
                runner,
                "_sample_target_gpu",
                side_effect=RuntimeError("sampling failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                runner._run_scale_worker(
                    config_path=self.config,
                    dataset_path=self.dataset,
                    legacy_run_root=self.legacy,
                    output_path=self.external / "failed-output.json",
                    stdout_path=self.external / "failed.stdout.log",
                    stderr_path=self.external / "failed.stderr.log",
                    device="cuda:0",
                    batch_size=1,
                    source_scenes=1,
                    target_scenes_per_city=1,
                    checkpoint_count=1,
                    checkpoint_arm="endpoint",
                    gpu_sample_interval_seconds=0.5,
                )
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=10.0)
        process.kill.assert_not_called()

    def test_scale_worker_refuses_shared_gpu_before_child_launch(self) -> None:
        occupied = {
            "gpu_index": 0,
            "gpu_uuid": "GPU-test",
            "gpu_utilization_percent": 10.0,
            "device_memory_bytes": 1024,
            "compute_processes": [
                {"nvidia_host_pid": 999, "process_vram_bytes": 1024}
            ],
        }
        with (
            mock.patch.object(runner, "_target_gpu_snapshot", return_value=occupied),
            mock.patch.object(runner.subprocess, "Popen") as popen,
        ):
            with self.assertRaisesRegex(RuntimeError, "empty exclusive target GPU"):
                runner._run_scale_worker(
                    config_path=self.config,
                    dataset_path=self.dataset,
                    legacy_run_root=self.legacy,
                    output_path=self.external / "shared-output.json",
                    stdout_path=self.external / "shared.stdout.log",
                    stderr_path=self.external / "shared.stderr.log",
                    device="cuda:0",
                    batch_size=1,
                    source_scenes=1,
                    target_scenes_per_city=1,
                    checkpoint_count=1,
                    checkpoint_arm="endpoint",
                    gpu_sample_interval_seconds=0.5,
                )
        popen.assert_not_called()

    def test_exact_child_cleanup_escalates_only_after_bounded_timeout(self) -> None:
        process = mock.Mock()
        process.returncode = None
        process.wait.side_effect = [
            runner.subprocess.TimeoutExpired("worker", 10.0),
            -9,
        ]
        runner._terminate_exact_child(process)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(
            process.wait.call_args_list,
            [mock.call(timeout=10.0), mock.call(timeout=10.0)],
        )

    def test_controls_use_spawn_and_resume_child_is_reaped(self) -> None:
        source_payload, parts, run, identities = self._control_material()
        state = self.external / "spawn-resume-state"
        state.mkdir()
        before = {
            int(process.pid)
            for process in multiprocessing.active_children()
            if process.pid is not None
        }
        real_get_context = multiprocessing.get_context
        with (
            mock.patch.object(
                runner.multiprocessing,
                "get_context",
                wraps=real_get_context,
            ) as get_context,
            mock.patch.object(
                runner.os,
                "fork",
                side_effect=AssertionError("fork must not be used"),
            ),
        ):
            results, details = runner._run_resume_control(
                state, run, identities, parts, source_payload
            )
        get_context.assert_called_once_with("spawn")
        self.assertTrue(results["output_equivalent"])
        self.assertEqual(details["interrupted_exit_code"], 75)
        self._assert_no_new_multiprocessing_children(before)

    def test_resume_failure_child_is_reaped_without_resuming(self) -> None:
        source_payload, parts, run, identities = self._control_material()
        state = self.external / "failed-resume-state"
        state.mkdir()
        before = {
            int(process.pid)
            for process in multiprocessing.active_children()
            if process.pid is not None
        }
        with mock.patch.object(
            runner, "_interrupted_writer", _failed_resume_worker
        ):
            with self.assertRaisesRegex(RuntimeError, "planned interruption"):
                runner._run_resume_control(
                    state, run, identities, parts, source_payload
                )
        self._assert_no_new_multiprocessing_children(before)

    def test_resume_start_error_reaps_only_child_with_an_assigned_pid(self) -> None:
        source_payload, parts, run, identities = self._control_material()
        state = self.external / "resume-start-error-state"
        state.mkdir()
        process = mock.Mock()
        process.pid = 4321
        process.start.side_effect = RuntimeError("spawn failed after PID assignment")
        context = mock.Mock()
        context.Process.return_value = process
        with (
            mock.patch.object(
                runner, "_control_process_context", return_value=context
            ),
            mock.patch.object(
                runner,
                "_finish_exact_control_child",
                return_value=-signal.SIGTERM,
            ) as finish,
        ):
            with self.assertRaisesRegex(RuntimeError, "PID assignment"):
                runner._run_resume_control(
                    state, run, identities, parts, source_payload
                )
        finish.assert_called_once_with(
            process,
            initial_timeout=runner.CONTROL_CHILD_JOIN_TIMEOUT_SECONDS,
        )

    def test_lock_readiness_timeout_terminates_and_reaps_exact_child(self) -> None:
        source_payload, parts, run, identities = self._control_material()
        state = self.external / "lock-timeout-state"
        state.mkdir()
        before = {
            int(process.pid)
            for process in multiprocessing.active_children()
            if process.pid is not None
        }
        with (
            mock.patch.object(
                runner, "_lock_holder", _lock_worker_never_ready
            ),
            mock.patch.object(
                runner, "CONTROL_CHILD_READY_TIMEOUT_SECONDS", 0.1
            ),
            mock.patch.object(
                runner, "CONTROL_CHILD_JOIN_TIMEOUT_SECONDS", 0.1
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "before timeout"):
                runner._run_lock_control(
                    state, run, identities[0], parts[0][2]
                )
        self._assert_no_new_multiprocessing_children(before)

    def test_lock_child_exit_before_readiness_is_reaped(self) -> None:
        _source_payload, parts, run, identities = self._control_material()
        state = self.external / "lock-early-exit-state"
        state.mkdir()
        before = {
            int(process.pid)
            for process in multiprocessing.active_children()
            if process.pid is not None
        }
        with mock.patch.object(
            runner, "_lock_holder", _lock_worker_exit_before_ready
        ):
            with self.assertRaisesRegex(RuntimeError, "before readiness"):
                runner._run_lock_control(
                    state, run, identities[0], parts[0][2]
                )
        self._assert_no_new_multiprocessing_children(before)

    def test_lock_contender_exception_releases_and_reaps_child(self) -> None:
        _source_payload, parts, run, identities = self._control_material()
        state = self.external / "lock-contender-error-state"
        state.mkdir()
        before = {
            int(process.pid)
            for process in multiprocessing.active_children()
            if process.pid is not None
        }
        with mock.patch.object(
            runner,
            "_probe_contender_lock",
            side_effect=RuntimeError("contender probe failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "contender probe failed"):
                runner._run_lock_control(
                    state, run, identities[0], parts[0][2]
                )
        self._assert_no_new_multiprocessing_children(before)
        store = runner.EvaluationResumeStore(state, run)
        with store.writer_lock():
            self.assertIsNotNone(store.load_completed_shard(identities[0]))

    def test_lock_release_failure_still_reaps_child(self) -> None:
        _source_payload, parts, run, identities = self._control_material()
        state = self.external / "lock-release-error-state"
        state.mkdir()
        before = {
            int(process.pid)
            for process in multiprocessing.active_children()
            if process.pid is not None
        }
        with mock.patch.object(
            runner,
            "_send_lock_release",
            side_effect=BrokenPipeError("release failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                runner._run_lock_control(
                    state, run, identities[0], parts[0][2]
                )
        self._assert_no_new_multiprocessing_children(before)
        store = runner.EvaluationResumeStore(state, run)
        with store.writer_lock():
            self.assertIsNotNone(store.load_completed_shard(identities[0]))

    def test_lock_start_failure_closes_every_parent_pipe_endpoint(self) -> None:
        _source_payload, parts, run, identities = self._control_material()
        state = self.external / "lock-start-error-state"
        state.mkdir()
        connections = [mock.Mock() for _index in range(4)]
        process = mock.Mock()
        process.start.side_effect = RuntimeError("spawn failed")
        context = mock.Mock()
        context.Pipe.side_effect = [
            (connections[0], connections[1]),
            (connections[2], connections[3]),
        ]
        context.Process.return_value = process
        with mock.patch.object(
            runner, "_control_process_context", return_value=context
        ):
            with self.assertRaisesRegex(RuntimeError, "spawn failed"):
                runner._run_lock_control(
                    state, run, identities[0], parts[0][2]
                )
        for connection in connections:
            connection.close.assert_called_once_with()
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_lock_second_pipe_failure_closes_first_pipe_endpoints(self) -> None:
        _source_payload, parts, run, identities = self._control_material()
        state = self.external / "lock-pipe-error-state"
        state.mkdir()
        connections = [mock.Mock(), mock.Mock()]
        context = mock.Mock()
        context.Pipe.side_effect = [
            (connections[0], connections[1]),
            OSError("second pipe failed"),
        ]
        with mock.patch.object(
            runner, "_control_process_context", return_value=context
        ):
            with self.assertRaisesRegex(OSError, "second pipe failed"):
                runner._run_lock_control(
                    state, run, identities[0], parts[0][2]
                )
        for connection in connections:
            connection.close.assert_called_once_with()
        context.Process.assert_not_called()

    def test_lock_success_closes_all_parent_pipe_endpoints(self) -> None:
        _source_payload, parts, run, identities = self._control_material()
        state = self.external / "lock-fd-success-state"
        state.mkdir()
        real_context = multiprocessing.get_context("spawn")
        connections = []
        context = mock.Mock()

        def make_pipe(*, duplex):
            endpoints = real_context.Pipe(duplex=duplex)
            connections.extend(endpoints)
            return endpoints

        context.Pipe.side_effect = make_pipe
        context.Process.side_effect = real_context.Process
        with mock.patch.object(
            runner, "_control_process_context", return_value=context
        ):
            results, details = runner._run_lock_control(
                state, run, identities[0], parts[0][2]
            )
        self.assertTrue(results["second_writer_rejected"])
        self.assertEqual(details["first_writer_exit_code"], 0)
        self.assertTrue(all(connection.closed for connection in connections))

    def test_exact_control_child_that_ignores_terminate_is_killed(self) -> None:
        context = multiprocessing.get_context("spawn")
        ready_receiver, ready_sender = context.Pipe(duplex=False)
        process = context.Process(
            name="csi-pairs-kill-fallback-test",
            target=_ignore_sigterm_worker,
            args=(ready_sender,),
        )
        process.start()
        finished = False
        try:
            ready_sender.close()
            self.assertTrue(ready_receiver.poll(120.0))
            self.assertEqual(ready_receiver.recv_bytes(), b"1")
            ready_receiver.close()
            with mock.patch.object(
                runner, "CONTROL_CHILD_JOIN_TIMEOUT_SECONDS", 0.1
            ):
                exit_code = runner._finish_exact_control_child(
                    process, initial_timeout=0.1
                )
            finished = True
        finally:
            ready_receiver.close()
            ready_sender.close()
            if not finished:
                try:
                    runner._finish_exact_control_child(
                        process, initial_timeout=0.0
                    )
                except BaseException:
                    pass
        self.assertEqual(exit_code, -signal.SIGKILL)

    def test_real_output_controls_exercise_resume_integrity_primitives(self) -> None:
        source_payload = (
            b'{"schema_version":"real-formal-subset","rows":['
            + b'"authenticated"' * 32
            + b"]}\n"
        )
        source = self.external / "real-subset-output.json"
        source.write_bytes(source_payload)
        source_sha = sha256_file(source)
        parts = runner._split_payload(source_payload)
        run = self._run_identity()
        identities = runner._shard_identities(run, source_sha, parts)
        operations = {
            "resume": lambda root: runner._run_resume_control(
                root, run, identities, parts, source_payload
            ),
            "corrupt_checkpoint": lambda root: runner._run_corrupt_control(
                root, run, identities, parts, source_payload
            ),
            "stale_checkpoint": lambda root: runner._run_stale_control(
                root, run, source_sha, parts[0]
            ),
            "lock": lambda root: runner._run_lock_control(
                root, run, identities[0], parts[0][2]
            ),
            "progress": lambda root: runner._run_progress_control(root, run),
        }
        for name, operation in operations.items():
            with self.subTest(name=name):
                state = self.external / f"{name}-state"
                state.mkdir()
                results, details = operation(state)
                observation = runner._control_observation(
                    name=name,
                    identity=self.identity,
                    run=run,
                    source_path=source,
                    state_root=state,
                    results=results,
                    details=details,
                )
                runner.validate_control_observation(
                    observation, name=name, identity=self.identity
                )
                tampered = copy.deepcopy(observation)
                tampered["identity_sha256"] = "0" * 64
                with self.assertRaisesRegex(RuntimeError, "identity"):
                    runner.validate_control_observation(
                        tampered, name=name, identity=self.identity
                    )

    def test_subset_refresh_request_changes_only_candidate_identity_fields(self) -> None:
        frozen = {
            "schema_version": runner.subset.WORKER_REQUEST_SCHEMA,
            "config_path": str(self.config.resolve()),
            "dataset_path": str(self.dataset.resolve()),
            "legacy_run_root": str(self.legacy.resolve()),
            "device": "cuda:0",
            "batch_size": 16,
            "source_scenes": 1,
            "target_scenes_per_city": 1,
            "checkpoint_count": 1,
            "checkpoint_arm": "endpoint",
            "role": "frozen_legacy",
            "fragment_path": str(self.external / "old-fragment.json"),
            "expected_git_commit": self.identity["legacy_commit"],
            "expected_evaluation_sha256": "b" * 64,
        }
        final_fragment = self.external / "final-fragment.json"
        candidate_identity = {
            "git_commit": self.identity["new_commit"],
            "files": {
                "formal_evaluation.py": {"sha256": "c" * 64},
            },
        }
        refreshed = runner._candidate_request_from_frozen(
            frozen,
            fragment_path=final_fragment,
            candidate_identity=candidate_identity,
        )
        changed = {
            key
            for key in frozen
            if frozen[key] != refreshed[key]
        }
        self.assertEqual(
            changed,
            {
                "role",
                "fragment_path",
                "expected_git_commit",
                "expected_evaluation_sha256",
            },
        )
        self.assertEqual(refreshed["role"], "streaming_candidate")
        self.assertEqual(
            refreshed["expected_git_commit"], self.identity["new_commit"]
        )
        malformed = dict(frozen)
        malformed["unexpected"] = True
        with self.assertRaisesRegex(RuntimeError, "fields"):
            runner._candidate_request_from_frozen(
                malformed,
                fragment_path=final_fragment,
                candidate_identity=candidate_identity,
            )

    def test_prior_subset_reuse_reauthenticates_frozen_fragment_and_live_harness(self) -> None:
        fixture_root = self.root / "migration-fixture"
        fixture_root.mkdir()
        fixture = FormalMigrationFixture(fixture_root)
        identity = json.loads(
            fixture.migration_evidence["performance"].read_text(encoding="ascii")
        )["inputs"]["identity"]
        report_path = fixture.external_root / "prior-real-subset.json"
        fixture.write_real_subset_report(report_path, identity=identity)
        harness = fixture.new_source / "formal_evaluation_subset_compare.py"
        with mock.patch.object(runner.subset, "__file__", str(harness)):
            report, request, fragment = runner._authenticate_prior_subset_report(
                report_path,
                identity=identity,
                frozen_source_root=fixture.legacy_source.parent,
            )
        self.assertTrue(report["passed"])
        self.assertEqual(request["role"], "frozen_legacy")
        self.assertEqual(fragment["role"], "frozen_legacy")
        harness.write_text("COMPARATOR_HARNESS = 'changed'\n", encoding="ascii")
        with mock.patch.object(runner.subset, "__file__", str(harness)):
            with self.assertRaises(RuntimeError):
                runner._authenticate_prior_subset_report(
                    report_path,
                    identity=identity,
                    frozen_source_root=fixture.legacy_source.parent,
                )


if __name__ == "__main__":
    unittest.main()
