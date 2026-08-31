from __future__ import annotations

import contextlib
import fcntl
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from formal_v2.formal_cli import (
    _acquire_legacy_read_lock,
    _reject_unsafe_mutating_output_argument,
    _require_full_stage,
    _reserve_command_output,
    _run_authorized_full_chain,
    _run_authorized_full_chain_for_upstream,
    _run_evaluation_command,
    _run_evaluation_for_upstream,
    _run_with_legacy_read_lock,
    build_parser,
    main,
)


class CliMigrationControlTests(unittest.TestCase):
    def test_optional_operator_preflight_output_is_accepted(self) -> None:
        _reject_unsafe_mutating_output_argument("operator-preflight", None)

    def test_all_rejects_raw_symlink_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "prepared-run"
            target.mkdir()
            output = root / "output-link"
            output.symlink_to(target, target_is_directory=True)

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "all",
                        "--config",
                        str(root / "missing-config.json"),
                        "--output",
                        str(output),
                        "--adapter-manifest",
                        str(root / "missing-adapters.json"),
                        "--approval-manifest",
                        str(root / "missing-approval.json"),
                    ]
                )

            self.assertEqual(status, 2)
            self.assertIn("all output root cannot be a symlink", stderr.getvalue())

    def test_migrated_evaluation_rejects_raw_symlink_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "migrated-run"
            (target / "migration").mkdir(parents=True)
            output = root / "output-link"
            output.symlink_to(target, target_is_directory=True)

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "run-evaluation",
                        "--config",
                        str(root / "missing-config.json"),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(status, 2)
            self.assertIn(
                "migrated run-evaluation output root cannot be a symlink",
                stderr.getvalue(),
            )

    def test_ordinary_command_keeps_symlink_output_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "ordinary-run"
            target.mkdir()
            output = root / "output-link"
            output.symlink_to(target, target_is_directory=True)

            _reject_unsafe_mutating_output_argument("run-risk", output)

    def test_non_migrated_evaluation_keeps_symlink_output_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "local-run"
            target.mkdir()
            output = root / "output-link"
            output.symlink_to(target, target_is_directory=True)

            _reject_unsafe_mutating_output_argument("run-evaluation", output)

    def test_legacy_evaluation_reservation_still_rejects_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evaluation").mkdir()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                _reserve_command_output("run-evaluation", root)

    @staticmethod
    def _migrated_upstream(legacy_run: Path):
        return SimpleNamespace(
            migrated=True,
            migration=SimpleNamespace(
                legacy_run_root=legacy_run,
                checkpoint_inventory_sha256="c" * 64,
            ),
            qualification_gate=legacy_run / "qualification" / "gate.json",
            factorial_gate=legacy_run / "factorial" / "gate.json",
            checkpoint_index=legacy_run / "factorial" / "checkpoint_index.json",
            factorial_root=legacy_run / "factorial",
        )

    @staticmethod
    def _write_legacy_guard(legacy_run: Path) -> Path:
        legacy_run.mkdir()
        guard = (
            legacy_run.parent
            / f".{legacy_run.name}.csi-pairs-operation.lock.guard"
        )
        guard.write_bytes(b"immutable-guard-sentinel")
        return guard

    def _assert_exclusive_guard_is_blocked(self, guard: Path) -> None:
        descriptor = os.open(guard, os.O_RDONLY)
        try:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)

    def _assert_exclusive_guard_is_available(self, guard: Path) -> None:
        descriptor = os.open(guard, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def test_evaluation_status_is_a_read_only_parser_surface(self) -> None:
        parsed = build_parser().parse_args(
            ["evaluation-status", "--output", "/tmp/evaluation-run"]
        )
        self.assertEqual(parsed.command, "evaluation-status")
        self.assertEqual(parsed.output, "/tmp/evaluation-run")
        self.assertFalse(hasattr(parsed, "dataset"))

    def test_legal_scientific_and_completion_statuses_continue(self) -> None:
        results = (
            {"status": "PASS", "passed": True},
            {"status": "FAIL", "passed": False},
            {"status": "BLOCKED", "passed": False, "engineering_complete": True},
            {"status": "COMPLETE"},
            {"status": "DIAGNOSTIC_COMPLETE_NOT_DOMAIN_EVIDENCE"},
        )
        for result in results:
            with self.subTest(status=result["status"]):
                _require_full_stage(
                    result,
                    "completed stage",
                    SimpleNamespace(is_fixture=False),
                )

    def test_engineering_incomplete_blocked_stage_stops(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "engineering-incomplete"):
            _require_full_stage(
                {"status": "BLOCKED", "passed": False},
                "external baselines",
                SimpleNamespace(is_fixture=False),
            )

    def test_invalid_incomplete_or_unknown_stage_stops(self) -> None:
        for status in ("INVALID", "INCOMPLETE_FAIL_CLOSED", "ERROR", "PARTIAL"):
            with self.subTest(status=status):
                with self.assertRaisesRegex(RuntimeError, "incomplete or invalid"):
                    _require_full_stage(
                        {"status": status},
                        "engineering stage",
                        SimpleNamespace(is_fixture=False),
                    )

    def test_inconsistent_or_malformed_stage_status_stops(self) -> None:
        results = (
            {"status": "PASS", "passed": False},
            {"status": "PASS"},
            {"status": "FAIL", "passed": True},
            {"status": "BLOCKED", "passed": True},
            {"status": "COMPLETE", "passed": True},
            {
                "status": "DIAGNOSTIC_COMPLETE_NOT_DOMAIN_EVIDENCE",
                "passed": False,
            },
        )
        for result in results:
            with self.subTest(result=result):
                with self.assertRaisesRegex(RuntimeError, "inconsistent"):
                    _require_full_stage(
                        result,
                        "inconsistent stage",
                        SimpleNamespace(is_fixture=False),
                    )
        with self.assertRaisesRegex(RuntimeError, "malformed passed"):
            _require_full_stage(
                {"status": "PASS", "passed": "yes"},
                "malformed stage",
                SimpleNamespace(is_fixture=False),
            )

    def test_legacy_shared_lock_is_read_only_and_excludes_a_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            run = parent / "legacy-run"
            run.mkdir()
            lock = parent / ".legacy-run.csi-pairs-operation.lock"
            lock.write_bytes(b'{"legacy":"owner-sentinel"}\n')
            lock_before = lock.read_bytes()
            guard = lock.with_name(f"{lock.name}.guard")
            guard.write_bytes(b"immutable-guard-sentinel")
            before = guard.read_bytes()

            shared = _acquire_legacy_read_lock(run)
            descriptor = os.open(guard, os.O_RDONLY)
            try:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(descriptor)
                shared.release()
            self.assertEqual(guard.read_bytes(), before)
            self.assertEqual(lock.read_bytes(), lock_before)

            descriptor = os.open(guard, os.O_RDONLY)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def test_legacy_shared_lock_rejects_symlink_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            run = parent / "legacy-run"
            run.mkdir()
            target = parent / "guard-target"
            target.write_bytes(b"guard")
            guard = parent / ".legacy-run.csi-pairs-operation.lock.guard"
            guard.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "guard"):
                _acquire_legacy_read_lock(run)

    def test_migrated_evaluation_dispatches_exact_streaming_identity(self) -> None:
        config = {"config": "sentinel"}
        dataset = object()
        output = Path("/new-run")
        upstream = self._migrated_upstream(Path("/legacy-run"))
        identity = object()
        execution = SimpleNamespace(
            identity=identity,
            devices=("cuda:0", "cuda:1"),
            batch_size=1,
        )
        expected = {"status": "PASS", "passed": True}

        with patch(
            "formal_v2.formal_evaluation_identity."
            "build_migrated_evaluation_execution",
            return_value=execution,
        ) as build_execution, patch(
            "formal_v2.formal_evaluation_streaming."
            "run_streaming_formal_evaluation",
            return_value=expected,
        ) as run_streaming:
            actual = _run_evaluation_for_upstream(
                config,
                dataset,
                output,
                upstream,
                None,
                None,
            )

        self.assertIs(actual, expected)
        build_execution.assert_called_once_with(config, dataset, upstream)
        run_streaming.assert_called_once_with(
            config,
            dataset,
            output,
            upstream_root=upstream.migration.legacy_run_root,
            qualification_gate_path=upstream.qualification_gate,
            factorial_gate_path=upstream.factorial_gate,
            checkpoint_index_path=upstream.checkpoint_index,
            checkpoint_inventory_sha256=(
                upstream.migration.checkpoint_inventory_sha256
            ),
            run_identity=identity,
            execution_devices=execution.devices,
            batch_size=execution.batch_size,
        )

    def test_non_migrated_evaluation_keeps_legacy_evaluator(self) -> None:
        config = {"config": "sentinel"}
        dataset = object()
        output = Path("/new-run")
        upstream = SimpleNamespace(migrated=False, migration=None)
        qualification = {"stage": "qualification"}
        factorial = {"stage": "factorial"}
        expected = {"status": "PASS", "passed": True}

        with patch(
            "formal_v2.formal_evaluation.run_formal_evaluation",
            return_value=expected,
        ) as run_legacy:
            actual = _run_evaluation_for_upstream(
                config,
                dataset,
                output,
                upstream,
                qualification,
                factorial,
            )

        self.assertIs(actual, expected)
        run_legacy.assert_called_once_with(
            config,
            dataset,
            output,
            qualification,
            factorial,
        )

    def test_run_evaluation_reauthenticates_after_window_writer_under_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            legacy_run = Path(directory) / "legacy-run"
            guard = self._write_legacy_guard(legacy_run)
            before = guard.read_bytes()
            located_upstream = self._migrated_upstream(legacy_run)
            authenticated_upstream = self._migrated_upstream(legacy_run)
            args = SimpleNamespace(qualification_gate=None, factorial_gate=None)
            expected = {"status": "PASS", "passed": True}
            mutation = legacy_run / "writer-mutation"
            resolutions = 0

            def resolve(*_args):
                nonlocal resolutions
                resolutions += 1
                if resolutions == 1:
                    descriptor = os.open(guard, os.O_RDONLY)
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        mutation.write_bytes(b"changed-after-first-authentication")
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                    finally:
                        os.close(descriptor)
                    return located_upstream
                self.assertEqual(resolutions, 2)
                self.assertEqual(
                    mutation.read_bytes(), b"changed-after-first-authentication"
                )
                self._assert_exclusive_guard_is_blocked(guard)
                return authenticated_upstream

            def run_stage(*call_args):
                self.assertEqual(
                    call_args[3:], (authenticated_upstream, None, None)
                )
                self._assert_exclusive_guard_is_blocked(guard)
                return expected

            with patch(
                "formal_v2.formal_upstream.resolve_authenticated_upstream",
                side_effect=resolve,
            ) as resolver, patch(
                "formal_v2.formal_cli._run_evaluation_for_upstream",
                side_effect=run_stage,
            ):
                actual = _run_evaluation_command(
                    {"config": "sentinel"}, object(), Path("/new-run"), args
                )

            self.assertIs(actual, expected)
            self.assertEqual(resolver.call_count, 2)
            self.assertEqual(guard.read_bytes(), before)
            self._assert_exclusive_guard_is_available(guard)

    def test_local_run_evaluation_resolves_once_without_legacy_lock(self) -> None:
        config = {"config": "sentinel"}
        dataset = object()
        upstream = SimpleNamespace(
            migrated=False,
            migration=None,
            qualification_gate=Path("/local/qualification/gate.json"),
            factorial_gate=Path("/local/factorial/gate.json"),
        )
        args = SimpleNamespace(qualification_gate=None, factorial_gate=None)
        qualification = {"stage": "qualification"}
        factorial = {"stage": "factorial"}
        expected = {"status": "PASS", "passed": True}
        with patch(
            "formal_v2.formal_upstream.resolve_authenticated_upstream",
            return_value=upstream,
        ) as resolver, patch(
            "formal_v2.formal_cli.read_strict_json",
            side_effect=(qualification, factorial),
        ), patch(
            "formal_v2.formal_cli._acquire_legacy_read_lock"
        ) as acquire, patch(
            "formal_v2.formal_cli._run_evaluation_for_upstream",
            return_value=expected,
        ) as run_stage:
            actual = _run_evaluation_command(
                config, dataset, Path("/local"), args
            )

        self.assertIs(actual, expected)
        resolver.assert_called_once()
        acquire.assert_not_called()
        run_stage.assert_called_once_with(
            config,
            dataset,
            Path("/local"),
            upstream,
            qualification,
            factorial,
        )

    def test_failed_upstream_location_never_acquires_legacy_lock(self) -> None:
        args = SimpleNamespace(qualification_gate=None, factorial_gate=None)
        with patch(
            "formal_v2.formal_upstream.resolve_authenticated_upstream",
            side_effect=RuntimeError("unauthenticated migration"),
        ), patch(
            "formal_v2.formal_cli._acquire_legacy_read_lock"
        ) as acquire:
            with self.assertRaisesRegex(RuntimeError, "unauthenticated migration"):
                _run_evaluation_command({}, object(), Path("/new-run"), args)
        acquire.assert_not_called()

    def test_run_evaluation_releases_shared_lock_after_stage_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            legacy_run = Path(directory) / "legacy-run"
            guard = self._write_legacy_guard(legacy_run)
            upstream = self._migrated_upstream(legacy_run)
            args = SimpleNamespace(qualification_gate=None, factorial_gate=None)

            def fail_stage(*_args):
                self._assert_exclusive_guard_is_blocked(guard)
                raise RuntimeError("evaluation stage failed")

            with patch(
                "formal_v2.formal_upstream.resolve_authenticated_upstream",
                return_value=upstream,
            ), patch(
                "formal_v2.formal_cli._run_evaluation_for_upstream",
                side_effect=fail_stage,
            ):
                with self.assertRaisesRegex(RuntimeError, "evaluation stage failed"):
                    _run_evaluation_command(
                        {"config": "sentinel"},
                        object(),
                        Path("/new-run"),
                        args,
                    )

            self._assert_exclusive_guard_is_available(guard)

    def test_migrated_all_releases_shared_lock_after_stage_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            legacy_run = Path(directory) / "legacy-run"
            guard = self._write_legacy_guard(legacy_run)
            before = guard.read_bytes()
            located_upstream = self._migrated_upstream(legacy_run)
            authenticated_upstream = self._migrated_upstream(legacy_run)

            def fail_full_chain(*call_args):
                self.assertIs(call_args[-1], authenticated_upstream)
                self._assert_exclusive_guard_is_blocked(guard)
                raise RuntimeError("downstream stage failed")

            with patch(
                "formal_v2.formal_upstream.resolve_authenticated_upstream",
                side_effect=(located_upstream, authenticated_upstream),
            ) as resolver, patch(
                "formal_v2.formal_cli._run_authorized_full_chain_for_upstream",
                side_effect=fail_full_chain,
            ):
                with self.assertRaisesRegex(RuntimeError, "downstream stage failed"):
                    _run_authorized_full_chain(
                        {"config": "sentinel"},
                        object(),
                        Path("/new-run"),
                        object(),
                        object(),
                    )

            self.assertEqual(resolver.call_count, 2)
            self.assertEqual(guard.read_bytes(), before)
            self._assert_exclusive_guard_is_available(guard)

    def test_migrated_all_preserves_downstream_order_and_root_separation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_run = root / "legacy-run"
            new_run = root / "new-run"
            qualification_gate = legacy_run / "qualification" / "gate.json"
            factorial_gate = legacy_run / "factorial" / "gate.json"
            qualification_gate.parent.mkdir(parents=True)
            factorial_gate.parent.mkdir(parents=True)
            new_run.mkdir()
            qualification_gate.write_text(
                '{"origin":"legacy-qualification"}\n', encoding="ascii"
            )
            factorial_gate.write_text(
                '{"origin":"legacy-factorial"}\n', encoding="ascii"
            )
            upstream = self._migrated_upstream(legacy_run)
            config = {"config": "sentinel"}
            dataset = SimpleNamespace(is_fixture=False)
            args = SimpleNamespace(
                adapter_manifest="adapters.json",
                representation_baseline_config="representations.json",
                control_manifest="controls.json",
                scene_id_manifest="scene-id.json",
                shuffled_pair_manifest="shuffled.json",
                retention_manifest="retention.json",
            )
            calls = []

            def evaluation(*call_args):
                calls.append("evaluation")
                self.assertEqual(call_args[:3], (config, dataset, new_run))
                self.assertIs(call_args[3], upstream)
                self.assertEqual(
                    call_args[4]["origin"], "legacy-qualification"
                )
                self.assertEqual(call_args[5]["origin"], "legacy-factorial")
                return {"status": "FAIL", "passed": False}

            def path(*call_args):
                calls.append("path")
                self.assertEqual(
                    call_args,
                    (config, dataset, legacy_run / "factorial", new_run),
                )
                return {"status": "PASS", "passed": True}

            def destination_stage(name, result=None):
                def invoke(*call_args):
                    calls.append(name)
                    self.assertEqual(call_args[-1], new_run)
                    self.assertNotEqual(call_args[-1], legacy_run)
                    return result or {"status": "PASS", "passed": True}

                return invoke

            with (
                patch(
                    "formal_v2.formal_cli._run_evaluation_for_upstream",
                    side_effect=evaluation,
                ),
                patch(
                    "formal_v2.formal_risk.run_risk_contract",
                    side_effect=destination_stage("risk"),
                ),
                patch("formal_v2.formal_path.run_path_audit", side_effect=path),
                patch(
                    "formal_v2.formal_external.run_external_baselines",
                    side_effect=destination_stage("external"),
                ),
                patch(
                    "formal_v2.formal_representation_baselines."
                    "run_representation_baselines",
                    side_effect=destination_stage("representations"),
                ),
                patch(
                    "formal_v2.formal_controls.run_resource_controls",
                    side_effect=destination_stage("resources"),
                ),
                patch(
                    "formal_v2.formal_scene_id.run_scene_id_audit",
                    side_effect=destination_stage("scene_id"),
                ),
                patch(
                    "formal_v2.formal_claim_controls.run_shuffled_pair_control",
                    side_effect=destination_stage("shuffled"),
                ),
                patch(
                    "formal_v2.formal_claim_controls.run_retention_audit",
                    side_effect=destination_stage("retention"),
                ),
                patch(
                    "formal_v2.formal_claims.assemble_claim_evidence",
                    side_effect=destination_stage(
                        "claims", {"status": "COMPLETE"}
                    ),
                ),
            ):
                result = _run_authorized_full_chain_for_upstream(
                    config,
                    dataset,
                    new_run,
                    args,
                    None,
                    upstream,
                )

            self.assertEqual(result, {"status": "COMPLETE"})
            self.assertEqual(
                calls,
                [
                    "evaluation",
                    "risk",
                    "path",
                    "external",
                    "representations",
                    "resources",
                    "scene_id",
                    "shuffled",
                    "retention",
                    "claims",
                ],
            )

    def test_non_migrated_execution_does_not_take_legacy_lock(self) -> None:
        upstream = SimpleNamespace(migrated=False, migration=None)
        expected = object()
        with patch(
            "formal_v2.formal_cli._acquire_legacy_read_lock"
        ) as acquire:
            actual = _run_with_legacy_read_lock(upstream, lambda: expected)
        self.assertIs(actual, expected)
        acquire.assert_not_called()


if __name__ == "__main__":
    unittest.main()
