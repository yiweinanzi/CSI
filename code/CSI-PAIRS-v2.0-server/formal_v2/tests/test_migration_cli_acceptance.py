from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from formal_v2 import formal_migration as migration
from formal_v2 import formal_cli
from formal_v2.formal_cli import _acquire_output_lock, build_parser
from formal_v2.formal_io import write_json
from formal_v2.tests.test_formal_migration import (
    CONFIG_SHA256,
    NOW,
    SEEDS,
    FormalMigrationFixture,
)


class _FixedDateTime(datetime):
    current = NOW

    @classmethod
    def now(cls, tz=None):
        current = cls.current
        if tz is None:
            return current.replace(tzinfo=None)
        return current.astimezone(tz)


class MigrationCliAcceptanceTests(unittest.TestCase):
    @staticmethod
    def _arguments(output: Path, approval_flag: str = "--approval-manifest") -> list[str]:
        return [
            "accept-migration-request",
            "--request",
            str(output / "migration" / "request.json"),
            approval_flag,
            str(output.parent / "external-approval.json"),
            "--config",
            str(output.parent / "config.json"),
            "--dataset",
            str(output.parent / "dataset.npz"),
            "--protocol",
            str(output.parent / "protocol.md"),
            "--output",
            str(output),
        ]

    def test_parser_binds_every_acceptance_input_and_rejects_two_approvals(self) -> None:
        output = Path("/tmp/migration-cli-run")
        for approval_flag in ("--approval-manifest", "--approval"):
            with self.subTest(approval_flag=approval_flag):
                parsed = build_parser().parse_args(
                    self._arguments(output, approval_flag)
                )
                self.assertEqual(parsed.command, "accept-migration-request")
                self.assertEqual(
                    parsed.request, str(output / "migration" / "request.json")
                )
                self.assertEqual(
                    parsed.approval_manifest,
                    str(output.parent / "external-approval.json"),
                )
                self.assertEqual(parsed.config, str(output.parent / "config.json"))
                self.assertEqual(parsed.dataset, str(output.parent / "dataset.npz"))
                self.assertEqual(parsed.protocol, str(output.parent / "protocol.md"))
                self.assertEqual(parsed.output, str(output))

        ambiguous = self._arguments(output) + [
            "--approval",
            str(output.parent / "other-approval.json"),
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(ambiguous)

    def test_command_delegates_exact_bindings_while_holding_output_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "new-run"
            output.mkdir()
            config = {"seeds": [11, 22, 33]}
            config_digest = "a" * 64
            accepted_path = output / "migration" / "accepted.json"
            accepted_digest = "b" * 64

            def accept(*_args, **_kwargs):
                with self.assertRaisesRegex(FileExistsError, "operation lock exists"):
                    _acquire_output_lock(output)
                return {
                    "status": "ACCEPTED",
                    "accepted_path": str(accepted_path),
                    "accepted_sha256": accepted_digest,
                }

            stdout = io.StringIO()
            with patch(
                "formal_v2.formal_cli.load_formal_config", return_value=config
            ) as load_config, patch(
                "formal_v2.formal_evidence.config_sha256",
                return_value=config_digest,
            ) as hash_config, patch(
                "formal_v2.formal_migration.accept_migration_request",
                side_effect=accept,
            ) as accept_request, contextlib.redirect_stdout(stdout):
                status = formal_cli.main(self._arguments(output))

            self.assertEqual(status, 0)
            load_config.assert_called_once_with(str(root / "config.json"))
            hash_config.assert_called_once_with(config)
            accept_request.assert_called_once_with(
                str(output / "migration" / "request.json"),
                str(root / "external-approval.json"),
                new_run_root=output.resolve(),
                new_source_root=Path(formal_cli.__file__).resolve().parent,
                protocol_path=str(root / "protocol.md"),
                dataset_path=str(root / "dataset.npz"),
                config_sha256=config_digest,
                expected_seeds=config["seeds"],
            )
            self.assertEqual(
                json.loads(stdout.getvalue()),
                {
                    "status": "ACCEPTED",
                    "accepted_path": str(accepted_path),
                    "accepted_sha256": accepted_digest,
                },
            )

            released = _acquire_output_lock(output)
            released.release()

    def test_active_run_lock_excludes_acceptance_before_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-run"
            output.mkdir()
            held = _acquire_output_lock(output)
            stderr = io.StringIO()
            try:
                with patch(
                    "formal_v2.formal_cli.load_formal_config"
                ) as load_config, patch(
                    "formal_v2.formal_migration.accept_migration_request"
                ) as accept_request, contextlib.redirect_stderr(stderr):
                    status = formal_cli.main(self._arguments(output))
            finally:
                held.release()

            self.assertEqual(status, 2)
            self.assertIn("operation lock exists", stderr.getvalue())
            load_config.assert_not_called()
            accept_request.assert_not_called()

    def test_output_symlink_is_rejected_before_locking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target-run"
            target.mkdir()
            output = root / "linked-run"
            output.symlink_to(target, target_is_directory=True)
            stderr = io.StringIO()
            with patch(
                "formal_v2.formal_cli.load_formal_config"
            ) as load_config, contextlib.redirect_stderr(stderr):
                status = formal_cli.main(self._arguments(output))

            self.assertEqual(status, 2)
            self.assertIn("output root cannot be a symlink", stderr.getvalue())
            load_config.assert_not_called()

    def test_malformed_request_is_rejected_by_real_migration_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "new-run"
            migration_root = output / "migration"
            migration_root.mkdir(parents=True)
            (migration_root / "request.json").write_text("{}\n", encoding="ascii")
            config = {"seeds": [11, 22, 33]}
            stderr = io.StringIO()

            with patch(
                "formal_v2.formal_cli.load_formal_config", return_value=config
            ), patch(
                "formal_v2.formal_evidence.config_sha256", return_value="a" * 64
            ), contextlib.redirect_stderr(stderr):
                status = formal_cli.main(self._arguments(output))

            self.assertEqual(status, 2)
            self.assertIn("migration request fields must be exact", stderr.getvalue())
            self.assertFalse((migration_root / "accepted.json").exists())
            released = _acquire_output_lock(output)
            released.release()

    def _real_cli_arguments(
        self,
        fixture: FormalMigrationFixture,
        request: dict[str, object],
        approval: Path,
    ) -> list[str]:
        return [
            "accept-migration-request",
            "--request",
            str(request["request_path"]),
            "--approval-manifest",
            str(approval),
            "--config",
            str(fixture.root / "config.json"),
            "--dataset",
            str(fixture.dataset),
            "--protocol",
            str(fixture.protocol),
            "--output",
            str(fixture.new_run),
        ]

    def _run_real_cli(
        self,
        fixture: FormalMigrationFixture,
        arguments: list[str],
        *,
        now: datetime = NOW,
    ) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        _FixedDateTime.current = now
        with patch.object(
            formal_cli, "__file__", str(fixture.new_source / "formal_cli.py")
        ), patch.object(
            migration, "_RUNNING_SOURCE_ROOT", fixture.new_source
        ), patch.object(
            migration, "datetime", _FixedDateTime
        ), patch(
            "formal_v2.formal_cli.load_formal_config",
            return_value={"seeds": list(SEEDS)},
        ), patch(
            "formal_v2.formal_evidence.config_sha256",
            return_value=CONFIG_SHA256,
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = formal_cli.main(arguments)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_real_cli_accepts_once_and_refuses_to_overwrite_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = FormalMigrationFixture(Path(directory))
            with patch.object(
                migration, "_RUNNING_SOURCE_ROOT", fixture.new_source
            ):
                request = migration.write_migration_request(
                    **fixture.request_arguments()
                )
            approval = fixture.write_migration_approval(request)
            arguments = self._real_cli_arguments(fixture, request, approval)

            status, stdout, stderr = self._run_real_cli(fixture, arguments)
            self.assertEqual(status, 0)
            self.assertNotIn("error:", stderr)
            result = json.loads(stdout)
            self.assertEqual(result["status"], "ACCEPTED")
            self.assertEqual(
                Path(result["accepted_path"]),
                fixture.new_run / "migration" / "accepted.json",
            )

            replay_status, _stdout, replay_stderr = self._run_real_cli(
                fixture, arguments
            )
            self.assertEqual(replay_status, 2)
            self.assertIn("refusing to overwrite migration evidence", replay_stderr)

    def test_real_cli_rejects_bad_expired_or_run_internal_approval(self) -> None:
        cases = ("bad-schema", "expired", "run-internal")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = FormalMigrationFixture(Path(directory))
                with patch.object(
                    migration, "_RUNNING_SOURCE_ROOT", fixture.new_source
                ):
                    request = migration.write_migration_request(
                        **fixture.request_arguments()
                    )
                approval = fixture.write_migration_approval(request)
                now = NOW
                expected_error = ""
                if case == "bad-schema":
                    payload = json.loads(approval.read_text(encoding="utf-8"))
                    payload["schema_version"] = "wrong-migration-approval-schema"
                    write_json(approval, payload)
                    expected_error = "migration LLM approval binding mismatch"
                elif case == "expired":
                    now = datetime(2026, 8, 31, 2, 0, tzinfo=timezone.utc)
                    expected_error = "migration LLM approval chronology is invalid"
                else:
                    approval = fixture.new_run / "migration" / "approval.json"
                    fixture.write_migration_approval(request, path=approval)
                    expected_error = "must be external to both run roots"

                status, _stdout, stderr = self._run_real_cli(
                    fixture,
                    self._real_cli_arguments(fixture, request, approval),
                    now=now,
                )
                self.assertEqual(status, 2)
                self.assertIn(expected_error, stderr)
                self.assertFalse(
                    (fixture.new_run / "migration" / "accepted.json").exists()
                )


if __name__ == "__main__":
    unittest.main()
