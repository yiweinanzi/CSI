from __future__ import annotations

import fcntl
import os
import tempfile
import unittest
from pathlib import Path

from formal_v2 import formal_migration_postexit_runner as runner
from formal_v2.formal_io import read_strict_json, sha256_file, write_json
from formal_v2.tests.test_formal_migration import FormalMigrationFixture


class _InjectedCrash(RuntimeError):
    pass


class PostExitMigrationRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.fixture = FormalMigrationFixture(self.root)
        freeze = self.fixture.use_post_exit_oom_freeze()
        receipt = read_strict_json(freeze)
        self.assertIsInstance(receipt, dict)
        self.receipt = receipt
        self.identity = dict(receipt["inputs"]["identity"])
        self.pid = 2_000_000_000
        self.lock = (
            self.fixture.legacy_run.parent
            / f".{self.fixture.legacy_run.name}.csi-pairs-operation.lock"
        )
        self.guard = self.lock.with_name(f"{self.lock.name}.guard")
        self.archive_root = self.root / "lock-archive"
        self.archive_root.mkdir()
        self.archive = self.archive_root / "stale-operation-lock.json"
        self.output = self.root / "post-exit-receipts"
        self._write_lock(self.pid)
        self.lock_sha256 = sha256_file(self.lock)
        self.guard.write_bytes(b"")
        self.guard_identity = self._guard_identity()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _lock_payload(self, pid: int) -> dict[str, object]:
        return {
            "output_root": str(self.fixture.legacy_run.resolve()),
            "pid": pid,
            "schema_version": runner.OPERATION_LOCK_SCHEMA,
        }

    def _write_lock(self, pid: int, path: Path | None = None) -> Path:
        target = path or self.lock
        write_json(target, self._lock_payload(pid))
        return target

    def _guard_identity(self) -> tuple[int, int, int, int, str]:
        status = self.guard.stat(follow_symlinks=False)
        return (
            status.st_dev,
            status.st_ino,
            status.st_mode,
            status.st_size,
            sha256_file(self.guard),
        )

    def _arguments(self, **changes: object) -> dict[str, object]:
        termination = self.receipt["termination"]
        values: dict[str, object] = {
            "expected_identity": self.identity,
            "legacy_run_root": self.fixture.legacy_run,
            "output_dir": self.output,
            "operation_lock_path": self.lock,
            "archive_path": self.archive,
            "expected_lock_sha256": self.lock_sha256,
            "expected_lock_pid": self.pid,
            "pid_snapshot_path": self.receipt["process_identity"][
                "identity_snapshot"
            ]["path"],
            "monitor_path": self.receipt["monitor"]["artifact"]["path"],
            "kernel_oom_evidence_path": termination[
                "kernel_cgroup_oom_evidence"
            ]["path"],
            "exit_site_path": termination["exit_site_evidence"]["path"],
            "timing_path": termination["timing_evidence"]["path"],
            "supervisor_log_path": termination["supervisor_log_evidence"][
                "path"
            ],
            "supervisor_script_path": termination[
                "supervisor_script_evidence"
            ]["path"],
            "config_path": self.fixture.config,
            "command": "python -B post-exit-runner-test",
        }
        values.update(changes)
        return values

    def _assert_untouched_precondition_failure(self) -> None:
        self.assertTrue(self.lock.is_file())
        self.assertFalse(os.path.lexists(self.archive))
        self.assertEqual(self._guard_identity(), self.guard_identity)

    def test_archives_lock_writes_receipts_and_repeats_idempotently(self) -> None:
        first = runner.run_post_exit_migration(**self._arguments())
        self.assertEqual(first["status"], "COMPLETE")
        self.assertEqual(first["archive_state"], "ARCHIVED_NOW")
        self.assertFalse(os.path.lexists(self.lock))
        self.assertTrue(self.archive.is_file())
        self.assertEqual(self._guard_identity(), self.guard_identity)
        self.assertTrue((self.output / runner.INVENTORY_NAME).is_file())
        self.assertTrue((self.output / runner.FREEZE_NAME).is_file())

        second = runner.run_post_exit_migration(
            **self._arguments(expected_lock_sha256=sha256_file(self.archive))
        )
        self.assertEqual(second["archive_state"], "ARCHIVE_ALREADY_COMPLETE")
        self.assertEqual(self._guard_identity(), self.guard_identity)

    def test_recovers_after_crash_immediately_after_hard_link(self) -> None:
        def crash() -> None:
            raise _InjectedCrash("simulated crash after durable hard link")

        with self.assertRaises(_InjectedCrash):
            runner.run_post_exit_migration(
                **self._arguments(_after_archive_link=crash)
            )
        lock_status = self.lock.stat(follow_symlinks=False)
        archive_status = self.archive.stat(follow_symlinks=False)
        self.assertEqual(
            (lock_status.st_dev, lock_status.st_ino),
            (archive_status.st_dev, archive_status.st_ino),
        )
        self.assertEqual(lock_status.st_nlink, 2)
        self.assertEqual(self._guard_identity(), self.guard_identity)

        result = runner.run_post_exit_migration(**self._arguments())
        self.assertEqual(result["archive_state"], "RECOVERED_LINKED_ARCHIVE")
        self.assertFalse(os.path.lexists(self.lock))
        self.assertEqual(self.archive.stat().st_nlink, 1)

    def test_resumes_from_completed_archive_without_canonical_lock(self) -> None:
        os.link(self.lock, self.archive, follow_symlinks=False)
        os.unlink(self.lock)
        result = runner.run_post_exit_migration(
            **self._arguments(expected_lock_sha256=sha256_file(self.archive))
        )
        self.assertEqual(result["archive_state"], "ARCHIVE_ALREADY_COMPLETE")
        self.assertEqual(self._guard_identity(), self.guard_identity)

    def test_rejects_different_archive_inode(self) -> None:
        self._write_lock(self.pid, self.archive)
        with self.assertRaisesRegex(
            runner.PostExitMigrationError, "not the same two-link inode"
        ):
            runner.run_post_exit_migration(**self._arguments())
        self.assertTrue(self.lock.is_file())
        self.assertTrue(self.archive.is_file())
        self.assertNotEqual(self.lock.stat().st_ino, self.archive.stat().st_ino)
        self.assertEqual(self._guard_identity(), self.guard_identity)

    def test_rejects_wrong_lock_sha_or_json_without_archiving(self) -> None:
        with self.assertRaisesRegex(runner.PostExitMigrationError, "SHA-256 mismatch"):
            runner.run_post_exit_migration(
                **self._arguments(expected_lock_sha256="0" * 64)
            )
        self._assert_untouched_precondition_failure()

        self.lock.write_text("not-json\n", encoding="ascii")
        with self.assertRaisesRegex(runner.PostExitMigrationError, "strict JSON"):
            runner.run_post_exit_migration(
                **self._arguments(expected_lock_sha256=sha256_file(self.lock))
            )
        self._assert_untouched_precondition_failure()

    def test_rejects_symlink_lock_without_following_it(self) -> None:
        target = self.root / "symlink-lock-target"
        self.lock.replace(target)
        self.lock.symlink_to(target)
        with self.assertRaisesRegex(
            runner.PostExitMigrationError, "non-symlink"
        ):
            runner.run_post_exit_migration(
                **self._arguments(expected_lock_sha256=sha256_file(target))
            )
        self.assertTrue(self.lock.is_symlink())
        self.assertFalse(os.path.lexists(self.archive))
        self.assertEqual(self._guard_identity(), self.guard_identity)

    def test_rejects_a_present_exact_pid(self) -> None:
        present_pid = 1
        self._write_lock(present_pid)
        with self.assertRaisesRegex(runner.PostExitMigrationError, "PID .* is present"):
            runner.run_post_exit_migration(
                **self._arguments(
                    expected_lock_pid=present_pid,
                    expected_lock_sha256=sha256_file(self.lock),
                )
            )
        self._assert_untouched_precondition_failure()

    def test_rejects_an_occupied_guard_without_touching_lock(self) -> None:
        descriptor = os.open(self.guard, os.O_RDWR)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(
                runner.PostExitMigrationError, "active operation-lock guard holder"
            ):
                runner.run_post_exit_migration(**self._arguments())
            self._assert_untouched_precondition_failure()
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def test_rejects_raw_config_sha_identity_before_touching_lock(self) -> None:
        raw_identity = dict(self.identity)
        raw_identity["config_sha256"] = sha256_file(self.fixture.config)
        self.assertNotEqual(raw_identity["config_sha256"], self.identity["config_sha256"])
        with self.assertRaisesRegex(
            runner.PostExitMigrationError, "canonical migration identity"
        ):
            runner.run_post_exit_migration(
                **self._arguments(expected_identity=raw_identity)
            )
        self._assert_untouched_precondition_failure()
        self.assertFalse(os.path.lexists(self.output))


if __name__ == "__main__":
    unittest.main()
