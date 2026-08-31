from __future__ import annotations

import hashlib
import itertools
import json
import tempfile
import threading
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from formal_v2.formal_evaluation_resume import (
    CorruptShardError,
    EvaluationExecutionProfile,
    EvaluationResumeStore,
    EvaluationResumeError,
    EvaluationRunIdentity,
    EvaluationShardIdentity,
    FINAL_OUTPUT_QUARANTINE_SCHEMA,
    IncompleteShardError,
    NO_MIGRATION_SHA256,
    ProgressValidationError,
    PROGRESS_RECONCILIATION_SCHEMA,
    QUARANTINE_RECEIPT_SCHEMA,
    RUN_IDENTITY_SCHEMA,
    SHARD_MANIFEST_SCHEMA,
    ShardRange,
    StaleResumeError,
    WriterLockError,
    iter_shard_ranges,
    read_evaluation_status,
    write_atomic_json,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class EvaluationResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "resume"
        self.run = EvaluationRunIdentity(
            code_revision="a" * 40,
            source_tree_sha256=_digest("source-tree"),
            config_sha256=_digest("config"),
            dataset_sha256=_digest("dataset"),
            migration_accepted_sha256=_digest("migration-accepted"),
            legacy_checkpoint_inventory_sha256=_digest("legacy-checkpoint-inventory"),
            qualification_gate_sha256=_digest("qualification-gate"),
            factorial_gate_sha256=_digest("factorial-gate"),
            runtime_provenance_sha256=_digest("runtime-provenance"),
            run_nonce=_digest("run-nonce"),
            compute_plan_sha256=_digest("compute-plan"),
            execution_profile=EvaluationExecutionProfile(
                execution_devices=("cuda:0", "cuda:1"),
                batch_size=1,
            ),
            output_schema_id="formal-evaluation-v6",
            output_schema_sha256=_digest("output-schema"),
        )
        self.identity = EvaluationShardIdentity(
            run=self.run,
            checkpoint_sha256=_digest("checkpoint"),
            seed=20270001,
            arm="full",
            substage="response-effects",
            shard_id="scene-000-edge-000",
            range_start=0,
            range_stop=7,
        )
        self.store = EvaluationResumeStore(self.root, self.run)

    def test_run_identity_strongly_binds_every_upstream_authority(self):
        payload = self.run.as_dict()
        self.assertEqual(payload["schema_version"], RUN_IDENTITY_SCHEMA)
        self.assertEqual(EvaluationRunIdentity.from_dict(payload), self.run)
        bound_fields = (
            "migration_accepted_sha256",
            "legacy_checkpoint_inventory_sha256",
            "qualification_gate_sha256",
            "factorial_gate_sha256",
            "runtime_provenance_sha256",
            "run_nonce",
            "compute_plan_sha256",
        )
        for field in bound_fields:
            with self.subTest(field=field):
                changed = replace(self.run, **{field: _digest(f"changed-{field}")})
                self.assertNotEqual(changed.sha256, self.run.sha256)

        missing = dict(payload)
        missing.pop("runtime_provenance_sha256")
        with self.assertRaises(ValueError):
            EvaluationRunIdentity.from_dict(missing)
        extra = {**payload, "unexpected": _digest("unexpected")}
        with self.assertRaises(ValueError):
            EvaluationRunIdentity.from_dict(extra)

    def test_execution_profile_change_is_stale_without_changing_output_schema(self):
        with self.store.writer_lock():
            self.store.commit_shard_bytes(self.identity, b"trusted")
            self.store.initialize_progress(1)

        changed_profiles = (
            EvaluationExecutionProfile(
                execution_devices=("cuda:1", "cuda:0"),
                batch_size=1,
            ),
            EvaluationExecutionProfile(
                execution_devices=("cuda:0", "cuda:1"),
                batch_size=2,
            ),
        )
        for changed_profile in changed_profiles:
            with self.subTest(profile=changed_profile):
                changed_run = replace(
                    self.run,
                    execution_profile=changed_profile,
                )
                self.assertNotEqual(changed_run.sha256, self.run.sha256)
                self.assertEqual(
                    changed_run.output_schema_id,
                    self.run.output_schema_id,
                )
                self.assertEqual(
                    changed_run.output_schema_sha256,
                    self.run.output_schema_sha256,
                )
                changed_store = EvaluationResumeStore(self.root, changed_run)
                changed_shard = replace(self.identity, run=changed_run)
                with self.assertRaises(StaleResumeError):
                    changed_store.load_completed_shard(changed_shard)
                with self.assertRaises(StaleResumeError):
                    changed_store.read_status()

    def test_execution_profile_serialization_is_strict_and_authenticated(self):
        payload = self.run.as_dict()
        tampered = dict(payload)
        tampered["execution_profile_sha256"] = _digest("wrong-profile")
        with self.assertRaisesRegex(ValueError, "profile digest"):
            EvaluationRunIdentity.from_dict(tampered)
        malformed = dict(payload)
        malformed["execution_profile"] = {
            **payload["execution_profile"],
            "execution_devices": ["cuda:00"],
        }
        with self.assertRaisesRegex(ValueError, "canonical CUDA"):
            EvaluationRunIdentity.from_dict(malformed)

    def test_no_migration_requires_an_explicit_paired_sentinel(self):
        no_migration = replace(
            self.run,
            migration_accepted_sha256=NO_MIGRATION_SHA256,
            legacy_checkpoint_inventory_sha256=NO_MIGRATION_SHA256,
        )
        self.assertEqual(
            EvaluationRunIdentity.from_dict(no_migration.as_dict()), no_migration
        )
        with self.assertRaisesRegex(ValueError, "both be bound"):
            replace(self.run, migration_accepted_sha256=NO_MIGRATION_SHA256)
        with self.assertRaisesRegex(ValueError, "cannot use"):
            replace(self.run, qualification_gate_sha256=NO_MIGRATION_SHA256)
        with self.assertRaisesRegex(ValueError, "run_nonce"):
            replace(self.run, run_nonce="not-a-64-character-lowercase-hex-nonce")

    def test_each_upstream_authority_change_is_stale_for_shard_and_progress(self):
        with self.store.writer_lock():
            self.store.commit_shard_bytes(self.identity, b"trusted")
            self.store.initialize_progress(1)
        bound_fields = (
            "migration_accepted_sha256",
            "legacy_checkpoint_inventory_sha256",
            "qualification_gate_sha256",
            "factorial_gate_sha256",
            "runtime_provenance_sha256",
            "run_nonce",
            "compute_plan_sha256",
        )
        for field in bound_fields:
            with self.subTest(field=field):
                changed_run = replace(
                    self.run, **{field: _digest(f"stale-{field}")}
                )
                changed_store = EvaluationResumeStore(self.root, changed_run)
                changed_shard = replace(self.identity, run=changed_run)
                with self.assertRaises(StaleResumeError):
                    changed_store.load_completed_shard(changed_shard)
                with self.assertRaises(StaleResumeError):
                    changed_store.read_status()

    def test_empty_and_non_divisible_shard_plans(self):
        self.assertEqual(list(iter_shard_ranges(0, 4)), [])
        self.assertEqual(
            list(iter_shard_ranges(10, 4)),
            [
                ShardRange(index=0, start=0, stop=4),
                ShardRange(index=1, start=4, stop=8),
                ShardRange(index=2, start=8, stop=10),
            ],
        )
        with self.assertRaises(ValueError):
            list(iter_shard_ranges(-1, 4))
        with self.assertRaises(ValueError):
            list(iter_shard_ranges(1, 0))

    def test_shard_plan_is_lazy_and_does_not_materialize_pairwise_work(self):
        plan = iter_shard_ranges(10**18, 3)
        self.assertEqual(
            list(itertools.islice(plan, 4)),
            [
                ShardRange(index=0, start=0, stop=3),
                ShardRange(index=1, start=3, stop=6),
                ShardRange(index=2, start=6, stop=9),
                ShardRange(index=3, start=9, stop=12),
            ],
        )

    def test_atomic_commit_binds_every_identity_and_authenticates_payload(self):
        chunks = [b"header\n", b"a,b\n", b"1,2\n"]

        def write_chunks(handle):
            for chunk in chunks:
                handle.write(chunk)

        completed_at = datetime(2026, 8, 31, 1, 2, 3, tzinfo=timezone.utc)
        with self.store.writer_lock():
            result = self.store.commit_shard(
                self.identity, write_chunks, suffix=".csv", now=completed_at
            )
        self.assertFalse(result.reused)
        self.assertEqual(result.payload_path.read_bytes(), b"".join(chunks))
        manifest = json.loads(result.manifest_path.read_text(encoding="ascii"))
        self.assertEqual(manifest["schema_version"], SHARD_MANIFEST_SCHEMA)
        self.assertEqual(manifest["status"], "COMPLETE")
        self.assertEqual(manifest["identity"], self.identity.as_dict())
        self.assertEqual(manifest["identity_sha256"], self.identity.sha256)
        self.assertEqual(
            manifest["identity"]["run"]["source_tree_sha256"],
            self.run.source_tree_sha256,
        )
        self.assertEqual(
            manifest["identity"]["run"]["config_sha256"], self.run.config_sha256
        )
        self.assertEqual(
            manifest["identity"]["run"]["dataset_sha256"], self.run.dataset_sha256
        )
        self.assertEqual(
            manifest["identity"]["checkpoint_sha256"],
            self.identity.checkpoint_sha256,
        )
        self.assertEqual(manifest["identity"]["range_start"], 0)
        self.assertEqual(manifest["identity"]["range_stop"], 7)
        self.assertEqual(
            manifest["identity"]["run"]["output_schema_sha256"],
            self.run.output_schema_sha256,
        )
        loaded = self.store.load_completed_shard(self.identity, suffix=".csv")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertTrue(loaded.reused)
        self.assertEqual(loaded.manifest, manifest)
        self.assertEqual(self._temporary_files(), [])

    def test_failed_writer_publishes_nothing_and_removes_its_temporary_file(self):
        def fail_after_partial_write(handle):
            handle.write(b"partial")
            raise RuntimeError("simulated interruption")

        with self.store.writer_lock():
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.store.commit_shard(self.identity, fail_after_partial_write)
        payload, manifest = self.store.shard_paths(self.identity)
        self.assertFalse(payload.exists())
        self.assertFalse(manifest.exists())
        self.assertEqual(self._temporary_files(), [])

    def test_invalid_timestamp_is_rejected_before_the_payload_writer_runs(self):
        called = False

        def writer(handle):
            nonlocal called
            called = True
            handle.write(b"must-not-run")

        with self.store.writer_lock():
            with self.assertRaises(ValueError):
                self.store.commit_shard(
                    self.identity,
                    writer,
                    now=datetime(2026, 8, 31),
                )
        payload, manifest = self.store.shard_paths(self.identity)
        self.assertFalse(called)
        self.assertFalse(payload.exists())
        self.assertFalse(manifest.exists())

    def test_payload_without_manifest_and_orphan_temporary_file_are_rejected(self):
        payload, manifest = self.store.shard_paths(self.identity)
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b"half-written")
        with self.assertRaises(IncompleteShardError):
            self.store.load_completed_shard(self.identity)
        payload.unlink()
        temporary = payload.parent / f".{payload.name}.crashed.tmp"
        temporary.write_bytes(b"half-written")
        self.assertFalse(manifest.exists())
        with self.assertRaises(IncompleteShardError):
            self.store.load_completed_shard(self.identity)

    def test_manifest_without_payload_is_rejected(self):
        with self.store.writer_lock():
            result = self.store.commit_shard_bytes(self.identity, b"trusted")
        result.payload_path.unlink()
        with self.assertRaises(IncompleteShardError):
            self.store.load_completed_shard(self.identity)

    def test_corrupt_payload_is_rejected_by_size_and_sha(self):
        with self.store.writer_lock():
            result = self.store.commit_shard_bytes(self.identity, b"trusted")
        result.payload_path.write_bytes(b"changed")
        with self.assertRaises(CorruptShardError):
            self.store.load_completed_shard(self.identity)

    def test_corrupt_manifest_is_rejected(self):
        with self.store.writer_lock():
            result = self.store.commit_shard_bytes(self.identity, b"trusted")
        result.manifest_path.write_text("{not-json", encoding="ascii")
        with self.assertRaises(CorruptShardError):
            self.store.load_completed_shard(self.identity)

    def test_corrupt_payload_is_quarantined_and_only_that_shard_is_recomputed(self):
        sibling = replace(
            self.identity,
            shard_id="scene-001-edge-000",
            range_start=7,
            range_stop=14,
        )
        with self.store.writer_lock():
            damaged = self.store.commit_shard_bytes(self.identity, b"trusted-a")
            preserved = self.store.commit_shard_bytes(sibling, b"trusted-b")
        damaged.payload_path.write_bytes(b"corrupt-a")
        preserved_mtime = preserved.payload_path.stat().st_mtime_ns
        with self.store.writer_lock():
            self.assertIsNone(self.store.load_or_quarantine_shard(self.identity))
            loaded_sibling = self.store.load_or_quarantine_shard(sibling)
            replacement = self.store.commit_shard_bytes(
                self.identity, b"recomputed-a"
            )
        self.assertIsNotNone(loaded_sibling)
        self.assertTrue(loaded_sibling.reused)
        self.assertEqual(loaded_sibling.payload_path.read_bytes(), b"trusted-b")
        self.assertEqual(loaded_sibling.payload_path.stat().st_mtime_ns, preserved_mtime)
        self.assertEqual(replacement.payload_path.read_bytes(), b"recomputed-a")
        receipts = self._quarantine_receipts()
        self.assertEqual(len(receipts), 1)
        receipt = receipts[0]
        self.assertEqual(receipt["schema_version"], QUARANTINE_RECEIPT_SCHEMA)
        self.assertEqual(receipt["status"], "COMPLETE")
        self.assertEqual(receipt["reason"]["class"], "CorruptShardError")
        self.assertEqual(receipt["shard_identity_sha256"], self.identity.sha256)
        self.assertEqual(len(receipt["files"]), 2)
        for row in receipt["files"]:
            quarantined = self.root / row["quarantine_path"]
            self.assertTrue(quarantined.exists())
            self.assertEqual(quarantined.stat().st_size, row["bytes"])
            self.assertEqual(hashlib.sha256(quarantined.read_bytes()).hexdigest(), row["sha256"])

    def test_incomplete_payload_manifest_and_temp_are_quarantined_for_recompute(self):
        for index, failure in enumerate(("payload-only", "manifest-only", "temp-only")):
            with self.subTest(failure=failure):
                root = Path(self.temporary.name) / f"incomplete-{index}"
                store = EvaluationResumeStore(root, self.run)
                identity = replace(self.identity, shard_id=f"incomplete-{index}")
                payload, manifest = store.shard_paths(identity)
                if failure == "payload-only":
                    payload.parent.mkdir(parents=True)
                    payload.write_bytes(b"orphan-payload")
                elif failure == "manifest-only":
                    with store.writer_lock():
                        completed = store.commit_shard_bytes(identity, b"will-vanish")
                    completed.payload_path.unlink()
                else:
                    payload.parent.mkdir(parents=True)
                    (payload.parent / f".{payload.name}.crashed.tmp").write_bytes(
                        b"orphan-temp"
                    )
                with store.writer_lock():
                    self.assertIsNone(store.load_or_quarantine_shard(identity))
                    replacement = store.commit_shard_bytes(identity, b"recomputed")
                self.assertEqual(replacement.payload_path.read_bytes(), b"recomputed")
                receipts = sorted(
                    root.glob("quarantine/*/receipt.json")
                )
                self.assertEqual(len(receipts), 1)
                receipt = json.loads(receipts[0].read_text(encoding="ascii"))
                self.assertEqual(receipt["reason"]["class"], "IncompleteShardError")
                self.assertEqual(len(receipt["files"]), 1)

    def test_corrupt_manifest_is_quarantined_for_recompute(self):
        with self.store.writer_lock():
            completed = self.store.commit_shard_bytes(self.identity, b"trusted")
        completed.manifest_path.write_text("{broken", encoding="ascii")
        with self.store.writer_lock():
            self.assertIsNone(self.store.load_or_quarantine_shard(self.identity))
            replacement = self.store.commit_shard_bytes(self.identity, b"recomputed")
        self.assertEqual(replacement.payload_path.read_bytes(), b"recomputed")
        receipt = self._quarantine_receipts()[0]
        self.assertEqual(receipt["reason"]["class"], "CorruptShardError")

    def test_stale_shard_with_temporary_residue_is_not_quarantined_or_moved(self):
        with self.store.writer_lock():
            completed = self.store.commit_shard_bytes(self.identity, b"trusted")
        temporary = completed.payload_path.parent / (
            f".{completed.payload_path.name}.stale.tmp"
        )
        temporary.write_bytes(b"residue")
        changed_run = replace(self.run, config_sha256=_digest("stale-config"))
        changed_identity = replace(self.identity, run=changed_run)
        changed_store = EvaluationResumeStore(self.root, changed_run)
        with changed_store.writer_lock():
            with self.assertRaises(StaleResumeError):
                changed_store.load_or_quarantine_shard(changed_identity)
        self.assertTrue(completed.payload_path.is_file())
        self.assertTrue(completed.manifest_path.is_file())
        self.assertTrue(temporary.is_file())
        self.assertFalse(changed_store.quarantine_root.exists())

    def test_quarantine_requires_the_writer_lock(self):
        with self.assertRaises(WriterLockError):
            self.store.load_or_quarantine_shard(self.identity)

    def test_changed_checkpoint_or_run_identity_is_rejected_as_stale(self):
        with self.store.writer_lock():
            self.store.commit_shard_bytes(self.identity, b"trusted")
        new_checkpoint = EvaluationShardIdentity(
            run=self.run,
            checkpoint_sha256=_digest("new-checkpoint"),
            seed=self.identity.seed,
            arm=self.identity.arm,
            substage=self.identity.substage,
            shard_id=self.identity.shard_id,
            range_start=self.identity.range_start,
            range_stop=self.identity.range_stop,
        )
        with self.assertRaises(StaleResumeError):
            self.store.load_completed_shard(new_checkpoint)

        changed_run = replace(self.run, config_sha256=_digest("new-config"))
        changed_store = EvaluationResumeStore(self.root, changed_run)
        changed_identity = EvaluationShardIdentity(
            run=changed_run,
            checkpoint_sha256=self.identity.checkpoint_sha256,
            seed=self.identity.seed,
            arm=self.identity.arm,
            substage=self.identity.substage,
            shard_id=self.identity.shard_id,
            range_start=self.identity.range_start,
            range_stop=self.identity.range_stop,
        )
        with self.assertRaises(StaleResumeError):
            changed_store.load_completed_shard(changed_identity)

    def test_duplicate_resume_is_idempotent_and_does_not_call_writer(self):
        calls = 0

        def writer(handle):
            nonlocal calls
            calls += 1
            handle.write(b"once")

        with self.store.writer_lock():
            first = self.store.commit_shard(self.identity, writer)
            first_payload_mtime = first.payload_path.stat().st_mtime_ns
            first_manifest_mtime = first.manifest_path.stat().st_mtime_ns
            second = self.store.commit_shard(self.identity, writer)
        self.assertEqual(calls, 1)
        self.assertFalse(first.reused)
        self.assertTrue(second.reused)
        self.assertEqual(second.payload_path.stat().st_mtime_ns, first_payload_mtime)
        self.assertEqual(second.manifest_path.stat().st_mtime_ns, first_manifest_mtime)
        self.assertEqual(first.manifest, second.manifest)

    def test_atomic_json_replace_failure_preserves_previous_file(self):
        output = self.root.parent / "evaluation"
        target = output / "gate.json"
        write_atomic_json(target, {"status": "OLD"})
        previous = target.read_bytes()
        with mock.patch(
            "formal_v2.formal_evaluation_resume.os.replace",
            side_effect=OSError("simulated final JSON replace failure"),
        ):
            with self.assertRaisesRegex(OSError, "final JSON replace failure"):
                write_atomic_json(target, {"status": "NEW"})
        self.assertEqual(target.read_bytes(), previous)
        self.assertEqual(list(output.glob(f".{target.name}.*.tmp")), [])

    def test_final_output_temporaries_are_quarantined_with_bound_receipt(self):
        output = self.root.parent / "evaluation"
        output.mkdir()
        targets = (output / "a.csv", output / "b.csv")
        first = output / ".a.csv.worker.tmp"
        second = output / ".b.csv.worker.tmp"
        unrelated = output / ".other.csv.worker.tmp"
        first.write_bytes(b"partial-a")
        second.write_bytes(b"partial-b")
        unrelated.write_bytes(b"unrelated")

        with self.store.writer_lock():
            receipt_path = self.store.quarantine_final_output_temporaries(targets)
        self.assertIsNotNone(receipt_path)
        assert receipt_path is not None
        self.assertFalse(first.exists())
        self.assertFalse(second.exists())
        self.assertEqual(unrelated.read_bytes(), b"unrelated")
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
        self.assertEqual(receipt["schema_version"], FINAL_OUTPUT_QUARANTINE_SCHEMA)
        self.assertEqual(receipt["status"], "COMPLETE")
        self.assertEqual(receipt["run_identity"], self.run.as_dict())
        self.assertEqual(receipt["run_identity_sha256"], self.run.sha256)
        self.assertEqual(
            [row["original_path"] for row in receipt["files"]],
            ["evaluation/.a.csv.worker.tmp", "evaluation/.b.csv.worker.tmp"],
        )
        preserved = [
            (self.root / row["quarantine_path"]).read_bytes()
            for row in receipt["files"]
        ]
        self.assertEqual(preserved, [b"partial-a", b"partial-b"])

    def test_final_output_quarantine_fails_closed_on_symlink_or_nonfile(self):
        output = self.root.parent / "evaluation"
        output.mkdir()
        target = output / "a.csv"
        victim = self.root.parent / "victim"
        victim.write_bytes(b"preserve")
        linked_temporary = output / ".a.csv.link.tmp"
        linked_temporary.symlink_to(victim)
        with self.store.writer_lock():
            with self.assertRaisesRegex(EvaluationResumeError, "regular file"):
                self.store.quarantine_final_output_temporaries((target,))
        self.assertTrue(linked_temporary.is_symlink())
        self.assertEqual(victim.read_bytes(), b"preserve")

        linked_temporary.unlink()
        directory_temporary = output / ".a.csv.directory.tmp"
        directory_temporary.mkdir()
        with self.store.writer_lock():
            with self.assertRaisesRegex(EvaluationResumeError, "regular file"):
                self.store.quarantine_final_output_temporaries((target,))
        self.assertTrue(directory_temporary.is_dir())

        directory_temporary.rmdir()
        target.symlink_to(victim)
        with self.store.writer_lock():
            with self.assertRaisesRegex(EvaluationResumeError, "regular file"):
                self.store.quarantine_final_output_temporaries((target,))
        self.assertTrue(target.is_symlink())
        self.assertEqual(victim.read_bytes(), b"preserve")

    def test_semantic_quarantine_rejects_stale_shard_identity(self):
        with self.store.writer_lock():
            completed = self.store.commit_shard_bytes(self.identity, b"trusted")
        changed_run = replace(self.run, run_nonce=_digest("different-run-nonce"))
        changed_store = EvaluationResumeStore(self.root, changed_run)
        changed_identity = replace(self.identity, run=changed_run)
        with changed_store.writer_lock():
            with self.assertRaises(StaleResumeError):
                changed_store.quarantine_completed_shard(
                    changed_identity,
                    reason="dependent final output failed validation",
                )
        self.assertEqual(completed.payload_path.read_bytes(), b"trusted")
        self.assertTrue(completed.manifest_path.is_file())

    def test_mutations_require_lock_and_second_writer_is_refused(self):
        with self.assertRaises(WriterLockError):
            self.store.commit_shard_bytes(self.identity, b"unlocked")
        with self.assertRaises(WriterLockError):
            self.store.initialize_progress(1)

        second = EvaluationResumeStore(self.root, self.run)
        with self.store.writer_lock():
            with self.assertRaises(WriterLockError):
                with second.writer_lock():
                    self.fail("second writer unexpectedly acquired the lock")
        with second.writer_lock():
            second.initialize_progress(1)

    def test_writer_lock_is_reentrant_and_refuses_a_symlink_lock_file(self):
        with self.store.writer_lock():
            with self.store.writer_lock():
                self.store.initialize_progress(1)
        self.assertIsNotNone(self.store.read_status())

        other_root = Path(self.temporary.name) / "symlink-lock"
        other_root.mkdir()
        victim = Path(self.temporary.name) / "victim"
        victim.write_bytes(b"preserve")
        (other_root / ".evaluation-writer.lock").symlink_to(victim)
        linked_store = EvaluationResumeStore(other_root, self.run)
        with self.assertRaises(WriterLockError):
            with linked_store.writer_lock():
                self.fail("symlink lock file unexpectedly acquired")
        self.assertEqual(victim.read_bytes(), b"preserve")

    def test_progress_is_atomic_authentic_and_reports_real_rate_and_eta(self):
        started = datetime(2026, 8, 31, 0, 0, 0, tzinfo=timezone.utc)
        with self.store.writer_lock():
            initial = self.store.initialize_progress(10, now=started)
            duplicate = self.store.initialize_progress(
                10, now=started + timedelta(seconds=1)
            )
            running = self.store.update_progress(
                4,
                current_seed=self.identity.seed,
                current_arm=self.identity.arm,
                current_substage=self.identity.substage,
                current_shard=self.identity.shard_id,
                now=started + timedelta(seconds=5),
            )
        self.assertEqual(initial, duplicate)
        self.assertEqual(running["status"], "RUNNING")
        self.assertEqual(running["total_units"], 10)
        self.assertEqual(running["completed_units"], 4)
        self.assertEqual(running["percentage"], 40.0)
        self.assertEqual(running["throughput_units_per_second"], 0.8)
        self.assertEqual(running["eta_seconds"], 7.5)
        self.assertEqual(running["current_seed"], self.identity.seed)
        self.assertEqual(running["current_arm"], self.identity.arm)
        self.assertEqual(running["current_substage"], self.identity.substage)
        self.assertEqual(running["current_shard"], self.identity.shard_id)

        before = self.store.progress_path.stat().st_mtime_ns
        status = read_evaluation_status(self.root, expected_identity=self.run)
        after = self.store.progress_path.stat().st_mtime_ns
        self.assertEqual(status, running)
        self.assertEqual(before, after)
        self.assertEqual(self._temporary_files(), [])

        with self.store.writer_lock():
            complete = self.store.update_progress(
                10,
                current_seed=self.identity.seed,
                current_arm=self.identity.arm,
                current_substage=self.identity.substage,
                current_shard=self.identity.shard_id,
                now=started + timedelta(seconds=10),
            )
        self.assertEqual(complete["status"], "COMPLETE")
        self.assertEqual(complete["percentage"], 100.0)
        self.assertEqual(complete["throughput_units_per_second"], 1.0)
        self.assertEqual(complete["eta_seconds"], 0.0)
        self.assertEqual(
            complete["completed_at"], complete["estimated_completion_at"]
        )

    def test_corrupt_completed_shard_reconciles_progress_and_preserves_sibling(self):
        sibling = replace(
            self.identity,
            shard_id="scene-001-edge-000",
            range_start=7,
            range_stop=14,
        )
        started = datetime(2026, 8, 31, 0, 0, 0, tzinfo=timezone.utc)
        with self.store.writer_lock():
            damaged = self.store.commit_shard_bytes(self.identity, b"trusted-a")
            preserved = self.store.commit_shard_bytes(sibling, b"trusted-b")
            self.store.initialize_progress(2, now=started)
            complete = self.store.update_progress(
                2,
                current_seed=sibling.seed,
                current_arm=sibling.arm,
                current_substage=sibling.substage,
                current_shard=sibling.shard_id,
                now=started + timedelta(seconds=2),
            )
        previous_progress_bytes = self.store.progress_path.read_bytes()
        previous_progress_sha256 = hashlib.sha256(previous_progress_bytes).hexdigest()
        sibling_payload_bytes = preserved.payload_path.read_bytes()
        sibling_payload_sha256 = hashlib.sha256(sibling_payload_bytes).hexdigest()
        sibling_payload_mtime = preserved.payload_path.stat().st_mtime_ns
        sibling_manifest_bytes = preserved.manifest_path.read_bytes()
        sibling_manifest_sha256 = hashlib.sha256(sibling_manifest_bytes).hexdigest()
        sibling_manifest_mtime = preserved.manifest_path.stat().st_mtime_ns

        damaged.payload_path.write_bytes(b"corrupt-a")
        with self.store.writer_lock():
            self.assertIsNone(self.store.load_or_quarantine_shard(self.identity))
            authenticated_sibling = self.store.load_or_quarantine_shard(sibling)
            reconciled = self.store.reconcile_progress(
                1,
                current_seed=sibling.seed,
                current_arm=sibling.arm,
                current_substage=sibling.substage,
                current_shard=sibling.shard_id,
                reason="authenticated shard inventory decreased after corruption",
                now=started + timedelta(seconds=3),
            )

        self.assertIsNotNone(authenticated_sibling)
        self.assertEqual(complete["status"], "COMPLETE")
        self.assertEqual(reconciled["status"], "RUNNING")
        self.assertEqual(reconciled["completed_units"], 1)
        self.assertEqual(reconciled["percentage"], 50.0)
        self.assertEqual(reconciled["started_at"], complete["started_at"])
        self.assertIsNone(reconciled["completed_at"])

        receipts = self._progress_reconciliation_receipts()
        self.assertEqual(len(receipts), 1)
        receipt_path, receipt = receipts[0]
        self.assertEqual(
            receipt["schema_version"], PROGRESS_RECONCILIATION_SCHEMA
        )
        self.assertEqual(receipt["status"], "COMPLETE")
        self.assertEqual(receipt["run_identity"], self.run.as_dict())
        self.assertEqual(receipt["run_identity_sha256"], self.run.sha256)
        self.assertEqual(receipt["previous_completed_units"], 2)
        self.assertEqual(receipt["reauthenticated_completed_units"], 1)
        self.assertEqual(
            receipt["reason"],
            "authenticated shard inventory decreased after corruption",
        )
        archived_path = (
            receipt_path.parent / receipt["previous_progress"]["archive_path"]
        )
        self.assertEqual(archived_path.read_bytes(), previous_progress_bytes)
        self.assertEqual(
            receipt["previous_progress"]["sha256"], previous_progress_sha256
        )

        with self.store.writer_lock():
            replacement = self.store.commit_shard_bytes(
                self.identity, b"recomputed-a"
            )
            completed_again = self.store.update_progress(
                2,
                current_seed=self.identity.seed,
                current_arm=self.identity.arm,
                current_substage=self.identity.substage,
                current_shard=self.identity.shard_id,
                now=started + timedelta(seconds=4),
            )
        self.assertEqual(replacement.payload_path.read_bytes(), b"recomputed-a")
        self.assertEqual(completed_again["status"], "COMPLETE")
        self.assertEqual(completed_again["completed_units"], 2)
        self.assertEqual(completed_again["percentage"], 100.0)
        self.assertEqual(preserved.payload_path.read_bytes(), sibling_payload_bytes)
        self.assertEqual(
            hashlib.sha256(preserved.payload_path.read_bytes()).hexdigest(),
            sibling_payload_sha256,
        )
        self.assertEqual(
            preserved.payload_path.stat().st_mtime_ns, sibling_payload_mtime
        )
        self.assertEqual(preserved.manifest_path.read_bytes(), sibling_manifest_bytes)
        self.assertEqual(
            hashlib.sha256(preserved.manifest_path.read_bytes()).hexdigest(),
            sibling_manifest_sha256,
        )
        self.assertEqual(
            preserved.manifest_path.stat().st_mtime_ns, sibling_manifest_mtime
        )

    def test_progress_reconciliation_requires_lock_and_rejects_stale_identity(self):
        started = datetime(2026, 8, 31, tzinfo=timezone.utc)
        with self.store.writer_lock():
            self.store.initialize_progress(2, now=started)
            self.store.update_progress(
                1,
                current_seed=self.identity.seed,
                current_arm=self.identity.arm,
                current_substage=self.identity.substage,
                current_shard=self.identity.shard_id,
                now=started + timedelta(seconds=1),
            )
        with self.assertRaises(WriterLockError):
            self.store.reconcile_progress(
                0,
                current_seed=None,
                current_arm=None,
                current_substage=None,
                current_shard=None,
                reason="must hold writer lock",
            )

        changed_run = replace(self.run, dataset_sha256=_digest("stale-dataset"))
        changed_store = EvaluationResumeStore(self.root, changed_run)
        with changed_store.writer_lock():
            with self.assertRaises(StaleResumeError):
                changed_store.reconcile_progress(
                    0,
                    current_seed=None,
                    current_arm=None,
                    current_substage=None,
                    current_shard=None,
                    reason="must reject stale progress",
                )
        self.assertEqual(self._progress_reconciliation_receipts(), [])

    def test_progress_heartbeat_can_run_on_a_worker_thread_under_process_lock(self):
        failures = []
        with self.store.writer_lock():
            self.store.initialize_progress(2)

            def heartbeat():
                try:
                    self.store.update_progress(
                        0,
                        current_seed=20270001,
                        current_arm="full",
                        current_substage="scene",
                        current_shard="scene-001",
                    )
                except BaseException as error:
                    failures.append(error)

            worker = threading.Thread(target=heartbeat)
            worker.start()
            worker.join(timeout=5.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(failures, [])
            self.store.update_progress(
                1,
                current_seed=20270001,
                current_arm="full",
                current_substage="scene",
                current_shard="scene-001",
            )
        self.assertEqual(self.store.read_status()["completed_units"], 1)

    def test_empty_progress_is_immediately_complete(self):
        started = datetime(2026, 8, 31, tzinfo=timezone.utc)
        with self.store.writer_lock():
            progress = self.store.initialize_progress(0, now=started)
        self.assertEqual(progress["status"], "COMPLETE")
        self.assertEqual(progress["percentage"], 100.0)
        self.assertEqual(progress["eta_seconds"], 0.0)

    def test_failed_progress_replace_preserves_last_receipt_and_cleans_temp(self):
        started = datetime(2026, 8, 31, tzinfo=timezone.utc)
        with self.store.writer_lock():
            initial = self.store.initialize_progress(10, now=started)
            with mock.patch(
                "formal_v2.formal_evaluation_resume.os.replace",
                side_effect=OSError("simulated replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated replace failure"):
                    self.store.update_progress(
                        1,
                        current_seed=self.identity.seed,
                        current_arm=self.identity.arm,
                        current_substage=self.identity.substage,
                        current_shard=self.identity.shard_id,
                        now=started + timedelta(seconds=1),
                    )
        self.assertEqual(self.store.read_status(), initial)
        self.assertEqual(self._temporary_files(), [])

    def test_stale_or_corrupt_progress_is_rejected(self):
        with self.store.writer_lock():
            self.store.initialize_progress(10)
        changed_run = replace(
            self.run, dataset_sha256=_digest("different-dataset")
        )
        with self.assertRaises(StaleResumeError):
            read_evaluation_status(self.root, expected_identity=changed_run)
        with self.store.writer_lock():
            with self.assertRaises(StaleResumeError):
                self.store.initialize_progress(11)

        malformed = json.loads(self.store.progress_path.read_text(encoding="ascii"))
        malformed["elapsed_seconds"] = None
        self.store.progress_path.write_text(
            json.dumps(malformed), encoding="ascii"
        )
        with self.assertRaises(ProgressValidationError):
            read_evaluation_status(self.root, expected_identity=self.run)

        self.store.progress_path.write_text("{broken", encoding="ascii")
        with self.assertRaises(ProgressValidationError):
            read_evaluation_status(self.root, expected_identity=self.run)

    def test_status_read_does_not_create_a_missing_resume_root(self):
        missing = Path(self.temporary.name) / "missing"
        self.assertIsNone(read_evaluation_status(missing, expected_identity=self.run))
        self.assertFalse(missing.exists())

    def _temporary_files(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.name.endswith(".tmp")
        )

    def _quarantine_receipts(self) -> list[dict[str, object]]:
        return [
            json.loads(path.read_text(encoding="ascii"))
            for path in sorted(self.root.glob("quarantine/*/receipt.json"))
        ]

    def _progress_reconciliation_receipts(
        self,
    ) -> list[tuple[Path, dict[str, object]]]:
        return [
            (path, json.loads(path.read_text(encoding="ascii")))
            for path in sorted(
                self.root.glob("audit/progress-reconciliation/*/receipt.json")
            )
        ]


if __name__ == "__main__":
    unittest.main()
