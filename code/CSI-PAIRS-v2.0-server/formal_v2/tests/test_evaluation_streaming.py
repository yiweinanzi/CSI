from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
import weakref
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from formal_v2.formal_evaluation_fragments import (
    merge_csv_fragments,
    write_csv_fragment_stream,
)
from formal_v2.formal_evaluation_inventory import build_scene_pair_universe
from formal_v2.formal_evaluation_resume import (
    EvaluationExecutionProfile,
    EvaluationResumeStore,
    EvaluationRunIdentity,
    FINAL_OUTPUT_ARTIFACT_QUARANTINE_SCHEMA,
    FINAL_OUTPUT_QUARANTINE_SCHEMA,
    StaleResumeError,
)
from formal_v2.formal_evaluation_streaming import (
    EVALUATION_TABLE_FIELDS,
    OPTIONAL_SCENE_NUMERIC_FIELDS,
    SCENE_INTEGER_FIELDS,
    SCENE_TABLES,
    SCENE_TEXT_FIELDS,
    STATE_DIRECTORY,
    STREAMING_EVALUATION_SCHEMA,
    EvaluationWorkUnit,
    _ProbeBundle,
    _SceneInputs,
    _ProgressHeartbeat,
    _SceneEvaluation,
    _commit_probe_bundle,
    _commit_scene_bundle,
    _commit_table_fragment,
    _contract_identity,
    _evaluation_total_units,
    _evaluate_scene,
    _fit_probe_bundle,
    _fit_probe_bundle_exclusive,
    _finalization_identity,
    _frozen_evaluation_scenes,
    _load_scene_bundle,
    _load_or_fit_probes,
    _load_authenticated_finalization,
    _ordered_evaluation_scenes,
    _prepare_scene_evaluation,
    _predict_checkpoint_scenes,
    _probe_bundle_payload,
    _probe_contract_rows,
    _probe_state_identity,
    _progress_update_if_reached,
    _read_fragment_rows,
    _reconcile_progress_after_reauthentication,
    _restore_probe_bundle,
    _response_record,
    _scene_bundle_identity,
    _table_identity,
    _validate_scene_table_rows,
    _validate_runner_identity,
    build_evaluation_work_plan,
    evaluation_output_schema_sha256,
    run_evaluation_worker_coordinator,
    run_streaming_formal_evaluation,
)
from formal_v2.formal_model import torch
from formal_v2.formal_probes import (
    ActionResponseProbe,
    CompatibilityProbe,
    predict_binary_probe,
    predict_response_probe,
)
from formal_v2.tests.test_evaluation_inventory import _FakeDataset


class EvaluationStreamingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run_identity = _run_identity()
        self.store = EvaluationResumeStore(self.root / "state", self.run_identity)
        self.checkpoint = {"seed": 17, "arm": "endpoint", "sha256": "c" * 64}
        self.evidence = {
            "artifact_label": "streaming-test",
            "dataset_sha256": "d" * 64,
            "config_sha256": "e" * 64,
            "fixture": False,
            "scientific_use": "CANDIDATE_NOT_CLAIM",
            "source_tree_sha256": "f" * 64,
            "requirements_lock_sha256": "1" * 64,
            "runtime_provenance_sha256": "2" * 64,
        }

    def _run_empty_streaming(
        self,
        run_root,
        *,
        identity=None,
        execution_devices=("cuda:0", "cuda:1"),
        batch_size=1,
    ):
        run_root = Path(run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        input_root = run_root / "inputs"
        input_root.mkdir(exist_ok=True)
        qualification = input_root / "qualification.json"
        factorial = input_root / "factorial.json"
        checkpoints = input_root / "checkpoint-index.json"
        for path in (qualification, factorial, checkpoints):
            if not path.exists():
                path.write_text("{}\n", encoding="ascii")
        upstream = run_root / "upstream"
        upstream.mkdir(exist_ok=True)

        class EmptyDataset:
            is_fixture = False
            bank_ids = np.asarray([], dtype=str)

            @staticmethod
            def indices_for_role(_role):
                return np.asarray([], dtype=np.int64)

        selected_identity = self.run_identity if identity is None else identity
        synthetic_gate = _synthetic_evaluation_gate()
        with (
            mock.patch(
                "formal_v2.formal_evaluation_streaming.evidence_context",
                return_value=self.evidence,
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming._validate_runner_identity"
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._validate_checkpoint_index",
                return_value=[],
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._evaluation_gate",
                return_value=synthetic_gate,
            ) as evaluation_gate,
            mock.patch(
                "formal_v2.formal_evidence.require_stage_manifested_gate",
                side_effect=lambda _path, payload, *_args, **_kwargs: payload,
            ),
        ):
            result = run_streaming_formal_evaluation(
                {},
                EmptyDataset(),
                run_root,
                upstream_root=upstream,
                qualification_gate_path=qualification,
                factorial_gate_path=factorial,
                checkpoint_index_path=checkpoints,
                checkpoint_inventory_sha256=(
                    selected_identity.legacy_checkpoint_inventory_sha256
                ),
                run_identity=selected_identity,
                execution_devices=execution_devices,
                batch_size=batch_size,
            )
        return result, evaluation_gate.call_count

    def _persisted_scene_rows(self, table, rows):
        """Round-trip rows through the canonical gzip CSV representation."""

        path = self.root / "persisted-scene" / table
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            write_csv_fragment_stream(
                handle,
                rows,
                fieldnames=EVALUATION_TABLE_FIELDS[table],
                evidence=self.evidence,
            )
        return _read_fragment_rows(path, EVALUATION_TABLE_FIELDS[table])

    def _validate_scene_table_for_both_storage_paths(self, table, rows):
        """Validate one table before and after CSV serialization."""

        table_rows = rows[table]
        for persisted in (False, True):
            with self.subTest(persisted=persisted):
                candidate = (
                    self._persisted_scene_rows(table, table_rows)
                    if persisted
                    else table_rows
                )
                _validate_scene_table_rows(
                    table,
                    candidate,
                    self.checkpoint,
                    scene=12,
                    bank_id="bank-a",
                    persisted=persisted,
                    evidence=self.evidence if persisted else None,
                )

    def _assert_scene_table_rejected_on_both_storage_paths(self, table, rows):
        """Require the same malformed condition row to fail in both forms."""

        table_rows = rows[table]
        for persisted in (False, True):
            with self.subTest(persisted=persisted):
                candidate = (
                    self._persisted_scene_rows(table, table_rows)
                    if persisted
                    else table_rows
                )
                with self.assertRaises(RuntimeError):
                    _validate_scene_table_rows(
                        table,
                        candidate,
                        self.checkpoint,
                        scene=12,
                        bank_id="bank-a",
                        persisted=persisted,
                        evidence=self.evidence if persisted else None,
                    )

    def _dataset_pair_scene_rows(self, universe):
        rows = _scene_rows_for(
            universe.scene_index,
            universe.bank_id,
            0.5,
        )
        for table in (
            "compatibility_pair_effects.csv",
            "response_pair_effects.csv",
        ):
            template = rows[table][0]
            rows[table] = [
                {**template, **identity}
                for identity in universe.iter_expected_records(table)
            ]
        return rows

    def _rewrite_authenticated_pair_fragment(
        self,
        store,
        table,
        scene,
        rows,
    ):
        """Replace a pair fragment while synchronizing its authenticated receipts."""

        table_result = store.load_completed_shard(
            _table_identity(self.run_identity, self.checkpoint, table, scene),
            suffix=".csv.gz",
        )
        self.assertIsNotNone(table_result)
        encoded = io.BytesIO()
        row_count = write_csv_fragment_stream(
            encoded,
            rows,
            fieldnames=EVALUATION_TABLE_FIELDS[table],
            evidence=self.evidence,
        )
        table_bytes = encoded.getvalue()
        table_result.payload_path.write_bytes(table_bytes)
        table_manifest = json.loads(
            table_result.manifest_path.read_text(encoding="ascii")
        )
        table_manifest["payload"]["bytes"] = len(table_bytes)
        table_manifest["payload"]["sha256"] = hashlib.sha256(
            table_bytes
        ).hexdigest()
        table_result.manifest_path.write_text(
            json.dumps(table_manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )

        bundle_result = store.load_completed_shard(
            _scene_bundle_identity(self.run_identity, self.checkpoint, scene),
            suffix=".json",
        )
        self.assertIsNotNone(bundle_result)
        bundle = json.loads(bundle_result.payload_path.read_text(encoding="ascii"))
        bundle["tables"][table].update(
            {
                "bytes": len(table_bytes),
                "sha256": hashlib.sha256(table_bytes).hexdigest(),
                "row_count": row_count,
            }
        )
        bundle_bytes = (
            json.dumps(bundle, indent=2, sort_keys=True) + "\n"
        ).encode("ascii")
        bundle_result.payload_path.write_bytes(bundle_bytes)
        bundle_manifest = json.loads(
            bundle_result.manifest_path.read_text(encoding="ascii")
        )
        bundle_manifest["payload"]["bytes"] = len(bundle_bytes)
        bundle_manifest["payload"]["sha256"] = hashlib.sha256(
            bundle_bytes
        ).hexdigest()
        bundle_result.manifest_path.write_text(
            json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )

    def _refresh_final_output_authentication(self, run_root):
        """Refresh every superficial receipt after a coordinated final-output edit."""

        run_root = Path(run_root)
        output_dir = run_root / "evaluation"

        def encode(payload):
            return (
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("ascii")

        stage_manifest_path = output_dir / "manifest.json"
        stage_manifest = json.loads(stage_manifest_path.read_text(encoding="ascii"))
        for record in stage_manifest["files"]:
            artifact = output_dir / record["path"]
            payload = artifact.read_bytes()
            record["bytes"] = len(payload)
            record["sha256"] = hashlib.sha256(payload).hexdigest()
        stage_manifest_path.write_bytes(encode(stage_manifest))

        store = EvaluationResumeStore(
            run_root / STATE_DIRECTORY, self.run_identity
        )
        final_payload_path, final_manifest_path = store.shard_paths(
            _finalization_identity(self.run_identity), suffix=".json"
        )
        finalization = json.loads(final_payload_path.read_text(encoding="ascii"))
        for record in finalization["artifacts"]:
            artifact = run_root / record["path"]
            payload = artifact.read_bytes()
            record["bytes"] = len(payload)
            record["sha256"] = hashlib.sha256(payload).hexdigest()
        final_payload = encode(finalization)
        final_payload_path.write_bytes(final_payload)

        shard_manifest = json.loads(
            final_manifest_path.read_text(encoding="ascii")
        )
        shard_manifest["payload"]["bytes"] = len(final_payload)
        shard_manifest["payload"]["sha256"] = hashlib.sha256(
            final_payload
        ).hexdigest()
        final_manifest_path.write_bytes(encode(shard_manifest))

    def test_heartbeat_atomically_refreshes_long_unit_without_advancing_it(self):
        updates = []
        update_guard = threading.Lock()
        three_updates = threading.Event()
        with self.store.writer_lock():
            self.store.initialize_progress(4)
            original_update = self.store.update_progress

            def recording_update(*args, **kwargs):
                result = original_update(*args, **kwargs)
                with update_guard:
                    updates.append(result)
                    if len(updates) >= 3:
                        three_updates.set()
                return result

            with mock.patch.object(
                self.store, "update_progress", side_effect=recording_update
            ):
                heartbeat = _ProgressHeartbeat(
                    self.store,
                    0,
                    seed=17,
                    arm="endpoint",
                    substage="scene",
                    shard="scene-012",
                    interval_seconds=0.01,
                )
                with heartbeat:
                    self.assertTrue(three_updates.wait(1.0))
                    self.assertTrue(heartbeat.thread_alive)
            self.assertFalse(heartbeat.thread_alive)
            status = self.store.read_status()

        self.assertGreaterEqual(len(updates), 3)
        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual(status["status"], "RUNNING")
        self.assertEqual(status["completed_units"], 0)
        self.assertEqual(status["percentage"], 0.0)
        self.assertEqual(status["current_seed"], 17)
        self.assertEqual(status["current_arm"], "endpoint")
        self.assertEqual(status["current_substage"], "scene")
        self.assertEqual(status["current_shard"], "scene-012")
        self.assertGreater(status["elapsed_seconds"], updates[0]["elapsed_seconds"])
        self.assertTrue(
            status["eta_seconds"] is None
            or isinstance(status["eta_seconds"], float)
        )
        self.assertTrue(all(row["completed_units"] == 0 for row in updates))

    def test_heartbeat_uses_persisted_monotone_count(self):
        with self.store.writer_lock():
            self.store.initialize_progress(4)
            self.store.update_progress(
                2,
                current_seed=17,
                current_arm="endpoint",
                current_substage="scene",
                current_shard="scene-011",
            )
            with _ProgressHeartbeat(
                self.store,
                0,
                seed=17,
                arm="endpoint",
                substage="scene",
                shard="scene-012",
                interval_seconds=0.01,
            ):
                pass
            status = self.store.read_status()
        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual(status["completed_units"], 2)
        self.assertEqual(status["current_shard"], "scene-012")

    def test_heartbeat_failure_propagates_after_thread_stops(self):
        failed = threading.Event()
        calls = 0
        with self.store.writer_lock():
            self.store.initialize_progress(2)
            original_update = self.store.update_progress

            def failing_update(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls >= 2:
                    failed.set()
                    raise RuntimeError("heartbeat write failed")
                return original_update(*args, **kwargs)

            heartbeat = _ProgressHeartbeat(
                self.store,
                0,
                seed=17,
                arm="endpoint",
                substage="scene",
                shard="scene-012",
                interval_seconds=0.01,
            )
            with mock.patch.object(
                self.store, "update_progress", side_effect=failing_update
            ):
                with self.assertRaisesRegex(RuntimeError, "heartbeat write failed"):
                    with heartbeat:
                        self.assertTrue(failed.wait(1.0))
        self.assertFalse(heartbeat.thread_alive)

    def test_heartbeat_failure_does_not_mask_primary_exception(self):
        failed = threading.Event()
        calls = 0
        with self.store.writer_lock():
            self.store.initialize_progress(2)
            original_update = self.store.update_progress

            def failing_update(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls >= 2:
                    failed.set()
                    raise RuntimeError("secondary heartbeat failure")
                return original_update(*args, **kwargs)

            with mock.patch.object(
                self.store, "update_progress", side_effect=failing_update
            ):
                with self.assertRaisesRegex(ValueError, "primary computation failure"):
                    with _ProgressHeartbeat(
                        self.store,
                        0,
                        seed=17,
                        arm="endpoint",
                        substage="scene",
                        shard="scene-012",
                        interval_seconds=0.01,
                    ):
                        self.assertTrue(failed.wait(1.0))
                        raise ValueError("primary computation failure")

    def test_heartbeat_interval_is_bounded_to_sixty_seconds(self):
        for invalid in (0, -1, 60.01, float("inf"), True):
            with self.subTest(interval=invalid):
                with self.assertRaises(ValueError):
                    _ProgressHeartbeat(
                        self.store,
                        0,
                        seed=17,
                        arm="endpoint",
                        substage="scene",
                        shard="scene-012",
                        interval_seconds=invalid,
                    )

    def test_resume_progress_reconciles_to_currently_authenticated_units(self):
        with self.store.writer_lock():
            self.store.initialize_progress(5)
            self.store.update_progress(
                3,
                current_seed=17,
                current_arm="endpoint",
                current_substage="scene",
                current_shard="scene-012",
            )
            _reconcile_progress_after_reauthentication(self.store, 2)
            status = self.store.read_status()
        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual(status["status"], "RUNNING")
        self.assertEqual(status["completed_units"], 2)
        self.assertIsNone(status["current_shard"])

    def test_resume_progress_advances_to_authenticated_units(self):
        with self.store.writer_lock():
            self.store.initialize_progress(5)
            _reconcile_progress_after_reauthentication(self.store, 3)
            status = self.store.read_status()
        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual(status["completed_units"], 3)

    def test_runner_identity_uses_authenticated_canonical_checkpoint_inventory(self):
        qualification_path = self.root / "qualification.json"
        factorial_path = self.root / "factorial.json"
        checkpoint_index_path = self.root / "checkpoint_index.json"
        qualification_path.write_text('{"gate":"qualification"}\n', encoding="ascii")
        factorial_path.write_text('{"gate":"factorial"}\n', encoding="ascii")
        checkpoint_index_path.write_text('{"checkpoints":[]}\n', encoding="ascii")
        canonical_inventory = "5" * 64
        self.assertNotEqual(
            hashlib.sha256(checkpoint_index_path.read_bytes()).hexdigest(),
            canonical_inventory,
        )
        identity = replace(
            self.run_identity,
            qualification_gate_sha256=hashlib.sha256(
                qualification_path.read_bytes()
            ).hexdigest(),
            factorial_gate_sha256=hashlib.sha256(
                factorial_path.read_bytes()
            ).hexdigest(),
            legacy_checkpoint_inventory_sha256=canonical_inventory,
        )
        evidence = {
            "source_tree_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "dataset_sha256": "3" * 64,
            "runtime_provenance_sha256": "8" * 64,
        }
        _validate_runner_identity(
            identity,
            evidence,
            qualification_gate_path=qualification_path,
            factorial_gate_path=factorial_path,
            checkpoint_inventory_sha256=canonical_inventory,
        )
        with self.assertRaisesRegex(
            RuntimeError, "legacy_checkpoint_inventory_sha256 mismatch"
        ):
            _validate_runner_identity(
                identity,
                evidence,
                qualification_gate_path=qualification_path,
                factorial_gate_path=factorial_path,
                checkpoint_inventory_sha256="9" * 64,
            )

    def test_final_merge_is_a_distinct_last_progress_unit(self):
        total_units = _evaluation_total_units(2, 3)
        self.assertEqual(total_units, 9)
        with self.store.writer_lock():
            self.store.initialize_progress(total_units)
            before_merge = self.store.update_progress(
                total_units - 1,
                current_seed=None,
                current_arm=None,
                current_substage="final_merge",
                current_shard="canonical-output",
            )
            self.assertEqual(before_merge["status"], "RUNNING")
            complete = self.store.update_progress(
                total_units,
                current_seed=None,
                current_arm=None,
                current_substage=None,
                current_shard=None,
            )
        self.assertEqual(complete["status"], "COMPLETE")
        self.assertEqual(complete["percentage"], 100.0)

    def test_completed_restart_authenticates_finalization_without_rewriting(self):
        run_root = self.root / "completed-restart"
        first_gate, first_gate_calls = self._run_empty_streaming(run_root)
        self.assertEqual(first_gate, _synthetic_evaluation_gate())
        self.assertEqual(first_gate_calls, 1)
        output_dir = run_root / "evaluation"
        store = EvaluationResumeStore(
            run_root / STATE_DIRECTORY, self.run_identity
        )
        final_payload, final_manifest = store.shard_paths(
            _finalization_identity(self.run_identity), suffix=".json"
        )
        finalization = json.loads(final_payload.read_text(encoding="ascii"))
        self.assertEqual(len(finalization["artifacts"]), 11)
        self.assertEqual(
            [row["path"] for row in finalization["artifacts"]],
            sorted(
                [
                    *[f"evaluation/{name}" for name in EVALUATION_TABLE_FIELDS],
                    "evaluation/gate.json",
                    "evaluation/manifest.json",
                ]
            ),
        )
        observed_paths = [
            *[output_dir / name for name in EVALUATION_TABLE_FIELDS],
            output_dir / "gate.json",
            output_dir / "manifest.json",
            final_payload,
            final_manifest,
            store.progress_path,
        ]
        before = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in observed_paths
        }

        with mock.patch(
            "formal_v2.formal_evaluation_streaming.merge_csv_fragments",
            side_effect=AssertionError("completed restart attempted final merge"),
        ) as merge:
            second_gate, second_gate_calls = self._run_empty_streaming(run_root)
        self.assertEqual(second_gate, first_gate)
        self.assertEqual(second_gate_calls, 1)
        merge.assert_not_called()
        self.assertEqual(
            {
                path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in observed_paths
            },
            before,
        )
        status = store.read_status()
        self.assertEqual(status["status"], "COMPLETE")
        self.assertEqual(status["completed_units"], status["total_units"])
        self.assertEqual(
            list(
                (run_root / STATE_DIRECTORY).glob(
                    "audit/progress-reconciliation/*/receipt.json"
                )
            ),
            [],
        )

    def test_finalization_receipt_recovers_crash_before_progress_completion(self):
        run_root = self.root / "finalized-before-progress"

        def fail_final_progress(store, completed_units, **position):
            if completed_units == 1:
                raise RuntimeError("simulated crash after finalization receipt")
            return _progress_update_if_reached(
                store, completed_units, **position
            )

        with mock.patch(
            "formal_v2.formal_evaluation_streaming._progress_update_if_reached",
            side_effect=fail_final_progress,
        ):
            with self.assertRaisesRegex(RuntimeError, "after finalization receipt"):
                self._run_empty_streaming(run_root)

        output_dir = run_root / "evaluation"
        store = EvaluationResumeStore(
            run_root / STATE_DIRECTORY, self.run_identity
        )
        final_payload, final_manifest = store.shard_paths(
            _finalization_identity(self.run_identity), suffix=".json"
        )
        stable_paths = [
            *[output_dir / name for name in EVALUATION_TABLE_FIELDS],
            output_dir / "gate.json",
            output_dir / "manifest.json",
            final_payload,
            final_manifest,
        ]
        before = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in stable_paths
        }
        self.assertEqual(store.read_status()["status"], "RUNNING")

        with mock.patch(
            "formal_v2.formal_evaluation_streaming.merge_csv_fragments",
            side_effect=AssertionError("valid finalization was not resumed"),
        ) as merge:
            gate, gate_calls = self._run_empty_streaming(run_root)
        self.assertEqual(gate, _synthetic_evaluation_gate())
        self.assertEqual(gate_calls, 1)
        merge.assert_not_called()
        self.assertEqual(store.read_status()["status"], "COMPLETE")
        self.assertEqual(
            {
                path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in stable_paths
            },
            before,
        )

    def test_forged_final_gate_is_rejected_by_strict_scientific_replay(self):
        run_root = self.root / "forged-final-gate"
        expected_gate, _calls = self._run_empty_streaming(run_root)
        gate_path = run_root / "evaluation" / "gate.json"
        forged_gate = json.loads(gate_path.read_text(encoding="ascii"))
        forged_gate["gate_vector"] = {"G3": "PASS"}
        gate_path.write_text(
            json.dumps(forged_gate, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        self._refresh_final_output_authentication(run_root)

        repaired_gate, gate_calls = self._run_empty_streaming(run_root)

        self.assertEqual(repaired_gate, expected_gate)
        self.assertEqual(gate_calls, 2)
        self.assertEqual(
            json.loads(gate_path.read_text(encoding="ascii")), expected_gate
        )

    def test_final_csv_must_match_authenticated_fragment_merge(self):
        run_root = self.root / "forged-final-csv"
        expected_gate, _calls = self._run_empty_streaming(run_root)
        table_path = run_root / "evaluation" / "cgs_per_bank.csv"
        forged = io.StringIO(newline="")
        csv.writer(forged, dialect="excel").writerow(
            EVALUATION_TABLE_FIELDS["cgs_per_bank.csv"]
        )
        table_path.write_text(forged.getvalue(), encoding="utf-8")
        self._refresh_final_output_authentication(run_root)

        repaired_gate, gate_calls = self._run_empty_streaming(run_root)

        self.assertEqual(repaired_gate, expected_gate)
        self.assertEqual(gate_calls, 1)
        self.assertEqual(table_path.read_bytes(), b"")

    def test_damaged_final_output_invalidates_receipt_and_only_refinalizes(self):
        run_root = self.root / "damaged-final-output"
        self._run_empty_streaming(run_root)
        output_dir = run_root / "evaluation"
        damaged = output_dir / "cgs_per_bank.csv"
        damaged.write_bytes(b"damaged-final-output")
        repaired_gate, gate_calls = self._run_empty_streaming(run_root)
        self.assertEqual(repaired_gate, _synthetic_evaluation_gate())
        self.assertEqual(gate_calls, 1)
        self.assertEqual(damaged.read_bytes(), b"")

        state_root = run_root / STATE_DIRECTORY
        receipts = [
            json.loads(path.read_text(encoding="ascii"))
            for path in state_root.glob("quarantine/*/receipt.json")
        ]
        finalization_receipts = [
            receipt
            for receipt in receipts
            if receipt.get("shard_identity", {}).get("substage") == "finalization"
        ]
        self.assertEqual(len(finalization_receipts), 1)
        self.assertEqual(
            finalization_receipts[0]["run_identity_sha256"],
            self.run_identity.sha256,
        )
        artifact_receipts = [
            receipt
            for receipt in receipts
            if receipt.get("schema_version")
            == FINAL_OUTPUT_ARTIFACT_QUARANTINE_SCHEMA
        ]
        self.assertEqual(len(artifact_receipts), 1)
        self.assertEqual(len(artifact_receipts[0]["files"]), 1)
        preserved_damage = (
            state_root
            / artifact_receipts[0]["files"][0]["quarantine_path"]
        )
        self.assertEqual(preserved_damage.read_bytes(), b"damaged-final-output")
        store = EvaluationResumeStore(state_root, self.run_identity)
        with store.writer_lock():
            authenticated = _load_authenticated_finalization(
                store, self.run_identity, output_dir
            )
        self.assertEqual(authenticated, repaired_gate)
        self.assertEqual(store.read_status()["status"], "COMPLETE")

    def test_final_output_crash_temporaries_are_preserved_then_retry_succeeds(self):
        run_root = self.root / "merge-crash-retry"
        output_dir = run_root / "evaluation"
        output_dir.mkdir(parents=True)
        final_output_names = (*EVALUATION_TABLE_FIELDS, "gate.json", "manifest.json")
        temporaries = {
            output_dir / f".{name}.worker.tmp": f"partial-{name}".encode("ascii")
            for name in final_output_names
        }
        self.assertEqual(len(temporaries), 11)
        unrelated = output_dir / ".not-a-final-table.csv.worker.tmp"
        for path, payload in temporaries.items():
            path.write_bytes(payload)
        unrelated.write_bytes(b"unrelated")

        gate, gate_calls = self._run_empty_streaming(run_root)
        self.assertEqual(gate, _synthetic_evaluation_gate())
        self.assertEqual(gate_calls, 1)
        self.assertTrue(all(not path.exists() for path in temporaries))
        self.assertEqual(unrelated.read_bytes(), b"unrelated")
        receipt_paths = list(
            (run_root / STATE_DIRECTORY).glob("quarantine/*/receipt.json")
        )
        self.assertEqual(len(receipt_paths), 1)
        receipt = json.loads(receipt_paths[0].read_text(encoding="ascii"))
        self.assertEqual(receipt["schema_version"], FINAL_OUTPUT_QUARANTINE_SCHEMA)
        self.assertEqual(receipt["run_identity_sha256"], self.run_identity.sha256)
        self.assertEqual(
            [row["original_path"] for row in receipt["files"]],
            [
                f"evaluation/.{name}.worker.tmp"
                for name in sorted(final_output_names)
            ],
        )
        preserved = {
            row["original_path"]: (
                run_root / STATE_DIRECTORY / row["quarantine_path"]
            ).read_bytes()
            for row in receipt["files"]
        }
        self.assertEqual(
            preserved,
            {
                f"evaluation/{path.name}": payload
                for path, payload in temporaries.items()
            },
        )

    def test_finalization_shard_rejects_stale_run_identity(self):
        run_root = self.root / "stale-finalization"
        self._run_empty_streaming(run_root)
        original_store = EvaluationResumeStore(
            run_root / STATE_DIRECTORY, self.run_identity
        )
        final_payload, final_manifest = original_store.shard_paths(
            _finalization_identity(self.run_identity), suffix=".json"
        )
        before = (final_payload.read_bytes(), final_manifest.read_bytes())
        changed_identity = replace(self.run_identity, run_nonce="b" * 64)
        changed_store = EvaluationResumeStore(
            run_root / STATE_DIRECTORY, changed_identity
        )
        with changed_store.writer_lock():
            with self.assertRaises(StaleResumeError):
                _load_authenticated_finalization(
                    changed_store,
                    changed_identity,
                    run_root / "evaluation",
                )
        self.assertEqual(
            (final_payload.read_bytes(), final_manifest.read_bytes()), before
        )

    def test_runner_rejects_execution_profile_mismatch_before_output_creation(self):
        cases = (
            (("cuda:1", "cuda:0"), 1),
            (("cuda:0", "cuda:1"), 2),
        )
        for index, (devices, batch_size) in enumerate(cases):
            with self.subTest(devices=devices, batch_size=batch_size):
                run_root = self.root / f"profile-mismatch-{index}"
                with self.assertRaisesRegex(StaleResumeError, "execution profile"):
                    self._run_empty_streaming(
                        run_root,
                        execution_devices=devices,
                        batch_size=batch_size,
                    )
                self.assertFalse((run_root / "evaluation").exists())

    def test_two_gpu_work_plan_is_checkpoint_stable_and_defaults_to_batch_one(self):
        checkpoints = _work_checkpoints(4)
        plan = build_evaluation_work_plan(checkpoints, (12, 5))
        self.assertTrue(all(isinstance(unit, EvaluationWorkUnit) for unit in plan))
        self.assertEqual(
            [unit.canonical_index for unit in plan], list(range(len(plan)))
        )
        for checkpoint_index in range(4):
            checkpoint_units = [
                unit for unit in plan if unit.checkpoint_index == checkpoint_index
            ]
            self.assertEqual(
                {unit.execution_device for unit in checkpoint_units},
                {f"cuda:{checkpoint_index % 2}"},
            )
            self.assertEqual({unit.batch_size for unit in checkpoint_units}, {1})
            self.assertEqual(
                [(unit.substage, unit.scene) for unit in checkpoint_units],
                [("probe_state", None), ("scene", 12), ("scene", 5)],
            )
        repeated = build_evaluation_work_plan(checkpoints, (12, 5))
        self.assertEqual(plan, repeated)

    def test_deleted_scene_bundle_recomputes_only_the_deleted_scene(self):
        run_root = self.root / "scene-local-resume"
        input_root = run_root / "inputs"
        input_root.mkdir(parents=True)
        upstream = run_root / "upstream"
        upstream.mkdir()
        qualification = input_root / "qualification.json"
        factorial = input_root / "factorial.json"
        checkpoints = input_root / "checkpoint-index.json"
        qualification.write_text(
            '{"teacher_checkpoint":"unused.pt"}\n', encoding="ascii"
        )
        factorial.write_text("{}\n", encoding="ascii")
        checkpoints.write_text("{}\n", encoding="ascii")
        config = {
            "evaluation": {
                "probe_hidden_dim": 5,
                "probe_steps": 1,
            }
        }

        class Dataset:
            is_fixture = False
            bank_ids = np.asarray(
                [
                    *[f"unused-{index}" for index in range(5)],
                    "bank-b",
                    *[f"unused-{index}" for index in range(6, 12)],
                    "bank-a",
                ]
            )

            @staticmethod
            def indices_for_role(role):
                return {
                    "source_final_unseen_bank": np.asarray([12], dtype=np.int64),
                    "target": np.asarray([5], dtype=np.int64),
                    "source_encoder_train": np.asarray([], dtype=np.int64),
                }[role]

        store = EvaluationResumeStore(
            run_root / STATE_DIRECTORY, self.run_identity
        )
        rows = {
            12: _scene_rows_for(12, "bank-a", 0.25),
            5: _scene_rows_for(5, "bank-b", 0.75),
        }
        with store.writer_lock():
            _commit_probe_bundle(
                store,
                _probe_state_identity(self.run_identity, self.checkpoint, 0),
                _probe_bundle(hidden_dim=5),
                config,
            )
            for scene in (12, 5):
                _commit_scene_bundle(
                    store,
                    self.run_identity,
                    self.checkpoint,
                    scene=scene,
                    bank_id=str(Dataset.bank_ids[scene]),
                    rows_by_table=rows[scene],
                    evidence=self.evidence,
                )

        sibling_paths = store.shard_paths(
            _scene_bundle_identity(self.run_identity, self.checkpoint, 12),
            suffix=".json",
        )
        sibling_before = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in sibling_paths
        }
        deleted_paths = store.shard_paths(
            _scene_bundle_identity(self.run_identity, self.checkpoint, 5),
            suffix=".json",
        )
        for path in deleted_paths:
            path.unlink()

        teacher = mock.Mock(patch_spec=object())
        with (
            mock.patch(
                "formal_v2.formal_evaluation_streaming.evidence_context",
                return_value=self.evidence,
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming._validate_runner_identity"
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._validate_checkpoint_index",
                return_value=[self.checkpoint],
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.torch.cuda.set_device"
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.load_teacher_bundle",
                return_value=teacher,
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.fit_route_normalization",
                return_value=object(),
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._training_normalization",
                return_value=object(),
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._load_model",
                return_value=object(),
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming._prepare_and_evaluate_scene",
                return_value=rows[5],
            ) as prepare_scene,
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._evaluation_gate",
                return_value={"status": "PASS"},
            ),
        ):
            gate = run_streaming_formal_evaluation(
                config,
                Dataset(),
                run_root,
                upstream_root=upstream,
                qualification_gate_path=qualification,
                factorial_gate_path=factorial,
                checkpoint_index_path=checkpoints,
                checkpoint_inventory_sha256=(
                    self.run_identity.legacy_checkpoint_inventory_sha256
                ),
                run_identity=self.run_identity,
                execution_devices=("cuda:0", "cuda:1"),
            )

        self.assertEqual(gate, {"status": "PASS"})
        prepare_scene.assert_called_once()
        self.assertEqual(prepare_scene.call_args.kwargs["scene"], 5)
        self.assertEqual(
            {
                path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in sibling_paths
            },
            sibling_before,
        )
        with store.writer_lock():
            restored = _load_scene_bundle(
                store,
                self.run_identity,
                self.checkpoint,
                scene=5,
                bank_id="bank-b",
                evidence=self.evidence,
            )
        self.assertIsNotNone(restored)

    def test_worker_completion_order_cannot_change_coordinator_order(self):
        plan = build_evaluation_work_plan(_work_checkpoints(2), (12, 5))
        cuda_one_started = threading.Event()
        worker_threads = []
        worker_finished = []
        coordinator_threads = []
        coordinated = []
        guard = threading.Lock()
        caller_thread = threading.get_ident()

        def worker(unit):
            if unit.execution_device == "cuda:0" and unit.canonical_index == 0:
                self.assertTrue(cuda_one_started.wait(1.0))
            with guard:
                worker_threads.append(threading.get_ident())
                worker_finished.append(unit.canonical_index)
            if unit.execution_device == "cuda:1" and unit.canonical_index == 3:
                cuda_one_started.set()
            return f"result-{unit.canonical_index}"

        def coordinator(unit, result):
            coordinator_threads.append(threading.get_ident())
            coordinated.append((unit.canonical_index, result))
            return f"receipt-{unit.canonical_index}"

        receipts = run_evaluation_worker_coordinator(plan, worker, coordinator)
        self.assertEqual(worker_finished[0], 3)
        deterministic_device_round_order = [0, 3, 1, 4, 2, 5]
        self.assertEqual(
            coordinated,
            [
                (index, f"result-{index}")
                for index in deterministic_device_round_order
            ],
        )
        self.assertEqual(
            receipts, tuple(f"receipt-{index}" for index in range(len(plan)))
        )
        self.assertTrue(all(thread != caller_thread for thread in worker_threads))
        self.assertEqual(set(coordinator_threads), {caller_thread})

    def test_bounded_scheduler_keeps_both_devices_advancing_to_third_unit(self):
        plan = build_evaluation_work_plan(_work_checkpoints(2), (12, 5))
        third_unit_started = {
            "cuda:0": threading.Event(),
            "cuda:1": threading.Event(),
        }

        def worker(unit):
            if unit.substage == "scene" and unit.scene == 5:
                third_unit_started[unit.execution_device].set()
                other = "cuda:1" if unit.execution_device == "cuda:0" else "cuda:0"
                self.assertTrue(third_unit_started[other].wait(1.0))
            return unit.canonical_index

        receipts = run_evaluation_worker_coordinator(
            plan, worker, lambda _unit, result: result
        )
        self.assertEqual(receipts, tuple(range(len(plan))))
        self.assertTrue(all(event.is_set() for event in third_unit_started.values()))

    def test_worker_failure_prevents_buffered_later_results_from_committing(self):
        plan = build_evaluation_work_plan(_work_checkpoints(2), (12,))
        release_cuda_zero = threading.Event()
        committed = []

        def worker(unit):
            if unit.execution_device == "cuda:0" and unit.canonical_index == 0:
                self.assertTrue(release_cuda_zero.wait(1.0))
            if unit.execution_device == "cuda:1":
                release_cuda_zero.set()
                raise RuntimeError("worker failed")
            return unit.canonical_index

        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            run_evaluation_worker_coordinator(
                plan,
                worker,
                lambda unit, result: committed.append((unit, result)),
            )
        self.assertTrue(
            all(unit.canonical_index < 2 for unit, _result in committed)
        )

    def test_scheduler_fails_closed_on_invalid_plan_identity(self):
        plan = build_evaluation_work_plan(_work_checkpoints(2), (12,))
        duplicate = (plan[0], replace(plan[1], canonical_index=0), *plan[2:])
        with self.assertRaisesRegex(ValueError, "canonical indices"):
            run_evaluation_worker_coordinator(duplicate, lambda unit: unit, lambda *_: None)
        invalid_device = (
            replace(plan[0], execution_device="gpu:0"),
            *plan[1:],
        )
        with self.assertRaisesRegex(ValueError, "invalid evaluation worker device"):
            run_evaluation_worker_coordinator(
                invalid_device, lambda unit: unit, lambda *_: None
            )

    def test_nine_table_headers_match_authenticated_oracle(self):
        oracle_value = os.environ.get("CSI_EVALUATION_ORACLE_ROOT")
        if not oracle_value:
            self.skipTest("CSI_EVALUATION_ORACLE_ROOT is not configured")
        oracle = Path(oracle_value)
        self.assertEqual(len(EVALUATION_TABLE_FIELDS), 9)
        self.assertEqual(set(EVALUATION_TABLE_FIELDS), {path.name for path in oracle.glob("*.csv")})
        for table, expected in EVALUATION_TABLE_FIELDS.items():
            with self.subTest(table=table):
                with (oracle / table).open(encoding="utf-8", newline="") as handle:
                    observed = tuple(next(csv.reader(handle)))
                self.assertEqual(observed, expected)

    def test_probe_bundle_roundtrip_preserves_parameters_and_contract(self):
        config = {"evaluation": {"probe_hidden_dim": 5}}
        probes = _probe_bundle(hidden_dim=5)
        identity = _probe_state_identity(self.run_identity, self.checkpoint, 0)
        with self.store.writer_lock():
            path = _commit_probe_bundle(self.store, identity, probes, config)
        with mock.patch.object(torch, "load", wraps=torch.load) as load:
            restored = _restore_probe_bundle(
                path,
                config,
                seed=self.checkpoint["seed"],
                arm=self.checkpoint["arm"],
                expected_response_output_dim=2,
            )
        load.assert_called_once_with(
            path,
            map_location="cpu",
            weights_only=True,
        )
        self.assertEqual(restored.compatibility_selection, probes.compatibility_selection)
        self.assertEqual(tuple(restored.shortcut_probes), tuple(probes.shortcut_probes))
        self.assertEqual(tuple(restored.variant_probes), tuple(probes.variant_probes))
        _assert_module_equal(self, restored.compatibility_probe, probes.compatibility_probe)
        _assert_module_equal(self, restored.response_probe, probes.response_probe)
        self.assertEqual(
            restored.response_probe.zero_preserving,
            probes.response_probe.zero_preserving,
        )
        for name in probes.shortcut_probes:
            expected = probes.shortcut_probes[name]["probe"]
            observed = restored.shortcut_probes[name]["probe"]
            if expected is None:
                self.assertIsNone(observed)
            else:
                _assert_module_equal(self, observed, expected)
        for name in probes.variant_probes:
            _assert_module_equal(
                self, restored.variant_probes[name], probes.variant_probes[name]
            )

    def test_probe_bundle_rejects_hash_valid_semantic_metadata_mutations(self):
        config = {"evaluation": {"probe_hidden_dim": 5}}
        original = _probe_bundle_payload(
            _probe_bundle(hidden_dim=5),
            config,
            seed=self.checkpoint["seed"],
            arm=self.checkpoint["arm"],
        )

        def wrong_response_width(payload):
            main = _fill_module(ActionResponseProbe(4, 1, 5), 8.0)
            main.zero_preserving = True
            payload["response_probe"] = _response_record(main)
            for index, name in enumerate(
                ("without_map", "edit_only", "csi_only", "oracle_x")
            ):
                probe = _fill_module(ActionResponseProbe(4, 1, 5), 9.0 + index)
                probe.zero_preserving = name != "csi_only"
                payload["variant_probes"][name] = _response_record(probe)

        def nonfinite_parameter(payload):
            parameter = next(
                iter(payload["compatibility_probe"]["state_dict"].values())
            )
            parameter.reshape(-1)[0] = float("nan")

        mutations = {
            "missing-selection-fields": lambda payload: payload.__setitem__(
                "compatibility_selection", {}
            ),
            "selection-family-mismatch": lambda payload: payload[
                "compatibility_selection"
            ].__setitem__("selected_family", "linear"),
            "candidate-order": lambda payload: payload["compatibility_selection"][
                "candidate_metrics"
            ].reverse(),
            "constant-family": lambda payload: payload["shortcut_probes"][
                "constant"
            ].__setitem__("selected_family", "linear"),
            "shortcut-nonfinite": lambda payload: payload["shortcut_probes"][
                "csi_only"
            ].__setitem__("source_train_auroc", float("inf")),
            "main-zero-mode": lambda payload: payload["response_probe"].__setitem__(
                "zero_preserving", False
            ),
            "variant-zero-mode": lambda payload: payload["variant_probes"][
                "csi_only"
            ].__setitem__("zero_preserving", True),
            "response-width": wrong_response_width,
            "nonfinite-parameter": nonfinite_parameter,
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                payload = copy.deepcopy(original)
                mutate(payload)
                path = self.root / f"invalid-{name}.pt"
                with path.open("wb") as handle:
                    torch.save(payload, handle)
                with self.assertRaises((RuntimeError, ValueError)):
                    _restore_probe_bundle(
                        path,
                        config,
                        seed=self.checkpoint["seed"],
                        arm=self.checkpoint["arm"],
                        expected_response_output_dim=2,
                    )

    def test_probe_builds_are_exclusive_across_workers(self):
        start = threading.Barrier(2)
        lock = threading.Lock()
        guard = threading.Lock()
        active = 0
        maximum_active = 0
        errors = []

        def fake_fit(label):
            nonlocal active, maximum_active
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.05)
            with guard:
                active -= 1
            return label

        def invoke(label):
            try:
                start.wait(timeout=1.0)
                _fit_probe_bundle_exclusive(lock, label)
            except BaseException as error:
                errors.append(error)

        with mock.patch(
            "formal_v2.formal_evaluation_streaming._fit_probe_bundle",
            side_effect=fake_fit,
        ):
            threads = [
                threading.Thread(target=invoke, args=(label,))
                for label in ("cuda:0", "cuda:1")
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2.0)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(maximum_active, 1)

    def test_compatibility_corpora_are_released_before_response_corpus(self):
        class TrackedRows(dict):
            pass

        references = []

        def compatibility_dataset(*_args, **_kwargs):
            rows = TrackedRows(
                features=np.zeros((2, 3), dtype=np.float64),
                labels=np.asarray([0, 1], dtype=np.int64),
            )
            references.append(weakref.ref(rows))
            return rows

        def response_dataset(*_args, **_kwargs):
            import gc

            gc.collect()
            self.assertTrue(all(reference() is None for reference in references))
            features = np.zeros((2, 4), dtype=np.float64)
            return {
                "features": features,
                "no_action_features": features.copy(),
                "without_map_features": features.copy(),
                "without_map_zero_action_features": features.copy(),
                "edit_only_features": features.copy(),
                "edit_only_zero_action_features": features.copy(),
                "csi_only_features": features.copy(),
                "oracle_x_features": features.copy(),
                "oracle_x_zero_action_features": features.copy(),
                "targets": np.ones((2, 2), dtype=np.float64),
                "source_targets": np.zeros((2, 2), dtype=np.float64),
            }

        class Dataset:
            @staticmethod
            def indices_for_role(role):
                return {
                    "source_probe_train": np.asarray([0]),
                    "source_probe_selection": np.asarray([1]),
                }[role]

        selection = {
            "selected_family": "linear",
            "selection_nll": 0.4,
            "selection_auroc": 0.6,
            "candidate_metrics": [
                {"family": "linear", "selection_nll": 0.4, "selection_auroc": 0.6},
                {"family": "mlp2", "selection_nll": 0.5, "selection_auroc": 0.5},
            ],
        }
        with (
            mock.patch(
                "formal_v2.formal_evaluation_streaming.route_dataset",
                side_effect=(object(), object()),
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._compatibility_dataset",
                new=compatibility_dataset,
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.fit_select_compatibility_probe",
                new=lambda *_args, **_kwargs: (object(), selection),
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._prepare_alignment_shortcut_probes",
                new=lambda *_args, **_kwargs: {},
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._response_probe_dataset",
                new=response_dataset,
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.fit_action_response_probe",
                new=lambda *_args, **_kwargs: object(),
            ),
        ):
            _fit_probe_bundle(
                object(),
                Dataset(),
                object(),
                {"evaluation": {}},
                object(),
                object(),
                seed=17,
                batch_size=1,
            )

    def test_hash_valid_malformed_probe_contract_is_isolated_and_rebuilt(self):
        config = {"evaluation": {"probe_hidden_dim": 5, "probe_steps": 1}}
        table = "compatibility_probe_contract.csv"
        rows = _probe_contract_rows(
            _probe_bundle(hidden_dim=5),
            config,
            seed=self.checkpoint["seed"],
            arm=self.checkpoint["arm"],
        )[table]
        identity = _contract_identity(
            self.run_identity, self.checkpoint, table, 0
        )
        with self.store.writer_lock():
            original, _receipt = _commit_table_fragment(
                self.store,
                identity,
                rows,
                table=table,
                evidence=self.evidence,
                repair_reused_mismatch=True,
            )
        original_bytes = original.payload_path.read_bytes()
        original.payload_path.write_bytes(b"hash-valid-but-not-a-gzip")
        manifest = json.loads(original.manifest_path.read_text(encoding="ascii"))
        manifest["payload"]["bytes"] = original.payload_path.stat().st_size
        manifest["payload"]["sha256"] = hashlib.sha256(
            original.payload_path.read_bytes()
        ).hexdigest()
        original.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )

        with self.store.writer_lock():
            restored, _receipt = _commit_table_fragment(
                self.store,
                identity,
                rows,
                table=table,
                evidence=self.evidence,
                repair_reused_mismatch=True,
            )
        self.assertEqual(restored.payload_path.read_bytes(), original_bytes)
        receipts = [
            json.loads(path.read_text(encoding="ascii"))
            for path in self.store.quarantine_root.glob("*/receipt.json")
        ]
        self.assertEqual(len(receipts), 1)
        self.assertEqual(
            receipts[0]["shard_identity"]["substage"],
            "compatibility_probe_contract",
        )

    def test_hash_valid_semantic_probe_is_quarantined_and_refit(self):
        config = {"evaluation": {"probe_hidden_dim": 5}}
        original = _probe_bundle(hidden_dim=5)
        replacement = _probe_bundle(hidden_dim=5)
        identity = _probe_state_identity(self.run_identity, self.checkpoint, 0)
        with self.store.writer_lock():
            path = _commit_probe_bundle(
                self.store, identity, original, config
            )

        with path.open("wb") as handle:
            torch.save({"invalid": "probe-contract"}, handle)
        manifest_path = self.store.shard_paths(identity, suffix=".pt")[1]
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        manifest["payload"]["bytes"] = path.stat().st_size
        manifest["payload"]["sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )

        with (
            self.store.writer_lock(),
            mock.patch(
                "formal_v2.formal_evaluation_streaming._fit_probe_bundle",
                return_value=replacement,
            ) as fit,
        ):
            restored, reused = _load_or_fit_probes(
                self.store,
                self.run_identity,
                self.checkpoint,
                0,
                model=object(),
                dataset=object(),
                teacher=object(),
                config=config,
                normalization=object(),
                route_normalization=object(),
                batch_size=1,
            )

        self.assertIs(restored, replacement)
        self.assertFalse(reused)
        fit.assert_called_once()
        receipts = [
            json.loads(path.read_text(encoding="ascii"))
            for path in self.store.quarantine_root.glob("*/receipt.json")
        ]
        self.assertEqual(len(receipts), 1)
        self.assertEqual(
            receipts[0]["shard_identity"]["substage"], "probe_state"
        )
        authenticated = self.store.load_completed_shard(identity, suffix=".pt")
        self.assertIsNotNone(authenticated)
        _assert_module_equal(
            self,
            _restore_probe_bundle(
                authenticated.payload_path,
                config,
                seed=self.checkpoint["seed"],
                arm=self.checkpoint["arm"],
            ).response_probe,
            replacement.response_probe,
        )

    def test_completed_table_fragments_recover_into_atomic_scene_bundle(self):
        rows = _scene_rows(0.25)
        first_table = SCENE_TABLES[0]
        first_identity = _table_identity(
            self.run_identity, self.checkpoint, first_table, 12
        )
        with self.store.writer_lock():
            first, _receipt = _commit_table_fragment(
                self.store,
                first_identity,
                rows[first_table],
                table=first_table,
                evidence=self.evidence,
            )
            first_sha = first.manifest["payload"]["sha256"]
            _commit_scene_bundle(
                self.store,
                self.run_identity,
                self.checkpoint,
                scene=12,
                bank_id="bank-a",
                rows_by_table=rows,
                evidence=self.evidence,
            )
        with self.store.writer_lock():
            loaded = _load_scene_bundle(
                self.store,
                self.run_identity,
                self.checkpoint,
                scene=12,
                bank_id="bank-a",
                evidence=self.evidence,
            )
        self.assertIsNotNone(loaded)
        self.assertEqual(set(loaded["tables"]), set(SCENE_TABLES))
        completed = self.store.load_completed_shard(first_identity, suffix=".csv.gz")
        self.assertIsNotNone(completed)
        self.assertEqual(completed.manifest["payload"]["sha256"], first_sha)

    def test_hash_valid_semantic_bundle_is_quarantined_and_recomputed(self):
        rows = _scene_rows(0.25)
        with self.store.writer_lock():
            _commit_scene_bundle(
                self.store,
                self.run_identity,
                self.checkpoint,
                scene=12,
                bank_id="bank-a",
                rows_by_table=rows,
                evidence=self.evidence,
            )
        table_paths = {}
        for table in SCENE_TABLES:
            completed_table = self.store.load_completed_shard(
                _table_identity(
                    self.run_identity, self.checkpoint, table, 12
                ),
                suffix=".csv.gz",
            )
            self.assertIsNotNone(completed_table)
            table_paths[table] = (
                completed_table.payload_path,
                completed_table.manifest_path,
            )
        table_before = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for paths in table_paths.values()
            for path in paths
        }
        identity = _scene_bundle_identity(
            self.run_identity, self.checkpoint, 12
        )
        completed = self.store.load_completed_shard(identity, suffix=".json")
        self.assertIsNotNone(completed)
        bundle = json.loads(completed.payload_path.read_text(encoding="ascii"))
        bundle["small_rows"]["cgs_per_bank.csv"][0]["cgs_auroc"] = 0.75
        encoded = (
            json.dumps(
                bundle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
        completed.payload_path.write_bytes(encoded)
        manifest = json.loads(completed.manifest_path.read_text(encoding="ascii"))
        manifest["payload"]["bytes"] = len(encoded)
        manifest["payload"]["sha256"] = hashlib.sha256(encoded).hexdigest()
        completed.manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
            encoding="ascii",
        )
        with self.store.writer_lock():
            self.assertIsNone(
                _load_scene_bundle(
                    self.store,
                    self.run_identity,
                    self.checkpoint,
                    scene=12,
                    bank_id="bank-a",
                    evidence=self.evidence,
                )
            )
            _commit_scene_bundle(
                self.store,
                self.run_identity,
                self.checkpoint,
                scene=12,
                bank_id="bank-a",
                rows_by_table=rows,
                evidence=self.evidence,
            )
            restored = _load_scene_bundle(
                self.store,
                self.run_identity,
                self.checkpoint,
                scene=12,
                bank_id="bank-a",
                evidence=self.evidence,
            )

        self.assertIsNotNone(restored)
        changed_table = "cgs_per_bank.csv"
        self.assertEqual(
            table_paths[changed_table][0].read_bytes(),
            table_before[table_paths[changed_table][0]][0],
        )
        sibling_paths = [
            path
            for table, paths in table_paths.items()
            if table != changed_table
            for path in paths
        ]
        self.assertEqual(
            {
                path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in sibling_paths
            },
            {path: table_before[path] for path in sibling_paths},
        )
        receipts = [
            json.loads(path.read_text(encoding="ascii"))
            for path in self.store.quarantine_root.glob("*/receipt.json")
        ]
        self.assertEqual(
            {receipt["shard_identity"]["substage"] for receipt in receipts},
            {"cgs_per_bank", "scene_bundle"},
        )

    def test_hash_valid_semantic_table_quarantines_only_table_and_bundle(self):
        rows = _scene_rows(0.25)
        table = "cgs_per_bank.csv"
        with self.store.writer_lock():
            _commit_scene_bundle(
                self.store,
                self.run_identity,
                self.checkpoint,
                scene=12,
                bank_id="bank-a",
                rows_by_table=rows,
                evidence=self.evidence,
            )

        table_results = {
            name: self.store.load_completed_shard(
                _table_identity(
                    self.run_identity, self.checkpoint, name, 12
                ),
                suffix=".csv.gz",
            )
            for name in SCENE_TABLES
        }
        self.assertTrue(all(result is not None for result in table_results.values()))
        sibling_paths = [
            path
            for name, result in table_results.items()
            if name != table
            for path in (result.payload_path, result.manifest_path)
        ]
        siblings_before = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in sibling_paths
        }

        changed_row = dict(rows[table][0])
        changed_row["seed"] = 999
        encoded_table = io.BytesIO()
        row_count = write_csv_fragment_stream(
            encoded_table,
            [changed_row],
            fieldnames=EVALUATION_TABLE_FIELDS[table],
            evidence=self.evidence,
        )
        table_bytes = encoded_table.getvalue()
        changed_table = table_results[table]
        changed_table.payload_path.write_bytes(table_bytes)
        table_manifest = json.loads(
            changed_table.manifest_path.read_text(encoding="ascii")
        )
        table_manifest["payload"]["bytes"] = len(table_bytes)
        table_manifest["payload"]["sha256"] = hashlib.sha256(
            table_bytes
        ).hexdigest()
        changed_table.manifest_path.write_text(
            json.dumps(table_manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )

        bundle_identity = _scene_bundle_identity(
            self.run_identity, self.checkpoint, 12
        )
        bundle_result = self.store.load_completed_shard(
            bundle_identity, suffix=".json"
        )
        self.assertIsNotNone(bundle_result)
        bundle = json.loads(bundle_result.payload_path.read_text(encoding="ascii"))
        bundle["tables"][table]["bytes"] = len(table_bytes)
        bundle["tables"][table]["sha256"] = hashlib.sha256(
            table_bytes
        ).hexdigest()
        bundle["tables"][table]["row_count"] = row_count
        bundle_bytes = (
            json.dumps(bundle, indent=2, sort_keys=True) + "\n"
        ).encode("ascii")
        bundle_result.payload_path.write_bytes(bundle_bytes)
        bundle_manifest = json.loads(
            bundle_result.manifest_path.read_text(encoding="ascii")
        )
        bundle_manifest["payload"]["bytes"] = len(bundle_bytes)
        bundle_manifest["payload"]["sha256"] = hashlib.sha256(
            bundle_bytes
        ).hexdigest()
        bundle_result.manifest_path.write_text(
            json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )

        with self.store.writer_lock():
            self.assertIsNone(
                _load_scene_bundle(
                    self.store,
                    self.run_identity,
                    self.checkpoint,
                    scene=12,
                    bank_id="bank-a",
                    evidence=self.evidence,
                )
            )
            _commit_scene_bundle(
                self.store,
                self.run_identity,
                self.checkpoint,
                scene=12,
                bank_id="bank-a",
                rows_by_table=rows,
                evidence=self.evidence,
            )

        self.assertEqual(
            {
                path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in sibling_paths
            },
            siblings_before,
        )
        receipts = [
            json.loads(path.read_text(encoding="ascii"))
            for path in self.store.quarantine_root.glob("*/receipt.json")
        ]
        self.assertEqual(
            {
                receipt["shard_identity"]["substage"]
                for receipt in receipts
            },
            {"cgs_per_bank", "scene_bundle"},
        )

    def test_partial_replay_with_changed_rows_must_fail_closed(self):
        original = _scene_rows(0.25)
        changed = _scene_rows(0.75)
        first_table = SCENE_TABLES[0]
        with self.store.writer_lock():
            _commit_table_fragment(
                self.store,
                _table_identity(
                    self.run_identity, self.checkpoint, first_table, 12
                ),
                original[first_table],
                table=first_table,
                evidence=self.evidence,
            )
            with self.assertRaisesRegex(RuntimeError, "fragment|shard|receipt|mismatch"):
                _commit_scene_bundle(
                    self.store,
                    self.run_identity,
                    self.checkpoint,
                    scene=12,
                    bank_id="bank-a",
                    rows_by_table=changed,
                    evidence=self.evidence,
                )
        with self.store.writer_lock():
            self.assertIsNone(
                _load_scene_bundle(
                    self.store,
                    self.run_identity,
                    self.checkpoint,
                    scene=12,
                    bank_id="bank-a",
                    evidence=self.evidence,
                )
            )

    def test_scene_rows_are_bound_to_seed_arm_bank_and_scene(self):
        mutations = (
            ("cgs_per_bank.csv", "seed", 23),
            ("compatibility_route_distributions.csv", "arm", "full"),
            ("response_per_bank.csv", "bank_id", "wrong-bank"),
            ("compatibility_pair_effects.csv", "scene_index", 99),
            ("response_pair_effects.csv", "scene_index", 99),
        )
        for index, (table, field, value) in enumerate(mutations):
            with self.subTest(table=table, field=field):
                rows = _scene_rows(0.5)
                rows[table][0][field] = value
                store = EvaluationResumeStore(
                    self.root / f"identity-state-{index}", self.run_identity
                )
                with store.writer_lock():
                    with self.assertRaisesRegex(
                        RuntimeError, "seed|arm|bank|scene|identity"
                    ):
                        _commit_scene_bundle(
                            store,
                            self.run_identity,
                            self.checkpoint,
                            scene=12,
                            bank_id="bank-a",
                            rows_by_table=rows,
                            evidence=self.evidence,
                        )

    def test_scene_table_inventory_rejects_missing_and_extra_tables(self):
        for index, changed in enumerate(("missing", "extra")):
            with self.subTest(changed=changed):
                rows = _scene_rows(0.5)
                if changed == "missing":
                    rows.pop(SCENE_TABLES[-1])
                else:
                    rows["not_an_evaluation_table.csv"] = []
                store = EvaluationResumeStore(
                    self.root / f"inventory-state-{index}", self.run_identity
                )
                with store.writer_lock():
                    with self.assertRaisesRegex(RuntimeError, "inventory"):
                        _commit_scene_bundle(
                            store,
                            self.run_identity,
                            self.checkpoint,
                            scene=12,
                            bank_id="bank-a",
                            rows_by_table=rows,
                            evidence=self.evidence,
                        )

    def test_scene_table_row_count_and_order_contract_fails_closed(self):
        mutations = (
            ("cgs_per_bank.csv", lambda rows: rows.append(dict(rows[0]))),
            ("response_per_bank.csv", lambda rows: rows.clear()),
            ("alignment_shortcut_baselines.csv", lambda rows: rows.pop()),
            ("compatibility_route_distributions.csv", lambda rows: rows.reverse()),
            ("cgs_active_effect_bins.csv", lambda rows: rows.pop(0)),
            ("compatibility_pair_effects.csv", lambda rows: rows.clear()),
            ("response_pair_effects.csv", lambda rows: rows.append(dict(rows[0]))),
        )
        for index, (table, mutate) in enumerate(mutations):
            with self.subTest(table=table):
                rows = _scene_rows(0.5)
                mutate(rows[table])
                store = EvaluationResumeStore(
                    self.root / f"row-contract-state-{index}", self.run_identity
                )
                with store.writer_lock():
                    with self.assertRaisesRegex(
                        RuntimeError, "row|empty|unique|order"
                    ):
                        _commit_scene_bundle(
                            store,
                            self.run_identity,
                            self.checkpoint,
                            scene=12,
                            bank_id="bank-a",
                            rows_by_table=rows,
                            evidence=self.evidence,
                        )

    def test_dataset_pair_universe_scene_bundle_roundtrip(self):
        universe = build_scene_pair_universe(_FakeDataset(), 1)
        rows = self._dataset_pair_scene_rows(universe)
        store = EvaluationResumeStore(
            self.root / "dataset-pair-roundtrip",
            self.run_identity,
        )
        with store.writer_lock():
            committed = _commit_scene_bundle(
                store,
                self.run_identity,
                self.checkpoint,
                scene=universe.scene_index,
                bank_id=universe.bank_id,
                rows_by_table=rows,
                evidence=self.evidence,
                pair_universe=universe,
            )
            loaded = _load_scene_bundle(
                store,
                self.run_identity,
                self.checkpoint,
                scene=universe.scene_index,
                bank_id=universe.bank_id,
                evidence=self.evidence,
                pair_universe=universe,
            )

        self.assertEqual(loaded, committed)
        self.assertEqual(
            committed["pair_inventory"]["compatibility_pair_effects.csv"][
                "row_count"
            ],
            16,
        )
        self.assertEqual(
            committed["pair_inventory"]["response_pair_effects.csv"][
                "row_count"
            ],
            384,
        )

    def test_dataset_pair_universe_rejects_pair_fragment_attacks(self):
        universe = build_scene_pair_universe(_FakeDataset(), 1)
        table = "response_pair_effects.csv"

        for attack in ("missing", "illegal", "synchronized"):
            with self.subTest(attack=attack):
                store = EvaluationResumeStore(
                    self.root / f"dataset-pair-{attack}",
                    self.run_identity,
                )
                rows = self._dataset_pair_scene_rows(universe)
                with store.writer_lock():
                    _commit_scene_bundle(
                        store,
                        self.run_identity,
                        self.checkpoint,
                        scene=universe.scene_index,
                        bank_id=universe.bank_id,
                        rows_by_table=rows,
                        evidence=self.evidence,
                        pair_universe=universe,
                    )

                    table_identity = _table_identity(
                        self.run_identity,
                        self.checkpoint,
                        table,
                        universe.scene_index,
                    )
                    if attack == "missing":
                        # Leave the manifest behind so the resume store records
                        # the incomplete pair fragment in quarantine.
                        store.shard_paths(
                            table_identity,
                            suffix=".csv.gz",
                        )[0].unlink()
                    elif attack == "illegal":
                        tampered = [dict(row) for row in rows[table]]
                        tampered[-1]["query_index"] = 99
                        tampered[-1]["pair_id"] = (
                            f"{universe.bank_id}:3:2:2:99"
                        )
                        tampered.sort(key=lambda row: row["pair_id"])
                        self._rewrite_authenticated_pair_fragment(
                            store,
                            table,
                            universe.scene_index,
                            tampered,
                        )
                    else:
                        self._rewrite_authenticated_pair_fragment(
                            store,
                            table,
                            universe.scene_index,
                            rows[table][:-1],
                        )

                    self.assertIsNone(
                        _load_scene_bundle(
                            store,
                            self.run_identity,
                            self.checkpoint,
                            scene=universe.scene_index,
                            bank_id=universe.bank_id,
                            evidence=self.evidence,
                            pair_universe=universe,
                        )
                    )

                quarantined = {
                    json.loads(path.read_text(encoding="ascii"))[
                        "shard_identity"
                    ]["substage"]
                    for path in store.quarantine_root.glob("*/receipt.json")
                }
                self.assertIn("scene_bundle", quarantined)
                self.assertIn("response_pair_effects", quarantined)

    def test_route_optional_metrics_follow_condition_matrix(self):
        table = "compatibility_route_distributions.csv"
        required = {
            "matched_minus_alternative_mean": 0.25,
            "matched_minus_alternative_median": 0.25,
            "absolute_difference_p90": 0.25,
        }
        cases = (
            (
                "missing route is empty and has no pairs",
                "null",
                "MISSING",
                0,
                {
                    "matched_minus_alternative_mean": None,
                    "matched_minus_alternative_median": None,
                    "absolute_difference_p90": None,
                    "overclassification_rate": None,
                },
                True,
            ),
            (
                "assessed null route has all metrics",
                "null",
                "ASSESSED",
                1,
                {**required, "overclassification_rate": 0.25},
                True,
            ),
            (
                "assessed gray route omits overclassification",
                "gray",
                "ASSESSED",
                1,
                {**required, "overclassification_rate": None},
                True,
            ),
            (
                "assessed active route omits overclassification",
                "active",
                "ASSESSED",
                1,
                {**required, "overclassification_rate": None},
                True,
            ),
            (
                "missing route cannot contain pairs",
                "null",
                "MISSING",
                1,
                {**required, "overclassification_rate": None},
                False,
            ),
            (
                "missing route cannot contain a metric",
                "null",
                "MISSING",
                0,
                {
                    "matched_minus_alternative_mean": 0.25,
                    "matched_minus_alternative_median": None,
                    "absolute_difference_p90": None,
                    "overclassification_rate": None,
                },
                False,
            ),
            (
                "assessed route requires pairs",
                "null",
                "ASSESSED",
                0,
                {**required, "overclassification_rate": 0.25},
                False,
            ),
            (
                "assessed route requires its core metrics",
                "null",
                "ASSESSED",
                1,
                {
                    "matched_minus_alternative_mean": None,
                    "matched_minus_alternative_median": 0.25,
                    "absolute_difference_p90": 0.25,
                    "overclassification_rate": 0.25,
                },
                False,
            ),
            (
                "gray route cannot report overclassification",
                "gray",
                "ASSESSED",
                1,
                {**required, "overclassification_rate": 0.25},
                False,
            ),
            (
                "active route cannot report overclassification",
                "active",
                "ASSESSED",
                1,
                {**required, "overclassification_rate": 0.25},
                False,
            ),
        )
        for name, route, status, pair_count, updates, valid in cases:
            with self.subTest(case=name):
                rows = _scene_rows(0.5)
                row = rows[table][("null", "gray", "active").index(route)]
                row["route"] = route
                row["condition_status"] = status
                row["pair_count"] = pair_count
                row.update(updates)
                if valid:
                    self._validate_scene_table_for_both_storage_paths(table, rows)
                else:
                    self._assert_scene_table_rejected_on_both_storage_paths(
                        table, rows
                    )

    def test_response_pair_swap_metrics_follow_match_status_matrix(self):
        table = "response_pair_effects.csv"
        for status in ("exact", "fallback", "failed"):
            expected_missing = status != "exact"
            for action_missing in (False, True):
                for advantage_missing in (False, True):
                    valid = (
                        action_missing == expected_missing
                        and advantage_missing == expected_missing
                    )
                    with self.subTest(
                        status=status,
                        action_missing=action_missing,
                        advantage_missing=advantage_missing,
                    ):
                        rows = _scene_rows(0.5)
                        row = rows[table][0]
                        row["wrong_action_match_status"] = status
                        row["action_swap_mse"] = (
                            None if action_missing else 0.25
                        )
                        row["response_advantage_vs_action_swap"] = (
                            None if advantage_missing else 0.0
                        )
                        if valid:
                            self._validate_scene_table_for_both_storage_paths(
                                table, rows
                            )
                        else:
                            self._assert_scene_table_rejected_on_both_storage_paths(
                                table, rows
                            )

    def test_response_bank_exact_swap_metrics_follow_exact_counts(self):
        table = "response_per_bank.csv"
        metric_groups = (
            (
                "probe_action_swap_exact_count",
                (
                    "unified_response_probe_action_swap_exact_patch_nmse",
                    "probe_action_swap_active_patch_nmse",
                ),
            ),
            (
                "native_action_swap_exact_count",
                (
                    "native_target_free_action_swap_exact_full_channel_nmse",
                    "native_action_swap_full_channel_nmse",
                    "native_latent_action_swap_exact_target_nmse",
                    "native_latent_action_swap_nmse",
                ),
            ),
        )
        for count_field, metric_fields in metric_groups:
            for exact_count in (0, 1):
                expected_missing = exact_count == 0
                with self.subTest(
                    count_field=count_field,
                    exact_count=exact_count,
                    state="valid",
                ):
                    rows = _scene_rows(0.5)
                    row = rows[table][0]
                    row[count_field] = exact_count
                    for field in metric_fields:
                        row[field] = None if expected_missing else 0.25
                    self._validate_scene_table_for_both_storage_paths(table, rows)

                for missing_field in metric_fields:
                    with self.subTest(
                        count_field=count_field,
                        exact_count=exact_count,
                        missing_field=missing_field,
                    ):
                        rows = _scene_rows(0.5)
                        row = rows[table][0]
                        row[count_field] = exact_count
                        for field in metric_fields:
                            if field == missing_field:
                                row[field] = 0.25 if expected_missing else None
                            else:
                                row[field] = None if expected_missing else 0.25
                        self._assert_scene_table_rejected_on_both_storage_paths(
                            table, rows
                        )

    def test_response_bank_null_metrics_follow_null_patch_count(self):
        table = "response_per_bank.csv"
        metric_fields = (
            "native_null_delta_rms_mean",
            "native_null_violation_rate",
            "native_latent_null_delta_rms_mean",
            "native_latent_null_violation_rate",
        )
        for null_count in (0, 1):
            expected_missing = null_count == 0
            with self.subTest(null_count=null_count, state="valid"):
                rows = _scene_rows(0.5)
                row = rows[table][0]
                row["native_null_patch_count"] = null_count
                for field in metric_fields:
                    row[field] = None if expected_missing else 0.25
                self._validate_scene_table_for_both_storage_paths(table, rows)

            for missing_field in metric_fields:
                with self.subTest(
                    null_count=null_count,
                    missing_field=missing_field,
                ):
                    rows = _scene_rows(0.5)
                    row = rows[table][0]
                    row["native_null_patch_count"] = null_count
                    for field in metric_fields:
                        if field == missing_field:
                            row[field] = 0.25 if expected_missing else None
                        else:
                            row[field] = None if expected_missing else 0.25
                    self._assert_scene_table_rejected_on_both_storage_paths(
                        table, rows
                    )

    def test_response_bank_transition_skill_may_be_unavailable(self):
        table = "response_per_bank.csv"
        rows = _scene_rows(0.5)
        rows[table][0]["native_transition_skill"] = None
        self._validate_scene_table_for_both_storage_paths(table, rows)

    def test_checkpoint_then_bank_merge_is_canonical(self):
        class Dataset:
            bank_ids = np.asarray(["bank-z", "bank-a"])

            @staticmethod
            def indices_for_role(role):
                return {
                    "source_final_unseen_bank": np.asarray([0]),
                    "target": np.asarray([1]),
                }[role]

        self.assertEqual(_frozen_evaluation_scenes(Dataset()), (0, 1))
        scenes = _ordered_evaluation_scenes(Dataset())
        self.assertEqual(scenes, (1, 0))
        checkpoints = (
            self.checkpoint,
            {"seed": 17, "arm": "alignment", "sha256": "a" * 64},
        )
        table = "cgs_per_bank.csv"
        fragments = []
        with self.store.writer_lock():
            for checkpoint in checkpoints:
                for scene in scenes:
                    bank = str(Dataset.bank_ids[scene])
                    result, _receipt = _commit_table_fragment(
                        self.store,
                        _table_identity(
                            self.run_identity, checkpoint, table, scene
                        ),
                        [
                            {
                                "seed": checkpoint["seed"],
                                "arm": checkpoint["arm"],
                                "bank_id": bank,
                            }
                        ],
                        table=table,
                        evidence=self.evidence,
                    )
                    fragments.append(result.payload_path)
        output = self.root / table
        merge_csv_fragments(
            fragments,
            output,
            fieldnames=EVALUATION_TABLE_FIELDS[table],
        )
        with output.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            [(row["arm"], row["bank_id"]) for row in rows],
            [
                ("endpoint", "bank-a"),
                ("endpoint", "bank-z"),
                ("alignment", "bank-a"),
                ("alignment", "bank-z"),
            ],
        )

    def test_duplicate_evaluation_bank_fails_closed(self):
        class DuplicateDataset:
            bank_ids = np.asarray(["same-bank", "same-bank"])

            @staticmethod
            def indices_for_role(role):
                return {
                    "source_final_unseen_bank": np.asarray([0]),
                    "target": np.asarray([1]),
                }[role]

        with self.assertRaisesRegex(RuntimeError, "one unique scene per bank ID"):
            _ordered_evaluation_scenes(DuplicateDataset())

    def _assert_scene_prediction_parity(self, device):
        probes = _probe_bundle(hidden_dim=5)
        probes.compatibility_probe.to(device)
        probes.response_probe.to(device)
        for probe in probes.variant_probes.values():
            probe.to(device)

        compatibility_scenes = np.asarray([2, 2], dtype=np.int64)
        compatibility = {
            "scene_indices": compatibility_scenes,
            "features": np.linspace(-1.0, 1.0, 6, dtype=np.float64).reshape(2, 3),
        }
        response_scenes = np.asarray([2, 2, 2], dtype=np.int64)
        response = {
            "scene_indices": response_scenes,
            "source_targets": np.linspace(-0.5, 0.5, 6, dtype=np.float64).reshape(3, 2),
        }
        response["targets"] = response["source_targets"] + 0.25
        feature_names = (
            "features",
            "no_action_features",
            "action_swap_features",
            "without_map_features",
            "without_map_zero_action_features",
            "edit_only_features",
            "edit_only_zero_action_features",
            "csi_only_features",
            "oracle_x_features",
            "oracle_x_zero_action_features",
        )
        base = np.linspace(-2.0, 2.0, 12, dtype=np.float64).reshape(3, 4)
        for offset, name in enumerate(feature_names):
            response[name] = base + offset / 10.0

        with (
            mock.patch(
                "formal_v2.formal_evaluation_streaming.route_dataset",
                return_value=object(),
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._compatibility_dataset",
                return_value=compatibility,
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._response_probe_dataset",
                return_value=response,
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.predict_binary_probe",
                wraps=predict_binary_probe,
            ) as binary_prediction,
            mock.patch(
                "formal_v2.formal_evaluation_streaming.predict_response_probe",
                wraps=predict_response_probe,
            ) as response_prediction,
        ):
            prepared = _prepare_scene_evaluation(
                object(),
                probes,
                object(),
                object(),
                {},
                object(),
                object(),
                scene=2,
                batch_size=1,
            )

        self.assertEqual(binary_prediction.call_count, 1)
        self.assertEqual(binary_prediction.call_args.args[1].shape[0], 2)
        self.assertEqual(response_prediction.call_count, 7)
        self.assertTrue(
            all(call.args[1].shape[0] == 3 for call in response_prediction.call_args_list)
        )
        np.testing.assert_array_equal(
            prepared.compatibility_probabilities,
            predict_binary_probe(probes.compatibility_probe, compatibility["features"]),
        )
        expected_response = response["source_targets"] + predict_response_probe(
            probes.response_probe,
            response["features"],
            response["no_action_features"],
        )
        np.testing.assert_array_equal(prepared.response_prediction, expected_response)
        np.testing.assert_array_equal(
            prepared.response_swap_prediction,
            response["source_targets"]
            + predict_response_probe(
                probes.response_probe,
                response["action_swap_features"],
                response["no_action_features"],
            ),
        )
        np.testing.assert_array_equal(
            prepared.response_no_action_prediction,
            response["source_targets"]
            + predict_response_probe(
                probes.response_probe,
                response["no_action_features"],
                response["no_action_features"],
            ),
        )
        contrast_variants = {"without_map", "edit_only", "oracle_x"}
        for name, probe in probes.variant_probes.items():
            np.testing.assert_array_equal(
                prepared.response_variant_predictions[name],
                response["source_targets"]
                + predict_response_probe(
                    probe,
                    response[f"{name}_features"],
                    (
                        response[f"{name}_zero_action_features"]
                        if name in contrast_variants
                        else None
                    ),
                ),
            )

    def _assert_merged_and_scene_probe_parity(self, device):
        probes = _probe_bundle(hidden_dim=5)
        probes.compatibility_probe.to(device)
        probes.response_probe.to(device)
        for probe in probes.variant_probes.values():
            probe.to(device)
        generator = np.random.default_rng(81103)
        scene_lengths = (2, 5, 3)

        compatibility_scenes = [
            generator.normal(size=(length, 3)).astype(np.float64)
            for length in scene_lengths
        ]
        merged_compatibility = predict_binary_probe(
            probes.compatibility_probe,
            np.concatenate(compatibility_scenes, axis=0),
        )
        feature_names = (
            "main",
            "without_map",
            "edit_only",
            "csi_only",
            "oracle_x",
        )
        response_scenes = {
            name: [
                generator.normal(size=(length, 4)).astype(np.float64)
                for length in scene_lengths
            ]
            for name in feature_names
        }
        zero_scenes = {
            name: [
                generator.normal(size=(length, 4)).astype(np.float64)
                for length in scene_lengths
            ]
            for name in ("main", "without_map", "edit_only", "oracle_x")
        }
        response_probes = {
            "main": probes.response_probe,
            **probes.variant_probes,
        }
        scene_inputs = tuple(
            _SceneInputs(
                scene=index,
                routed=object(),
                compatibility={
                    "scene_indices": np.full(length, index, dtype=np.int64),
                    "features": compatibility_scenes[index],
                },
                response={
                    "scene_indices": np.full(length, index, dtype=np.int64),
                    "source_targets": np.zeros((length, 2), dtype=np.float32),
                    **{
                        f"{name}_features": response_scenes[name][index]
                        for name in feature_names
                    },
                    "features": response_scenes["main"][index],
                    "action_swap_features": response_scenes["main"][index],
                    "no_action_features": zero_scenes["main"][index],
                    **{
                        f"{name}_zero_action_features": zero_scenes[name][index]
                        for name in ("without_map", "edit_only", "oracle_x")
                    },
                },
            )
            for index, length in enumerate(scene_lengths)
        )
        prepared = _predict_checkpoint_scenes(probes, scene_inputs)
        scene_compatibility = np.concatenate(
            [
                prepared[index].compatibility_probabilities
                for index in range(len(scene_lengths))
            ],
            axis=0,
        )
        np.testing.assert_array_equal(scene_compatibility, merged_compatibility)

        for name, probe in response_probes.items():
            values = response_scenes[name]
            zeros = zero_scenes.get(name)
            merged = predict_response_probe(
                probe,
                np.concatenate(values, axis=0),
                np.concatenate(zeros, axis=0) if zeros is not None else None,
            )
            if name == "main":
                scene_values = np.concatenate(
                    [
                        prepared[index].response_prediction
                        for index in range(len(scene_lengths))
                    ],
                    axis=0,
                )
            else:
                scene_values = np.concatenate(
                    [
                        prepared[index].response_variant_predictions[name]
                        for index in range(len(scene_lengths))
                    ],
                    axis=0,
                )
            np.testing.assert_array_equal(scene_values, merged, err_msg=name)

    def test_scene_prediction_rejects_broadcastable_response_width(self):
        probes = _probe_bundle(hidden_dim=5)
        compatibility = {
            "scene_indices": np.asarray([2, 2], dtype=np.int64),
            "features": np.zeros((2, 3), dtype=np.float64),
        }
        response = {
            "scene_indices": np.asarray([2, 2, 2], dtype=np.int64),
            "targets": np.ones((3, 2), dtype=np.float64),
            "source_targets": np.zeros((3, 2), dtype=np.float64),
            "features": np.zeros((3, 4), dtype=np.float64),
            "no_action_features": np.zeros((3, 4), dtype=np.float64),
        }
        with (
            mock.patch(
                "formal_v2.formal_evaluation_streaming.route_dataset",
                return_value=object(),
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._compatibility_dataset",
                return_value=compatibility,
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._response_probe_dataset",
                return_value=response,
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.predict_binary_probe",
                return_value=np.zeros(2, dtype=np.float64),
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.predict_response_probe",
                return_value=np.zeros((3, 1), dtype=np.float64),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "shape"):
                _prepare_scene_evaluation(
                    object(),
                    probes,
                    object(),
                    object(),
                    {},
                    object(),
                    object(),
                    scene=2,
                    batch_size=1,
                )

    def test_scene_probe_predictions_are_bitwise_exact_on_cpu(self):
        self._assert_scene_prediction_parity("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA runtime is unavailable")
    def test_scene_probe_predictions_are_bitwise_exact_on_cuda(self):
        self._assert_scene_prediction_parity("cuda:0")

    def test_merged_and_scene_probe_predictions_match_on_cpu(self):
        self._assert_merged_and_scene_probe_parity("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA runtime is unavailable")
    def test_merged_and_scene_probe_predictions_match_on_cuda(self):
        self._assert_merged_and_scene_probe_parity("cuda:0")

    def test_scene_shard_consumes_cache_without_repredicting(self):
        class Dataset:
            bank_ids = np.asarray(["bank-a"])
            base_map_cluster_ids = np.asarray(["cluster-a"])

            @staticmethod
            def canonical_base_map_digest(_scene):
                return "base-digest"

        compatibility = {
            "scene_indices": np.asarray([0, 0]),
            "bank_ids": np.asarray(["bank-a", "bank-a"]),
            "routes": np.asarray(["active", "active"]),
            "labels": np.asarray([1, 0]),
            "native_training_scores": np.asarray([0.8, 0.2]),
            "native_scores": np.asarray([0.7, 0.3]),
            "city_ids": np.asarray(["city-a", "city-a"]),
            "pair_ids": np.asarray(["pair-a", "pair-a"]),
            "physical_distances": np.asarray([1.0, 1.0]),
        }
        response = {
            "scene_indices": np.asarray([0]),
            "bank_ids": np.asarray(["bank-a"]),
            "routes": np.asarray(["active"]),
            "wrong_action_match_status": np.asarray(["exact"]),
            "targets": np.asarray([[1.0, -1.0]]),
            "source_targets": np.asarray([[0.0, 0.0]]),
            "city_ids": np.asarray(["city-a"]),
        }
        prepared = _SceneEvaluation(
            scene=0,
            routed=object(),
            compatibility=compatibility,
            compatibility_probabilities=np.asarray([0.9, 0.1]),
            response=response,
            response_prediction=np.asarray([[0.75, -0.75]]),
            response_swap_prediction=np.asarray([[0.25, -0.25]]),
            response_no_action_prediction=np.asarray([[0.0, 0.0]]),
            response_variant_predictions={},
        )
        probes = _ProbeBundle(
            compatibility_probe=object(),
            compatibility_selection={"selected_family": "linear"},
            shortcut_probes={},
            response_probe=object(),
            variant_probes={},
        )
        with (
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._response_probe_dataset",
                side_effect=AssertionError("scene shard rebuilt response rows"),
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.predict_binary_probe",
                side_effect=AssertionError("scene shard reran binary prediction"),
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.predict_response_probe",
                side_effect=AssertionError("scene shard reran response prediction"),
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.binary_auroc",
                return_value=0.5,
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy.spearman_correlation",
                return_value=0.0,
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._evaluation_scope",
                return_value="source_final_unseen_bank",
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._canonical_bank_digest",
                return_value="bank-digest",
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._alignment_shortcut_rows",
                return_value=[],
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._active_effect_bin_rows",
                return_value=[],
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._paired_score_differences",
                return_value=np.asarray([0.0]),
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._response_effect_rows",
                return_value=[],
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._native_mask_cover_metrics",
                return_value={},
            ),
            mock.patch(
                "formal_v2.formal_evaluation_streaming.legacy._compatibility_effect_rows",
                return_value=[],
            ),
        ):
            rows = _evaluate_scene(
                object(),
                probes,
                Dataset(),
                object(),
                {"evaluation": {"null_score_equivalence_margin": 0.0}},
                object(),
                object(),
                scene=0,
                seed=17,
                arm="endpoint",
                batch_size=1,
                scene_evaluation=prepared,
            )

        self.assertEqual(set(rows), set(SCENE_TABLES))
        self.assertEqual(
            rows["response_per_bank.csv"][0][
                "unified_response_probe_active_patch_nmse"
            ],
            0.0625,
        )


def _work_checkpoints(count):
    arms = ("endpoint", "alignment", "response", "full")
    return tuple(
        {
            "seed": 20270001 + index,
            "arm": arms[index % len(arms)],
            "sha256": f"{index + 1:x}" * 64,
        }
        for index in range(count)
    )


def _run_identity():
    return EvaluationRunIdentity(
        code_revision="a" * 40,
        source_tree_sha256="1" * 64,
        config_sha256="2" * 64,
        dataset_sha256="3" * 64,
        migration_accepted_sha256="4" * 64,
        legacy_checkpoint_inventory_sha256="5" * 64,
        qualification_gate_sha256="6" * 64,
        factorial_gate_sha256="7" * 64,
        runtime_provenance_sha256="8" * 64,
        run_nonce="9" * 64,
        compute_plan_sha256="a" * 64,
        execution_profile=EvaluationExecutionProfile(
            execution_devices=("cuda:0", "cuda:1"),
            batch_size=1,
        ),
        output_schema_id=STREAMING_EVALUATION_SCHEMA,
        output_schema_sha256=evaluation_output_schema_sha256(),
    )


def _fill_module(module, offset):
    with torch.no_grad():
        for index, parameter in enumerate(module.parameters()):
            values = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(
                parameter.shape
            )
            parameter.copy_(values / max(1, parameter.numel()) + offset + index)
    module.eval()
    return module


def _probe_bundle(hidden_dim):
    compatibility = _fill_module(
        CompatibilityProbe(3, "mlp2", hidden_dim), 0.1
    )
    response = _fill_module(ActionResponseProbe(4, 2, hidden_dim), 0.2)
    response.zero_preserving = True
    shortcut_names = (
        "constant",
        "csi_only",
        "map_only",
        "scene_id_only",
        "edit_status_xor",
        "variant_id_matcher",
    )
    shortcuts = {}
    for index, name in enumerate(shortcut_names):
        shortcuts[name] = {
            "probe": (
                None
                if name == "constant"
                else _fill_module(
                    CompatibilityProbe(2, "linear", hidden_dim), 0.3 + index
                )
            ),
            "selected_family": "constant" if name == "constant" else "linear",
            "source_train_auroc": 0.5 + index / 100.0,
        }
    variants = {}
    for index, name in enumerate(
        ("without_map", "edit_only", "csi_only", "oracle_x")
    ):
        probe = _fill_module(ActionResponseProbe(4, 2, hidden_dim), 1.0 + index)
        probe.zero_preserving = name != "csi_only"
        variants[name] = probe
    return _ProbeBundle(
        compatibility_probe=compatibility,
        compatibility_selection={
            "selected_family": "mlp2",
            "selection_nll": 0.4,
            "selection_auroc": 0.6,
            "candidate_metrics": [
                {"family": "mlp2", "selection_nll": 0.4, "selection_auroc": 0.6},
                {"family": "linear", "selection_nll": 0.5, "selection_auroc": 0.55},
            ],
        },
        shortcut_probes=shortcuts,
        response_probe=response,
        variant_probes=variants,
    )


def _assert_module_equal(test, observed, expected):
    test.assertEqual(type(observed), type(expected))
    test.assertEqual(set(observed.state_dict()), set(expected.state_dict()))
    for name, value in expected.state_dict().items():
        test.assertTrue(torch.equal(observed.state_dict()[name], value), name)


def _scene_rows(cgs_auroc):
    rows = {}
    evidence_fields = {
        "artifact_label",
        "dataset_sha256",
        "config_sha256",
        "fixture",
        "scientific_use",
        "source_tree_sha256",
        "requirements_lock_sha256",
        "runtime_provenance_sha256",
    }
    for table in SCENE_TABLES:
        core_fields = [
            field
            for field in EVALUATION_TABLE_FIELDS[table]
            if field not in evidence_fields
        ]

        def row():
            value = {}
            for field in core_fields:
                if field in SCENE_TEXT_FIELDS:
                    value[field] = f"{field}-value"
                elif field in SCENE_INTEGER_FIELDS:
                    value[field] = 1
                elif field in OPTIONAL_SCENE_NUMERIC_FIELDS:
                    value[field] = None
                else:
                    value[field] = 0.25
            value.update({"seed": 17, "arm": "endpoint", "bank_id": "bank-a"})
            return value

        if table == "alignment_shortcut_baselines.csv":
            rows[table] = []
            for baseline in (
                "constant",
                "csi_only",
                "map_only",
                "scene_id_only",
                "edit_status_xor",
                "variant_id_matcher",
            ):
                value = row()
                value["baseline"] = baseline
                rows[table].append(value)
        elif table == "compatibility_route_distributions.csv":
            rows[table] = []
            for route in ("null", "gray", "active"):
                value = row()
                value["route"] = route
                value["condition_status"] = "ASSESSED"
                value["pair_count"] = 1
                value["matched_minus_alternative_mean"] = 0.25
                value["matched_minus_alternative_median"] = 0.25
                value["absolute_difference_p90"] = 0.25
                value["overclassification_rate"] = 0.25 if route == "null" else None
                rows[table].append(value)
        elif table == "cgs_active_effect_bins.csv":
            rows[table] = []
            for effect_bin in ("low", "medium_low", "medium_high", "high"):
                value = row()
                value["effect_bin"] = effect_bin
                rows[table].append(value)
        else:
            value = row()
            if table == "cgs_per_bank.csv":
                value["cgs_auroc"] = cgs_auroc
                value["route"] = "active"
            elif table in {
                "compatibility_pair_effects.csv",
                "response_pair_effects.csv",
            }:
                value["pair_id"] = "bank-a:0:1:0:0"
                value["scene_index"] = 12
                value["source_world"] = 0
                value["target_world"] = 1
                value["position_index"] = 0
                value["route"] = "active"
                if table == "response_pair_effects.csv":
                    value["query_index"] = 0
                    value["wrong_action_match_status"] = "exact"
                    value["wrong_action_world"] = 2
                    value["action_swap_mse"] = 0.25
                    value["response_advantage_vs_action_swap"] = 0.0
            elif table == "response_per_bank.csv":
                for prefix in ("probe", "native"):
                    value[f"{prefix}_action_swap_exact_count"] = 1
                    value[f"{prefix}_action_swap_fallback_count"] = 0
                    value[f"{prefix}_action_swap_failed_count"] = 0
                    value[f"{prefix}_action_swap_exact_fraction"] = 1.0
                for field in OPTIONAL_SCENE_NUMERIC_FIELDS:
                    if field in value and "null_" not in field:
                        value[field] = 0.25
                value["native_null_patch_count"] = 0
                value["native_null_delta_rms_mean"] = None
                value["native_null_violation_rate"] = None
                value["native_latent_null_delta_rms_mean"] = None
                value["native_latent_null_violation_rate"] = None
            rows[table] = [value]
    return rows


def _synthetic_evaluation_gate():
    return {
        "schema_version": "csi-pairs-v6-evaluation-gate-v3",
        "status": "PASS",
        "passed": True,
        "scientific_claim_status": "CANDIDATE_NOT_CLAIM",
        "gate_vector": {},
        "g3_subgates": {},
        "g4_subgates": {},
    }


def _scene_rows_for(scene, bank_id, cgs_auroc):
    rows = _scene_rows(cgs_auroc)
    for table_rows in rows.values():
        for row in table_rows:
            row["bank_id"] = str(bank_id)
            if "scene_index" in row:
                row["scene_index"] = int(scene)
            if "pair_id" in row:
                row["pair_id"] = f"{bank_id}:0:1:0:0"
    return rows


if __name__ == "__main__":
    unittest.main()
