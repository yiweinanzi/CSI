from __future__ import annotations

import copy
import gc
import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
import weakref
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

from formal_v2 import formal_evaluation_streaming as streaming
from formal_v2.formal_evaluation_compare import CSV_CONTRACTS
from formal_v2.formal_evaluation_streaming import EVALUATION_TABLE_FIELDS
from formal_v2.formal_evaluation_subset_compare import (
    CHECKPOINT_SELECTION_RULE,
    GATE_REQUIRED_TABLES,
    IMPLEMENTATION_EVIDENCE_FIELDS,
    REPORT_SCHEMA,
    SCENE_TABLES,
    SELECTION_RULE,
    WORKER_FRAGMENT_SCHEMA,
    WORKER_REQUEST_SCHEMA,
    _directory_tree_snapshot,
    _evaluate_streaming_checkpoint,
    _runtime_digest,
    _selection_sha256,
    compare_evaluation_tables,
    gate_aggregate_report,
    main,
    run_worker_request,
    select_legacy_checkpoints,
    select_real_evaluation_subset,
    source_identity,
    validate_report_destination,
    validate_worker_pair,
    write_report_atomic,
)


class _SelectionDataset:
    is_fixture = False
    scene_roles = np.asarray(
        [
            "target",
            "source_final_unseen_bank",
            "target",
            "target",
            "source_final_unseen_bank",
            "target",
        ]
    )
    city_ids = np.asarray(["target-b", "source-z", "target-a", "target-b", "source-a", "target-a"])
    bank_ids = np.asarray(["bank-z", "bank-y", "bank-c", "bank-a", "bank-b", "bank-d"])
    scene_ids = np.asarray([f"scene-{index}" for index in range(6)])
    base_map_cluster_ids = np.asarray([f"cluster-{index}" for index in range(6)])

    def indices_for_role(self, role):
        return np.flatnonzero(self.scene_roles == role)

    def canonical_base_map_digest(self, scene):
        return f"{int(scene) + 10:064x}"


class _FixtureSelectionDataset(_SelectionDataset):
    is_fixture = True


def _complete_tables(*, rows_per_table=1):
    result = {}
    for table, fields in EVALUATION_TABLE_FIELDS.items():
        float_fields = set(CSV_CONTRACTS[table].float_fields)
        rows = []
        for row_index in range(rows_per_table):
            row = {}
            for field in fields:
                if field == "candidate_metrics":
                    row[field] = {
                        "linear": {"nll": 0.25 + row_index, "auroc": 0.75}
                    }
                elif field in float_fields:
                    row[field] = 0.25 + row_index
                elif field == "seed":
                    row[field] = 100 + row_index
                elif field in {
                    "n",
                    "steps",
                    "hidden_dim",
                    "pair_count",
                    "scene_index",
                    "source_world",
                    "target_world",
                    "position_index",
                    "query_index",
                    "wrong_action_world",
                    "probe_action_swap_exact_count",
                    "probe_action_swap_fallback_count",
                    "probe_action_swap_failed_count",
                    "native_action_swap_exact_count",
                    "native_action_swap_fallback_count",
                    "native_action_swap_failed_count",
                    "native_null_patch_count",
                    "n_active_patches",
                }:
                    row[field] = row_index + 1
                elif field == "fixture":
                    row[field] = False
                else:
                    row[field] = f"{field}-{row_index}"
            rows.append(row)
        result[table] = rows
    return result


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_record(path: Path, *, identity_sha256: str | None = None) -> dict:
    stat = path.stat()
    record = {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_bytes(path.read_bytes()),
    }
    if identity_sha256 is not None:
        record["identity_sha256"] = identity_sha256
    return record


def _source_file_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_bytes(path.read_bytes()),
        "git_tracked": True,
    }


def _bind_synthetic_evidence(
    tables: dict,
    *,
    dataset_sha256: str,
    config_sha256: str,
    source_tree_sha256: str,
    requirements_lock_sha256: str,
    runtime_provenance_sha256: str,
) -> dict:
    output = copy.deepcopy(tables)
    values = {
        "artifact_label": "formal-subset-test",
        "dataset_sha256": dataset_sha256,
        "config_sha256": config_sha256,
        "fixture": False,
        "scientific_use": "CANDIDATE_NOT_CLAIM",
        "source_tree_sha256": source_tree_sha256,
        "requirements_lock_sha256": requirements_lock_sha256,
        "runtime_provenance_sha256": runtime_provenance_sha256,
    }
    for rows in output.values():
        for row in rows:
            for field, value in values.items():
                row[field] = value
    return output


def _synthetic_worker_pair(root: Path):
    frozen_root = root / "frozen-source"
    candidate_root = root / "candidate-source"
    common_root = root / "common"
    for directory in (frozen_root / "formal_v2", candidate_root / "formal_v2", common_root):
        directory.mkdir(parents=True)

    harness = common_root / "formal_evaluation_subset_compare.py"
    frozen_evaluation = frozen_root / "formal_v2" / "formal_evaluation.py"
    candidate_evaluation = candidate_root / "formal_v2" / "formal_evaluation.py"
    candidate_streaming = (
        candidate_root / "formal_v2" / "formal_evaluation_streaming.py"
    )
    harness.write_bytes(b"# committed comparator\n")
    frozen_evaluation.write_bytes(b"# frozen evaluation\n")
    candidate_evaluation.write_bytes(b"# candidate evaluation\n")
    candidate_streaming.write_bytes(b"# candidate streaming\n")

    config = common_root / "formal.json"
    dataset = common_root / "dataset.npz"
    config.write_bytes(b"{}\n")
    dataset.write_bytes(b"formal-dataset")
    dataset_sha256 = _sha256_bytes(dataset.read_bytes())
    config_sha256 = _sha256_bytes(config.read_bytes())
    shared_files = [
        _file_record(config, identity_sha256=config_sha256),
        _file_record(dataset, identity_sha256=dataset_sha256),
    ]
    input_bindings = {
        "config_sha256": config_sha256,
        "dataset_sha256": dataset_sha256,
        "files": sorted(shared_files, key=lambda row: row["path"]),
    }
    read_only_files = [
        {
            "path": row["path"],
            "before": dict(row),
            "after": {
                field: row[field]
                for field in ("path", "bytes", "mtime_ns", "sha256")
            },
            "unchanged": True,
        }
        for row in input_bindings["files"]
    ]

    scenes = [
        {
            "scene_index": 3,
            "scene_id": "source-scene",
            "scene_role": "source_final_unseen_bank",
            "city_id": "source-city",
            "bank_id": "bank-a",
            "base_map_cluster_id": "cluster-a",
            "canonical_base_map_digest": "3" * 64,
            "canonical_bank_digest": "4" * 64,
        },
        {
            "scene_index": 7,
            "scene_id": "target-scene",
            "scene_role": "target",
            "city_id": "target-city",
            "bank_id": "bank-b",
            "base_map_cluster_id": "cluster-b",
            "canonical_base_map_digest": "5" * 64,
            "canonical_bank_digest": "6" * 64,
        },
    ]
    checkpoints = [{"seed": 100, "arm": "full", "sha256": "7" * 64}]
    selection = {
        "scene_rule": SELECTION_RULE,
        "checkpoint_rule": CHECKPOINT_SELECTION_RULE,
        "scenes_in_execution_order": scenes,
        "checkpoints_in_execution_order": checkpoints,
        "sha256": _selection_sha256(scenes, checkpoints),
    }

    requirements_sha256 = "8" * 64
    frozen_tree = "1" * 64
    candidate_tree = "2" * 64
    frozen_runtime = {
        "schema_version": "runtime-test-v1",
        "source_tree_sha256": frozen_tree,
        "requirements_lock_sha256": requirements_sha256,
    }
    candidate_runtime = {
        "schema_version": "runtime-test-v1",
        "source_tree_sha256": candidate_tree,
        "requirements_lock_sha256": requirements_sha256,
    }
    frozen_runtime_sha = _runtime_digest(frozen_runtime)
    candidate_runtime_sha = _runtime_digest(candidate_runtime)
    base_tables = _complete_tables()
    frozen_tables = _bind_synthetic_evidence(
        base_tables,
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
        source_tree_sha256=frozen_tree,
        requirements_lock_sha256=requirements_sha256,
        runtime_provenance_sha256=frozen_runtime_sha,
    )
    candidate_tables = _bind_synthetic_evidence(
        base_tables,
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
        source_tree_sha256=candidate_tree,
        requirements_lock_sha256=requirements_sha256,
        runtime_provenance_sha256=candidate_runtime_sha,
    )

    request_frozen = common_root / "frozen.request.json"
    request_candidate = common_root / "candidate.request.json"
    request_frozen.write_bytes(b'{"role":"frozen_legacy"}\n')
    request_candidate.write_bytes(b'{"role":"streaming_candidate"}\n')
    harness_record = _source_file_record(harness)

    def fragment(role, source_root, commit, runtime, runtime_sha, files, tables):
        request_path = request_frozen if role == "frozen_legacy" else request_candidate
        request_record = _file_record(request_path)
        snapshot = {
            "root": str((root / "legacy" / "evaluation").resolve()),
            "entry_count": 1,
            "total_file_bytes": 10,
            "entries_sha256": "9" * 64,
        }
        return {
            "schema_version": WORKER_FRAGMENT_SCHEMA,
            "role": role,
            "created_utc": "2026-08-31T00:00:00Z",
            "request": {
                "path": request_record["path"],
                "bytes": request_record["bytes"],
                "sha256": request_record["sha256"],
            },
            "source_identity": {
                "source_root": str(source_root.resolve()),
                "git_root": str(source_root.resolve()),
                "git_commit": commit,
                "tracked_tree_clean": True,
                "tracked_status": [],
                "untracked_scientific_paths": [],
                "tracked_diff_sha256": hashlib.sha256(b"").hexdigest(),
                "files": files,
                "source_tree_sha256": runtime["source_tree_sha256"],
                "runtime_provenance_sha256": runtime_sha,
            },
            "runtime_provenance": runtime,
            "input_bindings": copy.deepcopy(input_bindings),
            "selection": copy.deepcopy(selection),
            "tables": tables,
            "same_source_merged_tables": (
                None if role == "frozen_legacy" else copy.deepcopy(tables)
            ),
            "read_only": {
                "passed": True,
                "files": copy.deepcopy(read_only_files),
                "legacy_evaluation": (
                    {
                        "root": snapshot["root"],
                        "before": snapshot,
                        "after": copy.deepcopy(snapshot),
                        "unchanged": True,
                    }
                    if role == "frozen_legacy"
                    else None
                ),
            },
            "resources": {
                "elapsed_seconds": 1.0,
                "maximum_resident_set_bytes": 1024,
                "maximum_cuda_allocated_bytes": 2048,
            },
        }

    frozen = fragment(
        "frozen_legacy",
        frozen_root,
        "a" * 40,
        frozen_runtime,
        frozen_runtime_sha,
        {
            "formal_evaluation_subset_compare.py": harness_record,
            "formal_evaluation.py": _source_file_record(frozen_evaluation),
        },
        frozen_tables,
    )
    candidate = fragment(
        "streaming_candidate",
        candidate_root,
        "b" * 40,
        candidate_runtime,
        candidate_runtime_sha,
        {
            "formal_evaluation_subset_compare.py": harness_record,
            "formal_evaluation.py": _source_file_record(candidate_evaluation),
            "formal_evaluation_streaming.py": _source_file_record(candidate_streaming),
        },
        candidate_tables,
    )
    expected = {
        "expected_frozen_root": frozen_root,
        "expected_candidate_root": candidate_root,
        "expected_frozen_commit": "a" * 40,
        "expected_candidate_commit": "b" * 40,
        "expected_frozen_evaluation_sha256": _sha256_bytes(
            frozen_evaluation.read_bytes()
        ),
    }
    return frozen, candidate, expected


class EvaluationSubsetSelectionTests(unittest.TestCase):
    def test_real_subset_is_stable_and_covers_source_and_every_target_city(self):
        canonical_bank = lambda _dataset, scene: f"{int(scene) + 100:064x}"
        with mock.patch(
            "formal_v2.formal_evaluation_subset_compare.legacy._canonical_bank_digest",
            side_effect=canonical_bank,
        ):
            selected = select_real_evaluation_subset(
                _SelectionDataset(), source_scenes=1, target_scenes_per_city=1
            )
            repeated = select_real_evaluation_subset(_SelectionDataset())

        self.assertEqual([row["bank_id"] for row in selected], ["bank-a", "bank-b", "bank-c"])
        self.assertEqual(
            {row["scene_role"] for row in selected},
            {"source_final_unseen_bank", "target"},
        )
        target_cities = {
            row["city_id"] for row in selected if row["scene_role"] == "target"
        }
        self.assertEqual(target_cities, {"target-a", "target-b"})
        self.assertEqual(repeated, selected)

    def test_fixture_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "synthetic/fixture"):
            select_real_evaluation_subset(_FixtureSelectionDataset())

    def test_checkpoint_selection_uses_seed_then_frozen_arm_order(self):
        rows = [
            {"seed": 2, "arm": "full", "sha256": "4" * 64},
            {"seed": 1, "arm": "full", "sha256": "3" * 64},
            {"seed": 1, "arm": "endpoint", "sha256": "1" * 64},
            {"seed": 1, "arm": "response", "sha256": "2" * 64},
        ]
        selected = select_legacy_checkpoints(
            rows,
            ("endpoint", "alignment", "response", "full"),
            checkpoint_count=2,
        )
        self.assertEqual(
            [(row["seed"], row["arm"]) for row in selected],
            [(1, "endpoint"), (1, "response")],
        )
        selected_full = select_legacy_checkpoints(
            rows,
            ("endpoint", "alignment", "response", "full"),
            checkpoint_count=2,
            checkpoint_arm="full",
        )
        self.assertEqual([row["seed"] for row in selected_full], [1, 2])


class EvaluationSubsetStreamingCheckpointTests(unittest.TestCase):
    def test_streaming_checkpoint_prepares_only_one_scene_at_a_time(self):
        class Dataset:
            scene_roles = np.asarray(
                ["source_final_unseen_bank", "target", "target"]
            )

        probes = mock.Mock()
        probes.compatibility_probe = object()
        probes.response_probe = object()
        probes.variant_probes = {
            name: object()
            for name in ("without_map", "edit_only", "csi_only", "oracle_x")
        }

        def response_prediction(_probe, features, _zero_action_features=None):
            return np.zeros((len(features), 2), dtype=np.float64)

        def compatibility_dataset(*args, **_kwargs):
            scene_values = np.asarray(args[5], dtype=np.int64)
            self.assertEqual(scene_values.shape, (1,))
            scene = int(scene_values[0])
            return {
                "scene_indices": np.full(2, scene, dtype=np.int64),
                "features": np.arange(6, dtype=np.float64).reshape(2, 3),
            }

        def response_dataset(*args, **_kwargs):
            scene_values = np.asarray(args[5], dtype=np.int64)
            self.assertEqual(scene_values.shape, (1,))
            scene = int(scene_values[0])
            response = {
                "scene_indices": np.full(3, scene, dtype=np.int64),
                "source_targets": np.arange(6, dtype=np.float64).reshape(3, 2),
                "targets": np.arange(6, dtype=np.float64).reshape(3, 2) + 1.0,
            }
            for offset, name in enumerate(
                (
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
            ):
                response[name] = (
                    np.arange(12, dtype=np.float64).reshape(3, 4) + offset
                )
            return response

        evaluated_scenes = []

        def scene_rows(*_args, **kwargs):
            scene = int(kwargs["scene"])
            prepared = kwargs["scene_evaluation"]
            self.assertEqual(prepared.scene, scene)
            self.assertEqual(prepared.compatibility["features"].shape[0], 2)
            self.assertEqual(prepared.response["features"].shape[0], 3)
            evaluated_scenes.append(scene)
            return {table: [{"scene": scene}] for table in SCENE_TABLES}

        with (
            mock.patch.object(
                streaming,
                "_prepare_scene_evaluation",
                wraps=streaming._prepare_scene_evaluation,
            ) as prepare,
            mock.patch.object(streaming, "route_dataset", return_value=object()) as route,
            mock.patch.object(
                streaming.legacy,
                "_compatibility_dataset",
                side_effect=compatibility_dataset,
            ) as compatibility_dataset,
            mock.patch.object(
                streaming.legacy,
                "_response_probe_dataset",
                side_effect=response_dataset,
            ) as response_dataset,
            mock.patch.object(
                streaming,
                "predict_binary_probe",
                return_value=np.zeros(2, dtype=np.float64),
            ) as compatibility_prediction,
            mock.patch.object(
                streaming,
                "predict_response_probe",
                side_effect=response_prediction,
            ) as predict_response,
            mock.patch.object(
                streaming, "_evaluate_scene", new=scene_rows
            ),
        ):
            tables = _evaluate_streaming_checkpoint(
                object(),
                probes,
                Dataset(),
                object(),
                {},
                object(),
                object(),
                (2, 0, 1),
                seed=17,
                arm="endpoint",
                batch_size=4,
            )

        self.assertEqual(prepare.call_count, 3)
        self.assertEqual(
            [call.kwargs["scene"] for call in prepare.call_args_list],
            [2, 0, 1],
        )
        self.assertEqual(route.call_count, 3)
        self.assertTrue(
            all(np.asarray(call.args[3]).shape == (1,) for call in route.call_args_list)
        )
        self.assertEqual(compatibility_dataset.call_count, 3)
        self.assertEqual(response_dataset.call_count, 3)
        self.assertEqual(compatibility_prediction.call_count, 3)
        self.assertTrue(
            all(call.args[1].shape == (2, 3) for call in compatibility_prediction.call_args_list)
        )
        self.assertEqual(predict_response.call_count, 21)
        self.assertTrue(
            all(call.args[1].shape[0] == 3 for call in predict_response.call_args_list)
        )
        self.assertEqual(evaluated_scenes, [2, 0, 1])
        expected_rows = [{"scene": 2}, {"scene": 0}, {"scene": 1}]
        self.assertEqual(tables, {table: expected_rows for table in SCENE_TABLES})

    def test_streaming_checkpoint_releases_each_scene_payload_before_the_next(self):
        class Dataset:
            scene_roles = np.asarray(
                ["source_final_unseen_bank", "target", "target"]
            )

        class Payload:
            def __init__(self, scene):
                self.scene = scene

        previous = None
        prepared_scenes = []
        evaluated_scenes = []

        def prepare_scene(*_args, **kwargs):
            nonlocal previous
            gc.collect()
            if previous is not None:
                self.assertIsNone(previous())
            payload = Payload(int(kwargs["scene"]))
            previous = weakref.ref(payload)
            prepared_scenes.append(payload.scene)
            return payload

        def evaluate_scene(*_args, **kwargs):
            payload = kwargs["scene_evaluation"]
            scene = int(kwargs["scene"])
            self.assertEqual(payload.scene, scene)
            evaluated_scenes.append(scene)
            return {table: [{"scene": scene}] for table in SCENE_TABLES}

        with (
            mock.patch.object(
                streaming, "_prepare_scene_evaluation", new=prepare_scene
            ),
            mock.patch.object(streaming, "_evaluate_scene", new=evaluate_scene),
        ):
            _evaluate_streaming_checkpoint(
                object(),
                object(),
                Dataset(),
                object(),
                {},
                object(),
                object(),
                (2, 0, 1),
                seed=17,
                arm="endpoint",
                batch_size=1,
            )

        gc.collect()
        self.assertIsNotNone(previous)
        self.assertIsNone(previous())
        self.assertEqual(prepared_scenes, [2, 0, 1])
        self.assertEqual(evaluated_scenes, [2, 0, 1])


class EvaluationSubsetTableCompareTests(unittest.TestCase):
    def test_complete_nine_tables_are_bitwise_exact(self):
        legacy = _complete_tables(rows_per_table=2)
        streaming = _complete_tables(rows_per_table=2)

        report = compare_evaluation_tables(legacy, streaming)
        aggregates = gate_aggregate_report(legacy, streaming)

        self.assertTrue(report["exact"])
        self.assertTrue(report["bitwise_float_equal"])
        self.assertTrue(report["field_contract_equal"])
        self.assertTrue(report["row_order_equal"])
        self.assertEqual(report["table_count"], 9)
        self.assertGreater(report["float_comparisons"], 0)
        self.assertEqual(report["max_absolute_difference"], 0.0)
        self.assertTrue(aggregates["exact"])
        self.assertEqual(aggregates["table_count"], 5)
        self.assertEqual(set(aggregates["tables"]), set(GATE_REQUIRED_TABLES))
        self.assertFalse(aggregates["formal_gate_executed"])

    def test_one_ulp_float_change_fails_and_reports_maximum_difference(self):
        legacy = _complete_tables()
        streaming = _complete_tables()
        streaming["cgs_per_bank.csv"][0]["cgs_auroc"] = float(
            np.nextafter(0.25, 1.0)
        )

        report = compare_evaluation_tables(legacy, streaming)
        table = report["tables"]["cgs_per_bank.csv"]

        self.assertFalse(report["exact"])
        self.assertFalse(report["bitwise_float_equal"])
        self.assertGreater(report["max_absolute_difference"], 0.0)
        self.assertEqual(table["difference_examples"][0]["reason"], "float_bits")
        self.assertNotEqual(
            table["legacy_scientific_ordered_sha256"],
            table["streaming_scientific_ordered_sha256"],
        )

    def test_cross_source_ignores_only_declared_implementation_evidence(self):
        legacy = _complete_tables()
        candidate = _complete_tables()
        for rows in legacy.values():
            for row in rows:
                row["source_tree_sha256"] = "1" * 64
                row["runtime_provenance_sha256"] = "2" * 64
        for rows in candidate.values():
            for row in rows:
                row["source_tree_sha256"] = "3" * 64
                row["runtime_provenance_sha256"] = "4" * 64

        report = compare_evaluation_tables(
            legacy,
            candidate,
            implementation_fields=IMPLEMENTATION_EVIDENCE_FIELDS,
        )
        aggregates = gate_aggregate_report(
            legacy,
            candidate,
            implementation_fields=IMPLEMENTATION_EVIDENCE_FIELDS,
        )

        self.assertTrue(report["exact"])
        self.assertTrue(aggregates["exact"])
        self.assertEqual(
            report["comparison_profile"], "cross_source_scientific_payload"
        )
        table = report["tables"]["cgs_per_bank.csv"]
        self.assertEqual(
            table["legacy_scientific_ordered_sha256"],
            table["streaming_scientific_ordered_sha256"],
        )
        self.assertNotEqual(
            table["legacy_full_ordered_sha256"],
            table["streaming_full_ordered_sha256"],
        )

        candidate["cgs_per_bank.csv"][0]["dataset_sha256"] = "changed"
        self.assertFalse(
            compare_evaluation_tables(
                legacy,
                candidate,
                implementation_fields=IMPLEMENTATION_EVIDENCE_FIELDS,
            )["exact"]
        )

    def test_row_reordering_and_field_reordering_fail_closed(self):
        legacy = _complete_tables(rows_per_table=2)
        streaming = _complete_tables(rows_per_table=2)
        streaming["compatibility_pair_effects.csv"] = list(
            reversed(streaming["compatibility_pair_effects.csv"])
        )
        reordered = streaming["response_probe_contract.csv"][0]
        streaming["response_probe_contract.csv"][0] = dict(
            reversed(tuple(reordered.items()))
        )

        report = compare_evaluation_tables(legacy, streaming)

        self.assertFalse(report["exact"])
        self.assertFalse(report["row_order_equal"])
        self.assertFalse(report["field_contract_equal"])
        self.assertFalse(
            report["tables"]["compatibility_pair_effects.csv"]["row_order_equal"]
        )
        self.assertFalse(
            report["tables"]["response_probe_contract.csv"]["field_order_equal"]
        )

    def test_incomplete_table_inventory_is_rejected(self):
        legacy = _complete_tables()
        streaming = _complete_tables()
        streaming.pop("response_per_bank.csv")
        with self.assertRaisesRegex(ValueError, "complete nine-table inventory"):
            compare_evaluation_tables(legacy, streaming)


class EvaluationSubsetWorkerPairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.frozen, self.candidate, self.expected = _synthetic_worker_pair(
            self.root
        )

    def tearDown(self):
        self.temporary.cleanup()

    def validate(self):
        validate_worker_pair(self.frozen, self.candidate, **self.expected)

    def test_distinct_workers_with_complete_tables_are_valid(self):
        self.validate()
        frozen_path = self.root / "frozen.fragment.json"
        candidate_path = self.root / "candidate.fragment.json"
        write_report_atomic(frozen_path, self.frozen)
        write_report_atomic(candidate_path, self.candidate)
        validate_worker_pair(
            json.loads(frozen_path.read_text(encoding="utf-8")),
            json.loads(candidate_path.read_text(encoding="utf-8")),
            **self.expected,
        )
        comparison = compare_evaluation_tables(
            self.frozen["tables"],
            self.candidate["tables"],
            implementation_fields=IMPLEMENTATION_EVIDENCE_FIELDS,
        )
        aggregates = gate_aggregate_report(
            self.frozen["tables"],
            self.candidate["tables"],
            implementation_fields=IMPLEMENTATION_EVIDENCE_FIELDS,
        )
        self.assertTrue(comparison["exact"])
        self.assertEqual(comparison["table_count"], 9)
        self.assertTrue(aggregates["exact"])
        self.assertEqual(aggregates["table_count"], 5)

    def test_same_source_root_or_commit_is_rejected(self):
        self.candidate["source_identity"]["source_root"] = self.frozen[
            "source_identity"
        ]["source_root"]
        self.expected["expected_candidate_root"] = self.expected[
            "expected_frozen_root"
        ]
        with self.assertRaisesRegex(RuntimeError, "distinct source roots"):
            self.validate()

        self.frozen, self.candidate, self.expected = _synthetic_worker_pair(
            self.root / "commit-case"
        )
        commit = self.frozen["source_identity"]["git_commit"]
        self.candidate["source_identity"]["git_commit"] = commit
        self.expected["expected_candidate_commit"] = commit
        with self.assertRaisesRegex(RuntimeError, "distinct commits"):
            self.validate()

    def test_input_or_selection_binding_change_is_rejected(self):
        self.candidate["input_bindings"]["dataset_sha256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            self.validate()

        self.frozen, self.candidate, self.expected = _synthetic_worker_pair(
            self.root / "selection-case"
        )
        self.frozen["selection"]["sha256"] = "0" * 64
        self.candidate["selection"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "selection digest"):
            self.validate()

    def test_runtime_source_and_read_only_bindings_are_rejected(self):
        self.candidate["source_identity"]["runtime_provenance_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "runtime provenance"):
            self.validate()

        self.frozen, self.candidate, self.expected = _synthetic_worker_pair(
            self.root / "read-only-case"
        )
        self.frozen["read_only"]["legacy_evaluation"]["after"][
            "entries_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "legacy evaluation read-only"):
            self.validate()

    def test_empty_table_or_incomplete_resources_are_rejected(self):
        self.candidate["tables"]["response_per_bank.csv"] = []
        with self.assertRaisesRegex(RuntimeError, "table is empty"):
            self.validate()

        self.frozen, self.candidate, self.expected = _synthetic_worker_pair(
            self.root / "resource-case"
        )
        self.candidate["resources"]["maximum_cuda_allocated_bytes"] = 0
        with self.assertRaisesRegex(RuntimeError, "resource evidence"):
            self.validate()

    def test_non_bitwise_float_produces_failed_authoritative_comparison(self):
        self.validate()
        self.candidate["tables"]["cgs_per_bank.csv"][0]["cgs_auroc"] = float(
            np.nextafter(0.25, 1.0)
        )
        validate_worker_pair(self.frozen, self.candidate, **self.expected)
        comparison = compare_evaluation_tables(
            self.frozen["tables"],
            self.candidate["tables"],
            implementation_fields=IMPLEMENTATION_EVIDENCE_FIELDS,
        )
        self.assertFalse(comparison["exact"])
        self.assertFalse(comparison["bitwise_float_equal"])


class EvaluationSubsetReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.legacy = self.root / "legacy"
        self.source = self.root / "source"
        self.external = self.root / "external"
        self.legacy.mkdir()
        self.source.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def test_report_destination_must_be_external(self):
        with self.assertRaisesRegex(ValueError, "outside the legacy run"):
            validate_report_destination(
                self.legacy / "report.json", self.legacy, source_root=self.source
            )
        with self.assertRaisesRegex(ValueError, "outside the source tree"):
            validate_report_destination(
                self.source / "report.json", self.legacy, source_root=self.source
            )
        protected = self.root / "dataset.npz"
        protected.write_bytes(b"dataset")
        with self.assertRaisesRegex(ValueError, "authenticated input"):
            validate_report_destination(
                protected,
                self.legacy,
                source_root=self.source,
                protected_files=(protected,),
            )
        accepted = validate_report_destination(
            self.external / "report.json", self.legacy, source_root=self.source
        )
        self.assertEqual(accepted, (self.external / "report.json").resolve())

    def test_atomic_report_replaces_complete_json(self):
        target = self.external / "report.json"
        write_report_atomic(target, {"schema_version": REPORT_SCHEMA, "status": "OLD"})
        write_report_atomic(target, {"schema_version": REPORT_SCHEMA, "status": "PASS"})

        payload = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_atomic_json_round_trip_preserves_scientific_field_order(self):
        target = self.external / "fragment.json"
        tables = _complete_tables()
        write_report_atomic(target, {"tables": tables})

        payload = json.loads(target.read_text(encoding="utf-8"))
        for table, fields in EVALUATION_TABLE_FIELDS.items():
            self.assertEqual(tuple(payload["tables"][table][0]), fields)

    def test_atomic_replace_failure_preserves_previous_report(self):
        target = self.external / "report.json"
        old = {"schema_version": REPORT_SCHEMA, "status": "OLD"}
        write_report_atomic(target, old)

        with mock.patch(
            "formal_v2.formal_evaluation_subset_compare.os.replace",
            side_effect=OSError("injected replace failure"),
        ):
            with self.assertRaisesRegex(OSError, "injected replace failure"):
                write_report_atomic(
                    target, {"schema_version": REPORT_SCHEMA, "status": "NEW"}
                )

        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), old)
        self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_recursive_snapshot_detects_in_place_child_overwrite(self):
        evaluation = self.root / "legacy-evaluation"
        evaluation.mkdir()
        child = evaluation / "existing.csv"
        child.write_bytes(b"original")
        before = _directory_tree_snapshot(evaluation)

        child.write_bytes(b"modified")
        after = _directory_tree_snapshot(evaluation)

        self.assertEqual(before["entry_count"], after["entry_count"])
        self.assertEqual(before["total_file_bytes"], after["total_file_bytes"])
        self.assertNotEqual(before["entries_sha256"], after["entries_sha256"])

    def test_worker_rejects_wrong_formal_device_contract_before_data_loading(self):
        request = self.external / "request.json"
        request.parent.mkdir(parents=True)
        request.write_text(
            json.dumps(
                {
                    "schema_version": WORKER_REQUEST_SCHEMA,
                    "role": "frozen_legacy",
                    "device": "cuda:1",
                }
            ),
            encoding="utf-8",
        )
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
            with self.assertRaisesRegex(RuntimeError, "CUDA_VISIBLE_DEVICES=0,1"):
                run_worker_request(request)

    def test_worker_rejects_single_visible_gpu_before_data_loading(self):
        request = self.external / "single-visible-gpu-request.json"
        request.parent.mkdir(parents=True)
        request.write_text(
            json.dumps(
                {
                    "schema_version": WORKER_REQUEST_SCHEMA,
                    "role": "frozen_legacy",
                    "device": "cuda:0",
                }
            ),
            encoding="utf-8",
        )
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaisesRegex(RuntimeError, "CUDA_VISIBLE_DEVICES=0,1"):
                run_worker_request(request)

    def test_compare_cli_reads_cross_source_report_shape(self):
        fake_report = {
            "status": "PASS",
            "passed": True,
            "report_path": str((self.external / "report.json").resolve()),
            "layers": {
                "cross_source_end_to_end": {
                    "comparison": {
                        "table_count": 9,
                        "max_absolute_difference": 0.0,
                    }
                }
            },
        }
        argv = [
            "compare",
            "--config",
            "config.json",
            "--dataset",
            "dataset.npz",
            "--legacy-run",
            "legacy",
            "--report",
            "report.json",
            "--frozen-source-root",
            "frozen",
        ]
        output = io.StringIO()
        with mock.patch(
            "formal_v2.formal_evaluation_subset_compare.run_cross_source_subset_comparison",
            return_value=fake_report,
        ) as run, redirect_stdout(output):
            result = main(argv)

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["table_count"], 9)
        run.assert_called_once()


class EvaluationSubsetSourceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_untracked_scientific_module_breaks_commit_closure(self):
        source = self.root / "source"
        formal = source / "formal_v2"
        formal.mkdir(parents=True)
        evaluation = formal / "formal_evaluation.py"
        evaluation.write_text("# tracked evaluation\n", encoding="utf-8")
        subprocess.run(("git", "init", "-q", str(source)), check=True)
        subprocess.run(("git", "-C", str(source), "add", "."), check=True)
        subprocess.run(
            (
                "git",
                "-C",
                str(source),
                "-c",
                "user.name=Subset Test",
                "-c",
                "user.email=subset@example.invalid",
                "commit",
                "-q",
                "-m",
                "fixture",
            ),
            check=True,
        )

        clean = source_identity(source, require_streaming=False)
        self.assertTrue(clean["tracked_tree_clean"])
        self.assertEqual(clean["untracked_scientific_paths"], [])
        self.assertEqual(
            set(clean["files"]["formal_evaluation.py"]),
            {"path", "bytes", "sha256", "git_tracked"},
        )

        injected = formal / "injected.py"
        injected.write_text("VALUE = 1\n", encoding="utf-8")
        dirty = source_identity(source, require_streaming=False)
        self.assertFalse(dirty["tracked_tree_clean"])
        self.assertEqual(
            dirty["untracked_scientific_paths"], ["formal_v2/injected.py"]
        )

    def test_symbolic_link_source_root_is_rejected(self):
        source = self.root / "source"
        source.mkdir()
        linked = self.root / "linked"
        linked.symlink_to(source, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "symbolic link"):
            source_identity(linked, require_streaming=False)


if __name__ == "__main__":
    unittest.main()
