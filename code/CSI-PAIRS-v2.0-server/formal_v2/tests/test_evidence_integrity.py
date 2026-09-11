from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from formal_v2.formal_claim_controls import (
    SHUFFLED_SYSTEMS,
    _require_distinct_checkpoint_hash,
    _retention_assessment,
    _run_adapter as run_claim_control_adapter,
    _shuffled_assessment,
    _validate_pairing_permutations,
    _validate_retention_probe_checkpoint,
)
from formal_v2.formal_controls import (
    CONTROL_IDS,
    G4_CONCAT_CONTROL_IDS,
    REPORT_ONLY_CONTROL_IDS,
    _control_superiority_interval,
    _validate_manifest as validate_resource_manifest,
    _validate_resource,
    _validate_resource_index,
    _validate_v3_profiler,
)
from formal_v2.external_adapters.resource_control import (
    _evaluate_localization,
    _payload_resource_parameter_count,
)
from formal_v2.formal_external import (
    _c1_model_assessment,
    _external_gate_state,
    _resolve_adapter_command,
    _validate_manifest as validate_external_manifest,
    _validate_six_condition_rows,
    run_external_baselines,
)
from formal_v2.formal_io import (
    sha256_file,
    write_csv,
    write_json,
)
from formal_v2.formal_statistics import (
    interval_decision,
    paired_sign_flip_test,
)
from formal_v2.external_adapters.wigatr_protocol import SIX_CONDITIONS as CONDITIONS


ROOT = Path(__file__).resolve().parents[2]


class EvidenceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.dataset = SimpleNamespace(
            bank_ids=np.asarray(["bank-a", "bank-b"]),
            city_ids=np.asarray(["city-a", "city-b"]),
            position_ids=np.asarray([["p0"], ["p1"]]),
            scene_roles=np.asarray(["source_final_unseen_bank", "target"]),
            position_roles=np.asarray([["query"], ["query"]]),
            base_map_cluster_ids=np.asarray(["cluster-a", "cluster-b"]),
        )
        self.units = {
            "u-a": SimpleNamespace(scene=0, position=0, csi_context_sha256="a" * 64),
            "u-b": SimpleNamespace(scene=1, position=0, csi_context_sha256="b" * 64),
        }
        self.contract = {
            (unit_id, condition): {
                "base_map_cluster_id": str(
                    self.dataset.base_map_cluster_ids[unit.scene]
                ),
                "map_sha256": f"{index + 1:064x}",
                "action_sha256": f"{index + 101:064x}",
            }
            for unit_id, unit in self.units.items()
            for index, condition in enumerate(CONDITIONS)
        }

    def _rows(self, value="7.0"):
        rows = []
        for unit_id, unit in self.units.items():
            for condition in CONDITIONS:
                contract = self.contract[(unit_id, condition)]
                rows.append(
                    {
                        "unit_id": unit_id,
                        "model_name": "honest-model",
                        "condition": condition,
                        "city_id": str(self.dataset.city_ids[unit.scene]),
                        "bank_id": str(self.dataset.bank_ids[unit.scene]),
                        "position_id": str(self.dataset.position_ids[unit.scene, unit.position]),
                        "localization_error_m": value,
                        "csi_context_sha256": unit.csi_context_sha256,
                        "base_map_cluster_id": contract["base_map_cluster_id"],
                        "map_sha256": contract["map_sha256"],
                        "action_sha256": contract["action_sha256"],
                        "query_count": "1",
                    }
                )
        return rows

    def test_resource_localization_authentication_is_constant_in_result_rows(self):
        evidence = {
            "dataset_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "fixture": False,
        }
        config = {
            "localization": {
                "label_draws": 1,
                "label_budgets": [0],
                "sigma_min": 0.001,
            },
            "model": {"state_dim": 2},
        }

        def evaluate(scene_count):
            dataset = SimpleNamespace(
                source_path=ROOT / "unused-dataset.npz",
                is_fixture=False,
                city_ids=np.full(scene_count, "target-a"),
                bank_ids=np.asarray([f"bank-{index}" for index in range(scene_count)]),
                base_map_cluster_ids=np.full(scene_count, "cluster-a"),
                position_ids=np.full((scene_count, 1), "query-a"),
                positions=np.zeros((scene_count, 1, 2), dtype=np.float64),
                indices_for_role=lambda role: np.arange(scene_count) if role == "target" else np.asarray([]),
                canonical_base_map_digest=lambda scene: "c" * 64,
            )
            representations = {
                scene: np.zeros((1, 2), dtype=np.float64)
                for scene in range(scene_count)
            }
            with (
                patch(
                    "formal_v2.external_adapters.resource_control.city_support_candidates",
                    return_value=[],
                ),
                patch(
                    "formal_v2.external_adapters.resource_control.eligible_query_indices",
                    return_value=np.asarray([0]),
                ),
                patch(
                    "formal_v2.external_adapters.resource_control.adapt_position_head",
                    return_value=object(),
                ),
                patch(
                    "formal_v2.external_adapters.resource_control.predict_position_distribution",
                    return_value=(np.zeros((1, 2)), None, None),
                ),
                patch(
                    "formal_v2.external_adapters.resource_control._canonical_bank_digest",
                    return_value="d" * 64,
                ),
                patch("formal_v2.external_adapters.resource_control.sha256_file") as file_hash,
                patch("formal_v2.external_adapters.resource_control.evidence_context") as context,
            ):
                rows = _evaluate_localization(
                    "equal_flop_alignment",
                    representations,
                    dataset,
                    config,
                    1,
                    None,
                    object(),
                    evidence["dataset_sha256"],
                    evidence["config_sha256"],
                    evidence["fixture"],
                )
            self.assertEqual(len(rows), scene_count)
            self.assertEqual(file_hash.call_count, 0)
            self.assertEqual(context.call_count, 0)
            self.assertTrue(
                all(
                    row["dataset_sha256"] == evidence["dataset_sha256"]
                    and row["config_sha256"] == evidence["config_sha256"]
                    for row in rows
                )
            )

        evaluate(1)
        evaluate(10_000)

    def test_shipped_wiser_is_style_control_and_not_c1_eligible(self):
        manifest = json.loads(
            (ROOT / "formal_v2/external_adapters/all_map_adapters_v1.json").read_text()
        )
        validate_external_manifest(manifest)
        wiser = next(row for row in manifest["adapters"] if row["model_name"] == "WiSER")
        self.assertEqual(wiser["implementation_status"], "style-controlled-implementation")
        self.assertFalse(wiser["c1_eligible"])
        self.assertEqual(
            [row["model_name"] for row in manifest["adapters"] if row["c1_eligible"]],
            ["Wi-GATr", "PMNet"],
        )
        resource_registry = json.loads(
            (ROOT / "formal_v2/configs/waibu_resources_v1.json").read_text()
        )
        wiser_resource = next(
            row for row in resource_registry["resources"]
            if row["resource_id"] == "arxiv:2606.04770v1"
        )
        self.assertEqual(
            wiser_resource["implementation_status"],
            "style-controlled-implementation",
        )
        self.assertEqual(
            wiser_resource["allowed_name"],
            "WiSER style-controlled implementation",
        )

    def test_external_baselines_authenticate_every_role_the_adapters_read(self):
        manifest_path = ROOT / "formal_v2/external_adapters/all_map_adapters_v1.json"
        expected_roles = (
            "source_encoder_train",
            "source_method_selection",
            "source_final_unseen_bank",
            "target",
        )
        with patch(
            "formal_v2.formal_external.validate_resource_registry",
            return_value=[],
        ), patch(
            "formal_v2.formal_data_verification.require_verified_roles_from_root",
            side_effect=RuntimeError("stop after role audit"),
        ) as require_roles, self.assertRaisesRegex(RuntimeError, "stop after role audit"):
            run_external_baselines({}, self.dataset, manifest_path, ROOT / "unused-run")
        require_roles.assert_called_once()
        self.assertEqual(require_roles.call_args.args[3], expected_roles)

    def test_external_gate_separates_scientific_block_from_engineering_failure(self):
        self.assertEqual(
            _external_gate_state(False, []),
            {
                "status": "BLOCKED",
                "passed": False,
                "engineering_complete": True,
            },
        )
        self.assertEqual(
            _external_gate_state(True, []),
            {
                "status": "PASS",
                "passed": True,
                "engineering_complete": True,
            },
        )
        self.assertEqual(
            _external_gate_state(
                False,
                [{"adapter_id": "failed-adapter", "reason": "nonzero exit"}],
            ),
            {
                "status": "INCOMPLETE_FAIL_CLOSED",
                "passed": False,
                "engineering_complete": False,
            },
        )
        self.assertEqual(
            _external_gate_state(
                True,
                [{"adapter_id": "sigmap-controlled-csi-pairs-v1", "reason": "nonzero exit"}],
                blocking_failures=[],
            ),
            {
                "status": "PASS",
                "passed": True,
                "engineering_complete": True,
            },
        )

    def test_external_adapter_command_must_execute_hashed_source(self):
        manifest = json.loads(
            (ROOT / "formal_v2/external_adapters/all_map_adapters_v1.json").read_text()
        )
        manifest["adapters"][0]["command"] = [
            "/usr/bin/env",
            "python3",
            "-c",
            "print('fabricated')",
            "{adapter_source}",
        ]
        with self.assertRaisesRegex(ValueError, "authenticated source"):
            validate_external_manifest(manifest)

    def test_external_adapter_id_cannot_escape_stage_output(self):
        source = ROOT / "formal_v2/external_adapters/all_map_adapters_v1.json"
        manifest = json.loads(source.read_text())
        manifest["adapters"][0]["adapter_id"] = "../../../qualification"
        with self.assertRaisesRegex(ValueError, "safe path component"):
            validate_external_manifest(manifest)

    def test_external_adapter_config_path_hash_and_command_are_frozen(self):
        source = ROOT / "formal_v2/external_adapters/all_map_adapters_v1.json"
        manifest = json.loads(source.read_text())
        validate_external_manifest(manifest)

        changed = json.loads(source.read_text())
        changed["adapters"][0]["adapter_config_path"] = str(
            ROOT / changed["adapters"][0]["adapter_config_path"]
        )
        with self.assertRaisesRegex(ValueError, "config path/hash"):
            validate_external_manifest(changed)

        changed = json.loads(source.read_text())
        changed["adapters"][0]["adapter_config_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "config path/hash"):
            validate_external_manifest(changed)

        changed = json.loads(source.read_text())
        command = changed["adapters"][0]["command"]
        command[command.index("{adapter_config}")] = (
            "{project_root}/formal_v2/configs/sigmap_controlled_v1.json"
        )
        with self.assertRaisesRegex(ValueError, "exactly one --config"):
            validate_external_manifest(changed)

        adapter = manifest["adapters"][0]
        _, resolved = _resolve_adapter_command(
            adapter,
            dataset_path=ROOT / "dataset.npz",
            output_path=ROOT / "output",
            run_root=ROOT / "run",
        )
        config_index = resolved.index("--config") + 1
        self.assertEqual(
            Path(resolved[config_index]),
            (ROOT / adapter["adapter_config_path"]).resolve(),
        )
        self.assertTrue(Path(resolved[config_index]).is_absolute())
        self.assertNotIn("{adapter_config}", resolved)

        changed = json.loads(source.read_text())
        changed["adapters"][0]["command"].extend(
            ["--config", "{adapter_config}"]
        )
        with self.assertRaisesRegex(ValueError, "exactly one --config"):
            validate_external_manifest(changed)


    def test_shipped_claim_control_manifests_bind_current_adapter_sources(self):
        adapter_root = ROOT / "formal_v2/external_adapters"
        for manifest_name in (
            "retention_control_v3.json",
            "shuffled_pair_control_v3.json",
        ):
            with self.subTest(manifest=manifest_name):
                manifest_path = adapter_root / manifest_name
                manifest = json.loads(manifest_path.read_text())
                source = manifest_path.parent / manifest["adapter_source_path"]
                source_hash = sha256_file(source)

                self.assertTrue(source.is_file())
                self.assertEqual(manifest["adapter_source_sha256"], source_hash)
                self.assertEqual(
                    manifest["implementation_revision"],
                    manifest["adapter_source_sha256"],
                )



    def test_six_equal_conditions_cannot_pass_c1(self):
        rows = self._rows()
        _validate_six_condition_rows(
            {"model_name": "honest-model"},
            rows,
            self.dataset,
            expected_unit_ids=set(self.units),
            expected_unit_contract=self.units,
            expected_condition_contract=self.contract,
        )
        assessment = _c1_model_assessment(
            {
                "evaluation": {
                    "bootstrap_resamples": 100,
                    "null_score_equivalence_margin": 0.05,
                    "null_overclassification_rate_max": 0.05,
                    "c1_active_error_minimum_m": 0.05,
                    "c1_null_error_equivalence_margin_m": 0.05,
                }
            },
            {"adapter_id": "a", "model_name": "honest-model", "c1_eligible": True},
            rows,
            self.dataset,
            expected_unit_contract=self.units,
        )
        self.assertFalse(assessment["active_effect_passed"])
        self.assertFalse(assessment["passed"])

    def test_same_map_or_action_reuse_is_rejected_by_outer_contract(self):
        rows = self._rows()
        active = next(row for row in rows if row["condition"] == "paired_active_alternative")
        active["map_sha256"] = next(row for row in rows if row["condition"] == "correct")["map_sha256"]
        with self.assertRaisesRegex(ValueError, "outer recomputation"):
            _validate_six_condition_rows(
                {"model_name": "honest-model"},
                rows,
                self.dataset,
                expected_unit_ids=set(self.units),
                expected_unit_contract=self.units,
                expected_condition_contract=self.contract,
            )





    def test_g4_excludes_generous_control_from_required_subgate(self):
        self.assertEqual(
            G4_CONCAT_CONTROL_IDS,
            ("parameter_matched_concat", "flop_matched_concat"),
        )
        self.assertEqual(REPORT_ONLY_CONTROL_IDS, ("generous_2x_concat",))

    def test_control_sign_flip_null_is_centered_on_superiority_margin(self):
        main_rows = []
        control_rows = []
        effects = (0.08, 0.12) * 6
        for index, effect in enumerate(effects):
            common = {
                "city_id": "city-a",
                "base_map_cluster_id": f"cluster-{index:02d}",
                "canonical_base_map_digest": f"cluster-{index:02d}",
                "canonical_bank_digest": f"bank-{index:02d}",
                "bank_id": f"bank-{index:02d}",
                "seed": 1,
                "budget": 8,
                "draw": 0,
            }
            main_rows.append(
                {**common, "arm": "full", "utility_neg_log_median": effect}
            )
            control_rows.append(
                {
                    **common,
                    "arm": "equal_flop_alignment",
                    "utility_neg_log_median": 0.0,
                }
            )

        clusters = np.asarray([row["base_map_cluster_id"] for row in main_rows])
        full = np.asarray(effects)
        control = np.zeros_like(full)
        zero_null = paired_sign_flip_test(clusters, full, control, seed=31)
        result = _control_superiority_interval(
            main_rows,
            control_rows,
            resamples=1000,
            seed=30,
            superiority_margin=0.1,
        )

        self.assertLess(zero_null["p_value_two_sided"], 0.05)
        self.assertGreater(result["p_value_two_sided"], 0.05)
        self.assertEqual(result["superiority_margin"], 0.1)
        self.assertFalse(
            interval_decision(result, threshold=0.1, relation="superiority")
            and result["p_value_two_sided"] < 0.05
        )

    def test_legacy_resource_manifest_without_source_and_replay_is_rejected(self):
        legacy = {
            "schema_version": "csi-pairs-v6-resource-controls-v1",
            "controls": [
                {"control_id": name, "command": ["true"]}
                for name in (
                    "equal_flop_alignment", "equal_flop_response",
                    "parameter_matched_concat", "flop_matched_concat", "generous_2x_concat",
                )
            ],
        }
        with self.assertRaisesRegex(ValueError, "schema mismatch|fields must be exact"):
            validate_resource_manifest(legacy)

    def test_claim_control_must_execute_the_hashed_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "adapter.py"
            source.write_text("raise SystemExit(0)\n", encoding="utf-8")
            digest = sha256_file(source)
            manifest = root / "manifest.json"
            write_json(
                manifest,
                {
                    "schema_version": "csi-pairs-v6-shuffled-pair-adapter-v2",
                    "command": ["true"],
                    "implementation_revision": digest,
                    "control_seed": 1,
                    "adapter_source_path": source.name,
                    "adapter_source_sha256": digest,
                },
            )
            with self.assertRaisesRegex(ValueError, "execute the authenticated"):
                run_claim_control_adapter(
                    {}, object(), manifest, root / "out", "csi-pairs-v6-shuffled-pair-adapter-v2", "results.json"
                )

    def test_claim_control_statistics_use_canonical_hierarchy(self):
        registry = {}
        shuffled_rows = []
        retention_rows = []
        pair_index = 0
        for cluster_index, cluster in enumerate(("a" * 64, "b" * 64)):
            for bank_index in range(2):
                pair_id = f"pair-{pair_index}"
                key = (7, pair_id)
                registry[key] = {
                    "seed": 7,
                    "pair_id": pair_id,
                    "base_map_cluster_id": f"raw-cluster-{cluster_index}",
                    "canonical_base_map_digest": cluster,
                    "canonical_bank_digest": f"{cluster_index + 1}{bank_index}" * 32,
                    "source_world": 0,
                    "target_world": 1,
                    "position_index": pair_index,
                }
                gains = {
                    "matched_model": 1.0,
                    "shuffled_model": 0.05,
                    **{system: 0.1 for system in SHUFFLED_SYSTEMS[2:]},
                }
                for metric in ("alignment_cgs", "response_probe"):
                    metric_gains = (
                        gains
                        if metric == "alignment_cgs"
                        else {key: gains[key] for key in ("matched_model", "shuffled_model")}
                    )
                    for system, gain in metric_gains.items():
                        for label, score in (("positive", gain), ("negative", 0.0)):
                            shuffled_rows.append(
                                {
                                    "seed": "7",
                                    "pair_id": pair_id,
                                    "metric": metric,
                                    "system": system,
                                    "pair_label": label,
                                    "score": str(score),
                                }
                            )
                for condition, cgs, response in (
                    ("correct", 1.0, 1.0),
                    ("map_swap", 0.3, 0.4),
                    ("map_removed", 0.2, 0.2),
                ):
                    retention_rows.append(
                        {
                            "seed": "7",
                            "pair_id": pair_id,
                            "condition": condition,
                            "cgs_score": str(cgs),
                            "response_score": str(response),
                        }
                    )
                pair_index += 1

        config = {
            "evaluation": {
                "bootstrap_resamples": 200,
                "familywise_alpha": 0.05,
                "shuffled_gain_fraction_max": 0.2,
                "retention_minimum_effect": 0.1,
            }
        }
        shuffled = _shuffled_assessment(config, shuffled_rows, registry)
        retention = _retention_assessment(config, retention_rows, registry)

        raw_split = {key: dict(row) for key, row in registry.items()}
        for index, row in enumerate(raw_split.values()):
            row["base_map_cluster_id"] = f"adversarial-raw-split-{index}"
        self.assertEqual(
            shuffled,
            _shuffled_assessment(config, shuffled_rows, raw_split),
        )
        self.assertEqual(
            retention,
            _retention_assessment(config, retention_rows, raw_split),
        )
        self.assertEqual(shuffled["base_map_cluster_count"], 2)
        self.assertEqual(retention["base_map_cluster_count"], 2)

        copied = {key: dict(row) for key, row in registry.items()}
        copied[(7, "copied-pair")] = dict(next(iter(registry.values())))
        with self.assertRaisesRegex(RuntimeError, "duplicates a canonical evaluation unit"):
            _shuffled_assessment(config, shuffled_rows, copied)

    def test_shuffled_permutation_registry_requires_both_complete_derangements(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "permutations.csv"
            rows = []
            for branch in ("alignment_h_map_edge", "response_action_target"):
                rows.extend(
                    {
                        "seed": 7,
                        "branch": branch,
                        "scene_id": "scene",
                        "edit_family": "family",
                        "effect_bucket": "bucket",
                        "original_pair_id": original,
                        "permuted_pair_id": permuted,
                    }
                    for original, permuted in (("a", "b"), ("b", "a"))
                )
            from formal_v2.formal_io import write_csv

            write_csv(path, rows)
            _validate_pairing_permutations(path, {7})
            rows[0]["permuted_pair_id"] = "a"
            write_csv(path, rows)
            with self.assertRaisesRegex(RuntimeError, "unshuffled"):
                _validate_pairing_permutations(path, {7})

            rows[0]["permuted_pair_id"] = "b"
            write_csv(
                path,
                [row for row in rows if row["branch"] == "alignment_h_map_edge"],
            )
            with self.assertRaisesRegex(RuntimeError, "both V6 branches"):
                _validate_pairing_permutations(path, {7})

    def test_shuffled_control_cannot_reuse_matched_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, "reused the matched"):
            _require_distinct_checkpoint_hash("a" * 64, "a" * 64)
        _require_distinct_checkpoint_hash("a" * 64, "b" * 64)

    def test_target_trained_retention_probe_checkpoint_is_rejected(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "probe.pt"
            evidence = {
                "dataset_sha256": "a" * 64,
                "config_sha256": "b" * 64,
                "fixture": False,
            }
            torch.save(
                {
                    "schema_version": "csi-pairs-v6-retention-probe-checkpoint-v1",
                    "seed": 7,
                    "kind": "compatibility",
                    **evidence,
                    "fit_role": "target",
                    "selection_role": "source_probe_selection",
                    "training_provenance_sha256": "c" * 64,
                    "full_checkpoint_sha256": "d" * 64,
                    "state_dict": {"weight": torch.ones(1)},
                },
                checkpoint,
            )
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                _validate_retention_probe_checkpoint(
                    checkpoint,
                    7,
                    "compatibility",
                    evidence,
                    "c" * 64,
                    "d" * 64,
                )

    def test_retention_probe_must_bind_its_seed_full_checkpoint(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "probe.pt"
            evidence = {
                "dataset_sha256": "a" * 64,
                "config_sha256": "b" * 64,
                "fixture": False,
            }
            torch.save(
                {
                    "schema_version": "csi-pairs-v6-retention-probe-checkpoint-v1",
                    "seed": 7,
                    "kind": "compatibility",
                    **evidence,
                    "fit_role": "source_probe_train",
                    "selection_role": "source_probe_selection",
                    "training_provenance_sha256": "c" * 64,
                    "full_checkpoint_sha256": "d" * 64,
                    "state_dict": {"weight": torch.ones(1)},
                },
                checkpoint,
            )
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                _validate_retention_probe_checkpoint(
                    checkpoint,
                    7,
                    "compatibility",
                    evidence,
                    "c" * 64,
                    "e" * 64,
                )

    def test_resource_control_commands_must_execute_bound_source_and_spec(self):
        manifest = {
            "schema_version": "csi-pairs-v6-resource-controls-v2",
            "controls": [
                {
                    "control_id": control_id,
                    "command": ["true"],
                    "replay_command": ["true"],
                    "adapter_source_path": "adapter.py",
                    "adapter_source_sha256": "a" * 64,
                    "architecture_spec_path": "spec.json",
                    "architecture_spec_sha256": "b" * 64,
                }
                for control_id in CONTROL_IDS
            ],
        }
        with self.assertRaisesRegex(ValueError, "execute the authenticated"):
            validate_resource_manifest(manifest)

    def test_shipped_v3_resource_manifest_binds_real_sources_and_specs(self):
        path = ROOT / "formal_v2/external_adapters/resource_controls_v3.json"
        validate_resource_manifest(json.loads(path.read_text()), path.parent)

    def test_v3_resource_index_rejects_incomplete_seed_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "resource_index.json"
            write_json(
                index,
                {
                    "schema_version": "csi-pairs-v6-resource-index-v3",
                    "control_id": "equal_flop_alignment",
                    "dataset_sha256": "a" * 64,
                    "config_sha256": "b" * 64,
                    "fixture": False,
                    "architecture_spec_sha256": "c" * 64,
                    "adapter_source_sha256": "d" * 64,
                    "records": [],
                },
            )
            with self.assertRaisesRegex(RuntimeError, "cover every seed"):
                _validate_resource_index(
                    "equal_flop_alignment",
                    index,
                    {
                        "dataset_sha256": "a" * 64,
                        "config_sha256": "b" * 64,
                        "fixture": False,
                    },
                    root,
                    {"seeds": [7, 11]},
                    architecture={},
                    architecture_sha256="c" * 64,
                    adapter_source_sha256="d" * 64,
                )

    def test_concat_resource_profiler_cannot_omit_bottleneck_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profiler.json"
            events = [
                {
                    "name": f"{role}_{suffix}",
                    "phase": phase,
                    "count": 1,
                    "flops": 1.0,
                }
                for role in ("alignment", "response")
                for suffix, phase in (
                    ("training_graph", "training"),
                    ("retained_inference_graph", "inference"),
                )
            ]
            write_json(
                path,
                {
                    "schema_version": "csi-pairs-v6-profiler-summary-v3",
                    "profiler": "torch.utils.flop_counter.FlopCounterMode",
                    "resource_accounting_scope": (
                        "representation_training_and_retained_inference_"
                        "excluding_common_localization_head"
                    ),
                    "events": events,
                    "measured_training_flops": 2.0,
                    "measured_inference_flops": 2.0,
                    "profiled_step_count": len(events),
                },
            )
            with self.assertRaisesRegex(RuntimeError, "event coverage"):
                _validate_v3_profiler("parameter_matched_concat", path)

    def test_resource_parameter_scope_excludes_common_head_but_keeps_bottleneck(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")
        payload = {
            "components": [
                {"state_dict": {"weight": torch.ones(2, 3), "bias": torch.ones(2)}}
            ],
            "bottleneck_state_dict": {
                "weight": torch.ones(2, 4),
                "bias": torch.ones(2),
            },
            "source_head_state_dict": {"large_common_head": torch.ones(100)},
        }
        self.assertEqual(_payload_resource_parameter_count(payload), 18)

    def test_single_tensor_checkpoint_cannot_authenticate_as_control_architecture(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            torch.save(
                {
                    "schema_version": "csi-pairs-v6-control-checkpoint-v2",
                    "control_id": "equal_flop_alignment",
                    "architecture_family": "single_branch_alignment",
                    "architecture_spec_sha256": "c" * 64,
                    "adapter_source_sha256": "d" * 64,
                    "dataset_sha256": "a" * 64,
                    "config_sha256": "b" * 64,
                    "fixture": False,
                    "state_dict": {"totally_wrong.single_tensor": torch.zeros(3)},
                },
                checkpoint,
            )
            profiler = root / "profiler.json"
            training = root / "training.json"
            write_json(profiler, {})
            write_json(training, {})
            resource = {
                "schema_version": "csi-pairs-v6-resource-record-v2",
                "control_id": "equal_flop_alignment",
                "dataset_sha256": "a" * 64,
                "config_sha256": "b" * 64,
                "fixture": False,
                "training_flops": 1.0,
                "inference_flops": 1.0,
                "parameters": 3,
                "wall_seconds": 1.0,
                "checkpoint_path": checkpoint.name,
                "checkpoint_sha256": sha256_file(checkpoint),
                "profiler_trace_path": profiler.name,
                "profiler_trace_sha256": sha256_file(profiler),
                "training_log_path": training.name,
                "training_log_sha256": sha256_file(training),
                "architecture_spec_sha256": "c" * 64,
                "adapter_source_sha256": "d" * 64,
            }
            architecture = {
                "architecture_family": "single_branch_alignment",
                "state_keys": ["encoder.weight", "encoder.bias", "head.weight", "head.bias"],
                "objective_terms": ["endpoint", "alignment"],
            }
            with self.assertRaisesRegex(RuntimeError, "architecture/state-key"):
                _validate_resource(
                    "equal_flop_alignment",
                    resource,
                    {"dataset_sha256": "a" * 64, "config_sha256": "b" * 64, "fixture": False},
                    root,
                    architecture=architecture,
                    architecture_sha256="c" * 64,
                    adapter_source_sha256="d" * 64,
                )


if __name__ == "__main__":
    unittest.main()
