from __future__ import annotations

import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from formal_v2.formal_evaluation import (
    _alignment_shortcut_rows,
    _evaluation_gate,
    _g3_primary_scope_intervals,
    _gray_cells_complete,
    _native_probe_correlation_macro,
    _periodic_power_spread,
    _protocol_transition_skill,
    _response_effect_rows,
    _shortcut_metadata_features,
    _transition_metrics,
)
from formal_v2.formal_factorial import (
    _canonical_bank_digest,
    _city_budget_arm_interval,
)
from formal_v2.formal_io import artifact_manifest, write_json
from formal_v2.formal_metrics import risk_coverage
from formal_v2.formal_path import (
    _cluster_slope_interval,
    _effect_row_identity,
    _exact_path_match,
    _mechanism_gate,
    _require_manifested_stage_artifact,
)
from formal_v2.formal_protocol import PatchSpec, headline_alignment_edge, patchify_csi
from formal_v2.formal_risk import (
    FirstPartyRiskFeatures,
    _bank_macro_risk_coverage,
    _bank_cluster_probability_metrics,
    _calibration_confidence_intervals,
    _cluster_ranking_random_contrast,
    _expected_random_rejection_aurc,
    _freeze_first_party_risk_features,
    _hierarchical_sample_weights,
    _make_first_party_risk_features,
    _simultaneous_hierarchical_family,
    _synchronized_hierarchical_bootstrap,
    _validate_first_party_risk_features,
    fit_constrained_risk_calibrator,
    fit_scalar_risk_calibrator,
    fit_temperature_no_intercept,
    frozen_map_proposals,
    generating_map_pair_margin,
    load_risk_feature_archive,
    risk_metrics,
    run_risk_contract,
    used_map_score_margin,
)


class RiskPathEvaluationIntegrityTests(unittest.TestCase):
    def test_angular_spread_wraps_the_fft_angle_boundary(self):
        power = np.asarray([1.0, 0.0, 0.0, 1.0])
        axis = np.linspace(-1.0, 1.0, power.size, endpoint=False)
        linear_mean = float(np.sum(power * axis) / np.sum(power))
        linear_spread = float(
            np.sqrt(np.sum(power * (axis - linear_mean) ** 2) / np.sum(power))
        )
        periodic_spread = _periodic_power_spread(power, axis, period=2.0)
        self.assertAlmostEqual(linear_spread, 0.75)
        self.assertAlmostEqual(periodic_spread, 0.25)

    def test_periodic_power_spread_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            _periodic_power_spread(
                np.asarray([1.0, -1.0]),
                np.asarray([-1.0, 0.0]),
                period=2.0,
            )

    def test_g3_scope_tests_use_registered_superiority_margins(self):
        cgs_rows = []
        response_rows = []
        effects = (0.08, 0.12) * 6
        for index, effect in enumerate(effects):
            common = {
                "seed": 1,
                "base_map_cluster_id": f"cluster-{index}",
                "canonical_base_map_digest": f"cluster-{index}",
                "bank_id": f"bank-{index}",
                "canonical_bank_digest": f"bank-{index}",
                "evaluation_scope": "source_final_unseen_bank",
            }
            cgs_rows.extend(
                (
                    {**common, "arm": "endpoint", "cgs_auroc": 0.0},
                    {**common, "arm": "alignment", "cgs_auroc": effect},
                )
            )
            response_rows.extend(
                (
                    {
                        **common,
                        "arm": "endpoint",
                        "native_target_free_full_channel_nmse": effect,
                    },
                    {
                        **common,
                        "arm": "response",
                        "native_target_free_full_channel_nmse": 0.0,
                    },
                )
            )
        zero = _g3_primary_scope_intervals(
            cgs_rows, response_rows, ["source_final_unseen_bank"], 100
        )["source_final_unseen_bank"]
        centered = _g3_primary_scope_intervals(
            cgs_rows,
            response_rows,
            ["source_final_unseen_bank"],
            100,
            alignment_superiority_margin=0.1,
            response_superiority_margin=0.1,
        )["source_final_unseen_bank"]
        self.assertLess(zero["alignment_superiority"]["p_value_two_sided"], 0.05)
        self.assertLess(zero["response_superiority"]["p_value_two_sided"], 0.05)
        self.assertGreater(
            centered["alignment_superiority"]["p_value_two_sided"], 0.05
        )
        self.assertGreater(
            centered["response_superiority"]["p_value_two_sided"], 0.05
        )
        self.assertEqual(centered["alignment_superiority"]["null_difference"], 0.1)
        self.assertEqual(centered["response_superiority"]["null_difference"], 0.1)

    def test_g5_city_test_uses_registered_improvement_margin(self):
        rows = []
        effects = (0.08, 0.12) * 6
        for index, effect in enumerate(effects):
            common = {
                "city_id": "target-a",
                "base_map_cluster_id": f"cluster-{index}",
                "canonical_base_map_digest": f"cluster-{index}",
                "bank_id": f"bank-{index}",
                "canonical_bank_digest": f"bank-{index}",
                "seed": 1,
                "budget": 8,
                "draw": 0,
            }
            rows.extend(
                (
                    {**common, "arm": "endpoint", "utility_neg_log_median": 0.0},
                    {**common, "arm": "full", "utility_neg_log_median": effect},
                )
            )
        zero = _city_budget_arm_interval(
            rows, "target-a", 8, "endpoint", 100, 1
        )
        centered = _city_budget_arm_interval(
            rows,
            "target-a",
            8,
            "endpoint",
            100,
            1,
            null_threshold=0.1,
        )
        self.assertLess(zero["p_value_two_sided"], 0.05)
        self.assertGreater(centered["p_value_two_sided"], 0.05)
        self.assertEqual(centered["null_difference"], 0.1)

    def test_g3_scope_intervals_cannot_hide_target_regression(self):
        cgs_rows = []
        response_rows = []
        for scope, alignment_value, response_value in (
            ("source_final_unseen_bank", 0.9, 0.1),
            ("target:target-a", 0.4, 0.9),
        ):
            for index in range(6):
                common = {
                    "seed": 1,
                    "base_map_cluster_id": f"{scope}-cluster-{index}",
                    "canonical_base_map_digest": f"{scope}-cluster-{index}",
                    "bank_id": f"{scope}-bank-{index}",
                    "canonical_bank_digest": f"{scope}-bank-{index}",
                    "evaluation_scope": scope,
                }
                cgs_rows.extend(
                    (
                        {**common, "arm": "endpoint", "cgs_auroc": 0.5},
                        {**common, "arm": "alignment", "cgs_auroc": alignment_value},
                    )
                )
                response_rows.extend(
                    (
                        {
                            **common,
                            "arm": "endpoint",
                            "native_target_free_full_channel_nmse": 0.5,
                        },
                        {
                            **common,
                            "arm": "response",
                            "native_target_free_full_channel_nmse": response_value,
                        },
                    )
                )
        intervals = _g3_primary_scope_intervals(
            cgs_rows,
            response_rows,
            ["source_final_unseen_bank", "target:target-a"],
            100,
        )
        self.assertGreater(
            intervals["source_final_unseen_bank"]["alignment_superiority"][
                "paired_mean_difference"
            ],
            0,
        )
        self.assertLess(
            intervals["target:target-a"]["alignment_superiority"][
                "paired_mean_difference"
            ],
            0,
        )
        self.assertLess(
            intervals["target:target-a"]["response_superiority"][
                "paired_mean_difference"
            ],
            0,
        )

    def test_g3_gate_cannot_pool_away_target_response_control_failure(self):
        strong = {
            "assessed": True,
            "cluster_count": 4,
            "paired_mean_difference": 0.4,
            "ci95_low": 0.2,
            "ci95_high": 0.6,
            "confidence_level": 0.95,
            "p_value_two_sided": 1e-12,
        }
        failed = {
            **strong,
            "paired_mean_difference": -0.1,
            "ci95_low": -0.2,
            "ci95_high": 0.0,
        }
        scoped = {
            scope: {
                "alignment_superiority": dict(strong),
                "response_superiority": dict(strong),
                "response_vs_copy": dict(
                    failed if scope == "target:target-a" else strong
                ),
                "response_vs_no_action": dict(strong),
                "response_vs_action_swap": dict(strong),
                "response_direction": dict(strong),
            }
            for scope in ("source_final_unseen_bank", "target:target-a")
        }
        dataset = SimpleNamespace(
            city_ids=np.asarray(["source-a", "target-a"]),
            scene_roles=np.asarray(["source_final_unseen_bank", "target"]),
            bank_ids=np.asarray(["source-bank", "target-bank"]),
            is_fixture=False,
            indices_for_role=lambda role: np.asarray(
                [0] if role == "source_final_unseen_bank" else [1]
            ),
        )
        config = {
            "seeds": [1],
            "factorial": {
                "arms": ["endpoint", "alignment", "response", "full"]
            },
            "qualification": {
                "minimum_geometry_matched_wrong_action_fraction": 0.5
            },
            "evaluation": {
                "bootstrap_resamples": 100,
                "familywise_alpha": 0.05,
                "minimum_alignment_superiority": 0.01,
                "minimum_response_superiority": 0.01,
                "minimum_native_probe_correlation": 0.0,
                "null_score_equivalence_margin": 0.01,
                "response_null_violation_rate_max": 0.05,
                "response_null_equivalence_margin": 0.01,
                "minimum_cgs_noninferiority": -0.01,
                "minimum_response_noninferiority": -0.01,
            },
        }
        response_row = {
            "arm": "response",
            "native_null_violation_rate": 0.0,
            "native_latent_null_violation_rate": 0.0,
            "native_null_delta_rms_mean": 0.0,
            "native_latent_null_delta_rms_mean": 0.0,
            "native_action_swap_exact_count": 1,
            "native_action_swap_exact_fraction": 1.0,
            "probe_action_swap_exact_count": 1,
            "probe_action_swap_exact_fraction": 1.0,
            "probe_oracle_x_active_patch_nmse": 0.0,
            "native_delta_relative_magnitude_error": 0.0,
            "native_sgcs": 1.0,
            "native_transition_skill": 1.0,
            "native_path_loss_change_mae": 0.0,
            "native_delay_spread_change_mae": 0.0,
            "native_angular_spread_change_mae": 0.0,
            "native_path_loss_direction_accuracy": 1.0,
            "native_delay_spread_direction_accuracy": 1.0,
            "native_angular_spread_direction_accuracy": 1.0,
        }
        response_rows = [response_row, {**response_row, "arm": "full"}]
        contracts = {
            "scene_id_only": "stable_hashed_bank_token_no_label_feature",
            "edit_status_xor": "separate_world_bits_and_natural_flags_no_xor",
            "variant_id_matcher": "separate_stable_hashed_variant_tokens_no_match_or_unk_override",
        }
        shortcut_rows = []
        for arm, bank in itertools.product(config["factorial"]["arms"], dataset.bank_ids):
            for baseline in (
                "constant",
                "csi_only",
                "map_only",
                "scene_id_only",
                "edit_status_xor",
                "variant_id_matcher",
            ):
                row = {
                    "seed": 1,
                    "arm": arm,
                    "canonical_bank_digest": bank,
                    "baseline": baseline,
                    "unseen_bank_auroc": 0.5,
                }
                if baseline in contracts:
                    row["identity_token_contract"] = contracts[baseline]
                shortcut_rows.append(row)
        null_safety = {
            arm: {"passed": True} for arm in config["factorial"]["arms"]
        }
        with (
            patch(
                "formal_v2.formal_evaluation._null_safety_by_arm",
                return_value=null_safety,
            ),
            patch(
                "formal_v2.formal_evaluation._paired_arm_comparison",
                return_value=dict(strong),
            ),
            patch(
                "formal_v2.formal_evaluation._within_arm_advantage_interval",
                return_value=dict(strong),
            ),
            patch(
                "formal_v2.formal_evaluation._within_arm_level_interval",
                return_value=dict(strong),
            ),
            patch(
                "formal_v2.formal_evaluation._g3_primary_scope_intervals",
                return_value=scoped,
            ),
            patch(
                "formal_v2.formal_evaluation._effect_bins_complete",
                return_value=True,
            ),
            patch(
                "formal_v2.formal_evaluation._gray_cells_complete",
                return_value=True,
            ),
            patch(
                "formal_v2.formal_evaluation._native_probe_correlation_macro",
                return_value=1.0,
            ),
            patch(
                "formal_v2.formal_evaluation._canonical_bank_digest",
                side_effect=lambda data, scene: str(data.bank_ids[scene]),
            ),
        ):
            gate = _evaluation_gate(
                config,
                dataset,
                [],
                [],
                [],
                response_rows,
                shortcut_rows,
                {"g4_subgates": {"complete": "PASS"}, "gate_vector": {"G5": "PASS"}},
                {},
                qualification_gate_sha256="a" * 64,
                factorial_gate_sha256="b" * 64,
            )
        self.assertFalse(gate["passed"])
        self.assertEqual(
            gate["g3_subgates"]["9_unpooled_source_and_target_primary_metrics"],
            "FAIL",
        )

    def test_gray_completeness_requires_every_unique_canonical_cell(self):
        expected = {
            (seed, arm, bank)
            for seed in (1, 2)
            for arm in ("endpoint", "alignment", "response", "full")
            for bank in ("bank-a", "bank-b")
        }
        rows = [
            {
                "seed": seed,
                "arm": arm,
                "canonical_bank_digest": bank,
                "route": "gray",
                "condition_status": "ASSESSED",
            }
            for seed, arm, bank in sorted(expected)
        ]
        self.assertTrue(_gray_cells_complete(rows, expected))
        self.assertFalse(_gray_cells_complete(rows[:-1], expected))
        self.assertFalse(_gray_cells_complete(rows + [dict(rows[0])], expected))

    def test_native_probe_correlation_uses_seed_bank_foundation_macro(self):
        rows = []
        values = {
            ("foundation-a", "bank-a", 1): 0.1,
            ("foundation-a", "bank-a", 2): 0.3,
            ("foundation-a", "bank-b", 1): 0.7,
            ("foundation-a", "bank-b", 2): 0.9,
            ("foundation-b", "bank-c", 1): 0.5,
            ("foundation-b", "bank-c", 2): 0.7,
        }
        for (foundation, bank, seed), value in values.items():
            rows.append(
                {
                    "seed": seed,
                    "arm": "alignment",
                    "bank_id": f"raw-{bank}",
                    "canonical_base_map_digest": foundation,
                    "canonical_bank_digest": bank,
                    "native_probe_spearman": value,
                }
            )
        expected = {
            (seed, "alignment", bank)
            for _, bank, seed in values
        }
        baseline = _native_probe_correlation_macro(rows, "alignment", expected)
        self.assertAlmostEqual(baseline, 0.55)
        duplicated = rows + [
            {**rows[0], "bank_id": "adversarial-renamed-copy"}
        ]
        self.assertEqual(
            _native_probe_correlation_macro(duplicated, "alignment", expected),
            baseline,
        )
        conflicting = rows + [
            {
                **rows[0],
                "bank_id": "adversarial-renamed-copy",
                "native_probe_spearman": 0.99,
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "conflicting duplicate canonical"):
            _native_probe_correlation_macro(conflicting, "alignment", expected)

    def test_shortcut_metadata_has_no_match_label_or_forced_unk_feature(self):
        dataset = SimpleNamespace(
            bank_ids=np.asarray(["bank-a", "bank-b"]),
            world_bits=np.asarray(
                [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64
            ),
            natural_world_index=np.asarray([3, 3]),
        )
        first = _shortcut_metadata_features(dataset, 0, 0, 0)
        alternative = _shortcut_metadata_features(dataset, 0, 0, 1)
        unseen_bank = _shortcut_metadata_features(dataset, 1, 0, 0)
        self.assertTrue(all(values.size > 1 for values in first))
        self.assertFalse(np.array_equal(first[1], alternative[1]))
        self.assertFalse(np.array_equal(first[2], alternative[2]))
        self.assertFalse(np.array_equal(first[0], unseen_bank[0]))
        self.assertFalse(np.all(unseen_bank[2] == 1.0))

    def test_shortcut_rows_preserve_their_exact_evaluation_scope(self):
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        features = np.arange(4, dtype=np.float64)[:, None]
        fields = (
            "constant_shortcut_features",
            "csi_only_shortcut_features",
            "map_only_shortcut_features",
            "scene_id_only_shortcut_features",
            "edit_status_xor_shortcut_features",
            "variant_id_match_shortcut_features",
        )
        source = {"labels": labels, **{field: features for field in fields}}
        evaluated = {
            "labels": labels,
            "base_map_cluster_ids": np.asarray(["cluster"] * 4),
            "canonical_base_map_digests": np.asarray(["a" * 64] * 4),
            "canonical_bank_digests": np.asarray(["b" * 64] * 4),
            "city_ids": np.asarray(["city-b"] * 4),
            **{field: features for field in fields},
        }
        with (
            patch(
                "formal_v2.formal_evaluation.fit_select_compatibility_probe",
                return_value=(object(), {"selected_family": "linear"}),
            ),
            patch(
                "formal_v2.formal_evaluation.predict_binary_probe",
                side_effect=lambda _probe, values: np.asarray(values)[:, 0],
            ),
        ):
            rows = _alignment_shortcut_rows(
                1,
                "full",
                "bank-b",
                "target:city-b",
                source,
                source,
                evaluated,
                np.ones(4, dtype=np.bool_),
                {},
            )
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {row["evaluation_scope"] for row in rows}, {"target:city-b"}
        )

    def test_risk_coverage_is_invariant_to_tied_score_row_order(self):
        errors = np.asarray([9.0, 1.0, 7.0, 2.0, 5.0, 3.0, 8.0, 4.0])
        scores = np.zeros_like(errors)
        first = risk_coverage(errors, scores)
        second = risk_coverage(errors[::-1], scores)
        self.assertEqual(first, second)

    def test_external_risk_archive_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "first-party"):
                load_risk_feature_archive(
                    Path(directory) / "fabricated.npz", {}, object(), directory
                )

    def test_risk_contract_rejects_unmarked_self_reported_features(self):
        with self.assertRaisesRegex(TypeError, "first-party"):
            FirstPartyRiskFeatures({"full": {"x": np.asarray([1.0])}})
        with self.assertRaises(TypeError):
            run_risk_contract({}, object(), ".", {})

    def test_first_party_risk_payload_is_deeply_immutable(self):
        source = np.asarray([1.0, 2.0])
        raw = {"full": {"score": source}}
        features = _freeze_first_party_risk_features(
            raw,
            mixture_freeze={"counts": [1, 2]},
            replay_binding={"payload_sha256": "0" * 64},
        )
        source[0] = 99.0
        raw["full"]["score"][1] = 88.0
        np.testing.assert_array_equal(features["full"]["score"], [1.0, 2.0])
        with self.assertRaises(TypeError):
            features["full"]["score"] = np.asarray([3.0])
        with self.assertRaises(ValueError):
            features["full"]["score"][0] = 3.0
        with self.assertRaises(ValueError):
            features["full"]["score"].setflags(write=True)
        with self.assertRaises(TypeError):
            features.replay_binding = {"payload_sha256": "1" * 64}
        with self.assertRaises(TypeError):
            del features.replay_binding

    def test_first_party_replay_binding_detects_checkpoint_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "dataset.bin"
            source.write_bytes(b"dataset")
            checkpoint_index = root / "factorial" / "checkpoint_index.json"
            checkpoint_index.parent.mkdir(parents=True)
            checkpoint_index.write_text('{"version":1}', encoding="ascii")
            dataset = SimpleNamespace(source_path=source, is_fixture=True)
            config = {"artifact_label": "test-risk-binding"}
            features = _make_first_party_risk_features(
                {"full": {"score": np.asarray([1.0, 2.0])}},
                config,
                dataset,
                root,
            )
            _validate_first_party_risk_features(features, config, dataset, root)
            checkpoint_index.write_text('{"version":2}', encoding="ascii")
            with self.assertRaisesRegex(RuntimeError, "checkpoint/dataset/config"):
                _validate_first_party_risk_features(features, config, dataset, root)

    def test_compatibility_margins_use_higher_is_better_sign(self):
        self.assertEqual(used_map_score_margin(5.0, np.asarray([2.0, 4.0])), 1.0)
        self.assertEqual(generating_map_pair_margin(5.0, 2.0), 3.0)

    def test_risk_ranking_ignores_observations_outside_common_support(self):
        labels = np.asarray([0, 1, 1, 0, 1, 0])
        probabilities = np.asarray([0.1, 0.8, 0.2, 0.2, 0.9, 0.7])
        errors = np.asarray([1.0, 4.0, 999.0, 2.0, 5.0, 888.0])
        support = np.asarray([True, True, False, True, True, False])
        clusters = np.asarray(["a", "a", "a", "b", "b", "b"])
        first = risk_metrics(labels, probabilities, errors, support, clusters)
        probabilities[~support] = np.asarray([1.0, 0.0])
        errors[~support] = np.asarray([1e9, 1e9])
        second = risk_metrics(labels, probabilities, errors, support, clusters)
        self.assertEqual(first, second)

    def test_path_gate_cannot_pass_without_provenance(self):
        config = {
            "path": {
                "minimum_trend_slope": 0.001,
                "equivalence_margin": 0.02,
                "balance_smd_max": 0.1,
                "bootstrap_resamples": 20,
                "minimum_effective_sample_size": 2,
                "covariate_overlap_minimum": 0.5,
            }
        }
        result = _mechanism_gate(
            config,
            [],
            [],
            [],
            [],
            [],
            [],
            {"passed": False, "registry_sha256": "0" * 64},
        )
        self.assertEqual(
            result["g7_subgates"]["1_registered_path_provenance_and_noop_epsilon"],
            "FAIL",
        )
        self.assertFalse(result["passed"])

    def test_path_gate_writes_fail_state_when_exact_swap_is_unavailable(self):
        config = {
            "qualification": {
                "minimum_geometry_matched_wrong_action_fraction": 0.5
            },
            "path": {
                "minimum_trend_slope": 0.001,
                "equivalence_margin": 0.02,
                "balance_smd_max": 0.1,
                "bootstrap_resamples": 20,
                "minimum_effective_sample_size": 2,
                "covariate_overlap_minimum": 0.5,
            },
        }
        compatibility = []
        response = []
        for cluster in ("a", "b"):
            for index, (bin_name, a_path) in enumerate(
                (("zero", 0.0), ("low", 0.2), ("medium", 0.6), ("high", 1.0))
            ):
                common = {
                    "seed": 1,
                    "arm": "full",
                    "route": "active",
                    "bank_id": f"bank-{cluster}",
                    "base_map_cluster_id": cluster,
                    "canonical_base_map_digest": cluster,
                    "canonical_bank_digest": f"bank-{cluster}",
                    "a_path": a_path,
                    "a_path_bin": bin_name,
                }
                compatibility.append(
                    {**common, "matched_minus_alternative": 0.01 * index}
                )
                response.append(
                    {
                        **common,
                        "response_advantage": 0.01 * index,
                        "response_advantage_vs_no_action": 0.01 * index,
                        "wrong_action_match_status": "fallback",
                    }
                )
        result = _mechanism_gate(
            config,
            compatibility,
            response,
            [],
            compatibility,
            response,
            [],
            {"passed": True, "registry_sha256": "0" * 64},
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["response_vs_action_swap_trend_slope"]["assessed"])

    def test_path_rejects_csv_changed_after_stage_manifest(self):
        evidence = {
            "artifact_label": "test",
            "dataset_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "fixture": False,
            "scientific_use": "CANDIDATE_NOT_CLAIM",
        }
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            artifact = stage / "compatibility_pair_effects.csv"
            artifact.write_text("value\n1\n", encoding="utf-8")
            write_json(
                stage / "manifest.json",
                {
                    "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
                    **evidence,
                    "files": artifact_manifest(stage, evidence=evidence),
                },
            )
            _require_manifested_stage_artifact(artifact, evidence)
            artifact.write_text("value\n999\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "sha256 mismatch"):
                _require_manifested_stage_artifact(artifact, evidence)

    def test_path_recomputes_effect_row_identity_from_dataset(self):
        dataset = SimpleNamespace(
            bank_ids=np.asarray(["bank"]),
            city_ids=np.asarray(["city"]),
            base_map_cluster_ids=np.asarray(["cluster"]),
            scene_roles=np.asarray(["target"]),
            position_roles=np.asarray([["query"]]),
            position_ids=np.asarray([["position"]]),
            positions=np.zeros((1, 1, 2), dtype=np.float64),
            world_bits=np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]]),
            maps=np.zeros((1, 4, 3, 2, 2), dtype=np.float64),
            csi=np.zeros((1, 4, 1, 8), dtype=np.float64),
            radio_config=np.zeros((1, 2), dtype=np.float64),
            bs_pose=np.zeros((1, 7), dtype=np.float64),
            map_channel_names=np.asarray(["occupancy", "height", "material"]),
            canonical_base_map_digest=lambda _scene: "foundation",
        )
        canonical_bank = _canonical_bank_digest(dataset, 0)
        row = {
            "seed": "1",
            "arm": "full",
            "pair_id": "bank:0:1:0:0",
            "scene_index": "0",
            "bank_id": "bank",
            "base_map_cluster_id": "cluster",
            "canonical_base_map_digest": "foundation",
            "canonical_bank_digest": canonical_bank,
            "city_id": "city",
            "source_world": "0",
            "target_world": "1",
            "position_index": "0",
            "route": "active",
            "matched_minus_alternative": "0.1",
        }
        identity = _effect_row_identity(
            row, dataset, Path("effects.csv"), "matched_minus_alternative", {}
        )
        self.assertEqual(identity["canonical_bank_digest"], canonical_bank)
        for field, value in (
            ("city_id", "other-city"),
            ("pair_id", "bank:0:1:0:1"),
        ):
            with self.assertRaises(RuntimeError):
                _effect_row_identity(
                    {**row, field: value},
                    dataset,
                    Path("effects.csv"),
                    "matched_minus_alternative",
                    {},
                )

    def test_nonheadline_rows_cannot_inflate_headline_path_overlap(self):
        config = {
            "qualification": {
                "minimum_geometry_matched_wrong_action_fraction": 0.5
            },
            "path": {
                "minimum_trend_slope": 0.001,
                "equivalence_margin": 0.02,
                "balance_smd_max": 0.1,
                "bootstrap_resamples": 40,
                "minimum_effective_sample_size": 2,
                "covariate_overlap_minimum": 0.5,
            },
        }
        matched_compatibility = []
        matched_response = []
        for cluster in ("a", "b"):
            for index, (bin_name, a_path) in enumerate(
                (("zero", 0.0), ("low", 0.2), ("medium", 0.6), ("high", 1.0))
            ):
                common = {
                    "seed": 1,
                    "arm": "full",
                    "route": "active",
                    "bank_id": f"bank-{cluster}",
                    "base_map_cluster_id": cluster,
                    "canonical_base_map_digest": cluster,
                    "canonical_bank_digest": f"bank-{cluster}",
                    "a_path": a_path,
                    "a_path_bin": bin_name,
                }
                value = 0.02 * index
                matched_compatibility.append(
                    {**common, "matched_minus_alternative": value}
                )
                matched_response.append(
                    {
                        **common,
                        "response_advantage": value,
                        "response_advantage_vs_action_swap": value,
                        "response_advantage_vs_no_action": value,
                        "wrong_action_match_status": "exact",
                    }
                )
        nonheadline = [
            {"arm": "endpoint", "route": "null"} for _ in range(200)
        ]
        compatibility = matched_compatibility + nonheadline
        response = matched_response + nonheadline
        original_compatibility = matched_compatibility * 10 + nonheadline
        original_response = matched_response * 10 + nonheadline
        self.assertGreater(
            len(compatibility) / len(original_compatibility),
            config["path"]["covariate_overlap_minimum"],
        )
        result = _mechanism_gate(
            config,
            compatibility,
            response,
            [{"standardized_mean_difference": 0.0}],
            original_compatibility,
            original_response,
            [],
            {"passed": True, "registry_sha256": "0" * 64},
        )
        self.assertAlmostEqual(result["compatibility_matched_overlap_fraction"], 0.1)
        self.assertAlmostEqual(result["exact_response_matched_overlap_fraction"], 0.1)
        self.assertEqual(
            result["g7_subgates"]["2_covariate_matching_balance_overlap_ess"],
            "FAIL",
        )

    def test_path_slope_is_invariant_to_duplicate_rows_from_one_seed(self):
        rows = []
        for cluster in ("a", "b"):
            for seed, slope in ((1, 0.1), (2, 0.3)):
                for a_path in (0.0, 0.2, 0.6, 1.0):
                    rows.append(
                        {
                            "seed": seed,
                            "bank_id": f"bank-{cluster}",
                            "base_map_cluster_id": cluster,
                            "canonical_base_map_digest": cluster,
                            "canonical_bank_digest": f"bank-{cluster}",
                            "a_path": a_path,
                            "effect": slope * a_path,
                        }
                    )
        original = _cluster_slope_interval(rows, "effect", 100, 901)
        duplicated = _cluster_slope_interval(
            rows + [dict(row) for row in rows if row["seed"] == 1],
            "effect",
            100,
            901,
        )
        self.assertAlmostEqual(original["estimate"], 0.2)
        for key in ("estimate", "ci95_low", "ci95_high"):
            self.assertAlmostEqual(original[key], duplicated[key], places=12)
        self.assertEqual(original["training_seed_count"], 2)
        self.assertIn("paired_training_seed_across_all_banks", original["resampling_layers"])

    def test_path_matching_never_borrows_a_bin_from_another_seed(self):
        rows = []
        for index, (bin_name, a_path) in enumerate(
            (("zero", 0.0), ("low", 0.2), ("medium", 0.6), ("high", 1.0))
        ):
            rows.append(
                {
                    "seed": 1,
                    "arm": "full",
                    "route": "active",
                    "bank_id": "bank",
                    "bit_index": 0,
                    "edit_family": "occupancy_add",
                    "a_path_bin": bin_name,
                    "a_path": a_path,
                    "pair_id": f"seed-1-{index}",
                    "delta_map": 0.1,
                    "bs_ue_distance_m": 1.0,
                    "ue_edit_distance_m": 1.0,
                    "los_indicator": 1.0,
                }
            )
        rows.append(
            {
                **rows[0],
                "seed": 2,
                "pair_id": "a-seed-2-zero-sorts-first-without-seed-stratum",
            }
        )
        matched = _exact_path_match(rows)
        self.assertEqual(len(matched), 4)
        self.assertEqual({row["seed"] for row in matched}, {1})

    def test_risk_curve_is_invariant_to_duplicate_canonical_bank(self):
        errors = np.asarray([1.0, 4.0, 2.0, 5.0])
        scores = np.asarray([0.1, 0.9, 0.2, 0.8])
        clusters = np.asarray(["a", "a", "b", "b"])
        banks = np.asarray(["bank-a", "bank-a", "bank-b", "bank-b"])
        units = np.asarray(["a0", "a1", "b0", "b1"])
        original = _bank_macro_risk_coverage(
            errors, scores, clusters, banks, units
        )
        duplicated = _bank_macro_risk_coverage(
            np.concatenate((errors, errors[:2])),
            np.concatenate((scores, scores[:2])),
            np.concatenate((clusters, clusters[:2])),
            np.concatenate((banks, banks[:2])),
            np.concatenate((units, units[:2])),
        )
        self.assertEqual(original, duplicated)

    def test_hierarchical_weights_equalize_seed_foundation_and_bank(self):
        seeds = np.asarray([0] * 6 + [1] * 7)
        clusters = np.asarray(["f0"] * 4 + ["f1"] * 2 + ["f0"] * 3 + ["f1"] * 4)
        banks = np.asarray(["a"] * 3 + ["b"] + ["c"] * 2 + ["a"] + ["b"] * 2 + ["c"] * 4)
        weights = _hierarchical_sample_weights(seeds, clusters, banks)
        self.assertAlmostEqual(float(np.sum(weights[seeds == 0])), 0.5)
        self.assertAlmostEqual(float(np.sum(weights[seeds == 1])), 0.5)
        self.assertAlmostEqual(
            float(np.sum(weights[clusters == "f0"])), 0.5
        )
        self.assertAlmostEqual(
            float(np.sum(weights[clusters == "f1"])), 0.5
        )
        self.assertAlmostEqual(float(np.sum(weights[banks == "a"])), 0.25)
        self.assertAlmostEqual(float(np.sum(weights[banks == "b"])), 0.25)
        self.assertAlmostEqual(
            float(np.sum(weights[(banks == "a") & (seeds == 0)])), 0.125
        )
        self.assertAlmostEqual(
            float(np.sum(weights[(banks == "a") & (seeds == 1)])), 0.125
        )

    def test_weighted_fit_is_invariant_to_complete_bank_replication(self):
        config = {
            "risk": {
                "epsilon": 1e-6,
                "l2_grid": [0.0, 0.05],
                "learning_rate": 0.05,
                "logistic_steps": 120,
                "temperature_steps": 120,
                "support_alpha": 0.2,
            }
        }
        d = np.asarray([3.0, 2.0, 1.0, 0.5, -0.5, -1.0, -2.0, -3.0])
        u = np.asarray([0.1, 0.2, 0.3, 0.8, 0.2, 0.9, 1.0, 1.2])
        y = np.asarray([0, 0, 0, 1, 0, 1, 1, 1])
        seeds = np.asarray([0] * 4 + [1] * 4)
        clusters = np.asarray(["f0"] * 4 + ["f1"] * 4)
        banks = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])

        def fit(indices):
            weights = _hierarchical_sample_weights(
                seeds[indices], clusters[indices], banks[indices]
            )
            temperature = fit_temperature_no_intercept(
                d[indices], y[indices], config, sample_weights=weights
            )
            scalar = fit_scalar_risk_calibrator(
                d[indices],
                y[indices],
                (d[indices], y[indices]),
                config,
                direction="nonpositive",
                fit_weights=weights,
                selection_weights=weights,
            )
            joint = fit_constrained_risk_calibrator(
                d[indices],
                u[indices],
                y[indices],
                (d[indices], u[indices], y[indices]),
                config,
                fit_weights=weights,
                selection_weights=weights,
            )
            return temperature, scalar, joint

        baseline = fit(np.arange(d.size))
        replicated = fit(np.concatenate((np.arange(d.size), [0, 1, 0, 1])))
        self.assertAlmostEqual(
            baseline[0].temperature, replicated[0].temperature, places=12
        )
        self.assertAlmostEqual(baseline[1].intercept, replicated[1].intercept, places=12)
        self.assertAlmostEqual(
            baseline[1].coefficient, replicated[1].coefficient, places=12
        )
        self.assertEqual(baseline[1].l2, replicated[1].l2)
        self.assertAlmostEqual(baseline[2].intercept, replicated[2].intercept, places=12)
        self.assertAlmostEqual(baseline[2].beta_d, replicated[2].beta_d, places=12)
        self.assertAlmostEqual(baseline[2].beta_u, replicated[2].beta_u, places=12)
        self.assertEqual(baseline[2].l2, replicated[2].l2)

    def test_aurc_is_ranked_within_seed_before_macro_aggregation(self):
        errors = np.asarray([1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0])
        scores = np.asarray([0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1])
        seeds = np.asarray([0] * 4 + [1] * 4)
        clusters = np.asarray(["foundation"] * 8)
        banks = np.asarray(["bank"] * 8)
        result = _bank_macro_risk_coverage(
            errors, scores, clusters, banks, seed_ids=seeds
        )
        expected = np.mean(
            [
                risk_coverage(errors[seeds == value], scores[seeds == value])["aurc"]
                for value in (0, 1)
            ]
        )
        self.assertAlmostEqual(result["aurc"], expected)
        self.assertNotAlmostEqual(
            result["aurc"], risk_coverage(errors, scores)["aurc"]
        )
        self.assertEqual(result["algorithm_seed_count"], 2)

    def test_random_rejection_aurc_is_exact_uniform_ordering_expectation(self):
        errors = np.asarray([1.0, 2.0, 5.0, 9.0])
        observed = []
        for order in itertools.permutations(range(errors.size)):
            scores = np.empty(errors.size, dtype=np.float64)
            scores[np.asarray(order)] = np.arange(errors.size)
            observed.append(risk_coverage(errors, scores)["aurc"])
        self.assertAlmostEqual(
            _expected_random_rejection_aurc(errors), float(np.mean(observed))
        )

    def test_common_support_requires_complete_seed_bank_grid(self):
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
        probabilities = np.asarray([0.1, 0.8, 0.2, 0.9, 0.2, 0.7, 0.3, 0.8])
        errors = np.arange(1.0, 9.0)
        seeds = np.asarray([0] * 4 + [1] * 4)
        clusters = np.asarray(["f0", "f0", "f1", "f1"] * 2)
        banks = np.asarray(["a", "a", "b", "b"] * 2)
        support = np.ones(8, dtype=np.bool_)
        support[(seeds == 1) & (clusters == "f1")] = False
        with self.assertRaisesRegex(RuntimeError, "complete paired seed-by-bank grid"):
            risk_metrics(
                labels,
                probabilities,
                errors,
                support,
                clusters,
                banks,
                seed_ids=seeds,
            )
        contrast = _cluster_ranking_random_contrast(
            errors,
            probabilities,
            clusters,
            20,
            7,
            bank_ids=banks,
            seed_ids=seeds,
        )
        self.assertNotIn("p_value_two_sided", contrast)
        self.assertNotIn("sign_flip_unit", contrast)
        self.assertIn("familywise_ci95_low", contrast)

    def test_calibration_macro_and_ci_have_explicit_seed_layer(self):
        labels = np.asarray([0, 1] + [0, 1] * 5)
        probabilities = np.asarray([0.1, 0.9] + [0.9, 0.1] * 5)
        seeds = np.asarray([0, 0] + [1] * 10)
        clusters = np.asarray(["f"] * 12)
        banks = np.asarray(["b"] * 12)
        metrics = _bank_cluster_probability_metrics(
            labels, probabilities, clusters, banks, seeds
        )
        expected_brier = np.mean(
            [
                np.mean((probabilities[seeds == value] - labels[seeds == value]) ** 2)
                for value in (0, 1)
            ]
        )
        self.assertAlmostEqual(metrics["brier"], expected_brier)
        self.assertNotAlmostEqual(
            metrics["brier"], np.mean((probabilities - labels) ** 2)
        )

        ci_labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
        ci_probabilities = np.asarray([0.1, 0.8, 0.2, 0.9, 0.2, 0.7, 0.3, 0.8])
        ci_seeds = np.asarray([0] * 4 + [1] * 4)
        ci_clusters = np.asarray(["f0", "f0", "f1", "f1"] * 2)
        ci_banks = np.asarray(["a", "a", "b", "b"] * 2)
        intervals = _calibration_confidence_intervals(
            ci_labels,
            ci_probabilities,
            np.ones(8, dtype=np.bool_),
            ci_clusters,
            40,
            17,
            ci_banks,
            seed_ids=ci_seeds,
        )
        self.assertEqual(intervals["calibration_ci_algorithm_seed_count"], 2)
        self.assertEqual(
            intervals["calibration_ci_bootstrap_hierarchy"],
            "canonical foundation, canonical bank within foundation, "
            "one synchronized global algorithm-seed draw",
        )

    def test_three_seeds_do_not_structurally_block_simultaneous_bounds(self):
        cells = [
            {
                "seed": str(seed),
                "cluster": foundation,
                "bank": bank,
                "value": value,
            }
            for foundation in ("f0", "f1")
            for bank in (f"{foundation}-a", f"{foundation}-b")
            for seed in (0, 1, 2)
            for value in (2.0,)
        ]
        family = _simultaneous_hierarchical_family(
            {"joint_vs_control": cells, "joint_vs_random": cells},
            60,
            19,
        )
        self.assertGreater(
            family["records"]["joint_vs_control"]["familywise_ci95_low"],
            0.1,
        )
        self.assertEqual(family["family"]["global_seed_count"], 3)
        self.assertNotIn("p_value_two_sided", family["records"]["joint_vs_control"])

    def test_synchronized_bootstrap_shares_one_seed_draw_across_banks(self):
        cells = [
            {
                "seed": str(seed),
                "cluster": foundation,
                "bank": bank,
                "value": float(seed),
            }
            for foundation in ("f0", "f1")
            for bank in (f"{foundation}-a", f"{foundation}-b")
            for seed in (0, 1, 2)
        ]
        result = _synchronized_hierarchical_bootstrap(
            {"first": cells, "second": cells}, 30, 23, trace=True
        )
        self.assertTrue(
            result["audit"]["shared_seed_draw_across_all_banks_and_hypotheses"]
        )
        self.assertEqual(len(result["audit"]["seed_draws"]), 30)
        expected = np.asarray(
            [
                np.mean([float(seed) for seed in draw])
                for draw in result["audit"]["seed_draws"]
            ]
        )
        np.testing.assert_allclose(result["samples"]["first"], expected)
        np.testing.assert_allclose(result["samples"]["second"], expected)

    def test_synchronized_family_ignores_copied_bank_cells(self):
        cells = [
            {
                "seed": str(seed),
                "cluster": foundation,
                "bank": bank,
                "value": float(seed + bank_index),
            }
            for foundation in ("f0", "f1")
            for bank_index, bank in enumerate((f"{foundation}-a", f"{foundation}-b"))
            for seed in (0, 1, 2)
        ]
        original = _synchronized_hierarchical_bootstrap(
            {"metric": cells}, 40, 29
        )
        copied = _synchronized_hierarchical_bootstrap(
            {"metric": cells + cells[:3]}, 40, 29
        )
        self.assertEqual(original["estimates"], copied["estimates"])
        np.testing.assert_array_equal(
            original["samples"]["metric"], copied["samples"]["metric"]
        )

    def test_all_calibration_metrics_ignore_duplicate_canonical_units(self):
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
        probabilities = np.asarray([0.1, 0.8, 0.2, 0.9, 0.15, 0.75, 0.3, 0.85])
        errors = np.asarray([1.0, 4.0, 2.0, 5.0, 1.5, 4.5, 2.5, 5.5])
        support = np.ones(8, dtype=np.bool_)
        clusters = np.asarray(["a"] * 4 + ["b"] * 4)
        banks = np.asarray(["bank-a"] * 4 + ["bank-b"] * 4)
        units = np.asarray([f"unit-{index}" for index in range(8)])
        original = risk_metrics(
            labels, probabilities, errors, support, clusters, banks, units
        )
        duplicate = np.asarray([0, 2])
        duplicated = risk_metrics(
            np.concatenate((labels, labels[duplicate])),
            np.concatenate((probabilities, probabilities[duplicate])),
            np.concatenate((errors, errors[duplicate])),
            np.concatenate((support, support[duplicate])),
            np.concatenate((clusters, clusters[duplicate])),
            np.concatenate((banks, banks[duplicate])),
            np.concatenate((units, units[duplicate])),
        )
        self.assertEqual(original, duplicated)
        original_ci = _calibration_confidence_intervals(
            labels, probabilities, support, clusters, 40, 123, banks, units
        )
        duplicated_ci = _calibration_confidence_intervals(
            np.concatenate((labels, labels[duplicate])),
            np.concatenate((probabilities, probabilities[duplicate])),
            np.concatenate((support, support[duplicate])),
            np.concatenate((clusters, clusters[duplicate])),
            40,
            123,
            np.concatenate((banks, banks[duplicate])),
            np.concatenate((units, units[duplicate])),
        )
        self.assertEqual(original_ci, duplicated_ci)

    def test_single_material_vocabulary_has_legal_frozen_proposals(self):
        world_map = np.zeros((3, 2, 2), dtype=np.float64)
        maps, actions, audit = frozen_map_proposals(
            world_map,
            np.asarray([1.0, 2.0]),
            np.asarray(["occupancy", "height", "material"]),
            1,
            {
                "risk": {
                    "proposal_seed": 7,
                    "proposal_height_step": 1.0,
                    "proposal_count": 4,
                }
            },
        )
        self.assertEqual(audit["proposal_count"], 4)
        np.testing.assert_array_equal(maps[:, 2], 0.0)
        self.assertEqual(actions.shape[1], 6)

    def test_perfect_response_has_complete_transition_metrics(self):
        spec = PatchSpec(
            antennas=2,
            subcarriers=2,
            patch_complex_size=2,
            patch_antenna_size=1,
            patch_subcarrier_size=2,
        )
        source_csi = np.asarray([[1.0, 0.5, 0.25, 0.75, 0.0, 0.1, 0.2, 0.3]])
        target_csi = np.asarray([[0.8, 0.3, 0.5, 0.9, 0.2, 0.2, 0.4, 0.1]])
        source = patchify_csi(source_csi, spec)
        target = patchify_csi(target_csi, spec)
        normalization = SimpleNamespace(
            patch_mean=np.zeros((spec.patch_count, spec.patch_dim)),
            patch_scale=np.ones((spec.patch_count, spec.patch_dim)),
        )
        result = _transition_metrics(
            target, source, source, target, normalization, spec
        )
        self.assertAlmostEqual(result["native_sgcs"], 1.0)
        self.assertNotIn("native_transition_skill", result)
        for name in ("path_loss", "delay_spread", "angular_spread"):
            self.assertAlmostEqual(result[f"native_{name}_change_mae"], 0.0)
            self.assertAlmostEqual(result[f"native_{name}_direction_accuracy"], 1.0)

    def test_copy_prediction_has_zero_delta_sgcs(self):
        spec = PatchSpec(
            antennas=2,
            subcarriers=2,
            patch_complex_size=2,
            patch_antenna_size=1,
            patch_subcarrier_size=2,
        )
        source = patchify_csi(
            np.asarray([[1.0, 0.5, 0.25, 0.75, 0.0, 0.1, 0.2, 0.3]]),
            spec,
        )
        target = patchify_csi(
            np.asarray([[0.8, 0.3, 0.5, 0.9, 0.2, 0.2, 0.4, 0.1]]),
            spec,
        )
        normalization = SimpleNamespace(
            patch_mean=np.zeros((spec.patch_count, spec.patch_dim)),
            patch_scale=np.ones((spec.patch_count, spec.patch_dim)),
        )
        result = _transition_metrics(
            source, source, source, target, normalization, spec
        )
        self.assertAlmostEqual(result["native_sgcs"], 0.0)
        self.assertNotIn("native_transition_skill", result)

    def test_transition_delta_uses_model_zero_action_not_source_truth(self):
        spec = PatchSpec(
            antennas=2,
            subcarriers=2,
            patch_complex_size=2,
            patch_antenna_size=1,
            patch_subcarrier_size=2,
        )
        source = patchify_csi(np.zeros((1, 8), dtype=np.float64), spec)
        target = patchify_csi(np.ones((1, 8), dtype=np.float64), spec)
        zero_action = patchify_csi(
            np.full((1, 8), 10.0, dtype=np.float64), spec
        )
        action = patchify_csi(
            np.full((1, 8), 11.0, dtype=np.float64), spec
        )
        normalization = SimpleNamespace(
            patch_mean=np.zeros((spec.patch_count, spec.patch_dim)),
            patch_scale=np.ones((spec.patch_count, spec.patch_dim)),
        )
        result = _transition_metrics(
            action, zero_action, source, target, normalization, spec
        )
        source_baseline = _transition_metrics(
            action, source, source, target, normalization, spec
        )
        self.assertAlmostEqual(result["native_sgcs"], 1.0)
        self.assertNotIn("native_transition_skill", result)
        self.assertNotAlmostEqual(
            result["native_path_loss_change_mae"],
            source_baseline["native_path_loss_change_mae"],
        )
        for name in ("path_loss", "delay_spread", "angular_spread"):
            self.assertTrue(np.isfinite(result[f"native_{name}_change_mae"]))
            self.assertTrue(np.isfinite(result[f"native_{name}_direction_accuracy"]))

    def test_protocol_transition_skill_perfect_latent_match(self):
        latent_source = np.zeros((2, 3, 4), dtype=np.float64)
        latent_target = np.ones((2, 3, 4), dtype=np.float64)
        include = np.asarray([True, True])
        self.assertEqual(
            _protocol_transition_skill(
                latent_target, latent_source, latent_target, include
            ),
            1.0,
        )

    def test_protocol_transition_skill_copy_latent(self):
        latent_source = np.zeros((2, 3, 4), dtype=np.float64)
        latent_target = np.ones((2, 3, 4), dtype=np.float64)
        include = np.asarray([True, True])
        self.assertEqual(
            _protocol_transition_skill(
                latent_source, latent_source, latent_target, include
            ),
            0.0,
        )

    def test_protocol_transition_skill_is_bank_level_ratio_not_mean(self):
        latent_source = np.asarray([[[0.0]], [[0.0]]], dtype=np.float64)
        latent_target = np.asarray([[[2.0]], [[4.0]]], dtype=np.float64)
        latent_prediction = np.asarray([[[1.0]], [[0.0]]], dtype=np.float64)
        include = np.asarray([True, True])
        unit_errors = np.sqrt(
            np.mean((latent_prediction - latent_target) ** 2, axis=(1, 2))
        )
        unit_denoms = np.sqrt(
            np.mean((latent_source - latent_target) ** 2, axis=(1, 2))
        )
        expected = 1.0 - float(np.sum(unit_errors) / np.sum(unit_denoms))
        unit_mean = float(np.mean(1.0 - unit_errors / unit_denoms))
        skill = _protocol_transition_skill(
            latent_prediction, latent_source, latent_target, include
        )
        self.assertAlmostEqual(skill, expected)
        self.assertNotAlmostEqual(skill, unit_mean)

    def test_protocol_transition_skill_empty_or_tiny_denom_is_none(self):
        latent = np.ones((2, 3, 4), dtype=np.float64)
        self.assertIsNone(
            _protocol_transition_skill(
                latent, latent, latent, np.asarray([False, False])
            )
        )
        self.assertIsNone(
            _protocol_transition_skill(
                latent, latent, latent, np.asarray([True, True])
            )
        )

    def test_protocol_transition_skill_ignores_non_teacher_sensitive_units(self):
        latent_source = np.zeros((2, 3, 4), dtype=np.float64)
        latent_target = np.ones((2, 3, 4), dtype=np.float64)
        latent_prediction = np.stack(
            (latent_target[0], np.full((3, 4), 99.0, dtype=np.float64))
        )
        include = np.asarray([True, False])
        self.assertEqual(
            _protocol_transition_skill(
                latent_prediction, latent_source, latent_target, include
            ),
            1.0,
        )

    def test_evaluation_gate_accepts_na_transition_skill(self):
        strong = {
            "assessed": True,
            "cluster_count": 4,
            "paired_mean_difference": 0.4,
            "ci95_low": 0.2,
            "ci95_high": 0.6,
            "confidence_level": 0.95,
            "p_value_two_sided": 1e-12,
        }
        scoped = {
            scope: {
                "alignment_superiority": dict(strong),
                "response_superiority": dict(strong),
                "response_vs_copy": dict(strong),
                "response_vs_no_action": dict(strong),
                "response_vs_action_swap": dict(strong),
                "response_direction": dict(strong),
            }
            for scope in ("source_final_unseen_bank", "target:target-a")
        }
        dataset = SimpleNamespace(
            city_ids=np.asarray(["source-a", "target-a"]),
            scene_roles=np.asarray(["source_final_unseen_bank", "target"]),
            bank_ids=np.asarray(["source-bank", "target-bank"]),
            is_fixture=False,
            indices_for_role=lambda role: np.asarray(
                [0] if role == "source_final_unseen_bank" else [1]
            ),
        )
        config = {
            "seeds": [1],
            "factorial": {
                "arms": ["endpoint", "alignment", "response", "full"]
            },
            "qualification": {
                "minimum_geometry_matched_wrong_action_fraction": 0.5
            },
            "evaluation": {
                "bootstrap_resamples": 100,
                "familywise_alpha": 0.05,
                "minimum_alignment_superiority": 0.01,
                "minimum_response_superiority": 0.01,
                "minimum_native_probe_correlation": 0.0,
                "null_score_equivalence_margin": 0.01,
                "response_null_violation_rate_max": 0.05,
                "response_null_equivalence_margin": 0.01,
                "minimum_cgs_noninferiority": -0.01,
                "minimum_response_noninferiority": -0.01,
            },
        }
        response_row = {
            "arm": "response",
            "native_null_violation_rate": 0.0,
            "native_latent_null_violation_rate": 0.0,
            "native_null_delta_rms_mean": 0.0,
            "native_latent_null_delta_rms_mean": 0.0,
            "native_action_swap_exact_count": 1,
            "native_action_swap_exact_fraction": 1.0,
            "probe_action_swap_exact_count": 1,
            "probe_action_swap_exact_fraction": 1.0,
            "probe_oracle_x_active_patch_nmse": 0.0,
            "native_delta_relative_magnitude_error": 0.0,
            "native_sgcs": 1.0,
            "native_transition_skill": None,
            "native_path_loss_change_mae": 0.0,
            "native_delay_spread_change_mae": 0.0,
            "native_angular_spread_change_mae": 0.0,
            "native_path_loss_direction_accuracy": 1.0,
            "native_delay_spread_direction_accuracy": 1.0,
            "native_angular_spread_direction_accuracy": 1.0,
        }
        contracts = {
            "scene_id_only": "stable_hashed_bank_token_no_label_feature",
            "edit_status_xor": "separate_world_bits_and_natural_flags_no_xor",
            "variant_id_matcher": "separate_stable_hashed_variant_tokens_no_match_or_unk_override",
        }
        shortcut_rows = []
        for arm, bank in itertools.product(config["factorial"]["arms"], dataset.bank_ids):
            for baseline in (
                "constant",
                "csi_only",
                "map_only",
                "scene_id_only",
                "edit_status_xor",
                "variant_id_matcher",
            ):
                row = {
                    "seed": 1,
                    "arm": arm,
                    "canonical_bank_digest": bank,
                    "baseline": baseline,
                    "unseen_bank_auroc": 0.5,
                }
                if baseline in contracts:
                    row["identity_token_contract"] = contracts[baseline]
                shortcut_rows.append(row)
        null_safety = {
            arm: {"passed": True} for arm in config["factorial"]["arms"]
        }

        def run_gate(row):
            with (
                patch(
                    "formal_v2.formal_evaluation._null_safety_by_arm",
                    return_value=null_safety,
                ),
                patch(
                    "formal_v2.formal_evaluation._paired_arm_comparison",
                    return_value=dict(strong),
                ),
                patch(
                    "formal_v2.formal_evaluation._within_arm_advantage_interval",
                    return_value=dict(strong),
                ),
                patch(
                    "formal_v2.formal_evaluation._within_arm_level_interval",
                    return_value=dict(strong),
                ),
                patch(
                    "formal_v2.formal_evaluation._g3_primary_scope_intervals",
                    return_value=scoped,
                ),
                patch(
                    "formal_v2.formal_evaluation._effect_bins_complete",
                    return_value=True,
                ),
                patch(
                    "formal_v2.formal_evaluation._gray_cells_complete",
                    return_value=True,
                ),
                patch(
                    "formal_v2.formal_evaluation._native_probe_correlation_macro",
                    return_value=1.0,
                ),
                patch(
                    "formal_v2.formal_evaluation._canonical_bank_digest",
                    side_effect=lambda data, scene: str(data.bank_ids[scene]),
                ),
            ):
                return _evaluation_gate(
                    config,
                    dataset,
                    [],
                    [],
                    [],
                    [row, {**row, "arm": "full"}],
                    shortcut_rows,
                    {"g4_subgates": {"complete": "PASS"}, "gate_vector": {"G5": "PASS"}},
                    {},
                    qualification_gate_sha256="a" * 64,
                    factorial_gate_sha256="b" * 64,
                )

        na_gate = run_gate(response_row)
        self.assertEqual(
            na_gate["g3_subgates"]["4_response_direction_and_magnitude"],
            "PASS",
        )
        finite_fail = run_gate({**response_row, "native_sgcs": float("nan")})
        self.assertEqual(
            finite_fail["g3_subgates"]["4_response_direction_and_magnitude"],
            "FAIL",
        )
        skill_nan = run_gate({**response_row, "native_transition_skill": float("nan")})
        self.assertEqual(
            skill_nan["g3_subgates"]["4_response_direction_and_magnitude"],
            "FAIL",
        )

    def test_nonexact_action_swap_is_excluded_from_response_effect(self):
        evaluated = {
            "pair_ids": np.asarray(["u", "u"]),
            "targets": np.asarray([[1.0], [2.0]]),
            "source_targets": np.asarray([[0.0], [0.0]]),
            "wrong_action_match_status": np.asarray(["fallback", "fallback"]),
            "wrong_action_world": np.asarray([3, 3]),
            "scene_indices": np.asarray([0, 0]),
            "bank_ids": np.asarray(["b", "b"]),
            "base_map_cluster_ids": np.asarray(["c", "c"]),
            "canonical_base_map_digests": np.asarray(["c", "c"]),
            "canonical_bank_digests": np.asarray(["b", "b"]),
            "city_ids": np.asarray(["city", "city"]),
            "source_worlds": np.asarray([0, 0]),
            "target_worlds": np.asarray([1, 1]),
            "positions": np.asarray([0, 0]),
            "queries": np.asarray([0, 1]),
            "routes": np.asarray(["active", "active"]),
        }
        rows = _response_effect_rows(
            1,
            "response",
            evaluated,
            np.asarray([[1.0], [2.0]]),
            np.asarray([[9.0], [9.0]]),
            np.asarray([[0.0], [0.0]]),
        )
        self.assertIsNone(rows[0]["action_swap_mse"])
        self.assertIsNone(rows[0]["response_advantage_vs_action_swap"])
        self.assertEqual(rows[0]["wrong_action_match_status"], "fallback")

    def test_headline_alignment_excludes_privileged_natural_incident_edges(self):
        dataset = SimpleNamespace(natural_world_index=np.asarray([2]))
        self.assertFalse(
            headline_alignment_edge(
                dataset,
                0,
                SimpleNamespace(source_world=2, target_world=3),
            )
        )
        self.assertTrue(
            headline_alignment_edge(
                dataset,
                0,
                SimpleNamespace(source_world=0, target_world=1),
            )
        )


if __name__ == "__main__":
    unittest.main()
