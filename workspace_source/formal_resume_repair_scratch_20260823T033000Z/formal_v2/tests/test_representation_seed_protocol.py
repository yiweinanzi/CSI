import json
import tempfile
import unittest
from pathlib import Path

from formal_v2.formal_representation_baselines import (
    _summarize_across_seeds,
    _validate_representation_outputs,
    load_representation_config,
)


ROOT = Path(__file__).resolve().parents[2]


class RepresentationSeedConfigTests(unittest.TestCase):
    def test_formal_config_uses_three_distinct_explicit_seeds(self):
        config = load_representation_config(
            ROOT / "formal_v2/configs/representation_baselines_v1.json"
        )
        self.assertNotIn("seed", config)
        self.assertGreaterEqual(len(config["seeds"]), 3)
        self.assertEqual(len(config["seeds"]), len(set(config["seeds"])))

    def test_formal_profile_rejects_a_single_seed(self):
        source = json.loads(
            (ROOT / "formal_v2/configs/representation_baselines_v1.json").read_text()
        )
        source["seeds"] = [17]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "single-seed-formal.json"
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "at least three independent algorithm seeds"):
                load_representation_config(path)

    def test_legacy_single_seed_is_normalized_only_for_smoke(self):
        smoke = load_representation_config(
            ROOT / "formal_v2/configs/representation_baselines_smoke_v1.json"
        )
        self.assertEqual(smoke["profile"], "software-smoke-only")
        self.assertEqual(smoke["seeds"], [20260901])

        source = json.loads(
            (ROOT / "formal_v2/configs/representation_baselines_v1.json").read_text()
        )
        source["schema_version"] = "csi-pairs-v6-representation-baselines-v1"
        source["seed"] = source.pop("seeds")[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy-formal.json"
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "software-smoke-only"):
                load_representation_config(path)


class RepresentationOutputGateTests(unittest.TestCase):
    def _complete_rows(self):
        models = ["model-a", "model-b"]
        seeds = [11, 22, 33]
        unit = ("target", "city-a", 4, 0)
        statuses = []
        metrics = []
        queries = []
        for model_name in models:
            for seed in seeds:
                statuses.append(
                    {
                        "model_name": model_name,
                        "algorithm_seed": seed,
                        "checkpoint_path": f"{model_name}/seed_{seed}/source_selected.pt",
                        "checkpoint_sha256": "a" * 64,
                        "status": "PASS",
                    }
                )
                metrics.append(
                    {
                        "model_name": model_name,
                        "algorithm_seed": seed,
                        "split_role": unit[0],
                        "city_id": unit[1],
                        "budget": unit[2],
                        "draw": unit[3],
                    }
                )
                queries.append(
                    {
                        "model_name": model_name,
                        "algorithm_seed": seed,
                        "split_role": unit[0],
                        "city_id": unit[1],
                        "budget": unit[2],
                        "draw": unit[3],
                        "independent_unit_id": "cluster-a",
                        "bank_id": "bank-a",
                        "position_id": "position-a",
                        "world": 0,
                    }
                )
        return models, seeds, (unit,), statuses, metrics, queries

    def test_gate_requires_complete_common_model_seed_units(self):
        models, seeds, units, statuses, metrics, queries = self._complete_rows()
        result = _validate_representation_outputs(
            statuses, metrics, queries, models, seeds, units
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["metric_units_complete"])
        self.assertTrue(result["query_units_complete"])

        missing = _validate_representation_outputs(
            statuses, metrics[:-1], queries, models, seeds, units
        )
        self.assertFalse(missing["passed"])
        self.assertFalse(missing["metric_units_complete"])
        self.assertTrue(any("metric-unit mismatch" in value for value in missing["errors"]))

    def test_gate_rejects_noncommon_and_duplicate_query_units(self):
        models, seeds, units, statuses, metrics, queries = self._complete_rows()
        queries[-1] = dict(queries[-1], position_id="different-position")
        result = _validate_representation_outputs(
            statuses, metrics, queries, models, seeds, units
        )
        self.assertFalse(result["query_units_complete"])
        self.assertTrue(any("not common" in value for value in result["errors"]))

    def test_gate_rejects_reused_checkpoint_paths(self):
        models, seeds, units, statuses, metrics, queries = self._complete_rows()
        statuses[-1]["checkpoint_path"] = statuses[0]["checkpoint_path"]
        result = _validate_representation_outputs(
            statuses, metrics, queries, models, seeds, units
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("reused" in value for value in result["errors"]))


class RepresentationAcrossSeedSummaryTests(unittest.TestCase):
    def test_summary_is_cluster_macro_then_seed_level(self):
        per_seed = {
            11: {"cluster-a": [1.0, 3.0], "cluster-b": [10.0]},
            22: {"cluster-a": [2.0], "cluster-b": [4.0]},
            33: {"cluster-a": [9.0], "cluster-b": [1.0]},
        }
        rows = []
        for seed, clusters in per_seed.items():
            for cluster, values in clusters.items():
                for value in values:
                    rows.append(
                        {
                            "model_name": "model-a",
                            "algorithm_seed": seed,
                            "paper_label": "controlled",
                            "implementation_status": "style-controlled-implementation",
                            "split_role": "target",
                            "city_id": "city-a",
                            "budget": 4,
                            "draw": 0,
                            "independent_unit_id": cluster,
                            "localization_error_m": value,
                        }
                    )
        summary = _summarize_across_seeds(rows, [11, 22, 33], bootstrap_draws=500)
        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["algorithm_seed_count"], 3)
        self.assertTrue(row["all_expected_algorithm_seeds_present"])
        self.assertEqual(row["cluster_count_min"], 2)
        self.assertEqual(row["cluster_count_max"], 2)
        self.assertAlmostEqual(row["cluster_macro_across_seed_mean_error_m"], 14.0 / 3.0)
        self.assertAlmostEqual(row["cluster_macro_across_seed_std_m"], (7.0 / 3.0) ** 0.5)
        self.assertLessEqual(
            row["cluster_macro_seed_bootstrap_ci95_low_m"],
            row["cluster_macro_across_seed_mean_error_m"],
        )
        self.assertGreaterEqual(
            row["cluster_macro_seed_bootstrap_ci95_high_m"],
            row["cluster_macro_across_seed_mean_error_m"],
        )
        self.assertFalse(row["claim_eligible"])


if __name__ == "__main__":
    unittest.main()
