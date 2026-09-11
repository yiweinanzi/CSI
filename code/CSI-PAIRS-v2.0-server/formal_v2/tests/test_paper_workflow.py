import copy
import csv
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

from formal_v2.formal_localization import _optimize_head
from formal_v2.formal_probes import CompatibilityProbe, _fit_binary, fit_action_response_probe
from formal_v2.paper_suite import (
    load_config, export_tables, localization_table, Sources, read_rows,
    PROJECT, REPO, digest, stage_gate, map_condition_panel, metric_panel,
)


class ScalarHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.value.expand(len(x), 2), torch.zeros((len(x), 2))


class HeadStateSelectionTests(unittest.TestCase):
    def test_overshooting_update_cannot_be_selected_using_previous_loss(self):
        head = ScalarHead()
        _optimize_head(head, np.zeros((4, 1)), np.zeros((4, 2)), steps=1,
                       learning_rate=10.0, sigma_min=0.001, ridge=0)
        self.assertEqual(float(head.value), 1.0)

    def test_final_improving_iterate_is_considered(self):
        head = ScalarHead()
        taken = _optimize_head(head, np.zeros((4, 1)), np.zeros((4, 2)), steps=1,
                               learning_rate=0.1, sigma_min=0.001, ridge=0)
        self.assertEqual(taken, 1)
        self.assertLess(float(head.value), 1.0)

    def test_nonfinite_objective_is_not_saved(self):
        with self.assertRaises(FloatingPointError):
            _optimize_head(ScalarHead(), np.zeros((4, 1)), np.full((4, 2), np.nan),
                           steps=2, learning_rate=0.1, sigma_min=0.001, ridge=0)


class ProbeRecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = {"evaluation": {"probe_learning_rate": 0.01, "probe_steps": 6, "probe_hidden_dim": 4}}
        self.x = np.random.default_rng(1).normal(size=(11, 3)).astype("float32")
        self.y = np.arange(11) % 2

    def initial(self):
        torch.manual_seed(7)
        return CompatibilityProbe(3, "mlp2", 4)

    def interrupt(self, event):
        if event["phase"] == "training" and event["step"] == 2:
            raise RuntimeError("injected interruption")

    def test_binary_resume_matches_uninterrupted_full_and_accumulated_training(self):
        for batch_rows in (None, 3):
            with self.subTest(batch_rows=batch_rows), tempfile.TemporaryDirectory() as temp:
                expected = self.initial()
                _fit_binary(expected, self.x, self.y, self.config, train_batch_rows=batch_rows)
                path = Path(temp) / "binary.pt"
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    _fit_binary(self.initial(), self.x, self.y, self.config, train_batch_rows=batch_rows,
                                checkpoint_path=path, checkpoint_interval=1, progress_callback=self.interrupt)
                actual, events = self.initial(), []
                _fit_binary(actual, self.x, self.y, self.config, train_batch_rows=batch_rows,
                            checkpoint_path=path, checkpoint_interval=1, progress_callback=events.append)
                self.assertEqual(events[0]["step"], 2)
                for name, value in expected.state_dict().items():
                    self.assertTrue(torch.equal(value, actual.state_dict()[name]), name)

    def test_resume_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "binary.pt"
            _fit_binary(self.initial(), self.x, self.y, self.config, checkpoint_path=path)
            changed = self.x.copy()
            changed[0, 0] += 0.01
            with self.assertRaisesRegex(ValueError, "changed"):
                _fit_binary(self.initial(), changed, self.y, self.config, checkpoint_path=path)

    def test_response_optimizer_and_zero_action_resume(self):
        targets = np.stack((self.y, -self.y), axis=1).astype("float32")
        for rows in (None, 4):
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as temp:
                options = dict(seed=17, zero_action_x=np.zeros_like(self.x), train_batch_rows=rows)
                expected = fit_action_response_probe(self.x, targets, self.config, **options)
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    fit_action_response_probe(self.x, targets, self.config, **options,
                                              checkpoint_dir=temp, checkpoint_interval=1, progress_callback=self.interrupt)
                actual = fit_action_response_probe(self.x, targets, self.config, **options,
                                                   checkpoint_dir=temp, checkpoint_interval=1)
                self.assertTrue(actual.zero_preserving)
                for name, value in expected.state_dict().items():
                    self.assertTrue(torch.equal(value, actual.state_dict()[name]), name)

    def test_corrupt_optimizer_state_cannot_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "binary.pt"
            _fit_binary(self.initial(), self.x, self.y, self.config, checkpoint_path=path)
            checkpoint = torch.load(path, weights_only=True)
            next(iter(checkpoint["optimizer"]["state"].values()))["exp_avg"].fill_(float("nan"))
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                _fit_binary(self.initial(), self.x, self.y, self.config, checkpoint_path=path)


class PaperTableTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def write_csv(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def example_row(self):
        return {"arm": "full", "city_id": "target-boston", "budget": 0,
                "training_seeds": 3, "label_draws": 1, "independent_base_map_clusters": 41,
                "mean_bank_median_error_m": 2.0, "mean_bank_p90_error_m": 4.0,
                "dataset_sha256": self.config["dataset_sha256"], "config_sha256": "x",
                "fixture": False, "scientific_use": "CANDIDATE_NOT_CLAIM"}

    def test_archive_exports_real_negative_results_and_missing_baselines(self):
        with tempfile.TemporaryDirectory() as temp:
            coverage = export_tables(self.config, REPO, Path(temp), archive=True)
            self.assertEqual(coverage["tables"]["main2"]["COMPLETE"], 64)
            self.assertEqual(coverage["tables"]["main2"]["MISSING"], 32)
            self.assertNotEqual(coverage["tables"]["main2"]["status"], "COMPLETE")
            with (Path(temp) / "main2.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            full = next(r for r in rows if r["method"] == "full" and r["city"] == "target-boston" and r["budget"] == "0" and r["metric"] == "mean_bank_median_error_m")
            self.assertAlmostEqual(float(full["value"]), 11.985595121032393)
            self.assertEqual(len(list(Path(temp).glob("*.tex"))), 6)

    def test_no_historical_fallback_for_current_run(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = localization_table(Sources(temp, self.config))
            self.assertTrue(all(r["status"] == "MISSING" for r in rows))

    def test_duplicates_and_nonfinite_values_rejected(self):
        for problem in ("duplicate", "nan"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as temp:
                row = self.example_row()
                if problem == "nan":
                    row["mean_bank_median_error_m"] = "NaN"
                self.write_csv(Path(temp) / "factorial/localization_summary.csv", [row, row] if problem == "duplicate" else [row])
                with self.assertRaises(ValueError):
                    localization_table(Sources(temp, self.config))

    def test_fixture_and_foreign_dataset_rejected(self):
        for key, value in (("fixture", True), ("dataset_sha256", "another-dataset")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                row = self.example_row()
                row[key] = value
                path = Path(temp) / "table.csv"
                self.write_csv(path, [row])
                with self.assertRaises(ValueError):
                    read_rows(path, self.config["dataset_sha256"])

    def test_diagnostic_baseline_cannot_be_mislabelled_as_main_result(self):
        with tempfile.TemporaryDirectory() as temp:
            row = self.example_row()
            row["arm"] = "Wi-GATr"
            self.write_csv(Path(temp) / "paper_baselines/localization_summary.csv", [row])
            with self.assertRaisesRegex(ValueError, "shared-query"):
                localization_table(Sources(temp, self.config))





    def test_partial_risk_rows_cannot_make_a_complete_appendix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = {"arm": "full", "risk_model": "joint", "dataset_sha256": self.config["dataset_sha256"], "config_sha256": "x"}
            row.update({name: 0.2 for name in ("aurc", "ece", "brier", "nll", "support_coverage", "risk_error_spearman")})
            self.write_csv(root / "risk/metrics.csv", [row])
            coverage = export_tables(self.config, root, root / "tables")
            self.assertEqual(coverage["tables"]["s4"]["status"], "PARTIAL")
            self.assertGreaterEqual(coverage["tables"]["s4"]["MISSING"], 11)

    def test_map_diagnostics_aggregate_banks_and_require_all_conditions(self):
        with tempfile.TemporaryDirectory() as temp:
            config = copy.deepcopy(self.config)
            config.update(map_models=["PMNet"], cities=["target-boston"], minimum_clusters_per_city=1)
            rows = []
            for condition in ("correct", "paired_active_alternative", "paired_null_alternative", "wrong_city", "geometry_destroyed", "empty"):
                for bank, errors in (("a", [2, 4]), ("b", [9])):
                    for i, error in enumerate(errors):
                        rows.append(dict(model_name="PMNet", city_id="target-boston", condition=condition, bank_id=bank, base_map_cluster_id="cluster", unit_id=f"{bank}-{i}", localization_error_m=error, dataset_sha256=config["dataset_sha256"]))
            path = Path(temp) / "external_baselines/six_condition_results.csv"
            self.write_csv(path, rows)
            result = map_condition_panel(Sources(temp, config))
            self.assertTrue(all(r["status"] == "COMPLETE" for r in result))
            self.assertEqual(result[0]["value"], 6.0)  # Equal bank means: (median(2,4) + 9)/2.
            self.write_csv(path, rows[:-1])
            self.assertTrue(all(r["status"] != "COMPLETE" for r in map_condition_panel(Sources(temp, config))))

    def test_target_panel_does_not_mix_source_final_banks(self):
        with tempfile.TemporaryDirectory() as temp:
            config = copy.deepcopy(self.config)
            config.update(arms=["full"], cities=["target-boston"], seeds=[1], minimum_clusters_per_city=1)
            rows = [dict(arm="full", city_id=city, bank_id="bank", seed=1, cgs_auroc=value, dataset_sha256=config["dataset_sha256"], config_sha256="x") for city, value in (("target-boston", .75), ("source-city", .99))]
            self.write_csv(Path(temp) / "evaluation/cgs_per_bank.csv", rows)
            result = metric_panel(Sources(temp, config), "evaluation/cgs_per_bank.csv", ["cgs_auroc"], "alignment")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["value"], .75)
            self.assertEqual(result[0]["status"], "COMPLETE")


if __name__ == "__main__":
    unittest.main()
