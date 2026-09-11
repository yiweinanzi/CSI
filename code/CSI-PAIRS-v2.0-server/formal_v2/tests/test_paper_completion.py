import copy
import json
from pathlib import Path
from unittest import mock
import numpy as np
import pytest
import torch
from formal_v2.tests.test_paper_core import example
from formal_v2 import formal_factorial as f
from formal_v2.formal_localization import fit_source_position_head, adapt_position_head, predict_position_distribution
from formal_v2.paper_core import feature_scene, source_arrays, run_job, load_profile, merge_outputs
from formal_v2.paper_suite import PROJECT, digest
torch.set_num_threads(2)


def test_standardized_head_respects_coordinate_units_and_frozen_source_statistics():
    config = {"model": {"hidden_dim": 16}, "localization": {"head_steps": 30,
        "learning_rate": .01, "sigma_min": .001, "ridge": .001, "standardize": True}}
    rng = np.random.default_rng(71)
    x = rng.normal(size=(24, 5)).astype("float32")
    y = rng.normal(size=(24, 2)).astype("float32")
    a = fit_source_position_head(x, y, config, seed=41)
    b = fit_source_position_head(x * 100 + 50, y * 200 + 1000, config, seed=41)
    mean_a, variance_a, _ = predict_position_distribution(a, x)
    mean_b, variance_b, _ = predict_position_distribution(b, x * 100 + 50)
    np.testing.assert_allclose(mean_b, mean_a * 200 + 1000, atol=.03)
    np.testing.assert_allclose(variance_b, variance_a * 200 ** 2, rtol=.005)
    adapted = adapt_position_head(b, x[:3] * 100 + 50, y[:3] * 200 + 1000, config)
    for name in ("feature_mean", "feature_scale", "position_mean", "position_scale"):
        torch.testing.assert_close(getattr(adapted, name), getattr(b, name), rtol=0, atol=0)


def test_gradient_pilot_and_shared_localization_are_executable(example):
    _, _, config, _, data, teacher, _, route_norm, norm, model, _ = example
    config = copy.deepcopy(config)
    config["factorial"]["loss_normalization"] = "source_gradient"
    config["factorial"]["full_joint_loss"] = "static_sum"
    config["localization"]["standardize"] = True
    train = f._build_corpus(data, data.indices_for_role("source_encoder_train"), teacher, config, route_norm, norm)
    selection = f._build_corpus(data, data.indices_for_role("source_method_selection"), teacher, config, route_norm, norm)
    pilot = f._pilot_scales(config, train, selection, 21, device="cpu")
    assert pilot["normalization_method"] == "source_gradient"
    assert len(pilot["source_gradient_ratios"]) == min(8, config["factorial"]["pilot_steps"])
    assert all(np.isfinite(pilot[key]) and pilot[key] > 0 for key in ("alignment_scale", "response_scale"))
    trained, row = f._train_arm(config, train, 17, "full", pilot, step_count=2, device="cpu")
    assert row["full_joint_loss"] == "static_sum"
    scenes = [int(s) for role in ("source_encoder_train", "target") for s in data.indices_for_role(role)]
    features = f._natural_representations(trained, data, scenes, norm, teacher.patch_spec, batch_rows=2)
    one = {**config, "seeds": [17]}
    ours, ours_samples = f._run_localization(one, data, {(17, "full"): trained}, norm, teacher.patch_spec, ["cpu"], methods=["full"])
    other, other_samples = f._run_localization(one, data, {}, None, None, ["cpu"], methods=["PMNet"], feature_provider=lambda *_: features)
    from formal_v2.paper_export import verify_comparison
    verify_comparison(ours, other)
    np.testing.assert_allclose([r["error_m"] for r in ours_samples], [r["error_m"] for r in other_samples], rtol=.001, atol=.001)
    other[0]["query_position_ids"] += ";unmatched"
    with pytest.raises(ValueError, match="coverage differs"):
        verify_comparison(ours, other)


@pytest.mark.parametrize("task", ["alignment:map_only", "alignment:variant_id_matcher", "response:without_map", "response:csi_only", "response:oracle_x"])
def test_extra_probes_really_fit_score_and_resume_independently(example, tmp_path, task):
    _, _, config, _, data, teacher, _, route_norm, norm, model, _ = example
    profile = load_profile(PROJECT / "formal_v2/configs/paper_all_probes.json")
    profile.update(probe_updates=[1], batch_rows=16, encoding_batch_rows=16)
    base = {"arm": "full", "seed": 17, "checkpoint_sha256": "fixture", "fixture": True, "scientific_use": "FORBIDDEN"}
    job = tmp_path / task.replace(":", "-")
    run_job(task, job, model, data, teacher, config, norm, route_norm, profile, base)
    with mock.patch("formal_v2.paper_core.select_probe", side_effect=AssertionError("Completed task must not refit")):
        run_job(task, job, model, data, teacher, config, norm, route_norm, profile, base)
    assert (job / "complete.json").exists()


def test_teacher_minibatch_and_disk_storage_match_memory_execution(example, tmp_path):
    from formal_v2.formal_teacher import train_teacher_bundle
    _, _, config, _, data, teacher, *_ = example
    config = copy.deepcopy(config)
    config["teacher"]["readout_protocol"] = "minibatch_updates"
    csi = data.csi[data.indices_for_role("source_encoder_train")]
    expected = train_teacher_bundle(csi, teacher.patch_spec, config, seed=98)
    actual = train_teacher_bundle(csi, teacher.patch_spec, config, seed=98, streaming_root=tmp_path)
    for name, value in expected.teacher.state_dict().items():
        torch.testing.assert_close(actual.teacher.state_dict()[name], value, rtol=0, atol=0)
    for name, value in expected.readout.state_dict().items():
        torch.testing.assert_close(actual.readout.state_dict()[name], value, rtol=0, atol=0)
    assert (tmp_path / "latent.npy").exists()
    def no_update(*_args, **_kwargs):
        raise AssertionError("Completed teacher and readout must resume without optimizer updates")
    with mock.patch.object(torch.optim.AdamW, "step", no_update):
        resumed = train_teacher_bundle(csi, teacher.patch_spec, config, seed=98, streaming_root=tmp_path)
    for name, value in expected.readout.state_dict().items():
        torch.testing.assert_close(resumed.readout.state_dict()[name], value, rtol=0, atol=0)


def test_all_six_table_commands_are_concrete_without_data_or_gpu(capsys):
    from formal_v2.paper_run import main
    assert main(["--dataset", "missing-on-local.npz", "--output", "server-only-output", "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert set(plan) == {"training", "probes", "controls", "maps", "risk", "tables"}
    assert len(plan["maps"]) == 5
    text = json.dumps(plan)
    assert "paper_maps" in text and "paper_controls" in text and "paper_risk" in text
    assert "qualification" not in text and "approval" not in text


def test_empty_export_is_six_incomplete_tables_not_fabricated_results(example, tmp_path):
    from formal_v2.paper_export import export
    _, _, config, path, *_ = example
    report = export(tmp_path, config, digest(path), fixture=True)
    assert len(report["tables"]) == 6
    assert all(row["status"] == "PARTIAL" for row in report["tables"].values())
    assert len(list((tmp_path / "fixture_tables").glob("*.tex"))) == 6
    assert report["scientific_use"] == "FORBIDDEN"


def test_first_party_risk_replay_uses_real_model_and_source_calibration(example, tmp_path):
    from formal_v2.paper_risk import main
    root, config_path, _, path, *_ = example
    profile = load_profile(PROJECT / "formal_v2/configs/paper_core.json")
    profile.update(probe_updates=[1], batch_rows=16, encoding_batch_rows=16)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    assert main(["--dataset", str(path), "--run-root", str(root), "--config", str(config_path),
                 "--profile", str(profile_path), "--device", "cpu", "--allow-fixture"]) == 0
    from formal_v2.paper_export import rows
    results = rows(root / "paper_risk/metrics.csv")
    assert {row["model_name"] for row in results} == {"joint", "d_only", "u_only"}
    assert {row["scope"] for row in results} == {"all_queries", "source_support"}
    assert all(row["scientific_use"] == "FORBIDDEN" for row in results)


def test_resource_and_shuffled_controls_train_localize_and_resume(example, tmp_path):
    from formal_v2.paper_train import main as train
    from formal_v2.paper_controls import main as controls
    _, config_path, _, path, *_ = example
    root = tmp_path / "run"
    assert train(["--dataset", str(path), "--config", str(config_path), "--output", str(root), "--allow-fixture", "--device", "cpu"]) == 0
    profile = load_profile(PROJECT / "formal_v2/configs/paper_core.json")
    profile.update(probe_updates=[1], batch_rows=16, encoding_batch_rows=16)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    args = ["--dataset", str(path), "--config", str(config_path), "--run-root", str(root), "--allow-fixture", "--device", "cpu",
            "--profile", str(profile_path), "--controls", "equal_flop_response", "generous_2x_concat", "shuffled_full", "original_recipe_full"]
    assert controls(args) == 0
    with mock.patch("formal_v2.formal_factorial._train_arm", side_effect=AssertionError("Completed control must not retrain")):
        assert controls(args) == 0
    assert len(list((root / "paper_controls").glob("*/*/complete.json"))) == 12


def test_six_table_export_complete_then_detects_missing_and_mismatched_comparison(example, tmp_path):
    # Synthetic exporter fixture, deliberately with Full worse than Response.
    # These values are never a model result and can only reach fixture_tables/.
    from formal_v2.formal_io import write_csv
    from formal_v2.formal_evidence import config_sha256
    from formal_v2.paper_export import export, rows
    from formal_v2.paper_controls import CONTROLS
    from formal_v2.paper_maps import METHODS
    from formal_v2.paper_core import SHORTCUTS, VARIANTS
    from formal_v2.external_adapters.wigatr_protocol import SIX_CONDITIONS
    _, _, config, path, *_ = example
    evidence = {"fixture": True, "scientific_use": "FORBIDDEN", "dataset_sha256": digest(path), "config_sha256": config_sha256(config)}
    arms = ("endpoint", "alignment", "response", "full")
    cities = ("target-boston", "target-seattle")
    def write(name, data):
        write_csv(tmp_path / name, [{**row, **evidence} for row in data])
    banks = []
    for seed in config["seeds"]:
        for arm in arms:
            for city in cities:
                for bank in range(2):
                    for budget in config["localization"]["label_budgets"]:
                        for draw in range(1 if budget == 0 else config["localization"]["label_draws"]):
                            error = (2 if arm == "full" else 1) + bank * .1 + seed * .001
                            banks.append({"seed": seed, "arm": arm, "city_id": city, "bank_id": f"{city}-{bank}",
                                "base_map_cluster_id": f"{city}-{bank}", "canonical_base_map_digest": f"{city}-{bank}",
                                "canonical_bank_digest": f"{city}-{bank}", "budget": budget, "draw": draw,
                                "support_position_ids": ";".join(str(i) for i in range(budget)), "query_unique_positions": 2,
                                "query_position_ids": "q1;q2", "median_error_m": error, "p90_error_m": error * 1.5,
                                "utility_neg_log_median": -float(np.log(error))})
    write("factorial/localization_per_bank.csv", banks)
    write("factorial/training_summary.csv", [{"seed": seed, "arm": arm, "parameters": 100, "measured_flops_per_step": 200, "elapsed_seconds": 1} for seed in config["seeds"] for arm in arms])
    base = [row for row in banks if row["budget"] == 0]
    write("core/evaluation/cgs_per_bank.csv", [{**row, "cgs_auroc": .5} for row in base])
    write("core/evaluation/compatibility_route_distributions.csv", [{**row, "overclassification_rate": .1} for row in base])
    fields = ["unified_response_probe_active_patch_nmse", "native_target_free_full_channel_nmse", "native_null_violation_rate", "native_no_action_full_channel_nmse", "native_action_swap_full_channel_nmse"]
    fields += ["probe_" + name + "_active_patch_nmse" for name in ("copy", *VARIANTS)]
    write("core/evaluation/response_per_bank.csv", [{**row, **dict.fromkeys(fields, .2)} for row in base])
    write("core/evaluation/alignment_shortcut_baselines.csv", [{**row, "baseline": name, "unseen_bank_auroc": .5} for row in base for name in SHORTCUTS])
    for seed in config["seeds"]:
        reference = [row for row in banks if row["seed"] == seed and row["arm"] == "full"]
        for name in METHODS:
            output = f"map_methods/{name}/{seed}"
            if name in {"Wi-GATr", "PMNet"}:
                write(output + "/localization_per_bank.csv", [{**row, "arm": name} for row in reference])
            write(output + "/six_condition_results.csv", [{**row, "unit_id": row["bank_id"] + ":q1", "model_name": name, "condition": condition, "localization_error_m": 1}
                for row in reference if row["budget"] == 0 for condition in SIX_CONDITIONS])
        for name in CONTROLS:
            output = f"paper_controls/{name}/{seed}"
            write(output + "/localization_per_bank.csv", [{**row, "arm": name} for row in reference])
            write(output + "/resources.csv", [{"seed": seed, "arm": name, "parameters": 100, "training_flops": 200, "parameter_ratio_to_full": 1, "training_flop_ratio_to_full": 1}])
            if name == "shuffled_full":
                write(output + "/alignment/cgs_per_bank.csv", [{**row, "cgs_auroc": .5} for row in reference if row["budget"] == 0])
                write(output + "/response/response_per_bank.csv", [{**row, "unified_response_probe_active_patch_nmse": .2} for row in reference if row["budget"] == 0])
    write("paper_risk/metrics.csv", [{"arm": arm, "city_id": city, "model_name": name, "scope": scope, "status": "COMPLETE", "training_seeds": 3,
          **dict.fromkeys(("aurc", "ece", "brier", "nll", "support_coverage", "risk_error_spearman"), .2)}
          for arm in arms for city in cities for name in ("joint", "d_only", "u_only") for scope in ("all_queries", "source_support")])
    result = export(tmp_path, config, digest(path), fixture=True)
    assert all(row["status"] == "COMPLETE" for row in result["tables"].values())
    displayed = rows(tmp_path / "fixture_tables/main2.csv")
    key = lambda arm: [float(row["value"]) for row in displayed if row["method"] == arm]
    assert all(full > response for full, response in zip(key("full"), key("response")))
    diagnostic_path = tmp_path / f"map_methods/PMNet/{config['seeds'][0]}/six_condition_results.csv"
    write_csv(diagnostic_path, rows(diagnostic_path)[1:])
    assert export(tmp_path, config, digest(path), fixture=True)["tables"]["s3"]["status"] == "PARTIAL"
    baseline_path = tmp_path / f"map_methods/PMNet/{config['seeds'][0]}/localization_per_bank.csv"
    changed = rows(baseline_path)
    changed[0]["query_position_ids"] = "wrong-query"
    write_csv(baseline_path, changed)
    with pytest.raises(ValueError, match="coverage differs"):
        export(tmp_path, config, digest(path), fixture=True)


def test_wigatr_training_handles_short_dataset_and_exact_optimizer_resume(tmp_path):
    from types import SimpleNamespace
    from formal_v2.external_adapters import wigatr_adapter as adapter
    from formal_v2 import baseline_resume
    class Batch:
        def __init__(self, x):
            self.x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
            self.y = self.x[:, 0] * 2
        @classmethod
        def from_data_list(cls, data):
            return cls(data)
        def to(self, device):
            self.x, self.y = self.x.to(device), self.y.to(device)
            return self
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            self.dropout = torch.nn.Dropout(.1)
            self.training_modes = []
        def forward(self, batch):
            self.training_modes.append(self.training)
            return self.linear(self.dropout(batch.x))[:, 0]
    runtime = {"torch": torch, "Batch": Batch, "DataLoader": lambda data, **_: [Batch(data)]}
    path = tmp_path / "synthetic-input"
    path.write_bytes(b"fixture, never scientific data")
    dataset = SimpleNamespace(source_path=path, is_fixture=True)
    config = {"training": {"seed": 11, "steps": 102, "batch_size": 4, "microbatch_size": 2,
        "learning_rate": .01, "weight_decay": .001, "clip_grad_norm": 1., "selection_every_steps": 1},
        "mesh": {"preprocessing": "synthetic test only"}}
    def fit(output):
        torch.manual_seed(19)
        model = Model()
        output.mkdir(exist_ok=True)
        adapter._fit_source_only_model(runtime, model, dataset, config, output, 1, 0., 1., None)
        return model
    with mock.patch.object(adapter, "_PowerDataset", return_value=[1., 2., 3.]):
        expected = fit(tmp_path / "full")
        save = baseline_resume.save
        def interrupt(*args, **kwargs):
            save(*args, **kwargs)
            if kwargs["step"] == 100:
                raise RuntimeError("injected interruption")
        with mock.patch.object(baseline_resume, "save", side_effect=interrupt), pytest.raises(RuntimeError, match="injected"):
            fit(tmp_path / "resume")
        actual = fit(tmp_path / "resume")
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)
    assert actual.training_modes == [True, True, False, True, True, False]


def test_controlled_baseline_normalization_and_real_training(example, tmp_path):
    from formal_v2.external_adapters import controlled_map_adapter as adapter
    _, _, _, _, data, *_ = example
    normalizer = adapter._fit_data_normalizer(data)
    scenes = data.indices_for_role("source_encoder_train")
    np.testing.assert_allclose(normalizer["csi_mean"], data.csi_clean[scenes].mean(axis=(0, 1, 2)))
    np.testing.assert_allclose(normalizer["csi_std"], data.csi_clean[scenes].std(axis=(0, 1, 2)))
    config = json.loads((PROJECT / "formal_v2/configs/sigmap_controlled_v1.json").read_text())
    config["training"].update(steps=2, batch_size=4, microbatch_size=2, selection_every_steps=1)
    config["model"]["hidden_dim"] = 8
    torch.manual_seed(14)
    model, metadata = adapter._build_model(config, data)
    path, result = adapter._train(model, config, data, normalizer, tmp_path, metadata)
    assert path.exists() and result["steps"] == 2
    prediction = adapter._sigmap_predict(model, data, 14, data.csi[14, 0, 0], data.maps[14, 0], normalizer)
    assert prediction.shape == (2,) and np.isfinite(prediction).all()


def test_map_stage_real_sigmap_six_conditions_and_resume(example, tmp_path):
    from formal_v2 import paper_maps
    from formal_v2.paper_export import rows
    root, config_path, config, path, *_ = example
    baseline = json.loads((PROJECT / "formal_v2/configs/sigmap_controlled_v1.json").read_text())
    baseline["training"].update(steps=2, batch_size=4, microbatch_size=2, selection_every_steps=1)
    baseline["model"]["hidden_dim"] = 8
    baseline_path = tmp_path / "sigmap-smoke.json"
    baseline_path.write_text(json.dumps(baseline))
    args = ["--dataset", str(path), "--run-root", str(root), "--config", str(config_path), "--method", "SigMap", "--device", "cpu", "--allow-fixture"]
    with mock.patch.dict(paper_maps.METHODS, {"SigMap": str(baseline_path)}):
        assert paper_maps.main(args) == 0
        with mock.patch.object(paper_maps, "MapMethod", side_effect=AssertionError("Completed map method must not refit")):
            assert paper_maps.main(args) == 0
    for seed in config["seeds"]:
        values = rows(root / f"map_methods/SigMap/{seed}/six_condition_results.csv")
        assert {row["condition"] for row in values} == set(paper_maps.SIX_CONDITIONS)
        assert all(row["observations"] == "measured_csi_same_for_all_methods" for row in values)


def test_shortcuts_match_original_features_without_calling_an_encoder(example):
    from formal_v2 import formal_evaluation as evaluation
    from formal_v2.paper_core import SHORTCUTS
    _, _, config, _, data, teacher, _, route_norm, norm, model, _ = example
    scene = int(data.indices_for_role("source_probe_train")[0])
    original = evaluation._compatibility_dataset(model, data, teacher, config, norm, [scene], route_normalization=route_norm)
    for name, (field, _) in SHORTCUTS.items():
        raw = feature_scene("alignment:" + name, None, data, teacher, config, norm, scene, route_norm, 16, train=True)
        np.testing.assert_array_equal(raw["features"], original[field])
        np.testing.assert_array_equal(raw["labels"], original["labels"])
        np.testing.assert_array_equal(raw["routes"], original["routes"])


def test_input_only_control_is_trained_once_for_four_arm_comparisons(example, tmp_path):
    from formal_v2 import paper_core
    root, config_path, config, path, _, _, teacher_path, _, _, _, upstream = example
    independent = tmp_path / "factorial"
    independent.mkdir()
    payload = torch.load(upstream / "full.pt", weights_only=True)
    checkpoints = []
    for arm in ("endpoint", "alignment", "response", "full"):
        target = independent / (arm + ".pt")
        torch.save({**payload, "arm": arm}, target)
        checkpoints.append({"seed": payload["seed"], "arm": arm, "path": target.name, "sha256": digest(target)})
    (independent / "checkpoint_index.json").write_text(json.dumps({"checkpoints": checkpoints}))
    profile = load_profile(PROJECT / "formal_v2/configs/paper_core.json")
    profile.update(tasks=["alignment:map_only"], probe_updates=[1], batch_rows=16, encoding_batch_rows=16)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    with mock.patch.object(paper_core, "select_probe", wraps=paper_core.select_probe) as fit:
        assert paper_core.main(["--dataset", str(path), "--factorial-root", str(independent), "--teacher", str(teacher_path),
            "--output", str(tmp_path / "core"), "--config", str(config_path), "--profile", str(profile_path), "--allow-fixture", "--device", "cpu"]) == 0
        assert fit.call_count == 1
