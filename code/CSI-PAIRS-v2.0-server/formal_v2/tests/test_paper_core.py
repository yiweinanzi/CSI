import copy
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from formal_v2 import formal_factorial as factorial
from formal_v2 import formal_evaluation as evaluation
from formal_v2.formal_config import load_formal_config
from formal_v2.formal_dataset import FormalDataset, FormalDatasetError, _channel_std_bounded
from formal_v2.formal_fixture import write_nonscientific_fixture
from formal_v2.formal_evidence import config_sha256
from formal_v2.formal_model import portable_state_dict
from formal_v2.formal_protocol import PatchSpec
from formal_v2.formal_routing import fit_route_normalization
from formal_v2.formal_teacher import train_teacher_bundle, save_teacher_bundle
from formal_v2.paper_arrays import MappedNPZ
from formal_v2.paper_core import main, feature_scene, source_arrays, load_profile, frozen_model
from formal_v2.paper_minibatch import fit_candidate
from formal_v2.paper_suite import digest, PROJECT


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("core-protocol-synthetic")
    config_path = PROJECT / "formal_v2/configs/formal_v2_smoke.json"
    config = load_formal_config(config_path)
    path = write_nonscientific_fixture(root / "fixture.npz")
    data = FormalDataset.load(path)
    scenes = data.indices_for_role("source_encoder_train")
    spec = PatchSpec.from_metadata(data.metadata)
    teacher = train_teacher_bundle(data.csi[scenes], spec, config, seed=12031)
    teacher_path = root / "teacher.pt"
    save_teacher_bundle(teacher_path, teacher, config, seed=12031)
    route_norm = fit_route_normalization(data, teacher)
    norm = factorial._training_normalization(data, scenes, route_norm, spec)
    corpus = factorial._build_corpus(data, scenes, teacher, config, route_norm, norm)
    seed = config["seeds"][0]
    model = factorial._new_model(config, corpus, seed, device="cpu")
    model.eval()
    upstream = root / "factorial"
    upstream.mkdir()
    checkpoint = upstream / "full.pt"
    payload = {"schema_version": "csi-pairs-formal-checkpoint-v2.1-v6", "dataset_sha256": digest(path),
               "config_sha256": config_sha256(config), "teacher_checkpoint_sha256": digest(teacher_path),
               "model_spec": factorial._model_spec(config, corpus), "state_dict": portable_state_dict(model),
               "normalization": factorial._normalization_record(norm), "arm": "full", "seed": seed, "fixture": True}
    torch.save(payload, checkpoint)
    index = {"checkpoints": [{"path": "full.pt", "sha256": digest(checkpoint), "arm": "full", "seed": seed}]}
    (upstream / "checkpoint_index.json").write_text(json.dumps(index))
    return root, config_path, config, path, data, teacher, teacher_path, route_norm, norm, model, upstream


def test_mapped_npz_preserves_dataset_values_and_detects_corruption(example, tmp_path):
    _, _, _, path, data, *_ = example
    mapped = FormalDataset.load(path, array_cache=tmp_path)
    for field in ("csi_repeat", "csi_clean", "maps", "positions", "path_power", "noop_maps"):
        actual = getattr(mapped, field)
        np.testing.assert_array_equal(actual, getattr(data, field))
        assert isinstance(actual.base, np.memmap)
    cache = MappedNPZ(path, tmp_path)
    with (cache.root / "maps.npy").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="cache changed"):
        MappedNPZ(path, tmp_path)


def test_core_feature_values_match_legacy_without_control_arrays(example):
    _, _, config, _, data, teacher, _, route_norm, norm, model, _ = example
    scene = int(data.indices_for_role("source_probe_train")[0])
    for task in ("alignment", "response"):
        core = feature_scene(task, model, data, teacher, config, norm, scene, route_norm, 16, train=True)
        legacy = (evaluation._compatibility_dataset if task == "alignment" else evaluation._response_probe_dataset)(model, data, teacher, config, norm, [scene], route_normalization=route_norm, batch_size=16, **({} if task == "alignment" else {"active_only": False}))
        for key in core:
            if isinstance(core[key], np.ndarray):
                np.testing.assert_array_equal(core[key], legacy[key], err_msg=key)
        assert not any("shortcut_features" in name for name in core)
        assert "without_map_features" not in core
        if task == "response":
            assert "oracle_x_features" not in core


def test_minibatch_is_one_batch_per_step_and_resume_is_exact(tmp_path):
    x = np.random.default_rng(5).normal(size=(101, 4)).astype("float32")
    y = (np.arange(101) % 2).astype("float32")
    options = dict(family="linear", hidden=8, seed=9, steps=30, batch_rows=7, learning_rate=.01, device="cpu")
    events = []
    expected = fit_candidate(x, y, None, **options, callback=events.append)
    assert sum(row["batch_rows"] for row in events) == 202  # Two full shuffled epochs, not 30 epochs.
    def interrupt(event):
        if event["step"] == 25:
            raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        fit_candidate(x, y, None, **options, checkpoint=tmp_path / "resume.pt", callback=interrupt)
    resumed = []
    actual = fit_candidate(x, y, None, **options, checkpoint=tmp_path / "resume.pt", callback=resumed.append)
    assert resumed[0]["step"] == 26
    for key, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[key])


def test_bounded_channel_std_matches_population_std():
    data = np.random.default_rng(10).normal(size=(3, 4, 11, 2, 7))
    np.testing.assert_allclose(_channel_std_bounded(data, rows=13), np.std(data, axis=(0, 1, 2, 3)), rtol=1e-14)
    assert np.array_equal(_channel_std_bounded(np.ones_like(data), rows=13), np.zeros(7))


def test_source_feature_builder_rejects_target_role(example, tmp_path):
    _, _, config, _, data, teacher, _, route_norm, norm, model, _ = example
    with pytest.raises(ValueError, match="cannot consume target"):
        source_arrays("alignment", "target", tmp_path, model, data, teacher, config, norm, route_norm, {})


def test_core_cli_runs_independent_tasks_without_heavy_gates(example, tmp_path):
    _, config_path, _, path, _, _, teacher_path, _, _, _, upstream = example
    profile = load_profile(PROJECT / "formal_v2/configs/paper_core.json")
    profile.update(probe_updates=[1, 2], batch_rows=16, encoding_batch_rows=16)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    output = tmp_path / "output"
    arguments = ["--dataset", str(path), "--factorial-root", str(upstream), "--teacher", str(teacher_path),
                 "--config", str(config_path), "--profile", str(profile_path), "--output", str(output), "--device", "cpu", "--allow-fixture"]
    with mock.patch("formal_v2.formal_evidence.evidence_context", side_effect=AssertionError("Heavy provenance must not run")):
        assert main(arguments + ["--tasks", "alignment"]) == 0
        assert (output / "evaluation/cgs_per_bank.csv").is_file()
        assert not (output / "evaluation/response_per_bank.csv").exists()
        assert main(arguments + ["--tasks", "response", "native"]) == 0
        assert main(arguments) == 0
    status = json.loads((output / "status.json").read_text())
    assert all(status["tasks"][name] == {"complete": 1, "total": 1} for name in ("alignment", "response", "native"))
    assert all(row["complete"] == 0 for name, row in status["tasks"].items() if ":" in name)
    import csv
    rows = list(csv.DictReader((output / "evaluation/response_per_bank.csv").open()))
    assert rows and all(r["scientific_use"] == "FORBIDDEN" for r in rows)
    assert all(r["native_target_free_full_channel_nmse"] != "" for r in rows)


def test_light_training_runs_all_arms_and_resumes_without_approval(example, tmp_path):
    from formal_v2.paper_train import main as train_main
    _, config_path, config, path, *_ = example
    output = tmp_path / "training"
    arguments = ["--dataset", str(path), "--config", str(config_path), "--output", str(output), "--device", "cpu", "--allow-fixture"]
    with mock.patch("formal_v2.formal_evidence.evidence_context", side_effect=AssertionError("No heavyweight training provenance")):
        assert train_main(arguments) == 0
        with mock.patch("formal_v2.formal_factorial._train_arm", side_effect=AssertionError("Completed arms must not retrain")):
            assert train_main(arguments + ["--localize"]) == 0
    index = json.loads((output / "factorial/checkpoint_index.json").read_text())
    assert {(r["seed"], r["arm"]) for r in index["checkpoints"]} == {(s, a) for s in config["seeds"] for a in ("endpoint", "alignment", "response", "full")}
    for row in index["checkpoints"]:
        checkpoint = output / "factorial" / row["path"]
        frozen_model(checkpoint, row["sha256"], digest(output / "teacher.pt"), digest(path), config, "cpu")
    assert (output / "factorial/localization_summary.csv").is_file()
    status = json.loads((output / "training_status.json").read_text())
    assert status["scientific_use"] == "FORBIDDEN"
    assert status["localization_complete"] is True
