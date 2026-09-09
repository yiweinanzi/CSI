"""Source-only research safety tests. Synthetic inputs never count as formal units."""

import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from formal_v2 import formal_factorial as factorial
from formal_v2.formal_cli import build_parser
from formal_v2.formal_config import ARMS, load_formal_config, validate_formal_config
from formal_v2.formal_dataset import FormalDataset, PHYSICAL_POSITION_ATOL_M
from formal_v2.formal_evaluation_resume import write_atomic_json
from formal_v2.formal_fixture import write_nonscientific_fixture
from formal_v2.formal_io import read_strict_json, sha256_file, write_csv
from formal_v2.formal_localization import HeteroscedasticPositionHead
from formal_v2.formal_model import portable_state_dict
from formal_v2.formal_protocol import PatchSpec
from formal_v2.formal_routing import fit_route_normalization
from formal_v2.formal_source_guard import SourceOnlyDataset
from formal_v2.formal_source_method_pilot import (
    PilotProgress, ProgressTrace, ResourceGuard, calibrate_scales, json_hash, load_completed_arm,
    preserve_initial_state, recipe_configs, source_validation_split, state_hash,
    validate_source_model, write_once,
)
from formal_v2.formal_teacher import train_teacher_bundle
from formal_v2.formal_training_resume import _atomic_torch_save


ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return read_strict_json(ROOT / "configs/source_method_pilot_20260909.json")


def small_dataset():
    values = np.stack((np.arange(8), np.zeros(8)), axis=1).astype(np.float64)
    return SimpleNamespace(
        scene_count=4, position_count=8,
        indices_for_role=lambda role: np.array([0]) if role == "source_encoder_train" else np.array([1, 2]),
        positions=np.stack((values, values, values, values + 100)),
        position_ids=np.array([["id-" + str(index) for index in range(8)]] * 4, dtype="U32"),
        city_ids=np.array(["A", "A", "A", "TARGET"]),
        world_bits=np.arange(4), map_channel_names=np.array(["occupancy"]),
        unclassified=np.arange(4), base_map_cluster_ids=np.array(["a", "b", "c", "d"]),
    )


def test_guard_denies_target_roles_slices_and_unsliced_conversion():
    guarded = SourceOnlyDataset(small_dataset())
    np.testing.assert_array_equal(guarded.positions[[1, 2]], small_dataset().positions[[1, 2]])
    assert guarded.observed_scenes == {1, 2}
    for role in ("target", "source_probe_train", "external_validation"):
        with pytest.raises(RuntimeError, match="Forbidden research role"):
            guarded.indices_for_role(role)
    for key in (3, -1, slice(None), [0, 3], Ellipsis, None, True):
        with pytest.raises(RuntimeError):
            guarded.positions[key]
    with pytest.raises(RuntimeError, match="unsliced"):
        np.asarray(guarded.positions)
    with pytest.raises(ValueError):
        guarded.positions[0][0] = 42
    with pytest.raises(RuntimeError, match="Unclassified"):
        guarded.unclassified
    # Global arrays must not be mistaken for scene arrays when dimensions happen to match.
    np.testing.assert_array_equal(guarded.world_bits, np.arange(4))


def test_guard_denies_overlapping_source_partitions():
    raw = small_dataset()
    raw.indices_for_role = lambda role: np.array([0])
    with pytest.raises(RuntimeError, match="overlap"):
        SourceOnlyDataset(raw)


def test_source_split_excludes_ids_and_coordinate_siblings_across_banks():
    raw = small_dataset()
    raw.position_ids[2] = ["alias-" + str(index) for index in range(8)]
    guarded = SourceOnlyDataset(raw)
    config = protocol()
    config["validation"]["label_budgets"] = [0, 2, 4]
    split = source_validation_split(guarded, config)
    assert split == source_validation_split(guarded, config)
    assert set(split) == {"A"}
    pool = split["A"]["support_pool"]
    assert len(pool) == 4
    coords = np.stack([guarded.positions[s, p] for s, p in pool])
    ids = {str(guarded.position_ids[s, p]) for s, p in pool}
    for scene, positions in split["A"]["validation_rows"].items():
        assert len(positions) == 4
        for position in positions:
            assert str(guarded.position_ids[int(scene), position]) not in ids
            assert not np.any(np.all(np.abs(coords - guarded.positions[int(scene), position]) <= PHYSICAL_POSITION_ATOL_M, axis=1))
    assert guarded.observed_scenes == {1, 2}


def test_source_split_rejects_inconsistent_ids_and_insufficient_capacity():
    raw = small_dataset()
    raw.positions[2, 0, 0] += 1
    config = protocol()
    config["validation"]["label_budgets"] = [0, 2, 4]
    with pytest.raises(RuntimeError, match="inconsistent coordinates"):
        source_validation_split(SourceOnlyDataset(raw), config)
    config["validation"]["label_budgets"] = [0, 128]
    with pytest.raises(RuntimeError, match="capacity"):
        source_validation_split(SourceOnlyDataset(small_dataset()), config)


def test_two_recipes_change_only_the_existing_margin_switch():
    base = load_formal_config(ROOT / "configs/formal_v2.json")
    configs = recipe_configs(base, protocol())
    assert configs["fixed_margin"] == base
    candidate = copy.deepcopy(configs["effect_aware_margin"])
    candidate["factorial"]["alignment_primary_margin"] = "fixed"
    assert candidate == base
    assert configs["fixed_margin"] is not base
    with pytest.raises(ValueError, match="V6 P0 requires"):
        validate_formal_config(configs["effect_aware_margin"])


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(sota_ready=True),
    lambda p: p.update(scientific_use="CLAIM"),
    lambda p: p.update(steps_per_arm=127),
    lambda p: p.update(arms=["full"]),
    lambda p: p.update(selection_role="target"),
    lambda p: p["allowed_roles"].append("target"),
    lambda p: p["validation"].update(label_budgets=[8]),
    lambda p: p["validation"].update(draws=1),
])
def test_protocol_mutations_cannot_relax_safety(mutation):
    config = protocol()
    mutation(config)
    with pytest.raises(ValueError):
        recipe_configs(load_formal_config(ROOT / "configs/formal_v2.json"), config)


def test_initial_state_is_actual_shared_state_and_never_overwritten(tmp_path):
    state = {"weight": torch.arange(8, dtype=torch.float32).reshape(2, 4)}
    path = tmp_path / "init.pt"
    digest = preserve_initial_state(path, state)
    file_digest = sha256_file(path)
    assert preserve_initial_state(path, copy.deepcopy(state)) == digest
    assert sha256_file(path) == file_digest
    with pytest.raises(RuntimeError, match="initialization changed"):
        preserve_initial_state(path, {"weight": state["weight"] + 1})
    assert sha256_file(path) == file_digest
    write_once(tmp_path / "identity.json", {"value": 1})
    with pytest.raises(RuntimeError, match="identity changed"):
        write_once(tmp_path / "identity.json", {"value": 2})


def test_completed_arm_requires_authenticated_full_schedule_and_artifacts(tmp_path):
    context = {"steps": 2, "identity": "synthetic smoke"}
    assert load_completed_arm(tmp_path, "full", context) is None
    state = {"weight": torch.ones(2)}
    state_path = tmp_path / "full.pt"
    _atomic_torch_save(state_path, state)
    rows = [{"median_error_m": 1.0, "scientific_use": "NON_CLAIM"}]
    csv_path = tmp_path / "full_source_validation.csv"
    write_csv(csv_path, rows)
    record = {
        "arm": "full", "context_sha256": json_hash(context),
        "training": {"steps": 2}, "loss_trace": [{"step": 1}, {"step": 2}],
        "state_sha256": sha256_file(state_path), "state_value_sha256": state_hash(state),
        "validation_csv_sha256": sha256_file(csv_path), "source_rows": rows,
        "scientific_use": "NON_CLAIM", "formal_units_added": 0,
    }
    record["record_sha256"] = json_hash(record)
    write_atomic_json(tmp_path / "full_result.json", record)
    assert load_completed_arm(tmp_path, "full", context) == record
    with pytest.raises(RuntimeError, match="different identity"):
        load_completed_arm(tmp_path, "full", {**context, "steps": 3})
    mutated = copy.deepcopy(record)
    mutated["source_rows"][0]["median_error_m"] = 0.01
    write_atomic_json(tmp_path / "full_result.json", mutated)
    with pytest.raises(RuntimeError, match="receipt changed"):
        load_completed_arm(tmp_path, "full", context)
    write_atomic_json(tmp_path / "full_result.json", record)
    write_csv(csv_path, [{"median_error_m": 2.0}])
    with pytest.raises(RuntimeError, match="artifact changed"):
        load_completed_arm(tmp_path, "full", context)


def test_resource_guard_accounts_for_cgroup_headroom_and_explicit_pause(tmp_path, monkeypatch):
    guard = ResourceGuard(protocol(), tmp_path)
    monkeypatch.setattr(guard, "memory_snapshot", lambda: {"host_available_bytes": 500 * 1024**3, "peak_rss_bytes": 1024, "cgroup_available_bytes": 32 * 1024**3})
    assert guard.is_set()
    assert "floor" in guard.reason
    guard = ResourceGuard(protocol(), tmp_path)
    (tmp_path / "PAUSE_REQUESTED").touch()
    assert guard.is_set()
    assert "Operator" in guard.reason


def test_cli_exposes_only_explicit_research_inputs():
    names = ("dataset", "config", "protocol", "output", "origin-server", "upstream-root")
    args = build_parser().parse_args(["run-source-method-pilot", *[entry for name in names for entry in ("--" + name, name)]])
    assert args.command == "run-source-method-pilot"
    assert args.origin_server == "origin-server"


def test_cli_dispatches_to_the_separate_source_runner(monkeypatch):
    from formal_v2 import formal_cli, formal_source_method_pilot

    observed = []
    monkeypatch.setattr(formal_source_method_pilot, "main", lambda argv: observed.append(argv) or 73)
    arguments = [entry for name in ("dataset", "config", "protocol", "output", "origin-server", "upstream-root") for entry in ("--" + name, name)]
    assert formal_cli.main(["run-source-method-pilot", *arguments]) == 73
    assert observed == [arguments]


@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_synthetic_source_integration_shared_control_training_and_resume(tmp_path, device):
    if device.startswith("cuda"):
        if torch.cuda.device_count() < 2:
            pytest.skip("Dedicated GPU 1 is required for this explicitly synthetic smoke")
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    config = load_formal_config(ROOT / "configs/formal_v2_smoke.json")
    raw = FormalDataset.load(write_nonscientific_fixture(tmp_path / "synthetic.npz"))
    guarded = SourceOnlyDataset(raw)
    scenes = guarded.indices_for_role("source_encoder_train")
    spec = PatchSpec.from_metadata(guarded.metadata)
    teacher = train_teacher_bundle(guarded.csi[scenes], spec, config, seed=12031)
    route_norm = fit_route_normalization(guarded, teacher)
    norm = factorial._training_normalization(guarded, scenes, route_norm, spec)
    train = factorial._build_corpus(guarded, scenes, teacher, config, route_norm, norm)
    selection = factorial._build_corpus(guarded, guarded.indices_for_role("source_method_selection"), teacher, config, route_norm, norm)
    settings = protocol()
    settings["device"] = device
    settings["validation"]["label_budgets"] = config["localization"]["label_budgets"]
    progress = PilotProgress(tmp_path, settings)
    guard = SimpleNamespace(is_set=lambda: False)
    calibration = calibrate_scales(config, train, selection, settings, tmp_path, progress, guard, {"test": "synthetic"})
    assert calibrate_scales(config, train, selection, settings, tmp_path, progress, guard, {"test": "synthetic"}) == calibration
    with pytest.raises(RuntimeError, match="identity"):
        calibrate_scales(config, train, selection, settings, tmp_path, progress, guard, {"test": "changed"})
    template = factorial._new_model(config, train, 37, device="cpu")
    initial = state_hash(template.state_dict())
    plan = factorial._make_plan(train, config["factorial"]["batch_size"], 37, 0)
    assert json_hash(asdict(plan)) == json_hash(asdict(factorial._make_plan(train, config["factorial"]["batch_size"], 37, 0)))
    states = {}
    for recipe, margin in (("fixed_margin", "fixed"), ("effect_aware_margin", "effect_aware")):
        config["factorial"]["alignment_primary_margin"] = margin
        pilot = {"alignment_scale": calibration["alignment_scales"][recipe], "response_scale": calibration["response_scale"], "alignment_null_tolerance": calibration["alignment_null_tolerance"]}
        for arm in ARMS:
            model = copy.deepcopy(template).to(device)
            assert state_hash(model.state_dict()) == initial
            trace = ProgressTrace(progress, recipe, arm, model) if device.startswith("cuda") else []
            checkpoint = tmp_path / (recipe + "-" + arm)
            context = {"test": "synthetic", "recipe": recipe}
            model, training = factorial._train_arm(config, train, 37, arm, pilot, step_count=2, prepared_model=model, loss_trace=trace, checkpoint_path=checkpoint, checkpoint_context=context, checkpoint_interval_steps=1)
            states[recipe, arm] = state_hash(model.state_dict())
            assert states[recipe, arm] != initial
            assert training["steps"] == 2 and len(trace) == 2
            assert training["execution_device"] == device
            resumed, receipt = factorial._train_arm(config, train, 37, arm, pilot, step_count=2, prepared_model=copy.deepcopy(template).to(device), loss_trace=[], checkpoint_path=checkpoint, checkpoint_context=context, checkpoint_interval_steps=1)
            assert state_hash(resumed.state_dict()) == states[recipe, arm]
            assert {key: value for key, value in receipt.items() if key != "elapsed_seconds"} == {key: value for key, value in training.items() if key != "elapsed_seconds"}
            # The durable receipt excludes the final checkpoint's own write latency.
            assert 0 <= receipt["elapsed_seconds"] <= training["elapsed_seconds"]
    assert all(states["fixed_margin", arm] == states["effect_aware_margin", arm] for arm in ("endpoint", "response"))
    split = source_validation_split(guarded, settings)
    head = HeteroscedasticPositionHead(config["model"]["state_dim"], max(8, config["model"]["hidden_dim"] // 2))
    head_sha = state_hash(head.state_dict())
    rows = validate_source_model(model, guarded, norm, spec, config, 37, split, head)
    assert state_hash(head.state_dict()) == head_sha
    assert rows and all(np.isfinite(row["median_error_m"]) for row in rows)
    assert {row["city"] for row in rows} == set(split)
    assert {row["k"] for row in rows} == set(config["localization"]["label_budgets"])
    assert guarded.observed_scenes <= guarded.allowed_scenes
