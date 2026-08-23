from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from formal_v2.formal_action_inverse_response import (
    PHYSICAL_RESPONSE_CONTRACT,
    PhysicalNullGate,
    PhysicalResponseModel,
    _fit_shared_response_scale,
    _gate_features,
    build_physical_response_rows,
    calibrate_gate_thresholds,
    load_physical_response_checkpoint,
    load_physical_response_config,
    physical_action_swap_predictions,
    predict_physical_response,
    save_physical_response_checkpoint,
)
from formal_v2.formal_evidence import QUALIFICATION_SCHEMA, require_formal_qualification
from formal_v2.formal_io import sha256_file
from formal_v2.formal_protocol import PatchSpec
from formal_v2.formal_routing import PRIMARY_ROUTE_CONTRACT


def test_physical_response_config_freezes_full_selection_coverage() -> None:
    config = load_physical_response_config()
    assert config["world_scope"] == "all"
    assert config["selection_position_stride"] == 1
    assert config["fit_position_stride"] == 16
    assert config["gate_ensemble_size"] == 3
    assert config["maximum_fit_bank_group_false_positive_rate"] == 0.05


def test_physical_gate_threshold_controls_every_fit_bank_and_group() -> None:
    probability = np.asarray(
        [
            0.01,
            0.02,
            0.03,
            0.04,
            0.90,
            0.01,
            0.02,
            0.03,
            0.70,
            0.80,
            0.01,
            0.02,
            0.03,
            0.04,
            0.60,
            0.01,
            0.02,
            0.03,
            0.50,
            0.55,
        ]
    )
    labels = np.zeros(probability.shape, dtype=np.float32)
    groups = np.asarray((["a"] * 5 + ["b"] * 5) * 2)
    banks = np.asarray(["bank-1"] * 10 + ["bank-2"] * 10)

    thresholds, audit = calibrate_gate_thresholds(
        probability,
        labels,
        groups,
        banks,
        maximum_false_positive_rate=0.20,
    )

    assert thresholds["a"] > 0.90
    assert thresholds["b"] > 0.80
    assert audit["maximum_fit_bank_group_false_positive_rate"] <= 0.20
    for bank in ("bank-1", "bank-2"):
        for group in ("a", "b"):
            selected = (banks == bank) & (groups == group)
            assert np.mean(probability[selected] >= thresholds[group]) <= 0.20


def test_physical_response_checkpoint_round_trip(tmp_path: Path) -> None:
    gates = [PhysicalNullGate(3), PhysicalNullGate(3)]
    with torch.no_grad():
        for index, gate in enumerate(gates):
            for parameter in gate.parameters():
                parameter.fill_(0.01 * (index + 1))
    model = PhysicalResponseModel(
        coefficient_calibration={(1, 4): np.asarray([1.0 + 2.0j])},
        gates=gates,
        feature_mean=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        feature_scale=np.asarray([2.0, 3.0, 4.0], dtype=np.float32),
        thresholds={"q0:m1-4": 0.75},
        response_scale=1.375,
        metadata={"contract": PHYSICAL_RESPONSE_CONTRACT},
    )
    checkpoint = tmp_path / "physical.pt"
    bindings = {"dataset_sha256": "a" * 64, "config_sha256": "b" * 64}

    save_physical_response_checkpoint(checkpoint, model, bindings)
    loaded, loaded_bindings = load_physical_response_checkpoint(checkpoint)

    assert loaded_bindings == bindings
    assert loaded.thresholds == model.thresholds
    assert loaded.response_scale == model.response_scale
    np.testing.assert_array_equal(loaded.feature_mean, model.feature_mean)
    np.testing.assert_array_equal(loaded.feature_scale, model.feature_scale)
    np.testing.assert_array_equal(
        loaded.coefficient_calibration[(1, 4)],
        model.coefficient_calibration[(1, 4)],
    )
    for expected, observed in zip(model.gates, loaded.gates, strict=True):
        for name, value in expected.state_dict().items():
            torch.testing.assert_close(observed.state_dict()[name], value)


def test_physical_action_swap_uses_same_source_and_position() -> None:
    world_bits = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)

    class Dataset:
        pass

    dataset = Dataset()
    dataset.world_bits = world_bits
    rows = []
    for bit, target in ((0, 1), (1, 2)):
        rows.append(
            {
                "key": (0, 0, target, bit, 7, 3),
                "scene": 0,
                "source_world": 0,
                "target_world": target,
                "bit": bit,
                "position": 7,
                "query": 3,
            }
        )
    prediction = np.asarray([[1.0, 2.0], [5.0, 6.0]], dtype=np.float32)
    wrong_action_plan = {
        "selections": {
            (0, 0, 0, 7): (None, "exact", 2),
            (0, 0, 1, 7): (None, "exact", 1),
        }
    }

    swapped, status = physical_action_swap_predictions(
        rows, prediction, dataset, wrong_action_plan
    )

    np.testing.assert_array_equal(swapped, prediction[::-1])
    np.testing.assert_array_equal(status, np.asarray(["exact", "exact"]))


class _SourceOnlyCSI:
    def __getitem__(self, key):
        _scene, world, _position = key
        if int(world) != 0:
            raise AssertionError("oracle response read target CSI")
        return np.zeros(64, dtype=np.float64)


def _oracle_test_rows(*, receiver_mode: str, oracle_path) -> list[dict]:
    spec = PatchSpec(2, 16, 2, 1, 2)
    receiver = np.asarray([12.5, -4.0], dtype=np.float64)
    dataset = SimpleNamespace(
        metadata={
            "representation": {
                "antenna_count": 2,
                "subcarrier_count": 16,
                "patch_complex_size": 2,
                "patch_antenna_size": 1,
                "patch_subcarrier_size": 2,
            }
        },
        position_count=1,
        positions=receiver.reshape(1, 1, 2),
        csi=_SourceOnlyCSI(),
        maps=np.zeros((1, 2, 3, 4, 4), dtype=np.float64),
        bank_ids=np.asarray(["bank-0"]),
        directed_edges=lambda _scene: (
            SimpleNamespace(source_world=0, target_world=1, bit_index=0),
        ),
    )
    path = SimpleNamespace(
        features=np.arange(1.0, 8.0),
        basis=np.ones((spec.antennas, spec.subcarriers), dtype=np.complex128),
    )
    dictionary = (
        receiver.reshape(1, 2),
        np.ones((1, spec.complex_values), dtype=np.complex128),
        np.ones((1, spec.complex_values), dtype=np.complex128),
    )
    scene_dictionaries = {0: {"rectangle": (0.0, 1.0, 2.0, 3.0), "dictionary": dictionary}}
    candidate_config = {
        "radio": {
            "transmitter_z_m": 10.0,
            "receiver_z_m": 1.5,
            "carrier_frequency_hz": 3.5e9,
            "subcarrier_spacing_hz": 30e3,
        },
        "interventions": {"height_m": 2.0},
        "map": {"minimum_reflection_incidence_cosine": 0.01},
    }
    route_lookup = {(0, 0, 1, 0, query): 2 for query in range(spec.patch_count)}
    with (
        mock.patch(
            "formal_v2.formal_action_inverse_response._scene_dictionaries",
            return_value=scene_dictionaries,
        ),
        mock.patch(
            "formal_v2.formal_action_inverse_response._oracle_receiver_dictionary",
            return_value=(
                None
                if oracle_path is None
                else (dictionary[0], dictionary[1], dictionary[2], path)
            ),
        ) as oracle_dictionary,
        mock.patch(
            "formal_v2.formal_action_inverse_response.material_direction",
            return_value=(1, 4),
        ),
        mock.patch(
            "formal_v2.formal_action_inverse_response.reflection_path",
            return_value=path,
        ),
    ):
        rows = build_physical_response_rows(
            dataset,
            (0,),
            {"model": {"mask_bank_seed": 17}},
            candidate_config,
            np.ones((spec.patch_count, spec.patch_dim), dtype=np.float64),
            route_lookup,
            position_stride=1,
            grid_spacing_m=4.0,
            decomposition_ridge=1e-6,
            include_targets=False,
            receiver_mode=receiver_mode,
        )
    if receiver_mode == "oracle":
        np.testing.assert_array_equal(oracle_dictionary.call_args.args[0], receiver)
    return rows


def test_oracle_rows_use_exact_receiver_without_reading_target_csi() -> None:
    rows = _oracle_test_rows(receiver_mode="oracle", oracle_path=True)

    assert len(rows) == 16
    assert {row["receiver_position_source"] for row in rows} == {"oracle"}
    for row in rows:
        np.testing.assert_array_equal(row["inferred_xy"], np.asarray([12.5, -4.0]))


def test_oracle_and_inferred_row_keys_align() -> None:
    inferred = _oracle_test_rows(receiver_mode="inferred", oracle_path=True)
    oracle = _oracle_test_rows(receiver_mode="oracle", oracle_path=True)

    assert [row["key"] for row in inferred] == [row["key"] for row in oracle]


def test_no_path_oracle_rows_produce_finite_zero_predictions() -> None:
    rows = _oracle_test_rows(receiver_mode="oracle", oracle_path=None)
    spec = PatchSpec(2, 16, 2, 1, 2)
    patch_scale = np.ones((spec.patch_count, spec.patch_dim), dtype=np.float64)
    feature_count = _gate_features(
        rows, np.zeros((len(rows), spec.patch_dim), dtype=np.float64)
    ).shape[1]
    gate = PhysicalNullGate(feature_count)
    with torch.no_grad():
        for parameter in gate.parameters():
            parameter.zero_()
    model = PhysicalResponseModel(
        coefficient_calibration={(1, 4): np.zeros(15, dtype=np.complex128)},
        gates=[gate],
        feature_mean=np.zeros(feature_count, dtype=np.float32),
        feature_scale=np.ones(feature_count, dtype=np.float32),
        thresholds={f"q{query}:m1-4": 0.25 for query in range(spec.patch_count)},
        response_scale=11.0,
        metadata={"contract": PHYSICAL_RESPONSE_CONTRACT},
    )

    prediction, probability = predict_physical_response(rows, model, patch_scale, spec)

    assert np.all(np.isfinite(prediction))
    assert np.all(np.isfinite(probability))
    np.testing.assert_array_equal(prediction, np.zeros_like(prediction))
    assert not any(row["path_available"] for row in rows)


def test_shared_response_scale_uses_only_active_train_targets() -> None:
    rows = [
        {"route": 2, "target": np.asarray([2.0, 4.0])},
        {"route": 0, "target": np.asarray([1000.0, 1000.0])},
        {"route": 2, "target": np.asarray([6.0, 8.0])},
    ]
    prediction = np.asarray([[1.0, 2.0], [5.0, 5.0], [3.0, 4.0]])

    scale, audit = _fit_shared_response_scale(rows, prediction)

    assert scale == 2.0
    assert audit["fit_active_rows"] == 2
    assert audit["selection_targets_read"] is False
    rows[1]["target"] = np.asarray([-1.0e9, 1.0e9])
    changed_scale, _ = _fit_shared_response_scale(rows, prediction)
    assert changed_scale == scale


def test_predict_applies_shared_scale_after_null_gate() -> None:
    rows = _oracle_test_rows(receiver_mode="oracle", oracle_path=True)[:2]
    spec = PatchSpec(2, 16, 2, 1, 2)
    patch_scale = np.ones((spec.patch_count, spec.patch_dim), dtype=np.float64)
    calibration = {(1, 4): np.ones(15, dtype=np.complex128)}
    from formal_v2.formal_action_inverse_response import _raw_predictions

    raw = _raw_predictions(rows, calibration, patch_scale, spec)
    feature_count = _gate_features(rows, raw).shape[1]
    pass_gate = PhysicalNullGate(feature_count)
    reject_gate = PhysicalNullGate(feature_count)
    with torch.no_grad():
        for gate, bias in ((pass_gate, 20.0), (reject_gate, -20.0)):
            for parameter in gate.parameters():
                parameter.zero_()
            gate.network[-1].bias.fill_(bias)
    common = dict(
        coefficient_calibration=calibration,
        feature_mean=np.zeros(feature_count, dtype=np.float32),
        feature_scale=np.ones(feature_count, dtype=np.float32),
        thresholds={f"q{row['query']}:m1-4": 0.5 for row in rows},
        response_scale=2.5,
        metadata={"contract": PHYSICAL_RESPONSE_CONTRACT},
    )
    passed, _ = predict_physical_response(
        rows, PhysicalResponseModel(gates=[pass_gate], **common), patch_scale, spec
    )
    rejected, _ = predict_physical_response(
        rows, PhysicalResponseModel(gates=[reject_gate], **common), patch_scale, spec
    )

    np.testing.assert_allclose(passed, 2.5 * raw)
    np.testing.assert_array_equal(rejected, np.zeros_like(rejected))


def test_nonfixture_qualification_authenticates_physical_checkpoint(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.npz"
    dataset_path.write_bytes(b"formal-candidate")
    dataset = SimpleNamespace(is_fixture=False, source_path=dataset_path)
    qualification = tmp_path / "qualification"
    teacher = qualification / "checkpoints" / "stage0_csi_teacher.pt"
    teacher.parent.mkdir(parents=True)
    teacher.write_bytes(b"teacher")
    candidate = tmp_path / "assets" / "generator_config.json"
    candidate.parent.mkdir()
    candidate.write_text("{}\n", encoding="ascii")
    checkpoint = (
        qualification
        / "checkpoints"
        / "qualification_probes"
        / "attempt_0001"
        / "physical_response.pt"
    )
    source = (
        Path(__file__).resolve().parents[1] / "formal_action_inverse_response.py"
    )
    physical_config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "physical_response_v1.json"
    )
    evidence = {
        "artifact_label": "test",
        "dataset_sha256": sha256_file(dataset_path),
        "config_sha256": "c" * 64,
        "fixture": False,
        "source_tree_sha256": "d" * 64,
        "requirements_lock_sha256": "e" * 64,
        "runtime_provenance_sha256": "f" * 64,
        "runtime_provenance": {},
    }
    bindings = {
        "dataset_sha256": evidence["dataset_sha256"],
        "formal_config_sha256": evidence["config_sha256"],
        "physical_config_sha256": sha256_file(physical_config_path),
        "candidate_config": str(candidate),
        "candidate_config_sha256": sha256_file(candidate),
        "model_source": str(source),
        "model_source_sha256": sha256_file(source),
    }
    model = PhysicalResponseModel(
        coefficient_calibration={(1, 4): np.asarray([1.0 + 0.0j])},
        gates=[PhysicalNullGate(1)],
        feature_mean=np.zeros(1, dtype=np.float32),
        feature_scale=np.ones(1, dtype=np.float32),
        thresholds={"q0:m1-4": 0.5},
        response_scale=1.0,
        metadata={"contract": PHYSICAL_RESPONSE_CONTRACT},
    )
    save_physical_response_checkpoint(checkpoint, model, bindings)
    physical = {
        "status": "FORMAL_QUALIFICATION_MODEL_FIT",
        "contract": PHYSICAL_RESPONSE_CONTRACT,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_round_trip_valid": True,
        "bindings": bindings,
        "physical_config": load_physical_response_config(),
        "physical_config_sha256": sha256_file(physical_config_path),
    }
    gate = {
        "schema_version": QUALIFICATION_SCHEMA,
        "passed": True,
        "scientific_use": "FORMAL_EXPERIMENT_ALLOWED",
        **evidence,
        "upstream_gates": {"G1": "PASS", "G2": "PASS"},
        "teacher_checkpoint": str(teacher),
        "teacher_checkpoint_sha256": sha256_file(teacher),
        "primary_route_contract": PRIMARY_ROUTE_CONTRACT,
        "physical_response": physical,
    }

    with mock.patch(
        "formal_v2.formal_evidence.evidence_context", return_value=evidence
    ):
        assert (
            require_formal_qualification(
                gate, {}, dataset, allow_nonscientific_fixture=False
            )
            is gate
        )
        checkpoint.write_bytes(b"tampered")
        with np.testing.assert_raises_regex(RuntimeError, "checkpoint hash mismatch"):
            require_formal_qualification(
                gate, {}, dataset, allow_nonscientific_fixture=False
            )
