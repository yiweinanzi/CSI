from __future__ import annotations

import numpy as np
import torch
from torch import nn

from artifacts.formal_readiness.tools import v5_action_inverse_response_probe as probe


def test_learned_gate_threshold_controls_every_fit_bank_and_group() -> None:
    probability = np.asarray(
        [
            0.01, 0.02, 0.03, 0.04, 0.90,
            0.01, 0.02, 0.03, 0.70, 0.80,
            0.01, 0.02, 0.03, 0.04, 0.60,
            0.01, 0.02, 0.03, 0.50, 0.55,
        ]
    )
    labels = np.zeros(probability.shape, dtype=np.float32)
    group_ids = np.asarray((["a"] * 5 + ["b"] * 5) * 2)
    bank_ids = np.asarray(["bank-1"] * 10 + ["bank-2"] * 10)

    thresholds, audit = probe._calibrate_learned_gate_thresholds(
        probability,
        labels,
        group_ids,
        bank_ids,
        maximum_false_positive_rate=0.20,
    )

    assert thresholds["a"] > 0.90
    assert thresholds["b"] > 0.80
    assert audit["strategy"] == (
        "maximum_fit_bank_query_material_conditional_null_quantile"
    )
    assert audit["maximum_fit_bank_group_false_positive_rate"] <= 0.20
    for bank_id in ("bank-1", "bank-2"):
        for group_id in ("a", "b"):
            selected = (bank_ids == bank_id) & (group_ids == group_id)
            assert np.mean(probability[selected] >= thresholds[group_id]) <= 0.20


def test_learned_gate_calibration_requires_each_fit_bank_group() -> None:
    with np.testing.assert_raises_regex(RuntimeError, "has no physical null rows"):
        probe._calibrate_learned_gate_thresholds(
            np.asarray([0.1, 0.2]),
            np.zeros(2, dtype=np.float32),
            np.asarray(["a", "b"]),
            np.asarray(["bank-1", "bank-2"]),
            maximum_false_positive_rate=0.20,
        )


def test_learned_gate_sampler_balances_route_and_fit_bank() -> None:
    labels = np.asarray([0, 0, 1, 1] * 3, dtype=np.float32)
    bank_ids = np.repeat(np.asarray(["a", "b", "c"]), 4)
    pools = probe._gate_index_pools(labels, bank_ids)
    indices = probe._bank_balanced_gate_indices(
        pools,
        12,
        np.random.default_rng(7),
    )
    assert int(np.sum(labels[indices] == 0.0)) == 6
    assert int(np.sum(labels[indices] == 1.0)) == 6
    for label in (0.0, 1.0):
        counts = [
            int(np.sum((labels[indices] == label) & (bank_ids[indices] == bank)))
            for bank in ("a", "b", "c")
        ]
        assert counts == [2, 2, 2]


def test_learned_gate_prediction_is_exact_zero_below_threshold(monkeypatch) -> None:
    class FirstFeatureLogit(nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return values[:, 0]

    monkeypatch.setattr(
        probe,
        "_gate_features",
        lambda rows, prediction: np.asarray([[-2.0], [2.0]], dtype=np.float32),
    )
    rows = [
        {"query": 0, "material": (1, 2)},
        {"query": 0, "material": (1, 2)},
    ]
    prediction = np.asarray([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)

    gated, probability = probe._learned_gate_predictions(
        rows,
        prediction,
        FirstFeatureLogit(),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        {probe._gate_calibration_group(rows[0]): 0.5},
    )

    np.testing.assert_array_equal(gated[0], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(gated[1], prediction[1])
    assert probability[0] < 0.5 < probability[1]


def test_learned_gate_ensemble_averages_before_applying_threshold(monkeypatch) -> None:
    class ConstantLogit(nn.Module):
        def __init__(self, value: float):
            super().__init__()
            self.value = nn.Parameter(torch.tensor(value), requires_grad=False)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.value.expand(values.shape[0])

    monkeypatch.setattr(
        probe,
        "_gate_features",
        lambda rows, prediction: np.zeros((2, 1), dtype=np.float32),
    )
    rows = [
        {"query": 0, "material": (1, 2)},
        {"query": 0, "material": (1, 2)},
    ]
    prediction = np.asarray([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)

    gated, probability = probe._learned_gate_ensemble_predictions(
        rows,
        prediction,
        [ConstantLogit(-2.0), ConstantLogit(2.0)],
        np.asarray([0.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        {probe._gate_calibration_group(rows[0]): 0.49},
    )

    np.testing.assert_allclose(probability, np.asarray([0.5, 0.5]), atol=1e-7)
    np.testing.assert_array_equal(gated, prediction)


def test_learned_gate_ensemble_rejects_empty_model_list() -> None:
    with np.testing.assert_raises_regex(ValueError, "must not be empty"):
        probe._learned_gate_ensemble_predictions(
            [{"query": 0, "material": (1, 2)}],
            np.ones((1, 2), dtype=np.float32),
            [],
            np.zeros(1, dtype=np.float32),
            np.ones(1, dtype=np.float32),
            {"q0:m1-2": 0.5},
        )
