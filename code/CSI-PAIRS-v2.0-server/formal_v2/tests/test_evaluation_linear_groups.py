from __future__ import annotations

import math
import time
import unittest

import numpy as np

from formal_v2.formal_evaluation import (
    _active_effect_bin_rows,
    _compatibility_effect_rows,
    _effect_bins_complete,
    _paired_score_differences,
    _response_effect_rows,
    _stable_pair_groups,
)
from formal_v2.formal_metrics import binary_auroc


def _oracle_paired_score_differences(scores, labels, pair_ids):
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    identifiers = np.asarray(pair_ids).astype(str)
    differences = []
    for pair_id in np.unique(identifiers):
        mask = identifiers == pair_id
        if int(np.sum(mask)) != 2 or set(targets[mask].tolist()) != {0, 1}:
            raise RuntimeError(
                "compatibility pair must contain one matched and one alternative score"
            )
        differences.append(
            float(
                values[mask & (targets == 1)][0]
                - values[mask & (targets == 0)][0]
            )
        )
    if not differences:
        raise RuntimeError("compatibility route has no complete paired scores")
    return np.asarray(differences, dtype=np.float64)


def _oracle_active_effect_bin_rows(
    seed, arm, bank, city, evaluation_scope, scores, labels, pair_ids, distances
):
    identifiers = np.asarray(pair_ids).astype(str)
    pair_order = np.unique(identifiers)
    if pair_order.size < 4:
        raise RuntimeError(
            "active CGS effect-bin report requires at least four paired units per bank"
        )
    pair_distance = np.asarray(
        [float(np.mean(np.asarray(distances)[identifiers == pair])) for pair in pair_order]
    )
    ranked = np.argsort(np.argsort(pair_distance, kind="stable"), kind="stable")
    bins = np.minimum(3, (4 * ranked) // pair_order.size)
    rows = []
    for bin_index, label in enumerate(
        ("low", "medium_low", "medium_high", "high")
    ):
        selected_pairs = pair_order[bins == bin_index]
        selected = np.isin(identifiers, selected_pairs)
        if not np.any(selected):
            raise RuntimeError("active CGS effect bin is empty")
        rows.append(
            {
                "seed": int(seed),
                "arm": str(arm),
                "bank_id": str(bank),
                "city_id": str(city),
                "evaluation_scope": str(evaluation_scope),
                "effect_bin": label,
                "pair_count": int(selected_pairs.size),
                "physical_distance_min": float(
                    np.min(np.asarray(distances)[selected])
                ),
                "physical_distance_max": float(
                    np.max(np.asarray(distances)[selected])
                ),
                "cgs_auroc": binary_auroc(
                    np.asarray(labels)[selected], np.asarray(scores)[selected]
                ),
            }
        )
    return rows


def _oracle_compatibility_effect_rows(seed, arm, evaluated, probabilities):
    rows = []
    pair_ids = evaluated["pair_ids"]
    for pair_id in np.unique(pair_ids):
        mask = pair_ids == pair_id
        difference = _oracle_paired_score_differences(
            probabilities[mask], evaluated["labels"][mask], pair_ids[mask]
        )[0]
        first = int(np.flatnonzero(mask)[0])
        rows.append(
            {
                "seed": int(seed),
                "arm": str(arm),
                "pair_id": str(pair_id),
                "scene_index": int(evaluated["scene_indices"][first]),
                "bank_id": str(evaluated["bank_ids"][first]),
                "base_map_cluster_id": str(
                    evaluated.get("base_map_cluster_ids", evaluated["bank_ids"])[first]
                ),
                "canonical_base_map_digest": str(
                    evaluated["canonical_base_map_digests"][first]
                ),
                "canonical_bank_digest": str(
                    evaluated["canonical_bank_digests"][first]
                ),
                "city_id": str(evaluated["city_ids"][first]),
                "source_world": int(evaluated["csi_worlds"][first]),
                "target_world": int(
                    evaluated["edge_targets"][first]
                    if evaluated["csi_worlds"][first]
                    == evaluated["edge_sources"][first]
                    else evaluated["edge_sources"][first]
                ),
                "position_index": int(evaluated["positions"][first]),
                "route": str(evaluated["routes"][first]),
                "physical_distance": float(
                    evaluated["physical_distances"][first]
                ),
                "matched_minus_alternative": float(difference),
            }
        )
    return rows


def _oracle_response_effect_rows(
    seed, arm, evaluated, prediction, action_swap_prediction, no_action_prediction
):
    rows = []
    for pair_id in np.unique(evaluated["pair_ids"]):
        mask = evaluated["pair_ids"] == pair_id
        first = int(np.flatnonzero(mask)[0])
        prediction_error = float(
            np.mean((prediction[mask] - evaluated["targets"][mask]) ** 2)
        )
        copy_error = float(
            np.mean(
                (
                    evaluated["source_targets"][mask]
                    - evaluated["targets"][mask]
                )
                ** 2
            )
        )
        swap_status = str(evaluated["wrong_action_match_status"][first])
        action_swap_error = (
            float(
                np.mean(
                    (
                        action_swap_prediction[mask]
                        - evaluated["targets"][mask]
                    )
                    ** 2
                )
            )
            if swap_status == "exact"
            else None
        )
        no_action_error = float(
            np.mean(
                (no_action_prediction[mask] - evaluated["targets"][mask]) ** 2
            )
        )
        rows.append(
            {
                "seed": int(seed),
                "arm": str(arm),
                "pair_id": str(pair_id),
                "scene_index": int(evaluated["scene_indices"][first]),
                "bank_id": str(evaluated["bank_ids"][first]),
                "base_map_cluster_id": str(
                    evaluated["base_map_cluster_ids"][first]
                ),
                "canonical_base_map_digest": str(
                    evaluated["canonical_base_map_digests"][first]
                ),
                "canonical_bank_digest": str(
                    evaluated["canonical_bank_digests"][first]
                ),
                "city_id": str(evaluated["city_ids"][first]),
                "source_world": int(evaluated["source_worlds"][first]),
                "target_world": int(evaluated["target_worlds"][first]),
                "position_index": int(evaluated["positions"][first]),
                "query_index": int(evaluated["queries"][first]),
                "route": str(evaluated["routes"][first]),
                "wrong_action_match_status": swap_status,
                "wrong_action_world": int(
                    evaluated["wrong_action_world"][first]
                ),
                "prediction_mse": prediction_error,
                "copy_mse": copy_error,
                "action_swap_mse": action_swap_error,
                "no_action_mse": no_action_error,
                "response_advantage": copy_error - prediction_error,
                "response_advantage_vs_action_swap": (
                    action_swap_error - prediction_error
                    if action_swap_error is not None
                    else None
                ),
                "response_advantage_vs_no_action": (
                    no_action_error - prediction_error
                ),
            }
        )
    return rows


def _assert_scalar_identical(actual, expected, path):
    if isinstance(expected, (float, np.floating)):
        if math.isnan(float(expected)):
            if not isinstance(actual, (float, np.floating)) or not math.isnan(
                float(actual)
            ):
                raise AssertionError(f"{path}: expected NaN, got {actual!r}")
            return
        actual_bits = np.asarray(float(actual), dtype=np.float64).view(np.uint64).item()
        expected_bits = (
            np.asarray(float(expected), dtype=np.float64).view(np.uint64).item()
        )
        if actual_bits != expected_bits:
            raise AssertionError(
                f"{path}: float bits differ: {actual!r} != {expected!r}"
            )
        return
    if actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def _assert_rows_identical(actual, expected):
    if len(actual) != len(expected):
        raise AssertionError(f"row count differs: {len(actual)} != {len(expected)}")
    for row_index, (actual_row, expected_row) in enumerate(zip(actual, expected)):
        if list(actual_row) != list(expected_row):
            raise AssertionError(
                f"row {row_index} fields differ: {list(actual_row)} != {list(expected_row)}"
            )
        for field, expected_value in expected_row.items():
            _assert_scalar_identical(
                actual_row[field], expected_value, f"row {row_index}.{field}"
            )


def _compatibility_fixture():
    return {
        "pair_ids": np.asarray(["z", "a", "z", "m", "a", "m"]),
        "labels": np.asarray([0, 1, 1, 1, 0, 0], dtype=np.int64),
        "scene_indices": np.asarray([90, 10, 91, 30, 11, 31]),
        "bank_ids": np.asarray(["bz", "ba", "bz", "bm", "ba", "bm"]),
        "base_map_cluster_ids": np.asarray(
            ["cz", "ca", "cz", "cm", "ca", "cm"]
        ),
        "canonical_base_map_digests": np.asarray(
            ["Cz", "Ca", "Cz", "Cm", "Ca", "Cm"]
        ),
        "canonical_bank_digests": np.asarray(
            ["Bz", "Ba", "Bz", "Bm", "Ba", "Bm"]
        ),
        "city_ids": np.asarray(
            ["city-z", "city-a", "city-z", "city-m", "city-a", "city-m"]
        ),
        "csi_worlds": np.asarray([2, 0, 2, 1, 0, 1]),
        "edge_sources": np.asarray([1, 0, 1, 1, 0, 1]),
        "edge_targets": np.asarray([2, 1, 2, 2, 1, 2]),
        "positions": np.asarray([9, 1, 9, 3, 1, 3]),
        "routes": np.asarray(["active", "null", "active", "gray", "null", "gray"]),
        "physical_distances": np.asarray([9.0, 1.0, 9.0, 3.0, 1.0, 3.0]),
    }


def _active_fixture():
    pair_count = 8
    group = np.repeat(np.arange(pair_count), 2)
    labels = np.tile(np.asarray([0, 1]), pair_count)
    scores = np.repeat(np.asarray([0.2, 0.2, 0.5, 0.5]), 4)
    distances_by_group = np.asarray(
        [0.0, 0.0, 1.0, 1.0, np.inf, np.inf, np.nan, np.nan]
    )
    distances = distances_by_group[group]
    permutation = np.random.default_rng(20260831).permutation(group.size)
    return (
        scores[permutation],
        labels[permutation],
        np.asarray([f"pair-{value:02d}" for value in group[permutation]]),
        distances[permutation],
    )


def _response_fixture(pair_count, group_width, *, nonfinite=False):
    group = np.repeat(np.arange(pair_count), group_width)
    member = np.tile(np.arange(group_width), pair_count)
    permutation = np.random.default_rng(7000 + pair_count).permutation(group.size)
    group = group[permutation]
    member = member[permutation]
    targets = np.column_stack(
        (group * 0.01 + member * 0.001, group * -0.02 + member * 0.003)
    )
    prediction = targets + np.column_stack(
        (0.1 + member * 0.002, -0.2 + member * 0.001)
    )
    source_targets = targets + np.column_stack(
        (0.4 + member * 0.003, -0.3 + member * 0.004)
    )
    action_swap = targets + np.column_stack(
        (-0.5 + member * 0.001, 0.6 + member * 0.002)
    )
    no_action = targets + np.column_stack(
        (0.7 + member * 0.002, -0.8 + member * 0.001)
    )
    if nonfinite:
        prediction[0, 0] = np.nan
        prediction[-1, 1] = np.inf
        source_targets[1, 0] = -np.inf
        action_swap[2, 1] = np.inf
        no_action[-2, 0] = np.nan
    evaluated = {
        "pair_ids": np.asarray([f"pair-{value:08d}" for value in group]),
        "targets": targets,
        "source_targets": source_targets,
        "wrong_action_match_status": np.where(group % 2 == 0, "exact", "fallback"),
        "wrong_action_world": group % 7,
        "scene_indices": group + 100,
        "bank_ids": np.asarray([f"bank-{value:08d}" for value in group]),
        "base_map_cluster_ids": np.asarray(
            [f"cluster-{value:08d}" for value in group]
        ),
        "canonical_base_map_digests": np.asarray(
            [f"map-digest-{value:08d}" for value in group]
        ),
        "canonical_bank_digests": np.asarray(
            [f"bank-digest-{value:08d}" for value in group]
        ),
        "city_ids": np.asarray([f"city-{value % 3}" for value in group]),
        "source_worlds": group % 4,
        "target_worlds": (group + 1) % 4,
        "positions": group % 13,
        "queries": member,
        "routes": np.asarray([("active" if value % 2 == 0 else "gray") for value in group]),
    }
    return evaluated, prediction, action_swap, no_action


def benchmark_response_grouping(base_pair_count=64, group_width=4):
    records = []
    for multiplier in (1, 2, 4):
        pair_count = base_pair_count * multiplier
        evaluated, prediction, action_swap, no_action = _response_fixture(
            pair_count, group_width
        )
        started = time.perf_counter()
        expected = _oracle_response_effect_rows(
            7, "full", evaluated, prediction, action_swap, no_action
        )
        oracle_seconds = time.perf_counter() - started
        started = time.perf_counter()
        actual = _response_effect_rows(
            7, "full", evaluated, prediction, action_swap, no_action
        )
        grouped_seconds = time.perf_counter() - started
        _assert_rows_identical(actual, expected)
        records.append(
            {
                "scale": f"{multiplier}N",
                "pair_count": pair_count,
                "sample_count": pair_count * group_width,
                "oracle_seconds": oracle_seconds,
                "grouped_seconds": grouped_seconds,
                "speedup": oracle_seconds / max(grouped_seconds, 1e-12),
            }
        )
    return records


class EvaluationLinearGroupTests(unittest.TestCase):
    def test_effect_bin_completeness_consumes_rows_once(self):
        cgs_rows = [
            {"seed": 1, "arm": "full", "bank_id": "a"},
            {"seed": 1, "arm": "full", "bank_id": "b"},
        ]
        effect_rows = [
            {
                "seed": 1,
                "arm": "full",
                "bank_id": bank,
                "effect_bin": effect_bin,
            }
            for bank in ("a", "b")
            for effect_bin in ("low", "medium_low", "medium_high", "high")
        ]

        class SinglePassRows:
            def __init__(self, rows):
                self.rows = rows
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                if self.iterations > 1:
                    raise AssertionError("effect rows were rescanned")
                return iter(self.rows)

        rows = SinglePassRows(effect_rows)
        self.assertTrue(_effect_bins_complete(cgs_rows, rows))
        self.assertEqual(rows.iterations, 1)
        self.assertFalse(_effect_bins_complete(cgs_rows, effect_rows[:-1]))
        empty_rows = SinglePassRows(effect_rows)
        self.assertFalse(_effect_bins_complete([], empty_rows))
        self.assertEqual(empty_rows.iterations, 0)

    def test_stable_groups_match_unique_order_and_preserve_member_order(self):
        identifiers = np.asarray(["z", "a", "z", "m", "a", "m", "z"])
        returned, pair_order, order, starts, stops = _stable_pair_groups(identifiers)
        np.testing.assert_array_equal(returned, identifiers)
        np.testing.assert_array_equal(pair_order, np.unique(identifiers))
        for pair_id, start, stop in zip(pair_order, starts, stops):
            np.testing.assert_array_equal(
                order[start:stop], np.flatnonzero(identifiers == pair_id)
            )

        returned, pair_order, order, starts, stops = _stable_pair_groups(
            np.asarray([], dtype="U1")
        )
        self.assertEqual(returned.size, 0)
        self.assertEqual(pair_order.size, 0)
        self.assertEqual(order.size, 0)
        self.assertEqual(starts.size, 0)
        self.assertEqual(stops.size, 0)

    def test_paired_scores_match_oracle_for_shuffle_ties_nan_and_inf(self):
        pair_ids = np.asarray(["b", "a", "c", "b", "a", "c", "d", "d"])
        labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0])
        scores = np.asarray([np.inf, 1.0, np.nan, np.inf, np.inf, 0.0, 0.5, 0.5])
        with self.assertWarns(RuntimeWarning):
            expected = _oracle_paired_score_differences(scores, labels, pair_ids)
        with self.assertWarns(RuntimeWarning):
            actual = _paired_score_differences(scores, labels, pair_ids)
        np.testing.assert_equal(actual, expected)

    def test_paired_scores_preserve_empty_single_and_duplicate_failures(self):
        cases = (
            (np.asarray([]), np.asarray([]), np.asarray([], dtype="U1")),
            (np.asarray([0.2]), np.asarray([1]), np.asarray(["a"])),
            (
                np.asarray([0.2, 0.3, 0.4]),
                np.asarray([0, 1, 1]),
                np.asarray(["a", "a", "a"]),
            ),
        )
        for scores, labels, pair_ids in cases:
            with self.subTest(size=pair_ids.size):
                with self.assertRaises(RuntimeError) as oracle_error:
                    _oracle_paired_score_differences(scores, labels, pair_ids)
                with self.assertRaises(RuntimeError) as grouped_error:
                    _paired_score_differences(scores, labels, pair_ids)
                self.assertEqual(str(grouped_error.exception), str(oracle_error.exception))

    def test_active_effect_rows_match_oracle_with_distance_and_score_ties(self):
        scores, labels, pair_ids, distances = _active_fixture()
        expected = _oracle_active_effect_bin_rows(
            11, "full", "bank", "city", "target:city", scores, labels, pair_ids, distances
        )
        actual = _active_effect_bin_rows(
            11, "full", "bank", "city", "target:city", scores, labels, pair_ids, distances
        )
        _assert_rows_identical(actual, expected)

    def test_compatibility_rows_match_oracle_for_shuffled_pairs(self):
        evaluated = _compatibility_fixture()
        probabilities = np.asarray([0.2, 0.9, 0.8, 0.6, 0.3, 0.4])
        expected = _oracle_compatibility_effect_rows(
            13, "alignment", evaluated, probabilities
        )
        actual = _compatibility_effect_rows(13, "alignment", evaluated, probabilities)
        _assert_rows_identical(actual, expected)
        self.assertEqual([row["pair_id"] for row in actual], ["a", "m", "z"])

    def test_compatibility_rows_preserve_empty_and_invalid_pair_behavior(self):
        self.assertEqual(
            _compatibility_effect_rows(
                1, "endpoint", {"pair_ids": np.asarray([], dtype="U1")}, np.asarray([])
            ),
            [],
        )
        evaluated = {"pair_ids": np.asarray(["a"]), "labels": np.asarray([1])}
        with self.assertRaises(RuntimeError) as oracle_error:
            _oracle_compatibility_effect_rows(1, "endpoint", evaluated, np.asarray([0.5]))
        with self.assertRaises(RuntimeError) as grouped_error:
            _compatibility_effect_rows(1, "endpoint", evaluated, np.asarray([0.5]))
        self.assertEqual(str(grouped_error.exception), str(oracle_error.exception))

    def test_response_rows_match_oracle_for_empty_single_and_nonfinite_values(self):
        empty = {"pair_ids": np.asarray([], dtype="U1")}
        self.assertEqual(
            _response_effect_rows(
                1, "response", empty, np.asarray([]), np.asarray([]), np.asarray([])
            ),
            [],
        )
        for pair_count, group_width, nonfinite in ((1, 1, False), (5, 3, True)):
            with self.subTest(
                pair_count=pair_count,
                group_width=group_width,
                nonfinite=nonfinite,
            ):
                evaluated, prediction, action_swap, no_action = _response_fixture(
                    pair_count, group_width, nonfinite=nonfinite
                )
                with np.errstate(invalid="ignore", over="ignore"):
                    expected = _oracle_response_effect_rows(
                        17, "response", evaluated, prediction, action_swap, no_action
                    )
                    actual = _response_effect_rows(
                        17, "response", evaluated, prediction, action_swap, no_action
                    )
                _assert_rows_identical(actual, expected)
                self.assertEqual(
                    [row["pair_id"] for row in actual],
                    np.unique(evaluated["pair_ids"]).tolist(),
                )

    def test_n_2n_4n_benchmark_helper_checks_oracle_equivalence(self):
        records = benchmark_response_grouping(base_pair_count=2, group_width=2)
        self.assertEqual([row["scale"] for row in records], ["1N", "2N", "4N"])
        self.assertEqual([row["sample_count"] for row in records], [4, 8, 16])
        self.assertTrue(all(row["oracle_seconds"] >= 0.0 for row in records))
        self.assertTrue(all(row["grouped_seconds"] >= 0.0 for row in records))


if __name__ == "__main__":
    unittest.main()
