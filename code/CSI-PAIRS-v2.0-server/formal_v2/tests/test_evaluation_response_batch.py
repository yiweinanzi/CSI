from __future__ import annotations

import math
import time
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from formal_v2.formal_dataset import FormalEdge
from formal_v2.formal_evaluation import (
    _batched_masked_states,
    _response_probe_dataset,
)
from formal_v2.formal_evaluation_batch import (
    batched_masked_states_from_context,
    encode_context_bank,
)
from formal_v2.formal_model import CSIPairsFormalModel
from formal_v2.formal_protocol import PatchSpec, frozen_mask_query_bank


class _ResponseDataset(SimpleNamespace):
    def directed_edges(self, scene):
        del scene
        yield FormalEdge(0, 0, 1, 0, 100, 1)
        yield FormalEdge(0, 1, 0, 0, 100, -1)

    def canonical_base_map_digest(self, scene):
        return f"canonical-scene-{int(scene)}"


def _response_model(device="cpu", *, patch_rows=2, patch_columns=2):
    torch.manual_seed(20260831)
    patch_count = int(patch_rows) * int(patch_columns)
    return CSIPairsFormalModel(
        patch_count=patch_count,
        patch_rows=int(patch_rows),
        patch_columns=int(patch_columns),
        patch_dim=2,
        map_channels=3,
        action_channels=12,
        radio_dim=4,
        latent_dim=8,
        state_dim=8,
        map_dim=8,
        hidden_dim=16,
        attention_heads=2,
    ).eval().to(device)


def _response_fixture(position_count=3, *, patch_rows=2, patch_columns=2):
    generator = np.random.default_rng(614 + int(position_count))
    spec = PatchSpec(
        antennas=int(patch_rows),
        subcarriers=int(patch_columns),
        patch_complex_size=1,
        patch_antenna_size=1,
        patch_subcarrier_size=1,
    )
    maps = np.zeros((1, 2, 3, 8, 8), dtype=np.float64)
    maps[0, 1, 0, 2:6, 1:5] = 1.0
    maps[0, 1, 1, 2:6, 1:5] = 2.0
    maps[0, 1, 2, 2:6, 1:5] = 1.0
    dataset = _ResponseDataset(
        csi=generator.normal(
            size=(1, 2, position_count, 2 * spec.patch_count)
        ),
        maps=maps,
        map_channel_names=np.asarray(["occupancy", "height", "material"]),
        positions=generator.normal(size=(1, position_count, 2)),
        radio_config=generator.normal(size=(1, 2)),
        bs_pose=generator.normal(size=(1, 2)),
        world_bits=np.asarray([[0], [1]], dtype=np.int8),
        bit_count=1,
        position_count=int(position_count),
        scene_roles=np.asarray(["source_probe_train"]),
        position_roles=np.full((1, position_count), "standard"),
        metadata={"assets": {"material_category_count": 4}},
        bank_ids=np.asarray(["bank-fixture"]),
        city_ids=np.asarray(["city-fixture"]),
        base_map_cluster_ids=np.asarray(["cluster-fixture"]),
    )
    normalization = SimpleNamespace(
        patch_mean=np.zeros((spec.patch_count, 2), dtype=np.float64),
        patch_scale=np.ones((spec.patch_count, 2), dtype=np.float64),
        map_scale=np.ones(3, dtype=np.float64),
        radio_mean=np.zeros(4, dtype=np.float64),
        radio_scale=np.ones(4, dtype=np.float64),
        action_scale=np.ones(12, dtype=np.float64),
        position_mean=np.zeros(2, dtype=np.float64),
        position_scale=np.ones(2, dtype=np.float64),
    )
    response_route = {}
    for source_world, target_world in ((0, 1), (1, 0)):
        for position in range(position_count):
            for query in range(spec.patch_count):
                response_route[(0, source_world, target_world, position, query)] = (
                    position + query
                ) % 3
    return (
        dataset,
        SimpleNamespace(patch_spec=spec, mask_bank=frozen_mask_query_bank(spec, 91)),
        normalization,
        SimpleNamespace(response_route=response_route),
    )


def _per_position_state_oracle(
    model,
    patch_bank,
    mask_bank,
    patch_indices,
    encoded_context,
    *,
    batch_size,
):
    entries = tuple(mask_bank)
    indices = np.asarray(patch_indices, dtype=np.int64)
    states = []
    queries = []
    start = 0
    while start < indices.size:
        patch_index = int(indices[start])
        stop = start + 1
        while stop < indices.size and int(indices[stop]) == patch_index:
            stop += 1
        state, query = _batched_masked_states(
            model,
            np.asarray(patch_bank)[patch_index],
            entries[start:stop],
            encoded_context,
            batch_size=batch_size,
        )
        states.append(state)
        queries.append(query)
        start = stop
    return torch.cat(states, dim=0), np.concatenate(queries)


def _run_response(model, fixture, batch_size, *, legacy=False, active_only=False):
    dataset, teacher, normalization, routed = fixture
    replacement = (
        mock.patch(
            "formal_v2.formal_evaluation._batched_masked_patch_bank_states",
            side_effect=_per_position_state_oracle,
        )
        if legacy
        else nullcontext()
    )
    with replacement:
        return _response_probe_dataset(
            model,
            dataset,
            teacher,
            {},
            normalization,
            np.asarray([0]),
            active_only=active_only,
            route_normalization=object(),
            routed=routed,
            batch_size=batch_size,
        )


def _compare_outputs(actual, expected, *, rtol, atol):
    max_abs_error = 0.0
    if set(actual) != set(expected):
        raise AssertionError("response output keys changed")
    for name in actual:
        actual_value = actual[name]
        expected_value = expected[name]
        if not isinstance(actual_value, np.ndarray):
            if actual_value != expected_value:
                raise AssertionError(f"response scalar changed: {name}")
            continue
        if np.issubdtype(actual_value.dtype, np.floating):
            np.testing.assert_allclose(
                actual_value,
                expected_value,
                rtol=rtol,
                atol=atol,
                err_msg=name,
            )
            if actual_value.size:
                max_abs_error = max(
                    max_abs_error,
                    float(np.max(np.abs(actual_value - expected_value))),
                )
        else:
            np.testing.assert_array_equal(actual_value, expected_value, err_msg=name)
    return max_abs_error


def _timed_response(model, fixture, batch_size, *, legacy):
    device = next(model.parameters()).device
    calls = []
    hook = model.csi_encoder.register_forward_pre_hook(
        lambda _module, arguments: calls.append(int(arguments[0].shape[0]))
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        output = _run_response(model, fixture, batch_size, legacy=legacy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        hook.remove()
    return output, time.perf_counter() - started, calls


def benchmark_response_probe_edge_batch(
    position_count=16,
    batch_size=256,
    device="cpu",
    *,
    patch_rows=2,
    patch_columns=2,
):
    query_count = int(patch_rows) * int(patch_columns)
    model = _response_model(
        device, patch_rows=patch_rows, patch_columns=patch_columns
    )
    fixture = _response_fixture(
        position_count, patch_rows=patch_rows, patch_columns=patch_columns
    )
    _run_response(
        model,
        _response_fixture(
            1, patch_rows=patch_rows, patch_columns=patch_columns
        ),
        batch_size,
        legacy=False,
    )
    expected, legacy_seconds, legacy_calls = _timed_response(
        model, fixture, batch_size, legacy=True
    )
    actual, batched_seconds, batched_calls = _timed_response(
        model, fixture, batch_size, legacy=False
    )
    return {
        "device": str(next(model.parameters()).device),
        "position_count": int(position_count),
        "query_count": query_count,
        "edge_count": 2,
        "state_variant_count": 4,
        "batch_size": int(batch_size),
        "legacy_state_calls": len(legacy_calls),
        "batched_state_calls": len(batched_calls),
        "expected_legacy_state_calls": 2
        * int(position_count)
        * 4
        * math.ceil(query_count / int(batch_size)),
        "expected_batched_state_calls": 2
        * 4
        * math.ceil((int(position_count) * query_count) / int(batch_size)),
        "legacy_seconds": legacy_seconds,
        "batched_seconds": batched_seconds,
        "speedup": legacy_seconds / max(batched_seconds, 1e-12),
        "max_abs_error": _compare_outputs(
            actual, expected, rtol=2e-5, atol=2e-6
        ),
    }


class EvaluationResponseBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._original_threads)

    def test_masked_patch_bank_batch_one_matches_per_row_oracle_bitwise(self):
        model = _response_model()
        fixture = _response_fixture(3)
        dataset, teacher, normalization, _ = fixture
        del dataset, normalization
        generator = torch.Generator().manual_seed(841)
        patch_bank = torch.randn(3, 4, 2, generator=generator)
        masks = np.vstack(
            [
                next(entry for entry in teacher.mask_bank if entry.query == query).mask
                for _position in range(3)
                for query in range(4)
            ]
        )
        patch_indices = np.repeat(np.arange(3), 4)
        maps = torch.randn(1, 3, 8, 8, generator=generator)
        radio = torch.randn(1, 4, generator=generator)
        context = encode_context_bank(model, maps, radio)
        expected, _ = _per_position_state_oracle(
            model,
            patch_bank,
            tuple(
                next(entry for entry in teacher.mask_bank if entry.query == query)
                for _position in range(3)
                for query in range(4)
            ),
            patch_indices,
            context,
            batch_size=1,
        )
        actual = batched_masked_states_from_context(
            model,
            patch_bank,
            masks,
            context,
            patch_indices=patch_indices,
            batch_size=1,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_response_batch_one_is_bitwise_exact_for_every_output(self):
        model = _response_model()
        fixture = _response_fixture(3)
        expected = _run_response(model, fixture, 1, legacy=True)
        actual = _run_response(model, fixture, 1, legacy=False)
        self.assertEqual(_compare_outputs(actual, expected, rtol=0.0, atol=0.0), 0.0)
        self.assertEqual(
            actual["pair_ids"].tolist(),
            [
                f"bank-fixture:{source}:{target}:{position}:{query}"
                for source, target in ((0, 1), (1, 0))
                for position in range(3)
                for query in range(4)
            ],
        )

    def test_per_edge_batch_reduces_state_calls_and_matches_oracle(self):
        model = _response_model()
        fixture = _response_fixture(5)
        expected, _, legacy_calls = _timed_response(
            model, fixture, 256, legacy=True
        )
        actual, _, batched_calls = _timed_response(
            model, fixture, 256, legacy=False
        )
        self.assertEqual(legacy_calls, [4] * 40)
        self.assertEqual(batched_calls, [20] * 8)
        self.assertLessEqual(
            _compare_outputs(actual, expected, rtol=2e-6, atol=3e-7), 3e-7
        )

    def test_active_route_keeps_position_then_query_enumeration(self):
        model = _response_model()
        fixture = _response_fixture(3)
        expected = _run_response(
            model, fixture, 1, legacy=True, active_only=True
        )
        actual = _run_response(
            model, fixture, 1, legacy=False, active_only=True
        )
        self.assertEqual(_compare_outputs(actual, expected, rtol=0.0, atol=0.0), 0.0)
        self.assertEqual(
            actual["pair_ids"].tolist(),
            [
                f"bank-fixture:{source}:{target}:{position}:{query}"
                for source, target in ((0, 1), (1, 0))
                for position in range(3)
                for query in range(4)
                if (position + query) % 3 == 2
            ],
        )
        self.assertTrue(np.all(actual["routes"] == "active"))

    def test_patch_and_context_indices_fail_closed(self):
        model = _response_model()
        generator = torch.Generator().manual_seed(144)
        patches = torch.randn(2, 4, 2, generator=generator)
        masks = torch.zeros(3, 4, dtype=torch.bool)
        context = encode_context_bank(
            model,
            torch.randn(1, 3, 8, 8, generator=generator),
            torch.randn(1, 4, generator=generator),
        )
        with self.assertRaisesRegex(ValueError, "patch_indices is required"):
            batched_masked_states_from_context(model, patches, masks, context)
        with self.assertRaisesRegex(ValueError, "patch_indices.*out-of-range"):
            batched_masked_states_from_context(
                model,
                patches,
                masks,
                context,
                patch_indices=np.asarray([0, 1, 2]),
            )
        with self.assertRaisesRegex(ValueError, "context_indices is required"):
            batched_masked_states_from_context(
                model,
                patches,
                masks,
                torch.cat((context, context), dim=0),
                patch_indices=np.asarray([0, 1, 0]),
            )


if __name__ == "__main__":
    unittest.main()
