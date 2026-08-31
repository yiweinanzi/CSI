from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from formal_v2.formal_evaluation import (
    _batched_action_variants,
    _batched_masked_states,
    _cached_encoded_context,
    _encode_native_fixed_actions,
    _masked_alignment_state_and_score,
)
from formal_v2.formal_evaluation_batch import (
    batched_action_predictions,
    batched_action_predictions_from_encoded,
    batched_model_states,
    batched_model_states_from_context,
    batched_query_states,
    batched_query_states_from_context,
    encode_action_bank,
    encode_context_bank,
    gather_query_states,
)
from formal_v2.formal_model import CSIPairsFormalModel
from formal_v2.formal_model import endpoint_per_sample


def _tiny_model(device="cpu"):
    torch.manual_seed(20260831)
    return CSIPairsFormalModel(
        patch_count=4,
        patch_rows=2,
        patch_columns=2,
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


def _fixture(sample_count=7):
    generator = torch.Generator(device="cpu").manual_seed(1911 + sample_count)
    context_count = 3
    action_count = 3
    queries = torch.as_tensor(
        [3, 0, 2, 1, 3, 1, 0][:sample_count], dtype=torch.long
    )
    if sample_count > 7:
        queries = torch.arange(sample_count, dtype=torch.long) % 4
    masks = torch.zeros(sample_count, 4, dtype=torch.bool)
    if sample_count:
        rows = torch.arange(sample_count)
        masks[rows, queries] = True
        masks[rows, (queries + 1) % 4] = True
    return {
        "visible": torch.randn(sample_count, 4, 2, generator=generator),
        "maps": torch.randn(context_count, 3, 8, 8, generator=generator),
        "radio": torch.randn(context_count, 4, generator=generator),
        "masks": masks,
        "queries": queries,
        "context_indices": torch.as_tensor(
            [2, 0, 1, 2, 0, 1, 2][:sample_count], dtype=torch.long
        )
        if sample_count <= 7
        else torch.arange(sample_count, dtype=torch.long) % context_count,
        "actions": torch.randn(action_count, 12, 8, 8, generator=generator),
        "action_indices": torch.as_tensor(
            [1, 0, 2, 1, 2, 0, 1][:sample_count], dtype=torch.long
        )
        if sample_count <= 7
        else (torch.arange(sample_count, dtype=torch.long) * 2 + 1) % action_count,
    }


def _model_device_and_dtype(model):
    parameter = next(model.parameters())
    return parameter.device, parameter.dtype


def _per_sample_oracle(model, values):
    device, dtype = _model_device_and_dtype(model)
    states = []
    query_states = []
    latent = []
    physical = []
    with torch.no_grad():
        for row in range(values["visible"].shape[0]):
            context = int(values["context_indices"][row])
            action = int(values["action_indices"][row])
            query = values["queries"][row : row + 1].to(device=device)
            state = model.state(
                values["visible"][row : row + 1].to(device=device, dtype=dtype),
                values["maps"][context : context + 1].to(
                    device=device, dtype=dtype
                ),
                values["radio"][context : context + 1].to(
                    device=device, dtype=dtype
                ),
                values["masks"][row : row + 1].to(device=device),
            )
            latent_value, physical_value = model.predict(
                state,
                values["actions"][action : action + 1].to(
                    device=device, dtype=dtype
                ),
                query,
            )
            states.append(state)
            query_states.append(state[0, int(query.item())])
            latent.append(latent_value)
            physical.append(physical_value)
    if not states:
        return (
            torch.empty(0, 4, 8, dtype=dtype),
            torch.empty(0, 8, dtype=dtype),
            torch.empty(0, 8, dtype=dtype),
            torch.empty(0, 2, dtype=dtype),
        )
    return (
        torch.cat(states, dim=0).cpu(),
        torch.stack(query_states, dim=0).cpu(),
        torch.cat(latent, dim=0).cpu(),
        torch.cat(physical, dim=0).cpu(),
    )


def _max_abs_error(actual, expected):
    if actual.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(actual - expected)).item())


def _batched_outputs(model, values, batch_size):
    contexts = encode_context_bank(
        model, values["maps"], values["radio"], batch_size=batch_size
    )
    states = batched_model_states_from_context(
        model,
        values["visible"],
        values["masks"],
        contexts,
        context_indices=values["context_indices"],
        batch_size=batch_size,
    )
    query_states = gather_query_states(states, values["queries"])
    actions = encode_action_bank(model, values["actions"], batch_size=batch_size)
    latent, physical = batched_action_predictions_from_encoded(
        model,
        states,
        actions,
        values["queries"],
        action_indices=values["action_indices"],
        batch_size=batch_size,
    )
    return states, query_states, latent, physical


def _mask_entries(values):
    return tuple(
        SimpleNamespace(
            mask=values["masks"][row].cpu().numpy(),
            query=int(values["queries"][row]),
        )
        for row in range(values["queries"].shape[0])
    )


def _masked_alignment_oracle(
    model,
    patches,
    maps,
    radio,
    action,
    entries,
    latent_targets,
    physical_targets,
    normalization,
):
    device, dtype = _model_device_and_dtype(model)
    representations = []
    errors = []
    with torch.no_grad():
        for entry in entries:
            visible = patches.clone().to(device=device, dtype=dtype)
            visible[:, entry.mask] = 0.0
            masks = torch.as_tensor(
                entry.mask[None, :], dtype=torch.bool, device=device
            )
            state = model.state(
                visible,
                maps.to(device=device, dtype=dtype),
                radio.to(device=device, dtype=dtype),
                masks,
            )
            query = torch.as_tensor([entry.query], dtype=torch.long, device=device)
            prediction_z, prediction_y = model.predict(
                state, action.to(device=device, dtype=dtype), query
            )
            target_z = (
                latent_targets[entry.query] - normalization.latent_mean
            ) / normalization.latent_scale
            errors.append(
                endpoint_per_sample(
                    prediction_z,
                    prediction_y,
                    torch.as_tensor(target_z[None], dtype=dtype, device=device),
                    torch.as_tensor(
                        physical_targets[entry.query][None],
                        dtype=dtype,
                        device=device,
                    ),
                    1.0,
                )[0]
            )
            representations.append(state[0].mean(dim=0))
    return (
        torch.mean(torch.stack(representations), dim=0).cpu().numpy(),
        -float(torch.mean(torch.stack(errors)).cpu()),
    )


def _action_variant_oracle(model, states, queries, actions):
    latent = []
    physical = []
    device, dtype = _model_device_and_dtype(model)
    with torch.no_grad():
        for action in actions:
            action_latent = []
            action_physical = []
            for row, query in enumerate(queries):
                latent_value, physical_value = model.predict(
                    states[row : row + 1].to(device=device, dtype=dtype),
                    action[None].to(device=device, dtype=dtype),
                    torch.as_tensor([query], dtype=torch.long, device=device),
                )
                action_latent.append(latent_value)
                action_physical.append(physical_value)
            latent.append(torch.cat(action_latent, dim=0).cpu())
            physical.append(torch.cat(action_physical, dim=0).cpu())
    return torch.stack(latent), torch.stack(physical)


def benchmark_evaluation_batch(sample_count=32, batch_size=8, device="cpu"):
    model = _tiny_model(device)
    values = _fixture(sample_count)
    _per_sample_oracle(model, _fixture(1))
    _batched_outputs(model, _fixture(1), 1)

    execution_device = _model_device_and_dtype(model)[0]
    if execution_device.type == "cuda":
        torch.cuda.synchronize(execution_device)
    started = time.perf_counter()
    expected = _per_sample_oracle(model, values)
    if execution_device.type == "cuda":
        torch.cuda.synchronize(execution_device)
    oracle_seconds = time.perf_counter() - started
    started = time.perf_counter()
    actual = _batched_outputs(model, values, batch_size)
    if execution_device.type == "cuda":
        torch.cuda.synchronize(execution_device)
    batched_seconds = time.perf_counter() - started
    return {
        "device": str(execution_device),
        "sample_count": sample_count,
        "batch_size": batch_size,
        "oracle_seconds": oracle_seconds,
        "batched_seconds": batched_seconds,
        "speedup": oracle_seconds / max(batched_seconds, 1e-12),
        "oracle_rows_per_second": sample_count / max(oracle_seconds, 1e-12),
        "batched_rows_per_second": sample_count / max(batched_seconds, 1e-12),
        "max_state_abs_error": _max_abs_error(actual[0], expected[0]),
        "max_query_state_abs_error": _max_abs_error(actual[1], expected[1]),
        "max_latent_abs_error": _max_abs_error(actual[2], expected[2]),
        "max_physical_abs_error": _max_abs_error(actual[3], expected[3]),
    }


class EvaluationBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._original_threads)

    def test_cpu_nondivisible_batches_match_oracle_and_cache_encoders(self):
        model = _tiny_model()
        values = _fixture(7)
        expected = _per_sample_oracle(model, values)
        context_batches = []
        state_batches = []
        action_batches = []
        context_hook = model.map_encoder.register_forward_pre_hook(
            lambda _module, arguments: context_batches.append(int(arguments[0].shape[0]))
        )
        state_hook = model.csi_encoder.register_forward_pre_hook(
            lambda _module, arguments: state_batches.append(int(arguments[0].shape[0]))
        )
        action_hook = model.action_encoder.register_forward_pre_hook(
            lambda _module, arguments: action_batches.append(int(arguments[0].shape[0]))
        )
        try:
            actual = _batched_outputs(model, values, batch_size=2)
        finally:
            context_hook.remove()
            state_hook.remove()
            action_hook.remove()

        for actual_value, expected_value in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                actual_value, expected_value, rtol=1e-6, atol=2e-7
            )
        self.assertEqual(context_batches, [2, 1])
        self.assertEqual(state_batches, [2, 2, 2, 1])
        self.assertEqual(action_batches, [2, 1])

    def test_evaluator_cache_batch_one_is_bitwise_exact_and_preserves_order(self):
        model = _tiny_model()
        values = _fixture(4)
        entries = _mask_entries(values)
        patches = values["visible"][0:1]
        maps = values["maps"][0:1]
        radio = values["radio"][0:1]
        action = values["actions"][0:1]
        generator = np.random.default_rng(411)
        latent_targets = generator.normal(size=(4, 8))
        physical_targets = generator.normal(size=(4, 2))
        normalization = SimpleNamespace(
            latent_mean=generator.normal(size=8),
            latent_scale=np.exp(generator.normal(size=8)),
        )
        expected = _masked_alignment_oracle(
            model,
            patches,
            maps,
            radio,
            action,
            entries,
            latent_targets,
            physical_targets,
            normalization,
        )

        context_batches = []
        action_batches = []
        context_hook = model.map_encoder.register_forward_pre_hook(
            lambda _module, arguments: context_batches.append(int(arguments[0].shape[0]))
        )
        action_hook = model.action_encoder.register_forward_pre_hook(
            lambda _module, arguments: action_batches.append(int(arguments[0].shape[0]))
        )
        try:
            cache = {}
            context = _cached_encoded_context(
                model, cache, ("fixture", 0), maps, radio
            )
            self.assertIs(
                context,
                _cached_encoded_context(model, cache, ("fixture", 0), maps, radio),
            )
            encoded_action = encode_action_bank(model, action)
            actual = _masked_alignment_state_and_score(
                model,
                patches,
                maps,
                radio,
                action,
                entries,
                latent_targets,
                physical_targets,
                normalization,
                encoded_context=context,
                encoded_zero_action=encoded_action,
                batch_size=1,
            )
        finally:
            context_hook.remove()
            action_hook.remove()

        np.testing.assert_array_equal(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        self.assertEqual(context_batches, [1])
        self.assertEqual(action_batches, [1])
        self.assertEqual([entry.query for entry in entries], [3, 0, 2, 1])

        states, queries = _batched_masked_states(
            model, patches, entries, context, batch_size=1
        )
        variant_actions = values["actions"][:3]
        expected_variants = _action_variant_oracle(
            model, states, queries, variant_actions
        )
        encoded_variants = encode_action_bank(
            model, variant_actions, batch_size=1
        )
        actual_variants = _batched_action_variants(
            model,
            states,
            queries,
            encoded_variants,
            batch_size=1,
        )
        for actual_value, expected_value in zip(
            actual_variants, expected_variants, strict=True
        ):
            torch.testing.assert_close(
                actual_value, expected_value, rtol=0.0, atol=0.0
            )

    def _assert_native_fixed_action_encoding_parity(self, device):
        model = _tiny_model(device)
        action = _fixture(1)["actions"][0:1].cpu().numpy()
        expected = torch.cat(
            (
                encode_action_bank(model, action, batch_size=1),
                encode_action_bank(model, np.zeros_like(action), batch_size=1),
            ),
            dim=0,
        )
        encoded_inputs = []
        hook = model.action_encoder.register_forward_pre_hook(
            lambda _module, arguments: encoded_inputs.append(
                arguments[0].detach().cpu().clone()
            )
        )
        try:
            actual = _encode_native_fixed_actions(
                model,
                action,
                batch_size=1,
            )
        finally:
            hook.remove()

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual([int(values.shape[0]) for values in encoded_inputs], [1, 1])
        torch.testing.assert_close(
            encoded_inputs[0], torch.as_tensor(action), rtol=0.0, atol=0.0
        )
        self.assertEqual(int(torch.count_nonzero(encoded_inputs[1])), 0)

    def test_native_fixed_actions_encode_one_row_at_a_time_on_cpu(self):
        self._assert_native_fixed_action_encoding_parity("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA runtime is unavailable")
    def test_native_fixed_actions_encode_one_row_at_a_time_on_cuda(self):
        self._assert_native_fixed_action_encoding_parity("cuda:0")

    def test_evaluator_batched_path_matches_oracle_and_retains_exception(self):
        model = _tiny_model()
        values = _fixture(4)
        entries = _mask_entries(values)
        patches = values["visible"][0:1]
        maps = values["maps"][0:1]
        radio = values["radio"][0:1]
        action = values["actions"][0:1]
        generator = np.random.default_rng(877)
        latent_targets = generator.normal(size=(4, 8))
        physical_targets = generator.normal(size=(4, 2))
        normalization = SimpleNamespace(
            latent_mean=generator.normal(size=8),
            latent_scale=np.exp(generator.normal(size=8)),
        )
        expected = _masked_alignment_oracle(
            model,
            patches,
            maps,
            radio,
            action,
            entries,
            latent_targets,
            physical_targets,
            normalization,
        )
        actual = _masked_alignment_state_and_score(
            model,
            patches,
            maps,
            radio,
            action,
            entries,
            latent_targets,
            physical_targets,
            normalization,
            batch_size=3,
        )
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=2e-7)
        self.assertAlmostEqual(actual[1], expected[1], places=6)
        with self.assertRaisesRegex(
            RuntimeError, "alignment mask bank must cover every query exactly once"
        ):
            _masked_alignment_state_and_score(
                model,
                patches,
                maps,
                radio,
                action,
                entries[:-1],
                latent_targets,
                physical_targets,
                normalization,
                batch_size=2,
            )

    def test_wrappers_preserve_cached_results_and_input_order(self):
        model = _tiny_model()
        values = _fixture(7)
        expected = _batched_outputs(model, values, batch_size=3)
        states = batched_model_states(
            model,
            values["visible"],
            values["maps"],
            values["radio"],
            values["masks"],
            context_indices=values["context_indices"],
            batch_size=3,
        )
        query_states = batched_query_states(
            model,
            values["visible"],
            values["maps"],
            values["radio"],
            values["masks"],
            values["queries"],
            context_indices=values["context_indices"],
            batch_size=3,
        )
        contexts = encode_context_bank(model, values["maps"], values["radio"])
        cached_queries = batched_query_states_from_context(
            model,
            values["visible"],
            values["masks"],
            values["queries"],
            contexts,
            context_indices=values["context_indices"],
            batch_size=3,
        )
        latent, physical = batched_action_predictions(
            model,
            states,
            values["actions"],
            values["queries"],
            action_indices=values["action_indices"],
            batch_size=3,
        )
        actual = states, query_states, latent, physical
        for actual_value, expected_value in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_value, expected_value, rtol=0.0, atol=0.0)
        torch.testing.assert_close(cached_queries, query_states, rtol=0.0, atol=0.0)

    def test_empty_inputs_return_typed_empty_outputs(self):
        model = _tiny_model()
        values = _fixture(0)
        values["maps"] = values["maps"][:0]
        values["radio"] = values["radio"][:0]
        values["actions"] = values["actions"][:0]
        contexts = encode_context_bank(model, values["maps"], values["radio"])
        actions = encode_action_bank(model, values["actions"])
        states = batched_model_states_from_context(
            model,
            values["visible"],
            values["masks"],
            contexts,
            context_indices=values["context_indices"],
        )
        query_states = gather_query_states(states, values["queries"])
        latent, physical = batched_action_predictions_from_encoded(
            model,
            states,
            actions,
            values["queries"],
            action_indices=values["action_indices"],
        )
        self.assertEqual(tuple(contexts.shape), (0, 17, 8))
        self.assertEqual(tuple(actions.shape), (0, 17, 8))
        self.assertEqual(tuple(states.shape), (0, 4, 8))
        self.assertEqual(tuple(query_states.shape), (0, 8))
        self.assertEqual(tuple(latent.shape), (0, 8))
        self.assertEqual(tuple(physical.shape), (0, 2))
        for value in (contexts, actions, states, query_states, latent, physical):
            self.assertEqual(value.device.type, "cpu")

    def test_singleton_and_row_aligned_banks_need_no_explicit_indices(self):
        model = _tiny_model()
        values = _fixture(7)
        singleton_context = encode_context_bank(
            model, values["maps"][:1], values["radio"][:1]
        )
        implicit_singleton = batched_model_states_from_context(
            model,
            values["visible"],
            values["masks"],
            singleton_context,
            batch_size=3,
        )
        explicit_singleton = batched_model_states_from_context(
            model,
            values["visible"],
            values["masks"],
            singleton_context,
            context_indices=torch.zeros(7, dtype=torch.long),
            batch_size=3,
        )
        torch.testing.assert_close(
            implicit_singleton, explicit_singleton, rtol=0.0, atol=0.0
        )

        row_contexts = encode_context_bank(
            model,
            values["maps"][values["context_indices"]],
            values["radio"][values["context_indices"]],
            batch_size=3,
        )
        implicit_rows = batched_model_states_from_context(
            model,
            values["visible"],
            values["masks"],
            row_contexts,
            batch_size=3,
        )
        explicit_rows = batched_model_states_from_context(
            model,
            values["visible"],
            values["masks"],
            row_contexts,
            context_indices=torch.arange(7),
            batch_size=3,
        )
        torch.testing.assert_close(implicit_rows, explicit_rows, rtol=0.0, atol=0.0)

        singleton_action = encode_action_bank(model, values["actions"][:1])
        implicit_prediction = batched_action_predictions_from_encoded(
            model, implicit_singleton, singleton_action, values["queries"], batch_size=3
        )
        explicit_prediction = batched_action_predictions_from_encoded(
            model,
            implicit_singleton,
            singleton_action,
            values["queries"],
            action_indices=torch.zeros(7, dtype=torch.long),
            batch_size=3,
        )
        for implicit, explicit in zip(
            implicit_prediction, explicit_prediction, strict=True
        ):
            torch.testing.assert_close(implicit, explicit, rtol=0.0, atol=0.0)

    def test_batch_mask_query_and_bank_indices_are_fail_closed(self):
        model = _tiny_model()
        values = _fixture(7)
        contexts = encode_context_bank(model, values["maps"], values["radio"])
        with self.assertRaisesRegex(ValueError, "positive integer"):
            encode_context_bank(model, values["maps"], values["radio"], batch_size=0)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            encode_action_bank(model, values["actions"], batch_size=True)
        with self.assertRaisesRegex(ValueError, "share a row count"):
            encode_context_bank(model, values["maps"], values["radio"][:2])
        with self.assertRaisesRegex(ValueError, "boolean"):
            batched_model_states_from_context(
                model, values["visible"], values["masks"].long(), contexts
            )
        with self.assertRaisesRegex(ValueError, "masks must be boolean"):
            batched_model_states_from_context(
                model, values["visible"], values["masks"][:, :3], contexts
            )
        with self.assertRaisesRegex(ValueError, "context_indices.*integer"):
            batched_model_states_from_context(
                model,
                values["visible"],
                values["masks"],
                contexts,
                context_indices=values["context_indices"].float(),
            )
        invalid_context = values["context_indices"].clone()
        invalid_context[-1] = contexts.shape[0]
        with self.assertRaisesRegex(ValueError, "context_indices.*out-of-range"):
            batched_model_states_from_context(
                model,
                values["visible"],
                values["masks"],
                contexts,
                context_indices=invalid_context,
            )
        states = batched_model_states_from_context(
            model,
            values["visible"],
            values["masks"],
            contexts,
            context_indices=values["context_indices"],
        )
        with self.assertRaisesRegex(ValueError, "queries.*integer"):
            gather_query_states(states, values["queries"].float())
        with self.assertRaisesRegex(ValueError, "queries.*shape"):
            gather_query_states(states, values["queries"][:-1])
        invalid_query = values["queries"].clone()
        invalid_query[0] = model.patch_count
        with self.assertRaisesRegex(ValueError, "queries.*out-of-range"):
            gather_query_states(states, invalid_query)
        encoded_actions = encode_action_bank(model, values["actions"])
        invalid_action = values["action_indices"].clone()
        invalid_action[0] = encoded_actions.shape[0]
        with self.assertRaisesRegex(ValueError, "action_indices.*out-of-range"):
            batched_action_predictions_from_encoded(
                model,
                states,
                encoded_actions,
                values["queries"],
                action_indices=invalid_action,
            )
        with self.assertRaisesRegex(ValueError, "action_indices is required"):
            batched_action_predictions_from_encoded(
                model,
                states,
                encoded_actions[:2],
                values["queries"],
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA runtime is unavailable")
    def test_cuda_batches_match_per_sample_oracle(self):
        model = _tiny_model("cuda:0")
        values = _fixture(7)
        expected = _per_sample_oracle(model, values)
        actual = _batched_outputs(model, values, batch_size=3)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            self.assertEqual(actual_value.device.type, "cpu")
            torch.testing.assert_close(
                actual_value, expected_value, rtol=2e-6, atol=3e-7
            )

    def test_benchmark_reports_error_and_performance(self):
        report = benchmark_evaluation_batch(sample_count=7, batch_size=3)
        self.assertEqual(report["sample_count"], 7)
        self.assertEqual(report["batch_size"], 3)
        self.assertGreaterEqual(report["oracle_seconds"], 0.0)
        self.assertGreaterEqual(report["batched_seconds"], 0.0)
        self.assertGreater(report["speedup"], 0.0)
        for name in (
            "max_state_abs_error",
            "max_query_state_abs_error",
            "max_latent_abs_error",
            "max_physical_abs_error",
        ):
            self.assertLessEqual(report[name], 5e-7)


if __name__ == "__main__":
    unittest.main()
