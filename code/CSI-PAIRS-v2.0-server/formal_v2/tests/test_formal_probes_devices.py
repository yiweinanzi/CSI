from __future__ import annotations

import copy
import unittest
from unittest import mock

import numpy as np
import torch

from formal_v2.formal_probes import (
    ActionResponseProbe,
    CompatibilityProbe,
    _fit_binary,
    fit_action_response_probe,
    fit_select_compatibility_probe,
    predict_binary_probe,
    predict_response_probe,
)


def _config(steps: int = 3) -> dict:
    return {
        "evaluation": {
            "probe_steps": int(steps),
            "probe_learning_rate": 0.001,
            "probe_hidden_dim": 8,
        }
    }


class ProbeDeviceTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20260831)
        self.features = rng.normal(size=(19, 7)).astype(np.float64)
        self.zero = rng.normal(size=(19, 7)).astype(np.float64)
        self.targets = rng.normal(size=(19, 3)).astype(np.float64)
        self.labels = np.asarray(([0, 1] * 10)[:19], dtype=np.int64)

    def test_default_cpu_path_is_exactly_preserved(self):
        default = fit_action_response_probe(
            self.features,
            self.targets,
            _config(),
            seed=41,
            zero_action_x=self.zero,
        )
        explicit = fit_action_response_probe(
            self.features,
            self.targets,
            _config(),
            seed=41,
            zero_action_x=self.zero,
            device="cpu",
        )
        for key, value in default.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[key]), key)
        np.testing.assert_array_equal(
            predict_response_probe(default, self.features, self.zero),
            predict_response_probe(explicit, self.features, self.zero),
        )

    def test_chunked_response_prediction_preserves_order(self):
        probe = fit_action_response_probe(
            self.features,
            self.targets,
            _config(),
            seed=42,
        )
        expected = predict_response_probe(probe, self.features)
        actual = predict_response_probe(probe, self.features, batch_rows=6)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_chunked_gradient_accumulation_matches_one_full_step(self):
        full = fit_action_response_probe(
            self.features,
            self.targets,
            _config(steps=1),
            seed=43,
            zero_action_x=self.zero,
        )
        chunked = fit_action_response_probe(
            self.features,
            self.targets,
            _config(steps=1),
            seed=43,
            zero_action_x=self.zero,
            train_batch_rows=6,
        )
        for key, value in full.state_dict().items():
            torch.testing.assert_close(
                chunked.state_dict()[key],
                value,
                rtol=2e-6,
                atol=2e-7,
                msg=lambda message, key=key: f"{key}: {message}",
            )

        torch.manual_seed(4301)
        full_binary = CompatibilityProbe(self.features.shape[1], "mlp2", 8)
        chunked_binary = copy.deepcopy(full_binary)
        _fit_binary(
            full_binary,
            self.features,
            self.labels,
            _config(steps=1),
        )
        _fit_binary(
            chunked_binary,
            self.features,
            self.labels,
            _config(steps=1),
            train_batch_rows=6,
        )
        for key, value in full_binary.state_dict().items():
            torch.testing.assert_close(
                chunked_binary.state_dict()[key],
                value,
                rtol=2e-6,
                atol=2e-7,
                msg=lambda message, key=key: f"binary {key}: {message}",
            )

    def test_compatibility_probe_supports_device_and_chunking(self):
        train_x = self.features[:15]
        train_y = self.labels[:15]
        selection_x = self.features[15:]
        selection_y = self.labels[15:]
        probe, record = fit_select_compatibility_probe(
            train_x,
            train_y,
            selection_x,
            selection_y,
            _config(steps=2),
            seed=44,
            device="cpu",
            train_batch_rows=5,
        )
        expected = predict_binary_probe(probe, selection_x)
        actual = predict_binary_probe(probe, selection_x, batch_rows=3)
        self.assertIn(record["selected_family"], {"linear", "mlp2"})
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_batch_rows_must_be_positive_integer(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            fit_action_response_probe(
                self.features,
                self.targets,
                _config(steps=1),
                seed=45,
                train_batch_rows=0,
            )
        probe = fit_action_response_probe(
            self.features,
            self.targets,
            _config(steps=1),
            seed=46,
        )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            predict_response_probe(probe, self.features, batch_rows=True)

    def test_mock_cuda_training_transfers_only_cpu_backed_batches(self):
        response_transfers = []

        def record_response_transfer(values, device):
            response_transfers.append(
                (int(values.shape[0]), values.device.type, str(device))
            )
            return values

        def keep_response_probe_on_cpu(probe, *_args, **_kwargs):
            return probe

        with (
            mock.patch(
                "formal_v2.formal_probes._resolve_probe_device",
                return_value=torch.device("cuda:0"),
            ),
            mock.patch.object(
                ActionResponseProbe,
                "to",
                autospec=True,
                side_effect=keep_response_probe_on_cpu,
            ),
            mock.patch(
                "formal_v2.formal_probes._transfer_float_batch",
                side_effect=record_response_transfer,
            ),
        ):
            response_probe = fit_action_response_probe(
                self.features,
                self.targets,
                _config(steps=1),
                seed=48,
                zero_action_x=self.zero,
                device="cuda:0",
                train_batch_rows=6,
            )

        self.assertEqual(len(response_transfers), 12)
        self.assertEqual(
            [rows for rows, _backing, _device in response_transfers],
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1],
        )
        self.assertTrue(
            all(backing == "cpu" for _rows, backing, _device in response_transfers)
        )
        self.assertTrue(
            all(device == "cuda:0" for _rows, _backing, device in response_transfers)
        )

        binary_transfers = []

        def record_binary_transfer(values, device):
            binary_transfers.append(
                (int(values.shape[0]), values.device.type, str(device))
            )
            return values

        binary_probe = CompatibilityProbe(
            self.features.shape[1],
            "linear",
            _config()["evaluation"]["probe_hidden_dim"],
        )
        with (
            mock.patch(
                "formal_v2.formal_probes._module_device",
                return_value=torch.device("cuda:0"),
            ),
            mock.patch(
                "formal_v2.formal_probes._transfer_float_batch",
                side_effect=record_binary_transfer,
            ),
        ):
            _fit_binary(
                binary_probe,
                self.features,
                self.labels,
                _config(steps=1),
                train_batch_rows=5,
            )

        self.assertEqual(len(binary_transfers), 8)
        self.assertEqual(
            [rows for rows, _backing, _device in binary_transfers],
            [5, 5, 5, 5, 5, 5, 4, 4],
        )
        self.assertTrue(
            all(backing == "cpu" for _rows, backing, _device in binary_transfers)
        )
        self.assertTrue(
            all(device == "cuda:0" for _rows, _backing, device in binary_transfers)
        )

        expected_response = predict_response_probe(
            response_probe, self.features, self.zero
        )
        expected_binary = predict_binary_probe(binary_probe, self.features)
        prediction_transfers = []

        def record_prediction_transfer(values, device):
            prediction_transfers.append(
                (int(values.shape[0]), values.device.type, str(device))
            )
            return values

        with (
            mock.patch(
                "formal_v2.formal_probes._module_device",
                return_value=torch.device("cuda:0"),
            ),
            mock.patch(
                "formal_v2.formal_probes._transfer_float_batch",
                side_effect=record_prediction_transfer,
            ),
        ):
            actual_response = predict_response_probe(
                response_probe,
                self.features,
                self.zero,
                batch_rows=7,
            )
            actual_binary = predict_binary_probe(
                binary_probe,
                self.features,
                batch_rows=7,
            )

        self.assertEqual(
            [rows for rows, _backing, _device in prediction_transfers],
            [7, 7, 7, 7, 5, 5, 7, 7, 5],
        )
        self.assertTrue(
            all(
                backing == "cpu"
                for _rows, backing, _device in prediction_transfers
            )
        )
        self.assertTrue(
            all(
                device == "cuda:0"
                for _rows, _backing, device in prediction_transfers
            )
        )
        np.testing.assert_allclose(
            actual_response, expected_response, rtol=1e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            actual_binary, expected_binary, rtol=1e-6, atol=1e-7
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_probe_path_returns_cpu_numpy_in_input_order(self):
        probe = fit_action_response_probe(
            self.features,
            self.targets,
            _config(steps=1),
            seed=47,
            zero_action_x=self.zero,
            device="cuda:0",
            train_batch_rows=7,
        )
        values = predict_response_probe(
            probe,
            self.features,
            self.zero,
            batch_rows=5,
        )
        self.assertEqual(values.shape, self.targets.shape)
        self.assertIsInstance(values, np.ndarray)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertEqual(next(probe.parameters()).device, torch.device("cuda:0"))
        del probe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
