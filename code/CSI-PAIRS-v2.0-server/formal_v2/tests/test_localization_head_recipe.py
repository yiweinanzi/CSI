from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import numpy as np
import torch

from formal_v2.external_adapters import resource_control
from formal_v2.formal_localization import (
    HeteroscedasticPositionHead,
    _optimize_head,
    adapt_position_head,
    scheduled_head_steps,
)


class LocalizationHeadRecipeTests(unittest.TestCase):
    def _config(self, **overrides):
        localization = {
            "head_steps": 2000,
            "head_steps_per_labeled_point": 50,
            "early_stop_patience": 50,
            "learning_rate": 0.01,
            "sigma_min": 1e-3,
            "ridge": 0.01,
        }
        localization.update(overrides)
        return {"localization": localization, "model": {"hidden_dim": 16}}

    def test_ridge_reaches_adamw(self):
        captured = {}
        real = torch.optim.AdamW

        def factory(*args, **kwargs):
            captured.update(kwargs)
            return real(*args, **kwargs)

        head = HeteroscedasticPositionHead(4, 8)
        x = np.zeros((6, 4), dtype=np.float32)
        y = np.zeros((6, 2), dtype=np.float32)
        with patch("formal_v2.formal_localization.torch.optim.AdamW", side_effect=factory):
            taken = _optimize_head(
                head,
                x,
                y,
                steps=2,
                learning_rate=0.01,
                sigma_min=1e-3,
                ridge=0.01,
                early_stop_patience=0,
            )
        self.assertEqual(taken, 2)
        self.assertEqual(captured["weight_decay"], 0.01)
        self.assertEqual(captured["lr"], 0.01)

    def test_ridge_shrinks_head_weights_versus_unregularized(self):
        x = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(8, 2)
        x = np.concatenate((x, x[:, :1], x[:, 1:]), axis=1)
        y = np.stack((x[:, 0], x[:, 1]), axis=1)

        def weight_norm(ridge):
            torch.manual_seed(7)
            head = HeteroscedasticPositionHead(4, 8)
            _optimize_head(
                head,
                x,
                y,
                steps=25,
                learning_rate=0.05,
                sigma_min=1e-3,
                ridge=ridge,
                early_stop_patience=0,
            )
            return float(
                sum(parameter.detach().pow(2).sum() for parameter in head.parameters())
            )

        self.assertLess(weight_norm(1.0), weight_norm(0.0))

    def test_small_k_step_cap(self):
        config = self._config()
        self.assertEqual(scheduled_head_steps(config, 8), 400)
        self.assertEqual(scheduled_head_steps(config, 0), 0)
        self.assertEqual(scheduled_head_steps(config, 200), 2000)
        source = HeteroscedasticPositionHead(4, 8)
        support_x = np.zeros((8, 4), dtype=np.float32)
        support_y = np.zeros((8, 2), dtype=np.float32)
        with patch("formal_v2.formal_localization._optimize_head") as spy:
            adapt_position_head(source, support_x, support_y, config)
        self.assertEqual(spy.call_args.kwargs["steps"], 400)
        self.assertEqual(spy.call_args.kwargs["ridge"], 0.01)
        self.assertEqual(spy.call_args.kwargs["early_stop_patience"], 50)

    def test_k0_does_not_update_the_target_head(self):
        source = HeteroscedasticPositionHead(4, 8)
        empty_x = np.empty((0, 4), dtype=np.float32)
        empty_y = np.empty((0, 2), dtype=np.float32)
        with patch("formal_v2.formal_localization._optimize_head") as spy:
            adapted = adapt_position_head(source, empty_x, empty_y, self._config())
        spy.assert_not_called()
        for name, expected in source.state_dict().items():
            torch.testing.assert_close(
                adapted.state_dict()[name], expected, rtol=0.0, atol=0.0
            )

    def test_resource_control_uses_the_same_head_recipe(self):
        source = inspect.getsource(resource_control._fit_source_localizer)
        self.assertIn("weight_decay", source)
        self.assertIn('["ridge"]', source)
        self.assertIn("scheduled_head_steps", source)


if __name__ == "__main__":
    unittest.main()
