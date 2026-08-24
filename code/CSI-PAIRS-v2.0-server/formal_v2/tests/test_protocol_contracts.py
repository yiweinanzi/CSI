from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from formal_v2.formal_dataset import (
    FormalDataset,
    FormalDatasetError,
    SCENE_ROLES,
    SOURCE_ROLES,
)
from formal_v2.formal_factorial import (
    INFORMATION_BUDGET_CHECKLIST_SCHEMA,
    _information_budget_checklist,
    run_formal_factorial,
)
from formal_v2.formal_model import CSIPairsFormalModel
from formal_v2.formal_protocol import PatchSpec
from formal_v2.formal_teacher import (
    CSIMaskedTeacher,
    CSIReadout,
    TeacherBundle,
    teacher_targets,
)


_TEACHER_FORBIDDEN = frozenset({"map", "maps", "position", "x", "action", "edit"})
_RECEIVER_FORBIDDEN = frozenset({"position", "x", "receiver"})
_TARGET_CSI_FORBIDDEN = frozenset({"csi", "target_csi", "h_v", "H_v", "hv"})


def _named_parameters(function):
    return [
        name
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    ]


class TeacherInvarianceTests(unittest.TestCase):
    def test_teacher_forward_and_targets_signatures_exclude_map_position_action(self):
        forward_names = _named_parameters(CSIMaskedTeacher.forward)
        self.assertEqual(forward_names, ["self", "patches", "mask"])
        self.assertFalse(_TEACHER_FORBIDDEN.intersection(forward_names))
        for parameter in inspect.signature(CSIMaskedTeacher.forward).parameters.values():
            self.assertIn(
                parameter.kind,
                {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD},
            )

        target_names = _named_parameters(teacher_targets)
        self.assertEqual(target_names, ["bundle", "csi"])
        self.assertFalse(_TEACHER_FORBIDDEN.intersection(target_names))
        for parameter in inspect.signature(teacher_targets).parameters.values():
            self.assertIn(
                parameter.kind,
                {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD},
            )

    def test_teacher_targets_depend_only_on_csi(self):
        spec = PatchSpec(2, 2, 1)
        teacher = CSIMaskedTeacher(2, 2, 2, 8, 2, 1, 1).eval()
        bundle = TeacherBundle(
            teacher=teacher,
            readout=CSIReadout(8, spec.patch_dim),
            patch_spec=spec,
            seed=0,
            pretrain_mask_bank=(),
            mask_bank=(),
            audit_mask_bank=(),
            channel_mean=np.zeros(2 * spec.complex_values, dtype=np.float64),
            channel_scale=np.ones(2 * spec.complex_values, dtype=np.float64),
            reconstruction_nmse=0.0,
            readout_nmse=0.0,
        )
        rng = np.random.default_rng(24)
        csi = rng.standard_normal((4, 2 * spec.complex_values))
        other = rng.standard_normal((4, 2 * spec.complex_values))
        first = teacher_targets(bundle, csi)
        second = teacher_targets(bundle, csi)
        different = teacher_targets(bundle, other)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))


class Section10ContractTests(unittest.TestCase):
    def _tiny_model(self):
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
        ).eval()

    def test_state_and_encode_context_signatures_exclude_receiver_position(self):
        state_names = _named_parameters(CSIPairsFormalModel.state)
        self.assertEqual(state_names, ["self", "visible_patches", "maps", "radio", "masks"])
        self.assertFalse(_RECEIVER_FORBIDDEN.intersection(state_names))
        context_names = _named_parameters(CSIPairsFormalModel.encode_context)
        self.assertEqual(context_names, ["self", "maps", "radio"])
        self.assertFalse(_RECEIVER_FORBIDDEN.intersection(context_names))

    def test_encode_context_map_token_layout_is_independent_of_receiver_x(self):
        model = self._tiny_model()
        maps = torch.randn(2, 3, 8, 8)
        radio = torch.randn(2, 4)
        with torch.no_grad():
            first = model.encode_context(maps, radio)
            second = model.encode_context(maps, radio)
        self.assertEqual(first.shape, second.shape)
        self.assertGreater(first.shape[1], 1)
        torch.testing.assert_close(first[:, :-1], second[:, :-1], rtol=0.0, atol=0.0)
        self.assertEqual(first[:, :-1].shape[1], second[:, :-1].shape[1])

    def test_state_and_predict_signatures_exclude_target_csi(self):
        state_names = _named_parameters(CSIPairsFormalModel.state)
        predict_names = _named_parameters(CSIPairsFormalModel.predict)
        self.assertEqual(state_names, ["self", "visible_patches", "maps", "radio", "masks"])
        self.assertEqual(predict_names, ["self", "state", "signed_edit", "query"])
        self.assertFalse(_TARGET_CSI_FORBIDDEN.intersection(state_names))
        self.assertFalse(_TARGET_CSI_FORBIDDEN.intersection(predict_names))

    def test_state_and_predict_allowlist_is_closed_without_target_csi(self):
        model = self._tiny_model()
        patches = torch.randn(2, 4, 2)
        maps = torch.randn(2, 3, 8, 8)
        radio = torch.randn(2, 4)
        masks = torch.zeros(2, 4, dtype=torch.bool)
        action = torch.zeros(2, 12, 8, 8)
        query = torch.tensor([0, 1])
        with torch.no_grad():
            first_state = model.state(patches, maps, radio, masks)
            first_prediction = model.predict(first_state, action, query)
            second_state = model.state(patches, maps, radio, masks)
            second_prediction = model.predict(second_state, action, query)
        torch.testing.assert_close(first_state, second_state, rtol=0.0, atol=0.0)
        torch.testing.assert_close(first_prediction[0], second_prediction[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(first_prediction[1], second_prediction[1], rtol=0.0, atol=0.0)

    def test_maps_validation_rejects_a_position_axis(self):
        forged = SimpleNamespace(
            csi_repeat=np.zeros((2, 2, 2, 2, 2)),
            maps=np.zeros((2, 2, 3, 3, 4, 4)),
        )
        with self.assertRaisesRegex(
            FormalDatasetError,
            r"maps must have shape \[scene, world, map_channel, row, column\]",
        ):
            FormalDataset.validate(forged)


class InformationBudgetChecklistTests(unittest.TestCase):
    def _rows(self):
        return _information_budget_checklist(
            {"localization": {"label_budgets": [0, 8, 32, 128]}}
        )

    def test_k0_row_has_no_position_labels_or_head_updates(self):
        row = next(item for item in self._rows() if item["label_budget"] == 0)
        self.assertFalse(row["uses_position_labels"])
        self.assertFalse(row["allows_position_head_param_updates"])

    def test_positive_k_rows_adapt_position_head_only(self):
        for row in self._rows():
            if row["label_budget"] == 0:
                continue
            self.assertTrue(row["uses_position_labels"])
            self.assertTrue(row["allows_position_head_param_updates"])
            self.assertFalse(row["allows_encoder_param_updates"])
            self.assertFalse(row["uses_target_norm_stats"])

    def test_every_budget_uses_natural_map_radio_and_unlabeled_target_csi(self):
        rows = self._rows()
        self.assertEqual([row["label_budget"] for row in rows], [0, 8, 32, 128])
        for row in rows:
            self.assertTrue(row["uses_map"])
            self.assertTrue(row["uses_bs_pose"])
            self.assertTrue(row["uses_unlabeled_target_csi"])
            self.assertFalse(row["uses_reference_library"])
            self.assertFalse(row["allows_encoder_param_updates"])
            self.assertFalse(row["uses_target_norm_stats"])
            self.assertEqual(row["schema_version"], INFORMATION_BUDGET_CHECKLIST_SCHEMA)

    def test_checklist_is_written_before_factorial_manifest(self):
        source = inspect.getsource(run_formal_factorial)
        self.assertLess(
            source.index("information_budget_checklist.json"),
            source.index("artifact_manifest(output_dir"),
        )


class UnseenEditAxisTests(unittest.TestCase):
    def test_scene_roles_do_not_invent_unseen_edit(self):
        # unseen-edit holdout is a generator/registry obligation; runtime roles are bank-level.
        self.assertNotIn("unseen_edit", SCENE_ROLES)
        self.assertEqual(len(SOURCE_ROLES), 7)
        self.assertEqual(SCENE_ROLES, SOURCE_ROLES + ("target", "external_validation"))
        self.assertEqual(
            SOURCE_ROLES,
            (
                "source_encoder_train",
                "source_method_selection",
                "source_probe_train",
                "source_probe_selection",
                "source_calibration_fit",
                "source_calibration_selection",
                "source_final_unseen_bank",
            ),
        )


if __name__ == "__main__":
    unittest.main()
