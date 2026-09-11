from __future__ import annotations

from formal_v2.tests.platform_support import symlink_or_skip

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from formal_v2.formal_baselines import (
    RidgeRegressor,
    RidgeSufficientStatistics,
    ZeroPreservingRidgeRegressor,
    ZeroPreservingRidgeSufficientStatistics,
)
from formal_v2.formal_config import load_formal_config, validate_formal_config
from formal_v2.formal_dataset import FormalDataset, FormalDatasetError
from formal_v2.formal_fixture import write_nonscientific_fixture
from formal_v2.formal_features import (
    assemble_protocol_response_feature_matrix,
    assemble_protocol_response_features,
    multichannel_spatial_features,
    protocol_response_features,
)
from formal_v2.formal_protocol import (
    PatchSpec,
    delay_angle_power,
    frozen_mask_query_bank,
    patchify_csi,
    typed_signed_edit,
)
from formal_v2.formal_probes import (
    fit_action_response_probe,
    predict_response_probe,
)
from formal_v2.formal_response_probe import (
    CoordinateResponseProbe,
    ProbeTrainingConfig,
    coordinate_preserving_action_features,
    coordinate_preserving_map_features,
    fit_coordinate_response_probe,
    predict_coordinate_response_probe,
)
from formal_v2.formal_qualification import (
    QUALIFICATION_FINAL_ARTIFACT_NAMES,
    _action_geometry_profile,
    _build_records,
    _build_wrong_action_plan,
    _frozen_readout_alignment_distances,
    _iter_qualification_record_batches,
    _load_resumed_teacher,
    _prepare_qualification_output,
    _response_gate_rows,
    _reserve_qualification_probe_attempt,
    _route_noise_floor_rows,
    _route_coverage_passed,
    _select_wrong_action,
    _teacher_resume_receipt,
    _wrong_action_distance,
    qualification_blocking_scenes,
)
from formal_v2.formal_routing import (
    CompactEdgeTensorMapping,
    fit_route_normalization,
    route_dataset,
)
from formal_v2.formal_teacher import (
    CSIMaskedTeacher,
    CSIReadout,
    TEACHER_CHECKPOINT_SCHEMA,
    _encode_full_in_batches,
    _full_batch_readout_step,
    _masked_reconstruction_nmse_in_batches,
    load_teacher_bundle,
    normalized_teacher_patches,
    random_teacher_pretraining_masks,
    save_teacher_bundle,
    teacher_targets,
    train_teacher_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "formal_v2_smoke.json"


def _archive_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


class DatasetIdentityAndNoiseTests(unittest.TestCase):
    def test_teacher_sensitivity_uses_frozen_readout_recoverable_csi(self) -> None:
        latent = np.asarray(
            [
                [[[[0.0, 0.0]]], [[[1.0, 2.0]]]],
                [[[[3.0, 4.0]]], [[[4.0, 6.0]]]],
            ],
            dtype=np.float32,
        ).reshape(2, 2, 1, 2)
        readout = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            readout.weight.copy_(torch.eye(2))
        edge = SimpleNamespace(source_world=0, target_world=1)
        keys = {
            (7, 0, 1, 0): (99.0, 0.0),
            (7, 0, 1, 1): (99.0, 0.0),
        }
        dataset = SimpleNamespace(directed_edges=lambda _scene: (edge,))
        bundle = SimpleNamespace(readout=readout)
        routed = SimpleNamespace(
            teacher_latent={7: latent}, alignment_distances=keys
        )

        observed = _frozen_readout_alignment_distances(
            dataset, bundle, routed, 7, batch_rows=1
        )

        self.assertAlmostEqual(observed[(7, 0, 1, 0)], 5.0 / np.sqrt(2.0))
        self.assertAlmostEqual(observed[(7, 0, 1, 1)], 5.0 / np.sqrt(2.0))

    def test_ridge_mean_loss_is_invariant_to_duplicated_rows(self) -> None:
        features = np.asarray(
            [[0.0, 1.0], [1.0, -1.0], [2.0, 0.5], [3.0, 2.0]],
            dtype=np.float64,
        )
        targets = np.asarray([[0.5], [1.0], [-0.25], [2.0]], dtype=np.float64)
        original = RidgeRegressor(alpha=0.2).fit(features, targets)
        repeated = RidgeRegressor(alpha=0.2).fit(
            np.tile(features, (5, 1)), np.tile(targets, (5, 1))
        )
        probe = np.asarray([[0.25, 0.75], [2.5, -0.5]], dtype=np.float64)
        np.testing.assert_allclose(
            original.predict(probe), repeated.predict(probe), rtol=1e-12, atol=1e-12
        )

    def test_streaming_ridge_matches_materialized_fit(self) -> None:
        rng = np.random.default_rng(9181)
        features = rng.normal(size=(173, 19))
        zero = rng.normal(size=features.shape)
        targets = rng.normal(size=(173, 4))
        batches = np.array_split(np.arange(features.shape[0]), 11)

        expected = RidgeRegressor(alpha=0.03).fit(features, targets)
        streaming = RidgeSufficientStatistics(alpha=0.03)
        for rows in batches:
            streaming.update(features[rows], targets[rows])
        actual = streaming.finalize()
        np.testing.assert_allclose(
            actual.predict(features), expected.predict(features), rtol=2e-12, atol=2e-12
        )

        expected_zero = ZeroPreservingRidgeRegressor(alpha=0.03).fit(
            features, zero, targets
        )
        streaming_zero = ZeroPreservingRidgeSufficientStatistics(alpha=0.03)
        for rows in batches:
            streaming_zero.update(features[rows], zero[rows], targets[rows])
        actual_zero = streaming_zero.finalize()
        np.testing.assert_allclose(
            actual_zero.predict_contrast(features, zero),
            expected_zero.predict_contrast(features, zero),
            rtol=2e-12,
            atol=2e-12,
        )

    def test_batched_response_features_are_row_exact(self) -> None:
        rng = np.random.default_rng(9182)
        count, patches, width = 7, 5, 4
        visible = rng.normal(size=(count, patches, width))
        masks = rng.random((count, patches)) > 0.4
        queries = np.arange(count) % patches
        map_features = rng.normal(size=23)
        action_features = rng.normal(size=17)
        radio = rng.normal(size=9)
        position = rng.normal(size=2)
        expected = np.vstack(
            [
                assemble_protocol_response_features(
                    visible[row],
                    map_features,
                    action_features,
                    radio,
                    masks[row],
                    int(queries[row]),
                    position=position,
                )
                for row in range(count)
            ]
        )
        actual = assemble_protocol_response_feature_matrix(
            visible,
            map_features,
            action_features,
            radio,
            masks,
            queries,
            position=position,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_zero_preserving_ridge_has_exact_null_and_same_model_swap(self) -> None:
        rng = np.random.default_rng(9171)
        zero = rng.normal(size=(32, 7))
        action = zero + rng.normal(size=(32, 7))
        target = (action - zero) @ rng.normal(size=(7, 3))
        model = ZeroPreservingRidgeRegressor(alpha=1e-6).fit(
            action, zero, target
        )
        np.testing.assert_array_equal(model.predict_contrast(zero, zero), 0.0)
        swapped = zero + np.roll(action - zero, 1, axis=0)
        self.assertEqual(model.predict_contrast(swapped, zero).shape, target.shape)

    def test_neural_response_probe_uses_exact_zero_action_contrast(self) -> None:
        config = load_formal_config(SMOKE_CONFIG)
        rng = np.random.default_rng(9172)
        zero = rng.normal(size=(16, 6))
        action = zero + rng.normal(size=(16, 6))
        target = rng.normal(size=(16, 2))
        probe = fit_action_response_probe(
            action, target, config, seed=9172, zero_action_x=zero
        )
        np.testing.assert_array_equal(
            predict_response_probe(probe, zero, zero), np.zeros_like(target)
        )
        with self.assertRaisesRegex(ValueError, "requires zero-action"):
            predict_response_probe(probe, action)

    def test_coordinate_action_features_preserve_location_and_direction(self) -> None:
        first = np.zeros((14, 16, 16), dtype=np.float64)
        second = np.zeros_like(first)
        reverse = np.zeros_like(first)
        first[8, 2:5, 3:7] = 1.0  # source concrete
        first[10, 2:5, 3:7] = 1.0  # target glass
        second[8, 10:13, 9:13] = 1.0
        second[10, 10:13, 9:13] = 1.0
        reverse[5, 2:5, 3:7] = 1.0  # source glass
        reverse[13, 2:5, 3:7] = 1.0  # target concrete
        first_features = coordinate_preserving_action_features(first)
        self.assertFalse(
            np.array_equal(first_features, coordinate_preserving_action_features(second))
        )
        self.assertFalse(
            np.array_equal(first_features, coordinate_preserving_action_features(reverse))
        )

    def test_coordinate_map_features_preserve_semantic_grid(self) -> None:
        source = np.zeros((3, 16, 16), dtype=np.float64)
        moved = np.zeros_like(source)
        source[:, 2:4, 4:6] = np.asarray([1.0, 18.0, 4.0])[:, None, None]
        moved[:, 10:12, 8:10] = np.asarray([1.0, 18.0, 4.0])[:, None, None]
        first = coordinate_preserving_map_features(
            source, material_categories=5, pooled_size=8
        )
        second = coordinate_preserving_map_features(
            moved, material_categories=5, pooled_size=8
        )
        self.assertEqual(first.shape, second.shape)
        self.assertFalse(np.array_equal(first, second))

    def test_coordinate_probe_is_exactly_zero_and_trains_by_bank_route(self) -> None:
        rng = np.random.default_rng(8124)
        count = 24
        arrays = {
            "csi": rng.normal(size=(count, 8)).astype(np.float32),
            "map": rng.normal(size=(count, 6)).astype(np.float32),
            "action": rng.normal(size=(count, 5)).astype(np.float32),
            "zero_action": np.zeros((count, 5), dtype=np.float32),
            "context": rng.normal(size=(count, 3)).astype(np.float32),
            "query": rng.normal(size=(count, 4)).astype(np.float32),
            "position": rng.normal(size=(count, 2)).astype(np.float32),
        }
        arrays["target"] = (
            arrays["action"][:, :2] + 0.1 * arrays["csi"][:, :2]
        ).astype(np.float32)
        probe = CoordinateResponseProbe(
            csi_dim=8,
            map_dim=6,
            action_dim=5,
            context_dim=3,
            query_dim=4,
            output_dim=2,
            hidden_dim=16,
        )
        receipt = fit_coordinate_response_probe(
            probe,
            arrays,
            bank=np.repeat(np.arange(2), count // 2),
            route=np.tile(np.asarray([0, 2]), count // 2),
            seed=8124,
            config=ProbeTrainingConfig(steps=2, batch_size=8, hidden_dim=16),
            device="cpu",
        )
        self.assertTrue(receipt["losses"])
        zero = dict(arrays)
        zero["action"] = zero["zero_action"]
        prediction = predict_coordinate_response_probe(
            probe, zero, device="cpu", batch_size=7
        )
        np.testing.assert_array_equal(prediction, np.zeros_like(prediction))

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = write_nonscientific_fixture(self.root / "fixture.npz")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_canonical_foundation_digest_is_content_based_and_stable(self) -> None:
        first = FormalDataset.load(self.path)
        second = FormalDataset.load(self.path)
        np.testing.assert_array_equal(
            first.canonical_base_map_digests,
            second.canonical_base_map_digests,
        )
        self.assertEqual(first.canonical_base_map_digests.shape, (first.scene_count,))
        self.assertTrue(all(len(value) == 64 for value in first.canonical_base_map_digests))

    def test_formal_profile_rejects_foundation_content_spoofed_across_splits(self) -> None:
        dataset = FormalDataset.load(self.path)
        dataset.metadata["fixture"] = False
        dataset.metadata["scientific_use"] = "CANDIDATE"
        with self.assertRaisesRegex(FormalDatasetError, "canonical foundation content"):
            dataset.validate()

    def test_one_copied_observation_residual_is_rejected(self) -> None:
        arrays = _archive_arrays(self.path)
        clean = arrays["csi_clean"]
        residual = arrays["csi_repeat"] - clean[:, :, :, None, :]
        residual[0, 1, 2, 1] = residual[0, 0, 2, 1]
        arrays["csi_repeat"] = clean[:, :, :, None, :] + residual
        malformed = self.root / "one-copied-observation.npz"
        np.savez_compressed(malformed, **arrays)
        with self.assertRaisesRegex(FormalDatasetError, "sibling observations"):
            FormalDataset.load(malformed)

    def test_scaled_copy_of_one_observation_residual_is_rejected(self) -> None:
        arrays = _archive_arrays(self.path)
        clean = arrays["csi_clean"]
        residual = arrays["csi_repeat"] - clean[:, :, :, None, :]
        residual[0, 1, 2, 1] = 1.01 * residual[0, 0, 2, 1]
        arrays["csi_repeat"] = clean[:, :, :, None, :] + residual
        malformed = self.root / "scaled-copied-observation.npz"
        np.savez_compressed(malformed, **arrays)
        with self.assertRaisesRegex(FormalDatasetError, "near-perfectly correlate"):
            FormalDataset.load(malformed)

    def test_noise_binding_digest_changes_for_one_observation(self) -> None:
        dataset = FormalDataset.load(self.path)
        repeated = dataset.csi_repeat[0].copy()
        repeated[1, 2, 1, 0] += 1e-6
        self.assertNotEqual(
            dataset.observation_noise_binding_digest(0),
            dataset.observation_noise_binding_digest(0, repeated),
        )

    def test_phase_invariant_gauge_is_fail_closed_for_raw_complex_profile(self) -> None:
        arrays = _archive_arrays(self.path)
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["representation"]["phase_gauge_rule"] = "phase_invariant_delay_angle_power"
        arrays["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        )
        malformed = self.root / "unsupported-gauge.npz"
        np.savez_compressed(malformed, **arrays)
        with self.assertRaisesRegex(FormalDatasetError, "phase-invariant targets are not implemented"):
            FormalDataset.load(malformed)

    def test_phase_reference_fields_are_complex_world_independent_and_sourced(self) -> None:
        dataset = FormalDataset.load(self.path)
        expected_shape = (dataset.scene_count, dataset.position_count)
        self.assertEqual(dataset.phase_reference_values.shape, expected_shape)
        self.assertEqual(dataset.phase_reference_values.dtype, np.dtype(np.complex128))
        self.assertTrue(np.all(np.abs(dataset.phase_reference_values) > 0.0))
        self.assertEqual(dataset.phase_reference_source_sha256.shape, expected_shape)
        self.assertTrue(
            all(len(str(value)) == 64 for value in dataset.phase_reference_source_sha256.flat)
        )

    def test_phase_reference_fields_reject_a_per_world_axis(self) -> None:
        for field in ("phase_reference_values", "phase_reference_source_sha256"):
            with self.subTest(field=field):
                arrays = _archive_arrays(self.path)
                arrays[field] = np.repeat(
                    arrays[field][:, None, :],
                    arrays["world_bits"].shape[0],
                    axis=1,
                )
                malformed = self.root / f"per-world-{field}.npz"
                np.savez_compressed(malformed, **arrays)
                with self.assertRaisesRegex(
                    FormalDatasetError,
                    rf"{field} must have world-independent shape",
                ):
                    FormalDataset.load(malformed)

    def test_phase_reference_values_reject_zero_or_noncomplex_storage(self) -> None:
        arrays = _archive_arrays(self.path)
        arrays["phase_reference_values"] = arrays["phase_reference_values"].copy()
        arrays["phase_reference_values"][0, 0] = 0.0j
        malformed = self.root / "zero-phase-reference.npz"
        np.savez_compressed(malformed, **arrays)
        with self.assertRaisesRegex(FormalDatasetError, "finite nonzero complex references"):
            FormalDataset.load(malformed)

        arrays = _archive_arrays(self.path)
        arrays["phase_reference_values"] = arrays["phase_reference_values"].real
        malformed = self.root / "real-only-phase-reference.npz"
        np.savez_compressed(malformed, **arrays)
        with self.assertRaisesRegex(FormalDatasetError, "complex numeric dtype"):
            FormalDataset.load(malformed)

    def test_formal_patch_grid_must_support_exact_seventy_five_percent_masks(self) -> None:
        arrays = _archive_arrays(self.path)
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["representation"]["patch_complex_size"] = 4
        metadata["representation"]["patch_antenna_size"] = 1
        metadata["representation"]["patch_subcarrier_size"] = 4
        arrays["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        )
        malformed = self.root / "inexact-mask-cardinality.npz"
        np.savez_compressed(malformed, **arrays)
        with self.assertRaisesRegex(FormalDatasetError, "exact 75% masking"):
            FormalDataset.load(malformed)


class FrozenMaskContractTests(unittest.TestCase):
    def test_teacher_position_encoding_is_fixed_and_persistent(self) -> None:
        model = CSIMaskedTeacher(2, 4, 2, 8, 2, 1, 1)
        self.assertNotIn("fixed_position", dict(model.named_parameters()))
        self.assertIn("fixed_position", dict(model.named_buffers()))
        self.assertIn("fixed_position", model.state_dict())
        self.assertFalse(model.fixed_position.requires_grad)

    def test_batched_teacher_encoder_matches_the_per_sample_reference(self) -> None:
        torch.manual_seed(9181)
        reference = CSIMaskedTeacher(2, 2, 2, 8, 2, 1, 1)
        batched = copy.deepcopy(reference)
        reference_input = torch.randn(4, 4, 2, requires_grad=True)
        batched_input = reference_input.detach().clone().requires_grad_(True)
        mask = torch.tensor(
            [
                [False, True, True, True],
                [False, False, True, True],
                [False, False, False, True],
                [False, False, False, False],
            ],
            dtype=torch.bool,
        )

        tokens = reference.patch_embedding(reference_input) + reference.positional_encoding()
        reference_rows = []
        for batch_index in range(tokens.shape[0]):
            visible = ~mask[batch_index]
            encoded = reference.encoder(tokens[batch_index : batch_index + 1, visible])
            full = reference.mask_token[None, :].expand(reference.patch_count, -1).clone()
            full[visible] = encoded[0]
            reference_rows.append(full)
        reference_latent = torch.stack(reference_rows)
        reference_prediction = reference.decoder_head(
            reference.decoder_norm(
                reference.decoder_transformer(
                    reference_latent + reference.positional_encoding()
                )
            )
        )
        batched_latent, batched_prediction = batched(batched_input, mask)

        torch.testing.assert_close(
            batched_latent, reference_latent, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            batched_prediction, reference_prediction, rtol=1e-5, atol=1e-6
        )
        reference_prediction.square().mean().backward()
        batched_prediction.square().mean().backward()
        torch.testing.assert_close(
            batched_input.grad, reference_input.grad, rtol=1e-5, atol=1e-6
        )
        for (reference_name, reference_parameter), (batched_name, batched_parameter) in zip(
            reference.named_parameters(), batched.named_parameters(), strict=True
        ):
            self.assertEqual(batched_name, reference_name)
            torch.testing.assert_close(
                batched_parameter.grad,
                reference_parameter.grad,
                rtol=1e-5,
                atol=1e-6,
            )

        with self.assertRaisesRegex(ValueError, "cannot hide every patch"):
            batched(torch.zeros(1, 4, 2), torch.ones(1, 4, dtype=torch.bool))

    def test_teacher_full_dataset_inference_is_chunk_equivalent(self) -> None:
        torch.manual_seed(9183)
        model = CSIMaskedTeacher(2, 2, 2, 8, 2, 1, 1).eval()
        values = torch.randn(7, 4, 2)
        mask_bank = frozen_mask_query_bank(PatchSpec(2, 2, 1), 9183)
        masks = torch.as_tensor(
            np.stack([mask_bank[index % len(mask_bank)].mask for index in range(7)]),
            dtype=torch.bool,
        )
        with torch.no_grad():
            expected_latent = model.encode_full(values)
            _, expected_reconstruction = model(values, masks)
            expected_nmse = float(
                (
                    torch.sum(
                        (values[masks] - expected_reconstruction[masks]) ** 2
                    )
                    / torch.sum(values[masks] ** 2).clamp_min(1e-12)
                ).item()
            )
            observed_latent = _encode_full_in_batches(
                model, values, batch_size=2
            )
            observed_nmse = _masked_reconstruction_nmse_in_batches(
                model, values, mask_bank, batch_size=2
            )
        torch.testing.assert_close(observed_latent, expected_latent)
        self.assertAlmostEqual(observed_nmse, expected_nmse, places=6)

    def test_chunked_readout_step_preserves_full_batch_objective(self) -> None:
        torch.manual_seed(9184)
        direct = CSIReadout(8, 2)
        chunked = copy.deepcopy(direct)
        latent = torch.randn(11, 4, 8)
        target = torch.randn(11, 4, 2)
        direct_optimizer = torch.optim.AdamW(direct.parameters(), lr=3e-4)
        chunked_optimizer = torch.optim.AdamW(chunked.parameters(), lr=3e-4)

        direct_optimizer.zero_grad(set_to_none=True)
        direct_loss = torch.mean((direct(latent) - target) ** 2)
        direct_loss.backward()
        direct_optimizer.step()
        observed_loss = _full_batch_readout_step(
            chunked,
            chunked_optimizer,
            latent,
            target,
            chunk_size=3,
        )

        self.assertAlmostEqual(observed_loss, float(direct_loss), places=6)
        for direct_value, chunked_value in zip(
            direct.parameters(), chunked.parameters(), strict=True
        ):
            torch.testing.assert_close(
                chunked_value, direct_value, rtol=1e-6, atol=1e-7
            )

    def test_teacher_pretraining_masks_are_random_per_sample_and_reproducible(self) -> None:
        first = random_teacher_pretraining_masks(
            np.random.default_rng(9182),
            batch_size=32,
            patch_count=8,
            fraction=0.75,
        )
        second = random_teacher_pretraining_masks(
            np.random.default_rng(9182),
            batch_size=32,
            patch_count=8,
            fraction=0.75,
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (32, 8))
        np.testing.assert_array_equal(first.sum(axis=1), np.full(32, 6))
        self.assertGreater(np.unique(first, axis=0).shape[0], 1)
        with self.assertRaisesRegex(ValueError, "exact mask cardinality"):
            random_teacher_pretraining_masks(
                np.random.default_rng(1),
                batch_size=2,
                patch_count=6,
                fraction=0.75,
            )

    def test_teacher_training_resamples_masks_at_every_step(self) -> None:
        config = load_formal_config(SMOKE_CONFIG)
        spec = PatchSpec(2, 4, 1)
        csi = np.arange(4 * 16, dtype=np.float64).reshape(4, 16) / 100.0
        with patch(
            "formal_v2.formal_teacher.random_teacher_pretraining_masks",
            wraps=random_teacher_pretraining_masks,
        ) as sampler, patch(
            "formal_v2.formal_teacher._masked_reconstruction_nmse_in_batches",
            wraps=_masked_reconstruction_nmse_in_batches,
        ) as audit:
            train_teacher_bundle(csi, spec, config, seed=33)
        self.assertEqual(sampler.call_count, config["teacher"]["steps"])
        self.assertEqual(audit.call_count, 1)
        observed_audit_bank = audit.call_args.args[2]
        expected_audit_bank = tuple(
            entry
            for entry in frozen_mask_query_bank(
                spec, int(config["model"]["mask_bank_seed"]) + 1
            )
            if entry.mode == "random_75"
        )
        self.assertTrue(observed_audit_bank)
        self.assertTrue(all(entry.mode == "random_75" for entry in observed_audit_bank))
        for observed, expected in zip(
            observed_audit_bank, expected_audit_bank, strict=True
        ):
            np.testing.assert_array_equal(observed.mask, expected.mask)

    def test_teacher_checkpoint_authenticates_the_complete_mask_contract(self) -> None:
        config = load_formal_config(SMOKE_CONFIG)
        spec = PatchSpec(2, 4, 1)
        csi = np.arange(4 * 16, dtype=np.float64).reshape(4, 16) / 100.0
        bundle = train_teacher_bundle(csi, spec, config, seed=34)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "teacher.pt"
            save_teacher_bundle(checkpoint, bundle, config, seed=34)
            loaded = load_teacher_bundle(checkpoint, config)
            for name, value in bundle.teacher.state_dict().items():
                torch.testing.assert_close(value, loaded.teacher.state_dict()[name])
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["pretraining_mask_sampler"][
                "resampled_each_optimization_step"
            ] = False
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "exact V6 pretraining mask contract"):
                load_teacher_bundle(checkpoint, config)

    def test_teacher_checkpoint_rejects_pre_fixed_position_schema(self) -> None:
        config = load_formal_config(SMOKE_CONFIG)
        spec = PatchSpec(2, 4, 1)
        csi = np.arange(4 * 16, dtype=np.float64).reshape(4, 16) / 100.0
        bundle = train_teacher_bundle(csi, spec, config, seed=39)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "teacher.pt"
            save_teacher_bundle(checkpoint, bundle, config, seed=39)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["schema_version"], TEACHER_CHECKPOINT_SCHEMA)
            payload["schema_version"] = "csi-pairs-stage0-teacher-v2.4-v6"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "schema is not V6-compatible"):
                load_teacher_bundle(checkpoint, config)

    def test_teacher_checkpoint_authenticates_source_train_normalization(self) -> None:
        config = load_formal_config(SMOKE_CONFIG)
        spec = PatchSpec(2, 4, 1)
        csi = np.arange(4 * 16, dtype=np.float64).reshape(4, 16) / 100.0
        bundle = train_teacher_bundle(csi, spec, config, seed=35)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "teacher.pt"
            save_teacher_bundle(checkpoint, bundle, config, seed=35)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["input_normalization"]["channel_scale"][0] = 0.0
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "normalization is invalid"):
                load_teacher_bundle(checkpoint, config)

    def test_teacher_checkpoint_rejects_training_config_or_seed_substitution(self) -> None:
        config = load_formal_config(SMOKE_CONFIG)
        spec = PatchSpec(2, 4, 1)
        csi = np.arange(4 * 16, dtype=np.float64).reshape(4, 16) / 100.0
        bundle = train_teacher_bundle(csi, spec, config, seed=37)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "teacher.pt"
            with self.assertRaisesRegex(ValueError, "seed does not match"):
                save_teacher_bundle(checkpoint, bundle, config, seed=38)
            save_teacher_bundle(checkpoint, bundle, config, seed=37)
            changed = load_formal_config(SMOKE_CONFIG)
            changed["teacher"]["learning_rate"] *= 2.0
            with self.assertRaisesRegex(RuntimeError, "training configuration"):
                load_teacher_bundle(checkpoint, changed)

    def test_teacher_source_normalization_is_scale_invariant(self) -> None:
        config = load_formal_config(SMOKE_CONFIG)
        spec = PatchSpec(2, 4, 1)
        csi = (
            np.arange(32 * 16, dtype=np.float64).reshape(32, 16)
            + np.sin(np.arange(32, dtype=np.float64))[:, None]
        )
        base = train_teacher_bundle(csi, spec, config, seed=36)
        small = train_teacher_bundle(csi * 1e-4, spec, config, seed=36)

        np.testing.assert_allclose(
            normalized_teacher_patches(base, csi),
            normalized_teacher_patches(small, csi * 1e-4),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            teacher_targets(base, csi),
            teacher_targets(small, csi * 1e-4),
            rtol=0.0,
            atol=1e-6,
        )
        self.assertAlmostEqual(base.reconstruction_nmse, small.reconstruction_nmse, places=6)
        self.assertAlmostEqual(base.readout_nmse, small.readout_nmse, places=6)

    def test_cached_protocol_features_are_exactly_equivalent(self) -> None:
        rng = np.random.default_rng(3201)
        patches = rng.normal(size=(4, 4))
        source_map = rng.normal(size=(5, 8, 8))
        action = rng.normal(size=(14, 8, 8))
        radio = rng.normal(size=9)
        mask = np.asarray([True, False, True, True])
        position = rng.normal(size=2)
        map_features = multichannel_spatial_features(source_map)
        action_features = multichannel_spatial_features(action)
        for include_csi, include_map, include_action, supplied_position in (
            (True, True, True, None),
            (True, True, True, position),
            (True, True, False, None),
            (False, True, True, None),
            (True, False, False, None),
        ):
            with self.subTest(
                include_csi=include_csi,
                include_map=include_map,
                include_action=include_action,
                position=supplied_position is not None,
            ):
                expected = protocol_response_features(
                    patches,
                    source_map,
                    action,
                    radio,
                    mask,
                    1,
                    position=supplied_position,
                    include_csi=include_csi,
                    include_map=include_map,
                    include_action=include_action,
                )
                observed = assemble_protocol_response_features(
                    patches,
                    map_features if include_map else None,
                    action_features if include_action else None,
                    radio,
                    mask,
                    1,
                    position=supplied_position,
                    include_csi=include_csi,
                )
                np.testing.assert_array_equal(observed, expected)

    def test_spatial_features_distinguish_translated_equal_edits(self) -> None:
        first = np.zeros((1, 16, 16), dtype=np.float64)
        second = np.zeros_like(first)
        first[0, 2:4, 3:5] = 1.0
        second[0, 11:13, 10:12] = 1.0
        self.assertFalse(
            np.array_equal(
                multichannel_spatial_features(first),
                multichannel_spatial_features(second),
            )
        )

    def test_teacher_and_model_mask_configuration_cannot_be_silently_ignored(self) -> None:
        teacher = load_formal_config(SMOKE_CONFIG)
        teacher["teacher"]["mask_fraction"] = 0.6
        with self.assertRaisesRegex(ValueError, "teacher.mask_fraction"):
            validate_formal_config(teacher)
        model = load_formal_config(SMOKE_CONFIG)
        model["model"]["mask_fraction"] = 0.1
        with self.assertRaisesRegex(ValueError, "model.mask_fraction"):
            validate_formal_config(model)

    def test_block_masks_are_complete_half_axis_blocks(self) -> None:
        spec = PatchSpec(4, 4, 1, 1, 1)
        bank = frozen_mask_query_bank(spec, 17)
        for entry in bank:
            grid = entry.mask.reshape(spec.patch_rows, spec.patch_columns)
            if entry.mode == "antenna_block_50":
                selected = np.flatnonzero(np.all(grid, axis=1))
                self.assertEqual(selected.size, 2)
                self.assertTrue(np.all(np.logical_or(np.all(grid, axis=1), ~np.any(grid, axis=1))))
                self.assertTrue(np.all(np.diff(selected) == 1))
            if entry.mode == "subcarrier_block_50":
                selected = np.flatnonzero(np.all(grid, axis=0))
                self.assertEqual(selected.size, 2)
                self.assertTrue(np.all(np.logical_or(np.all(grid, axis=0), ~np.any(grid, axis=0))))
                self.assertTrue(np.all(np.diff(selected) == 1))
            self.assertTrue(entry.mask[entry.query])


class QualificationCoverageTests(unittest.TestCase):
    def test_compact_routes_match_independent_scalar_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = write_nonscientific_fixture(
                Path(temporary) / "fixture.npz", positions=8
            )
            dataset = FormalDataset.load(fixture)
            config = load_formal_config(SMOKE_CONFIG)
            blocking = qualification_blocking_scenes(dataset)
            spec = PatchSpec.from_metadata(dataset.metadata)
            teacher = train_teacher_bundle(
                dataset.csi[blocking["teacher_train"]], spec, config, seed=117
            )
            normalization = fit_route_normalization(dataset, teacher)
            scene = int(blocking["method_selection"][0])
            routed = route_dataset(
                dataset,
                teacher,
                config,
                np.asarray([scene]),
                normalization=normalization,
            )

        self.assertIsInstance(routed.alignment_route, CompactEdgeTensorMapping)
        self.assertIsInstance(routed.response_route, CompactEdgeTensorMapping)
        patch_scale = patchify_csi(normalization.channel_scale, spec)
        thresholds = config["qualification"]
        latent = routed.teacher_latent[scene]
        patches = routed.physical_patches[scene]

        def code(value, null_key, active_key):
            if value <= float(thresholds[null_key]):
                return 0
            if value >= float(thresholds[active_key]):
                return 2
            return 1

        for edge in dataset.directed_edges(scene):
            for position in range(dataset.position_count):
                source_csi = dataset.csi[scene, edge.source_world, position]
                target_csi = dataset.csi[scene, edge.target_world, position]
                complex_difference = (
                    target_csi - source_csi
                ) / normalization.channel_scale
                delay_difference = (
                    delay_angle_power(target_csi, spec)
                    - delay_angle_power(source_csi, spec)
                ) / normalization.delay_angle_scale
                physical_a = float(
                    np.sqrt(
                        np.mean(
                            np.concatenate((complex_difference, delay_difference)) ** 2
                        )
                    )
                )
                latent_difference = (
                    latent[edge.target_world, position]
                    - latent[edge.source_world, position]
                ) / normalization.latent_scale
                latent_a = float(np.sqrt(np.mean(latent_difference**2)))
                alignment_key = (
                    scene,
                    edge.source_world,
                    edge.target_world,
                    position,
                )
                np.testing.assert_allclose(
                    routed.alignment_distances[alignment_key],
                    (physical_a, latent_a),
                    rtol=1e-12,
                    atol=1e-12,
                )
                self.assertEqual(
                    routed.alignment_route[alignment_key],
                    code(physical_a, "physical_null_rms_max", "physical_active_rms_min"),
                )
                self.assertEqual(
                    routed.alignment_teacher_stratum[alignment_key],
                    code(latent_a, "latent_null_rms_max", "latent_active_rms_min"),
                )
                for query in range(spec.patch_count):
                    physical_r = float(
                        np.sqrt(
                            np.mean(
                                (
                                    (
                                        patches[edge.target_world, position, query]
                                        - patches[edge.source_world, position, query]
                                    )
                                    / patch_scale[query]
                                )
                                ** 2
                            )
                        )
                    )
                    latent_r = float(
                        np.sqrt(np.mean(latent_difference[query] ** 2))
                    )
                    response_key = (*alignment_key, query)
                    np.testing.assert_allclose(
                        routed.response_distances[response_key],
                        (physical_r, latent_r),
                        rtol=1e-12,
                        atol=1e-12,
                    )
                    self.assertEqual(
                        routed.response_route[response_key],
                        code(
                            physical_r,
                            "response_physical_null_rms_max",
                            "response_physical_active_rms_min",
                        ),
                    )
                    self.assertEqual(
                        routed.response_teacher_stratum[response_key],
                        code(
                            latent_r,
                            "response_latent_null_rms_max",
                            "response_latent_active_rms_min",
                        ),
                    )

    def test_route_normalization_must_match_the_teacher_source_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = write_nonscientific_fixture(Path(temporary) / "fixture.npz")
            dataset = FormalDataset.load(fixture)
            config = load_formal_config(SMOKE_CONFIG)
            scenes = qualification_blocking_scenes(dataset)["teacher_train"]
            teacher = train_teacher_bundle(
                dataset.csi[scenes],
                PatchSpec.from_metadata(dataset.metadata),
                config,
                seed=119,
            )
            dataset.csi_clean[int(scenes[0]), 0, 0, 0] += 1e-3
            with self.assertRaisesRegex(RuntimeError, "source_encoder_train CSI"):
                fit_route_normalization(dataset, teacher)

    def test_qualification_response_uses_frozen_psi_for_input_and_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = write_nonscientific_fixture(
                Path(temporary) / "fixture.npz", positions=8
            )
            dataset = FormalDataset.load(fixture)
            config = load_formal_config(SMOKE_CONFIG)
            blocking = qualification_blocking_scenes(dataset)
            teacher = train_teacher_bundle(
                dataset.csi[blocking["teacher_train"]],
                PatchSpec.from_metadata(dataset.metadata),
                config,
                seed=118,
            )
            normalization = fit_route_normalization(dataset, teacher)
            scene = int(blocking["teacher_train"][0])
            routed = route_dataset(
                dataset,
                teacher,
                config,
                np.asarray([scene]),
                normalization=normalization,
            )
            records = _build_records(
                dataset, np.asarray([scene]), routed, teacher.mask_bank
            )
            streamed_chunks = {}
            for batch in _iter_qualification_record_batches(
                dataset,
                np.asarray([scene]),
                routed,
                teacher.mask_bank,
                batch_rows=13,
            ):
                for name, values in batch.items():
                    streamed_chunks.setdefault(name, []).append(values)
            streamed = {
                name: np.concatenate(chunks, axis=0)
                for name, chunks in streamed_chunks.items()
            }

        edge = next(iter(dataset.directed_edges(scene)))
        patches = normalized_teacher_patches(teacher, dataset.csi[scene])
        source = patches[edge.source_world, 0]
        target = patches[edge.target_world, 0]
        query = 0
        entry = next(
            item
            for item in teacher.mask_bank
            if item.mode == "random_75" and item.query == query
        )
        visible = source.copy()
        visible[entry.mask] = 0.0
        np.testing.assert_array_equal(
            records[0]["target_delta"], target[query] - source[query]
        )
        np.testing.assert_array_equal(
            records[0]["no_x"][: visible.size], visible.ravel()
        )
        self.assertEqual(streamed["scene"].shape[0], len(records))
        for name in ("scene", "route", "wrong_action_match_status"):
            np.testing.assert_array_equal(
                streamed[name], np.asarray([record[name] for record in records])
            )
        for name in (
            "target_delta",
            "no_x",
            "no_x_zero_action",
            "oracle_x",
            "oracle_x_zero_action",
            "map_edit_only",
            "map_edit_only_zero_action",
            "csi_only",
            "variant_id_only",
            "action_swap",
        ):
            np.testing.assert_array_equal(
                streamed[name], np.vstack([record[name] for record in records])
            )

    def test_teacher_latent_replacement_cannot_change_primary_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = write_nonscientific_fixture(Path(temporary) / "fixture.npz")
            dataset = FormalDataset.load(fixture)
            config = load_formal_config(SMOKE_CONFIG)
            blocking = qualification_blocking_scenes(dataset)
            spec = PatchSpec.from_metadata(dataset.metadata)
            teacher = train_teacher_bundle(
                dataset.csi[blocking["teacher_train"]],
                spec,
                config,
                seed=120,
            )
            normalization = fit_route_normalization(dataset, teacher)
            scenes = blocking["method_selection"]
            baseline = route_dataset(
                dataset,
                teacher,
                config,
                scenes,
                normalization=normalization,
            )

            from formal_v2.formal_teacher import teacher_targets as real_teacher_targets

            def teacher_blind(bundle, csi):
                return np.zeros_like(real_teacher_targets(bundle, csi))

            with patch(
                "formal_v2.formal_routing.teacher_targets",
                side_effect=teacher_blind,
            ):
                replaced = route_dataset(
                    dataset,
                    teacher,
                    config,
                    scenes,
                    normalization=normalization,
                )

        self.assertEqual(baseline.alignment_route, replaced.alignment_route)
        self.assertEqual(baseline.response_route, replaced.response_route)
        self.assertTrue(any(value == 2 for value in baseline.alignment_route.values()))
        self.assertTrue(
            all(value == 0 for value in replaced.alignment_teacher_stratum.values())
        )
        self.assertTrue(
            all(value == 0 for value in replaced.response_teacher_stratum.values())
        )

    def test_route_low_thresholds_are_audited_in_their_native_noise_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = write_nonscientific_fixture(Path(temporary) / "fixture.npz")
            dataset = FormalDataset.load(fixture)
            config = load_formal_config(SMOKE_CONFIG)
            blocking = qualification_blocking_scenes(dataset)
            spec = PatchSpec.from_metadata(dataset.metadata)
            teacher = train_teacher_bundle(
                dataset.csi[blocking["teacher_train"]],
                spec,
                config,
                seed=121,
            )
            normalization = fit_route_normalization(dataset, teacher)
            rows = _route_noise_floor_rows(
                dataset,
                teacher,
                blocking["method_selection"],
                normalization,
                config,
            )

        self.assertTrue(rows)
        expected = {
            "alignment_physical_noise_floor_rms",
            "alignment_latent_noise_floor_rms",
            "response_physical_noise_floor_rms",
            "response_latent_noise_floor_rms",
        }
        self.assertTrue(expected.issubset(rows[0]))
        self.assertTrue(all(row["source_role"] == "source_method_selection" for row in rows))
        self.assertTrue(all(row["quantile"] == config["qualification"]["noise_floor_quantile"] for row in rows))
        self.assertTrue(all(row["passed"] for row in rows))

        zero_thresholds = load_formal_config(SMOKE_CONFIG)
        for key in (
            "physical_null_rms_max",
            "latent_null_rms_max",
            "response_physical_null_rms_max",
            "response_latent_null_rms_max",
        ):
            zero_thresholds["qualification"][key] = 0.0
        failed = _route_noise_floor_rows(
            dataset,
            teacher,
            blocking["method_selection"],
            normalization,
            zero_thresholds,
        )
        self.assertTrue(all(not row["passed"] for row in failed))

    def test_per_bank_train_route_coverage_requires_all_four_denominators(self) -> None:
        qualification = {
            "minimum_active_units_per_bank": 2,
            "minimum_null_units_per_bank": 2,
        }
        row = {
            "alignment_active_units": 2,
            "alignment_null_units": 2,
            "response_patch_active_units": 2,
            "response_patch_null_units": 2,
        }
        self.assertTrue(_route_coverage_passed(row, qualification))
        row["response_patch_null_units"] = 1
        self.assertFalse(_route_coverage_passed(row, qualification))


    def test_wrong_action_matching_distinguishes_exact_fallback_and_failed(self) -> None:
        reference = np.zeros((12, 6, 6), dtype=np.float64)
        reference[2, 1:3, 1:3] = 1.0
        exact = np.zeros_like(reference)
        exact[2, 3:5, 3:5] = 1.0
        fallback = np.zeros_like(reference)
        fallback[2, 3:6, 3:5] = 1.0
        failed = np.zeros_like(reference)
        failed[0, 3:5, 3:5] = 1.0
        self.assertEqual(_wrong_action_distance(reference, exact)[:2], (0, 0))
        self.assertEqual(_wrong_action_distance(reference, fallback)[:2], (0, 1))
        self.assertEqual(_wrong_action_distance(reference, failed)[0], 1)
        self.assertNotEqual(
            _action_geometry_profile(reference)["family"],
            _action_geometry_profile(failed)["family"],
        )

    def test_material_from_to_planes_are_part_of_wrong_action_semantics(self) -> None:
        reference = np.zeros((10, 4, 4), dtype=np.float64)
        candidate = np.zeros_like(reference)
        reference[4, 1:3, 1:3] = 1.0
        reference[8, 1:3, 1:3] = 1.0
        candidate[5, 1:3, 1:3] = 1.0
        candidate[9, 1:3, 1:3] = 1.0
        self.assertEqual(reference.shape[0], 4 + 2 * 3)
        self.assertEqual(_wrong_action_distance(reference, candidate)[0], 1)
        self.assertNotEqual(
            _action_geometry_profile(reference)["family"],
            _action_geometry_profile(candidate)["family"],
        )

    def test_reversible_material_direction_is_not_the_same_signed_edit_family(self) -> None:
        forward = np.zeros((14, 6, 6), dtype=np.float64)
        reverse = np.zeros_like(forward)
        # Five material categories: concrete (4) <-> glass (1).
        forward[4 + 4, 1:3, 2:4] = 1.0
        forward[4 + 5 + 1, 1:3, 2:4] = 1.0
        reverse[4 + 1, 3:5, 1:3] = 1.0
        reverse[4 + 5 + 4, 3:5, 1:3] = 1.0
        self.assertEqual(_wrong_action_distance(forward, reverse)[0], 1)
        self.assertNotEqual(
            _action_geometry_profile(forward)["family"],
            _action_geometry_profile(reverse)["family"],
        )

    def test_wrong_action_selection_is_receiver_position_specific(self) -> None:
        world_bits = (
            (np.arange(8, dtype=np.int64)[:, None] >> np.arange(3)) & 1
        )
        maps = np.zeros((1, 8, 3, 5, 5), dtype=np.float64)
        primitive_cells = ((1, 1), (1, 3), (3, 1))
        for world, bits in enumerate(world_bits):
            for bit, (row, column) in enumerate(primitive_cells):
                maps[0, world, 0, row, column] = float(bits[bit])
        dataset = SimpleNamespace(
            world_bits=world_bits,
            bit_count=3,
            maps=maps,
            map_channel_names=np.asarray(["occupancy", "height", "material"]),
            metadata={
                "representation": {
                    "map_origin_xy_m": [0.0, 0.0],
                    "map_resolution_m": 1.0,
                }
            },
        )
        correct = typed_signed_edit(
            maps[0, 0],
            maps[0, 1],
            dataset.map_channel_names,
            1,
        )
        _, first_status, first_world = _select_wrong_action(
            dataset,
            0,
            0,
            0,
            correct,
            1,
            receiver_position=np.asarray([2.5, 0.5, 0.0]),
        )
        _, second_status, second_world = _select_wrong_action(
            dataset,
            0,
            0,
            0,
            correct,
            1,
            receiver_position=np.asarray([0.5, 2.5, 0.0]),
        )
        self.assertEqual((first_status, first_world), ("exact", 2))
        self.assertEqual((second_status, second_world), ("exact", 4))
        self.assertNotEqual(first_world, second_world)
        self.assertEqual(
            _select_wrong_action(dataset, 0, 0, 0, correct, 1)[1],
            "fallback",
        )

    def test_wrong_action_plan_is_equivalent_to_direct_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = write_nonscientific_fixture(Path(temporary) / "fixture.npz")
            dataset = FormalDataset.load(fixture)
            scene = int(qualification_blocking_scenes(dataset)["teacher_train"][0])
            material_categories = int(
                dataset.metadata["assets"]["material_category_count"]
            )
            plan = _build_wrong_action_plan(
                dataset, np.asarray([scene]), material_categories
            )
            selected_plan = _build_wrong_action_plan(
                dataset,
                np.asarray([scene]),
                material_categories,
                positions=np.asarray([0]),
            )
            self.assertEqual(
                {key[-1] for key in selected_plan["selections"]},
                {0},
            )
            for edge in dataset.directed_edges(scene):
                correct = typed_signed_edit(
                    dataset.maps[scene, edge.source_world],
                    dataset.maps[scene, edge.target_world],
                    dataset.map_channel_names,
                    material_categories,
                )
                for position in range(dataset.position_count):
                    expected_action, expected_status, expected_world = _select_wrong_action(
                        dataset,
                        scene,
                        edge.source_world,
                        edge.bit_index,
                        correct,
                        material_categories,
                        receiver_position=dataset.positions[scene, position],
                    )
                    actual_features, actual_status, actual_world = plan["selections"][
                        (scene, edge.source_world, edge.bit_index, position)
                    ]
                    np.testing.assert_array_equal(
                        actual_features,
                        multichannel_spatial_features(expected_action),
                    )
                    self.assertEqual(actual_status, expected_status)
                    self.assertEqual(actual_world, expected_world)

    def test_failed_wrong_actions_cannot_enter_the_swap_comparison_denominator(self) -> None:
        dataset = SimpleNamespace(
            scene_ids=np.asarray(["scene"]),
            bank_ids=np.asarray(["bank"]),
        )
        records = [
            {"scene": 0, "route": 2, "wrong_action_match_status": "failed"},
            {"scene": 0, "route": 0, "wrong_action_match_status": "failed"},
        ]
        target = np.asarray([[1.0], [0.0]])
        predictions = {
            "no_x": np.asarray([[1.0], [0.0]]),
            "copy": np.zeros((2, 1)),
            "no_action": np.zeros((2, 1)),
            "action_swap": np.zeros((2, 1)),
            "oracle_x": np.asarray([[1.0], [0.0]]),
        }
        metrics, _, gates = _response_gate_rows(
            dataset,
            records,
            target,
            predictions,
            {
                "response_physical_null_rms_max": 0.1,
                "no_x_min_relative_improvement": 0.0,
                "oracle_min_relative_improvement": 0.0,
                "null_violation_rate_max": 0.0,
            },
        )
        self.assertIsNone(gates[0]["relative_improvement_vs_action_swap"])
        self.assertEqual(gates[0]["action_swap_exact_common_denominator"], 0)
        self.assertFalse(gates[0]["passed"])
        null_metrics = [row for row in metrics if row["stratum"] == "null"]
        self.assertTrue(null_metrics)
        self.assertTrue(
            all(row["physical_patch_delta_nmse"] is None for row in null_metrics)
        )


class QualificationResumeSafetyTests(unittest.TestCase):
    @staticmethod
    def _resume_layout(root: Path) -> Path:
        qualification = root / "qualification"
        checkpoints = qualification / "checkpoints"
        checkpoints.mkdir(parents=True)
        (qualification / "data_contract.json").write_text("{}\n", encoding="utf-8")
        (qualification / "teacher_resume.json").write_text("{}\n", encoding="utf-8")
        (checkpoints / "stage0_csi_teacher.pt").write_bytes(b"checkpoint")
        return qualification

    def test_new_qualification_refuses_to_overwrite_owned_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            qualification = Path(temporary) / "qualification"
            (qualification / "rt_calibration").mkdir(parents=True)
            _prepare_qualification_output(qualification, resume=False)
            contract = qualification / "data_contract.json"
            contract.write_text("preserve me\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "earlier attempt"):
                _prepare_qualification_output(qualification, resume=False)
            self.assertEqual(contract.read_text(encoding="utf-8"), "preserve me\n")

    def test_resume_refuses_every_partial_or_complete_final_artifact(self) -> None:
        for name in QUALIFICATION_FINAL_ARTIFACT_NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                qualification = self._resume_layout(Path(temporary))
                (qualification / name).write_text("partial\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "partial final evidence"):
                    _prepare_qualification_output(qualification, resume=True)

    def test_resume_requires_regular_authenticated_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qualification = self._resume_layout(root)
            receipt = qualification / "teacher_resume.json"
            receipt.unlink()
            target = root / "outside-receipt.json"
            target.write_text("{}\n", encoding="utf-8")
            symlink_or_skip(receipt, target)
            with self.assertRaisesRegex(RuntimeError, "regular authenticated artifact"):
                _prepare_qualification_output(qualification, resume=True)

    def test_resume_authenticates_receipt_checkpoint_and_teacher_seed(self) -> None:
        config = load_formal_config(SMOKE_CONFIG)
        spec = PatchSpec(2, 4, 1)
        csi = np.arange(8 * 16, dtype=np.float64).reshape(8, 16) / 100.0
        teacher_seed = 137
        bundle = train_teacher_bundle(csi, spec, config, seed=teacher_seed)
        evidence = {"dataset_sha256": "a" * 64, "source_tree_sha256": "b" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "teacher.pt"
            receipt_path = root / "teacher_resume.json"
            save_teacher_bundle(checkpoint, bundle, config, teacher_seed)
            receipt = _teacher_resume_receipt(checkpoint, evidence, teacher_seed)
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
            )
            loaded = _load_resumed_teacher(
                checkpoint,
                receipt_path,
                config,
                evidence,
                teacher_seed,
                "cpu",
            )
            self.assertEqual(loaded.seed, teacher_seed)

            receipt["dataset_sha256"] = "c" * 64
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "receipt no longer matches"):
                _load_resumed_teacher(
                    checkpoint,
                    receipt_path,
                    config,
                    evidence,
                    teacher_seed,
                    "cpu",
                )

            receipt_path.write_text(
                json.dumps(
                    _teacher_resume_receipt(checkpoint, evidence, teacher_seed + 1),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "teacher seed no longer matches"):
                _load_resumed_teacher(
                    checkpoint,
                    receipt_path,
                    config,
                    evidence,
                    teacher_seed + 1,
                    "cpu",
                )

            receipt_path.write_text(
                json.dumps(
                    _teacher_resume_receipt(checkpoint, evidence, teacher_seed),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            with checkpoint.open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "receipt no longer matches"):
                _load_resumed_teacher(
                    checkpoint,
                    receipt_path,
                    config,
                    evidence,
                    teacher_seed,
                    "cpu",
                )

    def test_probe_attempts_are_unique_and_skip_existing_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            qualification = Path(temporary) / "qualification"
            (qualification / "checkpoints").mkdir(parents=True)
            first = _reserve_qualification_probe_attempt(qualification)
            second = _reserve_qualification_probe_attempt(qualification)
            self.assertEqual(first.name, "attempt_0001")
            self.assertEqual(second.name, "attempt_0002")
            symlink_or_skip(second.parent / "attempt_0003", Path(temporary) / "missing")
            fourth = _reserve_qualification_probe_attempt(qualification)
            self.assertEqual(fourth.name, "attempt_0004")

    def test_probe_root_cannot_redirect_through_a_symbolic_link(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qualification = root / "qualification"
            checkpoints = qualification / "checkpoints"
            checkpoints.mkdir(parents=True)
            outside = root / "outside"
            outside.mkdir()
            symlink_or_skip(checkpoints / "qualification_probes", outside)
            with self.assertRaisesRegex(RuntimeError, "probe directory"):
                _reserve_qualification_probe_attempt(qualification)


if __name__ == "__main__":
    unittest.main()
    _iter_qualification_record_batches,
