from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from formal_v2.formal_config import load_formal_config, validate_formal_config
from formal_v2.formal_dataset import FormalDataset, FormalDatasetError
from formal_v2.formal_fixture import write_nonscientific_fixture
from formal_v2.formal_protocol import (
    PatchSpec,
    frozen_mask_query_bank,
    typed_signed_edit,
)
from formal_v2.formal_qualification import (
    _action_geometry_profile,
    _response_gate_rows,
    _route_noise_floor_rows,
    _route_coverage_passed,
    _select_wrong_action,
    _wrong_action_distance,
    qualification_blocking_scenes,
)
from formal_v2.formal_routing import fit_route_normalization, route_dataset
from formal_v2.formal_teacher import (
    load_teacher_bundle,
    random_teacher_pretraining_masks,
    save_teacher_bundle,
    train_teacher_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "formal_v2_smoke.json"


def _archive_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


class DatasetIdentityAndNoiseTests(unittest.TestCase):
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
        ) as sampler:
            train_teacher_bundle(csi, spec, config, seed=33)
        self.assertEqual(sampler.call_count, config["teacher"]["steps"])

    def test_teacher_checkpoint_authenticates_the_complete_mask_contract(self) -> None:
        config = load_formal_config(SMOKE_CONFIG)
        spec = PatchSpec(2, 4, 1)
        csi = np.arange(4 * 16, dtype=np.float64).reshape(4, 16) / 100.0
        bundle = train_teacher_bundle(csi, spec, config, seed=34)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "teacher.pt"
            save_teacher_bundle(checkpoint, bundle, config, seed=34)
            load_teacher_bundle(checkpoint, config)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["pretraining_mask_sampler"][
                "resampled_each_optimization_step"
            ] = False
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "exact V6 pretraining mask contract"):
                load_teacher_bundle(checkpoint, config)

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
        _, _, gates = _response_gate_rows(
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


if __name__ == "__main__":
    unittest.main()
