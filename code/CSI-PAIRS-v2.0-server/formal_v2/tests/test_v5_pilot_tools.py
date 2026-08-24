from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

try:
    from artifacts.formal_readiness.tools import v5_four_bank_rt_audit as audit
    from artifacts.formal_readiness.tools import v5_four_bank_rt_pilot as pilot
    from artifacts.formal_readiness.tools import v5_expanded_training_config as expanded_config
    from artifacts.formal_readiness.tools import qualification_streaming_bank_pilot as qualification_pilot
    from artifacts.formal_readiness.tools import v5_nonlinear_response_probe as nonlinear_pilot
    from artifacts.formal_readiness.tools import v5_physics_response_probe as physics_probe
    from artifacts.formal_readiness.tools import (
        v5_formal_model_response_pilot as formal_response_pilot,
    )
    from artifacts.formal_readiness.tools import (
        v5_formal_action_overfit_diagnostic as action_overfit,
    )
    from artifacts.formal_readiness.tools import (
        v5_response_generalization_diagnostic as generalization_diagnostic,
    )
    from artifacts.formal_readiness.tools import v5_teacher_pilot_train as teacher_pilot
except ModuleNotFoundError as error:
    raise unittest.SkipTest(str(error)) from error
from formal_v2.formal_factorial import _centered_branch_mse
from formal_v2.sionna_osm_candidate import load_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "formal_v2" / "configs" / "sionna_osm_formal_candidate_v5.json"


class V5PilotToolTests(unittest.TestCase):
    def test_rt_audit_accepts_bounded_variable_role_windows(self) -> None:
        self.assertEqual(
            audit._classified_scene_indices(
                {"scene_indices": list(range(16))}, "normalization"
            ),
            list(range(16)),
        )
        self.assertEqual(
            audit._classified_scene_indices(
                {"scene_indices": [32, 33, 34, 35]}, "evaluation"
            ),
            [32, 33, 34, 35],
        )
        for invalid in ([0], [0, 0], [1, 0], [0, "1"], list(range(33))):
            with self.assertRaises(RuntimeError):
                audit._classified_scene_indices(
                    {"scene_indices": invalid}, "invalid"
                )

    def test_expanded_training_config_adds_only_encoder_role_banks(self) -> None:
        base = load_config(CONFIG_PATH)
        expanded = expanded_config.expanded_training_config(base, 16)
        ledger = pilot.candidate.expected_scene_ledger(expanded)
        self.assertEqual(len(ledger), 227)
        self.assertEqual(
            sum(row["role"] == "source_encoder_train" for row in ledger), 32
        )
        self.assertEqual(
            sum(row["role"] == "source_method_selection" for row in ledger), 8
        )
        self.assertEqual(
            sum(row["role"] == "source_final_unseen_bank" for row in ledger), 41
        )
        for original, changed in zip(base["cities"], expanded["cities"], strict=True):
            if original["split_group"] != "source":
                self.assertEqual(changed, original)
                continue
            original_roles = dict(original["source_role_bank_counts"])
            changed_roles = dict(changed["source_role_bank_counts"])
            changed_roles["source_encoder_train"] = original_roles[
                "source_encoder_train"
            ]
            self.assertEqual(changed_roles, original_roles)

    def test_centered_branch_loss_detects_action_collapse_without_false_margin(self) -> None:
        target = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        collapsed = torch.zeros_like(target)
        self.assertGreater(float(_centered_branch_mse(collapsed, target, (2,))), 0.0)
        self.assertEqual(float(_centered_branch_mse(target, target, (2,))), 0.0)
        equivalent = torch.ones_like(target)
        self.assertEqual(
            float(_centered_branch_mse(collapsed, equivalent, (2,))), 0.0
        )

    def test_formal_generator_defaults_to_v5_candidate(self) -> None:
        script = (
            ROOT / "formal_v2" / "scripts" / "generate_sionna_osm_formal_candidate.sh"
        ).read_text(encoding="ascii")
        self.assertIn(
            "formal_v2/configs/sionna_osm_formal_candidate_v5.json}", script
        )
        self.assertNotIn(
            "formal_v2/configs/sionna_osm_formal_candidate_v4.json}", script
        )

    def _prepared_root(
        self,
        root: Path,
        *,
        deficient: bool = False,
        weak_quality: bool = False,
        indices: list[int] | None = None,
        bank_start: int = 0,
    ) -> Path:
        prepared = root / "prepared"
        asset_root = prepared / "assets"
        bank_root = asset_root / "banks"
        bank_root.mkdir(parents=True)
        config = load_config(CONFIG_PATH)
        (asset_root / "generator_config.json").write_text(
            json.dumps(config, sort_keys=True) + "\n", encoding="ascii"
        )
        indices = [4, 5, 6, 7] if indices is None else indices
        classification = {
            "schema_version": "csi-pairs-v5-formal-index-pilot-classification-v1",
            **pilot.CLASSIFICATION,
            "city_id": "source-austin",
            "bank_start": bank_start,
            "selection_prefix_count": len(indices),
            "selected_window_count": len(indices),
            "scene_indices": indices,
        }
        (prepared / "PILOT_CLASSIFICATION.json").write_text(
            json.dumps(classification, sort_keys=True) + "\n", encoding="ascii"
        )
        for offset, scene_index in enumerate(indices):
            scene_id = f"osm-sionna-source-austin-bank-{scene_index:02d}"
            directory = bank_root / scene_id
            directory.mkdir()
            counts = [24, 25, 26, 27]
            if deficient and offset == 0:
                counts[2] = 23
            record = {
                "scene_id": scene_id,
                "scene_index": scene_index,
                "geometry_registration": {
                    "algorithm": "controlled-four-primitive-matched-branching-v5-clearance",
                    "minimum_direct_path_vertical_clearance_m": float(
                        config["map"][
                            "receiver_minimum_direct_path_vertical_clearance_m"
                        ]
                    ),
                    "minimum_exact_positions_per_primitive": 24,
                    "minimum_selected_quality_mean_per_primitive": 0.0048,
                    "per_primitive_counts_are_blocking": True,
                    "primitive_records": [
                        {
                            "selected_joint_position_count": value,
                            "selected_quality_mean": (
                                0.0047 if weak_quality and index == 2 else 0.005
                            ),
                        }
                        for index, value in enumerate(counts)
                    ],
                },
            }
            (directory / "bank.json").write_text(
                json.dumps(record, sort_keys=True) + "\n", encoding="ascii"
            )
        return prepared

    def test_bootstrap_runtime_forwards_script_and_all_arguments(self) -> None:
        with mock.patch.object(pilot.candidate, "ensure_sionna_runtime") as ensure:
            pilot._bootstrap_runtime(["--output", "/tmp/pilot"])
        ensure.assert_called_once_with(
            [str(Path(pilot.__file__).resolve()), "--output", "/tmp/pilot"]
        )

    def test_render_worker_uses_an_isolated_drjit_home(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pilot"
            output.mkdir()
            with (
                mock.patch.object(pilot.os, "getpid", return_value=12345),
                mock.patch.dict(pilot.os.environ, {"HOME": "/original-home"}),
            ):
                pilot._initialize_render_worker(str(output))
                expected = output / "runtime-cache" / "drjit" / "worker-12345"
                self.assertEqual(pilot.os.environ["HOME"], str(expected.resolve()))
                self.assertTrue(expected.is_dir())

    def test_prepare_selects_requested_window_from_deterministic_prefix(self) -> None:
        config = load_config(CONFIG_PATH)
        ledger = [
            row
            for row in pilot.candidate.expected_scene_ledger(config)
            if row["city_id"] == "source-austin"
        ]
        selected = [{**row, "geometry_registration": {}} for row in ledger[:8]]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "pilot"
            with (
                mock.patch.object(pilot.candidate, "_download_city_osm") as download,
                mock.patch.object(pilot.candidate, "_parse_osm_buildings", return_value=[]),
                mock.patch.object(
                    pilot.candidate,
                    "_select_city_banks",
                    return_value=(selected, {"selected_bank_count": 8}),
                ) as choose,
                mock.patch.object(pilot, "_write_bank_assets"),
            ):
                raw = root.parent / "austin.json"
                raw.write_text('{"elements":[]}', encoding="ascii")
                download.return_value = (raw, "query", {"elements": []}, "cache")
                records = pilot.prepare(
                    CONFIG_PATH, root, "source-austin", 4, 4, None
                )
            self.assertEqual(
                [int(row["scene_index"]) for row in records], [12, 13, 14, 15]
            )
            self.assertEqual(choose.call_args.args[-1], ledger[:8])
            classification = json.loads(
                (root / "PILOT_CLASSIFICATION.json").read_text(encoding="ascii")
            )
            self.assertEqual(classification["bank_start"], 4)
            self.assertEqual(classification["selection_prefix_count"], 8)
            self.assertIsNone(classification["radio_override"])

    def test_prepare_records_a_forbidden_wideband_pilot_override(self) -> None:
        config = load_config(CONFIG_PATH)
        ledger = [
            row
            for row in pilot.candidate.expected_scene_ledger(config)
            if row["city_id"] == "source-austin"
        ]
        selected = [{**row, "geometry_registration": {}} for row in ledger[:4]]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "pilot"
            with (
                mock.patch.object(pilot.candidate, "_download_city_osm") as download,
                mock.patch.object(pilot.candidate, "_parse_osm_buildings", return_value=[]),
                mock.patch.object(
                    pilot.candidate,
                    "_select_city_banks",
                    return_value=(selected, {"selected_bank_count": 4}),
                ),
                mock.patch.object(pilot, "_write_bank_assets"),
            ):
                raw = root.parent / "austin.json"
                raw.write_text('{"elements":[]}', encoding="ascii")
                download.return_value = (raw, "query", {"elements": []}, "cache")
                pilot.prepare(
                    CONFIG_PATH,
                    root,
                    "source-austin",
                    4,
                    0,
                    None,
                    1_000_000.0,
                )
            generated = json.loads(
                (root / "assets" / "generator_config.json").read_text(encoding="ascii")
            )
            self.assertEqual(generated["radio"]["subcarrier_spacing_hz"], 1_000_000.0)
            classification = json.loads(
                (root / "PILOT_CLASSIFICATION.json").read_text(encoding="ascii")
            )
            self.assertEqual(
                classification["radio_override"],
                {
                    "effective_bandwidth_hz": 16_000_000.0,
                    "field": "radio.subcarrier_spacing_hz",
                    "original_value_hz": 30_000.0,
                    "override_value_hz": 1_000_000.0,
                    "scope": "FORBIDDEN_DEVELOPMENT_PILOT_ONLY",
                },
            )

    def test_post_rt_exclusion_preserves_original_rank_order(self) -> None:
        candidates = [
            (100.0, 10.0, 0.0, 0.0, []),
            (99.0, 10.0, 300.0, 0.0, []),
            (98.0, 10.0, 600.0, 0.0, []),
        ]

        def qualify(row, bank_index):
            return {
                "bank_index_within_city": bank_index,
                "bs_utm_xy_m": [row[2], row[3]],
                "scene_index": bank_index,
            }

        selected, selection_audit = pilot._select_excluding_post_rt_centers(
            pilot.candidate._select_reflection_qualified_bank_candidates,
            candidates,
            2,
            map_extent_m=256.0,
            qualify=qualify,
            excluded_centers=((0.0, 0.0),),
        )
        self.assertEqual(
            [row["bs_utm_xy_m"] for row in selected],
            [[300.0, 0.0], [600.0, 0.0]],
        )
        self.assertEqual(selection_audit["candidates_examined"], 3)
        self.assertEqual(selection_audit["post_rt_exclusion_count"], 1)
        self.assertEqual(
            [row["candidate_rank"] for row in selection_audit["selected_candidate_records"]],
            [1, 2],
        )

    def test_load_prepared_accepts_only_config_bound_v3_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self._prepared_root(Path(temporary))
            records = pilot.load_prepared(prepared, CONFIG_PATH, 4)
        self.assertEqual([row["scene_index"] for row in records], [4, 5, 6, 7])

    def test_load_prepared_selects_strict_subwindow_from_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self._prepared_root(Path(temporary), indices=list(range(8)))
            records = pilot.load_prepared(
                prepared,
                CONFIG_PATH,
                4,
                bank_start=4,
                city_id="source-austin",
            )
            self.assertEqual([row["scene_index"] for row in records], [4, 5, 6, 7])
            with self.assertRaisesRegex(RuntimeError, "outside the prepared inventory"):
                pilot.load_prepared(
                    prepared,
                    CONFIG_PATH,
                    4,
                    bank_start=6,
                    city_id="source-austin",
                )

    def test_load_prepared_defaults_to_inventory_declared_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self._prepared_root(
                Path(temporary), indices=[32, 33, 34, 35], bank_start=16
            )
            records = pilot.load_prepared(prepared, CONFIG_PATH, 4)
            self.assertEqual(
                [row["scene_index"] for row in records], [32, 33, 34, 35]
            )

    def test_copy_prepared_assets_copies_only_selected_bank_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self._prepared_root(root, indices=list(range(8)))
            raw = prepared / "assets" / "raw_osm"
            raw.mkdir()
            (raw / "source-austin.json").write_text("{}", encoding="ascii")
            records = pilot.load_prepared(
                prepared,
                CONFIG_PATH,
                4,
                bank_start=4,
                city_id="source-austin",
            )
            output = root / "window"
            output.mkdir()
            pilot._copy_prepared_assets(prepared, output, records)
            copied = sorted(
                path.parent.name
                for path in (output / "assets" / "banks").glob("*/bank.json")
            )
            self.assertEqual(
                copied,
                [f"osm-sionna-source-austin-bank-{index:02d}" for index in range(4, 8)],
            )

    def test_load_prepared_rejects_any_primitive_below_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self._prepared_root(Path(temporary), deficient=True)
            with self.assertRaisesRegex(RuntimeError, "failed V5 geometry qualification"):
                pilot.load_prepared(prepared, CONFIG_PATH, 4)

    def test_load_prepared_rejects_any_primitive_below_quality_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self._prepared_root(Path(temporary), weak_quality=True)
            with self.assertRaisesRegex(RuntimeError, "failed V5 geometry qualification"):
                pilot.load_prepared(prepared, CONFIG_PATH, 4)

    def test_qualification_pilot_selects_banks_across_both_source_cities(self) -> None:
        dataset = SimpleNamespace(
            scene_roles=np.asarray(
                ["source_encoder_train"] * 4 + ["source_method_selection"] * 4
            ),
            city_ids=np.asarray(
                [
                    "source-chicago",
                    "source-chicago",
                    "source-austin",
                    "source-austin",
                    "source-chicago",
                    "source-chicago",
                    "source-austin",
                    "source-austin",
                ]
            ),
        )
        train = qualification_pilot._select_city_stratified_scenes(
            dataset,
            np.asarray([0, 1, 2, 3]),
            2,
            expected_role="source_encoder_train",
        )
        selection = qualification_pilot._select_city_stratified_scenes(
            dataset,
            np.asarray([4, 5, 6, 7]),
            2,
            expected_role="source_method_selection",
        )
        self.assertEqual(train.tolist(), [0, 2])
        self.assertEqual(selection.tolist(), [4, 6])
        self.assertEqual(set(dataset.city_ids[train]), {"source-chicago", "source-austin"})
        self.assertEqual(
            set(dataset.city_ids[selection]), {"source-chicago", "source-austin"}
        )

    def test_qualification_pilot_rejects_two_bank_single_city_selection(self) -> None:
        dataset = SimpleNamespace(
            scene_roles=np.asarray(["source_encoder_train"] * 2),
            city_ids=np.asarray(["source-chicago", "source-chicago"]),
        )
        with self.assertRaisesRegex(RuntimeError, "requires 2 source cities"):
            qualification_pilot._select_city_stratified_scenes(
                dataset,
                np.asarray([0, 1]),
                2,
                expected_role="source_encoder_train",
            )

    def test_qualification_pilot_writes_only_strict_json_exclusively(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pilot.json"
            with self.assertRaises(ValueError):
                qualification_pilot._write_strict_json_exclusive(
                    output, {"invalid": float("nan")}
                )
            self.assertFalse(output.exists())
            qualification_pilot._write_strict_json_exclusive(
                output, {"valid": None}
            )
            self.assertEqual(json.loads(output.read_text()), {"valid": None})
            with self.assertRaises(FileExistsError):
                qualification_pilot._write_strict_json_exclusive(
                    output, {"valid": None}
                )

    def test_wideband_teacher_pilot_rejects_non_training_role(self) -> None:
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit_root = root / "fit"
            snapshot = fit_root / "assets" / "generator_config.json"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text(json.dumps(config), encoding="ascii")
            args = SimpleNamespace(
                fit_root=fit_root,
                candidate_config=CONFIG_PATH,
                formal_config=ROOT / "formal_v2" / "configs" / "formal_v2.json",
                output=root / "output",
                device="cpu",
                seed=20271010,
            )
            with (
                mock.patch.object(teacher_pilot, "configure_reproducible_runtime"),
                mock.patch.object(
                    teacher_pilot,
                    "_load_split",
                    return_value=(
                        [{}],
                        [
                            {
                                "role": "source_method_selection",
                                "base_map_cluster_id": "cluster",
                            }
                        ],
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "only source_encoder_train"
                ):
                    teacher_pilot.run(args)

    def test_wideband_teacher_pilot_rejects_config_snapshot_mismatch(self) -> None:
        config = load_config(CONFIG_PATH)
        mutated = json.loads(json.dumps(config))
        mutated["radio"]["subcarrier_spacing_hz"] = 1_000_000.0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit_root = root / "fit"
            snapshot = fit_root / "assets" / "generator_config.json"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text(json.dumps(mutated), encoding="ascii")
            args = SimpleNamespace(
                fit_root=fit_root,
                candidate_config=CONFIG_PATH,
                formal_config=ROOT / "formal_v2" / "configs" / "formal_v2.json",
                output=root / "output",
                device="cpu",
                seed=20271010,
            )
            with mock.patch.object(
                teacher_pilot, "configure_reproducible_runtime"
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "differs from fit snapshot"
                ):
                    teacher_pilot.run(args)

    def test_wideband_teacher_pilot_multiroot_inventory_is_unique(self) -> None:
        first = Path("/tmp/teacher-first")
        second = Path("/tmp/teacher-second")
        self.assertEqual(
            teacher_pilot._normalize_roots([first, second]),
            (first.resolve(), second.resolve()),
        )
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            teacher_pilot._normalize_roots([first, first])

    def test_formal_response_pilot_position_stride_requires_real_coverage(self) -> None:
        self.assertEqual(
            formal_response_pilot._evaluation_positions(256, 4).size,
            64,
        )
        with self.assertRaisesRegex(ValueError, "at least 16 positions"):
            formal_response_pilot._evaluation_positions(256, 32)

    def test_formal_response_pilot_expands_ranked_cached_actions(self) -> None:
        encoded = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        expanded = formal_response_pilot._expand_cached_action(encoded, 4)
        self.assertEqual(tuple(expanded.shape), (4, 2, 3))
        for row in expanded:
            torch.testing.assert_close(row, encoded, rtol=0.0, atol=0.0)
        with self.assertRaisesRegex(ValueError, "non-scalar action"):
            formal_response_pilot._expand_cached_action(torch.tensor(1.0), 4)
        with self.assertRaisesRegex(ValueError, "positive batch"):
            formal_response_pilot._expand_cached_action(encoded, 0)

    def test_formal_response_active_focus_is_normalized_and_differentiable(self) -> None:
        active_loss = torch.tensor(0.0125, requires_grad=True)
        focused = formal_response_pilot._normalized_active_focus(
            active_loss, 0.0125, 2.0
        )
        torch.testing.assert_close(focused, torch.tensor(2.0))
        focused.backward()
        torch.testing.assert_close(active_loss.grad, torch.tensor(160.0))
        with self.assertRaisesRegex(ValueError, "scale must be positive"):
            formal_response_pilot._normalized_active_focus(active_loss, 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "multiplier must be positive"):
            formal_response_pilot._normalized_active_focus(
                active_loss, 1.0, float("nan")
            )

    def test_formal_response_continuation_requires_bound_optimizer_checkpoint(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.json"
            formal = root / "formal.json"
            teacher = root / "teacher.pt"
            for path, value in (
                (candidate, "candidate"),
                (formal, "formal"),
                (teacher, "teacher"),
            ):
                path.write_text(value, encoding="ascii")
            run_root = root / "source-run"
            run_root.mkdir()
            checkpoint = run_root / "response_model_before_resume.pt"
            torch.save(
                {
                    "schema_version": formal_response_pilot.CHECKPOINT_SCHEMA,
                    **formal_response_pilot.CLASSIFICATION,
                    "model_contract": formal_response_pilot.MODEL_CONTRACT,
                    "training_objective_contract": (
                        formal_response_pilot.TRAINING_OBJECTIVE_CONTRACT
                    ),
                    "model_source_sha256": formal_response_pilot.sha256_file(
                        formal_response_pilot.MODEL_SOURCE
                    ),
                    "training_source_sha256": formal_response_pilot.sha256_file(
                        formal_response_pilot.TRAINING_SOURCE
                    ),
                    "training_dependency_sha256": (
                        formal_response_pilot._training_dependency_sha256()
                    ),
                    "seed": 20270001,
                    "completed_steps": 512,
                    "response_scale": 2.0,
                    "response_active_scale": 0.0125,
                    "branch_contrast_scale": 0.002,
                    "response_delta_multiplier": 1.0,
                    "response_null_multiplier": 1.0,
                    "response_active_focus_multiplier": 1.0,
                    "response_branch_contrast_multiplier": 1.0,
                    "model_state": {"weight": torch.ones(1)},
                    "optimizer_state": {"state": {}, "param_groups": []},
                },
                checkpoint,
            )
            receipt = {
                **formal_response_pilot.CLASSIFICATION,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": formal_response_pilot.sha256_file(checkpoint),
                "seed": 20270001,
                "steps": 512,
                "checkpoint_completed_steps": 512,
                "candidate_config_sha256": formal_response_pilot.sha256_file(
                    candidate
                ),
                "formal_config_sha256": formal_response_pilot.sha256_file(formal),
                "teacher_sha256": formal_response_pilot.sha256_file(teacher),
                "model_contract": formal_response_pilot.MODEL_CONTRACT,
                "training_objective_contract": (
                    formal_response_pilot.TRAINING_OBJECTIVE_CONTRACT
                ),
                "model_source_sha256": formal_response_pilot.sha256_file(
                    formal_response_pilot.MODEL_SOURCE
                ),
                "training_source_sha256": formal_response_pilot.sha256_file(
                    formal_response_pilot.TRAINING_SOURCE
                ),
                "training_dependency_sha256": (
                    formal_response_pilot._training_dependency_sha256()
                ),
                "response_active_scale": 0.0125,
                "response_active_focus_multiplier": 1.0,
            }
            (run_root / "pilot.json").write_text(
                json.dumps(receipt), encoding="ascii"
            )
            payload, receipt_path, loaded_receipt = (
                formal_response_pilot._validated_initial_checkpoint(
                    checkpoint.resolve(),
                    seed=20270001,
                    candidate_path=candidate,
                    formal_path=formal,
                    teacher_path=teacher,
                )
            )
            self.assertEqual(payload["completed_steps"], 512)
            self.assertEqual(receipt_path, run_root / "pilot.json")
            self.assertEqual(loaded_receipt["seed"], 20270001)

            receipt["response_active_focus_multiplier"] = 2.0
            (run_root / "pilot.json").write_text(
                json.dumps(receipt), encoding="ascii"
            )
            with self.assertRaisesRegex(RuntimeError, "payload binding failed"):
                formal_response_pilot._validated_initial_checkpoint(
                    checkpoint.resolve(),
                    seed=20270001,
                    candidate_path=candidate,
                    formal_path=formal,
                    teacher_path=teacher,
                )
            receipt["response_active_focus_multiplier"] = 1.0

            receipt.pop("checkpoint_completed_steps")
            receipt["initial_completed_steps"] = 128
            receipt["additional_training_steps"] = 384
            (run_root / "pilot.json").write_text(
                json.dumps(receipt), encoding="ascii"
            )
            derived, _, _ = formal_response_pilot._validated_initial_checkpoint(
                checkpoint.resolve(),
                seed=20270001,
                candidate_path=candidate,
                formal_path=formal,
                teacher_path=teacher,
            )
            self.assertEqual(derived["completed_steps"], 512)

            receipt["checkpoint_sha256"] = "0" * 64
            (run_root / "pilot.json").write_text(
                json.dumps(receipt), encoding="ascii"
            )
            with self.assertRaisesRegex(RuntimeError, "receipt binding failed"):
                formal_response_pilot._validated_initial_checkpoint(
                    checkpoint.resolve(),
                    seed=20270001,
                    candidate_path=candidate,
                    formal_path=formal,
                    teacher_path=teacher,
                )

            checkpoint_payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            checkpoint_payload["schema_version"] = (
                "csi-pairs-v5-formal-response-pilot-checkpoint-v1"
            )
            torch.save(checkpoint_payload, checkpoint)
            receipt["checkpoint_sha256"] = formal_response_pilot.sha256_file(
                checkpoint
            )
            (run_root / "pilot.json").write_text(
                json.dumps(receipt), encoding="ascii"
            )
            with self.assertRaisesRegex(RuntimeError, "payload binding failed"):
                formal_response_pilot._validated_initial_checkpoint(
                    checkpoint.resolve(),
                    seed=20270001,
                    candidate_path=candidate,
                    formal_path=formal,
                    teacher_path=teacher,
                )

    def test_action_overfit_wrong_action_preserves_source_state(self) -> None:
        corpus = SimpleNamespace(
            response_targets_by_source_query={(1, 2, 3, 4): (5, 6, 7)}
        )
        units = ((1, 2, 6, 3, 4),)
        self.assertEqual(
            action_overfit._wrong_action_units(corpus, units),
            ((1, 2, 5, 3, 4),),
        )

    def test_action_overfit_gradient_audit_includes_response_attention(self) -> None:
        class ResponseAuditModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.response_state_attention = torch.nn.Linear(1, 1, bias=False)
                self.response_spatial_attention = torch.nn.Linear(1, 1, bias=False)

        model = ResponseAuditModel()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        groups = action_overfit._gradient_groups(model)
        self.assertEqual(groups["action"], 2.0**0.5)

    def test_nonlinear_pilot_rejects_role_boundary_violation(self) -> None:
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit_root = root / "fit"
            eval_root = root / "eval"
            for split in (fit_root, eval_root):
                snapshot = split / "assets" / "generator_config.json"
                snapshot.parent.mkdir(parents=True)
                snapshot.write_text(json.dumps(config), encoding="ascii")
            args = SimpleNamespace(
                fit_root=fit_root,
                eval_root=eval_root,
                candidate_config=CONFIG_PATH,
                formal_config=ROOT / "formal_v2" / "configs" / "formal_v2.json",
                output=root / "output",
                stride=16,
            )
            wrong_role = [
                {
                    "scene_index": 1,
                    "scene_id": "fit-scene",
                    "bank_id": "fit-bank",
                    "role": "source_method_selection",
                    "base_map_cluster_id": "fit-cluster",
                }
            ]
            eval_role = [
                {
                    "scene_index": 2,
                    "scene_id": "eval-scene",
                    "bank_id": "eval-bank",
                    "role": "source_method_selection",
                    "base_map_cluster_id": "eval-cluster",
                }
            ]
            with (
                mock.patch.object(nonlinear_pilot, "configure_reproducible_runtime"),
                mock.patch.object(
                    nonlinear_pilot,
                    "_load_split",
                    side_effect=[([{}], wrong_role), ([{}], eval_role)],
                ),
                mock.patch.object(
                    nonlinear_pilot, "_split_bank_bindings", return_value=[]
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "only source_encoder_train"
                ):
                    nonlinear_pilot.run(args)

    def test_nonlinear_pilot_multiroot_inventory_is_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            self.assertEqual(
                nonlinear_pilot._normalize_roots([first, second]),
                (first.resolve(), second.resolve()),
            )
            with self.assertRaisesRegex(RuntimeError, "must be unique"):
                nonlinear_pilot._normalize_roots([first, first])

    def test_generalization_position_holdout_is_bank_balanced(self) -> None:
        records = [
            {"position_index": position}
            for _bank in range(3)
            for position in range(0, 64, 16)
            for _query in range(2)
        ]
        holdout = generalization_diagnostic._position_holdout_mask(
            records, stride=16, fold=1, folds=4
        )
        self.assertEqual(int(np.sum(holdout)), 6)
        selected = [
            row["position_index"]
            for row, keep in zip(records, holdout, strict=True)
            if keep
        ]
        self.assertEqual(set(selected), {16})

    def test_generalization_position_holdout_rejects_wrong_stride(self) -> None:
        with self.assertRaisesRegex(ValueError, "requested stride"):
            generalization_diagnostic._position_holdout_mask(
                [{"position_index": 8}], stride=16, fold=0, folds=4
            )

    def test_physics_probe_decomposes_each_edit_against_direct_path(self) -> None:
        cache = physics_probe.PhysicsCache.__new__(physics_probe.PhysicsCache)
        cache.spec = physics_probe.PatchSpec(2, 2, 1, 1, 1)
        cache.dataset = SimpleNamespace(
            csi=np.zeros((1, 1, 1, 8), dtype=np.float64)
        )
        direct = np.ones((2, 2), dtype=np.complex128)
        first = np.eye(2, dtype=np.complex128)
        second = np.fliplr(first)
        cache.dictionaries = {(0, 0): ([direct, first, second], [-1, 0, 1])}
        cache.masks = {0: np.ones((2, 2), dtype=np.bool_)}
        cache.ridge = 1.0e-4
        with mock.patch.object(
            physics_probe,
            "decompose_visible_csi",
            side_effect=[np.asarray((1.0, 2.0)), np.asarray((3.0, 4.0))],
        ) as decompose:
            coefficients = cache.source_coefficients(0, 0, 0, 0)
        self.assertEqual(coefficients, {0: 2.0 + 0.0j, 1: 4.0 + 0.0j})
        self.assertEqual(len(decompose.call_args_list), 2)
        self.assertEqual(decompose.call_args_list[0].args[1], [direct, first])
        self.assertEqual(decompose.call_args_list[1].args[1], [direct, second])
        self.assertIs(cache.source_coefficients(0, 0, 0, 0), coefficients)
        self.assertEqual(len(decompose.call_args_list), 2)


class V5SplitAuditTests(unittest.TestCase):
    def test_external_fit_scale_controls_evaluation_route(self) -> None:
        config = load_config(CONFIG_PATH)
        formal = json.loads(
            (ROOT / "formal_v2" / "configs" / "formal_v2.json").read_text(
                encoding="ascii"
            )
        )
        bits = np.asarray(config["world_bits"], dtype=np.float64)
        evaluation = np.repeat(bits.sum(axis=1)[None, :, None, None], 64, axis=3)
        small_fit = np.repeat(
            np.where(np.arange(16) % 2 == 0, -1.0, 1.0)[None, :, None, None],
            64,
            axis=3,
        )
        large_fit = 100.0 * small_fit
        _, small_routes, _ = audit._physical_routes(
            evaluation, config, formal, normalization_csi=small_fit
        )
        _, large_routes, _ = audit._physical_routes(
            evaluation, config, formal, normalization_csi=large_fit
        )
        self.assertEqual(small_routes[(0, 0, 1, 0, 0)], 2)
        self.assertEqual(large_routes[(0, 0, 1, 0, 0)], 0)

    def test_audit_rejects_self_normalization_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "must be different"):
                audit.audit(
                    root,
                    root,
                    CONFIG_PATH,
                    ROOT / "formal_v2" / "configs" / "formal_v2.json",
                    root / "audit.json",
                )


if __name__ == "__main__":
    unittest.main()
