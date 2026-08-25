from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from formal_v2.formal_claims import (
    CLAIM_DEPENDENCIES,
    _claim_state,
    _semantic_status,
    _validate_critical_chain_binding,
    _validate_external_manifest_binding,
    _validate_stage_bound_input,
    assemble_claim_evidence,
)
from formal_v2.formal_cli import (
    COMMAND_OUTPUT_PATHS,
    DEFAULT_RESOURCE_CONTROL_MANIFEST,
    DEFAULT_RETENTION_MANIFEST,
    DEFAULT_SCENE_ID_MANIFEST,
    DEFAULT_SHUFFLED_PAIR_MANIFEST,
    SELF_RESERVING_DIRECTORY_COMMANDS,
    _acquire_output_lock,
    _reserve_command_output,
    _reserve_full_run_output,
    _result_exit_code,
    _sionna_export_lock_root,
    build_parser,
    main as formal_cli_main,
)
from formal_v2.formal_controls import CONTROL_IDS, _validate_manifest as validate_control_manifest
from formal_v2.formal_config import load_formal_config, validate_formal_config
from formal_v2.formal_data_verification import require_data_verification, require_verified_roles
from formal_v2.formal_dataset import FormalDataset, FormalDatasetError, SOURCE_ROLES
from formal_v2.formal_evidence import (
    CLAIM_IDS,
    GATE_IDS,
    QUALIFICATION_SCHEMA,
    complete_gate_vector,
    configure_reproducible_runtime,
    evidence_context,
    require_manifested_formal_qualification,
    require_formal_qualification,
    require_stage_manifested_gate,
    runtime_provenance,
)
from formal_v2.formal_factorial import (
    _target_cluster_coverage,
    city_support_candidates,
    eligible_query_indices,
)
from formal_v2.formal_evaluation import _eligible_evaluation_positions, _paired_score_differences
from formal_v2.formal_external_validity import (
    _authenticate_sionna_runtime,
    _cluster_direction_interval,
    _validate_manifest as validate_external_validity_manifest,
)
from formal_v2.formal_external import (
    _c1_model_assessment,
    _validate_manifest as validate_external_manifest,
    _validate_execution_manifest as validate_external_execution_manifest,
    _validate_six_condition_rows,
)
from formal_v2.external_adapters.controlled_map_adapter import (
    _batch as controlled_batch,
    _build_model as build_controlled_model,
    _fit_data_normalizer,
    _loss as controlled_loss,
    _train as train_controlled_model,
    _training_task,
    load_controlled_map_config,
)
from formal_v2.external_adapters.representation_models import CSIMAE, build_representation_model
from formal_v2.external_adapters.controlled_map_models import (
    deterministic_prefix_product,
)
from formal_v2.external_adapters.wigatr_adapter import (
    _audit_mesh_preprocessing,
    _require_official_cuda,
    _verify_vendor_tree,
    inverse_localize_power,
)
from formal_v2.external_adapters.sionna_external_validity import (
    _numpy_cfr as sionna_numpy_cfr,
    _point3 as sionna_point3,
    _quaternion_to_euler as sionna_quaternion_to_euler,
    load_scene_manifest,
)
from formal_v2.external_adapters.wigatr_protocol import (
    SIX_CONDITIONS,
    build_six_condition_units,
    condition_map,
    geometry_destroyed_map,
    grid_to_cellwise_triangular_mesh,
    grid_to_triangular_mesh,
    load_wigatr_config,
    relative_total_power_db,
    require_compact_mesh_matches_map_surface,
    require_surface_ledger_equivalence,
)
from formal_v2.formal_fixture import write_nonscientific_fixture
from formal_v2.formal_io import (
    StrictJsonError,
    artifact_manifest,
    parse_strict_json,
    read_strict_json,
    sha256_file,
    write_csv,
    write_json,
)
from formal_v2.formal_model import (
    CSIPairsFormalModel,
    _CoordinateActionMoments,
    required_mean,
)
from formal_v2.formal_path import _noop_path_threshold, localization_path_incidence, path_incidence
from formal_v2.formal_protocol import (
    PatchSpec,
    delay_angle_power,
    frozen_mask_query_bank,
    patchify_csi,
    typed_signed_edit,
    unpatchify_csi,
)
from formal_v2.formal_representation_baselines import (
    _fit_normalizer as fit_representation_normalizer,
    _make_batch as make_representation_batch,
    _warm_start_contra,
    load_representation_config,
)
from formal_v2.formal_resources import validate_resource_registry_structure
from formal_v2.formal_qualification import qualification_blocking_scenes
from formal_v2.formal_literature import _validate_manifest as validate_literature_manifest
from formal_v2.formal_llm_judge import C13_JUDGE_ATTESTATION
from formal_v2.formal_rt_calibration import (
    _join_and_assess as assess_rt_calibration,
    _read_partition_contract as read_rt_partition_contract,
    _validate_partition_independence as validate_rt_partition_independence,
    _validate_manifest as validate_rt_calibration_manifest,
    run_rt_calibration_gate,
)
from formal_v2.formal_risk import (
    TemperatureCalibration,
    _coverage_error_monotonic,
    fit_constrained_risk_calibrator,
    frozen_map_proposals,
    randomized_candidate_labels,
)
from formal_v2.formal_routing import (
    PRIMARY_ROUTE_CONTRACT,
    route_code,
    teacher_sensitivity_code,
)
from formal_v2.formal_scene_id import (
    _model_assessment as scene_id_model_assessment,
    _validate_manifest as validate_scene_id_manifest,
    _validate_rows as validate_scene_id_rows,
    _validate_training_provenance as validate_scene_id_training_provenance,
)
from formal_v2.sionna_scene_export import export_sionna_scenes
from formal_v2.formal_statistics import (
    exact_factorial_utilities,
    hierarchical_factorial_interval,
    holm_adjust,
    paired_sign_flip_test,
)
from formal_v2.formal_teacher import CSIMaskedTeacher


ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = ROOT / "formal_v2" / "configs" / "formal_v2_smoke.json"


class ConfigTests(unittest.TestCase):
    def test_full_run_shell_requires_an_explicit_two_phase_mode(self):
        script = ROOT / "formal_v2" / "scripts" / "run_formal_v2.sh"
        completed = subprocess.run(
            ["/bin/bash", str(script)],
            cwd=ROOT,
            env={},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 5)
        self.assertIn("CSI_PAIRS_FULL_RUN_PHASE=prepare", completed.stderr)
        self.assertNotIn("CSI_PAIRS_FORMAL_OUTPUT", completed.stderr)

    def test_config_is_v6_and_has_complete_sections(self):
        config = load_formal_config(SMOKE_CONFIG)
        self.assertEqual(config["schema_version"], "csi-pairs-formal-config-v2.3-v6")
        self.assertEqual(len(config["seeds"]), 3)
        self.assertEqual(config["factorial"]["arms"], ["endpoint", "alignment", "response", "full"])
        self.assertEqual(set(("teacher", "evaluation", "risk", "path")).difference(config), set())
        self.assertIn("external_validity", config)
        self.assertIn("literature", config)
        self.assertEqual(
            config["data"][
                "minimum_independent_base_map_clusters_per_target_city"
            ],
            2,
        )

    def test_formal_power_minima_are_frozen_in_config(self):
        config = load_formal_config(
            ROOT / "formal_v2" / "configs" / "formal_v2.json"
        )
        self.assertEqual(
            config["data"][
                "minimum_independent_base_map_clusters_per_target_city"
            ],
            41,
        )
        self.assertEqual(config["data"]["minimum_banks_per_target_city"], 41)
        self.assertEqual(config["data"]["minimum_banks_per_source_role"], 8)
        self.assertEqual(
            config["data"]["minimum_independent_source_final_unseen_clusters"],
            41,
        )
        self.assertEqual(
            config["data"]["minimum_independent_external_validation_clusters"],
            32,
        )
        rows = [
            {
                "city_id": city,
                "base_map_cluster_id": f"declared-{city}-{index}",
                "canonical_base_map_digest": f"canonical-{city}-{index}",
            }
            for city in ("target-a", "target-b")
            for index in range(2)
        ]
        smoke = load_formal_config(SMOKE_CONFIG)
        self.assertTrue(_target_cluster_coverage(smoke, rows)["passed"])
        self.assertFalse(_target_cluster_coverage(config, rows)["passed"])

    def test_rejects_two_seeds(self):
        config = load_formal_config(SMOKE_CONFIG)
        config["seeds"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "at least three"):
            validate_formal_config(config)

    def test_rejects_nonfactorial_order(self):
        config = load_formal_config(SMOKE_CONFIG)
        config["factorial"]["arms"] = ["full", "endpoint", "alignment", "response"]
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_formal_config(config)

    def test_fixture_can_cover_cross_source_city_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            default_path = write_nonscientific_fixture(
                Path(temporary) / "default.npz"
            )
            default_dataset = FormalDataset.load(default_path)
            self.assertEqual(
                set(default_dataset.city_ids[:-4].tolist()),
                {"source-a", "source-b"},
            )
            path = write_nonscientific_fixture(
                Path(temporary) / "fixture.npz",
                scene_count=2 * len(SOURCE_ROLES) + 4,
                source_banks_per_role=2,
            )
            dataset = FormalDataset.load(path)
            for role in SOURCE_ROLES:
                indices = dataset.indices_for_role(role)
                self.assertEqual(indices.size, 2)
                self.assertEqual(
                    set(dataset.city_ids[indices].tolist()),
                    {"source-a", "source-b"},
                )

    def test_fixture_receivers_remain_inside_common_free_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_nonscientific_fixture(
                Path(temporary) / "many-scenes.npz",
                scene_count=96,
                positions=8,
            )
            dataset = FormalDataset.load(path)
            representation = dataset.metadata["representation"]
            origin = np.asarray(representation["map_origin_xy_m"])
            resolution = float(representation["map_resolution_m"])
            grid = (dataset.positions - origin) / resolution
            cells = np.floor(grid).astype(np.int64)
            offsets = grid - cells
            self.assertTrue(np.all(offsets > 0.09))
            self.assertTrue(np.all(offsets < 0.91))

            occupancy = tuple(dataset.map_channel_names.tolist()).index("occupancy")
            for scene in range(dataset.scene_count):
                columns = cells[scene, :, 0]
                rows = cells[scene, :, 1]
                values = dataset.maps[scene, :, occupancy][:, rows, columns]
                self.assertTrue(np.all(values < 0.5))

    def test_rejects_teacher_state_initialization_mismatch(self):
        config = load_formal_config(SMOKE_CONFIG)
        config["teacher"]["latent_dim"] += 1
        with self.assertRaisesRegex(ValueError, "shared CSI initialization"):
            validate_formal_config(config)

    def test_rejects_non_v6_teacher_mask_fraction(self):
        config = load_formal_config(SMOKE_CONFIG)
        config["teacher"]["mask_fraction"] = 0.25
        with self.assertRaisesRegex(ValueError, "teacher.mask_fraction"):
            validate_formal_config(config)

    def test_rejects_disabling_clean_physical_targets(self):
        config = load_formal_config(SMOKE_CONFIG)
        config["data"]["require_clean_csi"] = False
        with self.assertRaisesRegex(ValueError, "requires data.require_clean_csi=true"):
            validate_formal_config(config)

    def test_rejects_zero_risk_audit_stratum(self):
        config = load_formal_config(SMOKE_CONFIG)
        config["risk"]["audit_mixture"]["gray"] = 0.0
        config["risk"]["audit_mixture"]["correct"] += 0.2
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            validate_formal_config(config)

    def test_rejects_single_point_path_coverage_audit(self):
        config = load_formal_config(SMOKE_CONFIG)
        config["path"]["power_coverage"] = 1.0
        with self.assertRaisesRegex(ValueError, "strictly between"):
            validate_formal_config(config)

    def test_individual_cli_stages_refuse_to_overwrite_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for command, relative_path in COMMAND_OUTPUT_PATHS.items():
                with self.subTest(command=command):
                    target = output / relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if command == "inspect-data":
                        target.write_text("existing evidence\n", encoding="utf-8")
                    else:
                        target.mkdir()
                    with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                        _reserve_command_output(command, output)
                    if target.is_dir():
                        target.rmdir()
                    else:
                        target.unlink()
                    reserved = _reserve_command_output(command, output)
                    self.assertEqual(reserved, target)
                    if command in SELF_RESERVING_DIRECTORY_COMMANDS:
                        self.assertFalse(target.exists())
                    else:
                        self.assertTrue(target.exists())
                        if target.is_dir():
                            target.rmdir()
                        else:
                            target.unlink()
            self.assertIsNone(_reserve_command_output("all", output))

    def test_stage_output_reservation_is_atomic(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"

            def reserve(_):
                try:
                    return _reserve_command_output("run-risk", output)
                except FileExistsError:
                    return None

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(reserve, range(2)))
            self.assertEqual(sum(result is not None for result in results), 1)
            self.assertEqual(sum(result is None for result in results), 1)

    def test_output_root_operation_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            first = _acquire_output_lock(output)
            with self.assertRaisesRegex(FileExistsError, "operation lock exists"):
                _acquire_output_lock(output)
            first.unlink()
            second = _acquire_output_lock(output)
            self.assertTrue(second.is_file())
            second.unlink()

    def test_output_root_operation_lock_recovers_after_owner_crash(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            crashed = _acquire_output_lock(output)
            crashed.guard_handle.close()
            crashed.released = True
            self.assertTrue(crashed.path.is_file())
            recovered = _acquire_output_lock(output)
            self.assertTrue(recovered.is_file())
            recovered.unlink()

    def test_resource_verification_respects_run_root_lock_and_releases_after_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            lock = _acquire_output_lock(output)
            status = formal_cli_main(
                [
                    "verify-waibu-resources",
                    "--registry",
                    str(ROOT / "formal_v2" / "configs" / "waibu_resources_v1.json"),
                    "--waibu-root",
                    str(ROOT / "waibu"),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(status, 2)
            self.assertFalse((output / "waibu_resources").exists())
            lock.unlink()

            invalid_registry = root / "invalid-registry.json"
            invalid_registry.write_text("{}\n", encoding="utf-8")
            status = formal_cli_main(
                [
                    "verify-waibu-resources",
                    "--registry",
                    str(invalid_registry),
                    "--waibu-root",
                    str(ROOT / "waibu"),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(status, 2)
            self.assertFalse(
                (output.parent / f".{output.name}.csi-pairs-operation.lock").exists()
            )

    def test_resource_verification_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            marker = output / "waibu_resources" / "keep.txt"
            marker.parent.mkdir(parents=True)
            marker.write_text("existing evidence\n", encoding="utf-8")
            status = formal_cli_main(
                [
                    "verify-waibu-resources",
                    "--registry",
                    str(ROOT / "formal_v2" / "configs" / "waibu_resources_v1.json"),
                    "--waibu-root",
                    str(ROOT / "waibu"),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(status, 2)
            self.assertEqual(marker.read_text(encoding="utf-8"), "existing evidence\n")
            self.assertFalse(
                (output.parent / f".{output.name}.csi-pairs-operation.lock").exists()
            )

    def test_resource_lock_excludes_resource_and_normal_stage_concurrently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = write_nonscientific_fixture(root / "fixture.npz")
            output = root / "run"
            started = threading.Event()
            release = threading.Event()

            def blocked_verifier(*_args):
                started.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("test verifier release timed out")
                return {"status": "PASS", "passed": True}

            resource_args = [
                "verify-waibu-resources",
                "--registry",
                str(ROOT / "formal_v2" / "configs" / "waibu_resources_v1.json"),
                "--waibu-root",
                str(ROOT / "waibu"),
                "--output",
                str(output),
            ]
            inspect_args = [
                "inspect-data",
                "--config",
                str(SMOKE_CONFIG),
                "--dataset",
                str(dataset),
                "--output",
                str(output),
            ]
            with patch(
                "formal_v2.formal_resources.verify_waibu_resources",
                side_effect=blocked_verifier,
            ) as verifier, ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(formal_cli_main, resource_args)
                self.assertTrue(started.wait(timeout=5))
                self.assertEqual(formal_cli_main(resource_args), 2)
                self.assertEqual(formal_cli_main(inspect_args), 2)
                release.set()
                self.assertEqual(first.result(timeout=5), 0)
            self.assertEqual(verifier.call_count, 1)
            self.assertFalse((output / "data_contract.json").exists())
            self.assertFalse(
                (output.parent / f".{output.name}.csi-pairs-operation.lock").exists()
            )

    def test_sionna_export_respects_parent_run_root_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = write_nonscientific_fixture(root / "fixture.npz")
            output = root / "run" / "inputs"
            lock = _acquire_output_lock(output.parent)
            status = formal_cli_main(
                [
                    "export-sionna-scenes",
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                    "--license-id",
                    "GENERATED-FIXTURE-NO-EXTERNAL-ASSET",
                    "--carrier-frequency-hz",
                    "3500000000",
                    "--subcarrier-spacing-hz",
                    "30000",
                ]
            )
            self.assertEqual(status, 2)
            self.assertFalse(output.exists())
            lock.unlink()

    def test_sionna_export_lock_root_only_treats_inputs_as_reserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.assertEqual(_sionna_export_lock_root(root / "run" / "inputs"), root / "run")
            self.assertEqual(_sionna_export_lock_root(root / "scenes"), root / "scenes")

    def test_failed_sionna_export_does_not_leave_partial_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = write_nonscientific_fixture(root / "fixture.npz")
            output = root / "run" / "inputs"
            status = formal_cli_main(
                [
                    "export-sionna-scenes",
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                    "--license-id",
                    "GENERATED-FIXTURE-NO-EXTERNAL-ASSET",
                    "--carrier-frequency-hz",
                    "3500000000",
                    "--subcarrier-spacing-hz",
                    "30000",
                ]
            )
            self.assertEqual(status, 2)
            self.assertFalse(output.exists())
            self.assertFalse(
                (output.parent.parent / f".{output.parent.name}.csi-pairs-operation.lock").exists()
            )

    def test_full_run_reservation_accepts_only_new_root_or_staged_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh = root / "fresh"
            _reserve_full_run_output(fresh)
            self.assertTrue(fresh.is_dir())
            with self.assertRaisesRegex(FileExistsError, "single pre-staged inputs"):
                _reserve_full_run_output(fresh)

            staged = root / "staged"
            inputs = staged / "inputs"
            inputs.mkdir(parents=True)
            _reserve_full_run_output(staged)
            self.assertEqual(list(staged.iterdir()), [inputs])
            (staged / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "single pre-staged inputs"):
                _reserve_full_run_output(staged)

    def test_cli_calls_the_overwrite_guard_before_running_a_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = write_nonscientific_fixture(root / "fixture.npz")
            output = root / "run"
            marker = output / "risk" / "keep.txt"
            marker.parent.mkdir(parents=True)
            marker.write_text("existing evidence\n", encoding="utf-8")
            status = formal_cli_main(
                [
                    "run-risk",
                    "--config",
                    str(SMOKE_CONFIG),
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(status, 2)
            self.assertEqual(marker.read_text(encoding="utf-8"), "existing evidence\n")
            self.assertFalse(
                (output.parent / f".{output.name}.csi-pairs-operation.lock").exists()
            )

    def test_make_fixture_refuses_to_overwrite_an_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "fixture.npz"
            target.write_bytes(b"existing fixture bytes")
            status = formal_cli_main(["make-fixture", "--output", str(target)])
            self.assertEqual(status, 2)
            self.assertEqual(target.read_bytes(), b"existing fixture bytes")

    def test_make_fixture_normalizes_npz_suffix_before_exclusive_create(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "fixture.npz"
            existing.write_bytes(b"existing fixture bytes")
            status = formal_cli_main(
                ["make-fixture", "--output", str(root / "fixture")]
            )
            self.assertEqual(status, 2)
            self.assertEqual(existing.read_bytes(), b"existing fixture bytes")

            created = write_nonscientific_fixture(root / "fresh-fixture")
            self.assertEqual(created, root / "fresh-fixture.npz")
            self.assertTrue(created.is_file())
            self.assertFalse((root / "fresh-fixture").exists())

    def test_concurrent_fixture_writers_cannot_share_one_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "fixture.npz"

            def write(seed):
                try:
                    return write_nonscientific_fixture(target, seed=seed)
                except FileExistsError:
                    return None

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(write, (101, 202)))
            self.assertEqual(sum(result == target for result in results), 1)
            self.assertEqual(sum(result is None for result in results), 1)
            FormalDataset.load(target)

    def test_dry_run_wrapper_accepts_only_authenticated_fixture_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dry-run"
            environment = os.environ.copy()
            environment.update(
                {
                    "CSI_PAIRS_PYTHON": sys.executable,
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            completed = subprocess.run(
                [
                    "bash",
                    str(ROOT / "formal_v2" / "scripts" / "run_formal_v2_dry_run.sh"),
                    str(output),
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                # The wrapper performs the complete fixture qualification chain.
                timeout=300,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            summary = json.loads(completed.stdout.splitlines()[-1])
            self.assertEqual(
                summary,
                {
                    "dry_run_status": "EXPECTED_FAIL_CLOSED",
                    "scientific_use": "FORBIDDEN",
                    "status": "PASS",
                },
            )
            gate = json.loads(
                (output / "qualification" / "gate.json").read_text(encoding="utf-8")
            )
            self.assertEqual(gate["status"], "DRY_RUN_FAIL_NOT_EVIDENCE")
            self.assertFalse(gate["passed"])
            self.assertTrue(gate["fixture"])
            self.assertEqual(gate["scientific_use"], "FORBIDDEN")


class StrictJsonTests(unittest.TestCase):
    def test_duplicate_key_rejected(self):
        with self.assertRaisesRegex(StrictJsonError, "duplicate object key"):
            parse_strict_json('{"x":1,"x":2}')

    def test_nan_and_infinity_rejected(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value), self.assertRaises(StrictJsonError):
                parse_strict_json('{"x":' + value + "}")


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = write_nonscientific_fixture(self.root / "fixture.npz")
        self.dataset = FormalDataset.load(self.fixture)

    def tearDown(self):
        self.temporary.cleanup()

    def test_fixture_is_permanently_forbidden(self):
        report = self.dataset.contract_report()
        self.assertTrue(report["fixture"])
        self.assertEqual(report["scientific_use"], "FORBIDDEN")

    def test_exact_seven_source_roles_are_present(self):
        self.assertEqual(len(SOURCE_ROLES), 7)
        self.assertEqual(set(self.dataset.scene_roles).intersection(SOURCE_ROLES), set(SOURCE_ROLES))

    def test_maps_radio_and_coordinates_are_typed(self):
        self.assertEqual(self.dataset.maps.ndim, 5)
        self.assertTrue({"occupancy", "height", "material"}.issubset(self.dataset.map_channel_names))
        self.assertGreater(self.dataset.radio_config.shape[1], 0)
        self.assertEqual(self.dataset.metadata["representation"]["coordinate_system"], "bs_centered_right_handed_meters")

    def test_hypercube_edges_are_bidirectional_hamming_one(self):
        edges = list(self.dataset.directed_edges(0))
        pairs = {(edge.source_world, edge.target_world) for edge in edges}
        for edge in edges:
            self.assertIn((edge.target_world, edge.source_world), pairs)
            self.assertEqual(
                int(np.sum(self.dataset.world_bits[edge.source_world] != self.dataset.world_bits[edge.target_world])),
                1,
            )

    def test_qualification_blocking_allowlist_excludes_all_other_roles(self):
        blocking = qualification_blocking_scenes(self.dataset)
        roles = set(self.dataset.scene_roles[np.concatenate(tuple(blocking.values()))])
        self.assertEqual(roles, {"source_encoder_train", "source_method_selection"})
        self.assertNotIn("target", roles)
        self.assertNotIn("external_validation", roles)

    def test_city_level_support_pool_spans_banks_without_multiplying_k(self):
        candidates = city_support_candidates(self.dataset, "target-a")
        self.assertGreaterEqual(len({scene for scene, _ in candidates}), 2)
        selected = candidates[:2]
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({self.dataset.position_ids[scene, position] for scene, position in selected}), 2)

    def test_support_sibling_ids_are_removed_from_every_query_bank(self):
        scene = int(self.dataset.indices_for_role("target")[0])
        query = int(np.flatnonzero(self.dataset.position_roles[scene] == "query")[0])
        support_ids = {str(self.dataset.position_ids[scene, query])}
        remaining = eligible_query_indices(self.dataset, scene, support_ids)
        self.assertNotIn(query, remaining.tolist())

    def test_support_sibling_coordinates_are_removed_even_if_id_is_relabelled(self):
        target_scenes = self.dataset.indices_for_role("target")
        first = int(target_scenes[0])
        second = int(
            next(
                scene
                for scene in target_scenes
                if scene != first
                and self.dataset.city_ids[scene] == self.dataset.city_ids[first]
            )
        )
        support = int(
            np.flatnonzero(self.dataset.position_roles[first] == "support_pool")[0]
        )
        query = int(np.flatnonzero(self.dataset.position_roles[second] == "query")[0])
        support_id = str(self.dataset.position_ids[first, support])
        self.dataset.positions[second, query] = self.dataset.positions[first, support]
        self.assertNotEqual(
            str(self.dataset.position_ids[second, query]),
            support_id,
        )
        remaining = eligible_query_indices(self.dataset, second, {support_id})
        self.assertNotIn(query, remaining.tolist())

    def test_target_metric_iterator_excludes_support_pool(self):
        for scene_value in self.dataset.indices_for_role("target"):
            scene = int(scene_value)
            selected = _eligible_evaluation_positions(self.dataset, scene)
            self.assertTrue(np.all(self.dataset.position_roles[scene, selected] == "query"))
            support = set(
                np.flatnonzero(self.dataset.position_roles[scene] == "support_pool").tolist()
            )
            self.assertTrue(set(selected.tolist()).isdisjoint(support))

    def test_target_position_identity_cannot_cross_support_and_query_banks(self):
        arrays = _archive_arrays(self.fixture)
        target_scenes = np.flatnonzero(arrays["scene_roles"] == "target")
        first = int(target_scenes[0])
        same_city = target_scenes[
            arrays["city_ids"][target_scenes] == arrays["city_ids"][first]
        ]
        first, second = map(int, same_city[:2])
        position_id = str(arrays["position_ids"][first, 0])
        position = arrays["position_ids"].shape[1] - 1
        arrays["position_ids"] = arrays["position_ids"].astype("<U64")
        arrays["position_ids"][second, position] = position_id
        arrays["positions"] = arrays["positions"].copy()
        arrays["positions"][second, position] = arrays["positions"][first, 0]
        arrays["position_roles"] = arrays["position_roles"].astype("<U32")
        current = str(arrays["position_roles"][first, 0])
        arrays["position_roles"][second, position] = (
            "query" if current == "support_pool" else "support_pool"
        )
        malformed = self.root / "cross-bank-support-query-leak.npz"
        np.savez_compressed(malformed, **arrays)
        with self.assertRaisesRegex(FormalDatasetError, "cross support_pool/query"):
            FormalDataset.load(malformed)

    def test_target_physical_position_cannot_be_relabelled_across_banks(self):
        arrays = _archive_arrays(self.fixture)
        target_scenes = np.flatnonzero(arrays["scene_roles"] == "target")
        first = int(target_scenes[0])
        second = int(
            next(
                scene
                for scene in target_scenes
                if scene != first and arrays["city_ids"][scene] == arrays["city_ids"][first]
            )
        )
        support = int(np.flatnonzero(arrays["position_roles"][first] == "support_pool")[0])
        query = int(np.flatnonzero(arrays["position_roles"][second] == "query")[0])
        self.assertNotEqual(
            str(arrays["position_ids"][first, support]),
            str(arrays["position_ids"][second, query]),
        )
        arrays["positions"] = arrays["positions"].copy()
        arrays["positions"][second, query] = arrays["positions"][first, support] + np.asarray(
            (0.75e-9, -0.75e-9)
        )
        malformed = self.root / "same-coordinate-different-id.npz"
        np.savez_compressed(malformed, **arrays)
        with self.assertRaisesRegex(
            FormalDatasetError,
            "same city-level BS-centered coordinate.*one position_id",
        ):
            FormalDataset.load(malformed)

    def test_target_support_capacity_uses_unique_physical_positions(self):
        smoke = load_formal_config(SMOKE_CONFIG)
        required = max(smoke["localization"]["label_budgets"])
        self.dataset.validate_target_support_capacity(required)
        formal = load_formal_config(ROOT / "formal_v2" / "configs" / "formal_v2.json")
        formal_required = max(formal["localization"]["label_budgets"])
        self.assertEqual(formal_required, 128)
        with self.assertRaisesRegex(FormalDatasetError, "fewer than configured max k"):
            self.dataset.validate_target_support_capacity(formal_required)

    def test_support_spatial_index_matches_frozen_brute_force_selection(self):
        for city in sorted(
            set(
                str(value)
                for value in self.dataset.city_ids[
                    self.dataset.scene_roles == "target"
                ]
            )
        ):
            selected = []
            identifiers = set()
            coordinates = []
            for scene_value in self.dataset.indices_for_role("target"):
                scene = int(scene_value)
                if str(self.dataset.city_ids[scene]) != city:
                    continue
                for position_value in np.flatnonzero(
                    self.dataset.position_roles[scene] == "support_pool"
                ):
                    position = int(position_value)
                    identifier = str(self.dataset.position_ids[scene, position])
                    coordinate = self.dataset.positions[scene, position]
                    if identifier in identifiers or any(
                        np.allclose(
                            coordinate,
                            existing,
                            rtol=0.0,
                            atol=1e-9,
                        )
                        for existing in coordinates
                    ):
                        continue
                    identifiers.add(identifier)
                    coordinates.append(coordinate)
                    selected.append((scene, position))
            self.assertEqual(
                self.dataset.unique_target_support_positions(city), selected
            )

    def test_target_city_cluster_minimum_uses_canonical_foundations(self):
        self.dataset.validate(
            minimum_independent_base_map_clusters_per_target_city=2
        )
        with self.assertRaisesRegex(FormalDatasetError, "canonical base-map clusters"):
            self.dataset.validate(
                minimum_independent_base_map_clusters_per_target_city=3
            )

    def test_every_source_role_itself_must_cover_two_source_cities(self):
        for role in SOURCE_ROLES:
            with self.subTest(role=role):
                arrays = _archive_arrays(self.fixture)
                role_scenes = np.flatnonzero(arrays["scene_roles"] == role)
                self.assertGreaterEqual(role_scenes.size, 2)
                arrays["city_ids"] = arrays["city_ids"].astype("<U64")
                arrays["city_ids"][role_scenes] = arrays["city_ids"][
                    role_scenes[0]
                ]
                malformed = self.root / f"single-city-{role}.npz"
                np.savez_compressed(malformed, **arrays)
                with self.assertRaisesRegex(FormalDatasetError, role):
                    FormalDataset.load(malformed)

    def test_embedded_metadata_duplicate_is_rejected(self):
        arrays = _archive_arrays(self.fixture)
        raw = str(arrays["metadata_json"].item())
        needle = '"dataset_id":"NONSCIENTIFIC-CODE-FIXTURE"'
        arrays["metadata_json"] = np.asarray(
            raw.replace(needle, '"dataset_id":"first","dataset_id":"NONSCIENTIFIC-CODE-FIXTURE"')
        )
        malformed = self.root / "duplicate.npz"
        np.savez_compressed(malformed, **arrays)
        with self.assertRaisesRegex(FormalDatasetError, "strict JSON"):
            FormalDataset.load(malformed)

    def test_unexpected_npz_array_is_rejected(self):
        arrays = _archive_arrays(self.fixture)
        arrays["undeclared_side_channel"] = np.asarray([1])
        malformed = self.root / "unexpected.npz"
        np.savez_compressed(malformed, **arrays)
        with self.assertRaisesRegex(FormalDatasetError, "array fields must be exact"):
            FormalDataset.load(malformed)

    def test_copied_sibling_noise_is_rejected(self):
        arrays = _archive_arrays(self.fixture)
        clean = arrays["csi_clean"]
        residual = arrays["csi_repeat"] - clean[:, :, :, None, :]
        residual[:, 1] = residual[:, 0]
        arrays["csi_repeat"] = clean[:, :, :, None, :] + residual
        malformed = self.root / "copied_noise.npz"
        np.savez_compressed(malformed, **arrays)
        with self.assertRaisesRegex(FormalDatasetError, "noise realization"):
            FormalDataset.load(malformed)

    def test_canonical_digest_mismatch_is_rejected(self):
        arrays = _archive_arrays(self.fixture)
        arrays["maps"] = arrays["maps"].copy()
        arrays["maps"][0, 0, 0, 1, 1] += 1
        malformed = self.root / "map_digest.npz"
        np.savez_compressed(malformed, **arrays)
        with self.assertRaisesRegex(FormalDatasetError, "canonical map digest"):
            FormalDataset.load(malformed)

    def test_base_map_cluster_cannot_cross_roles(self):
        arrays = _archive_arrays(self.fixture)
        arrays["base_map_cluster_ids"] = arrays["base_map_cluster_ids"].copy()
        arrays["base_map_cluster_ids"][2] = arrays["base_map_cluster_ids"][0]
        malformed = self.root / "cluster_leak.npz"
        np.savez_compressed(malformed, **arrays)
        with self.assertRaisesRegex(FormalDatasetError, "cross scene roles"):
            FormalDataset.load(malformed)

    def test_common_free_space_failure_is_rejected(self):
        arrays = _archive_arrays(self.fixture)
        arrays["free_space"] = arrays["free_space"].copy()
        arrays["free_space"][0, 1, 2] = False
        malformed = self.root / "occupied.npz"
        np.savez_compressed(malformed, **arrays)
        with self.assertRaisesRegex(FormalDatasetError, "free in every sibling"):
            FormalDataset.load(malformed)

    def test_nonfixture_pathless_clean_unit_is_rejected(self):
        arrays = _archive_arrays(self.fixture)
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["fixture"] = False
        metadata["scientific_use"] = "CANDIDATE"
        arrays["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        )
        arrays["path_ids"] = arrays["path_ids"].copy()
        arrays["path_power"] = arrays["path_power"].copy()
        arrays["path_surface_ids"] = arrays["path_surface_ids"].copy()
        arrays["path_ids"][0, 0, 0] = -1
        arrays["path_power"][0, 0, 0] = 0.0
        arrays["path_surface_ids"][0, 0, 0] = -1
        malformed = self.root / "nonfixture-pathless.npz"
        np.savez_compressed(malformed, **arrays)
        with self.assertRaisesRegex(FormalDatasetError, "registered RT path"):
            FormalDataset.load(malformed)

    def test_nonfixture_all_zero_clean_unit_is_rejected(self):
        arrays = _archive_arrays(self.fixture)
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["fixture"] = False
        metadata["scientific_use"] = "CANDIDATE"
        arrays["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        )
        arrays["csi_clean"] = arrays["csi_clean"].copy()
        arrays["csi_clean"][0, 0, 0] = 0.0
        malformed = self.root / "nonfixture-zero-csi.npz"
        np.savez_compressed(malformed, **arrays)
        with self.assertRaisesRegex(FormalDatasetError, "all-zero CSI"):
            FormalDataset.load(malformed)


class ProtocolTests(unittest.TestCase):
    def test_patch_roundtrip(self):
        spec = PatchSpec(antennas=2, subcarriers=4, patch_complex_size=1)
        csi = np.arange(32, dtype=float).reshape(2, 16)
        np.testing.assert_array_equal(unpatchify_csi(patchify_csi(csi, spec), spec), csi)

    def test_patchify_uses_two_dimensional_antenna_subcarrier_tiles(self):
        spec = PatchSpec(
            antennas=2,
            subcarriers=4,
            patch_complex_size=4,
            patch_antenna_size=2,
            patch_subcarrier_size=2,
        )
        real = np.arange(8, dtype=float)
        csi = np.concatenate((real, 100 + real))
        patches = patchify_csi(csi, spec)
        np.testing.assert_array_equal(patches[0, 0::2], [0, 1, 4, 5])
        np.testing.assert_array_equal(patches[1, 0::2], [2, 3, 6, 7])
        np.testing.assert_array_equal(unpatchify_csi(patches, spec), csi)

    def test_delay_angle_power_is_invariant_to_shared_phase_rotation(self):
        spec = PatchSpec(2, 4, 1)
        values = np.arange(1, 9) + 1j * np.arange(9, 17)
        first = np.concatenate((values.real, values.imag))
        rotated = values * np.exp(1j * 0.73)
        second = np.concatenate((rotated.real, rotated.imag))
        np.testing.assert_allclose(
            delay_angle_power(first, spec),
            delay_angle_power(second, spec),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_mask_bank_has_three_modes_query_hidden_and_full_coverage(self):
        spec = PatchSpec(2, 4, 1)
        bank = frozen_mask_query_bank(spec, 7)
        self.assertEqual({entry.mode for entry in bank}, {"random_75", "antenna_block_50", "subcarrier_block_50"})
        self.assertTrue(all(entry.mask[entry.query] for entry in bank))
        for mode in {entry.mode for entry in bank}:
            self.assertEqual({entry.query for entry in bank if entry.mode == mode}, set(range(spec.patch_count)))

    def test_typed_action_has_inverse_direction_and_categorical_from_to(self):
        source = np.zeros((3, 4, 4))
        target = source.copy()
        source[0, 1, 1] = 1
        source[1, 1, 1] = 2
        source[2, 1, 1] = 1
        target[1, 1, 1] = 5
        target[2, 1, 1] = 3
        names = np.asarray(["occupancy", "height", "material"])
        forward = typed_signed_edit(source, target, names, 4)
        reverse = typed_signed_edit(target, source, names, 4)
        self.assertGreater(forward[2, 1, 1], 0)
        self.assertGreater(reverse[3, 1, 1], 0)
        self.assertEqual(forward[4 + 1, 1, 1], 1)
        self.assertEqual(forward[4 + 4 + 3, 1, 1], 1)

    def test_model_api_has_no_target_route_or_position_input_and_dual_outputs(self):
        signature = inspect.signature(CSIPairsFormalModel.state)
        self.assertEqual(list(signature.parameters), ["self", "visible_patches", "maps", "radio", "masks"])
        model = CSIPairsFormalModel(
            patch_count=8,
            patch_dim=2,
            map_channels=3,
            action_channels=12,
            radio_dim=4,
            latent_dim=12,
            state_dim=12,
            map_dim=8,
            hidden_dim=24,
            attention_heads=3,
        )
        state = model.state(
            torch.zeros(2, 8, 2), torch.zeros(2, 3, 8, 8), torch.zeros(2, 4), torch.ones(2, 8, dtype=torch.bool)
        )
        latent, physical = model.predict(state, torch.zeros(2, 12, 8, 8), torch.tensor([0, 7]))
        self.assertEqual(tuple(latent.shape), (2, 12))
        self.assertEqual(tuple(physical.shape), (2, 2))

    def test_cached_context_and_action_paths_are_bit_exact(self):
        model = CSIPairsFormalModel(
            patch_count=8,
            patch_dim=2,
            map_channels=3,
            action_channels=12,
            radio_dim=4,
            latent_dim=12,
            state_dim=12,
            map_dim=8,
            hidden_dim=24,
            attention_heads=3,
        ).eval()
        patches = torch.randn(2, 8, 2)
        maps = torch.randn(2, 3, 8, 8)
        radio = torch.randn(2, 4)
        masks = torch.zeros(2, 8, dtype=torch.bool)
        actions = torch.randn(2, 12, 8, 8)
        query = torch.tensor([1, 6])
        with torch.no_grad():
            direct_state = model.state(patches, maps, radio, masks)
            cached_state = model.state_from_context(
                patches, masks, model.encode_context(maps, radio)
            )
            direct_outputs = model.predict(direct_state, actions, query)
            cached_outputs = model.predict_from_action(
                cached_state, model.encode_action(actions), query
            )
        torch.testing.assert_close(direct_state, cached_state, rtol=0.0, atol=0.0)
        for direct, cached in zip(direct_outputs, cached_outputs, strict=True):
            torch.testing.assert_close(direct, cached, rtol=0.0, atol=0.0)

    def test_full_and_prepooled_spatial_inputs_are_model_equivalent(self):
        torch.manual_seed(20270817)
        model = CSIPairsFormalModel(
            patch_count=8,
            patch_dim=2,
            map_channels=3,
            action_channels=12,
            radio_dim=4,
            latent_dim=12,
            state_dim=12,
            map_dim=8,
            hidden_dim=24,
            attention_heads=3,
        ).eval()
        patches = torch.randn(2, 8, 2)
        maps = torch.randn(2, 3, 32, 48)
        radio = torch.randn(2, 4)
        masks = torch.zeros(2, 8, dtype=torch.bool)
        actions = torch.randn(2, 12, 32, 48)
        query = torch.tensor([1, 6])
        pooled_maps = model.map_encoder.input_pool(maps)
        pooled_actions = model.action_input_pool(actions)
        with torch.no_grad():
            direct_state = model.state(patches, maps, radio, masks)
            pooled_state = model.state(patches, pooled_maps, radio, masks)
            direct_outputs = model.predict(direct_state, actions, query)
            pooled_outputs = model.predict(pooled_state, pooled_actions, query)
        torch.testing.assert_close(direct_state, pooled_state, rtol=0.0, atol=0.0)
        for direct, pooled in zip(direct_outputs, pooled_outputs, strict=True):
            torch.testing.assert_close(direct, pooled, rtol=0.0, atol=0.0)

    def test_zero_action_encoding_and_response_residual_are_bit_exact_zero(self):
        model = CSIPairsFormalModel(
            patch_count=8,
            patch_dim=2,
            map_channels=3,
            action_channels=12,
            radio_dim=4,
            latent_dim=12,
            state_dim=12,
            map_dim=8,
            hidden_dim=24,
            attention_heads=3,
        ).eval()
        state = torch.randn(3, 8, 12)
        query = torch.tensor([0, 3, 7])
        encoded = model.encode_action(torch.zeros(3, 12, 8, 8))
        self.assertEqual(tuple(encoded.shape), (3, 17, 8))
        residual = model.predict_action_residual(state, encoded, query)
        self.assertEqual(int(torch.count_nonzero(encoded)), 0)
        self.assertTrue(all(int(torch.count_nonzero(value)) == 0 for value in residual))
        identity = model.predict_identity(state, query)
        prediction = model.predict_from_action(state, encoded, query)
        for direct, zero_action in zip(identity, prediction, strict=True):
            torch.testing.assert_close(direct, zero_action, rtol=0.0, atol=0.0)

    def test_endpoint_gradient_is_isolated_from_response_residual(self):
        model = CSIPairsFormalModel(
            patch_count=4,
            patch_dim=4,
            map_channels=3,
            action_channels=12,
            radio_dim=4,
            latent_dim=8,
            state_dim=8,
            map_dim=8,
            hidden_dim=16,
            attention_heads=2,
        )
        state = torch.randn(2, 4, 8, requires_grad=True)
        latent, physical = model.predict_identity(state, torch.tensor([1, 3]))
        (latent.square().mean() + physical.square().mean()).backward()
        response_prefixes = (
            "action_encoder",
            "action_pool",
            "response_",
        )
        response_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith(response_prefixes)
        ]
        self.assertTrue(response_parameters)
        self.assertTrue(
            all(
                parameter.grad is None or int(torch.count_nonzero(parameter.grad)) == 0
                for parameter in response_parameters
            )
        )
        self.assertGreater(
            sum(
                int(torch.count_nonzero(parameter.grad))
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
                and name.startswith(("predictor", "latent_head", "physical_head"))
            ),
            0,
        )

    def test_response_residual_is_sensitive_to_wrong_action_coordinates(self):
        torch.manual_seed(20270815)
        model = CSIPairsFormalModel(
            patch_count=4,
            patch_dim=4,
            map_channels=3,
            action_channels=12,
            radio_dim=4,
            latent_dim=8,
            state_dim=8,
            map_dim=8,
            hidden_dim=16,
            attention_heads=2,
        ).eval()
        state = torch.randn(2, 4, 8)
        actions = torch.zeros(2, 12, 16, 16)
        actions[0, 1, 2:5, 3:6] = 1.0
        actions[1, 1, 10:13, 11:14] = 1.0
        encoded = model.encode_action(actions)
        latent, physical = model.predict_action_residual(
            state[0:1].expand(2, -1, -1), encoded, torch.tensor([2, 2])
        )
        self.assertFalse(torch.equal(encoded[0], encoded[1]))
        self.assertFalse(torch.equal(latent[0], latent[1]))
        self.assertFalse(torch.equal(physical[0], physical[1]))

    def test_action_pool_preserves_the_frozen_spatial_token_order(self):
        model = CSIPairsFormalModel(
            patch_count=4,
            patch_dim=4,
            map_channels=3,
            action_channels=12,
            radio_dim=4,
            latent_dim=8,
            state_dim=8,
            map_dim=8,
            hidden_dim=16,
            attention_heads=2,
        )
        self.assertEqual(model.action_token_count, 16)
        self.assertEqual(
            model.action_pool[0].in_features,
            16 * model.map_dim + 12 * model.action_moments.feature_count,
        )
        encoded = model.encode_action(torch.zeros(2, 12, 16, 16))
        self.assertEqual(tuple(encoded.shape), (2, 17, model.map_dim))

    def test_response_conditioning_depends_on_state_for_the_same_action(self):
        torch.manual_seed(20270818)
        model = CSIPairsFormalModel(
            patch_count=4,
            patch_dim=4,
            map_channels=3,
            action_channels=12,
            radio_dim=4,
            latent_dim=8,
            state_dim=8,
            map_dim=8,
            hidden_dim=16,
            attention_heads=2,
        ).eval()
        action = torch.zeros(2, 12, 16, 16)
        action[:, 1, 3:7, 5:9] = 1.0
        encoded = model.encode_action(action)
        state = torch.stack((torch.zeros(4, 8), torch.ones(4, 8)))
        latent, physical = model.predict_action_residual(
            state, encoded, torch.tensor([2, 2])
        )
        self.assertFalse(torch.equal(latent[0], latent[1]))
        self.assertFalse(torch.equal(physical[0], physical[1]))

    def test_response_conditioning_reads_nonquery_source_state(self):
        torch.manual_seed(20270820)
        model = CSIPairsFormalModel(
            patch_count=4,
            patch_dim=4,
            map_channels=3,
            action_channels=12,
            radio_dim=4,
            latent_dim=8,
            state_dim=8,
            map_dim=8,
            hidden_dim=16,
            attention_heads=2,
        ).eval()
        action = torch.zeros(2, 12, 16, 16)
        action[:, 1, 3:7, 5:9] = 1.0
        encoded = model.encode_action(action)
        state = torch.zeros(2, 4, 8)
        state[1, 0] = 3.0
        latent, physical = model.predict_action_residual(
            state, encoded, torch.tensor([2, 2])
        )
        self.assertFalse(torch.equal(latent[0], latent[1]))
        self.assertFalse(torch.equal(physical[0], physical[1]))

    def test_action_response_batch_samples_are_isolated(self):
        torch.manual_seed(20270819)
        model = CSIPairsFormalModel(
            patch_count=4,
            patch_dim=4,
            map_channels=3,
            action_channels=12,
            radio_dim=4,
            latent_dim=8,
            state_dim=8,
            map_dim=8,
            hidden_dim=16,
            attention_heads=2,
        ).eval()
        state = torch.randn(2, 4, 8)
        action = torch.randn(2, 12, 16, 16)
        query = torch.tensor([1, 3])
        with torch.no_grad():
            original = model.predict(state, action, query)
            action[1] = torch.randn_like(action[1]) * 100.0
            state[1] = torch.randn_like(state[1]) * 100.0
            repeated = model.predict(state, action, query)
        for before, after in zip(original, repeated, strict=True):
            torch.testing.assert_close(before[0], after[0], rtol=0.0, atol=0.0)

    def test_response_action_has_a_zero_preserving_direct_gradient_path(self):
        torch.manual_seed(20270816)
        model = CSIPairsFormalModel(
            patch_count=4,
            patch_dim=4,
            map_channels=3,
            action_channels=12,
            radio_dim=4,
            latent_dim=8,
            state_dim=8,
            map_dim=8,
            hidden_dim=16,
            attention_heads=2,
        )
        for parameter in model.response_state.parameters():
            parameter.data.zero_()
        for parameter in model.response_predictor.parameters():
            parameter.data.zero_()
        state = torch.zeros(1, 4, 8)
        query = torch.tensor([2])
        action = torch.zeros(1, 12, 16, 16)
        action[:, 1, 3:7, 5:9] = 1.0
        encoded = model.encode_action(action)
        latent, physical = model.predict_action_residual(state, encoded, query)
        loss = latent.square().mean() + physical.square().mean()
        loss.backward()

        self.assertGreater(int(torch.count_nonzero(latent)), 0)
        self.assertGreater(int(torch.count_nonzero(physical)), 0)
        self.assertGreater(
            sum(
                int(torch.count_nonzero(parameter.grad))
                for parameter in model.response_action.parameters()
                if parameter.grad is not None
            ),
            0,
        )

    def test_model_has_no_batchnorm_and_batch_samples_are_isolated(self):
        model = CSIPairsFormalModel(
            patch_count=4,
            patch_dim=4,
            map_channels=3,
            action_channels=12,
            radio_dim=4,
            latent_dim=8,
            state_dim=8,
            map_dim=8,
            hidden_dim=16,
            attention_heads=2,
        ).eval()
        self.assertFalse(
            any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules())
        )
        patches = torch.randn(2, 4, 4)
        maps = torch.randn(2, 3, 8, 8)
        radio = torch.randn(2, 4)
        masks = torch.zeros(2, 4, dtype=torch.bool)
        with torch.no_grad():
            first = model.state(patches, maps, radio, masks)[0].clone()
            patches[1] = torch.randn_like(patches[1]) * 100
            maps[1] = torch.randn_like(maps[1]) * 100
            radio[1] = torch.randn_like(radio[1]) * 100
            repeated = model.state(patches, maps, radio, masks)[0]
        torch.testing.assert_close(first, repeated, rtol=0.0, atol=0.0)

    def test_action_moments_preserve_absolute_edit_coordinates(self):
        actions = torch.zeros(2, 3, 16, 16)
        actions[0, 1, 3:5, 4:6] = 1.0
        actions[1, 1, 10:12, 11:13] = 1.0
        moments = _CoordinateActionMoments()(actions)
        self.assertFalse(torch.equal(moments[0], moments[1]))
        channel = moments.reshape(2, 3, -1)[:, 1]
        self.assertLess(float(channel[0, 4]), float(channel[1, 4]))
        self.assertLess(float(channel[0, 5]), float(channel[1, 5]))

    def test_f_inherits_complete_teacher_encoder_and_uses_context_pose(self):
        teacher = CSIMaskedTeacher(2, 2, 2, 8, 2, 2, 1)
        model = CSIPairsFormalModel(
            patch_count=4,
            patch_rows=2,
            patch_columns=2,
            patch_dim=2,
            map_channels=3,
            action_channels=12,
            radio_dim=11,
            latent_dim=8,
            state_dim=8,
            map_dim=8,
            hidden_dim=16,
            attention_heads=2,
            csi_encoder_layers=2,
        ).eval()
        model.initialize_csi_from_teacher(teacher)
        self.assertNotIn("csi_fixed_position", dict(model.named_parameters()))
        self.assertIn("csi_fixed_position", dict(model.named_buffers()))
        self.assertIn("csi_fixed_position", model.state_dict())
        self.assertFalse(model.csi_fixed_position.requires_grad)
        torch.testing.assert_close(
            teacher.fixed_position,
            model.csi_fixed_position,
            rtol=0.0,
            atol=0.0,
        )
        for name, value in teacher.encoder.state_dict().items():
            torch.testing.assert_close(value, model.csi_encoder.state_dict()[name])
        patches = torch.zeros(1, 4, 2)
        maps = torch.zeros(1, 3, 8, 8)
        masks = torch.zeros(1, 4, dtype=torch.bool)
        first = model.state(patches, maps, torch.zeros(1, 11), masks)
        changed_pose = torch.zeros(1, 11)
        changed_pose[0, -1] = 1.0
        second = model.state(patches, maps, changed_pose, masks)
        self.assertFalse(torch.equal(first, second))

    def test_route_thresholds_are_direction_independent(self):
        forward = route_code(0.2, 0.05, 0.1)
        reverse = route_code(abs(-0.2), 0.05, 0.1)
        self.assertEqual(forward, reverse)

    def test_primary_route_is_physical_only_and_teacher_is_a_separate_stratum(self):
        self.assertEqual(route_code(0.2, 0.05, 0.1), 2)
        self.assertEqual(route_code(0.01, 0.05, 0.1), 0)
        self.assertEqual(route_code(0.075, 0.05, 0.1), 1)
        self.assertEqual(teacher_sensitivity_code(0.2, 0.05, 0.1), 2)
        self.assertEqual(teacher_sensitivity_code(0.01, 0.05, 0.1), 0)

    def test_paired_score_difference_requires_one_match_and_one_alternative(self):
        difference = _paired_score_differences(
            np.asarray([0.8, 0.2]), np.asarray([1, 0]), np.asarray(["pair", "pair"])
        )
        np.testing.assert_allclose(difference, [0.6])
        with self.assertRaisesRegex(RuntimeError, "one matched"):
            _paired_score_differences(
                np.asarray([0.8, 0.7]), np.asarray([1, 1]), np.asarray(["pair", "pair"])
            )

    def test_q_comp_random_order_and_candidate_swap_complement(self):
        labels = randomized_candidate_labels(100, 1234)
        np.testing.assert_array_equal(labels, randomized_candidate_labels(100, 1234))
        self.assertEqual(set(labels.tolist()), {0, 1})
        calibration = TemperatureCalibration(2.5)
        probability = calibration.predict(np.asarray([-2.0, 0.0, 3.0]))
        swapped = calibration.predict(np.asarray([2.0, -0.0, -3.0]))
        np.testing.assert_allclose(probability + swapped, np.ones(3), atol=1e-12)

    def test_empty_conditional_mean_fails(self):
        with self.assertRaisesRegex(RuntimeError, "empty conditional mean"):
            required_mean(torch.empty(0), "active")


class StatisticsTests(unittest.TestCase):
    def test_exact_j_equal_weights_city_k_bank_seed_draw(self):
        rows = _factorial_rows()
        report = exact_factorial_utilities(rows, [0, 8])
        self.assertAlmostEqual(report["utilities"]["endpoint"], 0.0)
        self.assertAlmostEqual(report["interaction"], 0.25)

    def test_hierarchical_bootstrap_declares_all_layers(self):
        report = hierarchical_factorial_interval(_factorial_rows(), [0, 8], 20, 3)
        self.assertEqual(
            report["resampling_layers"],
            ["base_map_cluster_within_fixed_city", "training_seed", "k_positive_label_draw"],
        )

    def test_holm_adjustment_is_monotone_in_sorted_order(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertTrue(all(0 <= value <= 1 for value in adjusted))
        self.assertGreaterEqual(adjusted[0], 0.01)

    def test_holm_adjustment_rejects_invalid_families(self):
        for values in ([], [float("nan")], [float("inf")], [-0.01], [1.01]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    holm_adjust(values)

    def test_exact_sign_flip_does_not_use_monte_carlo_plus_one_correction(self):
        result = paired_sign_flip_test(
            np.asarray(["cluster-a", "cluster-b"]),
            np.asarray([1.0, 2.0]),
            np.zeros(2),
            seed=7,
        )
        self.assertEqual(result["method"], "exact")
        self.assertEqual(result["draws"], 4)
        self.assertEqual(result["p_value_two_sided"], 0.5)

    def test_monte_carlo_sign_flip_uses_plus_one_correction(self):
        result = paired_sign_flip_test(
            np.asarray([f"cluster-{index}" for index in range(17)]),
            np.ones(17),
            np.zeros(17),
            seed=0,
            maximum_draws=3,
        )
        self.assertEqual(result["method"], "monte_carlo")
        self.assertEqual(result["draws"], 3)
        self.assertEqual(result["finite_sample_correction"], "plus_one_monte_carlo")
        self.assertEqual(result["p_value_two_sided"], 0.25)

    def test_sign_flip_null_is_centered_on_registered_effect_margin(self):
        clusters = np.asarray([f"cluster-{index}" for index in range(12)])
        effects = np.asarray((0.08, 0.12) * 6)
        zero_null = paired_sign_flip_test(clusters, effects, np.zeros(12), seed=7)
        margin_null = paired_sign_flip_test(
            clusters,
            effects,
            np.zeros(12),
            seed=7,
            null_difference=0.1,
        )
        self.assertLess(zero_null["p_value_two_sided"], 0.05)
        self.assertGreater(margin_null["p_value_two_sided"], 0.05)
        self.assertEqual(margin_null["null_difference"], 0.1)

    def test_scientific_gate_exit_code_is_fail_closed(self):
        self.assertEqual(_result_exit_code({"status": "PASS", "passed": True}), 0)
        self.assertEqual(_result_exit_code({"status": "FAIL", "passed": False}), 1)
        self.assertEqual(_result_exit_code({"status": "BLOCKED", "passed": False}), 1)
        self.assertEqual(
            _result_exit_code(
                {
                    "status": "COMPLETE",
                    "gate_vector": {"G0": "PASS", "G1": "FAIL"},
                }
            ),
            1,
        )

    def test_splitting_identical_bank_rows_inside_one_cluster_does_not_reweight_j(self):
        rows = _factorial_rows()
        original = exact_factorial_utilities(rows, [0, 8])
        duplicated = rows + [
            {**row, "bank_id": row["bank_id"] + "-duplicate"}
            for row in rows
            if row["city_id"] == "a" and row["base_map_cluster_id"] == "a1"
        ]
        repeated = exact_factorial_utilities(duplicated, [0, 8])
        self.assertEqual(original["utilities"], repeated["utilities"])
        self.assertEqual(original["interaction"], repeated["interaction"])


class EvidenceAndPathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = write_nonscientific_fixture(self.root / "fixture.npz")
        self.dataset = FormalDataset.load(self.path)
        self.config = load_formal_config(SMOKE_CONFIG)

    def tearDown(self):
        self.temporary.cleanup()

    def _rt_unit(
        self,
        unit_id,
        scene_id,
        source_asset_id,
        batch_id,
        source_record_id,
        raw_unit_id,
        payload,
    ):
        source_asset = self.root / source_asset_id
        write_json(
            source_asset,
            {
                "schema_version": "csi-pairs-v6-rt-source-asset-v1",
                "records": [
                    {
                        "generation_or_acquisition_batch_id": batch_id,
                        "source_record_id": source_record_id,
                        "raw_unit_id": raw_unit_id,
                        "payload": payload,
                    }
                ],
            },
        )
        return {
            "unit_id": unit_id,
            "scene_id": scene_id,
            "source": {
                "asset_path": source_asset_id,
                "asset_sha256": sha256_file(source_asset),
                "generation_or_acquisition_batch_id": batch_id,
                "source_record_id": source_record_id,
                "raw_unit_id": raw_unit_id,
            },
            "payload": payload,
        }

    def test_gate_and_claim_identifiers_are_fixed(self):
        self.assertEqual(GATE_IDS, tuple(f"G{i}" for i in range(9)))
        self.assertEqual(CLAIM_IDS, tuple(f"C{i}" for i in range(1, 14)))
        self.assertEqual(set(CLAIM_DEPENDENCIES), set(CLAIM_IDS))
        self.assertEqual(set(complete_gate_vector()), set(GATE_IDS))
        self.assertEqual(
            CLAIM_DEPENDENCIES["C8"],
            ("G1_G2", "G3", "G4", "G5"),
        )

    def test_c8_stages_bind_one_qualification_factorial_evaluation_chain(self):
        for name in ("qualification", "factorial", "evaluation", "controls"):
            (self.root / name).mkdir()
        qualification = self.root / "qualification" / "gate.json"
        factorial = self.root / "factorial" / "gate.json"
        evaluation = self.root / "evaluation" / "gate.json"
        controls = self.root / "controls" / "gate.json"
        qualification.write_bytes(b"qualification")
        factorial.write_bytes(b"factorial")
        evaluation.write_bytes(b"evaluation")
        controls.write_bytes(b"controls")
        g5 = {"qualification_gate_sha256": sha256_file(qualification)}
        g3 = {
            **g5,
            "factorial_gate_sha256": sha256_file(factorial),
        }
        g4 = {
            "factorial_gate_sha256": sha256_file(factorial),
            "evaluation_gate_sha256": sha256_file(evaluation),
        }
        _validate_critical_chain_binding(factorial, g5, "G5")
        _validate_critical_chain_binding(evaluation, g3, "G3")
        _validate_critical_chain_binding(controls, g4, "G4")
        factorial.write_bytes(b"different-factorial-run")
        with self.assertRaisesRegex(RuntimeError, "G3 factorial binding"):
            _validate_critical_chain_binding(evaluation, g3, "G3")
        with self.assertRaisesRegex(RuntimeError, "G4 factorial binding"):
            _validate_critical_chain_binding(controls, g4, "G4")

    def test_resource_verification_failure_returns_nonzero(self):
        output = self.root / "resource-failure"
        with patch(
            "formal_v2.formal_resources.verify_waibu_resources",
            return_value={"status": "FAIL", "passed": False},
        ):
            return_code = formal_cli_main(
                [
                    "verify-waibu-resources",
                    "--registry",
                    str(self.root / "registry.json"),
                    "--waibu-root",
                    str(self.root / "waibu"),
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(return_code, 1)

    def test_prepare_stops_before_data_verification_when_resources_fail(self):
        output = self.root / "all-resource-failure"
        argv = [
            "prepare-full-run",
            "--config",
            str(SMOKE_CONFIG),
            "--dataset",
            str(self.path),
            "--output",
            str(output),
            "--allow-nonscientific-fixture",
            "--compute-plan",
            str(self.root / "compute.json"),
            "--verifier-manifest",
            str(self.root / "verifier.json"),
            "--adapter-manifest",
            str(self.root / "adapters.json"),
            "--external-validity-manifest",
            str(self.root / "external-validity.json"),
            "--literature-resource-manifest",
            str(self.root / "literature.json"),
            "--rt-calibration-manifest",
            str(self.root / "rt-calibration.json"),
        ]
        with (
            patch(
                "formal_v2.formal_run_approval.preflight_full_run",
                return_value={"status": "PASS"},
            ),
            patch(
                "formal_v2.formal_resources.verify_waibu_resources",
                return_value={"status": "FAIL", "passed": False},
            ) as resource_verifier,
            patch(
                "formal_v2.formal_data_verification.run_data_verification"
            ) as data_verifier,
        ):
            self.assertEqual(formal_cli_main(argv), 2)
        resource_verifier.assert_called_once()
        data_verifier.assert_not_called()

    def test_claim_semantics_recheck_g0_c2_c11_and_g8_evidence(self):
        g0 = {
            "status": "PASS",
            "passed": True,
            "input_manifest_sha256": "a" * 64,
            "databases": ["Crossref", "OpenAlex", "Semantic Scholar"],
            "query_count": 1,
            "search_receipt_count": 3,
            "search_receipts_verified": True,
            "c13_requires_llm_judge": True,
            "decision": {
                "no_direct_overlap": True,
                "rt_path_ready": True,
                "map_path_ready": True,
                "external_validity_path_ready": True,
                "novelty_scope": "local paired interventions",
            },
        }
        self.assertEqual(_semantic_status("G0", g0), "PASS")
        g0["decision"]["rt_path_ready"] = False
        self.assertEqual(_semantic_status("G0", g0), "FAIL")

        scene = {
            "status": "PASS",
            "passed": True,
            "models_assessed": 1,
            "input_manifest_sha256": "b" * 64,
            "model_assessments": [
                {
                    "passed": True,
                    "base_map_cluster_count": 3,
                    "scene_id_minus_map_ci95_high": 0.1,
                    "scene_id_error_noninferiority_margin_m": 0.25,
                    "map_swap_id_swap_direction_cosine_ci95_low": 0.9,
                    "minimum_swap_direction_cosine": 0.8,
                    "adapter_source_sha256": "c" * 64,
                    "model_checkpoint_sha256": "d" * 64,
                    "training_provenance_sha256": "e" * 64,
                }
            ],
        }
        self.assertEqual(_semantic_status("scene_id_mechanism", scene), "PASS")
        scene["model_assessments"][0]["scene_id_minus_map_ci95_high"] = 0.3
        self.assertEqual(_semantic_status("scene_id_mechanism", scene), "FAIL")

        rt = {
            "status": "PASS",
            "passed": True,
            "fit_validation_independence": {
                "verified": True,
                "rule": "raw_partition_unit_scene_source_and_raw_unit_disjoint_v2",
                "unit_id_overlap": [],
                "scene_id_overlap": [],
                "source_asset_sha256_overlap": [],
                "source_record_identity_overlap": [],
                "raw_unit_identity_overlap": [],
                "canonical_payload_sha256_overlap_count": 0,
            },
            "protocol_sha256": "a" * 64,
            "input_manifest_sha256": "b" * 64,
            "adapter_source_sha256": "c" * 64,
            "fit_dataset_sha256": "d" * 64,
            "validation_inputs_sha256": "e" * 64,
            "validation_reference_sha256": "1" * 64,
            "fitted_parameters_sha256": "f" * 64,
            "simulated_statistics_path": "simulated_statistics.csv",
            "simulated_statistics_sha256": "2" * 64,
            "validated_statistics_path": "validated_statistics.csv",
            "validated_statistics_sha256": "3" * 64,
            "fit_unit_count": 2,
            "fit_scene_count": 2,
            "validation_unit_count": 2,
            "validation_scene_count": 2,
            "aggregation": "mean_absolute_error_per_unit",
            "statistics": {
                name: {"passed": True}
                for name in ("path_loss", "delay_spread", "angular_spread", "visible_path_count")
            },
        }
        self.assertEqual(_semantic_status("rt_calibration", rt), "PASS")
        rt["statistics"]["path_loss"]["passed"] = False
        self.assertEqual(_semantic_status("rt_calibration", rt), "FAIL")

        g8 = {
            "status": "PASS",
            "passed": True,
            "execution_mode": "authenticated_sionna_adapter",
            "engine_family": "sionna",
            "adapter_source_path": "/adapter.py",
            "active_direction_cluster_count": 3,
            "active_direction_agreement_ci95_low": 0.85,
            "minimum_active_direction_agreement": 0.8,
            "adapter_source_sha256": "a" * 64,
            "rt_scene_manifest_path": "rt_scene_manifest.json",
            "rt_scene_manifest_sha256": "b" * 64,
            "external_csi_path": "external_csi.npz",
            "external_csi_sha256": "c" * 64,
            "external_engine_config_path": None,
            "external_engine_config_sha256": None,
            "external_csi_contract": "outer-recomputed-direction-and-effect-from-raw-csi-v1",
            "external_scene_count": 2,
            "external_runtime_provenance_path": "runtime_provenance.json",
            "external_runtime_provenance_sha256": "d" * 64,
            "external_runtime_environment_sha256": "e" * 64,
            "external_runtime_provenance": {"environment_sha256": "e" * 64},
            "null_equivalence": {"passed": True, "base_map_cluster_count": 3},
        }
        self.assertEqual(_semantic_status("G8", g8), "PASS")
        archive = dict(g8)
        archive.update(
            {
                "execution_mode": "authenticated_precomputed_rt_archive",
                "engine_family": "wireless-insite",
                "adapter_source_path": None,
                "adapter_source_sha256": None,
                "external_engine_config_path": "external_engine_config.bin",
                "external_engine_config_sha256": "f" * 64,
                "external_runtime_provenance_path": None,
                "external_runtime_provenance_sha256": None,
                "external_runtime_environment_sha256": None,
                "external_runtime_provenance": None,
            }
        )
        self.assertEqual(_semantic_status("G8", archive), "FAIL")
        archive["external_runtime_provenance"] = {
            "environment_sha256": "e" * 64
        }
        self.assertEqual(_semantic_status("G8", archive), "FAIL")
        g8["active_direction_agreement_ci95_low"] = 0.7
        self.assertEqual(_semantic_status("G8", g8), "FAIL")

    def test_c13_follows_the_same_supported_rule_as_other_claims(self):
        self.assertEqual(_claim_state("C13", ["PASS"], False), "SUPPORTED")
        self.assertEqual(_claim_state("C13", ["PASS"], True), "SOFTWARE_ONLY")
        self.assertEqual(_claim_state("C12", ["PASS"], False), "SUPPORTED")

    def test_not_assessed_from_later_stage_cannot_erase_upstream_gate(self):
        (self.root / "qualification").mkdir()
        (self.root / "evaluation").mkdir()
        (self.root / "controls").mkdir()
        context = evidence_context(self.config, self.dataset, "FORBIDDEN")
        checkpoint = self.root / "qualification" / "teacher.pt"
        checkpoint.write_bytes(b"teacher")
        write_json(
            self.root / "qualification" / "gate.json",
            {
                "schema_version": QUALIFICATION_SCHEMA,
                "status": "DRY_RUN_PASS_NOT_EVIDENCE",
                "passed": True,
                **context,
                "upstream_gates": complete_gate_vector({"G1": "PASS", "G2": "PASS"}),
                "teacher_checkpoint": str(checkpoint),
                "teacher_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "primary_route_contract": PRIMARY_ROUTE_CONTRACT,
                "physical_response": {
                    "status": "NOT_ASSESSED_FIXTURE_FORBIDDEN",
                    "formal_physical_response_required_for_nonfixture": True,
                },
            },
        )
        write_json(
            self.root / "qualification" / "manifest.json",
            {
                "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
                **context,
                "files": artifact_manifest(self.root / "qualification", evidence=context),
            },
        )
        write_json(
            self.root / "evaluation" / "gate.json",
            {
                **context,
                "gate_vector": complete_gate_vector({"G1": "PASS", "G2": "PASS", "G3": "PASS"}),
            },
        )
        write_json(
            self.root / "controls" / "gate.json",
            {**context, "gate_vector": complete_gate_vector({"G4": "PASS"})},
        )
        result = assemble_claim_evidence(self.config, self.dataset, self.root)
        self.assertEqual(result["gate_vector"]["G1"], "PASS")
        self.assertEqual(result["gate_vector"]["G2"], "PASS")
        self.assertEqual(result["gate_vector"]["G3"], "BLOCKED")
        self.assertEqual(result["gate_vector"]["G4"], "BLOCKED")
        self.assertIn("G3", result["evidence_errors"])
        self.assertIn("G4", result["evidence_errors"])

    def test_evidence_context_propagates_fixture_forbidden(self):
        evidence = evidence_context(self.config, self.dataset, "FORMAL_EXPERIMENT_ALLOWED")
        self.assertTrue(evidence["fixture"])
        self.assertEqual(evidence["scientific_use"], "FORBIDDEN")
        self.assertEqual(len(evidence["dataset_sha256"]), 64)
        self.assertEqual(len(evidence["config_sha256"]), 64)

    def test_fixture_runtime_cache_never_applies_to_nonfixture_data(self):
        authenticated = {
            "requirements_lock_sha256": "a" * 64,
            "source_tree_sha256": "b" * 64,
        }
        nonfixture = SimpleNamespace(is_fixture=False, source_path=self.path)
        with (
            patch.dict(os.environ, {"CSI_PAIRS_FIXTURE_RUNTIME_CACHE": "1"}),
            patch("formal_v2.formal_evidence._FIXTURE_RUNTIME_CACHE", None),
            patch(
                "formal_v2.formal_evidence.runtime_provenance",
                return_value={"unvalidated": True},
            ) as provenance,
            patch(
                "formal_v2.formal_evidence.validate_runtime_provenance",
                return_value=authenticated,
            ) as validate,
        ):
            first = evidence_context(self.config, self.dataset, "FORBIDDEN")
            second = evidence_context(self.config, self.dataset, "FORBIDDEN")
            self.assertEqual(first["runtime_provenance"], authenticated)
            self.assertEqual(second["runtime_provenance"], authenticated)
            self.assertEqual(provenance.call_count, 1)
            self.assertEqual(validate.call_count, 1)

            evidence_context(self.config, nonfixture, "CANDIDATE")
            evidence_context(self.config, nonfixture, "CANDIDATE")
            self.assertEqual(provenance.call_count, 3)
            self.assertEqual(validate.call_count, 3)

    def test_direct_evidence_entrypoint_configures_deterministic_torch(self):
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        evidence = evidence_context(self.config, self.dataset, "FORBIDDEN")

        self.assertTrue(evidence["runtime_provenance"]["torch"]["deterministic_algorithms"])
        self.assertTrue(evidence["runtime_provenance"]["torch"]["cudnn_deterministic"])
        self.assertTrue(evidence["runtime_provenance"]["python_dont_write_bytecode"])

    def test_evidence_context_rejects_unlocked_main_runtime(self):
        configure_reproducible_runtime()
        original = runtime_provenance()
        mutations = []

        wrong_python = json.loads(json.dumps(original))
        wrong_python["python_version"] = "3.11.9"
        mutations.append((wrong_python, "CPython 3.12"))

        wrong_version = json.loads(json.dumps(original))
        wrong_version["installed_distributions"]["torch"]["version"] = "2.0.0"
        mutations.append((wrong_version, "torch must be exactly"))

        missing_record = json.loads(json.dumps(original))
        missing_record["installed_distributions"]["numpy"]["record_sha256"] = None
        mutations.append((missing_record, "numpy has no RECORD provenance"))

        wrong_wheel = json.loads(json.dumps(original))
        wrong_wheel["installed_distributions"]["numpy"]["wheel_sha256"] = "0" * 64
        mutations.append((wrong_wheel, "numpy wheel hash is not authenticated"))

        nondeterministic = json.loads(json.dumps(original))
        nondeterministic["torch"]["deterministic_algorithms"] = False
        mutations.append((nondeterministic, "deterministic torch state"))

        reduced_precision = json.loads(json.dumps(original))
        reduced_precision["torch"]["float32_matmul_precision"] = "high"
        mutations.append((reduced_precision, "deterministic torch state"))

        unsupported_platform = json.loads(json.dumps(original))
        unsupported_platform["platform_system"] = "Windows"
        unsupported_platform["platform_machine"] = "AMD64"
        mutations.append((unsupported_platform, "supports only macOS 14\\+ arm64"))

        obsolete_platform = json.loads(json.dumps(original))
        if original["platform_system"] == "Darwin":
            obsolete_platform["platform_mac_version"] = "13.6"
            mutations.append((obsolete_platform, "requires macOS 14.0 or newer"))
            malformed_platform = json.loads(json.dumps(original))
            malformed_platform["platform_mac_version"] = "14.beta"
            mutations.append((malformed_platform, "no valid macOS version"))
        else:
            obsolete_platform["platform_libc_version"] = "2.27"
            mutations.append((obsolete_platform, "requires glibc 2.28 or newer"))
            malformed_platform = json.loads(json.dumps(original))
            malformed_platform["platform_libc_version"] = "2.28-custom"
            mutations.append((malformed_platform, "no valid glibc version"))

        for runtime, message in mutations:
            with self.subTest(message=message):
                with (
                    patch("formal_v2.formal_evidence._FIXTURE_RUNTIME_CACHE", None),
                    patch(
                        "formal_v2.formal_evidence.runtime_provenance",
                        return_value=runtime,
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        evidence_context(self.config, self.dataset, "FORBIDDEN")

    def test_candidate_or_forbidden_nonfixture_cannot_start_factorial(self):
        arrays = _archive_arrays(self.path)
        arrays["maps"] = arrays["maps"].copy()
        arrays["noop_maps"] = arrays["noop_maps"].copy()
        arrays["canonical_map_sha256"] = arrays["canonical_map_sha256"].copy()
        arrays["noop_map_sha256"] = arrays["noop_map_sha256"].copy()
        for scene in range(arrays["maps"].shape[0]):
            # The toy fixture intentionally reuses foundations. Give this synthetic
            # non-fixture candidate a distinct invariant foundation per scene so this
            # test reaches the evidence gate it is intended to exercise.
            offset = 0.001 * (scene + 1)
            arrays["maps"][scene, :, 1, 0, 0] += offset
            arrays["noop_maps"][scene, :, 1, 0, 0] += offset
            for world in range(arrays["maps"].shape[1]):
                arrays["canonical_map_sha256"][scene, world] = hashlib.sha256(
                    np.ascontiguousarray(arrays["maps"][scene, world], dtype="<f8").tobytes()
                ).hexdigest()
                arrays["noop_map_sha256"][scene, world] = hashlib.sha256(
                    np.ascontiguousarray(arrays["noop_maps"][scene, world], dtype="<f8").tobytes()
                ).hexdigest()
        metadata = parse_strict_json(str(arrays["metadata_json"].item()))
        metadata["fixture"] = False
        metadata["scientific_use"] = "CANDIDATE"
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
        candidate_path = self.root / "candidate.npz"
        np.savez_compressed(candidate_path, **arrays)
        candidate = FormalDataset.load(candidate_path)
        context = evidence_context(self.config, candidate, "FORBIDDEN")
        gate = {
            "schema_version": QUALIFICATION_SCHEMA,
            "passed": True,
            **context,
            "upstream_gates": complete_gate_vector({"G1": "PASS", "G2": "PASS"}),
            "teacher_checkpoint": "/not/read/by-this-check",
            "teacher_checkpoint_sha256": "0" * 64,
            "primary_route_contract": PRIMARY_ROUTE_CONTRACT,
        }
        with self.assertRaisesRegex(RuntimeError, "FORMAL_EXPERIMENT_ALLOWED"):
            require_formal_qualification(gate, self.config, candidate, allow_nonscientific_fixture=False)

    def test_path_incidence_and_local_aggregation_are_bounded(self):
        edge = next(self.dataset.directed_edges(0))
        value = path_incidence(
            self.dataset, 0, edge.source_world, edge.target_world, 0, edge.bit_index
        )
        local = localization_path_incidence(self.dataset, 0, edge.source_world, 0)
        self.assertGreaterEqual(value, 0)
        self.assertLessEqual(value, 1)
        self.assertGreaterEqual(local, 0)
        self.assertLessEqual(local, 1)

    def test_noop_path_threshold_comes_from_registered_retrace(self):
        self.assertEqual(_noop_path_threshold(self.config, self.dataset, 0.99), 0.0)

    def test_frozen_map_proposal_is_deterministic_and_uses_typed_actions(self):
        scene = int(self.dataset.indices_for_role("source_calibration_fit")[0])
        world = int(self.dataset.natural_world_index[scene])
        first_maps, first_actions, first_audit = frozen_map_proposals(
            self.dataset.maps[scene, world],
            self.dataset.radio_config[scene],
            self.dataset.map_channel_names,
            int(self.dataset.metadata["assets"]["material_category_count"]),
            self.config,
        )
        second_maps, second_actions, second_audit = frozen_map_proposals(
            self.dataset.maps[scene, world],
            self.dataset.radio_config[scene],
            self.dataset.map_channel_names,
            int(self.dataset.metadata["assets"]["material_category_count"]),
            self.config,
        )
        np.testing.assert_array_equal(first_maps, second_maps)
        np.testing.assert_array_equal(first_actions, second_actions)
        self.assertEqual(first_audit, second_audit)
        self.assertEqual(
            first_audit["proposal_count"], self.config["risk"]["proposal_count"]
        )
        self.assertEqual(first_actions.shape[1], 4 + 2 * 4)

    def test_risk_support_threshold_is_frozen_from_selection(self):
        fit_d = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        fit_u = np.asarray([0.0, 0.2, 0.1, 0.4, 0.3, 0.7])
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        selection_d = np.asarray([-5.0, -4.0, 4.0, 5.0])
        selection_u = np.asarray([-2.0, -1.0, 1.0, 2.0])
        calibrator = fit_constrained_risk_calibrator(
            fit_d,
            fit_u,
            labels,
            (selection_d, selection_u, np.asarray([0, 0, 1, 1])),
            self.config,
        )
        transformed = np.clip(
            calibrator.standardization.transform(
                np.column_stack((selection_d, selection_u))
            ),
            -5.0,
            5.0,
        )
        expected = np.quantile(
            calibrator.support.squared_distance(transformed),
            1.0 - float(self.config["risk"]["support_alpha"]),
        )
        self.assertAlmostEqual(calibrator.support.threshold, expected)

    def test_coverage_monotonicity_checks_median_and_p90(self):
        retained = {
            "0.9": {"median_error": 3.0, "p90_error": 6.0},
            "0.75": {"median_error": 2.0, "p90_error": 5.0},
            "0.5": {"median_error": 1.0, "p90_error": 4.0},
        }
        self.assertTrue(_coverage_error_monotonic(retained, 0.0))
        retained["0.5"]["p90_error"] = 5.5
        self.assertFalse(_coverage_error_monotonic(retained, 0.0))

    def test_target_verification_failure_is_explicitly_nonblocking(self):
        from formal_v2.formal_data_verification import run_data_verification

        gate = run_data_verification(
            self.config,
            self.dataset,
            ROOT / "formal_v2/configs/fixture_verifier.json",
            self.root / "verified-target-contract",
        )
        gate["role_status"]["target"] = "FAIL"
        gate["nonblocking_scene_failures"] = ["target-scene"]
        self.assertIs(require_data_verification(gate, self.config, self.dataset), gate)
        with self.assertRaisesRegex(RuntimeError, "target"):
            require_verified_roles(gate, self.config, self.dataset, ("target",))

    def test_teacher_checkpoint_hash_is_authenticated(self):
        checkpoint = self.root / "teacher.pt"
        checkpoint.write_bytes(b"frozen-teacher")
        context = evidence_context(self.config, self.dataset, "FORBIDDEN")
        gate = {
            "schema_version": QUALIFICATION_SCHEMA,
            "passed": True,
            **context,
            "upstream_gates": complete_gate_vector({"G1": "PASS", "G2": "PASS"}),
            "teacher_checkpoint": str(checkpoint),
            "teacher_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "primary_route_contract": PRIMARY_ROUTE_CONTRACT,
            "physical_response": {
                "status": "NOT_ASSESSED_FIXTURE_FORBIDDEN",
                "formal_physical_response_required_for_nonfixture": True,
            },
        }
        require_formal_qualification(
            gate, self.config, self.dataset, allow_nonscientific_fixture=True
        )
        checkpoint.write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            require_formal_qualification(
                gate, self.config, self.dataset, allow_nonscientific_fixture=True
            )

    def test_manifested_qualification_can_authenticate_recorded_main_runtime(self):
        stage = self.root / "cross-interpreter-qualification"
        checkpoint = stage / "checkpoints" / "teacher.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"recorded-main-runtime-teacher")
        context = evidence_context(self.config, self.dataset, "FORBIDDEN")
        gate = {
            "schema_version": QUALIFICATION_SCHEMA,
            "passed": True,
            **context,
            "upstream_gates": complete_gate_vector({"G1": "PASS", "G2": "PASS"}),
            "teacher_checkpoint": str(checkpoint),
            "teacher_checkpoint_sha256": sha256_file(checkpoint),
            "primary_route_contract": PRIMARY_ROUTE_CONTRACT,
            "physical_response": {
                "status": "NOT_ASSESSED_FIXTURE_FORBIDDEN",
                "formal_physical_response_required_for_nonfixture": True,
            },
        }
        write_json(stage / "gate.json", gate)
        write_json(
            stage / "manifest.json",
            {
                "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
                **context,
                "files": artifact_manifest(stage, evidence=context),
            },
        )

        with patch(
            "formal_v2.formal_evidence.evidence_context",
            side_effect=AssertionError("must not inspect the adapter interpreter as main"),
        ):
            authenticated = require_manifested_formal_qualification(
                gate,
                self.config,
                self.dataset,
                allow_nonscientific_fixture=True,
                evidence_runtime=context["runtime_provenance"],
            )
        self.assertIs(authenticated, gate)

    def test_failed_fixture_qualification_only_allows_explicit_software_execution(self):
        checkpoint = self.root / "software-teacher.pt"
        checkpoint.write_bytes(b"software-only-teacher")
        context = evidence_context(self.config, self.dataset, "FORBIDDEN")
        gate = {
            "schema_version": QUALIFICATION_SCHEMA,
            "passed": False,
            **context,
            "upstream_gates": complete_gate_vector({"G1": "FAIL", "G2": "FAIL"}),
            "teacher_checkpoint": str(checkpoint),
            "teacher_checkpoint_sha256": sha256_file(checkpoint),
            "primary_route_contract": PRIMARY_ROUTE_CONTRACT,
            "physical_response": {
                "status": "NOT_ASSESSED_FIXTURE_FORBIDDEN",
                "formal_physical_response_required_for_nonfixture": True,
            },
        }
        require_formal_qualification(
            gate, self.config, self.dataset, allow_nonscientific_fixture=True
        )
        with self.assertRaisesRegex(RuntimeError, "upstream qualification"):
            require_formal_qualification(
                gate, self.config, self.dataset, allow_nonscientific_fixture=False
            )

    def test_manifested_gate_rejects_payload_or_file_mutation(self):
        stage = self.root / "authenticated-stage"
        stage.mkdir()
        context = evidence_context(self.config, self.dataset, "FORBIDDEN")
        gate = {"schema_version": "test-gate-v1", "passed": True, **context}
        write_json(stage / "gate.json", gate)
        write_json(
            stage / "manifest.json",
            {
                "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
                **context,
                "files": artifact_manifest(stage, evidence=context),
            },
        )
        require_stage_manifested_gate(
            stage / "gate.json",
            gate,
            self.config,
            self.dataset,
            schema_version="test-gate-v1",
        )
        mutated = {**gate, "passed": False}
        with self.assertRaisesRegex(RuntimeError, "differs"):
            require_stage_manifested_gate(
                stage / "gate.json",
                mutated,
                self.config,
                self.dataset,
                schema_version="test-gate-v1",
            )
        write_json(stage / "gate.json", mutated)
        with self.assertRaisesRegex(RuntimeError, "changed after manifesting"):
            require_stage_manifested_gate(
                stage / "gate.json",
                mutated,
                self.config,
                self.dataset,
                schema_version="test-gate-v1",
            )

    def test_manifested_gate_authenticates_complete_stage_inventory(self):
        stage = self.root / "complete-stage"
        stage.mkdir()
        context = evidence_context(self.config, self.dataset, "FORBIDDEN")
        gate = {"schema_version": "test-gate-v1", "passed": True, **context}
        write_json(stage / "gate.json", gate)
        (stage / "evidence.csv").write_text("value\n1\n", encoding="utf-8")
        write_json(
            stage / "manifest.json",
            {
                "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
                **context,
                "files": artifact_manifest(stage, evidence=context),
            },
        )
        require_stage_manifested_gate(
            stage / "gate.json",
            gate,
            self.config,
            self.dataset,
            schema_version="test-gate-v1",
        )
        (stage / "unmanifested.csv").write_text("value\n2\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "inventory is incomplete"):
            require_stage_manifested_gate(
                stage / "gate.json",
                gate,
                self.config,
                self.dataset,
                schema_version="test-gate-v1",
            )

    def test_claim_reauthenticates_copied_stage_input_manifest(self):
        stage = self.root / "bound-stage"
        stage.mkdir()
        source = stage / "adapter_manifest.json"
        write_json(source, {"schema_version": "adapter-v1"})
        payload = {
            "input_manifest_path": source.name,
            "input_manifest_sha256": sha256_file(source),
        }
        write_json(
            stage / "manifest.json",
            {
                "files": [{"path": source.name, "sha256": sha256_file(source)}],
            },
        )
        _validate_stage_bound_input(stage / "gate.json", payload)
        write_json(source, {"schema_version": "tampered"})
        with self.assertRaisesRegex(RuntimeError, "hash-mismatched"):
            _validate_stage_bound_input(stage / "gate.json", payload)

    def test_control_and_scene_id_manifests_are_fail_closed(self):
        controls = {
            "schema_version": "csi-pairs-v6-resource-controls-v1",
            "controls": [{"control_id": name, "command": ["true"]} for name in CONTROL_IDS],
        }
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            validate_control_manifest(controls)
        scene_manifest = {
                "schema_version": "csi-pairs-v6-scene-id-adapters-v3",
                "adapters": [
                    {
                        "adapter_id": "m",
                        "model_name": "model",
                        "implementation_revision": "a" * 64,
                        "adapter_source_path": "adapter.py",
                        "adapter_source_sha256": "a" * 64,
                        "model_checkpoint_path": "model.pt",
                        "model_checkpoint_sha256": "b" * 64,
                        "training_provenance_path": "training.json",
                        "training_provenance_sha256": "c" * 64,
                        "command": ["{python}", "{adapter_source}"],
                    }
                ],
            }
        validate_scene_id_manifest(scene_manifest)
        unsafe_scene_manifest = json.loads(json.dumps(scene_manifest))
        unsafe_scene_manifest["adapters"][0]["adapter_id"] = "../../../qualification"
        with self.assertRaisesRegex(ValueError, "safe path component"):
            validate_scene_id_manifest(unsafe_scene_manifest)
        duplicate_scene_manifest = json.loads(json.dumps(scene_manifest))
        duplicate_scene_manifest["adapters"].append(
            dict(duplicate_scene_manifest["adapters"][0])
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_scene_id_manifest(duplicate_scene_manifest)
        validate_external_validity_manifest(
            {
                "schema_version": "csi-pairs-v6-external-validity-adapter-v3",
                "evidence_type": "independent_rt_engine",
                "source_revision": "revision",
                "license_id": "license",
                "adapter_source_path": "adapter.py",
                "adapter_source_sha256": "a" * 64,
                "command": [
                    "{project_root}/formal_v2/external_adapters/"
                    ".runtime-sionna/venv/bin/python",
                    "{adapter_source}",
                ],
            }
        )

    def test_scene_id_rows_are_exactly_paired_to_held_out_positions(self):
        evidence = evidence_context(self.config, self.dataset, "FORBIDDEN")
        adapter = {"adapter_id": "test", "model_name": "model", "command": ["true"]}
        rows = []
        for scene_value in self.dataset.indices_for_role("source_final_unseen_bank"):
            scene = int(scene_value)
            for position in np.flatnonzero(
                self.dataset.position_roles[scene] == "standard"
            ):
                unit = (
                    f"{self.dataset.bank_ids[scene]}:"
                    f"{self.dataset.position_ids[scene, int(position)]}"
                )
                rows.extend(
                    {
                        "unit_id": unit,
                        "model_name": "model",
                        "condition": condition,
                        "bank_id": str(self.dataset.bank_ids[scene]),
                        "position_id": str(
                            self.dataset.position_ids[scene, int(position)]
                        ),
                        "localization_error_m": "1.0",
                        "prediction_x": str(index + 1),
                        "prediction_y": "0.0",
                        "source_role": "source_final_unseen_bank",
                        "dataset_sha256": str(evidence["dataset_sha256"]),
                        "config_sha256": str(evidence["config_sha256"]),
                    }
                    for index, condition in enumerate(
                        ("map", "scene_id", "map_swap", "id_swap")
                    )
                )
        validate_scene_id_rows(adapter, [dict(row) for row in rows], self.dataset, evidence)
        negative = [dict(row) for row in rows]
        negative[0]["localization_error_m"] = "-1.0"
        with self.assertRaisesRegex(RuntimeError, "nonnegative"):
            validate_scene_id_rows(adapter, negative, self.dataset, evidence)
        with self.assertRaisesRegex(RuntimeError, "duplicates"):
            validate_scene_id_rows(adapter, [dict(row) for row in rows] + [dict(rows[0])], self.dataset, evidence)

        scalar_only = [{**row, "response_score": "1.0"} for row in rows]
        for row in scalar_only:
            row.pop("prediction_x")
            row.pop("prediction_y")
        with self.assertRaisesRegex(RuntimeError, "columns must be exact"):
            validate_scene_id_rows(adapter, scalar_only, self.dataset, evidence)

    def test_scene_id_training_provenance_must_bind_training_artifacts(self):
        evidence = evidence_context(self.config, self.dataset, "FORBIDDEN")
        training = self.root / "training.json"
        write_json(training, {"source_roles": ["source_encoder_train"]})
        adapter = {
            "adapter_id": "m",
            "model_name": "model",
            "adapter_source_sha256": "a" * 64,
            "model_checkpoint_sha256": "b" * 64,
        }
        provenance = self.root / "provenance.json"
        payload = {
            "schema_version": "csi-pairs-v6-scene-id-training-provenance-v1",
            "adapter_id": "m",
            "model_name": "model",
            "dataset_sha256": evidence["dataset_sha256"],
            "config_sha256": evidence["config_sha256"],
            "fixture": evidence["fixture"],
            "training_role": "source_encoder_train",
            "selection_role": "source_method_selection",
            "evaluation_role": "source_final_unseen_bank",
            "target_data_used": False,
            "checkpoint_rule": "source_selection_then_frozen",
            "adapter_source_sha256": "a" * 64,
            "model_checkpoint_sha256": "b" * 64,
            "training_record_path": str(training),
            "training_record_sha256": sha256_file(training),
            "training_command_sha256": "c" * 64,
            "external_execution_manifest_sha256": "d" * 64,
        }
        write_json(provenance, payload)
        validate_scene_id_training_provenance(adapter, provenance, evidence)
        write_json(training, {"source_roles": ["target"]})
        with self.assertRaisesRegex(RuntimeError, "training record"):
            validate_scene_id_training_provenance(adapter, provenance, evidence)

    def test_scene_id_gate_uses_cluster_macro_confidence_intervals(self):
        rows = []
        for cluster_index, cluster in enumerate(("a", "b", "c", "d")):
            for repeat in range(1 if cluster != "a" else 20):
                unit = f"{cluster}-{repeat}"
                for condition, error, prediction in (
                    ("map", 1.0, (0.0, 0.0)),
                    ("scene_id", 1.05, (0.0, 0.0)),
                    ("map_swap", 0.0, (float(cluster_index + 1), 0.0)),
                    ("id_swap", 0.0, (float(cluster_index + 1), 0.0)),
                ):
                    rows.append(
                        {
                            "unit_id": unit,
                            "condition": condition,
                            "localization_error_m": error,
                            "prediction_x": prediction[0],
                            "prediction_y": prediction[1],
                            "canonical_unit_id": unit,
                            "canonical_base_map_digest": cluster,
                            "canonical_bank_digest": f"bank-{cluster}",
                        }
                    )
        result = scene_id_model_assessment(
            self.config, {"adapter_id": "a", "model_name": "m"}, rows
        )
        self.assertEqual(result["base_map_cluster_count"], 4)
        self.assertAlmostEqual(result["scene_id_minus_map_error_m"], 0.05)
        self.assertTrue(result["passed"])

    def test_external_direction_gate_cannot_be_inflated_by_duplicate_rows(self):
        rows = []
        for cluster, matches, count in (("a", 1, 20), ("b", 0, 1)):
            rows.extend(
                {
                    "canonical_unit_id": f"{cluster}-{index}",
                    "canonical_base_map_digest": cluster,
                    "canonical_bank_digest": f"bank-{cluster}",
                    "route": "active",
                    "primary_direction": "1",
                    "external_direction": "1" if matches else "-1",
                    "primary_effect": "0.1",
                    "external_effect": "0.1",
                }
                for index in range(count)
            )
        interval = _cluster_direction_interval(rows, 100)
        self.assertEqual(interval["base_map_cluster_count"], 2)
        self.assertAlmostEqual(interval["estimate"], 0.5)

    def test_rt_calibration_manifest_requires_bound_input_and_source_paths(self):
        manifest = {
            "schema_version": "csi-pairs-v6-rt-calibration-adapter-v6",
            "protocol_path": "protocol.json",
            "protocol_sha256": "a" * 64,
            "fit_dataset_path": "fit.bin",
            "fit_dataset_sha256": "b" * 64,
            "validation_inputs_path": "validation-inputs.bin",
            "validation_inputs_sha256": "c" * 64,
            "validation_reference_path": "validation-reference.csv",
            "validation_reference_sha256": "e" * 64,
            "adapter_source_path": "adapter.py",
            "adapter_source_sha256": "d" * 64,
            "design_record_path": "CALIBRATION_DESIGN.md",
            "design_record_sha256": "f" * 64,
            "license_review_path": "LICENSE_REVIEW.md",
            "license_review_sha256": "0" * 64,
            "command": [
                "{python}",
                "{adapter_source}",
                "--fit",
                "{fit_dataset}",
                "--validation-inputs",
                "{validation_inputs}",
                "--protocol",
                "{protocol}",
                "--output",
                "{output}",
            ],
        }
        validate_rt_calibration_manifest(manifest)
        forged = dict(manifest)
        forged["command"] = ["python3", "-c", "print('fabricated')"]
        with self.assertRaisesRegex(ValueError, "authenticated adapter"):
            validate_rt_calibration_manifest(forged)
        forged = dict(manifest)
        forged["command"] = [*manifest["command"], "--reference", "/tmp/held-out.csv"]
        with self.assertRaisesRegex(ValueError, "exclude validation references"):
            validate_rt_calibration_manifest(forged)
        leaked = {**manifest, "command": [*manifest["command"], "--secret", "/tmp/gold.csv"]}
        with self.assertRaisesRegex(ValueError, "authenticated adapter"):
            validate_rt_calibration_manifest(leaked)
        del manifest["fit_dataset_path"]
        with self.assertRaisesRegex(ValueError, "fields"):
            validate_rt_calibration_manifest(manifest)

    def test_rt_calibration_stage_isolates_reference_and_binds_outputs_end_to_end(self):
        fit = self.root / "rt-fit.json"
        validation_inputs = self.root / "rt-validation-inputs.json"
        validation_reference = self.root / "rt-validation-reference.csv"
        fit_a = self._rt_unit(
            "fit-a", "fit-scene-a", "fit-a-source.json", "fit-batch", "fit-record-a", "fit-raw-a", {"observation": 1}
        )
        fit_b = self._rt_unit(
            "fit-b", "fit-scene-b", "fit-b-source.json", "fit-batch", "fit-record-b", "fit-raw-b", {"observation": 2}
        )
        validation_a = self._rt_unit(
            "unit-a", "validation-scene-a", "validation-a-source.json", "validation-batch", "validation-record-a", "validation-raw-a", {"observation": 3}
        )
        validation_b = self._rt_unit(
            "unit-b", "validation-scene-b", "validation-b-source.json", "validation-batch", "validation-record-b", "validation-raw-b", {"observation": 4}
        )
        write_json(
            fit,
            {
                "schema_version": "csi-pairs-v6-rt-calibration-partition-v3",
                "partition": "fit",
                "units": [fit_a, fit_b],
            },
        )
        write_json(
            validation_inputs,
            {
                "schema_version": "csi-pairs-v6-rt-calibration-partition-v3",
                "partition": "validation",
                "units": [validation_a, validation_b],
            },
        )
        write_csv(
            validation_reference,
            [
                {
                    "unit_id": unit_id,
                    "path_loss": 1.0 + index,
                    "delay_spread": 2.0 + index,
                    "angular_spread": 3.0 + index,
                    "visible_path_count": 4 + index,
                }
                for index, unit_id in enumerate(("unit-a", "unit-b"))
            ],
        )
        protocol = self.root / "rt-protocol.json"
        write_json(
            protocol,
            {
                "schema_version": "csi-pairs-v6-rt-calibration-protocol-v4",
                "frozen_utc": "2026-08-06T00:00:00Z",
                "absolute_tolerances": {
                    "path_loss": 0.2,
                    "delay_spread": 0.2,
                    "angular_spread": 0.2,
                    "visible_path_count": 0.2,
                },
                "exclusion_rules": [],
                "minimum_validation_units": 2,
                "aggregation": "mean_absolute_error_per_unit",
            },
        )
        adapter = self.root / "rt_adapter.py"
        adapter.write_text(
            "import argparse, hashlib, json\n"
            "from pathlib import Path\n"
            "p=argparse.ArgumentParser(); p.add_argument('--output'); p.add_argument('--fit'); p.add_argument('--validation-inputs'); p.add_argument('--protocol'); a=p.parse_args()\n"
            "out=Path(a.output); fitted=out/'fitted.json'; fitted.write_text('{\"gain\":1.0}', encoding='utf-8')\n"
            "sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()\n"
            "sim=out/'simulated_statistics.csv'; secret=list(Path(a.validation_inputs).parent.glob('*reference*.csv'))\n"
            "if secret: sim.write_bytes(secret[0].read_bytes())\n"
            "else: sim.write_text('unit_id,path_loss,delay_spread,angular_spread,visible_path_count\\nunit-a,1.05,2.05,3.05,4\\nunit-b,2.05,3.05,4.05,5\\n',encoding='utf-8')\n"
            "payload={'schema_version':'csi-pairs-v6-rt-calibration-adapter-result-v5','fit_dataset_sha256':sha(a.fit),'validation_inputs_sha256':sha(a.validation_inputs),'fitted_parameters_path':fitted.name,'fitted_parameters_sha256':sha(fitted),'simulated_statistics_path':sim.name,'simulated_statistics_sha256':sha(sim)}\n"
            "(out/'adapter_result.json').write_text(json.dumps(payload,sort_keys=True),encoding='utf-8')\n",
            encoding="utf-8",
        )
        design_record = self.root / "CALIBRATION_DESIGN.md"
        design_record.write_text(
            "# C11 calibration design\n\n"
            "This design was pre-registered.\n\n"
            "Fit units are disjoint Denver scenes. Validation units are disjoint Miami scenes.\n\n"
            "## Frozen tolerances\n\n"
            "All four statistic tolerances are frozen before validation.\n"
            + "Design evidence.\n" * 40,
            encoding="utf-8",
        )
        license_review = self.root / "LICENSE_REVIEW.md"
        license_review.write_text(
            "# C11 license review\n\n"
            "Sionna, DiffeRT, and OpenStreetMap license records were reviewed.\n"
            "Both engine outputs are simulations and not measurements.\n"
            + "License evidence.\n" * 40,
            encoding="utf-8",
        )
        manifest = self.root / "rt-manifest.json"
        write_json(
            manifest,
            {
                "schema_version": "csi-pairs-v6-rt-calibration-adapter-v6",
                "protocol_path": str(protocol),
                "protocol_sha256": sha256_file(protocol),
                "fit_dataset_path": str(fit),
                "fit_dataset_sha256": sha256_file(fit),
                "validation_inputs_path": str(validation_inputs),
                "validation_inputs_sha256": sha256_file(validation_inputs),
                "validation_reference_path": str(validation_reference),
                "validation_reference_sha256": sha256_file(validation_reference),
                "adapter_source_path": str(adapter),
                "adapter_source_sha256": sha256_file(adapter),
                "design_record_path": str(design_record),
                "design_record_sha256": sha256_file(design_record),
                "license_review_path": str(license_review),
                "license_review_sha256": sha256_file(license_review),
                "command": [
                    "{python}",
                    "{adapter_source}",
                    "--output",
                    "{output}",
                    "--fit",
                    "{fit_dataset}",
                    "--validation-inputs",
                    "{validation_inputs}",
                    "--protocol",
                    "{protocol}",
                ],
            },
        )
        output = self.root / "formal-run"
        gate = run_rt_calibration_gate(self.config, self.dataset, manifest, output)
        self.assertTrue(gate["passed"])
        self.assertEqual(len(gate["statistics"]), 4)
        self.assertEqual(gate["validation_unit_count"], 2)
        self.assertTrue(gate["fit_validation_independence"]["verified"])
        self.assertAlmostEqual(
            gate["statistics"]["path_loss"]["mean_absolute_error_per_unit"], 0.05
        )
        _validate_stage_bound_input(
            output / "qualification/rt_calibration/gate.json",
            gate,
            self.config,
            "rt_calibration",
        )
        validation_reference.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "validation reference"):
            _validate_stage_bound_input(
                output / "qualification/rt_calibration/gate.json",
                gate,
                self.config,
                "rt_calibration",
            )

    def test_rt_calibration_rejects_unit_error_cancellation(self):
        statistics = (
            "path_loss",
            "delay_spread",
            "angular_spread",
            "visible_path_count",
        )
        reference = {
            "unit-a": {"unit_id": "unit-a", **{name: 0.0 for name in statistics}},
            "unit-b": {"unit_id": "unit-b", **{name: 100.0 for name in statistics}},
        }
        simulated = {
            "unit-a": {"unit_id": "unit-a", **{name: 100.0 for name in statistics}},
            "unit-b": {"unit_id": "unit-b", **{name: 0.0 for name in statistics}},
        }
        validated, assessments = assess_rt_calibration(
            reference,
            simulated,
            {name: 0.01 for name in statistics},
        )
        self.assertTrue(all(row["absolute_error_path_loss"] == 100.0 for row in validated))
        self.assertEqual(assessments["path_loss"]["reference_mean"], 50.0)
        self.assertEqual(assessments["path_loss"]["simulated_mean"], 50.0)
        self.assertEqual(assessments["path_loss"]["mean_absolute_error_per_unit"], 100.0)
        self.assertFalse(assessments["path_loss"]["passed"])

    def test_rt_calibration_rejects_fit_validation_scene_or_unit_overlap(self):
        fit_path = self.root / "raw-fit.json"
        validation_path = self.root / "raw-validation.json"
        fit_unit = self._rt_unit(
            "fit-a", "shared-scene", "raw-fit-source.json", "fit-batch", "fit-record", "fit-raw", {"raw": 1}
        )
        validation_unit = self._rt_unit(
            "validation-a", "shared-scene", "raw-validation-source.json", "validation-batch", "validation-record", "validation-raw", {"raw": 2}
        )
        write_json(fit_path, {
            "schema_version": "csi-pairs-v6-rt-calibration-partition-v3",
            "partition": "fit",
            "units": [fit_unit],
        })
        write_json(validation_path, {
            "schema_version": "csi-pairs-v6-rt-calibration-partition-v3",
            "partition": "validation",
            "units": [validation_unit],
        })
        # A dishonest sidecar is irrelevant: the outer runner derives identity from raw inputs.
        write_json(self.root / "lying-sidecar.json", {
            "fit": [{"unit_id": "fake-fit", "scene_id": "fake-fit-scene"}],
            "validation": [{"unit_id": "fake-validation", "scene_id": "fake-validation-scene"}],
        })
        fit = read_rt_partition_contract(fit_path, "fit")
        validation = read_rt_partition_contract(validation_path, "validation")
        with self.assertRaisesRegex(RuntimeError, "scene_overlap"):
            validate_rt_partition_independence(fit, validation, {"validation-a"})
        validation_unit["unit_id"] = "fit-a"
        validation_unit["scene_id"] = "validation-scene"
        write_json(validation_path, {
            "schema_version": "csi-pairs-v6-rt-calibration-partition-v3",
            "partition": "validation",
            "units": [validation_unit],
        })
        validation = read_rt_partition_contract(validation_path, "validation")
        with self.assertRaisesRegex(RuntimeError, "unit_overlap"):
            validate_rt_partition_independence(fit, validation, {"fit-a"})

    def test_rt_calibration_rejects_renamed_or_copied_raw_sources(self):
        shared = self._rt_unit(
            "fit-a",
            "fit-scene",
            "shared-source.json",
            "batch-a",
            "record-a",
            "raw-a",
            {"observation": [1, 2, 3]},
        )
        renamed = json.loads(json.dumps(shared))
        renamed["unit_id"] = "validation-renamed"
        renamed["scene_id"] = "validation-scene-renamed"
        fit_path = self.root / "renamed-fit.json"
        validation_path = self.root / "renamed-validation.json"
        write_json(
            fit_path,
            {
                "schema_version": "csi-pairs-v6-rt-calibration-partition-v3",
                "partition": "fit",
                "units": [shared],
            },
        )
        write_json(
            validation_path,
            {
                "schema_version": "csi-pairs-v6-rt-calibration-partition-v3",
                "partition": "validation",
                "units": [renamed],
            },
        )
        fit = read_rt_partition_contract(fit_path, "fit")
        validation = read_rt_partition_contract(validation_path, "validation")
        with self.assertRaisesRegex(RuntimeError, "source_asset_sha256_overlap"):
            validate_rt_partition_independence(
                fit, validation, {"validation-renamed"}
            )

        copied = json.loads(json.dumps(renamed))
        copied_asset = self.root / "renamed-copy.json"
        copied_asset.write_bytes((self.root / "shared-source.json").read_bytes())
        copied["source"]["asset_path"] = copied_asset.name
        write_json(
            validation_path,
            {
                "schema_version": "csi-pairs-v6-rt-calibration-partition-v3",
                "partition": "validation",
                "units": [copied],
            },
        )
        validation = read_rt_partition_contract(validation_path, "validation")
        with self.assertRaisesRegex(RuntimeError, "source_asset_sha256_overlap"):
            validate_rt_partition_independence(
                fit, validation, {"validation-renamed"}
            )

    def test_rt_calibration_rejects_payload_not_present_in_source_asset(self):
        unit = self._rt_unit(
            "fit-a",
            "fit-scene",
            "bound-source.json",
            "fit-batch",
            "fit-record",
            "fit-raw",
            {"observation": [1, 2, 3]},
        )
        unit["payload"] = {"observation": [9, 9, 9]}
        fit_path = self.root / "unbound-payload-fit.json"
        write_json(
            fit_path,
            {
                "schema_version": "csi-pairs-v6-rt-calibration-partition-v3",
                "partition": "fit",
                "units": [unit],
            },
        )
        with self.assertRaisesRegex(
            RuntimeError, "payload differs from its authenticated source record"
        ):
            read_rt_partition_contract(fit_path, "fit")

    def test_rt_calibration_allows_equal_measurements_from_distinct_raw_units(self):
        fit_unit = self._rt_unit(
            "fit-a",
            "fit-scene",
            "independent-fit-source.json",
            "fit-batch",
            "fit-record",
            "fit-raw",
            {"a": 1, "nested": {"x": 2, "y": 3}},
        )
        validation_unit = self._rt_unit(
            "validation-a",
            "validation-scene",
            "independent-validation-source.json",
            "validation-batch",
            "validation-record",
            "validation-raw",
            {"nested": {"y": 3, "x": 2}, "a": 1},
        )
        fit_path = self.root / "canonical-fit.json"
        validation_path = self.root / "canonical-validation.json"
        write_json(
            fit_path,
            {
                "schema_version": "csi-pairs-v6-rt-calibration-partition-v3",
                "partition": "fit",
                "units": [fit_unit],
            },
        )
        write_json(
            validation_path,
            {
                "schema_version": "csi-pairs-v6-rt-calibration-partition-v3",
                "partition": "validation",
                "units": [validation_unit],
            },
        )
        fit = read_rt_partition_contract(fit_path, "fit")
        validation = read_rt_partition_contract(validation_path, "validation")
        independence = validate_rt_partition_independence(
            fit, validation, {"validation-a"}
        )
        self.assertTrue(independence["verified"])

    def test_rt_calibration_rejects_same_logical_raw_unit_under_another_wrapper(self):
        fit_unit = self._rt_unit(
            "fit-a",
            "fit-scene",
            "wrapper-fit-source.json",
            "shared-batch",
            "shared-record",
            "shared-raw",
            {"observation": {"real": 1, "imag": 2}},
        )
        validation_unit = self._rt_unit(
            "validation-a",
            "validation-scene",
            "wrapper-validation-source.json",
            "shared-batch",
            "shared-record",
            "shared-raw",
            {"wrapped": {"observation": {"imag": 2, "real": 1}}},
        )
        fit_path = self.root / "wrapper-fit.json"
        validation_path = self.root / "wrapper-validation.json"
        write_json(
            fit_path,
            {
                "schema_version": "csi-pairs-v6-rt-calibration-partition-v3",
                "partition": "fit",
                "units": [fit_unit],
            },
        )
        write_json(
            validation_path,
            {
                "schema_version": "csi-pairs-v6-rt-calibration-partition-v3",
                "partition": "validation",
                "units": [validation_unit],
            },
        )
        fit = read_rt_partition_contract(fit_path, "fit")
        validation = read_rt_partition_contract(validation_path, "validation")
        with self.assertRaisesRegex(RuntimeError, "raw_unit_identity_overlap"):
            validate_rt_partition_independence(fit, validation, {"validation-a"})

    def test_literature_gate_rejects_unbound_or_contradictory_novelty_records(self):
        content = self.root / "paper.pdf"
        content.write_bytes(b"%PDF-1.4\n% test paper\n")
        from datetime import datetime, timezone
        from formal_v2.formal_evidence import _source_tree_sha256

        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        query = "map-conditioned CSI"
        receipt_specs = {
            "Crossref": (
                "https://api.crossref.org/works?query=map-conditioned%20CSI",
                {"message": {"items": [{"DOI": "10.1/test"}]}},
            ),
            "OpenAlex": (
                "https://api.openalex.org/works?search=map-conditioned%20CSI",
                {"results": [{"id": "https://openalex.org/W1"}]},
            ),
            "Semantic Scholar": (
                "https://api.semanticscholar.org/graph/v1/paper/search?query=map-conditioned%20CSI",
                {"data": [{"paperId": "paper-1"}]},
            ),
        }
        receipts = []
        for index, (database, (url, payload)) in enumerate(receipt_specs.items()):
            receipt_path = self.root / f"receipt-{index}.json"
            write_json(receipt_path, payload)
            receipts.append(
                {
                    "database": database,
                    "query": query,
                    "searched_utc": now,
                    "retrieval_url": url,
                    "result_count": 1,
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": sha256_file(receipt_path),
                }
            )
        review = self.root / "LLM_JUDGE_REVIEW.md"
        novelty_scope = "paired local geometry supervision"
        review.write_text(
            "# C13 llm-judge review\n\n"
            "- Judge family: codex\n"
            "- Judge identity: test-judge\n"
            "- Authorization basis: bound coding-agent session\n"
            f"- Review completed UTC: {now}\n"
            f"- Project dataset SHA-256: {sha256_file(self.dataset.source_path)}\n"
            f"- Project source-tree SHA-256: {_source_tree_sha256()}\n\n"
            "paper.pdf was reviewed record by record for relevance, license, and redistribution.\n\n"
            "- Licenses reviewed for every local PDF/source resource: true\n"
            "- No direct overlap with the frozen C13 claim: true\n"
            "- RT path ready: true\n"
            "- Map path ready: true\n"
            "- External-validity path ready: true\n"
            f"- Allowed novelty scope: {novelty_scope}\n"
            "- Conflicts or unresolved restrictions: none\n\n"
            f"{C13_JUDGE_ATTESTATION}\n\n"
            "- Judge signature or authenticated identity: codex:test-judge\n"
            f"- Signed UTC: {now}\n"
            + "Review detail.\n" * 20,
            encoding="utf-8",
        )
        manifest = {
            "schema_version": "csi-pairs-v6-literature-resource-manifest-v4",
            "search_completed_utc": now,
            "databases": list(self.config["literature"]["required_databases"]),
            "queries": [query],
            "search_receipts": receipts,
            "records": [
                {
                    "citation_key": "paper",
                    "title": "Paper",
                    "doi_or_url": "https://example.org/paper",
                    "verified_utc": now,
                    "content_path": str(content),
                    "content_sha256": sha256_file(content),
                    "relation_to_claim": "adjacent_nonoverlap",
                    "implementation_status": "integrated",
                }
            ],
            "resource_plan": {
                "gpu_hours": 1,
                "storage_gb": 1,
                "seed_count": 3,
                "failure_policy": "fail closed",
                "adapter_owners": ["research-team"],
            },
            "licenses_reviewed": True,
            "llm_judge_path": str(review),
            "llm_judge_sha256": sha256_file(review),
            "decision": {
                "no_direct_overlap": True,
                "rt_path_ready": True,
                "map_path_ready": True,
                "external_validity_path_ready": True,
                "novelty_scope": novelty_scope,
            },
        }
        validate_literature_manifest(self.config, manifest, self.root, self.dataset)
        review_bytes = review.read_bytes()
        review.write_bytes(review_bytes + b"tampered")
        with self.assertRaisesRegex(ValueError, "llm-judge review is missing or hash-mismatched"):
            validate_literature_manifest(self.config, manifest, self.root, self.dataset)
        review.write_bytes(review_bytes)
        original_url = manifest["search_receipts"][0]["retrieval_url"]
        manifest["search_receipts"][0]["retrieval_url"] = (
            "https://api.crossref.org/works?query=unrelated"
        )
        with self.assertRaisesRegex(ValueError, "bind the frozen query"):
            validate_literature_manifest(self.config, manifest, self.root, self.dataset)
        manifest["search_receipts"][0]["retrieval_url"] = original_url
        manifest["decision"]["no_direct_overlap"] = False
        with self.assertRaisesRegex(ValueError, "contradicts"):
            validate_literature_manifest(self.config, manifest, self.root, self.dataset)
        manifest["decision"]["no_direct_overlap"] = True
        stage = self.root / "literature-stage"
        stage.mkdir()
        bound = stage / "literature_manifest.json"
        write_json(bound, manifest)
        payload = {
            "input_manifest_path": bound.name,
            "input_manifest_sha256": sha256_file(bound),
            "llm_judge_path": str(review.resolve()),
            "llm_judge_sha256": sha256_file(review),
            "llm_judge": "codex:test-judge",
            "llm_judge_family": "codex",
            "llm_judge_completed_utc": now,
            "llm_judge_signature": "codex:test-judge",
            "llm_judge_signed_utc": now,
        }
        write_json(
            stage / "manifest.json",
            {"files": [{"path": bound.name, "sha256": sha256_file(bound)}]},
        )
        _validate_stage_bound_input(stage / "gate.json", payload, self.config, "G0")
        content.write_bytes(b"changed paper")
        with self.assertRaisesRegex(ValueError, "hash-mismatched"):
            _validate_stage_bound_input(stage / "gate.json", payload, self.config, "G0")

    def test_cli_exposes_all_fail_closed_stages(self):
        help_text = build_parser().format_help()
        for command in (
            "prepare-full-run",
            "run-evaluation",
            "run-risk",
            "run-path",
            "run-external-baselines",
            "verify-data",
            "run-resource-controls",
            "run-scene-id-audit",
            "run-external-validity",
            "run-literature-resources",
            "run-rt-calibration",
            "run-shuffled-pair-control",
            "run-retention-audit",
            "assemble-claims",
        ):
            self.assertIn(command, help_text)
        parser = build_parser()
        full_argv = [
            "all",
            "--config",
            "config.json",
            "--output",
            "output",
            "--verifier-manifest",
            "verifier.json",
            "--adapter-manifest",
            "adapters.json",
            "--external-validity-manifest",
            "external-validity.json",
            "--literature-resource-manifest",
            "literature.json",
            "--rt-calibration-manifest",
            "rt-calibration.json",
        ]
        with patch("builtins.print") as denied_message:
            self.assertEqual(formal_cli_main(full_argv), 2)
        self.assertIn("--approval-manifest", denied_message.call_args.args[0])
        full_args = parser.parse_args(
            [*full_argv, "--approve-full-experiment"]
        )
        self.assertTrue(full_args.approve_full_experiment)
        qualification = parser.parse_args(
            [
                "qualify",
                "--config",
                "config.json",
                "--output",
                "output",
                "--resume",
            ]
        )
        self.assertTrue(qualification.resume)

    def test_cli_qualify_resume_refuses_a_completed_gate_without_overwrite(self):
        output = self.root / "completed-qualification"
        fixture = write_nonscientific_fixture(self.root / "resume-cli-fixture.npz")
        gate = output / "qualification" / "gate.json"
        gate.parent.mkdir(parents=True)
        gate.write_bytes(b"completed gate bytes")
        with patch("builtins.print") as error_message:
            status = formal_cli_main(
                [
                    "qualify",
                    "--config",
                    str(SMOKE_CONFIG),
                    "--dataset",
                    str(fixture),
                    "--output",
                    str(output),
                    "--resume",
                ]
            )
        self.assertEqual(status, 2)
        self.assertIn("refuses a completed qualification", error_message.call_args.args[0])
        self.assertEqual(gate.read_bytes(), b"completed gate bytes")

    def test_cli_ships_first_party_claim_and_scene_id_defaults(self):
        parser = build_parser()
        common = ["--config", str(SMOKE_CONFIG), "--output", str(self.root / "run")]
        shuffled = parser.parse_args(["run-shuffled-pair-control", *common])
        retention = parser.parse_args(["run-retention-audit", *common])
        scene_id = parser.parse_args(["run-scene-id-audit", *common])
        resource = parser.parse_args(["run-resource-controls", *common])
        self.assertEqual(shuffled.claim_control_manifest, DEFAULT_SHUFFLED_PAIR_MANIFEST)
        self.assertEqual(retention.claim_control_manifest, DEFAULT_RETENTION_MANIFEST)
        self.assertEqual(scene_id.scene_id_manifest, DEFAULT_SCENE_ID_MANIFEST)
        self.assertEqual(resource.control_manifest, DEFAULT_RESOURCE_CONTROL_MANIFEST)


class WiGATrAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / "fixture.npz"
        write_nonscientific_fixture(self.path)
        self.dataset = FormalDataset.load(self.path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_paper_dose_config_and_vendor_snapshot_are_frozen(self):
        config = load_wigatr_config(
            ROOT / "formal_v2/configs/wigatr_official_v1.json"
        )
        self.assertEqual(config["model"]["num_blocks"], 32)
        self.assertEqual(config["model"]["hidden_mv_channels"], 16)
        self.assertEqual(config["model"]["hidden_s_channels"], 32)
        self.assertEqual(config["model"]["num_heads"], 8)
        self.assertEqual(config["training"]["steps"], 200000)
        self.assertEqual(config["training"]["batch_size"], 64)
        self.assertEqual(config["training"]["microbatch_size"], 16)
        self.assertEqual(
            config["mesh"]["preprocessing"],
            "surface-ledger-equivalent-rectangle-compaction-v1",
        )
        vendor = ROOT / "formal_v2/external_adapters/vendor/Wi-GATr"
        _verify_vendor_tree(vendor)
        self.assertTrue((vendor / "LICENSE").is_file())

    def test_official_wigatr_fails_before_training_without_cuda_attention(self):
        unavailable = {"torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))}
        with self.assertRaisesRegex(RuntimeError, "requires an NVIDIA CUDA device"):
            _require_official_cuda(unavailable)
        available = {"torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))}
        _require_official_cuda(available)

    def test_relative_total_power_is_invariant_to_common_phase(self):
        real = np.asarray([1.2, -0.4, 0.7, 2.1])
        imaginary = np.asarray([-0.3, 0.8, 1.1, -0.6])
        angle = 0.71
        rotated_real = real * np.cos(angle) - imaginary * np.sin(angle)
        rotated_imaginary = real * np.sin(angle) + imaginary * np.cos(angle)
        original = relative_total_power_db(
            np.concatenate((real, imaginary)), 1e-12
        )
        rotated = relative_total_power_db(
            np.concatenate((rotated_real, rotated_imaginary)), 1e-12
        )
        self.assertAlmostEqual(float(original), float(rotated), places=12)

    def test_2p5d_mesh_has_top_exposed_sides_and_materials(self):
        maps = np.zeros((3, 3, 3), dtype=np.float64)
        maps[0, 1, 1] = 1.0
        maps[1, 1, 1] = 2.0
        maps[2, 1, 1] = 3.0
        mesh, materials = grid_to_triangular_mesh(
            maps,
            ("occupancy", "height", "material"),
            resolution_m=0.5,
            origin_xy_m=(-0.75, -0.75),
            occupancy_threshold=0.5,
            minimum_height_m=0.001,
        )
        self.assertEqual(mesh.shape, (10, 3, 3))
        self.assertTrue(np.all(materials == 3))
        self.assertAlmostEqual(float(mesh[..., 2].min()), 0.0)
        self.assertAlmostEqual(float(mesh[..., 2].max()), 2.0)
        empty_mesh, empty_materials = grid_to_triangular_mesh(
            np.zeros_like(maps),
            ("occupancy", "height", "material"),
            resolution_m=0.5,
            origin_xy_m=(-0.75, -0.75),
            occupancy_threshold=0.5,
            minimum_height_m=0.001,
        )
        self.assertEqual(empty_mesh.shape, (0, 3, 3))
        self.assertEqual(empty_materials.shape, (0,))

    def test_compact_mesh_preserves_surface_area_and_material(self):
        maps = np.zeros((3, 4, 4), dtype=np.float64)
        maps[0, 1:3, 1:3] = 1.0
        maps[1, 1:3, 1:3] = 2.0
        maps[2, 1:3, 1:3] = 3.0
        arguments = {
            "resolution_m": 0.5,
            "origin_xy_m": (-1.0, -1.0),
            "occupancy_threshold": 0.5,
            "minimum_height_m": 0.001,
        }
        reference, reference_materials = grid_to_cellwise_triangular_mesh(
            maps, ("occupancy", "height", "material"), **arguments
        )
        compact, compact_materials = grid_to_triangular_mesh(
            maps, ("occupancy", "height", "material"), **arguments
        )

        def area(mesh):
            return 0.5 * np.linalg.norm(
                np.cross(mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0]),
                axis=1,
            ).sum()

        self.assertLess(len(compact), len(reference))
        self.assertAlmostEqual(float(area(compact)), float(area(reference)), places=6)
        self.assertTrue(np.all(reference_materials == 3))
        self.assertTrue(np.all(compact_materials == 3))
        require_surface_ledger_equivalence(
            reference,
            reference_materials,
            compact,
            compact_materials,
            resolution_m=arguments["resolution_m"],
            origin_xy_m=arguments["origin_xy_m"],
        )
        digest, atomic_quads = require_compact_mesh_matches_map_surface(
            maps,
            ("occupancy", "height", "material"),
            compact,
            compact_materials,
            **arguments,
        )
        self.assertEqual(len(digest), 64)
        self.assertEqual(atomic_quads * 2, len(reference))
        altered_materials = compact_materials.copy()
        altered_materials[:2] += 1
        with self.assertRaisesRegex(RuntimeError, "canonical surface ledger"):
            require_surface_ledger_equivalence(
                reference,
                reference_materials,
                compact,
                altered_materials,
                resolution_m=arguments["resolution_m"],
                origin_xy_m=arguments["origin_xy_m"],
            )
        with self.assertRaisesRegex(RuntimeError, "map surface"):
            require_compact_mesh_matches_map_surface(
                maps,
                ("occupancy", "height", "material"),
                compact,
                altered_materials,
                **arguments,
            )

    def test_wigatr_mesh_audit_covers_the_complete_fixture_map_bank(self):
        config = load_wigatr_config(ROOT / "formal_v2/configs/wigatr_official_v1.json")
        audit = _audit_mesh_preprocessing(self.dataset, config)
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(
            audit["map_count"], self.dataset.scene_count * self.dataset.world_count
        )
        self.assertEqual(
            sum(audit["role_map_counts"].values()), audit["map_count"]
        )
        self.assertLessEqual(
            audit["compact_face_count"], audit["reference_face_count"]
        )

    def test_geometry_destroyed_preserves_joint_cell_statistics(self):
        maps = np.arange(3 * 4 * 4, dtype=np.float64).reshape(3, 4, 4)
        first = geometry_destroyed_map(maps, "fixed-unit")
        second = geometry_destroyed_map(maps, "fixed-unit")
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, maps))
        original_cells = sorted(map(tuple, maps.reshape(3, -1).T.tolist()))
        destroyed_cells = sorted(map(tuple, first.reshape(3, -1).T.tolist()))
        self.assertEqual(original_cells, destroyed_cells)

    def test_common_unit_registry_excludes_target_support(self):
        routes = {}
        scenes = np.concatenate(
            (
                self.dataset.indices_for_role("source_final_unseen_bank"),
                self.dataset.indices_for_role("target"),
            )
        )
        for scene_value in scenes:
            scene = int(scene_value)
            for edge in self.dataset.directed_edges(scene):
                for position in range(self.dataset.position_count):
                    routes[(scene, edge.source_world, edge.target_world, position)] = (
                        2 if edge.bit_index == 0 else 0
                    )
        units = build_six_condition_units(
            self.dataset, SimpleNamespace(alignment_route=routes)
        )
        self.assertGreater(len(units), 0)
        for unit in units:
            if str(self.dataset.scene_roles[unit.scene]) == "target":
                self.assertEqual(
                    str(self.dataset.position_roles[unit.scene, unit.position]),
                    "query",
                )
        unit = units[0]
        maps = [condition_map(self.dataset, unit, name) for name in SIX_CONDITIONS]
        self.assertEqual(len(maps), 6)
        self.assertTrue(np.array_equal(maps[-1], np.zeros_like(maps[-1])))

    def test_external_rows_must_cover_the_frozen_registry(self):
        scene = int(self.dataset.indices_for_role("target")[0])
        position = int(np.flatnonzero(self.dataset.position_roles[scene] == "query")[0])
        unit_id = "registered-unit"
        rows = []
        condition_contract = {}
        for index, condition in enumerate(SIX_CONDITIONS):
            map_sha256 = f"{index + 1:064x}"
            action_sha256 = f"{index + 101:064x}"
            row = {
                "unit_id": unit_id,
                "model_name": "Wi-GATr",
                "condition": condition,
                "city_id": str(self.dataset.city_ids[scene]),
                "bank_id": str(self.dataset.bank_ids[scene]),
                "position_id": str(self.dataset.position_ids[scene, position]),
                "localization_error_m": "1.0",
                "csi_context_sha256": "a" * 64,
                "base_map_cluster_id": str(self.dataset.base_map_cluster_ids[scene]),
                "map_sha256": map_sha256,
                "action_sha256": action_sha256,
                "query_count": "1",
            }
            rows.append(row)
            condition_contract[(unit_id, condition)] = {
                "base_map_cluster_id": str(self.dataset.base_map_cluster_ids[scene]),
                "map_sha256": map_sha256,
                "action_sha256": action_sha256,
            }
        _validate_six_condition_rows(
            {"model_name": "Wi-GATr"},
            rows,
            self.dataset,
            expected_unit_ids={unit_id},
            expected_unit_contract={
                unit_id: SimpleNamespace(
                    scene=scene,
                    position=position,
                    csi_context_sha256="a" * 64,
                )
            },
            expected_condition_contract=condition_contract,
        )
        with self.assertRaisesRegex(ValueError, "common unit registry"):
            _validate_six_condition_rows(
                {"model_name": "Wi-GATr"},
                rows,
                self.dataset,
                expected_unit_ids={unit_id, "missing-unit"},
            )
        rows[0]["csi_context_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "frozen csi_context_sha256"):
            _validate_six_condition_rows(
                {"model_name": "Wi-GATr"},
                rows,
                self.dataset,
                expected_unit_ids={unit_id},
                expected_unit_contract={
                    unit_id: SimpleNamespace(
                        scene=scene,
                        position=position,
                        csi_context_sha256="a" * 64,
                    )
                },
            )
        rows[0]["csi_context_sha256"] = "a" * 64
        support = int(
            np.flatnonzero(self.dataset.position_roles[scene] == "support_pool")[0]
        )
        rows[0]["position_id"] = str(self.dataset.position_ids[scene, support])
        with self.assertRaisesRegex(ValueError, "support_pool"):
            _validate_six_condition_rows(
                {"model_name": "Wi-GATr"}, rows, self.dataset
            )

    def test_execution_manifest_binds_config_training_checkpoint_and_results(self):
        output = self.root / "adapter"
        output.mkdir()
        config_path = output / "adapter_config.json"
        training_path = output / "training_record.json"
        checkpoint_path = output / "checkpoint.pt"
        result_path = output / "six_condition_results.csv"
        write_json(config_path, {"schema_version": "adapter-config"})
        write_json(
            training_path,
            {
                "train_role": "source_encoder_train",
                "selection_role": "source_method_selection",
                "target_roles_read": [],
            },
        )
        checkpoint_path.write_bytes(b"checkpoint")
        write_csv(result_path, [{"result": 1}])
        command = ["python", "adapter.py"]
        adapter = {
            "adapter_id": "wigatr",
            "model_name": "Wi-GATr",
            "implementation_status": "official-code-adaptation",
            "source_revision": "revision",
            "command": command,
            "adapter_config_sha256": sha256_file(config_path),
        }
        execution = {
            "schema_version": "csi-pairs-v6-external-execution-v2",
            "adapter_id": "wigatr",
            "model_name": "Wi-GATr",
            "implementation_status": "official-code-adaptation",
            "source_revision": "revision",
            "dataset_sha256": sha256_file(self.path),
            "adapter_config_path": config_path.name,
            "adapter_config_sha256": sha256_file(config_path),
            "training_record_path": training_path.name,
            "training_record_sha256": sha256_file(training_path),
            "checkpoint_path": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "command_sha256": hashlib.sha256(
                json.dumps(command, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "results_sha256": sha256_file(result_path),
        }
        write_json(output / "execution_manifest.json", execution)
        validate_external_execution_manifest(
            adapter, output, result_path, self.dataset
        )
        substituted = dict(execution)
        substituted["adapter_config_sha256"] = "b" * 64
        write_json(output / "execution_manifest.json", substituted)
        with self.assertRaisesRegex(RuntimeError, "outer frozen hash"):
            validate_external_execution_manifest(
                adapter, output, result_path, self.dataset
            )
        write_json(config_path, {"schema_version": "substituted-adapter-config"})
        substituted["adapter_config_sha256"] = sha256_file(config_path)
        write_json(output / "execution_manifest.json", substituted)
        with self.assertRaisesRegex(RuntimeError, "outer frozen hash"):
            validate_external_execution_manifest(
                adapter, output, result_path, self.dataset
            )
        write_json(config_path, {"schema_version": "adapter-config"})
        write_json(output / "execution_manifest.json", execution)
        training = json.loads(training_path.read_text())
        training["target_roles_read"] = ["target"]
        write_json(training_path, training)
        execution["training_record_sha256"] = sha256_file(training_path)
        write_json(output / "execution_manifest.json", execution)
        with self.assertRaisesRegex(RuntimeError, "source-only"):
            validate_external_execution_manifest(
                adapter, output, result_path, self.dataset
            )

    def test_wigatr_execution_manifest_binds_independently_probed_runtime(self):
        output = self.root / "wigatr-runtime-adapter"
        output.mkdir()
        config_path = output / "adapter_config.json"
        training_path = output / "training_record.json"
        checkpoint_path = output / "checkpoint.pt"
        result_path = output / "six_condition_results.csv"
        runtime_path = output / "runtime_provenance.json"
        mesh_audit_path = output / "mesh_surface_audit.json"
        write_json(config_path, {"schema_version": "adapter-config"})
        checkpoint_path.write_bytes(b"checkpoint")
        write_csv(result_path, [{"result": 1}])
        runtime = {"environment_sha256": "a" * 64}
        write_json(runtime_path, runtime)
        role_map_counts = {
            str(role): int(np.sum(self.dataset.scene_roles == role))
            * self.dataset.world_count
            for role in sorted(set(str(value) for value in self.dataset.scene_roles))
        }
        mesh_audit = {
            "schema_version": "csi-pairs-v6-wigatr-mesh-surface-audit-v1",
            "status": "PASS",
            "passed": True,
            "dataset_sha256": sha256_file(self.path),
            "preprocessing": "surface-ledger-equivalent-rectangle-compaction-v1",
            "scene_count": self.dataset.scene_count,
            "world_count": self.dataset.world_count,
            "map_count": self.dataset.scene_count * self.dataset.world_count,
            "role_map_counts": role_map_counts,
            "reference_face_count": 4,
            "compact_face_count": 2,
            "canonical_surface_ledger_sha256": "c" * 64,
            "rule": "complete fixture map bank exact canonical ledger comparison",
        }
        write_json(mesh_audit_path, mesh_audit)
        write_json(
            training_path,
            {
                "train_role": "source_encoder_train",
                "selection_role": "source_method_selection",
                "target_roles_read": [],
                "mesh_surface_audit_path": mesh_audit_path.name,
                "mesh_surface_audit_sha256": sha256_file(mesh_audit_path),
            },
        )
        command = [
            "{project_root}/formal_v2/external_adapters/.venv-wigatr/bin/python",
            "adapter.py",
        ]
        adapter = {
            "adapter_id": "wigatr",
            "model_name": "Wi-GATr",
            "implementation_status": "official-code-adaptation",
            "source_revision": "revision",
            "command": command,
            "adapter_config_sha256": sha256_file(config_path),
        }
        execution = {
            "schema_version": "csi-pairs-v6-external-execution-v4",
            "adapter_id": "wigatr",
            "model_name": "Wi-GATr",
            "implementation_status": "official-code-adaptation",
            "source_revision": "revision",
            "dataset_sha256": sha256_file(self.path),
            "adapter_config_path": config_path.name,
            "adapter_config_sha256": sha256_file(config_path),
            "training_record_path": training_path.name,
            "training_record_sha256": sha256_file(training_path),
            "checkpoint_path": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "command_sha256": hashlib.sha256(
                json.dumps(command, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "results_sha256": sha256_file(result_path),
            "runtime_provenance_path": runtime_path.name,
            "runtime_provenance_sha256": sha256_file(runtime_path),
            "runtime_environment_sha256": runtime["environment_sha256"],
            "mesh_surface_audit_path": mesh_audit_path.name,
            "mesh_surface_audit_sha256": sha256_file(mesh_audit_path),
        }
        write_json(output / "execution_manifest.json", execution)
        with (
            patch(
                "formal_v2.formal_external_runtime.validate_external_runtime",
                return_value=runtime,
            ),
            patch(
                "formal_v2.formal_external_runtime.probe_external_runtime",
                return_value=runtime,
            ),
        ):
            validate_external_execution_manifest(
                adapter, output, result_path, self.dataset
            )
            tampered_audit = dict(mesh_audit)
            tampered_audit["map_count"] -= 1
            write_json(mesh_audit_path, tampered_audit)
            training = read_strict_json(training_path)
            training["mesh_surface_audit_sha256"] = sha256_file(mesh_audit_path)
            write_json(training_path, training)
            execution["mesh_surface_audit_sha256"] = sha256_file(mesh_audit_path)
            execution["training_record_sha256"] = sha256_file(training_path)
            write_json(output / "execution_manifest.json", execution)
            with self.assertRaisesRegex(RuntimeError, "audit contract"):
                validate_external_execution_manifest(
                    adapter, output, result_path, self.dataset
                )
            write_json(mesh_audit_path, mesh_audit)
            training["mesh_surface_audit_sha256"] = sha256_file(mesh_audit_path)
            write_json(training_path, training)
            execution["mesh_surface_audit_sha256"] = sha256_file(mesh_audit_path)
            execution["training_record_sha256"] = sha256_file(training_path)
            write_json(output / "execution_manifest.json", execution)
            substituted = {"environment_sha256": "b" * 64}
            write_json(runtime_path, substituted)
            execution["runtime_provenance_sha256"] = sha256_file(runtime_path)
            execution["runtime_environment_sha256"] = substituted[
                "environment_sha256"
            ]
            write_json(output / "execution_manifest.json", execution)
            with self.assertRaisesRegex(RuntimeError, "independently probed"):
                validate_external_execution_manifest(
                    adapter, output, result_path, self.dataset
                )

    def test_inverse_localizer_api_has_no_true_position_argument(self):
        parameters = inspect.signature(inverse_localize_power).parameters
        self.assertNotIn("true_position", parameters)

    def test_inverse_localizer_optimizes_only_from_public_bounds_and_power(self):
        class ToyBatch:
            @classmethod
            def from_data_list(cls, _values):
                return cls()

            def to(self, _device):
                return self

        class ToyPowerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, _batch, overrides):
                receiver = overrides["rx"]
                value = receiver[0] + 0.5 * receiver[1] + 0.0 * self.anchor
                return value.reshape(1, 1)

        runtime = {
            "torch": torch,
            "Batch": ToyBatch,
            "tokenize_scene": lambda *_args, **_kwargs: object(),
        }
        config = load_wigatr_config(
            ROOT / "formal_v2/configs/wigatr_official_v1.json"
        )
        config["inverse"]["steps"] = 80
        config["inverse"]["restarts"] = 3
        prediction = inverse_localize_power(
            runtime,
            ToyPowerModel(),
            self.dataset,
            self.dataset.maps[0, 0],
            self.dataset.bs_pose[0, :3],
            observed_power=1.25,
            bounds=((-4.0, 4.0), (-4.0, 4.0)),
            config=config,
            num_materials=int(
                self.dataset.metadata["assets"]["material_category_count"]
            ),
            restart_salt="test-without-target-position",
        )
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertTrue(np.all(prediction >= -4.0))
        self.assertTrue(np.all(prediction <= 4.0))
        self.assertLess(
            abs(float(prediction[0] + 0.5 * prediction[1]) - 1.25), 0.05
        )


class WaibuIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "fixture.npz"
        write_nonscientific_fixture(self.path)
        self.dataset = FormalDataset.load(self.path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_every_waibu_resource_has_frozen_authentication_metadata(self):
        registry = parse_strict_json(
            (ROOT / "formal_v2/configs/waibu_resources_v1.json").read_text()
        )
        rows = validate_resource_registry_structure(registry)
        self.assertEqual(len(rows), 11)
        self.assertTrue(all(len(row["sha256"]) == 64 for row in rows))
        self.assertEqual(
            {
                row["file"]
                for row in rows
                if not row["redistribution_allowed"]
            },
            {
                "2502.11965v2.pdf",
                "2505.09160v2.pdf",
                "2601.03789v1.pdf",
                "2604.07086v1.pdf",
            },
        )

    def test_representation_registry_preserves_paper_labels_and_roles(self):
        config = load_representation_config(
            ROOT / "formal_v2/configs/representation_baselines_v1.json"
        )
        self.assertEqual(
            [row["model_name"] for row in config["models"]],
            ["CSI-MAE", "CSI-CLIP", "CSI-CLIP++", "ContraWiMAE", "WWM"],
        )
        self.assertEqual(config["source_roles"]["pretrain"], "source_encoder_train")
        self.assertEqual(config["source_roles"]["selection"], "source_method_selection")

    def test_csi_mae_controlled_model_runs_masked_backward(self):
        config = load_representation_config(
            ROOT / "formal_v2/configs/representation_baselines_smoke_v1.json"
        )["models"][0]
        spec = PatchSpec.from_metadata(self.dataset.metadata)
        model = build_representation_model(
            "CSI-MAE",
            spec,
            self.dataset.maps.shape[2],
            self.dataset.radio_config.shape[1] + 9,
            config,
        )
        scene = int(self.dataset.indices_for_role("source_encoder_train")[0])
        normalizer = fit_representation_normalizer(
            self.dataset, self.dataset.indices_for_role("source_encoder_train")
        )
        batch = make_representation_batch(
            self.dataset,
            [(scene, 0, 0), (scene, 1, 1)],
            normalizer,
            torch.device("cpu"),
        )
        loss = model.pretraining_loss(batch, config["mask_fraction"])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.encode(batch).shape, (2, config["dim"]))
        self.assertIsInstance(model, CSIMAE)
        self.assertNotIn("fixed_position", dict(model.encoder.named_parameters()))
        self.assertEqual(model.decoder_head.in_features, config["decoder_dim"])
        torch.manual_seed(17)
        masks = model._mask(16, torch.device("cpu"), config["mask_fraction"])
        self.assertEqual(masks.shape, (16, spec.patch_count))
        self.assertTrue(torch.all(masks.sum(dim=1) == round(config["mask_fraction"] * spec.patch_count)))
        self.assertGreater(torch.unique(masks, dim=0).shape[0], 1)

    def test_all_representation_models_run_backward_with_real_gradients(self):
        rows = load_representation_config(
            ROOT / "formal_v2/configs/representation_baselines_v1.json"
        )["models"]
        spec = PatchSpec.from_metadata(self.dataset.metadata)
        normalizer = fit_representation_normalizer(
            self.dataset, self.dataset.indices_for_role("source_encoder_train")
        )
        scene = int(self.dataset.indices_for_role("source_encoder_train")[0])
        batch = make_representation_batch(
            self.dataset,
            [(scene, 0, 0), (scene, 1, 1)],
            normalizer,
            torch.device("cpu"),
        )
        for source in rows:
            with self.subTest(model=source["model_name"]):
                row = dict(source)
                row.update(
                    dim=8,
                    heads=2,
                    encoder_layers=1,
                    decoder_layers=1,
                    decoder_dim=8,
                    resnet_width=2,
                )
                model = build_representation_model(
                    row["model_name"],
                    spec,
                    self.dataset.maps.shape[2],
                    self.dataset.radio_config.shape[1] + self.dataset.bs_pose.shape[1] + 2,
                    row,
                )
                loss = model.pretraining_loss(
                    batch, min(float(row["mask_fraction"]), 0.75)
                )
                loss.backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(gradients)
                self.assertTrue(any(torch.any(gradient != 0) for gradient in gradients))
                self.assertEqual(model.encode(batch).shape, (2, 8))

    def test_formal_contrawimae_requires_wimae_warm_start(self):
        config = parse_strict_json(
            (ROOT / "formal_v2/configs/representation_baselines_v1.json").read_text()
        )
        contra = next(row for row in config["models"] if row["model_name"] == "ContraWiMAE")
        self.assertEqual(contra["warm_start_epochs"], 3000)
        contra["warm_start_epochs"] = 0
        temporary = Path(self.temporary.name) / "bad-representation.json"
        write_json(temporary, config)
        with self.assertRaisesRegex(ValueError, "warm-start"):
            load_representation_config(temporary)

    def test_contrawimae_executes_source_only_wimae_warm_start(self):
        config = load_representation_config(
            ROOT / "formal_v2/configs/representation_baselines_v1.json"
        )
        row = dict(next(value for value in config["models"] if value["model_name"] == "ContraWiMAE"))
        row.update(
            dim=8,
            heads=2,
            encoder_layers=1,
            decoder_layers=1,
            decoder_dim=8,
            batch_size=2,
            warmup_epochs=0,
            warm_start_epochs=1,
        )
        spec = PatchSpec.from_metadata(self.dataset.metadata)
        model = build_representation_model(
            "ContraWiMAE",
            spec,
            self.dataset.maps.shape[2],
            self.dataset.radio_config.shape[1] + self.dataset.bs_pose.shape[1] + 2,
            row,
        )
        normalizer = fit_representation_normalizer(
            self.dataset, self.dataset.indices_for_role("source_encoder_train")
        )
        train_scene = int(self.dataset.indices_for_role("source_encoder_train")[0])
        selection_scene = int(self.dataset.indices_for_role("source_method_selection")[0])
        record = _warm_start_contra(
            model,
            "ContraWiMAE",
            row,
            self.dataset,
            normalizer,
            [(train_scene, 0, 0), (train_scene, 1, 1)],
            [(selection_scene, 0, 0), (selection_scene, 1, 1)],
            torch.device("cpu"),
            113,
        )
        self.assertTrue(record["required"])
        self.assertTrue(record["completed"])
        self.assertEqual(record["selected_epoch"], 1)
        self.assertEqual(record["target_roles_read"], [])

    def test_all_controlled_map_models_have_real_gradients(self):
        scene = int(self.dataset.indices_for_role("source_encoder_train")[0])
        normalizer = _fit_data_normalizer(self.dataset)
        files = (
            "controlled_map_smoke_v1.json",
            "wiser_controlled_v1.json",
            "rfir_controlled_v1.json",
        )
        for file_name in files:
            config = load_controlled_map_config(ROOT / "formal_v2/configs" / file_name)
            config["model"].update(
                {
                    "hidden_dim": 16,
                    "scene_dim": 16,
                    "heads": 4,
                    "layers": 1,
                    "grid_size": 4,
                    "corridor_tokens": 4,
                    "tap_count": 2,
                    "maximum_primitives": 16,
                }
            )
            model, _ = build_controlled_model(config, self.dataset)
            if config["method"] == "wiser":
                self.assertIsInstance(model.cir.decoder, torch.nn.TransformerDecoder)
            if config["method"] == "rfir":
                self.assertTrue(hasattr(model, "material_opacity"))
                start = torch.zeros(2, 3)
                end = torch.ones(2, 1, 3)
                origin = torch.zeros(2, 2)
                clear = model._segment_visibility(torch.zeros(2, 1, 4, 4), start, end, origin, 0.25)
                blocked = model._segment_visibility(torch.ones(2, 1, 4, 4), start, end, origin, 0.25)
                self.assertTrue(torch.all(clear > blocked))
            batch = controlled_batch(
                self.dataset,
                [(scene, 0, 0), (scene, 1, 1)],
                normalizer,
                torch.device("cpu"),
                config["method"],
            )
            loss = controlled_loss(model, config["method"], batch)
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
            self.assertTrue(torch.isfinite(loss), config["method"])
            self.assertTrue(
                any(
                    value is not None and torch.any(value != 0)
                    for value in gradients
                ),
                config["method"],
            )
            if config["method"] in {"wiser", "rfir"}:
                with torch.no_grad():
                    first = model(
                        batch["maps"], batch["tx"], batch["context"], batch["receiver"],
                        batch["origin"], batch["resolution"],
                    )
                    second = model(
                        batch["maps"], batch["tx"], batch["context"] + 1.0, batch["receiver"],
                        batch["origin"], batch["resolution"],
                    )
                first_tensor = first[1] if isinstance(first, tuple) else first
                second_tensor = second[1] if isinstance(second, tuple) else second
                self.assertFalse(torch.equal(first_tensor, second_tensor), config["method"])
        self.assertEqual(_training_task("wiser", 1, 100, 5), "radiomap")
        self.assertEqual(_training_task("wiser", 11, 100, 5), "cir")
        self.assertIn(_training_task("wiser", 25, 100, 5), {"radiomap", "cir"})

    def test_controlled_map_memory_and_precision_contract_is_frozen(self):
        expectations = {
            "sigmap_controlled_v1.json": (128, 16, "bf16"),
            "wiser_controlled_v1.json": (64, 16, "float32"),
            "rfir_controlled_v1.json": (32, 16, "float32"),
        }
        for name, expected in expectations.items():
            with self.subTest(config=name):
                config = load_controlled_map_config(ROOT / "formal_v2/configs" / name)
                observed = (
                    config["training"]["batch_size"],
                    config["training"]["microbatch_size"],
                    config["training"]["precision"],
                )
                self.assertEqual(observed, expected)

    def test_controlled_map_train_records_effective_batch_accumulation(self):
        config = load_controlled_map_config(
            ROOT / "formal_v2/configs/controlled_map_smoke_v1.json"
        )
        config["training"].update(steps=1, selection_every_steps=1)
        model, metadata = build_controlled_model(config, self.dataset)
        normalizer = _fit_data_normalizer(self.dataset)
        train_scene = int(self.dataset.indices_for_role("source_encoder_train")[0])
        selection_scene = int(
            self.dataset.indices_for_role("source_method_selection")[0]
        )
        units = {
            "source_encoder_train": [
                (train_scene, 0, position)
                for position in range(config["training"]["batch_size"])
            ],
            "source_method_selection": [
                (selection_scene, 0, position)
                for position in range(config["training"]["microbatch_size"])
            ],
        }
        output = self.path.parent / "controlled-train"
        output.mkdir()
        with patch(
            "formal_v2.external_adapters.controlled_map_adapter._units",
            side_effect=lambda _dataset, role: units[role],
        ):
            checkpoint, record = train_controlled_model(
                model,
                config,
                self.dataset,
                normalizer,
                output,
                metadata,
            )
        self.assertTrue(checkpoint.is_file())
        self.assertEqual(record["batch_size"], 4)
        self.assertEqual(record["microbatch_size"], 2)
        self.assertEqual(record["gradient_accumulation_steps"], 2)
        self.assertEqual(record["configured_precision"], "float32")
        self.assertEqual(record["executed_precision"], "float32")
        self.assertFalse(record["autocast_enabled"])
        self.assertEqual(record["target_roles_read"], [])

    def test_deterministic_prefix_product_matches_forward_and_gradient(self):
        values = torch.tensor(
            [[0.91, 0.83, 0.77, 0.69, 0.61]],
            dtype=torch.float64,
            requires_grad=True,
        )
        reference_values = values.detach().clone().requires_grad_(True)
        weights = torch.arange(1, 6, dtype=torch.float64).reshape(1, -1)
        candidate = deterministic_prefix_product(values, dim=1)
        reference = torch.cumprod(reference_values, dim=1)
        candidate_gradient = torch.autograd.grad((candidate * weights).sum(), values)[0]
        reference_gradient = torch.autograd.grad(
            (reference * weights).sum(), reference_values
        )[0]
        self.assertTrue(torch.allclose(candidate, reference, rtol=1e-12, atol=1e-12))
        self.assertTrue(
            torch.allclose(
                candidate_gradient,
                reference_gradient,
                rtol=1e-12,
                atol=1e-12,
            )
        )
    def test_complete_map_manifest_has_five_distinct_map_models(self):
        manifest = parse_strict_json(
            (ROOT / "formal_v2/external_adapters/all_map_adapters_v1.json").read_text()
        )
        validate_external_manifest(manifest)
        self.assertEqual(
            {row["model_name"] for row in manifest["adapters"]},
            {"SigMap", "Wi-GATr", "PMNet", "WiSER", "RFIR"},
        )
        eligible = [row for row in manifest["adapters"] if row["c1_eligible"]]
        self.assertEqual(
            [row["model_name"] for row in eligible], ["Wi-GATr", "PMNet"]
        )
        changed = json.loads(json.dumps(manifest))
        changed["adapters"][0]["c1_eligible"] = True
        with self.assertRaisesRegex(ValueError, "style-controlled"):
            validate_external_manifest(changed)

    def test_c1_claim_requires_two_explicitly_eligible_external_models(self):
        base = {
            "status": "BLOCKED",
            "passed": False,
            "unique_passing_model_count": 4,
            "c1_eligible_model_count": 1,
            "c1_eligible_models": ["Wi-GATr"],
            "c1_required_eligible_model_count": 2,
            "c1_city_gate_contract": "all-evaluation-cities-must-pass-v1",
            "model_assessments": [
                {
                    "model_name": "Wi-GATr",
                    "c1_eligible": True,
                    "passed": True,
                    "active_effect_passed": True,
                    "null_safety_passed": True,
                    "base_map_cluster_count": 2,
                    "null_overclassification_rate_ci95_high": 0.01,
                    "null_overclassification_rate_max": 0.05,
                    "aggregation": "pooled-report-plus-simultaneous-per-city-gate",
                    "evaluation_city_count": 1,
                    "evaluation_cities": ["target-a"],
                    "all_cities_passed": True,
                    "city_assessments": {
                        "target-a": {
                            "city_id": "target-a",
                            "passed": True,
                            "active_effect_passed": True,
                            "null_safety_passed": True,
                            "base_map_cluster_count": 2,
                            "null_overclassification_rate_ci95_high": 0.01,
                            "null_overclassification_rate_max": 0.05,
                        }
                    },
                }
            ],
            "condition_input_contract": "outer-recomputed-map-and-action-sha256-v1",
            "condition_registry_path": "external_condition_registry.csv",
            "condition_registry_sha256": "c" * 64,
            "adapter_manifest_sha256": "a" * 64,
            "adapter_manifest_path": "adapter_manifest.json",
        }
        self.assertEqual(_semantic_status("external_baselines", base), "BLOCKED")
        base["c1_eligible_model_count"] = 2
        base["c1_eligible_models"] = ["Wi-GATr", "Faithful-2"]
        base["model_assessments"].append(
            {
                "model_name": "Faithful-2",
                "c1_eligible": True,
                "passed": True,
                "active_effect_passed": True,
                "null_safety_passed": True,
                "base_map_cluster_count": 2,
                "null_overclassification_rate_ci95_high": 0.01,
                "null_overclassification_rate_max": 0.05,
                "aggregation": "pooled-report-plus-simultaneous-per-city-gate",
                "evaluation_city_count": 1,
                "evaluation_cities": ["target-a"],
                "all_cities_passed": True,
                "city_assessments": {
                    "target-a": {
                        "city_id": "target-a",
                        "passed": True,
                        "active_effect_passed": True,
                        "null_safety_passed": True,
                        "base_map_cluster_count": 2,
                        "null_overclassification_rate_ci95_high": 0.01,
                        "null_overclassification_rate_max": 0.05,
                    }
                },
            }
        )
        base["status"] = "PASS"
        base["passed"] = True
        self.assertEqual(_semantic_status("external_baselines", base), "PASS")
        changed = json.loads(json.dumps(base))
        changed["model_assessments"][1]["city_assessments"]["target-a"][
            "passed"
        ] = False
        self.assertEqual(_semantic_status("external_baselines", changed), "FAIL")

    def test_c1_model_must_pass_in_every_evaluation_city(self):
        city_ids = np.asarray(["source-a"] * 20 + ["target-a"] * 4)
        dataset = SimpleNamespace(
            base_map_cluster_ids=np.asarray(
                [f"cluster-{index:02d}" for index in range(city_ids.size)]
            ),
            bank_ids=np.asarray(
                [f"bank-{index:02d}" for index in range(city_ids.size)]
            ),
            city_ids=city_ids,
        )
        contracts = {
            f"unit-{index:02d}": SimpleNamespace(scene=index)
            for index in range(city_ids.size)
        }
        rows = []
        conditions = (
            "correct",
            "paired_active_alternative",
            "paired_null_alternative",
            "wrong_city",
            "geometry_destroyed",
            "empty",
        )
        for index in range(city_ids.size):
            values = {
                "correct": 0.0,
                "paired_active_alternative": 1.0 if index < 20 else -0.1,
                "paired_null_alternative": 0.0,
                "wrong_city": 1.0,
                "geometry_destroyed": 1.0,
                "empty": 1.0,
            }
            rows.extend(
                {
                    "unit_id": f"unit-{index:02d}",
                    "condition": condition,
                    "localization_error_m": values[condition],
                }
                for condition in conditions
            )
        config = {
            "evaluation": {
                "bootstrap_resamples": 1000,
                "c1_active_error_minimum_m": 0.1,
                "c1_null_error_equivalence_margin_m": 0.01,
                "null_overclassification_rate_max": 0.05,
            }
        }
        result = _c1_model_assessment(
            config,
            {"adapter_id": "adapter", "model_name": "model", "c1_eligible": True},
            rows,
            dataset,
            expected_unit_contract=contracts,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(result["city_assessments"]["source-a"]["passed"])
        self.assertFalse(result["city_assessments"]["target-a"]["passed"])

    def test_c1_claim_reauthenticates_the_adapter_manifest_copy(self):
        stage = Path(self.temporary.name) / "external_baselines"
        stage.mkdir()
        manifest = parse_strict_json(
            (ROOT / "formal_v2/external_adapters/all_map_adapters_v1.json").read_text()
        )
        manifest_path = stage / "adapter_manifest.json"
        write_json(manifest_path, manifest)
        condition_path = stage / "external_condition_registry.csv"
        write_csv(
            condition_path,
            [{"unit_id": "u0", "condition": "correct", "map_sha256": "a" * 64}],
        )
        payload = {
            "adapter_manifest_path": manifest_path.name,
            "adapter_manifest_sha256": sha256_file(manifest_path),
            "condition_registry_path": condition_path.name,
            "condition_registry_sha256": sha256_file(condition_path),
        }
        write_json(
            stage / "manifest.json",
            {
                "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
                "files": [
                    {"path": manifest_path.name, "sha256": sha256_file(manifest_path)},
                    {"path": condition_path.name, "sha256": sha256_file(condition_path)},
                ],
            },
        )
        _validate_external_manifest_binding(stage / "gate.json", payload)
        forged = json.loads(json.dumps(manifest))
        forged["adapters"][0]["adapter_config_path"] = forged["adapters"][1][
            "adapter_config_path"
        ]
        write_json(manifest_path, forged)
        forged_digest = sha256_file(manifest_path)
        payload["adapter_manifest_sha256"] = forged_digest
        stage_manifest = json.loads((stage / "manifest.json").read_text())
        stage_manifest["files"][0]["sha256"] = forged_digest
        write_json(stage / "manifest.json", stage_manifest)
        with self.assertRaisesRegex(ValueError, "config path/hash"):
            _validate_external_manifest_binding(stage / "gate.json", payload)

        write_json(manifest_path, manifest)
        payload["adapter_manifest_sha256"] = sha256_file(manifest_path)
        stage_manifest["files"][0]["sha256"] = payload["adapter_manifest_sha256"]
        write_json(stage / "manifest.json", stage_manifest)
        manifest["adapters"][0]["c1_eligible"] = True
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            _validate_external_manifest_binding(stage / "gate.json", payload)

    def test_shipped_sionna_manifest_is_a_valid_g8_adapter(self):
        manifest = parse_strict_json(
            (ROOT / "formal_v2/configs/sionna_external_validity_adapter_v2.json").read_text()
        )
        validate_external_validity_manifest(manifest)
        self.assertEqual(manifest["command"][1], "{adapter_source}")

    def test_sionna_gate_rejects_runtime_record_not_matching_outer_probe(self):
        output = Path(self.temporary.name) / "sionna-runtime-output"
        output.mkdir()
        runtime_path = output / "runtime_provenance.json"
        runtime = {"environment_sha256": "a" * 64}
        write_json(runtime_path, runtime)
        command = ["/runtime/venv/bin/python", "adapter.py"]
        with patch(
            "formal_v2.formal_external_runtime.validate_external_runtime",
            return_value=runtime,
        ):
            path, authenticated = _authenticate_sionna_runtime(
                command, output, runtime
            )
            self.assertEqual(path, runtime_path)
            self.assertEqual(authenticated, runtime)
            with self.assertRaisesRegex(RuntimeError, "independently probed"):
                _authenticate_sionna_runtime(
                    command,
                    output,
                    {"environment_sha256": "b" * 64},
                )

    def test_sionna_scene_export_covers_and_authenticates_external_worlds(self):
        arrays = _archive_arrays(self.path)
        scene_count = self.dataset.scene_count
        for name, value in tuple(arrays.items()):
            if value.ndim and value.shape[0] == scene_count:
                arrays[name] = np.concatenate((value, value[-1:]), axis=0)
        arrays["scene_roles"] = arrays["scene_roles"].astype("<U32")
        arrays["scene_roles"][-1] = "external_validation"
        arrays["position_roles"] = arrays["position_roles"].astype("<U32")
        arrays["position_roles"][-1] = "standard"
        arrays["scene_ids"] = arrays["scene_ids"].astype("<U32")
        arrays["scene_ids"][-1] = "external-scene"
        arrays["bank_ids"] = arrays["bank_ids"].astype("<U32")
        arrays["bank_ids"][-1] = "external-bank"
        arrays["base_map_cluster_ids"] = arrays["base_map_cluster_ids"].astype("<U32")
        arrays["base_map_cluster_ids"][-1] = "external-cluster"
        arrays["city_ids"] = arrays["city_ids"].astype("<U32")
        arrays["city_ids"][-1] = "external-city"
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["external_reference"]["available"] = True
        metadata["external_reference"]["kind"] = "independent_rt_engine"
        metadata["external_reference"]["dataset_id"] = "fixture-external-copy"
        metadata["external_reference"]["pairing_rule"] = "same synthetic sibling worlds"
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
        dataset_path = Path(self.temporary.name) / "external-fixture.npz"
        np.savez(dataset_path, **arrays)
        dataset = FormalDataset.load(dataset_path)
        output = Path(self.temporary.name) / "sionna-scenes"
        manifest_path = export_sionna_scenes(
            dataset_path,
            output,
            license_id="GENERATED-FIXTURE-NO-EXTERNAL-ASSET",
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=30e3,
            receiver_z_m=1.5,
            max_depth=3,
            refraction=False,
        )
        manifest = load_scene_manifest(manifest_path, dataset)
        expected = len(dataset.indices_for_role("external_validation")) * dataset.world_count
        self.assertEqual(len(manifest["worlds"]), expected)
        self.assertTrue(all(Path(row["scene_xml"]).is_file() for row in manifest["worlds"]))
        duplicated = dict(manifest)
        duplicated["worlds"] = [*manifest["worlds"], dict(manifest["worlds"][0])]
        duplicated_path = Path(self.temporary.name) / "duplicated-sionna-manifest.json"
        write_json(duplicated_path, duplicated)
        with self.assertRaisesRegex(ValueError, "cover every"):
            load_scene_manifest(duplicated_path, dataset)
        changed = dict(manifest)
        changed["sionna_revision"] = "not-the-frozen-revision"
        changed_path = Path(self.temporary.name) / "changed-sionna-manifest.json"
        write_json(changed_path, changed)
        with self.assertRaisesRegex(ValueError, "provenance"):
            load_scene_manifest(changed_path, dataset)

    def test_sionna_device_positions_cross_the_mitsuba_boundary_as_python_floats(self):
        point = sionna_point3(np.asarray([1.0, 2.0, 3.0], dtype=np.float64))
        self.assertEqual(point, [1.0, 2.0, 3.0])
        self.assertTrue(all(type(value) is float for value in point))
        with self.assertRaisesRegex(ValueError, "three finite"):
            sionna_point3((1.0, 2.0, np.inf))

    def test_sionna_uses_scalar_first_wxyz_bs_pose_quaternions(self):
        identity = sionna_quaternion_to_euler([1.0, 0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(identity, [0.0, 0.0, 0.0], atol=1e-12))
        half_angle = np.pi / 4.0
        yaw_90 = sionna_quaternion_to_euler(
            [np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)]
        )
        self.assertTrue(np.allclose(yaw_90, [np.pi / 2.0, 0.0, 0.0], atol=1e-12))

    def test_sionna_cfr_uses_numpy_without_a_torch_runtime_dependency(self):
        class Paths:
            arguments = None

            def cfr(self, **arguments):
                self.arguments = arguments
                return np.ones((1, 1, 1, 1, 1, 2), dtype=np.complex64)

        paths = Paths()
        values = sionna_numpy_cfr(paths, np.asarray([0.0, 1.0]), 2.0)
        self.assertEqual(values.shape, (1, 1, 1, 1, 1, 2))
        self.assertEqual(paths.arguments["out_type"], "numpy")
        self.assertEqual(paths.arguments["num_time_steps"], 1)

    def test_cli_exposes_resource_and_representation_stages(self):
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("verify-waibu-resources", help_text)
        self.assertIn("run-representation-baselines", help_text)
        self.assertIn("export-sionna-scenes", help_text)


def _archive_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _factorial_rows():
    rows = []
    # City A has three banks, city B one; k=8 has two draws. Equal macro layers must ignore both imbalances.
    for arm in ("endpoint", "alignment", "response", "full"):
        for city, banks in (("a", ("a1", "a2", "a3")), ("b", ("b1",))):
            for bank in banks:
                for seed in (1, 2):
                    rows.append(_row(arm, city, bank, seed, 0, 0, 0.0))
                    for draw in (0, 1):
                        full_value = 1.0 if city == "a" else 0.0
                        value = full_value if arm == "full" else 0.0
                        rows.append(_row(arm, city, bank, seed, 8, draw, value))
    return rows


def _row(arm, city, bank, seed, budget, draw, utility):
    return {
        "arm": arm,
        "city_id": city,
        "bank_id": bank,
        "base_map_cluster_id": bank,
        "seed": seed,
        "budget": budget,
        "draw": draw,
        "utility_neg_log_median": utility,
    }


if __name__ == "__main__":
    unittest.main()
