from __future__ import annotations

import json
import inspect
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from formal_v2.formal_metrics import (
    binary_auroc,
    binary_nll,
    brier_score,
    expected_calibration_error,
    spearman_correlation,
)
from formal_v2.formal_factorial import _run_training_jobs
from formal_v2.formal_model import (
    _DeterministicAdaptiveAvgPool2d,
    portable_state_dict,
    resolve_execution_device,
    resolve_execution_devices,
)
from formal_v2.formal_qualification import _qualification_scientific_use
from formal_v2.sionna_osm_candidate import (
    ASSET_SCHEMA,
    CONFIG_SCHEMA_V5,
    PATH_ANGLE_QUANTIZATION_RAD,
    ReflectionGeometryQualificationError,
    RAW_OSM_RECEIPT_SCHEMA,
    SIONNA_DRJIT_THREADS,
    SIONNA_MITSUBA_VARIANT,
    _allocate_reflection_pair,
    _axis_aligned_direct_visibility_mask,
    _asset_selection_executor,
    _balanced_branching_indices,
    _controlled_primitive_rows,
    _download_city_osm,
    _extract_paths,
    _extruded_triangles,
    _filter_controlled_receiver_candidates,
    _geometry_catalog,
    _has_direct_geometric_visibility,
    _nearest_free_bs,
    _order_bank_candidates,
    _anchor_bits,
    _physical_states,
    _primitive_mapping,
    _require_rt_path_availability,
    _signed_wrong_action_coverage,
    _sionna_bootstrap_environment,
    _regeneration_executor,
    _select_nonoverlapping_bank_candidates,
    _select_reflection_qualified_bank_candidates,
    _sampled_formal_placement_bounds,
    _stable_path_id,
    _validate_shard_runtime,
    _visible_reflection_quality,
    build_engine_config,
    candidate_power_report,
    expected_scene_ledger,
    load_config,
    load_asset_manifest,
    render_bank,
    render_shard,
    sha256_file,
    validate_render_shard,
)


class MetricInputContractTests(unittest.TestCase):
    def test_binary_metrics_match_hand_calculations(self):
        labels = np.asarray([0, 1, 0, 1])
        probabilities = np.asarray([0.1, 0.9, 0.2, 0.8])
        self.assertEqual(binary_auroc(labels, probabilities), 1.0)
        expected_nll = -float(
            np.mean(
                labels * np.log(probabilities)
                + (1 - labels) * np.log(1 - probabilities)
            )
        )
        self.assertAlmostEqual(binary_nll(labels, probabilities), expected_nll)
        self.assertAlmostEqual(
            brier_score(labels, probabilities),
            float(np.mean((probabilities - labels) ** 2)),
        )
        self.assertAlmostEqual(
            expected_calibration_error(labels, probabilities, bins=2), 0.15
        )
        self.assertEqual(
            spearman_correlation(np.asarray([1, 2, 3]), np.asarray([3, 2, 1])),
            -1.0,
        )

    def test_metrics_reject_broadcast_nonfinite_and_invalid_values(self):
        labels = np.asarray([0, 1])
        probabilities = np.asarray([0.25, 0.75])
        for function in (binary_nll, brier_score, expected_calibration_error):
            with self.subTest(function=function.__name__, case="broadcast"):
                with self.assertRaises(ValueError):
                    function(labels[:, None], probabilities)
            with self.subTest(function=function.__name__, case="label"):
                with self.assertRaises(ValueError):
                    function(np.asarray([0, 2]), probabilities)
            with self.subTest(function=function.__name__, case="probability"):
                with self.assertRaises(ValueError):
                    function(labels, np.asarray([-0.1, 1.1]))
        with self.assertRaises(ValueError):
            binary_auroc(labels, np.asarray([0.1, math.inf]))
        with self.assertRaises(ValueError):
            spearman_correlation(np.asarray([1.0, math.nan]), np.asarray([1.0, 2.0]))


class ExecutionDeviceContractTests(unittest.TestCase):
    def test_deterministic_pool_matches_adaptive_average_reference(self):
        pool = _DeterministicAdaptiveAvgPool2d((4, 4))
        for shape in ((2, 3, 8, 8), (1, 2, 9, 11), (1, 1, 16, 12)):
            values = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
            values.requires_grad_(True)
            actual = pool(values)
            expected = torch.nn.functional.adaptive_avg_pool2d(values, (4, 4))
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=0.0))
            actual.sum().backward()
            self.assertTrue(torch.all(torch.isfinite(values.grad)))

    def test_fixture_defaults_to_cpu_and_formal_data_rejects_cpu(self):
        fixture = SimpleNamespace(is_fixture=True)
        formal = SimpleNamespace(is_fixture=False)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CSI_PAIRS_DEVICE", None)
            os.environ.pop("CSI_PAIRS_DEVICES", None)
            self.assertEqual(resolve_execution_device(fixture), torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "requires an NVIDIA CUDA device"):
            resolve_execution_device(formal, "cpu")

    def test_multi_gpu_inventory_is_explicit_unique_and_ordered(self):
        fixture = SimpleNamespace(is_fixture=True)
        with (
            patch.dict(os.environ, {"CSI_PAIRS_DEVICES": "cuda:0,cuda:1"}),
            patch("formal_v2.formal_model.torch.cuda.is_available", return_value=True),
            patch("formal_v2.formal_model.torch.cuda.device_count", return_value=2),
        ):
            self.assertEqual(
                resolve_execution_devices(fixture),
                (torch.device("cuda:0"), torch.device("cuda:1")),
            )
        with (
            patch.dict(os.environ, {"CSI_PAIRS_DEVICES": "cuda:0,cuda:0"}),
            patch("formal_v2.formal_model.torch.cuda.is_available", return_value=True),
            patch("formal_v2.formal_model.torch.cuda.device_count", return_value=2),
        ):
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                resolve_execution_devices(fixture)

    def test_checkpoint_state_is_an_independent_cpu_copy(self):
        module = torch.nn.Linear(3, 2)
        state = portable_state_dict(module)
        for name, value in module.state_dict().items():
            self.assertEqual(state[name].device.type, "cpu")
            self.assertNotEqual(state[name].data_ptr(), value.data_ptr())
            self.assertTrue(torch.equal(state[name], value))

    def test_two_device_scheduler_serializes_each_device_and_preserves_job_order(self):
        class Model:
            def __init__(self, job_index):
                self.job_index = job_index
                self.device = "cpu"

            def to(self, device):
                self.device = str(device)
                return self

        barrier = threading.Barrier(2)
        calls = {"cuda:0": [], "cuda:1": []}

        def train(
            _config,
            _corpus,
            seed,
            arm,
            _pilot,
            *,
            device,
            prepared_model,
            stop_event,
        ):
            self.assertEqual(prepared_model.device, device)
            self.assertFalse(stop_event.is_set())
            calls[device].append(prepared_model.job_index)
            barrier.wait(timeout=2.0)
            return prepared_model, {"seed": seed, "arm": arm, "execution_device": device}

        jobs = {"cuda:0": [], "cuda:1": []}
        expected = []
        for job_index in range(12):
            device = f"cuda:{job_index % 2}"
            seed = 100 + job_index // 4
            arm = ("endpoint", "alignment", "response", "full")[job_index % 4]
            model = Model(job_index)
            jobs[device].append((job_index, seed, arm, model))
            expected.append((seed, arm))
        with patch("formal_v2.formal_factorial._train_arm", side_effect=train):
            trained = _run_training_jobs({}, object(), {}, jobs)
        self.assertEqual(
            [(row[1]["seed"], row[1]["arm"]) for row in trained],
            expected,
        )
        self.assertEqual(calls["cuda:0"], [0, 2, 4, 6, 8, 10])
        self.assertEqual(calls["cuda:1"], [1, 3, 5, 7, 9, 11])
        self.assertTrue(all(model.device == "cpu" for model, _row in trained))


class SionnaFormalRendererContractTests(unittest.TestCase):
    def test_controlled_receiver_filter_preserves_clear_los_points(self):
        from shapely.geometry import Point, box

        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "sionna_osm_formal_candidate_v5.json"
        )
        primitive = box(9.0, -2.0, 11.0, 2.0)
        candidates = np.asarray(
            [[0.0, 8.0], [10.0, 3.0], [30.0, 0.0], [50.0, 0.0]],
            dtype=np.float64,
        )
        filtered = _filter_controlled_receiver_candidates(
            candidates, (primitive,), config
        )
        np.testing.assert_array_equal(filtered, candidates[[0, 3]])

    def test_axis_aligned_visibility_vectorization_is_shapely_exact(self):
        from shapely import linestrings
        from shapely.geometry import Polygon

        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "sionna_osm_formal_candidate_v5.json"
        )
        rng = np.random.default_rng(20260815)
        candidates = np.concatenate(
            (
                rng.uniform(-128.0, 128.0, size=(512, 2)),
                np.asarray(
                    [
                        [0.0, 0.0],
                        [0.0, 128.0],
                        [128.0, 0.0],
                        [-128.0, 0.0],
                        [0.0, -128.0],
                    ],
                    dtype=np.float64,
                ),
            )
        )
        transmitter_z = float(config["radio"]["transmitter_z_m"])
        receiver_z = float(config["radio"]["receiver_z_m"])
        height = float(config["interventions"]["height_m"])
        minimum_clearance = float(
            config["map"]["receiver_minimum_direct_path_vertical_clearance_m"]
        )
        for scene_index in (0, 2, 7, 19, 35):
            rows = next(_controlled_primitive_rows(config, scene_index))
            polygons = tuple(
                Polygon(row["local_exterior_xy_m"]) for row in rows
            )
            actual = _axis_aligned_direct_visibility_mask(
                candidates,
                polygons,
                height=height,
                transmitter_z=transmitter_z,
                receiver_z=receiver_z,
                minimum_vertical_clearance_m=minimum_clearance,
            )
            lines = linestrings(
                np.stack((np.zeros_like(candidates), candidates), axis=1)
            )
            expected = np.asarray(
                [
                    _has_direct_geometric_visibility(
                        line,
                        [(polygon, height) for polygon in polygons],
                        transmitter_z=transmitter_z,
                        receiver_z=receiver_z,
                        minimum_vertical_clearance_m=minimum_clearance,
                    )
                    for line in lines
                ],
                dtype=np.bool_,
            )
            np.testing.assert_array_equal(actual, expected)

    def test_v5_controlled_layout_is_two_equal_orthogonal_pairs(self):
        from shapely.geometry import Polygon

        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "sionna_osm_formal_candidate_v5.json"
        )
        separation = float(config["interventions"]["center_separation_m"])
        for rows in _controlled_primitive_rows(config, 0):
            polygons = [Polygon(row["local_exterior_xy_m"]) for row in rows]
            self.assertEqual(len(polygons), 4)
            self.assertTrue(all(polygon.area == polygons[0].area for polygon in polygons))
            centers = np.asarray(
                [[polygon.centroid.x, polygon.centroid.y] for polygon in polygons]
            )
            shared_x = [
                (left, right)
                for left in range(4)
                for right in range(left + 1, 4)
                if math.isclose(centers[left, 0], centers[right, 0])
                and math.isclose(abs(centers[left, 1] - centers[right, 1]), separation)
            ]
            shared_y = [
                (left, right)
                for left in range(4)
                for right in range(left + 1, 4)
                if math.isclose(centers[left, 1], centers[right, 1])
                and math.isclose(abs(centers[left, 0] - centers[right, 0]), separation)
            ]
            self.assertEqual(len(shared_x), 1)
            self.assertEqual(len(shared_y), 1)
            self.assertEqual(set(shared_x[0]).intersection(shared_y[0]), set())

    def test_v5_engine_provenance_records_only_controlled_material_family(self):
        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "sionna_osm_formal_candidate_v5.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "asset_manifest.json").write_text("{}\n", encoding="ascii")
            manifest = {
                "banks": [{}] * 203,
                "config_sha256": "1" * 64,
                "source_config_sha256": "2" * 64,
                "raw_sources": [],
            }
            engine = build_engine_config(root, manifest, config)
        intervention_text = str(engine["interventions"]).lower()
        self.assertIn("concrete-to-glass", intervention_text)
        self.assertNotIn("wood", intervention_text)
        self.assertNotIn("metal", intervention_text)
        self.assertEqual(
            engine["geometry"]["receiver_sampling"]["algorithm"],
            "controlled-four-primitive-matched-branching-v5-clearance",
        )

    def test_four_primitive_design_is_minimum_for_eighty_percent_signed_swap(self):
        self.assertEqual(_signed_wrong_action_coverage(2), 0.5)
        self.assertEqual(_signed_wrong_action_coverage(3), 0.75)
        self.assertEqual(_signed_wrong_action_coverage(4), 0.875)

    def test_balanced_branching_selection_enforces_every_primitive_floor(self):
        quality = np.zeros((120, 4), dtype=np.float64)
        quality[:80, :2] = 10.0
        quality[80:104, 2:] = 1.0
        quality[104:, [0, 2]] = 1.0
        selected = _balanced_branching_indices(
            quality,
            np.arange(120, dtype=np.int64),
            count=96,
            minimum_per_primitive=24,
        )
        self.assertEqual(len(selected), 96)
        coverage = np.sum(quality[selected] > 0.0, axis=0)
        np.testing.assert_array_less(np.full(4, 23), coverage)

    def test_balanced_branching_selection_fails_closed_when_floor_is_impossible(self):
        quality = np.zeros((120, 4), dtype=np.float64)
        quality[:, :2] = 1.0
        quality[:23, 2:] = 1.0
        self.assertEqual(
            _balanced_branching_indices(
                quality,
                np.arange(120, dtype=np.int64),
                count=96,
                minimum_per_primitive=24,
            ),
            [],
        )

    def test_v5_anchor_xor_and_primitive_permutation_change_physical_state(self):
        mapping = _primitive_mapping(3, 4)
        anchor = _anchor_bits(3, 4)
        self.assertEqual(sorted(mapping.tolist()), [0, 1, 2, 3])
        physical_anchor = _physical_states(anchor, mapping, anchor)
        np.testing.assert_array_equal(physical_anchor, np.zeros(4, dtype=np.int64))
        changed = anchor.copy()
        changed[2] ^= 1
        physical_changed = _physical_states(changed, mapping, anchor)
        self.assertEqual(int(np.sum(physical_changed)), 1)
        self.assertEqual(physical_changed[mapping[2]], 1)

    def test_checked_in_v5_is_four_bit_controlled_hypercube(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        config = load_config(root / "sionna_osm_formal_candidate_v5.json")
        worlds = np.asarray(config["world_bits"], dtype=np.int64)
        self.assertEqual(worlds.shape, (16, 4))
        self.assertEqual(config["interventions"]["primitive_count"], 4)
        self.assertEqual(
            config["interventions"]["minimum_exact_positions_per_primitive"],
            24,
        )
        self.assertEqual(
            config["interventions"][
                "minimum_selected_quality_mean_per_primitive"
            ],
            0.0048,
        )
        self.assertEqual(config["map"]["receiver_grid_spacing_m"], 0.5)
        self.assertEqual(
            config["map"]["receiver_minimum_direct_path_vertical_clearance_m"],
            0.5,
        )
        self.assertEqual(
            config["map"]["bs_selection"],
            "sampled-formal-placement-capacity-v1",
        )
        self.assertEqual(config["map"]["bs_placement_sample_count"], 44)
        self.assertEqual(
            config["map"]["candidate_qualification_order"],
            "sparse-osm-first-v1",
        )
        self.assertGreaterEqual(_signed_wrong_action_coverage(worlds.shape[1]), 0.8)

    def test_sionna_renderer_preserves_controlled_scene_objects(self):
        source = inspect.getsource(render_bank)
        self.assertIn("load_scene(scene_xml, merge_shapes=False)", source)

    def test_direct_visibility_rejects_bank_103_rooftop_grazing_margin(self):
        from shapely.geometry import LineString, Polygon

        roof = Polygon(
            [
                (-16.679611, 13.480588),
                (-26.520405, 13.5522),
                (-58.904022, 13.431524),
                (-62.484392, 18.96938),
                (-66.262467, 24.772301),
                (-66.590427, 25.325232),
                (29.340804, 25.877315),
                (32.958297, 27.442032),
                (35.122569, 28.405262),
                (35.083255, 21.569475),
                (-16.93109, 20.925192),
                (-16.679611, 13.480588),
            ]
        )
        line = LineString(((0.0, 0.0), (-28.591, 27.599)))
        self.assertTrue(
            _has_direct_geometric_visibility(
                line,
                [(roof, 3.2)],
                transmitter_z=25.0,
                receiver_z=1.5,
            )
        )
        self.assertFalse(
            _has_direct_geometric_visibility(
                line,
                [(roof, 3.2)],
                transmitter_z=25.0,
                receiver_z=1.5,
                minimum_vertical_clearance_m=0.5,
            )
        )

    def test_rt_path_availability_reports_world_receiver_role_and_position(self):
        path_ids = np.asarray([[3, -1], [-1, -1], [5, 7]], dtype=np.int64)
        bank = {
            "scene_id": "bank-103",
            "receiver_geometry_roles": ["active", "background", "active"],
        }
        positions = np.asarray([[1.0, 2.0], [-28.591, 27.599], [3.0, 4.0]])
        with self.assertRaisesRegex(
            RuntimeError,
            r'"empty_receiver_indices":\[1\].*"trace_kind":"normal".*"world_index":0',
        ):
            _require_rt_path_availability(
                path_ids,
                bank,
                positions,
                world_index=0,
                trace_kind="normal",
            )

    def test_extrusion_orients_vertical_faces_outward(self):
        import pytest
        pytest.importorskip("shapely", minversion="2.1")
        from shapely.geometry import Polygon

        clockwise = Polygon([(0.0, 0.0), (0.0, 4.0), (6.0, 4.0), (6.0, 0.0)])
        triangles = _extruded_triangles(clockwise, 8.0)
        vertical = []
        for triangle in triangles:
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            if abs(float(normal[2])) < 1e-9:
                vertical.append((triangle, normal))
        self.assertEqual(len(vertical), 8)
        center = np.asarray(clockwise.centroid.coords[0])
        for triangle, normal in vertical:
            radial = triangle[:, :2].mean(axis=0) - center
            self.assertGreater(float(normal[:2] @ radial), 0.0)

    def test_extrusion_constrained_roof_exactly_preserves_concave_footprint(self):
        import pytest
        pytest.importorskip("shapely", minversion="2.1")
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        footprint = Polygon(
            [
                (0.0, 0.0),
                (8.0, 0.0),
                (8.0, 2.0),
                (2.0, 2.0),
                (2.0, 8.0),
                (0.0, 8.0),
                (0.0, 0.0),
            ]
        )
        triangles = _extruded_triangles(footprint, 6.0)
        roof = [
            Polygon(triangle[:, :2])
            for triangle in triangles
            if np.all(triangle[:, 2] == 6.0)
        ]
        roof_union = unary_union(roof)
        self.assertAlmostEqual(float(roof_union.area), float(footprint.area), places=9)
        self.assertLessEqual(float(roof_union.difference(footprint).area), 1e-10)
        self.assertLessEqual(float(footprint.difference(roof_union).area), 1e-10)

    def test_visible_reflection_requires_an_unblocked_exterior_wall(self):
        primitive = {
            "osm_id": 10,
            "height_m": 30.0,
            "local_exterior_xy_m": [
                [10.0, -10.0],
                [12.0, -10.0],
                [12.0, 10.0],
                [10.0, 10.0],
                [10.0, -10.0],
            ],
        }
        receiver = np.asarray([0.0, 10.0])
        clear_catalog = _geometry_catalog({"buildings": [primitive]})
        self.assertGreater(
            _visible_reflection_quality(
                receiver,
                0,
                clear_catalog,
                transmitter_z=25.0,
                receiver_z=1.5,
                minimum_incidence_cosine=0.15,
            ),
            0.0,
        )
        blocker = {
            "osm_id": 11,
            "height_m": 30.0,
            "local_exterior_xy_m": [
                [3.0, 1.0],
                [5.0, 1.0],
                [5.0, 3.0],
                [3.0, 3.0],
                [3.0, 1.0],
            ],
        }
        blocked_catalog = _geometry_catalog({"buildings": [primitive, blocker]})
        self.assertEqual(
            _visible_reflection_quality(
                receiver,
                0,
                blocked_catalog,
                transmitter_z=25.0,
                receiver_z=1.5,
                minimum_incidence_cosine=0.15,
            ),
            0.0,
        )

    def test_reflection_pair_allocation_enforces_disjoint_quotas(self):
        first = np.asarray([4.0, 3.0, 2.0, 1.0, 0.0])
        second = np.asarray([4.0, 3.0, 2.0, 0.0, 1.0])
        allocation = _allocate_reflection_pair(first, second, quota=2)
        self.assertIsNotNone(allocation)
        first_indices, second_indices, _ = allocation
        self.assertEqual(len(first_indices), 2)
        self.assertEqual(len(second_indices), 2)
        self.assertFalse(set(first_indices) & set(second_indices))

    def test_stable_path_identity_distinguishes_aoa_and_aod(self):
        surfaces = np.asarray([100, 102, -1], dtype=np.int64)
        vertices = np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        reference = np.asarray([1.1, 2.2, 1.3, -2.4], dtype=np.float64)
        changed_aoa = reference.copy()
        changed_aoa[1] += 2.0 * PATH_ANGLE_QUANTIZATION_RAD
        changed_aod = reference.copy()
        changed_aod[3] += 2.0 * PATH_ANGLE_QUANTIZATION_RAD
        reference_id = _stable_path_id(surfaces, 1.25e-7, vertices, reference)
        self.assertNotEqual(
            reference_id,
            _stable_path_id(surfaces, 1.25e-7, vertices, changed_aoa),
        )
        self.assertNotEqual(
            reference_id,
            _stable_path_id(surfaces, 1.25e-7, vertices, changed_aod),
        )

    def test_numerical_duplicate_paths_are_merged_without_losing_power(self):
        invalid = np.iinfo(np.uint32).max
        objects = np.full((3, 1, 1, 2), invalid, dtype=np.uint32)
        objects[0, 0, 0] = 7
        vertices = np.zeros((3, 1, 1, 2, 3), dtype=np.float64)
        vertices[0, 0, 0, 0] = [1.0, 2.0, 3.0]
        vertices[0, 0, 0, 1] = [1.0 + 1e-6, 2.0, 3.0]
        angles = np.asarray([[[1.0, 1.0 + 1e-6]]], dtype=np.float64)
        paths = SimpleNamespace(
            valid=np.ones((1, 1, 2), dtype=np.bool_),
            tau=np.asarray([[[1.25e-7, 1.25e-7 + 1e-14]]]),
            objects=objects,
            vertices=vertices,
            phi_r=angles,
            theta_r=np.asarray([[[1.1, 1.1]]]),
            phi_t=np.asarray([[[-2.0, -2.0]]]),
            theta_t=np.asarray([[[1.2, 1.2]]]),
            a=(
                np.asarray([[[[[1.0, 2.0]]]]]),
                np.zeros((1, 1, 1, 1, 2), dtype=np.float64),
            ),
        )
        path_ids, path_power, path_surfaces = _extract_paths(
            paths,
            {7: 102},
            position_count=1,
            max_paths=4,
            max_depth=3,
        )
        self.assertEqual(np.count_nonzero(path_ids[0] >= 0), 1)
        self.assertEqual(path_power[0, path_ids[0] >= 0].tolist(), [5.0])
        self.assertEqual(path_surfaces[0, path_ids[0] >= 0][0].tolist(), [102, -1, -1])

    def test_verified_candidate_earns_formal_use_only_after_g1_and_g2(self):
        candidate = SimpleNamespace(
            is_fixture=False,
            metadata={"scientific_use": "CANDIDATE"},
        )
        live = {
            "schema_version": "csi-pairs-v6-data-verification-gate-v1",
            "verification_mode": "live_independent_regeneration",
            "passed": True,
            "blocking_passed": True,
        }
        self.assertEqual(_qualification_scientific_use(candidate, False, live), "FORBIDDEN")
        self.assertEqual(
            _qualification_scientific_use(candidate, True, None),
            "FORBIDDEN",
        )
        self.assertEqual(
            _qualification_scientific_use(candidate, True, live),
            "FORMAL_EXPERIMENT_ALLOWED",
        )
        candidate.metadata["scientific_use"] = "FORBIDDEN"
        self.assertEqual(_qualification_scientific_use(candidate, True, live), "FORBIDDEN")
        candidate.metadata["scientific_use"] = "CANDIDATE"
        candidate.is_fixture = True
        self.assertEqual(_qualification_scientific_use(candidate, True, live), "FORBIDDEN")






    def test_generator_script_defaults_to_the_reviewed_sionna_runtime(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts/generate_sionna_osm_formal_candidate.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("CSI_PAIRS_SIONNA_PYTHON", script)
        self.assertIn(".runtime-sionna/venv/bin/python", script)
        self.assertIn("setup_sionna.sh first", script)
        self.assertIn("CSI_PAIRS_RENDER_WORKERS", script)
        self.assertIn('RENDER_WORKERS:-8', script)
        self.assertIn('OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"', script)
        self.assertIn('VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"', script)
        self.assertIn('-f "${shard_manifest}" && ! -L "${shard_manifest}"', script)
        self.assertIn('-f "${OUTPUT_ROOT}/dataset.generation.json"', script)
        self.assertIn('! -L "${OUTPUT_ROOT}/dataset.generation.json"', script)
        self.assertIn('recovered_raw_cache="${recovery}/assets-incomplete/raw_osm"', script)
        self.assertIn('[[ -z "${RAW_CACHE}" && -d "${recovered_raw_cache}" ]]', script)
        self.assertIn('ASSET_ARGS+=(--raw-cache "${recovered_raw_cache}")', script)
        self.assertIn('"${OUTPUT_ROOT}/runtime-cache/drjit"', script)
        self.assertIn('cache_home="${OUTPUT_ROOT}/runtime-cache/drjit/${shard_name}"', script)
        self.assertIn('HOME="${cache_home}" "${PYTHON_BIN}"', script)
        self.assertIn("formal render shards must contain exactly one atomic bank", inspect.getsource(render_shard))
        self.assertIn("formal_candidate_world_complete", inspect.getsource(render_bank))
        self.assertIn("PENDING_INDICES", script)
        self.assertIn("ACTIVE_BANK_BY_PID", script)
        self.assertIn("CSI_PAIRS_RENDER_RETRIES", script)
        self.assertIn('wait "${pid}"', script)
        self.assertNotIn("wait -n", script)
        self.assertIn("bank-%03d", script)
        self.assertIn("failed != 0 && ${#ACTIVE_BANK_BY_PID[@]} == 0", script)
        self.assertNotIn("base_count=$((SCENE_COUNT / RENDER_WORKERS))", script)

    def _run_generator_scheduler_probe(self, *, always_fail: bool) -> subprocess.CompletedProcess:
        project_root = Path(__file__).resolve().parents[1]
        source_script = project_root / "scripts/generate_sionna_osm_formal_candidate.sh"
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        script = root / "formal_v2/scripts/generate_sionna_osm_formal_candidate.sh"
        script.parent.mkdir(parents=True)
        shutil.copy2(source_script, script)
        config = root / "candidate.json"
        config.write_text("{}\n", encoding="ascii")
        fake_python = root / "fake-python"
        fake_python.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
joined=" $* "
value_after() {
  local wanted="$1"
  shift
  while (( $# > 0 )); do
    if [[ "$1" == "$wanted" ]]; then
      printf '%s\\n' "$2"
      return 0
    fi
    shift
  done
  return 1
}
if [[ "$joined" == *" formal_v2.sionna_runtime_lock "* ]]; then
  printf '/tmp/fake-libLLVM.so\\n'
elif [[ "$joined" == *" scene-count "* ]]; then
  printf '5\\n'
elif [[ "$joined" == *" prepare-assets "* ]]; then
  output="$(value_after --output "$@")"
  mkdir -p "$output"
  printf '{}\\n' >"$output/asset_manifest.json"
elif [[ "$joined" == *" render-shard "* ]]; then
  output="$(value_after --output "$@")"
  index="$(value_after --shard-index "$@")"
  state="${output%/*}/attempt-$index"
  attempt=1
  if [[ -f "$state" ]]; then
    attempt=$(( $(<"$state") + 1 ))
  fi
  printf '%s\\n' "$attempt" >"$state"
  sleep 0.03
  if [[ "$index" == "2" && ( "${FAKE_ALWAYS_FAIL:-0}" == "1" || "$attempt" == "1" ) ]]; then
    exit 17
  fi
  printf 'bank=%s\\n' "$index" >"$output"
  printf '{}\\n' >"${output%.npz}.manifest.json"
elif [[ "$joined" == *" merge "* ]]; then
  output="$(value_after --output "$@")"
  printf 'merged\\n' >"$output"
  printf '{}\\n' >"${output%.npz}.generation.json"
else
  exit 1
fi
""",
            encoding="ascii",
        )
        fake_python.chmod(0o755)
        environment = {
            **os.environ,
            "CSI_PAIRS_SIONNA_PYTHON": str(fake_python),
            "CSI_PAIRS_SIONNA_CONFIG": str(config),
            "CSI_PAIRS_RENDER_WORKERS": "4",
            "CSI_PAIRS_RENDER_RETRIES": "2",
            "FAKE_ALWAYS_FAIL": "1" if always_fail else "0",
        }
        output = root / "output"
        return subprocess.run(
            ["bash", str(script), str(output)],
            cwd=root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )

    @unittest.skipIf(os.name == "nt", "Linux Bash renderer scheduler requires POSIX Python paths")
    def test_generator_scheduler_retries_one_failed_bank_without_omitting_tail(self):
        result = self._run_generator_scheduler_probe(always_fail=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = Path(result.args[-1])
        self.assertEqual(
            sorted(path.name for path in (output / "shards").glob("bank-*.npz")),
            [f"bank-{index:03d}.npz" for index in range(5)],
        )
        self.assertEqual((output / "shards/attempt-2").read_text().strip(), "2")
        self.assertTrue((output / "dataset.npz").is_file())

    @unittest.skipIf(os.name == "nt", "Linux Bash renderer scheduler requires POSIX Python paths")
    def test_generator_scheduler_fails_closed_after_retry_budget(self):
        result = self._run_generator_scheduler_probe(always_fail=True)
        self.assertNotEqual(result.returncode, 0)
        output = Path(result.args[-1])
        self.assertEqual((output / "shards/attempt-2").read_text().strip(), "3")
        self.assertFalse((output / "dataset.npz").exists())
        self.assertIn("failed after 3 attempts", result.stderr)

    def test_render_shard_is_one_bank_atomic_and_resume_validation_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root / "assets"
            assets.mkdir()
            (assets / "asset_manifest.json").write_text("{}\n", encoding="utf-8")
            bank_path = assets / "bank.json"
            bank_path.write_text("{}\n", encoding="utf-8")
            manifest = {
                "banks": [
                    {"bank_record_path": "bank.json"},
                    {"bank_record_path": "bank.json"},
                ]
            }
            rendered = {"probe": np.asarray([1.0, 2.0], dtype=np.float64)}
            runtime = {"test_runtime": True}
            module = "formal_v2.sionna_osm_candidate"
            with (
                patch(f"{module}.load_asset_manifest", return_value=(assets, manifest, {})),
                patch(f"{module}.render_bank", return_value=rendered) as render,
                patch(f"{module}._sionna_runtime_record", return_value=runtime),
            ):
                with self.assertRaisesRegex(ValueError, "exactly one atomic bank"):
                    render_shard(assets, root / "multi.npz", 0, 2, 0)
                shard = render_shard(assets, root / "bank-000.npz", 0, 1, 0)
            render.assert_called_once()
            shard_manifest = json.loads(
                shard.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(shard_manifest["scene_count"], 1)
            self.assertEqual(shard_manifest["scene_start_inclusive"], 0)
            self.assertEqual(shard_manifest["scene_end_exclusive"], 1)
            self.assertEqual(shard_manifest["output_sha256"], sha256_file(shard))
            with np.load(shard, allow_pickle=False) as archive:
                self.assertEqual(archive["scene_indices"].tolist(), [0])
                np.testing.assert_array_equal(archive["probe"], rendered["probe"][None])

            with (
                patch(f"{module}.load_asset_manifest", return_value=(assets, manifest, {})),
                patch(f"{module}._validate_shard_runtime"),
            ):
                self.assertEqual(
                    validate_render_shard(assets, shard, 0, 1, 0), shard.resolve()
                )
                with self.assertRaisesRegex(ValueError, "registered resume slot"):
                    validate_render_shard(assets, shard, 1, 2, 1)
                shard_manifest["output_sha256"] = "0" * 64
                shard.with_suffix(".manifest.json").write_text(
                    json.dumps(shard_manifest), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    validate_render_shard(assets, shard, 0, 1, 0)

    def test_sionna_setup_has_native_macos_arm64_branch(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / "external_adapters/setup_sionna.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('SYSTEM="$(uname -s)"', script)
        self.assertIn('if [[ "${SYSTEM}" == "Darwin" ]]', script)
        self.assertIn("CSI_PAIRS_BREW", script)
        self.assertIn("--prefix llvm@18", script)
        self.assertIn("requirements-sionna-runtime-darwin-arm64.txt", script)
        self.assertIn("requirements-sionna-runtime-linux-x86_64.txt", script)
        self.assertIn("--require-hashes", script)
        self.assertIn("sionna_runtime_lock", script)
        self.assertIn("-m formal_v2.formal_external_runtime", script)
        self.assertIn("llvm_ad_mono_polarized", script)
        self.assertIn("drjit.set_thread_count(1)", script)

    def test_sionna_torch_runtime_lock_includes_direct_dependency_closure(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "filelock==3.32.2": "87dd94cf281e586d135fa51132b8e3d9a598b316e90377a288663c9321036c82",
            "fsspec==2026.7.0": "b57ddbafedfaef7018c1ecab32aa200a9d7ca26b77965f64e48b70061249d279",
            "mpmath==1.3.0": "a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c",
            "sympy==1.14.0": "e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5",
        }
        for name in (
            "requirements-sionna-runtime-darwin-arm64.txt",
            "requirements-sionna-runtime-linux-x86_64.txt",
        ):
            text = (root / name).read_text(encoding="utf-8")
            with self.subTest(lock=name):
                for requirement, digest in expected.items():
                    self.assertIn(requirement, text)
                    self.assertIn(f"--hash=sha256:{digest}", text)

    def test_a100_runbook_forbids_old_candidate_and_binds_two_devices(self):
        root = Path(__file__).resolve().parents[1]
        runbook = (root / "A100_RUNBOOK.md").read_text(encoding="utf-8")
        self.assertIn("old", runbook.lower())
        self.assertIn("pathless/all-zero", runbook)
        self.assertIn("CSI_PAIRS_DEVICES=cuda:0,cuda:1", runbook)
        self.assertIn("rtol=0", runbook)
        self.assertIn("atol=0", runbook)


    def test_regeneration_executor_uses_spawn_context(self):
        context = multiprocessing.get_context("spawn")
        executor = MagicMock()
        with (
            patch("formal_v2.sionna_osm_candidate.multiprocessing.get_context", return_value=context),
            patch("formal_v2.sionna_osm_candidate.ProcessPoolExecutor", return_value=executor) as factory,
        ):
            self.assertIs(_regeneration_executor(8), executor)
            factory.assert_called_once_with(max_workers=8, mp_context=context)

    def test_asset_selection_executor_uses_spawn_context_and_rejects_zero(self):
        context = multiprocessing.get_context("spawn")
        executor = MagicMock()
        with (
            patch("formal_v2.sionna_osm_candidate.multiprocessing.get_context", return_value=context),
            patch("formal_v2.sionna_osm_candidate.ProcessPoolExecutor", return_value=executor) as factory,
        ):
            self.assertIs(_asset_selection_executor(6), executor)
            factory.assert_called_once_with(max_workers=6, mp_context=context)
        with self.assertRaisesRegex(ValueError, "at least one city worker"):
            _asset_selection_executor(0)

    def test_bank_selection_skips_higher_scoring_overlapping_extent(self):
        candidates = [
            (100.0, 10.0, 0.0, 0.0, []),
            (99.0, 10.0, 244.0, 0.0, []),
            (98.0, 10.0, 300.0, 0.0, []),
            (97.0, 10.0, 0.0, 300.0, []),
        ]
        selected = _select_nonoverlapping_bank_candidates(
            candidates,
            3,
            map_extent_m=256.0,
        )
        self.assertEqual(
            [(row[2], row[3]) for row in selected],
            [(0.0, 0.0), (300.0, 0.0), (0.0, 300.0)],
        )

    def test_nearest_free_bs_spatial_index_preserves_clearance(self):
        from shapely.geometry import box
        from shapely.strtree import STRtree

        buildings = [{"polygon": box(-5.0, -5.0, 5.0, 5.0)}]
        tree = STRtree([row["polygon"] for row in buildings])
        self.assertEqual(_nearest_free_bs(0.0, 0.0, buildings, tree), (10.0, 0.0))
        self.assertEqual(
            _nearest_free_bs(100.0, 100.0, buildings, tree), (100.0, 100.0)
        )

    def test_v5_placement_aware_bs_avoids_blocked_first_free_point(self):
        from shapely.geometry import Point, box
        from shapely.strtree import STRtree

        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "sionna_osm_formal_candidate_v5.json"
        )
        buildings = [{"polygon": box(18.0, -2.0, 22.0, 2.0)}]
        tree = STRtree([row["polygon"] for row in buildings])
        bounds = np.repeat(
            np.asarray([[[18.0, -2.0, 22.0, 2.0]]], dtype=np.float64),
            4,
            axis=1,
        )
        selected = _nearest_free_bs(
            0.0,
            0.0,
            buildings,
            tree,
            config=config,
            placement_bounds=bounds,
        )
        self.assertNotEqual(selected, (0.0, 0.0))
        self.assertGreaterEqual(buildings[0]["polygon"].distance(Point(selected)), 4.0)

    def test_v5_sampled_formal_placement_inventory_is_exact_and_deterministic(self):
        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "sionna_osm_formal_candidate_v5.json"
        )
        first = _sampled_formal_placement_bounds(config)
        second = _sampled_formal_placement_bounds(config)
        self.assertEqual(first.shape, (44, 4, 4))
        np.testing.assert_array_equal(first, second)

    def test_sparse_candidate_order_reorders_without_changing_membership(self):
        candidates = [
            (300.0, 5.0, 0.0, 0.0, [{}] * 30),
            (100.0, 9.0, 300.0, 0.0, [{}] * 6),
            (110.0, 8.0, 600.0, 0.0, [{}] * 6),
        ]
        ordered = _order_bank_candidates(candidates, "sparse-osm-first-v1")
        self.assertEqual([row[2] for row in ordered], [300.0, 600.0, 0.0])
        self.assertEqual({id(row) for row in ordered}, {id(row) for row in candidates})
        self.assertEqual(
            _order_bank_candidates(candidates, "sparse-osm-first-v1"), ordered
        )

    def test_raw_osm_download_writes_and_reuses_bound_receipt(self):
        payload = json.dumps(
            {"elements": [], "osm3s": {"timestamp_osm_base": "2026-08-13T00:00:00Z"}}
        ).encode("utf-8")
        city = {
            "city_id": "source-chicago",
            "center_lat": 41.882,
            "center_lon": -87.629,
            "query_delta_lat": 0.01,
            "query_delta_lon": 0.01,
        }
        config = {"overpass_endpoint": "https://overpass.example/api/interpreter"}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return payload

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            with patch(
                "formal_v2.sionna_osm_candidate.urlrequest.urlopen",
                return_value=Response(),
            ) as download:
                raw_path, query, parsed, endpoint = _download_city_osm(
                    city, config, first
                )
            download.assert_called_once()
            self.assertEqual(parsed["elements"], [])
            self.assertEqual(endpoint, config["overpass_endpoint"])
            receipt_path = raw_path.with_suffix(".receipt.json")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema_version"], RAW_OSM_RECEIPT_SCHEMA)
            self.assertEqual(receipt["overpass_query"], query)
            self.assertEqual(receipt["sha256"], sha256_file(raw_path))

            with patch(
                "formal_v2.sionna_osm_candidate.urlrequest.urlopen"
            ) as unexpected_download:
                copied, copied_query, copied_payload, copied_endpoint = _download_city_osm(
                    city, config, second, first
                )
            unexpected_download.assert_not_called()
            self.assertEqual(copied_query, query)
            self.assertEqual(copied_payload, parsed)
            self.assertEqual(
                copied_endpoint, f"frozen-cache-sha256:{sha256_file(raw_path)}"
            )
            copied_receipt = json.loads(
                copied.with_suffix(".receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(copied_receipt["overpass_endpoint"], copied_endpoint)

    def test_raw_osm_cache_rejects_receipt_query_tampering(self):
        city = {
            "city_id": "source-chicago",
            "center_lat": 41.882,
            "center_lon": -87.629,
            "query_delta_lat": 0.01,
            "query_delta_lon": 0.01,
        }
        config = {"overpass_endpoint": "https://overpass.example/api/interpreter"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            output = root / "output"
            cache.mkdir()
            output.mkdir()
            raw = cache / "source-chicago.json"
            raw.write_text('{"elements": []}\n', encoding="ascii")
            receipt = {
                "schema_version": RAW_OSM_RECEIPT_SCHEMA,
                "status": "PASS",
                "city_id": city["city_id"],
                "overpass_query": "tampered-query",
                "overpass_endpoint": config["overpass_endpoint"],
                "bytes": raw.stat().st_size,
                "sha256": sha256_file(raw),
            }
            raw.with_suffix(".receipt.json").write_text(
                json.dumps(receipt), encoding="ascii"
            )
            with (
                patch(
                    "formal_v2.sionna_osm_candidate.urlrequest.urlopen",
                    side_effect=RuntimeError("offline"),
                ) as download,
                self.assertRaisesRegex(RuntimeError, "Overpass query failed"),
            ):
                _download_city_osm(city, config, output, cache)
            self.assertGreaterEqual(download.call_count, 1)

    def test_raw_osm_download_retries_after_non_json_response(self):
        valid_payload = json.dumps(
            {
                "elements": [],
                "osm3s": {"timestamp_osm_base": "2026-08-13T00:00:00Z"},
            }
        ).encode("utf-8")
        city = {
            "city_id": "external-miami",
            "center_lat": 25.775,
            "center_lon": -80.1937,
            "query_delta_lat": 0.01,
            "query_delta_lon": 0.01,
        }
        config = {"overpass_endpoint": "https://bad.example/api/interpreter"}

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return self.payload

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch(
                "formal_v2.sionna_osm_candidate.urlrequest.urlopen",
                side_effect=[Response(b"<html>redirect</html>"), Response(valid_payload)],
            ) as download:
                raw_path, _query, parsed, endpoint = _download_city_osm(
                    city, config, root
                )
            self.assertEqual(download.call_count, 2)
            self.assertEqual(parsed["elements"], [])
            self.assertEqual(
                endpoint, "https://overpass.kumi.systems/api/interpreter"
            )
            self.assertEqual(raw_path.read_bytes(), valid_payload)
            receipt = json.loads(
                raw_path.with_suffix(".receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["overpass_endpoint"], endpoint)

    def test_v5_asset_loader_rejects_tampered_bound_osm_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "generator_config.json"
            config_path.write_text("{}\n", encoding="ascii")
            raw_path = root / "raw_osm" / "source-chicago.json"
            raw_path.parent.mkdir()
            raw_path.write_text('{"elements": []}\n', encoding="ascii")
            receipt_path = raw_path.with_suffix(".receipt.json")
            query = "[out:json];way[building];out;"
            endpoint = "https://overpass.example/api/interpreter"
            receipt = {
                "schema_version": RAW_OSM_RECEIPT_SCHEMA,
                "status": "PASS",
                "city_id": "source-chicago",
                "overpass_query": query,
                "overpass_endpoint": endpoint,
                "bytes": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
            }
            receipt_path.write_text(json.dumps(receipt), encoding="ascii")
            ledger = {
                "scene_index": 0,
                "scene_id": "scene-0",
                "city_id": "source-chicago",
                "role": "source_encoder_train",
                "bank_id": "bank-0",
                "base_map_cluster_id": "cluster-0",
            }
            config = {
                "schema_version": CONFIG_SCHEMA_V5,
                "cities": [{"city_id": "source-chicago"}],
            }
            manifest = {
                "schema_version": ASSET_SCHEMA,
                "config_path": config_path.name,
                "config_sha256": sha256_file(config_path),
                "raw_sources": [
                    {
                        "city_id": "source-chicago",
                        "path": raw_path.relative_to(root).as_posix(),
                        "bytes": raw_path.stat().st_size,
                        "sha256": sha256_file(raw_path),
                        "receipt_path": receipt_path.relative_to(root).as_posix(),
                        "receipt_sha256": sha256_file(receipt_path),
                        "overpass_query": query,
                        "overpass_endpoint": endpoint,
                    }
                ],
                "banks": [{**ledger, "files": []}],
            }
            manifest_path = root / "asset_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="ascii")
            with (
                patch(
                    "formal_v2.sionna_osm_candidate.load_config",
                    return_value=config,
                ),
                patch(
                    "formal_v2.sionna_osm_candidate.expected_scene_ledger",
                    return_value=[ledger],
                ),
            ):
                load_asset_manifest(root)
                receipt["overpass_query"] = "tampered-query"
                receipt_path.write_text(json.dumps(receipt), encoding="ascii")
                manifest["raw_sources"][0]["receipt_sha256"] = sha256_file(receipt_path)
                manifest_path.write_text(json.dumps(manifest), encoding="ascii")
                with self.assertRaisesRegex(ValueError, "does not bind its source"):
                    load_asset_manifest(root)

    def test_bank_selection_skips_unqualified_candidate_and_continues_rank_order(self):
        candidates = [
            (100.0, 10.0, 0.0, 0.0, []),
            (99.0, 10.0, 300.0, 0.0, []),
            (98.0, 10.0, 600.0, 0.0, []),
        ]
        calls = []

        def qualify(candidate, bank_index):
            calls.append((candidate[0], bank_index))
            if candidate[0] == 100.0:
                raise ReflectionGeometryQualificationError(
                    "insufficient_visible_primitives", "deliberate test rejection"
                )
            return {"score": candidate[0], "bank_index": bank_index}

        selected, audit = _select_reflection_qualified_bank_candidates(
            candidates,
            2,
            map_extent_m=256.0,
            qualify=qualify,
        )
        self.assertEqual(
            selected,
            [
                {"score": 99.0, "bank_index": 0},
                {"score": 98.0, "bank_index": 1},
            ],
        )
        self.assertEqual(calls, [(100.0, 0), (99.0, 0), (98.0, 1)])
        self.assertEqual(audit["candidates_examined"], 3)
        self.assertEqual(audit["selected_bank_count"], 2)
        self.assertEqual(audit["rejected_reflection_geometry_count"], 1)
        self.assertEqual(
            audit["rejected_reflection_geometry_reason_counts"],
            {"insufficient_visible_primitives": 1},
        )
        self.assertEqual(
            audit["rejected_reflection_geometry_candidates"][0]["candidate_rank"], 0
        )

    def test_bank_selection_fails_only_after_exhausting_qualified_candidates(self):
        candidates = [
            (100.0, 10.0, 0.0, 0.0, []),
            (99.0, 10.0, 300.0, 0.0, []),
            (98.0, 10.0, 600.0, 0.0, []),
        ]
        calls = []

        def qualify(candidate, bank_index):
            calls.append((candidate[0], bank_index))
            if candidate[0] != 99.0:
                raise ReflectionGeometryQualificationError(
                    "insufficient_visible_primitives", "deliberate test rejection"
                )
            return {"score": candidate[0], "bank_index": bank_index}

        with self.assertRaisesRegex(
            RuntimeError,
            "only 1 disjoint reflection-qualified bank extents are available; 2 are required",
        ):
            _select_reflection_qualified_bank_candidates(
                candidates,
                2,
                map_extent_m=256.0,
                qualify=qualify,
            )
        self.assertEqual(calls, [(100.0, 0), (99.0, 0), (98.0, 1)])

    def test_checked_in_candidate_uses_single_thread_llvm(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "sionna_osm_formal_candidate_v3.json"
        )
        config = load_config(config_path)
        self.assertEqual(config["renderer"]["mitsuba_variant"], SIONNA_MITSUBA_VARIANT)
        self.assertEqual(config["renderer"]["drjit_threads"], SIONNA_DRJIT_THREADS)
        self.assertEqual(config["map"]["receiver_grid_spacing_m"], 1.5)
        self.assertEqual(config["map"]["receiver_primitive_quota"], 64)
        self.assertEqual(config["map"]["minimum_reflection_incidence_cosine"], 0.15)

    def test_power_designed_candidate_replaces_underpowered_34_bank_layout(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        old = load_config(root / "sionna_osm_formal_candidate_v3.json")
        expanded = load_config(root / "sionna_osm_formal_candidate_v4.json")
        old_report = candidate_power_report(old)
        expanded_report = candidate_power_report(expanded)

        self.assertEqual(len(expected_scene_ledger(old)), 34)
        self.assertFalse(old_report["passed"])
        self.assertEqual(old_report["source_role_counts"]["source_final_unseen_bank"], 2)
        self.assertEqual(old_report["target_city_counts"], {
            "target-boston": 8,
            "target-seattle": 8,
        })

        self.assertEqual(len(expected_scene_ledger(expanded)), 203)
        self.assertTrue(expanded_report["passed"])
        self.assertEqual(
            expanded_report["source_role_counts"]["source_final_unseen_bank"], 41
        )
        self.assertTrue(
            all(
                value >= 8
                for value in expanded_report["source_role_counts"].values()
            )
        )
        self.assertEqual(expanded_report["target_city_counts"], {
            "target-boston": 41,
            "target-seattle": 41,
        })
        self.assertEqual(expanded_report["external_validation_count"], 32)

    def test_bootstrap_requires_only_python_and_llvm_for_renderer(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            python = project / "formal_v2/external_adapters/.runtime-sionna/venv/bin/python"
            llvm = project / "libLLVM-test.so"
            python.parent.mkdir(parents=True)
            python.touch()
            llvm.touch()
            approved = {"libllvm_path": str(llvm.resolve())}
            with patch(
                "formal_v2.sionna_runtime_lock.approved_library_record",
                return_value=approved,
            ):
                resolved_python, environment = _sionna_bootstrap_environment(
                    project,
                    {"PYTHONPATH": "/existing"},
                    libllvm=llvm,
                )
        self.assertEqual(resolved_python, python.resolve())
        self.assertEqual(environment["DRJIT_LIBLLVM_PATH"], str(llvm.resolve()))
        self.assertEqual(environment["MI_DEFAULT_VARIANT"], SIONNA_MITSUBA_VARIANT)
        self.assertNotIn("LD_PRELOAD", environment)

    def test_runtime_contract_rejects_cuda_variant(self):
        with tempfile.TemporaryDirectory() as temporary:
            llvm = Path(temporary) / "libLLVM-test.so"
            llvm.write_bytes(b"llvm")
            runtime = {
                "python": "3.12.13",
                "sionna": "2.0.1",
                "sionna_rt": "1.2.1",
                "mitsuba": "3.7.1",
                "drjit": "1.2.0",
                "mitsuba_variant": SIONNA_MITSUBA_VARIANT,
                "drjit_thread_count": 1,
                "sionna_revision": "04ddb9312116b408093b9d3ad363a3df355093a6",
                "drjit_libllvm_path": str(llvm),
                "drjit_libllvm_sha256": sha256_file(llvm),
            }
            approved = {"libllvm_sha256": sha256_file(llvm)}
            with patch(
                "formal_v2.sionna_runtime_lock.approved_library_record",
                return_value=approved,
            ):
                _validate_shard_runtime(runtime)
                runtime["mitsuba_variant"] = "cuda_ad_mono_polarized"
                with self.assertRaisesRegex(ValueError, "frozen Sionna runtime"):
                    _validate_shard_runtime(runtime)


if __name__ == "__main__":
    unittest.main()
