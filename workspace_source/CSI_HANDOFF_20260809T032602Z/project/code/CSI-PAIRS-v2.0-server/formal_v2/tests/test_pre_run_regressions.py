from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import torch

from formal_v2.formal_external_runtime import _cuda_total_memory_bytes
from formal_v2.formal_metrics import (
    binary_auroc,
    binary_nll,
    brier_score,
    expected_calibration_error,
    spearman_correlation,
)
from formal_v2.formal_evidence import (
    CUDA_COMPAT_FILES,
    CUDA_COMPAT_PACKAGE_SHA256,
    CUDA_COMPAT_PACKAGE_URL,
    _cuda_driver_record,
)
from formal_v2.formal_model import (
    _DeterministicAdaptiveAvgPool2d,
    resolve_execution_device,
    resolve_execution_devices,
)
from formal_v2.sionna_osm_candidate import (
    SIONNA_EXACT_REGENERATION_VERIFIED,
    _sionna_bootstrap_environment,
    _set_world_materials,
    _stable_path_id,
    _write_scene_xml,
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
            expected_calibration_error(labels, probabilities, bins=2),
            0.15,
        )
        self.assertAlmostEqual(
            spearman_correlation(np.asarray([1, 2, 3]), np.asarray([3, 2, 1])),
            -1.0,
        )

    def test_binary_metrics_reject_broadcast_nonfinite_and_invalid_values(self):
        valid_labels = np.asarray([0, 1])
        valid_probabilities = np.asarray([0.25, 0.75])
        functions = (binary_nll, brier_score, expected_calibration_error)
        for function in functions:
            with self.subTest(function=function.__name__, failure="broadcast"):
                with self.assertRaises(ValueError):
                    function(valid_labels[:, None], valid_probabilities)
            with self.subTest(function=function.__name__, failure="label"):
                with self.assertRaises(ValueError):
                    function(np.asarray([0, 2]), valid_probabilities)
            with self.subTest(function=function.__name__, failure="probability"):
                with self.assertRaises(ValueError):
                    function(valid_labels, np.asarray([-0.1, 1.1]))
            with self.subTest(function=function.__name__, failure="nonfinite"):
                with self.assertRaises(ValueError):
                    function(valid_labels, np.asarray([math.nan, 0.5]))

    def test_rank_metrics_reject_non_vectors_and_nonfinite_scores(self):
        with self.assertRaises(ValueError):
            binary_auroc(np.asarray([[0], [1]]), np.asarray([[0.1], [0.9]]))
        with self.assertRaises(ValueError):
            binary_auroc(np.asarray([0, 1]), np.asarray([0.1, np.inf]))
        with self.assertRaises(ValueError):
            binary_auroc(np.asarray([0.0, 0.5]), np.asarray([0.1, 0.9]))
        with self.assertRaises(ValueError):
            spearman_correlation(np.asarray([1.0, np.nan]), np.asarray([1.0, 2.0]))
        with self.assertRaises(ValueError):
            expected_calibration_error(
                np.asarray([0, 1]), np.asarray([0.1, 0.9]), bins=0
            )


class ExecutionDeviceContractTests(unittest.TestCase):
    def test_deterministic_pool_matches_adaptive_average_reference(self):
        pool = _DeterministicAdaptiveAvgPool2d((4, 4))
        for shape in ((2, 3, 8, 8), (1, 2, 9, 11), (1, 1, 16, 12)):
            values = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
            expected = torch.nn.functional.adaptive_avg_pool2d(values, (4, 4))
            self.assertTrue(torch.allclose(pool(values), expected, atol=1e-6, rtol=0.0))

    def test_fixture_defaults_to_cpu_and_formal_data_rejects_cpu(self):
        fixture = SimpleNamespace(is_fixture=True)
        formal = SimpleNamespace(is_fixture=False)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CSI_PAIRS_DEVICE", None)
            self.assertEqual(resolve_execution_device(fixture), torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "requires an NVIDIA CUDA device"):
            resolve_execution_device(formal, "cpu")

    def test_cuda_request_fails_closed_when_runtime_cannot_initialize(self):
        fixture = SimpleNamespace(is_fixture=True)
        with patch("formal_v2.formal_model.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "cannot initialize CUDA"):
                resolve_execution_device(fixture, "cuda:0")

    def test_cuda_index_is_checked_against_visible_inventory(self):
        fixture = SimpleNamespace(is_fixture=True)
        with (
            patch("formal_v2.formal_model.torch.cuda.is_available", return_value=True),
            patch("formal_v2.formal_model.torch.cuda.device_count", return_value=2),
        ):
            self.assertEqual(resolve_execution_device(fixture, "cuda:1"), torch.device("cuda:1"))
            with self.assertRaisesRegex(RuntimeError, "outside the visible device inventory"):
                resolve_execution_device(fixture, "cuda:2")

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


class ExternalRuntimeRegressionTests(unittest.TestCase):
    def test_sionna_candidate_fails_closed_on_unverified_exact_regeneration(self):
        self.assertIs(SIONNA_EXACT_REGENERATION_VERIFIED, False)

    def test_sionna_stable_path_id_includes_quantized_vertices(self):
        surfaces = np.asarray((100, 102, -1), dtype=np.int64)
        vertices = np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (0.0, 0.0, 0.0)))
        base = _stable_path_id(surfaces, 12.345e-9, vertices)
        within_quantum = _stable_path_id(surfaces, 12.345e-9, vertices + 1e-7)
        distinct = _stable_path_id(
            surfaces,
            12.345e-9,
            vertices + np.asarray(((2e-5, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
        )
        self.assertEqual(base, within_quantum)
        self.assertNotEqual(base, distinct)

    def test_sionna_world_material_names_match_registered_itu_names(self):
        first = SimpleNamespace(radio_material=None)
        second = SimpleNamespace(radio_material=None)
        scene = SimpleNamespace(objects={"primitive-0": first, "primitive-1": second})
        _set_world_materials(scene, np.asarray((0, 1)))
        self.assertEqual(first.radio_material, "itu_concrete")
        self.assertEqual(second.radio_material, "itu_metal")
        _set_world_materials(scene, np.asarray((1, 0)))
        self.assertEqual(first.radio_material, "itu_glass")
        self.assertEqual(second.radio_material, "itu_wood")

    def test_sionna_scene_groups_have_distinct_initial_materials(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scene.xml"
            _write_scene_xml(path)
            root = ET.parse(path).getroot()
        materials = {
            shape.attrib["id"]: shape.find("ref").attrib["id"]
            for shape in root.findall("shape")
        }
        self.assertEqual(
            materials,
            {
                "mesh-ground": "mat-itu_wet_ground",
                "mesh-background": "mat-itu_concrete",
                "mesh-primitive-0": "mat-itu_brick",
                "mesh-primitive-1": "mat-itu_wood",
            },
        )
        self.assertEqual(len(set(materials.values())), 4)

    def test_sionna_bootstrap_prepends_authenticated_driver_and_optix(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary).resolve()
            runtime = project / "formal_v2/external_adapters/.runtime-sionna/venv"
            optix = project / (
                "formal_v2/external_adapters/.runtime-sionna/driver-libs/root/"
                "usr/lib/x86_64-linux-gnu"
            )
            python_path = runtime / "bin/python"
            driver = project / "libcuda.so.1"
            llvm = project / "libLLVM-18.so"
            python_path.parent.mkdir(parents=True)
            optix.mkdir(parents=True)
            for path in (python_path, driver, llvm, optix / "libnvoptix.so.1"):
                path.touch()
            python, environment = _sionna_bootstrap_environment(
                project,
                {
                    "LD_PRELOAD": "/tmp/other.so",
                    "LD_LIBRARY_PATH": "/tmp/lib",
                    "PYTHONPATH": "/tmp/python",
                },
                cuda_driver=driver,
                libllvm=llvm,
            )
        self.assertEqual(python, python_path)
        self.assertEqual(environment["LD_PRELOAD"].split(":")[0], str(driver))
        self.assertEqual(environment["LD_LIBRARY_PATH"].split(":")[0], str(optix))
        self.assertEqual(environment["DRJIT_LIBLLVM_PATH"], str(llvm))
        self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(environment["PYTHONPATH"].split(os.pathsep)[0], str(project))

    def test_cuda_compatibility_package_is_content_locked(self):
        self.assertEqual(
            CUDA_COMPAT_PACKAGE_SHA256,
            "7d77eb1ed96bbc4f639f7b1acea4b66883ab57deedf4c55cf506d3e7dc03a0f3",
        )
        self.assertEqual(CUDA_COMPAT_PACKAGE_URL.split("/")[-1], "cuda-compat-13-0-580.105.08-1.el8.x86_64.rpm")
        self.assertEqual(CUDA_COMPAT_FILES["libcuda.so.1"]["target"], "libcuda.so.580.105.08")
        self.assertEqual(
            CUDA_COMPAT_FILES["libcuda.so.580.105.08"]["sha256"],
            "df9183549feb062f4195e6cf130e0ef372de4a59e59dbe51554ad8c3c5b167db",
        )

    def test_configured_but_inactive_cuda_compatibility_fails_closed(self):
        with patch.dict(os.environ, {"CSI_PAIRS_CUDA_COMPAT_ROOT": "/missing"}):
            with self.assertRaisesRegex(RuntimeError, "cannot initialize CUDA"):
                _cuda_driver_record(False)

    def test_zero_property_memory_uses_allocator_inventory(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(mem_get_info=lambda index: (123, 42_949_672_960))
        )
        properties = SimpleNamespace(total_memory=0)
        self.assertEqual(
            _cuda_total_memory_bytes(fake_torch, 1, properties),
            42_949_672_960,
        )

    def test_wigatr_setup_removes_generated_editable_metadata(self):
        project = Path(__file__).resolve().parents[2]
        script = (
            project / "formal_v2" / "external_adapters" / "setup_wigatr.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("-name '*.egg-info' -exec rm -rf -- {} +", script)
        self.assertIn('ENV_DIR="$(cd "$(dirname "${ENV_DIR_INPUT}")"', script)

    def test_dry_run_uses_minimal_permitted_fixture_positions(self):
        project = Path(__file__).resolve().parents[2]
        script = (
            project / "formal_v2" / "scripts" / "run_formal_v2_dry_run.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--positions 8", script)
        self.assertIn("export OMP_NUM_THREADS=1", script)
        self.assertIn("export MKL_NUM_THREADS=1", script)
        self.assertIn("export OPENBLAS_NUM_THREADS=1", script)


if __name__ == "__main__":
    unittest.main()
