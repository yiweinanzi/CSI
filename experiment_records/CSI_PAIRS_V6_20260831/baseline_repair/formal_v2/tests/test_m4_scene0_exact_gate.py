from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
TOOL_PATH = REPOSITORY_ROOT / "artifacts/m4_local_sim/tools/verify_scene0_exact_gate.py"
REPLAY_TOOL_PATH = (
    REPOSITORY_ROOT / "artifacts/formal_readiness/tools/compare_sionna_shard_replay.py"
)
REGENERATION_TOOL_PATH = REPLAY_TOOL_PATH.with_name("compare_sionna_regeneration.py")
REGENERATION_TOOL_SHA256 = (
    "6d707c836b97149dc4e435289b6bc79ab7d39ffa2b2240790ef1830996c1b3a6"
)
SPEC = importlib.util.spec_from_file_location("m4_scene0_exact_gate", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load exact gate: {TOOL_PATH}")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
REGENERATION_SPEC = importlib.util.spec_from_file_location(
    "compare_sionna_regeneration", REGENERATION_TOOL_PATH
)
if REGENERATION_SPEC is None or REGENERATION_SPEC.loader is None:
    raise RuntimeError(f"cannot load regeneration comparator: {REGENERATION_TOOL_PATH}")
regeneration = importlib.util.module_from_spec(REGENERATION_SPEC)
REGENERATION_SPEC.loader.exec_module(regeneration)


class M4Scene0ExactGateTests(unittest.TestCase):
    def test_regeneration_status_requires_every_registered_array(self) -> None:
        fields = {
            "csi_clean", "csi_repeat", "path_ids", "path_power",
            "path_surface_ids", "noop_path_ids", "noop_path_power",
            "noop_path_surface_ids",
        }
        reference = {name: np.ones((2,), dtype=np.float64) for name in fields}
        regenerated = {name: value.copy() for name, value in reference.items()}
        self.assertTrue(
            regeneration._required_arrays_exact(reference, regenerated, fields)
        )
        regenerated["noop_path_power"][0] = 2.0
        self.assertFalse(
            regeneration._required_arrays_exact(reference, regenerated, fields)
        )

    def test_repository_replay_cli_has_frozen_dependency(self) -> None:
        self.assertTrue(REGENERATION_TOOL_PATH.is_file())
        self.assertEqual(gate._sha256(REGENERATION_TOOL_PATH), REGENERATION_TOOL_SHA256)
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(REPLAY_TOOL_PATH.parent)
        result = subprocess.run(
            [sys.executable, "-P", str(REPLAY_TOOL_PATH), "--help"],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--scene-index", result.stdout)

    def _evidence(self, root: Path) -> tuple[Path, Path, Path]:
        left = root / "process-a.npz"
        right = root / "process-b.npz"
        python = root / "runtime/bin/python"
        llvm = root / "runtime/lib/libLLVM.dylib"
        asset_manifest = root / "assets/diagnostic_asset_manifest.json"
        generator = root / "sionna_osm_candidate.py"
        renderer = root / "render_sionna_bank_backend_diagnostic.py"
        for path, content in (
            (python, b"python-runtime"),
            (llvm, b"reviewed-llvm"),
            (asset_manifest, b"asset-manifest"),
            (generator, b"generator-source"),
            (renderer, b"renderer-source"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        arrays = {
            name: np.ones((1,), dtype=np.float64)
            for name in sorted(gate.EXPECTED_NPZ_FIELDS)
        }
        arrays["scene_indices"] = np.asarray((0,), dtype=np.int64)
        arrays["path_ids"] = np.asarray((7,), dtype=np.int64)
        arrays["path_power"] = np.asarray((1.0,), dtype=np.float64)
        np.savez(left, **arrays)
        shutil.copyfile(left, right)
        left_hash = gate._sha256(left)
        right_hash = gate._sha256(right)
        runtime = {
            **gate.EXPECTED_RUNTIME,
            "python_executable": str(python.resolve()),
            "drjit_libllvm_path": str(llvm.resolve()),
            "drjit_libllvm_sha256": gate._sha256(llvm),
        }
        common = {
            "schema_version": "csi-pairs-sionna-bank-backend-diagnostic-v2",
            "status": gate.SCIENTIFIC_USE,
            "simulation_not_measurement": True,
            "scientific_use": gate.SCIENTIFIC_USE,
            "backend": "llvm",
            "mitsuba_variant": "llvm_ad_mono_polarized",
            "drjit_thread_count": 1,
            "scene_index": 0,
            "scene_id": "osm-sionna-source-chicago-bank-00",
            "asset_manifest_path": str(asset_manifest.resolve()),
            "asset_manifest_sha256": gate._sha256(asset_manifest),
            "generator_path": str(generator.resolve()),
            "generator_sha256": gate._sha256(generator),
            "tool_path": str(renderer.resolve()),
            "tool_sha256": gate._sha256(renderer),
            "runtime": runtime,
        }
        manifests = (
            {
                **common,
                "run_id": "a" * 32,
                "started_utc": "2026-08-09T00:00:00+00:00",
                "ended_utc": "2026-08-09T00:01:00+00:00",
                "output_path": str(left.resolve()),
                "output_bytes": left.stat().st_size,
                "output_sha256": left_hash,
            },
            {
                **common,
                "run_id": "b" * 32,
                "started_utc": "2026-08-09T00:02:00+00:00",
                "ended_utc": "2026-08-09T00:03:00+00:00",
                "output_path": str(right.resolve()),
                "output_bytes": right.stat().st_size,
                "output_sha256": right_hash,
            },
        )
        for path, manifest in zip((left, right), manifests):
            path.with_suffix(".manifest.json").write_text(
                json.dumps(manifest), encoding="ascii"
            )
        comparison = root / "comparison.json"
        comparison.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "process_a": {
                        "path": str(left.resolve()),
                        "bytes": left.stat().st_size,
                        "sha256": left_hash,
                    },
                    "process_b": {
                        "path": str(right.resolve()),
                        "bytes": right.stat().st_size,
                        "sha256": right_hash,
                    },
                    "arrays": {
                        name: {"exact": True}
                        for name in sorted(gate.EXPECTED_NPZ_FIELDS)
                    },
                }
            ),
            encoding="ascii",
        )
        return left, right, comparison

    def test_distinct_bound_runs_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            left, right, comparison = self._evidence(Path(temporary))
            report = gate.verify(left, right, comparison)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["checks"]["comparator_inputs_bound"])
        self.assertTrue(report["runtime_manifest_checks"]["distinct_run_ids"])

    def test_same_output_cannot_impersonate_two_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            left, _, comparison = self._evidence(Path(temporary))
            with self.assertRaisesRegex(ValueError, "distinct output paths"):
                gate.verify(left, left, comparison)

    def test_run_runtime_and_asset_identity_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            left, right, comparison = self._evidence(Path(temporary))
            manifest_path = right.with_suffix(".manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
            manifest["run_id"] = "a" * 32
            manifest["runtime"]["sionna_rt"] = "unreviewed"
            manifest["asset_manifest_sha256"] = "9" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="ascii")
            report = gate.verify(left, right, comparison)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["runtime_manifest_checks"]["distinct_run_ids"])
        self.assertFalse(report["runtime_manifest_checks"]["frozen_runtime"])
        self.assertFalse(report["runtime_manifest_checks"]["same_asset_manifest"])

    def test_bound_source_or_runtime_bytes_cannot_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            left, right, comparison = self._evidence(Path(temporary))
            manifest = json.loads(
                left.with_suffix(".manifest.json").read_text(encoding="ascii")
            )
            Path(manifest["runtime"]["drjit_libllvm_path"]).write_bytes(b"changed")
            Path(manifest["tool_path"]).write_bytes(b"changed-tool")
            report = gate.verify(left, right, comparison)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["runtime_manifest_checks"]["frozen_runtime"])
        self.assertFalse(report["runtime_manifest_checks"]["same_generator_and_tool"])


if __name__ == "__main__":
    unittest.main()
