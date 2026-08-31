from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from formal_v2.formal_claims import _semantic_status
from formal_v2.formal_external_validity import (
    _render_independent_adapter_command,
    _validate_independent_runtime,
    _validate_manifest,
    _verify_independent_adapter_inputs,
    require_claim_eligible_manifest,
    require_independent_primary_engine,
)
from formal_v2.formal_io import sha256_file, write_json
from formal_v2.external_adapters.differt_external_validity import (
    SOURCE_ASSET_SCHEMA,
    _horizontal_array_offsets,
    _physical_world_bits,
    _sionna_subcarrier_offsets_hz,
    _world_face_materials,
)
from formal_v2.external_adapters.prepare_differt_external_scenes import (
    ALLOWED_DATASET_FIELDS,
    _external_scene_indices,
    _independent_adapter_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "formal_v2/external_adapters/differt_external_validity.py"
ENGINE_CONFIG = ROOT / "formal_v2/configs/differt_external_engine_v1.json"
ENGINE_REVISION = "differt@673cc58ef61906b8ab0869dd206b3d032dbc01b2"


def _manifest(**overrides):
    payload = {
        "schema_version": "csi-pairs-v6-external-validity-independent-adapter-v1",
        "evidence_type": "independent_rt_engine",
        "engine_family": "differt",
        "source_revision": ENGINE_REVISION,
        "license_id": "MIT",
        "adapter_source_path": str(ADAPTER),
        "adapter_source_sha256": sha256_file(ADAPTER),
        "engine_config_path": "engine-config.json",
        "engine_config_sha256": "a" * 64,
        "rt_scene_manifest_path": "rt_scene_manifest.json",
        "rt_scene_manifest_sha256": "b" * 64,
        "command": [
            "{project_root}/formal_v2/external_adapters/.runtime-differt/venv/bin/python",
            "{adapter_source}",
            "--dataset",
            "{dataset}",
            "--output",
            "{output}",
            "--scene-manifest",
            "{scene_manifest}",
            "--engine-config",
            "{engine_config}",
        ],
    }
    payload.update(overrides)
    return payload


class DiffeRTExternalValidityTests(unittest.TestCase):
    def test_scene_preparation_emits_a_valid_claim_eligible_adapter_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "engine_config.json"
            scene = root / "rt_scene_manifest.json"
            config.write_bytes(ENGINE_CONFIG.read_bytes())
            scene.write_bytes(b"authenticated-scene-manifest")
            manifest = _independent_adapter_manifest(config, scene)
        _validate_manifest(manifest)
        require_claim_eligible_manifest(manifest)
        self.assertEqual(manifest["engine_config_path"], config.name)
        self.assertEqual(manifest["rt_scene_manifest_path"], scene.name)
        self.assertEqual(manifest["adapter_source_path"], ADAPTER.relative_to(ROOT).as_posix())

    def test_scene_preparation_accepts_full_v4_scope_and_enforces_frozen_minimum(self):
        roles = np.asarray(
            ["source_encoder_train"] * 8 + ["external_validation"] * 32
        )
        indices = _external_scene_indices(roles, 32)
        self.assertEqual(indices.tolist(), list(range(8, 40)))
        with self.assertRaisesRegex(RuntimeError, "found 32"):
            _external_scene_indices(roles, 33)
        with self.assertRaisesRegex(ValueError, "at least two"):
            _external_scene_indices(roles, 1)

    def test_subcarrier_grid_matches_frozen_sionna_even_fft_order(self):
        self.assertEqual(
            _sionna_subcarrier_offsets_hz(4, 30_000.0).tolist(),
            [-60_000.0, -30_000.0, 0.0, 30_000.0],
        )
        self.assertEqual(
            _sionna_subcarrier_offsets_hz(16, 1_000_000.0).tolist(),
            [float(value) for value in range(-8_000_000, 8_000_000, 1_000_000)],
        )

    def test_wideband_four_primitive_world_materials_follow_bit_mapping(self):
        bounds = np.asarray(
            [[0, 2], [2, 8], [8, 10], [10, 12], [12, 14], [14, 16]],
            dtype=np.int64,
        )
        mapping = np.asarray([2, 0, 3, 1], dtype=np.int64)
        bits = np.asarray([1, 0, 1, 0], dtype=np.int64)
        materials = _world_face_materials(bounds, 16, mapping, bits)
        np.testing.assert_array_equal(materials[:2], np.zeros(2, dtype=np.int64))
        np.testing.assert_array_equal(materials[2:8], np.ones(6, dtype=np.int64))
        np.testing.assert_array_equal(
            materials[8:], np.asarray([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        )

    def test_anchor_is_applied_before_four_primitive_mapping(self):
        bounds = np.asarray(
            [[0, 2], [2, 8], [8, 10], [10, 12], [12, 14], [14, 16]],
            dtype=np.int64,
        )
        mapping = np.asarray([2, 0, 3, 1], dtype=np.int64)
        logical = np.asarray([0, 1, 1, 0], dtype=np.int64)
        anchor = np.asarray([1, 1, 0, 1], dtype=np.int64)
        physical = _physical_world_bits(logical, anchor)
        np.testing.assert_array_equal(physical, np.asarray([1, 0, 1, 1]))
        materials = _world_face_materials(bounds, 16, mapping, physical)
        np.testing.assert_array_equal(
            materials[8:], np.asarray([1, 1, 2, 2, 2, 2, 2, 2], dtype=np.int64)
        )
        self.assertEqual(
            SOURCE_ASSET_SCHEMA, "csi-pairs-v6-differt-source-asset-v3"
        )
        self.assertIn("anchor_bits", ALLOWED_DATASET_FIELDS)

    def test_horizontal_half_wavelength_array_generalizes_from_two_antennas(self):
        offsets = _horizontal_array_offsets(2, 0.4)
        np.testing.assert_allclose(
            offsets,
            np.asarray([[0.0, -0.1, 0.0], [0.0, 0.1, 0.0]]),
        )

    def test_executable_manifest_is_claim_eligible_and_binds_staged_inputs(self):
        manifest = _manifest()
        _validate_manifest(manifest)
        require_claim_eligible_manifest(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = _render_independent_adapter_command(
                manifest,
                dataset=root / "dataset.npz",
                output=root / "output",
                output_root=root,
                adapter_source=ADAPTER,
                engine_config=root / "engine-config.json",
                scene_manifest=root / "scene.json",
            )
        for flag in ("--dataset", "--output", "--engine-config", "--scene-manifest"):
            self.assertEqual(command.count(flag), 1)
            self.assertTrue(Path(command[command.index(flag) + 1]).is_absolute())

        changed = _manifest()
        changed["command"][changed["command"].index("--engine-config") + 1] = (
            "unbound-config.json"
        )
        with self.assertRaisesRegex(ValueError, "engine-config"):
            _validate_manifest(changed)

    def test_same_primary_engine_is_rejected(self):
        dataset = SimpleNamespace(
            is_fixture=False,
            engine_config={
                "engine": {
                    "name": "DiffeRT exhaustive RT",
                    "source_revision": "some-other-differt-revision",
                }
            },
        )
        with self.assertRaisesRegex(RuntimeError, "not independent"):
            require_independent_primary_engine(dataset, _manifest())

    def test_engine_config_scene_and_source_assets_are_authenticated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset.npz"
            dataset_path.write_bytes(b"formal-dataset")
            config_path = root / "engine-config.json"
            config_path.write_bytes(ENGINE_CONFIG.read_bytes())
            asset_path = root / "world-0.npz"
            asset_path.write_bytes(b"independent-source-asset")
            dataset = SimpleNamespace(
                source_path=dataset_path,
                scene_ids=np.asarray(["external-a"]),
                world_count=1,
                canonical_map_sha256=np.asarray([["1" * 64]]),
                indices_for_role=lambda role: np.asarray([0], dtype=np.int64),
            )
            scene = {
                "schema_version": "csi-pairs-v6-independent-rt-scene-manifest-v1",
                "dataset_sha256": sha256_file(dataset_path),
                "engine_family": "differt",
                "engine_name": "DiffeRT",
                "engine_revision": ENGINE_REVISION,
                "configuration_sha256": sha256_file(config_path),
                "license_id": "MIT",
                "worlds": [
                    {
                        "scene_id": "external-a",
                        "world": 0,
                        "canonical_map_sha256": "1" * 64,
                        "source_asset_id": "external-a-world-0",
                        "source_asset_path": asset_path.name,
                        "source_asset_sha256": sha256_file(asset_path),
                    }
                ],
            }
            scene_path = root / "rt_scene_manifest.json"
            write_json(scene_path, scene)
            manifest = _manifest(
                engine_config_path=config_path.name,
                engine_config_sha256=sha256_file(config_path),
                rt_scene_manifest_path=scene_path.name,
                rt_scene_manifest_sha256=sha256_file(scene_path),
            )
            resolved = _verify_independent_adapter_inputs(manifest, root, dataset)
            self.assertEqual(resolved[0], scene_path.resolve())
            self.assertEqual(resolved[1], config_path.resolve())
            self.assertEqual(resolved[3], (asset_path.resolve(),))

            asset_path.write_bytes(b"tampered-source-asset")
            with self.assertRaisesRegex(RuntimeError, "hash-mismatched"):
                _verify_independent_adapter_inputs(manifest, root, dataset)
            asset_path.write_bytes(b"independent-source-asset")
            config_path.write_bytes(config_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(RuntimeError, "hash-mismatched"):
                _verify_independent_adapter_inputs(manifest, root, dataset)

    def test_runtime_provenance_digest_is_recomputed(self):
        versions = {
            "differt": "0.10.0",
            "differt-core": "0.10.0",
            "jax": "0.11.0",
            "jaxlib": "0.11.0",
            "numpy": "2.5.2",
            "warp-lang": "1.16.0",
        }
        record = {
            "schema_version": "csi-pairs-v6-differt-runtime-provenance-v1",
            "engine_family": "differt",
            "engine_name": "DiffeRT",
            "engine_revision": ENGINE_REVISION,
            "license_id": "MIT",
            "python_executable": "/frozen/bin/python",
            "python_prefix": "/frozen",
            "python_version": "3.12.13",
            "python_implementation": "CPython",
            "platform_system": "Linux",
            "platform_release": "test-kernel",
            "platform_machine": "x86_64",
            "jax_enable_x64": True,
            "jax_platforms": "cpu",
            "jax_devices": ["cpu:0"],
            "package_records": {
                name: {"version": version, "record_sha256": "c" * 64}
                for name, version in versions.items()
            },
            "requirements_sha256": "97afdca61c657fa23f1eeae3f64bfb37b54a048411c9b25edd230293bdecc573",
            "engine_provenance_sha256": "317ab9e495e6dee722a37bc471db77e2eedc9b3c6d699732f0d6c008bba0b2ed",
            "engine_config_sha256": "f" * 64,
        }
        record["environment_sha256"] = hashlib.sha256(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        _validate_independent_runtime(
            record,
            executable="/frozen/bin/python",
            engine_config_sha256="f" * 64,
        )
        record["environment_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
            _validate_independent_runtime(
                record,
                executable="/frozen/bin/python",
                engine_config_sha256="f" * 64,
            )

    def test_claim_semantics_accept_executable_independent_but_reject_archive(self):
        payload = {
            "status": "PASS",
            "passed": True,
            "execution_mode": "authenticated_independent_rt_adapter",
            "engine_family": "differt",
            "adapter_source_sha256": "a" * 64,
            "active_direction_cluster_count": 2,
            "active_direction_agreement_ci95_low": 0.85,
            "minimum_active_direction_agreement": 0.8,
            "rt_scene_manifest_path": "rt_scene_manifest.json",
            "rt_scene_manifest_sha256": "b" * 64,
            "external_csi_path": "external_csi.npz",
            "external_csi_sha256": "c" * 64,
            "external_engine_config_path": "external_engine_config.bin",
            "external_engine_config_sha256": "d" * 64,
            "external_csi_contract": "outer-recomputed-direction-and-effect-from-raw-csi-v1",
            "external_scene_count": 4,
            "external_runtime_provenance_path": "runtime_provenance.json",
            "external_runtime_provenance_sha256": "e" * 64,
            "external_runtime_environment_sha256": "f" * 64,
            "external_runtime_provenance": {"environment_sha256": "f" * 64},
            "null_equivalence": {"passed": True, "base_map_cluster_count": 2},
        }
        self.assertEqual(_semantic_status("G8", payload), "PASS")
        payload["execution_mode"] = "authenticated_precomputed_rt_archive"
        self.assertEqual(_semantic_status("G8", payload), "FAIL")


if __name__ == "__main__":
    unittest.main()
