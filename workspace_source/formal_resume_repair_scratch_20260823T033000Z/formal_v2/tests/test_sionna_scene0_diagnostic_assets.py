from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from formal_v2 import sionna_osm_candidate as candidate
from formal_v2.sionna_scene0_diagnostic_assets import (
    ASSET_RELATIVE_PATHS,
    DIAGNOSTIC_ASSET_SCHEMA,
    EXPECTED_SCENE,
    _byte_comparison,
    _read_ascii_triangle_ply,
    load_diagnostic_asset_manifest,
)


class Scene0DiagnosticAssetTests(unittest.TestCase):
    def _write_manifest(self, root: Path) -> tuple[Path, dict]:
        config = root / "generator_config.json"
        config.write_text("{}\n", encoding="ascii")
        bank_root = root / "banks" / EXPECTED_SCENE["scene_id"]
        (bank_root / "mesh").mkdir(parents=True)
        files = []
        for relative in ASSET_RELATIVE_PATHS:
            path = bank_root / relative
            path.write_text(f"{relative}\n", encoding="ascii")
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": candidate.sha256_file(path),
                }
            )
        row = {
            **EXPECTED_SCENE,
            "bank_record_path": str((bank_root / "bank.json").relative_to(root)),
            "scene_xml_path": str((bank_root / "scene.xml").relative_to(root)),
            "files": files,
        }
        manifest = {
            "schema_version": DIAGNOSTIC_ASSET_SCHEMA,
            "simulation_not_measurement": True,
            "scientific_use": "DIAGNOSTIC_NOT_FORMAL_EVIDENCE",
            "config_path": config.name,
            "config_sha256": candidate.sha256_file(config),
            "banks": [row],
        }
        manifest_path = root / "diagnostic_asset_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="ascii")
        return manifest_path, manifest

    def test_ascii_triangle_ply_is_fully_readable(self) -> None:
        triangles = np.asarray(
            (((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),),
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "triangle.ply"
            candidate._write_triangle_ply(path, triangles)
            report = _read_ascii_triangle_ply(path)
        self.assertEqual(report["vertex_count"], 3)
        self.assertEqual(report["face_count"], 1)
        self.assertTrue(report["all_values_finite"])
        self.assertTrue(report["all_faces_triangular"])

    def test_ascii_triangle_ply_rejects_out_of_range_face(self) -> None:
        malformed = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
0 1 0
3 0 1 3
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.ply"
            path.write_text(malformed, encoding="ascii")
            with self.assertRaisesRegex(ValueError, "invalid vertex indices"):
                _read_ascii_triangle_ply(path)

    def test_byte_comparison_reports_identity_and_first_difference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left"
            right = root / "right"
            left.write_bytes(b"abcdef")
            right.write_bytes(b"abcdef")
            identical = _byte_comparison(left, right)
            self.assertTrue(identical["byte_identical"])
            self.assertIsNone(identical["first_differing_byte_offset"])
            right.write_bytes(b"abcXefg")
            different = _byte_comparison(left, right)
            self.assertFalse(different["byte_identical"])
            self.assertEqual(different["first_differing_byte_offset"], 3)
            self.assertEqual(different["differing_byte_count_with_size_delta"], 2)

    def test_diagnostic_loader_requires_exact_hashed_scene_file_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = self._write_manifest(root)
            with patch.object(candidate, "load_config", return_value={}):
                loaded_root, loaded, _ = load_diagnostic_asset_manifest(root)
            self.assertEqual(loaded_root, root.resolve())
            self.assertEqual(loaded, manifest)

            manifest["banks"][0]["files"].pop()
            manifest_path.write_text(json.dumps(manifest), encoding="ascii")
            with (
                patch.object(candidate, "load_config", return_value={}),
                self.assertRaisesRegex(ValueError, "exact six scene files"),
            ):
                load_diagnostic_asset_manifest(root)

            manifest_path.write_text(json.dumps(loaded), encoding="ascii")
            changed = root / loaded["banks"][0]["files"][1]["path"]
            changed.write_text("changed\n", encoding="ascii")
            with (
                patch.object(candidate, "load_config", return_value={}),
                self.assertRaisesRegex(ValueError, "diagnostic asset file changed"),
            ):
                load_diagnostic_asset_manifest(root)


if __name__ == "__main__":
    unittest.main()
