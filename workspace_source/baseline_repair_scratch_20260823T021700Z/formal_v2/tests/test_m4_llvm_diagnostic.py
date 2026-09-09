from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from formal_v2 import render_sionna_bank_backend_diagnostic as diagnostic


class M4LLVMBackendDiagnosticTests(unittest.TestCase):
    def test_asset_manifest_record_binds_digest_to_absolute_path(self):
        payload = b'{"schema_version":"asset-test"}\n'
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "asset_manifest.json"
            manifest.write_bytes(payload)
            record = diagnostic._asset_manifest_record(Path(temporary))
        self.assertEqual(record["asset_manifest_path"], str(manifest.resolve()))
        self.assertEqual(
            record["asset_manifest_sha256"], hashlib.sha256(payload).hexdigest()
        )

    def test_llvm_path_requires_fixed_llvm_inputs_without_cuda(self):
        registry = diagnostic.candidate._read_json(
            Path(diagnostic.__file__).resolve().parent
            / "configs/sionna_llvm_approved_v1.json"
        )
        registered = {
            (row["sha256"], row["provenance"]) for row in registry["libraries"]
        }
        self.assertIn(
            (
                "26273678e919e90006fe2f5fc6e020cfc11a428103494d4dad6fa211b5d50451",
                "M4 LLVM 18.1.8 Homebrew arm64 scene-0 two-process exact replay audited on 2026-08-09",
            ),
            registered,
        )
        self.assertIn(
            (
                "e514c689a4469887f30396826cec7559ad6ddc1d9db1a0b243790bee7725ca88",
                "M4 LLVM 22.1.8 diagnostic runtime audited on 2026-08-09",
            ),
            registered,
        )
        self.assertIn(
            (
                "6e9dad310fa8fa8116b221e4e1212dbad2936a3d12f158c9c30887d80a631977",
                "Linux LLVM 18.1.8 two-process scene-0 exact replay: 23/23 arrays, NPZ payload, 2262 stable path signatures, and CSI all exact at zero tolerance; audited on 2026-08-10",
            ),
            registered,
        )
        args = argparse.Namespace(backend="llvm", drjit_threads=1)
        with tempfile.TemporaryDirectory() as temporary:
            llvm_path = Path(temporary) / "libLLVM.dylib"
            llvm_path.touch()
            environment = {"DRJIT_LIBLLVM_PATH": str(llvm_path)}
            with (
                patch.dict(os.environ, environment, clear=True),
                patch.object(
                    diagnostic, "_runtime_versions",
                    return_value=diagnostic._expected_runtime_versions(),
                ),
                patch(
                    "formal_v2.sionna_runtime_lock.approved_library_record",
                    return_value={"libllvm_path": str(llvm_path.resolve())},
                ),
                patch.object(
                    diagnostic.candidate,
                    "ensure_sionna_runtime",
                    side_effect=AssertionError("LLVM must not bootstrap CUDA"),
                ),
            ):
                self.assertEqual(
                    diagnostic._prepare_backend(args, ["diagnostic.py"]),
                    llvm_path.resolve(),
                )

    def test_cuda_backend_is_not_exposed_without_an_authenticated_setup(self):
        args = argparse.Namespace(backend="cuda", drjit_threads=None)
        with self.assertRaisesRegex(ValueError, "only the LLVM backend"):
            diagnostic._prepare_backend(args, ["diagnostic.py"])


if __name__ == "__main__":
    unittest.main()
