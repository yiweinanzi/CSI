from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from formal_v2.formal_cli import main as formal_cli_main
from formal_v2.formal_config import load_formal_config
from formal_v2.formal_data_verification import (
    _validate_manifest,
    require_data_verification,
    require_verified_roles_from_root,
    run_data_verification,
)
from formal_v2.formal_dataset import FormalDataset
from formal_v2.formal_evidence import evidence_context
from formal_v2.formal_fixture import write_nonscientific_fixture
from formal_v2.formal_io import read_strict_json, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "formal_v2/configs/formal_v2_smoke.json"
FIXTURE_VERIFIER = ROOT / "formal_v2/formal_fixture_verifier.py"


class DataVerifierAuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_path = write_nonscientific_fixture(self.root / "fixture.npz")
        self.dataset = FormalDataset.load(self.dataset_path)
        self.config = load_formal_config(CONFIG)
        self.source = self.root / "fixture_verifier.py"
        shutil.copyfile(FIXTURE_VERIFIER, self.source)
        self.manifest_path = self.root / "verifier.json"
        self._write_manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def _manifest(self):
        return {
            "schema_version": "csi-pairs-v6-data-verifier-v1",
            "command": [
                "{python}",
                "{verifier_source}",
                "--dataset",
                "{dataset}",
                "--output",
                "{output}",
            ],
            "verifier_source_path": self.source.name,
            "verifier_source_sha256": sha256_file(self.source),
            "engine_source_revision": "fixture-source-revision",
            "engine_license_id": "GENERATED-FIXTURE-NO-EXTERNAL-ASSET",
            "asset_license_ids": ["GENERATED-FIXTURE-NO-EXTERNAL-ASSET"],
            "rtol": 0.0,
            "atol": 0.0,
        }

    def _write_manifest(self):
        write_json(self.manifest_path, self._manifest())

    def _verified_run(self, name="run"):
        output = self.root / name
        gate = run_data_verification(
            self.config,
            self.dataset,
            self.manifest_path,
            output,
        )
        return output, gate

    def test_manifest_rejects_legacy_or_indirect_commands(self):
        manifest = self._manifest()
        legacy = dict(manifest)
        legacy.pop("verifier_source_path")
        legacy.pop("verifier_source_sha256")
        with self.assertRaisesRegex(ValueError, "fields must be exact"):
            _validate_manifest(legacy, self.dataset)

        forged = dict(manifest)
        forged["command"] = [
            "{python}",
            "-c",
            "from pathlib import Path; Path(r'{output}/regenerated.npz').touch()",
            "{verifier_source}",
            "--dataset",
            "{dataset}",
        ]
        with self.assertRaisesRegex(ValueError, "directly execute"):
            _validate_manifest(forged, self.dataset)

        duplicated = dict(manifest)
        duplicated["command"] = [*manifest["command"], "{dataset}"]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            _validate_manifest(duplicated, self.dataset)

    def test_gate_binds_manifest_and_source_and_downstream_reauthenticates(self):
        output, gate = self._verified_run()
        gate_path = output / "data_verification/gate.json"
        self.assertEqual(gate["verifier_source_path"], str(self.source.resolve()))
        self.assertEqual(gate["verifier_source_sha256"], sha256_file(self.source))
        self.assertEqual(
            gate["verifier_manifest_sha256"],
            sha256_file(gate["verifier_manifest_path"]),
        )
        disk_gate = read_strict_json(gate_path)
        self.assertIs(
            require_data_verification(
                disk_gate,
                self.config,
                self.dataset,
                gate_path=gate_path,
            ),
            disk_gate,
        )
        require_verified_roles_from_root(output, self.config, self.dataset, ("target",))

    def test_verifier_detects_regenerated_phase_reference_tampering_per_scene(self):
        mutations = {
            "phase_reference_values": (
                "arrays['phase_reference_values'] = arrays['phase_reference_values'].copy()\n"
                "    arrays['phase_reference_values'][0, 0] *= np.exp(0.25j)"
            ),
            "phase_reference_source_sha256": (
                "arrays['phase_reference_source_sha256'] = "
                "arrays['phase_reference_source_sha256'].astype('U64').copy()\n"
                "    arrays['phase_reference_source_sha256'][0, 0] = '0' * 64"
            ),
        }
        marker = "    np.savez_compressed(target, **arrays)"
        original = FIXTURE_VERIFIER.read_text(encoding="utf-8")
        self.assertIn(marker, original)
        for field, mutation in mutations.items():
            with self.subTest(field=field):
                self.source.write_text(
                    original.replace(marker, f"    {mutation}\n{marker}"),
                    encoding="utf-8",
                )
                self._write_manifest()
                with patch(
                    "formal_v2.formal_data_verification.evidence_context",
                    return_value={},
                ):
                    output, gate = self._verified_run(f"tampered-{field}")
                self.assertFalse(gate["passed"])
                with (output / "data_verification/per_scene.csv").open(
                    newline="", encoding="utf-8"
                ) as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(rows[0][f"{field}_match"], "False")

    def test_source_tampering_blocks_qualify_cli_and_downstream(self):
        output, _ = self._verified_run()
        self.source.write_text("raise SystemExit('tampered')\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "source hash mismatch"):
            require_verified_roles_from_root(
                output, self.config, self.dataset, ("target",)
            )
        self.assertEqual(
            formal_cli_main(
                [
                    "qualify",
                    "--config",
                    str(CONFIG),
                    "--dataset",
                    str(self.dataset_path),
                    "--output",
                    str(output),
                ]
            ),
            2,
        )
        self.assertFalse((output / "qualification/gate.json").exists())

    def test_handwritten_or_manifest_tampered_gate_is_rejected(self):
        context = evidence_context(self.config, self.dataset, "FORBIDDEN")
        handwritten = {
            "schema_version": "csi-pairs-v6-data-verification-gate-v1",
            "passed": True,
            "blocking_passed": True,
            **context,
            "blocking_roles": ["source_encoder_train", "source_method_selection"],
            "target_and_other_roles_are_nonblocking": True,
            "role_status": {
                "source_encoder_train": "PASS",
                "source_method_selection": "PASS",
            },
        }
        with self.assertRaisesRegex(RuntimeError, "authenticated verifier fields"):
            require_data_verification(handwritten, self.config, self.dataset)

        output, gate = self._verified_run("tampered-run")
        bound = Path(gate["verifier_manifest_path"])
        payload = json.loads(bound.read_text(encoding="utf-8"))
        payload["rtol"] = 0.5
        write_json(bound, payload)
        gate_path = output / "data_verification/gate.json"
        with self.assertRaisesRegex(RuntimeError, "manifest hash mismatch"):
            require_data_verification(
                read_strict_json(gate_path),
                self.config,
                self.dataset,
                gate_path=gate_path,
            )


if __name__ == "__main__":
    unittest.main()
