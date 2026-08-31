from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from formal_v2.formal_cli import main as formal_cli_main
from formal_v2.formal_config import load_formal_config
from formal_v2.formal_data_verification import (
    REGENERATED_FIELDS,
    _compare_regenerated_archive,
    _validate_manifest,
    require_data_verification,
    require_verified_roles_from_root,
    run_data_verification,
)
from formal_v2.formal_dataset import FormalDataset
from formal_v2.formal_evidence import config_sha256, evidence_context, _source_tree_sha256
from formal_v2.formal_fixture import write_nonscientific_fixture
from formal_v2.formal_io import read_strict_json, sha256_file, write_json
from formal_v2.formal_precomputed_regeneration_verifier import validate_receipt


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "formal_v2/configs/formal_v2_smoke.json"
FIXTURE_VERIFIER = ROOT / "formal_v2/formal_fixture_verifier.py"
PRECOMPUTED_VERIFIER = ROOT / "formal_v2/formal_precomputed_regeneration_verifier.py"


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

    def _precomputed_bundle(self, origin, gate):
        stage = origin / "data_verification"
        bundle = self.root / "portable-source"
        bundle.mkdir()
        shutil.copyfile(stage / "regenerated.npz", bundle / "regenerated.npz")
        shutil.copyfile(PRECOMPUTED_VERIFIER, bundle / PRECOMPUTED_VERIFIER.name)
        registry = ROOT / "formal_v2/configs/sionna_llvm_approved_v1.json"
        renderer = {
            "schema_version": "csi-pairs-sionna-renderer-runtime-receipt-v1",
            "profile": "sionna",
            "platform_system": "Darwin",
            "platform_machine": "arm64",
            "python_version": "3.12.13",
            "environment_sha256": "1" * 64,
            "critical_distributions": {
                "drjit": {"version": "1.2.0", "record_sha256": "2" * 64},
                "h5py": {"version": "3.15.1", "record_sha256": "3" * 64},
                "mitsuba": {"version": "3.7.1", "record_sha256": "4" * 64},
                "sionna": {"version": "2.0.1", "record_sha256": "5" * 64},
                "sionna-rt": {"version": "1.2.1", "record_sha256": "6" * 64},
                "torch": {"version": "2.9.1", "record_sha256": "7" * 64},
            },
            "lock_files": {"sionna_approved_libllvm_registry": sha256_file(registry)},
            "libllvm_sha256": "e514c689a4469887f30396826cec7559ad6ddc1d9db1a0b243790bee7725ca88",
            "libllvm_registry_sha256": sha256_file(registry),
            "libllvm_approval_provenance": "M4 LLVM 22.1.8 diagnostic runtime audited on 2026-08-09",
        }
        receipt = {
            "schema_version": "csi-pairs-v6-precomputed-regeneration-receipt-v1",
            "status": "PASS",
            "fixture": True,
            "scientific_use": "FORBIDDEN",
            "dataset_sha256": sha256_file(self.dataset_path),
            "dataset_bytes": self.dataset_path.stat().st_size,
            "config_sha256": config_sha256(self.config),
            "source_tree_sha256": _source_tree_sha256(),
            "origin_gate_sha256": sha256_file(stage / "gate.json"),
            "origin_manifest_sha256": sha256_file(stage / "manifest.json"),
            "origin_per_scene_sha256": sha256_file(stage / "per_scene.csv"),
            "origin_verifier_source_sha256": gate["verifier_source_sha256"],
            "regenerated_path": "regenerated.npz",
            "regenerated_sha256": sha256_file(stage / "regenerated.npz"),
            "regenerated_bytes": (stage / "regenerated.npz").stat().st_size,
            "scene_count": self.dataset.scene_count,
            "role_status": gate["role_status"],
            "rtol": 0.0,
            "atol": 0.0,
            "engine_source_revision": gate["engine_source_revision"],
            "engine_license_id": gate["engine_license_id"],
            "asset_license_ids": gate["asset_license_ids"],
            "renderer_runtime": renderer,
        }
        receipt_path = bundle / "verification_receipt.json"
        write_json(receipt_path, receipt)
        manifest = {
            "schema_version": "csi-pairs-v6-precomputed-data-verifier-v1",
            "command": [
                "{python}",
                "{verifier_source}",
                "--dataset",
                "{dataset}",
                "--output",
                "{output}",
                "--receipt",
                "{verification_receipt}",
            ],
            "verifier_source_path": PRECOMPUTED_VERIFIER.name,
            "verifier_source_sha256": sha256_file(PRECOMPUTED_VERIFIER),
            "verification_receipt_path": receipt_path.name,
            "verification_receipt_sha256": sha256_file(receipt_path),
            "engine_source_revision": gate["engine_source_revision"],
            "engine_license_id": gate["engine_license_id"],
            "asset_license_ids": gate["asset_license_ids"],
            "rtol": 0.0,
            "atol": 0.0,
        }
        write_json(bundle / "verifier.json", manifest)
        return bundle

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

    def test_live_gate_cannot_be_relocated_with_stale_absolute_bindings(self):
        output, _gate = self._verified_run("live-origin")
        relocated = self.root / "live-relocated"
        shutil.copytree(output, relocated)
        gate_path = relocated / "data_verification/gate.json"
        with self.assertRaisesRegex(
            RuntimeError,
            "does not bind its colocated verifier manifest",
        ):
            require_data_verification(
                read_strict_json(gate_path),
                self.config,
                self.dataset,
                gate_path=gate_path,
            )

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

    def test_regenerated_archive_members_are_decompressed_once(self):
        output, _gate = self._verified_run("single-decompression")
        with np.load(
            output / "data_verification/regenerated.npz", allow_pickle=False
        ) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}

        class TrackingArchive:
            files = list(arrays)

            def __init__(self):
                self.reads = {name: 0 for name in arrays}

            def __getitem__(self, name):
                self.reads[name] += 1
                return arrays[name]

        tracked = TrackingArchive()
        engine_match, rows = _compare_regenerated_archive(
            self.dataset, tracked, 0.0, 0.0
        )
        self.assertTrue(engine_match)
        self.assertTrue(all(row["passed"] for row in rows))
        self.assertEqual(set(tracked.reads), REGENERATED_FIELDS)
        self.assertTrue(all(count == 1 for count in tracked.reads.values()))

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
        with self.assertRaisesRegex(RuntimeError, "requires live independent regeneration"):
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

    def test_precomputed_regeneration_bundle_is_relocatable_and_rechecked(self):
        origin, origin_gate = self._verified_run("portable-origin")
        source = self._precomputed_bundle(origin, origin_gate)
        relocated = self.root / "portable-relocated"
        shutil.move(source, relocated)

        output = self.root / "portable-consumer"
        gate = run_data_verification(
            self.config,
            self.dataset,
            relocated / "verifier.json",
            output,
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["blocking_passed"])
        self.assertTrue(gate["diagnostic_comparison_passed"])
        self.assertEqual(gate["status"], "DIAGNOSTIC_NOT_CLAIM")
        self.assertEqual(
            gate["verification_mode"],
            "precomputed_regeneration_replay_diagnostic",
        )
        self.assertEqual(
            gate["verification_receipt_sha256"],
            sha256_file(relocated / "verification_receipt.json"),
        )
        with self.assertRaisesRegex(RuntimeError, "live independent regeneration"):
            require_verified_roles_from_root(
                output, self.config, self.dataset, ("target",)
            )

        regenerated = relocated / "regenerated.npz"
        regenerated.write_bytes(regenerated.read_bytes() + b"tampered")
        with self.assertRaisesRegex(RuntimeError, "regenerated hash mismatch"):
            run_data_verification(
                self.config,
                self.dataset,
                relocated / "verifier.json",
                self.root / "portable-tampered",
            )

    def test_nonfixture_precomputed_receipt_must_match_registered_origin(self):
        origin, origin_gate = self._verified_run("registered-origin")
        bundle = self._precomputed_bundle(origin, origin_gate)
        receipt_path = bundle / "verification_receipt.json"
        receipt = read_strict_json(receipt_path)
        receipt.update(
            {
                "fixture": False,
                "scientific_use": "CANDIDATE_NOT_CLAIM",
                "scene_count": self.dataset.scene_count,
                "source_tree_sha256": "a" * 64,
            }
        )
        receipt["renderer_runtime"]["libllvm_registry_sha256"] = "b" * 64
        receipt["renderer_runtime"]["lock_files"][
            "sionna_approved_libllvm_registry"
        ] = "b" * 64
        write_json(receipt_path, receipt)
        with self.assertRaisesRegex(RuntimeError, "registry differs"):
            validate_receipt(
                receipt_path,
                self.dataset_path,
                require_registration=False,
            )
        approved_registry = ROOT / "formal_v2/configs/sionna_llvm_approved_v1.json"
        approved_registry_sha256 = sha256_file(approved_registry)
        receipt["renderer_runtime"][
            "libllvm_registry_sha256"
        ] = approved_registry_sha256
        receipt["renderer_runtime"]["lock_files"][
            "sionna_approved_libllvm_registry"
        ] = approved_registry_sha256
        write_json(receipt_path, receipt)
        unregistered = self.root / "unregistered_candidate_evidence.json"
        write_json(unregistered, {})
        with patch(
            "formal_v2.formal_precomputed_regeneration_verifier.REGISTERED_CANDIDATE_EVIDENCE",
            unregistered,
        ):
            with self.assertRaisesRegex(RuntimeError, "not registered"):
                validate_receipt(receipt_path, self.dataset_path)
        validate_receipt(
            receipt_path,
            self.dataset_path,
            require_registration=False,
        )
        registry = self.root / "candidate_evidence.json"
        write_json(
            registry,
            {
                "schema_version": "csi-pairs-m4-llvm22-candidate-evidence-v1",
                "status": "STATIC_REGISTRY_PASS",
                "scientific_use": "CANDIDATE_NOT_CLAIM",
                "candidate": {
                    "dataset_sha256": receipt["dataset_sha256"],
                    "dataset_bytes": receipt["dataset_bytes"],
                    "fixture": False,
                    "scene_banks": receipt["scene_count"],
                },
                "live_independent_regeneration": {
                    "status": "REPORTED_PASS_EXTERNAL_ARTIFACTS_NOT_VERIFIED",
                    "verification_mode": "live_independent_regeneration",
                    "gate_sha256": receipt["origin_gate_sha256"],
                    "stage_manifest_sha256": receipt["origin_manifest_sha256"],
                    "origin_per_scene_sha256": receipt["origin_per_scene_sha256"],
                    "regenerated_sha256": receipt["regenerated_sha256"],
                    "source_tree_sha256": receipt["source_tree_sha256"],
                    "role_status": receipt["role_status"],
                    "rtol": 0.0,
                    "atol": 0.0,
                },
                "portable_replay": {
                    "status": "DIAGNOSTIC_REPLAY_PASS",
                    "verification_mode": "precomputed_regeneration_replay_diagnostic",
                    "verification_receipt_sha256": sha256_file(receipt_path),
                },
            },
        )
        with patch(
            "formal_v2.formal_precomputed_regeneration_verifier.REGISTERED_CANDIDATE_EVIDENCE",
            registry,
        ):
            validate_receipt(receipt_path, self.dataset_path)
            receipt["origin_gate_sha256"] = "f" * 64
            write_json(receipt_path, receipt)
            with self.assertRaisesRegex(RuntimeError, "not registered"):
                validate_receipt(receipt_path, self.dataset_path)


if __name__ == "__main__":
    unittest.main()
