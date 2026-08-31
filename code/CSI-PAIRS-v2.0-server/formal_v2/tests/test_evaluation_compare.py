from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from formal_v2.formal_evaluation_compare import (
    CSV_CONTRACTS,
    REQUIRED_ARTIFACTS,
    compare_evaluation_artifacts,
    compare_json_values,
)
from formal_v2.formal_io import sha256_file


class EvaluationArtifactCompareTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.reference = self.root / "reference" / "evaluation"
        self.candidate = self.root / "candidate" / "evaluation"
        self.reference.mkdir(parents=True)
        self.candidate.mkdir(parents=True)
        self._build_evaluation(self.reference, row_count=2)
        self._build_evaluation(self.candidate, row_count=2)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _identity(source_tree="a" * 64):
        runtime = {
            "schema_version": "runtime-v1",
            "source_tree_sha256": source_tree,
            "requirements_lock_sha256": "d" * 64,
            "python_version": "3.12.13",
        }
        runtime_payload = json.dumps(
            runtime,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return {
            "artifact_label": "csi-pairs-v6-test-fixture",
            "dataset_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "fixture": True,
            "scientific_use": "FORBIDDEN",
            "source_tree_sha256": source_tree,
            "requirements_lock_sha256": "d" * 64,
            "runtime_provenance_sha256": hashlib.sha256(runtime_payload).hexdigest(),
            "runtime_provenance": runtime,
        }

    def _row(self, contract, index, source_tree="a" * 64):
        identity = self._identity(source_tree)
        row = {}
        for field in contract.fields:
            if field == "seed":
                row[field] = str(100 + index)
            elif field == "arm":
                row[field] = "full"
            elif field in contract.float_fields:
                row[field] = str(1.0 + index)
            elif field in contract.structured_fields:
                row[field] = repr(
                    [
                        {
                            "family": "linear",
                            "selection_nll": 1.0 + index,
                            "selection_auroc": 0.5,
                        }
                    ]
                )
            elif field in identity and field != "runtime_provenance":
                row[field] = identity[field]
            elif field == "fixture":
                row[field] = "True"
            elif field in {"pair_id", "bank_id", "baseline", "effect_bin", "probe", "route"}:
                row[field] = f"{field}-{index}"
            else:
                row[field] = f"value-{field}-{index}"
        return row

    def _write_csv(self, root, name, rows):
        contract = CSV_CONTRACTS[name]
        with (root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=contract.fields)
            writer.writeheader()
            writer.writerows(rows)

    def _gate(self, source_tree="a" * 64):
        return {
            "schema_version": "csi-pairs-v6-evaluation-gate-test-v1",
            **self._identity(source_tree),
            "status": "FAIL",
            "passed": False,
            "gate_vector": {"G3": "FAIL", "G4": "NOT_ASSESSED"},
            "metrics": {
                "score": 1.0,
                "interval": [0.25, 1.75],
                "count": 2,
            },
        }

    def _build_evaluation(self, root, row_count=1, source_tree="a" * 64):
        for name, contract in CSV_CONTRACTS.items():
            self._write_csv(
                root,
                name,
                [self._row(contract, index, source_tree) for index in range(row_count)],
            )
        (root / "gate.json").write_text(
            json.dumps(self._gate(source_tree), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._write_manifest(root, source_tree)

    def _write_manifest(self, root, source_tree="a" * 64):
        identity = self._identity(source_tree)
        files = []
        for relative in REQUIRED_ARTIFACTS:
            path = root / relative
            if not path.is_file():
                continue
            files.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    **identity,
                }
            )
        payload = {
            "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
            **identity,
            "files": files,
        }
        (root / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _read_rows(self, root, name):
        with (root / name).open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_exact_artifacts_pass_and_write_machine_report(self):
        output = self.root / "audit" / "equivalence.json"
        report = compare_evaluation_artifacts(
            self.reference, self.candidate, output_path=output
        )
        self.assertTrue(report["equivalent"])
        self.assertTrue(report["exact_match"])
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["summary"]["csv_passed"], 9)
        self.assertEqual(report["summary"]["csv_exact"], 9)
        on_disk = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, report)
        self.assertEqual(on_disk["schema_version"], "csi-pairs-v6-evaluation-equivalence-report-v1")

    def test_explicit_tolerance_covers_csv_structured_field_and_gate_paths(self):
        name = "compatibility_probe_contract.csv"
        rows = self._read_rows(self.candidate, name)
        rows[0]["selection_nll"] = "1.0000005"
        structured = json.loads(rows[0]["candidate_metrics"].replace("'", '"'))
        structured[0]["selection_nll"] = 1.0000005
        rows[0]["candidate_metrics"] = repr(structured)
        self._write_csv(self.candidate, name, rows)
        gate = json.loads((self.candidate / "gate.json").read_text(encoding="utf-8"))
        gate["metrics"]["score"] = 1.0000005
        (self.candidate / "gate.json").write_text(
            json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._write_manifest(self.candidate)

        exact = compare_evaluation_artifacts(self.reference, self.candidate)
        tolerant = compare_evaluation_artifacts(
            self.reference,
            self.candidate,
            absolute_tolerance=1e-6,
        )
        self.assertFalse(exact["equivalent"])
        self.assertTrue(tolerant["equivalent"])
        self.assertFalse(tolerant["exact_match"])
        field = tolerant["csv"][name]["float_fields"]["selection_nll"]
        self.assertAlmostEqual(field["max_abs_diff"], 5e-7, places=12)
        self.assertEqual(field["changed_count"], 1)
        self.assertEqual(field["outside_tolerance_count"], 0)
        structured_report = tolerant["csv"][name]["structured_fields"]["candidate_metrics"]
        self.assertTrue(structured_report["equivalent"])
        self.assertFalse(structured_report["exact"])
        self.assertAlmostEqual(
            tolerant["gate"]["comparison"]["max_abs_diff"], 5e-7, places=12
        )

    def test_primary_key_uniqueness_row_order_and_nonfloat_are_fail_closed(self):
        name = "response_probe_contract.csv"
        rows = self._read_rows(self.candidate, name)
        rows.reverse()
        rows[1]["seed"] = rows[0]["seed"]
        rows[1]["arm"] = rows[0]["arm"]
        rows[1]["probe"] = rows[0]["probe"]
        rows[0]["steps"] = "changed"
        self._write_csv(self.candidate, name, rows)
        self._write_manifest(self.candidate)
        report = compare_evaluation_artifacts(self.reference, self.candidate)
        csv_report = report["csv"][name]
        self.assertFalse(report["equivalent"])
        self.assertFalse(csv_report["row_order_match"])
        self.assertEqual(csv_report["candidate_duplicate_key_count"], 1)
        self.assertFalse(csv_report["primary_key_set_match"])
        self.assertGreater(csv_report["nonfloat_mismatch_count"], 0)

    def test_schema_missing_file_invalid_float_and_gate_type_changes_fail(self):
        name = "cgs_active_effect_bins.csv"
        rows = self._read_rows(self.candidate, name)
        rows[0]["cgs_auroc"] = "nan"
        contract = CSV_CONTRACTS[name]
        shortened = contract.fields[:-1]
        with (self.candidate / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=shortened, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        (self.candidate / "response_pair_effects.csv").unlink()
        gate = json.loads((self.candidate / "gate.json").read_text(encoding="utf-8"))
        gate["metrics"]["count"] = 2.0
        (self.candidate / "gate.json").write_text(
            json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._write_manifest(self.candidate)
        report = compare_evaluation_artifacts(
            self.reference, self.candidate, absolute_tolerance=1.0
        )
        self.assertFalse(report["equivalent"])
        self.assertFalse(report["csv"][name]["schema_match"])
        self.assertTrue(report["csv"]["response_pair_effects.csv"]["errors"])
        self.assertGreater(
            report["gate"]["comparison"]["structural_mismatch_count"], 0
        )
        self.assertFalse(report["manifest"]["candidate"]["valid"])

    def test_manifest_hash_validation_and_identity_layers_are_distinct(self):
        name = "cgs_per_bank.csv"
        with (self.candidate / name).open("a", encoding="utf-8") as handle:
            handle.write("tamper\n")
        report = compare_evaluation_artifacts(self.reference, self.candidate)
        self.assertFalse(report["manifest"]["candidate"]["valid"])
        self.assertIn(
            "artifact byte count does not match manifest",
            report["manifest"]["candidate"]["required_file_records"][name]["errors"],
        )

        self._build_evaluation(self.candidate, row_count=2, source_tree="f" * 64)
        layered = compare_evaluation_artifacts(self.reference, self.candidate)
        self.assertTrue(layered["manifest"]["frozen_experiment_identity"]["match"])
        self.assertFalse(layered["manifest"]["implementation_identity"]["match"])
        self.assertTrue(layered["manifest"]["comparable"])
        self.assertFalse(layered["manifest"]["exact"])
        self.assertTrue(layered["equivalent"])
        self.assertFalse(layered["exact_match"])
        self.assertEqual(layered["status"], "PASS")
        self.assertTrue(layered["gate"]["comparison"]["equivalent"])
        self.assertFalse(
            layered["gate"]["implementation_evidence"]["comparison"]["exact"]
        )
        self.assertTrue(
            all(
                value["equivalent"] and not value["exact"]
                for value in layered["csv"].values()
            )
        )

    def test_implementation_identity_must_bind_to_its_own_manifest(self):
        name = "response_probe_contract.csv"
        rows = self._read_rows(self.candidate, name)
        rows[0]["source_tree_sha256"] = "f" * 64
        self._write_csv(self.candidate, name, rows)
        self._write_manifest(self.candidate)
        csv_unbound = compare_evaluation_artifacts(self.reference, self.candidate)
        self.assertFalse(csv_unbound["equivalent"])
        binding = csv_unbound["csv"][name]["identity_binding"]["candidate"]
        self.assertFalse(binding["valid"])
        self.assertEqual(binding["mismatch_count"], 1)
        self.assertEqual(binding["differences"][0]["field"], "source_tree_sha256")

        self._build_evaluation(self.candidate, row_count=2)
        gate = json.loads((self.candidate / "gate.json").read_text(encoding="utf-8"))
        gate["runtime_provenance_sha256"] = "f" * 64
        (self.candidate / "gate.json").write_text(
            json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._write_manifest(self.candidate)
        gate_unbound = compare_evaluation_artifacts(self.reference, self.candidate)
        self.assertFalse(gate_unbound["equivalent"])
        self.assertFalse(
            gate_unbound["gate"]["identity_binding"]["candidate"]["valid"]
        )

    def test_new_implementation_cannot_hide_csv_scientific_differences(self):
        self._build_evaluation(self.candidate, row_count=2, source_tree="f" * 64)
        float_name = "cgs_active_effect_bins.csv"
        rows = self._read_rows(self.candidate, float_name)
        rows[0]["cgs_auroc"] = "1.0000000001"
        self._write_csv(self.candidate, float_name, rows)
        self._write_manifest(self.candidate, source_tree="f" * 64)
        float_change = compare_evaluation_artifacts(self.reference, self.candidate)
        self.assertFalse(float_change["equivalent"])
        self.assertEqual(
            float_change["csv"][float_name]["float_fields"]["cgs_auroc"][
                "outside_tolerance_count"
            ],
            1,
        )

        self._build_evaluation(self.candidate, row_count=2, source_tree="f" * 64)
        nonfloat_name = "response_probe_contract.csv"
        rows = self._read_rows(self.candidate, nonfloat_name)
        rows[0]["steps"] = "scientifically-changed"
        self._write_csv(self.candidate, nonfloat_name, rows)
        self._write_manifest(self.candidate, source_tree="f" * 64)
        nonfloat_change = compare_evaluation_artifacts(self.reference, self.candidate)
        self.assertFalse(nonfloat_change["equivalent"])
        self.assertEqual(
            nonfloat_change["csv"][nonfloat_name]["nonfloat_mismatch_count"], 1
        )

    def test_new_implementation_cannot_hide_gate_scientific_difference(self):
        self._build_evaluation(self.candidate, row_count=2, source_tree="f" * 64)
        gate = json.loads((self.candidate / "gate.json").read_text(encoding="utf-8"))
        gate["metrics"]["score"] = 1.0000000001
        (self.candidate / "gate.json").write_text(
            json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._write_manifest(self.candidate, source_tree="f" * 64)
        report = compare_evaluation_artifacts(self.reference, self.candidate)
        self.assertFalse(report["equivalent"])
        self.assertFalse(report["gate"]["comparison"]["equivalent"])
        self.assertEqual(
            report["gate"]["comparison"]["differences"][0]["path"],
            "/metrics/score",
        )

    def test_manifest_runtime_digest_binding_is_fail_closed(self):
        manifest_path = self.candidate / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["runtime_provenance"]["python_version"] = "3.13.0"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = compare_evaluation_artifacts(self.reference, self.candidate)
        self.assertFalse(report["equivalent"])
        self.assertIn(
            "manifest runtime_provenance_sha256 does not bind runtime_provenance",
            report["manifest"]["candidate"]["errors"],
        )

    def test_manifest_nonimplementation_metadata_remains_scientific(self):
        manifest_path = self.candidate / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["scientific_contract_note"] = "changed"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = compare_evaluation_artifacts(self.reference, self.candidate)
        self.assertFalse(report["equivalent"])
        comparison = report["manifest"]["scientific_metadata"]
        self.assertFalse(comparison["equivalent"])
        self.assertEqual(
            comparison["differences"][0]["path"], "/scientific_contract_note"
        )

    def test_json_comparison_is_path_aware_and_default_exact(self):
        reference = {"a": [1.0, {"count": 2}], "flag": True}
        candidate = {"a": [1.0000005, {"count": 2}], "flag": True}
        exact = compare_json_values(reference, candidate)
        tolerant = compare_json_values(
            reference, candidate, absolute_tolerance=1e-6
        )
        self.assertFalse(exact["equivalent"])
        self.assertTrue(tolerant["equivalent"])
        self.assertEqual(tolerant["differences"][0]["path"], "/a/0")
        wrong_type = compare_json_values(reference, {"a": [1.0, {"count": 2.0}], "flag": True})
        self.assertFalse(wrong_type["equivalent"])
        self.assertEqual(wrong_type["structural_mismatch_count"], 1)

    def test_report_cannot_be_written_into_compared_artifacts(self):
        with self.assertRaisesRegex(ValueError, "outside both evaluation roots"):
            compare_evaluation_artifacts(
                self.reference,
                self.candidate,
                output_path=self.candidate / "comparison.json",
            )


if __name__ == "__main__":
    unittest.main()
