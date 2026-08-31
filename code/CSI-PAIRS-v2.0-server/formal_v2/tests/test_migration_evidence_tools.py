from __future__ import annotations

import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from formal_v2 import formal_migration_evidence as evidence
from formal_v2.formal_evaluation_compare import (
    CSV_CONTRACTS,
    REQUIRED_ARTIFACTS,
    compare_evaluation_artifacts,
)
from formal_v2.formal_io import sha256_file, write_json
from formal_v2.tests.test_formal_migration import (
    CREATED_UTC,
    FormalMigrationFixture,
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


class MigrationEvidenceToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = FormalMigrationFixture(self.root)
        self.paths = self.fixture.migration_evidence
        self.identity = self._payload("performance")["inputs"]["identity"]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _payload(self, name: str) -> dict[str, object]:
        return json.loads(self.paths[name].read_text(encoding="utf-8"))

    def _validate(self, name: str, payload: dict[str, object]) -> None:
        evidence.validate_evidence_report(
            name,
            payload,
            expected_identity=self.identity,
            legacy_run_root=self.fixture.legacy_run,
            new_run_root=self.fixture.new_run,
        )

    def _control_trace(self) -> Path:
        return Path(self._payload("resume")["inputs"]["artifacts"][0]["path"])

    def _replace_comparator(
        self,
        payload: dict[str, object],
        key: str,
        comparator: dict[str, object],
        label: str,
    ) -> None:
        old = payload["results"][key]
        path = self.fixture.external_root / f"mutated-{label}.json"
        write_json(path, comparator)
        binding = evidence.bind_file(
            path,
            legacy_run_root=self.fixture.legacy_run,
            new_run_root=self.fixture.new_run,
            require_external=True,
        )
        payload["results"][key] = binding
        for index, item in enumerate(payload["inputs"]["artifacts"]):
            if item == old:
                payload["inputs"]["artifacts"][index] = binding
                return
        self.fail("comparator input binding was not found")

    def _formal_report(self, payload: dict[str, object]) -> dict[str, object]:
        path = Path(payload["results"]["formal_subset_report"]["path"])
        return json.loads(path.read_text(encoding="utf-8"))

    def _extended_binding(self, path: Path) -> dict[str, object]:
        binding = evidence.bind_file(path)
        return {
            "path": binding["path"],
            "bytes": binding["bytes"],
            "mtime_ns": path.resolve().stat().st_mtime_ns,
            "sha256": binding["sha256"],
        }

    def _replace_fragment(
        self,
        report: dict[str, object],
        side: str,
        fragment: dict[str, object],
        label: str,
    ) -> None:
        path = self.fixture.external_root / f"mutated-{label}-{side}-fragment.json"
        path.write_text(
            json.dumps(fragment, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
            encoding="ascii",
        )
        report["workers"][side]["fragment"] = self._extended_binding(path)

    def _write_real_comparator_report(
        self, label: str, *, fixture: bool
    ) -> Path:
        pair_root = self.fixture.external_root / label
        reference = pair_root / "reference" / "evaluation"
        candidate = pair_root / "candidate" / "evaluation"
        reference.mkdir(parents=True)
        candidate.mkdir(parents=True)

        def comparison_identity(source_tree: str) -> dict[str, object]:
            runtime = {
                "schema_version": "runtime-v1",
                "source_tree_sha256": source_tree,
                "requirements_lock_sha256": self.identity[
                    "requirements_lock_sha256"
                ],
            }
            return {
                "artifact_label": label,
                "dataset_sha256": self.identity["dataset_sha256"],
                "config_sha256": self.identity["config_sha256"],
                "fixture": fixture,
                "scientific_use": (
                    "FORBIDDEN" if fixture else "FORMAL_EXPERIMENT_ALLOWED"
                ),
                "source_tree_sha256": source_tree,
                "requirements_lock_sha256": self.identity[
                    "requirements_lock_sha256"
                ],
                "runtime_provenance_sha256": _canonical_sha256(runtime),
                "runtime_provenance": runtime,
            }

        def row(contract, identity):
            result = {}
            for field in contract.fields:
                if field in contract.float_fields:
                    result[field] = "1.0"
                elif field in contract.structured_fields:
                    result[field] = "[]"
                elif field in identity and field != "runtime_provenance":
                    result[field] = str(identity[field])
                elif field in contract.primary_key:
                    result[field] = f"{field}-key"
                else:
                    result[field] = f"value-{field}"
            return result

        def build(root: Path, identity: dict[str, object]) -> None:
            for name, contract in CSV_CONTRACTS.items():
                with (root / name).open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=contract.fields)
                    writer.writeheader()
                    writer.writerow(row(contract, identity))
            write_json(
                root / "gate.json",
                {
                    "schema_version": "csi-pairs-v6-test-gate-v1",
                    **identity,
                    "status": "PASS",
                    "passed": True,
                    "metrics": {"score": 1.0, "count": 1},
                },
            )
            records = []
            for relative in REQUIRED_ARTIFACTS:
                artifact = root / relative
                records.append(
                    {
                        "path": relative,
                        "bytes": artifact.stat().st_size,
                        "sha256": sha256_file(artifact),
                        **identity,
                    }
                )
            write_json(
                root / "manifest.json",
                {
                    "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
                    **identity,
                    "files": records,
                },
            )

        build(
            reference,
            comparison_identity(str(self.identity["legacy_source_tree_sha256"])),
        )
        build(
            candidate,
            comparison_identity(str(self.identity["new_source_tree_sha256"])),
        )
        output = pair_root / "comparison.json"
        report = compare_evaluation_artifacts(
            reference,
            candidate,
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
            output_path=output,
        )
        self.assertTrue(report["equivalent"])
        self.assertEqual(report["summary"]["csv_passed"], 9)
        return output

    def test_all_production_writers_round_trip_through_strict_validation(self) -> None:
        self.assertEqual(set(self.paths), set(evidence.MIGRATION_EVIDENCE_KEYS))
        for name in evidence.MIGRATION_EVIDENCE_KEYS:
            with self.subTest(name=name):
                payload = self._payload(name)
                self._validate(name, payload)
                self.assertEqual(
                    json.loads(self.paths[name].read_text(encoding="utf-8")), payload
                )

    def test_equivalence_writer_accepts_reports_from_real_comparator(self) -> None:
        smoke = self._write_real_comparator_report("real-smoke", fixture=True)
        formal = self.fixture.external_root / "real-formal-subset.json"
        self.fixture.write_real_subset_report(
            formal, identity=self.identity, label="real-formal-subset"
        )
        target = self.fixture.external_root / "real-equivalence-evidence.json"
        payload = evidence.write_equivalence_report(
            target,
            expected_identity=self.identity,
            command="python -B -m formal_v2.formal_evaluation_compare",
            smoke_report_path=smoke,
            formal_subset_report_path=formal,
            created_utc=CREATED_UTC,
        )
        self._validate("equivalence", payload)

    def test_writers_are_exclusive_and_require_external_evidence_paths(self) -> None:
        trace = self._control_trace()
        with self.assertRaises(FileExistsError):
            evidence.write_resume_report(
                self.paths["resume"],
                expected_identity=self.identity,
                command="python -B resume-check",
                artifact_paths=[trace],
                interrupted=True,
                resumed=True,
                completed=True,
                reused_shard_count=1,
                recomputed_shard_count=0,
                output_equivalent=True,
                created_utc=CREATED_UTC,
            )
        for root in (self.fixture.legacy_run, self.fixture.new_run):
            with self.subTest(root=root), self.assertRaises(RuntimeError):
                evidence.write_resume_report(
                    root / "inside-run.json",
                    expected_identity=self.identity,
                    command="python -B resume-check",
                    artifact_paths=[trace],
                    interrupted=True,
                    resumed=True,
                    completed=True,
                    reused_shard_count=1,
                    recomputed_shard_count=0,
                    output_equivalent=True,
                    created_utc=CREATED_UTC,
                )

    def test_placeholder_results_are_rejected_for_every_technical_schema(self) -> None:
        for name in evidence.MIGRATION_TECHNICAL_EVIDENCE_SCHEMAS:
            with self.subTest(name=name):
                payload = self._payload(name)
                payload["results"] = {"validated": True}
                with self.assertRaises(RuntimeError):
                    self._validate(name, payload)

    def test_performance_requires_exact_doubling_and_complete_measurements(self) -> None:
        payload = self._payload("performance")
        mutations = []

        missing_scale = copy.deepcopy(payload)
        missing_scale["results"]["scales"].pop()
        mutations.append(("missing scale", missing_scale))

        reordered = copy.deepcopy(payload)
        reordered["results"]["scales"][0], reordered["results"]["scales"][1] = (
            reordered["results"]["scales"][1],
            reordered["results"]["scales"][0],
        )
        mutations.append(("reordered scales", reordered))

        not_doubling = copy.deepcopy(payload)
        not_doubling["results"]["scales"][1]["sample_count"] += 1
        mutations.append(("not doubling", not_doubling))

        for field in (
            "cpu_seconds",
            "peak_rss_bytes",
            "execution_device",
            "gpu_utilization_percent",
            "peak_vram_bytes",
            "output",
        ):
            missing = copy.deepcopy(payload)
            missing["results"]["scales"][0].pop(field)
            mutations.append((f"missing {field}", missing))

        for label, mutated in mutations:
            with self.subTest(label=label), self.assertRaises(RuntimeError):
                self._validate("performance", mutated)

    def test_performance_recomputes_exponents_and_enforces_strict_limit(self) -> None:
        mismatch = self._payload("performance")
        mismatch["results"]["empirical_time_exponent"] += 0.01
        with self.assertRaises(RuntimeError):
            self._validate("performance", mismatch)

        superlinear = self._payload("performance")
        superlinear["results"]["scales"][2]["wall_seconds"] = 8.0
        superlinear["results"]["empirical_time_exponent"] = 1.5
        with self.assertRaises(RuntimeError):
            self._validate("performance", superlinear)

        quadratic_rss = self._payload("performance")
        quadratic_rss["results"]["scales"][2]["peak_rss_bytes"] = 16_000
        quadratic_rss["results"]["empirical_peak_rss_exponent"] = 2.0
        with self.assertRaises(RuntimeError):
            self._validate("performance", quadratic_rss)

        invalid_gpu = self._payload("performance")
        invalid_gpu["results"]["scales"][0]["gpu_utilization_percent"] = 100.1
        with self.assertRaises(RuntimeError):
            self._validate("performance", invalid_gpu)

        missing_vram = self._payload("performance")
        missing_vram["results"]["gpu_benchmark"].pop("peak_vram_bytes")
        with self.assertRaises(RuntimeError):
            self._validate("performance", missing_vram)

    def test_performance_output_and_comparator_report_tampering_are_rejected(self) -> None:
        performance = self._payload("performance")
        output = Path(performance["results"]["scales"][0]["output"]["path"])
        output.write_bytes(output.read_bytes() + b"tampered")
        with self.assertRaises(RuntimeError):
            self._validate("performance", performance)

        equivalence = self._payload("equivalence")
        comparator = Path(equivalence["results"]["smoke_report"]["path"])
        comparator.write_bytes(comparator.read_bytes() + b"tampered")
        with self.assertRaises(RuntimeError):
            self._validate("equivalence", equivalence)

    def test_equivalence_requires_zero_tolerance_and_all_nine_csvs(self) -> None:
        payload = self._payload("equivalence")
        smoke_path = Path(payload["results"]["smoke_report"]["path"])
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        smoke["tolerance"]["absolute"] = 1e-9
        self._replace_comparator(payload, "smoke_report", smoke, "tolerance")
        with self.assertRaises(RuntimeError):
            self._validate("equivalence", payload)

        payload = self._payload("equivalence")
        smoke_path = Path(payload["results"]["smoke_report"]["path"])
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        smoke["csv"].pop(next(iter(smoke["csv"])))
        smoke["summary"]["csv_passed"] = 8
        smoke["summary"]["csv_total"] = 8
        self._replace_comparator(payload, "smoke_report", smoke, "eight-csv")
        with self.assertRaises(RuntimeError):
            self._validate("equivalence", payload)

    def test_equivalence_requires_fixture_smoke_and_formal_nonfixture_identity(self) -> None:
        payload = self._payload("equivalence")
        smoke_path = Path(payload["results"]["smoke_report"]["path"])
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        smoke["manifest"]["reference"]["identity"]["fixture"] = False
        self._replace_comparator(payload, "smoke_report", smoke, "nonfixture-smoke")
        with self.assertRaises(RuntimeError):
            self._validate("equivalence", payload)

    def test_smoke_oracle_has_an_independent_manifest_source_and_runtime_binding(self) -> None:
        payload = self._payload("equivalence")
        oracle = payload["results"]["smoke_oracle"]
        self.assertNotEqual(
            oracle["identity"]["source_tree_sha256"],
            self.identity["legacy_source_tree_sha256"],
        )
        oracle["identity"]["source_tree_sha256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            self._validate("equivalence", payload)

        payload = self._payload("equivalence")
        manifest = Path(payload["results"]["smoke_oracle"]["manifest"]["path"])
        manifest.write_bytes(manifest.read_bytes() + b"tampered")
        with self.assertRaises(RuntimeError):
            self._validate("equivalence", payload)

    def test_formal_subset_requires_distinct_authenticated_worker_sources(self) -> None:
        mutations = []

        same_commit = self._payload("equivalence")
        report = self._formal_report(same_commit)
        report["workers"]["new"]["source_identity"]["git_commit"] = report[
            "workers"
        ]["legacy"]["source_identity"]["git_commit"]
        mutations.append(("same-commit", same_commit, report))

        same_root = self._payload("equivalence")
        report = self._formal_report(same_root)
        report["workers"]["new"]["source_identity"]["source_root"] = report[
            "workers"
        ]["legacy"]["source_identity"]["source_root"]
        mutations.append(("same-root", same_root, report))

        missing_harness = self._payload("equivalence")
        report = self._formal_report(missing_harness)
        report["workers"]["legacy"]["source_identity"]["files"].pop(
            "formal_evaluation_subset_compare.py"
        )
        mutations.append(("missing-harness", missing_harness, report))

        missing_bytes = self._payload("equivalence")
        report = self._formal_report(missing_bytes)
        report["workers"]["new"]["source_identity"]["files"][
            "formal_evaluation_streaming.py"
        ].pop("bytes")
        mutations.append(("missing-source-bytes", missing_bytes, report))

        for label, payload, report in mutations:
            with self.subTest(label=label):
                self._replace_comparator(
                    payload, "formal_subset_report", report, label
                )
                with self.assertRaises(RuntimeError):
                    self._validate("equivalence", payload)

    def test_formal_subset_authenticates_authority_runtime_and_worker_artifacts(self) -> None:
        cases = []

        payload = self._payload("equivalence")
        report = self._formal_report(payload)
        report["authoritative_layer"] = "layers.same_source_decomposition"
        cases.append(("authority", payload, report))

        payload = self._payload("equivalence")
        report = self._formal_report(payload)
        report["workers"]["new"]["runtime_provenance"][
            "source_tree_sha256"
        ] = "0" * 64
        cases.append(("runtime", payload, report))

        payload = self._payload("equivalence")
        report = self._formal_report(payload)
        report["workers"]["legacy"]["source_identity"]["files"][
            "formal_evaluation.py"
        ]["sha256"] = "0" * 64
        cases.append(("source-file-sha", payload, report))

        payload = self._payload("equivalence")
        report = self._formal_report(payload)
        report["workers"]["legacy"]["request"]["sha256"] = "0" * 64
        cases.append(("request-sha", payload, report))

        payload = self._payload("equivalence")
        report = self._formal_report(payload)
        report["workers"]["new"]["fragment"]["sha256"] = "0" * 64
        cases.append(("fragment-sha", payload, report))

        for label, payload, report in cases:
            with self.subTest(label=label):
                self._replace_comparator(
                    payload, "formal_subset_report", report, label
                )
                with self.assertRaises(RuntimeError):
                    self._validate("equivalence", payload)

    def test_formal_subset_binds_formal_inputs_and_selection(self) -> None:
        cases = []

        payload = self._payload("equivalence")
        report = self._formal_report(payload)
        report["shared_inputs"]["dataset_sha256"] = "0" * 64
        cases.append(("dataset", payload, report))

        payload = self._payload("equivalence")
        report = self._formal_report(payload)
        report["shared_inputs"]["config_sha256"] = "0" * 64
        cases.append(("config", payload, report))

        payload = self._payload("equivalence")
        report = self._formal_report(payload)
        report["shared_inputs"]["files"][0]["sha256"] = "0" * 64
        cases.append(("input-file", payload, report))

        payload = self._payload("equivalence")
        report = self._formal_report(payload)
        report["selection"]["scene_rule"] = "unreviewed-selection"
        cases.append(("scene-rule", payload, report))

        payload = self._payload("equivalence")
        report = self._formal_report(payload)
        report["selection"]["sha256"] = "0" * 64
        cases.append(("selection-digest", payload, report))

        payload = self._payload("equivalence")
        report = self._formal_report(payload)
        report["selection"]["scenes_in_execution_order"].reverse()
        cases.append(("scene-order", payload, report))

        for label, payload, report in cases:
            with self.subTest(label=label):
                self._replace_comparator(
                    payload, "formal_subset_report", report, label
                )
                with self.assertRaises(RuntimeError):
                    self._validate("equivalence", payload)

    def test_formal_subset_recomputes_nine_tables_and_gate_aggregates(self) -> None:
        payload = self._payload("equivalence")
        report = self._formal_report(payload)
        fragment_path = Path(report["workers"]["new"]["fragment"]["path"])
        fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
        fragment["tables"].pop(next(iter(fragment["tables"])))
        self._replace_fragment(report, "new", fragment, "missing-table")
        self._replace_comparator(
            payload, "formal_subset_report", report, "missing-table-report"
        )
        with self.assertRaises(RuntimeError):
            self._validate("equivalence", payload)

        payload = self._payload("equivalence")
        report = self._formal_report(payload)
        fragment_path = Path(report["workers"]["new"]["fragment"]["path"])
        fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
        table = next(iter(CSV_CONTRACTS))
        float_field = CSV_CONTRACTS[table].float_fields[0]
        fragment["tables"][table][0][float_field] = 0.5
        self._replace_fragment(report, "new", fragment, "float-bits")
        self._replace_comparator(
            payload, "formal_subset_report", report, "float-bits-report"
        )
        with self.assertRaises(RuntimeError):
            self._validate("equivalence", payload)

        payload = self._payload("equivalence")
        report = self._formal_report(payload)
        gate = report["layers"]["cross_source_end_to_end"][
            "gate_required_aggregates"
        ]
        gate["table_count"] -= 1
        first = next(iter(gate["tables"].values()))
        first["legacy_ordered_sha256"] = "0" * 64
        self._replace_comparator(
            payload, "formal_subset_report", report, "gate-aggregate"
        )
        with self.assertRaises(RuntimeError):
            self._validate("equivalence", payload)

    def test_formal_subset_requires_read_only_and_positive_resource_evidence(self) -> None:
        payload = self._payload("equivalence")
        report = self._formal_report(payload)
        report["read_only"]["legacy"]["files"][0]["after"]["bytes"] += 1
        self._replace_comparator(
            payload, "formal_subset_report", report, "read-only"
        )
        with self.assertRaises(RuntimeError):
            self._validate("equivalence", payload)

        payload = self._payload("equivalence")
        report = self._formal_report(payload)
        fragment_path = Path(report["workers"]["legacy"]["fragment"]["path"])
        fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
        fragment["read_only"]["legacy_evaluation"]["after"]["entry_count"] += 1
        self._replace_fragment(report, "legacy", fragment, "evaluation-snapshot")
        self._replace_comparator(
            payload, "formal_subset_report", report, "evaluation-snapshot-report"
        )
        with self.assertRaises(RuntimeError):
            self._validate("equivalence", payload)

        payload = self._payload("equivalence")
        report = self._formal_report(payload)
        report["workers"]["new"]["resources"][
            "maximum_cuda_allocated_bytes"
        ] = 0
        self._replace_comparator(
            payload, "formal_subset_report", report, "resources"
        )
        with self.assertRaises(RuntimeError):
            self._validate("equivalence", payload)

    def test_each_control_report_rejects_missing_or_false_required_state(self) -> None:
        cases = {
            "resume": ("resumed", False),
            "corrupt_checkpoint": ("corruption_detected", False),
            "stale_checkpoint": ("stale_identity_detected", False),
            "lock": ("second_writer_rejected", False),
            "progress": ("atomic", False),
        }
        for name, (field, invalid) in cases.items():
            with self.subTest(name=name, case="false"):
                payload = self._payload(name)
                payload["results"][field] = invalid
                with self.assertRaises(RuntimeError):
                    self._validate(name, payload)
            with self.subTest(name=name, case="missing"):
                payload = self._payload(name)
                payload["results"].pop(field)
                with self.assertRaises(RuntimeError):
                    self._validate(name, payload)

    def test_diff_binds_the_patch_bytes_and_pass_declarations(self) -> None:
        payload = self._payload("base_to_new_diff")
        payload["results"]["reviewed"] = False
        with self.assertRaises(RuntimeError):
            self._validate("base_to_new_diff", payload)

        payload = self._payload("base_to_new_diff")
        patch = Path(payload["results"]["diff"]["path"])
        patch.write_text("changed after review\n", encoding="ascii")
        with self.assertRaises(RuntimeError):
            self._validate("base_to_new_diff", payload)

    def test_inventory_rejects_incompleteness_and_artifact_tampering(self) -> None:
        incomplete = self._payload("legacy_evaluation_inventory")
        incomplete["results"]["files"] = []
        incomplete["results"]["files_sha256"] = _canonical_sha256([])
        incomplete["inputs"]["artifacts"] = []
        with self.assertRaises(RuntimeError):
            self._validate("legacy_evaluation_inventory", incomplete)

        payload = self._payload("legacy_evaluation_inventory")
        artifact = Path(payload["inputs"]["artifacts"][0]["path"])
        artifact.write_bytes(artifact.read_bytes() + b"tampered")
        with self.assertRaises(RuntimeError):
            self._validate("legacy_evaluation_inventory", payload)

    def test_inventory_detects_files_added_after_freeze(self) -> None:
        payload = self._payload("legacy_evaluation_inventory")
        (self.fixture.legacy_run / "evaluation" / "late-file.json").write_text(
            "{}\n", encoding="ascii"
        )
        with self.assertRaises(RuntimeError):
            self._validate("legacy_evaluation_inventory", payload)

    def test_empty_legacy_evaluation_inventory_is_valid_until_a_file_appears(self) -> None:
        legacy = self.root / "empty-legacy"
        new = self.root / "empty-new"
        external = self.root / "empty-evidence"
        (legacy / "evaluation").mkdir(parents=True)
        new.mkdir()
        external.mkdir()
        identity = {
            **self.identity,
            "legacy_run_root": str(legacy.resolve()),
            "new_run_root": str(new.resolve()),
        }
        target = external / "inventory.json"
        payload = evidence.write_legacy_evaluation_inventory_report(
            target,
            expected_identity=identity,
            command="python -B inventory",
            created_utc=CREATED_UTC,
        )
        self.assertEqual(payload["inputs"]["artifacts"], [])
        self.assertEqual(payload["results"]["files"], [])
        self.assertEqual(
            payload["results"]["files_sha256"], _canonical_sha256([])
        )
        evidence.validate_evidence_report(
            "legacy_evaluation_inventory",
            payload,
            expected_identity=identity,
            legacy_run_root=legacy,
            new_run_root=new,
        )
        (legacy / "evaluation" / "appeared.log").write_text(
            "late output\n", encoding="ascii"
        )
        with self.assertRaises(RuntimeError):
            evidence.validate_evidence_report(
                "legacy_evaluation_inventory",
                payload,
                expected_identity=identity,
                legacy_run_root=legacy,
                new_run_root=new,
            )

    def test_freeze_rejects_inventory_digest_and_observation_mismatches(self) -> None:
        digest_mismatch = self._payload("legacy_evaluation_freeze")
        digest_mismatch["legacy_inventory_sha256"] = "f" * 64
        with self.assertRaises(RuntimeError):
            self._validate("legacy_evaluation_freeze", digest_mismatch)

        unobserved = self._payload("legacy_evaluation_freeze")
        unobserved["lock_state"]["observed"] = False
        unobserved["results"]["lock_state_observed"] = False
        with self.assertRaises(RuntimeError):
            self._validate("legacy_evaluation_freeze", unobserved)

        pid_mismatch = self._payload("legacy_evaluation_freeze")
        pid_mismatch["lock_state"]["owner_pid_identity"] = "different-pid"
        with self.assertRaises(RuntimeError):
            self._validate("legacy_evaluation_freeze", pid_mismatch)

        not_running = self._payload("legacy_evaluation_freeze")
        not_running["results"]["process_state"] = "EXITED"
        with self.assertRaises(RuntimeError):
            self._validate("legacy_evaluation_freeze", not_running)

    def test_complete_identity_and_absolute_file_bindings_are_required(self) -> None:
        payload = self._payload("resume")
        payload["inputs"]["identity"].pop("protocol_sha256")
        with self.assertRaises(RuntimeError):
            self._validate("resume", payload)

        payload = self._payload("resume")
        payload["inputs"]["artifacts"][0]["path"] = "relative.log"
        with self.assertRaises(RuntimeError):
            self._validate("resume", payload)


if __name__ == "__main__":
    unittest.main()
