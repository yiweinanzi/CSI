from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from formal_v2 import formal_evidence
from formal_v2.scripts import build_v6_requirement_matrix as matrix


SERVER_ROOT = Path(__file__).resolve().parents[2]


class AtomicRequirementMatrixTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rows = [{field: field for field in matrix.FIELDNAMES}]

    def tearDown(self):
        self.temporary.cleanup()

    def test_claim_contract_uses_nonrecursive_delivery_provenance(self):
        contract = json.loads(
            (SERVER_ROOT / "artifacts/v2_0_claim_evidence_contract.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            contract["schema_version"],
            "csi-pairs-v6-claim-evidence-contract-v2.6",
        )
        self.assertNotIn("audited_code_sha", contract)
        self.assertEqual(
            contract["review_base_sha"],
            "bf5764afb52cc5c29fd41f230b66f9d869dfb8cb",
        )
        self.assertIn("SHA256SUMS", contract["delivery_binding"])

    def test_existing_private_target_leaves_public_target_absent(self):
        public = self.root / "public.csv"
        private = self.root / "private.csv"
        private.write_text("preserve\n", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            matrix.write_matrix_pair(public, self.rows, private, self.rows)

        self.assertFalse(public.exists())
        self.assertEqual(private.read_text(encoding="utf-8"), "preserve\n")

    def test_second_publish_failure_rolls_back_first_target(self):
        public = self.root / "public.csv"
        private = self.root / "private.csv"
        real_link = os.link
        calls = 0

        def fail_second_link(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected second-output failure")
            return real_link(source, destination)

        with patch.object(matrix.os, "link", side_effect=fail_second_link):
            with self.assertRaisesRegex(OSError, "injected second-output failure"):
                matrix.write_matrix_pair(public, self.rows, private, self.rows)

        self.assertFalse(public.exists())
        self.assertFalse(private.exists())
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_output_paths_must_be_distinct(self):
        output = self.root / "matrix.csv"
        with self.assertRaisesRegex(ValueError, "different paths"):
            matrix.write_matrix_pair(output, self.rows, output, self.rows)
        self.assertFalse(output.exists())

    def test_normative_clause_uses_semantic_family_and_exact_clause_hash(self):
        source = self.root / "reader.md"
        source.write_text(
            "## 0. Private heading\n必须保持非因果 claim 边界。\n",
            encoding="utf-8",
        )
        source_spec = {
            "sha256": matrix.sha256_file(source),
            "prefix": "TEST",
        }
        with patch.dict(matrix.SOURCE_SPECS, {"reader": source_spec}):
            rows = matrix.build_rows("reader", source, matrix.Path.cwd(), True)

        normative = next(
            row for row in rows if matrix.is_normative_signal(row["normative_signal"])
        )
        self.assertEqual(normative["evidence_family"], "claim_boundary")
        self.assertEqual(normative["status"], "EXACT")
        self.assertEqual(
            normative["mapping_basis"],
            "semantic-family:claim_boundary;"
            f"clause-sha256:{normative['source_clause_sha256']}",
        )
        for field in (
            "runtime_entry",
            "code_locations",
            "config_or_schema_locations",
            "regression_test_locations",
            "dynamic_evidence",
        ):
            self.assertTrue(normative[field])
            self.assertNotIn("NOT_APPLICABLE", normative[field])

    def test_unknown_normative_subsection_has_no_section_fallback(self):
        source = self.root / "reader.md"
        source.write_text(
            "## 0. Private heading\n### 0.7 Unmapped subsection\n必须执行新规则。\n",
            encoding="utf-8",
        )
        source_spec = {
            "sha256": matrix.sha256_file(source),
            "prefix": "TEST",
        }
        with patch.dict(matrix.SOURCE_SPECS, {"reader": source_spec}):
            with self.assertRaisesRegex(RuntimeError, "no semantic trace family"):
                matrix.build_rows("reader", source, matrix.Path.cwd(), True)

    def test_markdown_table_data_cells_are_normative_without_keyword_guessing(self):
        source = self.root / "reader.md"
        source.write_text(
            "## 0. Overview\n"
            "| Requirement | Evidence |\n"
            "|---|---|\n"
            "| Per-position map construction | Mandatory regression coverage |\n",
            encoding="utf-8",
        )
        source_spec = {
            "sha256": matrix.sha256_file(source),
            "prefix": "TEST",
        }
        with patch.dict(matrix.SOURCE_SPECS, {"reader": source_spec}):
            rows = matrix.build_rows("reader", source, matrix.Path.cwd(), True)

        header = [row for row in rows if row["source_line"] == "2"]
        data = [row for row in rows if row["source_line"] == "4"]
        self.assertTrue(header)
        self.assertTrue(data)
        self.assertTrue(all(not matrix.is_normative_signal(row["normative_signal"]) for row in header))
        self.assertTrue(all(row["normative_signal"] == "NORMATIVE_TABLE_CELL" for row in data))
        self.assertTrue(all(row["evidence_family"] != "not_normative_context" for row in data))

    def test_rq_and_claim_gate_clauses_keep_external_evidence_boundaries(self):
        family_for_clause = matrix.evidence_family_for_clause

        self.assertEqual(
            family_for_clause(
                8,
                "RQ1：现有模型是否反事实一致地使用地图",
                "paired-context p_fail 是否随真实失败上升。",
            ),
            "rq_wrong_map",
        )
        self.assertEqual(
            family_for_clause(
                8,
                "RQ5：paired context 下 compatibility 能否转化为可靠的定位风险",
                "q_comp 与定位风险必须分表。",
            ),
            "rq_risk",
        )
        self.assertEqual(
            family_for_clause(
                14,
                "14. Claim-Evidence Gate",
                "不能拿 q_comp 代替定位风险。",
            ),
            "claim_gates",
        )
        self.assertEqual(
            family_for_clause(5, "5.3 q_comp 只比较一对候选地图", "q_comp 必须单独拟合。"),
            "q_comp",
        )
        self.assertEqual(
            family_for_clause(5, "5.4 p_fail 预测定位会不会失败", "p_fail 必须单独拟合。"),
            "p_fail",
        )

    def test_result_rows_and_frozen_gauge_are_separately_classified(self):
        source = self.root / "reader.md"
        source.write_text(
            "## 3. Routing\n### 3.1 Teacher\n"
            "所有 delta target 必须采用 pair-consistent phase gauge。\n"
            "## 13. Main results\nPlanned result panel.\n",
            encoding="utf-8",
        )
        source_spec = {
            "sha256": matrix.sha256_file(source),
            "prefix": "TEST",
        }
        with patch.dict(matrix.SOURCE_SPECS, {"reader": source_spec}):
            rows = matrix.build_rows("reader", source, matrix.Path.cwd(), True)

        gauge = next(row for row in rows if row["evidence_family"] == "gauge")
        self.assertEqual(gauge["status"], "EXACT")
        self.assertEqual(gauge["blocking_type"], "NONE")
        results = [row for row in rows if row["section"] == "13"]
        self.assertTrue(results)
        self.assertEqual({row["evidence_family"] for row in results}, {"results"})
        self.assertEqual({row["status"] for row in results}, {"MISSING"})

    def test_public_rows_redact_clause_and_heading_source_text(self):
        source = self.root / "reader.md"
        source.write_text("## 0. Private heading\n必须保留私有条款。\n", encoding="utf-8")
        source_spec = {
            "sha256": matrix.sha256_file(source),
            "prefix": "TEST",
        }
        with patch.dict(matrix.SOURCE_SPECS, {"reader": source_spec}):
            public_rows = matrix.build_rows("reader", source, matrix.Path.cwd(), False)
            private_rows = matrix.build_rows("reader", source, matrix.Path.cwd(), True)

        self.assertTrue(public_rows)
        self.assertEqual(
            {row["original_norm"] for row in public_rows},
            {"[OMITTED_FROM_PUBLIC_REPOSITORY]"},
        )
        self.assertEqual(
            {row["section_heading"] for row in public_rows},
            {"[OMITTED_FROM_PUBLIC_REPOSITORY]"},
        )
        self.assertTrue(any("私有条款" in row["original_norm"] for row in private_rows))
        self.assertEqual(
            {row["section_heading"] for row in private_rows},
            {"0. Private heading"},
        )
        context = next(
            row for row in private_rows if not matrix.is_normative_signal(row["normative_signal"])
        )
        self.assertEqual(context["evidence_family"], "not_normative_context")
        self.assertEqual(context["blocking_type"], "NOT_NORMATIVE_CONTEXT")

    def test_tracked_public_matrix_has_atomic_fail_closed_evidence(self):
        server_root = Path(__file__).resolve().parents[2]
        repository_root = server_root.parents[1]
        authority = (
            repository_root
            / "Idea1-CSI-PAIRS-冻结版-零基础阅读稿-v6_VSCode兼容版.md"
        )
        artifact = server_root / "artifacts" / "v6_atomic_requirement_matrix_2026-08-08.csv"
        with artifact.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        if authority.is_file():
            expected_rows = matrix.build_rows(
                "reader",
                authority,
                server_root,
                False,
            )
            self.assertEqual(rows, expected_rows)
        else:
            expected_rows = rows
        self.assertEqual(len(rows), len(expected_rows))
        self.assertEqual(tuple(rows[0]), matrix.FIELDNAMES)
        self.assertGreaterEqual(len({row["evidence_family"] for row in rows}), 40)
        self.assertEqual(
            {row["original_norm"] for row in rows},
            {"[OMITTED_FROM_PUBLIC_REPOSITORY]"},
        )
        self.assertEqual(
            {row["section_heading"] for row in rows},
            {"[OMITTED_FROM_PUBLIC_REPOSITORY]"},
        )
        for row in rows:
            if row["section"] == "13":
                self.assertEqual(row["evidence_family"], "results")
                self.assertEqual(row["status"], "MISSING")
            elif matrix.is_normative_signal(row["normative_signal"]):
                self.assertNotEqual(row["evidence_family"], "not_normative_context")
                self.assertEqual(
                    row["mapping_basis"],
                    f"semantic-family:{row['evidence_family']};"
                    f"clause-sha256:{row['source_clause_sha256']}",
                )
            else:
                self.assertEqual(row["evidence_family"], "not_normative_context")
                self.assertEqual(row["status"], "PARTIAL/PROXY")
                self.assertEqual(row["blocking_type"], "NOT_NORMATIVE_CONTEXT")
            if row["status"] == "EXACT":
                self.assertEqual(row["blocking_type"], "NONE")
                for field in (
                    "runtime_entry",
                    "code_locations",
                    "config_or_schema_locations",
                    "regression_test_locations",
                    "dynamic_evidence",
                ):
                    self.assertTrue(row[field])
                    self.assertNotIn("NOT_APPLICABLE", row[field])
        gauge_rows = [row for row in rows if row["evidence_family"] == "gauge"]
        self.assertTrue(gauge_rows)
        self.assertEqual({row["status"] for row in gauge_rows}, {"EXACT"})
        self.assertEqual({row["blocking_type"] for row in gauge_rows}, {"NONE"})


class RequirementsLockTests(unittest.TestCase):
    def setUp(self):
        self.server_root = Path(__file__).resolve().parents[2]
        self.macos_lock = self.server_root / "formal_v2" / "requirements-lock.txt"
        self.linux_lock = (
            self.server_root
            / "formal_v2"
            / "requirements-lock-linux-x86_64-cu121.txt"
        )
        self.lock = formal_evidence._requirements_lock_path()

    def test_hashed_lock_has_complete_supported_platform_closures(self):
        macos = formal_evidence._locked_requirement_versions(
            self.macos_lock,
            sys_platform_value="darwin",
            machine_value="arm64",
        )
        linux = formal_evidence._locked_requirement_versions(
            self.linux_lock,
            sys_platform_value="linux",
            machine_value="x86_64",
        )

        common = {
            "filelock",
            "fsspec",
            "iniconfig",
            "Jinja2",
            "MarkupSafe",
            "mpmath",
            "networkx",
            "numpy",
            "packaging",
            "pluggy",
            "Pygments",
            "pytest",
            "scipy",
            "shapely",
            "setuptools",
            "sympy",
            "torch",
            "typing_extensions",
        }
        linux_only = {
            "nvidia-cublas-cu12",
            "nvidia-cuda-cupti-cu12",
            "nvidia-cuda-nvrtc-cu12",
            "nvidia-cuda-runtime-cu12",
            "nvidia-cudnn-cu12",
            "nvidia-cufft-cu12",
            "nvidia-curand-cu12",
            "nvidia-cusolver-cu12",
            "nvidia-cusparse-cu12",
            "nvidia-nccl-cu12",
            "nvidia-nvjitlink-cu12",
            "nvidia-nvtx-cu12",
            "triton",
        }
        self.assertEqual(set(macos), common)
        self.assertEqual(set(linux), common | linux_only)
        self.assertEqual(macos["torch"], "2.13.0")
        self.assertEqual(linux["torch"], "2.5.1+cu121")
        self.assertNotIn("triton", macos)
        self.assertEqual(linux["triton"], "3.1.0")
        self.assertEqual(linux["nvidia-cudnn-cu12"], "9.1.0.70")
        self.assertEqual(linux["nvidia-nccl-cu12"], "2.21.5")

    def test_runtime_selects_the_target_specific_lock(self):
        self.assertEqual(
            formal_evidence._requirements_lock_path(
                sys_platform_value="darwin", machine_value="arm64"
            ),
            self.macos_lock,
        )
        self.assertEqual(
            formal_evidence._requirements_lock_path(
                sys_platform_value="linux", machine_value="x86_64"
            ),
            self.linux_lock,
        )

    def test_torch_local_version_is_exact_when_the_lock_includes_it(self):
        self.assertTrue(
            formal_evidence._torch_module_version_matches_lock(
                "2.5.1+cu121", "2.5.1+cu121"
            )
        )
        self.assertFalse(
            formal_evidence._torch_module_version_matches_lock(
                "2.5.1+cu118", "2.5.1+cu121"
            )
        )
        self.assertTrue(
            formal_evidence._torch_module_version_matches_lock(
                "2.13.0+cu130", "2.13.0"
            )
        )
        self.assertFalse(
            formal_evidence._torch_module_version_matches_lock("2.13.1", "2.13.0")
        )

    def test_unhashed_or_unknown_marker_lock_entries_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "requirements-lock.txt"
            candidate.write_text("numpy==2.3.5\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not exact and hashed"):
                formal_evidence._locked_requirement_versions(candidate)

            candidate.write_text(
                'numpy==2.3.5 ; sys_platform == "darwin" '
                '--hash=sha256:' + "a" * 64 + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "marker is unsupported"):
                formal_evidence._locked_requirement_versions(candidate)

    def test_unsupported_platform_cannot_select_the_common_lock_subset(self):
        with self.assertRaisesRegex(RuntimeError, "supports only macOS arm64"):
            formal_evidence._locked_requirement_versions(
                self.lock,
                sys_platform_value="win32",
                machine_value="AMD64",
            )

    def test_installer_report_authenticates_each_selected_wheel(self):
        records = formal_evidence._locked_requirement_records(self.lock)
        report_path = Path(sys.prefix) / formal_evidence.INSTALL_REPORT_NAME
        receipt = formal_evidence._wheel_receipt_from_install_report(
            report_path,
            records,
        )
        self.assertEqual(
            set(receipt),
            {formal_evidence._normalize_distribution_name(name) for name in records},
        )

        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["install"][0]["download_info"]["archive_info"]["hashes"][
            "sha256"
        ] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            mutated = Path(temporary) / "install-report.json"
            mutated.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not match the wheel lock"):
                formal_evidence._wheel_receipt_from_install_report(
                    mutated,
                    records,
                )

    def test_distribution_record_detects_installed_file_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            package = prefix / "example" / "module.py"
            package.parent.mkdir()
            package.write_bytes(b"reviewed-wheel-content\n")
            dist_info = prefix / "example-1.0.dist-info"
            dist_info.mkdir()
            encoded = base64.urlsafe_b64encode(
                hashlib.sha256(package.read_bytes()).digest()
            ).rstrip(b"=").decode("ascii")
            record_text = (
                f"example/module.py,sha256={encoded},{package.stat().st_size}\n"
                "example/__pycache__/module.cpython-312.pyc,,\n"
                "example-1.0.dist-info/RECORD,,\n"
            )
            record = dist_info / "RECORD"
            record.write_text(record_text, encoding="utf-8")

            class Distribution:
                metadata = {"Name": "example"}

                @staticmethod
                def read_text(name):
                    return record_text if name == "RECORD" else None

                @staticmethod
                def locate_file(relative):
                    return prefix / relative

            formal_evidence._validated_distribution_record(
                Distribution(),
                prefix.resolve(),
            )
            package.write_bytes(b"mutated-installed-content\n")
            with self.assertRaisesRegex(RuntimeError, "differs from RECORD"):
                formal_evidence._validated_distribution_record(
                    Distribution(),
                    prefix.resolve(),
                )

    def test_install_entrypoint_requires_hashes_and_wheels(self):
        setup = (self.server_root / "formal_v2" / "scripts" / "setup_formal_v2.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--require-hashes", setup)
        self.assertIn("--only-binary=:all:", setup)
        self.assertIn("--report", setup)
        self.assertIn("CSI_PAIRS_PIP_CERT", setup)
        self.assertNotIn("--trusted-host", setup)

    def test_source_ci_uses_the_hashed_setup_entrypoint(self):
        repository_root = self.server_root.parents[1]
        if not (repository_root / ".git").exists():
            self.skipTest("source CI workflow is outside the standalone server delivery")
        workflow = (
            repository_root / ".github" / "workflows" / "formal-v2-cpu-ci.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("setup_formal_v2.sh", workflow)

    def test_server_builder_stages_public_trace_dependencies(self):
        builder = (
            self.server_root / "formal_v2" / "scripts" / "build_server_bundle.sh"
        ).read_text(encoding="utf-8")
        for relative in (
            "code_to_paper_reverse_matrix.md",
            "formal_experiment_blockers.md",
            "paper_to_code_traceability.md",
            "source_conflict_register.md",
            "v6_atomic_requirement_matrix_2026-08-08.csv",
        ):
            with self.subTest(relative=relative):
                self.assertIn(relative, builder)

    def test_server_builder_is_independent_of_calling_directory(self):
        builder = self.server_root / "formal_v2" / "scripts" / "build_server_bundle.sh"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "server.zip"
            completed = subprocess.run(
                [str(builder), str(output)],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.is_file())
            self.assertTrue(Path(f"{output}.sha256").is_file())


if __name__ == "__main__":
    unittest.main()
