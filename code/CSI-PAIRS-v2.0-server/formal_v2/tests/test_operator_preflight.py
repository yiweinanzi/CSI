from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from formal_v2.formal_io import sha256_file
from formal_v2.formal_operator_preflight import (
    REPORT_KEYS,
    SCIENTIFIC_USE,
    required_prepare_full_run_manifest_flags,
    run_operator_preflight,
)
from formal_v2.formal_run_approval import REQUIRED_FULL_RUN_INPUT_NAMES


class OperatorPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_missing_dataset_is_structured_failure(self):
        missing = self.root / "absent.npz"
        report = run_operator_preflight(missing)
        self.assertEqual(set(report), set(REPORT_KEYS))
        self.assertFalse(report["dataset"]["ok"])
        self.assertFalse(report["dataset"]["exists"])
        self.assertFalse(report["dataset"]["regular_file"])
        self.assertIsNone(report["dataset"]["size_bytes"])
        self.assertIsNone(report["dataset"]["sha256"])
        self.assertFalse(report["dataset_load"]["loaded"])
        self.assertFalse(report["formal_ready_to_prepare"])
        self.assertEqual(report["scientific_use"], SCIENTIFIC_USE)
        self.assertNotIn("FORMAL_GO", report)
        self.assertIn("does not exist", report["dataset"]["message"])

    def test_win32_reports_formal_host_false(self):
        missing = self.root / "absent.npz"
        with (
            patch("formal_v2.formal_operator_preflight.sys.platform", "win32"),
            patch(
                "formal_v2.formal_operator_preflight.platform.machine",
                return_value="AMD64",
            ),
        ):
            report = run_operator_preflight(missing)
        self.assertFalse(report["formal_host"])
        self.assertFalse(report["platform"]["formal_host"])
        self.assertFalse(report["platform"]["ok"])
        self.assertEqual(report["platform"]["system"], "win32")
        self.assertIn("win32 is not a formal host", report["platform"]["message"])
        self.assertFalse(report["formal_ready_to_prepare"])

    def test_report_has_exact_keys(self):
        payload = self.root / "not-npz.bin"
        payload.write_bytes(b"not-an-npz")
        output = self.root / "inventory"
        report = run_operator_preflight(payload, output_root=output)
        self.assertEqual(tuple(sorted(report)), tuple(sorted(REPORT_KEYS)))
        written = json.loads((output / "operator_preflight.json").read_text(encoding="utf-8"))
        self.assertEqual(set(written), set(REPORT_KEYS))
        self.assertTrue(report["dataset"]["ok"])
        self.assertEqual(report["dataset"]["size_bytes"], 10)
        self.assertEqual(report["dataset"]["sha256"], sha256_file(payload))
        self.assertFalse(report["dataset_load"]["loaded"])
        self.assertEqual(
            report["required_prepare_full_run_manifest_flags"],
            [f"--{name.replace('_', '-')}" for name in REQUIRED_FULL_RUN_INPUT_NAMES],
        )
        self.assertEqual(
            report["required_prepare_full_run_manifest_flags"],
            required_prepare_full_run_manifest_flags(),
        )
        self.assertFalse(report["compute_plan"]["measurement_validated"])
        self.assertIn("not measurement-validated", report["compute_plan"]["message"])
        for name in (
            "platform",
            "python",
            "dataset",
            "dataset_load",
            "waibu",
            "environment",
            "compute_plan",
        ):
            self.assertIn("ok", report[name])
            self.assertIn("message", report[name])
        self.assertNotIn("FORMAL_GO", report)
        self.assertEqual(report["scientific_use"], "NOT_ASSESSED")


if __name__ == "__main__":
    unittest.main()
