import tempfile
import unittest
from pathlib import Path
import numpy as np

from formal_v2.formal_evaluation_repair import require_execution_only_changes, validate_corpus_arrays, load_validation_corpus


class EvaluationRepairTests(unittest.TestCase):
    def test_validation_corpus_manifest_must_match_the_D_receipt(self):
        path = Path(self.temporary.name) / "manifest.json"
        path.write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "manifest identity differs"):
            load_validation_corpus(path, {}, expected_sha256="0" * 64)

    def test_dual_gpu_overlap_uses_real_kernel_interval_intersection(self):
        from formal_v2.tools.validate_formal_dual_probe import kernel_overlap

        events = [
            {"cat": "kernel", "args": {"device": 0}, "ts": 1, "dur": 4},
            {"cat": "kernel", "args": {"device": 0}, "ts": 2, "dur": 2},
            {"cat": "kernel", "args": {"device": 1}, "ts": 3, "dur": 3},
            {"cat": "gpu_memcpy", "args": {"device": 1}, "ts": 1, "dur": 50},
        ]
        result = kernel_overlap(events)
        self.assertEqual(result["actual_kernel_overlap_microseconds"], 2)
        self.assertEqual(result["kernel_counts"], {"0": 2, "1": 1})

    def test_validation_corpus_retains_full_shape_and_original_dtypes(self):
        arrays = {
            "train_features": np.ones((4, 3), dtype=np.float64),
            "train_labels": np.array([0, 1, 0, 1], dtype=np.int64),
            "selection_features": np.ones((2, 3), dtype=np.float64),
            "selection_labels": np.array([0, 1], dtype=np.int64),
        }
        validate_corpus_arrays(arrays)
        for replacement in (np.ones((3, 3)), np.ones((4, 3), dtype=np.float32), np.full((4, 3), np.nan)):
            with self.assertRaises(RuntimeError):
                validate_corpus_arrays({**arrays, "train_features": replacement})
        with self.assertRaises(RuntimeError):
            validate_corpus_arrays({**arrays, "query_features": np.ones((1, 3))})

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.origin = Path(self.temporary.name) / "old"
        self.evaluation = Path(self.temporary.name) / "new"
        for root in (self.origin, self.evaluation):
            root.mkdir()
            (root / "formal_metrics.py").write_text("def binary_auroc():\n    return 1\ndef other():\n    return 2\n")
            (root / "formal_evaluation.py").write_text("def _prepare_alignment_shortcut_probes():\n    return 1\ndef localization():\n    return 2\n")
            (root / "formal_probes.py").write_text("class CompatibilityProbe:\n    pass\nclass ActionResponseProbe:\n    pass\n")
            (root / "formal_model.py").write_text("WEIGHT = 1\n")

    def test_execution_only_change_is_reported_with_both_hashes(self):
        path = self.evaluation / "formal_metrics.py"
        path.write_text(path.read_text().replace("return 1", "return 3"))
        changes = require_execution_only_changes(self.origin, self.evaluation)
        self.assertEqual(len(changes), 1)
        self.assertNotEqual(changes[0]["upstream_sha256"], changes[0]["evaluation_sha256"])

    def test_model_method_and_configuration_changes_are_rejected(self):
        for name in ("formal_model.py", "config.json"):
            with self.subTest(name=name):
                path = self.evaluation / name
                original = path.read_text() if path.exists() else None
                path.write_text("changed")
                with self.assertRaisesRegex(RuntimeError, "non-evaluation"):
                    require_execution_only_changes(self.origin, self.evaluation)
                if original is None:
                    path.unlink()
                else:
                    path.write_text(original)

    def test_unrelated_metric_or_localization_function_cannot_change(self):
        for name in ("formal_metrics.py", "formal_evaluation.py"):
            path = self.evaluation / name
            original = path.read_text()
            path.write_text(original.replace("return 2", "return 4"))
            with self.assertRaisesRegex(RuntimeError, "unrelated functions"):
                require_execution_only_changes(self.origin, self.evaluation)
            path.write_text(original)

    def test_probe_architecture_change_is_rejected(self):
        path = self.evaluation / "formal_probes.py"
        path.write_text(path.read_text().replace("pass", "WIDTH = 9", 1))
        with self.assertRaisesRegex(RuntimeError, "architecture"):
            require_execution_only_changes(self.origin, self.evaluation)
