import tempfile
import unittest
from pathlib import Path

from formal_v2.formal_evaluation_repair import require_execution_only_changes


class EvaluationRepairTests(unittest.TestCase):
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
