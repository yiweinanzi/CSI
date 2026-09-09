"""Config-only tests for the diagnostic Sionna visibility protocol.

The tested protocol is a simulation diagnostic and is not formal scientific evidence.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


WORKING_COPY_ROOT = Path(__file__).resolve().parents[4]
TOOL_PATH = (
    WORKING_COPY_ROOT
    / "artifacts"
    / "formal_readiness"
    / "tools"
    / "diagnose_sionna_visibility_one_factor.py"
)
CONFIG_PATH = (
    WORKING_COPY_ROOT
    / "artifacts"
    / "formal_readiness"
    / "configs"
    / "sionna_visibility_one_factor_v1.json"
)
SPEC = importlib.util.spec_from_file_location("sionna_visibility_one_factor", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load diagnostic tool: {TOOL_PATH}")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class SionnaVisibilityOneFactorConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG_PATH.read_text(encoding="ascii"))

    def _write(self, value: dict, root: Path) -> Path:
        path = root / "config.json"
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="ascii",
        )
        return path

    def _assert_rejected(self, value: dict, message: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(value, Path(temporary))
            with self.assertRaisesRegex(ValueError, message):
                diagnostic.load_diagnostic_config(path)

    def test_registered_config_has_exact_one_factor_coverage(self) -> None:
        loaded = diagnostic.load_diagnostic_config(CONFIG_PATH)
        rows = diagnostic.condition_rows(loaded)
        self.assertEqual(len(rows), 1 + len(diagnostic.FACTOR_NAMES))
        self.assertIsNone(rows[0]["single_change"])
        baseline = rows[0]["factors"]
        changed = []
        for row in rows[1:]:
            differences = [
                name
                for name in diagnostic.FACTOR_NAMES
                if not diagnostic._same_value(row["factors"][name], baseline[name])
            ]
            self.assertEqual(differences, [row["single_change"]["factor"]])
            changed.extend(differences)
        self.assertEqual(set(changed), set(diagnostic.FACTOR_NAMES))
        self.assertEqual(len(changed), len(set(changed)))
        self.assertEqual(loaded["scene_index"], 27)
        self.assertEqual(loaded["solver_seed"], 2026107904)
        self.assertTrue(loaded["simulation_not_measurement"])
        self.assertEqual(loaded["scientific_use"], diagnostic.SCIENTIFIC_USE)

    def test_extra_root_field_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.config)
        mutated["unregistered_override"] = True
        self._assert_rejected(mutated, "fields must be exact")

    def test_variant_from_must_equal_typed_baseline_value(self) -> None:
        mutated = copy.deepcopy(self.config)
        mutated["variants"][3]["single_change"]["from"] = 3.0
        self._assert_rejected(mutated, "does not bind from")

    def test_duplicate_factor_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.config)
        mutated["variants"][1]["single_change"] = copy.deepcopy(
            mutated["variants"][0]["single_change"]
        )
        self._assert_rejected(mutated, "exactly one variant")

    def test_variant_cannot_carry_a_private_seed(self) -> None:
        mutated = copy.deepcopy(self.config)
        mutated["variants"][0]["solver_seed"] = 99
        self._assert_rejected(mutated, "fields must be exact")

    def test_runtime_must_be_fixed_llvm_single_thread(self) -> None:
        mutated = copy.deepcopy(self.config)
        mutated["drjit_threads"] = True
        self._assert_rejected(mutated, "requires LLVM with one Dr.Jit thread")

    def test_fixed_solver_booleans_are_type_strict(self) -> None:
        mutated = copy.deepcopy(self.config)
        mutated["fixed_solver_settings"]["los"] = 1
        self._assert_rejected(mutated, "fixed_solver_settings differ")

    def test_capacity_metrics_separate_raw_visibility_from_storage(self) -> None:
        raw = np.asarray((0, 1, 16, 17, 80), dtype=np.int64)
        metrics = diagnostic._condition_metrics(raw, 16)
        self.assertEqual(metrics["visible_positions"], 4)
        self.assertEqual(metrics["no_path_positions"], 1)
        self.assertEqual(metrics["no_path_rate"], 0.2)
        capacity = metrics["capacity_diagnostics"]
        self.assertEqual(capacity["raw_path_total"], 114)
        self.assertEqual(capacity["retained_path_total"], 49)
        self.assertEqual(capacity["truncated_path_total"], 65)
        self.assertEqual(capacity["capacity_reached_positions"], 3)
        self.assertEqual(capacity["saturated_positions"], 3)
        self.assertEqual(capacity["saturation_rate"], 0.6)
        self.assertEqual(capacity["capacity_exceeded_positions"], 2)
        self.assertFalse(capacity["physical_visibility_changed_by_storage_projection"])
        raw_histogram = metrics["path_count_distribution"]["raw_solver_paths"]["histogram"]
        stored_histogram = metrics["path_count_distribution"][
            "stored_after_capacity_projection"
        ]["histogram"]
        self.assertEqual([row["path_count"] for row in raw_histogram], [0, 1, 16, 17, 80])
        self.assertEqual([row["path_count"] for row in stored_histogram], [0, 1, 16])


if __name__ == "__main__":
    unittest.main()
