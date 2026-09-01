from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

if "fcntl" not in sys.modules:
    try:
        import fcntl as _fcntl
    except ImportError:
        _fcntl = types.ModuleType("fcntl")
        _fcntl.LOCK_EX = 2
        _fcntl.LOCK_SH = 1
        _fcntl.LOCK_UN = 8
        _fcntl.LOCK_NB = 4
        _fcntl.flock = lambda *_args, **_kwargs: None
        sys.modules["fcntl"] = _fcntl

from formal_v2.formal_evaluation_streaming import (
    PUBLIC_EVALUATION_STATUS_SCHEMA,
    _write_public_evaluation_status,
)


class PublicEvaluationStatusTests(unittest.TestCase):
    def test_status_json_records_fraction_and_last_shard_without_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SimpleNamespace(
                root=root / "evaluation_state",
                read_status=lambda: {
                    "status": "RUNNING",
                    "total_units": 10,
                    "completed_units": 3,
                    "current_shard": "seed-1/full/scene-4",
                    "current_seed": 1,
                    "current_arm": "full",
                    "current_substage": "scene",
                },
            )
            with mock.patch(
                "formal_v2.formal_evaluation_streaming.write_atomic_json",
                side_effect=lambda path, payload: Path(path).write_text(
                    json.dumps(payload, sort_keys=True) + "\n",
                    encoding="ascii",
                ),
            ):
                _write_public_evaluation_status(store)
            payload = json.loads(
                (root / "evaluation" / "status.json").read_text(encoding="ascii")
            )
        self.assertEqual(payload["schema_version"], PUBLIC_EVALUATION_STATUS_SCHEMA)
        self.assertEqual(payload["fraction_complete"], 0.3)
        self.assertEqual(payload["last_shard"], "seed-1/full/scene-4")
        self.assertIsNone(payload["authoritative_gate"])
        self.assertNotIn("passed", payload)
        self.assertIn("gate.json", payload["note"])


if __name__ == "__main__":
    unittest.main()
