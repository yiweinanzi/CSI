from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

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

from formal_v2.formal_cli import (
    _require_full_stage,
    _resolve_all_start_mode,
    _run_evaluation_for_upstream,
    _write_engineering_preflight_skip,
    _write_engineering_unapproved_marker,
    build_parser,
)


class CliEvaluationDefaultTests(TestCase):
    def test_non_migrated_evaluation_defaults_to_streaming(self) -> None:
        config = {"config": "sentinel"}
        dataset = object()
        output = Path("/new-run")
        upstream = SimpleNamespace(
            migrated=False,
            migration=None,
            qualification_gate=Path("/new-run/qualification/gate.json"),
            factorial_gate=Path("/new-run/factorial/gate.json"),
            checkpoint_index=Path("/new-run/factorial/checkpoint_index.json"),
        )
        execution = SimpleNamespace(
            identity=SimpleNamespace(legacy_checkpoint_inventory_sha256="0" * 64),
            devices=("cuda:0",),
            batch_size=256,
        )
        expected = {"status": "PASS", "passed": True}

        with patch(
            "formal_v2.formal_evaluation_identity.build_local_evaluation_execution",
            return_value=execution,
        ) as build_local, patch(
            "formal_v2.formal_evaluation_streaming.run_streaming_formal_evaluation",
            return_value=expected,
        ) as run_streaming, patch(
            "formal_v2.formal_evaluation.run_formal_evaluation",
        ) as run_legacy:
            actual = _run_evaluation_for_upstream(
                config,
                dataset,
                output,
                upstream,
                {"stage": "qualification"},
                {"stage": "factorial"},
            )

        self.assertIs(actual, expected)
        run_legacy.assert_not_called()
        build_local.assert_called_once_with(config, dataset, output, upstream)
        self.assertEqual(run_streaming.call_args.kwargs["batch_size"], 256)

    def test_all_rejects_legacy_evaluator_flag(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "all",
                    "--config",
                    "config.json",
                    "--output",
                    "output",
                    "--adapter-manifest",
                    "adapters.json",
                    "--engineering-run",
                    "--legacy-evaluator",
                ]
            )

    def test_legacy_evaluator_flag_keeps_importable_legacy_module(self) -> None:
        expected = {"status": "PASS", "passed": True}
        with patch(
            "formal_v2.formal_evaluation.run_formal_evaluation",
            return_value=expected,
        ) as run_legacy:
            actual = _run_evaluation_for_upstream(
                {"config": "sentinel"},
                object(),
                Path("/new-run"),
                SimpleNamespace(migrated=False, migration=None),
                {"stage": "qualification"},
                {"stage": "factorial"},
                legacy_evaluator=True,
            )
        self.assertIs(actual, expected)
        run_legacy.assert_called_once()

    def test_continue_on_stage_fail_records_and_continues(self) -> None:
        failures: list[dict[str, object]] = []
        _require_full_stage(
            {"status": "INVALID"},
            "engineering stage",
            SimpleNamespace(is_fixture=False),
            continue_on_fail=True,
            failures=failures,
        )
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["stage"], "engineering stage")
        with self.assertRaisesRegex(RuntimeError, "incomplete or invalid"):
            _require_full_stage(
                {"status": "INVALID"},
                "engineering stage",
                SimpleNamespace(is_fixture=False),
            )

    def test_engineering_run_is_a_start_mode_and_writes_nonclaim_marker(self) -> None:
        import tempfile

        parsed = build_parser().parse_args(
            [
                "all",
                "--config",
                "config.json",
                "--output",
                "output",
                "--adapter-manifest",
                "adapters.json",
                "--engineering-run",
            ]
        )
        self.assertEqual(_resolve_all_start_mode(parsed), "engineering")
        with tempfile.TemporaryDirectory() as directory:
            marker = _write_engineering_unapproved_marker(
                Path(directory),
                reason="--engineering-run",
            )
            payload = marker.read_text(encoding="ascii")
            skip = _write_engineering_preflight_skip(Path(directory))
            skip_payload = skip.read_text(encoding="ascii")
        self.assertIn("ENGINEERING_UNAPPROVED", payload)
        self.assertIn("NON_CLAIM", payload)
        self.assertIn("ENGINEERING_PREFLIGHT_SKIPPED", str(skip))
        self.assertIn("SKIPPED", skip_payload)
        self.assertIn("preflight_full_run", skip_payload)
