from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from formal_v2.formal_sionna_osm_verifier import _run_process_scheduler


class IsolatedSionnaVerifierTests(unittest.TestCase):
    def _fake_command(self, state: Path, scene_index: int, output: Path) -> list[str]:
        script = (
            "from pathlib import Path; import sys; "
            "scene=int(sys.argv[1]); out=Path(sys.argv[2]); state=Path(sys.argv[3]); "
            "attempt=int(state.read_text())+1 if state.exists() else 1; "
            "state.write_text(str(attempt)); "
            "sys.exit(17) if scene==2 and attempt==1 else out.write_text(str(scene))"
        )
        return [sys.executable, "-c", script, str(scene_index), str(output), str(state)]

    def test_scheduler_retries_only_failed_bank_and_preserves_attempt_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def command(scene_index: int, output: Path) -> list[str]:
                return self._fake_command(root / f"state-{scene_index}", scene_index, output)

            paths = _run_process_scheduler(
                5,
                root / "run",
                3,
                command,
                lambda path: Path(path).read_text(encoding="utf-8"),
                poll_interval=0.01,
            )
            self.assertEqual([path.read_text() for path in paths], ["0", "1", "2", "3", "4"])
            self.assertEqual((root / "state-2").read_text(), "2")
            failed = root / "run/bank_attempts/bank-002/attempt-01.stderr.txt"
            self.assertTrue(failed.is_file())
            self.assertTrue((root / "run/bank_attempts/bank-002/attempt-02.npz").is_file())

    def test_scheduler_fails_closed_after_retry_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def command(scene_index: int, output: Path) -> list[str]:
                return [sys.executable, "-c", "raise SystemExit(23)"]

            with self.assertRaisesRegex(RuntimeError, "failed after 2 isolated attempts"):
                _run_process_scheduler(
                    1,
                    root / "run",
                    1,
                    command,
                    lambda path: Path(path).read_bytes(),
                    max_attempts=2,
                    poll_interval=0.01,
                )
            self.assertEqual(
                len(list((root / "run/bank_attempts/bank-000").glob("attempt-*.stderr.txt"))),
                2,
            )


if __name__ == "__main__":
    unittest.main()
