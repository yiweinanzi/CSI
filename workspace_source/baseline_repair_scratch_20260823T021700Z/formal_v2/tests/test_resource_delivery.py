from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from formal_v2.fetch_waibu_resources import fetch_missing_resources
from formal_v2.formal_cli import main as formal_cli_main
from formal_v2.formal_resources import (
    stage_redistributable_resources,
    validate_resource_registry,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "formal_v2" / "configs" / "waibu_resources_v1.json"


class ResourceDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        self.content: dict[str, bytes] = {}
        for row in self.registry["resources"]:
            content = f"test-only-resource:{row['file']}\n".encode("utf-8")
            self.content[row["file"]] = content
            row["sha256"] = hashlib.sha256(content).hexdigest()
        self.registry_path = self.root / "registry.json"
        self.registry_path.write_text(
            json.dumps(self.registry, sort_keys=True),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _materialize(self, target: Path, *, include_restricted: bool) -> None:
        target.mkdir()
        for row in self.registry["resources"]:
            if row["redistribution_allowed"] or include_restricted:
                (target / row["file"]).write_bytes(self.content[row["file"]])

    def test_delivery_mode_allows_only_restricted_files_to_be_missing(self):
        source = self.root / "source"
        self._materialize(source, include_restricted=False)
        rows = validate_resource_registry(self.registry, source, mode="delivery")
        self.assertEqual(
            {
                row["file"]
                for row in rows
                if row["status"] == "LOCAL_FETCH_REQUIRED"
            },
            {
                row["file"]
                for row in self.registry["resources"]
                if not row["redistribution_allowed"]
            },
        )
        with self.assertRaisesRegex(ValueError, "formal resources are missing"):
            validate_resource_registry(self.registry, source, mode="formal")

        allowed = next(
            row for row in self.registry["resources"] if row["redistribution_allowed"]
        )
        (source / allowed["file"]).unlink()
        with self.assertRaisesRegex(ValueError, "delivery resources are missing"):
            validate_resource_registry(self.registry, source, mode="delivery")

    def test_delivery_staging_never_copies_restricted_resources(self):
        source = self.root / "source"
        destination = self.root / "delivery"
        self._materialize(source, include_restricted=True)
        rows = stage_redistributable_resources(
            self.registry_path,
            source,
            destination,
        )
        self.assertEqual(
            {path.name for path in destination.iterdir()},
            {
                row["file"]
                for row in self.registry["resources"]
                if row["redistribution_allowed"]
            },
        )
        self.assertTrue(
            all(
                row["status"] == "LOCAL_FETCH_REQUIRED"
                for row in rows
                if not row["redistribution_allowed"]
            )
        )

    def test_fetcher_downloads_missing_files_and_authenticates_before_rename(self):
        source = self.root / "source"
        self._materialize(source, include_restricted=False)
        requested: list[str] = []

        def open_registered(request, *, timeout):
            self.assertEqual(timeout, 17.0)
            requested.append(request.full_url)
            row = next(
                row
                for row in self.registry["resources"]
                if row["source_url"] == request.full_url
            )
            return io.BytesIO(self.content[row["file"]])

        with patch(
            "formal_v2.fetch_waibu_resources.urlrequest.urlopen",
            side_effect=open_registered,
        ):
            result = fetch_missing_resources(
                self.registry_path,
                source,
                timeout_seconds=17.0,
            )
        restricted = {
            row["file"]
            for row in self.registry["resources"]
            if not row["redistribution_allowed"]
        }
        self.assertEqual(set(result["fetched"]), restricted)
        self.assertEqual(
            set(requested),
            {
                row["source_url"]
                for row in self.registry["resources"]
                if not row["redistribution_allowed"]
            },
        )
        self.assertTrue(
            all(
                row["status"] == "PASS"
                for row in validate_resource_registry(
                    self.registry,
                    source,
                    mode="formal",
                )
            )
        )

    def test_fetcher_refuses_to_overwrite_existing_mismatched_bytes(self):
        source = self.root / "source"
        source.mkdir()
        first = self.registry["resources"][0]
        target = source / first["file"]
        target.write_bytes(b"do-not-overwrite")
        with patch(
            "formal_v2.fetch_waibu_resources.urlrequest.urlopen"
        ) as urlopen:
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                fetch_missing_resources(self.registry_path, source)
        urlopen.assert_not_called()
        self.assertEqual(target.read_bytes(), b"do-not-overwrite")

    def test_formal_resource_cli_fails_until_all_local_inputs_exist(self):
        source = self.root / "source"
        self._materialize(source, include_restricted=False)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            missing_exit = formal_cli_main(
                [
                    "verify-waibu-resources",
                    "--registry",
                    str(self.registry_path),
                    "--waibu-root",
                    str(source),
                    "--output",
                    str(self.root / "missing-run"),
                ]
            )
        self.assertEqual(missing_exit, 2)
        self.assertFalse((self.root / "missing-run" / "waibu_resources").exists())
        self.assertFalse((self.root / "missing-run" / "waibu_resources" / "gate.json").exists())

        for row in self.registry["resources"]:
            path = source / row["file"]
            if not path.exists():
                path.write_bytes(self.content[row["file"]])
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            complete_exit = formal_cli_main(
                [
                    "verify-waibu-resources",
                    "--registry",
                    str(self.registry_path),
                    "--waibu-root",
                    str(source),
                    "--output",
                    str(self.root / "complete-run"),
                ]
            )
        self.assertEqual(complete_exit, 0)


if __name__ == "__main__":
    unittest.main()
