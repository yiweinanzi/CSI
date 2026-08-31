from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from formal_v2 import formal_migration as migration
from formal_v2 import formal_upstream as upstream
from formal_v2.tests.test_formal_migration import (
    ACCEPTED_UTC,
    CONFIG_SHA256,
    NOW,
    FormalMigrationFixture,
)


class FormalUpstreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = FormalMigrationFixture(Path(self.temporary.name))
        self.dataset = SimpleNamespace(source_path=self.fixture.dataset)
        self.config = {"seeds": [20270001, 20270002, 20270003]}
        self.patches = (
            mock.patch.object(
                migration, "_RUNNING_SOURCE_ROOT", self.fixture.new_source
            ),
            mock.patch.object(
                upstream, "_RUNNING_SOURCE_ROOT", self.fixture.new_source
            ),
            mock.patch.object(upstream, "config_sha256", return_value=CONFIG_SHA256),
        )
        for patcher in self.patches:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self.patches):
            patcher.stop()
        self.temporary.cleanup()

    def _accept(self) -> dict[str, object]:
        request = migration.write_migration_request(
            **self.fixture.request_arguments()
        )
        approval = self.fixture.write_migration_approval(request)
        return migration.accept_migration_request(
            request["request_path"],
            approval,
            **self.fixture.acceptance_arguments(),
            accepted_utc=ACCEPTED_UTC,
            now=NOW,
        )

    def test_default_without_migration_keeps_local_paths(self) -> None:
        resolved = upstream.resolve_authenticated_upstream(
            self.config, self.dataset, self.fixture.new_run
        )
        self.assertFalse(resolved.migrated)
        self.assertEqual(
            resolved.qualification_root, self.fixture.new_run / "qualification"
        )
        self.assertEqual(resolved.factorial_root, self.fixture.new_run / "factorial")

    def test_request_without_accepted_receipt_forbids_local_fallback(self) -> None:
        migration.write_migration_request(**self.fixture.request_arguments())
        with self.assertRaises(RuntimeError):
            upstream.resolve_authenticated_upstream(
                self.config, self.dataset, self.fixture.new_run
            )

    def test_accepted_receipt_resolves_only_authenticated_legacy_upstream(self) -> None:
        self._accept()
        resolved = upstream.resolve_authenticated_upstream(
            self.config, self.dataset, self.fixture.new_run
        )
        self.assertTrue(resolved.migrated)
        self.assertEqual(resolved.run_root, self.fixture.new_run)
        self.assertEqual(
            resolved.qualification_root, self.fixture.legacy_run / "qualification"
        )
        self.assertEqual(resolved.factorial_root, self.fixture.factorial_root)
        self.assertEqual(resolved.checkpoint_index, self.fixture.checkpoint_index)
        self.assertEqual(
            resolved.qualification_evidence["source_tree_sha256"],
            self.fixture.legacy_source_identity["source_tree_sha256"],
        )
        self.assertEqual(
            resolved.factorial_evidence["source_tree_sha256"],
            self.fixture.legacy_source_identity["source_tree_sha256"],
        )

    def test_migrated_run_rejects_local_upstream_shadow(self) -> None:
        self._accept()
        (self.fixture.new_run / "factorial").mkdir()
        with self.assertRaises(RuntimeError):
            upstream.resolve_authenticated_upstream(
                self.config, self.dataset, self.fixture.new_run
            )


if __name__ == "__main__":
    unittest.main()
