from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from formal_v2.anonymous_release import (
    project_anonymity_tokens,
    scan_tree,
    scan_zip,
)


class AnonymousReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tokens = {"private-owner", "private-owner/project", "author@example.org"}
        self.project_shas = {"a" * 40, "b" * 40}

    def tearDown(self):
        self.temporary.cleanup()

    def test_tree_scan_checks_names_binary_metadata_and_project_shas(self):
        safe = self.root / "safe"
        safe.mkdir()
        (safe / "code.py").write_text("print('anonymous')\n", encoding="utf-8")
        self.assertEqual(
            scan_tree(safe, tokens=self.tokens, project_shas=self.project_shas),
            [],
        )

        (safe / "figure.png").write_bytes(b"header\x00author@example.org\x00")
        (safe / ("a" * 40 + ".txt")).write_text("clean", encoding="utf-8")
        (safe / "short-sha.txt").write_text("revision " + "b" * 7, encoding="utf-8")
        violations = scan_tree(
            safe,
            tokens=self.tokens,
            project_shas=self.project_shas,
        )
        self.assertTrue(any("author@example.org" in value for value in violations))
        self.assertTrue(any("project Git commit" in value for value in violations))
        self.assertTrue(any("b" * 7 in value for value in violations))

    def test_zip_scan_checks_entry_names_comments_payloads_and_symlinks(self):
        archive = self.root / "anonymous.zip"
        with zipfile.ZipFile(archive, "w") as package:
            package.comment = b"private-owner/project"
            info = zipfile.ZipInfo("paper/source.tex")
            info.comment = b"author@example.org"
            package.writestr(info, "project head " + "b" * 40)
            symlink = zipfile.ZipInfo("paper/link")
            symlink.external_attr = (0o120777 << 16) | 0xA0000000
            package.writestr(symlink, "source.tex")
            package.writestr("../identity.txt", "payload")
        violations = scan_zip(
            archive,
            tokens=self.tokens,
            project_shas=self.project_shas,
        )
        self.assertTrue(any("zip-comment" in value for value in violations))
        self.assertTrue(any("zip-entry-comment" in value for value in violations))
        self.assertTrue(any("project Git commit" in value for value in violations))
        self.assertTrue(any("symlink is forbidden" in value for value in violations))
        self.assertTrue(any("unsafe archive path" in value for value in violations))

    def test_absolute_home_paths_and_generated_files_are_rejected(self):
        release = self.root / "release"
        release.mkdir()
        personal_path = "/" + "Users" + "/someone/work/project"
        (release / "trace.txt").write_text(
            "built at " + personal_path,
            encoding="utf-8",
        )
        cache = release / "__pycache__"
        cache.mkdir()
        (cache / "module.pyc").write_bytes(b"bytecode")
        violations = scan_tree(
            release,
            tokens=self.tokens,
            project_shas=self.project_shas,
        )
        self.assertTrue(any("personal absolute path" in value for value in violations))
        self.assertTrue(any("forbidden release path" in value for value in violations))
        self.assertTrue(any("bytecode is forbidden" in value for value in violations))

    def test_project_tokens_include_author_and_committer_names_and_emails(self):
        repository = self.root / "repository"
        repository.mkdir()
        subprocess.run(
            ["git", "init", "--quiet"],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        (repository / "tracked.txt").write_text("content\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "tracked.txt"],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        environment = {
            **os.environ,
            "GIT_AUTHOR_NAME": "futaoran",
            "GIT_AUTHOR_EMAIL": "private-author@example.org",
            "GIT_COMMITTER_NAME": "Private Committer",
            "GIT_COMMITTER_EMAIL": "private-committer@example.org",
        }
        subprocess.run(
            ["git", "commit", "--quiet", "-m", "test identity"],
            cwd=repository,
            env=environment,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        tokens, shas = project_anonymity_tokens(repository)

        self.assertTrue(
            {
                "futaoran",
                "private-author@example.org",
                "private committer",
                "private-committer@example.org",
            }.issubset(tokens)
        )
        self.assertEqual(len(shas), 1)

    def test_project_tokens_ignore_generic_github_merge_committer(self):
        repository = self.root / "repository"
        repository.mkdir()
        subprocess.run(
            ["git", "init", "--quiet"],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        (repository / "tracked.txt").write_text("content\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "tracked.txt"],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        environment = {
            **os.environ,
            "GIT_AUTHOR_NAME": "Project Author",
            "GIT_AUTHOR_EMAIL": "project-author@example.org",
            "GIT_COMMITTER_NAME": "GitHub",
            "GIT_COMMITTER_EMAIL": "noreply@github.com",
        }
        subprocess.run(
            ["git", "commit", "--quiet", "-m", "merge-ref identity"],
            cwd=repository,
            env=environment,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        tokens, _ = project_anonymity_tokens(repository)

        self.assertIn("project author", tokens)
        self.assertIn("project-author@example.org", tokens)
        self.assertNotIn("github", tokens)
        self.assertNotIn("noreply@github.com", tokens)

    def test_built_supplement_contains_a_runnable_public_test_suite(self):
        project_root = Path(__file__).resolve().parents[2]
        git_probe = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if git_probe.returncode != 0 or git_probe.stdout.strip() != "true":
            self.skipTest("anonymous source export requires the repository Git history")

        archive = self.root / "anonymous.zip"
        builder = project_root / "formal_v2/scripts/build_anonymous_supplement.sh"
        subprocess.run(
            ["bash", str(builder), str(archive)],
            cwd=project_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        extracted = self.root / "extracted"
        with zipfile.ZipFile(archive) as package:
            package.extractall(extracted)
        release_root = extracted / "CSI-PAIRS-anonymous-supplement"

        self.assertFalse(
            (release_root / "formal_v2/scripts/build_v6_requirement_matrix.py").exists()
        )
        self.assertFalse(
            (release_root / "formal_v2/scripts/v6_trace_registry.py").exists()
        )
        for relative in (
            "formal_evaluation_subset_compare.py",
            "formal_migration.py",
            "formal_migration_evidence.py",
        ):
            with self.subTest(relative=relative):
                self.assertFalse((release_root / "formal_v2" / relative).exists())
        self.assertFalse(
            (release_root / "formal_v2/tests/test_audit_artifacts.py").exists()
        )
        for relative in (
            "test_adapter_root_separation.py",
            "test_cli_migration_control.py",
            "test_evaluation_subset_compare.py",
            "test_formal_migration.py",
            "test_formal_upstream.py",
            "test_migration_cli_acceptance.py",
            "test_migration_evidence_tools.py",
        ):
            with self.subTest(relative=relative):
                self.assertFalse(
                    (release_root / "formal_v2/tests" / relative).exists()
                )
        self.assertFalse(
            (
                release_root
                / "formal_v2/tests/test_formal_data_pilot_projection.py"
            ).exists()
        )
        self.assertFalse(
            (release_root / "formal_v2/tests/test_v5_pilot_tools.py").exists()
        )
        self.assertFalse(
            (
                release_root
                / "formal_v2/tests/test_v5_action_inverse_response_probe.py"
            ).exists()
        )
        self.assertFalse(
            (release_root / "formal_v2/tests/test_v5_protocol_freeze.py").exists()
        )
        self.assertFalse(
            (
                release_root
                / "formal_v2/tests/test_v5_response_pilot_aggregate.py"
            ).exists()
        )
        self.assertFalse(
            (release_root / "formal_v2/external_adapters/.runtime-differt").exists()
        )
        strict_candidates = []
        configured = os.environ.get("CSI_PAIRS_ANONYMOUS_TEST_PYTHON")
        if configured:
            strict_candidates.append(Path(configured).absolute())
        strict_candidates.append(Path(sys.executable).resolve())
        strict_candidates.extend(
            path.absolute()
            for path in reversed(
                sorted(project_root.glob(".venv-core-formal-*/bin/python"))
            )
        )
        strict_python = next(
            (
                path
                for path in strict_candidates
                if path.is_file()
                and os.access(path, os.X_OK)
                and (path.parents[1] / "csi-pairs-install-report.json").is_file()
                and (
                    path.parents[1] / "csi-pairs-reviewed-wheel-manifest.json"
                ).is_file()
            ),
            None,
        )
        if strict_python is None:
            self.skipTest(
                "recursive public-suite validation requires the README's "
                "hash-locked environment"
            )
        runtime_root = strict_python.parents[1]
        bytecode_before = {
            path.relative_to(runtime_root)
            for path in runtime_root.rglob("*")
            if path.is_file() and path.suffix in {".pyc", ".pyo"}
        }
        recursive_cache = self.root / "recursive-pycache"
        completed = subprocess.run(
            [
                str(strict_python),
                "-B",
                "-m",
                "unittest",
                "discover",
                "-s",
                "formal_v2/tests",
                "-v",
            ],
            cwd=release_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": str(recursive_cache.resolve()),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            # This recursively runs the public suite on a hosted CPU runner.
            timeout=900,
        )
        bytecode_after = {
            path.relative_to(runtime_root)
            for path in runtime_root.rglob("*")
            if path.is_file() and path.suffix in {".pyc", ".pyo"}
        }
        self.assertEqual(bytecode_after, bytecode_before)
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
