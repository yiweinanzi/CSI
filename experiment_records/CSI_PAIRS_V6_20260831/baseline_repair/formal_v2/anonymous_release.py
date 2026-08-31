from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile


STATIC_FORBIDDEN_TOKENS = {
    "yiweinanzi",
    "taron0323",
    "immune-skillnet",
    "researcher@immune-skillnet.ai",
    "/users/futaoran",
    "origin/main",
    "base_sha",
    "audited_code_sha",
    "delivery_head_sha",
}
GENERIC_AUTOMATION_IDENTITY_TOKENS = {
    "github",
    "noreply@github.com",
}
FORBIDDEN_PATH_PARTS = {
    ".git",
    ".github",
    ".ds_store",
    "__pycache__",
}
ABSOLUTE_HOME_PATTERNS = (
    re.compile(rb"/Users/[A-Za-z0-9._-]+/"),
    re.compile(rb"/home/[A-Za-z0-9._-]+/"),
    re.compile(rb"[A-Za-z]:\\Users\\[A-Za-z0-9._-]+\\"),
)
GIT_ABBREVIATION_PATTERN = re.compile(
    rb"(?<![0-9a-f])[0-9a-f]{7,40}(?![0-9a-f])",
    flags=re.IGNORECASE,
)


def _git_lines(repository: Path, *arguments: str) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def project_anonymity_tokens(repository: str | Path) -> tuple[set[str], set[str]]:
    root = Path(repository).resolve()
    shas = {
        value.lower()
        for value in _git_lines(root, "rev-list", "--all")
        if re.fullmatch(r"[0-9a-fA-F]{40}", value)
    }
    head = _git_lines(root, "rev-parse", "HEAD")
    shas.update(value.lower() for value in head)
    if not shas:
        raise ValueError("project Git history produced no commit identifiers")

    tokens = set(STATIC_FORBIDDEN_TOKENS)
    git_identity_tokens = {
        value.lower()
        for value in _git_lines(
            root,
            "log",
            "--all",
            "--format=%an%n%ae%n%cn%n%ce",
        )
    }
    tokens.update(git_identity_tokens - GENERIC_AUTOMATION_IDENTITY_TOKENS)
    remotes = _git_lines(root, "remote", "-v")
    for line in remotes:
        parts = line.split()
        if len(parts) < 2:
            continue
        remote = parts[1].removesuffix(".git")
        tokens.add(remote.lower())
        match = re.search(r"(?:github\.com[/:])([^/\s]+/[^/\s]+)$", remote, flags=re.I)
        if match is not None:
            tokens.add(match.group(1).lower())
            tokens.add(match.group(1).split("/", 1)[0].lower())
    return tokens, shas


def _scan_payload(
    label: str,
    payload: bytes,
    *,
    tokens: set[str],
    project_shas: set[str],
) -> list[str]:
    lowered = payload.lower()
    violations = [
        f"{label}: identity/provenance token {token!r}"
        for token in sorted(tokens)
        if token and token.encode("utf-8") in lowered
    ]
    project_shas = {sha.lower() for sha in project_shas}
    commit_tokens = {
        match.group(0).decode("ascii").lower()
        for match in GIT_ABBREVIATION_PATTERN.finditer(lowered)
    }
    violations.extend(
        f"{label}: project Git commit or abbreviation {token}"
        for token in sorted(commit_tokens)
        if any(sha.startswith(token) for sha in project_shas)
    )
    for pattern in ABSOLUTE_HOME_PATTERNS:
        match = pattern.search(payload)
        if match is not None:
            violations.append(
                f"{label}: personal absolute path {match.group(0).decode('utf-8', errors='replace')!r}"
            )
    return violations


def _scan_name(name: str, *, tokens: set[str], project_shas: set[str]) -> list[str]:
    violations = _scan_payload(
        f"path:{name}",
        name.encode("utf-8", errors="surrogateescape"),
        tokens=tokens,
        project_shas=project_shas,
    )
    parts = {part.lower() for part in Path(name).parts}
    posix = PurePosixPath(name)
    if posix.is_absolute() or ".." in posix.parts or "\\" in name:
        violations.append(f"path:{name}: unsafe archive path")
    blocked = sorted(parts & FORBIDDEN_PATH_PARTS)
    if blocked:
        violations.append(f"path:{name}: forbidden release path component {blocked}")
    if any(part.endswith(".egg-info") for part in parts):
        violations.append(f"path:{name}: generated egg-info is forbidden")
    if name.lower().endswith((".pyc", ".pyo")):
        violations.append(f"path:{name}: bytecode is forbidden")
    return violations


def scan_tree(
    root: str | Path,
    *,
    tokens: set[str],
    project_shas: set[str],
) -> list[str]:
    release_root = Path(root)
    if not release_root.is_dir() or release_root.is_symlink():
        raise ValueError("anonymous release root must be a regular directory")
    violations: list[str] = []
    for path in sorted(release_root.rglob("*")):
        relative = path.relative_to(release_root).as_posix()
        violations.extend(
            _scan_name(relative, tokens=tokens, project_shas=project_shas)
        )
        if path.is_symlink():
            violations.append(f"path:{relative}: symlink is forbidden")
        elif path.is_file():
            violations.extend(
                _scan_payload(
                    f"file:{relative}",
                    path.read_bytes(),
                    tokens=tokens,
                    project_shas=project_shas,
                )
            )
    return violations


def scan_zip(
    archive_path: str | Path,
    *,
    tokens: set[str],
    project_shas: set[str],
) -> list[str]:
    archive = Path(archive_path)
    if not archive.is_file() or archive.is_symlink():
        raise ValueError("anonymous release ZIP must be a regular file")
    violations: list[str] = []
    with zipfile.ZipFile(archive) as package:
        violations.extend(
            _scan_payload(
                "zip-comment",
                package.comment,
                tokens=tokens,
                project_shas=project_shas,
            )
        )
        names: set[str] = set()
        for info in package.infolist():
            if info.filename in names:
                violations.append(f"zip-entry:{info.filename}: duplicate entry")
            names.add(info.filename)
            violations.extend(
                _scan_name(
                    info.filename,
                    tokens=tokens,
                    project_shas=project_shas,
                )
            )
            violations.extend(
                _scan_payload(
                    f"zip-entry-comment:{info.filename}",
                    info.comment,
                    tokens=tokens,
                    project_shas=project_shas,
                )
            )
            unix_mode = (info.external_attr >> 16) & 0o170000
            if unix_mode == 0o120000:
                violations.append(f"zip-entry:{info.filename}: symlink is forbidden")
            if not info.is_dir():
                violations.extend(
                    _scan_payload(
                        f"zip-entry:{info.filename}",
                        package.read(info),
                        tokens=tokens,
                        project_shas=project_shas,
                    )
                )
    return violations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed CSI-PAIRS anonymous release scan")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--tree")
    target.add_argument("--zip")
    parser.add_argument("--project-repository", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tokens, shas = project_anonymity_tokens(args.project_repository)
        violations = (
            scan_tree(args.tree, tokens=tokens, project_shas=shas)
            if args.tree
            else scan_zip(args.zip, tokens=tokens, project_shas=shas)
        )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if violations:
        print(
            json.dumps(
                {"status": "FAIL", "violations": violations[:50]},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps({"status": "PASS", "violations": []}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
