from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .formal_evidence import config_sha256
from .formal_io import read_strict_json

if TYPE_CHECKING:
    from .formal_migration import AuthenticatedLegacyUpstream


_RUNNING_SOURCE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AuthenticatedUpstream:
    run_root: Path
    qualification_root: Path
    factorial_root: Path
    migration: AuthenticatedLegacyUpstream | None

    @property
    def migrated(self) -> bool:
        return self.migration is not None

    @property
    def qualification_gate(self) -> Path:
        return self.qualification_root / "gate.json"

    @property
    def factorial_gate(self) -> Path:
        return self.factorial_root / "gate.json"

    @property
    def checkpoint_index(self) -> Path:
        return self.factorial_root / "checkpoint_index.json"

    @property
    def qualification_evidence(self) -> dict[str, object] | None:
        if self.migration is None:
            return None
        return dict(self.migration.qualification_evidence)

    @property
    def factorial_evidence(self) -> dict[str, object] | None:
        if self.migration is None:
            return None
        return dict(self.migration.factorial_evidence)

    def qualification_path(self, relative: str) -> Path:
        return _bound_stage_path(self.qualification_root, relative)

    def factorial_path(self, relative: str) -> Path:
        return _bound_stage_path(self.factorial_root, relative)


def resolve_authenticated_upstream(config, dataset, output_root) -> AuthenticatedUpstream:
    """Resolve local upstreams or authenticate the canonical migration receipt."""
    root = _regular_directory(output_root, "formal output root")
    migration_root = root / "migration"
    request_path = migration_root / "request.json"
    accepted_path = migration_root / "accepted.json"
    migration_present = (
        migration_root.exists()
        or migration_root.is_symlink()
        or request_path.exists()
        or request_path.is_symlink()
        or accepted_path.exists()
        or accepted_path.is_symlink()
    )
    if not migration_present:
        return AuthenticatedUpstream(
            run_root=root,
            qualification_root=root / "qualification",
            factorial_root=root / "factorial",
            migration=None,
        )
    if migration_root.is_symlink() or not migration_root.is_dir():
        raise RuntimeError("migration directory is missing, invalid, or a symlink")
    if not request_path.is_file() or request_path.is_symlink():
        raise RuntimeError("migration request is missing or invalid; local fallback is forbidden")
    if not accepted_path.is_file() or accepted_path.is_symlink():
        raise RuntimeError("migration receipt is not accepted; local fallback is forbidden")
    for name in ("qualification", "factorial"):
        shadow = root / name
        if shadow.exists() or shadow.is_symlink():
            raise RuntimeError(f"migrated run root contains forbidden local {name} shadow")

    request = read_strict_json(request_path)
    if not isinstance(request, dict):
        raise RuntimeError("migration request is malformed")
    protocol_path = request.get("protocol_path")
    if not isinstance(protocol_path, str) or not protocol_path:
        raise RuntimeError("migration request has no frozen protocol binding")
    from .formal_migration import authenticate_migration_receipt

    seeds = config.get("seeds") if isinstance(config, dict) else None
    authenticated = authenticate_migration_receipt(
        accepted_path,
        new_run_root=root,
        new_source_root=_RUNNING_SOURCE_ROOT,
        protocol_path=protocol_path,
        dataset_path=dataset.source_path,
        config_sha256=config_sha256(config),
        expected_seeds=seeds,
    )
    if authenticated.new_run_root != root:
        raise RuntimeError("migration receipt resolved a different destination root")
    qualification_root = authenticated.qualification_gate.parent
    if authenticated.factorial_root != authenticated.factorial_gate.parent:
        raise RuntimeError("migration factorial paths disagree")
    return AuthenticatedUpstream(
        run_root=root,
        qualification_root=qualification_root,
        factorial_root=authenticated.factorial_root,
        migration=authenticated,
    )


def _bound_stage_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("upstream stage path must be nonempty and relative")
    if ".." in Path(relative).parts:
        raise ValueError("upstream stage path cannot escape its stage root")
    candidate = root / relative
    resolved = candidate.resolve()
    if root.resolve() not in resolved.parents:
        raise RuntimeError("upstream stage path escapes its authenticated root")
    return resolved


def _regular_directory(path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise RuntimeError(f"{label} cannot be a symlink")
    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"{label} is missing or not a directory")
    return resolved
