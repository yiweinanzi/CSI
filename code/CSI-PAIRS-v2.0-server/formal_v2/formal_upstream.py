from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path




_RUNNING_SOURCE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AuthenticatedUpstream:
    run_root: Path
    qualification_root: Path
    factorial_root: Path
    migration: None = None

    @property
    def migrated(self) -> bool:
        return False

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
        return None

    @property
    def factorial_evidence(self) -> dict[str, object] | None:
        return None

    def qualification_path(self, relative: str) -> Path:
        return _bound_stage_path(self.qualification_root, relative)

    def factorial_path(self, relative: str) -> Path:
        return _bound_stage_path(self.factorial_root, relative)


def resolve_authenticated_upstream(config, dataset, output_root) -> AuthenticatedUpstream:
    """Resolve the current run; archived migrations remain read-only artifacts."""
    root = _regular_directory(output_root, "experiment output root")
    if (root / "migration").exists():
        raise ValueError("Archived migrated runs are read-only; use paper_core with explicit checkpoints and a fresh output")
    return AuthenticatedUpstream(root, root / "qualification", root / "factorial")


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
