from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import socket
import stat
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Iterator

from .formal_io import StrictJsonError, parse_strict_json, read_strict_json, sha256_file
from .formal_locks import LOCK_EX, LOCK_NB, LOCK_UN, flock


RUN_IDENTITY_SCHEMA = "csi-pairs-v6-evaluation-run-identity-v3"
EXECUTION_PROFILE_SCHEMA = "csi-pairs-v6-evaluation-execution-profile-v1"
SHARD_IDENTITY_SCHEMA = "csi-pairs-v6-evaluation-shard-identity-v2"
SHARD_MANIFEST_SCHEMA = "csi-pairs-v6-evaluation-shard-manifest-v2"
PROGRESS_SCHEMA = "csi-pairs-v6-evaluation-progress-v2"
WRITER_LOCK_SCHEMA = "csi-pairs-v6-evaluation-writer-lock-v2"
QUARANTINE_RECEIPT_SCHEMA = "csi-pairs-v6-evaluation-quarantine-receipt-v1"
FINAL_OUTPUT_QUARANTINE_SCHEMA = (
    "csi-pairs-v6-evaluation-final-output-quarantine-v1"
)
FINAL_OUTPUT_ARTIFACT_QUARANTINE_SCHEMA = (
    "csi-pairs-v6-evaluation-final-artifact-quarantine-v1"
)
PROGRESS_RECONCILIATION_SCHEMA = (
    "csi-pairs-v6-evaluation-progress-reconciliation-v1"
)
NO_MIGRATION_SHA256 = "0" * 64

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_REVISION_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_PATH_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SUFFIX_RE = re.compile(r"\.[A-Za-z0-9][A-Za-z0-9._-]{0,31}\Z")
_CUDA_DEVICE_RE = re.compile(r"cuda:(?:0|[1-9][0-9]*)\Z")


class EvaluationResumeError(RuntimeError):
    """Base class for evaluation checkpoint integrity failures."""


class WriterLockError(EvaluationResumeError):
    """Raised when another writer owns the evaluation resume root."""


class IncompleteShardError(EvaluationResumeError):
    """Raised when only part of an atomic shard commit is present."""


class CorruptShardError(EvaluationResumeError):
    """Raised when a committed shard or manifest fails validation."""


class StaleResumeError(EvaluationResumeError):
    """Raised when persisted state belongs to a different run identity."""


class ProgressValidationError(EvaluationResumeError):
    """Raised when the progress receipt is missing required invariants."""


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_text(value: object, label: str, *, maximum: int = 256) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 0x20 or ord(character) > 0x7E for character in value)
    ):
        raise ValueError(f"{label} must be nonempty printable ASCII")
    return value


def _require_code_revision(value: object) -> str:
    if type(value) is not str or _GIT_REVISION_RE.fullmatch(value) is None:
        raise ValueError("code_revision must be a full lowercase Git object ID")
    return value


def _require_path_component(value: object, label: str) -> str:
    if type(value) is not str or _PATH_COMPONENT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} is not a safe path component")
    return value


def _require_suffix(value: object) -> str:
    if type(value) is not str or _SUFFIX_RE.fullmatch(value) is None:
        raise ValueError("payload suffix must be a safe extension beginning with '.'")
    return value


def _utc_datetime(value: datetime | None = None) -> datetime:
    current = datetime.now(timezone.utc) if value is None else value
    if not isinstance(current, datetime) or current.tzinfo is None:
        raise ValueError("evaluation timestamps must be timezone-aware datetimes")
    return current.astimezone(timezone.utc)


def _timestamp(value: datetime | None = None) -> str:
    return _utc_datetime(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: object, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class EvaluationExecutionProfile:
    """Operational identity for resume isolation, not a metric definition."""

    execution_devices: tuple[str, ...]
    batch_size: int

    def __post_init__(self) -> None:
        if type(self.execution_devices) is not tuple or not self.execution_devices:
            raise ValueError("execution_devices must be a nonempty tuple")
        if any(
            type(device) is not str or _CUDA_DEVICE_RE.fullmatch(device) is None
            for device in self.execution_devices
        ):
            raise ValueError("execution_devices must contain canonical CUDA devices")
        if len(self.execution_devices) != len(set(self.execution_devices)):
            raise ValueError("execution_devices must be unique")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("execution batch_size must be a positive integer")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": EXECUTION_PROFILE_SCHEMA,
            "execution_devices": list(self.execution_devices),
            "batch_size": self.batch_size,
        }

    @property
    def sha256(self) -> str:
        return _json_sha256(self.as_dict())

    @classmethod
    def from_dict(cls, payload: object) -> "EvaluationExecutionProfile":
        required = {"schema_version", "execution_devices", "batch_size"}
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("evaluation execution profile fields are invalid")
        if payload["schema_version"] != EXECUTION_PROFILE_SCHEMA:
            raise ValueError("evaluation execution profile schema is invalid")
        devices = payload["execution_devices"]
        if not isinstance(devices, list):
            raise ValueError("evaluation execution profile devices are invalid")
        return cls(
            execution_devices=tuple(devices),
            batch_size=payload["batch_size"],
        )


@dataclass(frozen=True)
class EvaluationRunIdentity:
    """Immutable scientific and software identity shared by all shards."""

    code_revision: str
    source_tree_sha256: str
    config_sha256: str
    dataset_sha256: str
    migration_accepted_sha256: str
    legacy_checkpoint_inventory_sha256: str
    qualification_gate_sha256: str
    factorial_gate_sha256: str
    runtime_provenance_sha256: str
    run_nonce: str
    compute_plan_sha256: str
    execution_profile: EvaluationExecutionProfile
    output_schema_id: str
    output_schema_sha256: str

    def __post_init__(self) -> None:
        _require_code_revision(self.code_revision)
        _require_sha256(self.source_tree_sha256, "source_tree_sha256")
        _require_sha256(self.config_sha256, "config_sha256")
        _require_sha256(self.dataset_sha256, "dataset_sha256")
        _require_sha256(
            self.migration_accepted_sha256, "migration_accepted_sha256"
        )
        _require_sha256(
            self.legacy_checkpoint_inventory_sha256,
            "legacy_checkpoint_inventory_sha256",
        )
        _require_sha256(self.qualification_gate_sha256, "qualification_gate_sha256")
        _require_sha256(self.factorial_gate_sha256, "factorial_gate_sha256")
        _require_sha256(self.runtime_provenance_sha256, "runtime_provenance_sha256")
        _require_sha256(self.run_nonce, "run_nonce")
        _require_sha256(self.compute_plan_sha256, "compute_plan_sha256")
        if not isinstance(self.execution_profile, EvaluationExecutionProfile):
            raise TypeError(
                "execution_profile must be an EvaluationExecutionProfile"
            )
        no_migration = self.migration_accepted_sha256 == NO_MIGRATION_SHA256
        no_legacy_inventory = (
            self.legacy_checkpoint_inventory_sha256 == NO_MIGRATION_SHA256
        )
        if no_migration != no_legacy_inventory:
            raise ValueError(
                "migration acceptance and legacy checkpoint inventory must both be "
                "bound or both use NO_MIGRATION_SHA256"
            )
        for value, label in (
            (self.qualification_gate_sha256, "qualification_gate_sha256"),
            (self.factorial_gate_sha256, "factorial_gate_sha256"),
            (self.runtime_provenance_sha256, "runtime_provenance_sha256"),
            (self.run_nonce, "run_nonce"),
            (self.compute_plan_sha256, "compute_plan_sha256"),
        ):
            if value == NO_MIGRATION_SHA256:
                raise ValueError(f"{label} cannot use the no-migration sentinel")
        _require_text(self.output_schema_id, "output_schema_id")
        _require_sha256(self.output_schema_sha256, "output_schema_sha256")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": RUN_IDENTITY_SCHEMA,
            "code_revision": self.code_revision,
            "source_tree_sha256": self.source_tree_sha256,
            "config_sha256": self.config_sha256,
            "dataset_sha256": self.dataset_sha256,
            "migration_accepted_sha256": self.migration_accepted_sha256,
            "legacy_checkpoint_inventory_sha256": self.legacy_checkpoint_inventory_sha256,
            "qualification_gate_sha256": self.qualification_gate_sha256,
            "factorial_gate_sha256": self.factorial_gate_sha256,
            "runtime_provenance_sha256": self.runtime_provenance_sha256,
            "run_nonce": self.run_nonce,
            "compute_plan_sha256": self.compute_plan_sha256,
            "execution_profile": self.execution_profile.as_dict(),
            "execution_profile_sha256": self.execution_profile.sha256,
            "output_schema_id": self.output_schema_id,
            "output_schema_sha256": self.output_schema_sha256,
        }

    @property
    def sha256(self) -> str:
        return _json_sha256(self.as_dict())

    @classmethod
    def from_dict(cls, payload: object) -> "EvaluationRunIdentity":
        required = {
            "schema_version",
            "code_revision",
            "source_tree_sha256",
            "config_sha256",
            "dataset_sha256",
            "migration_accepted_sha256",
            "legacy_checkpoint_inventory_sha256",
            "qualification_gate_sha256",
            "factorial_gate_sha256",
            "runtime_provenance_sha256",
            "run_nonce",
            "compute_plan_sha256",
            "execution_profile",
            "execution_profile_sha256",
            "output_schema_id",
            "output_schema_sha256",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("evaluation run identity fields are invalid")
        if payload["schema_version"] != RUN_IDENTITY_SCHEMA:
            raise ValueError("evaluation run identity schema is invalid")
        execution_profile = EvaluationExecutionProfile.from_dict(
            payload["execution_profile"]
        )
        if payload["execution_profile_sha256"] != execution_profile.sha256:
            raise ValueError("evaluation execution profile digest is invalid")
        return cls(
            code_revision=payload["code_revision"],
            source_tree_sha256=payload["source_tree_sha256"],
            config_sha256=payload["config_sha256"],
            dataset_sha256=payload["dataset_sha256"],
            migration_accepted_sha256=payload["migration_accepted_sha256"],
            legacy_checkpoint_inventory_sha256=payload[
                "legacy_checkpoint_inventory_sha256"
            ],
            qualification_gate_sha256=payload["qualification_gate_sha256"],
            factorial_gate_sha256=payload["factorial_gate_sha256"],
            runtime_provenance_sha256=payload["runtime_provenance_sha256"],
            run_nonce=payload["run_nonce"],
            compute_plan_sha256=payload["compute_plan_sha256"],
            execution_profile=execution_profile,
            output_schema_id=payload["output_schema_id"],
            output_schema_sha256=payload["output_schema_sha256"],
        )


@dataclass(frozen=True)
class EvaluationShardIdentity:
    """Identity and half-open input range for one independently resumable shard."""

    run: EvaluationRunIdentity
    checkpoint_sha256: str
    seed: int
    arm: str
    substage: str
    shard_id: str
    range_start: int
    range_stop: int

    def __post_init__(self) -> None:
        if not isinstance(self.run, EvaluationRunIdentity):
            raise TypeError("run must be an EvaluationRunIdentity")
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        _require_path_component(self.arm, "arm")
        _require_path_component(self.substage, "substage")
        _require_path_component(self.shard_id, "shard_id")
        if (
            type(self.range_start) is not int
            or type(self.range_stop) is not int
            or self.range_start < 0
            or self.range_stop <= self.range_start
        ):
            raise ValueError("shard range must be a nonempty half-open range")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SHARD_IDENTITY_SCHEMA,
            "run": self.run.as_dict(),
            "run_identity_sha256": self.run.sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "seed": self.seed,
            "arm": self.arm,
            "substage": self.substage,
            "shard_id": self.shard_id,
            "range_start": self.range_start,
            "range_stop": self.range_stop,
        }

    @property
    def sha256(self) -> str:
        return _json_sha256(self.as_dict())

    @classmethod
    def from_dict(cls, payload: object) -> "EvaluationShardIdentity":
        required = {
            "schema_version",
            "run",
            "run_identity_sha256",
            "checkpoint_sha256",
            "seed",
            "arm",
            "substage",
            "shard_id",
            "range_start",
            "range_stop",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("evaluation shard identity fields are invalid")
        if payload["schema_version"] != SHARD_IDENTITY_SCHEMA:
            raise ValueError("evaluation shard identity schema is invalid")
        run = EvaluationRunIdentity.from_dict(payload["run"])
        if payload["run_identity_sha256"] != run.sha256:
            raise ValueError("evaluation shard run identity digest is invalid")
        return cls(
            run=run,
            checkpoint_sha256=payload["checkpoint_sha256"],
            seed=payload["seed"],
            arm=payload["arm"],
            substage=payload["substage"],
            shard_id=payload["shard_id"],
            range_start=payload["range_start"],
            range_stop=payload["range_stop"],
        )


@dataclass(frozen=True)
class ShardRange:
    index: int
    start: int
    stop: int


@dataclass(frozen=True)
class ShardResumeResult:
    payload_path: Path
    manifest_path: Path
    manifest: dict[str, object]
    reused: bool


def iter_shard_ranges(total_units: int, shard_size: int) -> Iterator[ShardRange]:
    """Yield an O(number-of-shards), lazy plan without pairwise materialization."""

    if type(total_units) is not int or total_units < 0:
        raise ValueError("total_units must be a nonnegative integer")
    if type(shard_size) is not int or shard_size <= 0:
        raise ValueError("shard_size must be a positive integer")
    for index, start in enumerate(range(0, total_units, shard_size)):
        yield ShardRange(index=index, start=start, stop=min(start + shard_size, total_units))


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _require_directory(path: Path, *, create: bool) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if not _lexists(current):
            if not create:
                raise EvaluationResumeError(
                    f"resume path must be a regular directory: {path}"
                )
            try:
                os.mkdir(current)
            except FileExistsError:
                pass
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise EvaluationResumeError(
                f"cannot inspect resume directory component: {current}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise EvaluationResumeError(
                f"resume path component must be a regular directory: {current}"
            )


def _require_safe_existing_directory_prefix(path: Path) -> None:
    """Reject links in an existing prefix while permitting absent descendants."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if not _lexists(current):
            return
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise EvaluationResumeError(
                f"resume path component must be a regular directory: {current}"
            )


def require_safe_directory_prefix(path: str | Path) -> None:
    """Reject symlinks/non-directories in the existing prefix without creating it."""

    _require_safe_existing_directory_prefix(Path(path))


def _fsync_directory(path: Path) -> None:
    # Windows cannot open directories with os.open. The file itself is fsynced
    # before replacement; directory fsync remains required on POSIX.
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, writer: Callable[[BinaryIO], None]) -> tuple[int, str]:
    _require_directory(path.parent, create=True)
    if _lexists(path) and (path.is_symlink() or not path.is_file()):
        raise EvaluationResumeError(f"atomic target must be a regular file: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            writer(handle)
            if handle.closed:
                raise EvaluationResumeError("payload writer closed the atomic file")
            handle.flush()
            os.fsync(handle.fileno())
        size = temporary.stat().st_size
        digest = sha256_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        temporary = None
        return size, digest
    finally:
        if temporary is not None and _lexists(temporary):
            temporary.unlink()


def _atomic_json(path: Path, payload: object) -> tuple[int, str]:
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    return _atomic_write(path, lambda handle: handle.write(encoded))


def write_atomic_json(path: str | Path, payload: object) -> tuple[int, str]:
    """Atomically publish durable strict JSON and return its size and digest."""

    return _atomic_json(Path(path), payload)


def _matching_temporary_files(path: Path) -> list[Path]:
    if not path.parent.is_dir():
        return []
    return sorted(path.parent.glob(f".{path.name}.*.tmp"))


def _digest_record(digest, kind: str, relative: str, payload: bytes) -> None:
    for value in (kind.encode("ascii"), os.fsencode(relative), payload):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)


def _directory_evidence(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total_bytes = 0
    for current, directory_names, file_names in os.walk(
        path, topdown=True, followlinks=False
    ):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current)
        for name in directory_names + file_names:
            item = current_path / name
            relative = item.relative_to(path).as_posix()
            metadata = os.lstat(item)
            if stat.S_ISLNK(metadata.st_mode):
                target = os.fsencode(os.readlink(item))
                _digest_record(digest, "symlink", relative, target)
            elif stat.S_ISREG(metadata.st_mode):
                item_sha256 = sha256_file(item).encode("ascii")
                total_bytes += metadata.st_size
                _digest_record(digest, "file", relative, item_sha256)
            elif stat.S_ISDIR(metadata.st_mode):
                _digest_record(digest, "directory", relative, b"")
            else:
                descriptor = _canonical_json_bytes(
                    {
                        "mode": metadata.st_mode,
                        "size": metadata.st_size,
                    }
                )
                _digest_record(digest, "other", relative, descriptor)
    return total_bytes, digest.hexdigest()


def _quarantine_file_evidence(path: Path) -> tuple[str, int, str]:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode):
        target = os.fsencode(os.readlink(path))
        return "symlink", len(target), hashlib.sha256(target).hexdigest()
    if stat.S_ISREG(metadata.st_mode):
        return "file", metadata.st_size, sha256_file(path)
    if stat.S_ISDIR(metadata.st_mode):
        byte_count, digest = _directory_evidence(path)
        return "directory", byte_count, digest
    descriptor = _canonical_json_bytes(
        {
            "mode": metadata.st_mode,
            "size": metadata.st_size,
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
        }
    )
    return "other", metadata.st_size, hashlib.sha256(descriptor).hexdigest()


def _read_strict_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise EvaluationResumeError(f"{label} must be a regular file: {path}")
    try:
        payload = read_strict_json(path)
    except StrictJsonError as error:
        raise EvaluationResumeError(f"{label} is not valid strict JSON: {path}") from error
    if not isinstance(payload, dict):
        raise EvaluationResumeError(f"{label} must contain a JSON object: {path}")
    return payload


class EvaluationResumeStore:
    """Atomic, identity-bound storage for resumable evaluation shards."""

    def __init__(self, root: str | Path, run_identity: EvaluationRunIdentity):
        if not isinstance(run_identity, EvaluationRunIdentity):
            raise TypeError("run_identity must be an EvaluationRunIdentity")
        self.root = Path(root)
        self.run_identity = run_identity
        self._lock_fd: int | None = None
        self._lock_owner_thread: int | None = None
        self._lock_owner_pid: int | None = None
        self._lock_depth = 0
        self._state_guard = threading.Lock()
        self._progress_guard = threading.Lock()

    @property
    def lock_path(self) -> Path:
        return self.root / ".evaluation-writer.lock"

    @property
    def progress_path(self) -> Path:
        return self.root / "evaluation_progress.json"

    @property
    def quarantine_root(self) -> Path:
        return self.root / "quarantine"

    @property
    def progress_reconciliation_root(self) -> Path:
        return self.root / "audit" / "progress-reconciliation"

    def shard_paths(
        self, identity: EvaluationShardIdentity, *, suffix: str = ".bin"
    ) -> tuple[Path, Path]:
        self._require_matching_run(identity)
        extension = _require_suffix(suffix)
        payload = (
            self.root
            / "shards"
            / f"seed-{identity.seed}"
            / identity.arm
            / identity.substage
            / f"{identity.shard_id}{extension}"
        )
        _require_safe_existing_directory_prefix(payload.parent)
        return payload, payload.with_name(f"{payload.name}.manifest.json")

    @contextmanager
    def writer_lock(self) -> Iterator["EvaluationResumeStore"]:
        """Acquire the sole nonblocking writer lease for this resume root."""

        current_thread = threading.get_ident()
        current_pid = os.getpid()
        with self._state_guard:
            if self._lock_fd is not None:
                if (
                    self._lock_owner_thread != current_thread
                    or self._lock_owner_pid != current_pid
                ):
                    raise WriterLockError("evaluation writer lock belongs to another caller")
                self._lock_depth += 1
                nested = True
            else:
                _require_directory(self.root, create=True)
                flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                try:
                    descriptor = os.open(self.lock_path, flags, 0o600)
                except OSError as error:
                    raise WriterLockError(
                        f"cannot open evaluation writer lock: {self.lock_path}"
                    ) from error
                try:
                    flock(descriptor, LOCK_EX | LOCK_NB)
                except OSError as error:
                    os.close(descriptor)
                    if error.errno in {errno.EACCES, errno.EAGAIN}:
                        raise WriterLockError(
                            f"evaluation resume root already has a writer: {self.root}"
                        ) from error
                    raise WriterLockError("cannot acquire evaluation writer lock") from error
                try:
                    owner = {
                        "schema_version": WRITER_LOCK_SCHEMA,
                        "pid": current_pid,
                        "hostname": socket.gethostname(),
                        "thread_id": current_thread,
                        "run_identity_sha256": self.run_identity.sha256,
                        "acquired_at": _timestamp(),
                    }
                    encoded = _canonical_json_bytes(owner) + b"\n"
                    os.ftruncate(descriptor, 0)
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    remaining = memoryview(encoded)
                    while remaining:
                        written = os.write(descriptor, remaining)
                        if written <= 0:
                            raise WriterLockError(
                                "cannot write evaluation writer lock owner"
                            )
                        remaining = remaining[written:]
                    os.fsync(descriptor)
                except BaseException:
                    try:
                        flock(descriptor, LOCK_UN)
                    finally:
                        os.close(descriptor)
                    raise
                self._lock_fd = descriptor
                self._lock_owner_thread = current_thread
                self._lock_owner_pid = current_pid
                self._lock_depth = 1
                nested = False
        try:
            yield self
        finally:
            with self._state_guard:
                if self._lock_fd is None:
                    raise WriterLockError("evaluation writer lock was released unexpectedly")
                self._lock_depth -= 1
                if not nested and self._lock_depth == 0:
                    descriptor = self._lock_fd
                    self._lock_fd = None
                    self._lock_owner_thread = None
                    self._lock_owner_pid = None
                    try:
                        flock(descriptor, LOCK_UN)
                    finally:
                        os.close(descriptor)

    def commit_shard(
        self,
        identity: EvaluationShardIdentity,
        writer: Callable[[BinaryIO], None],
        *,
        suffix: str = ".bin",
        now: datetime | None = None,
    ) -> ShardResumeResult:
        """Publish one shard, or reuse it without invoking writer when already valid."""

        self._require_writer_lock()
        if not callable(writer):
            raise TypeError("writer must be callable")
        existing = self.load_completed_shard(identity, suffix=suffix)
        if existing is not None:
            return existing
        completed_at = _timestamp(now)
        payload_path, manifest_path = self.shard_paths(identity, suffix=suffix)
        payload_bytes, payload_sha256 = _atomic_write(payload_path, writer)
        relative_payload = payload_path.relative_to(self.root).as_posix()
        manifest: dict[str, object] = {
            "schema_version": SHARD_MANIFEST_SCHEMA,
            "status": "COMPLETE",
            "identity": identity.as_dict(),
            "identity_sha256": identity.sha256,
            "payload": {
                "path": relative_payload,
                "bytes": payload_bytes,
                "sha256": payload_sha256,
            },
            "completed_at": completed_at,
        }
        _atomic_json(manifest_path, manifest)
        return ShardResumeResult(
            payload_path=payload_path,
            manifest_path=manifest_path,
            manifest=manifest,
            reused=False,
        )

    def commit_shard_bytes(
        self,
        identity: EvaluationShardIdentity,
        payload: bytes,
        *,
        suffix: str = ".bin",
        now: datetime | None = None,
    ) -> ShardResumeResult:
        if type(payload) is not bytes:
            raise TypeError("payload must be bytes")
        return self.commit_shard(
            identity,
            lambda handle: handle.write(payload),
            suffix=suffix,
            now=now,
        )

    def load_completed_shard(
        self, identity: EvaluationShardIdentity, *, suffix: str = ".bin"
    ) -> ShardResumeResult | None:
        """Validate and return an exact completed shard without changing disk state."""

        payload_path, manifest_path = self.shard_paths(identity, suffix=suffix)
        payload_exists = _lexists(payload_path)
        manifest_exists = _lexists(manifest_path)
        temporary_files = _matching_temporary_files(payload_path) + _matching_temporary_files(
            manifest_path
        )
        if not payload_exists and not manifest_exists:
            if temporary_files:
                raise IncompleteShardError(
                    f"temporary files remain for incomplete shard {identity.shard_id}"
                )
            return None
        if not payload_exists or not manifest_exists:
            raise IncompleteShardError(
                f"payload and manifest must both exist for shard {identity.shard_id}"
            )
        if temporary_files:
            raise IncompleteShardError(
                f"temporary files remain beside shard {identity.shard_id}"
            )
        try:
            manifest = _read_strict_object(manifest_path, "evaluation shard manifest")
        except EvaluationResumeError as error:
            raise CorruptShardError(str(error)) from error
        self._validate_manifest(
            manifest,
            expected_identity=identity,
            expected_payload_path=payload_path,
        )
        if payload_path.is_symlink() or not payload_path.is_file():
            raise CorruptShardError("evaluation shard payload must be a regular file")
        payload = manifest["payload"]
        assert isinstance(payload, dict)
        actual_bytes = payload_path.stat().st_size
        if actual_bytes != payload["bytes"] or sha256_file(payload_path) != payload["sha256"]:
            raise CorruptShardError(
                f"evaluation shard payload hash or size mismatch: {payload_path}"
            )
        return ShardResumeResult(
            payload_path=payload_path,
            manifest_path=manifest_path,
            manifest=manifest,
            reused=True,
        )

    def load_or_quarantine_shard(
        self,
        identity: EvaluationShardIdentity,
        *,
        suffix: str = ".bin",
        now: datetime | None = None,
    ) -> ShardResumeResult | None:
        """Load one shard or preserve a corrupt/incomplete shard before recompute.

        A valid stale identity is never quarantined. It remains in place and raises
        StaleResumeError so callers cannot silently replace evidence from another run.
        """

        self._require_writer_lock()
        try:
            return self.load_completed_shard(identity, suffix=suffix)
        except IncompleteShardError as error:
            self._raise_if_persisted_manifest_is_stale(identity, suffix=suffix)
            self._quarantine_shard(identity, suffix=suffix, error=error, now=now)
            return None
        except CorruptShardError as error:
            self._quarantine_shard(identity, suffix=suffix, error=error, now=now)
            return None

    def quarantine_completed_shard(
        self,
        identity: EvaluationShardIdentity,
        *,
        suffix: str = ".bin",
        reason: str,
        now: datetime | None = None,
    ) -> Path:
        """Preserve a valid shard invalidated by authenticated dependent state."""

        self._require_writer_lock()
        normalized_reason = _require_text(
            reason, "completed shard quarantine reason", maximum=1024
        )
        completed = self.load_completed_shard(identity, suffix=suffix)
        if completed is None:
            raise IncompleteShardError(
                f"completed shard is missing: {identity.shard_id}"
            )
        return self._quarantine_shard(
            identity,
            suffix=suffix,
            error=CorruptShardError(normalized_reason),
            now=now,
        )

    def quarantine_final_output_temporaries(
        self,
        targets: Iterable[str | Path],
        *,
        now: datetime | None = None,
    ) -> Path | None:
        """Preserve crash leftovers for an exact set of final output targets."""

        self._require_writer_lock()
        values = tuple(Path(value) for value in targets)
        if not values:
            return None
        absolute_targets = tuple(
            Path(os.path.abspath(os.fspath(value))) for value in values
        )
        if len(absolute_targets) != len(set(absolute_targets)):
            raise ValueError("final output quarantine targets must be unique")
        expected_parent = Path(
            os.path.abspath(os.fspath(self.root.parent / "evaluation"))
        )
        _require_directory(expected_parent, create=False)
        for target in absolute_targets:
            _require_path_component(target.name, "final output target name")
            if target.parent != expected_parent:
                raise ValueError(
                    "final output quarantine target is outside the evaluation directory"
                )
            if _lexists(target) and (target.is_symlink() or not target.is_file()):
                raise EvaluationResumeError(
                    f"final output target must be a regular file: {target}"
                )

        sources: list[Path] = []
        seen: set[str] = set()
        for target in absolute_targets:
            for candidate in _matching_temporary_files(target):
                absolute = os.path.abspath(os.fspath(candidate))
                if absolute in seen:
                    continue
                seen.add(absolute)
                metadata = os.lstat(candidate)
                if not stat.S_ISREG(metadata.st_mode):
                    raise EvaluationResumeError(
                        "final output temporary evidence must be a regular file: "
                        f"{candidate}"
                    )
                sources.append(candidate)
        sources.sort(key=lambda path: path.name)
        if not sources:
            return None

        return self._quarantine_final_output_files(
            sources,
            schema=FINAL_OUTPUT_QUARANTINE_SCHEMA,
            reason="incomplete atomic final-output merge",
            tag="final-output",
            now=now,
        )

    def quarantine_final_output_artifacts(
        self,
        paths: Iterable[str | Path],
        *,
        reason: str,
        now: datetime | None = None,
    ) -> Path:
        """Preserve damaged regular final artifacts before deterministic rebuild."""

        self._require_writer_lock()
        normalized_reason = _require_text(
            reason, "final output artifact quarantine reason", maximum=1024
        )
        sources = tuple(
            Path(os.path.abspath(os.fspath(Path(path)))) for path in paths
        )
        if not sources:
            raise ValueError("final output artifact quarantine paths must not be empty")
        if len(sources) != len(set(sources)):
            raise ValueError("final output artifact quarantine paths must be unique")
        expected_parent = Path(
            os.path.abspath(os.fspath(self.root.parent / "evaluation"))
        )
        _require_directory(expected_parent, create=False)
        for source in sources:
            _require_path_component(source.name, "final output artifact name")
            if source.parent != expected_parent:
                raise ValueError(
                    "final output artifact is outside the evaluation directory"
                )
            if not _lexists(source) or source.is_symlink() or not source.is_file():
                raise EvaluationResumeError(
                    f"final output artifact must be a regular file: {source}"
                )
        return self._quarantine_final_output_files(
            sorted(sources, key=lambda path: path.name),
            schema=FINAL_OUTPUT_ARTIFACT_QUARANTINE_SCHEMA,
            reason=normalized_reason,
            tag="final-artifact",
            now=now,
        )

    def _quarantine_final_output_files(
        self,
        sources: Iterable[Path],
        *,
        schema: str,
        reason: str,
        tag: str,
        now: datetime | None,
    ) -> Path:
        self._require_writer_lock()
        source_rows = tuple(sources)
        if not source_rows:
            raise ValueError("final output quarantine sources must not be empty")

        _require_directory(self.quarantine_root, create=True)
        observed = _utc_datetime(now)
        prefix = observed.strftime("%Y%m%dT%H%M%S%fZ")
        evidence_digest = hashlib.sha256(
            _canonical_json_bytes(
                {
                    "run_identity_sha256": self.run_identity.sha256,
                    "sources": [source.name for source in source_rows],
                }
            )
        ).hexdigest()
        base_name = f"{prefix}-{tag}-{evidence_digest[:16]}"
        incident_path = self.quarantine_root / base_name
        counter = 0
        while _lexists(incident_path):
            counter += 1
            incident_path = self.quarantine_root / f"{base_name}-{counter:03d}"
        staging_path = Path(
            tempfile.mkdtemp(
                prefix=f".{base_name}.",
                suffix=".tmp",
                dir=self.quarantine_root,
            )
        )
        final_relative = incident_path.relative_to(self.root)
        run_root = Path(os.path.abspath(os.fspath(self.root.parent)))
        file_rows = []
        for index, source in enumerate(source_rows):
            metadata = os.lstat(source)
            if not stat.S_ISREG(metadata.st_mode):
                raise EvaluationResumeError(
                    f"final output evidence must remain a regular file: {source}"
                )
            byte_count = metadata.st_size
            digest = sha256_file(source)
            destination_name = f"{index:03d}-{source.name}"
            file_rows.append(
                {
                    "original_path": source.relative_to(run_root).as_posix(),
                    "quarantine_path": (
                        final_relative / destination_name
                    ).as_posix(),
                    "kind": "file",
                    "bytes": byte_count,
                    "sha256": digest,
                    "destination_name": destination_name,
                }
            )
        common = {
            "schema_version": schema,
            "run_identity": self.run_identity.as_dict(),
            "run_identity_sha256": self.run_identity.sha256,
            "reason": reason,
            "quarantined_at": _timestamp(observed),
        }
        public_rows = [
            {key: value for key, value in row.items() if key != "destination_name"}
            for row in file_rows
        ]
        _atomic_json(
            staging_path / "receipt.prepared.json",
            {**common, "status": "PREPARED", "files": public_rows},
        )
        for source, row in zip(source_rows, file_rows, strict=True):
            os.replace(source, staging_path / row["destination_name"])
            _fsync_directory(source.parent)
            _fsync_directory(staging_path)
        _atomic_json(
            staging_path / "receipt.json",
            {**common, "status": "COMPLETE", "files": public_rows},
        )
        os.replace(staging_path, incident_path)
        _fsync_directory(self.quarantine_root)
        return incident_path / "receipt.json"

    def _raise_if_persisted_manifest_is_stale(
        self, identity: EvaluationShardIdentity, *, suffix: str
    ) -> None:
        payload_path, manifest_path = self.shard_paths(identity, suffix=suffix)
        if manifest_path.is_symlink() or not manifest_path.is_file():
            return
        try:
            manifest = _read_strict_object(
                manifest_path, "evaluation shard manifest"
            )
            self._validate_manifest(
                manifest,
                expected_identity=identity,
                expected_payload_path=payload_path,
            )
        except StaleResumeError:
            raise
        except (EvaluationResumeError, OSError, TypeError, ValueError):
            return

    def _quarantine_shard(
        self,
        identity: EvaluationShardIdentity,
        *,
        suffix: str,
        error: EvaluationResumeError,
        now: datetime | None,
    ) -> Path:
        self._require_writer_lock()
        if not isinstance(error, (IncompleteShardError, CorruptShardError)):
            raise TypeError("only incomplete or corrupt shards may be quarantined")
        payload_path, manifest_path = self.shard_paths(identity, suffix=suffix)
        candidates = (
            [payload_path, manifest_path]
            + _matching_temporary_files(payload_path)
            + _matching_temporary_files(manifest_path)
        )
        sources = []
        seen = set()
        for candidate in candidates:
            absolute = os.path.abspath(os.fspath(candidate))
            if absolute not in seen and _lexists(candidate):
                seen.add(absolute)
                sources.append(candidate)
        if not sources:
            raise IncompleteShardError(
                "quarantine found no remaining evidence for the failed shard"
            ) from error

        _require_directory(self.quarantine_root, create=True)
        observed = _utc_datetime(now)
        prefix = observed.strftime("%Y%m%dT%H%M%S%fZ")
        base_name = f"{prefix}-{identity.sha256[:16]}"
        incident_path = self.quarantine_root / base_name
        counter = 0
        while _lexists(incident_path):
            counter += 1
            incident_path = self.quarantine_root / f"{base_name}-{counter:03d}"
        staging_path = Path(
            tempfile.mkdtemp(
                prefix=f".{base_name}.",
                suffix=".tmp",
                dir=self.quarantine_root,
            )
        )
        final_relative = incident_path.relative_to(self.root)
        file_rows = []
        for index, source in enumerate(sources):
            kind, byte_count, digest = _quarantine_file_evidence(source)
            destination_name = f"{index:03d}-{source.name}"
            file_rows.append(
                {
                    "original_path": source.relative_to(self.root).as_posix(),
                    "quarantine_path": (
                        final_relative / destination_name
                    ).as_posix(),
                    "kind": kind,
                    "bytes": byte_count,
                    "sha256": digest,
                    "destination_name": destination_name,
                }
            )
        common = {
            "schema_version": QUARANTINE_RECEIPT_SCHEMA,
            "run_identity_sha256": self.run_identity.sha256,
            "shard_identity": identity.as_dict(),
            "shard_identity_sha256": identity.sha256,
            "payload_suffix": suffix,
            "reason": {
                "class": type(error).__name__,
                "message": str(error),
            },
            "quarantined_at": _timestamp(observed),
        }
        _atomic_json(
            staging_path / "receipt.prepared.json",
            {
                **common,
                "status": "PREPARED",
                "files": [
                    {key: value for key, value in row.items() if key != "destination_name"}
                    for row in file_rows
                ],
            },
        )
        for source, row in zip(sources, file_rows, strict=True):
            os.replace(source, staging_path / row["destination_name"])
            _fsync_directory(source.parent)
            _fsync_directory(staging_path)
        receipt = {
            **common,
            "status": "COMPLETE",
            "files": [
                {key: value for key, value in row.items() if key != "destination_name"}
                for row in file_rows
            ],
        }
        _atomic_json(staging_path / "receipt.json", receipt)
        os.replace(staging_path, incident_path)
        _fsync_directory(self.quarantine_root)
        return incident_path / "receipt.json"

    def initialize_progress(
        self, total_units: int, *, now: datetime | None = None
    ) -> dict[str, object]:
        """Create the authoritative progress receipt, idempotently."""

        self._require_writer_lock()
        if type(total_units) is not int or total_units < 0:
            raise ValueError("total_units must be a nonnegative integer")
        existing = read_evaluation_status(self.root, expected_identity=self.run_identity)
        if existing is not None:
            if existing["total_units"] != total_units:
                raise StaleResumeError("evaluation progress total_units changed")
            return existing
        started = _utc_datetime(now)
        complete = total_units == 0
        payload = {
            "schema_version": PROGRESS_SCHEMA,
            "run_identity": self.run_identity.as_dict(),
            "run_identity_sha256": self.run_identity.sha256,
            "status": "COMPLETE" if complete else "RUNNING",
            "total_units": total_units,
            "completed_units": 0,
            "percentage": 100.0 if complete else 0.0,
            "current_seed": None,
            "current_arm": None,
            "current_substage": None,
            "current_shard": None,
            "started_at": _timestamp(started),
            "updated_at": _timestamp(started),
            "completed_at": _timestamp(started) if complete else None,
            "elapsed_seconds": 0.0,
            "throughput_units_per_second": 0.0,
            "eta_seconds": 0.0 if complete else None,
            "estimated_completion_at": _timestamp(started) if complete else None,
        }
        _validate_progress(payload, self.run_identity)
        _atomic_json(self.progress_path, payload)
        return payload

    def update_progress(
        self,
        completed_units: int,
        *,
        current_seed: int | None,
        current_arm: str | None,
        current_substage: str | None,
        current_shard: str | None,
        now: datetime | None = None,
    ) -> dict[str, object]:
        """Atomically publish a heartbeat or shard-completion progress update."""

        self._require_process_writer_lock()
        with self._progress_guard:
            return self._update_progress_locked(
                completed_units,
                current_seed=current_seed,
                current_arm=current_arm,
                current_substage=current_substage,
                current_shard=current_shard,
                now=now,
            )

    def reconcile_progress(
        self,
        reauthenticated_completed_units: int,
        *,
        current_seed: int | None,
        current_arm: str | None,
        current_substage: str | None,
        current_shard: str | None,
        reason: str,
        now: datetime | None = None,
    ) -> dict[str, object]:
        """Lower progress to a freshly authenticated shard count.

        Normal progress remains monotone through :meth:`update_progress`. This
        recovery-only operation permits a decrease after shard validation has
        disproved the persisted count. The exact previous receipt is archived
        before the replacement is published.
        """

        self._require_writer_lock()
        normalized_reason = _require_text(
            reason, "reconciliation reason", maximum=1024
        )
        with self._progress_guard:
            progress = read_evaluation_status(
                self.root, expected_identity=self.run_identity
            )
            if progress is None:
                raise ProgressValidationError(
                    "evaluation progress is not initialized"
                )
            if type(reauthenticated_completed_units) is not int:
                raise ValueError(
                    "reauthenticated_completed_units must be an integer"
                )
            previous_completed = progress["completed_units"]
            total_units = progress["total_units"]
            assert type(previous_completed) is int and type(total_units) is int
            if (
                reauthenticated_completed_units < 0
                or reauthenticated_completed_units > total_units
            ):
                raise ValueError(
                    "reauthenticated_completed_units must be within total_units"
                )
            if reauthenticated_completed_units > previous_completed:
                raise ValueError(
                    "reconcile_progress cannot advance progress; use update_progress"
                )
            if reauthenticated_completed_units == previous_completed:
                return progress
            updated = _utc_datetime(now)
            replacement = self._progress_payload(
                progress,
                reauthenticated_completed_units,
                current_seed=current_seed,
                current_arm=current_arm,
                current_substage=current_substage,
                current_shard=current_shard,
                updated=updated,
            )
            self._archive_progress_reconciliation(
                progress,
                replacement,
                reason=normalized_reason,
                reconciled_at=updated,
            )
            _atomic_json(self.progress_path, replacement)
            return replacement

    def _update_progress_locked(
        self,
        completed_units: int,
        *,
        current_seed: int | None,
        current_arm: str | None,
        current_substage: str | None,
        current_shard: str | None,
        now: datetime | None,
    ) -> dict[str, object]:

        progress = read_evaluation_status(self.root, expected_identity=self.run_identity)
        if progress is None:
            raise ProgressValidationError("evaluation progress is not initialized")
        if type(completed_units) is not int:
            raise ValueError("completed_units must be an integer")
        previous_completed = progress["completed_units"]
        total_units = progress["total_units"]
        assert type(previous_completed) is int and type(total_units) is int
        if completed_units < previous_completed or completed_units > total_units:
            raise ValueError("completed_units must be monotone and within total_units")
        if progress["status"] == "COMPLETE":
            if completed_units != total_units:
                raise ValueError("completed evaluation progress cannot be reopened")
            return progress
        updated = _utc_datetime(now)
        previous_update = _parse_timestamp(progress["updated_at"], "updated_at")
        if updated < previous_update:
            raise ValueError("progress timestamp cannot move backwards")
        payload = self._progress_payload(
            progress,
            completed_units,
            current_seed=current_seed,
            current_arm=current_arm,
            current_substage=current_substage,
            current_shard=current_shard,
            updated=updated,
        )
        _atomic_json(self.progress_path, payload)
        return payload

    def _progress_payload(
        self,
        progress: dict[str, object],
        completed_units: int,
        *,
        current_seed: int | None,
        current_arm: str | None,
        current_substage: str | None,
        current_shard: str | None,
        updated: datetime,
    ) -> dict[str, object]:
        _validate_current_position(
            current_seed,
            current_arm,
            current_substage,
            current_shard,
        )
        started = _parse_timestamp(progress["started_at"], "started_at")
        previous_update = _parse_timestamp(progress["updated_at"], "updated_at")
        if updated < previous_update:
            raise ValueError("progress timestamp cannot move backwards")
        total_units = progress["total_units"]
        assert type(total_units) is int
        elapsed = (updated - started).total_seconds()
        throughput = completed_units / elapsed if elapsed > 0.0 else 0.0
        complete = completed_units == total_units
        if complete:
            eta = 0.0
            estimated_completion = updated
        elif throughput > 0.0:
            eta = (total_units - completed_units) / throughput
            estimated_completion = updated + timedelta(seconds=eta)
        else:
            eta = None
            estimated_completion = None
        payload = {
            **progress,
            "status": "COMPLETE" if complete else "RUNNING",
            "completed_units": completed_units,
            "percentage": (
                100.0 if total_units == 0 else 100.0 * completed_units / total_units
            ),
            "current_seed": current_seed,
            "current_arm": current_arm,
            "current_substage": current_substage,
            "current_shard": current_shard,
            "updated_at": _timestamp(updated),
            "completed_at": _timestamp(updated) if complete else None,
            "elapsed_seconds": elapsed,
            "throughput_units_per_second": throughput,
            "eta_seconds": eta,
            "estimated_completion_at": (
                _timestamp(estimated_completion)
                if estimated_completion is not None
                else None
            ),
        }
        _validate_progress(payload, self.run_identity)
        return payload

    def _archive_progress_reconciliation(
        self,
        previous: dict[str, object],
        replacement: dict[str, object],
        *,
        reason: str,
        reconciled_at: datetime,
    ) -> Path:
        self._require_writer_lock()
        previous_bytes = self.progress_path.read_bytes()
        previous_file_sha256 = hashlib.sha256(previous_bytes).hexdigest()
        try:
            archived_payload = parse_strict_json(previous_bytes.decode("utf-8"))
        except (UnicodeError, StrictJsonError) as error:
            raise ProgressValidationError(
                "evaluation progress changed before reconciliation archival"
            ) from error
        if archived_payload != previous:
            raise ProgressValidationError(
                "evaluation progress changed before reconciliation archival"
            )
        if sha256_file(self.progress_path) != previous_file_sha256:
            raise ProgressValidationError(
                "evaluation progress changed while reconciliation was archived"
            )
        _require_directory(self.progress_reconciliation_root, create=True)
        prefix = reconciled_at.strftime("%Y%m%dT%H%M%S%fZ")
        base_name = f"{prefix}-{previous_file_sha256[:16]}"
        incident_path = self.progress_reconciliation_root / base_name
        counter = 0
        while _lexists(incident_path):
            counter += 1
            incident_path = self.progress_reconciliation_root / (
                f"{base_name}-{counter:03d}"
            )
        staging_path = Path(
            tempfile.mkdtemp(
                prefix=f".{base_name}.",
                suffix=".tmp",
                dir=self.progress_reconciliation_root,
            )
        )
        archived_name = "previous-evaluation-progress.json"
        archived_path = staging_path / archived_name
        archived_bytes, archived_sha256 = _atomic_write(
            archived_path, lambda handle: handle.write(previous_bytes)
        )
        if (
            archived_bytes != len(previous_bytes)
            or archived_sha256 != previous_file_sha256
        ):
            raise ProgressValidationError(
                "archived evaluation progress does not match source receipt"
            )
        receipt = {
            "schema_version": PROGRESS_RECONCILIATION_SCHEMA,
            "status": "COMPLETE",
            "run_identity": self.run_identity.as_dict(),
            "run_identity_sha256": self.run_identity.sha256,
            "reason": reason,
            "reconciled_at": _timestamp(reconciled_at),
            "previous_completed_units": previous["completed_units"],
            "reauthenticated_completed_units": replacement["completed_units"],
            "previous_progress": {
                "source_path": self.progress_path.relative_to(self.root).as_posix(),
                "archive_path": archived_name,
                "bytes": archived_bytes,
                "sha256": archived_sha256,
                "payload_sha256": _json_sha256(previous),
            },
            "replacement_progress_payload_sha256": _json_sha256(replacement),
        }
        _atomic_json(staging_path / "receipt.json", receipt)
        os.replace(staging_path, incident_path)
        _fsync_directory(self.progress_reconciliation_root)
        return incident_path / "receipt.json"

    def read_status(self) -> dict[str, object] | None:
        return read_evaluation_status(self.root, expected_identity=self.run_identity)

    def _require_matching_run(self, identity: EvaluationShardIdentity) -> None:
        if not isinstance(identity, EvaluationShardIdentity):
            raise TypeError("identity must be an EvaluationShardIdentity")
        if identity.run != self.run_identity:
            raise StaleResumeError("shard identity belongs to a different evaluation run")

    def _require_writer_lock(self) -> None:
        if (
            self._lock_fd is None
            or self._lock_owner_pid != os.getpid()
            or self._lock_owner_thread != threading.get_ident()
        ):
            raise WriterLockError("evaluation writer lock is required for this operation")

    def _require_process_writer_lock(self) -> None:
        if self._lock_fd is None or self._lock_owner_pid != os.getpid():
            raise WriterLockError("evaluation writer lock is required for this operation")

    def _validate_manifest(
        self,
        manifest: dict[str, object],
        *,
        expected_identity: EvaluationShardIdentity,
        expected_payload_path: Path,
    ) -> None:
        required = {
            "schema_version",
            "status",
            "identity",
            "identity_sha256",
            "payload",
            "completed_at",
        }
        if set(manifest) != required:
            raise CorruptShardError("evaluation shard manifest fields are invalid")
        if manifest["schema_version"] != SHARD_MANIFEST_SCHEMA or manifest["status"] != "COMPLETE":
            raise CorruptShardError("evaluation shard manifest schema or status is invalid")
        try:
            persisted_identity = EvaluationShardIdentity.from_dict(manifest["identity"])
        except (TypeError, ValueError) as error:
            raise CorruptShardError("evaluation shard identity is invalid") from error
        if manifest["identity_sha256"] != persisted_identity.sha256:
            raise CorruptShardError("evaluation shard identity digest is invalid")
        if persisted_identity != expected_identity:
            raise StaleResumeError(
                f"evaluation shard identity is stale: {expected_identity.shard_id}"
            )
        payload = manifest["payload"]
        if not isinstance(payload, dict) or set(payload) != {"path", "bytes", "sha256"}:
            raise CorruptShardError("evaluation shard payload receipt is invalid")
        expected_relative = expected_payload_path.relative_to(self.root).as_posix()
        if (
            payload["path"] != expected_relative
            or type(payload["bytes"]) is not int
            or payload["bytes"] < 0
            or type(payload["sha256"]) is not str
            or _SHA256_RE.fullmatch(payload["sha256"]) is None
        ):
            raise CorruptShardError("evaluation shard payload binding is invalid")
        try:
            _parse_timestamp(manifest["completed_at"], "completed_at")
        except ValueError as error:
            raise CorruptShardError("evaluation shard completion timestamp is invalid") from error


def _validate_current_position(
    seed: object,
    arm: object,
    substage: object,
    shard: object,
) -> None:
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("current_seed must be null or a nonnegative integer")
    for value, label in (
        (arm, "current_arm"),
        (substage, "current_substage"),
        (shard, "current_shard"),
    ):
        if value is not None:
            _require_path_component(value, label)


def _validate_optional_nonnegative_float(value: object, label: str) -> None:
    if value is None:
        return
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be null or a finite nonnegative float")


def _validate_nonnegative_float(value: object, label: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative float")


def _validate_progress(
    payload: dict[str, object], expected_identity: EvaluationRunIdentity | None
) -> None:
    required = {
        "schema_version",
        "run_identity",
        "run_identity_sha256",
        "status",
        "total_units",
        "completed_units",
        "percentage",
        "current_seed",
        "current_arm",
        "current_substage",
        "current_shard",
        "started_at",
        "updated_at",
        "completed_at",
        "elapsed_seconds",
        "throughput_units_per_second",
        "eta_seconds",
        "estimated_completion_at",
    }
    if set(payload) != required or payload["schema_version"] != PROGRESS_SCHEMA:
        raise ProgressValidationError("evaluation progress fields or schema are invalid")
    try:
        identity = EvaluationRunIdentity.from_dict(payload["run_identity"])
    except (TypeError, ValueError) as error:
        raise ProgressValidationError("evaluation progress run identity is invalid") from error
    if payload["run_identity_sha256"] != identity.sha256:
        raise ProgressValidationError("evaluation progress identity digest is invalid")
    if expected_identity is not None and identity != expected_identity:
        raise StaleResumeError("evaluation progress belongs to a different run identity")
    total = payload["total_units"]
    completed = payload["completed_units"]
    percentage = payload["percentage"]
    if (
        type(total) is not int
        or total < 0
        or type(completed) is not int
        or completed < 0
        or completed > total
        or type(percentage) is not float
        or not math.isfinite(percentage)
    ):
        raise ProgressValidationError("evaluation progress counts are invalid")
    expected_percentage = 100.0 if total == 0 else 100.0 * completed / total
    if not math.isclose(percentage, expected_percentage, rel_tol=0.0, abs_tol=1e-12):
        raise ProgressValidationError("evaluation progress percentage is inconsistent")
    complete = completed == total
    expected_status = "COMPLETE" if complete else "RUNNING"
    if payload["status"] != expected_status:
        raise ProgressValidationError("evaluation progress status is inconsistent")
    try:
        _validate_current_position(
            payload["current_seed"],
            payload["current_arm"],
            payload["current_substage"],
            payload["current_shard"],
        )
        started = _parse_timestamp(payload["started_at"], "started_at")
        updated = _parse_timestamp(payload["updated_at"], "updated_at")
        if updated < started:
            raise ValueError("updated_at precedes started_at")
        completed_at = (
            _parse_timestamp(payload["completed_at"], "completed_at")
            if payload["completed_at"] is not None
            else None
        )
        estimated_at = (
            _parse_timestamp(payload["estimated_completion_at"], "estimated_completion_at")
            if payload["estimated_completion_at"] is not None
            else None
        )
        _validate_nonnegative_float(payload["elapsed_seconds"], "elapsed_seconds")
        _validate_nonnegative_float(
            payload["throughput_units_per_second"], "throughput_units_per_second"
        )
        _validate_optional_nonnegative_float(payload["eta_seconds"], "eta_seconds")
    except ValueError as error:
        raise ProgressValidationError(str(error)) from error
    elapsed = payload["elapsed_seconds"]
    throughput = payload["throughput_units_per_second"]
    eta = payload["eta_seconds"]
    assert type(elapsed) is float and type(throughput) is float
    expected_elapsed = (updated - started).total_seconds()
    if not math.isclose(elapsed, expected_elapsed, rel_tol=0.0, abs_tol=1e-9):
        raise ProgressValidationError("evaluation elapsed time is inconsistent")
    expected_throughput = completed / elapsed if elapsed > 0.0 else 0.0
    if not math.isclose(throughput, expected_throughput, rel_tol=1e-12, abs_tol=1e-15):
        raise ProgressValidationError("evaluation throughput is inconsistent")
    if complete:
        if completed_at is None or eta != 0.0 or estimated_at != completed_at:
            raise ProgressValidationError("completed evaluation timestamps are inconsistent")
    else:
        if completed_at is not None:
            raise ProgressValidationError("running evaluation cannot have completed_at")
        if throughput > 0.0:
            expected_eta = (total - completed) / throughput
            if (
                type(eta) is not float
                or not math.isclose(eta, expected_eta, rel_tol=1e-12, abs_tol=1e-12)
                or estimated_at is None
            ):
                raise ProgressValidationError("evaluation ETA is inconsistent")
            expected_estimated = updated + timedelta(seconds=eta)
            if abs((estimated_at - expected_estimated).total_seconds()) > 1e-6:
                raise ProgressValidationError("estimated completion timestamp is inconsistent")
        elif eta is not None or estimated_at is not None:
            raise ProgressValidationError("ETA requires positive observed throughput")


def read_evaluation_status(
    root: str | Path,
    *,
    expected_identity: EvaluationRunIdentity | None = None,
) -> dict[str, object] | None:
    """Read and authenticate progress without creating files or acquiring a lock."""

    base = Path(root)
    path = base / "evaluation_progress.json"
    if not _lexists(path):
        if _matching_temporary_files(path):
            raise ProgressValidationError("evaluation progress has only a temporary file")
        return None
    try:
        payload = _read_strict_object(path, "evaluation progress")
    except EvaluationResumeError as error:
        raise ProgressValidationError(str(error)) from error
    _validate_progress(payload, expected_identity)
    return payload
