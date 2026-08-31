from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

from . import formal_migration_evidence as evidence
from .formal_io import parse_strict_json, read_strict_json, sha256_file


RUNNER_SCHEMA = "csi-pairs-v6-post-exit-migration-runner-v1"
OPERATION_LOCK_SCHEMA = "csi-pairs-v6-operation-lock-v2"
INVENTORY_NAME = "legacy_evaluation_inventory.json"
FREEZE_NAME = "legacy_evaluation_post_exit_freeze.json"
_ALLOWED_OUTPUT_NAMES = frozenset({INVENTORY_NAME, FREEZE_NAME})
_HEX64 = frozenset("0123456789abcdef")
_PROC_ROOT = Path("/proc")


class PostExitMigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class _FileRecord:
    path: Path
    device: int
    inode: int
    links: int
    mode: int
    size: int
    sha256: str
    content: bytes


@dataclass(frozen=True)
class _GuardRecord:
    path: Path
    device: int
    inode: int
    links: int
    mode: int
    size: int
    mtime_ns: int
    sha256: str


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _regular_directory(path: str | Path, label: str) -> Path:
    requested = Path(path)
    if not requested.is_absolute() or requested.is_symlink() or not requested.is_dir():
        raise PostExitMigrationError(f"{label} must be an absolute regular directory")
    resolved = requested.resolve()
    if requested != resolved:
        raise PostExitMigrationError(f"{label} must use its canonical path")
    return resolved


def _regular_file(path: str | Path, label: str) -> Path:
    requested = Path(path)
    if not requested.is_absolute() or requested.is_symlink() or not requested.is_file():
        raise PostExitMigrationError(f"{label} must be an absolute regular file")
    resolved = requested.resolve()
    if requested != resolved:
        raise PostExitMigrationError(f"{label} must use its canonical path")
    return resolved


def _canonical_target(path: str | Path, label: str) -> Path:
    requested = Path(path)
    if not requested.is_absolute() or requested.is_symlink():
        raise PostExitMigrationError(f"{label} must be an absolute non-symlink path")
    parent = _regular_directory(requested.parent, f"{label} parent")
    target = parent / requested.name
    if requested != target:
        raise PostExitMigrationError(f"{label} must use its canonical path")
    return target


def _require_external(path: Path, legacy_root: Path, new_root: Path, label: str) -> None:
    if _inside(path, legacy_root) or _inside(path, new_root):
        raise PostExitMigrationError(f"{label} must be external to both run roots")


def _prepare_output_directory(
    path: str | Path, legacy_root: Path, new_root: Path
) -> Path:
    target = _canonical_target(path, "post-exit output directory")
    _require_external(target, legacy_root, new_root, "post-exit output directory")
    if _lexists(target):
        output = _regular_directory(target, "post-exit output directory")
    else:
        target.mkdir(mode=0o700)
        _fsync_directory(target.parent)
        output = _regular_directory(target, "post-exit output directory")
    unexpected = sorted(
        item.name
        for item in output.iterdir()
        if item.name not in _ALLOWED_OUTPUT_NAMES
    )
    if unexpected:
        raise PostExitMigrationError(
            "post-exit output directory contains unexpected entries: "
            + ", ".join(unexpected)
        )
    return output


def _read_descriptor(descriptor: int) -> bytes:
    size = os.fstat(descriptor).st_size
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise PostExitMigrationError("regular file became truncated while reading")
        chunks.append(chunk)
        offset += len(chunk)
    if os.fstat(descriptor).st_size != size:
        raise PostExitMigrationError("regular file size changed while reading")
    return b"".join(chunks)


def _read_file_record(path: Path, label: str) -> _FileRecord:
    try:
        before = os.lstat(path)
    except OSError as error:
        raise PostExitMigrationError(f"cannot inspect {label}: {path}") from error
    if not stat.S_ISREG(before.st_mode):
        raise PostExitMigrationError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PostExitMigrationError(f"cannot open {label}: {path}") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise PostExitMigrationError(f"{label} changed while opening")
        content = _read_descriptor(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        after.st_dev != opened.st_dev
        or after.st_ino != opened.st_ino
        or after.st_size != opened.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
    ):
        raise PostExitMigrationError(f"{label} changed while reading")
    return _FileRecord(
        path=path,
        device=after.st_dev,
        inode=after.st_ino,
        links=after.st_nlink,
        mode=stat.S_IMODE(after.st_mode),
        size=after.st_size,
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
    )


def _guard_record(path: Path, descriptor: int) -> _GuardRecord:
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode):
        raise PostExitMigrationError("operation-lock guard is not a regular file")
    content = _read_descriptor(descriptor)
    current = os.fstat(descriptor)
    return _GuardRecord(
        path=path,
        device=current.st_dev,
        inode=current.st_ino,
        links=current.st_nlink,
        mode=stat.S_IMODE(current.st_mode),
        size=current.st_size,
        mtime_ns=current.st_mtime_ns,
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _verify_guard_unchanged(
    path: Path, descriptor: int, snapshot: _GuardRecord
) -> None:
    try:
        observed = os.lstat(path)
    except OSError as error:
        raise PostExitMigrationError("operation-lock guard disappeared") from error
    after = _guard_record(path, descriptor)
    if (
        observed.st_dev != snapshot.device
        or observed.st_ino != snapshot.inode
        or after != snapshot
    ):
        raise PostExitMigrationError("operation-lock guard changed during migration")


@contextmanager
def _exclusive_guard(path: Path) -> Iterator[_GuardRecord]:
    if path.is_symlink() or not path.is_file():
        raise PostExitMigrationError("operation-lock guard is missing or invalid")
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode):
        raise PostExitMigrationError("operation-lock guard is not regular")
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PostExitMigrationError("cannot open operation-lock guard") from error
    locked = False
    try:
        opened = os.fstat(descriptor)
        if opened.st_dev != before.st_dev or opened.st_ino != before.st_ino:
            raise PostExitMigrationError("operation-lock guard changed while opening")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as error:
            raise PostExitMigrationError(
                "legacy run still has an active operation-lock guard holder"
            ) from error
        snapshot = _guard_record(path, descriptor)
        try:
            yield snapshot
        finally:
            _verify_guard_unchanged(path, descriptor, snapshot)
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX64 for character in value)
    ):
        raise PostExitMigrationError(f"{label} must be a lowercase SHA-256")
    return value


def _expected_lock_payload(legacy_root: Path, pid: int) -> dict[str, object]:
    return {
        "output_root": str(legacy_root),
        "pid": pid,
        "schema_version": OPERATION_LOCK_SCHEMA,
    }


def _validate_lock_record(
    record: _FileRecord,
    *,
    expected_sha256: str,
    expected_payload: Mapping[str, object],
    label: str,
) -> None:
    if record.sha256 != expected_sha256:
        raise PostExitMigrationError(f"{label} SHA-256 mismatch")
    try:
        text = record.content.decode("ascii")
        payload = parse_strict_json(text)
    except (UnicodeError, ValueError) as error:
        raise PostExitMigrationError(f"{label} is not canonical strict JSON") from error
    if payload != dict(expected_payload):
        raise PostExitMigrationError(f"{label} JSON identity mismatch")


def _require_pid_absent(pid: int) -> None:
    if _lexists(_PROC_ROOT / str(pid)):
        raise PostExitMigrationError(
            f"refusing stale-lock archival while PID {pid} is present"
        )


def _archive_stale_lock(
    lock_path: Path,
    archive_path: Path,
    *,
    expected_sha256: str,
    expected_payload: Mapping[str, object],
    expected_pid: int,
    after_link: Callable[[], None] | None = None,
) -> tuple[str, _FileRecord]:
    _require_pid_absent(expected_pid)
    lock_exists = _lexists(lock_path)
    archive_exists = _lexists(archive_path)
    if not lock_exists and not archive_exists:
        raise PostExitMigrationError("canonical lock and archive are both absent")

    lock = _read_file_record(lock_path, "canonical operation lock") if lock_exists else None
    archive = _read_file_record(archive_path, "operation-lock archive") if archive_exists else None
    if lock is not None:
        _validate_lock_record(
            lock,
            expected_sha256=expected_sha256,
            expected_payload=expected_payload,
            label="canonical operation lock",
        )
    if archive is not None:
        _validate_lock_record(
            archive,
            expected_sha256=expected_sha256,
            expected_payload=expected_payload,
            label="operation-lock archive",
        )

    if lock is not None and archive is not None:
        if (
            lock.device != archive.device
            or lock.inode != archive.inode
            or lock.links != 2
            or archive.links != 2
        ):
            raise PostExitMigrationError(
                "canonical lock and archive are not the same two-link inode"
            )
        state = "RECOVERED_LINKED_ARCHIVE"
    elif lock is not None:
        if lock.links != 1:
            raise PostExitMigrationError("canonical operation lock has unexpected hard links")
        archive_parent = os.stat(archive_path.parent, follow_symlinks=False)
        if archive_parent.st_dev != lock.device:
            raise PostExitMigrationError("operation-lock archive is on another filesystem")
        try:
            os.link(lock_path, archive_path, follow_symlinks=False)
        except OSError as error:
            raise PostExitMigrationError("cannot hard-link operation-lock archive") from error
        _fsync_directory(archive_path.parent)
        archive = _read_file_record(archive_path, "operation-lock archive")
        _validate_lock_record(
            archive,
            expected_sha256=expected_sha256,
            expected_payload=expected_payload,
            label="operation-lock archive",
        )
        if (
            archive.device != lock.device
            or archive.inode != lock.inode
            or archive.links != 2
        ):
            raise PostExitMigrationError("hard-linked archive identity mismatch")
        if after_link is not None:
            after_link()
        state = "ARCHIVED_NOW"
    else:
        assert archive is not None
        if archive.links != 1:
            raise PostExitMigrationError("completed operation-lock archive has extra links")
        state = "ARCHIVE_ALREADY_COMPLETE"

    if _lexists(lock_path):
        current_lock = _read_file_record(lock_path, "canonical operation lock")
        current_archive = _read_file_record(archive_path, "operation-lock archive")
        if (
            current_lock.device != current_archive.device
            or current_lock.inode != current_archive.inode
            or current_lock.links != 2
            or current_archive.links != 2
        ):
            raise PostExitMigrationError("operation-lock hard-link identity changed")
        _require_pid_absent(expected_pid)
        os.unlink(lock_path)
        _fsync_directory(lock_path.parent)

    if _lexists(lock_path):
        raise PostExitMigrationError("canonical operation lock still exists")
    archive = _read_file_record(archive_path, "completed operation-lock archive")
    _validate_lock_record(
        archive,
        expected_sha256=expected_sha256,
        expected_payload=expected_payload,
        label="completed operation-lock archive",
    )
    if archive.links != 1:
        raise PostExitMigrationError("completed operation-lock archive has extra links")
    return state, archive


def _load_or_write_inventory(
    path: Path,
    *,
    identity: Mapping[str, object],
    command: str,
    legacy_root: Path,
    new_root: Path,
) -> dict[str, object]:
    if _lexists(path):
        if path.is_symlink() or not path.is_file():
            raise PostExitMigrationError("existing legacy inventory is not regular")
        payload = read_strict_json(path)
        if not isinstance(payload, dict):
            raise PostExitMigrationError("existing legacy inventory is not an object")
    else:
        payload = evidence.write_legacy_evaluation_inventory_report(
            path,
            expected_identity=identity,
            command=command,
        )
    evidence.validate_evidence_report(
        "legacy_evaluation_inventory",
        payload,
        expected_identity=identity,
        legacy_run_root=legacy_root,
        new_run_root=new_root,
    )
    return payload


def _load_or_write_freeze(
    path: Path,
    *,
    identity: Mapping[str, object],
    command: str,
    inventory_path: Path,
    raw: Mapping[str, Path],
    legacy_root: Path,
    new_root: Path,
) -> dict[str, object]:
    if _lexists(path):
        if path.is_symlink() or not path.is_file():
            raise PostExitMigrationError("existing post-exit freeze receipt is not regular")
        payload = read_strict_json(path)
        if not isinstance(payload, dict):
            raise PostExitMigrationError("existing post-exit freeze receipt is not an object")
    else:
        payload = evidence.write_legacy_evaluation_post_exit_freeze_receipt(
            path,
            expected_identity=identity,
            command=command,
            inventory_path=inventory_path,
            pid_snapshot_path=raw["pid_snapshot"],
            monitor_path=raw["monitor"],
            kernel_oom_evidence_path=raw["kernel_oom_evidence"],
            exit_site_path=raw["exit_site"],
            timing_path=raw["timing"],
            supervisor_log_path=raw["supervisor_log"],
            supervisor_script_path=raw["supervisor_script"],
            config_path=raw["config"],
        )
    evidence.validate_evidence_report(
        "legacy_evaluation_freeze",
        payload,
        expected_identity=identity,
        legacy_run_root=legacy_root,
        new_run_root=new_root,
    )
    return payload


def run_post_exit_migration(
    *,
    expected_identity: Mapping[str, object],
    legacy_run_root: str | Path,
    output_dir: str | Path,
    operation_lock_path: str | Path,
    archive_path: str | Path,
    expected_lock_sha256: str,
    expected_lock_pid: int,
    pid_snapshot_path: str | Path,
    monitor_path: str | Path,
    kernel_oom_evidence_path: str | Path,
    exit_site_path: str | Path,
    timing_path: str | Path,
    supervisor_log_path: str | Path,
    supervisor_script_path: str | Path,
    config_path: str | Path,
    command: str,
    _after_archive_link: Callable[[], None] | None = None,
) -> dict[str, object]:
    if not isinstance(expected_identity, Mapping):
        raise PostExitMigrationError("expected identity must be an object")
    identity, identity_legacy_root, new_root = evidence._writer_identity(
        expected_identity
    )
    legacy_root = _regular_directory(legacy_run_root, "legacy run root")
    if legacy_root != identity_legacy_root:
        raise PostExitMigrationError("legacy run root differs from expected identity")
    if type(expected_lock_pid) is not int or expected_lock_pid <= 0:
        raise PostExitMigrationError("expected lock PID must be a positive integer")
    lock_sha256 = _require_sha256(expected_lock_sha256, "expected lock SHA-256")
    if not isinstance(command, str) or not command.strip():
        raise PostExitMigrationError("post-exit runner command must be nonempty")

    canonical_lock = legacy_root.parent / f".{legacy_root.name}.csi-pairs-operation.lock"
    supplied_lock = _canonical_target(operation_lock_path, "canonical operation lock")
    if supplied_lock != canonical_lock:
        raise PostExitMigrationError("operation lock is not canonical for the legacy run")
    guard_path = canonical_lock.with_name(f"{canonical_lock.name}.guard")
    archive = _canonical_target(archive_path, "operation-lock archive")
    _require_external(archive, legacy_root, new_root, "operation-lock archive")

    raw_values = {
        "pid_snapshot": pid_snapshot_path,
        "monitor": monitor_path,
        "kernel_oom_evidence": kernel_oom_evidence_path,
        "exit_site": exit_site_path,
        "timing": timing_path,
        "supervisor_log": supervisor_log_path,
        "supervisor_script": supervisor_script_path,
        "config": config_path,
    }
    raw = {
        name: _regular_file(value, f"post-exit {name} evidence")
        for name, value in raw_values.items()
    }
    if len(set(raw.values())) != len(raw):
        raise PostExitMigrationError("post-exit raw evidence paths must be unique")
    raw_bindings = {
        name: evidence.bind_file(
            path,
            legacy_run_root=legacy_root,
            new_run_root=new_root,
            require_external=True,
        )
        for name, path in raw.items()
    }
    try:
        evidence._validate_post_exit_config_binding(raw_bindings["config"], identity)
    except RuntimeError as error:
        raise PostExitMigrationError(
            "post-exit config does not match the canonical migration identity"
        ) from error
    output = _prepare_output_directory(output_dir, legacy_root, new_root)
    if _inside(archive, output) or archive == output:
        raise PostExitMigrationError("operation-lock archive must be outside receipt output")
    expected_payload = _expected_lock_payload(legacy_root, expected_lock_pid)
    inventory_path = output / INVENTORY_NAME
    freeze_path = output / FREEZE_NAME

    with _exclusive_guard(guard_path) as guard:
        inventory = _load_or_write_inventory(
            inventory_path,
            identity=identity,
            command=command,
            legacy_root=legacy_root,
            new_root=new_root,
        )
        if inventory["results"]["files"] != []:
            raise PostExitMigrationError(
                "legacy evaluation inventory must be empty before lock archival"
            )
        archive_state, archive_record = _archive_stale_lock(
            canonical_lock,
            archive,
            expected_sha256=lock_sha256,
            expected_payload=expected_payload,
            expected_pid=expected_lock_pid,
            after_link=_after_archive_link,
        )
        _load_or_write_freeze(
            freeze_path,
            identity=identity,
            command=command,
            inventory_path=inventory_path,
            raw=raw,
            legacy_root=legacy_root,
            new_root=new_root,
        )
        current_bindings = {
            name: evidence.bind_file(
                path,
                legacy_run_root=legacy_root,
                new_run_root=new_root,
                require_external=True,
            )
            for name, path in raw.items()
        }
        if current_bindings != raw_bindings:
            raise PostExitMigrationError("post-exit raw evidence changed during execution")
        if _lexists(canonical_lock):
            raise PostExitMigrationError("canonical operation lock reappeared")
        final_archive = _read_file_record(archive, "completed operation-lock archive")
        if (
            final_archive.sha256 != archive_record.sha256
            or final_archive.inode != archive_record.inode
        ):
            raise PostExitMigrationError("operation-lock archive changed after receipt creation")

    inventory_payload = read_strict_json(inventory_path)
    freeze_payload = read_strict_json(freeze_path)
    if not isinstance(inventory_payload, dict) or not isinstance(freeze_payload, dict):
        raise PostExitMigrationError("post-exit output reports are malformed")
    evidence.validate_evidence_report(
        "legacy_evaluation_inventory",
        inventory_payload,
        expected_identity=identity,
        legacy_run_root=legacy_root,
        new_run_root=new_root,
    )
    evidence.validate_evidence_report(
        "legacy_evaluation_freeze",
        freeze_payload,
        expected_identity=identity,
        legacy_run_root=legacy_root,
        new_run_root=new_root,
    )
    return {
        "schema_version": RUNNER_SCHEMA,
        "status": "COMPLETE",
        "archive_state": archive_state,
        "canonical_lock_absent": not _lexists(canonical_lock),
        "archive": {
            "path": str(archive),
            "bytes": archive_record.size,
            "sha256": archive_record.sha256,
            "device": archive_record.device,
            "inode": archive_record.inode,
        },
        "guard": {
            "path": str(guard.path),
            "device": guard.device,
            "inode": guard.inode,
            "unchanged": True,
        },
        "inventory": {
            "path": str(inventory_path),
            "sha256": sha256_file(inventory_path),
        },
        "freeze_receipt": {
            "path": str(freeze_path),
            "sha256": sha256_file(freeze_path),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Archive one authenticated stale legacy operation lock and create "
            "the strict post-exit OOM migration freeze receipt."
        )
    )
    parser.add_argument("--identity", required=True)
    parser.add_argument("--legacy-run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--operation-lock", required=True)
    parser.add_argument("--archive-path", required=True)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--expected-lock-pid", required=True, type=int)
    parser.add_argument("--pid-snapshot", required=True)
    parser.add_argument("--monitor", required=True)
    parser.add_argument("--kernel-oom-evidence", required=True)
    parser.add_argument("--exit-site", required=True)
    parser.add_argument("--timing", required=True)
    parser.add_argument("--supervisor-log", required=True)
    parser.add_argument("--supervisor-script", required=True)
    parser.add_argument("--config", required=True)
    return parser


def _require_python_contract() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise PostExitMigrationError(
            "post-exit migration runner requires PYTHONDONTWRITEBYTECODE=1 and python -B"
        )


def main(argv: Sequence[str] | None = None) -> int:
    _require_python_contract()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    identity_path = _regular_file(args.identity, "expected identity")
    identity = read_strict_json(identity_path)
    if not isinstance(identity, dict):
        raise PostExitMigrationError("expected identity must be a JSON object")
    command = shlex.join(
        [
            sys.executable,
            "-B",
            "-m",
            "formal_v2.formal_migration_postexit_runner",
            *arguments,
        ]
    )
    result = run_post_exit_migration(
        expected_identity=identity,
        legacy_run_root=args.legacy_run_root,
        output_dir=args.output_dir,
        operation_lock_path=args.operation_lock,
        archive_path=args.archive_path,
        expected_lock_sha256=args.expected_lock_sha256,
        expected_lock_pid=args.expected_lock_pid,
        pid_snapshot_path=args.pid_snapshot,
        monitor_path=args.monitor,
        kernel_oom_evidence_path=args.kernel_oom_evidence,
        exit_site_path=args.exit_site,
        timing_path=args.timing,
        supervisor_log_path=args.supervisor_log,
        supervisor_script_path=args.supervisor_script,
        config_path=args.config,
        command=command,
    )
    print(json.dumps(result, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
