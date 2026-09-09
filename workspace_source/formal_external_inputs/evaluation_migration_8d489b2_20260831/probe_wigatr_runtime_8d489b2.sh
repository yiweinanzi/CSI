#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT=/root/xunlian/Futaoran/CSI_EVALUATION_RUNTIME_FINAL_20260831/code/CSI-PAIRS-v2.0-server
RUNTIME_PYTHON=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python
EVIDENCE_ROOT=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_8d489b2_20260831
REPORT="$EVIDENCE_ROOT/wigatr_runtime_preflight.json"
MODE=${1:-validate}

case "$MODE" in
  write|validate) ;;
  *)
    printf 'WIGATR_RUNTIME_REFUSAL=mode must be write or validate\n' >&2
    exit 64
    ;;
esac

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0,1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "$RUNTIME_ROOT"
exec "$RUNTIME_PYTHON" -B - "$MODE" "$REPORT" <<'PY'
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from formal_v2.formal_external_runtime import validate_external_runtime
from formal_v2.formal_io import read_strict_json, sha256_file
from formal_v2.formal_migration import _source_identity
from formal_v2.formal_migration_evidence_runner import write_json_exclusive_atomic

SCHEMA = "csi-pairs-v6-migration-wigatr-runtime-preflight-v1"
EXPECTED_COMMIT = "8d489b2387e7bb6c988a41d9e0d57b8a6cffc4d4"
EXPECTED_SOURCE_SHA256 = "aa5b1d6a1062d68bb5de045b40f042e1a453d14a8d73632ad0be86dc7b07caff"
RUNTIME_ROOT = Path(
    "/root/xunlian/Futaoran/CSI_EVALUATION_RUNTIME_FINAL_20260831/code/CSI-PAIRS-v2.0-server"
)
SOURCE_ROOT = RUNTIME_ROOT / "formal_v2"
LINK = SOURCE_ROOT / "external_adapters/.venv-wigatr"
EXPECTED_TARGET = Path(
    "/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/formal_v2/external_adapters/.venv-wigatr"
)
EXECUTABLE = LINK / "bin/python"
LEGACY_REQUEST = Path(
    "/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server/runs/formal-v6-gpu-9850fff-20260825T071800Z/approval/request.json"
)
PATH_FIELDS = {"python_executable", "python_prefix", "environment_sha256"}
REPORT_FIELDS = {
    "schema_version",
    "status",
    "created_utc",
    "new_commit",
    "new_source_tree_sha256",
    "runtime_link",
    "legacy_approval_request",
    "legacy_runtime_environment_sha256",
    "current_runtime_environment_sha256",
    "path_only_rebinding",
    "invariant_fields",
    "legacy_python_executable_realpath",
    "current_python_executable_realpath",
    "legacy_python_prefix_realpath",
    "current_python_prefix_realpath",
    "runtime_provenance",
    "runtime_file_closure",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def require_static_identity() -> tuple[dict, dict]:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise RuntimeError("Wi-GATr preflight requires PYTHONDONTWRITEBYTECODE=1 and -B")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("Wi-GATr preflight requires CUDA_VISIBLE_DEVICES=0,1")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Wi-GATr preflight requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if not LINK.is_symlink() or Path(os.readlink(LINK)) != EXPECTED_TARGET:
        raise RuntimeError("Wi-GATr runtime link target is not the reviewed runtime")
    if not EXPECTED_TARGET.is_dir() or not EXECUTABLE.is_file() or not os.access(EXECUTABLE, os.X_OK):
        raise RuntimeError("Wi-GATr reviewed runtime is unavailable")
    source = _source_identity(SOURCE_ROOT)
    if (
        source["git_commit"] != EXPECTED_COMMIT
        or source["source_tree_sha256"] != EXPECTED_SOURCE_SHA256
    ):
        raise RuntimeError("Wi-GATr preflight source identity mismatch")
    legacy_request = read_strict_json(LEGACY_REQUEST)
    try:
        legacy_runtime = legacy_request["external_runtime_provenance"]["wigatr"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("legacy approval has no authenticated Wi-GATr runtime") from error
    validate_external_runtime(
        legacy_runtime,
        profile="wigatr",
        executable=legacy_runtime["python_executable"],
        require_execution_ready=True,
    )
    return source, legacy_runtime


def probe_current() -> dict:
    completed = subprocess.run(
        [
            str(EXECUTABLE),
            "-B",
            "-m",
            "formal_v2.formal_external_runtime",
            "--profile",
            "wigatr",
            "--project-root",
            str(RUNTIME_ROOT),
            "--require-execution-ready",
        ],
        cwd=RUNTIME_ROOT,
        env=dict(os.environ),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Wi-GATr runtime probe failed: " + completed.stderr.strip()
        )
    try:
        current = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Wi-GATr runtime probe emitted invalid JSON") from error
    return validate_external_runtime(
        current,
        profile="wigatr",
        executable=EXECUTABLE,
        require_execution_ready=True,
    )


def probe_file_closure() -> dict:
    program = r"""
import base64
import hashlib
import importlib.metadata
import json
import sys
import sysconfig
from pathlib import Path


def digest_file(path, algorithms):
    hashers = {name: hashlib.new(name) for name in algorithms}
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            for hasher in hashers.values():
                hasher.update(chunk)
    return {name: value.digest() for name, value in hashers.items()}


prefix = Path(sys.prefix).resolve()
roots = []
for kind in ("purelib", "platlib"):
    root = Path(sysconfig.get_path(kind)).resolve()
    if root != prefix and prefix not in root.parents:
        raise RuntimeError("distribution root escapes Wi-GATr prefix")
    if str(root) not in roots:
        roots.append(str(root))

rows = []
record_digests = {}
verified_record_hashes = 0
missing_record_hashes = 0
seen = {}
for distribution in importlib.metadata.distributions(path=roots):
    name = distribution.metadata.get("Name")
    if not isinstance(name, str) or not name.strip():
        continue
    key = name.lower().replace("_", "-")
    record = distribution.read_text("RECORD")
    record_sha256 = (
        hashlib.sha256(record.encode("utf-8")).hexdigest()
        if record is not None
        else None
    )
    if key in record_digests and record_digests[key] != record_sha256:
        raise RuntimeError("duplicate distribution RECORD differs")
    record_digests[key] = record_sha256
    files = distribution.files
    if files is None:
        continue
    for package_path in files:
        located = Path(distribution.locate_file(package_path))
        resolved = located.resolve(strict=True)
        if resolved != prefix and prefix not in resolved.parents:
            raise RuntimeError("distribution file escapes Wi-GATr prefix")
        if not resolved.is_file():
            raise RuntimeError("distribution RECORD entry is not a regular file")
        record_algorithm = package_path.hash.mode if package_path.hash else None
        algorithms = {"sha256"}
        if record_algorithm is not None:
            algorithms.add(record_algorithm)
        digests = digest_file(resolved, algorithms)
        size = resolved.stat().st_size
        if package_path.size is not None and size != package_path.size:
            raise RuntimeError("distribution RECORD size mismatch")
        if package_path.hash is not None:
            encoded = base64.urlsafe_b64encode(
                digests[record_algorithm]
            ).rstrip(b"=").decode("ascii")
            if encoded != package_path.hash.value:
                raise RuntimeError("distribution RECORD content hash mismatch")
            verified_record_hashes += 1
        else:
            missing_record_hashes += 1
        relative = resolved.relative_to(prefix).as_posix()
        sha256 = digests["sha256"].hex()
        prior = seen.get(relative)
        if prior is not None and prior != (size, sha256):
            raise RuntimeError("duplicate distribution file differs")
        seen[relative] = (size, sha256)
        rows.append(
            {
                "distribution": key,
                "path": relative,
                "bytes": size,
                "sha256": sha256,
                "record_hash_present": package_path.hash is not None,
            }
        )

rows.sort(
    key=lambda row: (
        row["distribution"],
        row["path"],
        row["bytes"],
        row["sha256"],
        row["record_hash_present"],
    )
)
payload = json.dumps(
    rows,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
).encode("ascii")
print(
    json.dumps(
        {
            "schema_version": "csi-pairs-v6-runtime-file-closure-v1",
            "status": "PASS",
            "distribution_count": len(record_digests),
            "file_entry_count": len(rows),
            "unique_file_count": len(seen),
            "total_unique_file_bytes": sum(value[0] for value in seen.values()),
            "verified_record_hash_count": verified_record_hashes,
            "missing_record_hash_count": missing_record_hashes,
            "record_sha256s": {
                key: record_digests[key] for key in sorted(record_digests)
            },
            "closure_sha256": hashlib.sha256(payload).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
)
"""
    completed = subprocess.run(
        [str(EXECUTABLE), "-B", "-c", program],
        cwd=RUNTIME_ROOT,
        env=dict(os.environ),
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Wi-GATr runtime file-closure probe failed: " + completed.stderr.strip()
        )
    try:
        closure = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Wi-GATr runtime file-closure probe emitted invalid JSON") from error
    required = {
        "schema_version",
        "status",
        "distribution_count",
        "file_entry_count",
        "unique_file_count",
        "total_unique_file_bytes",
        "verified_record_hash_count",
        "missing_record_hash_count",
        "record_sha256s",
        "closure_sha256",
    }
    if not isinstance(closure, dict) or set(closure) != required:
        raise RuntimeError("Wi-GATr runtime file-closure fields are invalid")
    numeric = (
        "distribution_count",
        "file_entry_count",
        "unique_file_count",
        "total_unique_file_bytes",
        "verified_record_hash_count",
        "missing_record_hash_count",
    )
    if (
        closure["schema_version"] != "csi-pairs-v6-runtime-file-closure-v1"
        or closure["status"] != "PASS"
        or any(type(closure[field]) is not int or closure[field] < 0 for field in numeric)
        or closure["distribution_count"] <= 0
        or closure["file_entry_count"] <= 0
        or closure["unique_file_count"] <= 0
        or closure["total_unique_file_bytes"] <= 0
        or closure["verified_record_hash_count"] <= 0
        or closure["file_entry_count"]
        != closure["verified_record_hash_count"] + closure["missing_record_hash_count"]
        or not isinstance(closure["record_sha256s"], dict)
        or not isinstance(closure["closure_sha256"], str)
        or len(closure["closure_sha256"]) != 64
    ):
        raise RuntimeError("Wi-GATr runtime file-closure result is invalid")
    return closure


def compare_runtime(legacy: dict, current: dict) -> list[str]:
    if set(legacy) != set(current):
        raise RuntimeError("Wi-GATr runtime field set changed")
    invariant_fields = sorted(set(legacy) - PATH_FIELDS)
    for field in invariant_fields:
        if legacy[field] != current[field]:
            raise RuntimeError(f"Wi-GATr runtime invariant changed: {field}")
    if os.path.realpath(legacy["python_executable"]) != os.path.realpath(
        current["python_executable"]
    ):
        raise RuntimeError("Wi-GATr interpreter realpath changed")
    if os.path.realpath(legacy["python_prefix"]) != os.path.realpath(
        current["python_prefix"]
    ):
        raise RuntimeError("Wi-GATr environment realpath changed")
    return invariant_fields


def build_report(source: dict, legacy: dict, current: dict, closure: dict) -> dict:
    invariant_fields = compare_runtime(legacy, current)
    expected_records = {
        key: value["record_sha256"]
        for key, value in current["installed_distributions"].items()
    }
    if closure["record_sha256s"] != expected_records:
        raise RuntimeError("Wi-GATr file closure does not bind every runtime RECORD")
    return {
        "schema_version": SCHEMA,
        "status": "PASS",
        "created_utc": utc_now(),
        "new_commit": source["git_commit"],
        "new_source_tree_sha256": source["source_tree_sha256"],
        "runtime_link": {
            "path": str(LINK),
            "target": os.readlink(LINK),
            "resolved_target": str(LINK.resolve()),
        },
        "legacy_approval_request": {
            "path": str(LEGACY_REQUEST),
            "sha256": sha256_file(LEGACY_REQUEST),
        },
        "legacy_runtime_environment_sha256": legacy["environment_sha256"],
        "current_runtime_environment_sha256": current["environment_sha256"],
        "path_only_rebinding": True,
        "invariant_fields": invariant_fields,
        "legacy_python_executable_realpath": os.path.realpath(
            legacy["python_executable"]
        ),
        "current_python_executable_realpath": os.path.realpath(
            current["python_executable"]
        ),
        "legacy_python_prefix_realpath": os.path.realpath(legacy["python_prefix"]),
        "current_python_prefix_realpath": os.path.realpath(current["python_prefix"]),
        "runtime_provenance": current,
        "runtime_file_closure": closure,
    }


def validate_report(
    report: object, source: dict, legacy: dict, current: dict, closure: dict
) -> None:
    if not isinstance(report, dict) or set(report) != REPORT_FIELDS:
        raise RuntimeError("Wi-GATr preflight report fields are invalid")
    expected = build_report(source, legacy, current, closure)
    for field in REPORT_FIELDS - {"created_utc"}:
        if report[field] != expected[field]:
            raise RuntimeError(f"Wi-GATr preflight report mismatch: {field}")
    created = report["created_utc"]
    if not isinstance(created, str) or not created.endswith("Z"):
        raise RuntimeError("Wi-GATr preflight timestamp is invalid")


mode = sys.argv[1]
report_path = Path(sys.argv[2])
source_identity, legacy_provenance = require_static_identity()
current_provenance = probe_current()
current_file_closure = probe_file_closure()
if mode == "write" and not report_path.exists() and not report_path.is_symlink():
    report = build_report(
        source_identity,
        legacy_provenance,
        current_provenance,
        current_file_closure,
    )
    write_json_exclusive_atomic(report_path, report)
else:
    if report_path.is_symlink() or not report_path.is_file():
        raise RuntimeError("Wi-GATr preflight report is missing or unsafe")
    report = read_strict_json(report_path)
    validate_report(
        report,
        source_identity,
        legacy_provenance,
        current_provenance,
        current_file_closure,
    )
print("WIGATR_RUNTIME_PREFLIGHT=PASS")
print("WIGATR_RUNTIME_PREFLIGHT_SHA256=" + sha256_file(report_path))
PY
