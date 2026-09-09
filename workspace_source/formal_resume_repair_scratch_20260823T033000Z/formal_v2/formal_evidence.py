from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import threading
import re
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from .formal_config import public_formal_config
from .formal_dataset import FormalDataset
from .formal_io import sha256_file
from .formal_runtime_integrity import (
    WHEELHOUSE_NAME,
    WHEEL_MANIFEST_NAME,
    validate_installed_wheel_closure,
)


QUALIFICATION_SCHEMA = "csi-pairs-formal-qualification-gate-v4-v6"
FACTORIAL_SCHEMA = "csi-pairs-formal-factorial-gate-v2.1-v6"
GATE_IDS = tuple(f"G{index}" for index in range(9))
CLAIM_IDS = tuple(f"C{index}" for index in range(1, 14))
ASSESSMENT_STATES = {"PASS", "FAIL", "BLOCKED", "NOT_ASSESSED"}
RUNTIME_PROVENANCE_SCHEMA = "csi-pairs-runtime-provenance-v3"
FIXTURE_RUNTIME_CACHE_ENV = "CSI_PAIRS_FIXTURE_RUNTIME_CACHE"
_FIXTURE_RUNTIME_CACHE: dict[str, object] | None = None
_FIXTURE_RUNTIME_CACHE_LOCK = threading.Lock()
RUNTIME_PROVENANCE_FIELDS = {
    "schema_version",
    "source_tree_sha256",
    "requirements_lock_sha256",
    "installer_report_path",
    "installer_report_sha256",
    "reviewed_wheelhouse_path",
    "reviewed_wheelhouse_sha256",
    "reviewed_wheel_manifest_path",
    "reviewed_wheel_manifest_sha256",
    "python_version",
    "python_implementation",
    "python_executable",
    "python_executable_name",
    "python_prefix",
    "python_dont_write_bytecode",
    "platform_system",
    "platform_release",
    "platform_machine",
    "platform_mac_version",
    "platform_libc_name",
    "platform_libc_version",
    "cublas_workspace_config",
    "torch",
    "installed_distributions",
}
TORCH_RUNTIME_FIELDS = {
    "version",
    "cuda_version",
    "cuda_available",
    "gpu_names",
    "deterministic_algorithms",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "cudnn_allow_tf32",
    "cuda_matmul_allow_tf32",
    "float32_matmul_precision",
}
LINUX_X86_64_LOCK_MARKER = (
    'sys_platform == "linux" and platform_machine == "x86_64"'
)
SUPPORTED_LOCK_TARGETS = {("darwin", "arm64"), ("linux", "x86_64")}
TARGET_REQUIREMENTS_LOCKS = {
    ("darwin", "arm64"): "requirements-lock.txt",
    ("linux", "x86_64"): "requirements-lock-linux-x86_64-cu121.txt",
}
INSTALL_REPORT_NAME = "csi-pairs-install-report.json"
_LOCK_ENTRY = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"==(?P<version>[A-Za-z0-9][A-Za-z0-9.!+_-]*)"
    r"(?:\s*;\s*(?P<marker>[^;]+?))?"
    r"(?P<hashes>(?:\s+--hash=sha256:[0-9a-f]{64})+)$"
)
EVIDENCE_AUTH_KEYS = (
    "artifact_label",
    "dataset_sha256",
    "config_sha256",
    "fixture",
    "source_tree_sha256",
    "requirements_lock_sha256",
    "runtime_provenance_sha256",
    "runtime_provenance",
)


def config_sha256(config: dict) -> str:
    payload = json.dumps(
        public_formal_config(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def configure_reproducible_runtime() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    require_source_tree_without_bytecode()
    try:
        import torch
    except ImportError:
        return
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def require_source_tree_without_bytecode(root: str | Path | None = None) -> None:
    source_root = Path(root).resolve() if root is not None else Path(__file__).resolve().parent
    forbidden = sorted(
        path
        for path in source_root.rglob("*.pyc")
        if path.is_file()
        and not any(
            part.startswith(".venv-") or part.startswith(".runtime-")
            for part in path.relative_to(source_root).parts
        )
    )
    if forbidden:
        relative = [path.relative_to(source_root).as_posix() for path in forbidden[:10]]
        raise RuntimeError(
            "project source tree contains forbidden bytecode; run only from a clean "
            f"source checkout with python -B: {relative}"
        )


def _logical_lock_lines(requirements: Path) -> list[str]:
    logical: list[str] = []
    pending: list[str] = []
    for raw_line in requirements.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            if pending:
                raise RuntimeError("formal requirements lock has an interrupted entry")
            continue
        continued = line.endswith("\\")
        fragment = line[:-1].strip() if continued else line
        if not fragment:
            raise RuntimeError("formal requirements lock has an empty continuation")
        pending.append(fragment)
        if not continued:
            logical.append(" ".join(pending))
            pending = []
    if pending:
        raise RuntimeError("formal requirements lock has an unterminated entry")
    return logical


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _locked_requirement_records(
    requirements: Path,
    *,
    sys_platform_value: str | None = None,
    machine_value: str | None = None,
) -> dict[str, dict[str, object]]:
    active_platform = sys.platform if sys_platform_value is None else sys_platform_value
    active_machine = platform.machine() if machine_value is None else machine_value
    if (active_platform, active_machine) not in SUPPORTED_LOCK_TARGETS:
        raise RuntimeError(
            "formal requirements lock supports only macOS arm64 and Linux x86_64; "
            f"observed {active_platform} {active_machine}"
        )
    expected: dict[str, dict[str, object]] = {}
    seen: set[str] = set()
    for line in _logical_lock_lines(requirements):
        match = _LOCK_ENTRY.fullmatch(line)
        if match is None:
            raise RuntimeError(f"formal requirements lock entry is not exact and hashed: {line!r}")
        name = match.group("name")
        normalized_name = _normalize_distribution_name(name)
        if normalized_name in seen:
            raise RuntimeError(f"formal requirements lock has duplicate package: {name}")
        seen.add(normalized_name)
        hashes = re.findall(r"--hash=sha256:([0-9a-f]{64})", match.group("hashes"))
        if not hashes or len(hashes) != len(set(hashes)):
            raise RuntimeError(f"formal requirements lock hashes are invalid: {name}")
        marker = match.group("marker")
        if marker is not None and marker != LINUX_X86_64_LOCK_MARKER:
            raise RuntimeError(f"formal requirements lock marker is unsupported: {marker!r}")
        if marker is not None and not (
            active_platform == "linux" and active_machine == "x86_64"
        ):
            continue
        expected[name] = {
            "version": match.group("version"),
            "hashes": tuple(hashes),
        }
    if not expected:
        raise RuntimeError("formal requirements lock has no active packages")
    return expected


def _locked_requirement_versions(
    requirements: Path,
    *,
    sys_platform_value: str | None = None,
    machine_value: str | None = None,
) -> dict[str, str]:
    records = _locked_requirement_records(
        requirements,
        sys_platform_value=sys_platform_value,
        machine_value=machine_value,
    )
    return {name: str(record["version"]) for name, record in records.items()}


def _requirements_lock_path(
    *,
    sys_platform_value: str | None = None,
    machine_value: str | None = None,
) -> Path:
    active_platform = sys.platform if sys_platform_value is None else sys_platform_value
    active_machine = platform.machine() if machine_value is None else machine_value
    try:
        filename = TARGET_REQUIREMENTS_LOCKS[(active_platform, active_machine)]
    except KeyError as error:
        raise RuntimeError(
            "formal requirements lock supports only macOS arm64 and Linux x86_64; "
            f"observed {active_platform} {active_machine}"
        ) from error
    return Path(__file__).resolve().parent / filename


def _torch_module_version_matches_lock(module_version: str, locked_version: str) -> bool:
    if "+" in locked_version:
        return module_version == locked_version
    return module_version.split("+", 1)[0] == locked_version


def _wheel_receipt_from_install_report(
    report_path: Path,
    expected: dict[str, dict[str, object]],
) -> dict[str, str]:
    if not report_path.is_file() or report_path.is_symlink():
        raise RuntimeError("main runtime hashed installer report is missing")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("main runtime hashed installer report is invalid") from error
    if (
        not isinstance(report, dict)
        or set(report) != {"version", "pip_version", "install", "environment"}
        or report.get("version") != "1"
        or not isinstance(report.get("pip_version"), str)
        or not isinstance(report.get("install"), list)
        or not isinstance(report.get("environment"), dict)
    ):
        raise RuntimeError("main runtime hashed installer report schema mismatch")

    expected_by_normalized = {
        _normalize_distribution_name(name): (name, record)
        for name, record in expected.items()
    }
    receipt: dict[str, str] = {}
    for item in report["install"]:
        if not isinstance(item, dict):
            raise RuntimeError("main runtime installer report entry is invalid")
        metadata = item.get("metadata")
        download = item.get("download_info")
        archive = download.get("archive_info") if isinstance(download, dict) else None
        hashes = archive.get("hashes") if isinstance(archive, dict) else None
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("name"), str)
            or not isinstance(metadata.get("version"), str)
            or not isinstance(download, dict)
            or not isinstance(download.get("url"), str)
            or not isinstance(hashes, dict)
            or set(hashes) != {"sha256"}
            or not isinstance(hashes.get("sha256"), str)
            or item.get("is_direct") is not False
            or item.get("requested") is not True
        ):
            raise RuntimeError("main runtime installer report wheel binding is invalid")
        normalized = _normalize_distribution_name(metadata["name"])
        if normalized not in expected_by_normalized or normalized in receipt:
            raise RuntimeError(
                "main runtime installer report package inventory is invalid"
            )
        lock_name, locked = expected_by_normalized[normalized]
        wheel_hash = hashes["sha256"]
        parsed_url = urlparse(download["url"])
        if (
            metadata["version"] != locked["version"]
            or wheel_hash not in locked["hashes"]
            or not parsed_url.path.lower().endswith(".whl")
            or parsed_url.username is not None
            or parsed_url.password is not None
        ):
            raise RuntimeError(
                f"main runtime installer report does not match the wheel lock: {lock_name}"
            )
        receipt[normalized] = wheel_hash
    if set(receipt) != set(expected_by_normalized):
        raise RuntimeError("main runtime installer report package inventory is incomplete")
    return receipt


def _validated_distribution_record(
    distribution: importlib.metadata.Distribution,
    prefix: Path,
) -> str:
    record = distribution.read_text("RECORD")
    if record is None:
        raise RuntimeError(
            f"main runtime distribution {distribution.metadata['Name']} has no RECORD"
        )
    seen: set[str] = set()
    unhashed_record: list[str] = []
    try:
        rows = list(csv.reader(record.splitlines()))
    except csv.Error as error:
        raise RuntimeError("main runtime distribution RECORD is malformed") from error
    if not rows:
        raise RuntimeError("main runtime distribution RECORD is empty")
    for row in rows:
        if len(row) != 3 or not row[0] or row[0] in seen:
            raise RuntimeError("main runtime distribution RECORD row is invalid")
        relative, encoded_hash, encoded_size = row
        seen.add(relative)
        candidate = Path(distribution.locate_file(relative))
        resolved = candidate.resolve()
        if resolved != prefix and prefix not in resolved.parents:
            raise RuntimeError(
                f"main runtime distribution RECORD path is unsafe: {relative}"
            )
        if not encoded_hash and not encoded_size:
            parts = Path(relative).parts
            if relative.endswith(".dist-info/RECORD"):
                if not candidate.is_file() or candidate.is_symlink():
                    raise RuntimeError(
                        "main runtime distribution RECORD file is unsafe or missing"
                    )
                unhashed_record.append(relative)
            elif relative.endswith(".pyc") and "__pycache__" in parts:
                if candidate.exists() and (
                    not candidate.is_file() or candidate.is_symlink()
                ):
                    raise RuntimeError(
                        f"main runtime generated bytecode path is unsafe: {relative}"
                    )
            else:
                raise RuntimeError(
                    f"main runtime distribution has unexpected unhashed row: {relative}"
                )
            continue
        if not candidate.is_file() or candidate.is_symlink():
            raise RuntimeError(
                f"main runtime distribution RECORD path is missing: {relative}"
            )
        if not encoded_hash.startswith("sha256=") or not encoded_size.isdigit():
            raise RuntimeError(
                f"main runtime distribution RECORD hash is invalid: {relative}"
            )
        expected_hash = encoded_hash.split("=", 1)[1]
        actual_hash = base64.urlsafe_b64encode(
            bytes.fromhex(sha256_file(resolved))
        ).rstrip(b"=").decode("ascii")
        if actual_hash != expected_hash or resolved.stat().st_size != int(encoded_size):
            raise RuntimeError(
                f"main runtime distribution file differs from RECORD: {relative}"
            )
    if len(unhashed_record) != 1:
        raise RuntimeError("main runtime distribution has unexpected unhashed RECORD rows")
    return hashlib.sha256(record.encode("utf-8")).hexdigest()


def runtime_provenance() -> dict[str, object]:
    requirements = _requirements_lock_path()
    expected = _locked_requirement_records(requirements)
    prefix = Path(sys.prefix).resolve()
    report_path = prefix / INSTALL_REPORT_NAME
    wheel_receipt = _wheel_receipt_from_install_report(report_path, expected)
    wheelhouse = prefix / WHEELHOUSE_NAME
    wheel_manifest_path = prefix / WHEEL_MANIFEST_NAME
    requirements_digest = sha256_file(requirements)
    closure = validate_installed_wheel_closure(
        prefix,
        wheelhouse,
        wheel_manifest_path,
        expected,
        requirements_digest,
    )
    reviewed_wheels = {
        str(record["normalized_name"]): record
        for record in closure["manifest"]["wheels"]
    }
    distributions = {}
    if requirements.is_file():
        for name in expected:
            normalized = _normalize_distribution_name(name)
            try:
                distribution = importlib.metadata.distribution(name)
                distributions[name] = {
                    "version": distribution.version,
                    "record_sha256": closure["record_digests"][normalized],
                    "wheel_sha256": reviewed_wheels[normalized]["sha256"],
                }
            except importlib.metadata.PackageNotFoundError:
                distributions[name] = {
                    "version": None,
                    "record_sha256": None,
                    "wheel_sha256": wheel_receipt[normalized],
                }
    torch_record = {
        "version": None,
        "cuda_version": None,
        "cuda_available": False,
        "gpu_names": [],
        "deterministic_algorithms": False,
        "cudnn_benchmark": None,
        "cudnn_deterministic": None,
        "cudnn_allow_tf32": None,
        "cuda_matmul_allow_tf32": None,
        "float32_matmul_precision": None,
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_record.update(
            {
                "version": str(torch.__version__),
                "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
                "cuda_available": cuda_available,
                "gpu_names": (
                    [
                        str(torch.cuda.get_device_name(index))
                        for index in range(torch.cuda.device_count())
                    ]
                    if cuda_available
                    else []
                ),
                "deterministic_algorithms": bool(
                    torch.are_deterministic_algorithms_enabled()
                ),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "cuda_matmul_allow_tf32": bool(
                    torch.backends.cuda.matmul.allow_tf32
                ),
                "float32_matmul_precision": str(
                    torch.get_float32_matmul_precision()
                ),
            }
        )
    except ImportError:
        pass
    libc_name, libc_version = platform.libc_ver()
    return {
        "schema_version": RUNTIME_PROVENANCE_SCHEMA,
        "source_tree_sha256": _source_tree_sha256(),
        "requirements_lock_sha256": requirements_digest,
        "installer_report_path": str(report_path),
        "installer_report_sha256": sha256_file(report_path),
        "reviewed_wheelhouse_path": str(wheelhouse),
        "reviewed_wheelhouse_sha256": closure["wheelhouse_sha256"],
        "reviewed_wheel_manifest_path": str(wheel_manifest_path),
        "reviewed_wheel_manifest_sha256": closure["manifest_sha256"],
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_name": Path(sys.executable).resolve().name,
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_dont_write_bytecode": bool(sys.dont_write_bytecode),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "platform_mac_version": platform.mac_ver()[0] or None,
        "platform_libc_name": libc_name or None,
        "platform_libc_version": libc_version or None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch": torch_record,
        "installed_distributions": distributions,
    }


def validate_runtime_provenance(runtime: object) -> dict[str, object]:
    """Require the evidence-producing interpreter to match the checked-in lock."""
    if not isinstance(runtime, dict) or set(runtime) != RUNTIME_PROVENANCE_FIELDS:
        raise RuntimeError("main runtime provenance fields must be exact")
    if runtime.get("schema_version") != RUNTIME_PROVENANCE_SCHEMA:
        raise RuntimeError("main runtime provenance schema mismatch")
    if runtime.get("python_implementation") != "CPython" or not str(
        runtime.get("python_version", "")
    ).startswith("3.12."):
        raise RuntimeError("formal evidence requires CPython 3.12")
    executable = runtime.get("python_executable")
    prefix = runtime.get("python_prefix")
    if (
        not isinstance(executable, str)
        or not Path(executable).is_absolute()
        or Path(executable).name != runtime.get("python_executable_name")
        or not isinstance(prefix, str)
        or not Path(prefix).is_absolute()
        or runtime.get("python_dont_write_bytecode") is not True
    ):
        raise RuntimeError("main runtime interpreter provenance is invalid")

    lock_platform, lock_machine = _validated_runtime_platform(runtime)

    requirements = _requirements_lock_path(
        sys_platform_value=lock_platform,
        machine_value=lock_machine,
    )
    if not requirements.is_file() or requirements.is_symlink():
        raise RuntimeError("formal requirements lock is missing or not a regular file")
    if runtime.get("requirements_lock_sha256") != sha256_file(requirements):
        raise RuntimeError("main runtime requirements-lock digest mismatch")
    if runtime.get("source_tree_sha256") != _source_tree_sha256():
        raise RuntimeError("main runtime source-tree digest mismatch")

    expected_records = _locked_requirement_records(
        requirements,
        sys_platform_value=lock_platform,
        machine_value=lock_machine,
    )
    expected = {
        name: str(record["version"])
        for name, record in expected_records.items()
    }

    report_path_value = runtime.get("installer_report_path")
    prefix_path = Path(str(runtime["python_prefix"])).resolve()
    if not isinstance(report_path_value, str):
        raise RuntimeError("main runtime installer report path is invalid")
    report_path = Path(report_path_value)
    if (
        not report_path.is_absolute()
        or report_path.resolve() != prefix_path / INSTALL_REPORT_NAME
        or not report_path.is_file()
        or report_path.is_symlink()
        or runtime.get("installer_report_sha256") != sha256_file(report_path)
    ):
        raise RuntimeError("main runtime installer report provenance mismatch")
    wheel_receipt = _wheel_receipt_from_install_report(report_path, expected_records)

    wheelhouse_path_value = runtime.get("reviewed_wheelhouse_path")
    manifest_path_value = runtime.get("reviewed_wheel_manifest_path")
    wheelhouse_path = prefix_path / WHEELHOUSE_NAME
    manifest_path = prefix_path / WHEEL_MANIFEST_NAME
    if (
        not isinstance(wheelhouse_path_value, str)
        or Path(wheelhouse_path_value).resolve() != wheelhouse_path
        or not wheelhouse_path.is_dir()
        or wheelhouse_path.is_symlink()
        or not isinstance(manifest_path_value, str)
        or Path(manifest_path_value).resolve() != manifest_path
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
        or runtime.get("reviewed_wheel_manifest_sha256") != sha256_file(manifest_path)
        or re.fullmatch(
            r"[0-9a-f]{64}", str(runtime.get("reviewed_wheelhouse_sha256", ""))
        )
        is None
    ):
        raise RuntimeError("main runtime reviewed-wheel provenance mismatch")

    installed = runtime.get("installed_distributions")
    if not isinstance(installed, dict) or set(installed) != set(expected):
        missing = sorted(set(expected).difference(installed if isinstance(installed, dict) else {}))
        unexpected = sorted(
            set(installed if isinstance(installed, dict) else {}).difference(expected)
        )
        raise RuntimeError(
            "main runtime locked-distribution inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, version in expected.items():
        record = installed[name]
        if not isinstance(record, dict) or set(record) != {
            "version",
            "record_sha256",
            "wheel_sha256",
        }:
            raise RuntimeError(f"main runtime distribution provenance is invalid: {name}")
        if record["version"] != version:
            raise RuntimeError(
                f"main runtime distribution {name} must be exactly {version}, "
                f"observed {record['version']}"
            )
        digest = record["record_sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(f"main runtime distribution {name} has no RECORD provenance")
        wheel_hash = record["wheel_sha256"]
        if (
            wheel_hash not in expected_records[name]["hashes"]
            or wheel_receipt[_normalize_distribution_name(name)] != wheel_hash
        ):
            raise RuntimeError(
                f"main runtime distribution {name} wheel hash is not authenticated"
            )
    if runtime.get("cublas_workspace_config") != ":4096:8":
        raise RuntimeError("main runtime CUBLAS deterministic workspace is not configured")
    torch_record = runtime.get("torch")
    if not isinstance(torch_record, dict) or set(torch_record) != TORCH_RUNTIME_FIELDS:
        raise RuntimeError("main runtime torch provenance fields must be exact")
    torch_version = torch_record.get("version")
    if (
        not isinstance(torch_version, str)
        or not _torch_module_version_matches_lock(torch_version, expected["torch"])
    ):
        raise RuntimeError("main runtime torch module version does not match the lock")
    if (
        torch_record.get("deterministic_algorithms") is not True
        or torch_record.get("cudnn_benchmark") is not False
        or torch_record.get("cudnn_deterministic") is not True
        or torch_record.get("cudnn_allow_tf32") is not False
        or torch_record.get("cuda_matmul_allow_tf32") is not False
        or torch_record.get("float32_matmul_precision") != "highest"
    ):
        raise RuntimeError("main runtime deterministic torch state is not configured")
    if type(torch_record.get("cuda_available")) is not bool or not isinstance(
        torch_record.get("gpu_names"), list
    ) or not all(isinstance(value, str) for value in torch_record["gpu_names"]):
        raise RuntimeError("main runtime CUDA inventory is invalid")
    return runtime


def _version_at_least(value: object, minimum: tuple[int, int], label: str) -> None:
    if not isinstance(value, str):
        raise RuntimeError(f"formal evidence has no valid {label} version")
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)*", value)
    if match is None:
        raise RuntimeError(f"formal evidence has no valid {label} version")
    observed = (int(match.group(1)), int(match.group(2)))
    if observed < minimum:
        raise RuntimeError(
            f"formal evidence requires {label} {minimum[0]}.{minimum[1]} or newer; "
            f"observed {value}"
        )


def _validated_runtime_platform(runtime: dict[str, object]) -> tuple[str, str]:
    system = runtime.get("platform_system")
    machine = runtime.get("platform_machine")
    if system == "Darwin" and machine == "arm64":
        _version_at_least(runtime.get("platform_mac_version"), (14, 0), "macOS")
        return "darwin", "arm64"
    if system == "Linux" and machine == "x86_64":
        if runtime.get("platform_libc_name") != "glibc":
            raise RuntimeError("formal evidence requires glibc on Linux x86_64")
        _version_at_least(runtime.get("platform_libc_version"), (2, 28), "glibc")
        return "linux", "x86_64"
    raise RuntimeError(
        "formal evidence supports only macOS 14+ arm64 and glibc 2.28+ Linux "
        f"x86_64; observed {system} {machine}"
    )


def _source_tree_sha256() -> str:
    root = Path(__file__).resolve().parent
    included_suffixes = {
        ".py", ".json", ".sh", ".txt", ".toml", ".lock", ".yaml", ".yml"
    }
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in included_suffixes
        and not any(
            part.startswith(".venv-") or part.startswith(".runtime-")
            for part in path.relative_to(root).parts
        )
        and "__pycache__" not in path.parts
    )
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def evidence_context(config: dict, dataset: FormalDataset, scientific_use: str) -> dict[str, object]:
    global _FIXTURE_RUNTIME_CACHE

    ceiling = "FORBIDDEN" if dataset.is_fixture else scientific_use
    configure_reproducible_runtime()
    cache_fixture_runtime = (
        dataset.is_fixture and os.environ.get(FIXTURE_RUNTIME_CACHE_ENV) == "1"
    )
    if cache_fixture_runtime:
        # A fixture can never become paper evidence. The first call still performs
        # the full wheel/RECORD closure; later calls in the same smoke process reuse
        # only that validated immutable payload.
        with _FIXTURE_RUNTIME_CACHE_LOCK:
            if _FIXTURE_RUNTIME_CACHE is None:
                _FIXTURE_RUNTIME_CACHE = validate_runtime_provenance(
                    runtime_provenance()
                )
            runtime = json.loads(
                json.dumps(
                    _FIXTURE_RUNTIME_CACHE,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
    else:
        with _FIXTURE_RUNTIME_CACHE_LOCK:
            _FIXTURE_RUNTIME_CACHE = None
        runtime = validate_runtime_provenance(runtime_provenance())
    runtime_payload = json.dumps(
        runtime,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "artifact_label": config["artifact_label"],
        "dataset_sha256": sha256_file(dataset.source_path),
        "config_sha256": config_sha256(config),
        "fixture": dataset.is_fixture,
        "scientific_use": ceiling,
        "source_tree_sha256": runtime["source_tree_sha256"],
        "requirements_lock_sha256": runtime["requirements_lock_sha256"],
        "runtime_provenance_sha256": hashlib.sha256(runtime_payload).hexdigest(),
        "runtime_provenance": runtime,
    }


def bind_rows(rows: Iterable[dict], evidence: dict[str, object]) -> list[dict]:
    output = []
    for source in rows:
        row = dict(source)
        for key, value in evidence.items():
            if isinstance(value, (dict, list)):
                continue
            if key in row and row[key] != value:
                raise ValueError(f"row attempts to override evidence field {key!r}")
            row[key] = value
        output.append(row)
    return output


def require_formal_qualification(
    gate: object,
    config: dict,
    dataset: FormalDataset,
    *,
    allow_nonscientific_fixture: bool,
) -> dict:
    if not isinstance(gate, dict):
        raise RuntimeError("qualification gate must be an object")
    if (
        not dataset.is_fixture
        and gate.get("scientific_use") != "FORMAL_EXPERIMENT_ALLOWED"
    ):
        raise RuntimeError("factorial requires scientific_use=FORMAL_EXPERIMENT_ALLOWED")
    required = {
        "schema_version",
        "passed",
        "scientific_use",
        "fixture",
        "dataset_sha256",
        "config_sha256",
        "upstream_gates",
        "teacher_checkpoint",
        "teacher_checkpoint_sha256",
        "artifact_label",
        "source_tree_sha256",
        "requirements_lock_sha256",
        "runtime_provenance_sha256",
        "runtime_provenance",
        "primary_route_contract",
        "physical_response",
    }
    missing = required.difference(gate)
    if missing:
        raise RuntimeError(f"qualification gate is missing authenticated fields: {sorted(missing)}")
    if gate["schema_version"] != QUALIFICATION_SCHEMA:
        raise RuntimeError("qualification gate schema is not V6-compatible")
    from .formal_routing import PRIMARY_ROUTE_CONTRACT

    if gate["primary_route_contract"] != PRIMARY_ROUTE_CONTRACT:
        raise RuntimeError("qualification gate primary route contract is not V6-compatible")
    expected = evidence_context(config, dataset, str(gate["scientific_use"]))
    for key in EVIDENCE_AUTH_KEYS:
        if gate[key] != expected[key]:
            raise RuntimeError(f"qualification gate {key} does not match the current run")
    software_fixture_execution = bool(
        dataset.is_fixture
        and allow_nonscientific_fixture
        and gate["scientific_use"] == "FORBIDDEN"
    )
    if not bool(gate["passed"]) and not software_fixture_execution:
        raise RuntimeError("formal factorial is blocked because the upstream qualification gate failed")
    if dataset.is_fixture:
        if not allow_nonscientific_fixture:
            raise RuntimeError("fixture training requires explicit software-test permission")
        if gate["scientific_use"] != "FORBIDDEN":
            raise RuntimeError("fixture qualification gate must remain FORBIDDEN")
    elif gate["scientific_use"] != "FORMAL_EXPERIMENT_ALLOWED":
        raise RuntimeError("factorial requires scientific_use=FORMAL_EXPERIMENT_ALLOWED")
    for gate_id, status in gate["upstream_gates"].items():
        if status not in ASSESSMENT_STATES:
            raise RuntimeError(f"invalid upstream assessment state for {gate_id}")
        if (
            gate_id in {"G1", "G2"}
            and status != "PASS"
            and not software_fixture_execution
        ):
            raise RuntimeError(f"required upstream gate {gate_id} is not PASS")
    teacher_checkpoint = Path(str(gate["teacher_checkpoint"]))
    if not teacher_checkpoint.is_file():
        raise RuntimeError("qualification teacher checkpoint is missing")
    if sha256_file(teacher_checkpoint) != gate["teacher_checkpoint_sha256"]:
        raise RuntimeError("qualification teacher checkpoint hash mismatch")
    physical = gate["physical_response"]
    if not isinstance(physical, dict):
        raise RuntimeError("qualification physical_response must be an object")
    if dataset.is_fixture:
        if (
            physical.get("status") != "NOT_ASSESSED_FIXTURE_FORBIDDEN"
            or physical.get("formal_physical_response_required_for_nonfixture") is not True
        ):
            raise RuntimeError("fixture qualification weakened physical-response exclusion")
    else:
        from .formal_action_inverse_response import (
            PHYSICAL_RESPONSE_CONTRACT,
            load_physical_response_checkpoint,
            load_physical_response_config,
        )

        checkpoint = Path(str(physical.get("checkpoint", "")))
        bindings = physical.get("bindings")
        physical_config = physical.get("physical_config")
        physical_config_path = (
            Path(__file__).resolve().parent / "configs" / "physical_response_v1.json"
        )
        if (
            physical.get("status") != "FORMAL_QUALIFICATION_MODEL_FIT"
            or physical.get("contract") != PHYSICAL_RESPONSE_CONTRACT
            or physical.get("checkpoint_round_trip_valid") is not True
            or checkpoint.is_symlink()
            or not checkpoint.is_file()
            or not isinstance(bindings, dict)
            or physical_config != load_physical_response_config()
            or physical.get("physical_config_sha256")
            != sha256_file(physical_config_path)
            or int(physical_config.get("selection_position_stride", -1)) != 1
            or physical_config.get("world_scope") != "all"
        ):
            raise RuntimeError("qualification physical-response contract is invalid")
        if sha256_file(checkpoint) != physical.get("checkpoint_sha256"):
            raise RuntimeError("qualification physical-response checkpoint hash mismatch")
        model_source = Path(str(bindings.get("model_source", "")))
        candidate_config = Path(str(bindings.get("candidate_config", "")))
        qualification_root = teacher_checkpoint.resolve().parent.parent
        physical_checkpoint_root = (
            qualification_root / "checkpoints" / "qualification_probes"
        ).resolve()
        if (
            bindings.get("dataset_sha256") != sha256_file(dataset.source_path)
            or bindings.get("formal_config_sha256") != expected["config_sha256"]
            or bindings.get("physical_config_sha256")
            != sha256_file(physical_config_path)
            or candidate_config.is_symlink()
            or not candidate_config.is_file()
            or sha256_file(candidate_config)
            != bindings.get("candidate_config_sha256")
            or model_source.is_symlink()
            or not model_source.is_file()
            or sha256_file(model_source) != bindings.get("model_source_sha256")
            or not checkpoint.resolve().is_relative_to(physical_checkpoint_root)
        ):
            raise RuntimeError("qualification physical-response binding mismatch")
        _model, checkpoint_bindings = load_physical_response_checkpoint(checkpoint)
        if checkpoint_bindings != bindings:
            raise RuntimeError("qualification physical-response payload binding mismatch")
    return gate


def complete_gate_vector(overrides: dict[str, str] | None = None) -> dict[str, str]:
    result = {gate_id: "NOT_ASSESSED" for gate_id in GATE_IDS}
    if overrides:
        for gate_id, status in overrides.items():
            if gate_id not in result:
                raise ValueError(f"unknown V6 gate: {gate_id}")
            if status not in ASSESSMENT_STATES:
                raise ValueError(f"invalid gate status: {status}")
            result[gate_id] = status
    return result


def blocked_claim_vector() -> dict[str, str]:
    return {claim_id: "BLOCKED" for claim_id in CLAIM_IDS}


def require_stage_manifested_gate(
    path: str | Path,
    payload: object,
    config: dict,
    dataset: FormalDataset,
    *,
    schema_version: str,
) -> dict:
    """Authenticate a gate against its stage manifest and current evidence context."""
    from .formal_io import read_strict_json

    gate_path = Path(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != schema_version:
        raise RuntimeError(f"gate schema mismatch for {gate_path}")
    if not gate_path.is_file():
        raise RuntimeError(f"gate file is missing: {gate_path}")
    on_disk = read_strict_json(gate_path)
    if on_disk != payload:
        raise RuntimeError(f"supplied gate payload differs from its manifested file: {gate_path}")
    expected = evidence_context(config, dataset, str(payload.get("scientific_use", "")))
    for key in EVIDENCE_AUTH_KEYS:
        if payload.get(key) != expected[key]:
            raise RuntimeError(f"gate {key} mismatch for {gate_path}")
    manifest_path = gate_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"gate has no stage manifest: {gate_path}")
    manifest = read_strict_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "csi-pairs-formal-stage-manifest-v2.1-v6"
    ):
        raise RuntimeError(f"stage manifest schema mismatch: {manifest_path}")
    for key in EVIDENCE_AUTH_KEYS:
        if manifest.get(key) != expected[key]:
            raise RuntimeError(f"stage manifest {key} mismatch: {manifest_path}")
    entries = manifest.get("files")
    _authenticate_stage_inventory(gate_path.parent, entries)
    relative = gate_path.name
    matches = [
        row
        for row in entries
        if isinstance(row, dict) and row.get("path") == relative
    ] if isinstance(entries, list) else []
    if len(matches) != 1 or matches[0].get("sha256") != sha256_file(gate_path):
        raise RuntimeError(f"gate is absent from or mismatched with its stage manifest: {gate_path}")
    return payload


def _authenticate_stage_inventory(stage_root: Path, entries: object) -> None:
    """Reauthenticate the completed stage while allowing later nested stages."""
    root = stage_root.resolve()
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"stage manifest has no authenticated inventory: {root}")
    expected: dict[str, Path] = {}
    for row in entries:
        if not isinstance(row, dict):
            raise RuntimeError(f"stage manifest inventory is malformed: {root}")
        relative = row.get("path")
        digest = row.get("sha256")
        size = row.get("bytes")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in expected
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or type(size) is not int
            or size < 0
        ):
            raise RuntimeError(f"stage manifest inventory is malformed: {root}")
        path = root / relative
        if path.is_symlink():
            raise RuntimeError(f"stage artifact is a symbolic link: {path}")
        resolved = path.resolve()
        if root not in resolved.parents or not resolved.is_file():
            raise RuntimeError(f"stage artifact is missing or escapes its root: {path}")
        if resolved.stat().st_size != size or sha256_file(resolved) != digest:
            raise RuntimeError(f"stage artifact changed after manifesting: {path}")
        expected[relative] = resolved

    # Evaluation/controls gain separately manifested retention/shuffle children
    # later in the full chain. They are not part of the parent stage inventory.
    later_nested_roots = {
        manifest_path.parent.resolve()
        for manifest_path in root.rglob("manifest.json")
        if manifest_path.parent.resolve() != root
        and not any(
            relative == manifest_path.parent.relative_to(root).as_posix()
            or relative.startswith(
                manifest_path.parent.relative_to(root).as_posix() + "/"
            )
            for relative in expected
        )
    }
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and not any(nested == path.resolve() or nested in path.resolve().parents for nested in later_nested_roots)
    }
    if set(expected) != actual:
        raise RuntimeError(
            "stage manifest inventory is incomplete: "
            f"missing={sorted(actual - set(expected))[:5]}, "
            f"unexpected={sorted(set(expected) - actual)[:5]}"
        )


def require_manifested_formal_qualification(
    gate: object,
    config: dict,
    dataset: FormalDataset,
    *,
    allow_nonscientific_fixture: bool,
) -> dict:
    """Validate qualification semantics and authenticate its complete stage output."""
    validated = require_formal_qualification(
        gate,
        config,
        dataset,
        allow_nonscientific_fixture=allow_nonscientific_fixture,
    )
    teacher_parent = Path(str(validated["teacher_checkpoint"])).parent
    stage_dir = teacher_parent.parent if teacher_parent.name == "checkpoints" else teacher_parent
    gate_path = stage_dir / "gate.json"
    return require_stage_manifested_gate(
        gate_path,
        validated,
        config,
        dataset,
        schema_version=QUALIFICATION_SCHEMA,
    )
