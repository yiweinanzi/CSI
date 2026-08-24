from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from .formal_io import read_strict_json, sha256_file, write_json
from .formal_run_approval import (
    COMPUTE_PLAN_SCHEMA,
    FORMAL_COMPUTE_COMPONENTS,
    REQUIRED_FULL_RUN_INPUT_NAMES,
)


OPERATOR_PREFLIGHT_SCHEMA = "csi-pairs-operator-preflight-v1"
SCIENTIFIC_USE = "NOT_ASSESSED"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "formal_v2.json"
WAIBU_ROOT = Path(__file__).resolve().parents[1] / "waibu"
FORMAL_HOSTS = {("linux", "x86_64"), ("darwin", "arm64")}
REQUIRED_ENV_NAMES = ("CSI_PAIRS_DEVICES", "CUDA_VISIBLE_DEVICES")
COMPUTE_PLAN_REQUIRED_KEYS = {
    "schema_version",
    "profile",
    "estimated_output_bytes",
    "minimum_free_disk_bytes",
    "estimated_wall_time_seconds",
    "authorized_wall_time_seconds",
    "required_gpu_count",
    "minimum_gpu_memory_bytes",
    "estimated_gpu_hours",
    "authorized_gpu_hours",
    "component_estimates",
    "required_environment_variables",
    "license_acknowledgements",
}
REPORT_KEYS = (
    "schema_version",
    "formal_host",
    "formal_ready_to_prepare",
    "scientific_use",
    "platform",
    "python",
    "dataset",
    "dataset_load",
    "waibu",
    "environment",
    "required_prepare_full_run_manifest_flags",
    "compute_plan",
)


def required_prepare_full_run_manifest_flags() -> list[str]:
    return [f"--{name.replace('_', '-')}" for name in REQUIRED_FULL_RUN_INPUT_NAMES]


def run_operator_preflight(
    dataset_path: str | Path,
    config_path: str | Path | None = None,
    output_root: str | Path | None = None,
    compute_plan_path: str | Path | None = None,
) -> dict:
    """Inventory-only operator check. Never claims science."""
    platform_check, formal_host = _platform_check()
    python_check = _python_check()
    dataset_check = _dataset_check(dataset_path)
    dataset_load = _dataset_load_check(dataset_path, config_path, dataset_check)
    waibu_check = _waibu_check()
    environment_check = _environment_check()
    flags = required_prepare_full_run_manifest_flags()
    compute_plan = _compute_plan_check(compute_plan_path)
    dataset_loaded = bool(dataset_load["loaded"]) and dataset_load["fixture"] is False
    formal_ready_to_prepare = bool(
        formal_host
        and dataset_loaded
        and environment_check["ok"]
    )
    report = {
        "schema_version": OPERATOR_PREFLIGHT_SCHEMA,
        "formal_host": formal_host,
        "formal_ready_to_prepare": formal_ready_to_prepare,
        "scientific_use": SCIENTIFIC_USE,
        "platform": platform_check,
        "python": python_check,
        "dataset": dataset_check,
        "dataset_load": dataset_load,
        "waibu": waibu_check,
        "environment": environment_check,
        "required_prepare_full_run_manifest_flags": flags,
        "compute_plan": compute_plan,
    }
    if set(report) != set(REPORT_KEYS):
        raise RuntimeError("operator preflight report keys must stay exact")
    if output_root is not None:
        _write_report(output_root, report)
    return report


def _check(ok: bool, message: str, **extra) -> dict:
    payload = {"ok": bool(ok), "message": message}
    payload.update(extra)
    return payload


def _platform_check() -> tuple[dict, bool]:
    system = sys.platform
    machine = platform.machine()
    formal_host = (system, machine) in FORMAL_HOSTS
    if system == "win32":
        formal_host = False
        message = (
            "win32 is not a formal host; formal execution requires Linux x86_64 "
            "or macOS arm64"
        )
    elif formal_host:
        message = f"platform {system} {machine} matches a formal host tuple"
    else:
        message = (
            f"platform {system} {machine} is not a formal host; "
            "formal execution requires Linux x86_64 or macOS arm64"
        )
    return (
        _check(
            formal_host,
            message,
            system=system,
            machine=machine,
            formal_host=formal_host,
        ),
        formal_host,
    )


def _python_check() -> dict:
    version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    formal_python = (sys.version_info.major, sys.version_info.minor) == (3, 12)
    message = (
        f"interpreter is {version}; formal execution requires CPython 3.12"
    )
    return _check(formal_python, message, version=version, formal_requires="3.12")


def _dataset_check(dataset_path: str | Path) -> dict:
    path = Path(dataset_path)
    exists = path.exists()
    regular_file = exists and path.is_file() and not path.is_symlink()
    size_bytes = None
    digest = None
    if regular_file:
        size_bytes = int(path.stat().st_size)
        digest = sha256_file(path)
        message = f"regular file present size_bytes={size_bytes} sha256={digest}"
        ok = True
    elif exists and path.is_symlink():
        message = "dataset path is a symlink; formal inputs must be regular files"
        ok = False
    elif exists:
        message = "dataset path exists but is not a regular file"
        ok = False
    else:
        message = f"dataset path does not exist: {path}"
        ok = False
    return _check(
        ok,
        message,
        path=str(path),
        exists=exists,
        regular_file=regular_file,
        size_bytes=size_bytes,
        sha256=digest,
    )


def _looks_like_npz(path: Path) -> bool:
    if path.suffix.lower() != ".npz":
        return False
    try:
        with path.open("rb") as handle:
            magic = handle.read(4)
    except OSError:
        return False
    return magic.startswith(b"PK")


def _dataset_load_check(
    dataset_path: str | Path,
    config_path: str | Path | None,
    dataset_check: dict,
) -> dict:
    path = Path(dataset_path)
    if not dataset_check["regular_file"]:
        return _check(
            False,
            "FormalDataset load not attempted because the dataset is not a regular file",
            loaded=False,
            fixture=None,
        )
    if not _looks_like_npz(path):
        return _check(
            False,
            "FormalDataset load not attempted because the file does not look like an NPZ",
            loaded=False,
            fixture=None,
        )
    require_clean_csi = True
    if config_path is not None:
        try:
            from .formal_config import load_formal_config

            config = load_formal_config(config_path)
            require_clean_csi = bool(config["data"]["require_clean_csi"])
        except (OSError, ValueError, KeyError):
            require_clean_csi = True
    try:
        from .formal_dataset import FormalDataset, FormalDatasetError
    except ImportError as error:
        return _check(
            False,
            f"FormalDataset import failed: {error}",
            loaded=False,
            fixture=None,
        )
    try:
        dataset = FormalDataset.load(path, require_clean_csi=require_clean_csi)
    except (FormalDatasetError, OSError) as error:
        return _check(
            False,
            f"FormalDataset.load failed: {error}",
            loaded=False,
            fixture=None,
        )
    fixture = bool(dataset.is_fixture)
    if fixture:
        message = "FormalDataset loaded as a fixture; fixture bytes cannot authorize prepare-full-run"
    else:
        message = "FormalDataset loaded; load success is not scientific PASS"
    return _check(True, message, loaded=True, fixture=fixture)


def _waibu_check() -> dict:
    root = WAIBU_ROOT
    if not root.exists():
        return _check(
            False,
            f"waibu directory does not exist: {root}",
            path=str(root),
            empty=True,
        )
    if not root.is_dir() or root.is_symlink():
        return _check(
            False,
            f"waibu path is not a regular directory: {root}",
            path=str(root),
            empty=True,
        )
    entries = [path.name for path in root.iterdir()]
    empty = len(entries) == 0
    if empty:
        message = f"waibu directory is empty: {root}"
    else:
        message = (
            f"waibu directory is not empty ({len(entries)} entries); "
            "this is not verify-waibu-resources authentication"
        )
    return _check(not empty, message, path=str(root), empty=empty, entries=entries)


def _environment_check() -> dict:
    present = {name: bool(os.environ.get(name)) for name in REQUIRED_ENV_NAMES}
    missing = [name for name, value in present.items() if not value]
    ok = not missing
    if ok:
        message = (
            "CSI_PAIRS_DEVICES and CUDA_VISIBLE_DEVICES are set; "
            "presence is not a GPU-capacity validation"
        )
    else:
        message = f"required environment variables are unset: {missing}"
    return _check(ok, message, **{f"{name}_set": present[name] for name in REQUIRED_ENV_NAMES})


def _compute_plan_check(compute_plan_path: str | Path | None) -> dict:
    if compute_plan_path is None:
        return _check(
            False,
            "compute-plan not supplied; not measurement-validated",
            supplied=False,
            schema_keys_present=False,
            measurement_validated=False,
            schema_version=COMPUTE_PLAN_SCHEMA,
            required_components=list(FORMAL_COMPUTE_COMPONENTS),
        )
    path = Path(compute_plan_path)
    try:
        plan = read_strict_json(path)
    except (OSError, ValueError) as error:
        return _check(
            False,
            f"compute-plan could not be read; not measurement-validated: {error}",
            supplied=True,
            schema_keys_present=False,
            measurement_validated=False,
            path=str(path),
            schema_version=COMPUTE_PLAN_SCHEMA,
            required_components=list(FORMAL_COMPUTE_COMPONENTS),
        )
    keys_present = isinstance(plan, dict) and set(plan) == COMPUTE_PLAN_REQUIRED_KEYS
    schema_match = keys_present and plan.get("schema_version") == COMPUTE_PLAN_SCHEMA
    if schema_match:
        message = (
            "compute-plan top-level keys and schema_version are present; "
            "not measurement-validated"
        )
    else:
        message = (
            "compute-plan is missing required keys or schema_version; "
            "not measurement-validated"
        )
    return _check(
        schema_match,
        message,
        supplied=True,
        schema_keys_present=keys_present,
        measurement_validated=False,
        path=str(path),
        schema_version=COMPUTE_PLAN_SCHEMA,
        required_components=list(FORMAL_COMPUTE_COMPONENTS),
    )


def _write_report(output_root: str | Path, report: dict) -> Path:
    output = Path(output_root)
    if output.suffix == ".json":
        target = output
    else:
        target = output / "operator_preflight.json"
    write_json(target, report)
    return target
