from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


SCHEMA = "csi-pairs-external-runtime-provenance-v1"
PROFILES = {"wigatr", "sionna"}
PROFILE_DISTRIBUTIONS = {
    "wigatr": {
        "gatr": "1.2.2",
        "torch": "2.0.1",
        "torch-geometric": "2.6.0",
        "wigatr": "1.0.0",
        "xformers": "0.0.20",
    },
    "sionna": {
        "h5py": "3.15.1",
        "sionna": "2.0.1",
        "sionna-rt": "1.2.1",
        "torch": "2.9.1+cpu",
    },
}


def _interpreter_site_package_roots() -> list[str]:
    prefix = Path(sys.prefix).resolve()
    roots = []
    for kind in ("purelib", "platlib"):
        configured = sysconfig.get_path(kind)
        if not isinstance(configured, str) or not configured:
            raise RuntimeError(f"external runtime {kind} path is unavailable")
        root = Path(configured).resolve()
        if root != prefix and prefix not in root.parents:
            raise RuntimeError(f"external runtime {kind} path escapes interpreter prefix")
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError(f"external runtime {kind} path is missing or unsafe")
        text = str(root)
        if text not in roots:
            roots.append(text)
    return roots


def collect_external_runtime(profile: str, project_root: str | Path) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"unsupported external runtime profile: {profile}")
    project = Path(project_root).resolve()
    distributions = {}
    for distribution in importlib.metadata.distributions(
        path=_interpreter_site_package_roots()
    ):
        name = distribution.metadata.get("Name")
        if not isinstance(name, str) or not name.strip():
            continue
        key = name.lower().replace("_", "-")
        record = distribution.read_text("RECORD")
        value = {
            "name": name,
            "version": distribution.version,
            "record_sha256": (
                hashlib.sha256(record.encode("utf-8")).hexdigest()
                if record is not None
                else None
            ),
        }
        if key in distributions and distributions[key] != value:
            raise RuntimeError(f"external runtime contains duplicate distribution metadata: {name}")
        distributions[key] = value

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    lock_files = _lock_files(profile, project)
    torch_record = _torch_record()
    base = {
        "schema_version": SCHEMA,
        "profile": profile,
        "python_executable": os.path.abspath(sys.executable),
        "python_prefix": os.path.abspath(sys.prefix),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "lock_files": lock_files,
        "installed_distributions": {
            key: distributions[key] for key in sorted(distributions)
        },
        "torch": torch_record,
    }
    payload = json.dumps(
        base,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {**base, "environment_sha256": hashlib.sha256(payload).hexdigest()}


def probe_external_runtime(
    executable: str | Path,
    profile: str,
    project_root: str | Path,
    *,
    require_execution_ready: bool = True,
) -> dict:
    executable_path = Path(executable).absolute()
    if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
        raise RuntimeError(f"external runtime interpreter is unavailable: {executable_path}")
    project = Path(project_root).resolve()
    completed = subprocess.run(
        [
            str(executable_path),
            "-m",
            "formal_v2.formal_external_runtime",
            "--profile",
            profile,
            "--project-root",
            str(project),
            *(
                ["--require-execution-ready"]
                if require_execution_ready
                else []
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=project,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"external runtime provenance probe failed for {profile}: {completed.stderr.strip()}"
        )
    try:
        record = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("external runtime provenance probe emitted invalid JSON") from error
    validate_external_runtime(
        record,
        profile=profile,
        executable=executable_path,
        require_execution_ready=require_execution_ready,
    )
    return record


def validate_external_runtime(
    record: object,
    *,
    profile: str,
    executable: str | Path | None = None,
    require_execution_ready: bool = False,
) -> dict:
    required = {
        "schema_version",
        "profile",
        "python_executable",
        "python_prefix",
        "python_version",
        "python_implementation",
        "platform_system",
        "platform_release",
        "platform_machine",
        "cuda_visible_devices",
        "cublas_workspace_config",
        "lock_files",
        "installed_distributions",
        "torch",
        "environment_sha256",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise RuntimeError("external runtime provenance fields must be exact")
    if record["schema_version"] != SCHEMA or record["profile"] != profile:
        raise RuntimeError("external runtime provenance schema or profile mismatch")
    expected_python = "3.10" if profile == "wigatr" else "3.12.13"
    python_matches = (
        str(record["python_version"]).startswith(expected_python + ".")
        if profile == "wigatr"
        else record["python_version"] == expected_python
    )
    if not python_matches:
        raise RuntimeError(f"{profile} runtime requires Python {expected_python}")
    if executable is not None and os.path.realpath(record["python_executable"]) != os.path.realpath(executable):
        raise RuntimeError("external runtime provenance interpreter mismatch")
    if not isinstance(record["lock_files"], dict) or not record["lock_files"]:
        raise RuntimeError("external runtime provenance has no environment lock")
    for label, digest in record["lock_files"].items():
        if (
            not isinstance(label, str)
            or not label
            or not _lower_sha256(digest)
        ):
            raise RuntimeError("external runtime lock-file provenance is invalid")
    distributions = record["installed_distributions"]
    if not isinstance(distributions, dict) or not distributions:
        raise RuntimeError("external runtime installed-distribution inventory is empty")
    for key, value in distributions.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, dict)
            or set(value) != {"name", "version", "record_sha256"}
            or not isinstance(value["name"], str)
            or not isinstance(value["version"], str)
            or (
                value["record_sha256"] is not None
                and not _lower_sha256(value["record_sha256"])
            )
        ):
            raise RuntimeError("external runtime distribution provenance is invalid")
    required_distributions = PROFILE_DISTRIBUTIONS[profile]
    if not set(required_distributions).issubset(distributions):
        raise RuntimeError(
            "external runtime is missing required distributions: "
            f"{sorted(set(required_distributions) - set(distributions))}"
        )
    for name, version in required_distributions.items():
        expected_version = (
            "2.9.1"
            if profile == "sionna"
            and name == "torch"
            and record["platform_system"] == "Darwin"
            else version
        )
        if distributions[name]["version"] != expected_version:
            raise RuntimeError(
                f"{profile} runtime distribution {name} must be exactly {expected_version}"
            )
        if distributions[name]["record_sha256"] is None:
            raise RuntimeError(
                f"{profile} runtime distribution {name} has no RECORD provenance"
            )
    torch_record = record["torch"]
    torch_fields = {
        "version",
        "cuda_version",
        "cudnn_version",
        "cuda_available",
        "deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cudnn_allow_tf32",
        "cuda_matmul_allow_tf32",
        "float32_matmul_precision",
        "nvidia_driver_versions",
        "devices",
    }
    if not isinstance(torch_record, dict) or set(torch_record) != torch_fields:
        raise RuntimeError("external runtime Torch provenance fields must be exact")
    if profile == "wigatr" and require_execution_ready and (
        torch_record["cuda_available"] is not True
        or not torch_record["cuda_version"]
        or not torch_record["cudnn_version"]
        or torch_record["deterministic_algorithms"] is not True
        or torch_record["cudnn_benchmark"] is not False
        or torch_record["cudnn_deterministic"] is not True
        or torch_record["cudnn_allow_tf32"] is not False
        or torch_record["cuda_matmul_allow_tf32"] is not False
        or torch_record["float32_matmul_precision"] != "highest"
        or not torch_record["nvidia_driver_versions"]
        or not torch_record["devices"]
    ):
        raise RuntimeError("Wi-GATr runtime is not deterministic CUDA-ready")
    if profile == "sionna" and distributions["sionna-rt"]["version"] != "1.2.1":
        raise RuntimeError("Sionna RT runtime version is not frozen to 1.2.1")
    without_hash = {key: value for key, value in record.items() if key != "environment_sha256"}
    payload = json.dumps(
        without_hash,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if record["environment_sha256"] != hashlib.sha256(payload).hexdigest():
        raise RuntimeError("external runtime environment digest mismatch")
    return record


def _lock_files(profile: str, project: Path) -> dict[str, str]:
    if profile == "wigatr":
        paths = {
            "wigatr_uv_lock": project / "formal_v2" / "external_adapters" / "vendor" / "Wi-GATr" / "uv.lock",
            "wigatr_vendor_manifest": project / "formal_v2" / "external_adapters" / "vendor" / "Wi-GATr" / "VENDOR_SHA256SUMS",
        }
    else:
        runtime = Path(sys.prefix).absolute().parent
        system = platform.system()
        machine = platform.machine()
        if (system, machine) == ("Darwin", "arm64"):
            runtime_lock = project / "formal_v2" / "requirements-sionna-runtime-darwin-arm64.txt"
        elif (system, machine) == ("Linux", "x86_64"):
            runtime_lock = project / "formal_v2" / "requirements-sionna-runtime-linux-x86_64.txt"
        else:
            raise RuntimeError(f"unsupported Sionna runtime target: {system} {machine}")
        paths = {
            "sionna_lrm_frozen_uv_lock": project
            / "formal_v2"
            / "external_adapters"
            / "sionna_lrm_uv.lock",
            "sionna_lrm_runtime_uv_lock": runtime
            / "src"
            / "sionna-large-radio-maps-main"
            / "uv.lock",
            "resource_registry": project / "formal_v2" / "configs" / "waibu_resources_v1.json",
            "sionna_source_archive": project / "waibu" / "sionna-main.zip",
            "sionna_lrm_source_archive": project / "waibu" / "sionna-large-radio-maps-main.zip",
            "sionna_runtime_requirements_lock": runtime_lock,
            "sionna_approved_libllvm_registry": project
            / "formal_v2"
            / "configs"
            / "sionna_llvm_approved_v1.json",
            "sionna_libllvm_runtime_record": runtime / "llvm_runtime.json",
        }
    output = {}
    for label, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"external runtime lock/provenance file is missing: {path}")
        output[label] = _sha256_file(path)
    if profile == "sionna" and (
        output["sionna_lrm_frozen_uv_lock"]
        != output["sionna_lrm_runtime_uv_lock"]
    ):
        raise RuntimeError("Sionna runtime uv.lock differs from the frozen project lock")
    if profile == "sionna":
        from .sionna_runtime_lock import require_runtime_record

        require_runtime_record(project, Path(sys.prefix).absolute().parent)
    return output


def _torch_record() -> dict:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("external runtime has no Torch installation") from error
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": str(properties.name),
                    "total_memory_bytes": _cuda_total_memory_bytes(
                        torch, index, properties
                    ),
                    "compute_capability": [int(properties.major), int(properties.minor)],
                }
            )
    return {
        "version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "cudnn_version": (
            int(torch.backends.cudnn.version())
            if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.version() is not None
            else None
        ),
        "cuda_available": bool(torch.cuda.is_available()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "nvidia_driver_versions": _nvidia_driver_versions(),
        "devices": devices,
    }


def _cuda_total_memory_bytes(torch_module, index: int, properties) -> int:
    reported = int(getattr(properties, "total_memory", 0))
    if reported > 0:
        return reported
    try:
        _free, total = torch_module.cuda.mem_get_info(int(index))
    except TypeError:
        with torch_module.cuda.device(int(index)):
            _free, total = torch_module.cuda.mem_get_info()
    total = int(total)
    if total <= 0:
        raise RuntimeError(f"CUDA device {index} reports non-positive total memory")
    return total


def _nvidia_driver_versions() -> list[str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        return []
    return sorted(
        {
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip()
        }
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lower_sha256(value) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CSI-PAIRS external-runtime provenance probe")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--require-execution-ready", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)
    try:
        record = collect_external_runtime(args.profile, args.project_root)
        validate_external_runtime(
            record,
            profile=args.profile,
            executable=sys.executable,
            require_execution_ready=bool(args.require_execution_ready),
        )
        encoded = json.dumps(record, sort_keys=True, ensure_ascii=True, allow_nan=False)
        if args.output:
            target = Path(args.output)
            if target.is_symlink():
                raise FileExistsError(f"refusing symbolic-link runtime provenance: {target}")
            if target.exists():
                if not args.verify_existing:
                    raise FileExistsError(
                        f"refusing to overwrite runtime provenance: {target}"
                    )
                existing = json.loads(target.read_text(encoding="utf-8"))
                validate_external_runtime(
                    existing,
                    profile=args.profile,
                    executable=sys.executable,
                    require_execution_ready=bool(args.require_execution_ready),
                )
                if existing != record:
                    raise RuntimeError(
                        "existing external runtime provenance differs from the current environment"
                    )
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("x", encoding="utf-8") as handle:
                    handle.write(encoded + "\n")
        print(encoded)
        return 0
    except Exception as error:
        print(f"external runtime probe error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
