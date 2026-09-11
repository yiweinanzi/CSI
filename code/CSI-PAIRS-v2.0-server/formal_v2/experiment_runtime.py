"""Small, portable run records; no installation or approval certification."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

SCHEMA = "csi-pairs-experiment-runtime-v1"
RUNTIME_PROVENANCE_FIELDS = {
    "schema_version", "source_tree_sha256", "requirements_lock_sha256",
    "dependency_record_kind", "python_version", "python_executable",
    "platform_system", "platform_machine", "installed_distributions", "torch",
}
TORCH_RUNTIME_FIELDS = {"version", "cuda_version", "cuda_available", "deterministic_algorithms",
                        "cudnn_deterministic", "cudnn_benchmark", "cudnn_allow_tf32",
                        "cuda_matmul_allow_tf32", "float32_matmul_precision"}


def configure_reproducible_runtime():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def source_digest():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".json", ".sh", ".txt", ".toml", ".lock", ".yaml", ".yml"}:
            continue
        if any(part.startswith((".venv", ".runtime", "__pycache__")) for part in path.relative_to(root).parts):
            continue
        name = path.relative_to(root).as_posix().encode()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def source_identity():
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                capture_output=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unversioned"
    runtime = runtime_provenance()
    return {"source_root": str(root), "git_commit": commit,
            "source_tree_sha256": runtime["source_tree_sha256"],
            "requirements_lock_sha256": runtime["requirements_lock_sha256"]}


def runtime_provenance():
    import torch
    versions = {}
    for name in ("numpy", "scipy", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    # Compatibility field for the retained stage schemas: a hash of the recorded
    # dependency versions, explicitly NOT an authenticated installation lock.
    versions_hash = hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()
    return {"schema_version": SCHEMA, "source_tree_sha256": source_digest(),
            "requirements_lock_sha256": versions_hash,
            "dependency_record_kind": "installed-version-summary",
            "python_version": platform.python_version(), "python_executable": sys.executable,
            "platform_system": platform.system(), "platform_machine": platform.machine(),
            "installed_distributions": versions,
            "torch": {"version": str(torch.__version__), "cuda_version": torch.version.cuda,
                      "cuda_available": bool(torch.cuda.is_available()),
                      "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                      "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                      "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                      "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                      "float32_matmul_precision": torch.get_float32_matmul_precision(),
                      "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}}


def validate_runtime_provenance(runtime):
    if not isinstance(runtime, dict) or runtime.get("schema_version") != SCHEMA:
        raise ValueError("Expected experiment runtime v1; use the historical checkout for archived certification records")
    if set(runtime) != RUNTIME_PROVENANCE_FIELDS:
        raise ValueError("Unexpected experiment runtime fields")
    for key in ("source_tree_sha256", "requirements_lock_sha256"):
        value = runtime.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid runtime {key}")
    versions = runtime.get("installed_distributions")
    if not isinstance(versions, dict) or set(versions) != {"numpy", "scipy", "torch"}:
        raise ValueError("Missing dependency versions")
    if hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest() != runtime["requirements_lock_sha256"]:
        raise ValueError("Dependency version record hash mismatch")
    if not isinstance(runtime.get("torch"), dict) or runtime.get("dependency_record_kind") != "installed-version-summary":
        raise ValueError("Invalid experiment runtime record")
    return runtime
