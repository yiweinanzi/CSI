from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path


REGISTRY_SCHEMA = "csi-pairs-sionna-approved-libllvm-v1"
RUNTIME_SCHEMA = "csi-pairs-sionna-libllvm-runtime-v1"
REGISTRY_RELATIVE = Path("formal_v2/configs/sionna_llvm_approved_v1.json")
RUNTIME_RECORD_NAME = "llvm_runtime.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_constant)
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _approved_entry(project_root: Path, library_sha256: str) -> tuple[dict, Path]:
    registry_path = project_root / REGISTRY_RELATIVE
    if registry_path.is_symlink() or not registry_path.is_file():
        raise RuntimeError("approved libLLVM registry is missing or unsafe")
    registry = _read_json(registry_path)
    if set(registry) != {"schema_version", "libraries"} or registry.get(
        "schema_version"
    ) != REGISTRY_SCHEMA:
        raise RuntimeError("approved libLLVM registry schema is invalid")
    libraries = registry["libraries"]
    if not isinstance(libraries, list):
        raise RuntimeError("approved libLLVM registry entries are invalid")
    expected_platform = (platform.system(), platform.machine())
    matches = [
        row
        for row in libraries
        if isinstance(row, dict)
        and set(row)
        == {"platform_system", "platform_machine", "sha256", "provenance"}
        and (row["platform_system"], row["platform_machine"]) == expected_platform
        and row["sha256"] == library_sha256
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "libLLVM SHA-256 is not pre-registered for "
            f"{expected_platform[0]} {expected_platform[1]}"
        )
    return matches[0], registry_path


def approved_library_record(project_root: str | Path, library: str | Path) -> dict:
    project = Path(project_root).resolve()
    candidate = Path(library)
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError("libLLVM must be an absolute regular non-symlink file")
    path = candidate.resolve()
    digest = _sha256(path)
    entry, registry_path = _approved_entry(project, digest)
    return {
        "schema_version": RUNTIME_SCHEMA,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "libllvm_path": str(path),
        "libllvm_sha256": digest,
        "approval_provenance": entry["provenance"],
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": _sha256(registry_path),
    }


def write_runtime_record(
    project_root: str | Path, runtime_root: str | Path, library: str | Path
) -> Path:
    target = Path(runtime_root).resolve() / RUNTIME_RECORD_NAME
    payload = approved_library_record(project_root, library)
    if target.exists() or target.is_symlink():
        existing = _read_json(target)
        if existing != payload:
            raise RuntimeError("existing libLLVM runtime record differs from approved library")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    return target


def require_runtime_record(
    project_root: str | Path, runtime_root: str | Path
) -> tuple[Path, dict]:
    project = Path(project_root).resolve()
    record_path = Path(runtime_root).resolve() / RUNTIME_RECORD_NAME
    if record_path.is_symlink() or not record_path.is_file():
        raise RuntimeError("approved libLLVM runtime record is missing; rerun setup_sionna.sh")
    record = _read_json(record_path)
    required = {
        "schema_version",
        "platform_system",
        "platform_machine",
        "libllvm_path",
        "libllvm_sha256",
        "approval_provenance",
        "registry_path",
        "registry_sha256",
    }
    if set(record) != required or record.get("schema_version") != RUNTIME_SCHEMA:
        raise RuntimeError("libLLVM runtime record schema is invalid")
    current = approved_library_record(project, record["libllvm_path"])
    if record != current:
        raise RuntimeError("libLLVM runtime record or approved registry changed")
    return Path(record["libllvm_path"]), record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Authenticate the Sionna libLLVM runtime")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    register.add_argument("--libllvm", required=True)
    commands.add_parser("environment")
    args = parser.parse_args(argv)
    if args.command == "register":
        path = write_runtime_record(args.project_root, args.runtime_root, args.libllvm)
        print(path)
    else:
        path, _record = require_runtime_record(args.project_root, args.runtime_root)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
