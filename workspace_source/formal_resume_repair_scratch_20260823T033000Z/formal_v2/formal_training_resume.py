from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .formal_io import read_strict_json, sha256_file
from .formal_model import portable_state_dict, torch


CHECKPOINT_SCHEMA = "csi-pairs-v6-factorial-resume-checkpoint-v1"
POINTER_SCHEMA = "csi-pairs-v6-factorial-resume-pointer-v1"


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _portable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _portable(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_portable(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_portable(child) for child in value)
    return value


def _require_regular_parent(path: Path) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError(f"resume checkpoint parent must be a regular directory: {parent}")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(path: Path, payload: object) -> None:
    _require_regular_parent(path)
    if path.is_symlink():
        raise RuntimeError(f"resume checkpoint slot cannot be a symlink: {path}")
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: object) -> None:
    _require_regular_parent(path)
    if path.is_symlink():
        raise RuntimeError(f"resume pointer cannot be a symlink: {path}")
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _paths(base: str | Path, generation: int) -> tuple[Path, Path]:
    base = Path(base)
    if generation < 0:
        raise ValueError("resume checkpoint generation must be nonnegative")
    pointer = base.with_name(f"{base.name}.latest.json")
    slot = base.with_name(f"{base.name}.slot{generation % 2}.pt")
    return pointer, slot


def load_factorial_resume(
    base: str | Path,
    *,
    context: dict,
    seed: int,
    arm: str,
    total_steps: int,
    batch_size: int,
    map_location,
) -> dict | None:
    pointer, _ = _paths(base, 0)
    if not pointer.exists() and not pointer.is_symlink():
        return None
    if pointer.is_symlink() or not pointer.is_file():
        raise RuntimeError("factorial resume pointer must be a regular file")
    receipt = read_strict_json(pointer)
    required = {
        "schema_version",
        "checkpoint_schema_version",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_bytes",
        "generation",
        "seed",
        "arm",
        "completed_steps",
        "total_steps",
        "batch_size",
        "context_sha256",
        "status",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise RuntimeError("factorial resume pointer fields are invalid")
    generation = receipt["generation"]
    if type(generation) is not int or generation < 0:
        raise RuntimeError("factorial resume generation is invalid")
    expected_pointer, checkpoint = _paths(base, generation)
    if expected_pointer != pointer or receipt["checkpoint_path"] != checkpoint.name:
        raise RuntimeError("factorial resume pointer selected an invalid slot")
    context_sha = _json_sha256(context)
    completed = receipt["completed_steps"]
    expected_status = "COMPLETE" if completed == total_steps else "IN_PROGRESS"
    if (
        receipt["schema_version"] != POINTER_SCHEMA
        or receipt["checkpoint_schema_version"] != CHECKPOINT_SCHEMA
        or type(receipt["checkpoint_path"]) is not str
        or type(receipt["checkpoint_sha256"]) is not str
        or len(receipt["checkpoint_sha256"]) != 64
        or type(receipt["checkpoint_bytes"]) is not int
        or receipt["checkpoint_bytes"] <= 0
        or receipt["seed"] != int(seed)
        or type(receipt["seed"]) is not int
        or receipt["arm"] != arm
        or type(receipt["arm"]) is not str
        or type(completed) is not int
        or not 0 < completed <= total_steps
        or receipt["total_steps"] != total_steps
        or type(receipt["total_steps"]) is not int
        or receipt["batch_size"] != batch_size
        or type(receipt["batch_size"]) is not int
        or receipt["context_sha256"] != context_sha
        or type(receipt["context_sha256"]) is not str
        or receipt["status"] != expected_status
    ):
        raise RuntimeError("factorial resume pointer binding is stale or invalid")
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise RuntimeError("factorial resume checkpoint slot is unavailable")
    if (
        checkpoint.stat().st_size != receipt["checkpoint_bytes"]
        or sha256_file(checkpoint) != receipt["checkpoint_sha256"]
    ):
        raise RuntimeError("factorial resume checkpoint hash or size mismatch")
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    payload_required = {
        "schema_version",
        "generation",
        "context",
        "context_sha256",
        "seed",
        "arm",
        "completed_steps",
        "total_steps",
        "batch_size",
        "model_state_dict",
        "optimizer_state_dict",
        "training_state",
    }
    if not isinstance(payload, dict) or set(payload) != payload_required:
        raise RuntimeError("factorial resume checkpoint fields are invalid")
    if (
        type(payload["generation"]) is not int
        or type(payload["seed"]) is not int
        or type(payload["arm"]) is not str
        or type(payload["completed_steps"]) is not int
        or type(payload["total_steps"]) is not int
        or type(payload["batch_size"]) is not int
        or type(payload["context_sha256"]) is not str
        or not isinstance(payload["context"], dict)
        or not isinstance(payload["model_state_dict"], dict)
        or not isinstance(payload["optimizer_state_dict"], dict)
        or not isinstance(payload["training_state"], dict)
    ):
        raise RuntimeError("factorial resume checkpoint value types are invalid")
    for key in (
        "generation",
        "context_sha256",
        "seed",
        "arm",
        "completed_steps",
        "total_steps",
        "batch_size",
    ):
        if payload[key] != receipt[key]:
            raise RuntimeError(f"factorial resume payload differs from pointer: {key}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA or payload["context"] != context:
        raise RuntimeError("factorial resume payload context is invalid")
    return payload


def save_factorial_resume(
    base: str | Path,
    *,
    previous_generation: int,
    context: dict,
    seed: int,
    arm: str,
    completed_steps: int,
    total_steps: int,
    batch_size: int,
    model,
    optimizer,
    training_state: dict,
) -> int:
    if not isinstance(context, dict):
        raise TypeError("factorial resume context must be a dictionary")
    if type(previous_generation) is not int or previous_generation < -1:
        raise ValueError("factorial resume previous generation is invalid")
    if type(seed) is not int or not isinstance(arm, str) or not arm:
        raise ValueError("factorial resume job identity is invalid")
    if (
        type(completed_steps) is not int
        or type(total_steps) is not int
        or type(batch_size) is not int
        or not 0 < completed_steps <= total_steps
        or batch_size < 1
    ):
        raise ValueError("factorial resume completed step is outside the formal schedule")
    if not isinstance(training_state, dict):
        raise TypeError("factorial resume training state must be a dictionary")
    generation = previous_generation + 1
    pointer, checkpoint = _paths(base, generation)
    context_sha = _json_sha256(context)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "generation": generation,
        "context": context,
        "context_sha256": context_sha,
        "seed": int(seed),
        "arm": arm,
        "completed_steps": int(completed_steps),
        "total_steps": int(total_steps),
        "batch_size": int(batch_size),
        "model_state_dict": portable_state_dict(model),
        "optimizer_state_dict": _portable(optimizer.state_dict()),
        "training_state": training_state,
    }
    _atomic_torch_save(checkpoint, payload)
    receipt = {
        "schema_version": POINTER_SCHEMA,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA,
        "checkpoint_path": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "generation": generation,
        "seed": int(seed),
        "arm": arm,
        "completed_steps": int(completed_steps),
        "total_steps": int(total_steps),
        "batch_size": int(batch_size),
        "context_sha256": context_sha,
        "status": "COMPLETE" if completed_steps == total_steps else "IN_PROGRESS",
    }
    _atomic_json(pointer, receipt)
    return generation
