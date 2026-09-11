"""Small, atomic model+optimizer checkpoints for deterministic probe fitting.

The caller owns the run/probe write lock. Checkpoints are recovery state, not
scientific completion receipts. A changed input, initialization or schedule
requires a different output directory instead of silently reusing weights.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def array_digest(array):
    if array is None:
        return None
    array = np.asarray(array)
    h = hashlib.sha256(str((array.shape, str(array.dtype))).encode())
    # Hash strided arrays in bounded C-order buffers; avoid a full feature copy.
    for block in np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"], order="C", buffersize=262144):
        h.update(np.ascontiguousarray(block).tobytes())
    return h.hexdigest()


def portable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(portable(item) for item in value)
    return value


def require_finite_state(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite probe recovery state")
    elif isinstance(value, dict):
        for child in value.values():
            require_finite_state(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            require_finite_state(child)


class ProbeResume:
    def __init__(self, path, model, optimizer, arrays, steps, batch_rows, *, interval=100, extra_identity=None):
        self.path = Path(path) if path is not None else None
        self.model, self.optimizer = model, optimizer
        self.steps, self.interval, self.start_step = steps, interval, 0
        if self.path is None:
            return
        if interval < 1 or steps < 1:
            raise ValueError("Probe checkpoint interval and steps must be positive")
        self.identity = {
            "schema": "csi-pairs-probe-recovery-v1",
            "initial_state": {name: array_digest(t.detach().cpu().numpy()) for name, t in model.state_dict().items()},
            "arrays": [array_digest(array) for array in arrays],
            "steps": steps, "batch_rows": batch_rows,
            "optimizer": type(optimizer).__name__,
            "hyperparameters": [{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups],
            "torch_version": str(torch.__version__),
            "device": str(next(model.parameters()).device),
        }
        if extra_identity is not None:
            self.identity["protocol"] = extra_identity
        self.identity_hash = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        if self.path.exists():
            value = torch.load(self.path, map_location="cpu", weights_only=True)
            if value.get("identity_hash") != self.identity_hash:
                raise ValueError("Probe resume input, initialization or training schedule changed")
            completed = value.get("completed_steps")
            if type(completed) is not int or not 0 <= completed <= steps:
                raise ValueError("Invalid completed probe step count")
            require_finite_state(value["model"])
            require_finite_state(value["optimizer"])
            model.load_state_dict(value["model"], strict=True)
            optimizer.load_state_dict(value["optimizer"])
            self.start_step = completed

    def save(self, completed):
        if self.path is None or (completed % self.interval and completed != self.steps):
            return
        value = {"identity_hash": self.identity_hash, "identity": self.identity, "completed_steps": completed,
                 "model": portable(self.model.state_dict()), "optimizer": portable(self.optimizer.state_dict())}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + ".tmp")
        with temp.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, self.path)
