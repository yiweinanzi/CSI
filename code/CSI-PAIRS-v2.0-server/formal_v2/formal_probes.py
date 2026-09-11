from __future__ import annotations

import copy
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .formal_metrics import binary_auroc, binary_nll
from .paper_probe_resume import ProbeResume


def _initialize_probe(factory, seed, device, rng_lock):
    # Training is deterministic and draws no randomness after initialization.
    with (rng_lock if rng_lock is not None else nullcontext()):
        if rng_lock is None:
            torch.manual_seed(int(seed))
            return factory().to(device=device)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            return factory().to(device=device)


def _progress(callback, phase, step, total_steps, rows, total_rows):
    if callback is not None:
        callback({
            "phase": phase,
            "step": step,
            "total_steps": total_steps,
            "completed_rows": rows,
            "total_rows": total_rows,
        })


def _training_rows(probe, features, targets, rows, zero, prefer_full, callback):
    device = _module_device(probe)
    if not prefer_full or device.type != "cuda":
        return rows
    count = int(features.shape[0])
    widths = [layer.out_features for layer in probe.modules() if isinstance(layer, nn.Linear)]
    inputs = (features.nbytes // features.itemsize + targets.size + (zero.size if zero is not None else 0)) * 4
    # Conservative admission estimate, not an OOM-and-retry mechanism.
    activations = count * sum(widths) * 4 * (16 if zero is not None else 8)
    parameters = sum(p.numel() * p.element_size() for p in probe.parameters()) * 8
    estimate = inputs + activations + parameters + 512 * 1024**2
    free, _total = torch.cuda.mem_get_info(device)
    selected = None if estimate <= int(free * 0.65) else rows
    if selected is None and estimate > int(free * 0.65):
        raise RuntimeError("full-batch probe does not fit the CUDA memory admission budget")
    if callback is not None:
        callback({
            "phase": "memory_admission", "device": str(device),
            "execution_mode": "full_batch" if selected is None else "full_gradient_accumulation",
            "batch_rows": count if selected is None else selected,
            "estimated_full_batch_bytes": estimate, "cuda_free_bytes": free,
        })
    return selected


def _training_tensors(arrays, device, cache_on_device):
    tensors = [None if a is None else _cpu_float_tensor(a) for a in arrays]
    if cache_on_device and device.type == "cuda":
        required = sum(t.numel() * t.element_size() for t in tensors if t is not None)
        free, _total = torch.cuda.mem_get_info(device)
        if required <= int(free * 0.35):
            return [None if t is None else t.to(device) for t in tensors]
    return tensors


def _training_slice(tensor, start, stop, device):
    values = tensor[start:stop]
    return values if values.device == device else _transfer_float_batch(values, device)


def _finish_training(probe, callback, steps, rows):
    device = _module_device(probe)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    _progress(callback, "training_complete", steps, steps, rows, rows)


class CompatibilityProbe(nn.Module):
    def __init__(self, input_dim: int, family: str, hidden_dim: int):
        super().__init__()
        if family == "linear":
            self.network = nn.Linear(input_dim, 1)
        elif family == "mlp2":
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"unknown compatibility probe family: {family}")
        self.family = family

    def forward(self, values):
        return self.network(values).squeeze(1)


class ActionResponseProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.zero_preserving = False

    def forward(self, values):
        return self.network(values)


def fit_select_compatibility_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    selection_x: np.ndarray,
    selection_y: np.ndarray,
    config: dict,
    *,
    seed: int,
    device: str | torch.device | None = None,
    train_batch_rows: int | None = None,
    rng_lock=None,
    progress_callback=None,
    prefer_full_batch: bool = False,
    checkpoint_dir=None,
    checkpoint_interval: int = 100,
) -> tuple[CompatibilityProbe, dict]:
    resolved_device = _resolve_probe_device(device)
    batch_rows = _validate_batch_rows(train_batch_rows, "train_batch_rows")
    candidates = []
    for offset, family in enumerate(("linear", "mlp2")):
        probe = _initialize_probe(
            lambda: CompatibilityProbe(
                train_x.shape[1], family, int(config["evaluation"]["probe_hidden_dim"])
            ),
            int(seed) + offset, resolved_device, rng_lock,
        )
        def report(event, family=family):
            if progress_callback is not None:
                progress_callback({**event, "family": family})
        _fit_binary(
            probe,
            train_x,
            train_y,
            config,
            train_batch_rows=batch_rows,
            progress_callback=report,
            prefer_full_batch=prefer_full_batch,
            checkpoint_path=(Path(checkpoint_dir) / (family + ".pt")) if checkpoint_dir is not None else None,
            checkpoint_interval=checkpoint_interval,
        )
        report({"phase": "prediction", "step": 0, "total_steps": 0})
        probabilities = predict_binary_probe(
            probe,
            selection_x,
            batch_rows=(batch_rows if resolved_device.type == "cuda" else None),
        )
        report({"phase": "statistics", "metric": "binary_nll_and_auroc"})
        candidates.append(
            {
                "family": family,
                "probe": copy.deepcopy(probe),
                "selection_nll": binary_nll(selection_y, probabilities),
                "selection_auroc": binary_auroc(selection_y, probabilities),
            }
        )
    candidates.sort(key=lambda row: (row["selection_nll"], row["family"]))
    selected = candidates[0]
    if progress_callback is not None:
        progress_callback({"phase": "probe_complete", "selected_family": selected["family"]})
    return selected["probe"], {
        "selected_family": selected["family"],
        "selection_nll": selected["selection_nll"],
        "selection_auroc": selected["selection_auroc"],
        "candidate_metrics": [
            {key: value for key, value in row.items() if key != "probe"} for row in candidates
        ],
    }


def fit_action_response_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    config: dict,
    *,
    seed: int,
    zero_action_x: np.ndarray | None = None,
    device: str | torch.device | None = None,
    train_batch_rows: int | None = None,
    rng_lock=None,
    progress_callback=None,
    prefer_full_batch: bool = False,
    checkpoint_dir=None,
    checkpoint_interval: int = 100,
) -> ActionResponseProbe:
    resolved_device = _resolve_probe_device(device)
    batch_rows = _validate_batch_rows(train_batch_rows, "train_batch_rows")
    probe = _initialize_probe(
        lambda: ActionResponseProbe(
            train_x.shape[1], train_y.shape[1], int(config["evaluation"]["probe_hidden_dim"])
        ),
        seed, resolved_device, rng_lock,
    )
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=float(config["evaluation"]["probe_learning_rate"])
    )
    if zero_action_x is not None and zero_action_x.shape != train_x.shape:
        raise ValueError("zero-action probe features must match action features")
    probe.zero_preserving = zero_action_x is not None
    row_count = int(train_x.shape[0])
    batch_rows = _training_rows(
        probe, train_x, train_y, batch_rows, zero_action_x, prefer_full_batch, progress_callback
    )
    steps = int(config["evaluation"]["probe_steps"])
    recovery = ProbeResume((Path(checkpoint_dir) / "response.pt") if checkpoint_dir is not None else None,
                           probe, optimizer, (train_x, train_y, zero_action_x), steps, batch_rows,
                           interval=checkpoint_interval)
    _progress(progress_callback, "training", recovery.start_step, steps, 0, row_count)
    if batch_rows is None or batch_rows >= row_count:
        x = torch.as_tensor(train_x, dtype=torch.float32, device=resolved_device)
        y = torch.as_tensor(train_y, dtype=torch.float32, device=resolved_device)
        zero = (
            torch.as_tensor(
                zero_action_x,
                dtype=torch.float32,
                device=resolved_device,
            )
            if zero_action_x is not None
            else None
        )
        for step in range(recovery.start_step, steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = probe(x) - probe(zero) if zero is not None else probe(x)
            loss = torch.mean((prediction - y) ** 2)
            loss.backward()
            optimizer.step()
            recovery.save(step + 1)
            _progress(progress_callback, "training", step + 1, steps, row_count, row_count)
        probe.eval()
        _finish_training(probe, progress_callback, steps, row_count)
        return probe

    x_cpu, y_cpu, zero_cpu = _training_tensors(
        (train_x, train_y, zero_action_x), resolved_device, prefer_full_batch
    )
    denominator = int(y_cpu.numel())
    for step in range(recovery.start_step, steps):
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, row_count, batch_rows):
            stop = min(start + batch_rows, row_count)
            x_batch = _training_slice(x_cpu, start, stop, resolved_device)
            y_batch = _training_slice(y_cpu, start, stop, resolved_device)
            zero_batch = (
                _training_slice(zero_cpu, start, stop, resolved_device)
                if zero_cpu is not None
                else None
            )
            prediction = (
                probe(x_batch) - probe(zero_batch)
                if zero_batch is not None
                else probe(x_batch)
            )
            squared_error_sum = torch.sum((prediction - y_batch) ** 2)
            (squared_error_sum / denominator).backward()
            _progress(progress_callback, "accumulating", step, steps, stop, row_count)
        optimizer.step()
        recovery.save(step + 1)
        _progress(progress_callback, "training", step + 1, steps, row_count, row_count)
    probe.eval()
    _finish_training(probe, progress_callback, steps, row_count)
    return probe


def predict_binary_probe(
    probe: CompatibilityProbe,
    features: np.ndarray,
    *,
    batch_rows: int | None = None,
) -> np.ndarray:
    probe.eval()
    rows = _validate_batch_rows(batch_rows, "batch_rows")
    device = _module_device(probe)
    with torch.no_grad():
        row_count = int(features.shape[0])
        if rows is None or rows >= row_count:
            values = torch.as_tensor(features, dtype=torch.float32, device=device)
            return torch.sigmoid(probe(values)).cpu().numpy()
        values_cpu = _cpu_float_tensor(features)
        predictions = []
        for start in range(0, row_count, rows):
            stop = min(start + rows, row_count)
            values = _transfer_float_batch(values_cpu[start:stop], device)
            predictions.append(torch.sigmoid(probe(values)).cpu().numpy())
        return np.concatenate(predictions, axis=0)


def predict_response_probe(
    probe: ActionResponseProbe,
    features: np.ndarray,
    zero_action_features: np.ndarray | None = None,
    *,
    batch_rows: int | None = None,
) -> np.ndarray:
    probe.eval()
    if probe.zero_preserving and zero_action_features is None:
        raise ValueError("zero-preserving probe prediction requires zero-action features")
    if not probe.zero_preserving and zero_action_features is not None:
        raise ValueError("ordinary probe cannot be evaluated as a zero-action contrast")
    rows = _validate_batch_rows(batch_rows, "batch_rows")
    device = _module_device(probe)
    with torch.no_grad():
        if (
            zero_action_features is not None
            and zero_action_features.shape != features.shape
        ):
            raise ValueError("zero-action probe features must match action features")
        row_count = int(features.shape[0])
        if rows is None or rows >= row_count:
            feature_tensor = torch.as_tensor(
                features, dtype=torch.float32, device=device
            )
            zero = (
                torch.as_tensor(
                    zero_action_features,
                    dtype=torch.float32,
                    device=device,
                )
                if zero_action_features is not None
                else None
            )
            values = probe(feature_tensor)
            if zero is not None:
                values = values - probe(zero)
            return values.cpu().numpy()
        feature_cpu = _cpu_float_tensor(features)
        zero_cpu = (
            _cpu_float_tensor(zero_action_features)
            if zero_action_features is not None
            else None
        )
        predictions = []
        for start in range(0, row_count, rows):
            stop = min(start + rows, row_count)
            feature_tensor = _transfer_float_batch(
                feature_cpu[start:stop], device
            )
            values = probe(feature_tensor)
            if zero_cpu is not None:
                zero = _transfer_float_batch(zero_cpu[start:stop], device)
                values = values - probe(zero)
            predictions.append(values.cpu().numpy())
        return np.concatenate(predictions, axis=0)


def _fit_binary(probe, features, labels, config, *, train_batch_rows=None, progress_callback=None, prefer_full_batch=False, checkpoint_path=None, checkpoint_interval=100):
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=float(config["evaluation"]["probe_learning_rate"])
    )
    device = _module_device(probe)
    rows = _validate_batch_rows(train_batch_rows, "train_batch_rows")
    rows = _training_rows(probe, features, labels, rows, None, prefer_full_batch, progress_callback)
    row_count = int(features.shape[0])
    steps = int(config["evaluation"]["probe_steps"])
    recovery = ProbeResume(checkpoint_path, probe, optimizer, (features, labels), steps, rows, interval=checkpoint_interval)
    _progress(progress_callback, "training", recovery.start_step, steps, 0, row_count)
    if rows is None or rows >= row_count:
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        y = torch.as_tensor(labels, dtype=torch.float32, device=device)
        for step in range(recovery.start_step, steps):
            optimizer.zero_grad(set_to_none=True)
            logits = probe(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            optimizer.step()
            recovery.save(step + 1)
            _progress(progress_callback, "training", step + 1, steps, row_count, row_count)
        probe.eval()
        _finish_training(probe, progress_callback, steps, row_count)
        return

    x_cpu, y_cpu = _training_tensors((features, labels), device, prefer_full_batch)
    denominator = int(y_cpu.numel())
    for step in range(recovery.start_step, steps):
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, row_count, rows):
            stop = min(start + rows, row_count)
            x_batch = _training_slice(x_cpu, start, stop, device)
            y_batch = _training_slice(y_cpu, start, stop, device)
            loss_sum = torch.nn.functional.binary_cross_entropy_with_logits(
                probe(x_batch),
                y_batch,
                reduction="sum",
            )
            (loss_sum / denominator).backward()
            _progress(progress_callback, "accumulating", step, steps, stop, row_count)
        optimizer.step()
        recovery.save(step + 1)
        _progress(progress_callback, "training", step + 1, steps, row_count, row_count)
    probe.eval()
    _finish_training(probe, progress_callback, steps, row_count)


def _cpu_float_tensor(values) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device="cpu")


def _transfer_float_batch(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    if values.device.type != "cpu":
        raise RuntimeError("probe transfer batches must be CPU-backed")
    return values.to(device=device)


def _resolve_probe_device(device: str | torch.device | None) -> torch.device:
    try:
        resolved = torch.device("cpu" if device is None else device)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid probe device: {device!r}") from error
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("probe device must be cpu or cuda")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA probe execution was requested but CUDA is unavailable")
        index = 0 if resolved.index is None else int(resolved.index)
        if index < 0 or index >= int(torch.cuda.device_count()):
            raise ValueError("probe CUDA device index is outside the visible inventory")
        resolved = torch.device("cuda", index)
    return resolved


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration as error:
        raise RuntimeError("cannot infer a device for a parameterless probe") from error


def _validate_batch_rows(value: int | None, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer or None")
    return value
