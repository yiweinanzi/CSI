from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .formal_metrics import binary_auroc, binary_nll


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
) -> tuple[CompatibilityProbe, dict]:
    resolved_device = _resolve_probe_device(device)
    batch_rows = _validate_batch_rows(train_batch_rows, "train_batch_rows")
    candidates = []
    for offset, family in enumerate(("linear", "mlp2")):
        torch.manual_seed(int(seed) + offset)
        probe = CompatibilityProbe(
            train_x.shape[1], family, int(config["evaluation"]["probe_hidden_dim"])
        ).to(device=resolved_device)
        _fit_binary(
            probe,
            train_x,
            train_y,
            config,
            train_batch_rows=batch_rows,
        )
        probabilities = predict_binary_probe(
            probe,
            selection_x,
            batch_rows=(batch_rows if resolved_device.type == "cuda" else None),
        )
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
) -> ActionResponseProbe:
    torch.manual_seed(int(seed))
    resolved_device = _resolve_probe_device(device)
    batch_rows = _validate_batch_rows(train_batch_rows, "train_batch_rows")
    probe = ActionResponseProbe(
        train_x.shape[1], train_y.shape[1], int(config["evaluation"]["probe_hidden_dim"])
    ).to(device=resolved_device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=float(config["evaluation"]["probe_learning_rate"])
    )
    if zero_action_x is not None and zero_action_x.shape != train_x.shape:
        raise ValueError("zero-action probe features must match action features")
    probe.zero_preserving = zero_action_x is not None
    row_count = int(train_x.shape[0])
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
        for _ in range(int(config["evaluation"]["probe_steps"])):
            optimizer.zero_grad(set_to_none=True)
            prediction = probe(x) - probe(zero) if zero is not None else probe(x)
            loss = torch.mean((prediction - y) ** 2)
            loss.backward()
            optimizer.step()
        probe.eval()
        return probe

    x_cpu = _cpu_float_tensor(train_x)
    y_cpu = _cpu_float_tensor(train_y)
    zero_cpu = (
        _cpu_float_tensor(zero_action_x) if zero_action_x is not None else None
    )
    denominator = int(y_cpu.numel())
    for _ in range(int(config["evaluation"]["probe_steps"])):
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, row_count, batch_rows):
            stop = min(start + batch_rows, row_count)
            x_batch = _transfer_float_batch(x_cpu[start:stop], resolved_device)
            y_batch = _transfer_float_batch(y_cpu[start:stop], resolved_device)
            zero_batch = (
                _transfer_float_batch(zero_cpu[start:stop], resolved_device)
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
        optimizer.step()
    probe.eval()
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


def _fit_binary(probe, features, labels, config, *, train_batch_rows=None):
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=float(config["evaluation"]["probe_learning_rate"])
    )
    device = _module_device(probe)
    rows = _validate_batch_rows(train_batch_rows, "train_batch_rows")
    row_count = int(features.shape[0])
    if rows is None or rows >= row_count:
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        y = torch.as_tensor(labels, dtype=torch.float32, device=device)
        for _ in range(int(config["evaluation"]["probe_steps"])):
            optimizer.zero_grad(set_to_none=True)
            logits = probe(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            optimizer.step()
        probe.eval()
        return

    x_cpu = _cpu_float_tensor(features)
    y_cpu = _cpu_float_tensor(labels)
    denominator = int(y_cpu.numel())
    for _ in range(int(config["evaluation"]["probe_steps"])):
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, row_count, rows):
            stop = min(start + rows, row_count)
            x_batch = _transfer_float_batch(x_cpu[start:stop], device)
            y_batch = _transfer_float_batch(y_cpu[start:stop], device)
            loss_sum = torch.nn.functional.binary_cross_entropy_with_logits(
                probe(x_batch),
                y_batch,
                reduction="sum",
            )
            (loss_sum / denominator).backward()
        optimizer.step()
    probe.eval()


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
