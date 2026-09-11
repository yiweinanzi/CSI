from __future__ import annotations

import copy

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as functional

HEAD_STEP_FLOOR = 50
DEFAULT_HEAD_STEPS_PER_LABELED_POINT = 50
DEFAULT_EARLY_STOP_PATIENCE = 50


class HeteroscedasticPositionHead(nn.Module):
    def __init__(self, representation_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(representation_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean = nn.Linear(hidden_dim, 2)
        self.raw_scale = nn.Linear(hidden_dim, 2)

    def forward(self, representation: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.network(representation)
        return self.mean(hidden), self.raw_scale(hidden)


class StandardizedPositionHead(HeteroscedasticPositionHead):
    """Keep source-fitted feature/coordinate units fixed during adaptation."""

    def __init__(self, features, positions, hidden_dim):
        super().__init__(features.shape[1], hidden_dim)
        for name, values in (("feature", features), ("position", positions)):
            self.register_buffer(name + "_mean", torch.as_tensor(values.mean(axis=0), dtype=torch.float32))
            self.register_buffer(name + "_scale", torch.as_tensor(np.maximum(values.std(axis=0), 1e-3), dtype=torch.float32))

    def forward(self, representation):
        return super().forward((representation - self.feature_mean) / self.feature_scale)


def scheduled_head_steps(config: dict, labeled_count: int) -> int:
    labeled = int(labeled_count)
    if labeled <= 0:
        return 0
    localization = config["localization"]
    head_steps = int(localization["head_steps"])
    per_point = int(
        localization.get("head_steps_per_labeled_point", DEFAULT_HEAD_STEPS_PER_LABELED_POINT)
    )
    return min(int(head_steps), max(HEAD_STEP_FLOOR, labeled * max(1, per_point)))


def _head_early_stop_patience(config: dict, scheduled_steps: int) -> int:
    localization = config["localization"]
    if int(scheduled_steps) >= int(localization["head_steps"]):
        return 0
    return int(localization.get("early_stop_patience", DEFAULT_EARLY_STOP_PATIENCE))


def _head_train_kwargs(config: dict, labeled_count: int) -> dict:
    localization = config["localization"]
    steps = scheduled_head_steps(config, labeled_count)
    return {
        "steps": steps,
        "learning_rate": float(localization["learning_rate"]),
        "sigma_min": float(localization["sigma_min"]),
        "ridge": float(localization["ridge"]),
        "early_stop_patience": _head_early_stop_patience(config, steps),
    }


def fit_source_position_head(
    representations: np.ndarray,
    positions: np.ndarray,
    config: dict,
    *,
    seed: int,
) -> HeteroscedasticPositionHead:
    torch.manual_seed(int(seed))
    hidden = max(8, int(config["model"]["hidden_dim"]) // 2)
    features = np.asarray(representations, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    head = (StandardizedPositionHead(features, positions, hidden)
            if config["localization"].get("standardize", False)
            else HeteroscedasticPositionHead(representations.shape[1], hidden))
    _optimize_head(
        head,
        features,
        np.asarray(positions, dtype=np.float32),
        **_head_train_kwargs(config, features.shape[0]),
    )
    return head


def adapt_position_head(
    source_head: HeteroscedasticPositionHead,
    support_representations: np.ndarray,
    support_positions: np.ndarray,
    config: dict,
) -> HeteroscedasticPositionHead:
    head = copy.deepcopy(source_head)
    if support_representations.size:
        features = np.asarray(support_representations, dtype=np.float32)
        _optimize_head(
            head,
            features,
            np.asarray(support_positions, dtype=np.float32),
            **_head_train_kwargs(config, features.shape[0]),
        )
    return head


def predict_position_distribution(
    head: HeteroscedasticPositionHead,
    representations: np.ndarray,
    sigma_min: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    head.eval()
    with torch.no_grad():
        mean, raw_scale = head(torch.as_tensor(representations, dtype=torch.float32))
        scale = functional.softplus(raw_scale) + float(sigma_min)
        variance = scale**2
        if isinstance(head, StandardizedPositionHead):
            mean = mean * head.position_scale + head.position_mean
            variance = variance * head.position_scale.square()
        uncertainty = torch.sum(torch.log(variance), dim=1)
    return mean.numpy(), variance.numpy(), uncertainty.numpy()


def _optimize_head(
    head: HeteroscedasticPositionHead,
    representations: np.ndarray,
    positions: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
    sigma_min: float,
    ridge: float,
    early_stop_patience: int = 0,
) -> int:
    head.train()
    x = torch.as_tensor(representations, dtype=torch.float32)
    y = torch.as_tensor(positions, dtype=torch.float32)
    if isinstance(head, StandardizedPositionHead):
        y = (y - head.position_mean) / head.position_scale
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(learning_rate),
        weight_decay=float(ridge),
    )
    taken = 0
    best = float("inf")
    stale = 0
    patience = int(early_stop_patience)
    best_state = None
    for _ in range(int(steps)):
        mean, raw_scale = head(x)
        scale = functional.softplus(raw_scale) + float(sigma_min)
        gaussian_nll = torch.mean(torch.sum(torch.log(scale) + 0.5 * ((y - mean) / scale) ** 2, dim=1))
        huber = functional.huber_loss(mean, y)
        loss = gaussian_nll + huber
        current = float(loss.detach())
        if not np.isfinite(current):
            raise FloatingPointError("Position-head objective became nonfinite")
        if current < best - 1e-8:
            best = current
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in head.state_dict().items()
            }
            stale = 0
        elif patience > 0:
            stale += 1
            if stale >= patience:
                break
        # The loss describes the state BEFORE optimizer.step(). Save that exact
        # state, not the next iterate (which may overshoot on a tiny support set).
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        taken += 1
    # Include the final updated iterate; there is no subsequent loop iteration
    # to score it. This adds only one forward pass to the optimization schedule.
    with torch.no_grad():
        mean, raw_scale = head(x)
        scale = functional.softplus(raw_scale) + float(sigma_min)
        final_loss = torch.mean(torch.sum(torch.log(scale) + 0.5 * ((y - mean) / scale) ** 2, dim=1)) + functional.huber_loss(mean, y)
        if not torch.isfinite(final_loss):
            raise FloatingPointError("Final position-head objective became nonfinite")
        if float(final_loss) < best:
            best_state = {name: tensor.detach().clone() for name, tensor in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return taken
