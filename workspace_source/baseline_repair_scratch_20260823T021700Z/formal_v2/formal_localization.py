from __future__ import annotations

import copy

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as functional


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


def fit_source_position_head(
    representations: np.ndarray,
    positions: np.ndarray,
    config: dict,
    *,
    seed: int,
) -> HeteroscedasticPositionHead:
    torch.manual_seed(int(seed))
    hidden = max(8, int(config["model"]["hidden_dim"]) // 2)
    head = HeteroscedasticPositionHead(representations.shape[1], hidden)
    _optimize_head(
        head,
        np.asarray(representations, dtype=np.float32),
        np.asarray(positions, dtype=np.float32),
        steps=int(config["localization"]["head_steps"]),
        learning_rate=float(config["localization"]["learning_rate"]),
        sigma_min=float(config["localization"]["sigma_min"]),
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
        _optimize_head(
            head,
            np.asarray(support_representations, dtype=np.float32),
            np.asarray(support_positions, dtype=np.float32),
            steps=int(config["localization"]["head_steps"]),
            learning_rate=float(config["localization"]["learning_rate"]),
            sigma_min=float(config["localization"]["sigma_min"]),
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
) -> None:
    head.train()
    x = torch.as_tensor(representations, dtype=torch.float32)
    y = torch.as_tensor(positions, dtype=torch.float32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(learning_rate))
    for _ in range(int(steps)):
        mean, raw_scale = head(x)
        scale = functional.softplus(raw_scale) + float(sigma_min)
        gaussian_nll = torch.mean(torch.sum(torch.log(scale) + 0.5 * ((y - mean) / scale) ** 2, dim=1))
        huber = functional.huber_loss(mean, y)
        loss = gaussian_nll + huber
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    head.eval()
