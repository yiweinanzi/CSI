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
) -> tuple[CompatibilityProbe, dict]:
    candidates = []
    for offset, family in enumerate(("linear", "mlp2")):
        torch.manual_seed(int(seed) + offset)
        probe = CompatibilityProbe(
            train_x.shape[1], family, int(config["evaluation"]["probe_hidden_dim"])
        )
        _fit_binary(probe, train_x, train_y, config)
        probabilities = predict_binary_probe(probe, selection_x)
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
) -> ActionResponseProbe:
    torch.manual_seed(int(seed))
    probe = ActionResponseProbe(
        train_x.shape[1], train_y.shape[1], int(config["evaluation"]["probe_hidden_dim"])
    )
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=float(config["evaluation"]["probe_learning_rate"])
    )
    x = torch.as_tensor(train_x, dtype=torch.float32)
    y = torch.as_tensor(train_y, dtype=torch.float32)
    for _ in range(int(config["evaluation"]["probe_steps"])):
        prediction = probe(x)
        loss = torch.mean((prediction - y) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    probe.eval()
    return probe


def predict_binary_probe(probe: CompatibilityProbe, features: np.ndarray) -> np.ndarray:
    probe.eval()
    with torch.no_grad():
        logits = probe(torch.as_tensor(features, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


def predict_response_probe(probe: ActionResponseProbe, features: np.ndarray) -> np.ndarray:
    probe.eval()
    with torch.no_grad():
        return probe(torch.as_tensor(features, dtype=torch.float32)).numpy()


def _fit_binary(probe, features, labels, config):
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=float(config["evaluation"]["probe_learning_rate"])
    )
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    for _ in range(int(config["evaluation"]["probe_steps"])):
        logits = probe(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    probe.eval()
