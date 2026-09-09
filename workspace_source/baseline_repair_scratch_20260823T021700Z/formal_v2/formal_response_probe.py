from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


RESPONSE_PROBE_SCHEMA = "csi-pairs-coordinate-preserving-response-probe-v1"


def coordinate_preserving_action_features(action: np.ndarray) -> np.ndarray:
    """Encode every typed action plane without discarding its map coordinates."""
    values = np.asarray(action, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("typed action must have shape [plane, row, column]")
    size = int(values.shape[-1])
    axis = (np.arange(size, dtype=np.float64) + 0.5) / size * 2.0 - 1.0
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    output = []
    for plane in values:
        weight = np.abs(plane)
        support = weight > 1e-12
        total = float(np.sum(weight))
        if total > 0.0:
            center_x = float(np.sum(weight * xx) / total)
            center_y = float(np.sum(weight * yy) / total)
            scale_x = float(np.sqrt(np.sum(weight * (xx - center_x) ** 2) / total))
            scale_y = float(np.sqrt(np.sum(weight * (yy - center_y) ** 2) / total))
            rows, columns = np.nonzero(support)
            bbox = (
                float(axis[int(columns.min())]),
                float(axis[int(columns.max())]),
                float(axis[int(rows.min())]),
                float(axis[int(rows.max())]),
            )
            signed_mean = float(np.sum(plane) / total)
            magnitude_mean = float(np.mean(weight[support]))
            magnitude_max = float(np.max(weight[support]))
        else:
            center_x = center_y = scale_x = scale_y = 0.0
            bbox = (0.0, 0.0, 0.0, 0.0)
            signed_mean = magnitude_mean = magnitude_max = 0.0
        output.extend(
            (
                float(np.mean(support)),
                float(np.log1p(total)),
                signed_mean,
                magnitude_mean,
                magnitude_max,
                center_x,
                center_y,
                scale_x,
                scale_y,
                *bbox,
            )
        )
    return np.asarray(output, dtype=np.float32)


def coordinate_preserving_map_features(
    source_map: np.ndarray,
    *,
    material_categories: int,
    pooled_size: int = 16,
) -> np.ndarray:
    """Pool raw semantic map channels on a fixed coordinate grid."""
    values = np.asarray(source_map, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 3 or values.shape[1] != values.shape[2]:
        raise ValueError("source map must have shape [channel, row, column]")
    if values.shape[-1] % int(pooled_size):
        raise ValueError("pooled map size must divide the source map size")
    occupancy = values[0]
    height = np.clip(values[1] / 120.0, 0.0, 1.0)
    material = values[2]
    if not np.allclose(material, np.round(material)):
        raise ValueError("material map must be categorical")
    channels = [occupancy, height]
    channels.extend(material == category for category in range(int(material_categories)))
    stacked = np.stack(channels).astype(np.float32)
    factor = values.shape[-1] // int(pooled_size)
    pooled = stacked.reshape(
        stacked.shape[0], pooled_size, factor, pooled_size, factor
    ).mean(axis=(2, 4))
    return pooled.reshape(-1)


def normalized_radio_bs_context(radio_config: np.ndarray, bs_pose: np.ndarray) -> np.ndarray:
    radio = np.asarray(radio_config, dtype=np.float64).ravel().copy()
    pose = np.asarray(bs_pose, dtype=np.float64).ravel().copy()
    if radio.size < 4 or pose.size != 7:
        raise ValueError("radio/BS context has the wrong shape")
    radio[:4] /= np.asarray([1.0e10, 16.0, 128.0, 1.0e6])
    pose[:3] /= 256.0
    return np.concatenate((radio, pose)).astype(np.float32)


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(output_dim)),
            nn.LayerNorm(int(output_dim)),
            nn.GELU(),
            nn.Linear(int(output_dim), int(output_dim)),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class CoordinateResponseProbe(nn.Module):
    """Nonlinear protocol probe with an exact zero-action contrast."""

    def __init__(
        self,
        *,
        csi_dim: int,
        map_dim: int,
        action_dim: int,
        context_dim: int,
        query_dim: int,
        output_dim: int,
        hidden_dim: int = 192,
        include_csi: bool = True,
        include_map: bool = True,
        include_action: bool = True,
        include_position: bool = False,
        zero_preserving: bool = True,
    ):
        super().__init__()
        width = int(hidden_dim)
        self.include_csi = bool(include_csi)
        self.include_map = bool(include_map)
        self.include_action = bool(include_action)
        self.include_position = bool(include_position)
        self.zero_preserving = bool(zero_preserving)
        self.csi_encoder = _Encoder(csi_dim, width) if self.include_csi else None
        self.map_encoder = _Encoder(map_dim, width) if self.include_map else None
        self.action_encoder = _Encoder(action_dim, width) if self.include_action else None
        side_dim = int(context_dim) + int(query_dim) + (2 if self.include_position else 0)
        self.side_encoder = _Encoder(side_dim, width)
        block_count = (
            int(self.include_csi)
            + int(self.include_map)
            + int(self.include_action)
            + 1
        )
        interaction_count = int(self.include_action) * (
            int(self.include_csi) + int(self.include_map) + 1
        )
        self.head = nn.Sequential(
            nn.Linear(width * (block_count + interaction_count), 2 * width),
            nn.LayerNorm(2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, int(output_dim)),
        )

    def _response(
        self,
        csi: torch.Tensor,
        source_map: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
        query: torch.Tensor,
        position: torch.Tensor | None,
    ) -> torch.Tensor:
        side_values = [context, query]
        if self.include_position:
            if position is None:
                raise ValueError("oracle response probe requires receiver position")
            side_values.append(position)
        side = self.side_encoder(torch.cat(side_values, dim=1))
        blocks = []
        csi_latent = self.csi_encoder(csi) if self.csi_encoder is not None else None
        map_latent = self.map_encoder(source_map) if self.map_encoder is not None else None
        action_latent = self.action_encoder(action) if self.action_encoder is not None else None
        if csi_latent is not None:
            blocks.append(csi_latent)
        if map_latent is not None:
            blocks.append(map_latent)
        if action_latent is not None:
            blocks.append(action_latent)
        blocks.append(side)
        if action_latent is not None:
            if csi_latent is not None:
                blocks.append(action_latent * csi_latent)
            if map_latent is not None:
                blocks.append(action_latent * map_latent)
            blocks.append(action_latent * side)
        return self.head(torch.cat(blocks, dim=1))

    def forward(
        self,
        csi: torch.Tensor,
        source_map: torch.Tensor,
        action: torch.Tensor,
        zero_action: torch.Tensor,
        context: torch.Tensor,
        query: torch.Tensor,
        position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction = self._response(csi, source_map, action, context, query, position)
        if self.zero_preserving:
            prediction = prediction - self._response(
                csi, source_map, zero_action, context, query, position
            )
        return prediction


@dataclass(frozen=True)
class ProbeTrainingConfig:
    steps: int = 4000
    batch_size: int = 512
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    hidden_dim: int = 192


def fit_coordinate_response_probe(
    probe: CoordinateResponseProbe,
    arrays: dict[str, np.ndarray],
    *,
    bank: np.ndarray,
    route: np.ndarray,
    seed: int,
    config: ProbeTrainingConfig,
    device: str | torch.device,
) -> dict:
    """Fit with bank-balanced and route-balanced minibatches."""
    target = np.asarray(arrays["target"], dtype=np.float32)
    routes = np.asarray(route, dtype=np.int64)
    banks = np.asarray(bank, dtype=np.int64)
    if target.shape[0] != routes.size or routes.shape != banks.shape:
        raise ValueError("training target/bank/route lengths differ")
    strata = {
        (int(bank_value), int(route_value)): np.flatnonzero(
            (banks == bank_value) & (routes == route_value)
        )
        for bank_value in np.unique(banks)
        for route_value in (0, 1, 2)
    }
    strata = {key: value for key, value in strata.items() if value.size}
    if not strata or not any(key[1] == 0 for key in strata) or not any(
        key[1] == 2 for key in strata
    ):
        raise ValueError("response probe training requires nonempty null and active strata")
    execution_device = torch.device(device)
    probe.to(execution_device)
    probe.train()
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    generator = np.random.default_rng(int(seed))
    keys = sorted(strata)
    route_weight = {0: 0.35, 1: 0.15, 2: 0.50}
    key_probability = np.asarray(
        [route_weight[key[1]] / sum(other[1] == key[1] for other in keys) for key in keys],
        dtype=np.float64,
    )
    key_probability /= key_probability.sum()
    losses = []
    for step in range(int(config.steps)):
        chosen_keys = generator.choice(len(keys), size=int(config.batch_size), p=key_probability)
        indices = np.empty(int(config.batch_size), dtype=np.int64)
        for offset, key_index in enumerate(chosen_keys):
            candidates = strata[keys[int(key_index)]]
            indices[offset] = candidates[generator.integers(candidates.size)]
        batch = {
            name: torch.as_tensor(value[indices], dtype=torch.float32, device=execution_device)
            for name, value in arrays.items()
        }
        prediction = probe(
            batch["csi"],
            batch["map"],
            batch["action"],
            batch["zero_action"],
            batch["context"],
            batch["query"],
            batch.get("position"),
        )
        error = torch.mean((prediction - batch["target"]) ** 2, dim=1)
        loss = torch.mean(error)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 5.0)
        optimizer.step()
        if step % 100 == 0 or step + 1 == int(config.steps):
            losses.append({"step": step + 1, "mse": float(loss.detach().cpu())})
    probe.eval()
    return {"losses": losses, "strata": {str(key): int(value.size) for key, value in strata.items()}}


def predict_coordinate_response_probe(
    probe: CoordinateResponseProbe,
    arrays: dict[str, np.ndarray],
    *,
    device: str | torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    output = []
    execution_device = torch.device(device)
    probe.to(execution_device)
    probe.eval()
    with torch.no_grad():
        for start in range(0, arrays["csi"].shape[0], int(batch_size)):
            stop = min(start + int(batch_size), arrays["csi"].shape[0])
            batch = {
                name: torch.as_tensor(value[start:stop], dtype=torch.float32, device=execution_device)
                for name, value in arrays.items()
                if name != "target"
            }
            prediction = probe(
                batch["csi"],
                batch["map"],
                batch["action"],
                batch["zero_action"],
                batch["context"],
                batch["query"],
                batch.get("position"),
            )
            output.append(prediction.cpu().numpy())
    return np.concatenate(output, axis=0)


def save_coordinate_response_probe(
    path: str | Path,
    probe: CoordinateResponseProbe,
    metadata: dict,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite response probe: {target}")
    torch.save(
        {
            "schema_version": RESPONSE_PROBE_SCHEMA,
            "state_dict": {name: value.detach().cpu() for name, value in probe.state_dict().items()},
            "metadata": dict(metadata),
        },
        target,
    )
