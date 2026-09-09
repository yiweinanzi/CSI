from __future__ import annotations

from dataclasses import dataclass
import os

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as functional
except ImportError:
    torch = None
    Tensor = object
    nn = None
    functional = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("formal V6 training requires PyTorch")


def resolve_execution_device(dataset, requested: str | None = None):
    """Resolve the reviewed training device and fail closed for formal data."""
    require_torch()
    value = requested if requested is not None else os.environ.get("CSI_PAIRS_DEVICE")
    if value is None and os.environ.get("CSI_PAIRS_DEVICES"):
        value = os.environ["CSI_PAIRS_DEVICES"].split(",", 1)[0].strip()
    if value is None:
        value = "cpu" if bool(dataset.is_fixture) else "cuda:0"
    try:
        device = torch.device(str(value))
    except (RuntimeError, TypeError) as error:
        raise RuntimeError(f"invalid CSI_PAIRS_DEVICE value: {value!r}") from error
    if device.type not in {"cpu", "cuda"}:
        raise RuntimeError("CSI-PAIRS formal execution supports only cpu or cuda devices")
    if not bool(dataset.is_fixture) and device.type != "cuda":
        raise RuntimeError("non-fixture formal execution requires an NVIDIA CUDA device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but the reviewed PyTorch runtime cannot initialize CUDA"
            )
        index = 0 if device.index is None else int(device.index)
        if index < 0 or index >= int(torch.cuda.device_count()):
            raise RuntimeError(
                f"requested CUDA device index {index} is outside the visible device inventory"
            )
        device = torch.device("cuda", index)
    return device


def resolve_execution_devices(dataset) -> tuple:
    configured = os.environ.get("CSI_PAIRS_DEVICES")
    if configured is None:
        return (resolve_execution_device(dataset),)
    values = [value.strip() for value in configured.split(",")]
    if not values or any(not value for value in values):
        raise RuntimeError("CSI_PAIRS_DEVICES must be a comma-separated device list")
    devices = tuple(resolve_execution_device(dataset, value) for value in values)
    if len(set(devices)) != len(devices):
        raise RuntimeError("CSI_PAIRS_DEVICES cannot contain duplicate devices")
    if len(devices) > 1 and any(device.type != "cuda" for device in devices):
        raise RuntimeError("multi-device CSI-PAIRS execution requires only CUDA devices")
    return devices


def module_device(module):
    require_torch()
    try:
        return next(module.parameters()).device
    except StopIteration as error:
        raise RuntimeError("cannot infer a device from a parameterless module") from error


def tensor_for_module(module, values, *, dtype=None):
    return torch.as_tensor(values, dtype=dtype, device=module_device(module))


def batch_for_module(module, batch: dict) -> dict:
    device = module_device(module)
    return {
        key: value.to(device=device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def portable_state_dict(module) -> dict:
    return {
        key: value.detach().to(device="cpu", copy=True)
        for key, value in module.state_dict().items()
    }


if nn is not None:

    class _SpatialTokenEncoder(nn.Module):
        def __init__(self, input_channels: int, output_dim: int, hidden_dim: int):
            super().__init__()
            width = max(8, hidden_dim // 4)
            self.network = nn.Sequential(
                nn.Conv2d(input_channels, width, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(width, output_dim, kernel_size=3, padding=1),
                nn.GELU(),
                _DeterministicAdaptiveAvgPool2d((4, 4)),
            )

        def forward(self, values: Tensor) -> Tensor:
            encoded = self.network(values)
            return encoded.flatten(2).transpose(1, 2)


    class _DeterministicAdaptiveAvgPool2d(nn.Module):
        def __init__(self, output_size: tuple[int, int]):
            super().__init__()
            self.output_size = tuple(int(value) for value in output_size)

        def forward(self, values: Tensor) -> Tensor:
            output_rows, output_columns = self.output_size
            input_rows, input_columns = values.shape[-2:]
            if input_rows < output_rows or input_columns < output_columns:
                raise ValueError("spatial inputs must be at least as large as the pooled grid")
            rows = []
            for row in range(output_rows):
                row_start = (row * input_rows) // output_rows
                row_end = ((row + 1) * input_rows + output_rows - 1) // output_rows
                columns = []
                for column in range(output_columns):
                    column_start = (column * input_columns) // output_columns
                    column_end = (
                        ((column + 1) * input_columns + output_columns - 1)
                        // output_columns
                    )
                    columns.append(
                        values[
                            ...,
                            row_start:row_end,
                            column_start:column_end,
                        ].mean(dim=(-2, -1))
                    )
                rows.append(torch.stack(columns, dim=-1))
            return torch.stack(rows, dim=-2)


    class CSIPairsFormalModel(nn.Module):
        """V6 F/P contract with patch CSI, typed maps/actions, radio state and query."""

        def __init__(
            self,
            *,
            patch_count: int,
            patch_rows: int = 1,
            patch_columns: int | None = None,
            patch_dim: int,
            map_channels: int,
            action_channels: int,
            radio_dim: int,
            latent_dim: int,
            state_dim: int,
            map_dim: int,
            hidden_dim: int,
            attention_heads: int,
            csi_encoder_layers: int = 1,
        ):
            super().__init__()
            if state_dim % attention_heads:
                raise ValueError("state_dim must be divisible by attention_heads")
            self.patch_count = int(patch_count)
            self.patch_rows = int(patch_rows)
            self.patch_columns = (
                self.patch_count // self.patch_rows
                if patch_columns is None
                else int(patch_columns)
            )
            if self.patch_rows * self.patch_columns != self.patch_count:
                raise ValueError("patch_rows x patch_columns must equal patch_count")
            if int(latent_dim) != int(state_dim):
                raise ValueError("V6 requires F to inherit the complete teacher CSI encoder")
            self.patch_dim = int(patch_dim)
            self.latent_dim = int(latent_dim)
            self.state_dim = int(state_dim)
            self.csi_patch_embedding = nn.Linear(patch_dim, state_dim)
            self.csi_mask_token = nn.Parameter(torch.zeros(state_dim))
            self.csi_row_position = nn.Parameter(torch.zeros(1, self.patch_rows, state_dim))
            self.csi_column_position = nn.Parameter(torch.zeros(1, self.patch_columns, state_dim))
            csi_layer = nn.TransformerEncoderLayer(
                d_model=state_dim,
                nhead=attention_heads,
                dim_feedforward=max(4 * state_dim, 16),
                batch_first=True,
                norm_first=True,
                dropout=0.0,
                activation="gelu",
            )
            self.csi_encoder = nn.TransformerEncoder(
                csi_layer, num_layers=int(csi_encoder_layers)
            )
            self.map_encoder = _SpatialTokenEncoder(map_channels, map_dim, hidden_dim)
            self.map_projection = nn.Linear(map_dim, state_dim)
            self.radio_encoder = nn.Sequential(
                nn.Linear(radio_dim, state_dim), nn.LayerNorm(state_dim), nn.GELU()
            )
            self.fusion = nn.MultiheadAttention(
                embed_dim=state_dim,
                num_heads=attention_heads,
                batch_first=True,
                dropout=0.0,
            )
            self.state_norm = nn.LayerNorm(state_dim)
            self.action_encoder = _SpatialTokenEncoder(action_channels, map_dim, hidden_dim)
            self.action_pool = nn.Sequential(
                nn.Linear(map_dim, map_dim), nn.LayerNorm(map_dim), nn.GELU()
            )
            self.query_embedding = nn.Embedding(patch_count, state_dim)
            self.predictor = nn.Sequential(
                nn.Linear(state_dim + map_dim + state_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.latent_head = nn.Linear(hidden_dim, latent_dim)
            self.physical_head = nn.Linear(hidden_dim, patch_dim)

        def initialize_csi_from_teacher(self, teacher) -> None:
            if teacher.patch_embedding.weight.shape != self.csi_patch_embedding.weight.shape:
                raise ValueError("teacher and F CSI encoders are not shape-compatible")
            if teacher.row_position.shape != self.csi_row_position.shape:
                raise ValueError("teacher and F antenna positions are not shape-compatible")
            if teacher.column_position.shape != self.csi_column_position.shape:
                raise ValueError("teacher and F subcarrier positions are not shape-compatible")
            self.csi_patch_embedding.load_state_dict(teacher.patch_embedding.state_dict())
            self.csi_encoder.load_state_dict(teacher.encoder.state_dict(), strict=True)
            with torch.no_grad():
                self.csi_mask_token.copy_(teacher.mask_token)
                self.csi_row_position.copy_(teacher.row_position)
                self.csi_column_position.copy_(teacher.column_position)

        def csi_position(self) -> Tensor:
            return (
                self.csi_row_position[:, :, None, :]
                + self.csi_column_position[:, None, :, :]
            ).reshape(1, self.patch_count, self.state_dim)

        def state(self, visible_patches: Tensor, maps: Tensor, radio: Tensor, masks: Tensor) -> Tensor:
            if masks.dtype != torch.bool or masks.shape != visible_patches.shape[:2]:
                raise ValueError("masks must be boolean [batch, patch]")
            csi_tokens = self.csi_patch_embedding(visible_patches)
            csi_tokens = torch.where(masks[..., None], self.csi_mask_token[None, None, :], csi_tokens)
            csi_tokens = self.csi_encoder(csi_tokens + self.csi_position())
            map_tokens = self.map_projection(self.map_encoder(maps))
            radio_token = self.radio_encoder(radio)[:, None, :]
            context = torch.cat((map_tokens, radio_token), dim=1)
            fused, _ = self.fusion(csi_tokens, context, context, need_weights=False)
            return self.state_norm(csi_tokens + fused)

        def predict(self, state: Tensor, signed_edit: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
            if query.ndim != 1 or query.shape[0] != state.shape[0]:
                raise ValueError("query must have shape [batch]")
            batch = torch.arange(state.shape[0], device=state.device)
            query_state = state[batch, query]
            action = self.action_encoder(signed_edit).mean(dim=1)
            action = self.action_pool(action)
            features = self.predictor(
                torch.cat((query_state, action, self.query_embedding(query)), dim=1)
            )
            return self.latent_head(features), self.physical_head(features)

        def retained_representation(
            self,
            patches: Tensor,
            maps: Tensor,
            radio: Tensor,
        ) -> Tensor:
            masks = torch.zeros(patches.shape[:2], dtype=torch.bool, device=patches.device)
            return self.state(patches, maps, radio, masks).mean(dim=1)


@dataclass(frozen=True)
class LossWeights:
    endpoint_physical: float
    natural_endpoint: float
    alignment: float
    alignment_null: float
    response: float
    response_physical: float
    response_delta: float
    response_delta_physical: float
    response_null: float
    response_null_physical: float
    active_margin: float
    effect_margin_scale: float
    effect_margin_cap: float
    alignment_primary_margin: str
    alignment_null_tolerance: float
    response_latent_null_tolerance: float
    response_physical_null_tolerance: float
    alignment_scale: float
    response_scale: float


def squared_rms_error(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.mean((prediction - target) ** 2, dim=-1)


def endpoint_per_sample(
    prediction_z: Tensor,
    prediction_y: Tensor,
    target_z: Tensor,
    target_y: Tensor,
    physical_weight: float,
) -> Tensor:
    return squared_rms_error(prediction_z, target_z) + float(physical_weight) * squared_rms_error(
        prediction_y, target_y
    )


def alignment_active_quartet_loss(
    score_uu: Tensor,
    score_uv: Tensor,
    score_vv: Tensor,
    score_vu: Tensor,
    margin: float | Tensor,
) -> Tensor:
    return functional.relu(margin - score_uu + score_uv) + functional.relu(
        margin - score_vv + score_vu
    )


def alignment_null_quartet_loss(
    score_uu: Tensor,
    score_uv: Tensor,
    score_vv: Tensor,
    score_vu: Tensor,
    tolerance: float,
) -> Tensor:
    first = functional.relu(torch.abs(score_uu - score_uv) - float(tolerance)) ** 2
    second = functional.relu(torch.abs(score_vv - score_vu) - float(tolerance)) ** 2
    return first + second


def response_component_losses(
    prediction_z: Tensor,
    prediction_y: Tensor,
    identity_z: Tensor,
    identity_y: Tensor,
    target_z: Tensor,
    target_y: Tensor,
    source_z: Tensor,
    source_y: Tensor,
    route: Tensor,
    weights: LossWeights,
) -> dict[str, Tensor]:
    target = squared_rms_error(prediction_z, target_z)
    physical = squared_rms_error(prediction_y, target_y)
    predicted_delta_z = prediction_z - identity_z
    predicted_delta_y = prediction_y - identity_y
    true_delta_z = target_z - source_z
    true_delta_y = target_y - source_y
    delta = squared_rms_error(predicted_delta_z, true_delta_z) + float(
        weights.response_delta_physical
    ) * squared_rms_error(predicted_delta_y, true_delta_y)
    active = route == 2
    null = route == 0
    active_delta = required_mean(delta[active], "response active patch")
    latent_norm = torch.sqrt(torch.mean(predicted_delta_z**2, dim=-1) + 1e-12)
    physical_norm = torch.sqrt(torch.mean(predicted_delta_y**2, dim=-1) + 1e-12)
    null_penalty = (
        functional.relu(latent_norm - float(weights.response_latent_null_tolerance)) ** 2
        + float(weights.response_null_physical)
        * functional.relu(physical_norm - float(weights.response_physical_null_tolerance)) ** 2
    )
    return {
        "target": torch.mean(target),
        "physical": torch.mean(physical),
        "active_delta": active_delta,
        "null": required_mean(null_penalty[null], "response null patch"),
    }


def required_mean(values: Tensor, name: str) -> Tensor:
    if values.numel() == 0:
        raise RuntimeError(f"empty conditional mean: {name}")
    return torch.mean(values)
