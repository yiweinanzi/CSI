from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .formal_model import module_device, portable_state_dict
from .formal_protocol import MaskQuery, PatchSpec, frozen_mask_query_bank, patchify_csi


TEACHER_CHECKPOINT_SCHEMA = "csi-pairs-stage0-teacher-v2.3-v6"
PRETRAINING_MASK_SAMPLER = "independent_per_sample_without_replacement"


class CSIMaskedTeacher(nn.Module):
    """CSI-only asymmetric 2D masked autoencoder."""

    def __init__(
        self,
        patch_rows: int,
        patch_columns: int,
        patch_dim: int,
        latent_dim: int,
        heads: int,
        encoder_layers: int,
        decoder_layers: int,
    ):
        super().__init__()
        if latent_dim % heads:
            raise ValueError("teacher latent_dim must be divisible by attention heads")
        self.patch_rows = int(patch_rows)
        self.patch_columns = int(patch_columns)
        self.patch_count = self.patch_rows * self.patch_columns
        self.patch_dim = int(patch_dim)
        self.latent_dim = int(latent_dim)
        self.patch_embedding = nn.Linear(patch_dim, latent_dim)
        self.mask_token = nn.Parameter(torch.zeros(latent_dim))
        self.row_position = nn.Parameter(torch.zeros(1, self.patch_rows, latent_dim))
        self.column_position = nn.Parameter(torch.zeros(1, self.patch_columns, latent_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=heads,
            dim_feedforward=max(4 * latent_dim, 16),
            batch_first=True,
            norm_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(encoder_layers))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=heads,
            dim_feedforward=max(4 * latent_dim, 16),
            batch_first=True,
            norm_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.decoder_transformer = nn.TransformerEncoder(
            decoder_layer, num_layers=int(decoder_layers)
        )
        self.decoder_norm = nn.LayerNorm(latent_dim)
        self.decoder_head = nn.Linear(latent_dim, patch_dim)

    def positional_encoding(self) -> Tensor:
        return (
            self.row_position[:, :, None, :] + self.column_position[:, None, :, :]
        ).reshape(1, self.patch_count, self.latent_dim)

    def encode(self, patches: Tensor, mask: Tensor) -> Tensor:
        if patches.ndim != 3 or patches.shape[1:] != (self.patch_count, self.patch_dim):
            raise ValueError("teacher patches have the wrong shape")
        if mask.shape != patches.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("teacher mask must be boolean [batch, patch]")
        tokens = self.patch_embedding(patches) + self.positional_encoding()
        encoded_rows = []
        for batch_index in range(tokens.shape[0]):
            visible = ~mask[batch_index]
            if not bool(torch.any(visible)):
                raise ValueError("teacher mask cannot hide every patch")
            encoded_visible = self.encoder(tokens[batch_index : batch_index + 1, visible])
            full = self.mask_token[None, :].expand(self.patch_count, -1).clone()
            full[visible] = encoded_visible[0]
            encoded_rows.append(full)
        return torch.stack(encoded_rows, dim=0)

    def encode_full(self, patches: Tensor) -> Tensor:
        mask = torch.zeros(patches.shape[:2], dtype=torch.bool, device=patches.device)
        return self.encode(patches, mask)

    def forward(self, patches: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(patches, mask)
        decoded = self.decoder_transformer(latent + self.positional_encoding())
        return latent, self.decoder_head(self.decoder_norm(decoded))


class CSIReadout(nn.Module):
    def __init__(self, latent_dim: int, patch_dim: int):
        super().__init__()
        self.network = nn.Linear(latent_dim, patch_dim)

    def forward(self, latent: Tensor) -> Tensor:
        return self.network(latent)


@dataclass(frozen=True)
class TeacherBundle:
    teacher: CSIMaskedTeacher
    readout: CSIReadout
    patch_spec: PatchSpec
    pretrain_mask_bank: tuple[MaskQuery, ...]
    mask_bank: tuple[MaskQuery, ...]
    audit_mask_bank: tuple[MaskQuery, ...]
    reconstruction_nmse: float
    readout_nmse: float


def random_teacher_pretraining_masks(
    rng: np.random.Generator,
    batch_size: int,
    patch_count: int,
    fraction: float,
) -> np.ndarray:
    """Sample an independent exact-cardinality random mask for every example."""
    batch = int(batch_size)
    patches = int(patch_count)
    if batch < 1:
        raise ValueError("teacher mask batch_size must be positive")
    if patches < 2:
        raise ValueError("teacher masking requires at least two patches")
    hidden_exact = float(fraction) * patches
    if not hidden_exact.is_integer():
        raise ValueError("teacher masking requires an exact mask cardinality")
    hidden = int(hidden_exact)
    if hidden < 1 or hidden >= patches:
        raise ValueError("teacher mask fraction must leave at least one visible patch")
    masks = np.zeros((batch, patches), dtype=np.bool_)
    for index in range(batch):
        masks[index, rng.choice(patches, size=hidden, replace=False)] = True
    return masks


def train_teacher_bundle(
    csi: np.ndarray,
    patch_spec: PatchSpec,
    config: dict,
    *,
    seed: int,
    device: str | torch.device = "cpu",
) -> TeacherBundle:
    teacher_config = config["teacher"]
    model_config = config["model"]
    latent_dim = int(teacher_config["latent_dim"])
    heads = _compatible_heads(latent_dim, int(model_config["attention_heads"]))
    torch.manual_seed(int(seed))
    execution_device = torch.device(device)
    teacher = CSIMaskedTeacher(
        patch_spec.patch_rows,
        patch_spec.patch_columns,
        patch_spec.patch_dim,
        latent_dim,
        heads,
        int(teacher_config["encoder_layers"]),
        int(teacher_config["decoder_layers"]),
    ).to(execution_device)
    patches = patchify_csi(np.asarray(csi, dtype=np.float32), patch_spec).reshape(
        -1, patch_spec.patch_count, patch_spec.patch_dim
    )
    tensor = torch.as_tensor(patches, dtype=torch.float32, device=execution_device)
    mask_bank = frozen_mask_query_bank(
        patch_spec, int(model_config["mask_bank_seed"])
    )
    pretrain_mask_bank = tuple(
        entry
        for entry in mask_bank
        if entry.mode == "random_75"
    )
    audit_mask_bank = frozen_mask_query_bank(
        patch_spec, int(model_config["mask_bank_seed"]) + 1
    )
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=float(teacher_config["learning_rate"]))
    rng = np.random.default_rng(int(seed) + 43001)
    batch_size = min(int(teacher_config["batch_size"]), tensor.shape[0])
    for step in range(int(teacher_config["steps"])):
        indices = rng.integers(0, tensor.shape[0], size=batch_size)
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=execution_device)
        batch = tensor[index_tensor]
        masks = random_teacher_pretraining_masks(
            rng,
            batch_size,
            patch_spec.patch_count,
            float(teacher_config["mask_fraction"]),
        )
        mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=execution_device)
        _, prediction = teacher(batch, mask_tensor)
        error = (prediction - batch) ** 2
        loss = torch.mean(error[mask_tensor])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        audit_masks = torch.as_tensor(
            np.stack(
                [
                    pretrain_mask_bank[index % len(pretrain_mask_bank)].mask
                    for index in range(tensor.shape[0])
                ]
            ),
            dtype=torch.bool,
            device=execution_device,
        )
        latent_masked, reconstruction = teacher(tensor, audit_masks)
        reconstruction_nmse = _masked_nmse(tensor, reconstruction, audit_masks)
        full_latent = teacher.encode_full(tensor)

    torch.manual_seed(int(seed) + 1)
    readout = CSIReadout(latent_dim, patch_spec.patch_dim).to(execution_device)
    readout_optimizer = torch.optim.AdamW(readout.parameters(), lr=float(teacher_config["learning_rate"]))
    for _ in range(int(teacher_config["steps"])):
        prediction = readout(full_latent)
        loss = torch.mean((prediction - tensor) ** 2)
        readout_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        readout_optimizer.step()
    readout.eval()
    for parameter in readout.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        readout_nmse = _nmse(tensor, readout(full_latent))
    return TeacherBundle(
        teacher=teacher,
        readout=readout,
        patch_spec=patch_spec,
        pretrain_mask_bank=pretrain_mask_bank,
        mask_bank=mask_bank,
        audit_mask_bank=audit_mask_bank,
        reconstruction_nmse=reconstruction_nmse,
        readout_nmse=readout_nmse,
    )


def save_teacher_bundle(path: str | Path, bundle: TeacherBundle, config: dict, seed: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": TEACHER_CHECKPOINT_SCHEMA,
            "seed": int(seed),
            "patch_spec": {
                "antennas": bundle.patch_spec.antennas,
                "subcarriers": bundle.patch_spec.subcarriers,
                "patch_complex_size": bundle.patch_spec.patch_complex_size,
                "patch_antenna_size": bundle.patch_spec.patch_antenna_size,
                "patch_subcarrier_size": bundle.patch_spec.patch_subcarrier_size,
            },
            "teacher_config": dict(config["teacher"]),
            "model_attention_heads": int(config["model"]["attention_heads"]),
            "pretraining_mask_sampler": {
                "sampler": PRETRAINING_MASK_SAMPLER,
                "fraction": float(config["teacher"]["mask_fraction"]),
                "hidden_patch_count_rule": "exact_integer(fraction * patch_count)",
                "rng": "numpy.default_rng",
                "rng_seed": int(seed) + 43001,
                "resampled_each_optimization_step": True,
            },
            "teacher_state": portable_state_dict(bundle.teacher),
            "readout_state": portable_state_dict(bundle.readout),
            "reconstruction_nmse": bundle.reconstruction_nmse,
            "readout_nmse": bundle.readout_nmse,
        },
        target,
    )


def load_teacher_bundle(
    path: str | Path,
    config: dict,
    *,
    device: str | torch.device = "cpu",
) -> TeacherBundle:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != TEACHER_CHECKPOINT_SCHEMA:
        raise RuntimeError("teacher checkpoint schema is not V6-compatible")
    sampler = payload.get("pretraining_mask_sampler")
    expected_sampler = {
        "sampler": PRETRAINING_MASK_SAMPLER,
        "fraction": float(config["teacher"]["mask_fraction"]),
        "hidden_patch_count_rule": "exact_integer(fraction * patch_count)",
        "rng": "numpy.default_rng",
        "rng_seed": int(payload["seed"]) + 43001,
        "resampled_each_optimization_step": True,
    }
    if sampler != expected_sampler:
        raise RuntimeError("teacher checkpoint lacks the exact V6 pretraining mask contract")
    spec = PatchSpec(**payload["patch_spec"])
    latent_dim = int(payload["teacher_config"]["latent_dim"])
    heads = _compatible_heads(latent_dim, int(payload["model_attention_heads"]))
    execution_device = torch.device(device)
    teacher = CSIMaskedTeacher(
        spec.patch_rows,
        spec.patch_columns,
        spec.patch_dim,
        latent_dim,
        heads,
        int(payload["teacher_config"]["encoder_layers"]),
        int(payload["teacher_config"]["decoder_layers"]),
    ).to(execution_device)
    readout = CSIReadout(latent_dim, spec.patch_dim).to(execution_device)
    teacher.load_state_dict(payload["teacher_state"])
    readout.load_state_dict(payload["readout_state"])
    teacher.eval()
    readout.eval()
    for module in (teacher, readout):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return TeacherBundle(
        teacher=teacher,
        readout=readout,
        patch_spec=spec,
        pretrain_mask_bank=tuple(
            entry
            for entry in frozen_mask_query_bank(spec, int(config["model"]["mask_bank_seed"]))
            if entry.mode == "random_75"
        ),
        mask_bank=frozen_mask_query_bank(
            spec, int(config["model"]["mask_bank_seed"])
        ),
        audit_mask_bank=frozen_mask_query_bank(
            spec, int(config["model"]["mask_bank_seed"]) + 1
        ),
        reconstruction_nmse=float(payload["reconstruction_nmse"]),
        readout_nmse=float(payload["readout_nmse"]),
    )


def teacher_targets(bundle: TeacherBundle, csi: np.ndarray) -> np.ndarray:
    patches = patchify_csi(np.asarray(csi, dtype=np.float32), bundle.patch_spec)
    original_shape = patches.shape[:-2]
    tensor = torch.as_tensor(
        patches.reshape(-1, bundle.patch_spec.patch_count, bundle.patch_spec.patch_dim),
        dtype=torch.float32,
        device=module_device(bundle.teacher),
    )
    with torch.no_grad():
        latent = bundle.teacher.encode_full(tensor).cpu().numpy()
    return latent.reshape(*original_shape, bundle.patch_spec.patch_count, -1)


def masked_reconstruction_nmse(
    bundle: TeacherBundle,
    csi: np.ndarray,
    *,
    audit: bool = True,
) -> float:
    patches = patchify_csi(np.asarray(csi, dtype=np.float32), bundle.patch_spec).reshape(
        -1, bundle.patch_spec.patch_count, bundle.patch_spec.patch_dim
    )
    device = module_device(bundle.teacher)
    tensor = torch.as_tensor(patches, dtype=torch.float32, device=device)
    source_bank = bundle.audit_mask_bank if audit else bundle.pretrain_mask_bank
    bank = tuple(entry for entry in source_bank if entry.mode == "random_75")
    masks = torch.as_tensor(
        np.stack([bank[index % len(bank)].mask for index in range(tensor.shape[0])]),
        dtype=torch.bool,
        device=device,
    )
    with torch.no_grad():
        _, prediction = bundle.teacher(tensor, masks)
    return _masked_nmse(tensor, prediction, masks)


def _compatible_heads(dimension: int, requested: int) -> int:
    for heads in range(min(dimension, requested), 0, -1):
        if dimension % heads == 0:
            return heads
    return 1


def _masked_nmse(target: Tensor, prediction: Tensor, mask: Tensor) -> float:
    numerator = torch.sum((target[mask] - prediction[mask]) ** 2)
    denominator = torch.sum(target[mask] ** 2).clamp_min(1e-12)
    return float((numerator / denominator).item())


def _nmse(target: Tensor, prediction: Tensor) -> float:
    numerator = torch.sum((target - prediction) ** 2)
    denominator = torch.sum(target**2).clamp_min(1e-12)
    return float((numerator / denominator).item())
