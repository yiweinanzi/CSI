from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .formal_model import module_device, portable_state_dict
from .formal_protocol import MaskQuery, PatchSpec, frozen_mask_query_bank, patchify_csi


TEACHER_CHECKPOINT_SCHEMA = "csi-pairs-stage0-teacher-v2.5-v6"
PRETRAINING_MASK_SAMPLER = "independent_per_sample_without_replacement"
TEACHER_INPUT_NORMALIZATION = "source_encoder_train_per_channel_zscore"
TEACHER_INPUT_SCALE_FLOOR = 1e-9
TEACHER_INFERENCE_BATCH_SIZE = 256
READOUT_FULL_BATCH_CHUNK_SIZE = 32_768


def _sincos_axis(length: int, dimension: int) -> Tensor:
    """Return a deterministic 1D sine-cosine table without trainable IDs."""
    if length < 1 or dimension < 1:
        raise ValueError("position table dimensions must be positive")
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(0, dimension, 2, dtype=torch.float32)
        / max(dimension, 1)
    )
    angles = positions * frequencies.unsqueeze(0)
    table = torch.zeros(length, dimension, dtype=torch.float32)
    table[:, 0::2] = torch.sin(angles[:, : table[:, 0::2].shape[1]])
    table[:, 1::2] = torch.cos(angles[:, : table[:, 1::2].shape[1]])
    return table


def fixed_2d_sincos_position(patch_rows: int, patch_columns: int, dimension: int) -> Tensor:
    """Build the fixed 2D position encoding required by the CSI-MAE protocol."""
    if dimension < 2:
        raise ValueError("2D position encoding requires dimension >= 2")
    row_dimension = dimension // 2
    column_dimension = dimension - row_dimension
    rows = _sincos_axis(int(patch_rows), row_dimension)[:, None, :].expand(
        -1, int(patch_columns), -1
    )
    columns = _sincos_axis(int(patch_columns), column_dimension)[None, :, :].expand(
        int(patch_rows), -1, -1
    )
    return torch.cat((rows, columns), dim=-1).reshape(1, -1, dimension)


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
        self.register_buffer(
            "fixed_position",
            fixed_2d_sincos_position(self.patch_rows, self.patch_columns, latent_dim),
            persistent=True,
        )
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
        return self.fixed_position

    def encode(self, patches: Tensor, mask: Tensor) -> Tensor:
        if patches.ndim != 3 or patches.shape[1:] != (self.patch_count, self.patch_dim):
            raise ValueError("teacher patches have the wrong shape")
        if mask.shape != patches.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("teacher mask must be boolean [batch, patch]")
        tokens = self.patch_embedding(patches) + self.positional_encoding()
        visible = ~mask
        visible_counts = visible.sum(dim=1)
        if bool(torch.any(visible_counts == 0)):
            raise ValueError("teacher mask cannot hide every patch")

        # TransformerEncoder requires a rectangular token tensor. Group rows by
        # visible cardinality so each group can be encoded as one real batch.
        grouped_rows = []
        grouped_indices = []
        for visible_count in torch.unique(visible_counts, sorted=True).tolist():
            count = int(visible_count)
            batch_indices = torch.nonzero(
                visible_counts == count, as_tuple=False
            ).squeeze(1)
            group_visible = visible.index_select(0, batch_indices)
            visible_indices = torch.nonzero(
                group_visible, as_tuple=False
            )[:, 1].reshape(batch_indices.shape[0], count)
            gather_indices = visible_indices.unsqueeze(-1).expand(
                -1, -1, self.latent_dim
            )
            visible_tokens = tokens.index_select(0, batch_indices).gather(
                1, gather_indices
            )
            encoded_visible = self.encoder(visible_tokens)
            full = self.mask_token.reshape(1, 1, -1).expand(
                batch_indices.shape[0], self.patch_count, -1
            )
            grouped_rows.append(full.scatter(1, gather_indices, encoded_visible))
            grouped_indices.append(batch_indices)

        row_indices = torch.cat(grouped_indices)
        rows = torch.cat(grouped_rows, dim=0)
        return rows.index_select(0, torch.argsort(row_indices))

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
    seed: int
    pretrain_mask_bank: tuple[MaskQuery, ...]
    mask_bank: tuple[MaskQuery, ...]
    audit_mask_bank: tuple[MaskQuery, ...]
    channel_mean: np.ndarray
    channel_scale: np.ndarray
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


def _batch_bounds(row_count: int, batch_size: int):
    rows = int(row_count)
    batch = int(batch_size)
    if rows < 1 or batch < 1:
        raise ValueError("teacher batch dimensions must be positive")
    for start in range(0, rows, batch):
        yield start, min(start + batch, rows)


def _encode_full_in_batches(
    teacher: CSIMaskedTeacher,
    patches: Tensor,
    *,
    batch_size: int = TEACHER_INFERENCE_BATCH_SIZE,
    output_path=None,
) -> Tensor:
    output = (torch.from_numpy(np.lib.format.open_memmap(output_path, mode="w+", dtype="float32",
        shape=(patches.shape[0], teacher.patch_count, teacher.latent_dim))) if output_path else torch.empty(
        (patches.shape[0], teacher.patch_count, teacher.latent_dim),
        dtype=patches.dtype,
        device=patches.device,
    ))
    for start, end in _batch_bounds(patches.shape[0], batch_size):
        output[start:end].copy_(teacher.encode_full(patches[start:end].to(next(teacher.parameters()).device)))
    return output


def _masked_reconstruction_nmse_in_batches(
    teacher: CSIMaskedTeacher,
    patches: Tensor,
    mask_bank: tuple[MaskQuery, ...],
    *,
    batch_size: int = TEACHER_INFERENCE_BATCH_SIZE,
) -> float:
    if not mask_bank:
        raise ValueError("teacher reconstruction audit requires a mask bank")
    device = next(teacher.parameters()).device
    numerator = torch.zeros((), dtype=torch.float64, device=device)
    denominator = torch.zeros((), dtype=torch.float64, device=device)
    for start, end in _batch_bounds(patches.shape[0], batch_size):
        masks = torch.as_tensor(
            np.stack(
                [mask_bank[index % len(mask_bank)].mask for index in range(start, end)]
            ),
            dtype=torch.bool,
            device=device,
        )
        batch = patches[start:end].to(device)
        _, reconstruction = teacher(batch, masks)
        target = batch[masks].to(dtype=torch.float64)
        prediction = reconstruction[masks].to(dtype=torch.float64)
        numerator += torch.sum((target - prediction) ** 2)
        denominator += torch.sum(target**2)
    return float((numerator / denominator.clamp_min(1e-12)).item())


def _full_batch_readout_step(
    readout: CSIReadout,
    optimizer: torch.optim.Optimizer,
    latent: Tensor,
    target: Tensor,
    *,
    chunk_size: int = READOUT_FULL_BATCH_CHUNK_SIZE,
) -> float:
    if latent.shape[:2] != target.shape[:2]:
        raise ValueError("teacher readout latent and target shapes are incompatible")
    optimizer.zero_grad(set_to_none=True)
    device = next(readout.parameters()).device
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    element_count = int(target.numel())
    for start, end in _batch_bounds(target.shape[0], chunk_size):
        error = readout(latent[start:end].to(device)) - target[start:end].to(device)
        loss = torch.sum(error**2) / float(element_count)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite teacher readout loss")
        loss.backward()
        total_loss += torch.sum(error.detach().to(dtype=torch.float64) ** 2)
    optimizer.step()
    return float((total_loss / float(element_count)).item())


def _readout_nmse_in_batches(
    readout: CSIReadout,
    latent: Tensor,
    target: Tensor,
    *,
    chunk_size: int = READOUT_FULL_BATCH_CHUNK_SIZE,
) -> float:
    device = next(readout.parameters()).device
    numerator = torch.zeros((), dtype=torch.float64, device=device)
    denominator = torch.zeros((), dtype=torch.float64, device=device)
    for start, end in _batch_bounds(target.shape[0], chunk_size):
        batch = target[start:end].to(device)
        prediction = readout(latent[start:end].to(device))
        numerator += torch.sum(
            (batch.to(dtype=torch.float64) - prediction.to(dtype=torch.float64)) ** 2
        )
        denominator += torch.sum(batch.to(dtype=torch.float64) ** 2)
    return float((numerator / denominator.clamp_min(1e-12)).item())


def _teacher_resume(path, model, optimizer, rng):
    if path is None or not path.exists():
        return 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    rng.bit_generator.state = state["numpy_rng"]
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["step"]


def _save_teacher_resume(path, model, optimizer, rng, step, total):
    if path is None or (step % 100 and step != total):
        return
    from .formal_training_resume import _atomic_torch_save
    _atomic_torch_save(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": step, "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})
    print(f"teacher {path.stem} step {step}/{total}", flush=True)


def train_teacher_bundle(
    csi: np.ndarray,
    patch_spec: PatchSpec,
    config: dict,
    *,
    seed: int,
    device: str | torch.device = "cpu",
    streaming_root=None,
) -> TeacherBundle:
    teacher_config = config["teacher"]
    model_config = config["model"]
    latent_dim = int(teacher_config["latent_dim"])
    heads = _compatible_heads(latent_dim, int(model_config["attention_heads"]))
    training_seed = int(seed)
    torch.manual_seed(training_seed)
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
    channel_mean, channel_scale = _fit_teacher_input_normalization(csi, patch_spec)
    if streaming_root is None:
        patches = normalized_teacher_patches_from_statistics(csi, patch_spec, channel_mean, channel_scale).reshape(-1, patch_spec.patch_count, patch_spec.patch_dim)
        tensor = torch.as_tensor(patches, dtype=torch.float32, device=execution_device)
    else:
        streaming_root = Path(streaming_root)
        streaming_root.mkdir(parents=True, exist_ok=True)
        raw = csi.reshape(-1, csi.shape[-1])
        patches = np.lib.format.open_memmap(streaming_root / "patches.npy", mode="w+", dtype="float32",
            shape=(len(raw), patch_spec.patch_count, patch_spec.patch_dim))
        for start, end in _batch_bounds(len(raw), 512):
            patches[start:end] = normalized_teacher_patches_from_statistics(raw[start:end], patch_spec, channel_mean, channel_scale)
        patches.flush()
        tensor = torch.from_numpy(patches)
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
    audit_pretraining_mask_bank = tuple(
        entry for entry in audit_mask_bank if entry.mode == "random_75"
    )
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=float(teacher_config["learning_rate"]))
    rng = np.random.default_rng(training_seed + 43001)
    batch_size = min(int(teacher_config["batch_size"]), tensor.shape[0])
    teacher_resume = streaming_root / "teacher_resume.pt" if streaming_root else None
    start_step = _teacher_resume(teacher_resume, teacher, optimizer, rng)
    for step in range(start_step, int(teacher_config["steps"])):
        indices = rng.integers(0, tensor.shape[0], size=batch_size)
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=tensor.device)
        batch = tensor[index_tensor].to(execution_device)
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
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite teacher training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _save_teacher_resume(teacher_resume, teacher, optimizer, rng, step + 1, int(teacher_config["steps"]))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        reconstruction_nmse = _masked_reconstruction_nmse_in_batches(
            teacher,
            tensor,
            audit_pretraining_mask_bank,
        )
        full_latent = _encode_full_in_batches(teacher, tensor,
            output_path=streaming_root / "latent.npy" if streaming_root else None)

    torch.manual_seed(training_seed + 1)
    readout = CSIReadout(latent_dim, patch_spec.patch_dim).to(execution_device)
    readout_optimizer = torch.optim.AdamW(readout.parameters(), lr=float(teacher_config["learning_rate"]))
    readout_resume = streaming_root / "readout_resume.pt" if streaming_root else None
    start_step = _teacher_resume(readout_resume, readout, readout_optimizer, rng)
    for step in range(start_step, int(teacher_config["steps"])):
        if teacher_config.get("readout_protocol") == "minibatch_updates":
            indices = torch.as_tensor(rng.integers(0, tensor.shape[0], size=batch_size), device=tensor.device)
            _full_batch_readout_step(readout, readout_optimizer, full_latent[indices].to(execution_device), tensor[indices].to(execution_device))
        else:
            _full_batch_readout_step(readout, readout_optimizer, full_latent, tensor)
        _save_teacher_resume(readout_resume, readout, readout_optimizer, rng, step + 1, int(teacher_config["steps"]))
    readout.eval()
    for parameter in readout.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        readout_nmse = _readout_nmse_in_batches(readout, full_latent, tensor)
    return TeacherBundle(
        teacher=teacher,
        readout=readout,
        patch_spec=patch_spec,
        seed=training_seed,
        pretrain_mask_bank=pretrain_mask_bank,
        mask_bank=mask_bank,
        audit_mask_bank=audit_mask_bank,
        channel_mean=channel_mean,
        channel_scale=channel_scale,
        reconstruction_nmse=reconstruction_nmse,
        readout_nmse=readout_nmse,
    )


def save_teacher_bundle(path: str | Path, bundle: TeacherBundle, config: dict, seed: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if int(seed) != bundle.seed:
        raise ValueError("teacher checkpoint seed does not match the trained bundle")
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
            "input_normalization": {
                "name": TEACHER_INPUT_NORMALIZATION,
                "source_roles": ["source_encoder_train"],
                "scale_floor": TEACHER_INPUT_SCALE_FLOOR,
                "channel_mean": bundle.channel_mean.tolist(),
                "channel_scale": bundle.channel_scale.tolist(),
                "physical_target": "source-normalized clean raw-CSI patch Psi(H)_q",
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
    if payload.get("teacher_config") != dict(config["teacher"]):
        raise RuntimeError("teacher checkpoint training configuration does not match")
    if payload.get("model_attention_heads") != int(config["model"]["attention_heads"]):
        raise RuntimeError("teacher checkpoint attention configuration does not match")
    if not isinstance(payload.get("seed"), int) or int(payload["seed"]) < 0:
        raise RuntimeError("teacher checkpoint seed is invalid")
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
    channel_mean, channel_scale = _load_teacher_input_normalization(
        payload.get("input_normalization"), spec
    )
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
        seed=int(payload["seed"]),
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
        channel_mean=channel_mean,
        channel_scale=channel_scale,
        reconstruction_nmse=float(payload["reconstruction_nmse"]),
        readout_nmse=float(payload["readout_nmse"]),
    )


def teacher_targets(bundle: TeacherBundle, csi: np.ndarray) -> np.ndarray:
    patches = normalized_teacher_patches(bundle, csi)
    original_shape = patches.shape[:-2]
    tensor = torch.as_tensor(
        patches.reshape(-1, bundle.patch_spec.patch_count, bundle.patch_spec.patch_dim),
        dtype=torch.float32,
        device=module_device(bundle.teacher),
    )
    with torch.no_grad():
        latent = _encode_full_in_batches(bundle.teacher, tensor).cpu().numpy()
    return latent.reshape(*original_shape, bundle.patch_spec.patch_count, -1)


def masked_reconstruction_nmse(
    bundle: TeacherBundle,
    csi: np.ndarray,
    *,
    audit: bool = True,
) -> float:
    patches = normalized_teacher_patches(bundle, csi).reshape(
        -1, bundle.patch_spec.patch_count, bundle.patch_spec.patch_dim
    )
    device = module_device(bundle.teacher)
    tensor = torch.as_tensor(patches, dtype=torch.float32, device=device)
    source_bank = bundle.audit_mask_bank if audit else bundle.pretrain_mask_bank
    bank = tuple(entry for entry in source_bank if entry.mode == "random_75")
    with torch.no_grad():
        return _masked_reconstruction_nmse_in_batches(bundle.teacher, tensor, bank)


def normalized_teacher_patches(bundle: TeacherBundle, csi: np.ndarray) -> np.ndarray:
    """Apply the frozen source-train physical target transform Psi."""
    return normalized_teacher_patches_from_statistics(
        csi,
        bundle.patch_spec,
        bundle.channel_mean,
        bundle.channel_scale,
    )


def normalized_teacher_patches_from_statistics(
    csi: np.ndarray,
    spec: PatchSpec,
    channel_mean: np.ndarray,
    channel_scale: np.ndarray,
) -> np.ndarray:
    values = np.asarray(csi, dtype=np.float64)
    mean = np.asarray(channel_mean, dtype=np.float64)
    scale = np.asarray(channel_scale, dtype=np.float64)
    expected = (2 * spec.complex_values,)
    if mean.shape != expected or scale.shape != expected:
        raise ValueError("teacher input normalization has the wrong channel shape")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("teacher input normalization must be finite")
    if np.any(scale <= 0.0):
        raise ValueError("teacher input normalization scale must be positive")
    normalized = (values - mean) / scale
    return patchify_csi(normalized, spec)


def _fit_teacher_input_normalization(
    csi: np.ndarray, spec: PatchSpec
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(csi, dtype=np.float64)
    expected_channels = 2 * spec.complex_values
    if values.ndim < 2 or values.shape[-1] != expected_channels:
        raise ValueError("teacher CSI has the wrong channel shape")
    flattened = values.reshape(-1, expected_channels)
    if flattened.shape[0] < 2 or not np.all(np.isfinite(flattened)):
        raise ValueError("teacher CSI normalization requires finite source-train samples")
    channel_mean = np.zeros(expected_channels, dtype=np.float64)
    for start, end in _batch_bounds(len(flattened), 1024):
        channel_mean += flattened[start:end].sum(axis=0) / len(flattened)
    variance = np.zeros_like(channel_mean)
    for start, end in _batch_bounds(len(flattened), 1024):
        variance += np.square(flattened[start:end] - channel_mean).sum(axis=0) / len(flattened)
    channel_scale = np.sqrt(variance)
    channel_scale[channel_scale < TEACHER_INPUT_SCALE_FLOOR] = 1.0
    return channel_mean, channel_scale


def _load_teacher_input_normalization(
    record: object, spec: PatchSpec
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(record, dict):
        raise RuntimeError("teacher checkpoint lacks source-train input normalization")
    expected_keys = {
        "name",
        "source_roles",
        "scale_floor",
        "channel_mean",
        "channel_scale",
        "physical_target",
    }
    if set(record) != expected_keys:
        raise RuntimeError("teacher checkpoint input normalization contract is incomplete")
    if (
        record["name"] != TEACHER_INPUT_NORMALIZATION
        or record["source_roles"] != ["source_encoder_train"]
        or float(record["scale_floor"]) != TEACHER_INPUT_SCALE_FLOOR
        or record["physical_target"]
        != "source-normalized clean raw-CSI patch Psi(H)_q"
    ):
        raise RuntimeError("teacher checkpoint input normalization contract is invalid")
    channel_mean = np.asarray(record["channel_mean"], dtype=np.float64)
    channel_scale = np.asarray(record["channel_scale"], dtype=np.float64)
    try:
        normalized_teacher_patches_from_statistics(
            np.zeros((1, 2 * spec.complex_values), dtype=np.float64),
            spec,
            channel_mean,
            channel_scale,
        )
    except ValueError as error:
        raise RuntimeError(
            f"teacher checkpoint input normalization is invalid: {error}"
        ) from error
    return channel_mean, channel_scale


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
