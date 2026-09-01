"""Source-only representation baselines for descriptive localization comparison.

None of these models are C1-eligible. WWMJEPA is a restricted WWM-inspired
method: the teacher/target encoder sees source position
(``include_position=True``) while the online encoder does not. That is the
intended inspired recipe, not a silent bug, and it is not a strict
no-position C1 baseline. SigMap/WiSER/RFIR stay style-controlled and
C1-ineligible in the map-adapter registry.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from formal_v2.formal_protocol import PatchSpec


@dataclass(frozen=True)
class BaselineBatch:
    csi: Tensor
    maps: Tensor
    radio: Tensor
    bs_pose: Tensor
    position: Tensor


def csi_grid(csi: Tensor, spec: PatchSpec) -> Tensor:
    if csi.ndim != 2 or csi.shape[1] != 2 * spec.complex_values:
        raise ValueError("CSI tensor does not match the frozen antenna/subcarrier grid")
    real = csi[:, : spec.complex_values].reshape(-1, spec.antennas, spec.subcarriers)
    imag = csi[:, spec.complex_values :].reshape(-1, spec.antennas, spec.subcarriers)
    return torch.stack((real, imag), dim=1)


def patchify_tensor(csi: Tensor, spec: PatchSpec) -> Tensor:
    grid = csi_grid(csi, spec)
    patches = grid.unfold(2, int(spec.patch_antenna_size), int(spec.patch_antenna_size))
    patches = patches.unfold(3, int(spec.patch_subcarrier_size), int(spec.patch_subcarrier_size))
    return patches.permute(0, 2, 3, 1, 4, 5).reshape(csi.shape[0], spec.patch_count, spec.patch_dim)


def fixed_2d_sincos_position(spec: PatchSpec, dim: int, device=None) -> Tensor:
    if dim % 4:
        raise ValueError("fixed two-dimensional sine-cosine position dimension must be divisible by four")
    axis_dim = dim // 2
    frequency = torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device)
    frequency = torch.pow(10000.0, -frequency / axis_dim)

    def embed(length: int) -> Tensor:
        position = torch.arange(length, dtype=torch.float32, device=device)[:, None]
        angle = position * frequency[None]
        return torch.cat((torch.sin(angle), torch.cos(angle)), dim=1)

    row = embed(spec.patch_rows)[:, None].expand(-1, spec.patch_columns, -1)
    column = embed(spec.patch_columns)[None].expand(spec.patch_rows, -1, -1)
    return torch.cat((row, column), dim=-1).reshape(1, spec.patch_count, dim)


class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_channels: int, channels: int, stride: int):
        super().__init__()
        output_channels = channels * self.expansion
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and input_channels == output_channels
            else nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        )

    def forward(self, value: Tensor) -> Tensor:
        return F.relu(self.skip(value) + self.network(value), inplace=True)


class CSIResNetEncoder(nn.Module):
    """Two-channel residual encoder used by the CSI-CLIP controlled adaptation."""

    def __init__(self, spec: PatchSpec, width: int, depth: int, output_dim: int):
        super().__init__()
        if depth != 50:
            raise ValueError("CSI-CLIP paper-spec encoder must be ResNet-50")
        self.spec = spec
        self.stem = nn.Sequential(
            nn.Conv2d(2, width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        blocks = []
        input_channels = width
        for stage, count in enumerate((3, 4, 6, 3)):
            channels = width * (2**stage)
            for block in range(count):
                stride = 2 if stage > 0 and block == 0 else 1
                blocks.append(ResNetBottleneck(input_channels, channels, stride))
                input_channels = channels * ResNetBottleneck.expansion
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Linear(input_channels, output_dim)

    def forward(self, csi: Tensor) -> Tensor:
        hidden = self.blocks(self.stem(csi_grid(csi, self.spec)))
        return self.output(hidden.mean(dim=(-2, -1)))


class PatchTransformerEncoder(nn.Module):
    def __init__(self, spec: PatchSpec, dim: int, heads: int, layers: int, output_dim: int | None = None):
        super().__init__()
        if dim % heads:
            raise ValueError("transformer dimension must be divisible by its head count")
        self.spec = spec
        self.dim = int(dim)
        self.patch_embedding = nn.Linear(spec.patch_dim, dim)
        self.register_buffer("fixed_position", fixed_2d_sincos_position(spec, dim), persistent=True)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=0.0,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Identity() if output_dim is None else nn.Linear(dim, output_dim)

    def position(self) -> Tensor:
        return self.fixed_position

    def tokens(self, csi: Tensor) -> Tensor:
        return self.patch_embedding(patchify_tensor(csi, self.spec)) + self.position()

    def forward(self, csi: Tensor, visible: Tensor | None = None) -> Tensor:
        tokens = self.tokens(csi)
        if visible is not None:
            if visible.ndim == 1:
                if visible.shape[0] != self.spec.patch_count:
                    raise ValueError("visible patch selector has the wrong shape")
                tokens = tokens[:, visible]
            elif visible.ndim == 2:
                if visible.shape[0] != tokens.shape[0]:
                    raise ValueError("per-sample visible patch selector has the wrong batch size")
                tokens = torch.gather(tokens, 1, visible[..., None].expand(-1, -1, tokens.shape[-1]))
            else:
                raise ValueError("visible patch selector has the wrong rank")
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        encoded = self.norm(self.transformer(torch.cat((cls, tokens), dim=1)))
        return self.output(encoded[:, 0])


class CSIMAE(nn.Module):
    """Paper-spec two-dimensional asymmetric masked autoencoder."""

    def __init__(self, spec: PatchSpec, dim: int, heads: int, encoder_layers: int, decoder_layers: int, decoder_dim: int):
        super().__init__()
        self.spec = spec
        self.encoder = PatchTransformerEncoder(spec, dim, heads, encoder_layers)
        decoder_heads = min(heads, decoder_dim)
        while decoder_dim % decoder_heads:
            decoder_heads -= 1
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=2 * decoder_dim,
            dropout=0.0,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.encoder_to_decoder = nn.Linear(dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.register_buffer("decoder_position", fixed_2d_sincos_position(spec, decoder_dim), persistent=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, decoder_layers)
        self.decoder_head = nn.Linear(decoder_dim, spec.patch_dim)

    def _mask(self, batch_size: int, device: torch.device, mask_fraction: float) -> Tensor:
        count = max(1, min(self.spec.patch_count - 1, round(mask_fraction * self.spec.patch_count)))
        order = torch.rand(batch_size, self.spec.patch_count, device=device).argsort(dim=1)
        mask = torch.zeros(batch_size, self.spec.patch_count, dtype=torch.bool, device=device)
        mask.scatter_(1, order[:, :count], True)
        return mask

    def pretraining_loss(self, batch: BaselineBatch, mask_fraction: float = 0.75) -> Tensor:
        patches = patchify_tensor(batch.csi, self.spec)
        mask = self._mask(batch.csi.shape[0], batch.csi.device, mask_fraction)
        tokens = self.encoder.tokens(batch.csi)
        visible_index = (~mask).to(torch.int64).argsort(dim=1, descending=True)[:, : int((~mask).sum(dim=1)[0])]
        visible = torch.gather(tokens, 1, visible_index[..., None].expand(-1, -1, tokens.shape[-1]))
        cls = self.encoder.cls.expand(tokens.shape[0], -1, -1)
        visible_tokens = self.encoder.norm(self.encoder.transformer(torch.cat((cls, visible), dim=1)))[:, 1:]
        visible_tokens = self.encoder_to_decoder(visible_tokens)
        full = self.mask_token.expand(tokens.shape[0], self.spec.patch_count, -1).clone()
        full.scatter_(1, visible_index[..., None].expand(-1, -1, full.shape[-1]), visible_tokens)
        decoded = self.decoder(full + self.decoder_position)
        prediction = self.decoder_head(decoded)
        return F.mse_loss(prediction[mask], patches[mask])

    def encode(self, batch: BaselineBatch) -> Tensor:
        return self.encoder(batch.csi)


class CSIClip(nn.Module):
    """CIR/CSI consistency model with independent two-channel residual encoders."""

    def __init__(self, spec: PatchSpec, width: int, depth: int, dim: int):
        super().__init__()
        self.spec = spec
        self.csi_encoder = CSIResNetEncoder(spec, width, depth, dim)
        self.cir_encoder = CSIResNetEncoder(spec, width, depth, dim)
        self.log_temperature = nn.Parameter(torch.tensor(-2.659260036932778))

    def _cir(self, csi: Tensor) -> Tensor:
        grid = csi_grid(csi, self.spec)
        complex_csi = torch.complex(grid[:, 0], grid[:, 1])
        cir = torch.fft.ifft(complex_csi, dim=-1, norm="ortho")
        flat_real = cir.real.reshape(csi.shape[0], -1)
        flat_imag = cir.imag.reshape(csi.shape[0], -1)
        return torch.cat((flat_real, flat_imag), dim=1)

    def pretraining_loss(self, batch: BaselineBatch, mask_fraction: float = 0.0) -> Tensor:
        del mask_fraction
        csi = F.normalize(self.csi_encoder(batch.csi), dim=1)
        cir = F.normalize(self.cir_encoder(self._cir(batch.csi)), dim=1)
        temperature = self.log_temperature.exp().clamp(1e-3, 1.0)
        logits = csi @ cir.T / temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    def encode(self, batch: BaselineBatch) -> Tensor:
        return self.csi_encoder(batch.csi)


class CSIClipPlus(nn.Module):
    """Frozen V6 CSI-CLIP++-style ViT controlled implementation."""

    def __init__(self, spec: PatchSpec, dim: int, heads: int, layers: int):
        super().__init__()
        self.spec = spec
        self.csi_encoder = PatchTransformerEncoder(spec, dim, heads, layers)
        self.cir_encoder = PatchTransformerEncoder(spec, dim, heads, layers)
        self.log_temperature = nn.Parameter(torch.tensor(-2.659260036932778))

    def _cir(self, csi: Tensor) -> Tensor:
        grid = csi_grid(csi, self.spec)
        values = torch.fft.ifft(torch.complex(grid[:, 0], grid[:, 1]), dim=-1, norm="ortho")
        return torch.cat((values.real.reshape(csi.shape[0], -1), values.imag.reshape(csi.shape[0], -1)), dim=1)

    def pretraining_loss(self, batch: BaselineBatch, mask_fraction: float = 0.0) -> Tensor:
        del mask_fraction
        first = F.normalize(self.csi_encoder(batch.csi), dim=1)
        second = F.normalize(self.cir_encoder(self._cir(batch.csi)), dim=1)
        logits = first @ second.T / self.log_temperature.exp().clamp(1e-3, 1.0)
        labels = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    def encode(self, batch: BaselineBatch) -> Tensor:
        return self.csi_encoder(batch.csi)


class ContraWiMAE(nn.Module):
    """Asymmetric MAE with reconstruction and masked-view contrastive objectives."""

    def __init__(
        self,
        spec: PatchSpec,
        dim: int,
        heads: int,
        encoder_layers: int,
        decoder_layers: int,
        decoder_dim: int,
        reconstruction_weight: float,
        minimum_snr_db: float,
        maximum_snr_db: float,
    ):
        super().__init__()
        self.mae = CSIMAE(spec, dim, heads, encoder_layers, decoder_layers, decoder_dim)
        self.projector = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.log_temperature = nn.Parameter(torch.tensor(-1.6094379124341003))
        self.reconstruction_weight = float(reconstruction_weight)
        self.minimum_snr_db = float(minimum_snr_db)
        self.maximum_snr_db = float(maximum_snr_db)

    def _noisy_view(self, csi: Tensor) -> Tensor:
        snr_db = torch.empty(csi.shape[0], 1, device=csi.device).uniform_(
            self.minimum_snr_db, self.maximum_snr_db
        )
        signal_power = torch.mean(csi**2, dim=1, keepdim=True).clamp_min(1e-12)
        noise_power = signal_power / torch.pow(10.0, snr_db / 10.0)
        return csi + torch.randn_like(csi) * torch.sqrt(noise_power)

    def pretraining_loss(self, batch: BaselineBatch, mask_fraction: float = 0.75) -> Tensor:
        reconstruction = self.mae.pretraining_loss(batch, mask_fraction)
        mask_a = self.mae._mask(batch.csi.shape[0], batch.csi.device, mask_fraction)
        mask_b = self.mae._mask(batch.csi.shape[0], batch.csi.device, mask_fraction)
        visible_a = (~mask_a).to(torch.int64).argsort(dim=1, descending=True)[:, : int((~mask_a).sum(dim=1)[0])]
        visible_b = (~mask_b).to(torch.int64).argsort(dim=1, descending=True)[:, : int((~mask_b).sum(dim=1)[0])]
        view_a = F.normalize(self.projector(self.mae.encoder(self._noisy_view(batch.csi), visible_a)), dim=1)
        view_b = F.normalize(self.projector(self.mae.encoder(self._noisy_view(batch.csi), visible_b)), dim=1)
        logits = view_a @ view_b.T / self.log_temperature.exp().clamp(1e-3, 1.0)
        labels = torch.arange(logits.shape[0], device=logits.device)
        contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        return self.reconstruction_weight * reconstruction + (1.0 - self.reconstruction_weight) * contrastive

    def encode(self, batch: BaselineBatch) -> Tensor:
        return self.mae.encode(batch)

    def warm_start_loss(self, batch: BaselineBatch, mask_fraction: float) -> Tensor:
        return self.mae.pretraining_loss(batch, mask_fraction)


class ModalityExpertBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
            for _ in range(3)
        )

    def forward(self, tokens: Tensor, modality: Tensor) -> Tensor:
        attended, _ = self.attention(self.norm1(tokens), self.norm1(tokens), self.norm1(tokens), need_weights=False)
        tokens = tokens + attended
        normalized = self.norm2(tokens)
        update = torch.zeros_like(tokens)
        for index, expert in enumerate(self.experts):
            selected = modality == index
            if torch.any(selected):
                update[:, selected] = expert(normalized[:, selected])
        return tokens + update


class WWMEncoder(nn.Module):
    def __init__(self, spec: PatchSpec, map_channels: int, context_dim: int, dim: int, heads: int, layers: int):
        super().__init__()
        self.spec = spec
        self.dim = dim
        self.csi = nn.Linear(spec.patch_dim, dim)
        self.map = nn.Linear(map_channels, dim)
        self.context = nn.Linear(context_dim, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.modality_embedding = nn.Parameter(torch.zeros(3, dim))
        self.blocks = nn.ModuleList(ModalityExpertBlock(dim, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)

    def tokenize(self, batch: BaselineBatch, include_position: bool) -> tuple[Tensor, Tensor]:
        csi = self.csi(patchify_tensor(batch.csi, self.spec)) + self.modality_embedding[0]
        pooled = F.adaptive_avg_pool2d(batch.maps, (4, 4)).flatten(2).transpose(1, 2)
        map_tokens = self.map(pooled) + self.modality_embedding[1]
        position = batch.position if include_position else torch.zeros_like(batch.position)
        context = torch.cat((batch.radio, batch.bs_pose, position), dim=1)
        context_token = self.context(context)[:, None] + self.modality_embedding[2]
        cls = self.cls.expand(csi.shape[0], -1, -1)
        tokens = torch.cat((cls, csi, map_tokens, context_token), dim=1)
        modality = torch.cat(
            (
                torch.tensor([2], device=tokens.device),
                torch.zeros(csi.shape[1], dtype=torch.long, device=tokens.device),
                torch.ones(map_tokens.shape[1], dtype=torch.long, device=tokens.device),
                torch.tensor([2], device=tokens.device),
            )
        )
        return tokens, modality

    def forward(self, batch: BaselineBatch, include_position: bool = False) -> tuple[Tensor, Tensor]:
        tokens, modality = self.tokenize(batch, include_position)
        for block in self.blocks:
            tokens = block(tokens, modality)
        tokens = self.norm(tokens)
        return tokens[:, 0], tokens


class WWMJEPA(nn.Module):
    """Restricted WWM-inspired same-world JEPA with CSI, map, radio/BS, and trajectory tokens.

    The teacher target uses ``include_position=True``; the online encoder and
    ``encode`` path stay ``include_position=False``. This is an inspired
    style-controlled method, not a faithful no-position WWM reproduction and
    not a C1-eligible baseline.
    """

    claim_eligible = False
    restricted = True
    c1_eligible = False

    def __init__(self, spec: PatchSpec, map_channels: int, context_dim: int, dim: int, heads: int, layers: int, ema: float):
        super().__init__()
        self.online = WWMEncoder(spec, map_channels, context_dim, dim, heads, layers)
        self.target = copy.deepcopy(self.online)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.predictor = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.ema = float(ema)

    def pretraining_loss(self, batch: BaselineBatch, mask_fraction: float = 0.0) -> Tensor:
        del mask_fraction
        online, _ = self.online(batch, include_position=False)
        with torch.no_grad():
            target, _ = self.target(batch, include_position=True)
        return F.l1_loss(self.predictor(online), target.detach())

    @torch.no_grad()
    def update_target(self) -> None:
        for target, online in zip(self.target.parameters(), self.online.parameters()):
            target.mul_(self.ema).add_(online, alpha=1.0 - self.ema)

    def encode(self, batch: BaselineBatch) -> Tensor:
        representation, _ = self.online(batch, include_position=False)
        return representation


class CSIOnly(nn.Module):
    def __init__(self, spec: PatchSpec, dim: int, heads: int, layers: int):
        super().__init__()
        self.encoder = PatchTransformerEncoder(spec, dim, heads, layers)

    def pretraining_loss(self, batch: BaselineBatch, mask_fraction: float = 0.0) -> Tensor:
        del mask_fraction
        representation = self.encoder(batch.csi)
        return torch.mean(representation**2) * 0.0

    def encode(self, batch: BaselineBatch) -> Tensor:
        return self.encoder(batch.csi)


def build_representation_model(name: str, spec: PatchSpec, map_channels: int, context_dim: int, config: dict) -> nn.Module:
    dim = int(config["dim"])
    heads = int(config["heads"])
    layers = int(config["encoder_layers"])
    if name == "CSI-MAE":
        return CSIMAE(spec, dim, heads, layers, int(config["decoder_layers"]), int(config["decoder_dim"]))
    if name == "CSI-CLIP":
        return CSIClip(spec, int(config["resnet_width"]), int(config["resnet_depth"]), dim)
    if name == "CSI-CLIP++":
        return CSIClipPlus(spec, dim, heads, layers)
    if name == "ContraWiMAE":
        return ContraWiMAE(
            spec,
            dim,
            heads,
            layers,
            int(config["decoder_layers"]),
            int(config["decoder_dim"]),
            float(config["reconstruction_weight"]),
            float(config["minimum_snr_db"]),
            float(config["maximum_snr_db"]),
        )
    if name == "WWM":
        return WWMJEPA(
            spec,
            map_channels,
            context_dim,
            dim,
            heads,
            layers,
            float(config["ema"]),
        )
    if name == "CSI-only":
        return CSIOnly(spec, dim, heads, layers)
    raise ValueError(f"unknown representation baseline: {name}")
