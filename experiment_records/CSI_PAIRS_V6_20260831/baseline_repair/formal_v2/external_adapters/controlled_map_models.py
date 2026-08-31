from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def complex_csi(csi: Tensor) -> Tensor:
    if csi.shape[-1] % 2:
        raise ValueError("CSI must store all real values followed by all imaginary values")
    half = csi.shape[-1] // 2
    return torch.complex(csi[..., :half], csi[..., half:])


def real_imag_csi(csi: Tensor) -> Tensor:
    return torch.cat((csi.real, csi.imag), dim=-1)


def relative_power_db(csi: Tensor, floor: float = 1e-12) -> Tensor:
    values = complex_csi(csi)
    return 10.0 * torch.log10(torch.mean(values.abs() ** 2, dim=-1).clamp_min(floor))


def deterministic_prefix_product(values: Tensor, dim: int = -1) -> Tensor:
    """Inclusive prefix product using deterministic elementwise CUDA kernels."""
    normalized_dim = dim if dim >= 0 else values.ndim + dim
    if normalized_dim < 0 or normalized_dim >= values.ndim:
        raise IndexError("prefix-product dimension is out of range")
    prefix = values.movedim(normalized_dim, -1)
    offset = 1
    while offset < prefix.shape[-1]:
        shifted = torch.cat(
            (torch.ones_like(prefix[..., :offset]), prefix[..., :-offset]),
            dim=-1,
        )
        prefix = prefix * shifted
        offset *= 2
    return prefix.movedim(-1, normalized_dim)


class MapCNN(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, maps: Tensor) -> Tensor:
        return self.network(maps).flatten(1)


class SigMapLocator(nn.Module):
    """Map-conditioned CSI fingerprint locator used for the SigMap control."""

    def __init__(self, csi_dim: int, map_channels: int, context_dim: int, hidden: int):
        super().__init__()
        self.csi = nn.Sequential(
            nn.Linear(csi_dim, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden),
        )
        self.map = MapCNN(map_channels, hidden)
        self.context = nn.Sequential(nn.Linear(context_dim, hidden), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, csi: Tensor, maps: Tensor, context: Tensor) -> Tensor:
        return self.fusion(torch.cat((self.csi(csi), self.map(maps), self.context(context)), dim=1))

    def training_loss(self, csi: Tensor, maps: Tensor, context: Tensor, receiver_xy: Tensor) -> Tensor:
        return F.huber_loss(self(csi, maps, context), receiver_xy)


class SparseSceneEncoder(nn.Module):
    def __init__(self, map_channels: int, context_dim: int, dim: int, heads: int, layers: int, grid_size: int):
        super().__init__()
        if dim % heads:
            raise ValueError("WiSER scene dimension must be divisible by its head count")
        self.grid_size = int(grid_size)
        self.map_projection = nn.Linear(map_channels + 3, dim)
        self.scale_embedding = nn.Parameter(torch.zeros(4, dim))
        self.tx_projection = nn.Sequential(nn.Linear(context_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(
            dim, heads, 4 * dim, batch_first=True, norm_first=True, dropout=0.0, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, maps: Tensor, wireless_context: Tensor, origin: Tensor, resolution: float) -> tuple[Tensor, Tensor, Tensor]:
        batch, channels, rows, columns = maps.shape
        token_levels = []
        xyz_levels = []
        valid_levels = []
        for level, size in enumerate(
            (self.grid_size, max(1, self.grid_size // 2), max(1, self.grid_size // 4), max(1, self.grid_size // 8))
        ):
            pooled = F.adaptive_avg_pool2d(maps, (size, size))
            y, x = torch.meshgrid(
                torch.arange(size, device=maps.device),
                torch.arange(size, device=maps.device),
                indexing="ij",
            )
            scale_x = columns / size * float(resolution)
            scale_y = rows / size * float(resolution)
            xy = torch.stack(((x + 0.5) * scale_x, (y + 0.5) * scale_y), dim=-1).reshape(1, -1, 2)
            xy = xy + origin[:, None, :]
            height = (
                pooled[:, 1:2].flatten(2).transpose(1, 2)
                if channels > 1
                else torch.zeros(batch, size**2, 1, device=maps.device)
            )
            xyz = torch.cat((xy.expand(batch, -1, -1), height), dim=-1)
            features = pooled.flatten(2).transpose(1, 2)
            valid = features[..., 0] > 1e-4
            empty = ~torch.any(valid, dim=1)
            if torch.any(empty):
                valid[empty, 0] = True
            token_levels.append(
                self.map_projection(torch.cat((features, xyz), dim=-1)) + self.scale_embedding[level]
            )
            xyz_levels.append(xyz)
            valid_levels.append(valid)
        tokens = torch.cat(token_levels, dim=1)
        xyz = torch.cat(xyz_levels, dim=1)
        valid = torch.cat(valid_levels, dim=1)
        tokens = tokens + self.tx_projection(wireless_context)[:, None]
        memory = self.norm(self.encoder(tokens, src_key_padding_mask=~valid))
        memory = memory.masked_fill((~valid)[..., None], 0.0)
        return memory, xyz, valid


class RayCorridorDecoder(nn.Module):
    def __init__(self, dim: int, heads: int, corridor_tokens: int):
        super().__init__()
        self.corridor_tokens = int(corridor_tokens)
        self.query = nn.Sequential(nn.Linear(10, dim), nn.GELU(), nn.Linear(dim, dim))
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.power = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, memory: Tensor, voxel_xyz: Tensor, valid: Tensor, tx_xyz: Tensor, rx_xy: Tensor) -> tuple[Tensor, Tensor]:
        rx_xyz = torch.cat((rx_xy, torch.full_like(rx_xy[:, :1], 1.5)), dim=1)
        segment = rx_xyz - tx_xyz
        relative = voxel_xyz - tx_xyz[:, None]
        projection = torch.sum(relative * segment[:, None], dim=-1) / torch.sum(segment**2, dim=-1, keepdim=True).clamp_min(1e-8)
        projection = projection.clamp(0.0, 1.0)
        nearest = tx_xyz[:, None] + projection[..., None] * segment[:, None]
        distance = torch.linalg.vector_norm(voxel_xyz - nearest, dim=-1).masked_fill(~valid, float("inf"))
        count = min(self.corridor_tokens, memory.shape[1])
        indices = torch.topk(distance, count, largest=False, dim=1).indices
        corridor = torch.gather(memory, 1, indices[..., None].expand(-1, -1, memory.shape[-1]))
        geometry = torch.cat((tx_xyz, rx_xyz, segment, torch.linalg.vector_norm(segment, dim=1, keepdim=True)), dim=1)
        query = self.query(geometry)[:, None]
        attended, _ = self.attention(query, corridor, corridor, need_weights=False)
        fused = torch.cat((query[:, 0], attended[:, 0]), dim=1)
        return self.power(fused)[:, 0], fused


class CIRSetDecoder(nn.Module):
    def __init__(self, scene_dim: int, fused_dim: int, heads: int, subcarriers: int, antennas: int, tap_count: int):
        super().__init__()
        self.subcarriers = int(subcarriers)
        self.antennas = int(antennas)
        self.tap_count = int(tap_count)
        self.query = nn.Parameter(torch.zeros(tap_count, fused_dim))
        self.memory_projection = nn.Linear(scene_dim, fused_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=fused_dim,
            nhead=heads,
            dim_feedforward=4 * fused_dim,
            dropout=0.0,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=2)
        self.condition = nn.Linear(fused_dim, fused_dim)
        self.tap_head = nn.Linear(fused_dim, 5)

    def predict_taps(self, memory: Tensor, fused: Tensor) -> Tensor:
        queries = self.query[None].expand(fused.shape[0], -1, -1)
        queries = queries + self.condition(fused)[:, None]
        decoded = self.decoder(queries, self.memory_projection(memory))
        return self.tap_head(decoded)

    def forward(self, memory: Tensor, fused: Tensor) -> Tensor:
        values = self.predict_taps(memory, fused)
        existence = torch.sigmoid(values[..., 0])
        delay = torch.sigmoid(values[..., 1])
        power_db = -120.0 + 120.0 * torch.sigmoid(values[..., 2])
        phase_offset = math.pi * torch.tanh(values[..., 3])
        arrival_angle = math.pi * torch.tanh(values[..., 4])
        amplitude = torch.pow(10.0, power_db / 20.0) * torch.exp(1j * phase_offset) * existence
        frequency = torch.linspace(0.0, 1.0, self.subcarriers, device=fused.device)
        phase = torch.exp(-2j * math.pi * delay[..., None] * frequency)
        antenna = torch.arange(self.antennas, device=fused.device, dtype=fused.dtype)
        array = torch.exp(1j * math.pi * antenna[None, :, None] * torch.sin(arrival_angle)[:, None])
        per_tap = amplitude[:, None, :, None] * array[..., None] * phase[:, None]
        return torch.sum(per_tap, dim=2).reshape(fused.shape[0], -1)

    def matching_loss(self, memory: Tensor, fused: Tensor, target_csi: Tensor) -> Tensor:
        from scipy.optimize import linear_sum_assignment

        values = self.predict_taps(memory, fused)
        predicted_existence = values[..., 0]
        predicted_delay = torch.sigmoid(values[..., 1])
        predicted_power = -120.0 + 120.0 * torch.sigmoid(values[..., 2])
        target_grid = target_csi.reshape(-1, self.antennas, self.subcarriers)
        target_cir = torch.fft.ifft(target_grid, dim=-1, norm="ortho")
        target_power = torch.mean(target_cir.abs() ** 2, dim=1).clamp_min(1e-12)
        target_db = 10.0 * torch.log10(target_power)
        count = min(self.tap_count, self.subcarriers)
        peak_power, peak_index = torch.topk(target_db, count, dim=1)
        peak_delay = peak_index.to(target_db.dtype) / max(self.subcarriers - 1, 1)
        active = peak_power >= (torch.max(peak_power, dim=1, keepdim=True).values - 30.0)
        losses = []
        for batch_index in range(values.shape[0]):
            cost = (
                10.0 * torch.abs(predicted_delay[batch_index, :, None] - peak_delay[batch_index, None])
                + 0.05 * torch.abs(predicted_power[batch_index, :, None] - peak_power[batch_index, None])
            )
            predicted_index, target_index = linear_sum_assignment(cost.detach().cpu().numpy())
            predicted_index = torch.as_tensor(predicted_index, device=values.device)
            target_index = torch.as_tensor(target_index, device=values.device)
            matched_active = active[batch_index, target_index]
            existence_target = torch.zeros(self.tap_count, device=values.device)
            existence_target[predicted_index] = matched_active.to(values.dtype)
            existence_loss = F.binary_cross_entropy_with_logits(
                predicted_existence[batch_index], existence_target
            )
            if torch.any(matched_active):
                selected_predicted = predicted_index[matched_active]
                selected_target = target_index[matched_active]
                delay_loss = F.smooth_l1_loss(
                    predicted_delay[batch_index, selected_predicted], peak_delay[batch_index, selected_target]
                )
                power_loss = F.smooth_l1_loss(
                    predicted_power[batch_index, selected_predicted], peak_power[batch_index, selected_target]
                )
            else:
                delay_loss = predicted_delay[batch_index].sum() * 0.0
                power_loss = predicted_power[batch_index].sum() * 0.0
            losses.append(existence_loss + 10.0 * delay_loss + 0.05 * power_loss)
        return torch.stack(losses).mean()


class WiSERForward(nn.Module):
    """Sparse scene memory with ray-corridor radiomap and DETR-style tap readouts."""

    def __init__(self, map_channels: int, context_dim: int, dim: int, heads: int, layers: int, grid_size: int, corridor_tokens: int, antennas: int, subcarriers: int, tap_count: int):
        super().__init__()
        self.scene = SparseSceneEncoder(map_channels, context_dim, dim, heads, layers, grid_size)
        self.radiomap = RayCorridorDecoder(dim, heads, corridor_tokens)
        self.cir = CIRSetDecoder(dim, 2 * dim, heads, subcarriers, antennas, tap_count)

    def forward(self, maps: Tensor, tx_xyz: Tensor, wireless_context: Tensor, rx_xy: Tensor, origin: Tensor, resolution: float) -> tuple[Tensor, Tensor]:
        memory, voxel_xyz, valid = self.scene(maps, wireless_context, origin, resolution)
        power, fused = self.radiomap(memory, voxel_xyz, valid, tx_xyz, rx_xy)
        return power, self.cir(memory, fused)

    def training_loss(self, csi: Tensor, maps: Tensor, tx_xyz: Tensor, wireless_context: Tensor, rx_xy: Tensor, origin: Tensor, resolution: float, task: str = "joint") -> Tensor:
        memory, voxel_xyz, valid = self.scene(maps, wireless_context, origin, resolution)
        predicted_power, fused = self.radiomap(memory, voxel_xyz, valid, tx_xyz, rx_xy)
        predicted_csi = self.cir(memory, fused)
        target_complex = complex_csi(csi)
        target_power = 10.0 * torch.log10(torch.mean(target_complex.abs() ** 2, dim=1).clamp_min(1e-12))
        power_loss = F.huber_loss(predicted_power, target_power)
        csi_scale = torch.sqrt(torch.mean(target_complex.abs() ** 2, dim=1, keepdim=True).clamp_min(1e-12))
        csi_loss = torch.mean(torch.abs(predicted_csi / csi_scale - target_complex / csi_scale) ** 2)
        set_loss = self.cir.matching_loss(memory, fused, target_complex)
        if task == "radiomap":
            return power_loss
        if task == "cir":
            return csi_loss + set_loss
        if task != "joint":
            raise ValueError("WiSER training task is invalid")
        return power_loss + csi_loss + set_loss


class RFIRForward(nn.Module):
    """Differentiable 2.5D RF-BSDF renderer over map-cell Gaussian primitives."""

    def __init__(self, material_count: int, antennas: int, subcarriers: int, context_dim: int, hidden: int, maximum_primitives: int):
        super().__init__()
        self.material_reflection = nn.Embedding(material_count, 2)
        self.material_roughness = nn.Embedding(material_count, 1)
        self.material_log_scale = nn.Embedding(material_count, 3)
        self.material_opacity = nn.Embedding(material_count, 1)
        self.attenuation = nn.Sequential(nn.Linear(5, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.array_phase = nn.Parameter(torch.zeros(antennas))
        self.context_modulation = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * subcarriers)
        )
        self.subcarriers = int(subcarriers)
        self.maximum_primitives = int(maximum_primitives)

    def _segment_visibility(
        self,
        occupancy: Tensor,
        start: Tensor,
        end: Tensor,
        origin: Tensor,
        resolution: float,
        samples: int = 8,
    ) -> Tensor:
        batch, _, rows, columns = occupancy.shape
        if start.ndim == 2:
            start = start[:, None].expand(-1, end.shape[1], -1)
        fractions = torch.linspace(0.1, 0.9, samples, device=occupancy.device, dtype=occupancy.dtype)
        points = start[:, :, None] + fractions[None, None, :, None] * (end - start)[:, :, None]
        xy = (points[..., :2] - origin[:, None, None]) / float(resolution)
        grid_x = 2.0 * (xy[..., 0] + 0.5) / max(columns, 1) - 1.0
        grid_y = 2.0 * (xy[..., 1] + 0.5) / max(rows, 1) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)
        density = F.grid_sample(
            occupancy,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[:, 0]
        return torch.exp(-float(resolution) * torch.sum(density, dim=-1))

    def _primitives(self, maps: Tensor, origin: Tensor, resolution: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, _, rows, columns = maps.shape
        occupancy = maps[:, 0] > 0.5
        height = maps[:, 1]
        material = maps[:, 2].round().long().clamp_min(0)
        y, x = torch.meshgrid(
            torch.arange(rows, device=maps.device), torch.arange(columns, device=maps.device), indexing="ij"
        )
        centers_xy = torch.stack(((x + 0.5) * resolution, (y + 0.5) * resolution), dim=-1).reshape(-1, 2)
        centers = []
        materials = []
        valid = []
        areas = []
        for index in range(batch):
            selected = torch.nonzero(occupancy[index].flatten(), as_tuple=False).flatten()[: self.maximum_primitives]
            count = selected.numel()
            pad = self.maximum_primitives - count
            xyz = torch.cat((centers_xy[selected] + origin[index], height[index].flatten()[selected, None]), dim=1)
            xyz = F.pad(xyz, (0, 0, 0, pad))
            mat = F.pad(material[index].flatten()[selected], (0, pad))
            mask = torch.cat((torch.ones(count, device=maps.device), torch.zeros(pad, device=maps.device)))
            area = mask * (resolution**2)
            centers.append(xyz)
            materials.append(mat)
            valid.append(mask)
            areas.append(area)
        return torch.stack(centers), torch.stack(materials), torch.stack(valid), torch.stack(areas)

    def forward(self, maps: Tensor, tx_xyz: Tensor, wireless_context: Tensor, rx_xy: Tensor, origin: Tensor, resolution: float) -> Tensor:
        rx_xyz = torch.cat((rx_xy, torch.full_like(rx_xy[:, :1], 1.5)), dim=1)
        centers, material, valid, area = self._primitives(maps, origin, resolution)
        tx_leg = centers - tx_xyz[:, None]
        rx_leg = rx_xyz[:, None] - centers
        d_tx = torch.linalg.vector_norm(tx_leg, dim=-1).clamp_min(0.25)
        d_rx = torch.linalg.vector_norm(rx_leg, dim=-1).clamp_min(0.25)
        direct_distance = torch.linalg.vector_norm(rx_xyz - tx_xyz, dim=-1).clamp_min(0.25)
        reflection_raw = self.material_reflection(material)
        reflection = torch.complex(reflection_raw[..., 0], reflection_raw[..., 1])
        roughness = F.softplus(self.material_roughness(material)[..., 0])
        gaussian_scale = F.softplus(self.material_log_scale(material)) * float(resolution)
        opacity = torch.sigmoid(self.material_opacity(material)[..., 0])
        incidence = torch.abs(tx_leg[..., 2]) / d_tx
        departure = torch.abs(rx_leg[..., 2]) / d_rx
        directional = torch.pow((incidence * departure).clamp_min(1e-4), 1.0 + roughness)
        projected_cross_section = math.pi * gaussian_scale[..., 0] * gaussian_scale[..., 1]
        projected_cross_section = projected_cross_section * torch.sqrt(
            incidence.square() + (gaussian_scale[..., 2] / gaussian_scale[..., 0].clamp_min(1e-6)).square()
        )
        tx_visibility = self._segment_visibility(maps[:, :1], tx_xyz, centers, origin, resolution)
        rx_visibility = self._segment_visibility(maps[:, :1], rx_xyz, centers, origin, resolution)
        visibility = tx_visibility * rx_visibility
        geometry = torch.stack((d_tx, d_rx, incidence, departure, roughness), dim=-1)
        attenuation = self.attenuation(geometry)
        attenuation_complex = torch.complex(attenuation[..., 0], attenuation[..., 1])
        primitive_alpha = valid * (1.0 - torch.exp(-opacity * projected_cross_section.clamp_min(0.0)))
        order = torch.argsort(d_tx + d_rx, dim=1)
        sorted_alpha = torch.gather(primitive_alpha, 1, order)
        sorted_transmittance = deterministic_prefix_product(
            torch.cat((torch.ones_like(sorted_alpha[:, :1]), 1.0 - sorted_alpha[:, :-1]), dim=1),
            dim=1,
        )
        transmittance = torch.zeros_like(sorted_transmittance).scatter(1, order, sorted_transmittance)
        scatter_amplitude = (
            valid * area * projected_cross_section * directional * visibility * transmittance / (d_tx * d_rx)
        )
        frequency = torch.linspace(0.0, 1.0, self.subcarriers, device=maps.device)
        scatter_phase = torch.exp(-2j * math.pi * (d_tx + d_rx)[..., None] * frequency / 10.0)
        scattered = torch.sum(
            scatter_amplitude[..., None] * reflection[..., None] * attenuation_complex[..., None] * scatter_phase,
            dim=1,
        )
        direct_phase = torch.exp(-2j * math.pi * direct_distance[:, None] * frequency / 10.0)
        direct = direct_phase / direct_distance[:, None]
        modulation = self.context_modulation(wireless_context).reshape(-1, self.subcarriers, 2)
        modulation = 1.0 + 0.1 * torch.complex(modulation[..., 0], modulation[..., 1])
        scalar = (direct + scattered) * modulation
        array = torch.exp(1j * self.array_phase)[None, :, None]
        return (array * scalar[:, None]).reshape(maps.shape[0], -1)

    def training_loss(self, csi: Tensor, maps: Tensor, tx_xyz: Tensor, wireless_context: Tensor, rx_xy: Tensor, origin: Tensor, resolution: float) -> Tensor:
        prediction = self(maps, tx_xyz, wireless_context, rx_xy, origin, resolution)
        target = complex_csi(csi)
        scale = torch.sqrt(torch.mean(target.abs() ** 2, dim=1, keepdim=True).clamp_min(1e-12))
        signal_loss = torch.mean(torch.abs(prediction / scale - target / scale) ** 2)
        power_loss = F.huber_loss(
            10.0 * torch.log10(torch.mean(prediction.abs() ** 2, dim=1).clamp_min(1e-12)),
            10.0 * torch.log10(torch.mean(target.abs() ** 2, dim=1).clamp_min(1e-12)),
        )
        return signal_loss + power_loss
