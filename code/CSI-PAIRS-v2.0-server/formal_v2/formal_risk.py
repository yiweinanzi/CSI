from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .formal_evidence import (
    bind_rows,
    evidence_context,
    require_manifested_formal_qualification,
)
from .formal_io import artifact_manifest, parse_strict_json, sha256_file, write_csv, write_json
from .formal_metrics import (
    binary_nll,
    brier_score,
    expected_calibration_error,
    risk_coverage,
    spearman_correlation,
)
from .formal_model import resolve_execution_device, tensor_for_module
from .formal_protocol import typed_signed_edit
from .formal_routing import ROUTE_NAMES


_FIRST_PARTY_RISK_TOKEN = object()


def _plain_risk_metadata(value):
    if isinstance(value, Mapping):
        return {str(key): _plain_risk_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_risk_metadata(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze_risk_metadata(value):
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_risk_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_risk_metadata(item) for item in value)
    return value


def _risk_feature_payload_sha256(payload, mixture_freeze=None, blocking_strata=None):
    digest = hashlib.sha256()
    for arm in sorted(payload):
        arm_name = str(arm).encode("utf-8")
        digest.update(len(arm_name).to_bytes(8, "big"))
        digest.update(arm_name)
        fields = payload[arm]
        if not isinstance(fields, Mapping):
            raise TypeError("first-party risk arm payload must be a mapping")
        for field in sorted(fields):
            values = np.asarray(fields[field])
            if values.dtype.hasobject:
                raise TypeError("first-party risk payload cannot contain object arrays")
            contiguous = np.ascontiguousarray(values)
            header = json.dumps(
                [str(field), contiguous.dtype.str, list(contiguous.shape)],
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
            digest.update(len(header).to_bytes(8, "big"))
            digest.update(header)
            raw = contiguous.tobytes()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    metadata = json.dumps(
        {
            "mixture_freeze": _plain_risk_metadata(mixture_freeze),
            "blocking_strata": _plain_risk_metadata(blocking_strata),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest.update(metadata)
    return digest.hexdigest()


class FirstPartyRiskFeatures(Mapping):
    """Immutable replay payload carrying module-private provenance."""

    __slots__ = (
        "_payload",
        "_provenance",
        "_sealed",
        "blocking_strata",
        "mixture_freeze",
        "replay_binding",
    )

    def __init__(
        self,
        payload=None,
        *,
        mixture_freeze=None,
        blocking_strata=None,
        replay_binding=None,
        _token=None,
    ):
        if _token is not _FIRST_PARTY_RISK_TOKEN:
            raise TypeError(
                "FirstPartyRiskFeatures can only be constructed by first-party replay"
            )
        frozen = {}
        for arm, fields in (payload or {}).items():
            if not isinstance(fields, Mapping):
                raise TypeError("first-party risk arm payload must be a mapping")
            frozen_fields = {}
            for field, source in fields.items():
                source_values = np.asarray(source)
                values = np.ascontiguousarray(source_values)
                if values.dtype.hasobject:
                    raise TypeError("first-party risk payload cannot contain object arrays")
                immutable = np.frombuffer(
                    values.tobytes(), dtype=values.dtype
                ).reshape(source_values.shape)
                immutable.setflags(write=False)
                frozen_fields[str(field)] = immutable
            frozen[str(arm)] = MappingProxyType(frozen_fields)
        object.__setattr__(self, "_payload", MappingProxyType(frozen))
        object.__setattr__(self, "_provenance", _FIRST_PARTY_RISK_TOKEN)
        object.__setattr__(
            self, "mixture_freeze", _freeze_risk_metadata(mixture_freeze)
        )
        object.__setattr__(
            self, "blocking_strata", _freeze_risk_metadata(blocking_strata)
        )
        object.__setattr__(
            self, "replay_binding", _freeze_risk_metadata(replay_binding)
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise TypeError("first-party risk replay payload is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, "_sealed", False):
            raise TypeError("first-party risk replay payload is immutable")
        object.__delattr__(self, name)

    def __getitem__(self, key):
        return self._payload[key]

    def __iter__(self):
        return iter(self._payload)

    def __len__(self):
        return len(self._payload)


def _freeze_first_party_risk_features(
    payload,
    *,
    mixture_freeze=None,
    blocking_strata=None,
    replay_binding=None,
):
    return FirstPartyRiskFeatures(
        payload,
        mixture_freeze=mixture_freeze,
        blocking_strata=blocking_strata,
        replay_binding=replay_binding,
        _token=_FIRST_PARTY_RISK_TOKEN,
    )


def _risk_replay_binding(config, dataset, output_root, payload, mixture, blocking):
    from .formal_upstream import resolve_authenticated_upstream

    root = Path(output_root)
    upstream = resolve_authenticated_upstream(config, dataset, root)
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    )
    return {
        "schema_version": "csi-pairs-v6-risk-replay-binding-v1",
        "dataset_sha256": evidence["dataset_sha256"],
        "config_sha256": evidence["config_sha256"],
        "checkpoint_index_sha256": sha256_file(
            upstream.checkpoint_index
        ),
        "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
        "payload_sha256": _risk_feature_payload_sha256(
            payload, mixture_freeze=mixture, blocking_strata=blocking
        ),
    }


def _make_first_party_risk_features(
    payload,
    config,
    dataset,
    output_root,
    *,
    mixture_freeze=None,
    blocking_strata=None,
):
    binding = _risk_replay_binding(
        config,
        dataset,
        output_root,
        payload,
        mixture_freeze,
        blocking_strata,
    )
    return _freeze_first_party_risk_features(
        payload,
        mixture_freeze=mixture_freeze,
        blocking_strata=blocking_strata,
        replay_binding=binding,
    )


def _validate_first_party_risk_features(features, config, dataset, output_root):
    if (
        type(features) is not FirstPartyRiskFeatures
        or features._provenance is not _FIRST_PARTY_RISK_TOKEN
    ):
        raise RuntimeError("risk replay lacks first-party module provenance")
    expected = _risk_replay_binding(
        config,
        dataset,
        output_root,
        features,
        features.mixture_freeze,
        features.blocking_strata,
    )
    actual = _plain_risk_metadata(features.replay_binding)
    if actual != expected:
        raise RuntimeError(
            "first-party risk replay binding does not match payload/checkpoint/dataset/config"
        )


class RiskStrataUnavailable(RuntimeError):
    def __init__(self, role, counts):
        self.role = str(role)
        self.counts = {str(key): int(value) for key, value in counts.items()}
        super().__init__(
            f"risk role {self.role} lacks one or more frozen audit conditions: "
            f"{self.counts}"
        )


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float

    def predict(self, score_difference: np.ndarray) -> np.ndarray:
        values = np.asarray(score_difference, dtype=np.float64) / float(self.temperature)
        return _sigmoid(values)


@dataclass(frozen=True)
class RobustStandardization:
    median: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.median) / self.scale


@dataclass(frozen=True)
class SupportModel:
    center: np.ndarray
    covariance: np.ndarray
    threshold: float

    def squared_distance(self, values: np.ndarray) -> np.ndarray:
        delta = np.asarray(values, dtype=np.float64) - self.center
        inverse = np.linalg.inv(self.covariance)
        return np.einsum("bi,ij,bj->b", delta, inverse, delta)


@dataclass(frozen=True)
class ConstrainedRiskCalibration:
    intercept: float
    beta_d: float
    beta_u: float
    standardization: RobustStandardization
    support: SupportModel
    l2: float

    def predict(self, d_used: np.ndarray, uncertainty: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = np.column_stack((d_used, uncertainty))
        standardized = np.clip(self.standardization.transform(raw), -5.0, 5.0)
        logits = self.intercept + self.beta_d * standardized[:, 0] + self.beta_u * standardized[:, 1]
        inside = self.support.squared_distance(standardized) <= self.support.threshold
        return _sigmoid(logits), inside



@dataclass(frozen=True)
class ScalarRiskCalibration:
    intercept: float
    coefficient: float
    median: float
    scale: float
    l2: float
    direction: str

    def predict(self, values: np.ndarray) -> np.ndarray:
        standardized = np.clip(
            (np.asarray(values, dtype=np.float64) - self.median) / self.scale,
            -5.0,
            5.0,
        )
        return _sigmoid(self.intercept + self.coefficient * standardized)


def frozen_map_proposals(
    supplied_map: np.ndarray,
    public_radio_config: np.ndarray,
    map_channel_names: np.ndarray,
    material_categories: int,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate a frozen candidate set without CSI, route, identity, position, or outcome inputs."""
    world_map = np.asarray(supplied_map, dtype=np.float64)
    radio = np.asarray(public_radio_config, dtype=np.float64)
    material_categories = int(material_categories)
    if material_categories < 1:
        raise ValueError("map proposal material vocabulary must be nonempty")
    names = [str(value) for value in np.asarray(map_channel_names).tolist()]
    if world_map.ndim != 3 or not np.all(np.isfinite(world_map)) or not np.all(np.isfinite(radio)):
        raise ValueError("map proposal inputs must be finite typed map and public radio arrays")
    for required in ("occupancy", "height", "material"):
        if required not in names:
            raise ValueError(f"map proposal requires {required!r} channel")
    occupancy_index = names.index("occupancy")
    height_index = names.index("height")
    material_index = names.index("material")
    seed_material = np.ascontiguousarray(world_map).tobytes() + np.ascontiguousarray(radio).tobytes()
    digest_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    rng = np.random.default_rng(digest_seed ^ int(config["risk"]["proposal_seed"]))
    cells = [(row, column) for row in range(world_map.shape[1]) for column in range(world_map.shape[2])]
    order = rng.permutation(len(cells))
    height_step = float(config["risk"]["proposal_height_step"])
    candidates = []
    candidate_digests = set()

    def append_candidate(candidate):
        if np.array_equal(candidate, world_map):
            return False
        digest = hashlib.sha256(np.ascontiguousarray(candidate).tobytes()).hexdigest()
        if digest in candidate_digests:
            return False
        candidate_digests.add(digest)
        candidates.append(candidate)
        return True
    for cell_index in order:
        row, column = cells[int(cell_index)]
        occupied = world_map[occupancy_index, row, column] >= 0.5
        operations = ("remove", "height_down", "height_up", "material") if occupied else ("add",)
        for operation in operations:
            candidate = world_map.copy()
            if operation == "remove":
                candidate[occupancy_index, row, column] = 0.0
                candidate[height_index, row, column] = 0.0
                candidate[material_index, row, column] = 0.0
            elif operation == "add":
                candidate[occupancy_index, row, column] = 1.0
                positive = world_map[height_index][world_map[height_index] > 0]
                candidate[height_index, row, column] = float(np.median(positive)) if positive.size else height_step
                materials = world_map[material_index][world_map[occupancy_index] >= 0.5].astype(np.int64)
                candidate[material_index, row, column] = (
                    int(np.bincount(materials, minlength=material_categories).argmax())
                    if materials.size
                    else 0
                )
            elif operation == "height_down":
                candidate[height_index, row, column] = max(0.0, candidate[height_index, row, column] - height_step)
            elif operation == "height_up":
                candidate[height_index, row, column] += height_step
            else:
                current = int(round(candidate[material_index, row, column]))
                candidate[material_index, row, column] = (current + 1) % int(material_categories)
            append_candidate(candidate)
            if len(candidates) >= int(config["risk"]["proposal_count"]):
                break
        if len(candidates) >= int(config["risk"]["proposal_count"]):
            break
    if not candidates:
        raise RuntimeError("frozen map proposal library produced no legal candidates")
    primary_count = len(candidates)
    fallback_index = 0
    requested = int(config["risk"]["proposal_count"])
    while len(candidates) < requested:
        row, column = cells[int(order[fallback_index % len(order)])]
        cycle = fallback_index // len(order) + 1
        candidate = world_map.copy()
        if world_map[occupancy_index, row, column] >= 0.5:
            candidate[height_index, row, column] = (
                world_map[height_index, row, column] + height_step * (cycle + 1)
            )
        else:
            candidate[occupancy_index, row, column] = 1.0
            candidate[height_index, row, column] = height_step * (cycle + 1)
            candidate[material_index, row, column] = (
                0
                if material_categories == 1
                else 1 + cycle % (material_categories - 1)
            )
        append_candidate(candidate)
        fallback_index += 1
        if fallback_index > requested * len(cells) * 2:
            raise RuntimeError(
                "frozen proposal fallback could not realize the registered count"
            )
    maps = np.asarray(candidates)
    actions = np.asarray(
        [typed_signed_edit(world_map, candidate, map_channel_names, material_categories) for candidate in maps]
    )
    return maps, actions, {
        "proposal_count": int(len(maps)),
        "primary_count": int(primary_count),
        "fallback_count": int(len(maps) - primary_count),
        "proposal_set_sha256": hashlib.sha256(
            np.ascontiguousarray(maps).tobytes()
        ).hexdigest(),
    }


def used_map_score_margin(used_score, candidate_scores):
    candidates = np.asarray(candidate_scores, dtype=np.float64)
    if candidates.ndim != 1 or candidates.size == 0 or not np.all(np.isfinite(candidates)):
        raise ValueError(
            "used-map margin requires a nonempty finite candidate-score vector"
        )
    return float(np.min(float(used_score) - candidates))


def generating_map_pair_margin(matched_score, alternative_score):
    values = np.asarray([matched_score, alternative_score], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("paired compatibility scores must be finite")
    return float(values[0] - values[1])


def _normalized_sample_weights(sample_weights, size):
    if sample_weights is None:
        return None
    weights = np.asarray(sample_weights, dtype=np.float64)
    if (
        weights.shape != (int(size),)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or float(np.sum(weights)) <= 0.0
    ):
        raise ValueError("sample weights must be aligned, finite, nonnegative, and nonzero")
    return weights / float(np.sum(weights))


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        return np.mean(values, axis=0)
    return np.sum(values * np.asarray(weights)[(...,) + (None,) * (values.ndim - 1)], axis=0)


def _weighted_quantile(values, quantile, weights):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("weighted quantile requires one-dimensional values")
    if weights is None:
        return float(np.quantile(values, quantile))
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(ordered_weights)
    threshold = float(quantile) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(ordered_values[min(index, ordered_values.size - 1)])


def _weighted_binary_nll(labels, probabilities, weights):
    if weights is None:
        return binary_nll(labels, probabilities)
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return float(np.sum(np.asarray(weights) * (-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))))


def _hierarchical_sample_weights(seed_ids, cluster_ids, bank_ids):
    """Equalize units, seeds within bank, banks within foundation, then foundations."""
    seeds = np.asarray(seed_ids).astype(str)
    clusters = np.asarray(cluster_ids).astype(str)
    banks = np.asarray(bank_ids).astype(str)
    if not (seeds.shape == clusters.shape == banks.shape) or seeds.ndim != 1:
        raise ValueError("hierarchical sample identifiers must be aligned vectors")
    if seeds.size == 0:
        raise ValueError("hierarchical sample weighting requires observations")
    weights = np.zeros(seeds.size, dtype=np.float64)
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        cluster_mask = clusters == cluster
        cluster_banks = np.unique(banks[cluster_mask])
        for bank in cluster_banks:
            bank_mask = cluster_mask & (banks == bank)
            bank_seeds = np.unique(seeds[bank_mask])
            for seed_id in bank_seeds:
                selected = bank_mask & (seeds == seed_id)
                weights[selected] = (
                    1.0
                    / unique_clusters.size
                    / cluster_banks.size
                    / bank_seeds.size
                    / int(np.sum(selected))
                )
    return weights / float(np.sum(weights))


def fit_temperature_no_intercept(
    score_difference: np.ndarray,
    labels: np.ndarray,
    config: dict,
    *,
    sample_weights=None,
) -> TemperatureCalibration:
    d = np.asarray(score_difference, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if d.shape != y.shape or d.ndim != 1:
        raise ValueError("temperature calibration inputs must be aligned vectors")
    weights = _normalized_sample_weights(sample_weights, d.size)
    log_temperature = 0.0
    learning_rate = float(config["risk"]["learning_rate"])
    for _ in range(int(config["risk"]["temperature_steps"])):
        temperature = np.exp(log_temperature)
        probability = _sigmoid(d / temperature)
        gradient = _weighted_mean((probability - y) * (-d / temperature), weights)
        log_temperature -= learning_rate * gradient
        log_temperature = float(np.clip(log_temperature, -8.0, 8.0))
    return TemperatureCalibration(float(np.exp(log_temperature)))


def fit_constrained_risk_calibrator(
    d_used: np.ndarray,
    uncertainty: np.ndarray,
    labels: np.ndarray,
    selection: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: dict,
    *,
    fit_weights=None,
    selection_weights=None,
) -> ConstrainedRiskCalibration:
    raw = np.column_stack((d_used, uncertainty)).astype(np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if raw.shape[0] != y.size:
        raise ValueError("risk calibration fit inputs must align")
    weights = _normalized_sample_weights(fit_weights, y.size)
    median = (
        np.median(raw, axis=0)
        if weights is None
        else np.asarray(
            [_weighted_quantile(raw[:, index], 0.5, weights) for index in range(2)]
        )
    )
    mad = (
        np.median(np.abs(raw - median), axis=0)
        if weights is None
        else np.asarray(
            [
                _weighted_quantile(np.abs(raw[:, index] - median[index]), 0.5, weights)
                for index in range(2)
            ]
        )
    )
    scale = 1.4826 * mad + float(config["risk"]["epsilon"])
    standardization = RobustStandardization(median=median, scale=scale)
    x = np.clip(standardization.transform(raw), -5.0, 5.0)
    selection_d, selection_u, selection_y = selection
    selection_x = np.clip(
        standardization.transform(np.column_stack((selection_d, selection_u))), -5.0, 5.0
    )
    selection_y = np.asarray(selection_y, dtype=np.float64)
    selection_weight_values = _normalized_sample_weights(
        selection_weights, selection_y.size
    )
    if selection_x.shape[0] != selection_y.size:
        raise ValueError("risk calibration selection inputs must align")
    candidates = []
    for l2 in config["risk"]["l2_grid"]:
        beta = np.zeros(3, dtype=np.float64)
        for _ in range(int(config["risk"]["logistic_steps"])):
            probability = _sigmoid(beta[0] + x @ beta[1:])
            gradient = np.asarray(
                [
                    _weighted_mean(probability - y, weights),
                    _weighted_mean((probability - y) * x[:, 0], weights)
                    + float(l2) * beta[1],
                    _weighted_mean((probability - y) * x[:, 1], weights)
                    + float(l2) * beta[2],
                ]
            )
            beta -= float(config["risk"]["learning_rate"]) * gradient
            beta[1] = min(beta[1], 0.0)
            beta[2] = max(beta[2], 0.0)
        selection_probability = _sigmoid(beta[0] + selection_x @ beta[1:])
        candidates.append(
            (
                _weighted_binary_nll(
                    selection_y, selection_probability, selection_weight_values
                ),
                float(l2),
                beta.copy(),
            )
        )
    _, selected_l2, selected_beta = min(candidates, key=lambda item: (item[0], item[1]))
    if weights is None:
        covariance = np.cov(x, rowvar=False)
        center = np.mean(x, axis=0)
    else:
        center = _weighted_mean(x, weights)
        delta = x - center
        covariance = np.einsum("b,bi,bj->ij", weights, delta, delta)
    covariance = covariance + float(config["risk"]["epsilon"]) * np.eye(2)
    inverse = np.linalg.inv(covariance)
    selection_delta = selection_x - center
    selection_distances = np.einsum(
        "bi,ij,bj->b", selection_delta, inverse, selection_delta
    )
    support = SupportModel(
        center=center,
        covariance=covariance,
        threshold=float(
            _weighted_quantile(
                selection_distances,
                1.0 - float(config["risk"]["support_alpha"]),
                selection_weight_values,
            )
        ),
    )
    return ConstrainedRiskCalibration(
        intercept=float(selected_beta[0]),
        beta_d=float(selected_beta[1]),
        beta_u=float(selected_beta[2]),
        standardization=standardization,
        support=support,
        l2=selected_l2,
    )


def fit_scalar_risk_calibrator(
    values: np.ndarray,
    labels: np.ndarray,
    selection: tuple[np.ndarray, np.ndarray],
    config: dict,
    *,
    direction: str,
    fit_weights=None,
    selection_weights=None,
) -> ScalarRiskCalibration:
    """Fit and select a restricted one-feature control independently."""
    if direction not in {"nonpositive", "nonnegative"}:
        raise ValueError("scalar risk coefficient direction is invalid")
    raw = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    selection_values, selection_y = selection
    if raw.shape != y.shape or raw.ndim != 1:
        raise ValueError("scalar risk fit inputs must be aligned vectors")
    weights = _normalized_sample_weights(fit_weights, raw.size)
    selection_values = np.asarray(selection_values, dtype=np.float64)
    selection_y = np.asarray(selection_y, dtype=np.float64)
    if selection_values.shape != selection_y.shape or selection_values.ndim != 1:
        raise ValueError("scalar risk selection inputs must be aligned vectors")
    selection_weight_values = _normalized_sample_weights(
        selection_weights, selection_values.size
    )
    median = _weighted_quantile(raw, 0.5, weights)
    scale = float(
        1.4826 * _weighted_quantile(np.abs(raw - median), 0.5, weights)
        + float(config["risk"]["epsilon"])
    )
    x = np.clip((raw - median) / scale, -5.0, 5.0)
    selection_x = np.clip(
        (selection_values - median) / scale,
        -5.0,
        5.0,
    )
    candidates = []
    for l2 in config["risk"]["l2_grid"]:
        beta = np.zeros(2, dtype=np.float64)
        for _ in range(int(config["risk"]["logistic_steps"])):
            probability = _sigmoid(beta[0] + beta[1] * x)
            gradient = np.asarray(
                [
                    _weighted_mean(probability - y, weights),
                    _weighted_mean((probability - y) * x, weights)
                    + float(l2) * beta[1],
                ]
            )
            beta -= float(config["risk"]["learning_rate"]) * gradient
            beta[1] = min(beta[1], 0.0) if direction == "nonpositive" else max(beta[1], 0.0)
        probability = _sigmoid(beta[0] + beta[1] * selection_x)
        candidates.append(
            (
                _weighted_binary_nll(
                    selection_y, probability, selection_weight_values
                ),
                float(l2),
                beta.copy(),
            )
        )
    _, selected_l2, selected_beta = min(candidates, key=lambda item: (item[0], item[1]))
    return ScalarRiskCalibration(
        intercept=float(selected_beta[0]),
        coefficient=float(selected_beta[1]),
        median=median,
        scale=scale,
        l2=selected_l2,
        direction=direction,
    )


_RISK_HIERARCHY = (
    "canonical unit, equal algorithm seeds within canonical bank, "
    "equal canonical banks within foundation, equal canonical foundations"
)


def _risk_seed_array(seed_ids, size):
    if seed_ids is None:
        return np.asarray(["single-seed"] * int(size))
    seeds = np.asarray(seed_ids).astype(str)
    if seeds.shape != (int(size),):
        raise ValueError("risk seed IDs must align with observations")
    return seeds


def _require_complete_seed_bank_grid(
    seed_ids,
    cluster_ids,
    bank_ids,
    *,
    eligible=None,
    minimum_per_cell=1,
    context="risk hierarchy",
):
    seeds = np.asarray(seed_ids).astype(str)
    clusters = np.asarray(cluster_ids).astype(str)
    banks = np.asarray(bank_ids).astype(str)
    if not (seeds.shape == clusters.shape == banks.shape) or seeds.ndim != 1:
        raise ValueError(f"{context} identifiers must be aligned vectors")
    mask = (
        np.ones(seeds.size, dtype=np.bool_)
        if eligible is None
        else np.asarray(eligible, dtype=np.bool_)
    )
    if mask.shape != seeds.shape:
        raise ValueError(f"{context} eligibility must align with identifiers")
    expected_seeds = set(np.unique(seeds).tolist())
    missing = []
    for cluster, bank in sorted(set(zip(clusters.tolist(), banks.tolist()))):
        available = {
            seed_id
            for seed_id in expected_seeds
            if int(
                np.sum(
                    mask
                    & (seeds == seed_id)
                    & (clusters == cluster)
                    & (banks == bank)
                )
            )
            >= int(minimum_per_cell)
        }
        if available != expected_seeds:
            missing.append(
                {
                    "cluster": cluster,
                    "bank": bank,
                    "missing_seeds": sorted(expected_seeds - available),
                }
            )
    if missing:
        raise RuntimeError(
            f"{context} lacks a complete paired seed-by-bank grid: {missing}"
        )


def _hierarchical_cell_macro(rows, fields):
    if not rows:
        raise RuntimeError("hierarchical risk aggregation has no cells")
    canonical = {}
    for row in rows:
        key = (str(row["cluster"]), str(row["bank"]), str(row["seed"]))
        values = {field: float(row[field]) for field in fields}
        if key in canonical:
            if any(
                not np.isclose(
                    canonical[key][field], values[field], rtol=0.0, atol=1e-12,
                    equal_nan=True,
                )
                for field in fields
            ):
                raise RuntimeError(
                    f"duplicated hierarchical risk cell {key!r} is inconsistent"
                )
            continue
        canonical[key] = values
    cell_rows = [
        {"cluster": key[0], "bank": key[1], "seed": key[2], **values}
        for key, values in sorted(canonical.items())
    ]
    _require_complete_seed_bank_grid(
        np.asarray([row["seed"] for row in cell_rows]),
        np.asarray([row["cluster"] for row in cell_rows]),
        np.asarray([row["bank"] for row in cell_rows]),
        context="hierarchical risk macro",
    )
    foundation_rows = []
    for cluster in sorted({row["cluster"] for row in cell_rows}):
        cluster_cells = [row for row in cell_rows if row["cluster"] == cluster]
        bank_rows = []
        for bank in sorted({row["bank"] for row in cluster_cells}):
            seed_cells = [row for row in cluster_cells if row["bank"] == bank]
            bank_rows.append(
                {
                    field: float(np.mean([row[field] for row in seed_cells]))
                    for field in fields
                }
            )
        foundation_rows.append(
            {
                field: float(np.mean([row[field] for row in bank_rows]))
                for field in fields
            }
        )
    return {
        field: float(np.mean([row[field] for row in foundation_rows]))
        for field in fields
    }


def _canonical_hierarchy_value_cells(rows):
    canonical = {}
    for row in rows:
        key = (str(row["cluster"]), str(row["bank"]), str(row["seed"]))
        value = float(row["value"])
        if not np.isfinite(value):
            raise RuntimeError(f"hierarchical bootstrap cell {key!r} is non-finite")
        if key in canonical and not np.isclose(
            canonical[key], value, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(
                f"duplicated hierarchical bootstrap cell {key!r} is inconsistent"
            )
        canonical[key] = value
    output = [
        {"cluster": key[0], "bank": key[1], "seed": key[2], "value": value}
        for key, value in sorted(canonical.items())
    ]
    if not output:
        raise RuntimeError("synchronized hierarchical bootstrap has no cells")
    _require_complete_seed_bank_grid(
        np.asarray([row["seed"] for row in output]),
        np.asarray([row["cluster"] for row in output]),
        np.asarray([row["bank"] for row in output]),
        context="synchronized hierarchical bootstrap",
    )
    if len({row["cluster"] for row in output}) < 2:
        raise RuntimeError(
            "synchronized hierarchical bootstrap requires two canonical foundations"
        )
    return output


def _synchronized_hierarchical_bootstrap(
    hypotheses, resamples, seed, *, trace=False
):
    if not hypotheses or int(resamples) < 2:
        raise ValueError("synchronized hierarchical bootstrap requires hypotheses and resamples")
    cells = {
        str(name): _canonical_hierarchy_value_cells(rows)
        for name, rows in hypotheses.items()
    }
    seed_sets = [
        {row["seed"] for row in rows}
        for rows in cells.values()
    ]
    seeds = sorted(seed_sets[0])
    if not seeds or any(values != set(seeds) for values in seed_sets):
        raise RuntimeError(
            "synchronized hierarchical bootstrap requires one complete global seed set"
        )
    layouts = {}
    lookups = {}
    hypothesis_layout = {}
    for name, rows in cells.items():
        foundations = sorted({row["cluster"] for row in rows})
        layout = tuple(
            (
                foundation,
                tuple(
                    sorted(
                        {
                            row["bank"]
                            for row in rows
                            if row["cluster"] == foundation
                        }
                    )
                ),
            )
            for foundation in foundations
        )
        layouts[layout] = layout
        hypothesis_layout[name] = layout
        lookups[name] = {
            (row["cluster"], row["bank"], row["seed"]): row["value"]
            for row in rows
        }
    estimates = {
        name: _hierarchical_cell_macro(rows, ("value",))["value"]
        for name, rows in cells.items()
    }
    samples = {
        name: np.empty(int(resamples), dtype=np.float64) for name in cells
    }
    rng = np.random.default_rng(int(seed))
    traced_seed_draws = []
    digest = hashlib.sha256()
    for index in range(int(resamples)):
        structure_draws = {}
        for layout in sorted(layouts, key=repr):
            selected_foundations = [
                layout[value][0]
                for value in rng.integers(0, len(layout), size=len(layout))
            ]
            banks_by_foundation = dict(layout)
            structure_draws[layout] = [
                (
                    foundation,
                    tuple(
                        banks_by_foundation[foundation][value]
                        for value in rng.integers(
                            0,
                            len(banks_by_foundation[foundation]),
                            size=len(banks_by_foundation[foundation]),
                        )
                    ),
                )
                for foundation in selected_foundations
            ]
        selected_seeds = tuple(
            seeds[value]
            for value in rng.integers(0, len(seeds), size=len(seeds))
        )
        encoded_seed_draw = json.dumps(
            selected_seeds, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        digest.update(len(encoded_seed_draw).to_bytes(8, "big"))
        digest.update(encoded_seed_draw)
        if trace:
            traced_seed_draws.append(list(selected_seeds))
        for name in sorted(cells):
            foundation_values = []
            for foundation, selected_banks in structure_draws[
                hypothesis_layout[name]
            ]:
                foundation_values.append(
                    float(
                        np.mean(
                            [
                                np.mean(
                                    [
                                        lookups[name][
                                            (foundation, bank, selected_seed)
                                        ]
                                        for selected_seed in selected_seeds
                                    ]
                                )
                                for bank in selected_banks
                            ]
                        )
                    )
                )
            samples[name][index] = float(np.mean(foundation_values))
    audit = {
        "resamples": int(resamples),
        "resampling_layers": [
            "canonical_foundation",
            "canonical_bank_within_foundation",
            "one_global_paired_algorithm_seed_draw_shared_by_all_banks_and_hypotheses",
        ],
        "global_seed_count": len(seeds),
        "layout_count": len(layouts),
        "shared_seed_draw_across_all_banks_and_hypotheses": True,
        "seed_draws_sha256": digest.hexdigest(),
    }
    if trace:
        audit["seed_draws"] = traced_seed_draws
    return {"estimates": estimates, "samples": samples, "audit": audit}


def _simultaneous_hierarchical_family(
    hypotheses,
    resamples,
    seed,
    *,
    alpha=0.05,
    trace=False,
):
    bootstrapped = _synchronized_hierarchical_bootstrap(
        hypotheses, resamples, seed, trace=trace
    )
    estimates = bootstrapped["estimates"]
    samples = bootstrapped["samples"]
    scales = {
        name: float(np.std(values, ddof=1))
        for name, values in samples.items()
    }
    standardized = []
    for name in sorted(samples):
        scale = scales[name]
        if scale <= 1e-15:
            standardized.append(np.zeros(int(resamples), dtype=np.float64))
        else:
            standardized.append(
                np.abs(samples[name] - estimates[name]) / scale
            )
    maximum_deviation = np.max(np.column_stack(standardized), axis=1)
    critical = float(np.percentile(maximum_deviation, 100.0 * (1.0 - float(alpha))))
    records = {}
    for name in sorted(samples):
        scale = scales[name]
        records[name] = {
            "estimate": float(estimates[name]),
            "ci95_low": float(np.percentile(samples[name], 2.5)),
            "ci95_high": float(np.percentile(samples[name], 97.5)),
            "familywise_ci95_low": float(estimates[name] - critical * scale),
            "familywise_ci95_high": float(estimates[name] + critical * scale),
            "bootstrap_standard_error": scale,
        }
    family = {
        **bootstrapped["audit"],
        "schema_version": "csi-pairs-v6-risk-synchronized-family-v1",
        "hypothesis_count": len(records),
        "familywise_alpha": float(alpha),
        "familywise_method": (
            "synchronized_hierarchical_bootstrap_studentized_max_absolute_deviation"
        ),
        "simultaneous_critical_value": critical,
        "point_estimate_hierarchy": _RISK_HIERARCHY,
    }
    return {"records": records, "family": family}


def _bank_cluster_mean(values, cluster_ids, bank_ids, seed_ids=None):
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(cluster_ids).astype(str)
    banks = np.asarray(bank_ids).astype(str)
    if values.shape != clusters.shape or values.shape != banks.shape:
        raise ValueError("bank-cluster values and identifiers must align")
    seeds = _risk_seed_array(seed_ids, values.size)
    rows = []
    for seed_id in np.unique(seeds):
        for cluster in np.unique(clusters[seeds == seed_id]):
            for bank in np.unique(banks[(seeds == seed_id) & (clusters == cluster)]):
                selected = (
                    (seeds == seed_id) & (clusters == cluster) & (banks == bank)
                )
                rows.append(
                    {
                        "seed": str(seed_id),
                        "cluster": str(cluster),
                        "bank": str(bank),
                        "value": float(np.mean(values[selected])),
                    }
                )
    return _hierarchical_cell_macro(rows, ("value",))["value"]


def _bank_cluster_probability_metrics(
    labels, probabilities, cluster_ids, bank_ids, seed_ids=None
):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    clusters = np.asarray(cluster_ids).astype(str)
    banks = np.asarray(bank_ids).astype(str)
    if not (y.shape == p.shape == clusters.shape == banks.shape):
        raise ValueError("calibration values and canonical identifiers must align")
    seeds = _risk_seed_array(seed_ids, y.size)
    _require_complete_seed_bank_grid(
        seeds, clusters, banks, context="calibration metric"
    )
    rows = []
    for seed_id in np.unique(seeds):
        for cluster in np.unique(clusters[seeds == seed_id]):
            for bank in np.unique(banks[(seeds == seed_id) & (clusters == cluster)]):
                selected = (
                    (seeds == seed_id) & (clusters == cluster) & (banks == bank)
                )
                rows.append(
                    {
                        "seed": str(seed_id),
                        "cluster": str(cluster),
                        "bank": str(bank),
                        "ece": expected_calibration_error(y[selected], p[selected]),
                        "brier": brier_score(y[selected], p[selected]),
                        "nll": binary_nll(y[selected], p[selected]),
                    }
                )
    result = _hierarchical_cell_macro(rows, ("ece", "brier", "nll"))
    result.update(
        {
            "algorithm_seed_count": int(len(np.unique(seeds))),
            "base_map_cluster_count": int(len(np.unique(clusters))),
            "canonical_bank_count": int(
                len(set(zip(clusters.tolist(), banks.tolist())))
            ),
            "algorithm_seed_bank_cell_count": int(
                len(set(zip(seeds.tolist(), clusters.tolist(), banks.tolist())))
            ),
            "aggregation": _RISK_HIERARCHY,
        }
    )
    return result


def risk_metrics(
    labels,
    probabilities,
    errors,
    inside_support,
    cluster_ids=None,
    bank_ids=None,
    unit_ids=None,
    seed_ids=None,
):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    error = np.asarray(errors, dtype=np.float64)
    inside = np.asarray(inside_support, dtype=np.bool_)
    if not (y.shape == p.shape == error.shape == inside.shape) or y.ndim != 1:
        raise ValueError("risk metric inputs must be aligned one-dimensional arrays")
    clusters = None if cluster_ids is None else np.asarray(cluster_ids).astype(str)
    if bank_ids is not None and clusters is None:
        raise ValueError("canonical bank IDs require canonical cluster IDs")
    banks = (
        None
        if clusters is None
        else (
            np.asarray(bank_ids).astype(str)
            if bank_ids is not None
            else np.asarray([f"cluster-bank:{value}" for value in clusters])
        )
    )
    if clusters is not None and (clusters.shape != y.shape or banks.shape != y.shape):
        raise ValueError("risk canonical identifiers must align with observations")
    seeds = None if clusters is None else _risk_seed_array(seed_ids, y.size)
    if unit_ids is not None:
        audited = [y, p, error, inside]
        if clusters is not None:
            audited.extend((clusters, banks, seeds))
        indices = _unique_risk_unit_indices(unit_ids, *audited)
        y, p, error, inside = (values[indices] for values in (y, p, error, inside))
        if clusters is not None:
            clusters, banks, seeds = clusters[indices], banks[indices], seeds[indices]
    if not np.any(inside):
        raise RuntimeError("risk calibration has empty common support")
    if clusters is not None:
        _require_complete_seed_bank_grid(
            seeds,
            clusters,
            banks,
            eligible=inside,
            minimum_per_cell=2,
            context="risk common support",
        )
    if clusters is None:
        probability_metrics = {
            "ece": expected_calibration_error(y[inside], p[inside]),
            "brier": brier_score(y[inside], p[inside]),
            "nll": binary_nll(y[inside], p[inside]),
            "support_coverage": float(np.mean(inside)),
        }
    else:
        probability_metrics = _bank_cluster_probability_metrics(
            y[inside], p[inside], clusters[inside], banks[inside], seeds[inside]
        )
        probability_metrics["support_coverage"] = _bank_cluster_mean(
            inside.astype(np.float64), clusters, banks, seeds
        )
    supported_error = error[inside]
    supported_probability = p[inside]
    ranking = (
        _bank_macro_risk_coverage(
            supported_error,
            supported_probability,
            clusters[inside],
            banks[inside],
            seed_ids=seeds[inside],
        )
        if clusters is not None
        else risk_coverage(supported_error, supported_probability)
    )
    probability_metrics.update(ranking)
    if clusters is None:
        probability_metrics["risk_error_spearman"] = spearman_correlation(
            supported_probability, supported_error
        )
    else:
        spearman_rows = []
        supported_clusters = clusters[inside]
        supported_banks = banks[inside]
        supported_seeds = seeds[inside]
        for seed_id in np.unique(supported_seeds):
            for cluster in np.unique(supported_clusters[supported_seeds == seed_id]):
                for bank in np.unique(
                    supported_banks[
                        (supported_seeds == seed_id)
                        & (supported_clusters == cluster)
                    ]
                ):
                    selected = (
                        (supported_seeds == seed_id)
                        & (supported_clusters == cluster)
                        & (supported_banks == bank)
                    )
                    spearman_rows.append(
                        {
                            "seed": str(seed_id),
                            "cluster": str(cluster),
                            "bank": str(bank),
                            "spearman": spearman_correlation(
                                supported_probability[selected], supported_error[selected]
                            ),
                        }
                    )
        probability_metrics["risk_error_spearman"] = _hierarchical_cell_macro(
            spearman_rows, ("spearman",)
        )["spearman"]
    return probability_metrics


def _bank_macro_risk_coverage(
    errors,
    risk_scores,
    cluster_ids,
    bank_ids=None,
    unit_ids=None,
    seed_ids=None,
):
    error = np.asarray(errors, dtype=np.float64)
    risk = np.asarray(risk_scores, dtype=np.float64)
    clusters = np.asarray(cluster_ids).astype(str)
    banks = (
        np.asarray(bank_ids).astype(str)
        if bank_ids is not None
        else np.asarray([f"cluster-bank:{value}" for value in clusters])
    )
    units = None if unit_ids is None else np.asarray(unit_ids).astype(str)
    seeds = _risk_seed_array(seed_ids, error.size)
    if (
        error.shape != risk.shape
        or error.shape != clusters.shape
        or error.shape != banks.shape
        or error.shape != seeds.shape
        or (units is not None and error.shape != units.shape)
    ):
        raise ValueError("bank-macro risk inputs must be aligned")
    if units is not None:
        indices = _unique_risk_unit_indices(
            units, error, risk, clusters, banks, seeds
        )
        error, risk, clusters, banks, seeds = (
            values[indices] for values in (error, risk, clusters, banks, seeds)
        )
    _require_complete_seed_bank_grid(
        seeds,
        clusters,
        banks,
        minimum_per_cell=2,
        context="risk-coverage ranking",
    )
    flat_fields = ["aurc"] + [
        f"{coverage}_{key}"
        for coverage in ("0.9", "0.75", "0.5")
        for key in ("median_error", "p90_error", "effective_coverage")
    ]
    cell_rows = []
    for seed_id in np.unique(seeds):
        for cluster in np.unique(clusters[seeds == seed_id]):
            for bank in np.unique(banks[(seeds == seed_id) & (clusters == cluster)]):
                selected = (
                    (seeds == seed_id) & (clusters == cluster) & (banks == bank)
                )
                if int(np.sum(selected)) < 2:
                    raise RuntimeError(
                        f"risk seed/bank {seed_id!r}/{bank!r} has fewer than two observations"
                    )
                curve = risk_coverage(error[selected], risk[selected])
                row = {
                    "seed": str(seed_id),
                    "cluster": str(cluster),
                    "bank": str(bank),
                    "aurc": float(curve["aurc"]),
                }
                for coverage in ("0.9", "0.75", "0.5"):
                    for key in ("median_error", "p90_error", "effective_coverage"):
                        row[f"{coverage}_{key}"] = float(
                            curve["retained"][coverage][key]
                        )
                cell_rows.append(row)
    macro = _hierarchical_cell_macro(cell_rows, flat_fields)
    return {
        "aurc": macro["aurc"],
        "retained": {
            coverage: {
                key: macro[f"{coverage}_{key}"]
                for key in ("median_error", "p90_error", "effective_coverage")
            }
            for coverage in ("0.9", "0.75", "0.5")
        },
        "algorithm_seed_count": int(len(np.unique(seeds))),
        "base_map_cluster_count": int(len(np.unique(clusters))),
        "canonical_bank_count": int(
            len(set(zip(clusters.tolist(), banks.tolist())))
        ),
        "algorithm_seed_bank_cell_count": int(len(cell_rows)),
        "aggregation": _RISK_HIERARCHY,
    }


def _cluster_mean_interval(
    values,
    clusters,
    resamples,
    seed,
    bank_ids=None,
    unit_ids=None,
    seed_ids=None,
):
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters).astype(str)
    banks = (
        np.asarray(bank_ids).astype(str)
        if bank_ids is not None
        else np.asarray([f"cluster-bank:{value}" for value in clusters])
    )
    seeds = _risk_seed_array(seed_ids, values.size)
    if not (values.shape == clusters.shape == banks.shape == seeds.shape):
        raise ValueError("cluster interval identifiers must align with values")
    if unit_ids is not None:
        indices = _unique_risk_unit_indices(
            unit_ids, values, clusters, banks, seeds
        )
        values, clusters, banks, seeds = (
            item[indices] for item in (values, clusters, banks, seeds)
        )
    cells = []
    for seed_id in np.unique(seeds):
        for cluster in np.unique(clusters[seeds == seed_id]):
            for bank in np.unique(banks[(seeds == seed_id) & (clusters == cluster)]):
                selected = (
                    (seeds == seed_id) & (clusters == cluster) & (banks == bank)
                )
                value = float(np.mean(values[selected]))
                cells.append(
                    {
                        "seed": str(seed_id),
                        "cluster": str(cluster),
                        "bank": str(bank),
                        "value": value,
                    }
                )
    family = _simultaneous_hierarchical_family(
        {"value": cells}, resamples, seed
    )
    interval = family["records"]["value"]
    available_seeds = np.unique(seeds)
    unique_clusters = np.unique(clusters)
    return {
        **interval,
        "algorithm_seed_count": int(available_seeds.size),
        "base_map_cluster_count": int(unique_clusters.size),
        "canonical_bank_count": int(
            len(set(zip(clusters.tolist(), banks.tolist())))
        ),
        "algorithm_seed_bank_cell_count": int(len(cells)),
        "bootstrap_hierarchy": (
            "canonical foundation, canonical bank within foundation, "
            "one synchronized global algorithm-seed draw"
        ),
        "aggregation": _RISK_HIERARCHY,
        "familywise_method": family["family"]["familywise_method"],
        "seed_draws_sha256": family["family"]["seed_draws_sha256"],
    }


def _expected_random_rejection_aurc(errors):
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("expected random-rejection AURC requires finite errors")
    # Every prefix of a uniformly random ordering has expected error mean(values).
    return float(np.mean(values) * (1.0 - 1.0 / values.size))


def _cluster_ranking_random_contrast(
    errors,
    first_scores,
    clusters,
    resamples,
    seed,
    null_threshold=0.0,
    bank_ids=None,
    unit_ids=None,
    seed_ids=None,
    _cells_only=False,
):
    error = np.asarray(errors, dtype=np.float64)
    first = np.asarray(first_scores, dtype=np.float64)
    cluster = np.asarray(clusters).astype(str)
    banks = (
        np.asarray(bank_ids).astype(str)
        if bank_ids is not None
        else np.asarray([f"cluster-bank:{value}" for value in cluster])
    )
    seeds = _risk_seed_array(seed_ids, error.size)
    if unit_ids is not None:
        indices = _unique_risk_unit_indices(
            unit_ids, error, first, cluster, banks, seeds
        )
        error, first, cluster, banks, seeds = (
            values[indices] for values in (error, first, cluster, banks, seeds)
        )
    _require_complete_seed_bank_grid(
        seeds,
        cluster,
        banks,
        minimum_per_cell=2,
        context="random-rejection AURC contrast",
    )
    cells = []
    for seed_id in np.unique(seeds):
        for value in np.unique(cluster[seeds == seed_id]):
            for bank in np.unique(banks[(seeds == seed_id) & (cluster == value)]):
                selected = (
                    (seeds == seed_id) & (cluster == value) & (banks == bank)
                )
                cells.append(
                    {
                        "seed": str(seed_id),
                        "cluster": str(value),
                        "bank": str(bank),
                        "improvement": float(
                            _expected_random_rejection_aurc(error[selected])
                            - risk_coverage(error[selected], first[selected])["aurc"]
                        ),
                    }
                )
    value_cells = [
        {
            "seed": row["seed"],
            "cluster": row["cluster"],
            "bank": row["bank"],
            "value": row["improvement"],
        }
        for row in cells
    ]
    if _cells_only:
        return value_cells
    interval = _cluster_mean_interval(
        np.asarray([row["value"] for row in value_cells]),
        np.asarray([row["cluster"] for row in value_cells]),
        resamples,
        seed,
        bank_ids=np.asarray([row["bank"] for row in value_cells]),
        seed_ids=np.asarray([row["seed"] for row in value_cells]),
    )
    interval["baseline"] = "exact expected AURC under uniform random ordering"
    interval["registered_margin"] = float(null_threshold)
    return interval


def _cluster_ranking_contrast(
    errors,
    first_scores,
    second_scores,
    clusters,
    resamples,
    seed,
    null_threshold=0.0,
    bank_ids=None,
    unit_ids=None,
    seed_ids=None,
    _cells_only=False,
):
    error = np.asarray(errors, dtype=np.float64)
    first = np.asarray(first_scores, dtype=np.float64)
    second = np.asarray(second_scores, dtype=np.float64)
    cluster = np.asarray(clusters).astype(str)
    banks = (
        np.asarray(bank_ids).astype(str)
        if bank_ids is not None
        else np.asarray([f"cluster-bank:{value}" for value in cluster])
    )
    seeds = _risk_seed_array(seed_ids, error.size)
    if unit_ids is not None:
        indices = _unique_risk_unit_indices(
            unit_ids, error, first, second, cluster, banks, seeds
        )
        error, first, second, cluster, banks, seeds = (
            values[indices]
            for values in (error, first, second, cluster, banks, seeds)
        )
    unique = np.unique(cluster)
    if unique.size < 2:
        raise RuntimeError("AURC contrast requires at least two base-map clusters")
    cells = []
    for seed_id in np.unique(seeds):
        for value in np.unique(cluster[seeds == seed_id]):
            for bank in np.unique(banks[(seeds == seed_id) & (cluster == value)]):
                selected = (
                    (seeds == seed_id) & (cluster == value) & (banks == bank)
                )
                if int(np.sum(selected)) < 2:
                    raise RuntimeError(
                        f"risk seed/bank {seed_id!r}/{bank!r} has fewer than two observations"
                    )
                first_aurc = risk_coverage(error[selected], first[selected])["aurc"]
                second_aurc = risk_coverage(error[selected], second[selected])["aurc"]
                cells.append(
                    {
                        "seed": str(seed_id),
                        "cluster": str(value),
                        "bank": str(bank),
                        "improvement": float(second_aurc - first_aurc),
                    }
                )
    value_cells = [
        {
            "seed": row["seed"],
            "cluster": row["cluster"],
            "bank": row["bank"],
            "value": row["improvement"],
        }
        for row in cells
    ]
    if _cells_only:
        return value_cells
    interval = _cluster_mean_interval(
        np.asarray([row["value"] for row in value_cells]),
        np.asarray([row["cluster"] for row in value_cells]),
        resamples,
        seed,
        bank_ids=np.asarray([row["bank"] for row in value_cells]),
        seed_ids=np.asarray([row["seed"] for row in value_cells]),
    )
    interval["registered_margin"] = float(null_threshold)
    return interval


def _cluster_value_contrast(
    first,
    second,
    clusters,
    resamples,
    seed,
    null_threshold=0.0,
    bank_ids=None,
    unit_ids=None,
    seed_ids=None,
    _cells_only=False,
):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    cluster = np.asarray(clusters).astype(str)
    banks = (
        np.asarray(bank_ids).astype(str)
        if bank_ids is not None
        else np.asarray([f"cluster-bank:{value}" for value in cluster])
    )
    seeds = _risk_seed_array(seed_ids, first.size)
    if unit_ids is not None:
        indices = _unique_risk_unit_indices(
            unit_ids, first, second, cluster, banks, seeds
        )
        first, second, cluster, banks, seeds = (
            values[indices] for values in (first, second, cluster, banks, seeds)
        )
    cells = []
    difference = first - second
    for seed_id in np.unique(seeds):
        for value in np.unique(cluster[seeds == seed_id]):
            for bank in np.unique(banks[(seeds == seed_id) & (cluster == value)]):
                selected = (
                    (seeds == seed_id) & (cluster == value) & (banks == bank)
                )
                cells.append(
                    {
                        "seed": str(seed_id),
                        "cluster": str(value),
                        "bank": str(bank),
                        "difference": float(np.mean(difference[selected])),
                    }
                )
    value_cells = [
        {
            "seed": row["seed"],
            "cluster": row["cluster"],
            "bank": row["bank"],
            "value": row["difference"],
        }
        for row in cells
    ]
    if _cells_only:
        return value_cells
    interval = _cluster_mean_interval(
        np.asarray([row["value"] for row in value_cells]),
        np.asarray([row["cluster"] for row in value_cells]),
        resamples,
        seed,
        bank_ids=np.asarray([row["bank"] for row in value_cells]),
        seed_ids=np.asarray([row["seed"] for row in value_cells]),
    )
    interval["registered_margin"] = float(null_threshold)
    return interval


def _cluster_monotonic_margin(
    errors,
    scores,
    clusters,
    resamples,
    seed,
    null_threshold=0.0,
    bank_ids=None,
    unit_ids=None,
    seed_ids=None,
    _cells_only=False,
):
    error = np.asarray(errors, dtype=np.float64)
    risk = np.asarray(scores, dtype=np.float64)
    cluster = np.asarray(clusters).astype(str)
    banks = (
        np.asarray(bank_ids).astype(str)
        if bank_ids is not None
        else np.asarray([f"cluster-bank:{value}" for value in cluster])
    )
    seeds = _risk_seed_array(seed_ids, error.size)
    if unit_ids is not None:
        indices = _unique_risk_unit_indices(
            unit_ids, error, risk, cluster, banks, seeds
        )
        error, risk, cluster, banks, seeds = (
            values[indices] for values in (error, risk, cluster, banks, seeds)
        )
    cells = []
    for seed_id in np.unique(seeds):
        for value in np.unique(cluster[seeds == seed_id]):
            for bank in np.unique(banks[(seeds == seed_id) & (cluster == value)]):
                selected = (
                    (seeds == seed_id) & (cluster == value) & (banks == bank)
                )
                if int(np.sum(selected)) < 2:
                    raise RuntimeError(
                        f"risk seed/bank {seed_id!r}/{bank!r} has fewer than two observations"
                    )
                curve = risk_coverage(error[selected], risk[selected])["retained"]
                medians = [
                    curve[key]["median_error"] for key in ("0.9", "0.75", "0.5")
                ]
                p90s = [
                    curve[key]["p90_error"] for key in ("0.9", "0.75", "0.5")
                ]
                cells.append(
                    {
                        "seed": str(seed_id),
                        "cluster": str(value),
                        "bank": str(bank),
                        "margin": float(
                            min(
                                medians[0] - medians[1],
                                medians[1] - medians[2],
                                p90s[0] - p90s[1],
                                p90s[1] - p90s[2],
                            )
                        ),
                    }
                )
    value_cells = [
        {
            "seed": row["seed"],
            "cluster": row["cluster"],
            "bank": row["bank"],
            "value": row["margin"],
        }
        for row in cells
    ]
    if _cells_only:
        return value_cells
    interval = _cluster_mean_interval(
        np.asarray([row["value"] for row in value_cells]),
        np.asarray([row["cluster"] for row in value_cells]),
        resamples,
        seed,
        bank_ids=np.asarray([row["bank"] for row in value_cells]),
        seed_ids=np.asarray([row["seed"] for row in value_cells]),
    )
    interval["registered_margin"] = float(null_threshold)
    return interval


def _unique_risk_unit_indices(unit_ids, *arrays):
    units = np.asarray(unit_ids).astype(str)
    values = [np.asarray(value) for value in arrays]
    if any(value.shape != units.shape for value in values):
        raise ValueError("risk unit IDs must align with all audited arrays")
    indices = []
    for unit in np.unique(units):
        matches = np.flatnonzero(units == unit)
        for value in values:
            if np.issubdtype(value.dtype, np.number):
                consistent = np.allclose(value[matches], value[matches[0]])
            else:
                consistent = np.all(value[matches] == value[matches[0]])
            if not consistent:
                raise RuntimeError(
                    f"copied risk unit {unit!r} has inconsistent audited values"
                )
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=np.int64)


def run_risk_contract(
    config: dict,
    dataset,
    output_root: str | Path,
) -> dict:
    """Replay risk features internally, then fit source-only calibrators and audit k=0."""
    from .formal_data_verification import require_verified_roles_from_root

    require_verified_roles_from_root(
        output_root,
        config,
        dataset,
        (
            "source_probe_train",
            "source_probe_selection",
            "source_method_selection",
            "source_calibration_fit",
            "source_calibration_selection",
            "target",
        ),
    )
    arm_inputs = build_first_party_risk_features(config, dataset, output_root)
    _validate_first_party_risk_features(arm_inputs, config, dataset, output_root)
    output_dir = Path(output_root) / "risk"
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    )
    replay_binding = _plain_risk_metadata(arm_inputs.replay_binding)
    write_json(output_dir / "replay_binding.json", replay_binding)
    blocking = getattr(arm_inputs, "blocking_strata", None)
    if blocking is not None:
        blocking = _plain_risk_metadata(blocking)
        gate = {
            "schema_version": "csi-pairs-v6-risk-gate-v2",
            "status": "FAIL",
            "passed": False,
            **evidence,
            "gate": "G6",
            "c9_subgates": {
                "1_frozen_candidates_k0_common_denominator": "FAIL",
                "2_calibration_metrics_and_reliability_ci": "FAIL",
                "3_common_support_and_full_oos_noninferiority": "FAIL",
                "4_aurc_baselines_and_coverage_monotonicity": "FAIL",
            },
            "feature_generation": "first-party checkpoint/data/proposal replay",
            "risk_score_source": "source-trained frozen unified compatibility probe",
            "native_energy_role": "diagnostic_only",
            "proposal_count_contract_verified": False,
            "blocking_reason": "registered risk audit mixture has an empty stratum",
            "stratum_coverage": blocking,
            "replay_binding_path": "replay_binding.json",
            "replay_binding_sha256": sha256_file(
                output_dir / "replay_binding.json"
            ),
        }
        write_json(output_dir / "stratum_coverage.json", blocking)
        write_json(output_dir / "gate.json", gate)
        write_json(
            output_dir / "manifest.json",
            {
                "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
                **evidence,
                "files": artifact_manifest(output_dir, evidence=evidence),
            },
        )
        return gate
    if set(arm_inputs) != {"endpoint", "alignment", "response", "full"}:
        raise RuntimeError("risk replay must contain exactly the four frozen arms")
    mixture_freeze = getattr(arm_inputs, "mixture_freeze", None)
    if (
        not isinstance(mixture_freeze, Mapping)
        or mixture_freeze.get("schema_version")
        != "csi-pairs-v6-risk-mixture-freeze-v1"
        or mixture_freeze.get("frozen_role") != "source_method_selection"
        or mixture_freeze.get("registered_proportions")
        != config["risk"]["audit_mixture"]
        or not mixture_freeze.get("bank_ids")
    ):
        raise RuntimeError("risk mixture was not frozen on source_method_selection")
    mixture_freeze = _plain_risk_metadata(mixture_freeze)
    write_json(output_dir / "mixture_freeze.json", mixture_freeze)
    reference = arm_inputs["endpoint"]
    for arm, data in arm_inputs.items():
        for field in (
            "fit_unit_id",
            "fit_canonical_unit_id",
            "fit_canonical_base_map_digest",
            "fit_canonical_bank_digest",
            "fit_seed",
            "selection_unit_id",
            "selection_canonical_unit_id",
            "selection_canonical_base_map_digest",
            "selection_canonical_bank_digest",
            "selection_seed",
            "target_unit_id",
            "target_canonical_unit_id",
            "target_base_map_cluster_id",
            "target_canonical_base_map_digest",
            "target_bank_id",
            "target_canonical_bank_digest",
            "target_city_id",
            "target_position_role",
            "target_seed",
            "q_target_unit_id",
            "q_target_canonical_unit_id",
            "q_target_canonical_base_map_digest",
            "q_target_canonical_bank_digest",
            "q_target_seed",
            "q_fit_unit_id",
            "q_fit_canonical_unit_id",
            "q_fit_canonical_base_map_digest",
            "q_fit_canonical_bank_digest",
            "q_fit_seed",
            "fit_proposal_count",
            "fit_proposal_fallback_count",
            "fit_proposal_set_sha256",
            "selection_proposal_count",
            "selection_proposal_fallback_count",
            "selection_proposal_set_sha256",
            "target_proposal_count",
            "target_proposal_fallback_count",
            "target_proposal_set_sha256",
        ):
            if not np.array_equal(np.asarray(data[field]), np.asarray(reference[field])):
                raise RuntimeError(f"four-arm common risk denominator mismatch: {arm}/{field}")
    rows = []
    q_rows = []
    proposal_rows = []
    native_diagnostic_rows = []
    inside_by_arm = {}
    component_predictions = {}
    for arm, data in arm_inputs.items():
        for split in ("fit", "selection", "target"):
            counts = np.asarray(data[f"{split}_proposal_count"], dtype=np.int64)
            fallback = np.asarray(
                data[f"{split}_proposal_fallback_count"], dtype=np.int64
            )
            digests = np.asarray(data[f"{split}_proposal_set_sha256"]).astype(str)
            units = np.asarray(data[f"{split}_unit_id"]).astype(str)
            if (
                not np.all(counts == int(config["risk"]["proposal_count"]))
                or np.any(fallback < 0)
                or any(
                    len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                    for value in digests
                )
            ):
                raise RuntimeError("first-party proposal replay violates its frozen contract")
            proposal_rows.extend(
                {
                    "arm": arm,
                    "split": split,
                    "unit_id": unit,
                    "proposal_count": int(count),
                    "fallback_count": int(fallback_count),
                    "proposal_set_sha256": digest,
                }
                for unit, count, fallback_count, digest in zip(
                    units, counts, fallback, digests
                )
            )
            native_diagnostic_rows.extend(
                {
                    "arm": arm,
                    "split": split,
                    "unit_id": unit,
                    "native_d_used": float(value),
                    "main_score_source": "frozen_unified_compatibility_probe",
                    "diagnostic_score_source": "arm_native_alignment_energy",
                }
                for unit, value in zip(units, data[f"{split}_native_d"])
            )
        fit_indices = _unique_risk_unit_indices(
            data["fit_canonical_unit_id"],
            data["fit_d"],
            data["fit_u"],
            data["fit_y"],
            data["fit_canonical_base_map_digest"],
            data["fit_canonical_bank_digest"],
            data["fit_seed"],
        )
        selection_indices = _unique_risk_unit_indices(
            data["selection_canonical_unit_id"],
            data["selection_d"],
            data["selection_u"],
            data["selection_y"],
            data["selection_canonical_base_map_digest"],
            data["selection_canonical_bank_digest"],
            data["selection_seed"],
        )
        q_fit_indices = _unique_risk_unit_indices(
            data["q_fit_canonical_unit_id"],
            data["q_fit_d"],
            data["q_fit_y"],
            data["q_fit_canonical_base_map_digest"],
            data["q_fit_canonical_bank_digest"],
            data["q_fit_seed"],
        )
        q_target_indices = _unique_risk_unit_indices(
            data["q_target_canonical_unit_id"],
            data["q_target_d"],
            data["q_target_y"],
            data["q_target_canonical_base_map_digest"],
            data["q_target_canonical_bank_digest"],
            data["q_target_seed"],
        )
        for split, indices in (
            ("fit", fit_indices),
            ("selection", selection_indices),
            ("q_fit", q_fit_indices),
            ("q_target", q_target_indices),
        ):
            _require_complete_seed_bank_grid(
                np.asarray(data[f"{split}_seed"])[indices],
                np.asarray(data[f"{split}_canonical_base_map_digest"])[indices],
                np.asarray(data[f"{split}_canonical_bank_digest"])[indices],
                context=f"{split} calibration replay",
            )
        fit_weights = _hierarchical_sample_weights(
            np.asarray(data["fit_seed"])[fit_indices],
            np.asarray(data["fit_canonical_base_map_digest"])[fit_indices],
            np.asarray(data["fit_canonical_bank_digest"])[fit_indices],
        )
        selection_weights = _hierarchical_sample_weights(
            np.asarray(data["selection_seed"])[selection_indices],
            np.asarray(data["selection_canonical_base_map_digest"])[selection_indices],
            np.asarray(data["selection_canonical_bank_digest"])[selection_indices],
        )
        q_fit_weights = _hierarchical_sample_weights(
            np.asarray(data["q_fit_seed"])[q_fit_indices],
            np.asarray(data["q_fit_canonical_base_map_digest"])[q_fit_indices],
            np.asarray(data["q_fit_canonical_bank_digest"])[q_fit_indices],
        )
        temperature = fit_temperature_no_intercept(
            np.asarray(data["q_fit_d"])[q_fit_indices],
            np.asarray(data["q_fit_y"])[q_fit_indices],
            config,
            sample_weights=q_fit_weights,
        )
        q_probability = temperature.predict(
            np.asarray(data["q_target_d"])[q_target_indices]
        )
        q_metrics = _bank_cluster_probability_metrics(
            np.asarray(data["q_target_y"])[q_target_indices],
            q_probability,
            np.asarray(data["q_target_canonical_base_map_digest"])[q_target_indices],
            np.asarray(data["q_target_canonical_bank_digest"])[q_target_indices],
            np.asarray(data["q_target_seed"])[q_target_indices],
        )
        q_rows.append(
            {
                "arm": arm,
                "route": "active_paired_only",
                "temperature": temperature.temperature,
                **q_metrics,
                "candidate_order_randomized": True,
                "probability_semantics": "first_candidate_is_generating_map",
            }
        )
        calibrator = fit_constrained_risk_calibrator(
            np.asarray(data["fit_d"])[fit_indices],
            np.asarray(data["fit_u"])[fit_indices],
            np.asarray(data["fit_y"])[fit_indices],
            (
                np.asarray(data["selection_d"])[selection_indices],
                np.asarray(data["selection_u"])[selection_indices],
                np.asarray(data["selection_y"])[selection_indices],
            ),
            config,
            fit_weights=fit_weights,
            selection_weights=selection_weights,
        )
        d_only = fit_scalar_risk_calibrator(
            np.asarray(data["fit_d"])[fit_indices],
            np.asarray(data["fit_y"])[fit_indices],
            (
                np.asarray(data["selection_d"])[selection_indices],
                np.asarray(data["selection_y"])[selection_indices],
            ),
            config,
            direction="nonpositive",
            fit_weights=fit_weights,
            selection_weights=selection_weights,
        )
        u_only = fit_scalar_risk_calibrator(
            np.asarray(data["fit_u"])[fit_indices],
            np.asarray(data["fit_y"])[fit_indices],
            (
                np.asarray(data["selection_u"])[selection_indices],
                np.asarray(data["selection_y"])[selection_indices],
            ),
            config,
            direction="nonnegative",
            fit_weights=fit_weights,
            selection_weights=selection_weights,
        )
        probabilities, inside = calibrator.predict(data["target_d"], data["target_u"])
        inside_by_arm[arm] = inside
        components = {
            "d_only": d_only.predict(data["target_d"]),
            "u_only": u_only.predict(data["target_u"]),
            "joint": probabilities,
        }
        if not np.allclose(probabilities, components["joint"]):
            raise RuntimeError("joint risk probability implementation mismatch")
        component_predictions[arm] = components
        support_indices = _unique_risk_unit_indices(
            data["target_canonical_unit_id"],
            inside,
            data["target_canonical_base_map_digest"],
            data["target_canonical_bank_digest"],
            data["target_seed"],
        )
        rows.append(
            {
                "arm": arm,
                "beta_d": calibrator.beta_d,
                "beta_u": calibrator.beta_u,
                "l2": calibrator.l2,
                "d_only_intercept": d_only.intercept,
                "d_only_beta": d_only.coefficient,
                "d_only_l2": d_only.l2,
                "u_only_intercept": u_only.intercept,
                "u_only_beta": u_only.coefficient,
                "u_only_l2": u_only.l2,
                "individual_support_coverage": _bank_cluster_mean(
                    inside[support_indices].astype(np.float64),
                    np.asarray(data["target_canonical_base_map_digest"])[support_indices],
                    np.asarray(data["target_canonical_bank_digest"])[support_indices],
                    np.asarray(data["target_seed"])[support_indices],
                ),
            }
        )
    common = np.logical_and.reduce([inside_by_arm[arm] for arm in sorted(inside_by_arm)])
    metrics = []
    ranking_checks = []
    ranking_hypotheses = {}
    ranking_bindings = []
    calibration_hypotheses = {}
    calibration_bindings = []
    resamples = int(config["risk"]["bootstrap_resamples"])
    alpha = float(config["evaluation"]["familywise_alpha"])
    for arm, data in arm_inputs.items():
        for model_name, probabilities in component_predictions[arm].items():
            result = risk_metrics(
                data["target_y"],
                probabilities,
                data["target_error"],
                common,
                data["target_canonical_base_map_digest"],
                data["target_canonical_bank_digest"],
                data["target_canonical_unit_id"],
                data["target_seed"],
            )
            metric_row = {"arm": arm, "risk_model": model_name, **result}
            if model_name == "joint":
                cells_by_metric = _calibration_metric_value_cells(
                    data["target_y"],
                    probabilities,
                    common,
                    data["target_canonical_base_map_digest"],
                    data["target_canonical_bank_digest"],
                    data["target_canonical_unit_id"],
                    data["target_seed"],
                )
                keys = {}
                for metric_name, cells in cells_by_metric.items():
                    key = f"{arm}|joint|{metric_name}"
                    calibration_hypotheses[key] = cells
                    keys[metric_name] = key
                calibration_bindings.append((metric_row, keys))
            else:
                metric_row.update(
                    _calibration_confidence_intervals(
                        data["target_y"],
                        probabilities,
                        common,
                        data["target_canonical_base_map_digest"],
                        resamples,
                        92000 + len(metrics),
                        data["target_canonical_bank_digest"],
                        data["target_canonical_unit_id"],
                        data["target_seed"],
                    )
                )
            metrics.append(metric_row)
        cities = sorted(set(np.asarray(data["target_city_id"]).astype(str).tolist()))
        for city_index, city in enumerate(cities):
            selected = (
                (np.asarray(data["target_city_id"]).astype(str) == city)
                & common
            )
            if not np.any(selected):
                raise RuntimeError(
                    f"risk common support is empty for arm={arm!r}, city={city!r}"
                )
            error = np.asarray(data["target_error"], dtype=np.float64)[selected]
            joint = component_predictions[arm]["joint"][selected]
            u_only = component_predictions[arm]["u_only"][selected]
            clusters = np.asarray(
                data["target_canonical_base_map_digest"]
            ).astype(str)[selected]
            banks = np.asarray(data["target_canonical_bank_digest"]).astype(str)[
                selected
            ]
            units = np.asarray(data["target_canonical_unit_id"]).astype(str)[selected]
            seeds = np.asarray(data["target_seed"]).astype(str)[selected]
            joint_metrics = _bank_macro_risk_coverage(
                error, joint, clusters, banks, units, seeds
            )
            u_metrics = _bank_macro_risk_coverage(
                error, u_only, clusters, banks, units, seeds
            )
            comparison_seed = 93000 + city_index + 101 * len(ranking_checks)
            joint_vs_u_cells = _cluster_ranking_contrast(
                error,
                joint,
                u_only,
                clusters,
                resamples,
                comparison_seed,
                bank_ids=banks,
                unit_ids=units,
                seed_ids=seeds,
                _cells_only=True,
            )
            joint_vs_random_cells = _cluster_ranking_random_contrast(
                error,
                joint,
                clusters,
                resamples,
                comparison_seed + 1,
                bank_ids=banks,
                unit_ids=units,
                seed_ids=seeds,
                _cells_only=True,
            )
            monotonic_cells = _cluster_monotonic_margin(
                error,
                joint,
                clusters,
                resamples,
                comparison_seed + 2,
                bank_ids=banks,
                unit_ids=units,
                seed_ids=seeds,
                _cells_only=True,
            )
            prefix = f"{arm}|{city}"
            keys = {
                "joint_vs_u_only": f"{prefix}|joint_vs_u_only",
                "joint_vs_random": f"{prefix}|joint_vs_random",
                "coverage_monotonic_margin": f"{prefix}|coverage_monotonic_margin",
            }
            ranking_hypotheses[keys["joint_vs_u_only"]] = joint_vs_u_cells
            ranking_hypotheses[keys["joint_vs_random"]] = joint_vs_random_cells
            ranking_hypotheses[keys["coverage_monotonic_margin"]] = monotonic_cells
            row = {
                "arm": arm,
                "city_id": city,
                "joint_aurc": joint_metrics["aurc"],
                "u_only_aurc": u_metrics["aurc"],
                "random_rejection_definition": (
                    "exact expected AURC under uniform random ordering within "
                    "each algorithm-seed/canonical-foundation/canonical-bank cell"
                ),
                "retained": joint_metrics["retained"],
            }
            ranking_checks.append(row)
            ranking_bindings.append((row, keys))
    ranking_family = _simultaneous_hierarchical_family(
        ranking_hypotheses, resamples, 93000, alpha=alpha
    )
    ranking_margin = float(config["risk"]["aurc_minimum_improvement"])
    monotonic_margin = -float(config["risk"]["coverage_monotonic_tolerance"])
    for row, keys in ranking_bindings:
        row["joint_vs_u_only"] = {
            **ranking_family["records"][keys["joint_vs_u_only"]],
            "registered_margin": ranking_margin,
        }
        row["joint_vs_random"] = {
            **ranking_family["records"][keys["joint_vs_random"]],
            "registered_margin": ranking_margin,
            "baseline": "exact expected AURC under uniform random ordering",
        }
        row["coverage_monotonic_margin"] = {
            **ranking_family["records"][keys["coverage_monotonic_margin"]],
            "registered_margin": monotonic_margin,
        }
        row["random_rejection_aurc"] = (
            row["joint_aurc"] + row["joint_vs_random"]["estimate"]
        )
        row["joint_beats_u_only"] = bool(
            row["joint_vs_u_only"]["familywise_ci95_low"] > ranking_margin
        )
        row["joint_beats_random"] = bool(
            row["joint_vs_random"]["familywise_ci95_low"] > ranking_margin
        )
        row["coverage_error_monotonic"] = bool(
            row["coverage_monotonic_margin"]["familywise_ci95_low"]
            >= monotonic_margin
        )
    calibration_family = _simultaneous_hierarchical_family(
        calibration_hypotheses, resamples, 92000, alpha=alpha
    )
    for row, keys in calibration_bindings:
        for metric_name, key in keys.items():
            record = calibration_family["records"][key]
            if not np.isclose(
                float(row[metric_name]), record["estimate"], rtol=0.0, atol=1e-12
            ):
                raise RuntimeError(
                    f"calibration family estimate mismatch for {key!r}"
                )
            row[f"{metric_name}_ci95_low"] = record["ci95_low"]
            row[f"{metric_name}_ci95_high"] = record["ci95_high"]
            row[f"{metric_name}_familywise_ci95_low"] = record[
                "familywise_ci95_low"
            ]
            row[f"{metric_name}_familywise_ci95_high"] = record[
                "familywise_ci95_high"
            ]
        row["calibration_ci_familywise_method"] = calibration_family["family"][
            "familywise_method"
        ]
        row["calibration_ci_seed_draws_sha256"] = calibration_family["family"][
            "seed_draws_sha256"
        ]
    write_csv(output_dir / "calibrators.csv", bind_rows(rows, evidence))
    write_csv(output_dir / "q_comp.csv", bind_rows(q_rows, evidence))
    write_csv(output_dir / "metrics.csv", bind_rows(metrics, evidence))
    write_csv(output_dir / "proposal_audit.csv", bind_rows(proposal_rows, evidence))
    write_csv(
        output_dir / "native_score_diagnostics.csv",
        bind_rows(native_diagnostic_rows, evidence),
    )
    write_json(
        output_dir / "ranking_checks.json",
        {
            "checks": ranking_checks,
            "familywise_inference": ranking_family["family"],
            **evidence,
        },
    )
    joint_metrics = [row for row in metrics if row["risk_model"] == "joint"]
    calibration_pass = all(
        row["ece_familywise_ci95_high"]
        <= float(config["risk"]["calibration_ece_max"])
        and row["brier_familywise_ci95_high"]
        <= float(config["risk"]["calibration_brier_max"])
        and row["nll_familywise_ci95_high"]
        <= float(config["risk"]["calibration_nll_max"])
        for row in joint_metrics
    )
    reference_clusters = np.asarray(
        arm_inputs["endpoint"]["target_canonical_base_map_digest"]
    ).astype(str)
    reference_seeds = np.asarray(
        arm_inputs["endpoint"]["target_seed"]
    ).astype(str)
    reference_banks = np.asarray(
        arm_inputs["endpoint"]["target_canonical_bank_digest"]
    ).astype(str)
    reference_units = np.asarray(
        arm_inputs["endpoint"]["target_canonical_unit_id"]
    ).astype(str)
    support_hypotheses = {
        "common_support": _cluster_value_contrast(
            common.astype(np.float64),
            np.zeros(common.size, dtype=np.float64),
            reference_clusters,
            resamples,
            94000,
            bank_ids=reference_banks,
            unit_ids=reference_units,
            seed_ids=reference_seeds,
            _cells_only=True,
        )
    }
    support_keys = {}
    for arm in ("alignment", "response"):
        key = f"outside_{arm}_minus_full"
        support_keys[arm] = key
        support_hypotheses[key] = _cluster_value_contrast(
            (~inside_by_arm[arm]).astype(np.float64),
            (~inside_by_arm["full"]).astype(np.float64),
            reference_clusters,
            resamples,
            94100,
            bank_ids=reference_banks,
            unit_ids=reference_units,
            seed_ids=reference_seeds,
            _cells_only=True,
        )
    support_family = _simultaneous_hierarchical_family(
        support_hypotheses, resamples, 94000, alpha=alpha
    )
    common_support_interval = {
        **support_family["records"]["common_support"],
        "registered_minimum": float(config["risk"]["common_support_minimum"]),
    }
    outside_intervals = {
        arm: _cluster_mean_interval(
            (~inside).astype(np.float64),
            reference_clusters,
            resamples,
            94010 + index,
            reference_banks,
            reference_units,
            seed_ids=reference_seeds,
        )
        for index, (arm, inside) in enumerate(sorted(inside_by_arm.items()))
    }
    outside = {arm: interval["estimate"] for arm, interval in outside_intervals.items()}
    support_margin = -float(config["risk"]["outside_support_noninferiority_max"])
    support_contrasts = {
        arm: {
            **support_family["records"][support_keys[arm]],
            "registered_margin": support_margin,
        }
        for arm in ("alignment", "response")
    }
    support_pass = bool(
        common_support_interval["familywise_ci95_low"]
        >= float(config["risk"]["common_support_minimum"])
        and all(
            value["familywise_ci95_low"] >= support_margin
            for value in support_contrasts.values()
        )
    )
    ranking_pass = all(
        row["joint_beats_u_only"]
        and row["joint_beats_random"]
        and row["coverage_error_monotonic"]
        for row in ranking_checks
    )
    candidate_pass = bool(
        set(arm_inputs) == {"endpoint", "alignment", "response", "full"}
        and all(np.all(np.asarray(data["target_budget"]) == 0) for data in arm_inputs.values())
        and all(
            row["beta_d"] < 0
            and row["beta_u"] > 0
            and row["d_only_beta"] < 0
            and row["u_only_beta"] > 0
            for row in rows
        )
    )
    c9_subgates = {
        "1_frozen_candidates_k0_common_denominator": "PASS" if candidate_pass else "FAIL",
        "2_calibration_metrics_and_reliability_ci": "PASS" if calibration_pass else "FAIL",
        "3_common_support_and_full_oos_noninferiority": "PASS" if support_pass else "FAIL",
        "4_aurc_baselines_and_coverage_monotonicity": "PASS" if ranking_pass else "FAIL",
    }
    passed = all(value == "PASS" for value in c9_subgates.values())
    gate = {
        "schema_version": "csi-pairs-v6-risk-gate-v2",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        **evidence,
        "gate": "G6",
        "common_support_coverage": common_support_interval["estimate"],
        "common_support_interval": common_support_interval,
        "outside_support_fraction": outside,
        "outside_support_intervals": outside_intervals,
        "outside_support_noninferiority": support_contrasts,
        "ranking_familywise_inference": ranking_family["family"],
        "calibration_familywise_inference": calibration_family["family"],
        "support_familywise_inference": support_family["family"],
        "c9_subgates": c9_subgates,
        "q_comp_scope": "active paired candidates only; no null/gray/wrong-city probability claim",
        "scope": "frozen k=0 paired-proposal audit only",
        "feature_generation": "first-party checkpoint/data/proposal replay",
        "risk_score_source": "source-trained frozen unified compatibility probe",
        "native_energy_role": "diagnostic_only",
        "proposal_count_contract_verified": True,
        "mixture_freeze_path": "mixture_freeze.json",
        "mixture_freeze_sha256": sha256_file(output_dir / "mixture_freeze.json"),
        "replay_binding_path": "replay_binding.json",
        "replay_binding_sha256": sha256_file(output_dir / "replay_binding.json"),
        "statistical_hierarchy": _RISK_HIERARCHY,
        "proposal_fallback_fraction": float(
            np.sum([row["fallback_count"] for row in proposal_rows])
            / max(np.sum([row["proposal_count"] for row in proposal_rows]), 1)
        ),
    }
    write_json(output_dir / "gate.json", gate)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
            **evidence,
            "files": artifact_manifest(output_dir, evidence=evidence),
        },
    )
    return gate


def run_risk_feature_adapter(manifest_path, config, dataset, output_root):
    from .formal_io import read_strict_json

    manifest = read_strict_json(manifest_path)
    required = {
        "schema_version",
        "command",
        "implementation_revision",
        "proposal_library_revision",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("risk-feature adapter manifest fields must be exact")
    if manifest["schema_version"] != "csi-pairs-v6-risk-feature-adapter-v1":
        raise ValueError("risk-feature adapter manifest schema mismatch")
    if not isinstance(manifest["command"], list) or not manifest["command"]:
        raise ValueError("risk-feature adapter command must be nonempty argv")
    for key in ("implementation_revision", "proposal_library_revision"):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ValueError(f"risk-feature adapter {key} must be nonempty")
    output_dir = Path(output_root) / "risk_feature_adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_sha256 = sha256_file(Path(__file__).resolve())
    if manifest["implementation_revision"] != source_sha256:
        raise RuntimeError(
            "risk features must bind the reviewed first-party formal_risk.py revision"
        )
    if manifest["proposal_library_revision"] != "csi-pairs-v6-map-proposal-v1":
        raise RuntimeError("risk proposal library revision is not frozen V6")
    write_json(
        output_dir / "replay_manifest.json",
        {
            "schema_version": "csi-pairs-v6-risk-first-party-replay-v1",
            "implementation_source_sha256": source_sha256,
            "proposal_library_revision": manifest["proposal_library_revision"],
            "external_command_ignored": True,
            "reason": "risk features are recomputed from authenticated checkpoints and data",
        },
    )
    return build_first_party_risk_features(config, dataset, output_root)


def _calibration_metric_value_cells(
    labels,
    probabilities,
    inside,
    cluster_ids,
    bank_ids=None,
    unit_ids=None,
    seed_ids=None,
):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    support = np.asarray(inside, dtype=np.bool_)
    clusters = np.asarray(cluster_ids).astype(str)
    banks = (
        np.asarray(bank_ids).astype(str)
        if bank_ids is not None
        else np.asarray([f"cluster-bank:{value}" for value in clusters])
    )
    seeds = _risk_seed_array(seed_ids, y.size)
    if not (
        y.shape == p.shape == support.shape == clusters.shape == banks.shape == seeds.shape
    ):
        raise ValueError("calibration interval inputs and canonical IDs must align")
    if unit_ids is not None:
        indices = _unique_risk_unit_indices(
            unit_ids, y, p, support, clusters, banks, seeds
        )
        y, p, support, clusters, banks, seeds = (
            values[indices] for values in (y, p, support, clusters, banks, seeds)
        )
    _require_complete_seed_bank_grid(
        seeds,
        clusters,
        banks,
        eligible=support,
        context="calibration common support",
    )
    supported_clusters = np.unique(clusters[support])
    if supported_clusters.size < 2:
        raise RuntimeError("calibration confidence interval requires two base-map clusters")
    cell_rows = []
    for seed_id in np.unique(seeds[support]):
        for cluster in np.unique(clusters[support & (seeds == seed_id)]):
            for bank in np.unique(
                banks[support & (seeds == seed_id) & (clusters == cluster)]
            ):
                selected = (
                    support
                    & (seeds == seed_id)
                    & (clusters == cluster)
                    & (banks == bank)
                )
                row = {
                    "seed": str(seed_id),
                    "cluster": str(cluster),
                    "bank": str(bank),
                    "ece": expected_calibration_error(y[selected], p[selected]),
                    "brier": brier_score(y[selected], p[selected]),
                    "nll": binary_nll(y[selected], p[selected]),
                }
                cell_rows.append(row)
    return {
        name: [
            {
                "seed": row["seed"],
                "cluster": row["cluster"],
                "bank": row["bank"],
                "value": row[name],
            }
            for row in cell_rows
        ]
        for name in ("ece", "brier", "nll")
    }


def _calibration_confidence_intervals(
    labels,
    probabilities,
    inside,
    cluster_ids,
    resamples,
    seed,
    bank_ids=None,
    unit_ids=None,
    seed_ids=None,
):
    hypotheses = _calibration_metric_value_cells(
        labels,
        probabilities,
        inside,
        cluster_ids,
        bank_ids,
        unit_ids,
        seed_ids,
    )
    family = _simultaneous_hierarchical_family(
        hypotheses, resamples, seed
    )
    intervals = {}
    for name, record in family["records"].items():
        intervals.update(
            {
                f"{name}_ci95_low": record["ci95_low"],
                f"{name}_ci95_high": record["ci95_high"],
                f"{name}_familywise_ci95_low": record["familywise_ci95_low"],
                f"{name}_familywise_ci95_high": record["familywise_ci95_high"],
            }
        )
    first_cells = next(iter(hypotheses.values()))
    seeds = {row["seed"] for row in first_cells}
    clusters = {row["cluster"] for row in first_cells}
    banks = {(row["cluster"], row["bank"]) for row in first_cells}
    intervals.update(
        {
            "calibration_ci_algorithm_seed_count": len(seeds),
            "calibration_ci_canonical_foundation_count": len(clusters),
            "calibration_ci_canonical_bank_count": len(banks),
            "calibration_ci_algorithm_seed_bank_cell_count": len(first_cells),
            "calibration_ci_bootstrap_hierarchy": (
                "canonical foundation, canonical bank within foundation, "
                "one synchronized global algorithm-seed draw"
            ),
            "calibration_ci_aggregation": _RISK_HIERARCHY,
            "calibration_ci_familywise_method": family["family"][
                "familywise_method"
            ],
            "calibration_ci_seed_draws_sha256": family["family"][
                "seed_draws_sha256"
            ],
        }
    )
    return intervals


def _coverage_error_monotonic(retained, tolerance):
    order = ("0.9", "0.75", "0.5")
    for metric in ("median_error", "p90_error"):
        values = [float(retained[coverage][metric]) for coverage in order]
        if any(values[index + 1] > values[index] + tolerance for index in range(2)):
            return False
    return True


def build_first_party_risk_features(config, dataset, output_root):
    """Recompute immutable risk features from the reviewed model and dataset code."""
    from .formal_evaluation import (
        _compatibility_dataset,
        _load_model,
        _masked_alignment_state_and_score,
        _normalized_scene_patches,
        _validate_checkpoint_index,
    )
    from .formal_factorial import (
        _natural_representations,
        _normalized_action,
        _normalized_map,
        _normalized_radio,
        _training_normalization,
    )
    from .formal_io import read_strict_json
    from .formal_localization import fit_source_position_head, predict_position_distribution
    from .formal_probes import fit_select_compatibility_probe, predict_binary_probe
    from .formal_protocol import patchify_csi, zero_typed_edit
    from .formal_routing import fit_route_normalization, route_dataset
    from .formal_teacher import load_teacher_bundle

    root = Path(output_root)
    from .formal_upstream import resolve_authenticated_upstream

    upstream = resolve_authenticated_upstream(config, dataset, root)
    qualification = read_strict_json(upstream.qualification_gate)
    if not upstream.migrated:
        qualification = require_manifested_formal_qualification(
            qualification,
            config,
            dataset,
            allow_nonscientific_fixture=True,
        )
    execution_device = resolve_execution_device(dataset)
    teacher = load_teacher_bundle(
        qualification["teacher_checkpoint"],
        config,
        device=execution_device,
    )
    route_normalization = fit_route_normalization(dataset, teacher)
    normalization = _training_normalization(
        dataset,
        dataset.indices_for_role("source_encoder_train"),
        route_normalization,
        teacher.patch_spec,
    )
    checkpoint_index = read_strict_json(upstream.checkpoint_index)
    checkpoint_rows = _validate_checkpoint_index(
        checkpoint_index, config, dataset, qualification
    )
    role_examples = {}
    routed_by_role = {}
    for role in (
        "source_method_selection",
        "source_calibration_fit",
        "source_calibration_selection",
        "target",
    ):
        scenes = dataset.indices_for_role(role)
        routed = route_dataset(
            dataset, teacher, config, scenes, normalization=route_normalization
        )
        routed_by_role[role] = routed
        try:
            role_examples[role] = _risk_audit_examples(
                dataset, routed, config, role
            )
        except RiskStrataUnavailable as error:
            blocking = {
                "role": error.role,
                "counts": error.counts,
                "required_proportions": config["risk"]["audit_mixture"],
            }
            return _make_first_party_risk_features(
                {},
                config,
                dataset,
                output_root,
                blocking_strata=blocking,
            )
    frozen_examples = role_examples["source_method_selection"]
    mixture_freeze = {
        "schema_version": "csi-pairs-v6-risk-mixture-freeze-v1",
        "frozen_role": "source_method_selection",
        "registered_proportions": config["risk"]["audit_mixture"],
        "sampling_seed": int(config["risk"]["proposal_seed"]),
        "sampling_rule": "per-canonical-bank deterministic hash order then equal bank macro",
        "counts": {
            condition: sum(row[4] == condition for row in frozen_examples)
            for condition in ("correct", "active", "gray", "null")
        },
        "bank_ids": sorted(
            {
                str(dataset.bank_ids[int(row[0])])
                for row in frozen_examples
            }
        ),
    }

    replayed = {}
    order_seed = int(config["risk"]["proposal_seed"]) + 701
    for checkpoint_row in checkpoint_rows:
        arm = str(checkpoint_row["arm"])
        seed = int(checkpoint_row["seed"])
        model = _load_model(
            upstream.factorial_root,
            checkpoint_row,
            qualification,
            config,
            dataset,
            device=execution_device,
        )
        compatibility_train = _compatibility_dataset(
            model,
            dataset,
            teacher,
            config,
            normalization,
            dataset.indices_for_role("source_probe_train"),
        )
        compatibility_selection = _compatibility_dataset(
            model,
            dataset,
            teacher,
            config,
            normalization,
            dataset.indices_for_role("source_probe_selection"),
        )
        compatibility_probe, compatibility_probe_record = (
            fit_select_compatibility_probe(
                compatibility_train["features"],
                compatibility_train["labels"],
                compatibility_selection["features"],
                compatibility_selection["labels"],
                config,
                seed=seed + 31001,
            )
        )
        source_scenes = [
            int(value) for value in dataset.indices_for_role("source_encoder_train")
        ]
        source_representations = _natural_representations(
            model, dataset, source_scenes, normalization, teacher.patch_spec
        )
        source_x = np.vstack(
            [source_representations[scene] for scene in source_scenes]
        )
        source_y = np.vstack([dataset.positions[scene] for scene in source_scenes])
        head = fit_source_position_head(source_x, source_y, config, seed=seed + 17003)
        arm_data = {}
        for prefix, role in (
            ("fit", "source_calibration_fit"),
            ("selection", "source_calibration_selection"),
            ("target", "target"),
        ):
            values = _replay_risk_examples(
                model,
                head,
                dataset,
                teacher,
                config,
                normalization,
                routed_by_role[role],
                role_examples[role],
                arm,
                seed,
                _masked_alignment_state_and_score,
                _normalized_scene_patches,
                _normalized_map,
                _normalized_radio,
                _normalized_action,
                zero_typed_edit,
                patchify_csi,
                predict_position_distribution,
                compatibility_probe,
                predict_binary_probe,
            )
            arm_data[f"{prefix}_d"] = values["d"]
            arm_data[f"{prefix}_u"] = values["u"]
            arm_data[f"{prefix}_y"] = values["y"]
            arm_data[f"{prefix}_unit_id"] = values["unit_id"]
            arm_data[f"{prefix}_canonical_unit_id"] = values[
                "canonical_unit_id"
            ]
            arm_data[f"{prefix}_canonical_base_map_digest"] = values[
                "canonical_cluster"
            ]
            arm_data[f"{prefix}_canonical_bank_digest"] = values[
                "canonical_bank"
            ]
            arm_data[f"{prefix}_seed"] = np.full(
                len(values["y"]), seed, dtype=np.int64
            )
            arm_data[f"{prefix}_native_d"] = values["native_d"]
            arm_data[f"{prefix}_proposal_count"] = values["proposal_count"]
            arm_data[f"{prefix}_proposal_fallback_count"] = values[
                "proposal_fallback_count"
            ]
            arm_data[f"{prefix}_proposal_set_sha256"] = values[
                "proposal_set_sha256"
            ]
            arm_data[f"{prefix}_role"] = np.asarray([role] * len(values["y"]))
            if prefix == "target":
                arm_data["target_error"] = values["error"]
                arm_data["target_budget"] = np.zeros(len(values["y"]), dtype=np.int64)
                arm_data["target_base_map_cluster_id"] = values["cluster"]
                arm_data["target_bank_id"] = values["bank"]
                arm_data["target_city_id"] = values["city"]
                arm_data["target_position_role"] = np.asarray(
                    ["query"] * len(values["y"])
                )
            active_indices = np.flatnonzero(values["condition"] == "active")
            if prefix in {"fit", "target"}:
                unique_active = _unique_risk_unit_indices(
                    values["canonical_unit_id"][active_indices],
                    values["paired_margin"][active_indices],
                    values["native_paired_margin"][active_indices],
                    values["canonical_cluster"][active_indices],
                    values["canonical_bank"][active_indices],
                )
                active_indices = active_indices[unique_active]
                labels = randomized_candidate_labels(
                    len(active_indices), order_seed + (0 if prefix == "fit" else 1)
                )
                margin = values["paired_margin"][active_indices]
                q_prefix = "q_fit" if prefix == "fit" else "q_target"
                arm_data[f"{q_prefix}_d"] = np.where(labels == 1, margin, -margin)
                arm_data[f"{q_prefix}_y"] = labels
                arm_data[f"{q_prefix}_unit_id"] = np.asarray(
                    [f"q:{value}" for value in values["unit_id"][active_indices]]
                )
                arm_data[f"{q_prefix}_canonical_unit_id"] = np.asarray(
                    [
                        f"q:{value}"
                        for value in values["canonical_unit_id"][active_indices]
                    ]
                )
                arm_data[f"{q_prefix}_canonical_base_map_digest"] = values[
                    "canonical_cluster"
                ][active_indices]
                arm_data[f"{q_prefix}_canonical_bank_digest"] = values[
                    "canonical_bank"
                ][active_indices]
                arm_data[f"{q_prefix}_seed"] = np.full(
                    len(labels), seed, dtype=np.int64
                )
                arm_data[f"{q_prefix}_role"] = np.asarray(
                    [role] * len(labels)
                )
                arm_data[f"{q_prefix}_native_d"] = np.where(
                    labels == 1,
                    values["native_paired_margin"][active_indices],
                    -values["native_paired_margin"][active_indices],
                )
        arm_data["compatibility_probe_family"] = np.asarray(
            [compatibility_probe_record["selected_family"]]
        )
        replayed.setdefault(arm, []).append(arm_data)
    payload = {
        arm: {
            field: np.concatenate([row[field] for row in rows], axis=0)
            for field in rows[0]
        }
        for arm, rows in replayed.items()
    }
    expected_arms = {"endpoint", "alignment", "response", "full"}
    if set(payload) != expected_arms:
        raise RuntimeError("first-party risk replay did not cover the four frozen arms")
    return _make_first_party_risk_features(
        payload,
        config,
        dataset,
        output_root,
        mixture_freeze=mixture_freeze,
    )


def _risk_audit_examples(dataset, routed, config, role, *, require_all_strata=True):
    from .formal_protocol import headline_alignment_edge

    groups = {name: [] for name in ("correct", "active", "gray", "null")}
    for scene_value in dataset.indices_for_role(role):
        scene = int(scene_value)
        positions = range(dataset.position_count)
        if role == "target":
            positions = np.flatnonzero(dataset.position_roles[scene] == "query").tolist()
        for world in range(dataset.world_count):
            for position in positions:
                groups["correct"].append((scene, world, world, int(position), "correct"))
        for edge in dataset.directed_edges(scene):
            if not headline_alignment_edge(dataset, scene, edge):
                continue
            for position in positions:
                key = (scene, edge.source_world, edge.target_world, int(position))
                condition = str(ROUTE_NAMES[routed.alignment_route[key]])
                if condition in groups:
                    groups[condition].append(
                        (
                            scene,
                            edge.source_world,
                            edge.target_world,
                            int(position),
                            condition,
                        )
                    )
    proportions = config["risk"]["audit_mixture"]
    if set(proportions) != set(groups) or not np.isclose(sum(proportions.values()), 1.0):
        raise RuntimeError("risk audit mixture is not a complete probability vector")
    if require_all_strata and any(not groups[name] for name in groups):
        raise RiskStrataUnavailable(
            role, {name: len(values) for name, values in groups.items()}
        )
    selected = []
    for scene_value in dataset.indices_for_role(role):
        scene = int(scene_value)
        scene_groups = {
            name: [row for row in values if int(row[0]) == scene]
            for name, values in groups.items()
        }
        if require_all_strata and any(not values for values in scene_groups.values()):
            raise RiskStrataUnavailable(
                role,
                {
                    f"{dataset.bank_ids[scene]}:{name}": len(values)
                    for name, values in scene_groups.items()
                },
            )
        if not require_all_strata:
            scene_groups = {name: values for name, values in scene_groups.items() if values}
        total = min(
            len(scene_groups[name]) / float(proportions[name])
            for name in scene_groups
        )
        for name in sorted(scene_groups):
            count = int(np.floor(total * float(proportions[name])))
            if require_all_strata and (count < 2 or count > len(scene_groups[name])):
                raise RiskStrataUnavailable(
                    role,
                    {
                        f"{dataset.bank_ids[scene]}:{condition}": len(values)
                        for condition, values in scene_groups.items()
                    },
                )
            ordered = sorted(
                scene_groups[name],
                key=lambda row: _risk_sampling_key(
                    dataset,
                    row,
                    int(config["risk"]["proposal_seed"]),
                ),
            )
            selected.extend(ordered[:max(1, count)])
    return sorted(selected, key=lambda row: (row[0], row[3], row[1], row[2], row[4]))


def _risk_sampling_key(dataset, row, seed):
    scene, observed_world, supplied_world, position, condition = row
    payload = "|".join(
        (
            str(seed),
            str(dataset.bank_ids[int(scene)]),
            "".join(str(int(value)) for value in dataset.world_bits[int(observed_world)]),
            "".join(str(int(value)) for value in dataset.world_bits[int(supplied_world)]),
            str(dataset.position_ids[int(scene), int(position)]),
            str(condition),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _replay_risk_examples(
    model,
    head,
    dataset,
    teacher,
    config,
    normalization,
    routed,
    examples,
    arm,
    seed,
    masked_score,
    normalized_patches,
    normalized_map,
    normalized_radio,
    normalized_action,
    zero_edit,
    patchify,
    predict_position,
    compatibility_probe,
    predict_compatibility,
):
    import torch
    from .formal_factorial import _canonical_bank_digest

    representations = []
    d_values = []
    native_d_values = []
    margins = []
    native_margins = []
    proposal_counts = []
    proposal_fallback_counts = []
    proposal_set_sha256s = []
    identifiers = []
    canonical_identifiers = []
    conditions = []
    clusters = []
    canonical_clusters = []
    banks = []
    canonical_banks = []
    cities = []
    zero = zero_edit(
        (1,),
        dataset.maps.shape[-1],
        int(dataset.metadata["assets"]["material_category_count"]),
    )
    zero_tensor = tensor_for_module(
        model,
        normalized_action(normalization, zero),
        dtype=torch.float32,
    )
    mask_bank = tuple(entry for entry in teacher.mask_bank if entry.mode == "random_75")
    bank_digests = {scene: _canonical_bank_digest(dataset, scene) for scene in {row[0] for row in examples}}
    for scene, observed_world, supplied_world, position, condition in examples:
        raw = patchify(dataset.csi[scene, observed_world, position], teacher.patch_spec)
        patches = (raw - normalization.patch_mean) / normalization.patch_scale
        map_value = normalized_map(normalization, dataset.maps[scene, supplied_world])
        radio_value = normalized_radio(
            normalization, dataset.radio_config[scene], dataset.bs_pose[scene]
        )
        with torch.no_grad():
            representation = model.retained_representation(
                tensor_for_module(model, patches[None], dtype=torch.float32),
                tensor_for_module(model, map_value[None], dtype=torch.float32),
                tensor_for_module(model, radio_value[None], dtype=torch.float32),
            )[0]
        representations.append(representation.detach().cpu().numpy())

        physical = normalized_patches(
            dataset,
            normalization,
            teacher.patch_spec,
            scene,
            observed_world,
            position,
        )
        latent = routed.teacher_latent[scene][observed_world, position]

        def scores(map_array):
            with torch.no_grad():
                feature, native_value = masked_score(
                    model,
                    tensor_for_module(model, patches[None], dtype=torch.float32),
                    tensor_for_module(
                        model,
                        normalized_map(normalization, map_array)[None],
                        dtype=torch.float32,
                    ),
                    tensor_for_module(model, radio_value[None], dtype=torch.float32),
                    zero_tensor,
                    mask_bank,
                    latent,
                    physical,
                    normalization,
                )
            unified_value = float(
                predict_compatibility(
                    compatibility_probe, np.asarray(feature)[None, :]
                )[0]
            )
            return unified_value, float(native_value)

        used_score, used_native_score = scores(dataset.maps[scene, supplied_world])
        proposal_maps, _, proposal_audit = frozen_map_proposals(
            dataset.maps[scene, supplied_world],
            dataset.radio_config[scene],
            dataset.map_channel_names,
            int(dataset.metadata["assets"]["material_category_count"]),
            config,
        )
        candidate_scores = [scores(value) for value in proposal_maps]
        candidate_unified = np.asarray([value[0] for value in candidate_scores])
        candidate_native = np.asarray([value[1] for value in candidate_scores])
        d_values.append(used_map_score_margin(used_score, candidate_unified))
        native_d_values.append(
            used_map_score_margin(used_native_score, candidate_native)
        )
        matched_score, matched_native_score = scores(
            dataset.maps[scene, observed_world]
        )
        margins.append(generating_map_pair_margin(matched_score, used_score))
        native_margins.append(
            generating_map_pair_margin(matched_native_score, used_native_score)
        )
        proposal_counts.append(int(proposal_audit["proposal_count"]))
        proposal_fallback_counts.append(int(proposal_audit["fallback_count"]))
        proposal_set_sha256s.append(str(proposal_audit["proposal_set_sha256"]))
        identifiers.append(
            f"risk:{seed}:{scene}:{observed_world}:{supplied_world}:{position}:{condition}"
        )
        conditions.append(condition)
        clusters.append(str(dataset.base_map_cluster_ids[scene]))
        canonical_clusters.append(dataset.canonical_base_map_digest(scene))
        banks.append(str(dataset.bank_ids[scene]))
        canonical_bank = bank_digests[scene]
        canonical_banks.append(canonical_bank)
        canonical_identifiers.append(
            "risk-canonical:"
            + ":".join(
                (
                    str(seed),
                    canonical_bank,
                    "".join(
                        str(int(value))
                        for value in dataset.world_bits[int(observed_world)]
                    ),
                    "".join(
                        str(int(value))
                        for value in dataset.world_bits[int(supplied_world)]
                    ),
                    str(dataset.position_ids[int(scene), int(position)]),
                    str(condition),
                )
            )
        )
        cities.append(str(dataset.city_ids[scene]))
    representation = np.asarray(representations, dtype=np.float64)
    prediction, _, uncertainty = predict_position(
        head,
        representation,
        sigma_min=float(config["localization"]["sigma_min"]),
    )
    truth = np.asarray(
        [dataset.positions[scene, position] for scene, _, _, position, _ in examples],
        dtype=np.float64,
    )
    error = np.linalg.norm(prediction - truth, axis=1)
    return {
        "d": np.asarray(d_values, dtype=np.float64),
        "native_d": np.asarray(native_d_values, dtype=np.float64),
        "u": np.asarray(uncertainty, dtype=np.float64),
        "y": (error > float(config["localization"]["failure_threshold_m"])).astype(np.int64),
        "error": error,
        "unit_id": np.asarray(identifiers),
        "canonical_unit_id": np.asarray(canonical_identifiers),
        "condition": np.asarray(conditions),
        "paired_margin": np.asarray(margins, dtype=np.float64),
        "native_paired_margin": np.asarray(native_margins, dtype=np.float64),
        "proposal_count": np.asarray(proposal_counts, dtype=np.int64),
        "proposal_fallback_count": np.asarray(
            proposal_fallback_counts, dtype=np.int64
        ),
        "proposal_set_sha256": np.asarray(proposal_set_sha256s),
        "cluster": np.asarray(clusters),
        "canonical_cluster": np.asarray(canonical_clusters),
        "bank": np.asarray(banks),
        "canonical_bank": np.asarray(canonical_banks),
        "city": np.asarray(cities),
    }


def load_risk_feature_archive(
    path: str | Path, config, dataset, output_root: str | Path
) -> dict[str, dict[str, np.ndarray]]:
    raise RuntimeError(
        "external risk feature archives are not admissible evidence; use first-party "
        "build_first_party_risk_features so proposals, model outputs, and labels are replayed"
    )
    # Kept below as a reader for forensic compatibility with legacy v3 artifacts.
    # It is deliberately unreachable from scientific execution.
    required_suffixes = {
        "fit_d",
        "fit_u",
        "fit_y",
        "selection_d",
        "selection_u",
        "selection_y",
        "target_d",
        "target_u",
        "target_y",
        "target_error",
        "q_fit_d",
        "q_fit_y",
        "q_target_d",
        "q_target_y",
        "fit_unit_id",
        "fit_role",
        "selection_unit_id",
        "selection_role",
        "target_unit_id",
        "target_role",
        "target_budget",
        "target_base_map_cluster_id",
        "target_city_id",
        "target_position_role",
        "q_fit_unit_id",
        "q_fit_role",
        "q_target_unit_id",
        "q_target_role",
    }
    arms = ("endpoint", "alignment", "response", "full")
    with np.load(Path(path), allow_pickle=False) as archive:
        expected = {f"{arm}__{suffix}" for arm in arms for suffix in required_suffixes}
        expected.update(
            {
                "schema_version",
                "dataset_sha256",
                "config_sha256",
                "candidate_order_seed",
                "q_comp_route_contract",
                "proposal_contract_json",
                "checkpoint_index_sha256",
                "evaluation_manifest_sha256",
                "mixture_contract_json",
            }
        )
        if set(archive.files) != expected:
            raise ValueError(
                f"risk feature archive fields must be exact; missing={sorted(expected-set(archive.files))}, "
                f"unexpected={sorted(set(archive.files)-expected)}"
            )
        if str(np.asarray(archive["schema_version"]).item()) != "csi-pairs-v6-risk-features-v3":
            raise ValueError("risk feature archive schema mismatch")
        evidence = evidence_context(config, dataset, "CANDIDATE_NOT_CLAIM")
        if str(np.asarray(archive["dataset_sha256"]).item()) != evidence["dataset_sha256"]:
            raise ValueError("risk feature dataset hash mismatch")
        if str(np.asarray(archive["config_sha256"]).item()) != evidence["config_sha256"]:
            raise ValueError("risk feature config hash mismatch")
        root = Path(output_root)
        from .formal_upstream import resolve_authenticated_upstream

        upstream = resolve_authenticated_upstream(config, dataset, root)
        bindings = (
            ("checkpoint_index_sha256", upstream.checkpoint_index),
            ("evaluation_manifest_sha256", root / "evaluation" / "manifest.json"),
        )
        for field, bound_path in bindings:
            if not bound_path.is_file() or str(np.asarray(archive[field]).item()) != sha256_file(bound_path):
                raise ValueError(f"risk feature {field} does not bind the executed model/probe")
        if str(np.asarray(archive["q_comp_route_contract"]).item()) != "active_paired_only":
            raise ValueError("q_comp may only be calibrated on active paired candidates")
        proposal_contract = parse_strict_json(str(np.asarray(archive["proposal_contract_json"]).item()))
        _validate_proposal_contract(proposal_contract)
        mixture_contract = parse_strict_json(str(np.asarray(archive["mixture_contract_json"]).item()))
        if mixture_contract != {
            "schema_version": "csi-pairs-v6-risk-mixture-v1",
            "frozen_role": "source_method_selection",
            "proportions": config["risk"]["audit_mixture"],
        }:
            raise ValueError("risk feature archive uses the wrong frozen audit mixture")
        order_seed = int(np.asarray(archive["candidate_order_seed"]).item())
        output = {
            arm: {suffix: np.asarray(archive[f"{arm}__{suffix}"]) for suffix in required_suffixes}
            for arm in arms
        }
        for arm in arms:
            data = output[arm]
            _validate_binary_vector(data["q_fit_y"], "q_fit_y")
            _validate_binary_vector(data["q_target_y"], "q_target_y")
            expected_fit = randomized_candidate_labels(
                len(data["q_fit_y"]), order_seed
            )
            expected_target = randomized_candidate_labels(
                len(data["q_target_y"]), order_seed + 1
            )
            if not np.array_equal(data["q_fit_y"].astype(np.int64), expected_fit):
                raise ValueError(f"{arm} q_comp fit candidate order does not match the registered seed")
            if not np.array_equal(data["q_target_y"].astype(np.int64), expected_target):
                raise ValueError(f"{arm} q_comp target candidate order does not match the registered seed")
            _validate_arm_shapes(arm, data)
            _validate_split_identity(arm, data, dataset)
        reference = output[arms[0]]
        for arm in arms[1:]:
            for field in ("target_unit_id", "q_target_unit_id"):
                if not np.array_equal(output[arm][field].astype(str), reference[field].astype(str)):
                    raise ValueError(f"four-arm common support mismatch in {field}")
        return output


def randomized_candidate_labels(count: int, seed: int) -> np.ndarray:
    if count < 2:
        raise ValueError("q_comp requires at least two active paired examples")
    labels = np.random.default_rng(int(seed)).integers(0, 2, size=int(count), dtype=np.int64)
    if np.all(labels == labels[0]):
        raise ValueError("registered q_comp randomization produced a degenerate candidate order")
    return labels


def _validate_binary_vector(values: np.ndarray, name: str) -> None:
    array = np.asarray(values)
    if array.ndim != 1 or not np.all(np.isin(array, (0, 1))):
        raise ValueError(f"{name} must be a one-dimensional binary vector")


def _validate_arm_shapes(arm: str, data: dict[str, np.ndarray]) -> None:
    groups = (
        ("fit_d", "fit_u", "fit_y"),
        ("selection_d", "selection_u", "selection_y"),
        ("target_d", "target_u", "target_y", "target_error"),
        ("q_fit_d", "q_fit_y"),
        ("q_target_d", "q_target_y"),
    )
    for group in groups:
        lengths = {len(np.asarray(data[name])) for name in group}
        if len(lengths) != 1 or next(iter(lengths)) < 2:
            raise ValueError(f"{arm} risk feature lengths disagree for {group}")
        for name in group:
            values = np.asarray(data[name])
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise ValueError(f"{arm} {name} must be a finite one-dimensional vector")


def _validate_split_identity(arm: str, data: dict[str, np.ndarray], dataset) -> None:
    contracts = (
        ("fit", "source_calibration_fit", "fit_y"),
        ("selection", "source_calibration_selection", "selection_y"),
        ("target", "target", "target_y"),
        ("q_fit", "source_calibration_fit", "q_fit_y"),
        ("q_target", "target", "q_target_y"),
    )
    for prefix, expected_role, length_field in contracts:
        identifiers = np.asarray(data[f"{prefix}_unit_id"]).astype(str)
        roles = np.asarray(data[f"{prefix}_role"]).astype(str)
        expected_length = len(np.asarray(data[length_field]))
        if identifiers.shape != (expected_length,) or roles.shape != (expected_length,):
            raise ValueError(f"{arm} {prefix} identity/role lengths do not match features")
        if np.any(identifiers == "") or len(set(identifiers.tolist())) != expected_length:
            raise ValueError(f"{arm} {prefix} unit IDs must be nonempty and unique")
        if set(roles.tolist()) != {expected_role}:
            raise ValueError(f"{arm} {prefix} uses the wrong permission role")
    budget = np.asarray(data["target_budget"])
    if budget.shape != np.asarray(data["target_y"]).shape or not np.all(budget == 0):
        raise ValueError(f"{arm} p_fail target audit must be strict k=0")
    target_length = len(np.asarray(data["target_y"]))
    for field in ("target_base_map_cluster_id", "target_city_id", "target_position_role"):
        values = np.asarray(data[field]).astype(str)
        if values.shape != (target_length,) or np.any(values == ""):
            raise ValueError(f"{arm} {field} must bind every target risk unit")
    if set(np.asarray(data["target_position_role"]).astype(str).tolist()) != {"query"}:
        raise ValueError(f"{arm} target support_pool leaked into risk denominator")
    allowed_clusters = set(
        str(value)
        for value in dataset.base_map_cluster_ids[dataset.indices_for_role("target")]
    )
    if not set(np.asarray(data["target_base_map_cluster_id"]).astype(str)).issubset(allowed_clusters):
        raise ValueError(f"{arm} risk archive has a non-target base-map cluster")


def _validate_proposal_contract(contract: object) -> None:
    expected = {
        "schema_version": "csi-pairs-v6-map-proposal-v1",
        "frozen_before_target": True,
        "k": 0,
        "input_allowlist": [
            "supplied_map",
            "public_radio_config",
            "registered_edit_library",
            "registered_seed",
        ],
        "forbidden_inputs": [
            "target_csi",
            "route",
            "effect_bin",
            "generating_side",
            "bank_id",
            "world_id",
            "receiver_position",
            "localization_result",
        ],
    }
    if contract != expected:
        raise ValueError("p_fail proposal contract is not the exact frozen map-only V6 contract")


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output
