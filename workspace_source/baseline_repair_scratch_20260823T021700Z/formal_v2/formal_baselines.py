from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def normalized_mse(target: np.ndarray, prediction: np.ndarray) -> float:
    numerator = float(np.sum((np.asarray(target) - np.asarray(prediction)) ** 2))
    denominator = float(np.sum(np.asarray(target) ** 2))
    if denominator <= 1e-15:
        return float("nan")
    return numerator / denominator


def _add_intercept(features: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(features.shape[0]), features])


@dataclass
class RidgeRegressor:
    alpha: float = 0.01
    weights: np.ndarray | None = None
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "RidgeRegressor":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        self.feature_mean = x.mean(axis=0)
        self.feature_scale = x.std(axis=0)
        self.feature_scale[self.feature_scale < 1e-9] = 1.0
        normalized = (x - self.feature_mean) / self.feature_scale
        design = _add_intercept(normalized)
        # ``alpha`` regularizes mean squared loss, so repeating identical rows
        # must not change the fitted diagnostic probe.
        penalty = np.eye(design.shape[1]) * float(self.alpha) * design.shape[0]
        penalty[0, 0] = 0.0
        self.weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None or self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("model has not been fitted")
        normalized = (np.asarray(features, dtype=np.float64) - self.feature_mean) / self.feature_scale
        return _add_intercept(normalized) @ self.weights

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        if self.weights is None or self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("model has not been fitted")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            weights=self.weights,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            alpha=np.asarray([self.alpha]),
            metadata=np.asarray([json.dumps(metadata or {}, sort_keys=True)]),
        )


@dataclass
class ZeroPreservingRidgeRegressor:
    """Linear response probe whose zero-action contrast is exactly zero."""

    alpha: float = 0.01
    weights: np.ndarray | None = None
    feature_scale: np.ndarray | None = None

    def fit(
        self,
        action_features: np.ndarray,
        zero_action_features: np.ndarray,
        targets: np.ndarray,
    ) -> "ZeroPreservingRidgeRegressor":
        action = np.asarray(action_features, dtype=np.float64)
        zero = np.asarray(zero_action_features, dtype=np.float64)
        target = np.asarray(targets, dtype=np.float64)
        if action.shape != zero.shape or action.ndim != 2:
            raise ValueError("action and zero-action features must have equal 2D shape")
        if target.ndim != 2 or target.shape[0] != action.shape[0]:
            raise ValueError("response targets must align with response features")
        contrast = action - zero
        self.feature_scale = np.sqrt(np.mean(contrast**2, axis=0))
        self.feature_scale[self.feature_scale < 1e-9] = 1.0
        design = contrast / self.feature_scale
        penalty = np.eye(design.shape[1]) * float(self.alpha) * design.shape[0]
        self.weights = np.linalg.solve(
            design.T @ design + penalty, design.T @ target
        )
        return self

    def predict_contrast(
        self,
        action_features: np.ndarray,
        zero_action_features: np.ndarray,
    ) -> np.ndarray:
        if self.weights is None or self.feature_scale is None:
            raise RuntimeError("model has not been fitted")
        action = np.asarray(action_features, dtype=np.float64)
        zero = np.asarray(zero_action_features, dtype=np.float64)
        if action.shape != zero.shape or action.ndim != 2:
            raise ValueError("action and zero-action features must have equal 2D shape")
        return ((action - zero) / self.feature_scale) @ self.weights

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        if self.weights is None or self.feature_scale is None:
            raise RuntimeError("model has not been fitted")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            weights=self.weights,
            feature_scale=self.feature_scale,
            alpha=np.asarray([self.alpha]),
            zero_action_rule=np.asarray(
                ["predict(action_features - zero_action_features); no intercept"]
            ),
            metadata=np.asarray([json.dumps(metadata or {}, sort_keys=True)]),
        )


class RidgeSufficientStatistics:
    """Bound-memory ridge fit with the same normalized objective as RidgeRegressor."""

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = float(alpha)
        self.count = 0
        self.feature_sum: np.ndarray | None = None
        self.feature_square_sum: np.ndarray | None = None
        self.feature_cross: np.ndarray | None = None
        self.feature_target_cross: np.ndarray | None = None
        self.target_sum: np.ndarray | None = None

    def update(self, features: np.ndarray, targets: np.ndarray) -> None:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("ridge statistic batches must be aligned 2D arrays")
        if x.shape[0] == 0:
            return
        if self.feature_sum is None:
            self.feature_sum = np.zeros(x.shape[1], dtype=np.float64)
            self.feature_square_sum = np.zeros(x.shape[1], dtype=np.float64)
            self.feature_cross = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
            self.feature_target_cross = np.zeros(
                (x.shape[1], y.shape[1]), dtype=np.float64
            )
            self.target_sum = np.zeros(y.shape[1], dtype=np.float64)
        if (
            x.shape[1] != self.feature_sum.shape[0]
            or y.shape[1] != self.target_sum.shape[0]
        ):
            raise ValueError("ridge statistic batch dimensions changed")
        self.count += int(x.shape[0])
        self.feature_sum += np.sum(x, axis=0)
        self.feature_square_sum += np.sum(x * x, axis=0)
        self.feature_cross += x.T @ x
        self.feature_target_cross += x.T @ y
        self.target_sum += np.sum(y, axis=0)

    def finalize(self) -> RidgeRegressor:
        if self.count <= 0 or self.feature_sum is None:
            raise RuntimeError("cannot fit ridge from empty statistics")
        count = float(self.count)
        mean = self.feature_sum / count
        variance = np.maximum(self.feature_square_sum / count - mean * mean, 0.0)
        scale = np.sqrt(variance)
        scale[scale < 1e-9] = 1.0
        centered_cross = self.feature_cross - count * np.outer(mean, mean)
        normalized_cross = centered_cross / np.outer(scale, scale)
        centered_target_cross = self.feature_target_cross - np.outer(
            mean, self.target_sum
        )
        normalized_target_cross = centered_target_cross / scale[:, None]
        normalized_sum = (self.feature_sum - count * mean) / scale
        design_cross = np.empty(
            (mean.size + 1, mean.size + 1), dtype=np.float64
        )
        design_cross[0, 0] = count
        design_cross[0, 1:] = normalized_sum
        design_cross[1:, 0] = normalized_sum
        design_cross[1:, 1:] = normalized_cross
        design_target_cross = np.vstack((self.target_sum, normalized_target_cross))
        penalty = np.eye(design_cross.shape[0]) * self.alpha * count
        penalty[0, 0] = 0.0
        model = RidgeRegressor(alpha=self.alpha)
        model.feature_mean = mean
        model.feature_scale = scale
        model.weights = np.linalg.solve(
            design_cross + penalty, design_target_cross
        )
        return model


class ZeroPreservingRidgeSufficientStatistics:
    """Bound-memory zero-preserving ridge fit over action contrasts."""

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = float(alpha)
        self.count = 0
        self.contrast_square_sum: np.ndarray | None = None
        self.contrast_cross: np.ndarray | None = None
        self.contrast_target_cross: np.ndarray | None = None

    def update(
        self,
        action_features: np.ndarray,
        zero_action_features: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        action = np.asarray(action_features, dtype=np.float64)
        zero = np.asarray(zero_action_features, dtype=np.float64)
        target = np.asarray(targets, dtype=np.float64)
        if (
            action.ndim != 2
            or action.shape != zero.shape
            or target.ndim != 2
            or target.shape[0] != action.shape[0]
        ):
            raise ValueError(
                "zero-preserving statistic batches must be aligned 2D arrays"
            )
        if action.shape[0] == 0:
            return
        contrast = action - zero
        if self.contrast_square_sum is None:
            self.contrast_square_sum = np.zeros(
                contrast.shape[1], dtype=np.float64
            )
            self.contrast_cross = np.zeros(
                (contrast.shape[1], contrast.shape[1]), dtype=np.float64
            )
            self.contrast_target_cross = np.zeros(
                (contrast.shape[1], target.shape[1]), dtype=np.float64
            )
        if (
            contrast.shape[1] != self.contrast_square_sum.shape[0]
            or target.shape[1] != self.contrast_target_cross.shape[1]
        ):
            raise ValueError("zero-preserving statistic batch dimensions changed")
        self.count += int(contrast.shape[0])
        self.contrast_square_sum += np.sum(contrast * contrast, axis=0)
        self.contrast_cross += contrast.T @ contrast
        self.contrast_target_cross += contrast.T @ target

    def finalize(self) -> ZeroPreservingRidgeRegressor:
        if self.count <= 0 or self.contrast_square_sum is None:
            raise RuntimeError("cannot fit zero-preserving ridge from empty statistics")
        count = float(self.count)
        scale = np.sqrt(self.contrast_square_sum / count)
        scale[scale < 1e-9] = 1.0
        normalized_cross = self.contrast_cross / np.outer(scale, scale)
        normalized_target_cross = self.contrast_target_cross / scale[:, None]
        penalty = np.eye(normalized_cross.shape[0]) * self.alpha * count
        model = ZeroPreservingRidgeRegressor(alpha=self.alpha)
        model.feature_scale = scale
        model.weights = np.linalg.solve(
            normalized_cross + penalty, normalized_target_cross
        )
        return model
