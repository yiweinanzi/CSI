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
        penalty = np.eye(design.shape[1]) * float(self.alpha)
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
