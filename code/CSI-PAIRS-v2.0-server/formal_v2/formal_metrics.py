from __future__ import annotations

import numpy as np


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    raw_y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    _validate_aligned_vectors(raw_y, s, "AUROC")
    if not np.all(np.isin(raw_y, (0, 1))):
        raise ValueError("AUROC labels must be binary")
    y = raw_y.astype(np.int64, copy=False)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("AUROC requires aligned labels containing both classes")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    for value in np.unique(s):
        tied = np.flatnonzero(s == value)
        ranks[tied] = np.mean(ranks[tied])
    positive = y == 1
    n_positive = int(np.sum(positive))
    n_negative = int(np.sum(~positive))
    return float((np.sum(ranks[positive]) - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def binary_nll(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y, p = _validate_binary_probabilities(labels, probabilities, "binary NLL")
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y, p = _validate_binary_probabilities(labels, probabilities, "Brier score")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    y, p = _validate_binary_probabilities(labels, probabilities, "ECE")
    if type(bins) is not int or bins < 1:
        raise ValueError("ECE bins must be a positive integer")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = y.size
    value = 0.0
    for index in range(int(bins)):
        mask = (p >= edges[index]) & (p < edges[index + 1] if index + 1 < bins else p <= 1.0)
        if np.any(mask):
            value += float(np.sum(mask) / total) * abs(float(np.mean(p[mask]) - np.mean(y[mask])))
    return value


def risk_coverage(errors: np.ndarray, risk_scores: np.ndarray) -> dict:
    error = np.asarray(errors, dtype=np.float64)
    risk = np.asarray(risk_scores, dtype=np.float64)
    if error.shape != risk.shape or error.ndim != 1 or error.size < 2:
        raise ValueError("risk-coverage requires aligned one-dimensional arrays")
    if not np.all(np.isfinite(error)) or not np.all(np.isfinite(risk)):
        raise ValueError("risk-coverage requires finite errors and risk scores")

    # A risk threshold cannot distinguish observations with the same score.  Use
    # the expected cumulative error under a random ordering inside every tie
    # group, so neither AURC nor retained summaries depend on input row order.
    order = np.argsort(risk, kind="mergesort")
    sorted_risk = risk[order]
    observed_error = error[order]
    sorted_error = observed_error.copy()
    group_ends = []
    start = 0
    for end in np.flatnonzero(np.r_[sorted_risk[1:] != sorted_risk[:-1], True]) + 1:
        sorted_error[start:end] = float(np.mean(observed_error[start:end]))
        group_ends.append(int(end))
        start = int(end)
    coverages = np.arange(1, error.size + 1, dtype=np.float64) / error.size
    selective_risk = np.cumsum(sorted_error) / np.arange(1, error.size + 1)
    aurc = float(np.trapezoid(selective_risk, coverages))
    retained = {}
    for coverage in (0.9, 0.75, 0.5):
        requested = max(1, int(np.floor(coverage * error.size)))
        count = next(end for end in group_ends if end >= requested)
        values = observed_error[:count]
        retained[str(coverage)] = {
            "median_error": float(np.median(values)),
            "p90_error": float(np.percentile(values, 90)),
            "effective_coverage": float(count / error.size),
        }
    return {"aurc": aurc, "retained": retained}


def spearman_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    _validate_aligned_vectors(first_array, second_array, "Spearman correlation")
    x = _average_ranks(first_array)
    y = _average_ranks(second_array)
    if x.size < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    for value in np.unique(values):
        mask = values == value
        ranks[mask] = np.mean(ranks[mask])
    return ranks


def _validate_aligned_vectors(first: np.ndarray, second: np.ndarray, name: str) -> None:
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape or first.size == 0:
        raise ValueError(f"{name} requires nonempty aligned one-dimensional arrays")
    try:
        finite_first = np.isfinite(first)
    except TypeError as error:
        raise ValueError(f"{name} requires numeric inputs") from error
    if not np.all(finite_first) or not np.all(np.isfinite(second)):
        raise ValueError(f"{name} requires finite inputs")


def _validate_binary_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    raw_labels = np.asarray(labels)
    p = np.asarray(probabilities, dtype=np.float64)
    _validate_aligned_vectors(raw_labels, p, name)
    if not np.all(np.isin(raw_labels, (0, 1))):
        raise ValueError(f"{name} labels must be binary")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError(f"{name} probabilities must lie in [0, 1]")
    return raw_labels.astype(np.float64, copy=False), p
