#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from formal_v2.formal_metrics import (
    binary_auroc,
    binary_nll,
    brier_score,
    expected_calibration_error,
    risk_coverage,
    spearman_correlation,
)


def _pairwise_auroc(labels: list[int], scores: list[float]) -> float:
    positive = [score for label, score in zip(labels, scores) if label == 1]
    negative = [score for label, score in zip(labels, scores) if label == 0]
    wins = sum(
        1.0 if pos > neg else 0.5 if pos == neg else 0.0
        for pos in positive
        for neg in negative
    )
    return wins / (len(positive) * len(negative))


def _trapezoid(values: list[float], step: float) -> float:
    return sum(
        0.5 * (values[index] + values[index + 1]) * step
        for index in range(len(values) - 1)
    )


def _expect_rejection(function, *args, **kwargs) -> bool:
    try:
        function(*args, **kwargs)
    except ValueError:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite metric validation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = [0, 1, 0, 1]
    probabilities = [0.1, 0.9, 0.2, 0.8]
    y = np.asarray(labels)
    p = np.asarray(probabilities)
    expected = {
        "auroc": _pairwise_auroc(labels, probabilities),
        "nll": -sum(
            label * math.log(probability)
            + (1 - label) * math.log(1 - probability)
            for label, probability in zip(labels, probabilities)
        )
        / len(labels),
        "brier": sum(
            (probability - label) ** 2
            for label, probability in zip(labels, probabilities)
        )
        / len(labels),
        "ece_2_bins": 0.5 * abs(0.15 - 0.0) + 0.5 * abs(0.85 - 1.0),
        "spearman_reverse": -1.0,
    }
    observed = {
        "auroc": binary_auroc(y, p),
        "nll": binary_nll(y, p),
        "brier": brier_score(y, p),
        "ece_2_bins": expected_calibration_error(y, p, bins=2),
        "spearman_reverse": spearman_correlation(
            np.asarray([1.0, 2.0, 3.0]), np.asarray([3.0, 2.0, 1.0])
        ),
    }

    errors = np.asarray([1.0, 2.0, 4.0, 8.0])
    risk = np.asarray([0.1, 0.2, 0.3, 0.4])
    cumulative = [1.0, 1.5, 7.0 / 3.0, 3.75]
    expected_aurc = _trapezoid(cumulative, 0.25)
    observed_aurc = risk_coverage(errors, risk)["aurc"]
    tied_first = risk_coverage(errors, np.zeros(4))
    tied_second = risk_coverage(errors[::-1], np.zeros(4))

    tolerance = 1e-12
    checks = {
        name: math.isclose(observed[name], value, rel_tol=0.0, abs_tol=tolerance)
        for name, value in expected.items()
    }
    checks.update(
        {
            "aurc_registered_trapezoid": math.isclose(
                observed_aurc, expected_aurc, rel_tol=0.0, abs_tol=tolerance
            ),
            "aurc_tie_order_invariance": tied_first == tied_second,
            "reject_broadcast_shape": _expect_rejection(
                binary_nll, y[:, None], p
            ),
            "reject_nonfinite": _expect_rejection(
                binary_auroc, y, np.asarray([0.1, np.nan, 0.2, 0.8])
            ),
            "reject_invalid_probability": _expect_rejection(
                brier_score, y, np.asarray([0.1, 1.1, 0.2, 0.8])
            ),
        }
    )
    payload = {
        "schema_version": "csi-pairs-independent-metric-validation-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scientific_result": False,
        "expected": expected,
        "observed": observed,
        "aurc": {
            "convention": "trapezoid over registered coverages 1/n through 1",
            "expected": expected_aurc,
            "observed": observed_aurc,
        },
        "checks": checks,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
