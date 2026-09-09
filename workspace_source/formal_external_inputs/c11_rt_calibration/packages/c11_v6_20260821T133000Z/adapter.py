from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


STATISTICS = ("path_loss", "delay_spread", "angular_spread", "visible_path_count")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _statistics(payload: dict, field: str) -> dict[str, float]:
    values = payload.get(field)
    if not isinstance(values, dict) or set(values) != set(STATISTICS):
        raise RuntimeError(f"payload field {field} does not contain the four frozen statistics")
    parsed = {name: float(values[name]) for name in STATISTICS}
    if not all(math.isfinite(value) for value in parsed.values()):
        raise RuntimeError(f"payload field {field} contains nonfinite values")
    return parsed


def _fit_affine(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    slope = 0.0 if denominator <= 1e-20 else sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)
    ) / denominator
    return float(slope), float(y_mean - slope * x_mean)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", required=True)
    parser.add_argument("--validation-inputs", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fit_path = Path(args.fit).resolve()
    validation_path = Path(args.validation_inputs).resolve()
    protocol_path = Path(args.protocol).resolve()
    fit = _read(fit_path)
    validation = _read(validation_path)
    protocol = _read(protocol_path)
    if fit.get("partition") != "fit" or validation.get("partition") != "validation":
        raise RuntimeError("calibration partition roles are invalid")
    if protocol.get("frozen_utc") != "2026-08-21T12:00:23Z":
        raise RuntimeError("calibration protocol is not the pre-registered version")

    fit_rows = []
    for unit in fit.get("units", []):
        payload = unit.get("payload", {})
        primary = _statistics(payload, "primary_statistics")
        independent = _statistics(payload, "independent_reference_statistics")
        fit_rows.append((primary, independent))
    if len(fit_rows) != 16:
        raise RuntimeError("calibration fit requires exactly 16 Denver scene units")
    parameters = {}
    for name in STATISTICS:
        slope, intercept = _fit_affine(
            [row[0][name] for row in fit_rows], [row[1][name] for row in fit_rows]
        )
        parameters[name] = {"slope": slope, "intercept": intercept}

    predictions = []
    for unit in validation.get("units", []):
        payload = unit.get("payload", {})
        if "independent_reference_statistics" in payload:
            raise RuntimeError("held-out validation payload leaks independent reference statistics")
        primary = _statistics(payload, "primary_statistics")
        predicted = {}
        for name in STATISTICS:
            value = parameters[name]["slope"] * primary[name] + parameters[name]["intercept"]
            if name != "path_loss":
                value = max(0.0, value)
            predicted[name] = int(round(value)) if name == "visible_path_count" else float(value)
        predictions.append((str(unit["unit_id"]), predicted))
    if len(predictions) != 16 or len({row[0] for row in predictions}) != 16:
        raise RuntimeError("calibration validation requires 16 unique Miami scene units")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    parameters_path = output / "fitted_parameters.json"
    simulated_path = output / "simulated_statistics.csv"
    result_path = output / "adapter_result.json"
    if any(path.exists() or path.is_symlink() for path in (parameters_path, simulated_path, result_path)):
        raise FileExistsError("refusing to overwrite calibration adapter output")
    parameters_payload = {
        "schema_version": "csi-pairs-c11-affine-calibration-v1",
        "fit_unit_count": 16,
        "fit_city": "external-denver",
        "validation_city": "external-miami",
        "parameters": parameters,
    }
    with parameters_path.open("x", encoding="utf-8") as handle:
        json.dump(parameters_payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    with simulated_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("unit_id", *STATISTICS))
        for unit_id, row in predictions:
            writer.writerow(
                (
                    unit_id,
                    format(row["path_loss"], ".17g"),
                    format(row["delay_spread"], ".17g"),
                    format(row["angular_spread"], ".17g"),
                    str(row["visible_path_count"]),
                )
            )
    result = {
        "schema_version": "csi-pairs-v6-rt-calibration-adapter-result-v5",
        "fit_dataset_sha256": _sha256(fit_path),
        "validation_inputs_sha256": _sha256(validation_path),
        "fitted_parameters_path": parameters_path.name,
        "fitted_parameters_sha256": _sha256(parameters_path),
        "simulated_statistics_path": simulated_path.name,
        "simulated_statistics_sha256": _sha256(simulated_path),
    }
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
