from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from formal_v2 import sionna_osm_candidate as candidate


SCHEMA = "csi-pairs-c11-sionna-scene-statistics-v1"
EXPECTED_DATASET_SHA256 = "060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receiver_statistics(paths) -> dict[str, float]:
    valid = np.asarray(paths.valid, dtype=np.bool_)
    delay = np.asarray(paths.tau, dtype=np.float64)
    angle = np.asarray(paths.phi_r, dtype=np.float64)
    real = np.asarray(paths.a[0], dtype=np.float64)
    imag = np.asarray(paths.a[1], dtype=np.float64)
    power = np.sum(real * real + imag * imag, axis=(1, 2, 3))
    if valid.ndim != 3 or valid.shape[1] != 1:
        raise RuntimeError(f"unexpected Sionna validity shape: {valid.shape}")
    valid = valid[:, 0, :]
    delay = delay[:, 0, :]
    angle = angle[:, 0, :]
    if power.shape != valid.shape or delay.shape != valid.shape or angle.shape != valid.shape:
        raise RuntimeError("Sionna path-statistic tensors are not aligned")

    rows = []
    for receiver in range(valid.shape[0]):
        mask = (
            valid[receiver]
            & np.isfinite(power[receiver])
            & np.isfinite(delay[receiver])
            & np.isfinite(angle[receiver])
            & (power[receiver] > 0.0)
        )
        values = power[receiver, mask]
        delays = delay[receiver, mask]
        angles = angle[receiver, mask]
        if values.size == 0:
            raise RuntimeError(f"Sionna receiver {receiver} has no finite powered path")
        total = float(np.sum(values))
        mean_delay = float(np.sum(values * delays) / total)
        delay_spread_ns = math.sqrt(
            max(0.0, float(np.sum(values * np.square(delays - mean_delay)) / total))
        ) * 1e9
        resultant = np.sum(values * np.exp(1j * angles)) / total
        resultant_length = min(1.0, max(np.finfo(np.float64).tiny, float(abs(resultant))))
        angular_spread_deg = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(resultant_length))))
        visible = int(np.count_nonzero(values >= float(np.max(values)) * 1e-3))
        rows.append(
            (
                -10.0 * math.log10(total),
                delay_spread_ns,
                angular_spread_deg,
                visible,
            )
        )
    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (valid.shape[0], 4) or not np.all(np.isfinite(values)):
        raise RuntimeError("Sionna receiver statistics are nonfinite")
    medians = np.median(values, axis=0)
    return {
        "path_loss": float(medians[0]),
        "delay_spread": float(medians[1]),
        "angular_spread": float(medians[2]),
        "visible_path_count": int(np.rint(medians[3])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--scene-index", required=True, type=int)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    if not dataset.is_file() or dataset.is_symlink() or _sha256(dataset) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("formal dataset is missing, linked, or hash-mismatched")
    root, manifest, frozen_config = candidate.load_asset_manifest(args.asset_root)
    scene_index = int(args.scene_index)
    if not 0 <= scene_index < len(manifest["banks"]):
        raise ValueError("scene index is out of range")
    row = manifest["banks"][scene_index]
    if row["role"] != "external_validation" or row["city_id"] not in {
        "external-denver",
        "external-miami",
    }:
        raise RuntimeError("C11 statistics require a frozen external-validation scene")
    bank_path = root / str(row["bank_record_path"])
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(frozen_config)
    natural_bits = candidate._anchor_bits(scene_index, 4)
    config["world_bits"] = [natural_bits.tolist()]

    captured = []
    original_extract = candidate._extract_paths

    def capture(paths, *positional, **keywords):
        if not captured:
            captured.append(_receiver_statistics(paths))
        return original_extract(paths, *positional, **keywords)

    candidate._extract_paths = capture
    try:
        candidate.render_bank(bank, root, config, scene_index)
    finally:
        candidate._extract_paths = original_extract
    if len(captured) != 1:
        raise RuntimeError("Sionna normal-path statistic capture did not occur exactly once")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA,
        "simulation_not_measurement": True,
        "engine_name": "Sionna RT",
        "engine_version": "2.0.1",
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "asset_manifest_sha256": _sha256(root / "asset_manifest.json"),
        "bank_record_sha256": _sha256(bank_path),
        "scene_xml_sha256": _sha256(root / str(row["scene_xml_path"])),
        "generator_source_sha256": _sha256(Path(candidate.__file__).resolve()),
        "scene_index": scene_index,
        "scene_id": row["scene_id"],
        "city_id": row["city_id"],
        "base_map_cluster_id": row["base_map_cluster_id"],
        "natural_world_bits": natural_bits.tolist(),
        "receiver_count": int(frozen_config["positions_per_bank"]),
        "statistics": captured[0],
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
