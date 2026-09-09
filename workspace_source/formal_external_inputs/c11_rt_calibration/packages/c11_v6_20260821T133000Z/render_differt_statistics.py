from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from formal_v2.external_adapters import differt_external_validity as adapter
from formal_v2.formal_dataset import FormalDataset


SCHEMA = "csi-pairs-c11-differt-scene-statistics-v1"
EXPECTED_DATASET_SHA256 = "060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scene_statistics(result) -> dict[str, float]:
    mask = np.asarray(result.mask, dtype=np.bool_)
    power_dbw = np.asarray(result.power, dtype=np.float64)
    delay = np.asarray(result.delay, dtype=np.float64)
    angle = np.deg2rad(np.asarray(result.aoa_az, dtype=np.float64))
    if not (mask.shape == power_dbw.shape == delay.shape == angle.shape) or mask.ndim != 3:
        raise RuntimeError("DiffeRT path-statistic tensors are not aligned")
    finite = mask & np.isfinite(power_dbw) & np.isfinite(delay) & np.isfinite(angle)
    power = np.where(finite, np.power(10.0, power_dbw / 10.0), 0.0)
    rows = []
    for receiver in range(mask.shape[1]):
        per_path_power = np.sum(power[:, receiver, :], axis=0)
        active = np.isfinite(per_path_power) & (per_path_power > 0.0)
        values = per_path_power[active]
        if values.size == 0:
            raise RuntimeError(f"DiffeRT receiver {receiver} has no finite powered path")
        selected_power = power[:, receiver, active]
        path_delay = np.sum(selected_power * delay[:, receiver, active], axis=0) / values
        phasors = np.sum(selected_power * np.exp(1j * angle[:, receiver, active]), axis=0)
        path_angle = np.angle(phasors)
        total = float(np.sum(values))
        mean_delay = float(np.sum(values * path_delay) / total)
        delay_spread_ns = math.sqrt(
            max(0.0, float(np.sum(values * np.square(path_delay - mean_delay)) / total))
        ) * 1e9
        resultant = np.sum(values * np.exp(1j * path_angle)) / total
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
    if values.shape != (mask.shape[1], 4) or not np.all(np.isfinite(values)):
        raise RuntimeError("DiffeRT receiver statistics are nonfinite")
    medians = np.median(values, axis=0)
    return {
        "path_loss": float(medians[0]),
        "delay_spread": float(medians[1]),
        "angular_spread": float(medians[2]),
        "visible_path_count": int(np.rint(medians[3])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--engine-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.is_file() or dataset_path.is_symlink() or _sha256(dataset_path) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("formal dataset is missing, linked, or hash-mismatched")
    dataset = FormalDataset.load(dataset_path)
    config_path = Path(args.engine_config).resolve()
    config = adapter.load_engine_config(config_path)
    manifest_path = Path(args.scene_manifest).resolve()
    manifest = adapter.load_scene_manifest(manifest_path, dataset, config_path)
    entries = {
        (str(row["scene_id"]), int(row["world"])): row for row in manifest["worlds"]
    }
    source_root = manifest_path.parent

    from differt.plugins import deepmimo

    original_export = deepmimo.export
    results = []
    for scene_value in dataset.indices_for_role("external_validation"):
        scene = int(scene_value)
        scene_id = str(dataset.scene_ids[scene])
        world = int(dataset.natural_world_index[scene])
        entry = entries[(scene_id, world)]
        asset_path = adapter._resolve_relative_source_asset(
            source_root, entry["source_asset_path"], entry["source_asset_sha256"]
        )
        center_frequency = float(dataset.radio_config[scene, 0])
        captured = []

        def capture_export(*positional, **keywords):
            result = original_export(*positional, **keywords)
            if math.isclose(float(keywords["frequency"]), center_frequency, rel_tol=0.0, abs_tol=0.5):
                captured.append(result)
            return result

        deepmimo.export = capture_export
        try:
            adapter._trace_source_asset(asset_path, config, dataset, scene, world)
        finally:
            deepmimo.export = original_export
        if len(captured) != 1:
            raise RuntimeError(f"DiffeRT carrier statistic capture failed for {scene_id}")
        results.append(
            {
                "scene_index": scene,
                "scene_id": scene_id,
                "city_id": str(dataset.city_ids[scene]),
                "base_map_cluster_id": str(dataset.base_map_cluster_ids[scene]),
                "natural_world_index": world,
                "source_asset_sha256": entry["source_asset_sha256"],
                "receiver_count": int(dataset.position_count),
                "statistics": _scene_statistics(captured[0]),
            }
        )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA,
        "simulation_not_measurement": True,
        "engine_name": adapter.ENGINE_NAME,
        "engine_revision": adapter.ENGINE_REVISION,
        "engine_license": adapter.ENGINE_LICENSE,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "scene_manifest_sha256": _sha256(manifest_path),
        "engine_config_sha256": _sha256(config_path),
        "adapter_source_sha256": _sha256(Path(adapter.__file__).resolve()),
        "scenes": results,
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
