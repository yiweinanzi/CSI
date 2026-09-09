from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np

from formal_v2.formal_dataset import FormalDataset
from formal_v2.formal_external_runtime import (
    collect_external_runtime,
    validate_external_runtime,
)
from formal_v2.formal_io import read_strict_json, sha256_file, write_json
from formal_v2.formal_protocol import PatchSpec
from formal_v2.formal_resources import validate_resource_registry


SCHEMA = "csi-pairs-v6-sionna-scene-manifest-v1"
SIONNA_REVISION = "04ddb9312116b408093b9d3ad363a3df355093a6"
SIONNA_RT_VERSION = "1.2.1"
SIONNA_ARCHIVES = {
    "sionna-main.zip": "fdbf89f307cc8933535af1587f00f1bcbd4b5edf7715275cd461bd4779f1fac7",
    "sionna-large-radio-maps-main.zip": "694ad17e7977e1c1adbdc8f93e6dcb1856e14cdf1b25f33da14c0e7aca80c33b",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sionna RT independent-engine G8 adapter")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-manifest", required=True)
    try:
        run(parser.parse_args(argv))
        return 0
    except Exception as error:
        print(f"Sionna external-validity error: {error}", file=sys.stderr)
        return 2


def run(args):
    _verify_sionna_resources()
    dataset = FormalDataset.load(args.dataset)
    manifest = load_scene_manifest(args.scene_manifest, dataset)
    _require_sionna_version(manifest["sionna_rt_version"])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("external_csi.npz", "runtime_provenance.json"):
        if (output / name).exists() or (output / name).is_symlink():
            raise FileExistsError(f"refusing to overwrite Sionna output: {name}")
    runtime_provenance = collect_external_runtime(
        "sionna", Path(__file__).resolve().parents[2]
    )
    validate_external_runtime(
        runtime_provenance,
        profile="sionna",
        executable=sys.executable,
        require_execution_ready=True,
    )
    scenes = dataset.indices_for_role("external_validation")
    cfr = _trace_all(dataset, manifest, scenes)
    external_csi = np.stack(
        [
            np.stack(
                [cfr[(int(scene), world)] for world in range(dataset.world_count)],
                axis=0,
            )
            for scene in scenes
        ],
        axis=0,
    )
    if not np.all(np.isfinite(external_csi)):
        raise RuntimeError("Sionna external-validity adapter produced nonfinite CSI")
    np.savez(
        output / "external_csi.npz",
        scene_ids=np.asarray(dataset.scene_ids[scenes], dtype=str),
        position_ids=np.asarray(dataset.position_ids[scenes], dtype=str),
        external_csi=np.asarray(external_csi, dtype=np.float64),
    )
    write_json(output / "runtime_provenance.json", runtime_provenance)


def load_scene_manifest(path: str | Path, dataset) -> dict:
    payload = read_strict_json(path)
    required = {
        "schema_version",
        "dataset_sha256",
        "engine_config_sha256",
        "sionna_revision",
        "sionna_rt_version",
        "license_id",
        "carrier_frequency_hz",
        "bandwidth_hz",
        "subcarrier_spacing_hz",
        "max_depth",
        "refraction",
        "receiver_z_m",
        "tx_array",
        "rx_array",
        "worlds",
    }
    if not isinstance(payload, dict) or set(payload) != required or payload["schema_version"] != SCHEMA:
        raise ValueError("Sionna scene manifest fields or schema are invalid")
    if payload["dataset_sha256"] != sha256_file(dataset.source_path):
        raise ValueError("Sionna scene manifest dataset hash mismatch")
    if payload["engine_config_sha256"] != dataset.metadata["engine"]["config_sha256"]:
        raise ValueError("Sionna scene manifest engine config hash mismatch")
    if payload["license_id"] != "Apache-2.0" or payload["sionna_revision"] != SIONNA_REVISION:
        raise ValueError("Sionna engine provenance is invalid")
    if payload["sionna_rt_version"] != SIONNA_RT_VERSION:
        raise ValueError("Sionna RT version is not frozen")
    spec = PatchSpec.from_metadata(dataset.metadata)
    for key in ("carrier_frequency_hz", "bandwidth_hz", "subcarrier_spacing_hz", "receiver_z_m"):
        if not isinstance(payload[key], (int, float)) or not np.isfinite(payload[key]) or payload[key] <= 0:
            raise ValueError(f"Sionna {key} must be positive")
    if round(payload["bandwidth_hz"] / payload["subcarrier_spacing_hz"]) != spec.subcarriers:
        raise ValueError("Sionna frequency grid differs from the formal CSI subcarrier grid")
    if not isinstance(payload["max_depth"], int) or payload["max_depth"] <= 0 or not isinstance(payload["refraction"], bool):
        raise ValueError("Sionna path-solver settings are invalid")
    array_fields = {"num_rows", "num_cols", "vertical_spacing", "horizontal_spacing", "pattern", "polarization"}
    for name in ("tx_array", "rx_array"):
        if not isinstance(payload[name], dict) or set(payload[name]) != array_fields:
            raise ValueError(f"Sionna {name} fields must be exact")
    antenna_product = (
        int(payload["tx_array"]["num_rows"])
        * int(payload["tx_array"]["num_cols"])
        * int(payload["rx_array"]["num_rows"])
        * int(payload["rx_array"]["num_cols"])
    )
    if antenna_product != spec.antennas:
        raise ValueError("Sionna Tx/Rx array product differs from the formal antenna axis")
    expected_worlds = {
        (str(dataset.scene_ids[int(scene)]), world)
        for scene in dataset.indices_for_role("external_validation")
        for world in range(dataset.world_count)
    }
    entries = payload["worlds"]
    entry_fields = {
        "scene_id", "world", "scene_xml", "scene_xml_sha256", "asset_manifest", "asset_manifest_sha256", "canonical_map_sha256"
    }
    if not isinstance(entries, list) or any(not isinstance(row, dict) or set(row) != entry_fields for row in entries):
        raise ValueError("Sionna world entries have invalid fields")
    observed_worlds = {
        (row["scene_id"], int(row["world"])) for row in entries
    }
    if len(entries) != len(expected_worlds) or observed_worlds != expected_worlds:
        raise ValueError("Sionna scene manifest does not cover every external-validation sibling world")
    for row in entries:
        scene = int(np.flatnonzero(dataset.scene_ids == row["scene_id"])[0])
        world = int(row["world"])
        if row["canonical_map_sha256"] != str(dataset.canonical_map_sha256[scene, world]):
            raise ValueError("Sionna world is not bound to its canonical map")
        scene_path = Path(row["scene_xml"]).resolve()
        assets_path = Path(row["asset_manifest"]).resolve()
        if sha256_file(scene_path) != row["scene_xml_sha256"] or sha256_file(assets_path) != row["asset_manifest_sha256"]:
            raise ValueError("Sionna scene or asset manifest hash mismatch")
        _validate_assets(read_strict_json(assets_path), scene_path.parent)
    return payload


def _validate_assets(payload, root):
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "assets"} or payload["schema_version"] != "csi-pairs-v6-sionna-assets-v1":
        raise ValueError("Sionna asset manifest schema mismatch")
    if not isinstance(payload["assets"], list) or not payload["assets"]:
        raise ValueError("Sionna asset manifest is empty")
    for row in payload["assets"]:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "license_id"}:
            raise ValueError("Sionna asset fields must be exact")
        path = (root / row["path"]).resolve()
        if root.resolve() not in path.parents or not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError("Sionna scene asset is missing, escapes its scene, or changed")
        if not str(row["license_id"]).strip():
            raise ValueError("Sionna scene asset license is missing")


def _verify_sionna_resources():
    project_root = Path(__file__).resolve().parents[2]
    rows = validate_resource_registry(
        read_strict_json(project_root / "formal_v2/configs/waibu_resources_v1.json"),
        project_root / "waibu",
    )
    actual = {
        row["file"]: row["actual_sha256"]
        for row in rows
        if row["file"] in SIONNA_ARCHIVES
    }
    if actual != SIONNA_ARCHIVES:
        raise RuntimeError("Sionna source archive authentication failed")


def _trace_all(dataset, manifest, scenes):
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene, subcarrier_frequencies

    entries = {(row["scene_id"], int(row["world"])): row for row in manifest["worlds"]}
    frequencies = subcarrier_frequencies(
        PatchSpec.from_metadata(dataset.metadata).subcarriers,
        float(manifest["subcarrier_spacing_hz"]),
    )
    solver = PathSolver()
    output = {}
    for scene_value in scenes:
        scene_index = int(scene_value)
        for world in range(dataset.world_count):
            entry = entries[(str(dataset.scene_ids[scene_index]), world)]
            scene = load_scene(entry["scene_xml"])
            scene.frequency = float(manifest["carrier_frequency_hz"])
            scene.bandwidth = float(manifest["bandwidth_hz"])
            scene.tx_array = PlanarArray(**manifest["tx_array"])
            scene.rx_array = PlanarArray(**manifest["rx_array"])
            scene.add(
                Transmitter(
                    "tx",
                    position=_point3(dataset.bs_pose[scene_index, :3]),
                    orientation=_quaternion_to_euler(dataset.bs_pose[scene_index, 3:]),
                )
            )
            for position in range(dataset.position_count):
                xy = dataset.positions[scene_index, position]
                scene.add(
                    Receiver(
                        f"rx-{position}",
                        position=_point3((xy[0], xy[1], manifest["receiver_z_m"])),
                    )
                )
            paths = solver(
                scene,
                max_depth=int(manifest["max_depth"]),
                refraction=bool(manifest["refraction"]),
            )
            values = _numpy_cfr(paths, frequencies, float(manifest["bandwidth_hz"]))
            output[(scene_index, world)] = _flatten_cfr(values, dataset.channel_count)
    return output


def _numpy_cfr(paths, frequencies, sampling_frequency):
    return np.asarray(
        paths.cfr(
            frequencies=frequencies,
            sampling_frequency=float(sampling_frequency),
            num_time_steps=1,
            out_type="numpy",
        )
    )


def _flatten_cfr(array, channel_count):
    values = np.asarray(array)
    if values.ndim != 6:
        raise RuntimeError(f"unexpected Sionna CFR rank: {values.shape}")
    # [rx, rx_ant, tx, tx_ant, time, subcarrier]
    values = values[:, :, 0, :, 0, :].reshape(values.shape[0], -1)
    joined = np.concatenate((values.real, values.imag), axis=1)
    if joined.shape[1] != channel_count:
        raise RuntimeError("Sionna CFR does not match the formal CSI channel axis")
    return joined


def _quaternion_to_euler(quaternion):
    # DATA_CONTRACT freezes bs_pose as scalar-first (w, x, y, z).
    w, x, y, z = (float(value) for value in quaternion)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [yaw, pitch, roll]


def _point3(values):
    point = [float(value) for value in values]
    if len(point) != 3 or not all(math.isfinite(value) for value in point):
        raise ValueError("Sionna device position must contain three finite coordinates")
    return point


def _require_sionna_version(expected):
    import importlib.metadata
    actual = importlib.metadata.version("sionna-rt")
    if actual != expected:
        raise RuntimeError(f"Sionna RT version mismatch: expected {expected}, got {actual}")


if __name__ == "__main__":
    raise SystemExit(main())
