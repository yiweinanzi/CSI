from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

from formal_v2.external_adapters.differt_external_validity import (
    ENGINE_FAMILY,
    ENGINE_LICENSE,
    ENGINE_NAME,
    ENGINE_REVISION,
    MATERIAL_NAMES,
    SCENE_MANIFEST_SCHEMA,
    SOURCE_ASSET_SCHEMA,
    STATE_MATERIALS,
    _horizontal_array_offsets,
    _physical_world_bits,
    _sionna_subcarrier_offsets_hz,
    _world_face_materials,
    load_engine_config,
)
from formal_v2.formal_io import read_strict_json, sha256_file, write_json


ALLOWED_DATASET_FIELDS = {
    "scene_ids",
    "scene_roles",
    "world_bits",
    "anchor_bits",
    "positions",
    "position_ids",
    "canonical_map_sha256",
    "radio_config",
    "bs_pose",
    "primitive_ids",
    "metadata_json",
}
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare CSI-PAIRS external worlds for the independent DiffeRT adapter"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--engine-config", required=True)
    parser.add_argument("--minimum-external-clusters", type=int, default=2)
    parser.add_argument("--output", required=True)
    try:
        prepare(parser.parse_args(argv))
        return 0
    except Exception as error:
        print(f"DiffeRT scene preparation error: {error}", file=sys.stderr)
        return 2


def prepare(args) -> None:
    from differt.geometry import Scene

    dataset_path = _regular_file(args.dataset, "formal dataset")
    asset_root = Path(args.asset_root).resolve()
    if asset_root.is_symlink() or not asset_root.is_dir():
        raise RuntimeError("Sionna/OSM asset root must be a regular directory")
    config_path = _regular_file(args.engine_config, "DiffeRT engine configuration")
    config = load_engine_config(config_path)
    output = Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite DiffeRT scene root: {output}")
    output.mkdir(parents=True)
    source_root = output / "source_assets"
    source_root.mkdir()

    dataset = _load_blind_dataset_view(dataset_path)
    generator_config_path = _regular_file(
        asset_root / "generator_config.json", "frozen Sionna generator configuration"
    )
    generator_config = read_strict_json(generator_config_path)
    intervention = generator_config.get("interventions", {})
    primitive_count = int(np.asarray(dataset["world_bits"]).shape[1])
    if (
        intervention.get("primitive_count") != primitive_count
        or intervention.get("state_materials") != list(STATE_MATERIALS)
    ):
        raise RuntimeError(
            "Sionna generator primitive/material contract differs from the V6 dataset"
        )
    generator_config_sha256 = sha256_file(generator_config_path)
    external = _external_scene_indices(
        dataset["scene_roles"], args.minimum_external_clusters
    )
    entries = []
    source_inputs = []
    for scene_value in external:
        scene = int(scene_value)
        scene_id = str(dataset["scene_ids"][scene])
        scene_xml = asset_root / "banks" / scene_id / "scene.xml"
        scene_xml = _regular_file(scene_xml, "DiffeRT source scene XML")
        loaded = Scene.load_xml(str(scene_xml))
        vertices = np.asarray(loaded.mesh.vertices, dtype=np.float32)
        triangles = np.asarray(loaded.mesh.triangles, dtype=np.int64)
        bounds = np.asarray(loaded.mesh.object_bounds, dtype=np.int64)
        if (
            bounds.shape != (2 + primitive_count, 2)
            or int(bounds[-1, 1]) != triangles.shape[0]
        ):
            raise RuntimeError(f"DiffeRT source object catalog is invalid: {scene_id}")
        radio = np.asarray(dataset["radio_config"][scene], dtype=np.float64)
        frequency, antennas, subcarriers, spacing = radio
        if int(antennas) < 1 or int(subcarriers) < 1:
            raise RuntimeError("DiffeRT adapter requires a positive channel layout")
        frequencies = frequency + _sionna_subcarrier_offsets_hz(
            int(subcarriers), float(spacing)
        )
        center = np.asarray(dataset["bs_pose"][scene, :3], dtype=np.float64)
        quaternion = np.asarray(dataset["bs_pose"][scene, 3:], dtype=np.float64)
        wavelength = float(config["speed_of_light_m_s"]) / float(frequency)
        local_offsets = _horizontal_array_offsets(int(antennas), wavelength)
        transmitters = np.asarray(
            center + _rotate_vectors(local_offsets, quaternion), dtype=np.float32
        )
        receivers = np.asarray(np.column_stack(
            (
                np.asarray(dataset["positions"][scene], dtype=np.float64),
                np.full(
                    len(dataset["positions"][scene]),
                    float(config["receiver_height_m"]),
                    dtype=np.float64,
                ),
            )
        ), dtype=np.float32)
        mapping = np.asarray(dataset["primitive_ids"][scene], dtype=np.int64)
        if (
            mapping.shape != (primitive_count,)
            or set(mapping.tolist()) != set(range(primitive_count))
        ):
            raise RuntimeError("DiffeRT primitive mapping is invalid")
        for world, bits in enumerate(np.asarray(dataset["world_bits"], dtype=np.int64)):
            anchor_bits = np.asarray(dataset["anchor_bits"][scene], dtype=np.int64)
            physical_bits = _physical_world_bits(bits, anchor_bits)
            face_materials = _world_face_materials(
                bounds, triangles.shape[0], mapping, physical_bits
            )
            metadata = {
                "schema_version": SOURCE_ASSET_SCHEMA,
                "dataset_sha256": sha256_file(dataset_path),
                "scene_id": scene_id,
                "world": world,
                "canonical_map_sha256": str(dataset["canonical_map_sha256"][scene, world]),
                "engine_family": ENGINE_FAMILY,
                "engine_revision": ENGINE_REVISION,
                "primary_csi_fields_read": False,
                "generator_config_sha256": generator_config_sha256,
                "primitive_count": primitive_count,
                "anchor_bits": anchor_bits.tolist(),
                "state_materials": list(STATE_MATERIALS),
                "antenna_count": int(antennas),
                "subcarrier_count": int(subcarriers),
            }
            asset = source_root / f"{scene_id}-world-{world}.npz"
            np.savez(
                asset,
                vertices=vertices,
                triangles=triangles,
                object_bounds=bounds,
                face_materials=face_materials,
                material_names=np.asarray(MATERIAL_NAMES, dtype="U32"),
                transmitters=transmitters,
                receivers=receivers,
                frequencies_hz=frequencies,
                metadata_json=np.asarray(
                    json.dumps(
                        metadata,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                ),
            )
            digest = sha256_file(asset)
            entries.append(
                {
                    "scene_id": scene_id,
                    "world": world,
                    "canonical_map_sha256": str(
                        dataset["canonical_map_sha256"][scene, world]
                    ),
                    "source_asset_id": f"differt:{scene_id}:world-{world}:{digest}",
                    "source_asset_path": asset.relative_to(output).as_posix(),
                    "source_asset_sha256": digest,
                }
            )
        source_inputs.append(
            {
                "scene_id": scene_id,
                "scene_xml_path": str(scene_xml),
                "scene_xml_sha256": sha256_file(scene_xml),
                "triangle_count": int(triangles.shape[0]),
            }
        )
    if len({row["source_asset_sha256"] for row in entries}) != len(entries):
        raise RuntimeError("DiffeRT sibling-world source assets are not byte-distinct")
    manifest = {
        "schema_version": SCENE_MANIFEST_SCHEMA,
        "dataset_sha256": sha256_file(dataset_path),
        "engine_family": ENGINE_FAMILY,
        "engine_name": ENGINE_NAME,
        "engine_revision": ENGINE_REVISION,
        "configuration_sha256": sha256_file(config_path),
        "license_id": ENGINE_LICENSE,
        "worlds": entries,
    }
    scene_manifest_path = output / "rt_scene_manifest.json"
    write_json(scene_manifest_path, manifest)
    frozen_config_path = output / "engine_config.json"
    shutil.copyfile(config_path, frozen_config_path)
    if sha256_file(frozen_config_path) != sha256_file(config_path):
        raise RuntimeError("frozen DiffeRT engine configuration copy is invalid")
    adapter_manifest_path = output / "differt_external_validity_adapter.json"
    adapter_manifest = _independent_adapter_manifest(
        frozen_config_path, scene_manifest_path
    )
    from formal_v2.formal_external_validity import _validate_manifest

    _validate_manifest(adapter_manifest)
    write_json(adapter_manifest_path, adapter_manifest)
    write_json(
        output / "preparation.json",
        {
            "schema_version": "csi-pairs-v6-differt-scene-preparation-v2",
            "status": "PASS",
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "engine_config_path": str(config_path),
            "engine_config_sha256": sha256_file(config_path),
            "generator_config_path": str(generator_config_path),
            "generator_config_sha256": generator_config_sha256,
            "frozen_engine_config_path": frozen_config_path.name,
            "frozen_engine_config_sha256": sha256_file(frozen_config_path),
            "adapter_manifest_path": adapter_manifest_path.name,
            "adapter_manifest_sha256": sha256_file(adapter_manifest_path),
            "preparer_source_sha256": sha256_file(Path(__file__).resolve()),
            "primary_csi_fields_read": False,
            "external_scene_count": len(external),
            "minimum_external_cluster_count": int(args.minimum_external_clusters),
            "world_count": len(entries),
            "source_inputs": source_inputs,
        },
    )


def _independent_adapter_manifest(
    engine_config_path: Path, scene_manifest_path: Path
) -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[2]
    adapter_source = (
        project_root / "formal_v2/external_adapters/differt_external_validity.py"
    ).resolve()
    if adapter_source.is_symlink() or not adapter_source.is_file():
        raise RuntimeError("DiffeRT external-validity adapter must be a regular file")
    return {
        "schema_version": "csi-pairs-v6-external-validity-independent-adapter-v1",
        "evidence_type": "independent_rt_engine",
        "engine_family": ENGINE_FAMILY,
        "source_revision": ENGINE_REVISION,
        "license_id": ENGINE_LICENSE,
        "adapter_source_path": adapter_source.relative_to(project_root).as_posix(),
        "adapter_source_sha256": sha256_file(adapter_source),
        "engine_config_path": Path(engine_config_path).name,
        "engine_config_sha256": sha256_file(engine_config_path),
        "rt_scene_manifest_path": Path(scene_manifest_path).name,
        "rt_scene_manifest_sha256": sha256_file(scene_manifest_path),
        "command": [
            "{project_root}/formal_v2/external_adapters/.runtime-differt/venv/bin/python",
            "{adapter_source}",
            "--dataset",
            "{dataset}",
            "--output",
            "{output}",
            "--scene-manifest",
            "{scene_manifest}",
            "--engine-config",
            "{engine_config}",
        ],
    }


def _load_blind_dataset_view(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = ALLOWED_DATASET_FIELDS - set(archive.files)
        if missing:
            raise RuntimeError(f"formal dataset omits DiffeRT preparation fields: {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in ALLOWED_DATASET_FIELDS}


def _external_scene_indices(
    scene_roles: np.ndarray, minimum_external_clusters: int
) -> np.ndarray:
    minimum = int(minimum_external_clusters)
    if minimum < 2:
        raise ValueError("DiffeRT preparation requires a minimum of at least two clusters")
    roles = np.asarray(scene_roles).astype(str)
    if roles.ndim != 1:
        raise RuntimeError("DiffeRT preparation scene roles must be one-dimensional")
    external = np.flatnonzero(roles == "external_validation")
    if len(external) < minimum:
        raise RuntimeError(
            "DiffeRT preparation requires at least "
            f"{minimum} external-validation banks; found {len(external)}"
        )
    return np.asarray(external, dtype=np.int64)


def _rotate_vectors(vectors: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("base-station quaternion must be four finite values")
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("base-station quaternion must have positive norm")
    w, x, y, z = q / norm
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    return np.asarray(vectors, dtype=np.float64) @ rotation.T


def _regular_file(value: str | Path, label: str) -> Path:
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular file")
    return path.resolve()


if __name__ == "__main__":
    raise SystemExit(main())
