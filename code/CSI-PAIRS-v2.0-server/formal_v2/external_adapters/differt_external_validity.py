from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import sys

import numpy as np

from formal_v2.formal_dataset import FormalDataset
from formal_v2.formal_io import read_strict_json, sha256_file, write_json


ENGINE_FAMILY = "differt"
ENGINE_NAME = "DiffeRT"
ENGINE_REVISION = "differt@673cc58ef61906b8ab0869dd206b3d032dbc01b2"
ENGINE_LICENSE = "MIT"
ENGINE_CONFIG_SCHEMA = "csi-pairs-v6-differt-engine-config-v1"
SCENE_MANIFEST_SCHEMA = "csi-pairs-v6-independent-rt-scene-manifest-v1"
SOURCE_ASSET_SCHEMA = "csi-pairs-v6-differt-source-asset-v3"
RUNTIME_SCHEMA = "csi-pairs-v6-differt-runtime-provenance-v1"
REQUIREMENTS_SHA256 = "97afdca61c657fa23f1eeae3f64bfb37b54a048411c9b25edd230293bdecc573"
ENGINE_PROVENANCE_SHA256 = "317ab9e495e6dee722a37bc471db77e2eedc9b3c6d699732f0d6c008bba0b2ed"
REQUIRED_DISTRIBUTIONS = {
    "differt": "0.10.0",
    "differt-core": "0.10.0",
    "jax": "0.11.0",
    "jaxlib": "0.11.0",
    "numpy": "2.5.2",
    "warp-lang": "1.16.0",
}
SOURCE_ASSET_ARRAYS = {
    "vertices",
    "triangles",
    "object_bounds",
    "face_materials",
    "material_names",
    "transmitters",
    "receivers",
    "frequencies_hz",
    "metadata_json",
}
MATERIAL_NAMES = (
    "itu_wet_ground",
    "itu_concrete",
    "itu_glass",
    "itu_wood",
    "itu_metal",
)
STATE_MATERIALS = ("concrete", "glass")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DiffeRT independent-engine G8 adapter")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    parser.add_argument("--scene-manifest")
    parser.add_argument("--engine-config", required=True)
    parser.add_argument("--probe-runtime", action="store_true")
    try:
        args = parser.parse_args(argv)
        if args.probe_runtime:
            print(
                json.dumps(
                    collect_runtime(args.engine_config),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
            return 0
        for name in ("dataset", "output", "scene_manifest"):
            if not getattr(args, name):
                parser.error(f"--{name.replace('_', '-')} is required")
        run(args)
        return 0
    except Exception as error:
        print(f"DiffeRT external-validity error: {error}", file=sys.stderr)
        return 2


def run(args) -> None:
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    config_path = _regular_file(args.engine_config, "DiffeRT engine configuration")
    config = load_engine_config(config_path)
    dataset = FormalDataset.load(args.dataset)
    manifest_path = _regular_file(args.scene_manifest, "DiffeRT scene manifest")
    manifest = load_scene_manifest(manifest_path, dataset, config_path)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "external_csi.npz"
    runtime_path = output / "runtime_provenance.json"
    for path in (result_path, runtime_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite DiffeRT output: {path.name}")

    scenes = np.asarray(dataset.indices_for_role("external_validation"), dtype=np.int64)
    entries = {
        (str(row["scene_id"]), int(row["world"])): row
        for row in manifest["worlds"]
    }
    source_root = manifest_path.parent
    external_csi = []
    for scene_value in scenes:
        scene = int(scene_value)
        worlds = []
        for world in range(dataset.world_count):
            entry = entries[(str(dataset.scene_ids[scene]), world)]
            asset_path = _resolve_relative_source_asset(
                source_root,
                entry["source_asset_path"],
                entry["source_asset_sha256"],
            )
            worlds.append(_trace_source_asset(asset_path, config, dataset, scene, world))
        external_csi.append(np.stack(worlds, axis=0))
    values = np.stack(external_csi, axis=0)
    expected_shape = (
        len(scenes),
        dataset.world_count,
        dataset.position_count,
        dataset.channel_count,
    )
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"DiffeRT external CSI shape or values are invalid: {values.shape}"
        )
    np.savez(
        result_path,
        scene_ids=np.asarray(dataset.scene_ids[scenes], dtype=str),
        position_ids=np.asarray(dataset.position_ids[scenes], dtype=str),
        external_csi=np.asarray(values, dtype=np.float64),
    )
    write_json(runtime_path, collect_runtime(config_path))


def load_engine_config(path: str | Path) -> dict:
    payload = read_strict_json(path)
    required = {
        "schema_version",
        "engine_family",
        "engine_name",
        "engine_revision",
        "license_id",
        "solver",
        "path_orders",
        "polarization",
        "receiver_height_m",
        "material_thickness_m",
        "array_model",
        "geometry_dtype",
        "index_dtype",
        "output_dtype",
        "jax_enable_x64",
        "jax_platforms",
        "speed_of_light_m_s",
        "free_space_impedance_ohm",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("DiffeRT engine configuration fields must be exact")
    if (
        payload["schema_version"] != ENGINE_CONFIG_SCHEMA
        or payload["engine_family"] != ENGINE_FAMILY
        or payload["engine_name"] != ENGINE_NAME
        or payload["engine_revision"] != ENGINE_REVISION
        or payload["license_id"] != ENGINE_LICENSE
        or payload["solver"] != "exhaustive"
        or payload["path_orders"] != [0, 1]
        or payload["polarization"] != "V"
        or payload["array_model"] != "two-element-horizontal-half-wavelength"
        or payload["geometry_dtype"] != "float32"
        or payload["index_dtype"] != "int32"
        or payload["output_dtype"] != "float64"
        or payload["jax_enable_x64"] is not True
        or payload["jax_platforms"] != "cpu"
    ):
        raise ValueError("DiffeRT engine configuration differs from the frozen profile")
    for key in (
        "receiver_height_m",
        "material_thickness_m",
        "speed_of_light_m_s",
        "free_space_impedance_ohm",
    ):
        if not isinstance(payload[key], (int, float)) or not math.isfinite(payload[key]):
            raise ValueError(f"DiffeRT {key} must be finite")
    if (
        float(payload["receiver_height_m"]) <= 0
        or float(payload["material_thickness_m"]) <= 0
        or float(payload["speed_of_light_m_s"]) <= 0
        or float(payload["free_space_impedance_ohm"]) <= 0
    ):
        raise ValueError("DiffeRT physical constants must be positive")
    return payload


def load_scene_manifest(path: str | Path, dataset, engine_config_path: str | Path) -> dict:
    source = _regular_file(path, "DiffeRT scene manifest")
    payload = read_strict_json(source)
    required = {
        "schema_version",
        "dataset_sha256",
        "engine_family",
        "engine_name",
        "engine_revision",
        "configuration_sha256",
        "license_id",
        "worlds",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload["schema_version"] != SCENE_MANIFEST_SCHEMA
        or payload["dataset_sha256"] != sha256_file(dataset.source_path)
        or payload["engine_family"] != ENGINE_FAMILY
        or payload["engine_name"] != ENGINE_NAME
        or payload["engine_revision"] != ENGINE_REVISION
        or payload["configuration_sha256"] != sha256_file(engine_config_path)
        or payload["license_id"] != ENGINE_LICENSE
    ):
        raise ValueError("DiffeRT scene manifest identity or provenance is invalid")
    expected = {
        (str(dataset.scene_ids[int(scene)]), world)
        for scene in dataset.indices_for_role("external_validation")
        for world in range(dataset.world_count)
    }
    entry_fields = {
        "scene_id",
        "world",
        "canonical_map_sha256",
        "source_asset_id",
        "source_asset_path",
        "source_asset_sha256",
    }
    rows = payload["worlds"]
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or set(row) != entry_fields for row in rows
    ):
        raise ValueError("DiffeRT scene manifest world fields are invalid")
    observed = {(str(row["scene_id"]), int(row["world"])) for row in rows}
    if len(rows) != len(expected) or observed != expected:
        raise ValueError("DiffeRT scene manifest does not cover every external world")
    lookup = {
        str(dataset.scene_ids[int(scene)]): int(scene)
        for scene in dataset.indices_for_role("external_validation")
    }
    asset_ids = []
    asset_paths = []
    asset_digests = []
    for row in rows:
        scene = lookup[str(row["scene_id"])]
        world = int(row["world"])
        if row["canonical_map_sha256"] != str(dataset.canonical_map_sha256[scene, world]):
            raise ValueError("DiffeRT source asset is not bound to its canonical map")
        if not isinstance(row["source_asset_id"], str) or not row["source_asset_id"]:
            raise ValueError("DiffeRT source asset identity must be nonempty")
        asset = _resolve_relative_source_asset(
            source.parent, row["source_asset_path"], row["source_asset_sha256"]
        )
        asset_ids.append(row["source_asset_id"])
        asset_paths.append(asset)
        asset_digests.append(row["source_asset_sha256"])
    if (
        len(set(asset_ids)) != len(rows)
        or len(set(asset_paths)) != len(rows)
        or len(set(asset_digests)) != len(rows)
    ):
        raise ValueError("DiffeRT sibling worlds must use distinct source assets")
    return payload


def collect_runtime(engine_config_path: str | Path) -> dict:
    config_path = _regular_file(engine_config_path, "DiffeRT engine configuration")
    load_engine_config(config_path)
    project_root = Path(__file__).resolve().parents[2]
    provenance_path = project_root / "formal_v2/external_adapters/DIFFERT_PROVENANCE.json"
    requirements_path = (
        project_root
        / "formal_v2/external_adapters/requirements-differt-runtime-linux-x86_64.txt"
    )
    if sha256_file(requirements_path) != REQUIREMENTS_SHA256:
        raise RuntimeError("DiffeRT frozen requirements lock hash mismatch")
    if sha256_file(provenance_path) != ENGINE_PROVENANCE_SHA256:
        raise RuntimeError("DiffeRT engine provenance hash mismatch")
    provenance = read_strict_json(provenance_path)
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema_version")
        != "csi-pairs-v6-differt-provenance-v1"
        or provenance.get("repository")
        != "https://github.com/jeertmans/DiffeRT"
        or provenance.get("release") != "v0.10.0"
        or provenance.get("commit")
        != "673cc58ef61906b8ab0869dd206b3d032dbc01b2"
        or provenance.get("license_id") != ENGINE_LICENSE
        or provenance.get("primary_engine_independent") is not True
    ):
        raise RuntimeError("DiffeRT engine provenance fields are invalid")
    package_records = {}
    for name, expected in REQUIRED_DISTRIBUTIONS.items():
        distribution = importlib.metadata.distribution(name)
        record = distribution.read_text("RECORD")
        if distribution.version != expected or record is None:
            raise RuntimeError(f"DiffeRT runtime distribution mismatch: {name}")
        package_records[name] = {
            "version": distribution.version,
            "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
        }
    import jax

    devices = [f"{device.platform}:{device.id}" for device in jax.devices()]
    base = {
        "schema_version": RUNTIME_SCHEMA,
        "engine_family": ENGINE_FAMILY,
        "engine_name": ENGINE_NAME,
        "engine_revision": ENGINE_REVISION,
        "license_id": ENGINE_LICENSE,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "jax_devices": devices,
        "package_records": package_records,
        "requirements_sha256": REQUIREMENTS_SHA256,
        "engine_provenance_sha256": ENGINE_PROVENANCE_SHA256,
        "engine_config_sha256": sha256_file(config_path),
    }
    if (
        base["python_version"] != "3.12.13"
        or base["platform_system"] != "Linux"
        or base["platform_machine"] != "x86_64"
        or base["jax_enable_x64"] is not True
        or base["jax_platforms"] != "cpu"
        or not devices
        or any(not value.startswith("cpu:") for value in devices)
    ):
        raise RuntimeError("DiffeRT runtime is not the frozen Linux CPU x64 profile")
    encoded = json.dumps(
        base,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return {**base, "environment_sha256": hashlib.sha256(encoded).hexdigest()}


def _horizontal_array_offsets(antenna_count: int, wavelength_m: float) -> np.ndarray:
    antennas = int(antenna_count)
    wavelength = float(wavelength_m)
    if antennas < 1 or not math.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("DiffeRT antenna count and wavelength must be positive")
    y = (
        np.arange(antennas, dtype=np.float64) - 0.5 * float(antennas - 1)
    ) * (0.5 * wavelength)
    return np.column_stack((np.zeros(antennas), y, np.zeros(antennas)))


def _physical_world_bits(
    world_bits: np.ndarray, anchor_bits: np.ndarray
) -> np.ndarray:
    logical = np.asarray(world_bits, dtype=np.int64)
    anchor = np.asarray(anchor_bits, dtype=np.int64)
    if (
        logical.ndim != 1
        or anchor.shape != logical.shape
        or logical.size < 1
        or not np.all((logical == 0) | (logical == 1))
        or not np.all((anchor == 0) | (anchor == 1))
    ):
        raise RuntimeError("DiffeRT logical world and anchor bits are invalid")
    return np.bitwise_xor(logical, anchor)


def _world_face_materials(
    object_bounds: np.ndarray,
    triangle_count: int,
    primitive_mapping: np.ndarray,
    world_bits: np.ndarray,
) -> np.ndarray:
    bounds = np.asarray(object_bounds, dtype=np.int64)
    mapping = np.asarray(primitive_mapping, dtype=np.int64)
    bits = np.asarray(world_bits, dtype=np.int64)
    primitives = int(bits.size)
    if (
        bits.ndim != 1
        or primitives < 1
        or mapping.shape != (primitives,)
        or set(mapping.tolist()) != set(range(primitives))
        or not np.all((bits == 0) | (bits == 1))
        or bounds.shape != (2 + primitives, 2)
        or int(bounds[0, 0]) != 0
        or int(bounds[-1, 1]) != int(triangle_count)
        or not np.array_equal(bounds[1:, 0], bounds[:-1, 1])
        or np.any(bounds[:, 1] <= bounds[:, 0])
    ):
        raise RuntimeError("DiffeRT object, primitive, or world-bit layout is invalid")
    states = np.empty(primitives, dtype=np.int64)
    for bit_index, value in enumerate(bits):
        states[int(mapping[bit_index])] = int(value)
    face_materials = np.empty(int(triangle_count), dtype=np.int64)
    face_materials[bounds[0, 0] : bounds[0, 1]] = 0
    face_materials[bounds[1, 0] : bounds[1, 1]] = 1
    for primitive, state in enumerate(states):
        start, stop = bounds[2 + primitive]
        face_materials[int(start) : int(stop)] = 1 if int(state) == 0 else 2
    return face_materials


def _trace_source_asset(path: Path, config: dict, dataset, scene: int, world: int) -> np.ndarray:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from differt.em import materials
    from differt.geometry import Mesh, Scene
    from differt.plugins import deepmimo

    jax.config.update("jax_enable_x64", True)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != SOURCE_ASSET_ARRAYS:
            raise RuntimeError("DiffeRT source asset arrays must be exact")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    required_metadata = {
        "schema_version",
        "dataset_sha256",
        "scene_id",
        "world",
        "canonical_map_sha256",
        "engine_family",
        "engine_revision",
        "primary_csi_fields_read",
        "generator_config_sha256",
        "primitive_count",
        "anchor_bits",
        "state_materials",
        "antenna_count",
        "subcarrier_count",
    }
    if (
        not isinstance(metadata, dict)
        or set(metadata) != required_metadata
        or metadata["schema_version"] != SOURCE_ASSET_SCHEMA
        or metadata["dataset_sha256"] != sha256_file(dataset.source_path)
        or metadata["scene_id"] != str(dataset.scene_ids[scene])
        or int(metadata["world"]) != world
        or metadata["canonical_map_sha256"]
        != str(dataset.canonical_map_sha256[scene, world])
        or metadata["engine_family"] != ENGINE_FAMILY
        or metadata["engine_revision"] != ENGINE_REVISION
        or metadata["primary_csi_fields_read"] is not False
        or not _lower_sha256(metadata["generator_config_sha256"])
        or int(metadata["primitive_count"]) != int(dataset.world_bits.shape[1])
        or metadata["anchor_bits"]
        != np.asarray(dataset.anchor_bits[scene], dtype=np.int64).tolist()
        or metadata["state_materials"] != list(STATE_MATERIALS)
        or int(metadata["antenna_count"]) != int(dataset.radio_config[scene, 1])
        or int(metadata["subcarrier_count"]) != int(dataset.radio_config[scene, 2])
    ):
        raise RuntimeError("DiffeRT source asset metadata is invalid")
    vertices = np.asarray(arrays["vertices"], dtype=np.float64)
    triangles = np.asarray(arrays["triangles"], dtype=np.int64)
    face_materials = np.asarray(arrays["face_materials"], dtype=np.int64)
    object_bounds = np.asarray(arrays["object_bounds"], dtype=np.int64)
    names = tuple(np.asarray(arrays["material_names"]).astype(str).tolist())
    transmitters = np.asarray(arrays["transmitters"], dtype=np.float64)
    receivers = np.asarray(arrays["receivers"], dtype=np.float64)
    frequencies = np.asarray(arrays["frequencies_hz"], dtype=np.float64)
    radio = np.asarray(dataset.radio_config[scene], dtype=np.float64)
    antenna_count = int(radio[1])
    subcarrier_count = int(radio[2])
    primitive_count = int(dataset.world_bits.shape[1])
    expected_frequencies = radio[0] + _sionna_subcarrier_offsets_hz(
        subcarrier_count, float(radio[3])
    )
    expected_face_materials = _world_face_materials(
        object_bounds,
        triangles.shape[0],
        dataset.primitive_ids[scene],
        _physical_world_bits(
            dataset.world_bits[world], dataset.anchor_bits[scene]
        ),
    )
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or face_materials.shape != (triangles.shape[0],)
        or object_bounds.shape != (2 + primitive_count, 2)
        or transmitters.shape != (antenna_count, 3)
        or receivers.shape != (dataset.position_count, 3)
        or frequencies.shape != (subcarrier_count,)
        or dataset.channel_count != 2 * antenna_count * subcarrier_count
        or not np.array_equal(frequencies, expected_frequencies)
        or not np.array_equal(face_materials, expected_face_materials)
        or not all(np.all(np.isfinite(value)) for value in (vertices, transmitters, receivers, frequencies))
    ):
        raise RuntimeError("DiffeRT source asset geometry or radio arrays are invalid")
    int32 = np.iinfo(np.int32)
    if (
        triangles.size
        and (int(triangles.min()) < 0 or int(triangles.max()) > int32.max)
    ) or (
        object_bounds.size
        and (int(object_bounds.min()) < 0 or int(object_bounds.max()) > int32.max)
    ) or (
        face_materials.size
        and (int(face_materials.min()) < 0 or int(face_materials.max()) > int32.max)
    ):
        raise RuntimeError("DiffeRT source asset indices do not fit the frozen int32 kernel")
    if names != MATERIAL_NAMES:
        raise RuntimeError("DiffeRT source asset material registry is invalid")
    mesh = Mesh(
        vertices=jnp.asarray(vertices, dtype=jnp.float32),
        triangles=jnp.asarray(triangles, dtype=jnp.int32),
        face_materials=jnp.asarray(face_materials, dtype=jnp.int32),
        material_names=names,
        object_bounds=jnp.asarray(object_bounds, dtype=jnp.int32),
        assume_unique_vertices=True,
    )
    rt_scene = Scene(
        transmitters=jnp.asarray(transmitters, dtype=jnp.float32),
        receivers=jnp.asarray(receivers, dtype=jnp.float32),
        mesh=mesh,
    )
    paths = [
        rt_scene.trace_paths(order=int(order), solver=str(config["solver"]))
        for order in config["path_orders"]
    ]
    thickness = float(config["material_thickness_m"])
    radio_materials = {
        name: replace(materials[name], thickness=thickness) for name in names
    }
    per_frequency = []
    for frequency in frequencies:
        result = deepmimo.export(
            paths=paths,
            scene=rt_scene,
            radio_materials=radio_materials,
            frequency=float(frequency),
            polarization=str(config["polarization"]),
        )
        power = np.asarray(result.power, dtype=np.float64)
        phase = np.deg2rad(np.asarray(result.phase, dtype=np.float64))
        mask = np.asarray(result.mask, dtype=np.bool_)
        magnitude = np.sqrt(
            np.power(10.0, power / 10.0)
            * float(config["free_space_impedance_ohm"])
        )
        coefficients = magnitude * np.exp(1j * phase)
        per_frequency.append(np.sum(np.where(mask, coefficients, 0.0), axis=-1))
    # [frequency, tx, rx] -> [rx, tx, frequency]
    cfr = np.transpose(np.stack(per_frequency, axis=0), (2, 1, 0))
    reference = np.asarray(dataset.phase_reference_values[scene], dtype=np.complex128)
    cfr = cfr * np.conjugate(reference)[:, None, None] / np.abs(reference)[:, None, None]
    flat = cfr.reshape(dataset.position_count, -1)
    joined = np.concatenate((flat.real, flat.imag), axis=1)
    if joined.shape != (dataset.position_count, dataset.channel_count):
        raise RuntimeError("DiffeRT CFR does not match the formal channel contract")
    return np.asarray(joined, dtype=np.float64)


def _resolve_relative_source_asset(root: Path, value: str, digest: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("DiffeRT source asset path must be nonempty")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("DiffeRT source asset path must be relative")
    source_root = Path(root).resolve()
    cursor = source_root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError("DiffeRT source asset must be a regular file")
    source = cursor.resolve()
    if source_root != source.parent and source_root not in source.parents:
        raise RuntimeError("DiffeRT source asset escapes its manifest root")
    if not _lower_sha256(digest) or not source.is_file() or sha256_file(source) != digest:
        raise RuntimeError("DiffeRT source asset is missing or hash-mismatched")
    return source


def _sionna_subcarrier_offsets_hz(count: int, spacing_hz: float) -> np.ndarray:
    if count <= 0 or not math.isfinite(spacing_hz) or spacing_hz <= 0:
        raise ValueError("subcarrier count and spacing must be positive")
    # Matches sionna.rt.subcarrier_frequencies for both even and odd FFT sizes.
    return (
        np.arange(count, dtype=np.float64) - int(count // 2)
    ) * float(spacing_hz)


def _regular_file(value: str | Path, label: str) -> Path:
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular file")
    return path.resolve()


def _lower_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


if __name__ == "__main__":
    raise SystemExit(main())
