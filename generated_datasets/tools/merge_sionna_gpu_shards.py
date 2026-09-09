#!/usr/bin/env python3
"""Merge two Sionna GPU shards into a complete non-scientific V6 dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "code" / "CSI-PAIRS-v2.0-server"
sys.path.insert(0, str(SERVER_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from formal_v2.formal_config import load_formal_config  # noqa: E402
from formal_v2.formal_dataset import (  # noqa: E402
    DATASET_SCHEMA_VERSION,
    FormalDataset,
    SOURCE_ROLES,
    _array_sha256,
)
from formal_v2.formal_fixture import write_nonscientific_fixture  # noqa: E402
from formal_v2.formal_io import sha256_file  # noqa: E402
from generate_sionna_gpu_shard import (  # noqa: E402
    ANTENNAS,
    BANDWIDTH_HZ,
    BS_GLOBAL_XYZ_M,
    CARRIER_FREQUENCY_HZ,
    CHANNELS,
    MAP_ORIGIN_XY_M,
    MAP_RESOLUTION_M,
    MAP_SIZE,
    MATERIAL_CATEGORY,
    MAX_DEPTH,
    MAX_PATHS,
    NOISE_STD,
    POSITIONS_PER_BANK,
    REPEATS,
    SCENE_COUNT,
    SCENE_NAME,
    SEED,
    SIONNA_REVISION,
    SUBCARRIERS,
    SUBCARRIER_SPACING_HZ,
    WORLDS,
    canonical_json,
)


FINAL_ARRAY_NAMES = (
    "csi_repeat",
    "csi_clean",
    "maps",
    "map_channel_names",
    "positions",
    "position_ids",
    "free_space",
    "radio_config",
    "bs_pose",
    "repeat_seeds",
    "phase_reference_ids",
    "phase_reference_values",
    "phase_reference_source_sha256",
    "base_map_cluster_ids",
    "canonical_map_sha256",
    "noop_maps",
    "noop_map_sha256",
    "engine_config_json",
    "path_ids",
    "path_power",
    "path_surface_ids",
    "noop_path_ids",
    "noop_path_power",
    "noop_path_surface_ids",
    "primitive_surface_ids",
    "world_bits",
    "primitive_ids",
    "anchor_bits",
    "natural_world_index",
    "scene_ids",
    "city_ids",
    "bank_ids",
    "scene_roles",
    "position_roles",
    "metadata_json",
)


def _read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")


def _load_shards(paths: list[Path]):
    rows = []
    worker_manifests = []
    for path_value in paths:
        path = path_value.resolve()
        manifest_path = path.with_suffix(".manifest.json")
        if not path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"shard or manifest is missing: {path}")
        manifest = _read_json(manifest_path)
        if manifest["schema_version"] != "csi-pairs-sionna-gpu-shard-manifest-v1":
            raise ValueError(f"invalid shard manifest: {manifest_path}")
        if manifest["output"]["sha256"] != sha256_file(path):
            raise ValueError(f"shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            shard = {name: np.array(archive[name], copy=True) for name in archive.files}
        if not np.array_equal(shard["world_bits"], WORLDS):
            raise ValueError(f"world hypercube mismatch: {path}")
        for local_row, scene_index in enumerate(shard["scene_indices"]):
            rows.append(
                (
                    int(scene_index),
                    {name: value[local_row] for name, value in shard.items() if name not in {"scene_indices", "world_bits"}},
                )
            )
        worker_manifests.append(manifest)
    rows.sort(key=lambda row: row[0])
    if [index for index, _ in rows] != list(range(SCENE_COUNT)):
        raise ValueError("shards must cover every scene index 0..35 exactly once")
    uuids = {manifest["gpu"]["uuid"] for manifest in worker_manifests}
    physical_indices = {manifest["gpu"]["physical_gpu_index"] for manifest in worker_manifests}
    if len(paths) != 2 or len(uuids) != 2 or physical_indices != {0, 1}:
        raise ValueError("merge requires exactly two shards from distinct physical GPUs 0 and 1")
    return rows, worker_manifests


def _stack(rows, name: str) -> np.ndarray:
    return np.stack([row[name] for _, row in rows], axis=0)


def _scene_ledger():
    roles = []
    cities = []
    for role in SOURCE_ROLES:
        roles.extend([role, role])
        cities.extend(["source-a", "source-b"])
    for offset in range(18):
        roles.append("target")
        cities.append("target-a" if offset % 2 == 0 else "target-b")
    for offset in range(4):
        roles.append("external_validation")
        cities.append("external-a" if offset % 2 == 0 else "external-b")
    return np.asarray(roles, dtype="U40"), np.asarray(cities, dtype="U32")


def _edge_effect_report(clean: np.ndarray, primitive_ids: np.ndarray) -> dict[str, object]:
    lookup = {tuple(int(value) for value in row): index for index, row in enumerate(WORLDS)}
    by_primitive = {0: [], 1: []}
    changed_positions = {0: 0, 1: 0}
    comparisons = {0: 0, 1: 0}
    for scene_index in range(SCENE_COUNT):
        for source_world, bits in enumerate(WORLDS):
            for bit_index in range(2):
                if int(bits[bit_index]) != 0:
                    continue
                target_bits = bits.copy()
                target_bits[bit_index] = 1
                target_world = lookup[tuple(int(value) for value in target_bits)]
                primitive = int(primitive_ids[scene_index, bit_index])
                delta = clean[scene_index, target_world] - clean[scene_index, source_world]
                position_rms = np.sqrt(np.mean(delta * delta, axis=1))
                by_primitive[primitive].append(float(np.sqrt(np.mean(delta * delta))))
                changed_positions[primitive] += int(np.sum(position_rms > 1e-12))
                comparisons[primitive] += int(position_rms.size)
    report = {}
    for primitive in (0, 1):
        values = np.asarray(by_primitive[primitive], dtype=np.float64)
        report[str(primitive)] = {
            "mean_edge_rms": float(np.mean(values)),
            "minimum_bank_edge_rms": float(np.min(values)),
            "maximum_bank_edge_rms": float(np.max(values)),
            "changed_position_comparisons": changed_positions[primitive],
            "total_position_comparisons": comparisons[primitive],
        }
        if not np.all(values > 1e-12):
            raise ValueError(f"primitive {primitive} has an inactive bank edge")
    return report


def _validate_formal(path: Path) -> dict[str, object]:
    config = load_formal_config(SERVER_ROOT / "formal_v2" / "configs" / "formal_v2.json")
    dataset = FormalDataset.load(path)
    dataset.validate(
        require_clean_csi=bool(config["data"]["require_clean_csi"]),
        minimum_repeats=int(config["data"]["minimum_repeats"]),
        minimum_target_cities=int(config["data"]["minimum_target_cities"]),
        minimum_source_cities=int(config["data"]["minimum_source_cities"]),
        minimum_banks_per_target_city=int(config["data"]["minimum_banks_per_target_city"]),
        minimum_independent_base_map_clusters_per_target_city=int(
            config["data"]["minimum_independent_base_map_clusters_per_target_city"]
        ),
        minimum_banks_per_source_role=int(config["data"]["minimum_banks_per_source_role"]),
        minimum_unique_support_positions_per_target_city=max(
            int(value) for value in config["localization"]["label_budgets"]
        ),
    )
    report = dataset.contract_report()
    report["unique_target_support_positions"] = {
        city: len(dataset.unique_target_support_positions(city))
        for city in report["target_cities"]
    }
    report["target_canonical_cluster_counts"] = {
        city: len(
            {
                dataset.canonical_base_map_digest(scene)
                for scene in dataset.indices_for_role("target")
                if str(dataset.city_ids[scene]) == city
            }
        )
        for city in report["target_cities"]
    }
    return report


def merge(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.resolve()
    contract_path = args.contract_report.resolve()
    manifest_path = args.manifest.resolve()
    for path in (output, contract_path, manifest_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite output: {path}")
    rows, worker_manifests = _load_shards(args.shard)
    generated_at = datetime.now(timezone.utc).isoformat()
    roles, cities = _scene_ledger()

    with tempfile.TemporaryDirectory(prefix=".sionna-merge-", dir=output.parent) as temporary:
        temporary_root = Path(temporary)
        base_path = write_nonscientific_fixture(
            temporary_root / "base.npz",
            seed=SEED,
            scene_count=SCENE_COUNT,
            positions=POSITIONS_PER_BANK,
            source_banks_per_role=2,
        )
        with np.load(base_path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}

        arrays["csi_repeat"] = _stack(rows, "csi_repeat")
        arrays["csi_clean"] = _stack(rows, "csi_clean")
        arrays["maps"] = _stack(rows, "maps")
        arrays["map_channel_names"] = np.asarray(
            ["occupancy", "height", "material"], dtype="U16"
        )
        arrays["positions"] = _stack(rows, "positions")
        arrays["free_space"] = _stack(rows, "free_space")
        arrays["radio_config"] = _stack(rows, "radio_config")
        arrays["bs_pose"] = _stack(rows, "bs_pose")
        arrays["repeat_seeds"] = _stack(rows, "repeat_seeds")
        arrays["phase_reference_ids"] = _stack(rows, "phase_reference_ids")
        arrays["phase_reference_values"] = _stack(rows, "phase_reference_values")
        arrays["phase_reference_source_sha256"] = _stack(
            rows, "phase_reference_source_sha256"
        )
        arrays["path_ids"] = _stack(rows, "path_ids")
        arrays["path_power"] = _stack(rows, "path_power")
        arrays["path_surface_ids"] = _stack(rows, "path_surface_ids")
        arrays["primitive_surface_ids"] = _stack(rows, "primitive_surface_ids")
        arrays["primitive_ids"] = _stack(rows, "primitive_ids")
        arrays["anchor_bits"] = _stack(rows, "anchor_bits")
        arrays["natural_world_index"] = _stack(rows, "natural_world_index")
        arrays["world_bits"] = WORLDS.copy()
        arrays["noop_maps"] = arrays["maps"].copy()
        arrays["noop_path_ids"] = arrays["path_ids"].copy()
        arrays["noop_path_power"] = arrays["path_power"].copy()
        arrays["noop_path_surface_ids"] = arrays["path_surface_ids"].copy()
        arrays["canonical_map_sha256"] = np.asarray(
            [
                [_array_sha256(arrays["maps"][scene, world]) for world in range(len(WORLDS))]
                for scene in range(SCENE_COUNT)
            ],
            dtype="U64",
        )
        arrays["noop_map_sha256"] = arrays["canonical_map_sha256"].copy()

        arrays["scene_roles"] = roles
        arrays["city_ids"] = cities
        arrays["scene_ids"] = np.asarray(
            [f"sionna-rt-scene-{scene:02d}" for scene in range(SCENE_COUNT)], dtype="U40"
        )
        arrays["bank_ids"] = np.asarray(
            [f"sionna-rt-bank-{scene:02d}" for scene in range(SCENE_COUNT)], dtype="U40"
        )
        arrays["base_map_cluster_ids"] = np.asarray(
            [f"sionna-rt-base-{scene:02d}" for scene in range(SCENE_COUNT)], dtype="U40"
        )
        arrays["position_roles"] = np.full(
            (SCENE_COUNT, POSITIONS_PER_BANK), "standard", dtype="U16"
        )
        arrays["position_ids"] = np.empty(
            (SCENE_COUNT, POSITIONS_PER_BANK), dtype="U96"
        )
        for scene in range(SCENE_COUNT):
            if roles[scene] == "target":
                arrays["position_roles"][scene, : POSITIONS_PER_BANK // 2] = "support_pool"
                arrays["position_roles"][scene, POSITIONS_PER_BANK // 2 :] = "query"
            for position in range(POSITIONS_PER_BANK):
                arrays["position_ids"][scene, position] = (
                    f"{cities[scene]}:{arrays['bank_ids'][scene]}:p{position:04d}"
                )

        shard_generator_hashes = sorted(
            {manifest["generator"]["sha256"] for manifest in worker_manifests}
        )
        if len(shard_generator_hashes) != 1:
            raise ValueError("workers used different shard generator sources")
        runtime_versions = {
            key: worker_manifests[0]["runtime"][key]
            for key in ("python", "sionna", "sionna_rt", "mitsuba", "drjit", "mitsuba_variant")
        }
        for manifest in worker_manifests[1:]:
            for key, value in runtime_versions.items():
                if manifest["runtime"][key] != value:
                    raise ValueError(f"worker runtime mismatch for {key}")
        scene_info = worker_manifests[0]["scene"]
        if any(
            manifest["scene"]["asset_tree_sha256"] != scene_info["asset_tree_sha256"]
            for manifest in worker_manifests
        ):
            raise ValueError("worker scene asset inventories differ")

        engine_config = {
            "profile": "full-dual-a100-sionna-rt-nonscientific-fixture",
            "fixture": True,
            "seed": SEED,
            "scene_count": SCENE_COUNT,
            "world_bits": WORLDS.tolist(),
            "positions_per_bank": POSITIONS_PER_BANK,
            "repeats": REPEATS,
            "runtime_versions": runtime_versions,
            "sionna_revision": SIONNA_REVISION,
            "scene": {
                "name": SCENE_NAME,
                "xml_sha256": scene_info["xml_sha256"],
                "asset_tree_sha256": scene_info["asset_tree_sha256"],
            },
            "radio": {
                "carrier_frequency_hz": CARRIER_FREQUENCY_HZ,
                "bandwidth_hz": BANDWIDTH_HZ,
                "subcarrier_spacing_hz": SUBCARRIER_SPACING_HZ,
                "tx_array": "1x2-iso-V-half-wavelength",
                "rx_array": "1x1-iso-V",
            },
            "path_solver": {
                "max_depth": MAX_DEPTH,
                "max_stored_paths": MAX_PATHS,
                "los": True,
                "specular_reflection": True,
                "diffuse_reflection": False,
                "refraction": False,
                "synthetic_array": True,
            },
            "interventions": {
                "primitive_0": "building_2 radio material brick-to-glass",
                "primitive_1": "building_6 radio material wood-to-marble",
            },
            "map": {
                "channels": ["occupancy", "height", "material"],
                "size": MAP_SIZE,
                "resolution_m": MAP_RESOLUTION_M,
                "origin_xy_m": MAP_ORIGIN_XY_M.tolist(),
                "raster_rule": "scene-object-aabb-v1",
                "foundation_variant_rule": "occupied-height-bank-tag-v1",
            },
            "noise": {
                "distribution": "independent-normal-real-channels",
                "standard_deviation": NOISE_STD,
            },
            "shards": [
                {
                    "sha256": manifest["output"]["sha256"],
                    "gpu_uuid": manifest["gpu"]["uuid"],
                    "physical_gpu_index": manifest["gpu"]["physical_gpu_index"],
                    "scene_start_inclusive": manifest["coverage"]["scene_start_inclusive"],
                    "scene_end_exclusive": manifest["coverage"]["scene_end_exclusive"],
                }
                for manifest in sorted(
                    worker_manifests, key=lambda row: row["gpu"]["physical_gpu_index"]
                )
            ],
            "generators": {
                "shard_sha256": shard_generator_hashes[0],
                "merge_sha256": sha256_file(Path(__file__).resolve()),
            },
        }
        engine_text = canonical_json(engine_config)
        arrays["engine_config_json"] = np.asarray(engine_text)
        metadata = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_id": "NONSCIENTIFIC-SIONNA-RT-DUAL-A100-FIXTURE",
            "dataset_version": "2.1-v6-sionna-rt-dual-a100",
            "scientific_use": "FORBIDDEN",
            "fixture": True,
            "engine": {
                "name": "NVIDIA-Sionna-RT-PathSolver",
                "version": runtime_versions["sionna_rt"],
                "source_revision": SIONNA_REVISION,
                "license_id": "Apache-2.0",
                "config_sha256": hashlib.sha256(engine_text.encode("ascii")).hexdigest(),
                "deterministic": True,
            },
            "representation": {
                "csi_layout": "real_then_imag",
                "csi_units": "unitless-linear-channel-coefficient",
                "phase_gauge_rule": "shared_complex_reference",
                "coordinate_system": "bs_centered_right_handed_meters",
                "position_units": "m",
                "map_units": "m",
                "clean_target_definition": "Sionna RT CFR after shared free-space geometric phase reference and before independent observation noise",
                "antenna_count": ANTENNAS,
                "subcarrier_count": SUBCARRIERS,
                "patch_complex_size": 1,
                "patch_antenna_size": 1,
                "patch_subcarrier_size": 1,
                "map_resolution_m": MAP_RESOLUTION_M,
                "map_origin_xy_m": MAP_ORIGIN_XY_M.tolist(),
                "alignment_physical_representation": "complex_csi_plus_delay_angle_power",
            },
            "assets": {
                "license_ids": ["Apache-2.0"],
                "provenance_uri": "package://sionna.rt.scene.simple_street_canyon",
                "material_library": "Sionna-RT-ITU-materials-categorical-v1",
                "material_category_count": len(MATERIAL_CATEGORY),
                "redistribution_allowed": True,
            },
            "generation": {
                "created_utc": generated_at,
                "generator_command": "two generate_sionna_gpu_shard.py workers on GPU 0/1 followed by merge_sionna_gpu_shards.py",
                "seed": SEED,
            },
            "external_reference": {
                "available": True,
                "kind": "same-Sionna-engine-disjoint-external-validation-banks-not-independent",
                "dataset_id": "NONSCIENTIFIC-SIONNA-RT-EXTERNAL-BANK-SUBSET",
                "pairing_rule": "four disjoint role-isolated banks generated by the same frozen Sionna RT configuration",
            },
        }
        arrays["metadata_json"] = np.asarray(canonical_json(metadata))

        if set(arrays) != set(FINAL_ARRAY_NAMES):
            raise RuntimeError(
                f"final array mismatch: missing={sorted(set(FINAL_ARRAY_NAMES)-set(arrays))}, "
                f"unexpected={sorted(set(arrays)-set(FINAL_ARRAY_NAMES))}"
            )
        edge_effects = _edge_effect_report(arrays["csi_clean"], arrays["primitive_ids"])
        candidate = temporary_root / "candidate.npz"
        with candidate.open("xb") as handle:
            np.savez_compressed(
                handle, **{name: arrays[name] for name in FINAL_ARRAY_NAMES}
            )
        contract = _validate_formal(candidate)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.link(candidate, output)

    contract["dataset"] = str(output)
    contract["dataset_sha256"] = sha256_file(output)
    contract["gpu_simulation"] = {
        "status": "PASS",
        "engine": "Sionna RT PathSolver",
        "trace_calls": SCENE_COUNT * len(WORLDS),
        "edge_effects": edge_effects,
    }
    _write_json(contract_path, contract)
    manifest = {
        "schema_version": "csi-pairs-sionna-dual-gpu-generation-manifest-v1",
        "generated_at": generated_at,
        "status": "PASS",
        "dataset": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_id": metadata["dataset_id"],
            "fixture": True,
            "scientific_use": "FORBIDDEN",
        },
        "coverage": contract["shape"],
        "simulation": {
            "engine": "Sionna RT PathSolver",
            "scene": SCENE_NAME,
            "trace_calls": SCENE_COUNT * len(WORLDS),
            "seed": SEED,
            "edge_effects": edge_effects,
        },
        "workers": worker_manifests,
        "validation": {
            "formal_config": str(
                SERVER_ROOT / "formal_v2" / "configs" / "formal_v2.json"
            ),
            "contract_report": str(contract_path),
            "contract_status": contract["status"],
            "independent_regeneration_gate": "NOT_RUN",
            "rt_calibration": "NOT_ASSESSED",
            "external_validity": "NOT_ASSESSED",
        },
        "scientific_limit": (
            "GPU ray tracing and formal structural validation completed, but this "
            "fixture is not measured data, independently calibrated RT evidence, or "
            "an independent-engine external-validity result."
        ),
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--contract-report", required=True, type=Path)
    manifest = merge(parser.parse_args())
    print(canonical_json({"status": "PASS", "dataset": manifest["dataset"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
