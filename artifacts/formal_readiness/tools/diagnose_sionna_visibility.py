#!/usr/bin/env python3
"""Compare bounded Sionna propagation settings on one frozen candidate bank."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from formal_v2 import sionna_osm_candidate as candidate


CONDITIONS = (
    {
        "name": "registered_depth3_specular",
        "max_depth": 3,
        "specular_reflection": True,
        "diffuse_reflection": False,
        "refraction": False,
        "diffraction": False,
        "edge_diffraction": False,
    },
    {
        "name": "depth6_specular",
        "max_depth": 6,
        "specular_reflection": True,
        "diffuse_reflection": False,
        "refraction": False,
        "diffraction": False,
        "edge_diffraction": False,
    },
    {
        "name": "depth3_specular_diffraction",
        "max_depth": 3,
        "specular_reflection": True,
        "diffuse_reflection": False,
        "refraction": False,
        "diffraction": True,
        "edge_diffraction": False,
    },
    {
        "name": "depth3_specular_edge_diffraction",
        "max_depth": 3,
        "specular_reflection": True,
        "diffuse_reflection": False,
        "refraction": False,
        "diffraction": True,
        "edge_diffraction": True,
    },
    {
        "name": "depth3_specular_refraction",
        "max_depth": 3,
        "specular_reflection": True,
        "diffuse_reflection": False,
        "refraction": True,
        "diffraction": False,
        "edge_diffraction": False,
    },
    {
        "name": "depth6_specular_diffraction",
        "max_depth": 6,
        "specular_reflection": True,
        "diffuse_reflection": False,
        "refraction": False,
        "diffraction": True,
        "edge_diffraction": False,
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _visibility(paths, positions: np.ndarray) -> tuple[np.ndarray, dict]:
    valid = np.asarray(paths.valid)
    if valid.ndim != 3 or valid.shape[:2] != (positions.shape[0], 1):
        raise RuntimeError(f"unexpected Sionna valid-path shape: {valid.shape}")
    visible = np.any(valid[:, 0], axis=-1)
    count = np.count_nonzero(valid[:, 0], axis=-1)
    radii = np.linalg.norm(positions, axis=-1)
    radial = {}
    edges = (0.0, 32.0, 64.0, 96.0, 128.0, float("inf"))
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (radii >= lower) & (radii < upper)
        units = int(np.count_nonzero(selected))
        no_path = int(np.count_nonzero(selected & (~visible)))
        radial[f"[{lower:g},{upper:g})"] = {
            "positions": units,
            "no_path_positions": no_path,
            "no_path_fraction": float(no_path / units) if units else 0.0,
        }
    no_path = int(np.count_nonzero(~visible))
    return visible, {
        "positions": int(visible.size),
        "visible_positions": int(np.count_nonzero(visible)),
        "no_path_positions": no_path,
        "no_path_fraction": float(no_path / visible.size),
        "path_count_min": int(np.min(count)),
        "path_count_median": float(np.median(count)),
        "path_count_p90": float(np.quantile(count, 0.9)),
        "path_count_max": int(np.max(count)),
        "radial_distance_bins_m": radial,
    }


def diagnose(asset_root: Path, scene_index: int) -> dict:
    from sionna.rt import (
        ITURadioMaterial,
        PathSolver,
        PlanarArray,
        Receiver,
        Transmitter,
        load_scene,
    )

    root, manifest, config = candidate.load_asset_manifest(asset_root)
    bank_row = manifest["banks"][scene_index]
    bank_path = root / str(bank_row["bank_record_path"])
    bank = candidate._read_json(bank_path)
    radio = config["radio"]
    scene_path = root / str(bank_row["scene_xml_path"])
    scene = load_scene(scene_path)
    for material_type in ("glass", "metal"):
        name = f"itu_{material_type}"
        if name not in scene.radio_materials:
            scene.add(
                ITURadioMaterial(
                    name=name,
                    itu_type=material_type,
                    thickness=candidate.SIONNA_MATERIAL_THICKNESS_M,
                )
            )
    scene.frequency = float(radio["carrier_frequency_hz"])
    scene.bandwidth = float(radio["subcarrier_spacing_hz"]) * int(radio["subcarriers"])
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=int(radio["tx_antennas"]),
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.add(Transmitter("tx", position=[0.0, 0.0, float(radio["transmitter_z_m"])]))
    positions = candidate._receiver_positions(bank, config, scene_index)
    for position_index, xy in enumerate(positions):
        scene.add(
            Receiver(
                f"rx-{position_index}",
                position=[float(xy[0]), float(xy[1]), float(radio["receiver_z_m"])],
            )
        )
    worlds = np.asarray(config["world_bits"], dtype=np.int64)
    natural_world = int(candidate._natural_world_index(scene_index, worlds))
    mapping = candidate._primitive_mapping(scene_index)
    candidate._set_world_materials(
        scene, candidate._physical_states(worlds[natural_world], mapping)
    )
    solver = PathSolver()
    rows = []
    visibility_by_name = {}
    for condition_index, condition in enumerate(CONDITIONS):
        started = time.monotonic()
        paths = solver(
            scene,
            max_depth=int(condition["max_depth"]),
            los=True,
            specular_reflection=bool(condition["specular_reflection"]),
            diffuse_reflection=bool(condition["diffuse_reflection"]),
            refraction=bool(condition["refraction"]),
            diffraction=bool(condition["diffraction"]),
            edge_diffraction=bool(condition["edge_diffraction"]),
            synthetic_array=True,
            seed=int(config["seed"]) + scene_index * 1000 + natural_world + condition_index * 100_000,
        )
        visible, summary = _visibility(paths, positions)
        visibility_by_name[str(condition["name"])] = visible
        summary.update(condition)
        summary["duration_seconds"] = time.monotonic() - started
        rows.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)

    baseline = visibility_by_name["registered_depth3_specular"]
    for row in rows:
        visible = visibility_by_name[str(row["name"])]
        row["newly_visible_vs_registered"] = int(np.count_nonzero(visible & (~baseline)))
        row["lost_visibility_vs_registered"] = int(np.count_nonzero((~visible) & baseline))

    tool_path = Path(__file__).resolve()
    generator_path = Path(candidate.__file__).resolve()
    return {
        "schema_version": "csi-pairs-sionna-visibility-diagnostic-v1",
        "status": "DIAGNOSTIC_NOT_PAPER_EVIDENCE",
        "simulation_not_measurement": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime": candidate._sionna_runtime_record(),
        "asset_root": str(root),
        "asset_manifest_sha256": _sha256(root / "asset_manifest.json"),
        "bank_record": str(bank_path),
        "bank_record_sha256": _sha256(bank_path),
        "scene_xml": str(scene_path),
        "scene_xml_sha256": _sha256(scene_path),
        "scene_index": scene_index,
        "scene_id": bank_row["scene_id"],
        "city_id": bank_row["city_id"],
        "natural_world_index": natural_world,
        "positions": int(positions.shape[0]),
        "generator_path": str(generator_path),
        "generator_sha256": _sha256(generator_path),
        "tool_path": str(tool_path),
        "tool_sha256": _sha256(tool_path),
        "conditions": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.asset_root.resolve(), args.scene_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="ascii") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
