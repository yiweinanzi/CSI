#!/usr/bin/env python3
"""Exhaustively report RT path visibility without assigning a scientific threshold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _summary(mask: np.ndarray, los: np.ndarray, path_count: np.ndarray) -> dict:
    total = int(mask.size)
    no_path = int(np.count_nonzero(~mask))
    return {
        "units": total,
        "visible_path_units": total - no_path,
        "no_path_units": no_path,
        "no_path_fraction": _fraction(no_path, total),
        "los_units": int(np.count_nonzero(los)),
        "los_fraction": _fraction(int(np.count_nonzero(los)), total),
        "path_count_min": int(np.min(path_count)) if total else 0,
        "path_count_median": float(np.median(path_count)) if total else 0.0,
        "path_count_p90": float(np.quantile(path_count, 0.9)) if total else 0.0,
        "path_count_max": int(np.max(path_count)) if total else 0,
    }


def audit(dataset_path: Path) -> dict:
    with np.load(dataset_path, allow_pickle=False) as archive:
        required = {
            "csi_clean",
            "path_ids",
            "path_power",
            "path_surface_ids",
            "noop_path_ids",
            "noop_path_power",
            "noop_path_surface_ids",
            "scene_ids",
            "city_ids",
            "scene_roles",
            "position_roles",
            "positions",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"missing required arrays: {missing}")
        csi = np.asarray(archive["csi_clean"])
        path_ids = np.asarray(archive["path_ids"])
        path_power = np.asarray(archive["path_power"])
        surfaces = np.asarray(archive["path_surface_ids"])
        noop_ids = np.asarray(archive["noop_path_ids"])
        noop_power = np.asarray(archive["noop_path_power"])
        noop_surfaces = np.asarray(archive["noop_path_surface_ids"])
        scene_ids = np.asarray(archive["scene_ids"]).astype(str)
        cities = np.asarray(archive["city_ids"]).astype(str)
        roles = np.asarray(archive["scene_roles"]).astype(str)
        position_roles = np.asarray(archive["position_roles"]).astype(str)
        positions = np.asarray(archive["positions"], dtype=np.float64)

    valid_slots = (path_ids >= 0) & (path_power > 0)
    visible = np.any(valid_slots, axis=-1)
    path_count = np.count_nonzero(valid_slots, axis=-1)
    zero_csi = np.all(csi == 0, axis=-1)
    signed_zero_csi = np.all(np.abs(csi) == 0, axis=-1)
    los_slots = valid_slots & np.all(surfaces < 0, axis=-1)
    los = np.any(los_slots, axis=-1)

    if visible.shape != zero_csi.shape:
        raise ValueError(f"path/CSI shape mismatch: {visible.shape} versus {zero_csi.shape}")

    report = {
        "schema_version": "csi-pairs-rt-visibility-audit-v1",
        "dataset": str(dataset_path.resolve()),
        "dataset_bytes": dataset_path.stat().st_size,
        "dataset_sha256": _sha256(dataset_path),
        "scientific_gate": "NOT_ASSIGNED_REQUIRES_FROZEN_PROTOCOL",
        "overall": _summary(visible, los, path_count),
        "consistency": {
            "no_path_but_nonzero_csi": int(np.count_nonzero((~visible) & (~signed_zero_csi))),
            "visible_path_but_zero_csi": int(np.count_nonzero(visible & signed_zero_csi)),
            "exact_zero_csi_units": int(np.count_nonzero(zero_csi)),
            "signed_zero_csi_units": int(np.count_nonzero(signed_zero_csi)),
            "path_ids_noop_exact": bool(np.array_equal(path_ids, noop_ids)),
            "path_power_noop_exact": bool(np.array_equal(path_power, noop_power)),
            "path_surfaces_noop_exact": bool(np.array_equal(surfaces, noop_surfaces)),
        },
        "all_world_position_visibility": {
            "scene_position_units": int(visible.shape[0] * visible.shape[2]),
            "no_path_in_all_worlds": int(np.count_nonzero(~np.any(visible, axis=1))),
            "visible_in_all_worlds": int(np.count_nonzero(np.all(visible, axis=1))),
            "mixed_visibility_across_worlds": int(
                np.count_nonzero(np.any(visible, axis=1) & (~np.all(visible, axis=1)))
            ),
        },
        "by_role": {},
        "by_city": {},
        "by_world": {},
        "by_scene": {},
        "target_position_role": {},
        "radial_distance_bins_m": {},
    }

    for label in sorted(set(roles)):
        selector = roles == label
        report["by_role"][label] = _summary(
            visible[selector], los[selector], path_count[selector]
        )
    for label in sorted(set(cities)):
        selector = cities == label
        report["by_city"][label] = _summary(
            visible[selector], los[selector], path_count[selector]
        )
    for world in range(visible.shape[1]):
        report["by_world"][str(world)] = _summary(
            visible[:, world], los[:, world], path_count[:, world]
        )
    for scene, scene_id in enumerate(scene_ids):
        row = _summary(visible[scene], los[scene], path_count[scene])
        row.update(
            {
                "city_id": cities[scene],
                "scene_role": roles[scene],
                "positions_with_no_path_in_all_worlds": int(
                    np.count_nonzero(~np.any(visible[scene], axis=0))
                ),
                "positions_visible_in_all_worlds": int(
                    np.count_nonzero(np.all(visible[scene], axis=0))
                ),
            }
        )
        report["by_scene"][scene_id] = row

    target_scene = roles == "target"
    for label in sorted(set(position_roles[target_scene].reshape(-1))):
        selected_visible = []
        selected_los = []
        selected_count = []
        for scene in np.flatnonzero(target_scene):
            selector = position_roles[scene] == label
            selected_visible.append(visible[scene, :, selector])
            selected_los.append(los[scene, :, selector])
            selected_count.append(path_count[scene, :, selector])
        report["target_position_role"][label] = _summary(
            np.concatenate(selected_visible),
            np.concatenate(selected_los),
            np.concatenate(selected_count),
        )

    radii = np.linalg.norm(positions, axis=-1)
    bin_edges = (0.0, 32.0, 64.0, 96.0, 128.0, float("inf"))
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        selector = (radii >= lower) & (radii < upper)
        expanded = np.broadcast_to(selector[:, None, :], visible.shape)
        label = f"[{lower:g},{upper:g})"
        report["radial_distance_bins_m"][label] = _summary(
            visible[expanded], los[expanded], path_count[expanded]
        )

    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="ascii") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["overall"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
