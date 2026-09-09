#!/usr/bin/env python3
"""Quantify deterministic-regeneration drift without relaxing the formal gate."""

from __future__ import annotations

import argparse
from collections import Counter
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


def _numeric(left: np.ndarray, right: np.ndarray) -> dict:
    left = np.asarray(left)
    right = np.asarray(right)
    delta = np.abs(left - right)
    scale = np.abs(left)
    nonzero = scale > 0
    relative = delta[nonzero] / scale[nonzero]
    return {
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "exact_fraction": float(np.mean(left == right)),
        "different_elements": int(np.count_nonzero(left != right)),
        "max_absolute_difference": float(np.max(delta, initial=0.0)),
        "mean_absolute_difference": float(np.mean(delta)),
        "p99_absolute_difference": float(np.quantile(delta, 0.99)),
        "max_relative_difference_on_nonzero_reference": (
            float(np.max(relative, initial=0.0))
        ),
        "p99_relative_difference_on_nonzero_reference": (
            float(np.quantile(relative, 0.99)) if relative.size else 0.0
        ),
        "reference_zero_regenerated_nonzero": int(np.count_nonzero((left == 0) & (right != 0))),
        "reference_nonzero_regenerated_zero": int(np.count_nonzero((left != 0) & (right == 0))),
    }


def _visibility(clean: np.ndarray) -> np.ndarray:
    return np.any(np.asarray(clean) != 0, axis=-1)


def _confusion(reference: np.ndarray, regenerated: np.ndarray) -> dict:
    reference = np.asarray(reference, dtype=bool)
    regenerated = np.asarray(regenerated, dtype=bool)
    return {
        "both_visible": int(np.count_nonzero(reference & regenerated)),
        "reference_only_visible": int(np.count_nonzero(reference & ~regenerated)),
        "regenerated_only_visible": int(np.count_nonzero(~reference & regenerated)),
        "both_invisible": int(np.count_nonzero(~reference & ~regenerated)),
        "exact_fraction": float(np.mean(reference == regenerated)),
    }


def _counter_overlap(left: Counter, right: Counter) -> tuple[int, int]:
    intersection = sum((left & right).values())
    union = sum((left | right).values())
    return intersection, union


def _path_metrics(
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    left_power: np.ndarray,
    right_power: np.ndarray,
    left_surfaces: np.ndarray,
    right_surfaces: np.ndarray,
) -> dict:
    units = int(np.prod(left_ids.shape[:-1]))
    left_ids = left_ids.reshape(units, left_ids.shape[-1])
    right_ids = right_ids.reshape(units, right_ids.shape[-1])
    left_power = left_power.reshape(units, left_power.shape[-1])
    right_power = right_power.reshape(units, right_power.shape[-1])
    left_surfaces = left_surfaces.reshape(units, left_surfaces.shape[-2], left_surfaces.shape[-1])
    right_surfaces = right_surfaces.reshape(units, right_surfaces.shape[-2], right_surfaces.shape[-1])
    id_intersection = id_union = surface_intersection = surface_union = 0
    exact_id_units = exact_surface_units = exact_count_units = 0
    matched_power_differences: list[float] = []
    left_count_total = right_count_total = 0
    for unit in range(units):
        left_valid = left_ids[unit] >= 0
        right_valid = right_ids[unit] >= 0
        left_count = int(np.count_nonzero(left_valid))
        right_count = int(np.count_nonzero(right_valid))
        left_count_total += left_count
        right_count_total += right_count
        exact_count_units += left_count == right_count
        left_id_set = set(int(value) for value in left_ids[unit, left_valid])
        right_id_set = set(int(value) for value in right_ids[unit, right_valid])
        id_intersection += len(left_id_set & right_id_set)
        id_union += len(left_id_set | right_id_set)
        exact_id_units += left_id_set == right_id_set
        left_surface_counter = Counter(
            tuple(int(value) for value in row)
            for row in left_surfaces[unit, left_valid]
        )
        right_surface_counter = Counter(
            tuple(int(value) for value in row)
            for row in right_surfaces[unit, right_valid]
        )
        overlap, union = _counter_overlap(left_surface_counter, right_surface_counter)
        surface_intersection += overlap
        surface_union += union
        exact_surface_units += left_surface_counter == right_surface_counter
        right_by_id = {
            int(path_id): float(power)
            for path_id, power in zip(right_ids[unit, right_valid], right_power[unit, right_valid])
        }
        for path_id, power in zip(left_ids[unit, left_valid], left_power[unit, left_valid]):
            if int(path_id) in right_by_id:
                matched_power_differences.append(abs(float(power) - right_by_id[int(path_id)]))
    matched = np.asarray(matched_power_differences, dtype=np.float64)
    return {
        "unit_count": units,
        "reference_path_count": left_count_total,
        "regenerated_path_count": right_count_total,
        "units_with_exact_path_count_fraction": exact_count_units / units,
        "path_id_weighted_intersection": id_intersection,
        "path_id_weighted_union": id_union,
        "path_id_weighted_jaccard": id_intersection / id_union if id_union else 1.0,
        "units_with_exact_path_id_set_fraction": exact_id_units / units,
        "surface_sequence_multiset_weighted_intersection": surface_intersection,
        "surface_sequence_multiset_weighted_union": surface_union,
        "surface_sequence_multiset_weighted_jaccard": (
            surface_intersection / surface_union if surface_union else 1.0
        ),
        "units_with_exact_surface_sequence_multiset_fraction": exact_surface_units / units,
        "matched_path_id_power_pairs": int(matched.size),
        "matched_path_id_power_max_absolute_difference": (
            float(np.max(matched, initial=0.0))
        ),
        "matched_path_id_power_p99_absolute_difference": (
            float(np.quantile(matched, 0.99)) if matched.size else None
        ),
    }


def _slice_report(reference: dict, regenerated: dict, indices: np.ndarray) -> dict:
    result = {
        "scene_indices": [int(value) for value in indices],
        "scene_count": int(indices.size),
        "csi_clean": _numeric(reference["csi_clean"][indices], regenerated["csi_clean"][indices]),
        "csi_repeat": _numeric(reference["csi_repeat"][indices], regenerated["csi_repeat"][indices]),
        "clean_visibility": _confusion(
            _visibility(reference["csi_clean"][indices]),
            _visibility(regenerated["csi_clean"][indices]),
        ),
    }
    for prefix in ("", "noop_"):
        label = "paths" if not prefix else "noop_paths"
        result[label] = _path_metrics(
            reference[f"{prefix}path_ids"][indices],
            regenerated[f"{prefix}path_ids"][indices],
            reference[f"{prefix}path_power"][indices],
            regenerated[f"{prefix}path_power"][indices],
            reference[f"{prefix}path_surface_ids"][indices],
            regenerated[f"{prefix}path_surface_ids"][indices],
        )
    residual_left = reference["csi_repeat"][indices] - reference["csi_clean"][indices, :, :, None, :]
    residual_right = regenerated["csi_repeat"][indices] - regenerated["csi_clean"][indices, :, :, None, :]
    result["observation_noise_residual"] = _numeric(residual_left, residual_right)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--regenerated", type=Path, required=True)
    parser.add_argument("--metadata-reference", type=Path)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    with np.load(args.reference, allow_pickle=False) as archive:
        reference = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(args.regenerated, allow_pickle=False) as archive:
        regenerated = {name: np.asarray(archive[name]) for name in archive.files}
    metadata_path = args.metadata_reference or args.reference
    with np.load(metadata_path, allow_pickle=False) as archive:
        metadata = {name: np.asarray(archive[name]) for name in archive.files}
    regenerated_required = {
        "csi_clean", "csi_repeat", "path_ids", "path_power",
        "path_surface_ids", "noop_path_ids", "noop_path_power", "noop_path_surface_ids",
    }
    if (
        not regenerated_required.issubset(reference)
        or regenerated_required - set(regenerated)
        or "scene_ids" not in metadata
    ):
        raise RuntimeError("input archives lack required comparison arrays")
    if reference["csi_clean"].shape != regenerated["csi_clean"].shape:
        raise RuntimeError("archive scene shapes differ")
    generation = json.loads(args.generation_manifest.read_text(encoding="utf-8"))
    groups: dict[str, np.ndarray] = {}
    for shard in generation["shards"]:
        shard_path = Path(shard["path"])
        with np.load(shard_path, allow_pickle=False) as shard_archive:
            scene_indices = np.asarray(shard_archive["scene_indices"], dtype=np.int64)
        groups[f"original_gpu{int(shard['physical_gpu_index'])}"] = scene_indices
    scene_count = reference["csi_clean"].shape[0]
    per_scene = []
    for scene in range(scene_count):
        row = _slice_report(reference, regenerated, np.asarray([scene], dtype=np.int64))
        row["scene_index"] = scene
        row["scene_id"] = str(metadata["scene_ids"][scene])
        per_scene.append(row)
    payload = {
        "schema_version": "csi-pairs-sionna-regeneration-drift-audit-v1",
        "purpose": "diagnostic-only; does not relax or replace the exact formal verification gate",
        "status": "FAIL" if not np.array_equal(reference["csi_clean"], regenerated["csi_clean"]) else "PASS",
        "reference": {"path": str(args.reference.resolve()), "sha256": _sha256(args.reference)},
        "regenerated": {"path": str(args.regenerated.resolve()), "sha256": _sha256(args.regenerated)},
        "metadata_reference": {
            "path": str(metadata_path.resolve()),
            "sha256": _sha256(metadata_path),
        },
        "generation_manifest": {
            "path": str(args.generation_manifest.resolve()),
            "sha256": _sha256(args.generation_manifest),
        },
        "overall": _slice_report(reference, regenerated, np.arange(scene_count, dtype=np.int64)),
        "by_original_generation_gpu": {
            label: _slice_report(reference, regenerated, indices)
            for label, indices in groups.items()
        },
        "per_scene": per_scene,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
