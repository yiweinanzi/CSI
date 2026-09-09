#!/usr/bin/env python3
"""Generate a full-scale, permanently non-scientific CSI-PAIRS V6 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = REPO_ROOT / "code" / "CSI-PAIRS-v2.0-server"
sys.path.insert(0, str(SERVER_ROOT))

from formal_v2.formal_config import load_formal_config  # noqa: E402
from formal_v2.formal_dataset import (  # noqa: E402
    FormalDataset,
    SOURCE_ROLES,
    _array_sha256,
)
from formal_v2.formal_fixture import write_nonscientific_fixture  # noqa: E402
from formal_v2.formal_io import parse_strict_json, sha256_file  # noqa: E402


SEED = 20270805
SOURCE_BANKS_PER_ROLE = 2
TARGET_BANKS_PER_CITY = 9
TARGET_CITY_COUNT = 2
EXTERNAL_VALIDATION_BANKS = 4
POSITIONS_PER_BANK = 30


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _load_arrays(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        names = list(archive.files)
        arrays = {name: np.array(archive[name], copy=True) for name in names}
    return names, arrays


def _make_foundations_unique(arrays: dict[str, np.ndarray]) -> None:
    channel_names = [str(value) for value in arrays["map_channel_names"].tolist()]
    height_channel = channel_names.index("height")
    maps = arrays["maps"]

    # The fixture boundary is occupied in every world, so this tag changes the
    # canonical foundation identity without consuming a receiver position.
    for scene in range(maps.shape[0]):
        tag_height = 2.0 + (scene + 1) / 1000.0
        maps[scene, :, height_channel, 0, 1] = tag_height

    arrays["noop_maps"] = maps.copy()
    arrays["canonical_map_sha256"] = np.asarray(
        [
            [_array_sha256(maps[scene, world]) for world in range(maps.shape[1])]
            for scene in range(maps.shape[0])
        ],
        dtype="U64",
    )
    arrays["noop_map_sha256"] = arrays["canonical_map_sha256"].copy()


def _assign_external_validation_banks(arrays: dict[str, np.ndarray]) -> None:
    external_start = (
        len(SOURCE_ROLES) * SOURCE_BANKS_PER_ROLE
        + TARGET_CITY_COUNT * TARGET_BANKS_PER_CITY
    )
    scene_count = int(arrays["scene_roles"].shape[0])
    if scene_count - external_start != EXTERNAL_VALIDATION_BANKS:
        raise RuntimeError("generated scene count does not match the external-bank plan")

    for offset, scene in enumerate(range(external_start, scene_count)):
        city = "external-a" if offset % 2 == 0 else "external-b"
        arrays["scene_roles"][scene] = "external_validation"
        arrays["city_ids"][scene] = city
        arrays["position_roles"][scene, :] = "standard"
        bank = str(arrays["bank_ids"][scene])
        for position in range(arrays["position_ids"].shape[1]):
            arrays["position_ids"][scene, position] = (
                f"{city}:{bank}:p{position:04d}"
            )


def _update_metadata(arrays: dict[str, np.ndarray]) -> None:
    engine_config = parse_strict_json(
        str(np.asarray(arrays["engine_config_json"]).item())
    )
    engine_config.update(
        {
            "profile": "full-structural-synthetic",
            "source_banks_per_role": SOURCE_BANKS_PER_ROLE,
            "target_banks_per_city": TARGET_BANKS_PER_CITY,
            "target_city_count": TARGET_CITY_COUNT,
            "external_validation_banks": EXTERNAL_VALIDATION_BANKS,
            "positions_per_bank": POSITIONS_PER_BANK,
            "foundation_variant_rule": "occupied-boundary-height-scene-tag-v1",
        }
    )
    engine_text = _canonical_json(engine_config)
    arrays["engine_config_json"] = np.asarray(engine_text)

    metadata = parse_strict_json(str(np.asarray(arrays["metadata_json"]).item()))
    metadata["dataset_id"] = "NONSCIENTIFIC-FULL-STRUCTURAL-FIXTURE"
    metadata["dataset_version"] = "2.1-v6-full-synthetic"
    metadata["engine"]["name"] = "deterministic-full-fixture-generator-not-rt"
    metadata["engine"]["config_sha256"] = hashlib.sha256(
        engine_text.encode("utf-8")
    ).hexdigest()
    metadata["generation"].update(
        {
            "created_utc": "2026-08-09T00:00:00Z",
            "generator_command": (
                "python3 generated_datasets/tools/generate_full_synthetic.py "
                "--output generated_datasets/csi_pairs_v2_1_v6_full_synthetic/"
                "csi_pairs_v2_1_v6_full_synthetic.npz"
            ),
            "seed": SEED,
        }
    )
    metadata["external_reference"] = {
        "available": True,
        "kind": "synthetic-fixture-mirror-not-independent-engine",
        "dataset_id": "NONSCIENTIFIC-SYNTHETIC-EXTERNAL-BANKS",
        "pairing_rule": "same deterministic fixture generator with disjoint role and city IDs",
    }
    arrays["metadata_json"] = np.asarray(_canonical_json(metadata))


def _validate_against_formal_structure(path: Path) -> dict[str, object]:
    config = load_formal_config(SERVER_ROOT / "formal_v2" / "configs" / "formal_v2.json")
    dataset = FormalDataset.load(path)
    dataset.validate(
        require_clean_csi=bool(config["data"]["require_clean_csi"]),
        minimum_repeats=int(config["data"]["minimum_repeats"]),
        minimum_target_cities=int(config["data"]["minimum_target_cities"]),
        minimum_source_cities=int(config["data"]["minimum_source_cities"]),
        minimum_banks_per_target_city=int(
            config["data"]["minimum_banks_per_target_city"]
        ),
        minimum_independent_base_map_clusters_per_target_city=int(
            config["data"]["minimum_independent_base_map_clusters_per_target_city"]
        ),
        minimum_banks_per_source_role=int(
            config["data"]["minimum_banks_per_source_role"]
        ),
        minimum_unique_support_positions_per_target_city=max(
            int(value) for value in config["localization"]["label_budgets"]
        ),
    )
    report = dataset.contract_report()
    support_counts = {
        city: len(dataset.unique_target_support_positions(city))
        for city in report["target_cities"]
    }
    target_cluster_counts = {
        city: len(
            {
                dataset.canonical_base_map_digest(scene)
                for scene in dataset.indices_for_role("target")
                if str(dataset.city_ids[scene]) == city
            }
        )
        for city in report["target_cities"]
    }
    return {
        **report,
        "unique_target_support_positions": support_counts,
        "target_canonical_cluster_counts": target_cluster_counts,
    }


def generate(output: Path) -> dict[str, object]:
    target = output if output.suffix == ".npz" else Path(f"{output}.npz")
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite dataset: {target}")

    scene_count = (
        len(SOURCE_ROLES) * SOURCE_BANKS_PER_ROLE
        + TARGET_CITY_COUNT * TARGET_BANKS_PER_CITY
        + EXTERNAL_VALIDATION_BANKS
    )
    with tempfile.TemporaryDirectory(prefix=".full-synthetic-", dir=target.parent) as temporary:
        temporary_root = Path(temporary)
        base_path = write_nonscientific_fixture(
            temporary_root / "base.npz",
            seed=SEED,
            scene_count=scene_count,
            positions=POSITIONS_PER_BANK,
            source_banks_per_role=SOURCE_BANKS_PER_ROLE,
        )
        names, arrays = _load_arrays(base_path)
        _assign_external_validation_banks(arrays)
        _make_foundations_unique(arrays)
        _update_metadata(arrays)

        candidate = temporary_root / "candidate.npz"
        with candidate.open("xb") as handle:
            np.savez_compressed(handle, **{name: arrays[name] for name in names})
        report = _validate_against_formal_structure(candidate)

        # Hard-link publication is atomic and refuses an existing destination.
        os.link(candidate, target)

    return {
        "status": "PASS",
        "dataset": str(target),
        "dataset_sha256": sha256_file(target),
        "bytes": target.stat().st_size,
        "fixture": True,
        "scientific_use": "FORBIDDEN",
        "formal_structure_validation": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(_canonical_json(generate(args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
