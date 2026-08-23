from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from urllib import parse as urlparse
from urllib import request as urlrequest
import xml.etree.ElementTree as ET

import numpy as np


CONFIG_SCHEMA = "csi-pairs-sionna-osm-candidate-config-v3"
CONFIG_SCHEMA_V4 = "csi-pairs-sionna-osm-candidate-config-v4"
CONFIG_SCHEMA_V5 = "csi-pairs-sionna-osm-candidate-config-v5"
CONFIG_SCHEMA_V6 = "csi-pairs-sionna-osm-candidate-config-v6"
CONTROLLED_CONFIG_SCHEMAS = {CONFIG_SCHEMA_V5, CONFIG_SCHEMA_V6}
POWER_CONFIG_SCHEMAS = {CONFIG_SCHEMA_V4, *CONTROLLED_CONFIG_SCHEMAS}
SUPPORTED_CONFIG_SCHEMAS = {CONFIG_SCHEMA, *POWER_CONFIG_SCHEMAS}
ASSET_SCHEMA = "csi-pairs-sionna-osm-asset-manifest-v3"
BANK_SCHEMA = "csi-pairs-sionna-osm-bank-v2"
SHARD_SCHEMA = "csi-pairs-sionna-osm-render-shard-v2"
RAW_OSM_RECEIPT_SCHEMA = "csi-pairs-raw-osm-download-receipt-v1"
DATASET_SCHEMA = "csi-pairs-formal-dataset-v2.1-v6"
SIONNA_REVISION = "04ddb9312116b408093b9d3ad363a3df355093a6"
SIONNA_VERSION = "2.0.1"
SIONNA_RT_VERSION = "1.2.1"
SIONNA_PYTHON_VERSION = "3.12.13"
MITSUBA_VERSION = "3.7.1"
DRJIT_VERSION = "1.2.0"
SIONNA_RUNTIME_RELATIVE = Path("formal_v2/external_adapters/.runtime-sionna/venv")
SIONNA_MATERIAL_THICKNESS_M = 0.1
PATH_VERTEX_QUANTIZATION_M = 1e-5
PATH_ANGLE_QUANTIZATION_RAD = 1e-5
SIONNA_MITSUBA_VARIANT = "llvm_ad_mono_polarized"
SIONNA_DRJIT_THREADS = 1
V5_MINIMUM_DIRECT_PATH_VERTICAL_CLEARANCE_M = 0.5
SOURCE_ROLES = (
    "source_encoder_train",
    "source_method_selection",
    "source_probe_train",
    "source_probe_selection",
    "source_calibration_fit",
    "source_calibration_selection",
    "source_final_unseen_bank",
)
MAP_CHANNEL_NAMES = np.asarray(("occupancy", "height", "material"), dtype="U16")
MATERIAL_CATEGORY = {
    "free_space": 0,
    "glass": 1,
    "wood": 2,
    "metal": 3,
    "concrete": 4,
}
STABLE_SURFACE_IDS = {
    "ground": 100,
    "background": 101,
    "primitive-0": 102,
    "primitive-1": 103,
}


def _stable_surface_ids(primitive_count: int) -> dict[str, int]:
    count = int(primitive_count)
    if count < 2:
        raise ValueError("a formal world bank requires at least two primitives")
    return {
        "ground": 100,
        "background": 101,
        **{f"primitive-{index}": 102 + index for index in range(count)},
    }
REGENERATED_FIELDS = (
    "maps",
    "csi_clean",
    "csi_repeat",
    "free_space",
    "phase_reference_ids",
    "phase_reference_values",
    "phase_reference_source_sha256",
    "engine_config_json",
    "noop_maps",
    "path_ids",
    "path_power",
    "path_surface_ids",
    "noop_path_ids",
    "noop_path_power",
    "noop_path_surface_ids",
)


class ReflectionGeometryQualificationError(RuntimeError):
    """A candidate bank cannot satisfy the registered reflection protocol."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: str | Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_json_constant)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _write_json_exclusive(path: str | Path, value: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        handle.write("\n")
    target.chmod(0o644)
    return target


def _write_json_atomic_exclusive(path: str | Path, value: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite JSON: {target}")
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            temporary.write(
                (
                    json.dumps(
                        value,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.link(temporary_path, target)
        target.chmod(0o644)
    finally:
        temporary_path.unlink(missing_ok=True)
    return target


def _write_text_exclusive(path: str | Path, value: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        handle.write(value)
    target.chmod(0o644)
    return target


def _write_npz_exclusive(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite NPZ: {target}")
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.link(temporary_path, target)
        target.chmod(0o644)
    finally:
        temporary_path.unlink(missing_ok=True)
    return target


def load_config(path: str | Path) -> dict:
    config = _read_json(path)
    if (
        not isinstance(config, dict)
        or config.get("schema_version") not in SUPPORTED_CONFIG_SCHEMAS
    ):
        raise ValueError("Sionna OSM candidate config schema mismatch")
    schema = str(config["schema_version"])
    required = {
        "schema_version",
        "seed",
        "overpass_endpoint",
        "map",
        "radio",
        "renderer",
        "positions_per_bank",
        "world_bits",
        "cities",
    }
    if schema in POWER_CONFIG_SCHEMAS:
        required.add("power_design")
    if schema in CONTROLLED_CONFIG_SCHEMAS:
        required.add("interventions")
    if schema == CONFIG_SCHEMA_V6:
        required.add("post_rt_exclusions")
    if set(config) != required:
        raise ValueError("Sionna OSM candidate config fields must be exact")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("candidate seed must be a nonnegative integer")
    if type(config["positions_per_bank"]) is not int or config["positions_per_bank"] < 256:
        raise ValueError("formal candidate requires at least 256 positions per bank")
    world_bits = np.asarray(config["world_bits"], dtype=np.int64)
    expected_bits = 4 if schema in CONTROLLED_CONFIG_SCHEMAS else 2
    expected_worlds = {
        tuple(int((world >> bit) & 1) for bit in range(expected_bits))
        for world in range(2**expected_bits)
    }
    if (
        world_bits.shape != (2**expected_bits, expected_bits)
        or set(map(tuple, world_bits.tolist())) != expected_worlds
    ):
        raise ValueError(
            f"candidate world_bits must be the complete {expected_bits}-bit hypercube"
        )
    cities = config["cities"]
    if not isinstance(cities, list) or len(cities) != 6:
        raise ValueError("candidate config requires exactly six city records")
    city_ids = set()
    counts = {"source": 0, "target": 0, "external": 0}
    for city in cities:
        expected = {
            "city_id",
            "split_group",
            "center_lat",
            "center_lon",
            "query_delta_lat",
            "query_delta_lon",
            "bank_count",
        }
        if (
            schema in POWER_CONFIG_SCHEMAS
            and isinstance(city, dict)
            and city.get("split_group") == "source"
        ):
            expected.add("source_role_bank_counts")
        if not isinstance(city, dict) or set(city) != expected:
            raise ValueError("city config fields must be exact")
        city_id = city["city_id"]
        if not isinstance(city_id, str) or not city_id or city_id in city_ids:
            raise ValueError("city ids must be unique nonempty strings")
        city_ids.add(city_id)
        group = city["split_group"]
        if group not in counts:
            raise ValueError("city split_group is invalid")
        counts[group] += 1
        if schema == CONFIG_SCHEMA:
            required_banks = 7 if group == "source" else 8 if group == "target" else 2
            if city["bank_count"] != required_banks:
                raise ValueError(f"{city_id} must register {required_banks} banks")
        else:
            if type(city["bank_count"]) is not int or int(city["bank_count"]) < 1:
                raise ValueError(f"{city_id} bank_count must be a positive integer")
            if group == "source":
                role_counts = city["source_role_bank_counts"]
                if (
                    not isinstance(role_counts, dict)
                    or set(role_counts) != set(SOURCE_ROLES)
                    or any(type(value) is not int or value < 1 for value in role_counts.values())
                    or sum(role_counts.values()) != int(city["bank_count"])
                ):
                    raise ValueError(
                        f"{city_id} source role counts must be positive, exact, and sum to bank_count"
                    )
    if counts != {"source": 2, "target": 2, "external": 2}:
        raise ValueError("candidate requires two source, target, and external cities")
    if schema in POWER_CONFIG_SCHEMAS:
        _validate_power_design(config)
    radio = config["radio"]
    expected_subcarriers = 16 if schema in CONTROLLED_CONFIG_SCHEMAS else 4
    if (
        radio["repeats"] < 3
        or radio["tx_antennas"] != 2
        or radio["subcarriers"] != expected_subcarriers
    ):
        raise ValueError(
            f"candidate radio shape must provide 3 repeats and a 2x{expected_subcarriers} complex grid"
        )
    map_config = config["map"]
    expected_map_fields = {
        "size",
        "resolution_m",
        "origin_xy_m",
        "candidate_grid_spacing_m",
        "candidate_grid_radius",
        "minimum_buildings_per_bank",
        "minimum_building_area_m2",
        "receiver_grid_spacing_m",
        "receiver_building_clearance_m",
        "receiver_minimum_bs_distance_m",
        "receiver_primitive_quota",
        "primitive_shortlist_size",
        "reflection_candidate_limit",
        "minimum_reflection_incidence_cosine",
    }
    if schema in CONTROLLED_CONFIG_SCHEMAS:
        expected_map_fields.update(
            {
                "bs_selection",
                "bs_placement_sample_count",
                "candidate_qualification_order",
                "receiver_minimum_direct_path_vertical_clearance_m",
            }
        )
    if set(map_config) != expected_map_fields:
        raise ValueError("candidate map config fields must be exact")
    if map_config["size"] != 256 or map_config["resolution_m"] != 1.0:
        raise ValueError("candidate map must use the frozen 256x256 one-meter grid")
    if schema in CONTROLLED_CONFIG_SCHEMAS and (
        map_config["bs_selection"]
        != "sampled-formal-placement-capacity-v1"
        or type(map_config["bs_placement_sample_count"]) is not int
        or not 4 <= int(map_config["bs_placement_sample_count"]) <= 128
        or map_config["candidate_qualification_order"]
        != "sparse-osm-first-v1"
    ):
        raise ValueError("V5 bank-placement selection policy is invalid")
    if (
        float(map_config["receiver_grid_spacing_m"]) <= 0.0
        or float(map_config["receiver_building_clearance_m"]) < 0.0
        or float(map_config["receiver_minimum_bs_distance_m"]) <= 0.0
        or (
            schema in CONTROLLED_CONFIG_SCHEMAS
            and float(
                map_config["receiver_minimum_direct_path_vertical_clearance_m"]
            )
            != V5_MINIMUM_DIRECT_PATH_VERTICAL_CLEARANCE_M
        )
        or type(map_config["receiver_primitive_quota"]) is not int
        or (
            schema not in CONTROLLED_CONFIG_SCHEMAS
            and 2 * int(map_config["receiver_primitive_quota"])
            > int(config["positions_per_bank"])
        )
        or type(map_config["primitive_shortlist_size"]) is not int
        or int(map_config["primitive_shortlist_size"]) < 2
        or type(map_config["reflection_candidate_limit"]) is not int
        or int(map_config["reflection_candidate_limit"])
        < int(map_config["receiver_primitive_quota"])
        or not 0.0 < float(map_config["minimum_reflection_incidence_cosine"]) <= 1.0
    ):
        raise ValueError("candidate receiver sampling config is invalid")
    if schema in CONTROLLED_CONFIG_SCHEMAS:
        _validate_controlled_interventions(config)
    if schema == CONFIG_SCHEMA_V6:
        _validate_post_rt_exclusions(config)
    renderer = config["renderer"]
    if renderer != {
        "mitsuba_variant": SIONNA_MITSUBA_VARIANT,
        "drjit_threads": SIONNA_DRJIT_THREADS,
        "verification_workers": 8,
    }:
        raise ValueError("formal Sionna renderer must use frozen single-thread LLVM workers")
    return config


def _validate_controlled_interventions(config: dict) -> None:
    intervention = config["interventions"]
    required = {
        "schema_version",
        "primitive_count",
        "geometry",
        "width_m",
        "depth_m",
        "height_m",
        "center_separation_m",
        "state_materials",
        "joint_reflection_quota",
        "minimum_exact_positions_per_primitive",
        "minimum_selected_quality_mean_per_primitive",
        "wrong_action_distance_relative_max",
    }
    if not isinstance(intervention, dict) or set(intervention) != required:
        raise ValueError("V5 controlled-intervention fields must be exact")
    if (
        intervention["schema_version"] != "csi-pairs-controlled-interventions-v1"
        or intervention["primitive_count"] != 4
        or intervention["geometry"]
        != "four_equal_axis_aligned_cuboids_two_orthogonal_pairs"
        or intervention["state_materials"] != ["concrete", "glass"]
        or not all(
            np.isfinite(float(intervention[name])) and float(intervention[name]) > 0.0
            for name in ("width_m", "depth_m", "height_m", "center_separation_m")
        )
        or float(intervention["center_separation_m"])
        <= float(intervention["width_m"])
        or type(intervention["joint_reflection_quota"]) is not int
        or not 2 <= int(intervention["joint_reflection_quota"]) < int(config["positions_per_bank"])
        or type(intervention["minimum_exact_positions_per_primitive"]) is not int
        or not 1
        <= int(intervention["minimum_exact_positions_per_primitive"])
        <= int(intervention["joint_reflection_quota"])
        or not np.isfinite(
            float(intervention["minimum_selected_quality_mean_per_primitive"])
        )
        or not 0.0
        < float(intervention["minimum_selected_quality_mean_per_primitive"])
        <= 1.0
        or not 0.0 < float(intervention["wrong_action_distance_relative_max"]) <= 0.05
        or _signed_wrong_action_coverage(int(intervention["primitive_count"])) < 0.8
    ):
        raise ValueError("V5 controlled-intervention design is invalid")


def _validate_post_rt_exclusions(config: dict) -> None:
    payload = config["post_rt_exclusions"]
    required = {
        "schema_version",
        "selection_policy",
        "failed_gate",
        "threshold",
        "cities",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("V6 post-RT exclusion fields must be exact")
    if (
        payload["schema_version"]
        != "csi-pairs-v5-post-rt-candidate-exclusions-config-v1"
        or payload["selection_policy"]
        != "exclude-bound-method-selection-rt-failed-bs-centers-v1"
        or payload["failed_gate"] != "minimum_active_branching_fraction"
        or float(payload["threshold"]) != 0.8
        or not isinstance(payload["cities"], list)
    ):
        raise ValueError("V6 post-RT exclusion policy is invalid")
    source_city_ids = {
        str(city["city_id"])
        for city in config["cities"]
        if city["split_group"] == "source"
    }
    rows = payload["cities"]
    if {str(row.get("city_id")) for row in rows if isinstance(row, dict)} != source_city_ids:
        raise ValueError("V6 post-RT exclusions must cover both source cities exactly")
    all_centers = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "city_id",
            "exclusion_manifest_sha256",
            "excluded_bs_utm_xy_m",
        }:
            raise ValueError("V6 post-RT exclusion city fields must be exact")
        digest = row["exclusion_manifest_sha256"]
        centers = row["excluded_bs_utm_xy_m"]
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not isinstance(centers, list)
            or not centers
        ):
            raise ValueError("V6 post-RT exclusion evidence binding is invalid")
        for center in centers:
            if (
                not isinstance(center, list)
                or len(center) != 2
                or any(type(value) not in (int, float) for value in center)
                or any(not np.isfinite(float(value)) for value in center)
                or any(float(value) != round(float(value), 6) for value in center)
            ):
                raise ValueError("V6 post-RT exclusion center is invalid")
            all_centers.append(tuple(float(value) for value in center))
    if len(set(all_centers)) != len(all_centers):
        raise ValueError("V6 post-RT exclusion centers must be globally unique")


def _validate_power_design(config: dict) -> None:
    design = config["power_design"]
    required = {
        "schema_version",
        "independent_unit",
        "test",
        "familywise_alpha",
        "holm_family_size_upper_bound",
        "worst_case_bonferroni_alpha",
        "assumed_correct_direction_probability",
        "required_power",
        "minimum_g3_scope_clusters",
        "exact_power_at_minimum",
        "minimum_source_role_clusters",
        "minimum_external_validation_clusters",
        "insufficient_clusters_policy",
    }
    if not isinstance(design, dict) or set(design) != required:
        raise ValueError("V4 power-design fields must be exact")
    if (
        design["schema_version"] != "csi-pairs-bank-power-design-v1"
        or design["independent_unit"] != "canonical_base_map_cluster"
        or design["test"] != "exact_two_sided_sign_test_worst_case_bonferroni"
        or float(design["familywise_alpha"]) != 0.05
        or int(design["holm_family_size_upper_bound"]) != 31
        or float(design["assumed_correct_direction_probability"]) != 0.8
        or float(design["required_power"]) != 0.8
        or int(design["minimum_g3_scope_clusters"]) != 41
        or int(design["minimum_source_role_clusters"]) != 8
        or int(design["minimum_external_validation_clusters"]) != 32
        or design["insufficient_clusters_policy"]
        != "downgrade_claim_no_position_repeat_or_seed_pseudoreplication"
    ):
        raise ValueError("V4 power-design registration differs from the frozen protocol")
    adjusted_alpha = float(design["familywise_alpha"]) / int(
        design["holm_family_size_upper_bound"]
    )
    if not math.isclose(
        float(design["worst_case_bonferroni_alpha"]),
        adjusted_alpha,
        rel_tol=0.0,
        abs_tol=1e-18,
    ):
        raise ValueError("V4 power-design Bonferroni alpha is inconsistent")
    minimum = int(design["minimum_g3_scope_clusters"])
    probability = float(design["assumed_correct_direction_probability"])
    power = exact_two_sided_sign_test_power(minimum, probability, adjusted_alpha)
    previous_power = exact_two_sided_sign_test_power(
        minimum - 1, probability, adjusted_alpha
    )
    if (
        not math.isclose(
            float(design["exact_power_at_minimum"]),
            power,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or power < float(design["required_power"])
        or previous_power >= float(design["required_power"])
    ):
        raise ValueError("V4 minimum G3 cluster count does not match the exact power design")

    source_cities = [city for city in config["cities"] if city["split_group"] == "source"]
    target_cities = [city for city in config["cities"] if city["split_group"] == "target"]
    external_cities = [city for city in config["cities"] if city["split_group"] == "external"]
    role_totals = {
        role: sum(int(city["source_role_bank_counts"][role]) for city in source_cities)
        for role in SOURCE_ROLES
    }
    if any(
        role_totals[role] < int(design["minimum_source_role_clusters"])
        for role in SOURCE_ROLES
    ):
        raise ValueError("V4 source roles do not meet the independent-cluster minimum")
    if role_totals["source_final_unseen_bank"] < minimum:
        raise ValueError("V4 source-final scope does not meet the G3 power minimum")
    if any(int(city["bank_count"]) < minimum for city in target_cities):
        raise ValueError("V4 target-city scope does not meet the G3 power minimum")
    if sum(int(city["bank_count"]) for city in external_cities) < int(
        design["minimum_external_validation_clusters"]
    ):
        raise ValueError("V4 external scope does not meet the G8 cluster minimum")


def exact_two_sided_sign_test_power(
    cluster_count: int,
    correct_direction_probability: float,
    alpha: float,
) -> float:
    count = int(cluster_count)
    probability = float(correct_direction_probability)
    threshold = float(alpha)
    if count < 1 or not 0.5 < probability < 1.0 or not 0.0 < threshold < 1.0:
        raise ValueError("sign-test power inputs are invalid")
    rejected = []
    denominator = 2**count
    for successes in range(count + 1):
        tail_start = max(successes, count - successes)
        p_value = min(
            1.0,
            2.0
            * sum(math.comb(count, value) for value in range(tail_start, count + 1))
            / denominator,
        )
        if p_value <= threshold:
            rejected.append(successes)
    return float(
        sum(
            math.comb(count, successes)
            * probability**successes
            * (1.0 - probability) ** (count - successes)
            for successes in rejected
        )
    )


def candidate_power_report(config: dict) -> dict[str, object]:
    ledger = expected_scene_ledger(config)
    source_role_counts = {
        role: sum(row["role"] == role for row in ledger) for role in SOURCE_ROLES
    }
    target_city_counts = {
        city: sum(row["role"] == "target" and row["city_id"] == city for row in ledger)
        for city in sorted({str(row["city_id"]) for row in ledger if row["role"] == "target"})
    }
    external_count = sum(row["role"] == "external_validation" for row in ledger)
    thresholds = {
        "minimum_source_role_clusters": 8,
        "minimum_g3_scope_clusters": 41,
        "minimum_external_validation_clusters": 32,
    }
    passed = bool(
        all(value >= thresholds["minimum_source_role_clusters"] for value in source_role_counts.values())
        and source_role_counts["source_final_unseen_bank"]
        >= thresholds["minimum_g3_scope_clusters"]
        and target_city_counts
        and all(
            value >= thresholds["minimum_g3_scope_clusters"]
            for value in target_city_counts.values()
        )
        and external_count >= thresholds["minimum_external_validation_clusters"]
    )
    return {
        "passed": passed,
        "scene_count": len(ledger),
        "source_role_counts": source_role_counts,
        "target_city_counts": target_city_counts,
        "external_validation_count": external_count,
        "thresholds": thresholds,
    }


def expected_scene_ledger(config: dict) -> list[dict[str, object]]:
    by_group: dict[str, list[dict]] = {"source": [], "target": [], "external": []}
    for city in config["cities"]:
        by_group[city["split_group"]].append(city)
    rows: list[dict[str, object]] = []
    source_offsets = {str(city["city_id"]): 0 for city in by_group["source"]}
    for role_index, role in enumerate(SOURCE_ROLES):
        for city in by_group["source"]:
            count = int(city.get("source_role_bank_counts", {}).get(role, 1))
            city_id = str(city["city_id"])
            for _ in range(count):
                bank_index = source_offsets[city_id]
                rows.append(_ledger_row(city, bank_index, role))
                source_offsets[city_id] += 1
    for city in by_group["target"]:
        for bank_index in range(int(city["bank_count"])):
            rows.append(_ledger_row(city, bank_index, "target"))
    for city in by_group["external"]:
        for bank_index in range(int(city["bank_count"])):
            rows.append(_ledger_row(city, bank_index, "external_validation"))
    configured_count = sum(int(city["bank_count"]) for city in config["cities"])
    if len(rows) != configured_count:
        raise RuntimeError("formal candidate ledger does not match configured bank counts")
    for scene_index, row in enumerate(rows):
        row["scene_index"] = scene_index
    return rows


def _ledger_row(city: dict, bank_index: int, role: str) -> dict[str, object]:
    city_id = str(city["city_id"])
    scene_id = f"osm-sionna-{city_id}-bank-{bank_index:02d}"
    return {
        "city_id": city_id,
        "bank_index_within_city": int(bank_index),
        "role": role,
        "scene_id": scene_id,
        "bank_id": scene_id.replace("scene", "bank"),
        "base_map_cluster_id": f"osm-foundation-{city_id}-{bank_index:02d}",
    }


def prepare_assets(
    config_path: str | Path,
    output_root: str | Path,
    raw_cache: str | Path | None = None,
) -> Path:
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_snapshot = _write_json_exclusive(output / "generator_config.json", config)
    raw_root = output / "raw_osm"
    raw_root.mkdir()
    bank_root = output / "banks"
    bank_root.mkdir()
    ledger = expected_scene_ledger(config)
    ledger_by_city = {
        str(city["city_id"]): sorted(
            (row for row in ledger if row["city_id"] == city["city_id"]),
            key=lambda row: int(row["bank_index_within_city"]),
        )
        for city in config["cities"]
    }
    frozen_sources = []
    for city in config["cities"]:
        raw_path, query, response, endpoint = _download_city_osm(
            city, config, raw_root, raw_cache
        )
        frozen_sources.append(
            (city, raw_path, query, endpoint, response)
        )

    tasks = [
        (
            city,
            config,
            str(raw_path),
            query,
            ledger_by_city[str(city["city_id"])],
        )
        for city, raw_path, query, _endpoint, _response in frozen_sources
    ]
    selected_by_city = {}
    with _asset_selection_executor(len(tasks)) as executor:
        futures = [executor.submit(_select_city_assets_task, task) for task in tasks]
        for future in as_completed(futures):
            city_id, selected, selection_audit, parsed_count, osm_timestamp = future.result()
            selected_by_city[city_id] = (
                selected,
                selection_audit,
                parsed_count,
                osm_timestamp,
            )
            print(
                canonical_json(
                    {
                        "event": "asset_city_complete",
                        "city_id": city_id,
                        "selected_bank_count": len(selected),
                    }
                ),
                flush=True,
            )
    if set(selected_by_city) != {str(city["city_id"]) for city in config["cities"]}:
        raise RuntimeError("parallel asset selection did not return every configured city")

    city_banks: dict[str, list[dict]] = {}
    raw_sources = []
    for city, raw_path, query, endpoint, _response in frozen_sources:
        city_id = str(city["city_id"])
        selected, selection_audit, parsed_count, osm_timestamp = selected_by_city[city_id]
        city_banks[city_id] = selected
        raw_sources.append(
            {
                "city_id": city_id,
                "path": str(raw_path.relative_to(output)),
                "bytes": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
                "receipt_path": str(
                    raw_path.with_suffix(".receipt.json").relative_to(output)
                ),
                "receipt_sha256": sha256_file(raw_path.with_suffix(".receipt.json")),
                "overpass_query": query,
                "overpass_endpoint": endpoint,
                "osm_base_timestamp": osm_timestamp,
                "parsed_closed_building_way_count": parsed_count,
                "bank_selection": selection_audit,
                "license_id": "ODbL-1.0",
            }
        )
    bank_rows = []
    for scene_index, row in enumerate(ledger):
        record = city_banks[str(row["city_id"])][int(row["bank_index_within_city"])]
        if any(record.get(key) != value for key, value in row.items()):
            raise RuntimeError("registered bank no longer matches the formal scene ledger")
        if int(record.get("scene_index", -1)) != scene_index:
            raise RuntimeError("registered bank scene index no longer matches ledger order")
        bank_dir = bank_root / str(row["scene_id"])
        bank_dir.mkdir()
        bank_json = _write_json_exclusive(bank_dir / "bank.json", record)
        _write_scene_assets(record, bank_dir)
        files = []
        for path in sorted(candidate for candidate in bank_dir.rglob("*") if candidate.is_file()):
            files.append(
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "license_id": "ODbL-1.0" if path == bank_json else "ODbL-1.0-DERIVED",
                }
            )
        bank_rows.append(
            {
                **row,
                "bank_record_path": str(bank_json.relative_to(output)),
                "bank_record_sha256": sha256_file(bank_json),
                "scene_xml_path": str((bank_dir / "scene.xml").relative_to(output)),
                "scene_xml_sha256": sha256_file(bank_dir / "scene.xml"),
                "files": files,
            }
        )
    attribution = _write_text_exclusive(
        output / "ATTRIBUTION.md",
        "# Asset attribution\n\n"
        "Building footprints and tags are from OpenStreetMap contributors and are used under "
        "the Open Data Commons Open Database License 1.0 (ODbL-1.0). "
        "https://www.openstreetmap.org/copyright\n\n"
        "Scene meshes are generated derivatives. No real RF measurements are included.\n",
    )
    manifest = {
        "schema_version": ASSET_SCHEMA,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_snapshot.relative_to(output)),
        "config_sha256": sha256_file(config_snapshot),
        "source_config_sha256": sha256_file(config_file),
        "licenses": [
            {
                "license_id": "ODbL-1.0",
                "url": "https://opendatacommons.org/licenses/odbl/1-0/",
                "attribution_url": "https://www.openstreetmap.org/copyright",
                "redistribution_allowed_with_terms": True,
            },
            {
                "license_id": "Apache-2.0",
                "url": "https://www.apache.org/licenses/LICENSE-2.0",
                "scope": "Sionna and Sionna RT engine source",
                "redistribution_allowed_with_terms": True,
            },
        ],
        "attribution_path": str(attribution.relative_to(output)),
        "attribution_sha256": sha256_file(attribution),
        "raw_sources": raw_sources,
        "banks": bank_rows,
    }
    manifest_path = _write_json_exclusive(output / "asset_manifest.json", manifest)
    return manifest_path


def _select_city_assets_task(payload):
    city, config, raw_path_value, query, ledger_rows = payload
    raw_path = Path(raw_path_value)
    response = _read_json(raw_path)
    if not isinstance(response, dict) or not isinstance(response.get("elements"), list):
        raise RuntimeError(f"frozen OSM response is malformed for {city['city_id']}")
    polygons = _parse_osm_buildings(response, city)
    selected, selection_audit = _select_city_banks(
        city,
        config,
        polygons,
        raw_path,
        query,
        ledger_rows,
    )
    return (
        str(city["city_id"]),
        selected,
        selection_audit,
        len(polygons),
        response.get("osm3s", {}).get("timestamp_osm_base", "unknown"),
    )


def _asset_selection_executor(workers: int):
    count = int(workers)
    if count < 1:
        raise ValueError("asset selection requires at least one city worker")
    return ProcessPoolExecutor(
        max_workers=count,
        mp_context=multiprocessing.get_context("spawn"),
    )


def _download_city_osm(
    city: dict,
    config: dict,
    raw_root: Path,
    raw_cache: str | Path | None = None,
) -> tuple[Path, str, dict, str]:
    south = float(city["center_lat"]) - float(city["query_delta_lat"])
    north = float(city["center_lat"]) + float(city["query_delta_lat"])
    west = float(city["center_lon"]) - float(city["query_delta_lon"])
    east = float(city["center_lon"]) + float(city["query_delta_lon"])
    query = (
        f'[out:json][timeout:120];way["building"]'
        f"({south:.7f},{west:.7f},{north:.7f},{east:.7f});out geom tags;"
    )
    cached = None if raw_cache is None else Path(raw_cache).resolve() / f"{city['city_id']}.json"
    cache_manifest_path = None if cached is None else cached.parent.parent / "asset_manifest.json"
    cache_receipt_path = None if cached is None else cached.with_suffix(".receipt.json")
    cache_receipt = None
    if (
        cached is not None
        and cached.is_file()
        and not cached.is_symlink()
        and cache_manifest_path is not None
        and cache_manifest_path.is_file()
        and not cache_manifest_path.is_symlink()
    ):
        cache_manifest = _read_json(cache_manifest_path)
        if isinstance(cache_manifest, dict):
            cache_receipt = next(
                (
                    row
                    for row in cache_manifest.get("raw_sources", [])
                    if isinstance(row, dict)
                    and row.get("city_id") == city["city_id"]
                    and row.get("overpass_query") == query
                    and row.get("sha256") == sha256_file(cached)
                ),
                None,
            )
    if (
        cache_receipt is None
        and cached is not None
        and cached.is_file()
        and not cached.is_symlink()
        and cache_receipt_path is not None
        and cache_receipt_path.is_file()
        and not cache_receipt_path.is_symlink()
    ):
        candidate_receipt = _read_json(cache_receipt_path)
        if (
            isinstance(candidate_receipt, dict)
            and set(candidate_receipt)
            == {
                "schema_version",
                "status",
                "city_id",
                "overpass_query",
                "overpass_endpoint",
                "bytes",
                "sha256",
            }
            and candidate_receipt["schema_version"] == RAW_OSM_RECEIPT_SCHEMA
            and candidate_receipt["status"] == "PASS"
            and candidate_receipt["city_id"] == city["city_id"]
            and candidate_receipt["overpass_query"] == query
            and candidate_receipt["bytes"] == cached.stat().st_size
            and candidate_receipt["sha256"] == sha256_file(cached)
            and isinstance(candidate_receipt["overpass_endpoint"], str)
            and (
                candidate_receipt["overpass_endpoint"].startswith("https://")
                or candidate_receipt["overpass_endpoint"]
                == f"frozen-cache-sha256:{sha256_file(cached)}"
            )
        ):
            cache_receipt = candidate_receipt
    if cache_receipt is not None:
        raw_path = raw_root / cached.name
        shutil.copy2(cached, raw_path)
        payload = raw_path.read_bytes()
        parsed = json.loads(payload.decode("utf-8"), parse_constant=_reject_json_constant)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("elements"), list):
            raise RuntimeError(f"cached Overpass response is malformed for {city['city_id']}")
        cache_endpoint = f"frozen-cache-sha256:{sha256_file(cached)}"
        _write_json_atomic_exclusive(
            raw_path.with_suffix(".receipt.json"),
            {
                "schema_version": RAW_OSM_RECEIPT_SCHEMA,
                "status": "PASS",
                "city_id": city["city_id"],
                "overpass_query": query,
                "overpass_endpoint": cache_endpoint,
                "bytes": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
            },
        )
        return raw_path, query, parsed, cache_endpoint
    body = urlparse.urlencode({"data": query}).encode("ascii")
    endpoints = []
    for endpoint in (
        str(config["overpass_endpoint"]),
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass.nchc.org.tw/api/interpreter",
    ):
        if endpoint not in endpoints:
            endpoints.append(endpoint)
    last_error: Exception | None = None
    payload = b""
    selected_endpoint = ""
    for endpoint in endpoints:
        print(
            canonical_json(
                {
                    "city_id": str(city["city_id"]),
                    "endpoint": endpoint,
                    "event": "osm_download_started",
                    "timeout_seconds": 180,
                }
            ),
            flush=True,
        )
        request = urlrequest.Request(
            endpoint,
            data=body,
            headers={"User-Agent": "CSI-PAIRS-formal-audit/1.0 (research reproducibility)"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(request, timeout=180) as response:
                candidate_payload = response.read()
            candidate_parsed = json.loads(
                candidate_payload.decode("utf-8"), parse_constant=_reject_json_constant
            )
            if not isinstance(candidate_parsed, dict) or not isinstance(
                candidate_parsed.get("elements"), list
            ):
                raise RuntimeError("Overpass response is not a JSON object with elements")
            payload = candidate_payload
            parsed = candidate_parsed
            selected_endpoint = endpoint
            print(
                canonical_json(
                    {
                        "bytes": len(payload),
                        "city_id": str(city["city_id"]),
                        "endpoint": endpoint,
                        "event": "osm_download_complete",
                    }
                ),
                flush=True,
            )
            break
        except Exception as error:  # The successful endpoint is frozen in the asset manifest.
            last_error = error
            print(
                canonical_json(
                    {
                        "city_id": str(city["city_id"]),
                        "endpoint": endpoint,
                        "error_type": type(error).__name__,
                        "event": "osm_download_endpoint_failed",
                    }
                ),
                flush=True,
            )
    if not payload:
        raise RuntimeError(f"Overpass query failed for {city['city_id']}: {last_error}")
    raw_path = raw_root / f"{city['city_id']}.json"
    with raw_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _write_json_atomic_exclusive(
        raw_path.with_suffix(".receipt.json"),
        {
            "schema_version": RAW_OSM_RECEIPT_SCHEMA,
            "status": "PASS",
            "city_id": city["city_id"],
            "overpass_query": query,
            "overpass_endpoint": selected_endpoint,
            "bytes": raw_path.stat().st_size,
            "sha256": sha256_file(raw_path),
        },
    )
    return raw_path, query, parsed, selected_endpoint


def _parse_osm_buildings(response: dict, city: dict) -> list[dict]:
    from pyproj import CRS, Transformer
    from shapely.geometry import Polygon

    longitude = float(city["center_lon"])
    latitude = float(city["center_lat"])
    zone = int(math.floor((longitude + 180.0) / 6.0) + 1)
    epsg = (32600 if latitude >= 0 else 32700) + zone
    transformer = Transformer.from_crs("EPSG:4326", CRS.from_epsg(epsg), always_xy=True)
    rows = []
    for element in response["elements"]:
        if element.get("type") != "way" or not isinstance(element.get("geometry"), list):
            continue
        geometry = element["geometry"]
        if len(geometry) < 4:
            continue
        lon_lat = [(float(point["lon"]), float(point["lat"])) for point in geometry]
        if lon_lat[0] != lon_lat[-1]:
            continue
        x, y = transformer.transform(
            [value[0] for value in lon_lat], [value[1] for value in lon_lat]
        )
        polygon = Polygon(zip(x, y))
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.geom_type != "Polygon" or polygon.is_empty:
            continue
        tags = element.get("tags") if isinstance(element.get("tags"), dict) else {}
        rows.append(
            {
                "osm_type": "way",
                "osm_id": int(element["id"]),
                "tags": {str(key): str(value) for key, value in sorted(tags.items())},
                "polygon": polygon,
                "height_m": _building_height(tags, int(element["id"])),
                "utm_epsg": epsg,
            }
        )
    return rows


def _building_height(tags: dict, osm_id: int) -> float:
    for key in ("height", "building:height"):
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(tags.get(key, "")))
        if match:
            return float(np.clip(float(match.group()), 3.0, 120.0))
    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(tags.get("building:levels", ""))
    )
    if match:
        return float(np.clip(float(match.group()) * 3.2, 3.0, 120.0))
    bucket = int.from_bytes(hashlib.sha256(str(osm_id).encode("ascii")).digest()[:2], "big") % 7
    return float(9.0 + 3.0 * bucket)


def _select_city_banks(
    city: dict,
    config: dict,
    buildings: list[dict],
    raw_path: Path,
    query: str,
    ledger_rows: list[dict],
) -> tuple[list[dict], dict]:
    from pyproj import CRS, Transformer
    from shapely.geometry import Point, box
    from shapely.strtree import STRtree

    if not buildings:
        raise RuntimeError(f"no closed OSM building ways were parsed for {city['city_id']}")
    epsg = int(buildings[0]["utm_epsg"])
    transformer = Transformer.from_crs("EPSG:4326", CRS.from_epsg(epsg), always_xy=True)
    center_x, center_y = transformer.transform(float(city["center_lon"]), float(city["center_lat"]))
    map_config = config["map"]
    radius = int(map_config["candidate_grid_radius"])
    spacing = float(map_config["candidate_grid_spacing_m"])
    half_extent = float(map_config["size"]) * float(map_config["resolution_m"]) / 2.0
    margin = 2.0
    building_polygons = [row["polygon"] for row in buildings]
    building_tree = STRtree(building_polygons)
    placement_bounds = (
        _sampled_formal_placement_bounds(config)
        if config["schema_version"] in CONTROLLED_CONFIG_SCHEMAS
        else None
    )
    candidates = []
    for grid_y in range(-radius, radius + 1):
        for grid_x in range(-radius, radius + 1):
            nominal_x = center_x + grid_x * spacing
            nominal_y = center_y + grid_y * spacing
            try:
                bs_x, bs_y = _nearest_free_bs(
                    nominal_x,
                    nominal_y,
                    buildings,
                    building_tree,
                    config=(
                        config
                        if config["schema_version"] in CONTROLLED_CONFIG_SCHEMAS
                        else None
                    ),
                    placement_bounds=placement_bounds,
                )
            except RuntimeError:
                continue
            extent = box(
                bs_x - half_extent + margin,
                bs_y - half_extent + margin,
                bs_x + half_extent - margin,
                bs_y + half_extent - margin,
            )
            selected = [
                buildings[int(index)]
                for index in building_tree.query(extent, predicate="contains")
                if float(buildings[int(index)]["polygon"].area)
                >= float(map_config["minimum_building_area_m2"])
            ]
            if len(selected) < int(map_config["minimum_buildings_per_bank"]):
                continue
            selected.sort(key=lambda row: (-float(row["polygon"].area), int(row["osm_id"])))
            score = sum(min(float(value["polygon"].area), 2500.0) for value in selected)
            center_clearance = min(
                float(value["polygon"].distance(Point(bs_x, bs_y))) for value in selected
            )
            candidates.append((score + 20.0 * len(selected), center_clearance, bs_x, bs_y, selected))
    policy = map_config.get(
        "candidate_qualification_order", "score-descending-v1"
    )
    candidates = _order_bank_candidates(candidates, policy)
    original_candidates = candidates
    (
        candidates,
        kept_candidate_ranks,
        excluded_candidate_rows,
        city_exclusions,
    ) = _apply_post_rt_candidate_exclusions(
        config, str(city["city_id"]), candidates
    )
    required = int(city["bank_count"])
    if len(ledger_rows) != required:
        raise RuntimeError(
            f"{city['city_id']} ledger has {len(ledger_rows)} rows; {required} are required"
        )
    for bank_index, ledger_row in enumerate(ledger_rows):
        if (
            ledger_row.get("city_id") != city["city_id"]
            or ledger_row.get("bank_index_within_city") != bank_index
            or type(ledger_row.get("scene_index")) is not int
        ):
            raise RuntimeError(f"{city['city_id']} scene ledger is not sequential and exact")
    if len(candidates) < required:
        raise RuntimeError(
            f"{city['city_id']} has only {len(candidates)} static-geometry candidates; "
            f"{required} are required"
        )

    def qualify(candidate: tuple, bank_index: int) -> dict:
        _, clearance, bs_x, bs_y, selected = candidate
        building_rows = []
        for building in selected:
            polygon = building["polygon"]
            coordinates = [
                [round(float(x - bs_x), 6), round(float(y - bs_y), 6)]
                for x, y in polygon.exterior.coords
            ]
            building_rows.append(
                {
                    "osm_type": building["osm_type"],
                    "osm_id": int(building["osm_id"]),
                    "height_m": round(float(building["height_m"]), 6),
                    "height_source": _height_source(building["tags"]),
                    "local_exterior_xy_m": coordinates,
                    "is_registered_primitive": False,
                    "tags": building["tags"],
                }
            )
        source = {
            "city_id": city["city_id"],
            "bank_index_within_city": bank_index,
            "utm_epsg": epsg,
            "bs_utm_xy_m": [round(float(bs_x), 6), round(float(bs_y), 6)],
            "bs_center_clearance_m": round(clearance, 6),
            "flat_ground_elevation_m": 0.0,
            "building_count": len(building_rows),
            "buildings": building_rows,
            "source": {
                "kind": "OpenStreetMap Overpass building footprints",
                "raw_path_name": raw_path.name,
                "raw_sha256": sha256_file(raw_path),
                "query_sha256": _sha256_bytes(query.encode("ascii")),
                "license_id": "ODbL-1.0",
                "attribution_url": "https://www.openstreetmap.org/copyright",
            },
        }
        ledger_row = ledger_rows[bank_index]
        record = {**source, **ledger_row, "schema_version": BANK_SCHEMA}
        register = (
            _register_controlled_intervention_geometry
            if config["schema_version"] in CONTROLLED_CONFIG_SCHEMAS
            else _register_reflection_geometry
        )
        return register(record, config, scene_index=int(ledger_row["scene_index"]))

    rows, audit = _select_reflection_qualified_bank_candidates(
        candidates,
        required,
        map_extent_m=2.0 * half_extent,
        qualify=qualify,
    )
    if excluded_candidate_rows:
        filtered_examined = int(audit["candidates_examined"])
        if not 1 <= filtered_examined <= len(kept_candidate_ranks):
            raise RuntimeError("post-RT filtered candidate audit is invalid")
        last_original_rank = kept_candidate_ranks[filtered_examined - 1]
        if any(
            int(row["candidate_rank"]) > last_original_rank
            for row in excluded_candidate_rows
        ):
            raise RuntimeError(
                "post-RT exclusion was not encountered before selection completed"
            )
        for rejection in audit.get("rejected_reflection_geometry_candidates", []):
            rejection["candidate_rank"] = kept_candidate_ranks[
                int(rejection["candidate_rank"])
            ]
        audit["candidate_count"] = len(original_candidates)
        audit["candidates_examined"] = last_original_rank + 1
        audit["post_rt_exclusion_count"] = len(excluded_candidate_rows)
        audit["post_rt_excluded_candidates"] = excluded_candidate_rows
        audit["post_rt_exclusion_manifest_sha256"] = city_exclusions[
            "exclusion_manifest_sha256"
        ]
        audit["selected_candidate_records"] = [
            {
                "bank_index_within_city": int(record["bank_index_within_city"]),
                "candidate_rank": next(
                    rank
                    for rank, candidate in enumerate(original_candidates)
                    if (
                        round(float(candidate[2]), 6),
                        round(float(candidate[3]), 6),
                    )
                    == tuple(float(value) for value in record["bs_utm_xy_m"])
                ),
                "bs_utm_xy_m": record["bs_utm_xy_m"],
                "scene_index": int(record["scene_index"]),
            }
            for record in rows
        ]
    audit.update(
        {
            "city_id": str(city["city_id"]),
            "algorithm": (
                "sparse-osm-first-placement-aware-greedy-nonoverlap-with-reflection-qualification-v1"
                if config["schema_version"] in CONTROLLED_CONFIG_SCHEMAS
                else "score-ranked-greedy-nonoverlap-with-reflection-qualification-v1"
            ),
            "map_extent_m": 2.0 * half_extent,
            "candidate_qualification_order": map_config.get(
                "candidate_qualification_order", "score-descending-v1"
            ),
            "bs_selection": map_config.get("bs_selection", "first-free-v1"),
        }
    )
    return rows, audit


def _order_bank_candidates(candidates: list[tuple], policy: str) -> list[tuple]:
    if policy == "score-descending-v1":
        key = lambda row: (-float(row[0]), -float(row[1]), float(row[2]), float(row[3]))
    elif policy == "sparse-osm-first-v1":
        key = lambda row: (
            len(row[4]),
            float(row[0]),
            -float(row[1]),
            float(row[2]),
            float(row[3]),
        )
    else:
        raise ValueError(f"unsupported bank candidate qualification order: {policy}")
    return sorted(candidates, key=key)


def _apply_post_rt_candidate_exclusions(
    config: dict, city_id: str, candidates: list[tuple]
) -> tuple[list[tuple], list[int], list[dict], dict | None]:
    ranks = list(range(len(candidates)))
    if config["schema_version"] != CONFIG_SCHEMA_V6:
        return candidates, ranks, [], None
    city_exclusions = next(
        (
            row
            for row in config["post_rt_exclusions"]["cities"]
            if row["city_id"] == city_id
        ),
        None,
    )
    if city_exclusions is None:
        return candidates, ranks, [], None
    excluded_centers = {
        tuple(round(float(value), 6) for value in center)
        for center in city_exclusions["excluded_bs_utm_xy_m"]
    }
    center_to_rank = {
        (round(float(row[2]), 6), round(float(row[3]), 6)): rank
        for rank, row in enumerate(candidates)
    }
    if len(center_to_rank) != len(candidates):
        raise RuntimeError("ordered asset candidate centers are not unique")
    missing = sorted(excluded_centers.difference(center_to_rank))
    if missing:
        raise RuntimeError(f"post-RT exclusion centers are absent for {city_id}: {missing}")
    excluded_ranks = {center_to_rank[center] for center in excluded_centers}
    kept_ranks = [rank for rank in ranks if rank not in excluded_ranks]
    filtered = [candidates[rank] for rank in kept_ranks]
    excluded_rows = [
        {
            "candidate_rank": rank,
            "bs_utm_xy_m": [
                round(float(candidates[rank][2]), 6),
                round(float(candidates[rank][3]), 6),
            ],
            "reason": "bound_post_rt_physical_qualification_failure",
        }
        for rank in sorted(excluded_ranks)
    ]
    return filtered, kept_ranks, excluded_rows, city_exclusions


def _select_reflection_qualified_bank_candidates(
    candidates: list[tuple],
    required: int,
    *,
    map_extent_m: float,
    qualify,
) -> tuple[list[object], dict]:
    if required < 1:
        raise ValueError("at least one reflection-qualified bank is required")
    chosen_candidates = []
    chosen_records = []
    overlap_rejections = 0
    reflection_rejections = []
    reason_counts: dict[str, int] = {}
    candidates_examined = 0
    for candidate_rank, candidate in enumerate(candidates):
        candidates_examined += 1
        candidate_x = float(candidate[2])
        candidate_y = float(candidate[3])
        overlaps = any(
            abs(candidate_x - float(existing[2])) < map_extent_m
            and abs(candidate_y - float(existing[3])) < map_extent_m
            for existing in chosen_candidates
        )
        if overlaps:
            overlap_rejections += 1
            if candidates_examined % 10 == 0:
                print(
                    canonical_json(
                        {
                            "event": "asset_candidate_progress",
                            "candidate_rank": candidate_rank,
                            "candidates_examined": candidates_examined,
                            "selected": len(chosen_records),
                            "required": required,
                            "overlap_rejections": overlap_rejections,
                            "reflection_rejections": len(reflection_rejections),
                        }
                    ),
                    flush=True,
                )
            continue
        bank_index = len(chosen_records)
        try:
            record = qualify(candidate, bank_index)
        except ReflectionGeometryQualificationError as error:
            reason_counts[error.reason_code] = reason_counts.get(error.reason_code, 0) + 1
            reflection_rejections.append(
                {
                    "candidate_rank": candidate_rank,
                    "candidate_score": round(float(candidate[0]), 9),
                    "bs_utm_xy_m": [round(candidate_x, 6), round(candidate_y, 6)],
                    "assigned_bank_index_within_city": bank_index,
                    "reason": error.reason_code,
                    "detail": str(error),
                }
            )
            print(
                canonical_json(
                    {
                        "event": "asset_candidate_rejected",
                        "candidate_rank": candidate_rank,
                        "candidates_examined": candidates_examined,
                        "selected": len(chosen_records),
                        "required": required,
                        "reason": error.reason_code,
                    }
                ),
                flush=True,
            )
            continue
        chosen_candidates.append(candidate)
        chosen_records.append(record)
        print(
            canonical_json(
                {
                    "event": "asset_candidate_selected",
                    "candidate_rank": candidate_rank,
                    "candidates_examined": candidates_examined,
                    "selected": len(chosen_records),
                    "required": required,
                }
            ),
            flush=True,
        )
        if len(chosen_records) == required:
            return chosen_records, {
                "candidate_count": len(candidates),
                "candidates_examined": candidates_examined,
                "required_bank_count": required,
                "selected_bank_count": len(chosen_records),
                "rejected_overlap_count": overlap_rejections,
                "rejected_reflection_geometry_count": len(reflection_rejections),
                "rejected_reflection_geometry_reason_counts": dict(
                    sorted(reason_counts.items())
                ),
                "rejected_reflection_geometry_candidates": reflection_rejections,
            }
    reason_summary = ", ".join(
        f"{name}={count}" for name, count in sorted(reason_counts.items())
    ) or "none"
    raise RuntimeError(
        f"only {len(chosen_records)} disjoint reflection-qualified bank extents are "
        f"available; {required} are required; overlap_rejections={overlap_rejections}; "
        f"reflection_rejections={len(reflection_rejections)} ({reason_summary})"
    )


def _select_nonoverlapping_bank_candidates(
    candidates: list[tuple],
    required: int,
    *,
    map_extent_m: float,
) -> list[tuple]:
    chosen = []
    for candidate in candidates:
        candidate_x = float(candidate[2])
        candidate_y = float(candidate[3])
        overlaps = any(
            abs(candidate_x - float(existing[2])) < map_extent_m
            and abs(candidate_y - float(existing[3])) < map_extent_m
            for existing in chosen
        )
        if overlaps:
            continue
        chosen.append(candidate)
        if len(chosen) == required:
            return chosen
    raise RuntimeError(
        f"only {len(chosen)} disjoint bank extents are available; {required} are required"
    )


def _sampled_formal_placement_bounds(config: dict) -> np.ndarray:
    bounds = np.asarray(
        [
            [
                (
                    min(point[0] for point in primitive["local_exterior_xy_m"]),
                    min(point[1] for point in primitive["local_exterior_xy_m"]),
                    max(point[0] for point in primitive["local_exterior_xy_m"]),
                    max(point[1] for point in primitive["local_exterior_xy_m"]),
                )
                for primitive in placement
            ]
            for placement in _controlled_primitive_rows(config, 0)
        ],
        dtype=np.float64,
    )
    if bounds.ndim != 3 or bounds.shape[1:] != (4, 4):
        raise RuntimeError("formal controlled-placement inventory is malformed")
    count = int(config["map"]["bs_placement_sample_count"])
    stride = max(1, int(math.ceil(bounds.shape[0] / count)))
    return bounds[::stride][:count]


def _nearest_free_bs(
    x: float,
    y: float,
    buildings: list[dict],
    tree=None,
    *,
    config: dict | None = None,
    placement_bounds: np.ndarray | None = None,
) -> tuple[float, float]:
    from shapely import box as vector_box
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    spatial_index = tree or STRtree([row["polygon"] for row in buildings])

    offsets = [(0.0, 0.0)]
    for radius in (10.0, 20.0, 30.0, 40.0, 50.0):
        offsets.extend(
            (radius * math.cos(angle), radius * math.sin(angle))
            for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        )
    sampled_bounds = None
    if config is not None:
        if (
            config.get("schema_version") not in CONTROLLED_CONFIG_SCHEMAS
            or config.get("map", {}).get("bs_selection")
            != "sampled-formal-placement-capacity-v1"
        ):
            raise ValueError("placement-aware BS selection requires the frozen V5 policy")
        sampled_bounds = (
            _sampled_formal_placement_bounds(config)
            if placement_bounds is None
            else np.asarray(placement_bounds, dtype=np.float64)
        )
        if sampled_bounds.ndim != 3 or sampled_bounds.shape[1:] != (4, 4):
            raise ValueError("sampled controlled-placement bounds are malformed")
    scored = []
    for offset_index, (dx, dy) in enumerate(offsets):
        bs_x = float(x + dx)
        bs_y = float(y + dy)
        point = Point(bs_x, bs_y)
        nearby = spatial_index.query(point, predicate="dwithin", distance=4.0)
        if all(
            float(buildings[int(index)]["polygon"].distance(point)) >= 4.0
            for index in nearby
        ):
            if sampled_bounds is None:
                return bs_x, bs_y
            flattened = sampled_bounds.reshape(-1, 4)
            primitive_boxes = vector_box(
                flattened[:, 0] + bs_x - 2.0,
                flattened[:, 1] + bs_y - 2.0,
                flattened[:, 2] + bs_x + 2.0,
                flattened[:, 3] + bs_y + 2.0,
            )
            collision_pairs = spatial_index.query(
                primitive_boxes, predicate="intersects"
            )
            blocked = np.zeros(sampled_bounds.shape[0], dtype=np.bool_)
            if collision_pairs.size:
                blocked[np.unique(collision_pairs[0] // 4)] = True
            scored.append((int(np.sum(~blocked)), -offset_index, bs_x, bs_y))
    if scored:
        selected = max(scored)
        return float(selected[2]), float(selected[3])
    raise RuntimeError("could not place a transmitter with four-meter building clearance")


def _height_source(tags: dict) -> str:
    if tags.get("height") or tags.get("building:height"):
        return "osm_height_tag_clamped_3_to_120_m"
    if tags.get("building:levels"):
        return "osm_levels_times_3.2_m_clamped_3_to_120_m"
    return "deterministic_osm_id_hash_fallback_9_to_27_m"


def _geometry_catalog(bank: dict) -> dict[str, object]:
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient
    from shapely.strtree import STRtree

    rows = list(bank["buildings"])
    polygons = []
    for row in rows:
        polygon = orient(Polygon(row["local_exterior_xy_m"]), sign=1.0)
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 0.0:
            raise RuntimeError(f"invalid bank polygon for OSM way {row['osm_id']}")
        polygons.append(polygon)
    return {
        "rows": rows,
        "polygons": polygons,
        "heights": np.asarray([float(row["height_m"]) for row in rows], dtype=np.float64),
        "osm_ids": np.asarray([int(row["osm_id"]) for row in rows], dtype=np.int64),
        "tree": STRtree(polygons),
    }


def _extend_geometry_catalog(base: dict[str, object], rows: list[dict]) -> dict[str, object]:
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient
    from shapely.strtree import STRtree

    added_polygons = []
    for row in rows:
        polygon = orient(Polygon(row["local_exterior_xy_m"]), sign=1.0)
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 0.0:
            raise RuntimeError(f"invalid bank polygon for OSM way {row['osm_id']}")
        added_polygons.append(polygon)
    polygons = [*base["polygons"], *added_polygons]
    return {
        "rows": [*base["rows"], *rows],
        "polygons": polygons,
        "heights": np.concatenate(
            (
                np.asarray(base["heights"], dtype=np.float64),
                np.asarray([float(row["height_m"]) for row in rows], dtype=np.float64),
            )
        ),
        "osm_ids": np.concatenate(
            (
                np.asarray(base["osm_ids"], dtype=np.int64),
                np.asarray([int(row["osm_id"]) for row in rows], dtype=np.int64),
            )
        ),
        "tree": STRtree(polygons),
    }


def _pre_render_receiver_candidates(
    bank: dict,
    config: dict,
    scene_index: int,
    catalog: dict[str, object] | None = None,
) -> np.ndarray:
    from shapely import linestrings, points

    geometry = catalog or _geometry_catalog(bank)
    polygons = geometry["polygons"]
    heights = geometry["heights"]
    tree = geometry["tree"]
    map_config = config["map"]
    radio = config["radio"]
    offset_x = ((scene_index % 17) - 8) * 0.013
    offset_y = (((scene_index * 7) % 19) - 9) * 0.011
    spacing = float(map_config["receiver_grid_spacing_m"])
    half_extent = 0.5 * float(map_config["size"]) * float(map_config["resolution_m"])
    grid_extent = half_extent - 12.0
    values = np.arange(-grid_extent, grid_extent + 0.5 * spacing, spacing)
    minimum_distance = float(map_config["receiver_minimum_bs_distance_m"])
    clearance = float(map_config["receiver_building_clearance_m"])
    transmitter_z = float(radio["transmitter_z_m"])
    receiver_z = float(radio["receiver_z_m"])
    grid_x, grid_y = np.meshgrid(values + offset_x, values + offset_y, indexing="xy")
    candidates = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)
    candidates = candidates[
        np.linalg.norm(candidates, axis=1) >= minimum_distance
    ]
    point_geometries = points(candidates)
    clearance_pairs = tree.query(
        point_geometries, predicate="dwithin", distance=clearance
    )
    if clearance_pairs.size:
        clear = np.ones(candidates.shape[0], dtype=np.bool_)
        clear[np.unique(clearance_pairs[0])] = False
        candidates = candidates[clear]

    line_coordinates = np.stack(
        (np.zeros_like(candidates), candidates), axis=1
    )
    line_geometries = linestrings(line_coordinates)
    blocker_pairs = tree.query(line_geometries)
    blocker_rows: dict[int, list[int]] = {}
    if blocker_pairs.size:
        for line_index, polygon_index in blocker_pairs.T:
            blocker_rows.setdefault(int(line_index), []).append(int(polygon_index))
    visible = np.ones(candidates.shape[0], dtype=np.bool_)
    for line_index, polygon_indices in blocker_rows.items():
        buildings = [
            (polygons[index], float(heights[index])) for index in polygon_indices
        ]
        visible[line_index] = _has_direct_geometric_visibility(
            line_geometries[line_index],
            buildings,
            transmitter_z=transmitter_z,
            receiver_z=receiver_z,
            minimum_vertical_clearance_m=float(
                map_config.get(
                    "receiver_minimum_direct_path_vertical_clearance_m", 0.0
                )
            ),
        )
    candidates = candidates[visible]
    if len(candidates) < int(config["positions_per_bank"]):
        raise RuntimeError(
            f"bank {bank['scene_id']} has only {len(candidates)} pre-render 3D-LoS positions"
        )
    return np.asarray(candidates, dtype=np.float64)


def _approximate_reflection_qualities(
    polygon,
    height: float,
    receivers: np.ndarray,
    *,
    transmitter_z: float,
    receiver_z: float,
    minimum_incidence_cosine: float,
) -> np.ndarray:
    coordinates = np.asarray(polygon.exterior.coords, dtype=np.float64)
    transmitter = np.zeros(2, dtype=np.float64)
    qualities = np.zeros(receivers.shape[0], dtype=np.float64)
    for first, second in zip(coordinates[:-1], coordinates[1:]):
        direction = second - first
        squared_length = float(direction @ direction)
        if squared_length <= 1e-8:
            continue
        left_normal = np.asarray((-direction[1], direction[0])) / math.sqrt(squared_length)
        transmitter_side = float((transmitter - first) @ left_normal)
        receiver_side = (receivers - first) @ left_normal
        # CCW polygon interiors are on the left; both endpoints must see the exterior face.
        valid = (transmitter_side < -1e-6) & (receiver_side < -1e-6)
        image = transmitter - 2.0 * transmitter_side * left_normal
        denominator = (receivers - image) @ left_normal
        valid &= np.abs(denominator) > 1e-9
        fraction = np.zeros(receivers.shape[0], dtype=np.float64)
        fraction[valid] = -float((image - first) @ left_normal) / denominator[valid]
        valid &= (fraction > 1e-6) & (fraction < 1.0 - 1e-6)
        reflection = image + fraction[:, None] * (receivers - image)
        wall_fraction = ((reflection - first) @ direction) / squared_length
        reflection_z = transmitter_z + fraction * (receiver_z - transmitter_z)
        valid &= (
            (wall_fraction > 1e-5)
            & (wall_fraction < 1.0 - 1e-5)
            & (reflection_z > 1e-3)
            & (reflection_z < height - 1e-3)
        )
        transmitter_leg = np.linalg.norm(reflection, axis=1)
        receiver_leg = np.linalg.norm(receivers - reflection, axis=1)
        incidence = np.abs(
            np.sum(
                reflection
                / np.maximum(transmitter_leg[:, None], np.finfo(np.float64).tiny)
                * left_normal,
                axis=1,
            )
        )
        valid &= incidence >= minimum_incidence_cosine
        quality = np.where(
            valid,
            incidence / np.maximum(transmitter_leg + receiver_leg, np.finfo(np.float64).tiny),
            0.0,
        )
        qualities = np.maximum(qualities, quality)
    return qualities


def _intersection_parameters(line, geometry) -> list[float]:
    from shapely.geometry import Point

    if geometry.is_empty:
        return []
    if hasattr(geometry, "geoms"):
        values = []
        for child in geometry.geoms:
            values.extend(_intersection_parameters(line, child))
        return values
    if not hasattr(geometry, "coords"):
        return []
    return [
        float(line.project(Point(coordinate), normalized=True))
        for coordinate in geometry.coords
    ]


def _has_clear_3d_segment(
    start_xy: np.ndarray,
    start_z: float,
    end_xy: np.ndarray,
    end_z: float,
    catalog: dict[str, object],
    primitive_index: int,
) -> bool:
    from shapely.geometry import LineString

    line = LineString((start_xy, end_xy))
    polygons = catalog["polygons"]
    heights = catalog["heights"]
    for index_value in catalog["tree"].query(line):
        index = int(index_value)
        intersection = line.intersection(polygons[index])
        if intersection.is_empty:
            continue
        if index == primitive_index:
            if float(getattr(intersection, "length", 0.0)) > 1e-5:
                return False
            continue
        parameters = _intersection_parameters(line, intersection)
        if not parameters:
            return False
        minimum_ray_height = min(
            start_z + parameter * (end_z - start_z) for parameter in parameters
        )
        if float(heights[index]) >= minimum_ray_height - 1e-6:
            return False
    return True


def _visible_reflection_quality(
    receiver: np.ndarray,
    primitive_index: int,
    catalog: dict[str, object],
    *,
    transmitter_z: float,
    receiver_z: float,
    minimum_incidence_cosine: float,
) -> float:
    polygon = catalog["polygons"][primitive_index]
    height = float(catalog["heights"][primitive_index])
    coordinates = np.asarray(polygon.exterior.coords, dtype=np.float64)
    transmitter = np.zeros(2, dtype=np.float64)
    best = 0.0
    for first, second in zip(coordinates[:-1], coordinates[1:]):
        direction = second - first
        squared_length = float(direction @ direction)
        if squared_length <= 1e-8:
            continue
        left_normal = np.asarray((-direction[1], direction[0])) / math.sqrt(squared_length)
        transmitter_side = float((transmitter - first) @ left_normal)
        receiver_side = float((receiver - first) @ left_normal)
        if transmitter_side >= -1e-6 or receiver_side >= -1e-6:
            continue
        image = transmitter - 2.0 * transmitter_side * left_normal
        denominator = float((receiver - image) @ left_normal)
        if abs(denominator) <= 1e-9:
            continue
        fraction = -float((image - first) @ left_normal) / denominator
        if not 1e-6 < fraction < 1.0 - 1e-6:
            continue
        reflection = image + fraction * (receiver - image)
        wall_fraction = float((reflection - first) @ direction / squared_length)
        reflection_z = transmitter_z + fraction * (receiver_z - transmitter_z)
        if (
            wall_fraction <= 1e-5
            or wall_fraction >= 1.0 - 1e-5
            or reflection_z <= 1e-3
            or reflection_z >= height - 1e-3
        ):
            continue
        transmitter_leg = float(np.linalg.norm(reflection))
        receiver_leg = float(np.linalg.norm(receiver - reflection))
        incidence = abs(float((reflection / transmitter_leg) @ left_normal))
        if incidence < minimum_incidence_cosine:
            continue
        if not _has_clear_3d_segment(
            transmitter,
            transmitter_z,
            reflection,
            reflection_z,
            catalog,
            primitive_index,
        ):
            continue
        if not _has_clear_3d_segment(
            reflection,
            reflection_z,
            receiver,
            receiver_z,
            catalog,
            primitive_index,
        ):
            continue
        best = max(best, incidence / (transmitter_leg + receiver_leg))
    return best


def _rank_positive(values: np.ndarray) -> list[int]:
    return sorted(
        np.flatnonzero(values > 0.0).tolist(),
        key=lambda index: (-float(values[index]), int(index)),
    )


def _allocate_reflection_pair(
    first: np.ndarray,
    second: np.ndarray,
    quota: int,
) -> tuple[list[int], list[int], tuple[float, float]] | None:
    allocations = []
    for reverse in (False, True):
        leading, trailing = (second, first) if reverse else (first, second)
        leading_indices = _rank_positive(leading)[:quota]
        unavailable = set(leading_indices)
        trailing_indices = [
            index for index in _rank_positive(trailing) if index not in unavailable
        ][:quota]
        if len(leading_indices) != quota or len(trailing_indices) != quota:
            continue
        if reverse:
            first_indices, second_indices = trailing_indices, leading_indices
        else:
            first_indices, second_indices = leading_indices, trailing_indices
        sums = (
            float(np.sum(first[np.asarray(first_indices, dtype=np.int64)])),
            float(np.sum(second[np.asarray(second_indices, dtype=np.int64)])),
        )
        allocations.append((min(sums), sum(sums), first_indices, second_indices, sums))
    if not allocations:
        return None
    selected = max(allocations, key=lambda row: (row[0], row[1]))
    return selected[2], selected[3], selected[4]


def _controlled_primitive_rows(config: dict, scene_index: int):
    """Yield deterministic equal-geometry primitive placements before RT."""
    from shapely.geometry import box

    intervention = config["interventions"]
    width = float(intervention["width_m"])
    depth = float(intervention["depth_m"])
    height = float(intervention["height_m"])
    separation = float(intervention["center_separation_m"])
    primitive_count = int(intervention["primitive_count"])
    base_id = -1_000_000_000 - primitive_count * int(scene_index)
    if primitive_count != 4:
        raise ValueError("the controlled square-array layout requires four primitives")
    # Two orthogonal primitive pairs give every selected active receiver two
    # unobstructed first-order branches before any RT output is inspected.
    offsets = (
        0.0,
        -8.0,
        8.0,
        -16.0,
        16.0,
        -24.0,
        24.0,
        -32.0,
        32.0,
        -40.0,
        40.0,
        -48.0,
        48.0,
        -4.0,
        4.0,
        -12.0,
        12.0,
        -20.0,
        20.0,
        -28.0,
        28.0,
        -36.0,
        36.0,
        -44.0,
        44.0,
    )
    radial_values = (104.0, 96.0, 88.0, 80.0, 112.0, 120.0, 72.0, 64.0, 56.0, 48.0, 40.0)
    orientations = tuple(range(4))
    order = np.arange(len(offsets) * len(radial_values) * len(orientations))
    rng = np.random.default_rng(int(config["seed"]) + int(scene_index) * 7919)
    rng.shuffle(order)
    candidates = []
    for radial in radial_values:
        for offset in offsets:
            for orientation in orientations:
                pair = (-0.5 * separation, 0.5 * separation)
                if orientation == 0:  # east and north
                    centers = (
                        (radial, offset + pair[0]),
                        (radial, offset + pair[1]),
                        (offset + pair[0], radial),
                        (offset + pair[1], radial),
                    )
                elif orientation == 1:  # north and west
                    centers = (
                        (offset + pair[0], radial),
                        (offset + pair[1], radial),
                        (-radial, offset + pair[0]),
                        (-radial, offset + pair[1]),
                    )
                elif orientation == 2:  # west and south
                    centers = (
                        (-radial, offset + pair[0]),
                        (-radial, offset + pair[1]),
                        (offset + pair[0], -radial),
                        (offset + pair[1], -radial),
                    )
                else:  # south and east
                    centers = (
                        (offset + pair[0], -radial),
                        (offset + pair[1], -radial),
                        (radial, offset + pair[0]),
                        (radial, offset + pair[1]),
                    )
                sizes = (width, depth)
                rows = []
                for primitive, (center_x, center_y) in enumerate(centers):
                    size_x, size_y = sizes
                    polygon = box(
                        center_x - size_x / 2,
                        center_y - size_y / 2,
                        center_x + size_x / 2,
                        center_y + size_y / 2,
                    )
                    coordinates = [
                        [round(float(x), 6), round(float(y), 6)]
                        for x, y in polygon.exterior.coords
                    ]
                    rows.append(
                        {
                            "osm_type": "controlled_intervention",
                            "osm_id": base_id - primitive,
                            "height_m": height,
                            "height_source": "registered_controlled_intervention",
                            "local_exterior_xy_m": coordinates,
                            "is_registered_primitive": True,
                            "tags": {
                                "csi_pairs:provenance": "generated_controlled_intervention",
                                "csi_pairs:primitive": str(primitive),
                            },
                        }
                    )
                candidates.append(rows)
    for index in order:
        yield candidates[int(index)]


def _maximin_indices(
    candidates: np.ndarray,
    pool: np.ndarray,
    count: int,
    initial: list[int] | None = None,
) -> list[int]:
    available = [int(value) for value in np.asarray(pool, dtype=np.int64).tolist()]
    selected = [] if initial is None else [int(value) for value in initial]
    if len(available) < int(count) - len(selected):
        return []
    while len(selected) < int(count):
        if selected:
            distances = np.min(
                np.sum(
                    (candidates[np.asarray(available)][:, None, :]
                    - candidates[np.asarray(selected)][None, :, :]) ** 2,
                    axis=2,
                ),
                axis=1,
            )
            best_offset = max(
                range(len(available)),
                key=lambda offset: (float(distances[offset]), -available[offset]),
            )
        else:
            best_offset = min(
                range(len(available)),
                key=lambda offset: (
                    float(np.sum(candidates[available[offset]] ** 2)), available[offset]
                ),
            )
        selected.append(available.pop(best_offset))
    return selected


def _receiver_distance_match(
    receivers: np.ndarray, primitive_polygons: tuple[object, ...], relative_max: float
) -> np.ndarray:
    """Return receiver-specific pairwise primitive distance matches."""
    centers = np.asarray(
        [[polygon.centroid.x, polygon.centroid.y] for polygon in primitive_polygons],
        dtype=np.float64,
    )
    distances = np.linalg.norm(receivers[:, None, :] - centers[None, :, :], axis=2)
    relative = np.abs(distances[:, :, None] - distances[:, None, :]) / np.maximum(
        distances[:, :, None], np.finfo(np.float64).eps
    )
    matched = relative <= float(relative_max)
    diagonal = np.arange(centers.shape[0])
    matched[:, diagonal, diagonal] = False
    return matched


def _balanced_branching_indices(
    quality: np.ndarray,
    pool: np.ndarray,
    count: int,
    minimum_per_primitive: int,
) -> list[int]:
    """Select a deterministic high-quality subset with primitive coverage floors."""
    values = np.asarray(quality, dtype=np.float64)
    available = [int(value) for value in np.asarray(pool, dtype=np.int64)]
    target = int(count)
    minimum = int(minimum_per_primitive)
    if values.ndim != 2 or len(available) < target or minimum < 1:
        return []
    active = values > 0.0
    if np.any(np.sum(active[available], axis=0) < minimum):
        return []
    selected: list[int] = []
    coverage = np.zeros(values.shape[1], dtype=np.int64)
    while len(selected) < target:
        deficits = np.maximum(minimum - coverage, 0)
        best = max(
            available,
            key=lambda index: (
                int(np.sum(deficits * active[index])),
                int(np.sum(active[index])),
                float(np.sum(values[index])),
                -index,
            ),
        )
        selected.append(best)
        available.remove(best)
        coverage += active[best].astype(np.int64)
    return selected if np.all(coverage >= minimum) else []


def _filter_controlled_receiver_candidates(
    candidates: np.ndarray,
    primitive_polygons: tuple[object, ...],
    config: dict,
) -> np.ndarray:
    """Apply only added-object checks to the OSM-only receiver pool."""
    from shapely import linestrings, points
    from shapely.geometry import box
    from shapely.strtree import STRtree

    values = np.asarray(candidates, dtype=np.float64)
    tree = STRtree(list(primitive_polygons))
    clearance = float(config["map"]["receiver_building_clearance_m"])
    transmitter_z = float(config["radio"]["transmitter_z_m"])
    receiver_z = float(config["radio"]["receiver_z_m"])
    height = float(config["interventions"]["height_m"])
    point_geometries = points(values)
    clearance_pairs = tree.query(
        point_geometries, predicate="dwithin", distance=clearance
    )
    if clearance_pairs.size:
        keep = np.ones(values.shape[0], dtype=np.bool_)
        keep[np.unique(clearance_pairs[0])] = False
        values = values[keep]

    if all(polygon.equals(box(*polygon.bounds)) for polygon in primitive_polygons):
        keep = _axis_aligned_direct_visibility_mask(
            values,
            primitive_polygons,
            height=height,
            transmitter_z=transmitter_z,
            receiver_z=receiver_z,
            minimum_vertical_clearance_m=float(
                config["map"].get(
                    "receiver_minimum_direct_path_vertical_clearance_m", 0.0
                )
            ),
        )
        return values[keep]

    line_geometries = linestrings(
        np.stack((np.zeros_like(values), values), axis=1)
    )
    blocker_pairs = tree.query(line_geometries)
    blockers_by_line: dict[int, list[int]] = {}
    if blocker_pairs.size:
        for line_index, primitive_index in blocker_pairs.T:
            blockers_by_line.setdefault(int(line_index), []).append(
                int(primitive_index)
            )
    keep = np.ones(values.shape[0], dtype=np.bool_)
    for line_index, primitive_indices in blockers_by_line.items():
        keep[line_index] = _has_direct_geometric_visibility(
            line_geometries[line_index],
            [
                (primitive_polygons[index], height)
                for index in primitive_indices
            ],
            transmitter_z=transmitter_z,
            receiver_z=receiver_z,
            minimum_vertical_clearance_m=float(
                config["map"].get(
                    "receiver_minimum_direct_path_vertical_clearance_m", 0.0
                )
            ),
        )
    return values[keep]


def _axis_aligned_direct_visibility_mask(
    receiver_xy: np.ndarray,
    primitive_polygons: tuple[object, ...],
    *,
    height: float,
    transmitter_z: float,
    receiver_z: float,
    minimum_vertical_clearance_m: float,
) -> np.ndarray:
    """Vectorized exact segment/rectangle clearance for controlled primitives."""
    values = np.asarray(receiver_xy, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("receiver coordinates must have shape [N, 2]")
    visible = np.ones(values.shape[0], dtype=np.bool_)
    for polygon in primitive_polygons:
        min_x, min_y, max_x, max_y = (float(value) for value in polygon.bounds)
        enter = np.zeros(values.shape[0], dtype=np.float64)
        leave = np.ones(values.shape[0], dtype=np.float64)
        intersects = np.ones(values.shape[0], dtype=np.bool_)
        for direction, lower, upper in (
            (values[:, 0], min_x, max_x),
            (values[:, 1], min_y, max_y),
        ):
            stationary = direction == 0.0
            intersects &= (~stationary) | (lower <= 0.0 <= upper)
            moving = ~stationary
            lower_parameter = np.full(values.shape[0], -np.inf, dtype=np.float64)
            upper_parameter = np.full(values.shape[0], np.inf, dtype=np.float64)
            first = np.zeros(values.shape[0], dtype=np.float64)
            second = np.zeros(values.shape[0], dtype=np.float64)
            np.divide(lower, direction, out=first, where=moving)
            np.divide(upper, direction, out=second, where=moving)
            lower_parameter[moving] = np.minimum(first[moving], second[moving])
            upper_parameter[moving] = np.maximum(first[moving], second[moving])
            enter = np.maximum(enter, lower_parameter)
            leave = np.minimum(leave, upper_parameter)
        intersects &= enter <= leave
        ray_height = float(transmitter_z) + leave * (
            float(receiver_z) - float(transmitter_z)
        )
        visible &= ~(
            intersects
            & (
                ray_height - float(height)
                <= float(minimum_vertical_clearance_m)
            )
        )
    return visible


def _signed_wrong_action_coverage(bit_count: int) -> float:
    """Exact-control ceiling when all primitives share one signed edit family."""
    bits = int(bit_count)
    worlds = np.asarray(
        [tuple((world >> bit) & 1 for bit in range(bits)) for world in range(2**bits)],
        dtype=np.int64,
    )
    eligible = 0
    total = 0
    for state in worlds:
        for correct_bit in range(bits):
            direction = 1 - 2 * int(state[correct_bit])
            total += 1
            eligible += int(
                any(
                    bit != correct_bit and 1 - 2 * int(state[bit]) == direction
                    for bit in range(bits)
                )
            )
    return eligible / total


def _register_controlled_intervention_geometry(
    bank: dict, config: dict, scene_index: int
) -> dict:
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree

    base_catalog = _geometry_catalog(bank)
    original_polygons = list(base_catalog["polygons"])
    original_tree = STRtree(original_polygons)
    joint_quota = int(config["interventions"]["joint_reflection_quota"])
    minimum_per_primitive = int(
        config["interventions"]["minimum_exact_positions_per_primitive"]
    )
    minimum_quality_mean = float(
        config["interventions"]["minimum_selected_quality_mean_per_primitive"]
    )
    background_quota = int(config["positions_per_bank"]) - joint_quota
    relative_max = float(
        config["interventions"]["wrong_action_distance_relative_max"]
    )
    radio = config["radio"]
    minimum_incidence = float(config["map"]["minimum_reflection_incidence_cosine"])
    failures: dict[str, int] = {}
    maxima = {
        "pairwise_matched_branching_candidates": 0,
        "exact_matched_branching_candidates": 0,
        "background_no_reflection_candidates": 0,
        "exact_per_primitive_candidates": [0, 0, 0, 0],
        "selected_per_primitive_candidates": [0, 0, 0, 0],
        "selected_quality_mean_per_primitive": [0.0, 0.0, 0.0, 0.0],
    }
    try:
        base_candidates = _pre_render_receiver_candidates(
            bank, config, scene_index, _geometry_catalog(bank)
        )
    except RuntimeError as error:
        raise ReflectionGeometryQualificationError(
            "receiver_pool_too_small", str(error)
        ) from error
    for placement_index, primitive_rows in enumerate(
        _controlled_primitive_rows(config, scene_index)
    ):
        if placement_index and placement_index % 100 == 0:
            print(
                canonical_json(
                    {
                        "event": "controlled_placement_progress",
                        "scene_id": str(bank["scene_id"]),
                        "scene_index": int(scene_index),
                        "placements_examined": placement_index,
                        "failure_counts": dict(sorted(failures.items())),
                        "maxima": maxima,
                    }
                ),
                flush=True,
            )
        primitive_polygons = tuple(
            Polygon(row["local_exterior_xy_m"]) for row in primitive_rows
        )
        if any(
            len(original_tree.query(polygon.buffer(2.0), predicate="intersects"))
            for polygon in primitive_polygons
        ):
            failures["overlaps_osm_geometry"] = failures.get("overlaps_osm_geometry", 0) + 1
            continue
        candidate_bank = {
            **bank,
            "buildings": [*bank["buildings"], *primitive_rows],
            "building_count": len(bank["buildings"]) + len(primitive_rows),
        }
        candidates = _filter_controlled_receiver_candidates(
            base_candidates, primitive_polygons, config
        )
        if candidates.shape[0] < int(config["positions_per_bank"]):
            failures["receiver_pool_too_small"] = failures.get("receiver_pool_too_small", 0) + 1
            continue
        pairwise_matched = _receiver_distance_match(
            candidates, primitive_polygons, relative_max
        )
        primitive_count = len(primitive_polygons)
        all_pairs_matched = np.all(
            np.sum(pairwise_matched, axis=2) == primitive_count - 1,
            axis=1,
        )
        primitive_indices = tuple(
            range(len(base_catalog["polygons"]), len(base_catalog["polygons"]) + primitive_count)
        )
        approximate = [
            _approximate_reflection_qualities(
                polygon,
                float(primitive_rows[index]["height_m"]),
                candidates,
                transmitter_z=float(radio["transmitter_z_m"]),
                receiver_z=float(radio["receiver_z_m"]),
                minimum_incidence_cosine=minimum_incidence,
            )
            for index, polygon in enumerate(primitive_polygons)
        ]
        approximate_stack = np.stack(approximate, axis=1)
        joint_pool = np.flatnonzero(
            all_pairs_matched & (np.sum(approximate_stack > 0.0, axis=1) >= 2)
        )
        background_pool = np.flatnonzero(
            all_pairs_matched & np.all(approximate_stack == 0.0, axis=1)
        )
        maxima["pairwise_matched_branching_candidates"] = max(
            maxima["pairwise_matched_branching_candidates"], int(joint_pool.size)
        )
        maxima["background_no_reflection_candidates"] = max(
            maxima["background_no_reflection_candidates"], int(background_pool.size)
        )
        if joint_pool.size < joint_quota or background_pool.size < background_quota:
            failures["approximate_quota_unavailable"] = failures.get(
                "approximate_quota_unavailable", 0
            ) + 1
            continue
        catalog = _extend_geometry_catalog(base_catalog, primitive_rows)
        exact = []
        exact_limit = int(config["map"]["reflection_candidate_limit"])
        for primitive_index, values in zip(primitive_indices, approximate, strict=True):
            quality = np.zeros(candidates.shape[0], dtype=np.float64)
            ranked = sorted(
                joint_pool.tolist(), key=lambda index: (-float(values[index]), int(index))
            )[:exact_limit]
            for receiver_index in ranked:
                quality[receiver_index] = _visible_reflection_quality(
                    candidates[receiver_index],
                    primitive_index,
                    catalog,
                    transmitter_z=float(radio["transmitter_z_m"]),
                    receiver_z=float(radio["receiver_z_m"]),
                    minimum_incidence_cosine=minimum_incidence,
                )
            exact.append(quality)
        exact_stack = np.stack(exact, axis=1)
        exact_joint = np.flatnonzero(
            all_pairs_matched & (np.sum(exact_stack > 0.0, axis=1) >= 2)
        )
        maxima["exact_matched_branching_candidates"] = max(
            maxima["exact_matched_branching_candidates"], int(exact_joint.size)
        )
        if exact_joint.size < joint_quota:
            failures["exact_joint_quota_unavailable"] = failures.get(
                "exact_joint_quota_unavailable", 0
            ) + 1
            continue
        exact_per_primitive = np.sum(exact_stack[exact_joint] > 0.0, axis=0)
        maxima["exact_per_primitive_candidates"] = np.maximum(
            np.asarray(maxima["exact_per_primitive_candidates"], dtype=np.int64),
            exact_per_primitive,
        ).tolist()
        joint = _balanced_branching_indices(
            exact_stack,
            exact_joint,
            joint_quota,
            minimum_per_primitive,
        )
        if not joint:
            failures["per_primitive_exact_quota_unavailable"] = failures.get(
                "per_primitive_exact_quota_unavailable", 0
            ) + 1
            continue
        per_primitive_joint_counts = np.sum(exact_stack[joint] > 0.0, axis=0)
        maxima["selected_per_primitive_candidates"] = np.maximum(
            np.asarray(maxima["selected_per_primitive_candidates"], dtype=np.int64),
            per_primitive_joint_counts,
        ).tolist()
        selected_quality_sums = np.sum(exact_stack[joint], axis=0)
        selected_quality_means = selected_quality_sums / per_primitive_joint_counts
        maxima["selected_quality_mean_per_primitive"] = np.maximum(
            np.asarray(
                maxima["selected_quality_mean_per_primitive"], dtype=np.float64
            ),
            selected_quality_means,
        ).tolist()
        if np.any(selected_quality_means < minimum_quality_mean):
            failures["per_primitive_quality_floor_unavailable"] = failures.get(
                "per_primitive_quality_floor_unavailable", 0
            ) + 1
            continue
        background = _maximin_indices(
            candidates,
            background_pool,
            int(config["positions_per_bank"]),
            joint,
        )
        if len(background) != int(config["positions_per_bank"]):
            failures["background_quota_unavailable"] = failures.get(
                "background_quota_unavailable", 0
            ) + 1
            continue
        background = background[len(joint) :]
        selected = [*joint, *background]
        positions = candidates[np.asarray(selected, dtype=np.int64)]
        roles = np.asarray(
            ["matched-branching-reflection"] * joint_quota
            + ["background-no-primitive-reflection"] * background_quota,
            dtype="U40",
        )
        rng = np.random.default_rng(int(config["seed"]) + scene_index * 1009)
        order = np.arange(len(selected))
        rng.shuffle(order)
        output = dict(candidate_bank)
        output["primitive_osm_ids"] = [int(row["osm_id"]) for row in primitive_rows]
        output["receiver_positions_xy_m"] = np.round(positions[order], 9).tolist()
        output["receiver_geometry_roles"] = roles[order].tolist()
        output["geometry_registration"] = {
            "algorithm": "controlled-four-primitive-matched-branching-v5-clearance",
            "selection_stage": "pre-render",
            "world_or_effect_conditioning": False,
            "minimum_direct_path_vertical_clearance_m": float(
                config["map"]["receiver_minimum_direct_path_vertical_clearance_m"]
            ),
            "candidate_position_count": int(candidates.shape[0]),
            "minimum_incidence_cosine": minimum_incidence,
            "joint_reflection_quota": joint_quota,
            "minimum_exact_branches_per_active_position": 2,
            "minimum_exact_positions_per_primitive": minimum_per_primitive,
            "minimum_selected_quality_mean_per_primitive": minimum_quality_mean,
            "per_primitive_counts_are_blocking": True,
            "background_no_reflection_quota": background_quota,
            "wrong_action_distance_relative_max": relative_max,
            "all_active_positions_match_every_wrong_primitive": True,
            "pre_render_signed_wrong_action_coverage": _signed_wrong_action_coverage(
                primitive_count
            ),
            "primitive_geometry": dict(config["interventions"]),
            "primitive_records": [
                {
                    "controlled_id": int(primitive_rows[offset]["osm_id"]),
                    "selected_joint_position_count": int(
                        per_primitive_joint_counts[offset]
                    ),
                    "selected_quality_sum": round(
                        float(selected_quality_sums[offset]), 12
                    ),
                    "selected_quality_mean": round(
                        float(selected_quality_means[offset]), 12
                    ),
                }
                for offset in range(primitive_count)
            ],
        }
        return output
    reason = ", ".join(f"{key}={value}" for key, value in sorted(failures.items()))
    maxima_summary = ", ".join(
        f"max_{key}={value}" for key, value in sorted(maxima.items())
    )
    raise ReflectionGeometryQualificationError(
        "controlled_intervention_placement_unavailable",
        f"bank {bank['scene_id']} has no pre-render controlled placement satisfying all quotas "
        f"({reason}; {maxima_summary})",
    )


def _register_reflection_geometry(bank: dict, config: dict, scene_index: int) -> dict:
    catalog = _geometry_catalog(bank)
    candidates = _pre_render_receiver_candidates(bank, config, scene_index, catalog)
    map_config = config["map"]
    radio = config["radio"]
    quota = int(map_config["receiver_primitive_quota"])
    shortlist_size = int(map_config["primitive_shortlist_size"])
    candidate_limit = int(map_config["reflection_candidate_limit"])
    minimum_incidence = float(map_config["minimum_reflection_incidence_cosine"])
    transmitter_z = float(radio["transmitter_z_m"])
    receiver_z = float(radio["receiver_z_m"])

    approximate = []
    for primitive_index, polygon in enumerate(catalog["polygons"]):
        values = _approximate_reflection_qualities(
            polygon,
            float(catalog["heights"][primitive_index]),
            candidates,
            transmitter_z=transmitter_z,
            receiver_z=receiver_z,
            minimum_incidence_cosine=minimum_incidence,
        )
        ranked = _rank_positive(values)
        approximate.append(
            (
                min(len(ranked), quota),
                float(np.sum(values[np.asarray(ranked[:quota], dtype=np.int64)])),
                -int(catalog["osm_ids"][primitive_index]),
                primitive_index,
                values,
            )
        )
    approximate.sort(reverse=True, key=lambda row: row[:3])
    exact: dict[int, np.ndarray] = {}
    for _, _, _, primitive_index, approximate_values in approximate[:shortlist_size]:
        values = np.zeros(candidates.shape[0], dtype=np.float64)
        for receiver_index in _rank_positive(approximate_values)[:candidate_limit]:
            values[receiver_index] = _visible_reflection_quality(
                candidates[receiver_index],
                primitive_index,
                catalog,
                transmitter_z=transmitter_z,
                receiver_z=receiver_z,
                minimum_incidence_cosine=minimum_incidence,
            )
        if int(np.sum(values > 0.0)) >= quota:
            exact[primitive_index] = values
    if len(exact) < 2:
        raise ReflectionGeometryQualificationError(
            "insufficient_visible_primitives",
            f"bank {bank['scene_id']} has only {len(exact)} primitives with {quota} "
            "pre-render visible first-order reflection positions",
        )

    pair_candidates = []
    valid_indices = sorted(exact, key=lambda index: int(catalog["osm_ids"][index]))
    for first_offset, first_index in enumerate(valid_indices):
        for second_index in valid_indices[first_offset + 1 :]:
            allocation = _allocate_reflection_pair(
                exact[first_index], exact[second_index], quota
            )
            if allocation is None:
                continue
            first_receivers, second_receivers, quality_sums = allocation
            score = (min(quality_sums), sum(quality_sums))
            pair_candidates.append(
                (
                    score,
                    -int(catalog["osm_ids"][first_index]),
                    -int(catalog["osm_ids"][second_index]),
                    first_index,
                    second_index,
                    first_receivers,
                    second_receivers,
                    quality_sums,
                )
            )
    if not pair_candidates:
        raise ReflectionGeometryQualificationError(
            "disjoint_reflection_quota_unavailable",
            f"bank {bank['scene_id']} cannot allocate disjoint reflection quotas",
        )
    selected_pair = max(pair_candidates, key=lambda row: row[:3])
    first_index, second_index = selected_pair[3], selected_pair[4]
    first_receivers, second_receivers = selected_pair[5], selected_pair[6]
    selected = [*first_receivers, *second_receivers]
    selected_set = set(selected)
    remaining = set(range(candidates.shape[0])) - selected_set
    active = np.zeros(candidates.shape[0], dtype=np.bool_)
    active[list(remaining)] = True
    minimum_squared_distance = np.full(candidates.shape[0], np.inf, dtype=np.float64)
    for index in selected:
        delta = candidates - candidates[index]
        minimum_squared_distance = np.minimum(
            minimum_squared_distance, np.sum(delta * delta, axis=1)
        )
    minimum_squared_distance[~active] = -np.inf
    count = int(config["positions_per_bank"])
    while len(selected) < count:
        index = int(np.argmax(minimum_squared_distance))
        if not np.isfinite(minimum_squared_distance[index]):
            raise ReflectionGeometryQualificationError(
                "receiver_candidate_pool_exhausted",
                f"bank {bank['scene_id']} receiver maximin sampler exhausted visible candidates",
            )
        selected.append(index)
        active[index] = False
        delta = candidates - candidates[index]
        minimum_squared_distance = np.minimum(
            minimum_squared_distance, np.sum(delta * delta, axis=1)
        )
        minimum_squared_distance[~active] = -np.inf

    roles = np.asarray(
        ["primitive-0-reflection"] * quota
        + ["primitive-1-reflection"] * quota
        + ["background-maximin"] * (count - 2 * quota),
        dtype="U32",
    )
    selected_array = candidates[np.asarray(selected, dtype=np.int64)]
    rng = np.random.default_rng(int(config["seed"]) + scene_index * 1009)
    order = np.arange(count)
    rng.shuffle(order)
    selected_array = selected_array[order]
    roles = roles[order]

    primitive_indices = (first_index, second_index)
    primitive_ids = [int(catalog["osm_ids"][index]) for index in primitive_indices]
    output = dict(bank)
    output["primitive_osm_ids"] = primitive_ids
    output["buildings"] = [
        {
            **row,
            "is_registered_primitive": int(row["osm_id"]) in set(primitive_ids),
        }
        for row in bank["buildings"]
    ]
    output["receiver_positions_xy_m"] = np.round(selected_array, 9).tolist()
    output["receiver_geometry_roles"] = roles.tolist()
    output["geometry_registration"] = {
        "algorithm": "visible-first-order-exterior-wall-reflection-v1",
        "selection_stage": "pre-render",
        "world_or_effect_conditioning": False,
        "candidate_position_count": int(candidates.shape[0]),
        "minimum_incidence_cosine": minimum_incidence,
        "primitive_quota": quota,
        "primitive_records": [
            {
                "osm_id": int(catalog["osm_ids"][index]),
                "visible_reflection_candidate_count": int(np.sum(exact[index] > 0.0)),
                "selected_position_count": quota,
                "selected_quality_sum": round(float(selected_pair[7][offset]), 12),
            }
            for offset, index in enumerate(primitive_indices)
        ],
    }
    return output


def _write_scene_assets(record: dict, bank_dir: Path) -> None:
    from shapely.geometry import Polygon

    mesh_root = bank_dir / "mesh"
    mesh_root.mkdir()
    primitive_ids = [int(value) for value in record["primitive_osm_ids"]]
    groups: dict[str, list[tuple[object, float]]] = {
        "background": [],
        **{f"primitive-{index}": [] for index in range(len(primitive_ids))},
    }
    for building in record["buildings"]:
        polygon = Polygon(building["local_exterior_xy_m"])
        osm_id = int(building["osm_id"])
        primitive_index = next(
            (index for index, value in enumerate(primitive_ids) if osm_id == value), None
        )
        group = "background" if primitive_index is None else f"primitive-{primitive_index}"
        groups[group].append((polygon, float(building["height_m"])))
    for group, values in groups.items():
        if not values:
            raise RuntimeError(f"scene group is empty: {group}")
        triangles = np.concatenate([_extruded_triangles(polygon, height) for polygon, height in values])
        _write_triangle_ply(mesh_root / f"{group}.ply", triangles)
    origin = (-128.0, -128.0)
    size = 256.0
    ground = np.asarray(
        (
            ((origin[0], origin[1], 0.0), (origin[0] + size, origin[1], 0.0), (origin[0] + size, origin[1] + size, 0.0)),
            ((origin[0], origin[1], 0.0), (origin[0] + size, origin[1] + size, 0.0), (origin[0], origin[1] + size, 0.0)),
        ),
        dtype=np.float32,
    )
    _write_triangle_ply(mesh_root / "ground.ply", ground)
    _write_scene_xml(bank_dir / "scene.xml", len(primitive_ids))


def _extruded_triangles(polygon, height: float) -> np.ndarray:
    from shapely import constrained_delaunay_triangles
    from shapely.geometry.polygon import orient
    from shapely.ops import unary_union

    polygon = orient(polygon, sign=1.0)
    triangles = []
    roof_triangles = list(constrained_delaunay_triangles(polygon).geoms)
    if not roof_triangles:
        raise RuntimeError("building footprint produced no constrained roof triangles")
    roof_union = unary_union(roof_triangles)
    area_tolerance = max(1e-8, float(polygon.area) * 1e-10)
    if (
        float(roof_union.difference(polygon).area) > area_tolerance
        or float(polygon.difference(roof_union).area) > area_tolerance
    ):
        raise RuntimeError("constrained roof triangulation does not preserve the footprint")
    for triangle in roof_triangles:
        points = list(orient(triangle, sign=1.0).exterior.coords)[:3]
        triangles.append([[float(x), float(y), float(height)] for x, y in points])
    coordinates = list(polygon.exterior.coords)
    for first, second in zip(coordinates[:-1], coordinates[1:]):
        x0, y0 = first
        x1, y1 = second
        triangles.append(((x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, height)))
        triangles.append(((x0, y0, 0.0), (x1, y1, height), (x0, y0, height)))
    values = np.asarray(triangles, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or values.shape[0] == 0:
        raise RuntimeError("building extrusion produced no triangles")
    return values


def _write_triangle_ply(path: Path, triangles: np.ndarray) -> None:
    values = np.asarray(triangles, dtype=np.float32)
    vertices = values.reshape(-1, 3)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {vertices.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {values.shape[0]}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend("{:.9g} {:.9g} {:.9g}".format(*vertex) for vertex in vertices)
    lines.extend(
        f"3 {3 * index} {3 * index + 1} {3 * index + 2}"
        for index in range(values.shape[0])
    )
    _write_text_exclusive(path, "\n".join(lines) + "\n")


def _write_scene_xml(path: Path, primitive_count: int = 2) -> None:
    scene = ET.Element("scene", version="2.1.0")
    colors = {
        "mat-itu_wet_ground": (0.18, 0.18, 0.16),
        "mat-itu_concrete": (0.54, 0.54, 0.54),
        "mat-itu_brick": (0.52, 0.24, 0.18),
    }
    if int(primitive_count) == 2:
        colors["mat-itu_wood"] = (0.31, 0.20, 0.10)
    for material_id, color in colors.items():
        bsdf = ET.SubElement(scene, "bsdf", type="diffuse", id=material_id)
        ET.SubElement(
            bsdf,
            "rgb",
            name="reflectance",
            value=" ".join(str(value) for value in color),
        )
    objects = [
        ("ground", "mat-itu_wet_ground"),
        ("background", "mat-itu_concrete"),
        *[(f"primitive-{index}", "mat-itu_concrete") for index in range(int(primitive_count))],
    ]
    for name, material in objects:
        shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-{name}")
        ET.SubElement(shape, "string", name="filename", value=f"mesh/{name}.ply")
        ET.SubElement(shape, "ref", id=material, name="bsdf")
        ET.SubElement(shape, "boolean", name="face_normals", value="true")
    tree = ET.ElementTree(scene)
    with path.open("xb") as handle:
        tree.write(handle, encoding="utf-8", xml_declaration=True)


def _sionna_bootstrap_environment(
    project_root: str | Path,
    current: dict[str, str] | None = None,
    *,
    libllvm: str | Path | None = None,
) -> tuple[Path, dict[str, str]]:
    root = Path(project_root).resolve()
    runtime = root / SIONNA_RUNTIME_RELATIVE
    python = runtime / "bin" / "python"
    environment = dict(os.environ if current is None else current)
    from .sionna_runtime_lock import approved_library_record, require_runtime_record

    if libllvm is None:
        llvm_path, _record = require_runtime_record(root, runtime.parent)
    else:
        record = approved_library_record(root, libllvm)
        llvm_path = Path(record["libllvm_path"])
    configured_llvm = environment.get("DRJIT_LIBLLVM_PATH")
    if configured_llvm and Path(configured_llvm).resolve() != llvm_path:
        raise RuntimeError("DRJIT_LIBLLVM_PATH differs from the approved runtime record")
    required = {
        "Sionna runtime Python": python,
        "Dr.Jit LLVM library": llvm_path,
    }
    missing = [label for label, candidate in required.items() if not candidate.exists()]
    if missing:
        raise RuntimeError("Sionna runtime prerequisites are missing: " + ", ".join(missing))
    environment["DRJIT_LIBLLVM_PATH"] = str(llvm_path)
    environment["MI_DEFAULT_VARIANT"] = SIONNA_MITSUBA_VARIANT
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    project_text = str(root)
    python_path = [value for value in environment.get("PYTHONPATH", "").split(os.pathsep) if value]
    environment["PYTHONPATH"] = os.pathsep.join(
        [project_text, *[value for value in python_path if value != project_text]]
    )
    return python, environment


def ensure_sionna_runtime(reexec_arguments: list[str]) -> None:
    project_root = Path(__file__).resolve().parents[1]
    python, environment = _sionna_bootstrap_environment(project_root)
    runtime_prefix = python.parent.parent.resolve()
    correct_python = Path(sys.prefix).resolve() == runtime_prefix
    llvm_configured = os.environ.get("DRJIT_LIBLLVM_PATH") == environment["DRJIT_LIBLLVM_PATH"]
    if correct_python and llvm_configured:
        return
    if os.environ.get("CSI_PAIRS_SIONNA_BOOTSTRAPPED") == "1":
        raise RuntimeError(
            "Sionna runtime bootstrap did not activate the fixed Python and LLVM backend"
        )
    environment["CSI_PAIRS_SIONNA_BOOTSTRAPPED"] = "1"
    os.execve(str(python), [str(python), *reexec_arguments], environment)


def load_asset_manifest(asset_root: str | Path) -> tuple[Path, dict, dict]:
    root = Path(asset_root).resolve()
    manifest_path = root / "asset_manifest.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != ASSET_SCHEMA:
        raise ValueError("asset manifest schema mismatch")
    config_path = root / str(manifest["config_path"])
    config = load_config(config_path)
    if sha256_file(config_path) != manifest["config_sha256"]:
        raise ValueError("asset config snapshot hash mismatch")
    raw_sources = manifest.get("raw_sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != len(config["cities"]):
        raise ValueError("asset manifest raw OSM source inventory mismatch")
    expected_city_ids = {str(row["city_id"]) for row in config["cities"]}
    if {str(row.get("city_id", "")) for row in raw_sources} != expected_city_ids:
        raise ValueError("asset manifest raw OSM city inventory mismatch")
    for row in raw_sources:
        raw_path = (root / str(row.get("path", ""))).resolve()
        if (
            not raw_path.is_relative_to(root)
            or raw_path.is_symlink()
            or not raw_path.is_file()
            or row.get("bytes") != raw_path.stat().st_size
            or row.get("sha256") != sha256_file(raw_path)
        ):
            raise ValueError(f"raw OSM source is missing or changed: {raw_path}")
        if config["schema_version"] not in CONTROLLED_CONFIG_SCHEMAS:
            continue
        receipt_path = (root / str(row.get("receipt_path", ""))).resolve()
        if (
            not receipt_path.is_relative_to(root)
            or receipt_path.is_symlink()
            or not receipt_path.is_file()
            or row.get("receipt_sha256") != sha256_file(receipt_path)
        ):
            raise ValueError(f"raw OSM receipt is missing or changed: {receipt_path}")
        receipt = _read_json(receipt_path)
        if (
            not isinstance(receipt, dict)
            or set(receipt)
            != {
                "schema_version",
                "status",
                "city_id",
                "overpass_query",
                "overpass_endpoint",
                "bytes",
                "sha256",
            }
            or receipt["schema_version"] != RAW_OSM_RECEIPT_SCHEMA
            or receipt["status"] != "PASS"
            or receipt["city_id"] != row["city_id"]
            or receipt["overpass_query"] != row["overpass_query"]
            or receipt["overpass_endpoint"] != row["overpass_endpoint"]
            or receipt["bytes"] != row["bytes"]
            or receipt["sha256"] != row["sha256"]
        ):
            raise ValueError(f"raw OSM receipt does not bind its source: {receipt_path}")
    expected = expected_scene_ledger(config)
    if len(manifest.get("banks", [])) != len(expected):
        raise ValueError("asset manifest bank count mismatch")
    for row, ledger in zip(manifest["banks"], expected):
        for key in ("scene_index", "scene_id", "city_id", "role", "bank_id", "base_map_cluster_id"):
            if row.get(key) != ledger[key]:
                raise ValueError(f"asset manifest ledger mismatch for {key}")
        for file_row in row["files"]:
            path = root / str(file_row["path"])
            if path.is_symlink() or not path.is_file() or sha256_file(path) != file_row["sha256"]:
                raise ValueError(f"asset file is missing or changed: {path}")
    return root, manifest, config


def validate_assets(asset_root: str | Path, config_path: str | Path) -> Path:
    root, _manifest, frozen_config = load_asset_manifest(asset_root)
    requested_config = load_config(config_path)
    if canonical_json(frozen_config) != canonical_json(requested_config):
        raise ValueError("asset root does not match the requested candidate config")
    return root / "asset_manifest.json"


def render_shard(
    asset_root: str | Path,
    output_path: str | Path,
    scene_start: int,
    scene_end: int,
    shard_index: int,
) -> Path:
    root, manifest, config = load_asset_manifest(asset_root)
    if not (0 <= scene_start < scene_end <= len(manifest["banks"])):
        raise ValueError("render shard scene range is invalid")
    if scene_end != scene_start + 1:
        raise ValueError("formal render shards must contain exactly one atomic bank")
    if shard_index < 0:
        raise ValueError("render shard index must be nonnegative")
    started = time.monotonic()
    arrays_by_scene = []
    for scene_index in range(scene_start, scene_end):
        row = manifest["banks"][scene_index]
        bank_record = _read_json(root / str(row["bank_record_path"]))
        rendered = render_bank(bank_record, root, config, scene_index)
        arrays_by_scene.append(rendered)
        print(
            canonical_json(
                {
                    "event": "formal_candidate_bank_complete",
                    "render_shard_index": shard_index,
                    "scene_index": scene_index,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            ),
            flush=True,
        )
    arrays = {"scene_indices": np.arange(scene_start, scene_end, dtype=np.int64)}
    for name in arrays_by_scene[0]:
        arrays[name] = np.stack([row[name] for row in arrays_by_scene], axis=0)
    output = _write_npz_exclusive(output_path, arrays)
    result = {
        "schema_version": SHARD_SCHEMA,
        "status": "PASS",
        "fixture": False,
        "scientific_use": "CANDIDATE",
        "simulation_not_measurement": True,
        "asset_manifest_sha256": sha256_file(root / "asset_manifest.json"),
        "scene_start_inclusive": scene_start,
        "scene_end_exclusive": scene_end,
        "scene_count": scene_end - scene_start,
        "duration_seconds": time.monotonic() - started,
        "output_path": str(output.resolve()),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "render_shard_index": shard_index,
        "runtime": _sionna_runtime_record(),
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
    }
    _write_json_atomic_exclusive(output.with_suffix(".manifest.json"), result)
    return output


def _sionna_runtime_record() -> dict:
    import drjit
    import mitsuba

    llvm_path = Path(os.environ["DRJIT_LIBLLVM_PATH"]).resolve()
    return {
        "python": os.sys.version.split()[0],
        "sionna": importlib.metadata.version("sionna"),
        "sionna_rt": importlib.metadata.version("sionna-rt"),
        "mitsuba": importlib.metadata.version("mitsuba"),
        "drjit": importlib.metadata.version("drjit"),
        "mitsuba_variant": mitsuba.variant(),
        "drjit_thread_count": int(drjit.thread_count()),
        "sionna_revision": SIONNA_REVISION,
        "drjit_libllvm_path": str(llvm_path),
        "drjit_libllvm_sha256": sha256_file(llvm_path),
    }


def _validate_shard_runtime(runtime: object) -> None:
    from .sionna_runtime_lock import approved_library_record

    required = {
        "python",
        "sionna",
        "sionna_rt",
        "mitsuba",
        "drjit",
        "mitsuba_variant",
        "drjit_thread_count",
        "sionna_revision",
        "drjit_libllvm_path",
        "drjit_libllvm_sha256",
    }
    if not isinstance(runtime, dict) or set(runtime) != required:
        raise ValueError("render shard runtime fields must be exact")
    expected = {
        "python": SIONNA_PYTHON_VERSION,
        "sionna": SIONNA_VERSION,
        "sionna_rt": SIONNA_RT_VERSION,
        "mitsuba": MITSUBA_VERSION,
        "drjit": DRJIT_VERSION,
        "mitsuba_variant": SIONNA_MITSUBA_VARIANT,
        "drjit_thread_count": SIONNA_DRJIT_THREADS,
        "sionna_revision": SIONNA_REVISION,
    }
    if any(runtime.get(key) != value for key, value in expected.items()):
        raise ValueError("render shard runtime differs from the frozen Sionna runtime")
    llvm_path = Path(str(runtime["drjit_libllvm_path"]))
    approved = approved_library_record(Path(__file__).resolve().parents[1], llvm_path)
    if (
        not llvm_path.is_absolute()
        or not llvm_path.is_file()
        or llvm_path.is_symlink()
        or runtime["drjit_libllvm_sha256"] != sha256_file(llvm_path)
        or runtime["drjit_libllvm_sha256"] != approved["libllvm_sha256"]
    ):
        raise ValueError("render shard LLVM library provenance is invalid")


def _configure_renderer(config: dict) -> None:
    import drjit
    import mitsuba

    renderer = config["renderer"]
    mitsuba.set_variant(str(renderer["mitsuba_variant"]))
    drjit.set_thread_count(int(renderer["drjit_threads"]))
    if (
        mitsuba.variant() != SIONNA_MITSUBA_VARIANT
        or int(drjit.thread_count()) != SIONNA_DRJIT_THREADS
    ):
        raise RuntimeError("Sionna renderer did not activate the frozen LLVM backend")


def render_bank(bank: dict, asset_root: Path, config: dict, scene_index: int) -> dict[str, np.ndarray]:
    bank_started = time.monotonic()
    _configure_renderer(config)
    from sionna.rt import (
        ITURadioMaterial,
        PathSolver,
        PlanarArray,
        Receiver,
        Transmitter,
        load_scene,
        subcarrier_frequencies,
    )

    radio = config["radio"]
    worlds = np.asarray(config["world_bits"], dtype=np.int64)
    scene_xml = asset_root / "banks" / str(bank["scene_id"]) / "scene.xml"
    # Controlled primitives must remain distinct scene objects so their radio
    # materials and path surface identities can be changed independently.
    rt_scene = load_scene(scene_xml, merge_shapes=False)
    primitive_count = len(bank["primitive_osm_ids"])
    stable_surface_ids = _stable_surface_ids(primitive_count)
    expected_objects = set(stable_surface_ids)
    if set(rt_scene.objects) != expected_objects:
        raise RuntimeError(
            f"Sionna object catalog mismatch for {bank['scene_id']}: "
            f"expected={sorted(expected_objects)}, actual={sorted(rt_scene.objects)}"
        )
    runtime_to_stable = {
        int(rt_scene.objects[name].object_id): stable_id
        for name, stable_id in stable_surface_ids.items()
    }
    material_types = (
        ("glass",)
        if config["schema_version"] in CONTROLLED_CONFIG_SCHEMAS
        else ("glass", "metal")
    )
    for material_type in material_types:
        name = f"itu_{material_type}"
        if name not in rt_scene.radio_materials:
            rt_scene.add(
                ITURadioMaterial(
                    name=name,
                    itu_type=material_type,
                    thickness=SIONNA_MATERIAL_THICKNESS_M,
                )
            )
    rt_scene.frequency = float(radio["carrier_frequency_hz"])
    bandwidth_hz = float(radio["subcarrier_spacing_hz"]) * int(radio["subcarriers"])
    rt_scene.bandwidth = bandwidth_hz
    rt_scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=int(radio["tx_antennas"]),
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    rt_scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    rt_scene.add(Transmitter("tx", position=[0.0, 0.0, float(radio["transmitter_z_m"])]))
    positions = _receiver_positions(bank, config, scene_index)
    for position_index, xy in enumerate(positions):
        rt_scene.add(
            Receiver(
                f"rx-{position_index}",
                position=[float(xy[0]), float(xy[1]), float(radio["receiver_z_m"])],
            )
        )
    specular = PathSolver()
    frequencies = subcarrier_frequencies(
        int(radio["subcarriers"]), float(radio["subcarrier_spacing_hz"])
    )
    world_count = worlds.shape[0]
    position_count = positions.shape[0]
    channels = 2 * int(radio["tx_antennas"]) * int(radio["subcarriers"])
    max_paths = int(radio["max_stored_paths"])
    max_depth = int(radio["max_depth"])
    clean = np.zeros((world_count, position_count, channels), dtype=np.float64)
    maps = np.zeros((world_count, 3, 256, 256), dtype=np.float64)
    noop_maps = np.zeros_like(maps)
    path_ids = np.full((world_count, position_count, max_paths), -1, dtype=np.int64)
    path_power = np.zeros_like(path_ids, dtype=np.float64)
    path_surfaces = np.full(
        (world_count, position_count, max_paths, max_depth), -1, dtype=np.int64
    )
    noop_ids = np.full_like(path_ids, -1)
    noop_power = np.zeros_like(path_power)
    noop_surfaces = np.full_like(path_surfaces, -1)
    phase_ids, phase_values, phase_sources = _phase_references(bank, positions, config)
    primitive_mapping = _primitive_mapping(scene_index, primitive_count)
    anchor_bits = _anchor_bits(scene_index, primitive_count)
    print(
        canonical_json(
            {
                "event": "formal_candidate_bank_started",
                "scene_id": str(bank["scene_id"]),
                "scene_index": int(scene_index),
                "world_count": int(world_count),
            }
        ),
        flush=True,
    )
    for world_index, bits in enumerate(worlds):
        physical_states = _physical_states(
            bits,
            primitive_mapping,
            (
                anchor_bits
                if config["schema_version"] in CONTROLLED_CONFIG_SCHEMAS
                else None
            ),
        )
        _set_world_materials(
            rt_scene,
            physical_states,
            controlled=config["schema_version"] in CONTROLLED_CONFIG_SCHEMAS,
        )
        maps[world_index] = _rasterize_world(bank, physical_states)
        traced = specular(
            rt_scene,
            max_depth=max_depth,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=False,
            synthetic_array=True,
            seed=int(config["seed"]) + scene_index * 1000 + world_index,
        )
        raw_cfr = _flatten_cfr(traced, frequencies, bandwidth_hz, position_count, radio)
        gauge_fixed = raw_cfr * np.conjugate(phase_values)[:, None] / np.abs(phase_values)[:, None]
        clean[world_index] = np.concatenate((gauge_fixed.real, gauge_fixed.imag), axis=1)
        path_ids[world_index], path_power[world_index], path_surfaces[world_index] = _extract_paths(
            traced, runtime_to_stable, position_count, max_paths, max_depth
        )
        _require_rt_path_availability(
            path_ids[world_index],
            bank,
            positions,
            world_index=world_index,
            trace_kind="normal",
        )
        noop_maps[world_index] = _rasterize_world(bank, physical_states)
        noop = specular(
            rt_scene,
            max_depth=max_depth,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=False,
            synthetic_array=True,
            seed=int(config["seed"]) + 50_000_000 + scene_index * 1000 + world_index,
        )
        noop_ids[world_index], noop_power[world_index], noop_surfaces[world_index] = _extract_paths(
            noop, runtime_to_stable, position_count, max_paths, max_depth
        )
        _require_rt_path_availability(
            noop_ids[world_index],
            bank,
            positions,
            world_index=world_index,
            trace_kind="noop",
        )
        print(
            canonical_json(
                {
                    "elapsed_seconds": round(time.monotonic() - bank_started, 3),
                    "event": "formal_candidate_world_complete",
                    "scene_id": str(bank["scene_id"]),
                    "scene_index": int(scene_index),
                    "world_count": int(world_count),
                    "world_index": int(world_index),
                    "worlds_complete": int(world_index + 1),
                }
            ),
            flush=True,
        )
    if np.any(np.all(path_ids < 0, axis=-1)) or np.any(np.all(noop_ids < 0, axis=-1)):
        raise RuntimeError(
            f"internal RT path-availability invariant failed for bank {bank['scene_id']}"
        )
    if np.any(np.linalg.norm(clean, axis=-1) == 0.0):
        raise RuntimeError(f"Sionna produced an all-zero clean channel for {bank['scene_id']}")
    repeat_seeds = _repeat_seeds(config, scene_index, world_count, position_count)
    repeated = _add_observation_noise(clean, repeat_seeds, float(radio["observation_noise_std"]))
    free_space = _validate_common_free_space(maps, positions)
    return {
        "csi_repeat": repeated,
        "csi_clean": clean,
        "maps": maps,
        "noop_maps": noop_maps,
        "positions": positions,
        "free_space": free_space,
        "radio_config": np.asarray(
            (
                float(radio["carrier_frequency_hz"]),
                int(radio["tx_antennas"]),
                int(radio["subcarriers"]),
                float(radio["subcarrier_spacing_hz"]),
            ),
            dtype=np.float64,
        ),
        "bs_pose": np.asarray(
            (0.0, 0.0, float(radio["transmitter_z_m"]), 1.0, 0.0, 0.0, 0.0),
            dtype=np.float64,
        ),
        "repeat_seeds": repeat_seeds,
        "phase_reference_ids": phase_ids,
        "phase_reference_values": phase_values,
        "phase_reference_source_sha256": phase_sources,
        "path_ids": path_ids,
        "path_power": path_power,
        "path_surface_ids": path_surfaces,
        "noop_path_ids": noop_ids,
        "noop_path_power": noop_power,
        "noop_path_surface_ids": noop_surfaces,
        "primitive_surface_ids": np.asarray(
            [[stable_surface_ids[f"primitive-{index}"]] for index in range(primitive_count)],
            dtype=np.int64,
        )[primitive_mapping],
        "primitive_ids": primitive_mapping,
        "anchor_bits": anchor_bits,
        "natural_world_index": np.asarray(
            _natural_world_index(scene_index, worlds, primitive_count), dtype=np.int64
        ),
    }


def _receiver_positions(bank: dict, config: dict, scene_index: int) -> np.ndarray:
    count = int(config["positions_per_bank"])
    positions = np.asarray(bank.get("receiver_positions_xy_m"), dtype=np.float64)
    roles = np.asarray(bank.get("receiver_geometry_roles"))
    if positions.shape != (count, 2) or roles.shape != (count,):
        raise RuntimeError(f"bank {bank['scene_id']} has malformed registered receiver geometry")
    if not np.all(np.isfinite(positions)) or np.unique(positions, axis=0).shape[0] != count:
        raise RuntimeError(f"bank {bank['scene_id']} registered receiver positions are invalid")
    if config["schema_version"] in CONTROLLED_CONFIG_SCHEMAS:
        joint_quota = int(config["interventions"]["joint_reflection_quota"])
        expected_role_counts = {
            "matched-branching-reflection": joint_quota,
            "background-no-primitive-reflection": count - joint_quota,
        }
        expected_algorithm = "controlled-four-primitive-matched-branching-v5-clearance"
    else:
        expected_role_counts = {
            "primitive-0-reflection": int(config["map"]["receiver_primitive_quota"]),
            "primitive-1-reflection": int(config["map"]["receiver_primitive_quota"]),
            "background-maximin": count
            - 2 * int(config["map"]["receiver_primitive_quota"]),
        }
        expected_algorithm = "visible-first-order-exterior-wall-reflection-v1"
    observed_role_counts = {
        role: int(np.sum(roles == role)) for role in expected_role_counts
    }
    if observed_role_counts != expected_role_counts or set(roles.tolist()) != set(
        expected_role_counts
    ):
        raise RuntimeError(f"bank {bank['scene_id']} receiver role counts are invalid")
    registration = bank.get("geometry_registration")
    if (
        not isinstance(registration, dict)
        or registration.get("algorithm") != expected_algorithm
        or registration.get("selection_stage") != "pre-render"
        or registration.get("world_or_effect_conditioning") is not False
        or (
            config["schema_version"] in CONTROLLED_CONFIG_SCHEMAS
            and registration.get("minimum_direct_path_vertical_clearance_m")
            != float(
                config["map"][
                    "receiver_minimum_direct_path_vertical_clearance_m"
                ]
            )
        )
    ):
        raise RuntimeError(f"bank {bank['scene_id']} receiver geometry provenance is invalid")
    if config["schema_version"] in CONTROLLED_CONFIG_SCHEMAS:
        catalog = _geometry_catalog(bank)
        from shapely import linestrings

        lines = linestrings(np.stack((np.zeros_like(positions), positions), axis=1))
        minimum_clearance = float(
            config["map"]["receiver_minimum_direct_path_vertical_clearance_m"]
        )
        invalid = []
        for index, line in enumerate(lines):
            polygon_indices = catalog["tree"].query(line)
            buildings = [
                (
                    catalog["polygons"][int(polygon_index)],
                    float(catalog["heights"][int(polygon_index)]),
                )
                for polygon_index in polygon_indices
            ]
            if not _has_direct_geometric_visibility(
                line,
                buildings,
                transmitter_z=float(config["radio"]["transmitter_z_m"]),
                receiver_z=float(config["radio"]["receiver_z_m"]),
                minimum_vertical_clearance_m=minimum_clearance,
            ):
                invalid.append(index)
        if invalid:
            raise RuntimeError(
                "registered receiver direct-path clearance contract failed: "
                + canonical_json(
                    {
                        "invalid_receiver_indices": invalid,
                        "minimum_vertical_clearance_m": minimum_clearance,
                        "scene_id": str(bank["scene_id"]),
                    }
                )
            )
    return positions


def _has_direct_geometric_visibility(
    line,
    buildings,
    *,
    transmitter_z: float,
    receiver_z: float,
    minimum_vertical_clearance_m: float = 0.0,
) -> bool:
    from shapely.geometry import Point

    for polygon, height in buildings:
        intersection = line.intersection(polygon)
        if intersection.is_empty:
            continue
        geometries = intersection.geoms if hasattr(intersection, "geoms") else (intersection,)
        coordinates = [
            coordinate
            for geometry in geometries
            if hasattr(geometry, "coords")
            for coordinate in geometry.coords
        ]
        if not coordinates:
            return False
        farthest = max(
            float(line.project(Point(coordinate), normalized=True))
            for coordinate in coordinates
        )
        ray_height = transmitter_z + farthest * (receiver_z - transmitter_z)
        if ray_height - height <= float(minimum_vertical_clearance_m):
            return False
    return True


def _require_rt_path_availability(
    path_ids: np.ndarray,
    bank: dict,
    positions: np.ndarray,
    *,
    world_index: int,
    trace_kind: str,
) -> None:
    values = np.asarray(path_ids)
    coordinates = np.asarray(positions, dtype=np.float64)
    roles = np.asarray(bank.get("receiver_geometry_roles"))
    if (
        values.ndim != 2
        or coordinates.shape != (values.shape[0], 2)
        or roles.shape != (values.shape[0],)
        or trace_kind not in {"normal", "noop"}
    ):
        raise RuntimeError("RT path-availability diagnostic inputs are malformed")
    empty = np.flatnonzero(np.all(values < 0, axis=1))
    if not empty.size:
        return
    detail = {
        "empty_receiver_count": int(empty.size),
        "empty_receiver_indices": empty.tolist(),
        "empty_receiver_positions_xy_m": coordinates[empty].tolist(),
        "empty_receiver_roles": roles[empty].tolist(),
        "scene_id": str(bank.get("scene_id", "")),
        "trace_kind": trace_kind,
        "world_index": int(world_index),
    }
    raise RuntimeError(
        "RT path-availability contract failed: " + canonical_json(detail)
    )


def _primitive_mapping(scene_index: int, primitive_count: int = 2) -> np.ndarray:
    count = int(primitive_count)
    if count == 2:
        return np.asarray((0, 1) if scene_index % 2 == 0 else (1, 0), dtype=np.int64)
    values = np.arange(count, dtype=np.int64)
    return np.roll(values, int(scene_index) % count)


def _anchor_bits(scene_index: int, bit_count: int = 2) -> np.ndarray:
    count = int(bit_count)
    if count == 2:
        return np.asarray(((scene_index // 2) % 2, scene_index % 2), dtype=np.int64)
    return np.asarray(
        [int((int(scene_index) >> bit) & 1) for bit in range(count)], dtype=np.int64
    )


def _natural_world_index(
    scene_index: int, worlds: np.ndarray, bit_count: int = 2
) -> int:
    anchor = _anchor_bits(scene_index, bit_count)
    matches = np.flatnonzero(np.all(worlds == anchor, axis=1))
    if matches.size != 1:
        raise RuntimeError("natural anchor is absent from the world hypercube")
    return int(matches[0])


def _physical_states(
    bits: np.ndarray,
    mapping: np.ndarray,
    anchor_bits: np.ndarray | None = None,
) -> np.ndarray:
    logical = np.asarray(bits, dtype=np.int64)
    if anchor_bits is not None:
        anchor = np.asarray(anchor_bits, dtype=np.int64)
        if anchor.shape != logical.shape:
            raise ValueError("anchor bits must match the logical world state")
        logical = np.bitwise_xor(logical, anchor)
    states = np.zeros(len(mapping), dtype=np.int64)
    for bit_index, value in enumerate(logical):
        states[int(mapping[bit_index])] = int(value)
    return states


def _set_world_materials(
    rt_scene, states: np.ndarray, *, controlled: bool = False
) -> None:
    for primitive, state in enumerate(np.asarray(states, dtype=np.int64)):
        if controlled or primitive == 0:
            materials = ("itu_concrete", "itu_glass")
        else:
            materials = ("itu_wood", "itu_metal")
        rt_scene.objects[f"primitive-{primitive}"].radio_material = materials[int(state)]


def _rasterize_world(bank: dict, states: np.ndarray) -> np.ndarray:
    from shapely import contains_xy
    from shapely.geometry import Polygon

    cells = np.arange(256, dtype=np.float64) - 127.5
    x, y = np.meshgrid(cells, cells, indexing="xy")
    occupancy = np.zeros((256, 256), dtype=np.float64)
    height = np.zeros_like(occupancy)
    material = np.zeros_like(occupancy)
    primitive_ids = [int(value) for value in bank["primitive_osm_ids"]]
    ordered = sorted(
        bank["buildings"],
        key=lambda row: (int(row["osm_id"]) in set(primitive_ids), int(row["osm_id"])),
    )
    for building in ordered:
        polygon = Polygon(building["local_exterior_xy_m"])
        mask = contains_xy(polygon, x, y)
        if not np.any(mask):
            continue
        occupancy[mask] = 1.0
        height[mask] = float(building["height_m"])
        osm_id = int(building["osm_id"])
        primitive_index = next(
            (index for index, value in enumerate(primitive_ids) if osm_id == value), None
        )
        if primitive_index is None:
            category = MATERIAL_CATEGORY["concrete"]
        elif len(primitive_ids) > 2 or primitive_index == 0:
            category = MATERIAL_CATEGORY[
                "concrete" if int(states[primitive_index]) == 0 else "glass"
            ]
        else:
            category = MATERIAL_CATEGORY[
                "wood" if int(states[primitive_index]) == 0 else "metal"
            ]
        material[mask] = category
    return np.stack((occupancy, height, material), axis=0)


def _phase_references(bank: dict, positions: np.ndarray, config: dict):
    radio = config["radio"]
    frequency = float(radio["carrier_frequency_hz"])
    tx = np.asarray((0.0, 0.0, float(radio["transmitter_z_m"])), dtype=np.float64)
    identifiers = np.empty(positions.shape[0], dtype="U128")
    references = np.empty(positions.shape[0], dtype=np.complex128)
    source_sha = np.empty(positions.shape[0], dtype="U64")
    speed_of_light = 299_792_458.0
    for index, xy in enumerate(positions):
        receiver = np.asarray((xy[0], xy[1], float(radio["receiver_z_m"])), dtype=np.float64)
        distance = float(np.linalg.norm(receiver - tx))
        reference = np.exp(-1j * 2.0 * np.pi * frequency * distance / speed_of_light)
        record = {
            "schema_version": "csi-pairs-free-space-complex-reference-v1",
            "scene_id": bank["scene_id"],
            "position_index": index,
            "carrier_frequency_hz": frequency,
            "speed_of_light_m_s": speed_of_light,
            "tx_bs_centered_xyz_m": tx.tolist(),
            "receiver_bs_centered_xyz_m": receiver.tolist(),
            "distance_m": distance,
        }
        identifiers[index] = f"free-space-phase:{bank['scene_id']}:p{index:04d}"
        references[index] = reference
        source_sha[index] = _sha256_bytes(canonical_json(record).encode("ascii"))
    return identifiers, references, source_sha


def _flatten_cfr(paths, frequencies, bandwidth_hz, position_count: int, radio: dict) -> np.ndarray:
    values = np.asarray(
        paths.cfr(
            frequencies=frequencies,
            sampling_frequency=bandwidth_hz,
            num_time_steps=1,
            out_type="numpy",
        )
    )
    expected = (
        position_count,
        1,
        1,
        int(radio["tx_antennas"]),
        1,
        int(radio["subcarriers"]),
    )
    if values.shape != expected:
        raise RuntimeError(f"unexpected Sionna CFR shape: expected={expected}, actual={values.shape}")
    return values[:, :, 0, :, 0, :].reshape(position_count, -1)


def _stable_path_record(
    surface_ids: np.ndarray,
    delay_s: float,
    interaction_vertices_m: np.ndarray,
    angles_rad: np.ndarray,
) -> bytes:
    vertices = np.asarray(interaction_vertices_m, dtype=np.float64)
    if vertices.shape != (len(surface_ids), 3) or not np.all(np.isfinite(vertices)):
        raise RuntimeError("stable path vertices must be finite [interaction, xyz]")
    angles = np.asarray(angles_rad, dtype=np.float64)
    if angles.shape != (4,) or not np.all(np.isfinite(angles)):
        raise RuntimeError("stable path angles must be finite [theta_r, phi_r, theta_t, phi_t]")
    quantized_vertices = np.rint(vertices / PATH_VERTEX_QUANTIZATION_M).astype("<i8")
    quantized_angles = np.rint(angles / PATH_ANGLE_QUANTIZATION_RAD).astype("<i8")
    record = np.concatenate(
        (
            np.asarray(surface_ids, dtype="<i8"),
            np.asarray((int(round(delay_s * 1e12)),), dtype="<i8"),
            quantized_angles,
            quantized_vertices.reshape(-1),
        )
    )
    return record.tobytes()


def _stable_path_id(
    surface_ids: np.ndarray,
    delay_s: float,
    interaction_vertices_m: np.ndarray,
    angles_rad: np.ndarray,
) -> int:
    record = _stable_path_record(surface_ids, delay_s, interaction_vertices_m, angles_rad)
    return int.from_bytes(hashlib.sha256(record).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _extract_paths(
    paths,
    runtime_to_stable: dict[int, int],
    position_count: int,
    max_paths: int,
    max_depth: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.asarray(paths.valid)
    delay = np.asarray(paths.tau)
    objects = np.asarray(paths.objects)
    vertices = np.asarray(paths.vertices)
    phi_r = np.asarray(paths.phi_r)
    theta_r = np.asarray(paths.theta_r)
    phi_t = np.asarray(paths.phi_t)
    theta_t = np.asarray(paths.theta_t)
    real = np.asarray(paths.a[0])
    imag = np.asarray(paths.a[1])
    expected_prefix = (max_depth, position_count, 1)
    if (
        valid.ndim != 3
        or delay.shape != valid.shape
        or phi_r.shape != valid.shape
        or theta_r.shape != valid.shape
        or phi_t.shape != valid.shape
        or theta_t.shape != valid.shape
        or objects.shape[:3] != expected_prefix
        or vertices.shape[:3] != expected_prefix
        or vertices.shape[-1] != 3
    ):
        raise RuntimeError(
            f"unexpected Sionna path tensors: valid={valid.shape}, delay={delay.shape}, "
            f"angles={(theta_r.shape, phi_r.shape, theta_t.shape, phi_t.shape)}, "
            f"objects={objects.shape}, vertices={vertices.shape}"
        )
    power = np.sum(real * real + imag * imag, axis=(1, 2, 3))
    output_ids = np.full((position_count, max_paths), -1, dtype=np.int64)
    output_power = np.zeros((position_count, max_paths), dtype=np.float64)
    output_surfaces = np.full((position_count, max_paths, max_depth), -1, dtype=np.int64)
    invalid_shape = np.iinfo(np.uint32).max
    for receiver in range(position_count):
        entries = []
        for path_index in np.flatnonzero(valid[receiver, 0]):
            surfaces = np.full(max_depth, -1, dtype=np.int64)
            interaction_vertices = np.zeros((max_depth, 3), dtype=np.float64)
            for depth, raw_value in enumerate(objects[:, receiver, 0, path_index]):
                raw_id = int(raw_value)
                if raw_id == invalid_shape:
                    continue
                if raw_id not in runtime_to_stable:
                    raise RuntimeError(f"unregistered Sionna surface id: {raw_id}")
                surfaces[depth] = runtime_to_stable[raw_id]
                interaction_vertices[depth] = vertices[depth, receiver, 0, path_index]
            angles = np.asarray(
                (
                    theta_r[receiver, 0, path_index],
                    phi_r[receiver, 0, path_index],
                    theta_t[receiver, 0, path_index],
                    phi_t[receiver, 0, path_index],
                ),
                dtype=np.float64,
            )
            record = _stable_path_record(
                surfaces,
                float(delay[receiver, 0, path_index]),
                interaction_vertices,
                angles,
            )
            path_id = _stable_path_id(
                surfaces,
                float(delay[receiver, 0, path_index]),
                interaction_vertices,
                angles,
            )
            entries.append((path_id, record, float(power[receiver, path_index]), surfaces))
        records_by_id = {}
        merged = {}
        for path_id, record, path_value, surfaces in entries:
            previous = records_by_id.setdefault(path_id, record)
            if previous != record:
                raise RuntimeError("stable Sionna path identifier digest collision")
            if record in merged:
                merged[record][1] += path_value
            else:
                merged[record] = [path_id, path_value, surfaces]
        entries = sorted(merged.values(), key=lambda row: row[1], reverse=True)[:max_paths]
        entries = sorted(entries, key=lambda row: row[0])
        for output_index, (path_id, path_value, surfaces) in enumerate(entries):
            output_ids[receiver, output_index] = path_id
            output_power[receiver, output_index] = path_value
            output_surfaces[receiver, output_index] = surfaces
    return output_ids, output_power, output_surfaces


def _repeat_seeds(
    config: dict, scene_index: int, world_count: int, position_count: int
) -> np.ndarray:
    repeat_count = int(config["radio"]["repeats"])
    seeds = np.empty((world_count, position_count, repeat_count), dtype=np.int64)
    for world in range(world_count):
        for position in range(position_count):
            for repeat in range(repeat_count):
                seeds[world, position, repeat] = (
                    int(config["seed"])
                    + scene_index * 10_000_000
                    + world * 100_000
                    + position * 100
                    + repeat
                )
    return seeds


def _add_observation_noise(
    clean: np.ndarray, repeat_seeds: np.ndarray, noise_std: float
) -> np.ndarray:
    repeated = np.empty((*clean.shape[:-1], repeat_seeds.shape[-1], clean.shape[-1]))
    for world in range(clean.shape[0]):
        for position in range(clean.shape[1]):
            for repeat in range(repeat_seeds.shape[-1]):
                noise = np.random.default_rng(
                    int(repeat_seeds[world, position, repeat])
                ).normal(0.0, noise_std, size=clean.shape[-1])
                repeated[world, position, repeat] = clean[world, position] + noise
    return repeated


def _validate_common_free_space(maps: np.ndarray, positions: np.ndarray) -> np.ndarray:
    free = np.ones((maps.shape[0], positions.shape[0]), dtype=np.bool_)
    for position_index, xy in enumerate(positions):
        column = int(math.floor(float(xy[0]) + 128.0))
        row = int(math.floor(float(xy[1]) + 128.0))
        if not (0 <= row < 256 and 0 <= column < 256):
            raise RuntimeError("receiver position lies outside the map")
        if np.any(maps[:, 0, row, column] >= 0.5):
            raise RuntimeError("receiver position intersects an occupied map cell")
    return free


def build_engine_config(asset_root: Path, manifest: dict, config: dict) -> dict:
    schema = config["schema_version"]
    record = {
        "profile": (
            "formal-candidate-sionna-rt-osm-controlled-four-primitive-v6-post-rt-exclusions"
            if schema == CONFIG_SCHEMA_V6
            else
            "formal-candidate-sionna-rt-osm-controlled-four-primitive-v5"
            if schema == CONFIG_SCHEMA_V5
            else
            "formal-candidate-sionna-rt-osm-urban-banks-v4"
            if schema == CONFIG_SCHEMA_V4
            else "formal-candidate-sionna-rt-osm-urban-banks-v3"
        ),
        "simulation_not_measurement": True,
        "fixture": False,
        "scientific_use": "CANDIDATE",
        "seed": int(config["seed"]),
        "scene_count": len(manifest["banks"]),
        "world_bits": config["world_bits"],
        "positions_per_bank": int(config["positions_per_bank"]),
        "asset_manifest_sha256": sha256_file(asset_root / "asset_manifest.json"),
        "asset_config_sha256": manifest["config_sha256"],
        "source_config_sha256": manifest["source_config_sha256"],
        "raw_osm_sources": [
            {
                "city_id": row["city_id"],
                "sha256": row["sha256"],
                "query_sha256": _sha256_bytes(row["overpass_query"].encode("ascii")),
                "overpass_endpoint": row["overpass_endpoint"],
                "osm_base_timestamp": row["osm_base_timestamp"],
                "license_id": row["license_id"],
            }
            for row in manifest["raw_sources"]
        ],
        "engine": {
            "name": "NVIDIA Sionna RT PathSolver",
            "sionna_revision": SIONNA_REVISION,
            "sionna_rt_version": SIONNA_RT_VERSION,
            "license_id": "Apache-2.0",
            "renderer": config["renderer"],
        },
        "radio": config["radio"],
        "propagation": {
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": False,
            "refraction": False,
            "synthetic_array": True,
            "max_depth": int(config["radio"]["max_depth"]),
        },
        "map": config["map"],
        "interventions": (
            {
                "design": config["interventions"],
                "primitive_states": "concrete-to-glass for every controlled primitive",
                "world_rendering": "complete state render for every four-bit hypercube node",
                "itu_material_thickness_m": SIONNA_MATERIAL_THICKNESS_M,
            }
            if schema in CONTROLLED_CONFIG_SCHEMAS
            else {
                "primitive_0": "OSM building material concrete-to-glass",
                "primitive_1": "OSM building material wood-to-metal",
                "world_rendering": "complete state render for every hypercube node",
                "itu_material_thickness_m": SIONNA_MATERIAL_THICKNESS_M,
            }
        ),
        "path_identity": {
            "surface_ids": "ordered stable registered surface identifiers",
            "delay_quantization_s": 1e-12,
            "angle_order": ["theta_r", "phi_r", "theta_t", "phi_t"],
            "angle_quantization_rad": PATH_ANGLE_QUANTIZATION_RAD,
            "interaction_vertex_quantization_m": PATH_VERTEX_QUANTIZATION_M,
            "numerical_duplicate_rule": "sum power for identical quantized physical records",
            "digest": "sha256 truncated to nonnegative signed int64",
        },
        "geometry": {
            "source": "OpenStreetMap closed building ways",
            "license_id": "ODbL-1.0",
            "terrain": "flat_ground_no_terrain_tile_source",
            "bank_selection": (
                "sparse-OSM-first placement-aware disjoint 256-meter scene extents within each city"
                if schema in CONTROLLED_CONFIG_SCHEMAS
                else "score-ranked disjoint 256-meter scene extents within each city"
            ),
            "bs_selection": config["map"].get("bs_selection", "first-free-v1"),
            "candidate_qualification_order": config["map"].get(
                "candidate_qualification_order", "score-descending-v1"
            ),
            "building_height_rules": [
                "OSM height tag clamped to [3,120] m",
                "OSM building:levels times 3.2 m clamped to [3,120] m",
                "deterministic OSM-id hash fallback in [9,27] m",
            ],
            "receiver_sampling": {
                "stage": "pre-render map-and-height-only",
                "world_or_effect_conditioning": False,
                "selection": (
                    "at least two exact visible first-order controlled-primitive branches per "
                    "active receiver, all-pair distance-matched wrong-action geometry, and "
                    "deterministic maximin null fill"
                    if schema in CONTROLLED_CONFIG_SCHEMAS
                    else "visible first-order exterior-wall reflection quotas and deterministic maximin fill"
                ),
                "algorithm": (
                    "controlled-four-primitive-matched-branching-v5-clearance"
                    if schema in CONTROLLED_CONFIG_SCHEMAS
                    else "visible-first-order-exterior-wall-reflection-v1"
                ),
                "grid_spacing_m": float(config["map"]["receiver_grid_spacing_m"]),
                "building_clearance_m": float(
                    config["map"]["receiver_building_clearance_m"]
                ),
                "minimum_direct_path_vertical_clearance_m": float(
                    config["map"].get(
                        "receiver_minimum_direct_path_vertical_clearance_m", 0.0
                    )
                ),
                "minimum_bs_distance_m": float(
                    config["map"]["receiver_minimum_bs_distance_m"]
                ),
                "primitive_quota": (
                    int(config["interventions"]["joint_reflection_quota"])
                    if schema in CONTROLLED_CONFIG_SCHEMAS
                    else int(config["map"]["receiver_primitive_quota"])
                ),
                "primitive_shortlist_size": int(
                    config["map"]["primitive_shortlist_size"]
                ),
                "reflection_candidate_limit": int(
                    config["map"]["reflection_candidate_limit"]
                ),
                "minimum_reflection_incidence_cosine": float(
                    config["map"]["minimum_reflection_incidence_cosine"]
                ),
                **(
                    {
                        "controlled_primitive_count": int(
                            config["interventions"]["primitive_count"]
                        ),
                        "background_no_reflection_quota": int(
                            config["positions_per_bank"]
                            - config["interventions"]["joint_reflection_quota"]
                        ),
                        "wrong_action_distance_relative_max": float(
                            config["interventions"][
                                "wrong_action_distance_relative_max"
                            ]
                        ),
                        "pre_render_signed_wrong_action_coverage": _signed_wrong_action_coverage(
                            int(config["interventions"]["primitive_count"])
                        ),
                    }
                    if schema in CONTROLLED_CONFIG_SCHEMAS
                    else {}
                ),
            },
        },
        "generator": {
            "module_path": "formal_v2/sionna_osm_candidate.py",
            "module_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    if schema in POWER_CONFIG_SCHEMAS:
        record["power_design"] = config["power_design"]
    if schema == CONFIG_SCHEMA_V6:
        record["post_rt_exclusions"] = config["post_rt_exclusions"]
    return record


def merge_shards(
    config_path: str | Path,
    asset_root: str | Path,
    shard_paths: list[str | Path],
    output_path: str | Path,
) -> Path:
    config = load_config(config_path)
    root, manifest, asset_config = load_asset_manifest(asset_root)
    if canonical_json(config) != canonical_json(asset_config):
        raise ValueError("merge config differs from the frozen asset config")
    rows = []
    shard_manifests = []
    for path_value in shard_paths:
        shard_rows, shard_manifest = _validated_render_shard_rows(path_value, root)
        rows.extend(shard_rows)
        shard_manifests.append(shard_manifest)
    rows.sort(key=lambda row: row[0])
    ledger = expected_scene_ledger(config)
    scene_count = len(ledger)
    if [row[0] for row in rows] != list(range(scene_count)):
        raise ValueError("render shards must cover every configured scene index exactly once")
    shard_indices = [int(row.get("render_shard_index", -1)) for row in shard_manifests]
    if sorted(shard_indices) != list(range(len(shard_manifests))):
        raise ValueError("render shard indices must be unique and contiguous from zero")
    if len({canonical_json(row["runtime"]) for row in shard_manifests}) != 1:
        raise ValueError("render shards used different Sionna runtimes")
    stack = lambda name: np.stack([row[name] for _, row in rows], axis=0)
    arrays = {
        name: stack(name)
        for name in (
            "csi_repeat",
            "csi_clean",
            "maps",
            "positions",
            "free_space",
            "radio_config",
            "bs_pose",
            "repeat_seeds",
            "phase_reference_ids",
            "phase_reference_values",
            "phase_reference_source_sha256",
            "noop_maps",
            "path_ids",
            "path_power",
            "path_surface_ids",
            "noop_path_ids",
            "noop_path_power",
            "noop_path_surface_ids",
            "primitive_surface_ids",
            "primitive_ids",
            "anchor_bits",
            "natural_world_index",
        )
    }
    arrays["map_channel_names"] = MAP_CHANNEL_NAMES
    arrays["world_bits"] = np.asarray(config["world_bits"], dtype=np.int64)
    arrays["scene_ids"] = np.asarray([row["scene_id"] for row in ledger], dtype="U96")
    arrays["city_ids"] = np.asarray([row["city_id"] for row in ledger], dtype="U64")
    arrays["bank_ids"] = np.asarray([row["bank_id"] for row in ledger], dtype="U96")
    arrays["base_map_cluster_ids"] = np.asarray(
        [row["base_map_cluster_id"] for row in ledger], dtype="U96"
    )
    arrays["scene_roles"] = np.asarray([row["role"] for row in ledger], dtype="U40")
    position_count = int(config["positions_per_bank"])
    arrays["position_roles"] = np.full(
        (scene_count, position_count), "standard", dtype="U16"
    )
    arrays["position_ids"] = np.empty((scene_count, position_count), dtype="U160")
    for scene, ledger_row in enumerate(ledger):
        if ledger_row["role"] == "target":
            arrays["position_roles"][scene, ::2] = "support_pool"
            arrays["position_roles"][scene, 1::2] = "query"
        for position, xy in enumerate(arrays["positions"][scene]):
            arrays["position_ids"][scene, position] = (
                f"{ledger_row['city_id']}:{ledger_row['scene_id']}:"
                f"x{float(xy[0]):+.6f}:y{float(xy[1]):+.6f}"
            )
    array_sha = lambda value: hashlib.sha256(
        np.ascontiguousarray(np.asarray(value, dtype="<f8")).tobytes()
    ).hexdigest()
    world_count = len(config["world_bits"])
    arrays["canonical_map_sha256"] = np.asarray(
        [
            [array_sha(arrays["maps"][scene, world]) for world in range(world_count)]
            for scene in range(scene_count)
        ],
        dtype="U64",
    )
    arrays["noop_map_sha256"] = np.asarray(
        [
            [array_sha(arrays["noop_maps"][scene, world]) for world in range(world_count)]
            for scene in range(scene_count)
        ],
        dtype="U64",
    )
    engine_config = build_engine_config(root, manifest, config)
    engine_text = canonical_json(engine_config)
    arrays["engine_config_json"] = np.asarray(engine_text)
    metadata = {
        "schema_version": DATASET_SCHEMA,
        "dataset_id": (
            "CSI-PAIRS-V6-SIONNA-OSM-FORMAL-CANDIDATE-CONTROLLED-V6"
            if config["schema_version"] == CONFIG_SCHEMA_V6
            else
            "CSI-PAIRS-V6-SIONNA-OSM-FORMAL-CANDIDATE-CONTROLLED-V5"
            if config["schema_version"] == CONFIG_SCHEMA_V5
            else
            "CSI-PAIRS-V6-SIONNA-OSM-FORMAL-CANDIDATE-REFLECTION-V4"
            if config["schema_version"] == CONFIG_SCHEMA_V4
            else "CSI-PAIRS-V6-SIONNA-OSM-FORMAL-CANDIDATE-REFLECTION-V3"
        ),
        "dataset_version": (
            "2026-08-15-v6"
            if config["schema_version"] == CONFIG_SCHEMA_V6
            else
            "2026-08-13-v5"
            if config["schema_version"] == CONFIG_SCHEMA_V5
            else
            "2026-08-11-v4"
            if config["schema_version"] == CONFIG_SCHEMA_V4
            else "2026-08-10-v3"
        ),
        "scientific_use": "CANDIDATE",
        "fixture": False,
        "engine": {
            "name": "NVIDIA-Sionna-RT-PathSolver",
            "version": SIONNA_RT_VERSION,
            "source_revision": SIONNA_REVISION,
            "license_id": "Apache-2.0",
            "config_sha256": _sha256_bytes(engine_text.encode("ascii")),
            "deterministic": True,
        },
        "representation": {
            "csi_layout": "real_then_imag",
            "csi_units": "dimensionless_complex_baseband_channel_coefficient",
            "phase_gauge_rule": "shared_complex_reference",
            "coordinate_system": "bs_centered_right_handed_meters",
            "position_units": "m",
            "map_units": "m",
            "clean_target_definition": "single-thread LLVM Sionna RT CFR before independent observation noise",
            "antenna_count": int(config["radio"]["tx_antennas"]),
            "subcarrier_count": int(config["radio"]["subcarriers"]),
            "patch_complex_size": 2,
            "patch_antenna_size": 1,
            "patch_subcarrier_size": 2,
            "map_resolution_m": float(config["map"]["resolution_m"]),
            "map_origin_xy_m": [float(value) for value in config["map"]["origin_xy_m"]],
            "alignment_physical_representation": "complex_csi_plus_delay_angle_power",
        },
        "assets": {
            "license_ids": ["ODbL-1.0"],
            "provenance_uri": "https://www.openstreetmap.org/copyright",
            "material_library": "Sionna-RT-ITU-materials-plus-OSM-building-footprints-v1",
            "material_category_count": len(MATERIAL_CATEGORY),
            "redistribution_allowed": True,
        },
        "generation": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "generator_command": "python -m formal_v2.sionna_osm_candidate merge",
            "seed": int(config["seed"]),
        },
        "external_reference": {
            "available": True,
            "kind": "independent-engine-rerender-required-not-yet-supplied",
            "dataset_id": "PENDING-INDEPENDENT-ENGINE-EVIDENCE",
            "pairing_rule": "same registered external banks, worlds, positions, radio configuration, and edit primitives",
        },
    }
    arrays["metadata_json"] = np.asarray(canonical_json(metadata))
    output = _write_npz_exclusive(output_path, arrays)
    merge_manifest = {
        "schema_version": "csi-pairs-sionna-osm-candidate-generation-v2",
        "status": "PASS",
        "fixture": False,
        "scientific_use": "CANDIDATE",
        "simulation_not_measurement": True,
        "output_path": str(output.resolve()),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "asset_manifest_path": str((root / "asset_manifest.json").resolve()),
        "asset_manifest_sha256": sha256_file(root / "asset_manifest.json"),
        "shards": [
            {
                "path": row["output_path"],
                "sha256": row["output_sha256"],
                "render_shard_index": row["render_shard_index"],
                "duration_seconds": row["duration_seconds"],
            }
            for row in sorted(shard_manifests, key=lambda value: value["render_shard_index"])
        ],
        "engine_config_sha256": metadata["engine"]["config_sha256"],
        "generator_sha256": sha256_file(Path(__file__).resolve()),
    }
    _write_json_exclusive(output.with_suffix(".generation.json"), merge_manifest)
    return output


def _validated_render_shard_rows(
    shard_path: str | Path,
    asset_root: Path,
) -> tuple[list[tuple[int, dict[str, np.ndarray]]], dict]:
    path = Path(shard_path).resolve()
    manifest_path = path.with_suffix(".manifest.json")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"render shard must be a regular non-symlink file: {path}")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("render shard manifest must be a regular non-symlink file")
    shard_manifest = _read_json(manifest_path)
    if not isinstance(shard_manifest, dict) or shard_manifest.get("schema_version") != SHARD_SCHEMA:
        raise ValueError("render shard manifest schema mismatch")
    if (
        shard_manifest.get("status") != "PASS"
        or shard_manifest.get("fixture") is not False
        or shard_manifest.get("scientific_use") != "CANDIDATE"
        or shard_manifest.get("simulation_not_measurement") is not True
    ):
        raise ValueError("render shard classification or status is invalid")
    if Path(str(shard_manifest.get("output_path", ""))).resolve() != path:
        raise ValueError("render shard manifest output path mismatch")
    if int(shard_manifest.get("output_bytes", -1)) != path.stat().st_size:
        raise ValueError("render shard manifest byte count mismatch")
    if shard_manifest.get("output_sha256") != sha256_file(path):
        raise ValueError("render shard hash mismatch")
    if shard_manifest.get("asset_manifest_sha256") != sha256_file(
        asset_root / "asset_manifest.json"
    ):
        raise ValueError("render shard asset manifest mismatch")
    generator = Path(str(shard_manifest.get("generator_path", ""))).resolve()
    if generator != Path(__file__).resolve() or shard_manifest.get(
        "generator_sha256"
    ) != sha256_file(Path(__file__).resolve()):
        raise ValueError("render shard generator source path/hash mismatch")
    _validate_shard_runtime(shard_manifest.get("runtime"))
    rows = []
    with np.load(path, allow_pickle=False) as archive:
        if "scene_indices" not in archive.files:
            raise ValueError("render shard archive lacks scene_indices")
        indices = np.asarray(archive["scene_indices"], dtype=np.int64)
        start = int(shard_manifest.get("scene_start_inclusive", -1))
        end = int(shard_manifest.get("scene_end_exclusive", -1))
        if (
            indices.ndim != 1
            or indices.tolist() != list(range(start, end))
            or int(shard_manifest.get("scene_count", -1)) != indices.size
        ):
            raise ValueError("render shard scene range does not match its archive")
        for name in archive.files:
            if name != "scene_indices" and np.asarray(archive[name]).shape[0] != indices.size:
                raise ValueError(f"render shard field {name} has the wrong scene axis")
        for local_index, scene_index in enumerate(indices):
            rows.append(
                (
                    int(scene_index),
                    {
                        name: np.array(archive[name][local_index], copy=True)
                        for name in archive.files
                        if name != "scene_indices"
                    },
                )
            )
    return rows, shard_manifest


def validate_render_shard(
    asset_root: str | Path,
    shard_path: str | Path,
    scene_start: int,
    scene_end: int,
    shard_index: int,
) -> Path:
    root, _manifest, _config = load_asset_manifest(asset_root)
    rows, shard_manifest = _validated_render_shard_rows(shard_path, root)
    if (
        [row[0] for row in rows] != list(range(int(scene_start), int(scene_end)))
        or int(shard_manifest.get("render_shard_index", -1)) != int(shard_index)
    ):
        raise ValueError("render shard does not match its registered resume slot")
    return Path(shard_path).resolve()


def validate_generated_dataset(
    config_path: str | Path,
    asset_root: str | Path,
    dataset_path: str | Path,
) -> Path:
    from .formal_dataset import FormalDataset

    config = load_config(config_path)
    root, manifest, frozen_config = load_asset_manifest(asset_root)
    if canonical_json(config) != canonical_json(frozen_config):
        raise ValueError("generated dataset config differs from its asset root")
    dataset_file = Path(dataset_path).resolve()
    generation_path = dataset_file.with_suffix(".generation.json")
    if (
        dataset_file.is_symlink()
        or not dataset_file.is_file()
        or generation_path.is_symlink()
        or not generation_path.is_file()
    ):
        raise ValueError("generated dataset and generation manifest must be regular files")
    generation = _read_json(generation_path)
    if (
        not isinstance(generation, dict)
        or generation.get("schema_version")
        != "csi-pairs-sionna-osm-candidate-generation-v2"
        or generation.get("status") != "PASS"
        or generation.get("fixture") is not False
        or generation.get("scientific_use") != "CANDIDATE"
        or generation.get("simulation_not_measurement") is not True
        or Path(str(generation.get("output_path", ""))).resolve() != dataset_file
        or generation.get("output_bytes") != dataset_file.stat().st_size
        or generation.get("output_sha256") != sha256_file(dataset_file)
        or generation.get("asset_manifest_sha256")
        != sha256_file(root / "asset_manifest.json")
        or generation.get("generator_sha256") != sha256_file(Path(__file__).resolve())
    ):
        raise ValueError("generated dataset manifest is invalid or stale")
    dataset = FormalDataset.load(dataset_file)
    if dataset.scene_count != len(expected_scene_ledger(config)):
        raise ValueError("generated dataset scene count differs from its config")
    if dataset.engine_config != build_engine_config(root, manifest, config):
        raise ValueError("generated dataset engine config differs from immutable assets")
    return dataset_file


def _regenerate_bank_task(payload):
    scene_index, root_value, bank_record_path, config = payload
    root = Path(root_value)
    bank = _read_json(root / str(bank_record_path))
    return int(scene_index), render_bank(bank, root, config, int(scene_index))


def _regeneration_executor(workers: int):
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    )


def regenerate_dataset(dataset_path: str | Path, output_dir: str | Path) -> Path:
    from .formal_dataset import FormalDataset

    dataset = FormalDataset.load(dataset_path)
    if dataset.is_fixture or dataset.metadata["scientific_use"] != "CANDIDATE":
        raise RuntimeError("Sionna OSM verifier only accepts the non-fixture CANDIDATE dataset")
    asset_root = Path(dataset.source_path).parent / "assets"
    root, manifest, config = load_asset_manifest(asset_root)
    expected_engine = build_engine_config(root, manifest, config)
    if expected_engine != dataset.engine_config:
        raise RuntimeError("dataset engine config does not reconstruct from immutable assets")
    rows = [None] * len(manifest["banks"])
    workers = min(
        int(config["renderer"]["verification_workers"]),
        len(manifest["banks"]),
    )
    tasks = [
        (scene_index, str(root), bank_row["bank_record_path"], config)
        for scene_index, bank_row in enumerate(manifest["banks"])
    ]
    with _regeneration_executor(workers) as executor:
        futures = [executor.submit(_regenerate_bank_task, task) for task in tasks]
        for future in as_completed(futures):
            scene_index, rendered = future.result()
            rows[scene_index] = rendered
            print(
                canonical_json(
                    {
                        "event": "independent_regeneration_bank_complete",
                        "scene_index": scene_index,
                    }
                ),
                flush=True,
            )
    if any(row is None for row in rows):
        raise RuntimeError("independent regeneration did not return every formal bank")
    arrays = {
        "maps": np.stack([row["maps"] for row in rows]),
        "csi_clean": np.stack([row["csi_clean"] for row in rows]),
        "csi_repeat": np.stack([row["csi_repeat"] for row in rows]),
        "free_space": np.stack([row["free_space"] for row in rows]),
        "phase_reference_ids": np.stack([row["phase_reference_ids"] for row in rows]),
        "phase_reference_values": np.stack([row["phase_reference_values"] for row in rows]),
        "phase_reference_source_sha256": np.stack(
            [row["phase_reference_source_sha256"] for row in rows]
        ),
        "engine_config_json": np.asarray(canonical_json(expected_engine)),
        "noop_maps": np.stack([row["noop_maps"] for row in rows]),
        "path_ids": np.stack([row["path_ids"] for row in rows]),
        "path_power": np.stack([row["path_power"] for row in rows]),
        "path_surface_ids": np.stack([row["path_surface_ids"] for row in rows]),
        "noop_path_ids": np.stack([row["noop_path_ids"] for row in rows]),
        "noop_path_power": np.stack([row["noop_path_power"] for row in rows]),
        "noop_path_surface_ids": np.stack([row["noop_path_surface_ids"] for row in rows]),
    }
    if set(arrays) != set(REGENERATED_FIELDS):
        raise RuntimeError("regeneration output fields drifted from the V6 contract")
    return _write_npz_exclusive(Path(output_dir) / "regenerated.npz", arrays)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a non-fixture formal-candidate CSI-PAIRS dataset from frozen OSM assets and Sionna RT"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    assets = commands.add_parser("prepare-assets")
    assets.add_argument("--config", required=True)
    assets.add_argument("--output", required=True)
    assets.add_argument("--raw-cache")
    render = commands.add_parser("render-shard")
    render.add_argument("--asset-root", required=True)
    render.add_argument("--output", required=True)
    render.add_argument("--scene-start", required=True, type=int)
    render.add_argument("--scene-end", required=True, type=int)
    render.add_argument("--shard-index", required=True, type=int)
    merge = commands.add_parser("merge")
    merge.add_argument("--config", required=True)
    merge.add_argument("--asset-root", required=True)
    merge.add_argument("--shard", action="append", required=True)
    merge.add_argument("--output", required=True)
    regenerate = commands.add_parser("regenerate")
    regenerate.add_argument("--dataset", required=True)
    regenerate.add_argument("--output", required=True)
    scene_count = commands.add_parser("scene-count")
    scene_count.add_argument("--config", required=True)
    validate_asset_command = commands.add_parser("validate-assets")
    validate_asset_command.add_argument("--asset-root", required=True)
    validate_asset_command.add_argument("--config", required=True)
    validate_shard_command = commands.add_parser("validate-shard")
    validate_shard_command.add_argument("--asset-root", required=True)
    validate_shard_command.add_argument("--shard", required=True)
    validate_shard_command.add_argument("--scene-start", required=True, type=int)
    validate_shard_command.add_argument("--scene-end", required=True, type=int)
    validate_shard_command.add_argument("--shard-index", required=True, type=int)
    validate_generation_command = commands.add_parser("validate-generation")
    validate_generation_command.add_argument("--config", required=True)
    validate_generation_command.add_argument("--asset-root", required=True)
    validate_generation_command.add_argument("--dataset", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "scene-count":
        print(len(expected_scene_ledger(load_config(args.config))))
        return 0
    if args.command in {"prepare-assets", "render-shard", "regenerate"}:
        if argv is not None:
            raise RuntimeError("RT commands require process argv for authenticated re-execution")
        ensure_sionna_runtime(["-m", "formal_v2.sionna_osm_candidate", *sys.argv[1:]])
    if args.command == "prepare-assets":
        output = prepare_assets(args.config, args.output, args.raw_cache)
    elif args.command == "render-shard":
        output = render_shard(
            args.asset_root,
            args.output,
            args.scene_start,
            args.scene_end,
            args.shard_index,
        )
    elif args.command == "merge":
        output = merge_shards(args.config, args.asset_root, args.shard, args.output)
    elif args.command == "regenerate":
        output = regenerate_dataset(args.dataset, args.output)
    elif args.command == "validate-assets":
        output = validate_assets(args.asset_root, args.config)
    elif args.command == "validate-shard":
        output = validate_render_shard(
            args.asset_root,
            args.shard,
            args.scene_start,
            args.scene_end,
            args.shard_index,
        )
    else:
        output = validate_generated_dataset(
            args.config, args.asset_root, args.dataset
        )
    print(canonical_json({"status": "PASS", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
