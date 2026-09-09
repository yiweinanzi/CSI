from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path


STATISTICS = ("path_loss", "delay_spread", "angular_spread", "visible_path_count")
DATASET_SHA256 = "060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac"
PROTOCOL_SHA256 = "040235346c65dcaca60c642b53ebef8a9a2c15c5fb6e2cb99b3d3358dbe5b96b"
SIONNA_ENGINE_NAME = "Sionna RT"
SIONNA_ENGINE_VERSION = "2.0.1"
SIONNA_GENERATOR_SOURCE_SHA256 = "9ae838e512250e1ed5a8542ce67c003180200b8a210dab504446ba7ed45409f8"
SIONNA_ASSET_MANIFEST_SHA256 = "4547a46687201f2ab7d3dc92031e6e58dbf64a4b75fd796f6b946c228da479e7"
DIFFERT_ENGINE_NAME = "DiffeRT"
DIFFERT_ENGINE_REVISION = "differt@673cc58ef61906b8ab0869dd206b3d032dbc01b2"
DIFFERT_ENGINE_LICENSE = "MIT"
DIFFERT_ADAPTER_SOURCE_SHA256 = "8b8ab7eb98611c92aab3c288aad4e462ec748b9d34c8f8c6b18888428a294a08"
DIFFERT_SCENE_MANIFEST_SHA256 = "50fd69590aecf5a02956c79d753cd0223e1ebf318f6e5df4fe70cacf034f4465"
DIFFERT_ENGINE_CONFIG_SHA256 = "630d4df6764d4d709aefe19614a6f640ec804b60853a309f25ec5d0ddc251d3f"
PACKAGE_TEMPLATE_FILES = (
    "protocol.json",
    "adapter.py",
    "CALIBRATION_DESIGN.md",
    "LICENSE_REVIEW.md",
    "assemble_package.py",
    "render_sionna_statistics.py",
    "render_differt_statistics.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")


def _stage_package_template(template_root: Path, root: Path) -> None:
    if template_root.is_symlink() or not template_root.is_dir():
        raise RuntimeError("C11 package template root must be a regular directory")
    if root.is_symlink():
        raise RuntimeError("C11 package output root must not be a symbolic link")
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise FileExistsError("refusing to overwrite a nonempty C11 package root")
    else:
        root.mkdir()
    for name in PACKAGE_TEMPLATE_FILES:
        source = template_root / name
        destination = root / name
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"C11 package template file is missing or unsafe: {name}")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite C11 package template: {name}")
        shutil.copy2(source, destination)


def _stats(value: object, label: str) -> dict[str, float | int]:
    if not isinstance(value, dict) or set(value) != set(STATISTICS):
        raise RuntimeError(f"{label} statistics fields are not exact")
    parsed = {name: float(value[name]) for name in STATISTICS}
    if not all(math.isfinite(number) for number in parsed.values()):
        raise RuntimeError(f"{label} statistics are nonfinite")
    if any(parsed[name] < 0.0 for name in STATISTICS if name != "path_loss"):
        raise RuntimeError(f"{label} contains a negative physical statistic")
    visible = parsed["visible_path_count"]
    if not visible.is_integer():
        raise RuntimeError(f"{label} visible path count is not an integer")
    return {
        "path_loss": parsed["path_loss"],
        "delay_spread": parsed["delay_spread"],
        "angular_spread": parsed["angular_spread"],
        "visible_path_count": int(visible),
    }


def _source_asset(
    root: Path,
    partition: str,
    scene: dict,
    payload: dict,
) -> tuple[dict, dict]:
    scene_id = str(scene["scene_id"])
    batch_id = (
        "c11-sionna-v6-differt-v0.10-fit-20260821"
        if partition == "fit"
        else "c11-sionna-v6-validation-inputs-20260821"
    )
    source_record_id = f"c11:{partition}:{scene_id}"
    raw_unit_id = f"c11-physical-cluster:{scene['base_map_cluster_id']}:natural-world"
    wrapper = {
        "schema_version": "csi-pairs-v6-rt-source-asset-v1",
        "records": [
            {
                "generation_or_acquisition_batch_id": batch_id,
                "source_record_id": source_record_id,
                "raw_unit_id": raw_unit_id,
                "payload": payload,
            }
        ],
    }
    asset = root / f"{partition}-{scene_id}.json"
    _write_json(asset, wrapper)
    source = {
        "asset_path": f"source_assets/{asset.name}",
        "asset_sha256": _sha256(asset),
        "generation_or_acquisition_batch_id": batch_id,
        "source_record_id": source_record_id,
        "raw_unit_id": raw_unit_id,
    }
    unit = {
        "unit_id": f"c11-{scene_id}",
        "scene_id": scene_id,
        "source": source,
        "payload": payload,
    }
    return unit, source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--template-root",
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument("--sionna-statistics-root", required=True)
    parser.add_argument("--differt-statistics", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    template_root = Path(args.template_root).resolve()
    _stage_package_template(template_root, root)
    protocol = root / "protocol.json"
    adapter = root / "adapter.py"
    design_record = root / "CALIBRATION_DESIGN.md"
    license_review = root / "LICENSE_REVIEW.md"
    if _sha256(protocol) != PROTOCOL_SHA256:
        raise RuntimeError("C11 pre-registered protocol hash changed")
    if not adapter.is_file() or adapter.is_symlink():
        raise RuntimeError("C11 adapter must be a regular file")
    for record, label in (
        (design_record, "calibration design record"),
        (license_review, "license review record"),
    ):
        if not record.is_file() or record.is_symlink():
            raise RuntimeError(f"C11 {label} must be a regular file")

    sionna_rows = {}
    for path in sorted(Path(args.sionna_statistics_root).resolve().glob("*.json")):
        row = _read(path)
        if (
            row.get("schema_version") != "csi-pairs-c11-sionna-scene-statistics-v1"
            or row.get("dataset_sha256") != DATASET_SHA256
            or row.get("simulation_not_measurement") is not True
            or row.get("engine_name") != SIONNA_ENGINE_NAME
            or row.get("engine_version") != SIONNA_ENGINE_VERSION
            or row.get("generator_source_sha256") != SIONNA_GENERATOR_SOURCE_SHA256
            or row.get("asset_manifest_sha256") != SIONNA_ASSET_MANIFEST_SHA256
            or not isinstance(row.get("scene_index"), int)
            or not isinstance(row.get("receiver_count"), int)
            or int(row.get("receiver_count", -1)) != 256
        ):
            raise RuntimeError(f"invalid Sionna statistic record: {path}")
        natural_bits = row.get("natural_world_bits")
        if (
            not isinstance(natural_bits, list)
            or len(natural_bits) != 4
            or any(type(value) is not int or value not in {0, 1} for value in natural_bits)
        ):
            raise RuntimeError(f"invalid Sionna natural-world bits: {path}")
        scene_id = str(row.get("scene_id", ""))
        if not scene_id or scene_id in sionna_rows:
            raise RuntimeError("Sionna scene statistics have missing or duplicate identities")
        row["statistics"] = _stats(row.get("statistics"), f"Sionna {scene_id}")
        row["record_path"] = str(path)
        row["record_sha256"] = _sha256(path)
        sionna_rows[scene_id] = row

    differt_path = Path(args.differt_statistics).resolve()
    differt = _read(differt_path)
    if (
        differt.get("schema_version") != "csi-pairs-c11-differt-scene-statistics-v1"
        or differt.get("dataset_sha256") != DATASET_SHA256
        or differt.get("simulation_not_measurement") is not True
        or differt.get("engine_name") != DIFFERT_ENGINE_NAME
        or differt.get("engine_revision") != DIFFERT_ENGINE_REVISION
        or differt.get("engine_license") != DIFFERT_ENGINE_LICENSE
        or differt.get("adapter_source_sha256") != DIFFERT_ADAPTER_SOURCE_SHA256
        or differt.get("scene_manifest_sha256") != DIFFERT_SCENE_MANIFEST_SHA256
        or differt.get("engine_config_sha256") != DIFFERT_ENGINE_CONFIG_SHA256
    ):
        raise RuntimeError("invalid DiffeRT statistics package")
    differt_rows = {}
    for row in differt.get("scenes", []):
        scene_id = str(row.get("scene_id", ""))
        if not scene_id or scene_id in differt_rows:
            raise RuntimeError("DiffeRT scene statistics have missing or duplicate identities")
        row["statistics"] = _stats(row.get("statistics"), f"DiffeRT {scene_id}")
        differt_rows[scene_id] = row
    if set(sionna_rows) != set(differt_rows) or len(sionna_rows) != 32:
        raise RuntimeError("C11 requires exactly 32 aligned Sionna/DiffeRT external scenes")

    source_root = root / "source_assets"
    if source_root.exists() or source_root.is_symlink():
        raise FileExistsError("refusing to overwrite C11 source-assets directory")
    source_root.mkdir()
    partitions = {"fit": [], "validation": []}
    reference_rows = []
    city_counts = {"external-denver": 0, "external-miami": 0}
    clusters = set()
    for scene_id in sorted(sionna_rows):
        primary = sionna_rows[scene_id]
        independent = differt_rows[scene_id]
        for field in ("city_id", "base_map_cluster_id", "receiver_count"):
            if primary.get(field) != independent.get(field):
                raise RuntimeError(f"C11 engine records disagree on {field}: {scene_id}")
        natural_world = sum(
            int(bit) << index for index, bit in enumerate(primary["natural_world_bits"])
        )
        if (
            primary.get("scene_index") != independent.get("scene_index")
            or independent.get("natural_world_index") != natural_world
        ):
            raise RuntimeError(f"C11 engine records disagree on natural world: {scene_id}")
        city = str(primary["city_id"])
        if city not in city_counts:
            raise RuntimeError(f"unexpected C11 city: {city}")
        cluster = str(primary["base_map_cluster_id"])
        if cluster in clusters:
            raise RuntimeError("C11 base-map clusters must be unique")
        clusters.add(cluster)
        city_counts[city] += 1
        partition = "fit" if city == "external-denver" else "validation"
        payload = {
            "schema_version": "csi-pairs-c11-calibration-unit-v1",
            "simulation_not_measurement": True,
            "scene_id": scene_id,
            "city_id": city,
            "base_map_cluster_id": cluster,
            "receiver_count": int(primary["receiver_count"]),
            "natural_world": "frozen_scene_anchor",
            "natural_world_index": natural_world,
            "natural_world_bits": primary["natural_world_bits"],
            "primary_engine": SIONNA_ENGINE_NAME,
            "primary_engine_version": SIONNA_ENGINE_VERSION,
            "primary_generator_source_sha256": primary["generator_source_sha256"],
            "primary_asset_manifest_sha256": primary["asset_manifest_sha256"],
            "primary_bank_record_sha256": primary["bank_record_sha256"],
            "primary_scene_xml_sha256": primary["scene_xml_sha256"],
            "primary_record_sha256": primary["record_sha256"],
            "primary_statistics": primary["statistics"],
        }
        if partition == "fit":
            payload.update(
                {
                    "independent_engine": DIFFERT_ENGINE_NAME,
                    "independent_engine_revision": DIFFERT_ENGINE_REVISION,
                    "independent_engine_license": DIFFERT_ENGINE_LICENSE,
                    "independent_adapter_source_sha256": differt[
                        "adapter_source_sha256"
                    ],
                    "independent_scene_manifest_sha256": differt[
                        "scene_manifest_sha256"
                    ],
                    "independent_engine_config_sha256": differt[
                        "engine_config_sha256"
                    ],
                    "independent_record_sha256": _sha256(differt_path),
                    "independent_source_asset_sha256": independent["source_asset_sha256"],
                    "independent_reference_statistics": independent["statistics"],
                }
            )
        unit, _source = _source_asset(source_root, partition, primary, payload)
        partitions[partition].append(unit)
        if partition == "validation":
            reference_rows.append((unit["unit_id"], independent["statistics"]))
    if city_counts != {"external-denver": 16, "external-miami": 16}:
        raise RuntimeError(f"C11 city split is not 16/16: {city_counts}")

    fit_path = root / "fit.json"
    validation_path = root / "validation.json"
    reference_path = root / "validation_reference.csv"
    manifest_path = root / "independent_rt_calibration.json"
    for path in (fit_path, validation_path, reference_path, manifest_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite C11 package output: {path.name}")
    _write_json(
        fit_path,
        {
            "schema_version": "csi-pairs-v6-rt-calibration-partition-v3",
            "partition": "fit",
            "units": partitions["fit"],
        },
    )
    _write_json(
        validation_path,
        {
            "schema_version": "csi-pairs-v6-rt-calibration-partition-v3",
            "partition": "validation",
            "units": partitions["validation"],
        },
    )
    with reference_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("unit_id", *STATISTICS))
        for unit_id, values in sorted(reference_rows):
            writer.writerow(
                (
                    unit_id,
                    format(float(values["path_loss"]), ".17g"),
                    format(float(values["delay_spread"]), ".17g"),
                    format(float(values["angular_spread"]), ".17g"),
                    str(int(values["visible_path_count"])),
                )
            )
    manifest = {
        "schema_version": "csi-pairs-v6-rt-calibration-adapter-v6",
        "protocol_path": "protocol.json",
        "protocol_sha256": _sha256(protocol),
        "fit_dataset_path": "fit.json",
        "fit_dataset_sha256": _sha256(fit_path),
        "validation_inputs_path": "validation.json",
        "validation_inputs_sha256": _sha256(validation_path),
        "validation_reference_path": "validation_reference.csv",
        "validation_reference_sha256": _sha256(reference_path),
        "adapter_source_path": "adapter.py",
        "adapter_source_sha256": _sha256(adapter),
        "design_record_path": "CALIBRATION_DESIGN.md",
        "design_record_sha256": _sha256(design_record),
        "license_review_path": "LICENSE_REVIEW.md",
        "license_review_sha256": _sha256(license_review),
        "command": [
            "{python}",
            "{adapter_source}",
            "--fit",
            "{fit_dataset}",
            "--validation-inputs",
            "{validation_inputs}",
            "--protocol",
            "{protocol}",
            "--output",
            "{output}",
        ],
    }
    _write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
