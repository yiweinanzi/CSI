from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .formal_evidence import evidence_context
from .formal_io import artifact_manifest, read_strict_json, sha256_file, write_csv, write_json


STATISTICS = ("path_loss", "delay_spread", "angular_spread", "visible_path_count")
STATISTIC_COLUMNS = ("unit_id", *STATISTICS)
PARTITION_SCHEMA = "csi-pairs-v6-rt-calibration-partition-v3"
SOURCE_ASSET_SCHEMA = "csi-pairs-v6-rt-source-asset-v1"


def run_rt_calibration_gate(config, dataset, manifest_path, output_root):
    manifest_file = Path(manifest_path).resolve()
    manifest = read_strict_json(manifest_file)
    _validate_manifest(manifest)
    protocol_path = _bound_input(
        manifest["protocol_path"], manifest["protocol_sha256"], manifest_file.parent, "protocol"
    )
    fit_dataset_path = _bound_input(
        manifest["fit_dataset_path"], manifest["fit_dataset_sha256"], manifest_file.parent, "fit dataset"
    )
    validation_inputs_path = _bound_input(
        manifest["validation_inputs_path"],
        manifest["validation_inputs_sha256"],
        manifest_file.parent,
        "validation inputs",
    )
    validation_reference_path = _bound_input(
        manifest["validation_reference_path"],
        manifest["validation_reference_sha256"],
        manifest_file.parent,
        "validation reference",
    )
    adapter_source_path = _bound_input(
        manifest["adapter_source_path"],
        manifest["adapter_source_sha256"],
        manifest_file.parent,
        "adapter source",
    )
    data_paths = {fit_dataset_path, validation_inputs_path, validation_reference_path}
    if len(data_paths) != 3:
        raise RuntimeError("RT calibration fit, validation inputs, and reference must be distinct files")
    protocol = read_strict_json(protocol_path)
    _validate_protocol(protocol)
    reference_rows = _read_statistics_csv(validation_reference_path, "validation reference")
    fit_partition = _read_partition_contract(fit_dataset_path, "fit")
    validation_partition = _read_partition_contract(validation_inputs_path, "validation")
    independence = _validate_partition_independence(
        fit_partition, validation_partition, set(reference_rows)
    )
    if len(reference_rows) < int(protocol["minimum_validation_units"]):
        raise RuntimeError("RT calibration validation reference has too few units")

    output_dir = Path(output_root) / "qualification" / "rt_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    bound_manifest = output_dir / "adapter_manifest.json"
    write_json(
        bound_manifest,
        {
            **manifest,
            "protocol_path": str(protocol_path),
            "fit_dataset_path": str(fit_dataset_path),
            "validation_inputs_path": str(validation_inputs_path),
            "validation_reference_path": str(validation_reference_path),
            "adapter_source_path": str(adapter_source_path),
        },
    )
    command = [
        value.format(
            dataset=str(dataset.source_path),
            output=str(output_dir),
            python=sys.executable,
            fit_dataset=str(fit_dataset_path),
            validation_inputs=str(validation_inputs_path),
            protocol=str(protocol_path),
            adapter_source=str(adapter_source_path),
        )
        for value in manifest["command"]
    ]
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=project_root,
        env=_adapter_environment(project_root),
    )
    (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    result_path = output_dir / "adapter_result.json"
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError("RT calibration adapter failed")
    result_record = read_strict_json(result_path)
    required_result = {
        "schema_version",
        "fit_dataset_sha256",
        "validation_inputs_sha256",
        "fitted_parameters_path",
        "fitted_parameters_sha256",
        "simulated_statistics_path",
        "simulated_statistics_sha256",
    }
    if not isinstance(result_record, dict) or set(result_record) != required_result:
        raise RuntimeError("RT calibration adapter-result fields must be exact")
    if result_record["schema_version"] != "csi-pairs-v6-rt-calibration-adapter-result-v5":
        raise RuntimeError("RT calibration adapter-result schema mismatch")
    for key in ("fit_dataset_sha256", "validation_inputs_sha256"):
        if result_record[key] != manifest[key]:
            raise RuntimeError(f"RT calibration result {key} mismatch")
    fitted_path = _bound_output_file(
        output_dir,
        result_record["fitted_parameters_path"],
        result_record["fitted_parameters_sha256"],
        "fitted parameters",
    )
    simulated_path = _bound_output_file(
        output_dir,
        result_record["simulated_statistics_path"],
        result_record["simulated_statistics_sha256"],
        "simulated statistics",
    )
    simulated_rows = _read_statistics_csv(simulated_path, "simulated statistics")
    validated_rows, assessments = _join_and_assess(
        reference_rows,
        simulated_rows,
        protocol["absolute_tolerances"],
    )
    validated_path = output_dir / "validated_statistics.csv"
    write_csv(validated_path, validated_rows)
    passed = all(value["passed"] for value in assessments.values())
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    )
    gate = {
        "schema_version": "csi-pairs-v6-rt-calibration-gate-v5",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        **evidence,
        "claim": "C11",
        "protocol_sha256": manifest["protocol_sha256"],
        "adapter_source_path": str(adapter_source_path),
        "adapter_source_sha256": manifest["adapter_source_sha256"],
        "input_manifest_path": bound_manifest.name,
        "input_manifest_sha256": sha256_file(bound_manifest),
        "fit_dataset_sha256": manifest["fit_dataset_sha256"],
        "validation_inputs_sha256": manifest["validation_inputs_sha256"],
        "validation_reference_sha256": manifest["validation_reference_sha256"],
        "fitted_parameters_sha256": sha256_file(fitted_path),
        "simulated_statistics_path": simulated_path.name,
        "simulated_statistics_sha256": sha256_file(simulated_path),
        "validated_statistics_path": validated_path.name,
        "validated_statistics_sha256": sha256_file(validated_path),
        "validation_unit_count": len(validated_rows),
        "aggregation": "mean_absolute_error_per_unit",
        "fit_unit_count": len(fit_partition),
        "fit_scene_count": len({row["scene_id"] for row in fit_partition.values()}),
        "validation_scene_count": len(
            {row["scene_id"] for row in validation_partition.values()}
        ),
        "fit_validation_independence": independence,
        "statistics": assessments,
    }
    write_json(output_dir / "gate.json", gate)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
            **evidence,
            "files": artifact_manifest(output_dir, evidence=evidence),
        },
    )
    return gate


def _validate_manifest(manifest):
    required = {
        "schema_version",
        "protocol_path",
        "protocol_sha256",
        "fit_dataset_path",
        "fit_dataset_sha256",
        "validation_inputs_path",
        "validation_inputs_sha256",
        "validation_reference_path",
        "validation_reference_sha256",
        "adapter_source_path",
        "adapter_source_sha256",
        "command",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("RT calibration manifest fields must be exact")
    if manifest["schema_version"] != "csi-pairs-v6-rt-calibration-adapter-v5":
        raise ValueError("RT calibration manifest schema mismatch")
    digest_keys = (
        "protocol_sha256",
        "fit_dataset_sha256",
        "validation_inputs_sha256",
        "validation_reference_sha256",
        "adapter_source_sha256",
    )
    for key in digest_keys:
        value = manifest[key]
        if not _lower_sha256(value):
            raise ValueError(f"RT calibration {key} must be lowercase SHA-256")
    if len({manifest[key] for key in digest_keys[1:4]}) != 3:
        raise ValueError("RT calibration fit, validation inputs, and reference must be distinct")
    if not isinstance(manifest["command"], list) or not manifest["command"]:
        raise ValueError("RT calibration command must be nonempty argv")
    command = manifest["command"]
    if (
        Path(manifest["adapter_source_path"]).suffix != ".py"
        or command[:2] != ["{python}", "{adapter_source}"]
        or any(
            command.count(placeholder) != 1
            for placeholder in (
                "{fit_dataset}",
                "{validation_inputs}",
                "{protocol}",
                "{output}",
            )
        )
        or any("validation_reference" in value for value in command)
    ):
        raise ValueError(
            "RT calibration command must directly execute the authenticated adapter, bind fit, "
            "validation inputs, protocol, and output exactly once, and exclude validation references"
        )
    for key in (
        "protocol_path",
        "fit_dataset_path",
        "validation_inputs_path",
        "validation_reference_path",
        "adapter_source_path",
    ):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ValueError(f"RT calibration {key} must be nonempty")


def _read_statistics_csv(path, label):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != STATISTIC_COLUMNS:
            raise RuntimeError(f"RT calibration {label} columns must be exact")
        source_rows = list(reader)
    if not source_rows:
        raise RuntimeError(f"RT calibration {label} is empty")
    rows = {}
    for source in source_rows:
        unit_id = source["unit_id"]
        if not unit_id or unit_id.strip() != unit_id or unit_id in rows:
            raise RuntimeError(f"RT calibration {label} unit IDs must be nonempty and unique")
        parsed = {"unit_id": unit_id}
        for statistic in STATISTICS:
            try:
                value = float(source[statistic])
            except (TypeError, ValueError) as error:
                raise RuntimeError(f"RT calibration {label} contains a nonnumeric statistic") from error
            if not np.isfinite(value):
                raise RuntimeError(f"RT calibration {label} statistics must be finite")
            if statistic != "path_loss" and value < 0:
                raise RuntimeError(f"RT calibration {label} contains a negative physical statistic")
            if statistic == "visible_path_count" and not value.is_integer():
                raise RuntimeError(f"RT calibration {label} visible path counts must be integers")
            parsed[statistic] = value
        rows[unit_id] = parsed
    return rows


def _read_partition_contract(path, partition):
    contract_path = Path(path).resolve()
    contract = read_strict_json(contract_path)
    required = {"schema_version", "partition", "units"}
    if not isinstance(contract, dict) or set(contract) != required:
        raise RuntimeError(f"RT calibration {partition} partition fields must be exact")
    if contract["schema_version"] != PARTITION_SCHEMA or contract["partition"] != partition:
        raise RuntimeError(f"RT calibration {partition} partition schema/role mismatch")
    if not isinstance(contract["units"], list) or not contract["units"]:
        raise RuntimeError(f"RT calibration {partition} partition must be nonempty")
    parsed = {}
    source_hashes = {}
    source_records = {}
    raw_unit_identities = set()
    for row in contract["units"]:
        if not isinstance(row, dict) or set(row) != {
            "unit_id",
            "scene_id",
            "source",
            "payload",
        }:
            raise RuntimeError(f"RT calibration {partition} unit fields must be exact")
        unit_id = row["unit_id"]
        scene_id = row["scene_id"]
        source = row["source"]
        if (
            not isinstance(unit_id, str)
            or not unit_id
            or unit_id.strip() != unit_id
            or unit_id in parsed
            or not isinstance(scene_id, str)
            or not scene_id
            or scene_id.strip() != scene_id
            or not isinstance(source, dict)
            or set(source)
            != {
                "asset_path",
                "asset_sha256",
                "generation_or_acquisition_batch_id",
                "source_record_id",
                "raw_unit_id",
            }
            or not isinstance(row["payload"], dict)
            or not row["payload"]
        ):
            raise RuntimeError(
                f"RT calibration {partition} units require unique IDs, stable scene/source identity, and nonempty inline payloads"
            )
        identity_values = (
            source["generation_or_acquisition_batch_id"],
            source["source_record_id"],
            source["raw_unit_id"],
        )
        if any(
            not isinstance(value, str) or not value or value.strip() != value
            for value in identity_values
        ):
            raise RuntimeError(
                f"RT calibration {partition} source identity fields must be stable nonempty strings"
            )
        source_path = source["asset_path"]
        source_sha256 = source["asset_sha256"]
        if not isinstance(source_path, str) or not source_path.strip() or not _lower_sha256(source_sha256):
            raise RuntimeError(
                f"RT calibration {partition} source asset reference is invalid"
            )
        asset = Path(source_path)
        if not asset.is_absolute():
            asset = contract_path.parent / asset
        asset = asset.resolve()
        if not asset.is_file():
            raise RuntimeError(f"RT calibration {partition} source asset is missing")
        actual_sha256 = source_hashes.get(asset)
        if actual_sha256 is None:
            actual_sha256 = sha256_file(asset)
            source_hashes[asset] = actual_sha256
        if actual_sha256 != source_sha256:
            raise RuntimeError(f"RT calibration {partition} source asset hash mismatch")
        records = source_records.get(asset)
        if records is None:
            records = _read_source_asset_records(asset, partition)
            source_records[asset] = records
        if identity_values not in records:
            raise RuntimeError(
                f"RT calibration {partition} source identity is absent from its authenticated asset"
            )
        canonical_payload = _canonical_json_bytes(row["payload"])
        if canonical_payload != records[identity_values]:
            raise RuntimeError(
                f"RT calibration {partition} payload differs from its authenticated source record"
            )
        raw_unit_identity = identity_values[2]
        if raw_unit_identity in raw_unit_identities:
            raise RuntimeError(
                f"RT calibration {partition} raw-unit identities must be unique"
            )
        raw_unit_identities.add(raw_unit_identity)
        parsed[unit_id] = {
            **row,
            "_source_asset_sha256": actual_sha256,
            "_source_record_identity": identity_values[:2],
            "_raw_unit_identity": raw_unit_identity,
            "_canonical_payload_sha256": hashlib.sha256(canonical_payload).hexdigest(),
        }
    return parsed


def _canonical_json_bytes(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _read_source_asset_records(asset, partition):
    payload = read_strict_json(asset)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "records"}
        or payload["schema_version"] != SOURCE_ASSET_SCHEMA
        or not isinstance(payload["records"], list)
        or not payload["records"]
    ):
        raise RuntimeError(
            f"RT calibration {partition} source asset contract is invalid"
        )
    records = {}
    required = {
        "generation_or_acquisition_batch_id",
        "source_record_id",
        "raw_unit_id",
        "payload",
    }
    for record in payload["records"]:
        if not isinstance(record, dict) or set(record) != required:
            raise RuntimeError(
                f"RT calibration {partition} source asset record fields must be exact"
            )
        identity = (
            record["generation_or_acquisition_batch_id"],
            record["source_record_id"],
            record["raw_unit_id"],
        )
        if (
            any(
                not isinstance(value, str) or not value or value.strip() != value
                for value in identity
            )
            or identity in records
            or not isinstance(record["payload"], dict)
            or not record["payload"]
        ):
            raise RuntimeError(
                f"RT calibration {partition} source asset records require unique stable identity and payload"
            )
        records[identity] = _canonical_json_bytes(record["payload"])
    return records


def _validate_partition_independence(fit_partition, validation_partition, reference_unit_ids):
    validation_units = set(validation_partition)
    if validation_units != set(reference_unit_ids):
        missing = sorted(set(reference_unit_ids).difference(validation_units))
        unexpected = sorted(validation_units.difference(reference_unit_ids))
        raise RuntimeError(
            "RT calibration validation partition differs from validation reference: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    unit_overlap = sorted(set(fit_partition).intersection(validation_partition))
    fit_scenes = {row["scene_id"] for row in fit_partition.values()}
    validation_scenes = {row["scene_id"] for row in validation_partition.values()}
    scene_overlap = sorted(fit_scenes.intersection(validation_scenes))
    source_asset_sha256_overlap = sorted(
        {row["_source_asset_sha256"] for row in fit_partition.values()}.intersection(
            row["_source_asset_sha256"] for row in validation_partition.values()
        )
    )
    source_record_identity_overlap = sorted(
        {row["_source_record_identity"] for row in fit_partition.values()}.intersection(
            row["_source_record_identity"]
            for row in validation_partition.values()
        )
    )
    raw_unit_identity_overlap = sorted(
        {row["_raw_unit_identity"] for row in fit_partition.values()}.intersection(
            row["_raw_unit_identity"] for row in validation_partition.values()
        )
    )
    payload_overlap = {
        row["_canonical_payload_sha256"] for row in fit_partition.values()
    }.intersection(
        row["_canonical_payload_sha256"] for row in validation_partition.values()
    )
    if (
        unit_overlap
        or scene_overlap
        or source_asset_sha256_overlap
        or source_record_identity_overlap
        or raw_unit_identity_overlap
    ):
        raise RuntimeError(
            "RT calibration raw fit and validation partitions are not independent: "
            f"unit_overlap={unit_overlap[:5]}, scene_overlap={scene_overlap[:5]}, "
            f"source_asset_sha256_overlap={source_asset_sha256_overlap[:5]}, "
            f"source_record_identity_overlap={source_record_identity_overlap[:5]}, "
            f"raw_unit_identity_overlap={raw_unit_identity_overlap[:5]}"
        )
    return {
        "verified": True,
        "rule": "raw_partition_unit_scene_source_and_raw_unit_disjoint_v2",
        "unit_id_overlap": [],
        "scene_id_overlap": [],
        "source_asset_sha256_overlap": [],
        "source_record_identity_overlap": [],
        "raw_unit_identity_overlap": [],
        "canonical_payload_sha256_overlap_count": len(payload_overlap),
    }


def _join_and_assess(reference_rows, simulated_rows, tolerances):
    if set(reference_rows) != set(simulated_rows):
        missing = sorted(set(reference_rows).difference(simulated_rows))
        unexpected = sorted(set(simulated_rows).difference(reference_rows))
        raise RuntimeError(
            "RT calibration simulated units differ from the frozen validation reference: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    validated = []
    for unit_id in sorted(reference_rows):
        validated.append(
            {
                "unit_id": unit_id,
                **{
                    f"reference_{statistic}": reference_rows[unit_id][statistic]
                    for statistic in STATISTICS
                },
                **{
                    f"simulated_{statistic}": simulated_rows[unit_id][statistic]
                    for statistic in STATISTICS
                },
                **{
                    f"absolute_error_{statistic}": abs(
                        simulated_rows[unit_id][statistic]
                        - reference_rows[unit_id][statistic]
                    )
                    for statistic in STATISTICS
                },
            }
        )
    assessments = {}
    for statistic in STATISTICS:
        reference = float(np.mean([row[f"reference_{statistic}"] for row in validated]))
        simulated = float(np.mean([row[f"simulated_{statistic}"] for row in validated]))
        tolerance = float(tolerances[statistic])
        per_unit_errors = [row[f"absolute_error_{statistic}"] for row in validated]
        mean_absolute_error = float(np.mean(per_unit_errors))
        assessments[statistic] = {
            "reference_mean": reference,
            "simulated_mean": simulated,
            "mean_absolute_error_per_unit": mean_absolute_error,
            "maximum_absolute_error_per_unit": float(np.max(per_unit_errors)),
            "absolute_tolerance": tolerance,
            "unit_count": len(per_unit_errors),
            "passed": mean_absolute_error <= tolerance,
        }
    return validated, assessments


def _bound_input(path_value, digest, manifest_root, label):
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(manifest_root) / path
    if path.is_symlink():
        raise RuntimeError(f"RT calibration {label} must be a regular file")
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError(f"RT calibration {label} is missing or hash-mismatched")
    return path


def _bound_output_file(output_dir, relative, digest, label):
    if not isinstance(relative, str) or Path(relative).name != relative or not _lower_sha256(digest):
        raise RuntimeError(f"RT calibration {label} reference is invalid")
    candidate = Path(output_dir) / relative
    if candidate.is_symlink():
        raise RuntimeError(f"RT calibration {label} must be a regular file")
    path = candidate.resolve()
    if path.parent != Path(output_dir).resolve() or not path.is_file():
        raise RuntimeError(f"RT calibration {label} is missing or escapes the stage output")
    if sha256_file(path) != digest:
        raise RuntimeError(f"RT calibration {label} hash mismatch")
    return path


def _adapter_environment(project_root):
    root = str(Path(project_root).resolve())
    existing = os.environ.get("PYTHONPATH", "")
    return {
        **os.environ,
        "PYTHONPATH": root if not existing else root + os.pathsep + existing,
    }


def _validate_protocol(protocol):
    required = {
        "schema_version",
        "frozen_utc",
        "absolute_tolerances",
        "exclusion_rules",
        "minimum_validation_units",
        "aggregation",
    }
    if not isinstance(protocol, dict) or set(protocol) != required:
        raise ValueError("RT calibration protocol fields must be exact")
    if protocol["schema_version"] != "csi-pairs-v6-rt-calibration-protocol-v4":
        raise ValueError("RT calibration protocol schema mismatch")
    if not isinstance(protocol["frozen_utc"], str) or not protocol["frozen_utc"].endswith("Z"):
        raise ValueError("RT calibration protocol requires a frozen UTC timestamp")
    tolerances = protocol["absolute_tolerances"]
    if not isinstance(tolerances, dict) or set(tolerances) != set(STATISTICS):
        raise ValueError("RT calibration protocol must freeze all four tolerances")
    if any(not np.isfinite(value) or float(value) < 0 for value in tolerances.values()):
        raise ValueError("RT calibration tolerances must be finite and nonnegative")
    if not isinstance(protocol["exclusion_rules"], list):
        raise ValueError("RT calibration exclusion rules must be frozen")
    if type(protocol["minimum_validation_units"]) is not int or protocol["minimum_validation_units"] < 2:
        raise ValueError("RT calibration requires at least two validation units")
    if protocol["aggregation"] != "mean_absolute_error_per_unit":
        raise ValueError("RT calibration aggregation must be mean_absolute_error_per_unit")


def _lower_sha256(value):
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )
