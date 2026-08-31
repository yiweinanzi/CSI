from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from formal_v2.formal_claim_controls import (
    _authenticated_qualification_gate,
    _validate_formal_checkpoint,
)
from formal_v2.formal_dataset import FormalDataset
from formal_v2.formal_evaluation import _compatibility_dataset, _response_probe_dataset
from formal_v2.formal_evidence import configure_reproducible_runtime, evidence_context
from formal_v2.formal_factorial import _training_normalization
from formal_v2.formal_io import read_strict_json, sha256_file, write_csv, write_json
from formal_v2.formal_probes import (
    fit_action_response_probe,
    fit_select_compatibility_probe,
    predict_binary_probe,
    predict_response_probe,
)
from formal_v2.formal_routing import fit_route_normalization
from formal_v2.formal_teacher import load_teacher_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="First-party frozen-F retention audit for CSI-PAIRS V6"
    )
    parser.add_argument("--dataset", required=True)
    roots = parser.add_mutually_exclusive_group(required=True)
    roots.add_argument("--run-root")
    roots.add_argument("--output-run-root")
    parser.add_argument("--upstream-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--context", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_retention_control(
        args.dataset,
        args.run_root,
        args.output,
        args.context,
        output_run_root=args.output_run_root,
        upstream_root=args.upstream_root,
    )
    return 0


def run_retention_control(
    dataset_path,
    run_root,
    output_root,
    context_path,
    *,
    output_run_root=None,
    upstream_root=None,
) -> Path:
    configure_reproducible_runtime()
    output = Path(output_root).resolve()
    context = read_strict_json(context_path)
    if set(context) != {
        "schema_version", "config", "artifact_label", "dataset_sha256",
        "config_sha256", "fixture", "scientific_use",
        "source_tree_sha256", "requirements_lock_sha256",
        "runtime_provenance_sha256", "runtime_provenance",
    } or context["schema_version"] != "csi-pairs-v6-claim-control-context-v1":
        raise RuntimeError("retention control context fields are not exact")
    config = context["config"]
    dataset = FormalDataset.load(
        dataset_path, require_clean_csi=bool(config["data"]["require_clean_csi"])
    )
    dataset.validate(
        require_clean_csi=bool(config["data"]["require_clean_csi"]),
        minimum_repeats=int(config["data"]["minimum_repeats"]),
        minimum_target_cities=int(config["data"]["minimum_target_cities"]),
        minimum_source_cities=int(config["data"]["minimum_source_cities"]),
        minimum_banks_per_target_city=int(
            config["data"]["minimum_banks_per_target_city"]
        ),
        minimum_independent_base_map_clusters_per_target_city=int(
            config["data"][
                "minimum_independent_base_map_clusters_per_target_city"
            ]
        ),
        minimum_banks_per_source_role=int(
            config["data"]["minimum_banks_per_source_role"]
        ),
        minimum_independent_source_final_unseen_clusters=int(
            config["data"]["minimum_independent_source_final_unseen_clusters"]
        ),
        minimum_independent_external_validation_clusters=int(
            config["data"]["minimum_independent_external_validation_clusters"]
        ),
    )
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    )
    for key in (
        "artifact_label", "dataset_sha256", "config_sha256", "fixture",
        "source_tree_sha256", "requirements_lock_sha256",
        "runtime_provenance_sha256", "runtime_provenance",
    ):
        if context[key] != evidence[key]:
            raise RuntimeError(f"retention control context {key} mismatch")

    output_run, upstream, authenticated = _resolve_run_roots(
        dataset,
        config,
        output,
        run_root=run_root,
        output_run_root=output_run_root,
        upstream_root=upstream_root,
    )
    qualification = _authenticated_qualification_gate(
        config, dataset, authenticated
    )
    teacher_path = Path(str(qualification["teacher_checkpoint"]))
    if (
        not teacher_path.is_file()
        or sha256_file(teacher_path) != qualification["teacher_checkpoint_sha256"]
    ):
        raise RuntimeError("retention control teacher checkpoint is not authenticated")
    teacher = load_teacher_bundle(teacher_path, config)
    route_normalization = fit_route_normalization(dataset, teacher)
    normalization = _training_normalization(
        dataset,
        dataset.indices_for_role("source_encoder_train"),
        route_normalization,
        teacher.patch_spec,
    )

    provenance = {
        "schema_version": "csi-pairs-v6-retention-probe-training-v1",
        "dataset_sha256": evidence["dataset_sha256"],
        "config_sha256": evidence["config_sha256"],
        "fixture": evidence["fixture"],
        "fit_role": "source_probe_train",
        "selection_role": "source_probe_selection",
        "evaluation_roles": ["source_final_unseen_bank", "target"],
        "formal_model_parameters_updated": False,
        "target_labels_used": False,
    }
    provenance_path = output / "probe_training_provenance.json"
    write_json(provenance_path, provenance)
    provenance_sha256 = sha256_file(provenance_path)

    checkpoint_index_path = upstream / "factorial" / "checkpoint_index.json"
    checkpoint_index = read_strict_json(checkpoint_index_path)
    full_rows = {
        int(row["seed"]): row
        for row in checkpoint_index.get("checkpoints", [])
        if row.get("arm") == "full"
    }
    if set(full_rows) != set(map(int, config["seeds"])):
        raise RuntimeError("retention control requires every registered Full checkpoint")
    if any(
        row.get("teacher_checkpoint_sha256")
        != qualification["teacher_checkpoint_sha256"]
        for row in full_rows.values()
    ):
        raise RuntimeError("retention control Full checkpoint teacher hash mismatch")

    registry_path = output_run / "evaluation" / "compatibility_pair_effects.csv"
    registry = _read_registry(registry_path)
    all_rows = []
    probe_rows = []
    for seed in map(int, config["seeds"]):
        checkpoint_row = full_rows[seed]
        checkpoint_path = upstream / "factorial" / checkpoint_row["path"]
        if sha256_file(checkpoint_path) != checkpoint_row["sha256"]:
            raise RuntimeError("retention Full checkpoint hash mismatch")
        payload = _validate_formal_checkpoint(
            checkpoint_path,
            checkpoint_row,
            evidence,
            teacher_checkpoint_sha256=qualification[
                "teacher_checkpoint_sha256"
            ],
            legacy_runtime=authenticated.migrated,
        )
        model = _model_from_payload(payload)

        compatibility_probe, response_probe = _fit_probes(
            config, dataset, teacher, normalization, model, seed
        )
        probe_hashes = {}
        for kind, probe in (
            ("compatibility", compatibility_probe),
            ("response", response_probe),
        ):
            path = output / "probes" / f"seed_{seed}" / f"{kind}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": "csi-pairs-v6-retention-probe-checkpoint-v1",
                    "seed": seed,
                    "kind": kind,
                    "dataset_sha256": evidence["dataset_sha256"],
                    "config_sha256": evidence["config_sha256"],
                    "fixture": evidence["fixture"],
                    "fit_role": "source_probe_train",
                    "selection_role": "source_probe_selection",
                    "training_provenance_sha256": provenance_sha256,
                    "full_checkpoint_sha256": checkpoint_row["sha256"],
                    "state_dict": probe.state_dict(),
                },
                path,
            )
            digest = sha256_file(path)
            probe_hashes[kind] = digest
            probe_rows.append(
                {
                    "seed": seed,
                    "kind": kind,
                    "path": str(path.relative_to(output)),
                    "sha256": digest,
                }
            )

        all_rows.extend(
            _evaluate_retention(
                config,
                dataset,
                teacher,
                normalization,
                model,
                compatibility_probe,
                response_probe,
                seed,
                registry,
                checkpoint_row["sha256"],
                probe_hashes,
            )
        )

    probe_index_path = output / "probe_checkpoint_index.json"
    write_json(
        probe_index_path,
        {
            "schema_version": "csi-pairs-v6-retention-probe-index-v1",
            "dataset_sha256": evidence["dataset_sha256"],
            "config_sha256": evidence["config_sha256"],
            "fixture": evidence["fixture"],
            "training_provenance_sha256": provenance_sha256,
            "probes": probe_rows,
        },
    )
    rows_path = output / "per_unit_results.csv"
    write_csv(rows_path, all_rows)
    result_path = output / "results.json"
    write_json(
        result_path,
        {
            "schema_version": "csi-pairs-v6-retention-results-v3",
            "dataset_sha256": evidence["dataset_sha256"],
            "config_sha256": evidence["config_sha256"],
            "fixture": evidence["fixture"],
            "per_unit_results_path": rows_path.name,
            "per_unit_results_sha256": sha256_file(rows_path),
            "pair_registry_sha256": sha256_file(
                registry_path
            ),
            "checkpoint_index_sha256": sha256_file(checkpoint_index_path),
            "adapter_source_sha256": sha256_file(Path(__file__).resolve()),
            "probe_checkpoint_index_path": probe_index_path.name,
            "probe_checkpoint_index_sha256": sha256_file(probe_index_path),
            "probe_training_provenance_path": provenance_path.name,
            "probe_training_provenance_sha256": provenance_sha256,
        },
    )
    return result_path


def _resolve_run_roots(
    dataset,
    config,
    output,
    *,
    run_root=None,
    output_run_root=None,
    upstream_root=None,
):
    if run_root is not None:
        if output_run_root is not None or upstream_root is not None:
            raise RuntimeError("retention roots cannot mix local and explicit modes")
        local_root = Path(run_root).resolve()
        if (local_root / "migration").exists():
            raise RuntimeError("retention migration requires explicit output and upstream roots")
        output_run_root = local_root
        upstream_root = local_root
    if output_run_root is None or upstream_root is None:
        raise RuntimeError("retention requires output and upstream roots")
    output_run = Path(output_run_root).resolve()
    upstream = Path(upstream_root).resolve()
    if output != output_run and output_run not in output.parents:
        raise RuntimeError("retention output escapes the output run root")
    from formal_v2.formal_upstream import resolve_authenticated_upstream

    authenticated = resolve_authenticated_upstream(config, dataset, output_run)
    expected_upstream = (
        authenticated.migration.legacy_run_root
        if authenticated.migrated
        else authenticated.run_root
    )
    if authenticated.run_root != output_run or expected_upstream != upstream:
        raise RuntimeError("retention output/upstream root binding mismatch")
    return output_run, upstream, authenticated


def _model_from_payload(payload):
    from formal_v2.formal_model import CSIPairsFormalModel

    model = CSIPairsFormalModel(**payload["model_spec"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def _fit_probes(config, dataset, teacher, normalization, model, seed):
    train_scenes = dataset.indices_for_role("source_probe_train")
    selection_scenes = dataset.indices_for_role("source_probe_selection")
    compatibility_train = _compatibility_dataset(
        model, dataset, teacher, config, normalization, train_scenes
    )
    compatibility_selection = _compatibility_dataset(
        model, dataset, teacher, config, normalization, selection_scenes
    )
    compatibility_probe, _ = fit_select_compatibility_probe(
        compatibility_train["features"],
        compatibility_train["labels"],
        compatibility_selection["features"],
        compatibility_selection["labels"],
        config,
        seed=seed + 61001,
    )
    response_train = _response_probe_dataset(
        model, dataset, teacher, config, normalization, train_scenes, active_only=True
    )
    response_probe = fit_action_response_probe(
        response_train["features"],
        response_train["targets"],
        config,
        seed=seed + 62001,
    )
    return compatibility_probe, response_probe


def _read_registry(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("arm") == "full" and row.get("route") == "active"
        ]
    registry = {(int(row["seed"]), row["pair_id"]): row for row in rows}
    if not rows or len(registry) != len(rows):
        raise RuntimeError("retention control pair registry is empty or duplicated")
    return registry


def _evaluate_retention(
    config,
    dataset,
    teacher,
    normalization,
    model,
    compatibility_probe,
    response_probe,
    seed,
    registry,
    checkpoint_sha256,
    probe_hashes,
):
    scenes = np.concatenate(
        (
            dataset.indices_for_role("source_final_unseen_bank"),
            dataset.indices_for_role("target"),
        )
    )
    compatibility = _compatibility_dataset(
        model, dataset, teacher, config, normalization, scenes, active_only=True
    )
    compatibility_scores = {
        "correct": predict_binary_probe(compatibility_probe, compatibility["features"]),
        "map_removed": predict_binary_probe(
            compatibility_probe, compatibility["without_map_features"]
        ),
    }
    compatibility_by_pair = {}
    for pair_id in np.unique(compatibility["pair_ids"]):
        mask = compatibility["pair_ids"] == pair_id
        positive = np.flatnonzero(mask & (compatibility["labels"] == 1))
        negative = np.flatnonzero(mask & (compatibility["labels"] == 0))
        if positive.size != 1 or negative.size != 1:
            raise RuntimeError("retention compatibility pair is not an exact quartet half")
        compatibility_by_pair[str(pair_id)] = {
            "correct": float(compatibility_scores["correct"][positive[0]]),
            "map_swap": float(compatibility_scores["correct"][negative[0]]),
            "map_removed": float(compatibility_scores["map_removed"][positive[0]]),
        }

    response = _response_probe_dataset(
        model, dataset, teacher, config, normalization, scenes, active_only=True
    )
    response_predictions = {
        "correct": predict_response_probe(response_probe, response["features"]),
        "map_swap": predict_response_probe(response_probe, response["map_swap_features"]),
        "map_removed": predict_response_probe(
            response_probe, response["without_map_features"]
        ),
    }
    response_by_transition = {}
    for condition, prediction in response_predictions.items():
        score = -np.mean((prediction - response["targets"]) ** 2, axis=1)
        for index in range(score.shape[0]):
            key = (
                str(response["bank_ids"][index]),
                int(response["source_worlds"][index]),
                int(response["target_worlds"][index]),
                int(response["positions"][index]),
            )
            response_by_transition.setdefault((condition, key), []).append(
                float(score[index])
            )

    rows = []
    expected_keys = {key for key in registry if key[0] == int(seed)}
    for key in sorted(expected_keys):
        registry_row = registry[key]
        pair_id = key[1]
        if pair_id not in compatibility_by_pair:
            raise RuntimeError("retention compatibility replay omitted a registered pair")
        transition = (
            str(registry_row["bank_id"]),
            int(registry_row["source_world"]),
            int(registry_row["target_world"]),
            int(registry_row["position_index"]),
        )
        for condition in ("correct", "map_swap", "map_removed"):
            response_values = response_by_transition.get((condition, transition))
            if not response_values:
                raise RuntimeError("retention response replay omitted a registered transition")
            rows.append(
                {
                    "seed": seed,
                    "pair_id": pair_id,
                    "condition": condition,
                    "cgs_score": compatibility_by_pair[pair_id][condition],
                    "response_score": float(np.mean(response_values)),
                    "checkpoint_sha256": checkpoint_sha256,
                    "compatibility_probe_sha256": probe_hashes["compatibility"],
                    "response_probe_sha256": probe_hashes["response"],
                    "representation_scope": "formal_model.retained_representation",
                }
            )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
