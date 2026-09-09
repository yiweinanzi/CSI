from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from formal_v2.formal_claim_controls import (
    SHUFFLED_SYSTEMS,
    SHUFFLED_FIXTURE_SUPPORT_RESULT_SCHEMA,
    SHUFFLED_FIXTURE_SUPPORT_STATUS,
    _registry_action_sha256,
    _validate_formal_checkpoint,
)
from formal_v2.formal_checkpoint_contract import (
    SHUFFLED_CHECKPOINT_FIELDS,
    SHUFFLED_CHECKPOINT_SCHEMA,
)
from formal_v2.formal_dataset import FormalDataset
from formal_v2.formal_evaluation import _compatibility_dataset, _response_probe_dataset
from formal_v2.formal_evidence import configure_reproducible_runtime, evidence_context
from formal_v2.formal_factorial import (
    _build_corpus,
    _model_spec,
    _normalization_record,
    _train_arm,
    _training_normalization,
)
from formal_v2.formal_io import read_strict_json, sha256_file, write_csv, write_json
from formal_v2.formal_probes import (
    fit_action_response_probe,
    fit_select_compatibility_probe,
    predict_binary_probe,
    predict_response_probe,
)
from formal_v2.formal_routing import fit_route_normalization
from formal_v2.formal_teacher import load_teacher_bundle


SHORTCUT_FEATURES = {
    "constant": "constant_shortcut_features",
    "csi_only": "csi_only_shortcut_features",
    "map_only": "map_only_shortcut_features",
    "scene_id_only": "scene_id_only_shortcut_features",
    "edit_status_xor": "edit_status_xor_shortcut_features",
    "variant_id_matcher": "variant_id_match_shortcut_features",
}


class InsufficientDerangementSupport(RuntimeError):
    def __init__(self, branch: str):
        self.branch = branch
        super().__init__(
            f"{branch} has no within-scene/edit-family/route/effect-bucket derangement"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="First-party two-branch shuffled-pair training control"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--control-seed", type=int, required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_shuffled_pair_control(
        args.dataset,
        args.run_root,
        args.output,
        args.context,
        control_seed=args.control_seed,
    )
    return 0


def run_shuffled_pair_control(
    dataset_path, run_root, output_root, context_path, *, control_seed
) -> Path:
    configure_reproducible_runtime()
    root = Path(run_root).resolve()
    output = Path(output_root).resolve()
    context = read_strict_json(context_path)
    if set(context) != {
        "schema_version", "config", "artifact_label", "dataset_sha256",
        "config_sha256", "fixture", "scientific_use",
        "source_tree_sha256", "requirements_lock_sha256",
        "runtime_provenance_sha256", "runtime_provenance",
    } or context["schema_version"] != "csi-pairs-v6-claim-control-context-v1":
        raise RuntimeError("shuffled control context fields are not exact")
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
            raise RuntimeError(f"shuffled control context {key} mismatch")

    qualification = read_strict_json(root / "qualification" / "gate.json")
    teacher_path = Path(str(qualification["teacher_checkpoint"]))
    if (
        not teacher_path.is_file()
        or sha256_file(teacher_path) != qualification["teacher_checkpoint_sha256"]
    ):
        raise RuntimeError("shuffled control teacher checkpoint is not authenticated")
    teacher = load_teacher_bundle(teacher_path, config)
    route_normalization = fit_route_normalization(dataset, teacher)
    normalization = _training_normalization(
        dataset,
        dataset.indices_for_role("source_encoder_train"),
        route_normalization,
        teacher.patch_spec,
    )
    corpus = _build_corpus(
        dataset,
        dataset.indices_for_role("source_encoder_train"),
        teacher,
        config,
        route_normalization,
        normalization,
    )
    pilot = read_strict_json(root / "factorial" / "frozen_pilot.json")
    _validate_pilot(pilot)

    checkpoint_index_path = root / "factorial" / "checkpoint_index.json"
    checkpoint_index = read_strict_json(checkpoint_index_path)
    matched_rows = {
        int(row["seed"]): row
        for row in checkpoint_index.get("checkpoints", [])
        if row.get("arm") == "full"
    }
    if set(matched_rows) != set(map(int, config["seeds"])):
        raise RuntimeError("shuffled control requires every registered Full checkpoint")

    pairing_by_seed = {}
    permutation_rows = []
    try:
        for seed in map(int, config["seeds"]):
            pairing, rows = _build_pairing_break(corpus, seed, int(control_seed))
            pairing_by_seed[seed] = pairing
            permutation_rows.extend(rows)
    except InsufficientDerangementSupport as error:
        if not dataset.is_fixture:
            raise
        return _write_fixture_support_result(
            output,
            root,
            evidence,
            checkpoint_index_path,
            error,
        )
    permutation_path = output / "pairing_permutations.csv"
    write_csv(permutation_path, permutation_rows)

    training_rows = []
    shuffled_models = {}
    for seed in map(int, config["seeds"]):
        model, training_row = _train_arm(
            config,
            corpus,
            seed,
            "full",
            pilot,
            pairing_break=pairing_by_seed[seed],
        )
        training_rows.append(training_row)
        shuffled_models[seed] = model
    training_summary_path = output / "training_summary.csv"
    write_csv(training_summary_path, training_rows)

    provenance_path = output / "training_provenance.json"
    write_json(
        provenance_path,
        {
            "schema_version": "csi-pairs-v6-shuffled-training-provenance-v1",
            "dataset_sha256": evidence["dataset_sha256"],
            "config_sha256": evidence["config_sha256"],
            "fixture": evidence["fixture"],
            "control_seed": int(control_seed),
            "training_roles": ["source_encoder_train"],
            "selection_role": "source_method_selection",
            "target_roles_used": [],
            "pairing_breaks": ["alignment_h_map_edge", "response_action_target"],
            "pairing_strata": ["scene", "edit_family", "effect_bucket"],
            "loss_form_preserved": True,
            "training_steps": int(config["factorial"]["steps"]),
            "permutation_registry_path": permutation_path.name,
            "permutation_registry_sha256": sha256_file(permutation_path),
            "training_summary_path": training_summary_path.name,
            "training_summary_sha256": sha256_file(training_summary_path),
        },
    )
    provenance_sha256 = sha256_file(provenance_path)

    shuffled_checkpoint_rows = []
    for seed in map(int, config["seeds"]):
        path = output / "checkpoints" / f"seed_{seed}" / "full_shuffled.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": SHUFFLED_CHECKPOINT_SCHEMA,
                "arm": "full",
                "seed": seed,
                "model_spec": _model_spec(config, corpus),
                "normalization": _normalization_record(normalization),
                "teacher_checkpoint_sha256": qualification[
                    "teacher_checkpoint_sha256"
                ],
                "checkpoint_rule": "fixed_final_step_no_target_selection",
                "state_dict": shuffled_models[seed].state_dict(),
                **evidence,
                "pairing_breaks": [
                    "alignment_h_map_edge",
                    "response_action_target",
                ],
                "training_provenance_sha256": provenance_sha256,
            },
            path,
        )
        reloaded = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(reloaded, dict)
            or set(reloaded) != SHUFFLED_CHECKPOINT_FIELDS
            or reloaded.get("schema_version") != SHUFFLED_CHECKPOINT_SCHEMA
            or int(reloaded.get("seed", -1)) != seed
        ):
            raise RuntimeError("shuffled checkpoint changed during serialization")
        shuffled_checkpoint_rows.append(
            {
                "seed": seed,
                "path": str(path.relative_to(output)),
                "sha256": sha256_file(path),
                "matched_checkpoint_sha256": matched_rows[seed]["sha256"],
            }
        )
    shuffled_index_path = output / "shuffled_checkpoint_index.json"
    write_json(
        shuffled_index_path,
        {
            "schema_version": "csi-pairs-v6-shuffled-checkpoint-index-v1",
            "dataset_sha256": evidence["dataset_sha256"],
            "config_sha256": evidence["config_sha256"],
            "fixture": evidence["fixture"],
            "training_provenance_sha256": provenance_sha256,
            "checkpoints": shuffled_checkpoint_rows,
        },
    )

    registry_path = root / "evaluation" / "compatibility_pair_effects.csv"
    registry = _read_registry(registry_path)
    result_rows = []
    shuffled_hashes = {
        int(row["seed"]): row["sha256"] for row in shuffled_checkpoint_rows
    }
    for seed in map(int, config["seeds"]):
        matched_path = root / "factorial" / matched_rows[seed]["path"]
        if sha256_file(matched_path) != matched_rows[seed]["sha256"]:
            raise RuntimeError("shuffled control matched checkpoint hash mismatch")
        matched_payload = _validate_formal_checkpoint(
            matched_path, matched_rows[seed], evidence
        )
        matched_model = _model_from_payload(matched_payload)
        result_rows.extend(
            _evaluate_seed(
                config,
                dataset,
                teacher,
                normalization,
                seed,
                matched_model,
                shuffled_models[seed],
                registry,
                matched_rows[seed]["sha256"],
                shuffled_hashes[seed],
            )
        )

    rows_path = output / "per_unit_results.csv"
    write_csv(rows_path, result_rows)
    result_path = output / "results.json"
    write_json(
        result_path,
        {
            "schema_version": "csi-pairs-v6-shuffled-pair-results-v3",
            "dataset_sha256": evidence["dataset_sha256"],
            "config_sha256": evidence["config_sha256"],
            "fixture": evidence["fixture"],
            "per_unit_results_path": rows_path.name,
            "per_unit_results_sha256": sha256_file(rows_path),
            "pair_registry_sha256": sha256_file(registry_path),
            "checkpoint_index_sha256": sha256_file(checkpoint_index_path),
            "adapter_source_sha256": sha256_file(Path(__file__).resolve()),
            "shuffled_checkpoint_index_path": shuffled_index_path.name,
            "shuffled_checkpoint_index_sha256": sha256_file(shuffled_index_path),
            "training_provenance_path": provenance_path.name,
            "training_provenance_sha256": provenance_sha256,
        },
    )
    return result_path


def _validate_pilot(pilot):
    if (
        pilot.get("schema_version") != "csi-pairs-v6-frozen-pilot-v1"
        or pilot.get("source_roles")
        != ["source_encoder_train", "source_method_selection"]
        or pilot.get("checkpoint_reused_for_final_training") is not False
    ):
        raise RuntimeError("shuffled control pilot is not the frozen source-only pilot")
    for key in ("alignment_scale", "response_scale"):
        if not np.isfinite(float(pilot.get(key, np.nan))) or float(pilot[key]) <= 0:
            raise RuntimeError("shuffled control pilot scale is invalid")
    null_tolerance = float(pilot.get("alignment_null_tolerance", np.nan))
    if not np.isfinite(null_tolerance) or null_tolerance < 0:
        raise RuntimeError("shuffled control pilot scale is invalid")


def _write_fixture_support_result(
    output, run_root, evidence, checkpoint_index_path, error
):
    registry_path = Path(run_root) / "evaluation" / "compatibility_pair_effects.csv"
    if not registry_path.is_file() or not checkpoint_index_path.is_file():
        raise RuntimeError("shuffled fixture support preflight inputs are missing")
    result_path = Path(output) / "results.json"
    write_json(
        result_path,
        {
            "schema_version": SHUFFLED_FIXTURE_SUPPORT_RESULT_SCHEMA,
            "status": SHUFFLED_FIXTURE_SUPPORT_STATUS,
            "dataset_sha256": evidence["dataset_sha256"],
            "config_sha256": evidence["config_sha256"],
            "fixture": evidence["fixture"],
            "scientific_use": evidence["scientific_use"],
            "reason": str(error),
            "failed_branch": error.branch,
            "required_pairing_strata": [
                "scene",
                "edit_family",
                "route",
                "effect_bucket",
            ],
            "pair_registry_sha256": sha256_file(registry_path),
            "checkpoint_index_sha256": sha256_file(checkpoint_index_path),
            "adapter_source_sha256": sha256_file(Path(__file__).resolve()),
        },
    )
    return result_path


def _model_from_payload(payload):
    from formal_v2.formal_model import CSIPairsFormalModel

    model = CSIPairsFormalModel(**payload["model_spec"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def _build_pairing_break(corpus, seed, control_seed):
    alignment_units = sorted(
        {
            (int(scene), *map(int, unit))
            for table in (corpus.alignment_active, corpus.alignment_null)
            for scene, units in table.items()
            for unit in units
        }
    )
    response_units = sorted(
        {
            (int(scene), *map(int, unit))
            for scene, units in corpus.response_all.items()
            for unit in units
        }
    )
    alignment, alignment_rows = _stratified_derangement(
        corpus, alignment_units, "alignment_h_map_edge", seed, control_seed
    )
    response, response_rows = _stratified_derangement(
        corpus, response_units, "response_action_target", seed, control_seed
    )
    return {"alignment": alignment, "response": response}, alignment_rows + response_rows


def _stratified_derangement(corpus, units, branch, seed, control_seed):
    preliminary = {}
    effects = {}
    for unit in units:
        scene, source, target, position, *query_value = unit
        edge = _edge(corpus.dataset, scene, source, target)
        query = query_value[0] if query_value else None
        if branch == "alignment_h_map_edge":
            route = int(corpus.routed.alignment_route[(scene, source, target, position)])
            effect = float(
                corpus.routed.alignment_distances[(scene, source, target, position)][0]
            )
        else:
            route = int(
                corpus.routed.response_route[
                    (scene, source, target, position, int(query))
                ]
            )
            effect = float(
                corpus.routed.response_distances[
                    (scene, source, target, position, int(query))
                ][0]
            )
        family = f"primitive-{int(edge.primitive_id)}-direction-{int(edge.direction)}"
        base = (scene, family, route, query)
        preliminary.setdefault(base, []).append(unit)
        effects[unit] = effect

    groups = {}
    for base, values in preliminary.items():
        ordered = sorted(values, key=lambda unit: (effects[unit], unit))
        bucket_count = _bucket_count(ordered, branch)
        for rank, unit in enumerate(ordered):
            bucket = min(bucket_count - 1, rank * bucket_count // len(ordered))
            groups.setdefault((*base, bucket), []).append(unit)

    mapping = {}
    rows = []
    rng = np.random.default_rng(
        int(control_seed) * 1000003 + int(seed) * 9176 + (1 if branch.startswith("alignment") else 2)
    )
    for group, values in sorted(groups.items()):
        ordered = [values[index] for index in rng.permutation(len(values))]
        partners = _valid_rotation(ordered, branch)
        scene, family, route, query, bucket = group
        for original, partner in zip(ordered, partners):
            mapping[tuple(original)] = tuple(partner)
            rows.append(
                {
                    "seed": int(seed),
                    "branch": branch,
                    "scene_id": str(corpus.dataset.scene_ids[int(scene)]),
                    "edit_family": family,
                    "effect_bucket": f"route-{route}-adaptive-quantile-{bucket}",
                    "original_pair_id": _unit_id(original),
                    "permuted_pair_id": _unit_id(partner),
                }
            )
    if set(mapping) != set(units):
        raise RuntimeError(f"{branch} permutation does not cover every training unit")
    return mapping, rows


def _bucket_count(values, branch):
    for count in range(min(4, len(values) // 2), 0, -1):
        chunks = [
            values[index * len(values) // count : (index + 1) * len(values) // count]
            for index in range(count)
        ]
        if all(_group_can_derange(chunk, branch) for chunk in chunks):
            return count
    raise InsufficientDerangementSupport(branch)


def _group_can_derange(values, branch):
    if len(values) < 2:
        return False
    if branch == "alignment_h_map_edge":
        return len({(unit[1], unit[2]) for unit in values}) >= 2
    return True


def _valid_rotation(values, branch):
    for offset in range(1, len(values)):
        partners = values[offset:] + values[:offset]
        if all(
            original != partner
            and (
                branch != "alignment_h_map_edge"
                or (original[1], original[2]) != (partner[1], partner[2])
            )
            for original, partner in zip(values, partners)
        ):
            return partners
    raise RuntimeError(f"{branch} stratum cannot form a complete derangement")


def _edge(dataset, scene, source, target):
    matches = [
        edge
        for edge in dataset.directed_edges(int(scene))
        if int(edge.source_world) == int(source) and int(edge.target_world) == int(target)
    ]
    if len(matches) != 1:
        raise RuntimeError("shuffled training unit does not identify one directed edge")
    return matches[0]


def _unit_id(unit):
    return ":".join(map(str, unit))


def _read_registry(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("arm") == "full" and row.get("route") == "active"
        ]
    registry = {(int(row["seed"]), row["pair_id"]): row for row in rows}
    if not rows or len(registry) != len(rows):
        raise RuntimeError("shuffled control pair registry is empty or duplicated")
    return registry


def _fit_compatibility(config, dataset, teacher, normalization, model, seed):
    train = _compatibility_dataset(
        model,
        dataset,
        teacher,
        config,
        normalization,
        dataset.indices_for_role("source_probe_train"),
    )
    selection = _compatibility_dataset(
        model,
        dataset,
        teacher,
        config,
        normalization,
        dataset.indices_for_role("source_probe_selection"),
    )
    probe, _ = fit_select_compatibility_probe(
        train["features"],
        train["labels"],
        selection["features"],
        selection["labels"],
        config,
        seed=seed,
    )
    return probe, train, selection


def _evaluate_seed(
    config,
    dataset,
    teacher,
    normalization,
    seed,
    matched_model,
    shuffled_model,
    registry,
    matched_checkpoint_sha256,
    shuffled_checkpoint_sha256,
):
    scenes = np.concatenate(
        (
            dataset.indices_for_role("source_final_unseen_bank"),
            dataset.indices_for_role("target"),
        )
    )
    alignment_scores = {}
    matched_probe, shortcut_train, shortcut_selection = _fit_compatibility(
        config, dataset, teacher, normalization, matched_model, seed + 71001
    )
    shuffled_probe, _, _ = _fit_compatibility(
        config, dataset, teacher, normalization, shuffled_model, seed + 72001
    )
    matched_evaluation = _compatibility_dataset(
        matched_model, dataset, teacher, config, normalization, scenes, active_only=True
    )
    shuffled_evaluation = _compatibility_dataset(
        shuffled_model, dataset, teacher, config, normalization, scenes, active_only=True
    )
    alignment_scores["matched_model"] = _pair_scores(
        matched_evaluation,
        predict_binary_probe(matched_probe, matched_evaluation["features"]),
    )
    alignment_scores["shuffled_model"] = _pair_scores(
        shuffled_evaluation,
        predict_binary_probe(shuffled_probe, shuffled_evaluation["features"]),
    )
    for index, (system, feature_name) in enumerate(SHORTCUT_FEATURES.items()):
        probe, _ = fit_select_compatibility_probe(
            shortcut_train[feature_name],
            shortcut_train["labels"],
            shortcut_selection[feature_name],
            shortcut_selection["labels"],
            config,
            seed=seed + 73001 + index,
        )
        alignment_scores[system] = _pair_scores(
            matched_evaluation,
            predict_binary_probe(probe, matched_evaluation[feature_name]),
        )

    response_scores = {
        "matched_model": _response_scores(
            config, dataset, teacher, normalization, matched_model, seed + 74001, scenes
        ),
        "shuffled_model": _response_scores(
            config, dataset, teacher, normalization, shuffled_model, seed + 75001, scenes
        ),
    }
    rows = []
    for key in sorted(key for key in registry if key[0] == int(seed)):
        pair_id = key[1]
        registry_row = registry[key]
        action_sha256 = _registry_action_sha256(dataset, registry_row)
        common = {
            "seed": seed,
            "pair_id": pair_id,
            "matched_checkpoint_sha256": matched_checkpoint_sha256,
            "shuffled_checkpoint_sha256": shuffled_checkpoint_sha256,
            "action_sha256": action_sha256,
        }
        for system in SHUFFLED_SYSTEMS:
            scores = alignment_scores.get(system, {}).get(pair_id)
            if scores is None:
                raise RuntimeError("shuffled Alignment replay omitted a registered pair")
            for label in ("positive", "negative"):
                rows.append(
                    {
                        **common,
                        "metric": "alignment_cgs",
                        "system": system,
                        "pair_label": label,
                        "score": scores[label],
                    }
                )
        transition = (
            str(registry_row["bank_id"]),
            int(registry_row["source_world"]),
            int(registry_row["target_world"]),
            int(registry_row["position_index"]),
        )
        for system in SHUFFLED_SYSTEMS[:2]:
            scores = response_scores[system].get(transition)
            if scores is None:
                raise RuntimeError("shuffled Response replay omitted a registered transition")
            for label in ("positive", "negative"):
                rows.append(
                    {
                        **common,
                        "metric": "response_probe",
                        "system": system,
                        "pair_label": label,
                        "score": scores[label],
                    }
                )
    return rows


def _pair_scores(evaluated, probabilities):
    output = {}
    for pair_id in np.unique(evaluated["pair_ids"]):
        mask = evaluated["pair_ids"] == pair_id
        positive = probabilities[mask & (evaluated["labels"] == 1)]
        negative = probabilities[mask & (evaluated["labels"] == 0)]
        if positive.size != 1 or negative.size != 1:
            raise RuntimeError("shuffled Alignment pair is not exactly balanced")
        output[str(pair_id)] = {
            "positive": float(positive[0]),
            "negative": float(negative[0]),
        }
    return output


def _response_scores(config, dataset, teacher, normalization, model, seed, scenes):
    train = _response_probe_dataset(
        model,
        dataset,
        teacher,
        config,
        normalization,
        dataset.indices_for_role("source_probe_train"),
        active_only=True,
    )
    probe = fit_action_response_probe(
        train["features"], train["targets"], config, seed=seed
    )
    evaluated = _response_probe_dataset(
        model, dataset, teacher, config, normalization, scenes, active_only=True
    )
    prediction = predict_response_probe(probe, evaluated["features"])
    model_score = -np.mean((prediction - evaluated["targets"]) ** 2, axis=1)
    copy_score = -np.mean(
        (evaluated["source_targets"] - evaluated["targets"]) ** 2, axis=1
    )
    grouped = {}
    for index in range(model_score.shape[0]):
        key = (
            str(evaluated["bank_ids"][index]),
            int(evaluated["source_worlds"][index]),
            int(evaluated["target_worlds"][index]),
            int(evaluated["positions"][index]),
        )
        grouped.setdefault(key, {"positive": [], "negative": []})
        grouped[key]["positive"].append(float(model_score[index]))
        grouped[key]["negative"].append(float(copy_score[index]))
    return {
        key: {label: float(np.mean(values)) for label, values in scores.items()}
        for key, scores in grouped.items()
    }


if __name__ == "__main__":
    raise SystemExit(main())
