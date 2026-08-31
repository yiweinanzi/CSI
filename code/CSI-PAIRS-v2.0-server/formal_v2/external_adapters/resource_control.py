from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional

from formal_v2.formal_dataset import FormalDataset
from formal_v2.formal_evidence import evidence_context
from formal_v2.formal_factorial import (
    _build_corpus,
    _canonical_bank_digest,
    _make_plan,
    _loss_weights,
    _measure_execution,
    _model_spec,
    _natural_representations,
    _new_model,
    _stable_city_seed,
    _train_arm,
    _training_normalization,
    city_support_candidates,
    eligible_query_indices,
)
from formal_v2.formal_io import read_strict_json, sha256_file, write_csv, write_json
from formal_v2.formal_localization import (
    HeteroscedasticPositionHead,
    adapt_position_head,
    predict_position_distribution,
)
from formal_v2.formal_routing import fit_route_normalization
from formal_v2.formal_teacher import load_teacher_bundle


CONTROL_ROLES = {
    "equal_flop_alignment": ("alignment",),
    "equal_flop_response": ("response",),
    "parameter_matched_concat": ("alignment", "response"),
    "flop_matched_concat": ("alignment", "response"),
    "generous_2x_concat": ("alignment", "response"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="First-party CSI-PAIRS V6 resource-control runner"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--control-id", required=True, choices=tuple(CONTROL_ROLES))
    run.add_argument("--dataset", required=True)
    roots = run.add_mutually_exclusive_group(required=True)
    roots.add_argument("--run-root")
    roots.add_argument("--output-run-root")
    run.add_argument("--upstream-root")
    run.add_argument("--output", required=True)
    run.add_argument("--architecture-spec", required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--control-id", required=True, choices=tuple(CONTROL_ROLES))
    replay.add_argument("--output", required=True)
    replay.add_argument("--architecture-spec", required=True)
    replay.add_argument("--replay-output", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "run":
        run_control(
            args.control_id,
            args.dataset,
            args.run_root,
            args.output,
            args.architecture_spec,
            output_run_root=args.output_run_root,
            upstream_root=args.upstream_root,
        )
    else:
        replay_control(
            args.control_id,
            args.output,
            args.architecture_spec,
            args.replay_output,
        )
    return 0


def run_control(
    control_id,
    dataset_path,
    run_root,
    output_root,
    architecture_path,
    *,
    output_run_root=None,
    upstream_root=None,
):
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    architecture_path = Path(architecture_path).resolve()
    architecture = read_strict_json(architecture_path)
    if architecture.get("control_id") != control_id:
        raise RuntimeError("resource-control architecture identity mismatch")
    provisional_root = upstream_root if upstream_root is not None else run_root
    if provisional_root is None:
        raise RuntimeError("resource control requires output and upstream roots")
    qualification = read_strict_json(
        Path(provisional_root).resolve() / "qualification" / "gate.json"
    )
    config = qualification["config"]
    dataset = FormalDataset.load(
        dataset_path,
        require_clean_csi=bool(config["data"]["require_clean_csi"]),
    )
    _output_run, upstream, _authenticated = _resolve_run_roots(
        dataset,
        config,
        output,
        run_root=run_root,
        output_run_root=output_run_root,
        upstream_root=upstream_root,
    )
    qualification = read_strict_json(upstream / "qualification" / "gate.json")
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    )
    teacher_path = Path(qualification["teacher_checkpoint"])
    if sha256_file(teacher_path) != qualification["teacher_checkpoint_sha256"]:
        raise RuntimeError("resource control teacher checkpoint is not authenticated")
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
    pilot = read_strict_json(upstream / "factorial" / "frozen_pilot.json")
    training_rows = _factorial_training_rows(
        upstream / "factorial" / "training_summary.csv"
    )
    model_specs, step_counts = _select_control_design(
        control_id, architecture, config, corpus, pilot, training_rows
    )
    source_hash = sha256_file(Path(__file__).resolve())
    architecture_hash = sha256_file(architecture_path)
    records = []
    localization_rows = []
    for seed in map(int, config["seeds"]):
        started = time.perf_counter()
        components = {}
        component_traces = {}
        component_rows = {}
        for role in CONTROL_ROLES[control_id]:
            trace = []
            model, row = _train_arm(
                config,
                corpus,
                seed,
                role,
                pilot,
                step_count=step_counts[role],
                model_spec=model_specs[role],
                loss_trace=trace,
            )
            components[role] = model
            component_traces[role] = trace
            component_rows[role] = row

        representations = _control_representations(
            control_id,
            components,
            dataset,
            normalization,
            teacher.patch_spec,
        )
        bottleneck, source_head, localization_trace, bottleneck_training_flops = (
            _fit_source_localizer(control_id, representations, dataset, config, seed)
        )
        localization_rows.extend(
            _evaluate_localization(
                control_id,
                representations,
                dataset,
                config,
                seed,
                bottleneck,
                source_head,
                evidence["dataset_sha256"],
                evidence["config_sha256"],
                evidence["fixture"],
            )
        )
        checkpoint_dir = output / "checkpoints" / f"seed_{seed}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / "control.pt"
        payload = {
            "schema_version": "csi-pairs-v6-control-checkpoint-v3",
            "control_id": control_id,
            "architecture_family": architecture["architecture_family"],
            "architecture_spec_sha256": architecture_hash,
            "adapter_source_sha256": source_hash,
            "dataset_sha256": evidence["dataset_sha256"],
            "config_sha256": evidence["config_sha256"],
            "fixture": evidence["fixture"],
            "seed": seed,
            "checkpoint_rule": "fixed_final_step_no_target_selection",
            "components": [
                {
                    "role": role,
                    "model_spec": model_specs[role],
                    "state_dict": components[role].state_dict(),
                }
                for role in CONTROL_ROLES[control_id]
            ],
            "bottleneck_state_dict": (
                None if bottleneck is None else bottleneck.state_dict()
            ),
            "source_head_state_dict": source_head.state_dict(),
        }
        torch.save(payload, checkpoint)
        checkpoint_hash = sha256_file(checkpoint)

        loss_paths = {}
        for role, trace in component_traces.items():
            path = checkpoint_dir / f"{role}_loss_trace.csv"
            write_csv(path, trace)
            loss_paths[role] = {
                "path": str(path.relative_to(output)),
                "sha256": sha256_file(path),
            }
        localization_path = checkpoint_dir / "localization_loss_trace.csv"
        write_csv(localization_path, localization_trace)
        training_log = checkpoint_dir / "training_log.json"
        write_json(
            training_log,
            {
                "schema_version": "csi-pairs-v6-control-training-log-v3",
                "control_id": control_id,
                "seed": seed,
                "component_optimizer_steps": {
                    role: int(component_rows[role]["steps"])
                    for role in CONTROL_ROLES[control_id]
                },
                "component_objectives": {
                    role: ["endpoint", role]
                    for role in CONTROL_ROLES[control_id]
                },
                "source_training_role": "source_encoder_train",
                "selection_role": "source_method_selection",
                "localizer_fit_role": "source_encoder_train",
                "target_selection_used": False,
                "fixed_final_checkpoint": True,
                "checkpoint_sha256": checkpoint_hash,
                "component_loss_traces": loss_paths,
                "localization_loss_trace_path": str(
                    localization_path.relative_to(output)
                ),
                "localization_loss_trace_sha256": sha256_file(localization_path),
            },
        )

        events = []
        for role, row in component_rows.items():
            training_flops = float(row["measured_flops_per_step"]) * int(row["steps"])
            inference_flops = _retained_inference_flops(
                components[role], dataset, normalization, teacher.patch_spec
            )
            events.extend(
                (
                    {
                        "name": f"{role}_training_graph",
                        "phase": "training",
                        "count": 1,
                        "flops": training_flops,
                    },
                    {
                        "name": f"{role}_retained_inference_graph",
                        "phase": "inference",
                        "count": 1,
                        "flops": inference_flops,
                    },
                )
            )
        if bottleneck is not None:
            events.extend(
                (
                    {
                        "name": "concat_bottleneck_training_graph",
                        "phase": "training",
                        "count": 1,
                        "flops": bottleneck_training_flops,
                    },
                    {
                        "name": "concat_bottleneck_inference_graph",
                        "phase": "inference",
                        "count": 1,
                        "flops": float(
                            2 * bottleneck.in_features * bottleneck.out_features
                        ),
                    },
                )
            )
        training_flops = float(
            sum(row["flops"] for row in events if row["phase"] == "training")
        )
        inference_flops = float(
            sum(row["flops"] for row in events if row["phase"] == "inference")
        )
        profiler = checkpoint_dir / "profiler_trace.json"
        write_json(
            profiler,
            {
                "schema_version": "csi-pairs-v6-profiler-summary-v3",
                "profiler": "torch.utils.flop_counter.FlopCounterMode",
                "resource_accounting_scope": (
                    "representation_training_and_retained_inference_"
                    "excluding_common_localization_head"
                ),
                "events": events,
                "measured_training_flops": training_flops,
                "measured_inference_flops": inference_flops,
                "profiled_step_count": len(events),
            },
        )
        parameter_count = _payload_resource_parameter_count(payload)
        records.append(
            {
                "seed": seed,
                "training_flops": training_flops,
                "inference_flops": inference_flops,
                "parameters": parameter_count,
                "wall_seconds": time.perf_counter() - started,
                "checkpoint_path": str(checkpoint.relative_to(output)),
                "checkpoint_sha256": checkpoint_hash,
                "profiler_trace_path": str(profiler.relative_to(output)),
                "profiler_trace_sha256": sha256_file(profiler),
                "training_log_path": str(training_log.relative_to(output)),
                "training_log_sha256": sha256_file(training_log),
            }
        )

    write_csv(output / "localization_per_bank.csv", localization_rows)
    index = {
        "schema_version": "csi-pairs-v6-resource-index-v3",
        "control_id": control_id,
        "dataset_sha256": evidence["dataset_sha256"],
        "config_sha256": evidence["config_sha256"],
        "fixture": evidence["fixture"],
        "architecture_spec_sha256": architecture_hash,
        "adapter_source_sha256": source_hash,
        "records": records,
    }
    write_json(output / "resource_index.json", index)
    return index


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
            raise RuntimeError("resource-control roots cannot mix local and explicit modes")
        local_root = Path(run_root).resolve()
        if (local_root / "migration").exists():
            raise RuntimeError(
                "resource-control migration requires explicit output and upstream roots"
            )
        output_run_root = local_root
        upstream_root = local_root
    if output_run_root is None or upstream_root is None:
        raise RuntimeError("resource control requires output and upstream roots")
    output_run = Path(output_run_root).resolve()
    upstream = Path(upstream_root).resolve()
    if output != output_run and output_run not in output.parents:
        raise RuntimeError("resource-control output escapes the output run root")
    from formal_v2.formal_upstream import resolve_authenticated_upstream

    authenticated = resolve_authenticated_upstream(config, dataset, output_run)
    expected_upstream = (
        authenticated.migration.legacy_run_root
        if authenticated.migrated
        else authenticated.run_root
    )
    if authenticated.run_root != output_run or expected_upstream != upstream:
        raise RuntimeError("resource-control output/upstream root binding mismatch")
    return output_run, upstream, authenticated


def replay_control(control_id, output_root, architecture_path, replay_output):
    output = Path(output_root).resolve()
    architecture_path = Path(architecture_path).resolve()
    index_path = output / "resource_index.json"
    index = read_strict_json(index_path)
    if index.get("control_id") != control_id:
        raise RuntimeError("resource replay control identity mismatch")
    parameters = []
    training_flops = []
    inference_flops = []
    for record in index["records"]:
        checkpoint = output / record["checkpoint_path"]
        if sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise RuntimeError("resource replay checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        parameters.append(_payload_resource_parameter_count(payload))
        profiler = read_strict_json(output / record["profiler_trace_path"])
        totals = {
            phase: float(
                sum(
                    int(event["count"]) * float(event["flops"])
                    for event in profiler["events"]
                    if event["phase"] == phase
                )
            )
            for phase in ("training", "inference")
        }
        training_flops.append(totals["training"])
        inference_flops.append(totals["inference"])
    replay = {
        "schema_version": "csi-pairs-v6-resource-replay-v2",
        "control_id": control_id,
        "resource_index_sha256": sha256_file(index_path),
        "architecture_spec_sha256": sha256_file(architecture_path),
        "adapter_source_sha256": sha256_file(Path(__file__).resolve()),
        "localization_per_bank_sha256": sha256_file(
            output / "localization_per_bank.csv"
        ),
        "recomputed_parameters": float(np.mean(parameters)),
        "recomputed_training_flops": float(np.mean(training_flops)),
        "recomputed_inference_flops": float(np.mean(inference_flops)),
    }
    write_json(replay_output, replay)
    return replay


def _factorial_training_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("resource controls require factorial training summaries")
    return rows


def _select_control_design(control_id, architecture, config, corpus, pilot, rows):
    base = _model_spec(config, corpus)
    full_rows = [row for row in rows if row["arm"] == "full"]
    if not full_rows:
        raise RuntimeError("resource controls require Full resource measurements")
    target_flops = float(
        np.mean(
            [float(row["measured_flops_per_step"]) * int(row["steps"]) for row in full_rows]
        )
    )
    roles = CONTROL_ROLES[control_id]
    if control_id.startswith("equal_flop"):
        role = roles[0]
        reference = [row for row in rows if row["arm"] == role]
        per_step = float(np.mean([float(row["measured_flops_per_step"]) for row in reference]))
        return {role: base}, {role: max(1, int(round(target_flops / per_step)))}
    if control_id == "generous_2x_concat":
        return {role: dict(base) for role in roles}, {
            role: int(config["factorial"]["steps"]) for role in roles
        }

    candidates = _candidate_specs(base, architecture)
    target_parameters = float(np.mean([float(row["parameters"]) for row in full_rows]))
    bottleneck_parameters = int(base["state_dim"]) * (2 * int(base["state_dim"]) + 1)
    if control_id == "parameter_matched_concat":
        selected = min(
            candidates,
            key=lambda value: abs(
                (2 * _model_parameter_count(value) + bottleneck_parameters)
                / target_parameters
                - 1.0
            ),
        )
    else:
        plan = _make_plan(
            corpus,
            int(config["factorial"]["batch_size"]),
            int(config["seeds"][0]),
            0,
        )
        measured = []
        for candidate in candidates:
            total = 0.0
            for role in roles:
                model = _new_model(
                    config,
                    corpus,
                    int(config["seeds"][0]),
                    model_spec=candidate,
                )
                weights = _loss_weights(
                    config,
                    1.0 if role == "alignment" else 0.0,
                    1.0 if role == "response" else 0.0,
                    float(pilot["alignment_scale"]),
                    float(pilot["response_scale"]),
                    float(pilot["alignment_null_tolerance"]),
                )
                execution = _measure_execution(model, corpus, plan, weights)
                if execution["flops"] is None:
                    raise RuntimeError("resource-control architecture search requires FLOP measurement")
                total += float(execution["flops"]) * int(
                    config["factorial"]["steps"]
                )
            total += _bottleneck_training_flops(config, corpus)
            measured.append((abs(total / target_flops - 1.0), candidate))
        selected = min(measured, key=lambda value: value[0])[1]
    return {role: dict(selected) for role in roles}, {
        role: int(config["factorial"]["steps"]) for role in roles
    }


def _candidate_specs(base, architecture):
    search = architecture["search_policy"]
    output = []
    for layers in search["csi_encoder_layers"]:
        for map_scale in search["map_dim_scales"]:
            for hidden_scale in search["hidden_dim_scales"]:
                candidate = dict(base)
                candidate["csi_encoder_layers"] = min(
                    int(layers), int(base["csi_encoder_layers"])
                )
                candidate["map_dim"] = max(4, int(round(base["map_dim"] * map_scale)))
                candidate["hidden_dim"] = max(
                    8, int(round(base["hidden_dim"] * hidden_scale))
                )
                output.append(candidate)
    unique = {tuple(sorted(value.items())): value for value in output}
    return list(unique.values())


def _model_parameter_count(spec):
    from formal_v2.formal_model import CSIPairsFormalModel

    model = CSIPairsFormalModel(**spec)
    return sum(parameter.numel() for parameter in model.parameters())


def _control_representations(
    control_id, components, dataset, normalization, patch_spec
):
    scenes = [
        int(value)
        for role in ("source_encoder_train", "target")
        for value in dataset.indices_for_role(role)
    ]
    by_role = {
        role: _natural_representations(
            model, dataset, scenes, normalization, patch_spec
        )
        for role, model in components.items()
    }
    if len(by_role) == 1:
        return next(iter(by_role.values()))
    return {
        scene: np.concatenate(
            (by_role["alignment"][scene], by_role["response"][scene]), axis=1
        )
        for scene in scenes
    }


def _fit_source_localizer(control_id, representations, dataset, config, seed):
    source_scenes = [int(value) for value in dataset.indices_for_role("source_encoder_train")]
    features = np.vstack([representations[scene] for scene in source_scenes]).astype(np.float32)
    targets = np.vstack([dataset.positions[scene] for scene in source_scenes]).astype(np.float32)
    state_dim = int(config["model"]["state_dim"])
    torch.manual_seed(int(seed) + 17003)
    bottleneck = (
        nn.Linear(features.shape[1], state_dim)
        if control_id.endswith("concat")
        else None
    )
    head = HeteroscedasticPositionHead(
        state_dim if bottleneck is not None else features.shape[1],
        max(8, int(config["model"]["hidden_dim"]) // 2),
    )
    parameters = list(head.parameters()) + (
        list(bottleneck.parameters()) if bottleneck is not None else []
    )
    optimizer = torch.optim.AdamW(
        parameters, lr=float(config["localization"]["learning_rate"])
    )
    x = torch.as_tensor(features)
    y = torch.as_tensor(targets)
    trace = []
    bottleneck_flops_per_step = 0.0
    if bottleneck is not None:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as counter, torch.no_grad():
            bottleneck(x)
        bottleneck_flops_per_step = float(counter.get_total_flops())
        if bottleneck_flops_per_step <= 0:
            raise RuntimeError("resource control could not measure bottleneck FLOPs")
    for step in range(int(config["localization"]["head_steps"])):
        def compute_loss():
            encoded = bottleneck(x) if bottleneck is not None else x
            mean, raw_scale = head(encoded)
            scale = functional.softplus(raw_scale) + float(
                config["localization"]["sigma_min"]
            )
            nll = torch.mean(
                torch.sum(
                    torch.log(scale) + 0.5 * ((y - mean) / scale) ** 2,
                    dim=1,
                )
            )
            return nll + functional.huber_loss(mean, y)

        loss = compute_loss()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = math.sqrt(
            max(
                sum(
                    float(torch.sum(parameter.grad.detach() ** 2))
                    for parameter in parameters
                    if parameter.grad is not None
                ),
                0.0,
            )
        )
        optimizer.step()
        trace.append(
            {"step": step + 1, "localization_loss": float(loss.detach()), "gradient_norm": gradient}
        )
    head.eval()
    if bottleneck is not None:
        bottleneck.eval()
    return (
        bottleneck,
        head,
        trace,
        bottleneck_flops_per_step * int(config["localization"]["head_steps"]),
    )


def _bottleneck_training_flops(config, corpus):
    source_scenes = [
        int(value) for value in corpus.dataset.indices_for_role("source_encoder_train")
    ]
    sample_count = sum(int(corpus.dataset.positions[scene].shape[0]) for scene in source_scenes)
    state_dim = int(config["model"]["state_dim"])
    bottleneck = nn.Linear(2 * state_dim, state_dim)
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as counter, torch.no_grad():
        bottleneck(torch.zeros((sample_count, 2 * state_dim), dtype=torch.float32))
    per_step = float(counter.get_total_flops())
    if per_step <= 0:
        raise RuntimeError("resource-control architecture search could not measure bottleneck FLOPs")
    return per_step * int(config["localization"]["head_steps"])


def _evaluate_localization(
    control_id,
    representations,
    dataset,
    config,
    seed,
    bottleneck,
    source_head,
    dataset_sha256,
    config_sha256,
    fixture,
):
    transformed = {
        scene: _apply_bottleneck(values, bottleneck)
        for scene, values in representations.items()
    }
    target_scenes = [int(value) for value in dataset.indices_for_role("target")]
    target_cities = sorted(set(dataset.city_ids[target_scenes].tolist()))
    rows = []
    for city in target_cities:
        candidates = city_support_candidates(dataset, city)
        city_scenes = [
            scene for scene in target_scenes if str(dataset.city_ids[scene]) == city
        ]
        for draw in range(int(config["localization"]["label_draws"])):
            rng = np.random.default_rng(
                int(seed) * 100003 + _stable_city_seed(city) + draw
            )
            order = [candidates[index] for index in rng.permutation(len(candidates))]
            for budget in config["localization"]["label_budgets"]:
                if int(budget) == 0 and draw > 0:
                    continue
                selected = order[: int(budget)]
                support_ids = [
                    str(dataset.position_ids[scene, position])
                    for scene, position in selected
                ]
                support_x = (
                    np.vstack(
                        [transformed[scene][position] for scene, position in selected]
                    )
                    if selected
                    else np.empty((0, int(config["model"]["state_dim"])))
                )
                support_y = (
                    np.vstack(
                        [dataset.positions[scene, position] for scene, position in selected]
                    )
                    if selected
                    else np.empty((0, 2))
                )
                head = adapt_position_head(
                    source_head, support_x, support_y, config
                )
                for scene in city_scenes:
                    query = eligible_query_indices(dataset, scene, set(support_ids))
                    prediction, _, _ = predict_position_distribution(
                        head,
                        transformed[scene][query],
                        sigma_min=float(config["localization"]["sigma_min"]),
                    )
                    errors = np.linalg.norm(
                        prediction - dataset.positions[scene, query], axis=1
                    )
                    median = float(np.median(errors))
                    rows.append(
                        {
                            "arm": control_id,
                            "city_id": city,
                            "bank_id": str(dataset.bank_ids[scene]),
                            "base_map_cluster_id": str(
                                dataset.base_map_cluster_ids[scene]
                            ),
                            "canonical_base_map_digest": str(
                                dataset.canonical_base_map_digest(scene)
                            ),
                            "canonical_bank_digest": _canonical_bank_digest(
                                dataset, scene
                            ),
                            "seed": seed,
                            "budget": int(budget),
                            "draw": 0 if int(budget) == 0 else draw,
                            "utility_neg_log_median": -math.log(
                                max(median, 1e-12)
                            ),
                            "dataset_sha256": dataset_sha256,
                            "config_sha256": config_sha256,
                            "fixture": fixture,
                        }
                    )
    return rows


def _apply_bottleneck(values, bottleneck):
    if bottleneck is None:
        return np.asarray(values, dtype=np.float64)
    with torch.no_grad():
        return bottleneck(torch.as_tensor(values, dtype=torch.float32)).numpy().astype(
            np.float64
        )


def _retained_inference_flops(model, dataset, normalization, patch_spec):
    scene = int(dataset.indices_for_role("source_encoder_train")[0])
    world = int(dataset.natural_world_index[scene])
    from formal_v2.formal_protocol import patchify_csi

    patches = patchify_csi(dataset.csi[scene, world, :1], patch_spec)
    patches = (patches - normalization.patch_mean) / normalization.patch_scale
    maps = dataset.maps[scene, world][None] / normalization.map_scale[None, :, None, None]
    radio = np.concatenate((dataset.radio_config[scene], dataset.bs_pose[scene]))[None]
    radio = (radio - normalization.radio_mean) / normalization.radio_scale
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model.retained_representation(
            torch.as_tensor(patches, dtype=torch.float32),
            torch.as_tensor(maps, dtype=torch.float32),
            torch.as_tensor(radio, dtype=torch.float32),
        )
    value = float(counter.get_total_flops())
    if value <= 0:
        raise RuntimeError("resource control could not measure retained inference FLOPs")
    return value


def _payload_resource_parameter_count(payload):
    tensors = []
    for component in payload["components"]:
        tensors.extend(component["state_dict"].values())
    if payload["bottleneck_state_dict"] is not None:
        tensors.extend(payload["bottleneck_state_dict"].values())
    if any(
        not isinstance(value, torch.Tensor)
        or value.numel() == 0
        or not bool(torch.isfinite(value).all())
        for value in tensors
    ):
        raise RuntimeError("resource-control checkpoint contains invalid tensors")
    return int(sum(value.numel() for value in tensors))


if __name__ == "__main__":
    raise SystemExit(main())
