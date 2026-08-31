from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .external_adapters.representation_models import BaselineBatch, ContraWiMAE, WWMJEPA, build_representation_model
from .formal_data_verification import require_verified_roles_from_root
from .formal_evidence import bind_rows, evidence_context
from .formal_io import artifact_manifest, read_strict_json, sha256_file, write_csv, write_json
from .formal_localization import adapt_position_head, fit_source_position_head, predict_position_distribution
from .formal_protocol import PatchSpec
from .formal_resources import validate_resource_registry


CONFIG_SCHEMA = "csi-pairs-v6-representation-baselines-v2"
LEGACY_SMOKE_CONFIG_SCHEMA = "csi-pairs-v6-representation-baselines-v1"
MODEL_NAMES = {"CSI-MAE", "CSI-CLIP", "CSI-CLIP++", "ContraWiMAE", "WWM"}
IMPLEMENTATION_STATUSES = {
    "paper-spec-controlled-implementation",
    "style-controlled-implementation",
}


def run_representation_baselines(config, dataset, adapter_config_path, output_root):
    from .formal_upstream import resolve_authenticated_upstream

    resolve_authenticated_upstream(config, dataset, output_root)
    required_roles = (
        "source_encoder_train",
        "source_method_selection",
        "source_probe_train",
        "source_probe_selection",
        "source_final_unseen_bank",
        "target",
    )
    require_verified_roles_from_root(output_root, config, dataset, required_roles)
    adapter_config_path = Path(adapter_config_path).resolve()
    adapter = load_representation_config(adapter_config_path)
    if not dataset.is_fixture and adapter["profile"] != "formal-paper-dose":
        raise RuntimeError("scientific representation execution requires the formal-paper-dose profile")
    project_root = Path(__file__).resolve().parents[1]
    resource_registry_path = Path(__file__).resolve().parent / "configs" / "waibu_resources_v1.json"
    resource_rows = validate_resource_registry(
        read_strict_json(resource_registry_path), project_root / "waibu"
    )
    resources = {row["resource_id"]: row for row in resource_rows}
    output_dir = Path(output_root) / "representation_baselines"
    output_dir.mkdir(parents=True, exist_ok=False)
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    )
    spec = PatchSpec.from_metadata(dataset.metadata)
    normalizer = _fit_normalizer(dataset, dataset.indices_for_role("source_encoder_train"))
    algorithm_seeds = tuple(int(value) for value in adapter["seeds"])
    support_draws = _target_support_draws(config, adapter, dataset)
    expected_metric_units = _expected_metric_units(config, adapter, dataset)
    status_rows = []
    metric_rows = []
    per_query_rows = []
    for model_config in adapter["models"]:
        model_name = model_config["model_name"]
        resource = resources.get(model_config["resource_id"])
        if resource is None or resource["status"] != "PASS":
            raise RuntimeError(f"representation baseline resource is not authenticated: {model_name}")
        allowed_labels = {resource["allowed_name"]}
        if model_name == "CSI-CLIP++":
            allowed_labels.add("CSI-CLIP++-style controlled implementation")
        if model_config["paper_label"] not in allowed_labels:
            raise RuntimeError(f"representation baseline label exceeds its authenticated provenance: {model_name}")
        model_output = output_dir / _slug(model_name)
        model_output.mkdir(parents=True, exist_ok=False)
        for algorithm_seed in algorithm_seeds:
            seed_output = model_output / f"seed_{algorithm_seed}"
            seed_output.mkdir(parents=True, exist_ok=False)
            torch.manual_seed(algorithm_seed)
            model = build_representation_model(
                model_name,
                spec,
                dataset.maps.shape[2],
                dataset.radio_config.shape[1] + dataset.bs_pose.shape[1] + 2,
                model_config,
            )
            checkpoint, training_record = _train_model(
                model,
                model_name,
                model_config,
                dataset,
                normalizer,
                seed_output,
                seed=algorithm_seed,
            )
            write_json(seed_output / "training_record.json", training_record)
            rows, queries, probe_record = _evaluate_localization(
                model,
                model_name,
                model_config,
                adapter,
                config,
                dataset,
                normalizer,
                support_draws,
                seed=algorithm_seed,
            )
            write_json(seed_output / "probe_record.json", probe_record)
            write_csv(seed_output / "localization_metrics.csv", bind_rows(rows, evidence))
            write_csv(seed_output / "localization_per_query.csv", bind_rows(queries, evidence))
            metric_rows.extend(rows)
            per_query_rows.extend(queries)
            status_rows.append(
                {
                    "model_name": model_name,
                    "algorithm_seed": algorithm_seed,
                    "paper_label": model_config["paper_label"],
                    "implementation_status": model_config["implementation_status"],
                    "resource_id": model_config["resource_id"],
                    "resource_sha256": resource["sha256"],
                    "checkpoint_path": str(checkpoint.relative_to(output_dir)),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "status": "PASS",
                }
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_csv(output_dir / "model_status.csv", bind_rows(status_rows, evidence))
    write_csv(output_dir / "localization_metrics.csv", bind_rows(metric_rows, evidence))
    write_csv(output_dir / "localization_per_query.csv", bind_rows(per_query_rows, evidence))
    summary_rows = _summarize_across_seeds(per_query_rows, algorithm_seeds)
    write_csv(output_dir / "localization_across_seed_summary.csv", bind_rows(summary_rows, evidence))
    completeness = _validate_representation_outputs(
        status_rows,
        metric_rows,
        per_query_rows,
        [row["model_name"] for row in adapter["models"]],
        algorithm_seeds,
        expected_metric_units,
    )
    summary_complete = bool(summary_rows) and all(
        int(row["algorithm_seed_count"]) == len(algorithm_seeds)
        and bool(row["all_expected_algorithm_seeds_present"])
        for row in summary_rows
    )
    passed = bool(completeness["passed"] and summary_complete)
    gate = {
        "schema_version": "csi-pairs-v6-representation-baseline-gate-v2",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        **evidence,
        "profile": adapter["profile"],
        "adapter_config_sha256": sha256_file(adapter_config_path),
        "resource_registry_sha256": sha256_file(resource_registry_path),
        "algorithm_seeds": list(algorithm_seeds),
        "algorithm_seed_count": len(algorithm_seeds),
        "expected_model_seed_count": len(adapter["models"]) * len(algorithm_seeds),
        "executed_model_seed_count": len(status_rows),
        "executed_model_count": len({row["model_name"] for row in status_rows}),
        "executed_models": sorted({row["model_name"] for row in status_rows}),
        "expected_metric_unit_count_per_model_seed": len(expected_metric_units),
        "common_metric_units_complete": completeness["metric_units_complete"],
        "common_query_units_complete": completeness["query_units_complete"],
        "completeness_errors": completeness["errors"],
        "across_seed_summary_complete": summary_complete,
        "c1_eligible_model_count": 0,
        "claim_eligible": False,
        "claim_scope": "descriptive representation/localization comparison only; this stage cannot satisfy C1 or any scientific claim gate",
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


def load_representation_config(path: str | Path) -> dict:
    payload = read_strict_json(path)
    if not isinstance(payload, dict):
        raise ValueError("representation baseline config fields must be exact")
    expected_fields = {
        "schema_version", "profile", "seeds", "source_roles", "probe", "models"
    }
    legacy_fields = {
        "schema_version", "profile", "seed", "source_roles", "probe", "models"
    }
    if set(payload) == legacy_fields:
        if (
            payload["schema_version"] != LEGACY_SMOKE_CONFIG_SCHEMA
            or payload["profile"] != "software-smoke-only"
        ):
            raise ValueError("legacy single-seed representation configs are software-smoke-only")
        payload = dict(payload)
        payload["seeds"] = [payload.pop("seed")]
    elif set(payload) != expected_fields:
        raise ValueError("representation baseline config fields must be exact")
    if payload["schema_version"] not in {CONFIG_SCHEMA, LEGACY_SMOKE_CONFIG_SCHEMA}:
        raise ValueError("representation baseline config schema mismatch")
    if payload["schema_version"] == LEGACY_SMOKE_CONFIG_SCHEMA and payload["profile"] != "software-smoke-only":
        raise ValueError("legacy representation config schema is software-smoke-only")
    if payload["profile"] not in {"formal-paper-dose", "software-smoke-only"}:
        raise ValueError("representation baseline profile is invalid")
    if payload["source_roles"] != {
        "pretrain": "source_encoder_train",
        "selection": "source_method_selection",
        "probe_train": "source_probe_train",
        "probe_selection": "source_probe_selection",
        "evaluation": ["source_final_unseen_bank", "target"],
    }:
        raise ValueError("representation baseline role ledger is not frozen V6")
    seeds = payload["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("representation algorithm seeds must be a nonempty list")
    for index, seed in enumerate(seeds):
        _positive_integer(seed, f"seeds[{index}]")
    if len(set(seeds)) != len(seeds):
        raise ValueError("representation algorithm seeds must be distinct")
    if payload["profile"] == "formal-paper-dose" and len(seeds) < 3:
        raise ValueError("formal representation execution requires at least three independent algorithm seeds")
    if set(payload["probe"]) != {"head_steps", "learning_rate", "hidden_dim", "label_draws"}:
        raise ValueError("representation probe config fields must be exact")
    for key in ("head_steps", "hidden_dim", "label_draws"):
        _positive_integer(payload["probe"][key], f"probe.{key}")
    _positive_number(payload["probe"]["learning_rate"], "probe.learning_rate")
    models = payload["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("representation models must be a nonempty list")
    expected_fields = {
        "model_name",
        "paper_label",
        "implementation_status",
        "resource_id",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "warmup_epochs",
        "mask_fraction",
        "dim",
        "heads",
        "encoder_layers",
        "decoder_layers",
        "decoder_dim",
        "warm_start_epochs",
        "preprocessing",
        "resnet_width",
        "resnet_depth",
        "ema",
        "reconstruction_weight",
        "minimum_snr_db",
        "maximum_snr_db",
    }
    names = []
    for row in models:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError("representation model config fields must be exact")
        if row["model_name"] not in MODEL_NAMES or row["model_name"] in names:
            raise ValueError("representation model name is invalid or duplicated")
        names.append(row["model_name"])
        if row["implementation_status"] not in IMPLEMENTATION_STATUSES:
            raise ValueError("representation implementation status is invalid")
        for key in ("epochs", "batch_size", "dim", "heads", "encoder_layers", "decoder_layers", "decoder_dim", "resnet_width", "resnet_depth"):
            _positive_integer(row[key], f"{row['model_name']}.{key}")
        if not isinstance(row["warm_start_epochs"], int) or isinstance(row["warm_start_epochs"], bool) or row["warm_start_epochs"] < 0:
            raise ValueError("warm-start epochs must be a nonnegative integer")
        if row["preprocessing"] not in {"zscore", "minmax-standardize"}:
            raise ValueError("representation preprocessing is unsupported")
        if row["model_name"] == "ContraWiMAE" and row["warm_start_epochs"] <= 0 and payload["profile"] == "formal-paper-dose":
            raise ValueError("formal ContraWiMAE requires reconstruction-only WiMAE warm-start")
        if row["model_name"] != "ContraWiMAE" and row["warm_start_epochs"] != 0:
            raise ValueError("only ContraWiMAE may request WiMAE warm-start")
        if row["decoder_dim"] % 4:
            raise ValueError("representation decoder dimension must support fixed two-dimensional position encoding")
        for key in ("learning_rate", "ema", "reconstruction_weight"):
            _positive_number(row[key], f"{row['model_name']}.{key}")
        if not isinstance(row["mask_fraction"], (int, float)) or isinstance(row["mask_fraction"], bool):
            raise ValueError("mask fraction must be numeric")
        if row["weight_decay"] < 0 or row["warmup_epochs"] < 0:
            raise ValueError("weight decay and warmup epochs must be nonnegative")
        if row["dim"] % row["heads"]:
            raise ValueError("representation dimension must be divisible by attention heads")
        if not 0.0 <= row["mask_fraction"] < 1.0 or not 0.0 < row["ema"] < 1.0:
            raise ValueError("mask fraction or EMA is out of range")
        if not 0.0 < row["reconstruction_weight"] < 1.0:
            raise ValueError("reconstruction weight must lie strictly between zero and one")
        if row["minimum_snr_db"] >= row["maximum_snr_db"]:
            raise ValueError("ContraWiMAE SNR range is empty")
    if payload["profile"] == "formal-paper-dose" and set(names) != MODEL_NAMES:
        raise ValueError("formal representation execution requires all five frozen baselines")
    return payload


def _expected_metric_units(formal_config, adapter, dataset):
    units = set()
    for scene, _, _ in _natural_position_units(dataset, "source_final_unseen_bank", query_only=False):
        units.add(("source_final_unseen_bank", str(dataset.city_ids[scene]), 0, 0))

    support_counts = {}
    for scene_value in dataset.indices_for_role("target"):
        scene = int(scene_value)
        city = str(dataset.city_ids[scene])
        support_counts.setdefault(city, set()).update(
            str(dataset.position_ids[scene, position])
            for position in np.flatnonzero(dataset.position_roles[scene] == "support_pool")
        )
    target_cities = {
        str(dataset.city_ids[scene])
        for scene, _, _ in _natural_position_units(dataset, "target", query_only=True)
    }
    for city in target_cities:
        for draw in range(int(adapter["probe"]["label_draws"])):
            for budget_value in formal_config["localization"]["label_budgets"]:
                budget = int(budget_value)
                if len(support_counts.get(city, set())) < budget:
                    if adapter["profile"] == "formal-paper-dose":
                        # Evaluation fails closed before the gate for this condition.
                        units.add(("target", city, budget, draw))
                    continue
                units.add(("target", city, budget, draw))
    return tuple(sorted(units))


def _validate_representation_outputs(
    status_rows,
    metric_rows,
    per_query_rows,
    model_names,
    algorithm_seeds,
    expected_metric_units,
):
    expected_identities = {
        (str(model_name), int(seed))
        for model_name in model_names
        for seed in algorithm_seeds
    }
    errors = []

    status_counts = {}
    checkpoint_paths = {}
    for row in status_rows:
        identity = (str(row.get("model_name")), int(row.get("algorithm_seed", -1)))
        status_counts[identity] = status_counts.get(identity, 0) + 1
        if row.get("status") != "PASS":
            errors.append(f"model-seed status is not PASS: {identity}")
        checkpoint_path = str(row.get("checkpoint_path", ""))
        checkpoint_sha256 = str(row.get("checkpoint_sha256", ""))
        checkpoint_paths.setdefault(checkpoint_path, []).append(identity)
        if f"seed_{identity[1]}/" not in checkpoint_path.replace("\\", "/"):
            errors.append(f"checkpoint path is not seed-scoped for {identity}: {checkpoint_path!r}")
        if len(checkpoint_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in checkpoint_sha256
        ):
            errors.append(f"checkpoint digest is invalid for {identity}")
    status_identities = set(status_counts)
    if status_identities != expected_identities:
        missing = sorted(expected_identities - status_identities)
        unexpected = sorted(status_identities - expected_identities)
        if missing:
            errors.append(f"missing model-seed status rows: {missing}")
        if unexpected:
            errors.append(f"unexpected model-seed status rows: {unexpected}")
    duplicated_status = sorted(identity for identity, count in status_counts.items() if count != 1)
    if duplicated_status:
        errors.append(f"non-unique model-seed status rows: {duplicated_status}")
    reused_checkpoints = sorted(
        path for path, identities in checkpoint_paths.items() if not path or len(identities) != 1
    )
    if reused_checkpoints:
        errors.append(f"checkpoint paths are empty or reused across model-seed runs: {reused_checkpoints}")

    expected_units = set(expected_metric_units)
    metric_units_by_identity = {identity: set() for identity in expected_identities}
    metric_counts = {}
    for row in metric_rows:
        identity = (str(row.get("model_name")), int(row.get("algorithm_seed", -1)))
        unit = (
            str(row.get("split_role")),
            str(row.get("city_id")),
            int(row.get("budget", -1)),
            int(row.get("draw", -1)),
        )
        metric_units_by_identity.setdefault(identity, set()).add(unit)
        metric_counts[(identity, unit)] = metric_counts.get((identity, unit), 0) + 1
    metric_units_complete = True
    for identity in sorted(expected_identities):
        observed = metric_units_by_identity.get(identity, set())
        if observed != expected_units:
            metric_units_complete = False
            missing = sorted(expected_units - observed)
            unexpected = sorted(observed - expected_units)
            errors.append(
                f"metric-unit mismatch for {identity}: missing={missing}, unexpected={unexpected}"
            )
    unexpected_metric_identities = sorted(set(metric_units_by_identity) - expected_identities)
    if unexpected_metric_identities:
        metric_units_complete = False
        errors.append(f"unexpected metric identities: {unexpected_metric_identities}")
    duplicated_metrics = sorted(key for key, count in metric_counts.items() if count != 1)
    if duplicated_metrics:
        metric_units_complete = False
        errors.append(f"duplicate metric rows: {duplicated_metrics}")

    query_units_by_identity = {identity: set() for identity in expected_identities}
    query_counts = {}
    for row in per_query_rows:
        identity = (str(row.get("model_name")), int(row.get("algorithm_seed", -1)))
        query_unit = (
            str(row.get("split_role")),
            str(row.get("city_id")),
            int(row.get("budget", -1)),
            int(row.get("draw", -1)),
            str(row.get("independent_unit_id")),
            str(row.get("bank_id")),
            str(row.get("position_id")),
            int(row.get("world", -1)),
        )
        query_units_by_identity.setdefault(identity, set()).add(query_unit)
        query_counts[(identity, query_unit)] = query_counts.get((identity, query_unit), 0) + 1
    reference_query_units = None
    query_units_complete = bool(per_query_rows)
    for identity in sorted(expected_identities):
        observed = query_units_by_identity.get(identity, set())
        observed_metric_units = {unit[:4] for unit in observed}
        if not observed or observed_metric_units != expected_units:
            query_units_complete = False
            errors.append(
                f"query metric-unit coverage mismatch for {identity}: "
                f"missing={sorted(expected_units - observed_metric_units)}, "
                f"unexpected={sorted(observed_metric_units - expected_units)}"
            )
        if reference_query_units is None:
            reference_query_units = observed
        elif observed != reference_query_units:
            query_units_complete = False
            errors.append(f"query units are not common across model-seed runs: {identity}")
    unexpected_query_identities = sorted(set(query_units_by_identity) - expected_identities)
    if unexpected_query_identities:
        query_units_complete = False
        errors.append(f"unexpected query identities: {unexpected_query_identities}")
    duplicated_queries = sorted(key for key, count in query_counts.items() if count != 1)
    if duplicated_queries:
        query_units_complete = False
        errors.append(f"duplicate per-query rows: {duplicated_queries}")

    return {
        "passed": not errors and metric_units_complete and query_units_complete,
        "metric_units_complete": metric_units_complete,
        "query_units_complete": query_units_complete,
        "errors": errors,
    }


def _summarize_across_seeds(per_query_rows, algorithm_seeds, *, bootstrap_draws=2000):
    cluster_values = {}
    metadata = {}
    for row in per_query_rows:
        cell = (
            str(row["model_name"]),
            int(row["algorithm_seed"]),
            str(row["split_role"]),
            str(row["city_id"]),
            int(row["budget"]),
            int(row["draw"]),
            str(row["independent_unit_id"]),
        )
        cluster_values.setdefault(cell, []).append(float(row["localization_error_m"]))
        model_key = cell[0]
        model_metadata = (
            str(row["paper_label"]),
            str(row["implementation_status"]),
        )
        if model_key in metadata and metadata[model_key] != model_metadata:
            raise RuntimeError(f"representation metadata changes within model {model_key}")
        metadata[model_key] = model_metadata

    seed_cells = {}
    for key, values in cluster_values.items():
        model_name, seed, role, city, budget, draw, _ = key
        seed_cell = (model_name, role, city, budget, draw, seed)
        seed_cells.setdefault(seed_cell, []).append(float(np.median(values)))

    across_seed = {}
    for key, cluster_medians in seed_cells.items():
        model_name, role, city, budget, draw, seed = key
        output_key = (model_name, role, city, budget, draw)
        across_seed.setdefault(output_key, {})[seed] = {
            "cluster_macro_error_m": float(np.mean(cluster_medians)),
            "cluster_count": len(cluster_medians),
        }

    expected_seeds = tuple(int(value) for value in algorithm_seeds)
    rows = []
    rng = np.random.default_rng(0x435349)
    for key in sorted(across_seed):
        model_name, role, city, budget, draw = key
        values_by_seed = across_seed[key]
        present_seeds = sorted(values_by_seed)
        values = np.asarray(
            [values_by_seed[seed]["cluster_macro_error_m"] for seed in present_seeds],
            dtype=np.float64,
        )
        if values.size:
            bootstrap = np.mean(
                values[rng.integers(0, values.size, size=(int(bootstrap_draws), values.size))],
                axis=1,
            )
            ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
        else:
            ci_low = ci_high = float("nan")
        cluster_counts = [values_by_seed[seed]["cluster_count"] for seed in present_seeds]
        paper_label, implementation_status = metadata[model_name]
        rows.append(
            {
                "model_name": model_name,
                "paper_label": paper_label,
                "implementation_status": implementation_status,
                "split_role": role,
                "city_id": city,
                "budget": budget,
                "draw": draw,
                "expected_algorithm_seed_count": len(expected_seeds),
                "algorithm_seed_count": len(present_seeds),
                "algorithm_seeds": "|".join(str(seed) for seed in present_seeds),
                "all_expected_algorithm_seeds_present": present_seeds == sorted(expected_seeds),
                "cluster_count_min": min(cluster_counts),
                "cluster_count_max": max(cluster_counts),
                "cluster_macro_across_seed_mean_error_m": float(np.mean(values)),
                "cluster_macro_across_seed_std_m": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "cluster_macro_seed_bootstrap_ci95_low_m": float(ci_low),
                "cluster_macro_seed_bootstrap_ci95_high_m": float(ci_high),
                "per_seed_cluster_macro_error_m": "|".join(
                    f"{seed}:{values_by_seed[seed]['cluster_macro_error_m']:.17g}"
                    for seed in present_seeds
                ),
                "claim_eligible": False,
            }
        )
    return rows


def _train_model(model, model_name, model_config, dataset, normalizer, output, *, seed):
    train_units = _world_position_units(dataset, "source_encoder_train")
    selection_units = _world_position_units(dataset, "source_method_selection")
    if not train_units or not selection_units:
        raise RuntimeError("representation pretraining roles are empty")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    warm_start = _warm_start_contra(
        model, model_name, model_config, dataset, normalizer, train_units, selection_units, device, seed
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    batch_size = int(model_config["batch_size"])
    steps_per_epoch = max(1, math.ceil(len(train_units) / batch_size))
    total_steps = int(model_config["epochs"]) * steps_per_epoch
    warmup_steps = int(model_config["warmup_epochs"]) * steps_per_epoch

    def factor(step):
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-6)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        floor = 0.01 if model_name == "ContraWiMAE" else 0.0
        return floor + (1.0 - floor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    rng = np.random.default_rng(int(seed))
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    training_loss = float("nan")
    for epoch in range(1, int(model_config["epochs"]) + 1):
        order = rng.permutation(len(train_units))
        model.train()
        for start in range(0, len(order), batch_size):
            chosen = [train_units[int(index)] for index in order[start : start + batch_size]]
            batch = _make_batch(
                dataset, chosen, normalizer, device, preprocessing=model_config["preprocessing"]
            )
            loss = model.pretraining_loss(batch, float(model_config["mask_fraction"]))
            if not torch.isfinite(loss):
                raise RuntimeError(f"{model_name} produced nonfinite pretraining loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if isinstance(model, WWMJEPA):
                model.update_target()
            scheduler.step()
            training_loss = float(loss.detach().cpu())
        selection_loss = _mean_pretraining_loss(
            model, dataset, selection_units, normalizer, device, model_config
        )
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError(f"{model_name} source-method-selection produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    checkpoint = output / "source_selected.pt"
    torch.save(
        {
            "schema_version": "csi-pairs-v6-representation-checkpoint-v2",
            "model_name": model_name,
            "algorithm_seed": int(seed),
            "paper_label": model_config["paper_label"],
            "implementation_status": model_config["implementation_status"],
            "dataset_sha256": sha256_file(dataset.source_path),
            "train_role": "source_encoder_train",
            "selection_role": "source_method_selection",
            "target_roles_read": [],
            "selected_epoch": best_epoch,
            "selection_loss": best_loss,
            "warm_start": warm_start,
            "normalizer": {key: value.tolist() for key, value in normalizer.items()},
            "model_config": model_config,
            "state_dict": best_state,
        },
        checkpoint,
    )
    return checkpoint, {
        "schema_version": "csi-pairs-v6-representation-training-record-v2",
        "model_name": model_name,
        "algorithm_seed": int(seed),
        "device": str(device),
        "epochs": int(model_config["epochs"]),
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "scheduler": "linear-warmup-cosine",
        "last_training_loss": training_loss,
        "selected_epoch": best_epoch,
        "selection_loss": best_loss,
        "warm_start": warm_start,
        "train_role": "source_encoder_train",
        "selection_role": "source_method_selection",
        "probe_roles_read": [],
        "target_roles_read": [],
    }


def _warm_start_contra(
    model,
    model_name,
    model_config,
    dataset,
    normalizer,
    train_units,
    selection_units,
    device,
    seed,
):
    epochs = int(model_config["warm_start_epochs"])
    if model_name != "ContraWiMAE":
        return {
            "required": False,
            "completed": True,
            "epochs": 0,
            "selected_epoch": 0,
            "selection_loss": None,
        }
    if not isinstance(model, ContraWiMAE) or epochs <= 0:
        raise RuntimeError("ContraWiMAE requires a reconstruction-only WiMAE warm-start")
    batch_size = int(model_config["batch_size"])
    steps_per_epoch = max(1, math.ceil(len(train_units) / batch_size))
    total_steps = epochs * steps_per_epoch
    warmup_steps = min(int(model_config["warmup_epochs"]) * steps_per_epoch, total_steps)
    optimizer = torch.optim.AdamW(
        model.mae.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )

    def factor(step):
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-6)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    rng = np.random.default_rng(int(seed) ^ 0x57494D41)
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_units))
        model.train()
        for start in range(0, len(order), batch_size):
            chosen = [train_units[int(index)] for index in order[start : start + batch_size]]
            batch = _make_batch(
                dataset,
                chosen,
                normalizer,
                device,
                preprocessing=model_config["preprocessing"],
            )
            loss = model.warm_start_loss(batch, float(model_config["mask_fraction"]))
            if not torch.isfinite(loss):
                raise RuntimeError("WiMAE warm-start produced nonfinite reconstruction loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.mae.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
        model.eval()
        values = []
        with torch.no_grad():
            for start in range(0, len(selection_units), batch_size):
                batch = _make_batch(
                    dataset,
                    selection_units[start : start + batch_size],
                    normalizer,
                    device,
                    preprocessing=model_config["preprocessing"],
                )
                values.append(float(model.warm_start_loss(batch, float(model_config["mask_fraction"]))))
        selection_loss = float(np.mean(values))
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.mae.state_dict().items()}
    if best_state is None:
        raise RuntimeError("WiMAE warm-start produced no source-selected checkpoint")
    model.mae.load_state_dict(best_state)
    model.to(device)
    return {
        "required": True,
        "completed": True,
        "epochs": epochs,
        "selected_epoch": best_epoch,
        "selection_loss": best_loss,
        "train_role": "source_encoder_train",
        "selection_role": "source_method_selection",
        "target_roles_read": [],
    }


def _evaluate_localization(model, model_name, model_config, adapter, formal_config, dataset, normalizer, support_draws, *, seed):
    source_units = _natural_position_units(dataset, "source_probe_train", query_only=False)
    source_selection_units = _natural_position_units(dataset, "source_probe_selection", query_only=False)
    preprocessing = model_config["preprocessing"]
    source_rep, source_position, _ = _represent(model, dataset, source_units, normalizer, preprocessing=preprocessing)
    selection_rep, selection_position, _ = _represent(
        model, dataset, source_selection_units, normalizer, preprocessing=preprocessing
    )
    head_config = {
        "model": {"hidden_dim": int(adapter["probe"]["hidden_dim"]) * 2},
        "localization": {
            "head_steps": int(adapter["probe"]["head_steps"]),
            "learning_rate": float(adapter["probe"]["learning_rate"]),
            "sigma_min": float(formal_config["localization"]["sigma_min"]),
        },
    }
    source_head = fit_source_position_head(source_rep, source_position, head_config, seed=int(seed))
    selection_mean, _, _ = predict_position_distribution(
        source_head, selection_rep, sigma_min=float(formal_config["localization"]["sigma_min"])
    )
    selection_error = np.linalg.norm(selection_mean - selection_position, axis=1)
    query_rows = []
    metric_rows = []
    source_final = _natural_position_units(dataset, "source_final_unseen_bank", query_only=False)
    _append_evaluation(
        model,
        source_head,
        model_name,
        model_config,
        dataset,
        normalizer,
        source_final,
        algorithm_seed=seed,
        budget=0,
        draw=0,
        rows=query_rows,
    )
    target_units = _natural_position_units(dataset, "target", query_only=True)
    target_cities = sorted({str(dataset.city_ids[scene]) for scene, _, _ in target_units})
    budgets = [int(value) for value in formal_config["localization"]["label_budgets"]]
    for city in target_cities:
        city_queries = [unit for unit in target_units if str(dataset.city_ids[unit[0]]) == city]
        for draw, ordered_support in support_draws[city].items():
            for budget in budgets:
                selected = ordered_support[:budget]
                if len(selected) < budget:
                    if adapter["profile"] == "formal-paper-dose":
                        raise RuntimeError(f"target city {city} lacks {budget} unique support positions")
                    continue
                if budget:
                    support_rep, support_position, _ = _represent(
                        model, dataset, selected, normalizer, preprocessing=preprocessing
                    )
                    head = adapt_position_head(source_head, support_rep, support_position, head_config)
                else:
                    head = source_head
                _append_evaluation(
                    model,
                    head,
                    model_name,
                    model_config,
                    dataset,
                    normalizer,
                    city_queries,
                    algorithm_seed=seed,
                    budget=budget,
                    draw=draw,
                    rows=query_rows,
                )
    grouped = {}
    for row in query_rows:
        key = (
            row["model_name"],
            row["algorithm_seed"],
            row["split_role"],
            row["city_id"],
            row["budget"],
            row["draw"],
        )
        grouped.setdefault(key, []).append(float(row["localization_error_m"]))
    for key, values in sorted(grouped.items()):
        model_key, algorithm_seed, role, city, budget, draw = key
        metric_rows.append(
            {
                "model_name": model_key,
                "algorithm_seed": algorithm_seed,
                "paper_label": model_config["paper_label"],
                "implementation_status": model_config["implementation_status"],
                "split_role": role,
                "city_id": city,
                "budget": budget,
                "draw": draw,
                "query_count": len(values),
                "median_error_m": float(np.median(values)),
                "p90_error_m": float(np.quantile(values, 0.9)),
            }
        )
    return metric_rows, query_rows, {
        "schema_version": "csi-pairs-v6-representation-probe-record-v2",
        "algorithm_seed": int(seed),
        "probe_train_role": "source_probe_train",
        "probe_selection_role": "source_probe_selection",
        "target_support_role": "support_pool",
        "target_evaluation_role": "query",
        "source_probe_selection_median_error_m": float(np.median(selection_error)),
        "head_steps": int(adapter["probe"]["head_steps"]),
        "label_draws": int(adapter["probe"]["label_draws"]),
        "target_normalization_statistics_used": False,
    }


def _append_evaluation(
    model,
    head,
    model_name,
    model_config,
    dataset,
    normalizer,
    units,
    *,
    algorithm_seed,
    budget,
    draw,
    rows,
):
    if not units:
        return
    representations, positions, selected_units = _represent(
        model, dataset, units, normalizer, preprocessing=model_config["preprocessing"]
    )
    means, variances, uncertainty = predict_position_distribution(head, representations)
    for index, (scene, world, position) in enumerate(selected_units):
        rows.append(
            {
                "model_name": model_name,
                "algorithm_seed": int(algorithm_seed),
                "paper_label": model_config["paper_label"],
                "implementation_status": model_config["implementation_status"],
                "split_role": str(dataset.scene_roles[scene]),
                "city_id": str(dataset.city_ids[scene]),
                "independent_unit_id": dataset.independent_unit_id(scene),
                "bank_id": str(dataset.bank_ids[scene]),
                "position_id": str(dataset.position_ids[scene, position]),
                "world": int(world),
                "budget": int(budget),
                "draw": int(draw),
                "localization_error_m": float(np.linalg.norm(means[index] - positions[index])),
                "predicted_x_m": float(means[index, 0]),
                "predicted_y_m": float(means[index, 1]),
                "variance_x_m2": float(variances[index, 0]),
                "variance_y_m2": float(variances[index, 1]),
                "uncertainty": float(uncertainty[index]),
            }
        )


def _represent(model, dataset, units, normalizer, batch_size=256, *, preprocessing="zscore"):
    device = next(model.parameters()).device
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(units), batch_size):
            batch_units = units[start : start + batch_size]
            batch = _make_batch(dataset, batch_units, normalizer, device, preprocessing=preprocessing)
            values.append(model.encode(batch).detach().cpu().numpy())
    representations = np.concatenate(values, axis=0)
    positions = np.asarray([dataset.positions[scene, position] for scene, _, position in units], dtype=np.float32)
    return representations, positions, units


def _mean_pretraining_loss(model, dataset, units, normalizer, device, model_config):
    model.eval()
    values = []
    batch_size = int(model_config["batch_size"])
    with torch.no_grad():
        for start in range(0, len(units), batch_size):
            batch = _make_batch(
                dataset,
                units[start : start + batch_size],
                normalizer,
                device,
                preprocessing=model_config["preprocessing"],
            )
            loss = model.pretraining_loss(batch, float(model_config["mask_fraction"]))
            values.append(float(loss.detach().cpu()))
    return float(np.mean(values))


def _fit_normalizer(dataset, scenes):
    csi = dataset.csi_clean[np.asarray(scenes, dtype=np.int64)]
    maps = dataset.maps[np.asarray(scenes, dtype=np.int64)]
    positions = dataset.positions[np.asarray(scenes, dtype=np.int64)]
    radio = dataset.radio_config[np.asarray(scenes, dtype=np.int64)]
    bs_pose = dataset.bs_pose[np.asarray(scenes, dtype=np.int64)]
    csi_min = np.min(csi, axis=(0, 1, 2))
    csi_max = np.max(csi, axis=(0, 1, 2))
    csi_range = np.maximum(csi_max - csi_min, 1e-6)
    csi_scaled = (csi - csi_min) / csi_range
    return {
        "csi_mean": np.mean(csi, axis=(0, 1, 2)),
        "csi_std": np.maximum(np.std(csi, axis=(0, 1, 2)), 1e-6),
        "csi_min": csi_min,
        "csi_range": csi_range,
        "csi_scaled_mean": np.mean(csi_scaled, axis=(0, 1, 2)),
        "csi_scaled_std": np.maximum(np.std(csi_scaled, axis=(0, 1, 2)), 1e-6),
        "map_mean": np.mean(maps, axis=(0, 1, 3, 4)),
        "map_std": np.maximum(np.std(maps, axis=(0, 1, 3, 4)), 1e-6),
        "position_mean": np.mean(positions, axis=(0, 1)),
        "position_std": np.maximum(np.std(positions, axis=(0, 1)), 1e-6),
        "radio_mean": np.mean(radio, axis=0),
        "radio_std": np.maximum(np.std(radio, axis=0), 1e-6),
        "bs_mean": np.mean(bs_pose, axis=0),
        "bs_std": np.maximum(np.std(bs_pose, axis=0), 1e-6),
    }


def _make_batch(dataset, units, normalizer, device, *, preprocessing="zscore"):
    csi = np.asarray([dataset.csi_clean[scene, world, position] for scene, world, position in units])
    maps = np.asarray([dataset.maps[scene, world] for scene, world, _ in units])
    radio = np.asarray([dataset.radio_config[scene] for scene, _, _ in units])
    bs_pose = np.asarray([dataset.bs_pose[scene] for scene, _, _ in units])
    positions = np.asarray([dataset.positions[scene, position] for scene, _, position in units])
    if preprocessing == "zscore":
        csi = (csi - normalizer["csi_mean"]) / normalizer["csi_std"]
    elif preprocessing == "minmax-standardize":
        csi = (csi - normalizer["csi_min"]) / normalizer["csi_range"]
        csi = (csi - normalizer["csi_scaled_mean"]) / normalizer["csi_scaled_std"]
    else:
        raise ValueError("representation preprocessing is unsupported")
    maps = (maps - normalizer["map_mean"][None, :, None, None]) / normalizer["map_std"][None, :, None, None]
    radio = (radio - normalizer["radio_mean"]) / normalizer["radio_std"]
    bs_pose = (bs_pose - normalizer["bs_mean"]) / normalizer["bs_std"]
    positions = (positions - normalizer["position_mean"]) / normalizer["position_std"]
    return BaselineBatch(
        csi=torch.as_tensor(csi, dtype=torch.float32, device=device),
        maps=torch.as_tensor(maps, dtype=torch.float32, device=device),
        radio=torch.as_tensor(radio, dtype=torch.float32, device=device),
        bs_pose=torch.as_tensor(bs_pose, dtype=torch.float32, device=device),
        position=torch.as_tensor(positions, dtype=torch.float32, device=device),
    )


def _world_position_units(dataset, role):
    return [
        (int(scene), world, position)
        for scene in dataset.indices_for_role(role)
        for world in range(dataset.world_count)
        for position in range(dataset.position_count)
    ]


def _natural_position_units(dataset, role, *, query_only):
    output = []
    for scene_value in dataset.indices_for_role(role):
        scene = int(scene_value)
        world = int(dataset.natural_world_index[scene])
        for position in range(dataset.position_count):
            if query_only and str(dataset.position_roles[scene, position]) != "query":
                continue
            output.append((scene, world, position))
    return output


def _target_support_draws(config, adapter, dataset):
    support_by_city = {}
    for scene_value in dataset.indices_for_role("target"):
        scene = int(scene_value)
        city = str(dataset.city_ids[scene])
        world = int(dataset.natural_world_index[scene])
        for position in np.flatnonzero(dataset.position_roles[scene] == "support_pool"):
            key = str(dataset.position_ids[scene, position])
            support_by_city.setdefault(city, {})
            if key in support_by_city[city]:
                previous = support_by_city[city][key]
                if not np.array_equal(dataset.positions[previous[0], previous[2]], dataset.positions[scene, position]):
                    raise RuntimeError("one target position_id maps to different coordinates within a city")
                continue
            support_by_city[city][key] = (scene, world, int(position))
    output = {}
    for city, keyed in support_by_city.items():
        ordered_keys = sorted(keyed)
        output[city] = {}
        for draw in range(int(adapter["probe"]["label_draws"])):
            # The support draw is shared across algorithms and algorithm seeds.
            seed = int(adapter["seeds"][0]) + draw + int.from_bytes(city.encode("utf-8"), "little") % 1000003
            order = np.random.default_rng(seed).permutation(len(ordered_keys))
            output[city][draw] = [keyed[ordered_keys[int(index)]] for index in order]
    return output


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _positive_number(value, name):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def _slug(value):
    return str(value).lower().replace("+", "plus").replace("-", "_")
