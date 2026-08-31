from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .formal_io import read_strict_json
from .formal_protocol import FROZEN_RANDOM_MASK_FRACTION


SCHEMA_VERSION = "csi-pairs-formal-config-v2.3-v6"
ARMS = ("endpoint", "alignment", "response", "full")
TOP_LEVEL_KEYS = {
    "schema_version",
    "artifact_label",
    "seeds",
    "data",
    "qualification",
    "teacher",
    "model",
    "factorial",
    "localization",
    "evaluation",
    "risk",
    "path",
    "external_validity",
    "literature",
}
DATA_KEYS = {
    "dataset",
    "minimum_repeats",
    "require_clean_csi",
    "minimum_target_cities",
    "minimum_source_cities",
    "minimum_banks_per_target_city",
    "minimum_independent_base_map_clusters_per_target_city",
    "minimum_banks_per_source_role",
    "minimum_independent_source_final_unseen_clusters",
    "minimum_independent_external_validation_clusters",
}
QUALIFICATION_KEYS = {
    "noise_floor_quantile",
    "repeat_noise_nmse_max",
    "clean_repeat_nmse_max",
    "physical_null_rms_max",
    "physical_active_rms_min",
    "latent_null_rms_max",
    "latent_active_rms_min",
    "response_physical_null_rms_max",
    "response_physical_active_rms_min",
    "response_latent_null_rms_max",
    "response_latent_active_rms_min",
    "minimum_active_units_per_bank",
    "minimum_null_units_per_bank",
    "teacher_reconstruction_nmse_max",
    "oracle_min_relative_improvement",
    "no_x_min_relative_improvement",
    "null_violation_rate_max",
    "shortcut_relative_improvement_max",
    "ridge",
    "bootstrap_resamples",
    "alignment_noop_quantile",
    "teacher_physical_active_agreement_min",
    "teacher_physical_null_agreement_min",
    "minimum_active_branching_fraction",
    "minimum_geometry_matched_wrong_action_fraction",
}
TEACHER_KEYS = {
    "steps",
    "batch_size",
    "learning_rate",
    "mask_fraction",
    "latent_dim",
    "encoder_layers",
    "decoder_layers",
}
MODEL_KEYS = {
    "state_dim",
    "map_dim",
    "hidden_dim",
    "mask_fraction",
    "learning_rate",
    "weight_decay",
    "attention_heads",
    "mask_bank_seed",
}
FACTORIAL_KEYS = {
    "arms",
    "steps",
    "batch_size",
    "alignment_weight",
    "response_weight",
    "active_margin",
    "effect_margin_scale",
    "effect_margin_cap",
    "alignment_primary_margin",
    "minimum_interaction_effect",
    "bootstrap_resamples",
    "pilot_steps",
    "alignment_bank_queries",
    "endpoint_physical_weight",
    "alignment_null_weight",
    "response_physical_weight",
    "response_delta_weight",
    "response_delta_physical_weight",
    "response_null_weight",
    "response_null_physical_weight",
    "natural_endpoint_weight",
}
LOCALIZATION_KEYS = {
    "label_budgets",
    "primary_budgets",
    "label_draws",
    "ridge",
    "maximum_city_regression",
    "minimum_city_improvement",
    "head_steps",
    "learning_rate",
    "failure_threshold_m",
    "sigma_min",
}
EVALUATION_KEYS = {
    "probe_steps",
    "probe_learning_rate",
    "probe_hidden_dim",
    "minimum_cgs_noninferiority",
    "minimum_response_noninferiority",
    "minimum_equal_flop_superiority",
    "minimum_concat_superiority",
    "null_score_equivalence_margin",
    "null_overclassification_rate_max",
    "c1_active_error_minimum_m",
    "c1_null_error_equivalence_margin_m",
    "resource_match_relative_tolerance",
    "scene_id_error_noninferiority_m",
    "scene_id_swap_direction_cosine_min",
    "shuffled_gain_fraction_max",
    "retention_minimum_effect",
    "bootstrap_resamples",
    "familywise_alpha",
    "minimum_alignment_superiority",
    "minimum_response_superiority",
    "response_null_equivalence_margin",
    "response_null_violation_rate_max",
    "minimum_native_probe_correlation",
}
RISK_KEYS = {
    "temperature_steps",
    "logistic_steps",
    "learning_rate",
    "l2_grid",
    "support_alpha",
    "common_support_minimum",
    "epsilon",
    "proposal_count",
    "proposal_height_step",
    "proposal_seed",
    "bootstrap_resamples",
    "calibration_ece_max",
    "calibration_brier_max",
    "calibration_nll_max",
    "outside_support_noninferiority_max",
    "aurc_minimum_improvement",
    "coverage_monotonic_tolerance",
    "audit_mixture",
}
PATH_KEYS = {
    "power_coverage",
    "zero_quantile",
    "balance_smd_max",
    "equivalence_margin",
    "minimum_trend_slope",
    "bootstrap_resamples",
    "minimum_effective_sample_size",
    "covariate_overlap_minimum",
}
EXTERNAL_VALIDITY_KEYS = {
    "minimum_active_direction_agreement",
    "null_equivalence_margin",
    "bootstrap_resamples",
}
LITERATURE_KEYS = {"maximum_search_age_days", "required_databases"}


def load_formal_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    config = read_strict_json(source)
    validate_formal_config(config)
    config["_config_path"] = str(source.resolve())
    return config


def validate_formal_config(config: object) -> None:
    if not isinstance(config, dict):
        raise ValueError("formal configuration root must be an object")
    public = {key for key in config if not key.startswith("_")}
    _exact_keys("configuration", public, TOP_LEVEL_KEYS)
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    if not isinstance(config["artifact_label"], str) or not config["artifact_label"].strip():
        raise ValueError("artifact_label must be a nonempty string")
    seeds = config["seeds"]
    if not _unique_nonnegative_int_list(seeds) or len(seeds) < 3:
        raise ValueError("seeds must contain at least three unique nonnegative integers")

    sections = (
        ("data", config["data"], DATA_KEYS),
        ("qualification", config["qualification"], QUALIFICATION_KEYS),
        ("teacher", config["teacher"], TEACHER_KEYS),
        ("model", config["model"], MODEL_KEYS),
        ("factorial", config["factorial"], FACTORIAL_KEYS),
        ("localization", config["localization"], LOCALIZATION_KEYS),
        ("evaluation", config["evaluation"], EVALUATION_KEYS),
        ("risk", config["risk"], RISK_KEYS),
        ("path", config["path"], PATH_KEYS),
        ("external_validity", config["external_validity"], EXTERNAL_VALIDITY_KEYS),
        ("literature", config["literature"], LITERATURE_KEYS),
    )
    for name, section, expected in sections:
        if not isinstance(section, dict):
            raise ValueError(f"{name} must be an object")
        _exact_keys(name, set(section), expected)

    data = config["data"]
    if not isinstance(data["dataset"], str) or not data["dataset"].strip():
        raise ValueError("data.dataset must be a nonempty path")
    _positive_int(data["minimum_repeats"], "data.minimum_repeats", minimum=2)
    if data["require_clean_csi"] is not True:
        raise ValueError("V6 requires data.require_clean_csi=true")
    _positive_int(data["minimum_target_cities"], "data.minimum_target_cities", minimum=2)
    _positive_int(data["minimum_source_cities"], "data.minimum_source_cities", minimum=2)
    _positive_int(
        data["minimum_banks_per_target_city"],
        "data.minimum_banks_per_target_city",
        minimum=2,
    )
    _positive_int(
        data["minimum_independent_base_map_clusters_per_target_city"],
        "data.minimum_independent_base_map_clusters_per_target_city",
        minimum=2,
    )
    _positive_int(
        data["minimum_banks_per_source_role"],
        "data.minimum_banks_per_source_role",
        minimum=1,
    )
    _positive_int(
        data["minimum_independent_source_final_unseen_clusters"],
        "data.minimum_independent_source_final_unseen_clusters",
        minimum=1,
    )
    _positive_int(
        data["minimum_independent_external_validation_clusters"],
        "data.minimum_independent_external_validation_clusters",
        minimum=0,
    )

    qualification = config["qualification"]
    _finite_number(
        qualification["noise_floor_quantile"],
        "qualification.noise_floor_quantile",
        minimum=0.0,
        maximum=1.0,
    )
    if not 0.0 < float(qualification["noise_floor_quantile"]) < 1.0:
        raise ValueError("qualification.noise_floor_quantile must be strictly between 0 and 1")
    for key in (
        "repeat_noise_nmse_max",
        "clean_repeat_nmse_max",
        "physical_null_rms_max",
        "physical_active_rms_min",
        "latent_null_rms_max",
        "latent_active_rms_min",
        "response_physical_null_rms_max",
        "response_physical_active_rms_min",
        "response_latent_null_rms_max",
        "response_latent_active_rms_min",
        "teacher_reconstruction_nmse_max",
        "oracle_min_relative_improvement",
        "no_x_min_relative_improvement",
        "null_violation_rate_max",
        "shortcut_relative_improvement_max",
        "ridge",
    ):
        _finite_number(qualification[key], f"qualification.{key}", minimum=0.0)
    if qualification["physical_null_rms_max"] >= qualification["physical_active_rms_min"]:
        raise ValueError("physical null threshold must be below the active threshold")
    if qualification["latent_null_rms_max"] >= qualification["latent_active_rms_min"]:
        raise ValueError("latent null threshold must be below the active threshold")
    if qualification["response_physical_null_rms_max"] >= qualification["response_physical_active_rms_min"]:
        raise ValueError("response physical null threshold must be below the active threshold")
    if qualification["response_latent_null_rms_max"] >= qualification["response_latent_active_rms_min"]:
        raise ValueError("response latent null threshold must be below the active threshold")
    for key in ("null_violation_rate_max", "shortcut_relative_improvement_max"):
        if float(qualification[key]) > 1.0:
            raise ValueError(f"qualification.{key} must not exceed 1")
    _positive_int(
        qualification["minimum_active_units_per_bank"],
        "qualification.minimum_active_units_per_bank",
    )
    _positive_int(
        qualification["minimum_null_units_per_bank"],
        "qualification.minimum_null_units_per_bank",
    )
    _positive_int(qualification["bootstrap_resamples"], "qualification.bootstrap_resamples", minimum=100)
    for key in (
        "alignment_noop_quantile",
        "teacher_physical_active_agreement_min",
        "teacher_physical_null_agreement_min",
        "minimum_active_branching_fraction",
        "minimum_geometry_matched_wrong_action_fraction",
    ):
        _finite_number(qualification[key], f"qualification.{key}", minimum=0.0, maximum=1.0)

    teacher = config["teacher"]
    for key in ("steps", "batch_size", "latent_dim", "encoder_layers", "decoder_layers"):
        _positive_int(teacher[key], f"teacher.{key}")
    _finite_number(teacher["learning_rate"], "teacher.learning_rate", minimum=1e-12)
    _finite_number(teacher["mask_fraction"], "teacher.mask_fraction", minimum=0.0, maximum=0.95)
    if float(teacher["mask_fraction"]) != FROZEN_RANDOM_MASK_FRACTION:
        raise ValueError(
            f"teacher.mask_fraction must equal the frozen V6 value {FROZEN_RANDOM_MASK_FRACTION}"
        )

    model = config["model"]
    for key in ("state_dim", "map_dim", "hidden_dim"):
        _positive_int(model[key], f"model.{key}", minimum=4)
    _finite_number(model["mask_fraction"], "model.mask_fraction", minimum=0.0, maximum=0.95)
    if float(model["mask_fraction"]) != FROZEN_RANDOM_MASK_FRACTION:
        raise ValueError(
            "model.mask_fraction is the frozen mask-bank contract and must equal "
            f"{FROZEN_RANDOM_MASK_FRACTION}"
        )
    _finite_number(model["learning_rate"], "model.learning_rate", minimum=1e-12)
    _finite_number(model["weight_decay"], "model.weight_decay", minimum=0.0)
    _positive_int(model["attention_heads"], "model.attention_heads")
    _positive_int(model["mask_bank_seed"], "model.mask_bank_seed", minimum=0)
    if int(model["state_dim"]) % int(model["attention_heads"]):
        raise ValueError("model.state_dim must be divisible by model.attention_heads")
    if int(teacher["latent_dim"]) != int(model["state_dim"]):
        raise ValueError("teacher.latent_dim must equal model.state_dim for shared CSI initialization")

    factorial = config["factorial"]
    if factorial["arms"] != list(ARMS):
        raise ValueError(f"factorial.arms must be exactly {list(ARMS)}")
    _positive_int(factorial["steps"], "factorial.steps")
    _positive_int(factorial["batch_size"], "factorial.batch_size", minimum=2)
    for key in (
        "alignment_weight",
        "response_weight",
        "active_margin",
        "effect_margin_scale",
        "effect_margin_cap",
        "minimum_interaction_effect",
    ):
        _finite_number(factorial[key], f"factorial.{key}", minimum=0.0)
    if factorial["alignment_primary_margin"] != "fixed":
        raise ValueError("V6 P0 requires factorial.alignment_primary_margin='fixed'")
    _positive_int(factorial["bootstrap_resamples"], "factorial.bootstrap_resamples", minimum=100)
    _positive_int(factorial["pilot_steps"], "factorial.pilot_steps")
    _positive_int(factorial["alignment_bank_queries"], "factorial.alignment_bank_queries")
    for key in (
        "endpoint_physical_weight",
        "alignment_null_weight",
        "response_physical_weight",
        "response_delta_weight",
        "response_delta_physical_weight",
        "response_null_weight",
        "response_null_physical_weight",
        "natural_endpoint_weight",
    ):
        _finite_number(factorial[key], f"factorial.{key}", minimum=0.0)

    localization = config["localization"]
    budgets = localization["label_budgets"]
    primary = localization["primary_budgets"]
    if not _unique_nonnegative_int_list(budgets) or budgets != sorted(budgets) or budgets[0] != 0:
        raise ValueError("localization.label_budgets must be sorted, unique, and begin with 0")
    if not _unique_nonnegative_int_list(primary) or not set(primary).issubset(budgets):
        raise ValueError("localization.primary_budgets must be a nonempty subset of label_budgets")
    _positive_int(localization["label_draws"], "localization.label_draws")
    _finite_number(localization["ridge"], "localization.ridge", minimum=0.0)
    _positive_int(localization["head_steps"], "localization.head_steps")
    _finite_number(localization["learning_rate"], "localization.learning_rate", minimum=1e-12)
    _finite_number(localization["failure_threshold_m"], "localization.failure_threshold_m", minimum=0.0)
    _finite_number(localization["sigma_min"], "localization.sigma_min", minimum=1e-12)
    _finite_number(
        localization["maximum_city_regression"],
        "localization.maximum_city_regression",
        minimum=0.0,
    )
    _finite_number(
        localization["minimum_city_improvement"],
        "localization.minimum_city_improvement",
        minimum=1e-12,
    )

    evaluation = config["evaluation"]
    _positive_int(evaluation["probe_steps"], "evaluation.probe_steps")
    _positive_int(evaluation["probe_hidden_dim"], "evaluation.probe_hidden_dim")
    _finite_number(evaluation["probe_learning_rate"], "evaluation.probe_learning_rate", minimum=1e-12)
    for key in (
        "minimum_cgs_noninferiority",
        "minimum_response_noninferiority",
        "minimum_equal_flop_superiority",
        "minimum_concat_superiority",
    ):
        _finite_number(evaluation[key], f"evaluation.{key}")
    _finite_number(
        evaluation["null_score_equivalence_margin"],
        "evaluation.null_score_equivalence_margin",
        minimum=0.0,
    )
    _finite_number(
        evaluation["null_overclassification_rate_max"],
        "evaluation.null_overclassification_rate_max",
        minimum=0.0,
        maximum=1.0,
    )
    _finite_number(
        evaluation["c1_active_error_minimum_m"],
        "evaluation.c1_active_error_minimum_m",
        minimum=0.0,
    )
    _finite_number(
        evaluation["c1_null_error_equivalence_margin_m"],
        "evaluation.c1_null_error_equivalence_margin_m",
        minimum=0.0,
    )
    _finite_number(
        evaluation["resource_match_relative_tolerance"],
        "evaluation.resource_match_relative_tolerance",
        minimum=0.0,
        maximum=1.0,
    )
    _finite_number(
        evaluation["scene_id_error_noninferiority_m"],
        "evaluation.scene_id_error_noninferiority_m",
        minimum=0.0,
    )
    _finite_number(
        evaluation["scene_id_swap_direction_cosine_min"],
        "evaluation.scene_id_swap_direction_cosine_min",
        minimum=-1.0,
        maximum=1.0,
    )
    _finite_number(
        evaluation["shuffled_gain_fraction_max"],
        "evaluation.shuffled_gain_fraction_max",
        minimum=0.0,
        maximum=1.0,
    )
    _finite_number(
        evaluation["retention_minimum_effect"],
        "evaluation.retention_minimum_effect",
        minimum=0.0,
    )
    _positive_int(
        evaluation["bootstrap_resamples"],
        "evaluation.bootstrap_resamples",
        minimum=100,
    )
    _finite_number(
        evaluation["familywise_alpha"],
        "evaluation.familywise_alpha",
        minimum=1e-12,
        maximum=0.5,
    )
    for key in ("minimum_alignment_superiority", "minimum_response_superiority"):
        _finite_number(evaluation[key], f"evaluation.{key}", minimum=1e-12)
    _finite_number(
        evaluation["response_null_equivalence_margin"],
        "evaluation.response_null_equivalence_margin",
        minimum=0.0,
    )
    _finite_number(
        evaluation["response_null_violation_rate_max"],
        "evaluation.response_null_violation_rate_max",
        minimum=0.0,
        maximum=1.0,
    )
    _finite_number(
        evaluation["minimum_native_probe_correlation"],
        "evaluation.minimum_native_probe_correlation",
        minimum=-1.0,
        maximum=1.0,
    )

    risk = config["risk"]
    for key in ("temperature_steps", "logistic_steps"):
        _positive_int(risk[key], f"risk.{key}")
    _finite_number(risk["learning_rate"], "risk.learning_rate", minimum=1e-12)
    if not isinstance(risk["l2_grid"], list) or not risk["l2_grid"]:
        raise ValueError("risk.l2_grid must be a nonempty list")
    for index, value in enumerate(risk["l2_grid"]):
        _finite_number(value, f"risk.l2_grid[{index}]", minimum=0.0)
    _finite_number(risk["support_alpha"], "risk.support_alpha", minimum=0.0, maximum=1.0)
    _finite_number(risk["common_support_minimum"], "risk.common_support_minimum", minimum=0.0, maximum=1.0)
    _finite_number(risk["epsilon"], "risk.epsilon", minimum=1e-15)
    _positive_int(risk["proposal_count"], "risk.proposal_count", minimum=1)
    _finite_number(risk["proposal_height_step"], "risk.proposal_height_step", minimum=1e-12)
    _positive_int(risk["proposal_seed"], "risk.proposal_seed", minimum=0)
    _positive_int(risk["bootstrap_resamples"], "risk.bootstrap_resamples", minimum=100)
    for key in (
        "calibration_ece_max",
        "calibration_brier_max",
        "calibration_nll_max",
        "outside_support_noninferiority_max",
        "coverage_monotonic_tolerance",
    ):
        _finite_number(risk[key], f"risk.{key}", minimum=0.0)
    _finite_number(
        risk["aurc_minimum_improvement"],
        "risk.aurc_minimum_improvement",
        minimum=1e-12,
    )
    mixture = risk["audit_mixture"]
    if not isinstance(mixture, dict) or set(mixture) != {"correct", "active", "gray", "null"}:
        raise ValueError("risk.audit_mixture must contain exact correct/active/gray/null proportions")
    for key, value in mixture.items():
        _finite_number(value, f"risk.audit_mixture.{key}", minimum=0.0)
        if float(value) <= 0.0:
            raise ValueError(f"risk.audit_mixture.{key} must be strictly positive")
    if not math.isclose(sum(float(value) for value in mixture.values()), 1.0, abs_tol=1e-12):
        raise ValueError("risk.audit_mixture proportions must sum to 1")

    path = config["path"]
    _finite_number(path["power_coverage"], "path.power_coverage", minimum=0.0, maximum=1.0)
    if not 0.0 < float(path["power_coverage"]) < 1.0:
        raise ValueError("path.power_coverage must be strictly between zero and one")
    _finite_number(path["zero_quantile"], "path.zero_quantile", minimum=0.0, maximum=1.0)
    _finite_number(path["balance_smd_max"], "path.balance_smd_max", minimum=0.0)
    _finite_number(path["equivalence_margin"], "path.equivalence_margin", minimum=0.0)
    _finite_number(path["minimum_trend_slope"], "path.minimum_trend_slope")
    _positive_int(path["bootstrap_resamples"], "path.bootstrap_resamples", minimum=100)
    _finite_number(
        path["minimum_effective_sample_size"],
        "path.minimum_effective_sample_size",
        minimum=2.0,
    )
    _finite_number(
        path["covariate_overlap_minimum"],
        "path.covariate_overlap_minimum",
        minimum=0.0,
        maximum=1.0,
    )

    external = config["external_validity"]
    _finite_number(
        external["minimum_active_direction_agreement"],
        "external_validity.minimum_active_direction_agreement",
        minimum=0.0,
        maximum=1.0,
    )
    _finite_number(
        external["null_equivalence_margin"],
        "external_validity.null_equivalence_margin",
        minimum=0.0,
    )
    _positive_int(
        external["bootstrap_resamples"],
        "external_validity.bootstrap_resamples",
        minimum=100,
    )
    literature = config["literature"]
    _positive_int(
        literature["maximum_search_age_days"],
        "literature.maximum_search_age_days",
        minimum=1,
    )
    if not isinstance(literature["required_databases"], list) or not literature["required_databases"]:
        raise ValueError("literature.required_databases must be a nonempty list")
    if any(not isinstance(value, str) or not value.strip() for value in literature["required_databases"]):
        raise ValueError("literature.required_databases must contain nonempty strings")


def resolve_dataset_path(config: dict[str, Any]) -> Path:
    path = Path(config["data"]["dataset"])
    if path.is_absolute():
        return path
    return (Path(config["_config_path"]).parent / path).resolve()


def public_formal_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _exact_keys(name: str, actual: set[str], expected: set[str]) -> None:
    if actual != expected:
        raise ValueError(
            f"{name} fields must be exact; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _positive_int(value: object, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite_number(
    value: object,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


def _unique_nonnegative_int_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(type(item) is int and item >= 0 for item in value)
        and len(value) == len(set(value))
    )
