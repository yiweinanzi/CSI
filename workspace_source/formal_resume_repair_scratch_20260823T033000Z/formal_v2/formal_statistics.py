from __future__ import annotations

import numpy as np


def paired_cluster_interval(
    cluster_ids: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    clusters, differences = _cluster_differences(cluster_ids, first, second)
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        selected = rng.integers(0, differences.size, size=differences.size)
        estimates[index] = float(np.mean(differences[selected]))
    return {
        "cluster_count": int(clusters.size),
        "paired_mean_difference": float(np.mean(differences)),
        "ci95_low": float(np.percentile(estimates, 100.0 * float(alpha) / 2.0)),
        "ci95_high": float(np.percentile(estimates, 100.0 * (1.0 - float(alpha) / 2.0))),
        "confidence_level": 1.0 - float(alpha),
    }


def paired_sign_flip_test(
    cluster_ids: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    seed: int,
    maximum_draws: int = 100000,
    null_difference: float = 0.0,
) -> dict[str, float | int | str]:
    clusters, differences = _cluster_differences(cluster_ids, first, second)
    center = float(null_difference)
    if not np.isfinite(center):
        raise ValueError("sign-flip null difference must be finite")
    centered_differences = differences - center
    observed = abs(float(np.mean(centered_differences)))
    count = differences.size
    if count <= 16:
        assignments = np.arange(2**count, dtype=np.uint64)[:, None]
        shifts = np.arange(count, dtype=np.uint64)[None, :]
        signs = 2.0 * ((assignments >> shifts) & 1).astype(np.float64) - 1.0
        null_values = np.abs(np.mean(signs * centered_differences[None, :], axis=1))
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(int(maximum_draws), count))
        null_values = np.abs(np.mean(signs * centered_differences[None, :], axis=1))
        method = "monte_carlo"
    exceed = int(np.sum(null_values >= observed - 1e-15))
    p_value = (
        float(exceed / null_values.size)
        if method == "exact"
        else float((exceed + 1) / (null_values.size + 1))
    )
    return {
        "cluster_count": int(clusters.size),
        "method": method,
        "draws": int(null_values.size),
        "finite_sample_correction": "none_exact_enumeration"
        if method == "exact"
        else "plus_one_monte_carlo",
        "null_difference": center,
        "p_value_two_sided": p_value,
    }


def interaction_cluster_interval(
    cluster_ids: np.ndarray,
    endpoint: np.ndarray,
    alignment: np.ndarray,
    response: np.ndarray,
    full: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    arrays = [np.asarray(value, dtype=np.float64) for value in (endpoint, alignment, response, full)]
    identifiers = np.asarray(cluster_ids).astype(str)
    if any(array.shape != identifiers.shape for array in arrays):
        raise ValueError("interaction arrays must share shape")
    clusters = np.unique(identifiers)
    if clusters.size < 2:
        raise ValueError("at least two independent scene-bank clusters are required")
    cluster_values = []
    for cluster in clusters:
        mask = identifiers == cluster
        means = [float(np.mean(array[mask])) for array in arrays]
        cluster_values.append(means[3] - means[1] - means[2] + means[0])
    values = np.asarray(cluster_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        selected = rng.integers(0, values.size, size=values.size)
        estimates[index] = float(np.mean(values[selected]))
    return {
        "cluster_count": int(clusters.size),
        "interaction": float(np.mean(values)),
        "ci95_low": float(np.percentile(estimates, 2.5)),
        "ci95_high": float(np.percentile(estimates, 97.5)),
    }


def _cluster_differences(
    cluster_ids: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    identifiers = np.asarray(cluster_ids).astype(str)
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if identifiers.shape != first_array.shape or first_array.shape != second_array.shape:
        raise ValueError("paired arrays and cluster_ids must share shape")
    if not np.all(np.isfinite(first_array)) or not np.all(np.isfinite(second_array)):
        raise ValueError("paired statistics require finite values")
    clusters = np.unique(identifiers)
    if clusters.size < 2:
        raise ValueError("at least two independent scene-bank clusters are required")
    differences = np.asarray(
        [float(np.mean(first_array[identifiers == cluster] - second_array[identifiers == cluster])) for cluster in clusters],
        dtype=np.float64,
    )
    return clusters, differences


def exact_factorial_utilities(rows: list[dict], primary_budgets: list[int] | tuple[int, ...]) -> dict:
    """Compute V6 J_a with equal city, k, bank, seed and draw layers."""
    _validate_canonical_bank_ownership(rows)
    budgets = tuple(int(value) for value in primary_budgets)
    arms = sorted({str(row["arm"]) for row in rows})
    cities = sorted({str(row["city_id"]) for row in rows})
    utilities = {
        arm: _macro_utility(rows, arm, cities, budgets)
        for arm in arms
    }
    required = {"endpoint", "alignment", "response", "full"}
    if not required.issubset(utilities):
        raise ValueError("factorial utility requires all four V6 arms")
    interaction = (
        utilities["full"]
        - utilities["alignment"]
        - utilities["response"]
        + utilities["endpoint"]
    )
    return {
        "utilities": utilities,
        "interaction": float(interaction),
        "full_vs_alignment": float(utilities["full"] - utilities["alignment"]),
        "full_vs_response": float(utilities["full"] - utilities["response"]),
        "cities": cities,
        "primary_budgets": list(budgets),
    }


def hierarchical_factorial_interval(
    rows: list[dict],
    primary_budgets: list[int] | tuple[int, ...],
    resamples: int,
    seed: int,
) -> dict:
    """Paired city-fixed bootstrap over base-map cluster, seed, and k>0 draw."""
    _validate_canonical_bank_ownership(rows)
    budgets = tuple(int(value) for value in primary_budgets)
    cities = sorted({str(row["city_id"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    draws = {
        (city, budget): sorted(
            {
                int(row["draw"])
                for row in rows
                if str(row["city_id"]) == city and int(row["budget"]) == budget
            }
        )
        for city in cities
        for budget in budgets
    }
    clusters = {
        city: sorted(
            {
                _independent_unit(row)
                for row in rows
                if str(row["city_id"]) == city
            }
        )
        for city in cities
    }
    lookup = _factorial_cell_lookup(rows, budgets)
    rng = np.random.default_rng(int(seed))
    samples = {name: np.empty(int(resamples), dtype=np.float64) for name in (
        "interaction", "full_vs_alignment", "full_vs_response"
    )}
    for index in range(int(resamples)):
        selected_seeds = [seeds[value] for value in rng.integers(0, len(seeds), size=len(seeds))]
        selected_clusters = {
            city: [
                clusters[city][value]
                for value in rng.integers(
                    0, len(clusters[city]), size=len(clusters[city])
                )
            ]
            for city in cities
        }
        selected_draws = {}
        for city in cities:
            for budget in budgets:
                available = draws[(city, budget)]
                if not available:
                    raise ValueError("missing label draw in hierarchical bootstrap")
                selected_draws[(city, budget)] = [
                    available[value]
                    for value in rng.integers(0, len(available), size=len(available))
                ]
        utility = {}
        for arm in ("endpoint", "alignment", "response", "full"):
            city_values = []
            for city in cities:
                budget_values = []
                for budget in budgets:
                    cluster_values = []
                    for cluster in selected_clusters[city]:
                        values = [
                            lookup[(arm, city, budget, cluster, selected_seed, selected_draw)]
                            for selected_seed in selected_seeds
                            for selected_draw in selected_draws[(city, budget)]
                        ]
                        cluster_values.append(float(np.mean(values)))
                    budget_values.append(float(np.mean(cluster_values)))
                city_values.append(float(np.mean(budget_values)))
            utility[arm] = float(np.mean(city_values))
        samples["interaction"][index] = utility["full"] - utility["alignment"] - utility["response"] + utility["endpoint"]
        samples["full_vs_alignment"][index] = utility["full"] - utility["alignment"]
        samples["full_vs_response"][index] = utility["full"] - utility["response"]
    exact = exact_factorial_utilities(rows, budgets)
    records = {
        "interaction": _interval_record(exact["interaction"], samples["interaction"]),
        "full_vs_alignment": _interval_record(exact["full_vs_alignment"], samples["full_vs_alignment"]),
        "full_vs_response": _interval_record(exact["full_vs_response"], samples["full_vs_response"]),
    }
    # The three primary G4 hypotheses share each synchronized bootstrap draw.
    # A max-deviation critical value gives one simultaneous 95% family rather
    # than treating three marginal 95% intervals as independent gates.
    estimates = {
        "interaction": float(exact["interaction"]),
        "full_vs_alignment": float(exact["full_vs_alignment"]),
        "full_vs_response": float(exact["full_vs_response"]),
    }
    maximum_deviation = np.max(
        np.column_stack(
            [
                np.abs(samples[name] - estimates[name])
                for name in ("interaction", "full_vs_alignment", "full_vs_response")
            ]
        ),
        axis=1,
    )
    simultaneous_critical = float(np.percentile(maximum_deviation, 95.0))
    for name, record in records.items():
        record["familywise_ci95_low"] = estimates[name] - simultaneous_critical
        record["familywise_ci95_high"] = estimates[name] + simultaneous_critical
    return {
        "resampling_layers": [
            "base_map_cluster_within_fixed_city",
            "training_seed",
            "k_positive_label_draw",
        ],
        "resamples": int(resamples),
        "familywise_method": "synchronized_bootstrap_max_absolute_deviation",
        "familywise_alpha": 0.05,
        "simultaneous_critical_value": simultaneous_critical,
        **records,
    }


def bank_only_factorial_interval(
    rows: list[dict],
    primary_budgets: list[int] | tuple[int, ...],
    resamples: int,
    seed: int,
) -> dict:
    _validate_canonical_bank_ownership(rows)
    collapsed = _collapsed_cells(rows, primary_budgets)
    rng = np.random.default_rng(int(seed))
    cities = sorted({key[1] for key in collapsed})
    budgets = tuple(int(value) for value in primary_budgets)
    clusters = {
        city: sorted({key[3] for key in collapsed if key[1] == city})
        for city in cities
    }
    values = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        sampled_clusters = {
            city: [
                clusters[city][item]
                for item in rng.integers(
                    0, len(clusters[city]), size=len(clusters[city])
                )
            ]
            for city in cities
        }
        utility = {}
        for arm in ("endpoint", "alignment", "response", "full"):
            utility[arm] = float(
                np.mean(
                    [
                        np.mean(
                            [
                                np.mean(
                                    [
                                        collapsed[(arm, city, budget, cluster)]
                                        for cluster in sampled_clusters[city]
                                    ]
                                )
                                for budget in budgets
                            ]
                        )
                        for city in cities
                    ]
                )
            )
        values[index] = utility["full"] - utility["alignment"] - utility["response"] + utility["endpoint"]
    exact = exact_factorial_utilities(rows, budgets)["interaction"]
    return {
        "interaction": _interval_record(exact, values),
        "resampling_layers": ["base_map_cluster_within_fixed_city"],
    }


def leave_one_factorial_sensitivity(
    rows: list[dict], primary_budgets: list[int] | tuple[int, ...]
) -> dict:
    _validate_canonical_bank_ownership(rows)
    seeds = sorted({int(row["seed"]) for row in rows})
    positive_draws = sorted({int(row["draw"]) for row in rows if int(row["budget"]) > 0})
    leave_seed = [
        {
            "omitted_seed": seed,
            "interaction": exact_factorial_utilities(
                [row for row in rows if int(row["seed"]) != seed], primary_budgets
            )["interaction"],
        }
        for seed in seeds
    ]
    leave_draw = [
        {
            "omitted_draw": draw,
            "interaction": exact_factorial_utilities(
                [row for row in rows if int(row["budget"]) == 0 or int(row["draw"]) != draw],
                primary_budgets,
            )["interaction"],
        }
        for draw in positive_draws
    ]
    return {"leave_one_seed": leave_seed, "leave_one_positive_budget_draw": leave_draw}


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Holm adjustment requires a nonempty one-dimensional p-value family")
    if not np.all(np.isfinite(values)):
        raise ValueError("Holm adjustment requires finite p-values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Holm adjustment requires p-values in [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = values.size
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * float(values[index])))
        adjusted[index] = running
    return adjusted.tolist()


def interval_decision(interval: dict, *, threshold: float, relation: str) -> bool:
    if relation == "superiority":
        return float(interval["ci95_low"]) > float(threshold)
    if relation == "noninferiority":
        return float(interval["ci95_low"]) >= -abs(float(threshold))
    if relation == "equivalence":
        margin = abs(float(threshold))
        return float(interval["ci95_low"]) > -margin and float(interval["ci95_high"]) < margin
    raise ValueError(f"unknown interval decision relation: {relation}")


def _macro_utility(rows, arm, cities, budgets):
    city_values = []
    for city in cities:
        budget_values = []
        for budget in budgets:
            clusters = sorted(
                {
                    _independent_unit(row)
                    for row in rows
                    if str(row["arm"]) == arm
                    and str(row["city_id"]) == city
                    and int(row["budget"]) == budget
                }
            )
            if not clusters:
                raise ValueError(f"missing factorial cell for {arm}/{city}/k={budget}")
            cluster_values = []
            for cluster in clusters:
                selected = [
                    row
                    for row in rows
                    if str(row["arm"]) == arm
                    and str(row["city_id"]) == city
                    and int(row["budget"]) == budget
                    and _independent_unit(row) == cluster
                ]
                cluster_values.append(_layered_cell_mean(selected))
            budget_values.append(float(np.mean(cluster_values)))
        city_values.append(float(np.mean(budget_values)))
    return float(np.mean(city_values))


def _collapsed_cells(rows, budgets):
    output = {}
    keys = {
        (
            str(row["arm"]),
            str(row["city_id"]),
            int(row["budget"]),
            _independent_unit(row),
        )
        for row in rows
        if int(row["budget"]) in set(int(value) for value in budgets)
    }
    for key in keys:
        selected = [
            row
            for row in rows
            if (
                str(row["arm"]),
                str(row["city_id"]),
                int(row["budget"]),
                _independent_unit(row),
            )
            == key
        ]
        output[key] = _layered_cell_mean(selected)
    return output


def _factorial_cell_lookup(rows, budgets):
    grouped = {}
    allowed = set(int(value) for value in budgets)
    for row in rows:
        if int(row["budget"]) not in allowed:
            continue
        key = (
            str(row["arm"]),
            str(row["city_id"]),
            int(row["budget"]),
            _independent_unit(row),
            int(row["seed"]),
            int(row["draw"]),
        )
        grouped.setdefault(key, {}).setdefault(_bank_unit(row), []).append(
            float(row["utility_neg_log_median"])
        )
    if not grouped:
        raise ValueError("factorial lookup is empty")
    return {
        key: _canonical_bank_macro_mean(by_bank)
        for key, by_bank in grouped.items()
    }


def _layered_cell_mean(rows: list[dict]) -> float:
    """Equal bank, positive-budget draw, and training-seed macro mean."""
    if not rows:
        raise ValueError("factorial macro cell is empty")
    seeds = sorted({int(row["seed"]) for row in rows})
    seed_values = []
    for seed in seeds:
        draws = sorted({int(row["draw"]) for row in rows if int(row["seed"]) == seed})
        draw_values = []
        for draw in draws:
            by_bank = {}
            for row in rows:
                if int(row["seed"]) == seed and int(row["draw"]) == draw:
                    by_bank.setdefault(_bank_unit(row), []).append(
                        float(row["utility_neg_log_median"])
                    )
            draw_values.append(_canonical_bank_macro_mean(by_bank))
        seed_values.append(float(np.mean(draw_values)))
    return float(np.mean(seed_values))


def _canonical_bank_macro_mean(by_bank: dict[str, list[float]]) -> float:
    values = []
    for bank, copies in by_bank.items():
        value = copies[0]
        if any(copy != value for copy in copies[1:]):
            raise ValueError(
                f"conflicting utility values for canonical bank {bank}"
            )
        values.append(value)
    return float(np.mean(values))


def _validate_canonical_bank_ownership(rows: list[dict]) -> None:
    owners = {}
    for row in rows:
        bank = _bank_unit(row)
        owner = (_independent_unit(row), str(row["city_id"]))
        previous = owners.setdefault(bank, owner)
        if previous != owner:
            raise ValueError(
                f"canonical bank {bank} is assigned to multiple independent units or cities"
            )


def _independent_unit(row: dict) -> str:
    digest = row.get("canonical_base_map_digest")
    if digest is not None and str(digest).strip():
        return "canonical-foundation:" + str(digest)
    value = row.get("base_map_cluster_id")
    if value is None or not str(value).strip():
        raise ValueError("factorial rows require base_map_cluster_id")
    return str(value)


def _bank_unit(row: dict) -> str:
    value = row.get("canonical_bank_digest")
    if value is not None and str(value).strip():
        return str(value)
    # Compatibility for historical artifacts; newly generated formal rows are
    # always content-bound by canonical_bank_digest.
    value = row.get("bank_id")
    if value is None or not str(value).strip():
        raise ValueError("factorial rows require canonical_bank_digest or bank_id")
    return "legacy-bank-id:" + str(value)


def _interval_record(estimate: float, samples: np.ndarray) -> dict[str, float]:
    return {
        "estimate": float(estimate),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
    }
