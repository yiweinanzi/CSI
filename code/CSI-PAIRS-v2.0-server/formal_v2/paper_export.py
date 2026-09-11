"""Six tables from experiment rows. Completeness never means a hypothesis won."""
from __future__ import annotations
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from .formal_config import ARMS
from .formal_evidence import config_sha256
from .paper_suite import cell, write_table, atomic_json, TITLES, numeric_tree, latex_escape
from .paper_controls import CONTROLS
from .paper_maps import METHODS
from .external_adapters.wigatr_protocol import SIX_CONDITIONS


def rows(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def macro(values, field):
    """Equal foundation, bank, seed and draw, never overweight duplicated banks."""
    groups = defaultdict(list)
    for row in values:
        value = row.get(field)
        if value in (None, "") or not math.isfinite(float(value)):
            continue
        key = (row.get("canonical_base_map_digest", row.get("base_map_cluster_id", "all")),
               row.get("canonical_bank_digest", row.get("bank_id", "all")), row.get("seed", "all"), row.get("draw", "0"))
        groups[key].append(float(value))
    for _ in range(4):
        reduced = defaultdict(list)
        for key, group in groups.items():
            reduced[key[:-1]].append(statistics.mean(group))
        groups = reduced
    return statistics.mean(groups[()]) if groups else None


def verify_comparison(reference, candidates):
    """Compare actual support and query identities, rather than approval receipts."""
    keys = ("seed", "city_id", "bank_id", "budget", "draw")
    expected = {tuple(row[k] for k in keys): (row["support_position_ids"], row["query_position_ids"]) for row in reference}
    actual = {tuple(row[k] for k in keys): (row["support_position_ids"], row["query_position_ids"]) for row in candidates}
    if len(actual) != len(candidates):
        raise ValueError("Duplicate baseline localization records")
    if actual != expected:
        raise ValueError("Baseline support/query coverage differs from the four-arm experiment")


def display_main1(output, values):
    fields = ("cgs_auroc", "overclassification_rate", "unified_response_probe_active_patch_nmse",
              "native_target_free_full_channel_nmse", "native_null_violation_rate")
    header = ["Method", "City", "AUROC", "Null overclassification", "Probe NMSE", "Native NMSE", "Native null violation"]
    index = {(row["method"], row["city"], row["metric"]): row for row in values}
    body = []
    for method, city in dict.fromkeys((row["method"], row["city"]) for row in values):
        line = [method, city.removeprefix("target-")]
        for field in fields:
            row = index[(method, city, field)]
            line.append("--" if row["value"] == "" else f"{row['value']:.4g}" + ("*" if row["status"] != "COMPLETE" else ""))
        body.append(line)
    note = "Equal foundation/bank/seed means. -- = missing; * = incomplete coverage. Rates are fractions."
    lines = ["# " + TITLES["main1"], "", note, "", "| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    (output / "main1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex = ["% " + note, r"\begin{tabular}{llrrrrr}", r"\toprule", " & ".join(latex_escape(v) for v in header) + r" \\", r"\midrule"]
    tex += [" & ".join(latex_escape(v) for v in row) + r" \\" for row in body]
    (output / "main1.tex").write_text("\n".join(tex + [r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")


def export(root, config, dataset_hash, *, fixture=False):
    root = Path(root)
    tables = {key: [] for key in TITLES}
    seeds = {str(seed) for seed in config["seeds"]}
    source_files = set()

    def read(path):
        result = rows(root / path)
        if result:
            source_files.add(str(path))
        for row in result:
            if row.get("dataset_sha256") != dataset_hash:
                raise ValueError(f"Wrong dataset in {path}")
            if row.get("config_sha256") != config_sha256(config):
                raise ValueError(f"Mixed experiment configurations in {path}")
            if not fixture and (row.get("scientific_use") == "FORBIDDEN" or row.get("fixture", "").lower() == "true"):
                raise ValueError("Fixture numbers cannot populate formal paper tables")
        return result

    banks = read("factorial/localization_per_bank.csv")
    cities = sorted({row["city_id"] for row in banks}) or ["target-boston", "target-seattle"]
    reference = [row for row in banks if row["arm"] == "full"]
    expected = {(row["seed"], row["city_id"], row["bank_id"]) for row in reference if int(row["budget"]) == 0}

    def metric(table, panel, method, city, budget, field, values, source, *, complete=True, output_field=None, note=""):
        value = macro(values, field)
        status = "COMPLETE" if value is not None and complete and all(row.get(field) not in (None, "") for row in values) else "PARTIAL" if value is not None else "MISSING"
        tables[table].append(cell(panel, method, city, budget, output_field or field,
            value if value is not None else "", status, len(values), source, note))

    for filename, fields, panel in (
        ("cgs_per_bank.csv", ["cgs_auroc"], "alignment"),
        ("compatibility_route_distributions.csv", ["overclassification_rate"], "null"),
        ("response_per_bank.csv", ["unified_response_probe_active_patch_nmse", "native_target_free_full_channel_nmse", "native_null_violation_rate"], "response")):
        source = "core/evaluation/" + filename
        records = read(source)
        for arm in ARMS:
            for city in cities:
                selected = [row for row in records if row["arm"] == arm and row["city_id"] == city]
                complete = {(row["seed"], row["city_id"], row["bank_id"]) for row in selected} == {key for key in expected if key[1] == city} and bool(selected)
                for field in fields:
                    metric("main1", panel, arm, city, "", field, selected, source, complete=complete)

    all_localization = list(banks)
    for name in ("Wi-GATr", "PMNet"):
        baseline = []
        for seed in config["seeds"]:
            baseline.extend(read(f"map_methods/{name}/{seed}/localization_per_bank.csv"))
        if baseline:
            # Partial runs can be exported; only compare shared cells until all seeds finish.
            matching_reference = [row for row in reference if row["seed"] in {v["seed"] for v in baseline}]
            verify_comparison(matching_reference, baseline)
        all_localization.extend(baseline)
    for method in (*ARMS, "Wi-GATr", "PMNet"):
        for city in cities:
            for budget in config["localization"]["label_budgets"]:
                selected = [row for row in all_localization if row["arm"] == method and row["city_id"] == city and int(row["budget"]) == budget]
                ref = [row for row in reference if row["city_id"] == city and int(row["budget"]) == budget]
                complete = len(selected) == len(ref) and {row["seed"] for row in selected} == seeds
                for field in ("median_error_m", "p90_error_m"):
                    metric("main2", "localization", method, city, budget, field, selected, "localization_per_bank.csv",
                        complete=complete, output_field="mean_bank_" + field,
                        note="Equal foundation/bank/seed/draw means. Power methods: native inverse + shared source/few-shot position head.")

    training = read("factorial/training_summary.csv")
    for arm in ARMS:
        for field in ("parameters", "measured_flops_per_step", "elapsed_seconds"):
            metric("s1", "encoder_cost", arm, "", "", field, [row for row in training if row["arm"] == arm], "factorial/training_summary.csv")
    if banks:
        from .formal_factorial import _factorial_statistics
        typed = [{**row, **{key: int(row[key]) for key in ("seed", "budget", "draw")},
                  **{key: float(row[key]) for key in ("median_error_m", "p90_error_m", "utility_neg_log_median")}} for row in banks]
        statistics_path = root / "factorial/factorial_statistics.json"
        if {row["seed"] for row in banks} == seeds:
            result = _factorial_statistics(config, typed)
            atomic_json(statistics_path, result)
            for field, value in numeric_tree(result["hierarchical_bootstrap"]):
                tables["s1"].append(cell("factorial_intervals", metric=field, value=value, status="COMPLETE", source="factorial/factorial_statistics.json"))
    for control in CONTROLS:
        control_banks, resources = [], []
        for seed in config["seeds"]:
            base = f"paper_controls/{control}/{seed}"
            control_banks.extend(read(base + "/localization_per_bank.csv"))
            resources.extend(read(base + "/resources.csv"))
            if control == "shuffled_full":
                for filename, field, task in (("cgs_per_bank.csv", "cgs_auroc", "alignment"), ("response_per_bank.csv", "unified_response_probe_active_patch_nmse", "response")):
                    records = read(base + "/" + task + "/" + filename)
                    for city in cities:
                        metric("s2", "shuffled_pair", control, city, "", field, [row for row in records if row["city_id"] == city], base, note=f"seed={seed}")
        for city in cities:
            for budget in config["localization"]["primary_budgets"]:
                selected = [row for row in control_banks if row["city_id"] == city and int(row["budget"]) == budget]
                metric("s1", "controlled_localization", control, city, budget, "median_error_m", selected,
                       f"paper_controls/{control}", complete={row["seed"] for row in selected} == seeds)
        for field in ("parameters", "training_flops", "parameter_ratio_to_full", "training_flop_ratio_to_full"):
            metric("s1", "control_cost", control, "", "", field, resources, f"paper_controls/{control}", complete={row["seed"] for row in resources} == seeds)

    shortcut = read("core/evaluation/alignment_shortcut_baselines.csv")
    response = read("core/evaluation/response_per_bank.csv")
    from .paper_core import SHORTCUTS, VARIANTS
    for arm in ARMS:
        for city in cities:
            for name in SHORTCUTS:
                selected = [row for row in shortcut if row["arm"] == arm and row["city_id"] == city and row["baseline"] == name]
                metric("s2", "shortcut:" + name, arm, city, "", "unseen_bank_auroc", selected, "core/evaluation/alignment_shortcut_baselines.csv", complete={row["seed"] for row in selected} == seeds)
            selected = [row for row in response if row["arm"] == arm and row["city_id"] == city]
            for name in ("copy", *VARIANTS):
                metric("s2", "response_control", arm, city, "", "probe_" + name + "_active_patch_nmse", selected, "core/evaluation/response_per_bank.csv", complete={row["seed"] for row in selected} == seeds)
            for field in ("native_no_action_full_channel_nmse", "native_action_swap_full_channel_nmse"):
                metric("s2", "response_control", arm, city, "", field, selected, "core/evaluation/response_per_bank.csv", complete={row["seed"] for row in selected} == seeds)

    map_records, diagnostic_units = {}, defaultdict(set)
    for name in METHODS:
        diagnostics = []
        for seed in config["seeds"]:
            diagnostics.extend(read(f"map_methods/{name}/{seed}/six_condition_results.csv"))
        map_records[name] = diagnostics
        seen = set()
        for row in diagnostics:
            key = (row["seed"], row["city_id"], row["condition"], row["unit_id"])
            if key in seen:
                raise ValueError("Duplicate map diagnostic unit")
            seen.add(key)
            diagnostic_units[row["city_id"]].add((row["seed"], row["unit_id"]))
    for name, diagnostics in map_records.items():
        for city in cities:
            for condition in SIX_CONDITIONS:
                selected = [row for row in diagnostics if row["city_id"] == city and row["condition"] == condition]
                # Percentiles within bank/seed first, then equal foundations.
                groups = defaultdict(list)
                for row in selected:
                    groups[(row["seed"], row["bank_id"])].append(row)
                import numpy as np
                summaries = [{**group[0], "median_error_m": float(np.median([float(r["localization_error_m"]) for r in group])),
                              "p90_error_m": float(np.percentile([float(r["localization_error_m"]) for r in group], 90))} for group in groups.values()]
                for field in ("median_error_m", "p90_error_m"):
                    metric("s3", condition, name, city, "native", field, summaries, f"map_methods/{name}",
                        complete={row["seed"] for row in selected} == seeds and
                        {(row["seed"], row["unit_id"]) for row in selected} == diagnostic_units[city])

    risk = read("paper_risk/metrics.csv")
    for arm in ARMS:
        for city in cities:
            for name in ("joint", "d_only", "u_only"):
                for scope in ("all_queries", "source_support"):
                    selected = [row for row in risk if (row["arm"], row["city_id"], row["model_name"], row["scope"]) == (arm, city, name, scope)]
                    for field in ("aurc", "ece", "brier", "nll", "support_coverage", "risk_error_spearman"):
                        if selected and selected[0]["status"] == "UNDEFINED":
                            tables["s4"].append(cell(scope + ":" + name, arm, city, 0, field, "N/A", "UNDEFINED", source="paper_risk/metrics.csv", note=selected[0]["reason"]))
                        else:
                            metric("s4", scope + ":" + name, arm, city, 0, field, selected, "paper_risk/metrics.csv",
                                   complete=bool(selected) and int(selected[0].get("training_seeds", 0)) == len(seeds))
    output = root / ("fixture_tables" if fixture else "tables")
    coverage = {"fixture": fixture, "scientific_use": "FORBIDDEN" if fixture else "CANDIDATE_NOT_CLAIM",
                "scientific_conclusion": "NOT_INFERRED_FROM_COMPLETION", "dataset_sha256": dataset_hash,
                "aggregation": "equal foundation, bank, seed, draw", "sources": sorted(source_files), "tables": {}}
    for name, values in tables.items():
        write_table(output, name, values)
        if name == "main1":
            display_main1(output, values)
        elif name == "main2":
            for extension in ("md", "tex"):
                path = output / f"main2.{extension}"
                text = path.read_text(encoding="utf-8").replace("Mean bank median (mean bank P90)", "Foundation/bank/seed/draw macro median (P90)")
                text += ("\n" if extension == "md" else "\n% ") + "Wi-GATr and PMNet: native power-inverse features + the shared position head; not native CSI-localization benchmarks.\n"
                path.write_text(text, encoding="utf-8")
        missing = sum(row["status"] in {"MISSING", "PARTIAL"} for row in values)
        coverage["tables"][name] = {"status": "PARTIAL" if missing else "COMPLETE", "incomplete_cells": missing,
                                     "undefined_cells": sum(row["status"] == "UNDEFINED" for row in values)}
    atomic_json(output / "coverage.json", coverage)
    return coverage
