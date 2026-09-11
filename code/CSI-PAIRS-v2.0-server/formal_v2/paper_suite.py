"""Shared table formatting and read-only historical export. New runs use paper_run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
from contextlib import contextmanager
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
DEFAULT_CONFIG = PROJECT / "formal_v2/configs/paper_suite.json"
FIELDS = ["panel", "method", "city", "budget", "metric", "value", "status", "n", "source", "note"]
TITLES = {
    "main1": "Four-arm alignment and response",
    "main2": "Cross-city localization: ablations and external baselines",
    "s1": "Interaction, resource cost and matched controls",
    "s2": "Shortcut and paired-supervision controls",
    "s3": "Six-condition map diagnostics (separate evaluation scope)",
    "s4": "Frozen k=0 risk and calibration",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def load_config(path=DEFAULT_CONFIG):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != "csi-pairs-paper-suite-v1":
        raise ValueError("Unsupported paper suite schema")
    seen = set()
    for stage in value["stages"]:
        if stage["id"] in seen:
            raise ValueError("Historical stage IDs must be unique")
        seen.add(stage["id"])
        if not set(stage["tables"]).issubset(TITLES):
            raise ValueError("Unknown paper table")
        for name in stage["outputs"]:
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Stage artifacts must stay inside the run root")
    return value


def read_rows(path, dataset_sha256):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if str(row.get("fixture", "")).lower() == "true" or row.get("scientific_use") == "FORBIDDEN":
            raise ValueError(f"Synthetic artifacts cannot populate paper tables: {path}")
        if row.get("dataset_sha256") != dataset_sha256:
            raise ValueError(f"Missing or inconsistent dataset identity: {path}")
    return rows


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite table metric")
    return result


def cell(panel, method="", city="", budget="", metric="", value="", status="MISSING", n="", source="", note=""):
    return dict(zip(FIELDS, (panel, method, city, budget, metric, value, status, n, source, note)))


class Sources:
    def __init__(self, root, config, archive=False):
        self.root, self.config, self.archive = Path(root), config, archive
        self.files = {}
        self.core_config = None
        self.evaluation_protocol = None

    def path(self, relative):
        # Archive mode is explicit, mutually exclusive with a live run. Never
        # silently fill a current experiment's holes with historical values.
        if self.archive:
            base = REPO / "artifacts/streaming_gpu_repair_20260908/sota-obstacle-audit-20260909"
            aliases = {
                "factorial/localization_summary.csv": base / "localization_summary.csv",
                "factorial/training_summary.csv": base / "training_summary.csv",
                "factorial/gate.json": base / "gate.json",
            }
            return aliases.get(relative)
        path = self.root / relative
        if relative.startswith("evaluation/"):
            core_path = self.root / "core" / relative
            if path.is_file() and core_path.is_file():
                raise ValueError("Both legacy and new core evaluation outputs exist; choose a single run")
            if core_path.is_file():
                return core_path
        return path

    def rows(self, relative, *, core=True):
        path = self.path(relative)
        if path is None or not path.is_file():
            return []
        rows = read_rows(path, self.config["dataset_sha256"])
        if relative.startswith("evaluation/") and rows:
            protocols = {r.get("evaluation_protocol_sha256") or "legacy-frozen-v6" for r in rows}
            if len(protocols) != 1:
                raise ValueError("Cannot mix probe evaluation protocols")
            protocol = next(iter(protocols))
            if self.evaluation_protocol is not None and self.evaluation_protocol != protocol:
                raise ValueError("Cannot mix probe evaluation protocols")
            self.evaluation_protocol = protocol
        if core and rows:
            identities = {r.get("config_sha256") for r in rows}
            if len(identities) != 1 or not next(iter(identities)):
                raise ValueError(f"Missing/mixed core configuration: {relative}")
            identity = next(iter(identities))
            if self.core_config is not None and self.core_config != identity:
                raise ValueError("Cannot mix core experiment configurations in paper tables")
            self.core_config = identity
        self.files[relative] = {"path": str(path.resolve()), "sha256": digest(path), "rows": len(rows)}
        return rows


def localization_table(sources):
    config = sources.config
    rows = sources.rows("factorial/localization_summary.csv")
    external = sources.rows("paper_baselines/localization_summary.csv", core=False)
    protocol_path = sources.path("paper_baselines/comparison_protocol.json")
    reference_path = sources.path("paper_protocol.json")
    if external:
        if not protocol_path or not protocol_path.is_file() or not reference_path or not reference_path.is_file():
            raise ValueError("Baseline main-table rows require a shared-query comparison protocol")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        for key in ("dataset_sha256", "query_registry_sha256", "support_registry_sha256", "metric_aggregation", "budgets", "seeds", "label_draws"):
            if key not in reference or protocol.get(key) != reference[key]:
                raise ValueError(f"Baseline comparison protocol differs: {key}")
        for key, expected in (("dataset_sha256", config["dataset_sha256"]), ("budgets", config["budgets"]), ("seeds", config["seeds"]), ("label_draws", config["label_draws"]), ("metric_aggregation", "mean_bank_median_and_p90")):
            if reference[key] != expected:
                raise ValueError(f"Main table protocol differs from suite: {key}")
        for key in ("query_registry_sha256", "support_registry_sha256"):
            value = reference[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Invalid comparison registry hash: {key}")
    index = {}
    for row in rows + external:
        method = row.get("arm") or row.get("model_name")
        key = (method, row["city_id"], int(row["budget"]))
        if key in index:
            raise ValueError(f"Duplicate localization cell: {key}")
        index[key] = row
    output = []
    for method in config["arms"] + config["baselines"]:
        for city in config["cities"]:
            for budget in config["budgets"]:
                row = index.get((method, city, budget))
                source = "factorial/localization_summary.csv" if method in config["arms"] else "paper_baselines/localization_summary.csv"
                complete = row is not None and (
                    int(row.get("training_seeds", 0)) == len(config["seeds"])
                    and int(row.get("label_draws", 0)) == (1 if budget == 0 else config["label_draws"])
                    and int(row.get("independent_base_map_clusters", 0)) >= config["minimum_clusters_per_city"]
                )
                for field in ("mean_bank_median_error_m", "mean_bank_p90_error_m"):
                    present = row is not None and row.get(field) not in (None, "")
                    output.append(cell("localization", method, city, budget, field,
                        number(row[field]) if present else "", "COMPLETE" if complete and present else "PARTIAL" if present else "MISSING",
                        row.get("independent_base_map_clusters", "") if row else "", source,
                        "" if complete else "Full seed/draw/cluster coverage required; six-condition diagnostic errors are not interchangeable."))
    return output


def metric_panel(sources, path, fields, panel, *, where=None, expected_arms=True):
    rows = sources.rows(path)
    rows = [row for row in rows if row.get("city_id") in sources.config["cities"]]
    if where:
        rows = [row for row in rows if where(row)]
    output = []
    if not rows:
        methods = sources.config["arms"] if expected_arms else [""]
        return [cell(panel, method=method, city=city, metric=field, source=path) for method in methods for city in sources.config["cities"] for field in fields]
    groups = {}
    for row in rows:
        key = (row.get("arm", row.get("model_name", "")), row.get("city_id", ""), row.get("baseline", row.get("condition", "")))
        groups.setdefault(key, []).append(row)
    for (method, city, condition), group in sorted(groups.items()):
        for field in fields:
            valid = [r for r in group if r.get(field) not in (None, "")]
            # Balance banks within each seed, then seeds. Preserve missingness.
            by_seed = {}
            for row in valid:
                by_seed.setdefault(str(row.get("seed", "")), {}).setdefault(str(row.get("bank_id", "")), []).append(number(row[field]))
            seed_means = [statistics.mean(statistics.mean(v) for v in banks.values()) for banks in by_seed.values()]
            seed_complete = set(by_seed) == {str(s) for s in sources.config["seeds"]}
            bank_complete = all(len(banks) >= sources.config["minimum_clusters_per_city"] for banks in by_seed.values())
            status = "COMPLETE" if len(valid) == len(group) and seed_complete and bank_complete else "PARTIAL" if valid else "MISSING"
            output.append(cell(panel + (":" + condition if condition else ""), method, city, 0 if panel == "risk" else "", field,
                statistics.mean(seed_means) if seed_means else "", status, len(valid), path,
                "Equal bank then seed means; table is incomplete if expected seeds are absent."))
    if expected_arms:
        present = {(r["method"], r["city"], r["metric"]) for r in output}
        for method in sources.config["arms"]:
            for city in sources.config["cities"]:
                for field in fields:
                    if (method, city, field) not in present:
                        output.append(cell(panel, method, city, metric=field, source=path))
    return output


def raw_metrics(sources, path, fields, panel, *, core=True):
    rows = sources.rows(path, core=core)
    if panel == "risk" and len({(r.get("arm"), r.get("risk_model")) for r in rows}) != len(rows):
        raise ValueError("Duplicate aggregate risk record")
    if not rows:
        return [cell(panel, metric=field, source=path) for field in fields]
    output = []
    for row in rows:
        for field in fields:
            present = row.get(field) not in (None, "")
            output.append(cell(panel + (":" + row["risk_model"] if row.get("risk_model") else ""), row.get("arm", row.get("model_name", row.get("control_id", ""))), row.get("city_id", ""), row.get("budget", 0 if panel == "risk" else ""), field,
                number(row[field]) if present else "", "COMPLETE" if present else "MISSING", row.get("seed", ""), path,
                "Per-record value; n column identifies training seed when supplied."))
    return output


def numeric_tree(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from numeric_tree(child, prefix + ("." if prefix else "") + key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from numeric_tree(child, prefix + f"[{index}]")
    elif type(value) in (float, int):
        yield prefix, number(value)


def json_panel(sources, path, panel, keys):
    source = sources.path(path)
    if source is None or not source.is_file():
        return [cell(panel, metric=key, source=path) for key in keys]
    record = json.loads(source.read_text(encoding="utf-8"))
    if record.get("dataset_sha256") != sources.config["dataset_sha256"] or record.get("fixture") is True:
        raise ValueError("Invalid JSON panel dataset: " + path)
    if sources.core_config is not None and record.get("config_sha256") != sources.core_config:
        raise ValueError("JSON panel config mismatch: " + path)
    sources.files[path] = {"path": str(source), "sha256": digest(source)}
    output = []
    for key in keys:
        values = list(numeric_tree(record.get(key), key))
        if not values:
            output.append(cell(panel, metric=key, source=path))
        for metric, value in values:
            output.append(cell(panel, metric=metric, value=value, status="COMPLETE", source=path,
                               note="Original statistical record; hypothesis status=" + str(record.get("status"))))
    return output


def latex_escape(value):
    return "".join({"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}.get(c, c) for c in str(value))


def write_table(output, key, rows):
    output.mkdir(parents=True, exist_ok=True)
    with (output / (key + ".csv")).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if key == "main2":
        write_localization_display(output, rows)
        return
    display = ["panel", "method", "city", "budget", "metric", "value", "status"]
    def fmt(value):
        return f"{value:.6g}" if isinstance(value, float) else str(value)
    lines = ["# " + TITLES[key], "", "Missing values are unrun/unavailable, never zero. See CSV and coverage.json for provenance.", "", "| " + " | ".join(display) + " |", "| " + " | ".join("---" for _ in display) + " |"]
    lines += ["| " + " | ".join(fmt(row[k]).replace("|", "\\|").replace("\n", " ") for k in display) + " |" for row in rows]
    (output / (key + ".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Longtable can span pages in the appendix; the CSV remains the full source.
    tex = [r"% Requires booktabs and longtable. Missing entries are not results.", r"\begin{longtable}{lllllll}", r"\caption{" + latex_escape(TITLES[key]) + r"}\\", r"\toprule", " & ".join(display) + r" \\", r"\midrule\endhead"]
    tex += [" & ".join(latex_escape(fmt(row[k])) for k in display) + r" \\" for row in rows]
    tex += [r"\bottomrule", r"\end{longtable}"]
    (output / (key + ".tex")).write_text("\n".join(tex) + "\n", encoding="utf-8")


def write_localization_display(output, rows):
    methods = list(dict.fromkeys(row["method"] for row in rows))
    columns = list(dict.fromkeys((row["city"], row["budget"]) for row in rows))
    index = {(r["method"], r["city"], r["budget"], r["metric"]): r for r in rows}
    body = []
    for method in methods:
        cells = [method]
        for city, budget in columns:
            pair = [index[(method, city, budget, field)] for field in ("mean_bank_median_error_m", "mean_bank_p90_error_m")]
            values = [f"{r['value']:.3f}" + ("*" if r["status"] == "PARTIAL" else "") if r["value"] != "" else "--" for r in pair]
            cells.append(values[0] + " (" + values[1] + ")")
        body.append(cells)
    header = ["Method"] + [f"{city.removeprefix('target-')} k={budget}" for city, budget in columns]
    note = "Mean bank median (mean bank P90), metres; lower is better. -- = missing, * = incomplete coverage. Historical values are not rerun results."
    lines = ["# " + TITLES["main2"], "", note, "", "| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    (output / "main2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex = ["% " + note, r"\begin{tabular}{l" + "r" * len(columns) + "}", r"\toprule", " & ".join(latex_escape(v) for v in header) + r" \\", r"\midrule"]
    tex += [" & ".join(latex_escape(v) for v in row) + r" \\" for row in body]
    tex += [r"\bottomrule", r"\end{tabular}"]
    (output / "main2.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def expected_raw_coverage(sources, rows, table):
    if table == "s4":
        expected = {(arm, "risk:" + risk) for arm in sources.config["arms"] for risk in ("joint", "d_only", "u_only")}
        actual = {(r["method"], r["panel"]) for r in rows}
        return [cell(panel, method=arm, metric="coverage", note="Expected risk model is absent.") for arm, panel in sorted(expected - actual)]
    return []


def map_condition_panel(sources):
    path = "external_baselines/six_condition_results.csv"
    rows = sources.rows(path, core=False)
    conditions = ("correct", "paired_active_alternative", "paired_null_alternative", "wrong_city", "geometry_destroyed", "empty")
    grouped, units, seen = {}, {}, set()
    for row in rows:
        key = (row["model_name"], row["city_id"], row["condition"])
        identity = (*key, row["unit_id"])
        if identity in seen:
            raise ValueError("Duplicate six-condition diagnostic unit")
        if row["condition"] not in conditions:
            raise ValueError("Unknown map diagnostic condition")
        seen.add(identity)
        units.setdefault(key, set()).add(row["unit_id"])
        cluster = row.get("canonical_base_map_digest") or row.get("base_map_cluster_id")
        if not cluster or not row.get("bank_id"):
            raise ValueError("Map diagnostics require bank and cluster identities")
        grouped.setdefault(key, {}).setdefault(cluster, {}).setdefault(row["bank_id"], []).append(number(row["localization_error_m"]))
    def percentile(values, fraction):
        values = sorted(values)
        index = (len(values) - 1) * fraction
        lower, upper = math.floor(index), math.ceil(index)
        return values[lower] + (values[upper] - values[lower]) * (index - lower)
    result = []
    for method in sources.config.get("map_models", sources.config["baselines"]):
        for city in sources.config["cities"]:
            sets = [units.get((method, city, condition), set()) for condition in conditions]
            paired = bool(sets[0]) and all(value == sets[0] for value in sets)
            for condition in conditions:
                groups = grouped.get((method, city, condition), {})
                for field, fraction in (("mean_cluster_bank_median_error_m", .5), ("mean_cluster_bank_p90_error_m", .9)):
                    estimates = [statistics.mean(percentile(errors, fraction) for errors in banks.values()) for banks in groups.values()]
                    complete = paired and len(groups) >= sources.config["minimum_clusters_per_city"]
                    result.append(cell("six_conditions:" + condition, method, city, "diagnostic", field,
                        statistics.mean(estimates) if estimates else "", "COMPLETE" if complete else "PARTIAL" if estimates else "MISSING",
                        len(groups), path, "Equal bank then base-map-cluster mean; native adapter run, separate from few-shot main table."))
    return result


def export_tables(config, run_root, output, *, archive=False):
    sources = Sources(run_root, config, archive)
    tables = {"main2": localization_table(sources)}
    tables["main1"] = metric_panel(sources, "evaluation/cgs_per_bank.csv", ["cgs_auroc"], "alignment")
    tables["main1"] += metric_panel(sources, "evaluation/compatibility_route_distributions.csv", ["overclassification_rate"], "null", where=lambda r: r.get("route") == "null")
    tables["main1"] += metric_panel(sources, "evaluation/response_per_bank.csv", ["unified_response_probe_active_patch_nmse", "native_target_free_full_channel_nmse", "native_null_violation_rate"], "response")
    tables["s1"] = raw_metrics(sources, "factorial/training_summary.csv", ["parameters", "measured_flops_per_step", "elapsed_seconds"], "training_cost")
    tables["s1"] += raw_metrics(sources, "controls/resource_summary.csv", ["parameters", "training_flops", "inference_flops", "wall_seconds"], "matched_resource")
    tables["s1"] += raw_metrics(sources, "controls/localization_per_bank.csv", ["median_error_m", "p90_error_m"], "matched_localization")
    gate_path = sources.path("factorial/gate.json")
    if gate_path and gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("dataset_sha256") != config["dataset_sha256"] or gate.get("config_sha256") != sources.core_config:
            raise ValueError("Factorial gate identity differs from exported tables")
        sources.files["factorial/gate.json"] = {"path": str(gate_path), "sha256": digest(gate_path)}
        for row in gate.get("g5_cell_report", {}).get("cells", []):
            for field in ("difference", "ci95_low", "ci95_high", "holm_adjusted_p"):
                tables["s1"].append(cell("full_minus_endpoint", "full", row["city_id"], row["budget"], field, number(row[field]), "COMPLETE", row["cluster_count"], "factorial/gate.json", "Hypothesis pass=" + str(row["passed"])))
    else:
        tables["s1"].append(cell("factorial_statistics", source="factorial/gate.json"))
    # Compute only descriptive contrasts from balanced summary cells. Do not
    # invent bootstrap intervals when the original statistics are unavailable.
    summary = sources.rows("factorial/localization_summary.csv")
    macro = {}
    for method in config["arms"]:
        selected = [r for r in summary if r["arm"] == method and r["city_id"] in config["cities"] and int(r["budget"]) in (0, 8)]
        if len(selected) == len(config["cities"]) * 2:
            macro[method] = statistics.mean(number(r["mean_utility_neg_log_median"]) for r in selected)
    if len(macro) == 4:
        for name, value in (("full_minus_alignment", macro["full"] - macro["alignment"]), ("full_minus_response", macro["full"] - macro["response"]), ("interaction", macro["full"] - macro["alignment"] - macro["response"] + macro["endpoint"])):
            tables["s1"].append(cell("factorial_point_estimates", "full", metric=name, value=value, status="COMPLETE", source="factorial/localization_summary.csv", note="Descriptive utility contrast at k=0/8; not a significance decision."))
    tables["s1"] += json_panel(sources, "factorial/factorial_statistics.json", "factorial_intervals", ["hierarchical_bootstrap"])
    tables["s2"] = metric_panel(sources, "evaluation/alignment_shortcut_baselines.csv", ["unseen_bank_auroc"], "alignment_shortcuts", expected_arms=False)
    shortcut_cells = {(r["method"], r["city"], r["panel"]) for r in tables["s2"]}
    for method in config["arms"]:
        for city in config["cities"]:
            for shortcut in ("constant", "csi_only", "map_only", "scene_id_only", "edit_status_xor", "variant_id_matcher"):
                panel = "alignment_shortcuts:" + shortcut
                if (method, city, panel) not in shortcut_cells:
                    tables["s2"].append(cell(panel, method, city, metric="unseen_bank_auroc", source="evaluation/alignment_shortcut_baselines.csv"))
    tables["s2"] += metric_panel(sources, "evaluation/response_per_bank.csv", ["probe_copy_active_patch_nmse", "probe_without_map_active_patch_nmse", "probe_edit_only_active_patch_nmse", "probe_csi_only_active_patch_nmse", "probe_oracle_x_active_patch_nmse", "native_no_action_full_channel_nmse", "native_action_swap_full_channel_nmse"], "response_controls")
    tables["s2"] += json_panel(sources, "controls/shuffled_pair/gate.json", "shuffled_pair", ["alignment_gain", "shuffled_alignment_gain", "shuffled_suppression_interval", "response_gain", "shuffled_response_gain", "shuffled_response_suppression_interval"])
    tables["s3"] = map_condition_panel(sources)
    tables["s4"] = raw_metrics(sources, "risk/metrics.csv", ["aurc", "ece", "brier", "nll", "support_coverage", "risk_error_spearman"], "risk")
    tables["s4"] += expected_raw_coverage(sources, tables["s4"], "s4")
    for key, stages in (("s1", ["resources"]), ("s2", ["evaluation", "shuffled_pair"]), ("s3", ["map_baselines"]), ("s4", ["risk"])):
        # A few present rows do not establish complete experimental coverage.
        # The stage completion receipt is independent of hypothesis success.
        for stage_id in stages:
            stage = next(s for s in config["stages"] if s["id"] == stage_id)
            complete = not archive and artifact_records(sources.root, stage) and stage_gate(sources.root, stage, config)
            if not complete:
                tables[key].append(cell("coverage", metric=stage_id, source=", ".join(stage["outputs"]), note="A complete stage receipt is absent; some rows alone cannot establish coverage."))
    coverage = {"schema_version": "csi-pairs-paper-table-coverage-v1", "protocol_id": config["protocol_id"], "archive_only": archive, "scientific_conclusion": "NOT_INFERRED_FROM_COMPLETION", "tables": {}, "sources": sources.files}
    for key in TITLES:
        rows = tables[key]
        write_table(Path(output), key, rows)
        counts = {status: sum(r["status"] == status for r in rows) for status in ("COMPLETE", "PARTIAL", "MISSING")}
        coverage["tables"][key] = {"title": TITLES[key], **counts, "status": "COMPLETE" if counts["MISSING"] == counts["PARTIAL"] == 0 else "PARTIAL" if counts["COMPLETE"] or counts["PARTIAL"] else "MISSING"}
    atomic_json(Path(output) / "coverage.json", coverage)
    return coverage




def artifact_records(root, stage):
    result = {}
    for relative in stage["outputs"]:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            return None
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        else:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                if not list(csv.DictReader(stream)):
                    return None
        result[relative] = digest(path)
    return result


def stage_gate(root, stage, config):
    parents = {Path(name).parts[0] for name in stage["outputs"]}
    relative = "controls/shuffled_pair/gate.json" if stage["id"] == "shuffled_pair" else next(iter(parents)) + "/gate.json" if len(parents) == 1 else ""
    path = root / relative
    if not relative or not path.is_file():
        return None
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("dataset_sha256") != config["dataset_sha256"] or gate.get("fixture") is True or gate.get("scientific_use") == "FORBIDDEN":
        return None
    status, passed = gate.get("status"), gate.get("passed")
    valid = (status == "PASS" and passed is True) or (status == "FAIL" and passed is False) or (status == "BLOCKED" and passed is False and gate.get("engineering_complete") is True)
    if not valid:
        return None
    for filename in stage["outputs"]:
        if filename.endswith(".csv"):
            read_rows(root / filename, config["dataset_sha256"])
    return {"path": relative, "sha256": digest(path), "hypothesis_status": status}




@contextmanager
def suite_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "suite.lock").open("a+") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write("0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)






def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only export of historical results. New experiments: python -m formal_v2.paper_run")
    parser.add_argument("command", choices=("export",))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args(argv)
    if args.archive == bool(args.run_root):
        parser.error("Choose --archive or --run-root for an existing legacy result")
    result = export_tables(load_config(args.config), args.run_root or REPO, args.output, archive=args.archive)
    print(json.dumps(result["tables"], ensure_ascii=False, indent=2))
    return 0 if all(row["status"] == "COMPLETE" for row in result["tables"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
