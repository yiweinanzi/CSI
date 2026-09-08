"""Bounded, source-only method research. Never a formal factorial replacement."""

import argparse
import copy
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import formal_factorial as factorial
from .formal_config import ARMS, load_formal_config, validate_formal_config
from .formal_dataset import FormalDataset, PHYSICAL_POSITION_ATOL_M
from .formal_evaluation_resume import write_atomic_json
from .formal_evidence import config_sha256, configure_reproducible_runtime, evidence_context
from .formal_io import read_strict_json, sha256_file, write_csv
from .formal_localization import (
    HeteroscedasticPositionHead, _head_train_kwargs, _optimize_head,
    adapt_position_head, predict_position_distribution,
)
from .formal_model import portable_state_dict
from .formal_routing import fit_route_normalization
from .formal_source_guard import SOURCE_RESEARCH_ROLES, SourceOnlyDataset
from .formal_teacher import load_teacher_bundle
from .formal_training_resume import _atomic_torch_save


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def state_hash(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        result.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode())
        result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def write_once(path, value):
    if path.exists():
        if read_strict_json(path) != value:
            raise RuntimeError("Research resume identity changed: " + str(path))
    else:
        write_atomic_json(path, value)


def preserve_initial_state(path, state):
    expected = state_hash(state)
    if path.exists():
        if state_hash(torch.load(path, map_location="cpu", weights_only=True)) != expected:
            raise RuntimeError("Actual paired initialization changed: " + str(path))
    else:
        _atomic_torch_save(path, state)
    return expected


def load_completed_arm(root, arm, context):
    path = root / (arm + "_result.json")
    if not path.exists():
        return None
    record = read_strict_json(path)
    content = {key: value for key, value in record.items() if key != "record_sha256"}
    if record.get("record_sha256") != json_hash(content):
        raise RuntimeError("Completed research receipt changed")
    if record.get("context_sha256") != json_hash(context) or record.get("arm") != arm:
        raise RuntimeError("Completed research arm has a different identity")
    if record.get("scientific_use") != "NON_CLAIM" or record.get("formal_units_added") != 0:
        raise RuntimeError("Completed research arm has an invalid claim ceiling")
    if record["training"]["steps"] != context["steps"] or [row["step"] for row in record["loss_trace"]] != list(range(1, context["steps"] + 1)):
        raise RuntimeError("Completed arm does not contain the full declared training schedule")
    for suffix, key in ((".pt", "state_sha256"), ("_source_validation.csv", "validation_csv_sha256")):
        if sha256_file(root / (arm + suffix)) != record[key]:
            raise RuntimeError("Completed research artifact changed: " + arm + suffix)
    state = torch.load(root / (arm + ".pt"), map_location="cpu", weights_only=True)
    if state_hash(state) != record["state_value_sha256"]:
        raise RuntimeError("Completed research state values changed")
    if not record["source_rows"] or any(not np.isfinite(row["median_error_m"]) for row in record["source_rows"]):
        raise RuntimeError("Completed source validation rows are invalid")
    return record


def recipe_configs(base, protocol):
    validate_formal_config(base)
    if protocol["schema_version"] != "csi-pairs-source-method-pilot-v1" or protocol["scientific_use"] != "NON_CLAIM" or protocol["sota_ready"] is not False:
        raise ValueError("This entry supports NON_CLAIM source research only")
    if config_sha256(base) != protocol["base_config_sha256"] or protocol["arms"] != list(ARMS):
        raise ValueError("Base config or four-arm contract changed")
    if protocol["allowed_roles"] != list(SOURCE_RESEARCH_ROLES):
        raise ValueError("Research cannot use other roles")
    if (protocol["training_role"], protocol["selection_role"]) != SOURCE_RESEARCH_ROLES:
        raise ValueError("Research training and selection roles cannot change")
    if protocol["seed"] not in base["seeds"] or protocol["steps_per_arm"] != 128:
        raise ValueError("This bounded protocol requires its fixed seed and 128 steps")
    if protocol["validation"]["label_budgets"] != base["localization"]["label_budgets"] or protocol["validation"]["draws"] != base["localization"]["label_draws"]:
        raise ValueError("Validation budgets and draw counts must retain the original grid")
    expected = [{"name": "fixed_margin", "alignment_primary_margin": "fixed"}, {"name": "effect_aware_margin", "alignment_primary_margin": "effect_aware"}]
    if protocol["recipes"] != expected:
        raise ValueError("Candidate set must match the preregistered two recipes")
    configs = {}
    for row in protocol["recipes"]:
        config = copy.deepcopy(base)
        # This declared research override is not a V6 P0 formal configuration.
        # The formal validator remains unchanged and still rejects effect_aware.
        config["factorial"]["alignment_primary_margin"] = row["alignment_primary_margin"]
        configs[row["name"]] = config
    return configs


def source_validation_split(dataset, protocol):
    scenes = [int(value) for value in dataset.indices_for_role("source_method_selection")]
    cities = sorted({str(dataset.city_ids[scene]) for scene in scenes})
    output = {}
    maximum_budget = max(protocol["validation"]["label_budgets"])
    for city in cities:
        city_scenes = [scene for scene in scenes if str(dataset.city_ids[scene]) == city]
        rows = sorted(((str(dataset.position_ids[scene, position]), scene, position) for scene in city_scenes for position in range(dataset.position_count)))
        unique, ids, coordinates, id_coordinates = [], set(), [], {}
        for identifier, scene, position in rows:
            coordinate = np.asarray(dataset.positions[scene, position], dtype=np.float64)
            if not np.isfinite(coordinate).all():
                raise RuntimeError("Source position is nonfinite")
            if identifier in id_coordinates and not np.all(np.abs(id_coordinates[identifier] - coordinate) <= PHYSICAL_POSITION_ATOL_M):
                raise RuntimeError("Source physical position ID has inconsistent coordinates")
            id_coordinates[identifier] = coordinate
            if identifier in ids or (coordinates and np.any(np.all(np.abs(np.asarray(coordinates) - coordinate) <= PHYSICAL_POSITION_ATOL_M, axis=1))):
                continue
            unique.append((scene, position))
            ids.add(identifier)
            coordinates.append(coordinate)
        if len(unique) // 2 < maximum_budget:
            raise RuntimeError("Source city lacks disjoint support and validation capacity")
        rng = np.random.default_rng(protocol["validation"]["partition_seed"] + factorial._stable_city_seed(city))
        order = rng.permutation(len(unique)).tolist()
        support_pool = [unique[index] for index in order[:len(unique) // 2]]
        pool_coordinates = np.asarray([dataset.positions[scene, position] for scene, position in support_pool])
        pool_ids = {str(dataset.position_ids[scene, position]) for scene, position in support_pool}
        validation = {}
        for scene in city_scenes:
            positions = np.asarray(dataset.positions[scene])
            sibling = np.any(np.all(np.abs(positions[:, None, :] - pool_coordinates[None, :, :]) <= PHYSICAL_POSITION_ATOL_M, axis=2), axis=1)
            selected = [position for position in range(dataset.position_count) if not sibling[position] and str(dataset.position_ids[scene, position]) not in pool_ids]
            if not selected:
                raise RuntimeError("Source support isolation leaves an empty validation bank")
            validation[str(scene)] = selected
        output[city] = {"support_pool": [list(row) for row in support_pool], "validation_rows": validation}
    return output


class PilotProgress:
    def __init__(self, output, protocol):
        self.output = output
        self.protocol = protocol
        self.started = time.monotonic()
        self.completed = 0

    def report(self, stage, **details):
        row = {
            "utc": datetime.now(timezone.utc).isoformat(), "stage": stage,
            "elapsed_seconds": time.monotonic() - self.started,
            "completed_arm_jobs": self.completed, "total_arm_jobs": 8,
            "device": self.protocol["device"], "scientific_use": "NON_CLAIM",
            "formal_units_added": 0, "sota_ready": False, **details,
        }
        with (self.output / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        write_atomic_json(self.output / "progress.json", row)
        print(json.dumps(row), flush=True)


class ResourceGuard:
    def __init__(self, protocol, output):
        self.protocol, self.output = protocol, output
        self.reason = None

    def memory_snapshot(self):
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        available = int(memory["MemAvailable"].split()[0]) * 1024
        snapshot = {"host_available_bytes": available, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        # The mounted v1 controller is namespace-relative on this host; also support v2.
        for limit_path, current_path in (
            ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
            ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ):
            if Path(limit_path).is_file() and Path(current_path).is_file():
                limit = Path(limit_path).read_text().strip()
                current = int(Path(current_path).read_text())
                if limit != "max" and int(limit) < 2**60:
                    snapshot["cgroup_available_bytes"] = max(0, int(limit) - current)
        return snapshot

    def is_set(self):
        snapshot = self.memory_snapshot()
        available = min(snapshot["host_available_bytes"], snapshot.get("cgroup_available_bytes", snapshot["host_available_bytes"]))
        if (self.output / "PAUSE_REQUESTED").exists():
            self.reason = "Operator requested a checkpoint-boundary pause"
        elif available < self.protocol["host_available_floor_gib"] * 1024**3:
            self.reason = "Host available memory fell below the declared floor"
        elif snapshot["peak_rss_bytes"] > self.protocol["host_rss_limit_gib"] * 1024**3:
            self.reason = "Research process exceeded its declared RSS envelope"
        return self.reason is not None


class ProgressTrace(list):
    def __init__(self, progress, recipe, arm, model):
        super().__init__()
        self.progress, self.recipe, self.arm, self.model = progress, recipe, arm, model

    def append(self, row):
        super().append(row)
        if row["step"] == 1 or row["step"] % 16 == 0:
            device = torch.device(self.progress.protocol["device"])
            gradients = [value.grad for value in self.model.parameters() if value.grad is not None]
            if not gradients or any(value.device != device for value in gradients):
                raise RuntimeError("Main-model pilot gradients are not on the assigned CUDA device")
            self.progress.report("backward_complete_before_optimizer_step", recipe=self.recipe, arm=self.arm, cuda_gradients_verified=True, **row)


def calibrate_scales(base, corpus, selection, protocol, output, progress, guard, identity):
    path = output / "calibration.json"
    state_path = output / "shared_calibration_model.pt"
    if path.exists():
        record = read_strict_json(path)
        if record["record_sha256"] != json_hash({key: value for key, value in record.items() if key != "record_sha256"}):
            raise RuntimeError("Completed calibration receipt changed")
        if record["identity_sha256"] != json_hash(identity) or sha256_file(state_path) != record["model_sha256"]:
            raise RuntimeError("Completed calibration identity or model changed")
        if state_hash(torch.load(state_path, map_location="cpu", weights_only=True)) != record["model_value_sha256"]:
            raise RuntimeError("Completed calibration model values changed")
        progress.report("reused_authenticated_calibration")
        return record
    progress.report("starting_calibration", resume_policy="Incomplete calibration restarts from its original initialization; arm checkpoints remain resumable")
    seed = int(protocol["seed"]) + 6001
    model = factorial._new_model(base, corpus, seed, device=protocol["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(base["model"]["learning_rate"]), weight_decay=float(base["model"]["weight_decay"]))
    for step in range(int(base["factorial"]["pilot_steps"])):
        if guard.is_set():
            raise RuntimeError(guard.reason)
        plan = factorial._make_plan(corpus, int(base["factorial"]["batch_size"]), seed, step)
        endpoint = factorial._identity_batch(corpus, plan.endpoint, [corpus.teacher.mask_bank[index] for index in plan.endpoint_masks])
        natural = factorial._identity_batch(corpus, plan.natural_endpoint, [corpus.teacher.mask_bank[index] for index in plan.natural_masks])
        physical = float(base["factorial"]["endpoint_physical_weight"])
        loss = factorial._endpoint_loss(model, endpoint, physical) + float(base["factorial"]["natural_endpoint_weight"]) * factorial._endpoint_loss(model, natural, physical)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite source-only calibration loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 16 == 0:
            if str(protocol["device"]).startswith("cuda"):
                torch.cuda.synchronize(protocol["device"])
            progress.report("shared_endpoint_calibration", step=step + 1, total_steps=int(base["factorial"]["pilot_steps"]))
    model.eval()
    identical_noop = all(np.array_equal(selection.dataset.maps[scene], selection.dataset.noop_maps[scene]) for scene in selection.scenes)
    kappa = 0.0 if identical_noop else factorial._estimate_noop_kappa(model, selection, float(base["qualification"]["alignment_noop_quantile"]))
    weights = factorial._loss_weights(base, 1.0, 1.0, 1.0, 1.0, kappa)
    values = {"fixed_margin": [], "effect_aware_margin": [], "response": []}
    with torch.no_grad():
        for step in range(int(base["factorial"]["pilot_steps"])):
            if guard.is_set():
                raise RuntimeError(guard.reason)
            plan = factorial._make_plan(selection, int(base["factorial"]["batch_size"]), seed + 1, step)
            components = factorial._loss_components(model, selection, plan, weights)
            null = float(weights.alignment_null) * float(components["alignment_null"])
            values["fixed_margin"].append(float(components["alignment_active_fixed"]) + null)
            values["effect_aware_margin"].append(float(components["alignment_active_effect_aware"]) + null)
            values["response"].append(float(components["response"]))
            if (step + 1) % 16 == 0:
                progress.report("shared_source_scale_calibration", step=step + 1, total_steps=int(base["factorial"]["pilot_steps"]))
    calibration_state = portable_state_dict(model)
    _atomic_torch_save(state_path, calibration_state)
    record = {
        "identity_sha256": json_hash(identity), "model_sha256": sha256_file(state_path),
        "model_value_sha256": state_hash(calibration_state), "pilot_seed": seed,
        "pilot_steps": int(base["factorial"]["pilot_steps"]), "alignment_null_tolerance": kappa,
        "noop_policy": "Exact identical normalized inputs imply zero deterministic score gap" if identical_noop else "Original full no-op scoring",
        "response_scale": max(float(np.mean(values["response"])), 1e-6),
        "alignment_scales": {name: max(float(np.mean(values[name])), 1e-6) for name in ("fixed_margin", "effect_aware_margin")},
        "selection_values": values, "checkpoint_reused_for_arm_training": False,
    }
    record["record_sha256"] = json_hash(record)
    write_atomic_json(path, record)
    del optimizer, model
    return record


def validate_source_model(model, dataset, normalization, patch_spec, config, seed, split, head_template, *, progress=None, guard=None, recipe=None, arm=None):
    train_scenes = dataset.indices_for_role("source_encoder_train").tolist()
    selection_scenes = dataset.indices_for_role("source_method_selection").tolist()
    representations = factorial._natural_representations(model, dataset, train_scenes + selection_scenes, normalization, patch_spec)
    train_x = np.vstack([representations[scene] for scene in train_scenes])
    train_y = np.vstack([dataset.positions[scene] for scene in train_scenes])
    head = copy.deepcopy(head_template)
    _optimize_head(head, train_x.astype(np.float32), train_y.astype(np.float32), **_head_train_kwargs(config, len(train_x)))
    rows = []
    for city, partition in sorted(split.items()):
        pool = partition["support_pool"]
        for draw in range(int(config["localization"]["label_draws"])):
            rng = np.random.default_rng(seed * 100003 + factorial._stable_city_seed(city) + draw)
            order = [pool[index] for index in rng.permutation(len(pool))]
            for budget in config["localization"]["label_budgets"]:
                if guard is not None and guard.is_set():
                    raise RuntimeError(guard.reason)
                if budget == 0 and draw > 0:
                    continue
                support = order[:budget]
                x = np.vstack([representations[s][p] for s, p in support]) if support else np.empty((0, train_x.shape[1]))
                y = np.vstack([dataset.positions[s, p] for s, p in support]) if support else np.empty((0, 2))
                adapted = adapt_position_head(head, x, y, config)
                for key, positions in partition["validation_rows"].items():
                    scene = int(key)
                    prediction, _, _ = predict_position_distribution(adapted, representations[scene][positions], sigma_min=float(config["localization"]["sigma_min"]))
                    errors = np.linalg.norm(prediction - dataset.positions[scene, positions], axis=1)
                    if not np.isfinite(errors).all():
                        raise RuntimeError("Source validation produced nonfinite meters")
                    rows.append({"city": city, "scene": scene, "bank": str(dataset.bank_ids[scene]), "k": budget, "draw": draw, "validation_rows": len(positions), "median_error_m": float(np.median(errors)), "p90_error_m": float(np.percentile(errors, 90)), "scientific_use": "NON_CLAIM"})
            if progress is not None:
                progress.report("source_validation_draw_complete", recipe=recipe, arm=arm, city=city, draw=draw, validation_metric_rows=len(rows))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "config", "protocol", "output", "origin-server", "upstream-root"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    configure_reproducible_runtime()
    base = load_formal_config(args.config)
    protocol = read_strict_json(args.protocol)
    configs = recipe_configs(base, protocol)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1" or torch.cuda.device_count() != 2 or protocol["device"] != "cuda:1":
        raise RuntimeError("This pilot requires visible devices 0,1 and dedicated logical cuda:1")
    torch.cuda.set_device(protocol["device"])
    if torch.cuda.mem_get_info(protocol["device"])[0] < 30 * 1024**3:
        raise RuntimeError("GPU 1 lacks the declared 30 GiB free-memory admission floor")
    server = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=server, text=True).strip():
        raise RuntimeError("Commit the isolated research source before starting the pilot")
    from .formal_cli import _acquire_output_lock, _acquire_legacy_read_lock

    if Path(args.output).is_symlink():
        raise RuntimeError("Source research output cannot be a symlink")
    output = Path(args.output).resolve()
    upstream_root = Path(args.upstream_root).resolve()
    if output == upstream_root or upstream_root in output.parents or output in upstream_root.parents:
        raise RuntimeError("Research output must not overlap original evidence")
    lock = _acquire_output_lock(output)
    upstream_lock = None
    progress = PilotProgress(output, protocol)
    guard = ResourceGuard(protocol, output)
    try:
        output.mkdir(exist_ok=True)
        upstream_lock = _acquire_legacy_read_lock(upstream_root)
        if guard.is_set():
            raise RuntimeError(guard.reason)
        write_once(output / "protocol.json", protocol)
        progress.report("authenticating_original_source_inputs")
        command = [sys.executable, "-B", str(server / "formal_v2/tools/verify_source_method_inputs.py")]
        for name in ("origin-server", "config", "dataset", "upstream-root"):
            command += ["--" + name, str(Path(getattr(args, name.replace("-", "_"))).resolve())]
        verified = subprocess.run(command, cwd=args.origin_server, capture_output=True, text=True)
        (output / "authentication.stderr.log").write_text(verified.stderr)
        if verified.returncode:
            raise RuntimeError("Source input authentication failed: " + verified.stderr[-4000:])
        origin = json.loads(verified.stdout)
        write_once(output / "source_origin.json", origin)
        if origin["dataset_sha256"] != protocol["dataset_sha256"] or origin["origin_commit"] != protocol["origin_commit"]:
            raise RuntimeError("Authenticated origin differs from the locked protocol")
        progress.report("loading_formal_npz")
        raw_dataset = FormalDataset.load(args.dataset, require_clean_csi=True)
        if raw_dataset.is_fixture or sha256_file(raw_dataset.source_path) != protocol["dataset_sha256"]:
            raise RuntimeError("Formal NPZ identity changed")
        evidence = evidence_context(base, raw_dataset, "NON_CLAIM")
        identity = {"protocol_sha256": sha256_file(output / "protocol.json"), "origin_sha256": sha256_file(output / "source_origin.json"), "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=server, text=True).strip(), "tracked_diff_sha256": hashlib.sha256(b"").hexdigest(), "python_executable": sys.executable, "device": protocol["device"], "evidence": evidence}
        write_once(output / "identity.json", identity)
        dataset = SourceOnlyDataset(raw_dataset)
        split = source_validation_split(dataset, protocol)
        write_once(output / "source_validation_split.json", split)
        progress.report("preparing_source_corpora", allowed_scene_count=len(dataset.allowed_scenes))
        if guard.is_set():
            raise RuntimeError(guard.reason)
        teacher = load_teacher_bundle(origin["teacher_checkpoint"], base, device=protocol["device"])
        route_norm = fit_route_normalization(dataset, teacher)
        norm_record = read_strict_json(upstream_root / "factorial/normalization.json")
        if sha256_file(upstream_root / "factorial/normalization.json") != origin["inventory"][str(upstream_root / "factorial/normalization.json")]:
            raise RuntimeError("Source normalization changed after authentication")
        if norm_record["source_roles"] != ["source_encoder_train"]:
            raise RuntimeError("Normalization was not fitted on the source training role")
        normalization = factorial.TrainingNormalization(**{key: np.asarray(value, dtype=np.float64) for key, value in norm_record.items() if key != "source_roles"})
        progress.report("preparing_source_training_corpus", **guard.memory_snapshot())
        train = factorial._build_corpus(dataset, dataset.indices_for_role("source_encoder_train"), teacher, base, route_norm, normalization)
        if guard.is_set():
            raise RuntimeError(guard.reason)
        progress.report("preparing_source_selection_corpus", **guard.memory_snapshot())
        selection = factorial._build_corpus(dataset, dataset.indices_for_role("source_method_selection"), teacher, base, route_norm, normalization)
        calibration = calibrate_scales(base, train, selection, protocol, output, progress, guard, identity)
        seed = int(protocol["seed"])
        template = factorial._new_model(base, train, seed, device="cpu")
        initial_state = portable_state_dict(template)
        initial_path = output / "shared_arm_initialization.pt"
        initial_sha = preserve_initial_state(initial_path, initial_state)
        torch.manual_seed(seed + 17003)
        head_template = HeteroscedasticPositionHead(int(base["model"]["state_dim"]), max(8, int(base["model"]["hidden_dim"]) // 2))
        head_sha = preserve_initial_state(output / "shared_head_initialization.pt", head_template.state_dict())
        plan_hashes = [json_hash(asdict(factorial._make_plan(train, int(base["factorial"]["batch_size"]), seed, step))) for step in range(protocol["steps_per_arm"])]
        write_once(output / "paired_plan_hashes.json", plan_hashes)
        records, final_states = [], {}
        for recipe, config in configs.items():
            recipe_root = output / recipe
            recipe_root.mkdir(exist_ok=True)
            write_once(recipe_root / "config.json", config)
            pilot = {"alignment_scale": calibration["alignment_scales"][recipe], "response_scale": calibration["response_scale"], "alignment_null_tolerance": calibration["alignment_null_tolerance"]}
            for arm in ARMS:
                if guard.is_set():
                    raise RuntimeError(guard.reason)
                context = {"identity": identity, "recipe": recipe, "pilot": pilot, "initialization_sha256": initial_sha, "head_initialization_sha256": head_sha, "plan_hashes_sha256": json_hash(plan_hashes), "source_validation_split_sha256": json_hash(split), "steps": protocol["steps_per_arm"]}
                reused = load_completed_arm(recipe_root, arm, context)
                if reused is not None:
                    records.append(reused)
                    final_states[recipe, arm] = reused["state_value_sha256"]
                    progress.completed += 1
                    progress.report("reused_authenticated_arm", recipe=recipe, arm=arm)
                    continue
                model = copy.deepcopy(template).to(protocol["device"])
                if state_hash(model.state_dict()) != initial_sha:
                    raise RuntimeError("Arm did not receive the shared actual initialization")
                trace = ProgressTrace(progress, recipe, arm, model)
                progress.report("training_arm", recipe=recipe, arm=arm, step=0, total_steps=protocol["steps_per_arm"], initialization_sha256=initial_sha)
                model, training = factorial._train_arm(config, train, seed, arm, pilot, step_count=protocol["steps_per_arm"], prepared_model=model, device=protocol["device"], loss_trace=trace, stop_event=guard, checkpoint_path=recipe_root / "resume" / arm, checkpoint_context=context, checkpoint_interval_steps=protocol["checkpoint_interval_steps"])
                torch.cuda.synchronize(protocol["device"])
                state = portable_state_dict(model)
                if state_hash(state) == initial_sha:
                    raise RuntimeError("Arm parameters did not update")
                state_path = recipe_root / (arm + ".pt")
                _atomic_torch_save(state_path, state)
                final_states[recipe, arm] = state_hash(state)
                progress.report("source_only_localization_validation", recipe=recipe, arm=arm, completed_training_steps=training["steps"])
                rows = validate_source_model(model, dataset, normalization, teacher.patch_spec, config, seed, split, head_template, progress=progress, guard=guard, recipe=recipe, arm=arm)
                csv_path = recipe_root / (arm + "_source_validation.csv")
                write_csv(csv_path, rows)
                record = {"recipe": recipe, "arm": arm, "context_sha256": json_hash(context), "training": training, "loss_trace": list(trace), "source_rows": rows, "state_sha256": sha256_file(state_path), "state_value_sha256": state_hash(state), "validation_csv_sha256": sha256_file(csv_path), "initialization_sha256": initial_sha, "scientific_use": "NON_CLAIM", "formal_units_added": 0}
                record["record_sha256"] = json_hash(record)
                write_atomic_json(recipe_root / (arm + "_result.json"), record)
                records.append(record)
                progress.completed += 1
                progress.report("arm_saved", recipe=recipe, arm=arm)
                del model
        controls_equal = {arm: final_states["fixed_margin", arm] == final_states["effect_aware_margin", arm] for arm in ("endpoint", "response")}
        if not all(controls_equal.values()):
            raise RuntimeError("Unchanged control arms diverged between recipes")
        summaries = []
        for record in records:
            for city in sorted(split):
                for budget in protocol["validation"]["label_budgets"]:
                    rows = [row for row in record["source_rows"] if row["city"] == city and row["k"] == budget]
                    summaries.append({"recipe": record["recipe"], "arm": record["arm"], "city": city, "k": budget, "mean_bank_median_error_m": float(np.mean([row["median_error_m"] for row in rows])), "rows": len(rows), "scientific_use": "NON_CLAIM"})
        write_csv(output / "source_validation_summary.csv", summaries)
        by_cell = {(row["recipe"], row["arm"], row["city"], row["k"]): row["mean_bank_median_error_m"] for row in summaries}
        wins = all(by_cell["effect_aware_margin", "full", city, budget] < by_cell["effect_aware_margin", arm, city, budget] for city in split for budget in protocol["validation"]["label_budgets"] for arm in ARMS if arm != "full")
        full_macro = {recipe: float(np.mean([row["mean_bank_median_error_m"] for row in summaries if row["recipe"] == recipe and row["arm"] == "full"])) for recipe in configs}
        result = {"status": "SOURCE_PILOT_COMPLETE_NON_CLAIM", "completed_arm_jobs": len(records), "controls_identical": controls_equal, "eligible_for_larger_source_pilot": wins and full_macro["effect_aware_margin"] < full_macro["fixed_margin"], "full_source_macro_m": full_macro, "observed_scenes": sorted(dataset.observed_scenes), "allowed_scenes": sorted(dataset.allowed_scenes), "formal_units_added": 0, "sota_ready": False, "scientific_use": "NON_CLAIM", "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        for source, expected in origin["inventory"].items():
            if sha256_file(source) != expected:
                raise RuntimeError("Original input changed during research: " + source)
        write_atomic_json(output / "result.json", result)
        progress.report(result["status"], eligible_for_larger_source_pilot=result["eligible_for_larger_source_pilot"])
        return 0
    except Exception as error:
        progress.report("PAUSED_RESOURCE_GUARD" if guard.reason else "FAILED", reason=guard.reason or type(error).__name__ + ": " + str(error))
        raise
    finally:
        if upstream_lock is not None:
            upstream_lock.release()
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
