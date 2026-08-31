from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from formal_v2 import formal_action_inverse_response as physical
from formal_v2 import formal_qualification as qualification
from formal_v2.formal_baselines import RidgeRegressor, ZeroPreservingRidgeRegressor
from formal_v2.formal_config import load_formal_config
from formal_v2.formal_dataset import FormalDataset
from formal_v2.formal_evidence import evidence_context
from formal_v2.formal_io import sha256_file
from formal_v2.formal_protocol import PatchSpec, patchify_csi


SOURCE = Path(
    "/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/"
    "code/CSI-PAIRS-v2.0-server"
)
OUTPUT = SOURCE / "runs/formal-v6-gpu-9850fff-20260825T071800Z"
DATASET = Path(
    "/root/xunlian/Futaoran/正式开始训练/"
    "CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz"
)
CONFIG = SOURCE / "formal_v2/configs/formal_v2.json"
ATTEMPT = OUTPUT / "qualification/checkpoints/qualification_probes/attempt_0001"
PHYSICAL_CHECKPOINT = ATTEMPT / "physical_response.pt"


def _preserved_save(expected: Path):
    expected = expected.resolve()

    def save(path, _metadata=None):
        observed = Path(path)
        if observed.is_symlink() or observed.resolve() != expected or not observed.is_file():
            raise RuntimeError("recovered qualification probe save target changed")

    return save


def _load_probe(path: Path, method: str, expected_evidence: dict):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing preserved qualification probe: {path}")
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"][0]))
        for key in (
            "config_sha256",
            "dataset_sha256",
            "source_tree_sha256",
            "requirements_lock_sha256",
            "runtime_provenance_sha256",
        ):
            if metadata.get(key) != expected_evidence[key]:
                raise RuntimeError(f"preserved qualification probe {method} has stale {key}")
        if (
            metadata.get("schema_version") != "csi-pairs-v6-qualification-probe-v2"
            or metadata.get("input_contract") != qualification._method_contract(method)
            or metadata.get("source_roles") != ["source_encoder_train"]
        ):
            raise RuntimeError(f"preserved qualification probe {method} metadata changed")
        alpha = float(payload["alpha"][0])
        if method in qualification.ZERO_PRESERVING_METHODS:
            model = ZeroPreservingRidgeRegressor(alpha=alpha)
            model.weights = np.asarray(payload["weights"], dtype=np.float64)
            model.feature_scale = np.asarray(payload["feature_scale"], dtype=np.float64)
        else:
            model = RidgeRegressor(alpha=alpha)
            model.weights = np.asarray(payload["weights"], dtype=np.float64)
            model.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float64)
            model.feature_scale = np.asarray(payload["feature_scale"], dtype=np.float64)
    if not all(
        np.all(np.isfinite(value))
        for value in (
            model.weights,
            model.feature_scale,
            getattr(model, "feature_mean", np.asarray([0.0])),
        )
    ):
        raise RuntimeError(f"preserved qualification probe {method} is non-finite")
    model.save = _preserved_save(path)
    return model


def _selection_only_physical_response(
    dataset,
    train_scenes,
    selection_scenes,
    train_routed,
    selection_routed,
    formal_config,
    train_wrong_actions,
    selection_wrong_actions,
    *,
    seed,
    device,
    physical_config=None,
):
    del seed
    config = (
        physical.load_physical_response_config()
        if physical_config is None
        else physical_config
    )
    candidate_config, candidate_path = physical.load_bound_candidate_config(dataset)
    model, bindings = physical.load_physical_response_checkpoint(PHYSICAL_CHECKPOINT)
    expected_evidence = evidence_context(formal_config, dataset, "FORBIDDEN")
    if (
        bindings.get("dataset_sha256") != sha256_file(dataset.source_path)
        or bindings.get("formal_config_sha256") != expected_evidence["config_sha256"]
        or bindings.get("physical_config_sha256")
        != sha256_file(SOURCE / "formal_v2/configs/physical_response_v1.json")
        or bindings.get("model_source_sha256")
        != sha256_file(SOURCE / "formal_v2/formal_action_inverse_response.py")
        or bindings.get("candidate_config_sha256") != sha256_file(candidate_path)
        or bindings.get("train_bank_ids")
        != [str(dataset.bank_ids[int(scene)]) for scene in train_scenes]
        or bindings.get("selection_bank_ids")
        != [str(dataset.bank_ids[int(scene)]) for scene in selection_scenes]
    ):
        raise RuntimeError("preserved physical-response checkpoint binding changed")
    for gate in model.gates:
        gate.to(device)
        gate.eval()

    spec = PatchSpec.from_metadata(dataset.metadata)
    patch_scale = patchify_csi(train_routed.normalization.channel_scale, spec)
    key_chunks = []
    prediction_chunks = []
    oracle_chunks = []
    swap_chunks = []
    status_chunks = []
    probability_quantiles = []
    for scene_value in selection_scenes:
        scene = int(scene_value)
        rows = physical.build_physical_response_rows(
            dataset,
            (scene,),
            formal_config,
            candidate_config,
            patch_scale,
            selection_routed.response_route,
            position_stride=int(config["selection_position_stride"]),
            grid_spacing_m=float(config["grid_spacing_m"]),
            decomposition_ridge=float(config["decomposition_ridge"]),
            include_targets=False,
        )
        prediction, probability = physical.predict_physical_response(
            rows, model, patch_scale, spec
        )
        swap, statuses = physical.physical_action_swap_predictions(
            rows, prediction, dataset, selection_wrong_actions
        )
        scene_keys = [row["key"] for row in rows]
        key_chunks.extend(scene_keys)
        prediction_chunks.append(prediction)
        swap_chunks.append(swap)
        status_chunks.append(statuses)
        probability_quantiles.append(
            {
                "scene": scene,
                "bank_id": str(dataset.bank_ids[scene]),
                "quantiles": {
                    str(value): float(np.quantile(probability, value))
                    for value in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
                },
            }
        )
        del rows, prediction, probability, swap, statuses
        gc.collect()

        oracle_rows = physical.build_physical_response_rows(
            dataset,
            (scene,),
            formal_config,
            candidate_config,
            patch_scale,
            selection_routed.response_route,
            position_stride=int(config["selection_position_stride"]),
            grid_spacing_m=float(config["grid_spacing_m"]),
            decomposition_ridge=float(config["decomposition_ridge"]),
            include_targets=False,
            receiver_mode="oracle",
        )
        if [row["key"] for row in oracle_rows] != scene_keys:
            raise RuntimeError("recovered oracle physical-response rows changed")
        oracle_prediction, _ = physical.predict_physical_response(
            oracle_rows, model, patch_scale, spec
        )
        oracle_chunks.append(oracle_prediction)
        print(
            json.dumps(
                {
                    "stage": "recovered_physical_response_selection_bank_complete",
                    "scene": scene,
                    "bank_id": str(dataset.bank_ids[scene]),
                    "rows": len(scene_keys),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del oracle_rows, oracle_prediction, scene_keys
        gc.collect()

    fit_positions = len(
        range(0, dataset.position_count, int(config["fit_position_stride"]))
    )
    fit_row_count = sum(
        sum(1 for _ in dataset.directed_edges(int(scene)))
        * fit_positions
        * spec.patch_count
        for scene in train_scenes
    )
    audit = {
        **model.metadata,
        "fit_row_count_including_gray": int(fit_row_count),
        "selection_row_count": len(key_chunks),
        "physical_config": config,
        "physical_config_sha256": sha256_file(
            SOURCE / "formal_v2/configs/physical_response_v1.json"
        ),
        "candidate_config": str(candidate_path),
        "candidate_config_sha256": sha256_file(candidate_path),
        "selection_probability_quantiles": probability_quantiles,
        "oracle_position_control": {
            "receiver_position_source": "frozen_ground_truth_receiver_xy",
            "target_csi_read": False,
            "route_label_read_by_predictor": False,
            "shared_fit_model_and_gate": True,
        },
        "train_wrong_action_plan_bound": train_wrong_actions is not None,
    }
    return (
        model,
        key_chunks,
        np.concatenate(prediction_chunks, axis=0),
        np.concatenate(swap_chunks, axis=0),
        np.concatenate(oracle_chunks, axis=0),
        np.concatenate(status_chunks, axis=0),
        audit,
    )


def _preserve_physical_checkpoint(path, model, bindings):
    target = Path(path)
    if (
        target.is_symlink()
        or target.resolve() != PHYSICAL_CHECKPOINT.resolve()
        or not target.is_file()
    ):
        raise RuntimeError("recovered physical-response save target changed")
    existing, existing_bindings = physical.load_physical_response_checkpoint(target)
    if (
        existing_bindings != bindings
        or existing.thresholds != model.thresholds
        or existing.metadata != model.metadata
        or existing.response_scale != model.response_scale
    ):
        raise RuntimeError("recovered physical-response checkpoint payload changed")
    return target


def main() -> None:
    config = load_formal_config(CONFIG)
    dataset = FormalDataset.load(DATASET, require_clean_csi=True)
    expected_evidence = evidence_context(config, dataset, "FORBIDDEN")
    if ATTEMPT.is_symlink() or not ATTEMPT.is_dir():
        raise RuntimeError("preserved qualification attempt is unavailable")
    models = {
        method: _load_probe(ATTEMPT / f"{method}.npz", method, expected_evidence)
        for method in qualification.QUALIFICATION_METHODS
    }

    qualification._reserve_qualification_probe_attempt = lambda _output: ATTEMPT
    qualification._fit_qualification_probes_streaming = lambda *_args, **_kwargs: models
    physical.fit_and_predict_qualification_response = _selection_only_physical_response
    physical.save_physical_response_checkpoint = _preserve_physical_checkpoint

    result = qualification.run_formal_qualification(
        config,
        dataset,
        OUTPUT,
        resume=True,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
