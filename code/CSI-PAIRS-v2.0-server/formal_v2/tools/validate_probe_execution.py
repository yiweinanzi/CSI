"""A/B engineering validation; synthetic inputs never become formal evidence."""

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path
from unittest import mock

import numpy as np
import scipy
from scipy.stats import rankdata
import torch

from formal_v2.formal_evidence import configure_reproducible_runtime
from formal_v2.formal_metrics import binary_auroc
from formal_v2.formal_probes import ActionResponseProbe, fit_action_response_probe, predict_response_probe


TOLERANCES = {
    "forward_loss_gradient_step": {"rtol": 1e-4, "atol": 5e-6},
    "complete_2000_step_output": {"rtol": 2e-3, "atol": 2e-4},
}


def differences(actual, expected, tolerance):
    a, b = np.asarray(actual), np.asarray(expected)
    maximum = float(np.max(np.abs(a - b)))
    scale = float(np.max(np.abs(b)))
    passed = bool(np.allclose(a, b, **tolerance))
    return {"max_abs": maximum, "max_abs_over_reference_max": maximum / max(scale, 1e-30), "passed": passed, "tolerance": tolerance}


def metrics():
    source = subprocess.check_output([
        "git", "show", "c5917018610b262f5263ccaeba2f087b2eb96a96:code/CSI-PAIRS-v2.0-server/formal_v2/formal_metrics.py",
    ], text=True)
    namespace = {}
    exec(compile(source, "original_formal_metrics.py", "exec"), namespace)
    original = namespace["binary_auroc"]
    rng = np.random.default_rng(417)
    cases = []
    for dtype in (np.float32, np.float64, np.int32, np.int64):
        for name, scores in (
            ("random", rng.normal(size=128)), ("ties", rng.integers(-2, 3, 128)),
            ("all_equal", np.ones(128)), ("perfect", np.arange(128) % 2),
            ("reverse", 1 - np.arange(128) % 2),
        ):
            labels = np.arange(128) % 2
            scores = scores.astype(dtype)
            wide = np.empty((128, 2), dtype=dtype)
            wide[:, 0] = scores
            for values in (scores, wide[:, 0]):
                positives, negatives = values[labels == 1, None], values[labels == 0][None, :]
                pairwise = float(np.mean((positives > negatives) + 0.5 * (positives == negatives)))
                old, new = original(labels, values), binary_auroc(labels, values)
                assert old == new == pairwise
                cases.append({"case": name, "dtype": str(dtype), "contiguous": bool(values.flags.c_contiguous), "auc": new})
    for count in (1, 999):
        labels = np.r_[np.ones(count), np.zeros(1000 - count)]
        scores = rng.normal(size=1000)
        assert binary_auroc(labels, scores) == original(labels, scores)
    for labels, scores in (([], []), ([1, 1], [1, 2]), ([0, 1], [1, np.nan]), ([0, 1], [1, np.inf]), ([0, 2], [1, 2])):
        for function in (original, binary_auroc):
            try:
                function(np.asarray(labels), np.asarray(scores))
            except ValueError:
                pass
            else:
                raise AssertionError("invalid AUROC input was accepted")
    timings = []
    for size in (8000, 16000, 32000, 64000):
        labels = np.arange(size) % 2
        scores = rng.normal(size=size)
        runs = {"old": [], "new": [], "scipy_rank_reference": []}
        for _ in range(3):
            for name, function in (("old", original), ("new", binary_auroc)):
                start = time.perf_counter()
                value = function(labels, scores)
                runs[name].append(time.perf_counter() - start)
                if name == "old":
                    expected = value
                else:
                    assert value == expected
            start = time.perf_counter()
            ranks = rankdata(scores, method="average")
            count = int(labels.sum())
            value = float((ranks[labels == 1].sum() - count * (count + 1) / 2) / (count * (size - count)))
            runs["scipy_rank_reference"].append(time.perf_counter() - start)
            assert value == expected
        timings.append({"rows": size, "seconds": runs})
    return {"passed": True, "scipy_version": scipy.__version__, "numpy_version": np.__version__, "cases": cases, "timings": timings, "timing_boundary": "input arrays generated once outside timing; function-internal validation/conversion/sorting included; same process and thread environment; three repeats; no end-to-end speed claim"}


def probes(output):
    configure_reproducible_runtime()
    torch.manual_seed(314)
    initial = ActionResponseProbe(7, 3, 8)
    state = copy.deepcopy(initial.state_dict())
    initial_path = output / "shared_initialization.pt"
    torch.save(state, initial_path)
    digest = hashlib.sha256(initial_path.read_bytes()).hexdigest()
    rng = np.random.default_rng(43)
    x = rng.normal(size=(33, 7)).astype(np.float32)
    zero = x.copy()
    zero[:, 4:] = 0
    y = rng.normal(size=(33, 3)).astype(np.float32)
    np.savez(output / "fixed_probe_inputs.npz", features=x, zero=zero, targets=y)

    def one_step(device):
        model = copy.deepcopy(initial).to(device)
        for key, value in state.items():
            assert torch.equal(model.state_dict()[key].cpu(), value)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        a, z, target = [torch.as_tensor(v, device=device) for v in (x, zero, y)]
        prediction = model(a) - model(z)
        loss = ((prediction - target) ** 2).mean()
        loss.backward()
        gradients = np.concatenate([p.grad.detach().cpu().numpy().ravel() for p in model.parameters()])
        optimizer.step()
        parameters = np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])
        return {"forward": prediction.detach().cpu().numpy(), "loss": np.asarray([loss.item()]), "gradient": gradients, "updated_parameters": parameters}

    cpu = one_step("cpu")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
        gpu = one_step("cuda:0")
        torch.cuda.synchronize(0)
    profile.export_chrome_trace(str(output / "cuda_forward_backward_update_trace.json"))
    comparisons = {key: differences(gpu[key], cpu[key], TOLERANCES["forward_loss_gradient_step"]) for key in cpu}
    assert all(value["passed"] for value in comparisons.values()), comparisons
    events = [event.name for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    assert events, "No CUDA execution events observed"
    before = np.concatenate([p.detach().numpy().ravel() for p in initial.parameters()])
    assert not np.array_equal(before, gpu["updated_parameters"])
    results, timings = {}, {}

    def initialized(_factory, _seed, device, _lock):
        model = copy.deepcopy(initial).to(device)
        for key, value in state.items():
            assert torch.equal(model.state_dict()[key].cpu(), value)
        return model

    config = {"evaluation": {"probe_steps": 2000, "probe_hidden_dim": 8, "probe_learning_rate": 0.001}}
    for name, device, batch in (("cpu_full", "cpu", None), ("gpu_full", "cuda:0", None), ("gpu_accumulated", "cuda:0", 7)):
        if device != "cpu":
            torch.cuda.synchronize(0)
        start = time.perf_counter()
        with mock.patch("formal_v2.formal_probes._initialize_probe", side_effect=initialized):
            model = fit_action_response_probe(x, y, config, seed=314, zero_action_x=zero, device=device, train_batch_rows=batch)
        results[name] = predict_response_probe(model, x, zero)
        if device != "cpu":
            torch.cuda.synchronize(0)
        timings[name] = time.perf_counter() - start
        np.save(output / (name + "_predictions.npy"), results[name])
    complete = {name: differences(value, results["cpu_full"], TOLERANCES["complete_2000_step_output"]) for name, value in results.items() if name != "cpu_full"}
    assert all(value["passed"] for value in complete.values()), complete
    return {"passed": True, "input_kind": "synthetic_smoke_not_formal", "dtype": "float32", "initialization_sha256": digest, "one_step_comparisons": comparisons, "complete_2000_step_comparisons": complete, "seconds": timings, "cuda_kernel_event_count": len(events), "cuda_kernel_examples": sorted(set(events))[:20], "precision": {"tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32, "deterministic": torch.are_deterministic_algorithms_enabled(), "amp": False}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("metrics", "probes"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "predeclared_tolerances.json").write_text(json.dumps(TOLERANCES, indent=2) + "\n")
    result = metrics() if args.stage == "metrics" else probes(output)
    result["scientific_use"] = "NON_CLAIM"
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
