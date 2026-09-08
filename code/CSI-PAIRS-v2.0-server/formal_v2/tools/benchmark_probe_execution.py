"""Synthetic execution benchmark only. Never emits a scientific gate or meters."""

import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from formal_v2.formal_evidence import configure_reproducible_runtime
from formal_v2.formal_probes import fit_action_response_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--rows", type=int, default=8192)
    args = parser.parse_args()
    if args.steps < 1 or args.rows < 2:
        parser.error("steps must be positive and rows must be at least two")
    configure_reproducible_runtime()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This benchmark requires two CUDA devices")
    rng = np.random.default_rng(91723)
    features = rng.normal(size=(args.rows, 256))
    zero = features.copy()
    zero[:, 128:] = 0
    targets = rng.normal(size=(args.rows, 32))
    config = {"evaluation": {
        "probe_steps": args.steps, "probe_hidden_dim": 128,
        "probe_learning_rate": 0.001,
    }}
    guard = threading.Lock()
    started = threading.Barrier(2)
    progress = {}

    def run(index):
        def report(event):
            with guard:
                progress[str(index)] = event

        started.wait(timeout=60)
        begin = time.monotonic()
        probe = fit_action_response_probe(
            features, targets, config, seed=618 + index,
            zero_action_x=zero, device=f"cuda:{index}",
            train_batch_rows=4096, rng_lock=guard, progress_callback=report,
        )
        torch.cuda.synchronize(index)
        return {
            "device": str(next(probe.parameters()).device),
            "steps": args.steps, "rows": args.rows,
            "seconds": time.monotonic() - begin,
            "all_parameters_finite": all(bool(torch.isfinite(p).all()) for p in probe.parameters()),
        }

    samples = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, index) for index in (0, 1)]
        while not all(future.done() for future in futures):
            gpu = subprocess.check_output([
                "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ], text=True, timeout=15).strip()
            with guard:
                current = {key: dict(value) for key, value in progress.items()}
            sample = {"gpu": gpu, "progress": current}
            samples.append(sample)
            print(json.dumps({"scientific_use": "NON_CLAIM", **sample}), flush=True)
            time.sleep(2)
        results = [future.result() for future in futures]
    print(json.dumps({
        "scientific_use": "NON_CLAIM", "synthetic_execution_benchmark": True,
        "results": results, "utilization_samples": len(samples),
        "note": "Not the formal NPZ, not experiment evidence, no localization metrics.",
    }), flush=True)


if __name__ == "__main__":
    main()
