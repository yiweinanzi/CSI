"""Pre-D integration smoke of the E worker/timing machinery, not formal E."""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/root/xunlian/Futaoran/CSI_STREAMING_GPU_FIX_20260908/code/CSI-PAIRS-v2.0-server')
from formal_v2.formal_evidence import configure_reproducible_runtime
from formal_v2.formal_probes import CompatibilityProbe
from formal_v2.tools.validate_formal_dual_probe import train_candidate, kernel_overlap, compare


def main():
    output = Path(__file__).parent / 'dual-execution-smoke'
    output.mkdir(exist_ok=False)
    configure_reproducible_runtime()
    rng = np.random.default_rng(417)
    arrays = {role + '_features': rng.normal(size=(257, 128)) for role in ('train', 'selection')}
    arrays.update({role + '_labels': np.arange(257) % 2 for role in ('train', 'selection')})
    config = {'evaluation': {'probe_steps': 2000, 'probe_learning_rate': .001}}
    initial = {}
    for offset, family in enumerate(('linear', 'mlp2')):
        torch.manual_seed(417 + offset)
        initial[family] = CompatibilityProbe(128, family, 128)
    serial = {family: train_candidate(model, arrays, config, 'cuda:0') for family, model in initial.items()}
    barrier = threading.Barrier(2)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {family: pool.submit(train_candidate, model, arrays, config, f'cuda:{index}', barrier) for index, (family, model) in enumerate(initial.items())}
            parallel = {family: future.result() for family, future in futures.items()}
    trace = output / 'trace.json'
    profile.export_chrome_trace(str(trace))
    overlap = kernel_overlap(json.loads(trace.read_text())['traceEvents'])
    comparisons = {family: compare(parallel[family]['probabilities'], serial[family]['probabilities']) for family in initial}
    result = {'scientific_use': 'NON_CLAIM', 'input_kind': 'synthetic_smoke_not_E', 'cuda_overlap': overlap, 'comparisons': comparisons}
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    assert all(row['passed'] for row in comparisons.values())
    assert overlap['actual_kernel_overlap_microseconds'] > 0


if __name__ == '__main__':
    main()
