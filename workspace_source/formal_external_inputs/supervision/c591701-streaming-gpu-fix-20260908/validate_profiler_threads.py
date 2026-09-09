"""Instrumentation smoke only, using the idle physical GPU 1."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/root/xunlian/Futaoran/CSI_STREAMING_GPU_REPAIR_EVIDENCE_20260908/code/CSI-PAIRS-v2.0-server')
from formal_v2.formal_evidence import configure_reproducible_runtime
from formal_v2.formal_probes import CompatibilityProbe
from formal_v2.tools.validate_formal_dual_probe import train_candidate, kernel_overlap


def main():
    output = Path(__file__).parent / 'profiler-worker-smoke'
    output.mkdir(exist_ok=False)
    configure_reproducible_runtime()
    torch.manual_seed(417)
    model = CompatibilityProbe(5, 'linear', 8)
    rng = np.random.default_rng(417)
    arrays = {role + '_features': rng.normal(size=(32, 5)) for role in ('train', 'selection')}
    arrays.update({role + '_labels': np.arange(32) % 2 for role in ('train', 'selection')})
    config = {'evaluation': {'probe_steps': 2000, 'probe_learning_rate': .001}}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(train_candidate, model, arrays, config, 'cuda:0').result()
    trace = output / 'trace.json'
    profile.export_chrome_trace(str(trace))
    observed = kernel_overlap(json.loads(trace.read_text())['traceEvents'])
    record = {'scientific_use': 'NON_CLAIM', 'input_kind': 'synthetic_instrumentation_smoke', 'worker_record': result['record'], 'cuda_trace': observed}
    (output / 'result.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(observed))
    assert observed['kernel_counts'].get('0', 0) > 0, observed


if __name__ == '__main__':
    main()
