"""E acceptance on authenticated full source features, never a formal-unit count."""

import argparse
import copy
import csv
import io
import json
import os
import resource
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from formal_v2.formal_config import load_formal_config
from formal_v2.formal_evaluation_repair import load_validation_corpus
from formal_v2.formal_evidence import configure_reproducible_runtime
from formal_v2.formal_evaluation_resume import write_atomic_json
from formal_v2.formal_io import sha256_file
from formal_v2.formal_metrics import binary_auroc, binary_nll
from formal_v2.formal_probes import CompatibilityProbe, _fit_binary, predict_binary_probe


TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return {"passed": bool(np.allclose(a, b, **TOLERANCE)), "max_abs": float(np.max(np.abs(a - b)))}


def kernel_overlap(events):
    intervals = {}
    for event in events:
        if event.get("cat") == "kernel":
            device = int(event["args"]["device"])
            intervals.setdefault(device, []).append((event["ts"], event["ts"] + event["dur"]))
    merged = {}
    for device, rows in intervals.items():
        merged[device] = []
        for start, end in sorted(rows):
            if merged[device] and start <= merged[device][-1][1]:
                merged[device][-1][1] = max(end, merged[device][-1][1])
            else:
                merged[device].append([start, end])
    a, b = merged.get(0, []), merged.get(1, [])
    i = j = 0
    overlap = 0.0
    while i < len(a) and j < len(b):
        overlap += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return {"kernel_counts": {str(k): len(v) for k, v in intervals.items()}, "actual_kernel_overlap_microseconds": overlap}


def sample_resources():
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemAvailable', 'MemTotal'):
            memory[key] = int(value.split()[0]) * 1024
    gpu = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits',
    ], text=True, timeout=10)
    return {"monotonic": time.monotonic(), "host_memory_bytes": memory, "process_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, "gpus": list(csv.reader(io.StringIO(gpu)))}


def train_candidate(initial, arrays, config, device, barrier=None):
    torch.cuda.set_device(device)
    setup_start = time.monotonic()
    model = copy.deepcopy(initial).to(device)
    assert all(torch.equal(value, model.state_dict()[name].cpu()) for name, value in initial.state_dict().items())
    events = []
    forward_devices = []

    def first_forward(module, inputs):
        forward_devices.append(str(inputs[0].device))
        assert inputs[0].device == next(module.parameters()).device
        handle.remove()

    handle = model.register_forward_pre_hook(first_forward)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    if barrier is not None:
        barrier.wait(timeout=60)
    start = time.monotonic()
    cuda_start, cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cuda_start.record()

    def report(event):
        if event.get('phase') != 'training' or event.get('step', 0) % 100 == 0:
            events.append({"monotonic": time.monotonic(), **event})

    _fit_binary(model, arrays['train_features'], arrays['train_labels'], config, train_batch_rows=4096, prefer_full_batch=True, progress_callback=report)
    cuda_end.record()
    cuda_end.synchronize()
    end = time.monotonic()
    assert forward_devices == [str(torch.device(device))]
    assert all(p.grad is not None and p.grad.device.type == 'cuda' and torch.isfinite(p.grad).all().item() for p in model.parameters())
    assert any(not torch.equal(value, model.state_dict()[name].cpu()) for name, value in initial.state_dict().items())
    assert any(row.get('phase') == 'training_complete' and row.get('step') == 2000 for row in events)
    assert all(row['execution_mode'] == 'full_batch' for row in events if row.get('phase') == 'memory_admission')
    prediction_start = time.monotonic()
    probabilities = predict_binary_probe(model, arrays['selection_features'], batch_rows=4096)
    prediction_end = time.monotonic()
    metrics = {'selection_nll': binary_nll(arrays['selection_labels'], probabilities), 'selection_auroc': binary_auroc(arrays['selection_labels'], probabilities)}
    return {
        'state': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        'probabilities': probabilities,
        'record': {'family': model.family, 'device': device, 'setup_and_barrier_seconds': start - setup_start, 'training_start_monotonic': start, 'training_end_monotonic': end, 'training_wall_seconds': end-start, 'cuda_training_interval_ms': cuda_start.elapsed_time(cuda_end), 'prediction_seconds': prediction_end-prediction_start, 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device), 'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device), 'actual_forward_devices': forward_devices, 'cuda_gradients_and_parameter_updates_verified': True, 'metrics': metrics, 'events': events},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--corpus-sha256', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_atomic_json(output / 'predeclared_tolerances.json', {'float32_same_hardware_tolerance': TOLERANCE, 'rank_threshold_auroc_and_family_changes_allowed': False, 'timing_note': 'CUDA intervals include dispatch gaps, not busy-kernel time; parallel run is instrumented; no end-to-end speed ratio.'})
    configure_reproducible_runtime()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1' or torch.cuda.device_count() != 2:
        raise RuntimeError('This recorded E protocol requires explicit CUDA_VISIBLE_DEVICES=0,1 and two logical devices')
    config = load_formal_config(args.config)
    if int(config['evaluation']['probe_steps']) != 2000:
        raise RuntimeError('E requires the original 2,000 complete updates')
    arrays, corpus, bundle = load_validation_corpus(args.corpus, config, expected_sha256=args.corpus_sha256)
    initial = {}
    for offset, family in enumerate(('linear', 'mlp2')):
        torch.manual_seed(int(corpus['checkpoint']['seed']) + 31001 + offset)
        initial[family] = CompatibilityProbe(arrays['train_features'].shape[1], family, int(config['evaluation']['probe_hidden_dim']))
        torch.save(initial[family].state_dict(), output / (family + '_shared_initialization.pt'))
    resources = []
    try:
        resources.append(sample_resources())
    except (OSError, subprocess.SubprocessError) as error:
        resources.append({'monitor_warning': str(error)})
    serial = {family: train_candidate(model, arrays, config, 'cuda:0') for family, model in initial.items()}
    barrier = threading.Barrier(2)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {family: pool.submit(train_candidate, model, arrays, config, f'cuda:{index}', barrier) for index, (family, model) in enumerate(initial.items())}
            while not all(future.done() for future in futures.values()):
                try:
                    resources.append(sample_resources())
                except (OSError, subprocess.SubprocessError) as error:
                    resources.append({'monitor_warning': str(error)})
                time.sleep(.5)
            concurrent = {family: future.result() for family, future in futures.items()}
    trace_path = output / 'dual_cuda_trace.json'
    profile.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text())
    overlap = kernel_overlap(trace['traceEvents'])
    comparisons = {}
    for family in initial:
        a, b = serial[family], concurrent[family]
        states = {key: compare(b['state'][key], value) for key, value in a['state'].items()}
        probabilities = compare(b['probabilities'], a['probabilities'])
        rankings_equal = bool(np.array_equal(np.argsort(a['probabilities'], kind='stable'), np.argsort(b['probabilities'], kind='stable')))
        threshold_changes = int(np.count_nonzero((a['probabilities'] >= .5) != (b['probabilities'] >= .5)))
        comparisons[family] = {'state': states, 'probabilities': probabilities, 'rankings_equal': rankings_equal, 'threshold_changes': threshold_changes, 'auroc_equal': a['record']['metrics']['selection_auroc'] == b['record']['metrics']['selection_auroc']}
        for mode, row in (('serial', a), ('concurrent', b)):
            torch.save(row['state'], output / (mode + '_' + family + '.pt'))
            np.save(output / (mode + '_' + family + '_probabilities.npy'), row['probabilities'])
    winners = {mode: min(rows, key=lambda name: (rows[name]['record']['metrics']['selection_nll'], name)) for mode, rows in (('serial', serial), ('concurrent', concurrent))}
    winner = winners['concurrent']
    formal_state = (
        {key: compare(concurrent[winner]['state'][key], value.numpy()) for key, value in bundle.compatibility_probe.state_dict().items()}
        if winner == bundle.compatibility_probe.family else {}
    )
    result = {
        'status': 'CHECKED_NOT_YET_PASSED', 'scientific_use': 'NON_CLAIM', 'sota_ready': False,
        'scope': 'Two independent compatibility candidate probes on full original source corpus; not two complete probe bundles and not additional formal units',
        'full_bundle_concurrency_approved': False, 'formal_completed_units_added': 0,
        'corpus_manifest_sha256': sha256_file(args.corpus), 'shapes': {key: list(value.shape) for key, value in arrays.items()},
        'serial': {name: row['record'] for name, row in serial.items()}, 'concurrent': {name: row['record'] for name, row in concurrent.items()},
        'comparisons': comparisons, 'winners': winners, 'formal_unit_selected_family': bundle.compatibility_probe.family,
        'formal_unit_state_comparison': formal_state, 'cuda_overlap': overlap,
        'resources': resources, 'host_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'trace_sha256': sha256_file(trace_path),
    }
    write_atomic_json(output / 'result.json', result)
    assert overlap['actual_kernel_overlap_microseconds'] > 0, overlap
    for record in comparisons.values():
        assert all(row['passed'] for row in record['state'].values()) and record['probabilities']['passed'], record
        assert record['rankings_equal'] and record['threshold_changes'] == 0 and record['auroc_equal'], record
    assert winners['serial'] == winners['concurrent'] == bundle.compatibility_probe.family, winners
    assert all(row['passed'] for row in formal_state.values()), formal_state
    assert concurrent[winner]['record']['metrics']['selection_auroc'] == bundle.compatibility_selection['selection_auroc']
    assert abs(concurrent[winner]['record']['metrics']['selection_nll'] - bundle.compatibility_selection['selection_nll']) <= 1e-6
    result['status'] = 'PASS_FULL_CORPUS_INDEPENDENT_CANDIDATE_DUAL_GPU'
    write_atomic_json(output / 'result.json', result)
    print(json.dumps({'status': result['status'], 'output': str(output), 'cuda_overlap': overlap, 'formal_completed_units_added': 0}))


if __name__ == '__main__':
    main()
