"""Additional B smoke evidence; shared states, no formal data substitution."""

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SERVER = Path('/root/xunlian/Futaoran/CSI_STREAMING_GPU_FIX_20260908/code/CSI-PAIRS-v2.0-server')
sys.path.insert(0, str(SERVER))
from formal_v2.formal_evidence import configure_reproducible_runtime
from formal_v2.formal_metrics import binary_auroc, binary_nll
from formal_v2.formal_probes import CompatibilityProbe, _fit_binary, predict_binary_probe


def main():
    root = Path(__file__).parent / 'B-binary-numerical'
    root.mkdir(exist_ok=False)
    tolerance = {'one_step': {'rtol': 1e-4, 'atol': 5e-6}, 'complete': {'rtol': 2e-3, 'atol': 2e-4}}
    (root / 'predeclared_tolerances.json').write_text(json.dumps(tolerance, indent=2))
    configure_reproducible_runtime()
    rng = np.random.default_rng(417)
    x = rng.normal(size=(129, 11)).astype(np.float32)
    y = (np.arange(len(x)) % 2).astype(np.float32)
    selection = rng.normal(size=(64, 11)).astype(np.float32)
    labels = (np.arange(len(selection)) % 2).astype(np.float32)
    np.savez(root / 'inputs.npz', x=x, y=y, selection=selection, labels=labels)
    config = {'evaluation': {'probe_steps': 2000, 'probe_learning_rate': 0.001}}
    results = []
    for family in ('linear', 'mlp2'):
        torch.manual_seed(417)
        initial = CompatibilityProbe(x.shape[1], family, 16)
        state_path = root / (family + '_initialization.pt')
        torch.save(initial.state_dict(), state_path)
        values = {}
        for device in ('cpu', 'cuda:1'):
            probe = copy.deepcopy(initial).to(device)
            assert all(torch.equal(initial.state_dict()[k], v.cpu()) for k, v in probe.state_dict().items())
            optimizer = torch.optim.AdamW(probe.parameters(), lr=0.001)
            a, b = [torch.as_tensor(v, device=device) for v in (x, y)]
            logits = probe(a)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, b)
            loss.backward()
            grads = np.concatenate([p.grad.cpu().numpy().ravel() for p in probe.parameters()])
            optimizer.step()
            assert all(p.device == a.device and p.grad.device == a.device for p in probe.parameters())
            assert all(s['exp_avg'].device == a.device and s['exp_avg_sq'].device == a.device for s in optimizer.state.values())
            updated = np.concatenate([p.detach().cpu().numpy().ravel() for p in probe.parameters()])
            assert not np.array_equal(updated, np.concatenate([p.detach().numpy().ravel() for p in initial.parameters()]))
            values[device] = {'forward': logits.detach().cpu().numpy(), 'loss': np.asarray([loss.item()]), 'gradient': grads, 'updated': updated}
        one_step = {}
        for key, reference in values['cpu'].items():
            actual = values['cuda:1'][key]
            passed = bool(np.allclose(actual, reference, **tolerance['one_step']))
            one_step[key] = {'max_abs': float(np.max(np.abs(actual - reference))), 'passed': passed}
            assert passed, one_step
        outputs = {}
        timings = {}
        for device in ('cpu', 'cuda:1'):
            probe = copy.deepcopy(initial).to(device)
            assert all(torch.equal(initial.state_dict()[k], v.cpu()) for k, v in probe.state_dict().items())
            start = time.perf_counter()
            _fit_binary(probe, x, y, config)
            prediction = predict_binary_probe(probe, selection)
            timings[device] = time.perf_counter() - start
            outputs[device] = prediction
            np.save(root / (family + '_' + device.replace(':', '-') + '_probabilities.npy'), prediction)
        cpu, gpu = outputs['cpu'], outputs['cuda:1']
        passed = bool(np.allclose(cpu, gpu, **tolerance['complete']))
        detail = {
            'family': family, 'initialization_sha256': hashlib.sha256(state_path.read_bytes()).hexdigest(),
            'one_step': one_step, 'complete_max_abs': float(np.max(np.abs(cpu-gpu))),
            'complete_passed': passed, 'training_steps': 2000, 'seconds': timings,
            'cpu_auroc': binary_auroc(labels, cpu), 'gpu_auroc': binary_auroc(labels, gpu),
            'cpu_nll': binary_nll(labels, cpu), 'gpu_nll': binary_nll(labels, gpu),
            'threshold_0_5_disagreements': int(np.count_nonzero((cpu >= .5) != (gpu >= .5))),
            'ranking_equal': bool(np.array_equal(np.argsort(cpu), np.argsort(gpu))),
        }
        results.append(detail)
        assert passed, detail
        assert detail['threshold_0_5_disagreements'] == 0 and detail['ranking_equal'], detail
    selected = {device: min(results, key=lambda row: (row[device + '_nll'], row['family']))['family'] for device in ('cpu', 'gpu')}
    assert selected['cpu'] == selected['gpu'], selected
    result = {'passed': True, 'scientific_use': 'NON_CLAIM', 'input_kind': 'synthetic_smoke', 'device': 'cuda:1', 'dtype': 'float32', 'amp': False, 'tf32': torch.backends.cuda.matmul.allow_tf32, 'selected_family': selected, 'results': results}
    (root / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
