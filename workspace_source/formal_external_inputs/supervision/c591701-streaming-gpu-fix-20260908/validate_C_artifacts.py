"""Validate completed original-scale C artifacts without refitting anything."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

SERVER = Path('/root/xunlian/Futaoran/CSI_STREAMING_GPU_FIX_20260908/code/CSI-PAIRS-v2.0-server')
sys.path.insert(0, str(SERVER))
from formal_v2.formal_config import load_formal_config
from formal_v2.formal_evaluation_streaming import _restore_compatibility_probe
from formal_v2.formal_io import sha256_file


def main():
    operator = Path(__file__).parent
    root = SERVER.parents[1] / 'runs/representative-C-1738cae'
    result = json.loads((root / 'representative_probe_result.json').read_text())
    assert result['status'] == 'REPRESENTATIVE_PROBE_COMPLETE'
    assert result['formal_work_unit_complete'] is False and result['sota_ready'] is False
    assert result['training_steps_per_candidate'] == 2000 and result['backbone_unchanged'] is True
    assert result['dataset_sha256'] == '060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac'
    assert sha256_file(root / 'evaluation_origin.json') == result['origin_receipt_sha256']
    assert sha256_file(result['checkpoint_path']) == result['checkpoint_sha256']
    assert sha256_file(root / 'representative_predictions.npy') == result['predictions_sha256']
    probabilities = np.load(root / 'representative_predictions.npy', allow_pickle=False)
    assert probabilities.shape == (result['selection_shape'][0],)
    assert probabilities.dtype == np.float32
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    config = load_formal_config(SERVER / 'formal_v2/configs/formal_v2.json')
    payload = torch.load(result['checkpoint_path'], map_location='cpu', weights_only=True)
    model = _restore_compatibility_probe(payload, int(config['evaluation']['probe_hidden_dim']))
    assert model.family == result['selection']['selected_family']
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all().item() for value in model.state_dict().values())
    events = [json.loads(line) for line in (operator / 'checks/C-representative-formal/stdout.log').read_text().splitlines() if line.startswith('{')]
    completed = [row for row in events if row.get('phase') == 'training_complete']
    assert {row['family'] for row in completed} == {'linear', 'mlp2'} and len(completed) == 2
    assert all(row['step'] == 2000 and row['total_steps'] == 2000 and row['total_rows'] == result['train_shape'][0] and row['device'] == 'cuda:0' for row in completed)
    assert not any(row['phase'][0] == 'accumulating' for row in result['phases'])
    observed = {
        'passed': True, 'acceptance': 'C_ONLY_NOT_D', 'formal_completed_units_added': 0,
        'result_sha256': sha256_file(root / 'representative_probe_result.json'),
        'checkpoint_sha256': result['checkpoint_sha256'], 'prediction_sha256': result['predictions_sha256'],
        'training_completions': completed, 'observed_accumulation': False,
        'train_shape': result['train_shape'], 'selection_shape': result['selection_shape'],
        'metrics': result['selection'], 'timings': result['phases'],
        'training_selection_wall_seconds': result['training_selection_wall_seconds'],
        'host_peak_rss_bytes': result['host_maxrss_bytes'],
        'cuda_peak_allocated_bytes': result['cuda_peak_allocated_bytes'],
        'cuda_peak_reserved_bytes': result['cuda_peak_reserved_bytes'],
        'backbone_unchanged': True, 'sota_ready': False,
    }
    destination = operator / 'C-artifact-validation.json'
    with destination.open('x') as handle:
        json.dump(observed, handle, indent=2)
    print(json.dumps(observed))


if __name__ == '__main__':
    main()
