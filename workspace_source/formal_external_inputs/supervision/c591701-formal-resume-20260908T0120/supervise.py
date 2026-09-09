import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/root/xunlian/Futaoran/CSI_MAIN_C591701_20260901')
SERVER = ROOT / 'code/CSI-PAIRS-v2.0-server'
RUN = ROOT / 'runs/c591701-formal-prepare-20260902T2040'
HERE = Path(__file__).resolve().parent
PYTHON = Path('/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python')
DATASET = Path('/root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz')
COMMIT = 'c5917018610b262f5263ccaeba2f087b2eb96a96'
APPROVAL = Path('/root/xunlian/Futaoran/formal_external_inputs/run_approvals/c591701-formal-prepare-20260902T2040/LLM_JUDGE_APPROVAL.json')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_status(path):
    if not path.is_file():
        return {'exists': False}
    try:
        value = json.loads(path.read_text())
        return {'exists': True, **{key: value[key] for key in ('status', 'passed', 'sota_ready', 'scientific_use', 'completed', 'total', 'progress', 'phase') if key in value}}
    except (OSError, ValueError) as error:
        return {'exists': True, 'read_warning': str(error)}


def main():
    guard = (HERE / 'supervisor.lock').open('a+')
    fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (HERE / 'launch.json').exists(), 'This supervision attempt has already launched'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT, text=True).strip()
    assert sha(DATASET) == '060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac'
    index = json.loads((RUN / 'factorial/checkpoint_index.json').read_text())
    assert len(index['checkpoints']) == 12
    records = []
    for row in index['checkpoints']:
        checkpoint = RUN / 'factorial' / row['path']
        assert sha(checkpoint) == row['sha256']
        pointer = RUN / 'factorial/resume' / f"seed_{row['seed']}" / (row['arm'] + '.latest.json')
        value = json.loads(pointer.read_text())
        assert value['status'] == 'COMPLETE' and value['completed_steps'] == 20000
        assert sha(pointer.parent / value['checkpoint_path']) == value['checkpoint_sha256']
        records.append({'seed': row['seed'], 'arm': row['arm'], 'sha256': row['sha256'], 'completed_steps': 20000})
    old_lock = RUN.parent / ('.' + RUN.name + '.csi-pairs-operation.lock')
    if old_lock.exists():
        owner = json.loads(old_lock.read_text())
        assert not Path('/proc/' + str(owner['pid'])).exists(), 'Previous writer PID is still present'
        shutil.copy2(old_lock, HERE / 'previous_operation_lock.json')
    shutil.copytree(RUN / 'factorial', HERE / 'before_resume_factorial')
    shutil.copytree(RUN / 'approval', HERE / 'before_resume_approval')
    write_json(HERE / 'before_resume.json', {'utc': now(), 'commit': COMMIT, 'dataset_sha256': sha(DATASET), 'checkpoints': records})
    env = os.environ.copy()
    env.update({
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONUNBUFFERED': '1',
        'CUDA_VISIBLE_DEVICES': '0,1',
        'CSI_PAIRS_DEVICES': 'cuda:0,cuda:1',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'OMP_NUM_THREADS': '4',
        'MKL_NUM_THREADS': '4',
        'OPENBLAS_NUM_THREADS': '4',
    })
    base = [str(PYTHON), '-B', '-m', 'formal_v2.formal_cli']
    preflight = base + ['operator-preflight', '--config', 'formal_v2/configs/formal_v2.json', '--dataset', str(DATASET), '--output', str(HERE / 'operator_preflight')]
    with (HERE / 'operator_preflight.log').open('wb') as log:
        result = subprocess.run(preflight, cwd=SERVER, env=env, stdout=log, stderr=subprocess.STDOUT)
    write_json(HERE / 'operator_preflight_exit.json', {'utc': now(), 'returncode': result.returncode})
    command = base + [
        'all', '--config', 'formal_v2/configs/formal_v2.json',
        '--dataset', str(DATASET), '--output', str(RUN),
        '--adapter-manifest', 'formal_v2/external_adapters/all_map_adapters_v1.json',
        '--verifier-manifest', 'formal_v2/configs/sionna_osm_verifier_v2.json',
        '--approval-manifest', str(APPROVAL), '--continue-on-stage-fail',
    ]
    with (HERE / 'all.log').open('wb') as log:
        child = subprocess.Popen(command, cwd=SERVER, env=env, stdout=log, stderr=subprocess.STDOUT)
        launch = {'utc': now(), 'supervisor_pid': os.getpid(), 'pid': child.pid, 'cwd': str(SERVER), 'command': command, 'environment': {k: env[k] for k in ('PYTHONDONTWRITEBYTECODE', 'CUDA_VISIBLE_DEVICES', 'CSI_PAIRS_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}, 'training_steps_already_complete': 240000}
        write_json(HERE / 'launch.json', launch)
        last_report = 0.0
        stages = {
            'data_verification': 'data_verification/gate.json',
            'qualification': 'qualification/gate.json',
            'approval': 'approval/accepted.json',
            'wrong_map': 'wrong_map/status.json',
            'factorial': 'factorial/gate.json',
            'evaluation': 'evaluation/gate.json',
            'risk': 'risk/gate.json',
            'path': 'path/gate.json',
            'external_baselines': 'external_baselines/gate.json',
            'representation_baselines': 'representation_baselines/gate.json',
            'controls': 'controls/gate.json',
            'scene_id': 'scene_id/gate.json',
            'shuffled_pair': 'controls/shuffled_pair/gate.json',
            'retention': 'evaluation/retention/gate.json',
            'claims': 'claims/claim_evidence.json',
        }
        while True:
            returncode = child.poll()
            try:
                gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
            except (OSError, subprocess.SubprocessError) as error:
                gpu = 'WARNING: ' + str(error)
            status = {'utc': now(), 'pid': child.pid, 'process_state': 'RUNNING' if returncode is None else 'EXITED', 'returncode': returncode, 'gpu': gpu, 'stages': {name: read_status(RUN / rel) for name, rel in stages.items()}, 'stage_failures': read_status(RUN / 'stage_failures.json'), 'scientific_use': 'CANDIDATE_NOT_CLAIM'}
            write_json(HERE / 'progress.json', status)
            if time.monotonic() - last_report >= 300 or returncode is not None:
                with (HERE / 'progress_5min.jsonl').open('a') as handle:
                    handle.write(json.dumps(status) + '\n')
                last_report = time.monotonic()
            if returncode is not None:
                write_json(HERE / 'exit.json', status)
                break
            time.sleep(30)
    return returncode


if __name__ == '__main__':
    raise SystemExit(main())
