"""Export immutable, scoped execution evidence; never export credentials."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    args = parser.parse_args()
    source = Path(__file__).parent
    repo = Path('/root/xunlian/Futaoran/CSI_STREAMING_GPU_REPAIR_EVIDENCE_20260908')
    dest = repo / 'artifacts/streaming_gpu_repair_20260908' / args.snapshot
    dest.mkdir(parents=True, exist_ok=False)
    names = ['old_scene/scene.json', 'old_task_stop.json', 'validate_binary_execution.py']
    for directory in ('A-auroc', 'B-numerical', 'B-binary-numerical'):
        names.extend(str(p.relative_to(source)) for p in (source / directory).iterdir() if p.is_file())
    for label in ('A-auroc', 'B-numerical', 'B-binary-numerical', 'regression-final', 'origin-authentication-v2'):
        names.extend('checks/' + label + '/' + name for name in ('record.json', 'stdout.log', 'stderr.log', 'tracked.patch'))
    inventory = {}
    for name in sorted(names):
        before = source / name
        after = dest / name
        after.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(before, after)
        digest = hashlib.sha256(after.read_bytes()).hexdigest()
        assert digest == hashlib.sha256(before.read_bytes()).hexdigest()
        inventory[name] = {'sha256': digest, 'size_bytes': after.stat().st_size}
    result = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'evaluation_code_commit': '1738caea131b9ed0d28994c89d17c3730586d323',
        'evaluation_source_tree_sha256': '98572664677378e50baead78828671d0cc92cc6e580809cb57b8e4c3ad8785eb',
        'upstream_training_commit': 'c5917018610b262f5263ccaeba2f087b2eb96a96',
        'dataset_sha256': '060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac',
        'scientific_use': 'NON_CLAIM', 'sota_ready': False,
        'A': 'PASS: fixed-score AUROC correctness and function timing',
        'B': 'PASS: synthetic shared-initialization CUDA numerical checks only',
        'C': 'NOT_INCLUDED: running separately; no completed formal probe in this snapshot',
        'D': 'NOT_STARTED', 'E': 'NOT_VERIFIED_ON_FORMAL_INPUT', 'F': 'NOT_STARTED',
        'original_meter_results': 'PRESERVED: Full did not beat Response-only in eight recorded cells; G5 FAIL unchanged',
        'files': inventory,
    }
    (dest / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(dest), 'files': len(inventory)}))


if __name__ == '__main__':
    main()
