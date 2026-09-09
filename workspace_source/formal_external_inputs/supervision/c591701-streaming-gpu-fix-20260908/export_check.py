"""Export a completed operator check with a generated hash manifest."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.label):
        raise ValueError('Invalid check label')
    source = Path(__file__).parent / 'checks' / args.label
    record = json.loads((source / 'record.json').read_text())
    dest = Path('/root/xunlian/Futaoran/CSI_STREAMING_GPU_REPAIR_EVIDENCE_20260908/artifacts/streaming_gpu_repair_20260908') / args.label
    dest.mkdir(exist_ok=False)
    inventory = {}
    for before in source.iterdir():
        if not before.is_file():
            continue
        after = dest / before.name
        shutil.copyfile(before, after)
        inventory[after.name] = hashlib.sha256(after.read_bytes()).hexdigest()
    manifest = {'check': args.label, 'returncode': record['returncode'], 'git_head_at_check': record['git_head'], 'tracked_diff_sha256': record['tracked_diff_sha256'], 'source': str(source), 'files_sha256': inventory, 'note': 'A completed command is not necessarily a passed acceptance; inspect exit code and scientific scope.'}
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'exported': str(dest), 'files': len(inventory)}))


if __name__ == '__main__':
    main()
