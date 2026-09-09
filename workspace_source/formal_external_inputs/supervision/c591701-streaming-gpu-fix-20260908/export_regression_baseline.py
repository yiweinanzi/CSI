import hashlib
import json
import shutil
from pathlib import Path


def main():
    source = Path(__file__).parent / 'checks'
    dest = Path('/root/xunlian/Futaoran/CSI_STREAMING_GPU_REPAIR_EVIDENCE_20260908/artifacts/streaming_gpu_repair_20260908/streaming-regression-baseline')
    dest.mkdir(exist_ok=False)
    failures = {}
    inventory = {}
    for label in ('streaming-regression-original', 'streaming-regression-repair'):
        log = (source / label / 'stdout.log').read_text()
        failures[label] = sorted(line for line in log.splitlines() if line.startswith('FAILED '))
        assert len(failures[label]) == 10
        assert json.loads((source / label / 'record.json').read_text())['returncode'] == 1
        (dest / label).mkdir()
        for name in ('record.json', 'stdout.log', 'stderr.log', 'tracked.patch'):
            target = dest / label / name
            shutil.copyfile(source / label / name, target)
            inventory[str(target.relative_to(dest))] = hashlib.sha256(target.read_bytes()).hexdigest()
    assert failures['streaming-regression-original'] == failures['streaming-regression-repair']
    result = {'original_and_repair_failed_tests_identical': True, 'failed_tests': failures['streaming-regression-original'], 'whole_module_passed': False, 'per_version': {'failed': 10, 'passed': 41, 'skipped': 1}, 'scope': 'Preexisting regression failures preserved, not bypassed in production; this comparison is not an all-green acceptance.', 'files_sha256': inventory}
    (dest / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(dest), 'comparison': 'identical_failure_set'}))


if __name__ == '__main__':
    main()
