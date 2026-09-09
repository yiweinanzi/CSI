import hashlib
import json
import shutil
from pathlib import Path


def main():
    source = Path('/root/xunlian/Futaoran/CSI_STREAMING_GPU_FIX_20260908/runs/representative-C-1738cae')
    dest = Path('/root/xunlian/Futaoran/CSI_STREAMING_GPU_REPAIR_EVIDENCE_20260908/artifacts/streaming_gpu_repair_20260908/C-formal-probe-artifacts')
    validation = json.loads((Path(__file__).parent / 'C-artifact-validation.json').read_text())
    assert validation['passed'] and validation['formal_completed_units_added'] == 0
    dest.mkdir(exist_ok=False)
    inventory = {}
    for name in ('evaluation_origin.json', 'representative_compatibility_probe.pt', 'representative_predictions.npy', 'representative_probe_result.json', 'representative_progress.json'):
        after = dest / name
        shutil.copyfile(source / name, after)
        inventory[name] = hashlib.sha256(after.read_bytes()).hexdigest()
    manifest = {'acceptance': 'C_PASS_NOT_D', 'scientific_use': 'NON_CLAIM', 'sota_ready': False, 'formal_completed_units_added': 0, 'files_sha256': inventory, 'validation': validation}
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(dest), 'files': len(inventory)}))


if __name__ == '__main__':
    main()
