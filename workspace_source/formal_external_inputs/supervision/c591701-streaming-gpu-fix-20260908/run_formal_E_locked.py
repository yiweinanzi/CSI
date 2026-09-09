"""Read-lock both provenance roots while executing formal E acceptance."""

import argparse
import json
import sys
from pathlib import Path

SERVER = Path('/root/xunlian/Futaoran/CSI_STREAMING_GPU_FIX_20260908/code/CSI-PAIRS-v2.0-server')
sys.path.insert(0, str(SERVER))
from formal_v2.formal_cli import _acquire_legacy_read_lock
from formal_v2.formal_config import load_formal_config
from formal_v2.formal_evaluation_repair import load_validation_corpus
from formal_v2.formal_evaluation_resume import write_atomic_json
from formal_v2.formal_io import read_strict_json, sha256_file
from formal_v2.tools.validate_formal_dual_probe import main as validate


def main():
    parser = argparse.ArgumentParser()
    for name in ('corpus', 'corpus-sha256', 'config', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    corpus_path = Path(args.corpus).resolve()
    if sha256_file(corpus_path) != args.corpus_sha256:
        raise RuntimeError('D corpus manifest changed before reader locking')
    manifest = read_strict_json(corpus_path)
    origin_path = Path(manifest['origin_receipt_path']).resolve()
    if sha256_file(origin_path) != manifest['origin_receipt_sha256']:
        raise RuntimeError('D origin receipt changed before reader locking')
    origin = read_strict_json(origin_path)
    locks = []
    try:
        locks.append(_acquire_legacy_read_lock(origin_path.parent))
        locks.append(_acquire_legacy_read_lock(origin['upstream']['origin_run']))
        validate()
        result_path = Path(args.output).resolve() / 'result.json'
        result = read_strict_json(result_path)
        try:
            load_validation_corpus(corpus_path, load_formal_config(args.config), expected_sha256=args.corpus_sha256)
        except Exception as error:
            result['status'] = 'FAIL_POSTCOMPUTE_INPUT_REAUTHENTICATION'
            result['reauthentication_error'] = type(error).__name__ + ': ' + str(error)
            write_atomic_json(result_path, result)
            raise
        result['input_read_locks_held'] = [str(origin_path.parent), origin['upstream']['origin_run']]
        result['postcompute_input_reauthentication'] = 'PASS'
        write_atomic_json(result_path, result)
        print(json.dumps({'wrapper_status': 'PASS', 'result': str(result_path), 'result_sha256': sha256_file(result_path)}))
    finally:
        for lock in reversed(locks):
            lock.release()


if __name__ == '__main__':
    main()
