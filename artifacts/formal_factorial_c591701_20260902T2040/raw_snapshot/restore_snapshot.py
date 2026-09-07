"""Verify archived objects and restore original paths into a new directory."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object_store', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    records = json.loads((args.object_store / 'source_inventory.json').read_text())
    expected_inventory = Path(__file__).resolve().parent.parent / 'source_inventory.json'
    if records != json.loads(expected_inventory.read_text()):
        raise ValueError('Archive inventory does not match the published snapshot')
    checked = set()
    for record in records:
        rel = PurePosixPath(record['path'])
        if rel.is_absolute() or '..' in rel.parts or not rel.parts:
            raise ValueError('Invalid archive path')
        digest = record['sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid SHA-256')
        obj = args.object_store / 'objects' / digest
        if obj.is_symlink() or obj.stat().st_size != record['bytes']:
            raise ValueError('Invalid archived object: ' + digest)
        if digest not in checked:
            with obj.open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != digest:
                    raise ValueError('Corrupt archived object: ' + digest)
            checked.add(digest)
    args.destination.mkdir(parents=True, exist_ok=False)
    for record in records:
        target = args.destination / record['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        with (args.object_store / 'objects' / record['sha256']).open('rb') as source, target.open('xb') as output:
            shutil.copyfileobj(source, output)
    print(f"Restored {len(records)} files; verified {len(checked)} unique SHA-256 objects.")


if __name__ == '__main__':
    main()
