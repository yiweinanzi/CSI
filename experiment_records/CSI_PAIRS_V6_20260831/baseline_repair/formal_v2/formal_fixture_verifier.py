from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from formal_v2.formal_data_verification import REGENERATED_FIELDS
from formal_v2.formal_io import parse_strict_json


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    source = Path(args.dataset)
    target = Path(args.output) / "regenerated.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as archive:
        metadata = parse_strict_json(str(np.asarray(archive["metadata_json"]).item()))
        if metadata.get("fixture") is not True or metadata.get("scientific_use") != "FORBIDDEN":
            raise RuntimeError("copy verifier is restricted to permanently forbidden fixtures")
        missing = REGENERATED_FIELDS.difference(archive.files)
        if missing:
            raise RuntimeError(f"fixture is missing regenerated fields: {sorted(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in REGENERATED_FIELDS}
    np.savez_compressed(target, **arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
