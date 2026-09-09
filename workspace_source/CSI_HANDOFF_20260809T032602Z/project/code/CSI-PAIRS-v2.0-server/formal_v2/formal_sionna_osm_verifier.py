from __future__ import annotations

import argparse
import sys

from formal_v2.sionna_osm_candidate import ensure_sionna_runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independently regenerate the frozen Sionna/OSM formal candidate"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if argv is not None:
        raise RuntimeError("the authenticated verifier requires process argv")
    ensure_sionna_runtime([__file__, *sys.argv[1:]])
    from formal_v2.sionna_osm_candidate import regenerate_dataset

    regenerate_dataset(args.dataset, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
