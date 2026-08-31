# CSI-PAIRS anonymous supplementary

This package contains the formal implementation, tests, frozen configurations, data contract,
paper source, and official conference style files. It contains no formal dataset, experiment result,
checkpoint, internal repository provenance, or author identity.

Third-party inputs are not bundled. Their immutable versions, source URLs, licenses, and SHA-256
digests are recorded in `formal_v2/configs/waibu_resources_v1.json`. Resources whose recorded
license does not grant downstream redistribution must be obtained by each user from the source URL.
Local byte authentication does not establish paper fidelity or scientific evidence.

Fetch missing inputs directly from those sources and authenticate the complete local set:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m formal_v2.fetch_waibu_resources \
  --registry formal_v2/configs/waibu_resources_v1.json \
  --waibu-root waibu
PYTHONDONTWRITEBYTECODE=1 python3 -m formal_v2.formal_cli verify-waibu-resources \
  --registry formal_v2/configs/waibu_resources_v1.json \
  --waibu-root waibu \
  --output runs/resource-auth-001
```

The fetcher refuses to overwrite mismatched files and removes a partial download unless its
SHA-256 matches the frozen registry. Local downloads must not be committed or redistributed.

Create the hash-locked Python 3.12 environment, then run the code tests:

```bash
formal_v2/scripts/setup_formal_v2.sh "$PWD/.venv"
PYTHONDONTWRITEBYTECODE=1 "$PWD/.venv/bin/python" -m unittest discover -s formal_v2/tests -v
```

Setup retains the exact lock-authenticated wheels, installs them offline with `--no-compile`, removes
all bytecode, and writes a wheel-derived closure manifest plus an informational pip report. Every
evidence context rehashes the retained wheels and the full installed-file closure. Rewriting a pip
report or installed `RECORD` cannot authorize modified files; extra distributions, files, startup
hooks, symlinks, and any `.pyc` are rejected along with unsupported platforms or nondeterministic
PyTorch state.

The optional fixture dry run is a fail-closed software check:

```bash
CSI_PAIRS_PYTHON=python3 \
  formal_v2/scripts/run_formal_v2_dry_run.sh runs/dry-run-001
```

The inner qualification must exit `1` and emit an authenticated `DRY_RUN_FAIL_NOT_EVIDENCE` gate
with `passed=false`, `fixture=true`, and `scientific_use=FORBIDDEN`. The wrapper exits `0` only after
verifying that exact state and its complete root manifest; this does not mean that qualification or
any scientific gate passed.

Fixtures are permanently marked `scientific_use=FORBIDDEN`. Formal claims remain blocked until the
non-fixture data, independent regeneration, calibration, external-validity, and statistical gates
complete under the frozen protocol.
