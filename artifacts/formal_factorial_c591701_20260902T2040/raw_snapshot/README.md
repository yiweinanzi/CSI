# Complete raw run archive

The ordered `snapshot.tar.zst.part-*` files are stored using Git LFS. Their
hashes, lengths and concatenated-stream hash are in `archive_manifest.json`.
The archive preserves every file from the training worktree's `runs/`,
`inspection_v6_20260901T/` and `operator_preflight_20260901T/` directories at the
snapshot time, including failed attempts, recovery slots and original lock
records. These are archival records, not live locks or an approved resumed run.

Files with identical bytes share one `objects/<sha256>` entry in the archive.
`source_inventory.json` records every original relative path and its exact
size and hash. No samples or training cells are removed by this storage scheme.

From this directory, with Git LFS, zstd and Python 3.11 or newer installed:

```bash
git lfs pull
(cd .. && sha256sum --check SHA256SUMS)
mkdir /path/to/empty/object-store
cat snapshot.tar.zst.part-* | zstd -d | tar -xf - -C /path/to/empty/object-store
python3 -B restore_snapshot.py /path/to/empty/object-store /path/to/new/snapshot
```

Run the full `sha256sum --check SHA256SUMS` from the parent artifact directory
to verify every published file. `restore_snapshot.py` checks all archived bytes
before recreating independent files and refuses to overwrite a destination.

The authoritative training NPZ lives outside the source worktree and is not
duplicated here; its SHA is recorded in the parent snapshot. Regenerated NPZ
files produced by the LIVE verifier are retained exactly as they were. Local
virtual environments and runtime symlinks are outside this archive.

This snapshot remains `CANDIDATE_NOT_CLAIM`, with `sota_ready=false`. Uploading
the complete existing files does not complete localization, C1 or claims.
