# CSI-PAIRS A100 datasets V2 verification status

Date: 2026-08-10

## Current binary status

No repository-verifiable V2 binary archive is certified by this source tree.
The approximately 20 GB ZIP is intentionally not committed, and the previously
reported archive size and SHA-256 came from a build whose control-plane files
were subsequently changed. Those values are stale and must not be used as a
trust anchor for the current revision.

```text
CURRENT_ARCHIVE_SHA256=NOT_FROZEN_REBUILD_REQUIRED
CURRENT_ARCHIVE_BYTES=NOT_FROZEN_REBUILD_REQUIRED
CURRENT_CLEAN_EXTRACTION=NOT_RUN_FOR_CURRENT_REVISION
FORMAL_TRAINING_READY=NO
SCIENTIFIC_EVIDENCE=NOT_ASSESSED
```

## Required rebuild acceptance

A replacement archive is accepted only after all of the following complete on
the exact reviewed source revision:

1. Builder preflight passes with the fixed component file counts and required
   candidate/fixture NPZ hashes.
2. The deterministic Zip64 build completes without a source size or SHA-256
   changing between planning and archive write.
3. The outer ZIP SHA-256 sidecar is generated and independently retained as the
   transfer trust anchor.
4. The archive is extracted into a new directory and the normal verifier hashes
   every manifest entry, rejects symlinks and extra files, opens every ZIP/NPZ
   container, and reports `PACKAGE_VERIFY=PASS` and `DEEP_HASH=PASS`.
5. The reported archive bytes, outer SHA-256, manifest payload bytes, file count,
   source revision, and clean-extraction log are recorded together without any
   later source normalization.

## Scientific boundary

Package verification establishes engineering integrity only. External baseline
paper use remains blocked until dataset-level license review and an immutable,
machine-readable split ledger assign every scenario/map/transmitter/
configuration/sequence/location group to exactly one partition and pass an
overlap audit. Fresh independent RT, G1/G2, resource, adapter, G8,
qualification, and explicit llm-judge approval gates also remain required.
