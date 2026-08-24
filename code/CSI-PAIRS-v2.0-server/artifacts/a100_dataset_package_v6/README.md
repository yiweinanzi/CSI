# CSI-PAIRS A100 datasets V2 delivery control plane

This directory defines the internal A100 data ZIP built outside Git. The ZIP
combines the public external-wireless core, frozen six-city OSM inputs, the
CPU LLVM22 34-bank reference candidate, and the dual-A100 Sionna fixture while
keeping their scientific roles separate.

The binary ZIP and dataset payloads are deliberately not committed to Git.
This directory is the reviewable control plane: policy, data card, registries,
build code, verification code, and the destination-host regeneration entry
point.

V2 adds an exact formal split ledger and a Chinese zero-basics guide. It keeps
one authoritative dataset payload and selects permissions through
`scene_roles` and `position_roles`; it does not create drifting train/test NPZ
copies.

```text
PACKAGE_INTEGRITY=TO_BE_BUILT_AND_VERIFIED
FORMAL_INPUT_READY=BLOCKED
FORMAL_TRAINING_READY=NO
SCIENTIFIC_EVIDENCE=NOT_ASSESSED
```

## Package contents

| Package path | Contents | Permitted use |
| --- | --- | --- |
| `primary_raw_inputs/` | Six frozen OpenStreetMap JSON responses | Regenerate a new 34-bank candidate under a pre-approved Linux runtime |
| `external_public_datasets/` | DeepMIMO, UrbanMIMOMap, RadioMapSeer, and DeepSense public data | Adapter engineering only until license review and an immutable split ledger pass |
| `cpu_llvm22_34bank_candidate/` | M4 CPU/LLVM22 34-bank candidate, assets, shards, logs, and inspection | `CANDIDATE_NOT_CLAIM` transfer diagnostic and reference only |
| `cpu_same_engine_verification_evidence/` | Same-host, same-engine zero-tolerance evidence without `regenerated.npz` | Internal consistency evidence only |
| `a100_sionna_fixture/` | Real dual-A100 Sionna RT fixture | Software, loader, and multi-GPU smoke tests only |
| `external_dataset_registry/` | Source identity, substitutions, roles, and mount policy | Review and audit |
| `scripts/` | Package verification and fail-closed A100 regeneration | Operational entry points |

The CPU reference contains the Boston and Seattle target partitions inside one
six-city archive. It is not split into independent city ZIPs. Its city-bank
counts are Austin 7, Chicago 7, Boston 8, Seattle 8, Denver 2, and Miami 2. Its
dataset SHA-256 is
`e89034302d6c4a64c838c244a60ad7c42b081e760ef2c9fc548440b0d5458d2e`.
The exact per-bank permissions and target support/query rule are frozen in
`FORMAL_MAIN_SPLIT_LEDGER.json` and explained in `SPLIT_GUIDE.zh-CN.md`.

## Build outside Git

Run from this repository checkout, pointing at the workspace that owns the
local datasets:

```bash
python3 artifacts/a100_dataset_package_v6/build_a100_dataset_package_v6.py \
  --workspace-root /absolute/path/to/ICLR2027 \
  --output /absolute/path/to/deliveries/CSI-PAIRS-A100-DATASETS-v2.zip
```

The builder refuses to overwrite either the ZIP or its external `.sha256`
record. It creates one ASCII top-level directory and uses Zip64 with stored
payload bytes for fast, lossless packaging of already-compressed datasets.

## Verify after clean extraction

```bash
unzip -q CSI-PAIRS-A100-DATASETS-v2.zip -d /new/empty/directory
cd /new/empty/directory/CSI-PAIRS-A100-DATASETS-v2
./scripts/VERIFY_PACKAGE.sh
```

The verifier recomputes all 455 source hashes, validates the exact manifest and
byte sizes, rejects symlinks and extra files, opens every ZIP/NPZ central
directory, and checks the scientific-use labels. `--deep-hash` is retained only
as a compatibility spelling; hashing cannot be skipped. Verification does not
prove leakage control, physical validity, independent-engine agreement,
model performance, or any paper claim.

## A100 progression

Run `./START_HERE.sh`, read `docs/DATASET_CARD.md`, then run
`scripts/A100_REGENERATE_AND_VERIFY.sh` only
from the extracted package on a reviewed Linux two-A100 host. The repository
currently has no pre-approved Linux x86_64 libLLVM entry, so the script must
fail closed until that destination library is reviewed in a separate commit.
Even a successful new generation and zero-tolerance verification leaves
`FORMAL_TRAINING_READY=NO`; formal G1/G2 qualification, independent RT/G8,
resource checks, and explicit llm-judge approval remain separate gates.
