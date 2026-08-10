# CSI-PAIRS A100 datasets V2

This is an internal Linux/A100 data handoff. It deliberately keeps five data
roles separate. Do not concatenate the directories into one training table.

```text
PACKAGE_INTEGRITY=VERIFY_AFTER_EXTRACTION
TRAINING_CORE_COMPLETE=true
ALL_ORIGINAL_SOURCES_COMPLETE=false
FORMAL_TRAINING_READY=NO
SCIENTIFIC_EVIDENCE=NOT_ASSESSED
```

## Included data

| Directory | Contents | Scientific role |
| --- | --- | --- |
| `external_public_datasets/` | DeepMIMO, UrbanMIMOMap map 0, RadioMapSeer public core, WWM metadata, and DeepSense 8/33 | Named baselines and preregistered external-domain work only |
| `primary_raw_inputs/` | Frozen six-city OpenStreetMap responses | Inputs for fresh A100 candidate generation; not CSI samples |
| `a100_sionna_fixture/` | Real dual-A100 Sionna RT output | `fixture=true`; software and GPU smoke tests only |
| `cpu_llvm22_34bank_candidate/` | CPU LLVM22 34-bank Boston/Seattle CSI-PAIRS candidate | `CANDIDATE_NOT_CLAIM`; diagnostic/reference use only |
| `cpu_same_engine_verification_evidence/` | Zero-tolerance same-engine replay records | Engineering evidence, not independent scientific validation |

Original China Mobile WWM payloads are not public and are not included.
DeepSense Scenarios 8 and 33 are an explicit, non-equivalent public substitute.
`RadioMap3DSeer.zip` is not the paid `IRT2HighRes.zip` source.

## Frozen train, selection, calibration, and test permissions

Read `docs/SPLIT_GUIDE.zh-CN.md` for a zero-basics explanation and
`docs/FORMAL_MAIN_SPLIT_LEDGER.json` for the machine-readable source of truth.
The CSI bytes remain in one NPZ. Select rows by `scene_roles` and
`position_roles`; never make separately edited train/test NPZ copies.

Austin and Chicago each contribute one bank to each of the seven mutually
exclusive source roles. Boston and Seattle are target cities: even position
indices are the 128-position `support_pool`, while odd indices are the 128
frozen `query` positions in every target bank. Denver and Miami are external
validation only. All sibling worlds, edges, repeats, masks, and queries follow
their highest-level bank or position permission.

## Start on the A100 host

From this extracted directory:

```bash
./START_HERE.sh
source ./scripts/MOUNT_DATASETS.sh
```

The first command runs the quick package verifier and reports visible NVIDIA
GPUs. The mount script exports separate roots for each dataset family. It does
not start training or alter an environment.

The large public sources remain in their upstream ZIP/NPZ form to keep this
handoff compact and fast to transfer. Extract a source into a new work
directory only when its named baseline needs it. Never fit preprocessing,
normalization, routing, or checkpoint selection on a held-out group.

To create a new CSI-PAIRS candidate, use a separate extracted server bundle
and fresh output directories:

```bash
./scripts/A100_REGENERATE_AND_VERIFY.sh \
  /absolute/path/CSI-PAIRS-v2.1-server \
  /new/path/a100-candidate \
  /new/path/a100-inspection \
  /new/path/a100-verification
```

This command intentionally stops before formal qualification or training.

## Verification levels

The normal verifier checks the exact manifest, byte sizes, critical JSON
status boundaries, NPZ readability, and ZIP central directories:

```bash
./scripts/VERIFY_PACKAGE.sh
```

Full per-file SHA-256 is optional because the package already ships with the
133 upstream checksums and the outer ZIP has one transfer checksum:

```bash
./scripts/VERIFY_PACKAGE.sh --deep-hash
```

Neither mode proves physical channel realism, leakage freedom, independent RT
agreement, model quality, or any paper result.

## Formal experiment boundary

This ZIP is A100-operational data, but it is not permission to start the formal
paper run. A formal CSI-PAIRS run still requires a fresh candidate generated
on the reviewed Linux runtime, independent RT calibration/G8 evidence, G1/G2
qualification, compute and license gates, an actual `qualification/gate.json`
with `QUALIFICATION_PASS` and `FORMAL_EXPERIMENT_ALLOWED`, and explicit human
approval of that exact run.
