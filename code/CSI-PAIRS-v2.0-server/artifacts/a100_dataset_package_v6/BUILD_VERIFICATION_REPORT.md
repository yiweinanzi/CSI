# CSI-PAIRS A100 datasets V2 build and verification report

Date: 2026-08-10 (Asia/Shanghai)

Scope: this report records the independently materialized V2 binary archive.
V2 adds an exact formal split ledger and Chinese split guide while retaining a
single authoritative copy of the CSI payload.

## Delivered archive

| Field | Value |
| --- | --- |
| Local archive | `$WORKSPACE/deliveries/CSI-PAIRS-A100-DATASETS-v2.zip` |
| Outer SHA-256 | `afaad08f5a788e5e6168c3fb59a7e42962fc330d7eb0ec4a1e1284b19648bbc4` |
| Archive bytes | `20,130,276,407` |
| Manifest payload bytes | `20,130,009,120` |
| Manifest files | `455` |
| ZIP members | `456` (`455` payload/control files plus `MANIFEST.json`) |
| Compression | Stored bytes with Zip64 |
| Distribution | Internal research-team transfer only; not committed to Git |

The external sidecar is
`$WORKSPACE/deliveries/CSI-PAIRS-A100-DATASETS-v2.zip.sha256`.
`shasum -a 256 -c` returned `OK`, and `unzip -tq` returned exit code 0.

## Exact component inventory

| Manifest role | Files | Purpose and status |
| --- | ---: | --- |
| `external_public_payload` | 133 | DeepMIMO, UrbanMIMOMap map 0, RadioMapSeer public core, WWM metadata, and DeepSense 8/33 substitute |
| `cpu_llvm22_34bank_candidate` | 266 | M4 CPU/LLVM22 reference candidate; `CANDIDATE_NOT_CLAIM` |
| `cpu_same_engine_verification_evidence` | 12 | Local zero-tolerance replay evidence without `regenerated.npz` |
| `a100_sionna_fixture` | 9 | Real dual-A100 Sionna output; fixture and software-test only |
| `primary_raw_inputs` | 8 | Six frozen OSM responses plus bundle documentation/checksums |
| `workspace_metadata` | 7 | Download, readiness, provenance, and code-license snapshots |
| `package_control_plane` | 20 | Data card, exact formal split ledger, Chinese split guide, policy, registries, verifier, mount and regeneration scripts |

The external groups are DeepMIMO 2 files, UrbanMIMOMap 121 files,
RadioMapSeer 5 files, and WWM/DeepSense 5 files. The package excludes Python
virtual environments, Git object databases, partial downloads, Qualcomm
Wi3R/WiPTR, paid `IRT2HighRes.zip`, unavailable original WWM payloads, and all
verification `regenerated.npz` files.

## Clean-extraction evidence

The ZIP was extracted into a newly created directory outside the source tree.
The extracted `scripts/VERIFY_PACKAGE.sh --deep-hash` result was:

```text
PACKAGE_VERIFY=PASS
MANIFEST_FILES=455
PAYLOAD_BYTES=20130009120
EXTERNAL_REGISTERED_FILES=133
ZIP_CONTAINERS_OPENED=10
NPZ_CONTAINERS_OPENED=138
FORMAL_SPLIT_LEDGER=PASS
SOURCE_PERMISSION_ROLES=7
TARGET_BANKS=16
TARGET_SUPPORT_POSITIONS_PER_BANK=128
TARGET_QUERY_POSITIONS_PER_BANK=128
DEEP_HASH=PASS
TRAINING_CORE_COMPLETE=true
ALL_ORIGINAL_SOURCES_COMPLETE=false
A100_PACKAGE_ENGINEERING_READY=YES
FORMAL_TRAINING_READY=NO
SCIENTIFIC_EVIDENCE=NOT_ASSESSED
```

All 455 manifest entries carry a source SHA-256. The clean-extraction verifier
recomputed every one, compared the exact file inventory and byte sizes, opened
all ZIP/NPZ central directories, checked the six OSM filenames, and rechecked
the candidate, fixture, substitution, license, split, and training-status boundaries.

## Frozen training and test permissions

The V2 archive keeps one authoritative CPU candidate rather than copying its
bytes into separately editable train/test archives. The exact role index is in
`docs/FORMAL_MAIN_SPLIT_LEDGER.json` and its zero-basics Chinese explanation is
in `docs/SPLIT_GUIDE.zh-CN.md`.

- Austin and Chicago each provide one independent bank to each of the seven
  source permissions: encoder train, method selection, probe train, probe
  selection, calibration fit, calibration selection, and final unseen-bank
  test.
- Boston and Seattle provide 16 target banks in total. Every target bank has
  128 even-index `support_pool` positions and 128 odd-index frozen `query`
  positions.
- Denver and Miami provide four `external_validation` banks and cannot affect
  training, normalization, calibration, thresholds, architecture, or checkpoint
  selection.
- The builder and extracted verifier expanded the ledger into all 34 expected
  bank records and matched them exactly against the candidate asset manifest
  and inspected role counts.

## Fresh structural loads

The extracted NPZ files were reloaded with the current
`formal_v2.formal_dataset.FormalDataset` implementation:

| Dataset | Contract | Shape | Target cities | Boundary |
| --- | --- | --- | --- | --- |
| CPU LLVM22 reference | PASS | 34 scenes, 4 worlds, 256 positions, 3 repeats, 16 channels | Boston, Seattle | `fixture=false`, `scientific_use=CANDIDATE` |
| Dual-A100 Sionna fixture | PASS | 36 scenes, 4 worlds, 30 positions, 3 repeats, 16 channels | fixture target-a/target-b | `fixture=true`, `scientific_use=FORBIDDEN` |

The CPU dataset SHA-256 is
`e89034302d6c4a64c838c244a60ad7c42b081e760ef2c9fc548440b0d5458d2e`.
The A100 fixture SHA-256 is
`ae3445739fa3415f7676f544bab28b102459d9819c40028b5ed7e23f7cd5cead`.

A host-specific `formal_cli inspect-data` attempt correctly failed closed
because the pre-existing local formal Python environment contained forbidden
bytecode cache files. No runtime files were deleted or altered. The direct
current-source loader check above is an additional structural check, not a
formal qualification result.

## Repository checks

- `formal_v2.tests.test_a100_dataset_package_v6`: 9/9 PASS.
- Package builder preflight: PASS, 455 files, 18.748 GiB payload.
- Delivered V2 archive manifest: 455/455 internally exact PASS; every entry
  has a SHA-256. After archive verification, the source tree normalized one
  trailing blank line in `external_dataset_registry/MOUNTING.md` and
  `scripts/A100_REGENERATE_AND_VERIFY.sh`. This whitespace-only source delta
  does not change behavior, but a later rebuild will have a different outer
  ZIP hash.
- Python syntax, Bash syntax, JSON parsing, and `git diff --check`: PASS.
- Binary ZIP and dataset payloads remain outside Git and outside the PR.

## Scientific boundary

This report proves package and data-contract engineering integrity only. It
does not prove measured-channel realism, independent-engine agreement,
leakage-free model evaluation, trained-model performance, or a paper claim.

The destination A100 host must still pre-approve Linux x86_64 libLLVM in a
separate reviewed commit, generate a fresh candidate from the six OSM inputs,
and pass fresh all-role zero-tolerance inspection and verification. Independent
RT calibration/G8, G1/G2, adapter, resource/license, qualification, and explicit
human-approval gates remain open. Therefore:

```text
FORMAL_INPUT_READY=BLOCKED
FORMAL_TRAINING_READY=NO
LAUNCH_READY=BLOCKED
SCIENTIFIC_EVIDENCE=NOT_ASSESSED
```
