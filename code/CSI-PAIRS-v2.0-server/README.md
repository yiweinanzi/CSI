# CSI-PAIRS V2.1 V6 server bundle

Status: formal code `CODE_READY_FOR_FORMAL_INPUT`;
`M4_DATA_PRODUCTION_READY=REPORTED`;
`EVIDENCE_REGISTRY_READY=YES`;
`FORMAL_CANDIDATE_READY=EXTERNAL_DEEP_VERIFICATION_REQUIRED`;
`FORMAL_INPUT_READY=BLOCKED`;
`FORMAL_TRAINING_READY=NO`;
`LAUNCH_READY=BLOCKED`; and scientific evidence `NOT_ASSESSED`.
Archived V1 fixture failures remain non-scientific history.

This internal research-delivery bundle is self-contained for the V2.1 code runtime. It is not an
anonymous ICLR supplementary artifact because it includes delivery provenance. Third-party files
without a downstream redistribution grant are never bundled; each user fetches them directly from
the immutable source URL before formal preflight. Use `formal_v2/scripts/build_anonymous_supplement.sh`
for the separate identity-scanned package that excludes all of `waibu/` and delivery audits. The
internal bundle intentionally contains no formal dataset, external model checkpoint, licensed scene
asset, or claimed result. The source repository contains identity-scrubbed candidate evidence under
`artifacts/m4_formal_candidate_v2/` and `artifacts/m4_llvm22_candidate_v1/`; the current portable candidate bytes remain a separate transfer. The top-level directory and three audit-listed
artifact filenames retain
`v2.0`/`v2_0` only as compatibility paths; their contents, schemas, runtime version, and generated
bundle root are V2.1.

The collaborator's GitHub tree is the sole implementation baseline. Historical local ICLR2027
code is not imported, copied, or required by this bundle.

## 1. Server requirements

- glibc 2.28+ Linux x86_64 or macOS 14+ arm64 with Python 3.12 and `venv` support;
- enough disk space for PyTorch and your formal data;
- CUDA is optional for the non-scientific dry run; non-fixture execution requires Linux CUDA;
- `sha256sum` for bundle verification.

For the reviewed two-A100 transfer layout, frozen raw OSM cache, deterministic
203-bank V4 data generation, and exact server commands, use
`formal_v2/A100_RUNBOOK.md`.

## 2. Verify and install

From the extracted `CSI-PAIRS-v2.1-server` directory:

```bash
sha256sum --check SHA256SUMS
formal_v2/scripts/setup_formal_v2.sh "$PWD/.venv"
```

The setup script refuses to overwrite an existing environment directory or run on an unsupported
platform. It selects `requirements-lock.txt` on macOS arm64 and
`requirements-lock-linux-x86_64-cu121.txt` on Linux x86_64. The Linux lock contains
PyTorch 2.5.1+cu121 and its complete CUDA 12.1/Triton dependency closure. Installation uses pip hash mode
and binary-only mode, so an unlisted transitive dependency, source archive, or changed wheel fails.
On macOS, the script uses the system `/etc/ssl/cert.pem` when present; set
`CSI_PAIRS_PIP_CERT` to a different regular CA bundle when required. TLS verification is never
disabled.
Every evidence-producing command then revalidates the main runtime before writing evidence:
CPython must be 3.12, every package named in the selected platform lock must have the exact
pinned version, the read-only pip installation report must bind every selected wheel SHA-256 to the
active platform allowlist, and every hashed installed file must still match its distribution
`RECORD` entry. Only generated `__pycache__` rows and `RECORD` itself may be unhashed.
The runtime record also binds the interpreter, supported OS/libc floor, source tree, lock/report
digests, CUDA inventory, CUBLAS workspace, TF32 controls, and deterministic PyTorch state. A
self-consistent same-version environment installed from an unreviewed wheel is rejected rather
than merely recorded.

## 3. Run code-only verification

Use a new output path:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PWD/.venv/bin/python" -m formal_v2.scripts.check_python_syntax formal_v2
PYTHONDONTWRITEBYTECODE=1 "$PWD/.venv/bin/python" -m unittest discover -s formal_v2/tests -v
PYTHONDONTWRITEBYTECODE=1 "$PWD/.venv/bin/python" -m pytest -p no:cacheprovider formal_v2/tests -v
```

Expected software outcome:

- all pure schema and semantic tests pass;
- no teacher, qualification, four-arm, localization, RT, or external-model experiment is run;
- no scientific gate changes state.

`formal_v2/scripts/verify_server_bundle.sh` also runs a non-scientific fixture dry run. Its inner
qualification must exit `1` with authenticated status `DRY_RUN_FAIL_NOT_EVIDENCE`, `passed=false`,
`fixture=true`, and `scientific_use=FORBIDDEN`. The wrapper returns `0` only after reauthenticating
that exact expected fail-closed gate and the complete root manifest. A successful wrapper therefore
means that software verification preserved the scientific prohibition; it is not a qualification
PASS or experimental evidence.

Stage-0 samples an independent, without-replacement 75% patch mask for every example at every
optimization step. Formal patch grids must contain a multiple of four patches so the mask
cardinality is exact. The checkpoint records this sampler contract together with the frozen
`source_encoder_train` per-channel CSI mean/scale used by the physical target transform
`Psi(H)`. Old fixed-bank or unnormalized Stage-0 checkpoints are rejected.

## 3.1 External papers, baselines, and Sionna facilities

The registry freezes ten third-party inputs, including four papers whose recorded licenses do not
grant this project downstream redistribution. The repository and both delivery formats omit those
four bytes. Fetch every missing input directly from its recorded source, then authenticate the full
ten-file local set before any formal preflight or external experiment:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PWD/.venv/bin/python" -m formal_v2.fetch_waibu_resources \
  --registry "$PWD/formal_v2/configs/waibu_resources_v1.json" \
  --waibu-root "$PWD/waibu"
```

Downloaded files are local inputs and are ignored by Git. Do not commit or redistribute them. The
fetcher refuses to overwrite existing bytes and accepts a download only when its SHA-256 matches the
registry. Source URL, license URL, and redistribution status remain separate from byte authentication:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PWD/.venv/bin/python" -m formal_v2.formal_cli verify-waibu-resources \
  --registry "$PWD/formal_v2/configs/waibu_resources_v1.json" \
  --waibu-root "$PWD/waibu" \
  --output "$PWD/runs/resource-auth-001"
```

The core environment runs CSI-MAE, CSI-CLIP, CSI-CLIP++, ContraWiMAE, WWM, SigMap, WiSER, and RFIR
controlled implementations. WWM and RFIR retain explicit inspired/style-controlled labels; WiSER is
a style-controlled 2D map/CSI diagnostic and is not C1-eligible.
Wi-GATr retains its separate Python 3.10
environment:

```bash
formal_v2/external_adapters/setup_wigatr.sh
```

The setup authenticates and imports the frozen environment and writes an idempotently verified
`runtime_provenance.json` under the environment root. Formal Wi-GATr execution requires an NVIDIA
CUDA device and visible driver because its frozen xFormers attention has no compatible CPU kernel.
The adapter records the actual interpreter, complete installed-distribution inventory, RECORD
digests, lock/vendor digests, Torch/CUDA/cuDNN/driver/GPU identity, and deterministic/TF32 state;
the outer runner probes the interpreter again and rejects a mismatch.

Sionna RT and the official large-radio-map tools use a separate Python 3.12 environment because of
their Mitsuba/Dr.Jit/Open3D stack:

```bash
formal_v2/external_adapters/setup_sionna.sh
PYTHONDONTWRITEBYTECODE=1 "$PWD/formal_v2/external_adapters/.runtime-sionna/venv/bin/python" -m formal_v2.sionna_facility \
  --runtime-root "$PWD/formal_v2/external_adapters/.runtime-sionna" verify
```

The runtime installs from the checked-in `sionna_lrm_uv.lock` with `uv sync --locked`, then installs
the fixed, wheel-hash-locked CPU PyTorch and h5py builds needed to reload the frozen Stage-0 teacher
and reproduce route assignments. It excludes the unrelated PyTorch CUDA dependency set. Setup also
requires libLLVM to match the platform-specific reviewed registry and writes a persistent runtime
record; an unregistered Linux library is a launch blocker, not an automatically trusted runtime.

Use `formal_v2/external_adapters/all_map_adapters_v1.json` for C1 and
`formal_v2/configs/representation_baselines_v1.json` for the representation comparison. Read
`formal_v2/WAIBU_INTEGRATION.md` for the exact mapping. Code, authenticated paper bytes, smoke
execution, and an installed simulator still do not constitute C1/G8 evidence.
The latest file-by-file runtime audit is `artifacts/waibu_integration_audit_2026-08-06.md`.

For primary data generated by Sionna RT, the shipped Sionna G8 adapter is not an
independent engine and is rejected by formal preflight. G8 then requires a
controlled real intervention or a genuinely different RT engine; it cannot be
closed by rerunning the same Sionna revision.
A precomputed independent-RT archive can be supplied through the diagnostic
contract in `formal_v2/A100_RUNBOOK.md`; the outer code binds its raw CSI,
scene assets, and engine configuration and recomputes statistics, but archive
mode is `DIAGNOSTIC_NOT_CLAIM` and is rejected by formal preflight and G8 claim
reauthentication. Formal G8 still requires an authenticated executable adapter
for the independent engine or a controlled real intervention.

Use `formal_cli export-sionna-scenes` to generate the hashed PLY/XML/assets manifest for every
`external_validation` sibling world before running the shipped
`formal_v2/configs/sionna_external_validity_adapter_v2.json` G8 adapter.
G8 PASS uses the lower cluster-bootstrap confidence bound for active direction agreement and a
cluster-level null-equivalence interval; repeated rows from one base map cannot increase its weight.
The built-in source-only SigMap scene-ID runner uses the V3 adapter/provenance contract and is
selected by default. RT-calibration and literature manifests use the evidence schemas documented
in `formal_v2/README.md`; legacy aggregate-only and RT V3 manifests are rejected. RT V4 additionally
requires structured fit/validation partitions with inline payloads and disjoint stable scene/unit IDs and
uses mean per-unit absolute error rather than a difference between aggregate means.

Shuffled-pair, retention, and all five resource controls are first-party executable adapters under
`formal_v2/external_adapters/`. Each run binds source, config, dataset, checkpoints, per-unit rows,
training traces, and replay artifacts. Resource accounting covers representation training and
retained inference while excluding the common localization head on both sides; concat bottleneck
parameters and measured training/inference FLOPs remain included. `generous_2x_concat` is
report-only.

## 4. Verify candidate evidence

The earlier 34-bank Apple Silicon candidate is bound by dataset SHA-256
`e5ec3d32bbb7c639f2fd6e6dcc23bc4bb6085cc76830a700e5b7103b17a37847`.
Its isolated zero-tolerance verification gate is bound by
`ac790e2ffacca784cba22bc31bc9798049bb219f3f88cfeb3cd0c69fce7c86a8`.
The submitted registry reports that all 34 scene banks and all nine role
groups passed at `rtol=0` and `atol=0`. Static verification checks the
committed hash registry and inventories only:

~~~bash
PYTHONDONTWRITEBYTECODE=1 python3 -B \
  artifacts/m4_formal_candidate_v2/verify_candidate_evidence.py
~~~

To authenticate the omitted local NPZ files, shard manifests, inspection, and
verification output, use the optional deep-verification command documented in
`artifacts/m4_formal_candidate_v2/README.md`. The repository does not
contain either NPZ or unsanitized host-local manifests.

That historical run used registered libLLVM `26273678...451`, but its bound
generator predates approval enforcement. Later registration cannot prove that
the runtime was preapproved at generation, so it remains blocked historical
evidence.

The newer registry reports dataset `e8903430...d2e` and an independent
regeneration under LLVM 22.1.8 library `e514c689...a88`.
`artifacts/m4_llvm22_candidate_v1/` records the claimed 34-scene inventory,
source/runtime fields, and a historical portable receipt. The recorded portable
bundle predates the current shipped verifier and must be re-exported before
diagnostic replay. The candidate, regenerated NPZ,
origin bundle, and runtime evidence remain outside Git, so source CI cannot
authenticate that execution. Transfer and replay are diagnostic content checks
only; formal use requires deep verification of the originals and a fresh live
independent regeneration on the destination.

If authenticated, a same-host, same-Sionna regeneration establishes
deterministic internal consistency only. It does not close independent RT calibration, G8,
destination A100 preflight, model execution, training, license review, or
scientific qualification.

## 5. Inspect formal data

Do not start training first. Validate the NPZ against the frozen contract:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PWD/.venv/bin/python" -m formal_v2.formal_cli inspect-data \
  --config "$PWD/formal_v2/configs/formal_v2.json" \
  --dataset /absolute/path/csi_pairs_formal_v2_1_v6.npz \
  --output "$PWD/runs/data-inspection-001"
```

Review the generated `data_contract.json`, the engine/config hash, license records, phase/gauge convention, scene roles, support/query isolation, repeats, natural anchors, and primitive permutations.

## 6. Prepare, review, and authorize a formal run

Formal execution is a two-phase protocol. `prepare-full-run` validates every late manifest, runtime
executable, license acknowledgement, credential name, pre-staged input, disk budget, CUDA GPU budget,
and output path before creating run artifacts. It then runs resources, G0, independent RT calibration,
independent data regeneration, G1/G2, and independent G8 external validity in that order. Only a
successful preparation writes
`OUTPUT/approval/request.json` and stops. The request binds a random run nonce plus the config,
dataset, source tree, core and external runtimes, GPU inventory, compute plan, teacher checkpoint, all
input manifests, pre-staged inputs, and every early gate hash.

The compute-plan JSON has schema `csi-pairs-full-run-compute-plan-v2` and these exact fields:

```json
{
  "schema_version": "csi-pairs-full-run-compute-plan-v2",
  "profile": "formal",
  "estimated_output_bytes": 0,
  "minimum_free_disk_bytes": 0,
  "estimated_wall_time_seconds": 0,
  "authorized_wall_time_seconds": 0,
  "required_gpu_count": 0,
  "minimum_gpu_memory_bytes": 0,
  "estimated_gpu_hours": 0,
  "authorized_gpu_hours": 0,
  "component_estimates": [],
  "required_environment_variables": [],
  "license_acknowledgements": []
}
```

The zeros are placeholders. A missing, zero, or measurement-placeholder compute
plan is advisory and does not block `prepare-full-run` or `all`. G0 literature
receipts/PDF hashes, independent data verification, C11 RT calibration, and G8
are likewise optional: skip them or keep a FAIL/NOT_ASSESSED record rather than
hard-stopping the chain. Those claims stay `BLOCKED`/`NOT_ASSESSED`; the runner
does not invent scientific PASS. Secret values are never recorded; only
environment-variable names and their presence enter the preflight.

Prepare into a new root, optionally containing only a pre-staged `inputs/` directory:

```bash
CSI_PAIRS_PYTHON="$PWD/.venv/bin/python" \
CSI_PAIRS_FORMAL_DATASET=/absolute/path/csi_pairs_formal_v2_1_v6.npz \
CSI_PAIRS_FORMAL_OUTPUT="$PWD/runs/formal-001" \
CSI_PAIRS_COMPUTE_PLAN=/absolute/path/compute-plan.json \
CSI_PAIRS_VERIFIER_MANIFEST=/absolute/path/independent_rt_verifier.json \
CSI_PAIRS_EXTERNAL_ADAPTER_MANIFEST=/absolute/path/external_adapters.json \
CSI_PAIRS_EXTERNAL_VALIDITY_MANIFEST=/absolute/path/external_validity.json \
CSI_PAIRS_LITERATURE_RESOURCE_MANIFEST=/absolute/path/literature.json \
CSI_PAIRS_RT_CALIBRATION_MANIFEST=/absolute/path/rt_calibration.json \
CSI_PAIRS_FULL_RUN_PHASE=prepare \
  formal_v2/scripts/run_formal_v2.sh
```

Qualification writes `route_noise_floor.csv`. G1 requires every registered route threshold to cover
the configured independent repeat-noise quantile in the same native units. Review the approval
request, `qualification/response_gate.csv`, `qualification/null_safety.csv`, G0, independent RT, and
the compute plan. An allowed LLM judge (`codex`, `claude-code`, or `cursor`) then creates an approval outside the run root:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PWD/.venv/bin/python" -m formal_v2.formal_cli create-run-approval \
  --request "$PWD/runs/formal-001/approval/request.json" \
  --output /absolute/path/formal-001-llm-judge-approval.json \
  --judge codex:bound-session \
  --expires-utc 2027-01-01T00:00:00Z \
  --attest-llm-judged
```

Repeat the same environment with `CSI_PAIRS_FULL_RUN_PHASE=run` and add:

```bash
CSI_PAIRS_LLM_JUDGE_APPROVAL_MANIFEST=/absolute/path/formal-001-llm-judge-approval.json
```

`all` resumes only that authenticated prepared root. It rejects Boolean-only authorization, stale or
expired approval, changed inputs/runtime/gates/teacher/compute plan, cross-run replay, and consumed
approval. It does not rerun G0, RT, data verification, G1/G2, or G8 after approval. Any formal stage
failure stops the chain. The deprecated `--approve-full-experiment` flag and
`CSI_PAIRS_APPROVE_FULL_EXPERIMENT` variable have no authorization power.

The script defaults to the shipped resource V3, shuffled-pair V3, retention V3, and built-in scene-ID
manifests. Formal scenes, independent RT inputs, external-validity input, installed Wi-GATr/Sionna
runtimes, licenses, CUDA capacity, and reviewed budget remain mandatory. Wi-GATr and PMNet use the
authenticated shipped adapter manifest and generate their source-only checkpoints/results inside the
authorized run; pre-generated result-only checkpoints are neither required nor accepted as a
substitute. Missing any required preflight input blocks preparation before training.

Never reuse an output directory except for the authenticated `prepare-full-run` to `all` transition.
Every individual evidence-producing stage holds an exclusive run-root operation lock. Fixture paths
are normalized to `.npz` before exclusive creation. Never replace paper placeholders with fixture
output. The inherited No-X failures and four warning names remain binding until untouched non-fixture
banks pass the registered gates.

The current bidirectional paper/code audit and five-layer readiness verdict are recorded in
`artifacts/v6_traceability_audit_2026-08-07.md`.
