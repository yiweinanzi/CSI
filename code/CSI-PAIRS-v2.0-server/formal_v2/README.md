> 当前实验入口与运行说明见 [服务器 README](../README.md)。新任务使用 `paper_run`；下文保留原始 V6 科学定义与历史流程背景，其中旧认证/启动脚本不再是运行前置条件。

# CSI-PAIRS V6 formal implementation (V2.1)

This directory implements the frozen V6 protocol contract. It intentionally rejects the earlier V2.0 single-channel/full-vector schema.

Implemented code surfaces:

- seven-way source permission ledger and city/bank/base-map-cluster validation;
- strict embedded JSON, independent data regeneration, engine/config/license binding, typed maps/radio/path/no-op data;
- CSI-only 2D asymmetric MAE Stage-0 teacher with exact per-sample 75% masks resampled every step,
  independent B_audit_hold, sampler-bound checkpoints, and frozen physical readout;
- patch-level mask/query banks, separate full-channel Alignment and patch Response routes;
- shared F/P with complete teacher-encoder initialization, map/radio/BS-pose fusion, typed signed actions, latent and physical patch outputs;
- bank/route-stratified Endpoint, Alignment quartet, and Response objectives;
- one frozen source-method-selection pilot and canonical no-op Alignment tolerance;
- strict four arms with common batch plans, measured resource fields, and fail-closed seven-part G4;
- city-level k, heteroscedastic localization, exact V6 J_a, multilevel and bank-only bootstrap, leave-one sensitivity;
- active CGS plus gray/null distributions, unified response probes, q_comp/p_fail calibration, path matching/equivalence, external/scene-ID/resource/claim controls, independent RT calibration and external-validity adapters, literature/resource G0, and C1-C13/G0-G8 assembly;
- authenticated `waibu/` resource inventory, five source-only representation baselines, five map-conditioned six-condition adapters (two C1-eligible), and Sionna RT/large-radio-map facilities.

Unavailable data, external models, independent RT calibration, or unrun controls produce `NOT_ASSESSED/BLOCKED`. Code presence is not scientific evidence. The executed factorial localization table (commit `9850fff`) may be shown as a descriptive negative or ablation; that is not “no results exist.” A SOTA or success claim still requires same-data comparison meters. Those in-repo adapters are wired and not yet run.

CLI stages are visible with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m formal_v2.formal_cli --help
```

Formal orchestration is split between `prepare-full-run` and `all`. Preparation performs the complete
static manifest/disk/CUDA/license/credential preflight, then runs resources, G0, independent RT,
independent data verification, G1/G2, and independent G8 external validity before emitting a
random-nonce approval request. `all`
requires an external, unexpired llm-judge approval bound to that exact request and prepared root; the
deprecated Boolean flag has no authorization power. After authentication, `all` runs the remaining
factorial, evaluation, risk, path, external-baseline, representation, resource-control, scene-ID,
shuffled-pair, retention, and claim stages without rerunning preapproval stages.
A formal stage failure stops the chain. Fixtures remain `FORBIDDEN` at every artifact layer. Code and
tests do not constitute scientific evidence.
G1 also writes a per-bank `route_noise_floor.csv` and requires all four physical/teacher audit null
thresholds to cover the registered quantile of independent repeat-pair noise in their native
alignment/response norms. Alignment primary inclusion uses full-channel physical distance;
Response primary inclusion uses query-patch physical distance. Teacher sensitivity remains a
separate audit stratum and auxiliary G2 condition. Overall repeat NMSE cannot substitute for this
test, and teacher sensitivity cannot select raw-CSI primary samples.

The preparation root must be new, except that a single pre-staged `inputs/` directory is allowed for
authenticated Sionna scenes and other immutable run inputs. Those bytes enter the approval request.
The only reuse allowed is `all` resuming its exactly authenticated prepared root; approval is
single-use and cross-root or modified-root replay is rejected.
Individual stage commands atomically reserve their registered output path and hold an exclusive
operation lock for the run root, so concurrent, interrupted, or completed evidence directories
cannot be silently mixed or overwritten. Fixture paths are normalized to `.npz` before exclusive
creation.

`external_adapters/all_map_adapters_v1.json` registers SigMap, Wi-GATr, PMNet, WiSER, and RFIR for the
same internally generated six-condition unit registry. `configs/representation_baselines_v1.json`
registers CSI-MAE, CSI-CLIP, CSI-CLIP++, ContraWiMAE, and WWM-inspired same-world prediction for
the unified localization comparison. Signal-only representation rows can never count toward C1.
WiSER is a style-controlled 2D map/CSI diagnostic and not a faithful implementation of the paper;
RFIR remains style-controlled because the data contract lacks its multi-view RGB 3DGS geometry stage.
Wi-GATr and PMNet are C1-eligible official-code adaptations. C1 remains `BLOCKED` until both produce
authenticated non-fixture checkpoints and pass the same per-city active/null gate. The shipped
`configs/sionna_external_validity_adapter_v2.json` is the standard G8 adapter manifest.
See `WAIBU_INTEGRATION.md` and `external_adapters/README.md` for provenance and execution limits.

The repository ships first-party shuffled-pair, retention, scene-ID, and five resource-control
adapters with authenticated default manifests. Shuffled Alignment and Response models are trained
independently from deranged pair registries; retention probes are source-only and bind the frozen
Full checkpoint; scene-ID reuses the source-trained SigMap checkpoint and evaluates unseen source
banks; resource controls emit one checkpoint, log, loss trace, profiler summary, and replay record
per seed. Aggregate or self-reported substitutes remain rejected. Factorial
localization exists as a historical negative / ablation. Evaluation, C1
baselines, and the remaining controls are still unrun, so those surfaces
are executable protocol rather than comparison evidence.

External evidence contracts are fail-closed. Scene-ID adapters must bind their implementation source
and trained checkpoint, cover each held-out position with one exact four-condition unit, and pass
base-map-cluster bootstrap intervals rather than row-level point estimates. RT calibration V5
manifests bind separate fit data, validation inputs, and an independent per-unit validation-reference
CSV. Fit data and validation inputs are themselves strict JSON partition contracts; every unit
contains its `unit_id`, stable `scene_id`, nonempty inline payload, and a V3 `source` record binding
the source asset path and SHA-256, generation/acquisition batch, source record, and physical raw-unit
ID. Each source asset is a strict record contract; the outer runner locates the declared record and
requires its identity and canonical payload to match the partition row exactly. It then requires
unit, scene, source-asset content, source-record, and raw-unit
identities to be disjoint while recording equal canonical payloads as a diagnostic rather than
rejecting equal measurements from genuinely distinct physical units. Validation units must match
the reference CSV exactly; no identity sidecar is accepted. The adapter cannot
receive the reference path; it emits per-unit simulated statistics. The outer runner preserves every
per-unit absolute error and evaluates each C11 statistic by frozen mean absolute error per unit, so
opposite signed errors cannot cancel. G8 adapters emit raw independent-engine CSI in
an exact NPZ contract; direction and effect are recomputed outside the adapter before cluster-macro
confidence intervals are evaluated. G0 requires one raw API receipt for every frozen database/query
pair plus authenticated PDF records. C13 is supported only when G0 PASSes with a bound
`LLM_JUDGE_REVIEW.md` from an allowed coding-agent family (`codex`, `claude-code`, or `cursor`).
Each validated input manifest is copied into its stage output and
reauthenticated during claim assembly.

Every stage records and reauthenticates the formal source-tree digest, requirements-lock digest,
Python/platform identity, retained reviewed-wheel inventory, installed-file closure,
Torch/CUDA/GPU identity, and determinism settings. Setup downloads only lock-authenticated wheels,
installs offline with bytecode compilation disabled, retains those wheels as the trust root, and
removes all generated bytecode. Each evidence context rehashes the retained wheels and compares the
wheel-derived file union with the complete import-active site-packages tree. Every call walks the
entire closure; unchanged inode/size/mode/mtime/ctime fingerprints may reuse a process-local digest,
while any ordinary mutation forces a byte-for-byte rehash. Rewritten installation
`RECORD` files or pip reports cannot authorize changed or additional code; extra distributions,
unreviewed startup hooks, symlinks, and any `.pyc` fail closed. The locked `setuptools` wheel's exact
reviewed `distutils-precedence.pth` is the sole wheel-provided startup path entry. The CLI enables
deterministic Torch algorithms, disables TF32 and cuDNN benchmarking, and refuses to combine gates
produced by a different recorded runtime.
Wi-GATr and Sionna additionally use their actual external interpreters to emit complete runtime
records. Preparation and the outer evidence runners independently probe those interpreters and bind
the exact lock files, package versions/RECORD digests, CUDA/cuDNN/driver state, and environment digest
into the approval request, Wi-GATr V3 execution manifest, and G8 V4 gate.

`scripts/build_server_bundle.sh` creates a deterministic internal research-delivery ZIP. It contains
delivery provenance and only authenticated third-party files with recorded downstream redistribution
permission, so it is not an anonymous submission artifact. It excludes every
`redistribution_allowed=false` resource even if that file exists in the builder's local `waibu/`
directory. `scripts/build_anonymous_supplement.sh` creates the separate deterministic anonymous
package and excludes all of `waibu/`, internal Git provenance, and identity-bearing delivery audits.
Use `PYTHONDONTWRITEBYTECODE=1 python3 -m formal_v2.fetch_waibu_resources --registry ... --waibu-root ...` to obtain omitted
inputs directly from their frozen source URLs; formal resource verification remains strict and fails
until all eleven local files authenticate.

C1 rows bind the exact supplied map and directed action by SHA-256; the outer runner recomputes both
from the frozen unit registry and makes cluster-macro active-effect/null-equivalence decisions.
Shuffled-pair and retention controls use complete per-pair rows bound to the evaluation registry,
adapter source, independently trained control checkpoints, and exact formal checkpoints; aggregate
self-reported effects are rejected. Resource controls require frozen architecture/state specs,
per-step loss traces, operator-level profiler events, complete seed coverage, and a source-bound
replay. The resource scope excludes the common localization head from both main and control totals,
but includes concat bottleneck parameters and its measured training/inference FLOPs.
`generous_2x_concat` is report-only, not G4 subgate 7.

Use `make-fixture --source-banks-per-role 2` only when a software smoke must exercise the
cross-source-city scene-ID path. The generated data and every derivative remain permanently
`scientific_use=FORBIDDEN`.

`scripts/run_formal_v2_dry_run.sh` treats the fixture qualification's nonzero result as part of the
verification contract, not as a scientific success. It accepts only exit `1` plus a fully
reauthenticated `DRY_RUN_FAIL_NOT_EVIDENCE` gate with `passed=false`, `fixture=true`, and
`scientific_use=FORBIDDEN`; any other exit or manifest state makes the wrapper fail. Consequently,
`verify_server_bundle.sh` may return `0` after this expected fail-closed check without promoting G1,
G2, or any claim.
