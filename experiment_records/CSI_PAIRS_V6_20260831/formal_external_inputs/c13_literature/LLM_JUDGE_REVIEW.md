# C13 LLM-as-judge novelty and license review

- Judge family: codex
- Judge identity: 01a02fac-6ad2-7571-be26-89200b1ceecc
- Authorization basis: bound coding-agent session
- Review completed UTC: 2026-08-24T09:08:00Z
- Project dataset SHA-256: 060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac
- Project source-tree SHA-256: 83e72f0bfb8cf5539ee1bc5166dfe8cbea394b088bb4d8dad301183bd297f1b7
- Licenses reviewed for every local PDF/source resource: true
- No direct overlap with the frozen C13 claim: true
- RT path ready: true
- Map path ready: true
- External-validity path ready: true
- Allowed novelty scope: An evidence-gated paired-scene-intervention protocol that, for retained no-position localization representations, retraces Hamming-1 scene edits at fixed receiver, radio, and propagation-nuisance state, routes active, gray, and null physical effects, and tests the 2x2 Alignment/Response supervision factors; this is not a claim to the first geometry-aware wireless model, wireless world model, editable RF renderer, channel foundation model, or map-conditioned localizer.
- Conflicts or unresolved restrictions: WWM uses causal/world-model language for same-world multimodal prediction and RFIR supports editable RF geometry, so the claim must remain limited to the frozen paired-intervention audit and supervision protocol. The arXiv non-exclusive-distribution licenses on 2502.11965v2.pdf, 2505.09160v2.pdf, 2601.03789v1.pdf, and 2604.07086v1.pdf do not establish third-party redistribution permission; keep those PDFs out of redistributed artifacts unless separate permission is obtained. CC BY 4.0 papers require attribution, a license link, and change indication. Source-code redistribution remains subject to the BSD-3-Clause-Clear, MIT, and Apache-2.0 notice and disclaimer conditions; Wi-GATr's license grants no express or implied patent license.

## Claim reviewed

The review covered the paper abstract, all three contribution bullets, the
complete Related Work section, and the Artifact-to-Claim Checklist in
`paper_v2/main.tex` (SHA-256
`9a0d4aeb3655d16788c9d234cc454c8f6d22b45026de93877f96206e31379183`).
The frozen claim is narrower than geometry-aware prediction or wireless world
modeling: it combines controlled paired scene edits, physical active/gray/null
routing, Alignment/Response supervision, a preregistered four-arm comparison,
and retained no-position localization representation tests.

## Paper records

### 2406.14995v2.pdf - Wi-GATr

- Relevance: Geometry-conditioned learned wireless simulation, including an
  inverse receiver-localization demonstration, is a strong external baseline.
- Overlap category: `baseline`.
- Implementation status: `adapter_ready`; the project uses an official-code
  adaptation and does not claim that adaptation as a new method.
- Substantive boundary: Wi-GATr maps full 3D scene meshes and transmitter and
  receiver geometry to channel statistics. It does not train or audit a
  retained no-position localization representation on paired Hamming-1 scene
  interventions with active/gray/null routing and a 2x2 supervision design.
- License: the local PDF SHA-256 is
  `6c60e162f4de8a7523626730e3c1725c8d730c56dbe445712656cd6710d22201`;
  the authoritative arXiv record for v2 identifies CC BY 4.0. Redistribution is
  allowed with attribution, license link, and change indication.

### 2502.11965v2.pdf - CSI-CLIP

- Relevance: CSI/CIR same-sample contrastive pretraining and CSI localization
  are representation-learning baselines.
- Overlap category: `baseline`.
- Implementation status: `integrated` as a paper-spec controlled baseline, not
  represented as official source reproduction.
- Substantive boundary: the positive pair is the same channel in frequency and
  delay domains. It does not change scene geometry, retrace an intervention,
  or supervise physical response to an edit.
- License: the local PDF SHA-256 is
  `1b6aac49344d04e60e07ce29f950593b5699daddaac04a01ab848e7be3426c2d`;
  the authoritative arXiv v2 record uses the arXiv non-exclusive distribution
  license. Third-party redistribution permission is not established.

### 2505.09160v2.pdf - WiMAE and ContraWiMAE

- Relevance: masked CSI reconstruction plus noise-based contrastive learning is
  a strong representation baseline for downstream channel tasks.
- Overlap category: `baseline`.
- Implementation status: `integrated` as a paper-spec controlled baseline.
- Substantive boundary: its paired views are masked/noisy realizations of CSI;
  there is no controlled scene edit, physical effect routing, or retained
  localization response audit.
- License: the local PDF SHA-256 is
  `b32ac897858212faf9cd7189f84f1bf94e522f8b8dfdaaa881b58cc798fde4d1`;
  the authoritative arXiv v2 record uses the arXiv non-exclusive distribution
  license. Third-party redistribution permission is not established.

### 2601.03789v1.pdf - CSI-MAE

- Relevance: masked complex-CSI pretraining, positioning, and cross-scenario
  transfer form a representation and localization baseline.
- Overlap category: `baseline`.
- Implementation status: `integrated` as a paper-spec controlled baseline.
- Substantive boundary: it reconstructs masked antenna/subcarrier patches from
  statistical channel data; it does not use paired world interventions or
  measure an edit-response skill.
- License: the local PDF SHA-256 is
  `5ec21848bb1e82bd8b3331cea71e6663ce41f72cb410f6db72187c8088ae7d93`;
  the authoritative arXiv v1 record uses the arXiv non-exclusive distribution
  license. Third-party redistribution permission is not established.

### 2603.25216v1.pdf - Wireless World Model

- Relevance: WWM jointly embeds CSI, 3D point clouds, and trajectories using
  JEPA/MMoE pretraining and explicitly describes geometry-to-channel learning
  as causal/world modeling.
- Overlap category: `adjacent_nonoverlap`.
- Implementation status: `integrated` only as a WWM-inspired same-world matched
  prediction control; no faithful-source claim is made.
- Substantive boundary: WWM predicts masked same-world multimodal semantics and
  downstream tasks. The reviewed paper does not hold receiver/radio/nuisance
  state fixed across registered Hamming-1 world edits, route active/gray/null
  effects, or test Alignment/Response factors in a retained no-position
  localization encoder. Its broad causal wording prevents any broader CSI-PAIRS
  priority claim.
- License: the local PDF SHA-256 is
  `711304aed7766057263502d36af63927e97dff10d683dc1659e89e5a46b2b102`;
  the authoritative arXiv v1 record identifies CC BY 4.0. Redistribution is
  allowed with attribution, license link, and change indication.

### 2604.07086v1.pdf - RF inverse rendering

- Relevance: RFIR explicitly decouples emission, geometry, and material RF
  properties and demonstrates editable/reconfigurable RF scenes.
- Overlap category: `adjacent_nonoverlap`.
- Implementation status: `integrated` as an RFIR-inspired controlled forward
  model, not a faithful reproduction.
- Substantive boundary: RFIR is an editable forward/inverse RF renderer over 3D
  Gaussians and RF-BSDF attributes. It does not propose paired-intervention
  supervision for a retained no-position localization representation or the
  frozen active/gray/null four-arm evidence protocol. Its editability claim
  rules out describing CSI-PAIRS as the first editable RF model.
- License: the local PDF SHA-256 is
  `24838f6a01eab58bf31d7fe54167dc13a245d3e6a6986da4e3023cafe2e0ff83`;
  the authoritative arXiv v1 record uses the arXiv non-exclusive distribution
  license. Third-party redistribution permission is not established.

### 2606.04770v1.pdf - WiSER

- Relevance: a shared transmitter-conditioned sparse 3D scene memory predicts
  co-registered radiomaps and unordered CIR tap sets.
- Overlap category: `baseline`.
- Implementation status: `integrated` as a style-controlled diagnostic; its
  TRELLIS scene encoder and complete training schedule are not reproduced.
- Substantive boundary: WiSER is a geometry-conditioned multi-view forward
  predictor queried with receiver coordinates. It does not learn response to
  controlled paired edits or audit a retained no-position localizer.
- License: the local PDF SHA-256 is
  `97e368043d0c111987d26993ec88c295c5bb997126e21711223ca5448d1645ec`;
  the authoritative arXiv v1 record identifies CC BY 4.0. Redistribution is
  allowed with attribution, license link, and change indication.

## Official source archives

### Wi-GATr-main.zip

- Relevance and category: official Wi-GATr baseline source, `baseline`.
- Implementation status: `adapter_ready` at revision
  `6daa5bd49d9499817d903ab335830221d01e9daa`.
- Binding: archive SHA-256
  `b78be8ed14d16aab6c8fea54c2aa90a22bed492a12a6e892ee4c225c60a8138c`.
- License decision: the archive's LICENSE is BSD-3-Clause-Clear. Local use and
  redistribution are allowed if copyright, conditions, and disclaimer are
  retained; names may not endorse derived products, and no patent license is
  granted.

### PMNet-a0e0c592.zip

- Relevance and category: official PMNet map-to-radiomap baseline source,
  `baseline`.
- Implementation status: `adapter_ready` at revision
  `a0e0c5926de721074beeb23f630f2d313f6508dd`.
- Binding: archive SHA-256
  `e48e283e596270a9932052cc41f213b0a22c0ebbf29ba9551c5562d8cb877331`.
- License decision: the archive's LICENSE is MIT. Local use and redistribution
  are allowed with the copyright and permission notice retained.

### sionna-main.zip

- Relevance and category: primary Sionna RT source facility, `facility`.
- Implementation status: `integrated` at revision
  `04ddb9312116b408093b9d3ad363a3df355093a6`.
- Binding: archive SHA-256
  `fdbf89f307cc8933535af1587f00f1bcbd4b5edf7715275cd461bd4779f1fac7`.
- License decision: the archive's LICENSE identifies Apache-2.0. Local use and
  redistribution are allowed subject to Apache-2.0 license, notice, attribution,
  and modification-marking requirements.

### sionna-large-radio-maps-main.zip

- Relevance and category: scene tiling and large-radio-map source facility,
  `facility`.
- Implementation status: `integrated` at revision
  `1ba19ae1df1d26302fcfbaab14efc2347313da5d`.
- Binding: archive SHA-256
  `694ad17e7977e1c1adbdc8f93e6dcb1856e14cdf1b25f33da14c0e7aca80c33b`.
- License decision: the archive's LICENSE identifies Apache-2.0. Local use and
  redistribution are allowed subject to Apache-2.0 license, notice, attribution,
  and modification-marking requirements.

## Readiness interpretation

`RT path ready: true` records that the pinned Sionna 2.0.1/Sionna-RT 1.2.1
runtime, source archives, lock files, scene assets, and executable adapters are
present and provenance-probe successfully. `Map path ready: true` records that
the authenticated 227-bank dataset assets and required map-conditioned adapter
and control manifests are present. `External-validity path ready: true` records
that the pinned DiffeRT 0.10.0 runtime, independent adapter, engine config, and
source scene manifest authenticate. These are implementation-path decisions,
not scientific gate results. Current-run C11 and G8 remain `NOT_ASSESSED` unless
their optional authenticated stages are actually executed; no C11 or G8 PASS is
asserted by this review.

This LLM-as-judge review bound the listed resources and the frozen C13
claim, verified the recorded license/redistribution decisions from the cited
sources, and made the novelty and readiness decisions above.
The judge family is one of: codex, claude-code, cursor.

- Judge signature or authenticated identity: codex:01a02fac-6ad2-7571-be26-89200b1ceecc
- Signed UTC: 2026-08-24T09:08:00Z
