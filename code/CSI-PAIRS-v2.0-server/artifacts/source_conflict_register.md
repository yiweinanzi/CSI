# CSI-PAIRS V6 source conflict register

Date: 2026-08-08 (Asia/Shanghai); reader amendment recorded 2026-08-24

Authority hashes:

- unique frozen V6 reader: `ffe96c57c7986a77c54b1c52b9708088c714efa2a0cd67561aaf84a5b73c548b` (SHA-256 of `Idea1-CSI-PAIRS-冻结版-零基础阅读稿-v6_VSCode兼容版.md`, hashed 2026-08-24 after C13 llm-judge amendment)
- prior unique frozen V6 reader (2026-08-24 route amendment, pre-C13 llm-judge): `6f83b3701946e0a15f66869f58c3ffea6437a2ba14b41979cd41be1af28b6054`
- prior unique frozen V6 reader (pre-2026-08-24 amendment): `5866888fac736bcb812ebe3630b38095ad4989979a9fdf68cabcdbfe286f737e`
- integrated Goal prompt: `79b759141bd31a75fbefc80365ef5e6467e6cfedfa58457fa78b4ddebb4bb132`

The optional private matrix retains clause text. The tracked matrix records source line, document
hash, and clause hash without duplicating the authority prose into every row.

## Frozen author decisions

| ID | Sources | Author decision | Frozen protocol | Evidence and gate effect |
|---|---|---|---|---|
| `SC-GAUGE-001` | reader lines 352-354 and 518; plan line 287; Goal lines 214-232 | `A`, confirmed 2026-08-08: use a world-independent shared RT complex reference. | The dataset carries one complex reference value, stable ID and source SHA per scene-position with no world axis. The loader rejects malformed records; the authenticated independent verifier must regenerate the references and referenced CSI before qualification. | Code, data contract and tests implement A. `PROTOCOL_READY=PASS`; `FORMAL_INPUT_READY` remains blocked until independent RT/reference regeneration passes. |
| `SC-ROUTE-002` | reader lines 962, 966 and 1506-1508; plan lines 648-650 and 983-985; Goal lines 203-212 | `R1`, confirmed 2026-08-08: retain two explicit Response estimands. | Native target-free mask-cover uses full-channel physical-only `r^A`; the unified retained-state probe uses query-level physical-only `r^{R,q}`. Teacher sensitivity remains an independent stratum. | Dedicated regression tests exercise both asymmetric route combinations so the two selectors cannot be collapsed silently. |

## Explicit amendment already applied

| ID | Sources | Resolution | Evidence |
|---|---|---|---|
| `SC-ROUTE-001` | frozen V6 joint physical/teacher active wording; Goal lines 203-212 | Closed as Goal-vs-code only before 2026-08-24. The frozen reader itself was amended 2026-08-24, so this is no longer “Goal vs unamended reader”. Active primary routes are physical-only `Route(δ)`; teacher is an audit stratum. | Reader revision note 2026-08-24; `formal_v2/formal_routing.py::PRIMARY_ROUTE_CONTRACT`; target-replacement and teacher-latent-replacement regression tests. |

## Reader amendment 2026-08-24

The unique frozen V6 reader was amended so `SC-ROUTE-001` is no longer a Goal-vs-unamended-reader split:

- Active primary routes = physical-only `Route(δ)` (Alignment: full-channel physical distance; Response: query-patch physical distance).
- Teacher = independent audit / qualification-auxiliary stratum; it does not decide active/null.
- RQ5 is aligned to the C9 four risk gates (no separate eight-item hard gate).
- Scene-ID localization displacement direction (cosine) and Alignment main score `random_75` are documented in the reader.
- Position geometric-role preselect is documented; there is no post-CSI / post-route filtering.

This amendment updates the authority reader hash only. It does not execute C1–C13, and it does not change `FORMAL_GO=NO-GO`.

The 2026-08-24 C13 follow-up replaces human review with an LLM-as-judge
(`codex` / `claude-code` / `cursor`). A personal human attestation is rejected.

No protocol conflict remains open. The A + R1 confirmation and the 2026-08-24 reader amendment close only the author-decision
blockers; formal data, independent RT evidence, executed external checkpoints, licenses and
compute remain separate launch requirements.
