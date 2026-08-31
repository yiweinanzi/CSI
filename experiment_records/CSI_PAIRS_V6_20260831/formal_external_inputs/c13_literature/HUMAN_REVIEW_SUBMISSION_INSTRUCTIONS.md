# C13 human-review submission instructions

Status: `INSTRUCTIONS_ONLY_NOT_A_REVIEW_OR_APPROVAL`

Only a real authorized reviewer may perform, complete, sign, or submit the C13
review. The linter below checks formatting and frozen hashes only; it does not
make a novelty, overlap, readiness, licence, redistribution, identity, or
authorization decision.

1. Personally review `C13_REVIEW_PACKET.md`, all seven authenticated papers,
   the four source archives, their upstream licence sources, and the frozen
   project claims.
2. Fill `HUMAN_REVIEW_TEMPLATE.md`, include all fourteen completed lines from
   `HUMAN_REVIEW_STRUCTURED_APPENDIX_TEMPLATE.md`, and initially save the result
   as `HUMAN_REVIEW.candidate.md`. Do not create the watched
   `HUMAN_REVIEW.md` yet.
3. Run the structural linter from the CSI-PAIRS repository:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  .venv-core-formal-20260819T091049Z/bin/python -P -B \
  artifacts/formal_readiness/tools/lint_c13_human_review_candidate.py \
  --candidate /root/xunlian/Futaoran/formal_external_inputs/c13_literature/HUMAN_REVIEW.candidate.md \
  --c13-root /root/xunlian/Futaoran/formal_external_inputs/c13_literature
```

4. Confirm the linter reports
   `STRUCTURE_PASS_NOT_AUTHENTICITY_OR_SCIENTIFIC_REVIEW`. Independently confirm
   that every decision, identity, timestamp, source and signature is genuine.
   `formal_pass_decisions_selected=false` is a legitimate reviewed result and
   must not be changed merely to let the experiment proceed; in that case the
   formal run remains blocked by design.
5. Submit atomically only after those checks:

```bash
mv --no-clobber -- \
  /root/xunlian/Futaoran/formal_external_inputs/c13_literature/HUMAN_REVIEW.candidate.md \
  /root/xunlian/Futaoran/formal_external_inputs/c13_literature/HUMAN_REVIEW.md
```

The detached C13 watcher polls the final path. A malformed final submission
will fail closed and will not be treated as human evidence.
