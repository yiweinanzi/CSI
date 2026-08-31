#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT=/root/xunlian/Futaoran/CSI_EVALUATION_RUNTIME_FINAL_20260831/code/CSI-PAIRS-v2.0-server
RUNTIME_PYTHON=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python
EVIDENCE_ROOT=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_8d489b2_20260831
REQUEST="$RUNTIME_ROOT/runs/formal-v6-streaming-8d489b2-d7d0b8affcd40f05/migration/request.json"
PROMPT="$EVIDENCE_ROOT/independent_migration_judge_prompt_8d489b2.txt"
APPROVAL_ROOT=/root/xunlian/Futaoran/formal_external_inputs/migration_approvals/formal-v6-streaming-8d489b2-d7d0b8affcd40f05
APPROVAL="$APPROVAL_ROOT/LLM_JUDGE_MIGRATION_APPROVAL.json"
REVIEW="$APPROVAL_ROOT/LLM_JUDGE_MIGRATION_REVIEW.md"
EVENT_LOG="$APPROVAL_ROOT/codex_judge_events.jsonl"

required=(
  "$REQUEST"
  "$PROMPT"
  "$EVIDENCE_ROOT/performance/performance.json"
  "$EVIDENCE_ROOT/equivalence.json"
  "$EVIDENCE_ROOT/controls/resume.json"
  "$EVIDENCE_ROOT/controls/corrupt_checkpoint.json"
  "$EVIDENCE_ROOT/controls/stale_checkpoint.json"
  "$EVIDENCE_ROOT/controls/lock.json"
  "$EVIDENCE_ROOT/controls/progress.json"
  "$EVIDENCE_ROOT/base_to_new_diff.json"
  "$EVIDENCE_ROOT/wigatr_runtime_preflight.json"
  "$EVIDENCE_ROOT/post_exit_freeze/legacy_evaluation_inventory.json"
  "$EVIDENCE_ROOT/post_exit_freeze/legacy_evaluation_post_exit_freeze.json"
)
for artifact in "${required[@]}"; do
  if [[ ! -f "$artifact" || -L "$artifact" ]]; then
    printf 'JUDGE_REFUSAL=required input missing or unsafe: %s\n' "$artifact" >&2
    exit 93
  fi
done

mkdir -p "$APPROVAL_ROOT"
for target in "$APPROVAL" "$REVIEW" "$EVENT_LOG"; do
  if [[ -e "$target" || -L "$target" ]]; then
    printf 'JUDGE_REFUSAL=exclusive output already exists: %s\n' "$target" >&2
    exit 94
  fi
done

cd "$RUNTIME_ROOT"
codex exec --ephemeral --json --color never \
  --model gpt-5.6-sol \
  -c 'model_reasoning_effort="ultra"' \
  --sandbox danger-full-access \
  --cd "$RUNTIME_ROOT" \
  --add-dir "$EVIDENCE_ROOT" \
  --add-dir "$APPROVAL_ROOT" \
  --output-last-message "$REVIEW" \
  - <"$PROMPT" >"$EVENT_LOG"

if [[ ! -f "$APPROVAL" || -L "$APPROVAL" ]]; then
  printf 'JUDGE_RESULT=REJECT_OR_NO_APPROVAL\n' >&2
  exit 95
fi

env PYTHONDONTWRITEBYTECODE=1 "$RUNTIME_PYTHON" -B -c \
  'from datetime import datetime, timezone; from formal_v2.formal_io import read_strict_json, sha256_file; from formal_v2.formal_migration import _validate_migration_approval; import sys; request=read_strict_json(sys.argv[1]); approval=read_strict_json(sys.argv[2]); _validate_migration_approval(approval, request, sha256_file(sys.argv[1]), now=datetime.now(timezone.utc)); print("JUDGE_APPROVAL_VALIDATED=" + sha256_file(sys.argv[2]))' \
  "$REQUEST" "$APPROVAL"
