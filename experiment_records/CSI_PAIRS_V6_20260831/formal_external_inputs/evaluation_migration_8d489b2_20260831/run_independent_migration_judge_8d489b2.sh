#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT=/root/xunlian/Futaoran/CSI_EVALUATION_RUNTIME_FINAL_20260831/code/CSI-PAIRS-v2.0-server
FROZEN_ROOT=/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server
RUNTIME_PYTHON=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python
EVIDENCE_ROOT=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_8d489b2_20260831
REQUEST="$RUNTIME_ROOT/runs/formal-v6-streaming-8d489b2-d7d0b8affcd40f05/migration/request.json"
PROMPT="$EVIDENCE_ROOT/independent_migration_judge_prompt_8d489b2.txt"
RUN_ALL="$EVIDENCE_ROOT/run_all_streaming_8d489b2.sh"
SUPERVISOR="$EVIDENCE_ROOT/supervise_all_streaming_8d489b2.sh"
WIGATR_PROBE="$EVIDENCE_ROOT/probe_wigatr_runtime_8d489b2.sh"
REQUEST_CREATOR="$EVIDENCE_ROOT/create_migration_request_8d489b2.sh"
GATE_VALIDATOR="$EVIDENCE_ROOT/validate_migration_gates_8d489b2.sh"
APPROVAL_ROOT=/root/xunlian/Futaoran/formal_external_inputs/migration_approvals/formal-v6-streaming-8d489b2-d7d0b8affcd40f05
APPROVAL="$APPROVAL_ROOT/LLM_JUDGE_MIGRATION_APPROVAL.json"
EXPECTED_JUDGE=codex:gpt-5.6-sol-ultra-independent-migration-review-8d489b2
EXPECTED_PROMPT_SHA256=6902850bd4ec66aa85cd429bb3ab5ccc38d6719f0496bae99bd44a7afbef5621
EXPECTED_RUN_ALL_SHA256=399a848fc5bb31cbb7c71a857f10584655164fdb6ae2b9321c765f7a459e6e1b
EXPECTED_SUPERVISOR_SHA256=22cb80762e7e11a4f728fc0746459b69880be43aa52d770b00ee9de77c9ed3ad
EXPECTED_WIGATR_PROBE_SHA256=099901cc3b68a7021728ce37c74a037e77f3047a24200da966e899b6f7debf0f
EXPECTED_REQUEST_CREATOR_SHA256=d6df54f8c483a80fa8b5459750dd38cbae48c0bddf8091113ef3ef9be49f9883
EXPECTED_GATE_VALIDATOR_SHA256=dbf185d225711803aeab886da1a1f56c267eb0af39c17d0912dc5c5357f65b05

export PYTHONDONTWRITEBYTECODE=1

scientific_pyc_count() {
  find "$RUNTIME_ROOT" "$FROZEN_ROOT" "$EVIDENCE_ROOT" \
    \( -type d \( -name .git -o -name '.venv*' -o -name '.runtime*' \) -prune \) \
    -o \( -type f \( -name '*.pyc' -o -name '*.pyo' \) -print \) \
    | wc -l
}

required=(
  "$REQUEST"
  "$PROMPT"
  "$RUN_ALL"
  "$SUPERVISOR"
  "$WIGATR_PROBE"
  "$REQUEST_CREATOR"
  "$GATE_VALIDATOR"
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

declare -A expected_sha256=(
  ["$PROMPT"]="$EXPECTED_PROMPT_SHA256"
  ["$RUN_ALL"]="$EXPECTED_RUN_ALL_SHA256"
  ["$SUPERVISOR"]="$EXPECTED_SUPERVISOR_SHA256"
  ["$WIGATR_PROBE"]="$EXPECTED_WIGATR_PROBE_SHA256"
  ["$REQUEST_CREATOR"]="$EXPECTED_REQUEST_CREATOR_SHA256"
  ["$GATE_VALIDATOR"]="$EXPECTED_GATE_VALIDATOR_SHA256"
)
for artifact in "${!expected_sha256[@]}"; do
  observed_sha256=$(sha256sum "$artifact" | awk '{print $1}')
  if [[ "$observed_sha256" != "${expected_sha256[$artifact]}" ]]; then
    printf 'JUDGE_REFUSAL=pinned input SHA-256 mismatch: %s\n' "$artifact" >&2
    exit 94
  fi
done

if [[ -L "$APPROVAL_ROOT" ]]; then
  printf 'JUDGE_REFUSAL=approval root is a symlink: %s\n' "$APPROVAL_ROOT" >&2
  exit 95
fi
mkdir -p "$APPROVAL_ROOT"
exec 9>"$APPROVAL_ROOT/independent_judge.lock"
if ! flock -n 9; then
  printf 'JUDGE_REFUSAL=independent judge already active\n' >&2
  exit 96
fi
if [[ -e "$APPROVAL" || -L "$APPROVAL" ]]; then
  printf 'JUDGE_REFUSAL=exclusive approval already exists: %s\n' "$APPROVAL" >&2
  exit 97
fi
if [[ "$(scientific_pyc_count)" != 0 ]]; then
  printf 'JUDGE_REFUSAL=scientific source or evidence contains bytecode\n' >&2
  exit 98
fi

ATTEMPT_ROOT=$(mktemp -d "$APPROVAL_ROOT/attempt-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
REVIEW="$ATTEMPT_ROOT/LLM_JUDGE_MIGRATION_REVIEW.md"
EVENT_LOG="$ATTEMPT_ROOT/codex_judge_events.jsonl"
STDERR_LOG="$ATTEMPT_ROOT/codex_judge_stderr.log"

cd /root/xunlian/Futaoran
if ! codex exec --ephemeral --json --color never \
  --model gpt-5.6-sol \
  -c 'model_reasoning_effort="ultra"' \
  --sandbox read-only \
  --cd /root/xunlian/Futaoran \
  --skip-git-repo-check \
  --output-last-message "$REVIEW" \
  - <"$PROMPT" >"$EVENT_LOG" 2>"$STDERR_LOG"
then
  printf 'JUDGE_RESULT=CODEX_EXEC_FAILED attempt=%s\n' "$ATTEMPT_ROOT" >&2
  exit 99
fi

if [[ ! -f "$REVIEW" || -L "$REVIEW" ]]; then
  printf 'JUDGE_RESULT=REVIEW_MISSING attempt=%s\n' "$ATTEMPT_ROOT" >&2
  exit 100
fi
if [[ "$(scientific_pyc_count)" != 0 ]]; then
  printf 'JUDGE_RESULT=BYTECODE_CREATED attempt=%s\n' "$ATTEMPT_ROOT" >&2
  exit 101
fi

cd "$RUNTIME_ROOT"
env PYTHONDONTWRITEBYTECODE=1 "$RUNTIME_PYTHON" -B - \
  "$REQUEST" "$REVIEW" "$APPROVAL" "$EXPECTED_JUDGE" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import sys

from formal_v2.formal_io import parse_strict_json, read_strict_json, sha256_file
from formal_v2.formal_migration import _validate_migration_approval
from formal_v2.formal_migration_evidence import write_json_exclusive_atomic

request_path = Path(sys.argv[1])
review_path = Path(sys.argv[2])
approval_path = Path(sys.argv[3])
expected_judge = sys.argv[4]
review = review_path.read_text(encoding="utf-8")
begin = "APPROVAL_JSON_BEGIN"
end = "APPROVAL_JSON_END"
if not review.startswith("DECISION=APPROVE"):
    raise RuntimeError("independent judge did not approve")
if review.count(begin) != 1 or review.count(end) != 1:
    raise RuntimeError("independent judge approval delimiters are invalid")
prefix, remainder = review.split(begin, 1)
encoded, suffix = remainder.split(end, 1)
if end in prefix or begin in suffix or not encoded.strip():
    raise RuntimeError("independent judge approval transport is malformed")
approval = parse_strict_json(encoded.strip())
if not isinstance(approval, dict) or approval.get("judge") != expected_judge:
    raise RuntimeError("independent judge identity mismatch")
request = read_strict_json(request_path)
_validate_migration_approval(
    approval,
    request,
    sha256_file(request_path),
    now=datetime.now(timezone.utc),
)
write_json_exclusive_atomic(approval_path, approval)
published = read_strict_json(approval_path)
_validate_migration_approval(
    published,
    request,
    sha256_file(request_path),
    now=datetime.now(timezone.utc),
)
print("JUDGE_APPROVAL_VALIDATED=" + sha256_file(approval_path))
print("JUDGE_REVIEW_SHA256=" + sha256_file(review_path))
PY
