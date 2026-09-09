#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT=/root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server
FROZEN_ROOT=/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server
RUNTIME_PYTHON=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python
EVIDENCE_ROOT=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_bounded_8cafdf4_20260901
REQUEST="$RUNTIME_ROOT/runs/formal-v6-streaming-8cafdf4-08828d387df6db2b/migration/request.json"
PROMPT="$EVIDENCE_ROOT/independent_migration_judge_prompt_8cafdf4.txt"
RUN_ALL="$EVIDENCE_ROOT/run_all_streaming_8cafdf4.sh"
SUPERVISOR="$EVIDENCE_ROOT/supervise_all_streaming_8cafdf4.sh"
WIGATR_PROBE="$EVIDENCE_ROOT/probe_wigatr_runtime_8cafdf4.sh"
REQUEST_CREATOR="$EVIDENCE_ROOT/create_migration_request_after_gates_8cafdf4.sh"
APPROVAL_ROOT=/root/xunlian/Futaoran/formal_external_inputs/migration_approvals/formal-v6-streaming-8cafdf4-08828d387df6db2b
APPROVAL="$APPROVAL_ROOT/LLM_JUDGE_MIGRATION_APPROVAL.json"
EXPECTED_JUDGE=codex:gpt-5.6-sol-ultra-independent-migration-review-8cafdf4
EXPECTED_PROMPT_SHA256=c9f03fb0679507882162f67c7a335e5ded601aeac5b6f94bdd06d7f58307ead8
EXPECTED_RUN_ALL_SHA256=f7fb8a1013e33df821352a4a9e83ecf9ad3fd47712b65611098d6c683ae28fd2
EXPECTED_SUPERVISOR_SHA256=2f4944a226ba36f6740074d97d2558027a4cfca44456131994730e6e69653119
EXPECTED_WIGATR_PROBE_SHA256=1bb6de6c260f6c8f1625eb9188ca58bc27a8dc996996112ad7606b4a6ebe99d7
EXPECTED_REQUEST_CREATOR_SHA256=99bdd95e4ad55c95012a9311b7ed96493f870a053d01e158bc82a112c6543e06

export PYTHONDONTWRITEBYTECODE=1

scientific_pyc_count() {
  find "$RUNTIME_ROOT" "$FROZEN_ROOT" "$EVIDENCE_ROOT" \
    \( -type d \( -name .git -o -name '.venv*' -o -name '.runtime*' \) -prune \) \
    -o \( -type f \( -name '*.pyc' -o -name '*.pyo' \) -print \) \
    | wc -l
}

while [[ ! -f "$REQUEST" ]]; do
  sleep 30
done

required=(
  "$REQUEST"
  "$PROMPT"
  "$RUN_ALL"
  "$SUPERVISOR"
  "$WIGATR_PROBE"
  "$REQUEST_CREATOR"
  "$EVIDENCE_ROOT/wigatr_runtime_preflight.json"
  "$EVIDENCE_ROOT/critical_regression_8cafdf4.xml"
  "$EVIDENCE_ROOT/performance/performance.json"
  "$EVIDENCE_ROOT/equivalence.json"
  "$EVIDENCE_ROOT/controls/resume.json"
  "$EVIDENCE_ROOT/controls/corrupt_checkpoint.json"
  "$EVIDENCE_ROOT/controls/stale_checkpoint.json"
  "$EVIDENCE_ROOT/controls/lock.json"
  "$EVIDENCE_ROOT/controls/progress.json"
  "$EVIDENCE_ROOT/base_to_new_diff.json"
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
exec "$RUNTIME_PYTHON" -B - "$REQUEST" "$REVIEW" "$APPROVAL" "$EXPECTED_JUDGE" <<'PY'
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
