#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT=/root/xunlian/Futaoran/CSI_EVALUATION_RUNTIME_FINAL_20260831/code/CSI-PAIRS-v2.0-server
RUNTIME_PYTHON=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python
FROZEN_ROOT=/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server
EVIDENCE_ROOT=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_8d489b2_20260831
ARCHIVE_ROOT=/root/xunlian/Futaoran/CSI_EVALUATION_INTEGRATION_20260831
ARCHIVED_SELF="$ARCHIVE_ROOT/experiment_records/CSI_PAIRS_V6_20260831/formal_external_inputs/evaluation_migration_8d489b2_20260831/accept_and_launch_after_judge_8d489b2.sh"
RUN_ROOT="$RUNTIME_ROOT/runs/formal-v6-streaming-8d489b2-d7d0b8affcd40f05"
REQUEST="$RUN_ROOT/migration/request.json"
ACCEPTED="$RUN_ROOT/migration/accepted.json"
APPROVAL_ROOT=/root/xunlian/Futaoran/formal_external_inputs/migration_approvals/formal-v6-streaming-8d489b2-d7d0b8affcd40f05
APPROVAL="$APPROVAL_ROOT/LLM_JUDGE_MIGRATION_APPROVAL.json"
TRANSPORT_RECEIPT="$APPROVAL_ROOT/LLM_JUDGE_TRANSPORT_RECEIPT.json"
PROMPT="$EVIDENCE_ROOT/independent_migration_judge_prompt_8d489b2.txt"
JUDGE_RUNNER="$EVIDENCE_ROOT/run_independent_migration_judge_8d489b2.sh"
RUN_ALL="$EVIDENCE_ROOT/run_all_streaming_8d489b2.sh"
SUPERVISOR="$EVIDENCE_ROOT/supervise_all_streaming_8d489b2.sh"
WIGATR_PROBE="$EVIDENCE_ROOT/probe_wigatr_runtime_8d489b2.sh"
REQUEST_CREATOR="$EVIDENCE_ROOT/create_migration_request_8d489b2.sh"
CONFIG="$RUNTIME_ROOT/formal_v2/configs/formal_v2.json"
DATASET=/root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz
PROTOCOL=/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/Idea1-CSI-PAIRS-冻结版-零基础阅读稿-v6_VSCode兼容版.md
COMPUTE_PLAN="$EVIDENCE_ROOT/compute_plan.json"
SUPERVISION_ROOT=/root/xunlian/Futaoran/formal_external_inputs/supervision/formal-v6-streaming-8d489b2-d7d0b8affcd40f05
COORDINATION_ROOT="$SUPERVISION_ROOT/acceptance-coordinator"
COORDINATION_RECEIPT="$COORDINATION_ROOT/acceptance_receipt.json"
LAUNCH_RECEIPT="$COORDINATION_ROOT/supervisor_launch_receipt.json"
EXPECTED_JUDGE=codex:gpt-5.6-sol-ultra-independent-migration-review-8d489b2
EXPECTED_PROMPT_SHA256=1d70d5452615709edd527b9fd8c7d52bc85c099d52e576fcd00a6271c1e91b8c
EXPECTED_JUDGE_RUNNER_SHA256=a1f1e7ae51a21849cfaba663473397aed6c30f850ec1ec31fb8e5dde977c409c
EXPECTED_RUN_ALL_SHA256=399a848fc5bb31cbb7c71a857f10584655164fdb6ae2b9321c765f7a459e6e1b
EXPECTED_SUPERVISOR_SHA256=fee5fe65e56d738bd4056dbc693fd2a343247c8d855471f15d2b9a76ba29fa37
EXPECTED_WIGATR_PROBE_SHA256=099901cc3b68a7021728ce37c74a037e77f3047a24200da966e899b6f7debf0f
EXPECTED_REQUEST_CREATOR_SHA256=c2f6e76609758de033a9d25d4bccd04d0518cc3767c081174d4ad6cfdcb12914
EXPECTED_RUNTIME_COMMIT=8d489b2387e7bb6c988a41d9e0d57b8a6cffc4d4
EXPECTED_SOURCE_SHA256=aa5b1d6a1062d68bb5de045b40f042e1a453d14a8d73632ad0be86dc7b07caff

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0,1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

refuse() {
  log "ACCEPTANCE_COORDINATOR_REFUSAL=$1" >&2
  exit 92
}

require_file_sha() {
  local path=$1
  local expected=$2
  [[ -f "$path" && ! -L "$path" ]] || refuse "MISSING_OR_UNSAFE:$path"
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] \
    || refuse "SHA256_MISMATCH:$path"
}

scientific_pyc_count() {
  find "$RUNTIME_ROOT" "$FROZEN_ROOT" "$EVIDENCE_ROOT" \
    \( -type d \( -name .git -o -name '.venv*' -o -name '.runtime*' \) -prune \) \
    -o \( -type f \( -name '*.pyc' -o -name '*.pyo' \) -print \) \
    | wc -l
}

static_preflight() {
  [[ -f "$ARCHIVED_SELF" && ! -L "$ARCHIVED_SELF" ]] \
    || refuse ARCHIVED_COORDINATOR_MISSING
  [[ -z "$(git -C "$ARCHIVE_ROOT" status --porcelain)" ]] \
    || refuse EVIDENCE_ARCHIVE_DIRTY
  cmp -s "$0" "$ARCHIVED_SELF" || refuse COORDINATOR_NOT_COMMITTED_COPY
  require_file_sha "$PROMPT" "$EXPECTED_PROMPT_SHA256"
  require_file_sha "$JUDGE_RUNNER" "$EXPECTED_JUDGE_RUNNER_SHA256"
  require_file_sha "$RUN_ALL" "$EXPECTED_RUN_ALL_SHA256"
  require_file_sha "$SUPERVISOR" "$EXPECTED_SUPERVISOR_SHA256"
  require_file_sha "$WIGATR_PROBE" "$EXPECTED_WIGATR_PROBE_SHA256"
  require_file_sha "$REQUEST_CREATOR" "$EXPECTED_REQUEST_CREATOR_SHA256"
  [[ "$(git -C "$RUNTIME_ROOT" rev-parse HEAD)" == "$EXPECTED_RUNTIME_COMMIT" ]] \
    || refuse RUNTIME_COMMIT_MISMATCH
  [[ -z "$(git -C "$RUNTIME_ROOT" status --porcelain --untracked-files=no)" ]] \
    || refuse RUNTIME_TRACKED_WORKTREE_DIRTY
  [[ "$(git -C "$RUNTIME_ROOT" ls-files --others --exclude-standard -- formal_v2)" == "formal_v2/external_adapters/.venv-wigatr" ]] \
    || refuse RUNTIME_UNTRACKED_SOURCE_MISMATCH
  [[ "$(scientific_pyc_count)" == 0 ]] || refuse BYTECODE_PRESENT
  (
    cd "$RUNTIME_ROOT"
    "$RUNTIME_PYTHON" -B -c \
      'from formal_v2.formal_migration import _source_identity; import sys; value=_source_identity("formal_v2"); sys.exit(0 if value["git_commit"]==sys.argv[1] and value["source_tree_sha256"]==sys.argv[2] else 1)' \
      "$EXPECTED_RUNTIME_COMMIT" "$EXPECTED_SOURCE_SHA256"
  ) || refuse RUNTIME_SOURCE_IDENTITY_MISMATCH
}

write_or_validate_transport_receipt() {
  cd "$RUNTIME_ROOT"
  "$RUNTIME_PYTHON" -B - \
    "$REQUEST" "$APPROVAL" "$APPROVAL_ROOT" "$TRANSPORT_RECEIPT" \
    "$EXPECTED_JUDGE" "$PROMPT" "$JUDGE_RUNNER" "$RUN_ALL" \
    "$SUPERVISOR" "$WIGATR_PROBE" "$REQUEST_CREATOR" <<'PY'
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

from formal_v2.formal_io import parse_strict_json, read_strict_json, sha256_file
from formal_v2.formal_migration import _validate_migration_approval
from formal_v2.formal_migration_evidence import write_json_exclusive_atomic


def binding(path_value):
    path = Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"transport input is missing or unsafe: {path}")
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


request_path = Path(sys.argv[1])
approval_path = Path(sys.argv[2])
approval_root = Path(sys.argv[3])
receipt_path = Path(sys.argv[4])
expected_judge = sys.argv[5]
prompt_path, runner_path, run_all_path, supervisor_path, probe_path, creator_path = (
    Path(value) for value in sys.argv[6:12]
)
if approval_root.is_symlink() or not approval_root.is_dir():
    raise RuntimeError("approval root is missing or unsafe")
request = read_strict_json(request_path)
approval = read_strict_json(approval_path)
if not isinstance(approval, dict) or approval.get("judge") != expected_judge:
    raise RuntimeError("approval judge identity mismatch")
_validate_migration_approval(
    approval,
    request,
    sha256_file(request_path),
    now=datetime.now(timezone.utc),
)

matches = []
for attempt in sorted(approval_root.glob("attempt-*")):
    if attempt.is_symlink() or not attempt.is_dir():
        raise RuntimeError(f"judge attempt directory is unsafe: {attempt}")
    review_path = attempt / "LLM_JUDGE_MIGRATION_REVIEW.md"
    event_path = attempt / "codex_judge_events.jsonl"
    stderr_path = attempt / "codex_judge_stderr.log"
    if not all(path.is_file() and not path.is_symlink() for path in (review_path, event_path, stderr_path)):
        continue
    review = review_path.read_text(encoding="utf-8")
    begin = "APPROVAL_JSON_BEGIN"
    end = "APPROVAL_JSON_END"
    if not review.startswith("DECISION=APPROVE") or review.count(begin) != 1 or review.count(end) != 1:
        continue
    encoded = review.split(begin, 1)[1].split(end, 1)[0].strip()
    try:
        transported = parse_strict_json(encoded)
    except Exception:
        continue
    if transported == approval:
        matches.append((attempt, review_path, event_path, stderr_path))
if len(matches) != 1:
    raise RuntimeError("approval does not bind exactly one preserved judge attempt")
attempt, review_path, event_path, stderr_path = matches[0]

events = []
for line_number, line in enumerate(event_path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    event = parse_strict_json(line)
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise RuntimeError(f"invalid Codex JSONL event at line {line_number}")
    events.append(event)
if not events:
    raise RuntimeError("Codex judge event log is empty")
event_types = Counter(str(event["type"]) for event in events)
if any(kind == "error" or kind.endswith(".failed") for kind in event_types):
    raise RuntimeError("Codex judge event log contains a failed top-level event")

core = {
    "schema_version": "csi-pairs-v6-independent-judge-transport-v1",
    "status": "PASS",
    "judge": expected_judge,
    "model": "gpt-5.6-sol",
    "reasoning_effort": "ultra",
    "sandbox": "read-only",
    "attempt_dir": str(attempt.resolve()),
    "request": binding(request_path),
    "approval": binding(approval_path),
    "review": binding(review_path),
    "event_log": binding(event_path),
    "stderr_log": binding(stderr_path),
    "event_count": len(events),
    "event_types": dict(sorted(event_types.items())),
    "prompt": binding(prompt_path),
    "helpers": {
        "judge_runner": binding(runner_path),
        "run_all": binding(run_all_path),
        "supervisor": binding(supervisor_path),
        "wigatr_probe": binding(probe_path),
        "request_creator": binding(creator_path),
    },
    "python_dont_write_bytecode": True,
}
if receipt_path.exists() or receipt_path.is_symlink():
    existing = read_strict_json(receipt_path)
    if not isinstance(existing, dict) or set(existing) != {*core, "created_utc"}:
        raise RuntimeError("existing transport receipt fields are invalid")
    if any(existing[key] != value for key, value in core.items()):
        raise RuntimeError("existing transport receipt bindings changed")
else:
    write_json_exclusive_atomic(
        receipt_path,
        {**core, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")},
    )
print("JUDGE_TRANSPORT_RECEIPT=" + str(receipt_path))
print("JUDGE_TRANSPORT_RECEIPT_SHA256=" + sha256_file(receipt_path))
PY
}

accept_and_authenticate() {
  local accept_attempt
  if [[ ! -e "$ACCEPTED" && ! -L "$ACCEPTED" ]]; then
    accept_attempt=$(mktemp -d "$COORDINATION_ROOT/accept-attempt-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
    cd "$RUNTIME_ROOT"
    if ! "$RUNTIME_PYTHON" -B -m formal_v2.formal_cli accept-migration-request \
      --request "$REQUEST" \
      --approval-manifest "$APPROVAL" \
      --config "$CONFIG" \
      --dataset "$DATASET" \
      --protocol "$PROTOCOL" \
      --output "$RUN_ROOT" \
      >"$accept_attempt/stdout.log" 2>"$accept_attempt/stderr.log"
    then
      refuse "OFFICIAL_ACCEPTANCE_FAILED:$accept_attempt"
    fi
  fi
  [[ -f "$ACCEPTED" && ! -L "$ACCEPTED" ]] || refuse ACCEPTED_RECEIPT_MISSING
  cd "$RUNTIME_ROOT"
  "$RUNTIME_PYTHON" -B - \
    "$ACCEPTED" "$RUN_ROOT" "$PROTOCOL" "$DATASET" "$CONFIG" \
    "$TRANSPORT_RECEIPT" "$COORDINATION_RECEIPT" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import sys

from formal_v2.formal_config import load_formal_config
from formal_v2.formal_evidence import config_sha256
from formal_v2.formal_io import read_strict_json, sha256_file
from formal_v2.formal_migration import authenticate_migration_receipt
from formal_v2.formal_migration_evidence import write_json_exclusive_atomic


def binding(path_value):
    path = Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"acceptance input is missing or unsafe: {path}")
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


accepted_path, run_root, protocol, dataset, config_path, transport_path, receipt_path = map(Path, sys.argv[1:8])
config = load_formal_config(config_path)
authenticate_migration_receipt(
    accepted_path,
    new_run_root=run_root,
    new_source_root=Path("formal_v2").resolve(),
    protocol_path=protocol,
    dataset_path=dataset,
    config_sha256=config_sha256(config),
    expected_seeds=config["seeds"],
)
core = {
    "schema_version": "csi-pairs-v6-acceptance-coordinator-v1",
    "status": "PASS",
    "official_acceptance_authenticated": True,
    "accepted": binding(accepted_path),
    "transport_receipt": binding(transport_path),
}
if receipt_path.exists() or receipt_path.is_symlink():
    existing = read_strict_json(receipt_path)
    if not isinstance(existing, dict) or set(existing) != {*core, "created_utc"}:
        raise RuntimeError("existing coordinator receipt fields are invalid")
    if any(existing[key] != value for key, value in core.items()):
        raise RuntimeError("existing coordinator receipt bindings changed")
else:
    write_json_exclusive_atomic(
        receipt_path,
        {**core, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")},
    )
print("MIGRATION_ACCEPTED_SHA256=" + sha256_file(accepted_path))
print("ACCEPTANCE_COORDINATOR_RECEIPT_SHA256=" + sha256_file(receipt_path))
PY
}

wait_for_idle_gpus() {
  local processes
  while true; do
    if ! processes=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>/dev/null); then
      refuse NVIDIA_PROCESS_QUERY_FAILED
    fi
    if [[ -z "${processes//[[:space:]]/}" ]]; then
      return 0
    fi
    log "ACCEPTANCE_COORDINATOR_WAIT=GPUS_OCCUPIED"
    sleep 60
  done
}

validate_disk_budget() {
  cd "$RUNTIME_ROOT"
  "$RUNTIME_PYTHON" -B - "$COMPUTE_PLAN" <<'PY'
from pathlib import Path
import os
import sys
from formal_v2.formal_io import read_strict_json

plan = read_strict_json(sys.argv[1])
stat = os.statvfs(Path(sys.argv[1]).parent)
free = stat.f_bavail * stat.f_frsize
minimum = int(plan["minimum_free_disk_bytes"])
if free < minimum:
    raise RuntimeError(f"disk budget failed: free={free}, minimum={minimum}")
print(f"DISK_BUDGET=PASS free={free} minimum={minimum}")
PY
}

launch_supervisor() {
  local boot_root
  local supervisor_pid
  local start_ticks
  mkdir -p "$SUPERVISION_ROOT"
  exec 8>"$SUPERVISION_ROOT/supervisor.lock"
  if ! flock -n 8; then
    log SUPERVISOR_ALREADY_ACTIVE
    return 0
  fi
  flock -u 8
  boot_root=$(mktemp -d "$COORDINATION_ROOT/supervisor-attempt-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
  nohup setsid "$SUPERVISOR" </dev/null >"$boot_root/stdout.log" 2>"$boot_root/stderr.log" &
  supervisor_pid=$!
  sleep 5
  [[ -r "/proc/$supervisor_pid/stat" ]] \
    || refuse "SUPERVISOR_EXITED_DURING_BOOT:$boot_root"
  start_ticks=$(awk '{print $22}' "/proc/$supervisor_pid/stat")
  cd "$RUNTIME_ROOT"
  "$RUNTIME_PYTHON" -B - \
    "$LAUNCH_RECEIPT" "$supervisor_pid" "$start_ticks" "$SUPERVISOR" \
    "$boot_root" "$ACCEPTED" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import sys
from formal_v2.formal_io import sha256_file
from formal_v2.formal_migration_evidence import write_json_exclusive_atomic

receipt, pid, ticks, supervisor, boot_root, accepted = sys.argv[1:7]
payload = {
    "schema_version": "csi-pairs-v6-supervisor-launch-v1",
    "status": "RUNNING",
    "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "pid": int(pid),
    "start_ticks": int(ticks),
    "supervisor_path": str(Path(supervisor).resolve()),
    "supervisor_sha256": sha256_file(supervisor),
    "boot_root": str(Path(boot_root).resolve()),
    "accepted_path": str(Path(accepted).resolve()),
    "accepted_sha256": sha256_file(accepted),
}
write_json_exclusive_atomic(receipt, payload)
print("SUPERVISOR_LAUNCH_RECEIPT_SHA256=" + sha256_file(receipt))
PY
  log "SUPERVISOR_STARTED pid=$supervisor_pid start_ticks=$start_ticks"
}

mkdir -p "$COORDINATION_ROOT"
exec 9>"$COORDINATION_ROOT/coordinator.lock"
if ! flock -n 9; then
  log ACCEPTANCE_COORDINATOR_ALREADY_ACTIVE
  exit 0
fi

static_preflight
log ACCEPTANCE_COORDINATOR_WAIT=LLM_APPROVAL
while [[ ! -f "$APPROVAL" || -L "$APPROVAL" ]]; do
  sleep 30
done
static_preflight
write_or_validate_transport_receipt
"$WIGATR_PROBE" validate
[[ "$(scientific_pyc_count)" == 0 ]] || refuse BYTECODE_CREATED_BY_APPROVAL_VALIDATION
accept_and_authenticate
validate_disk_budget
wait_for_idle_gpus
static_preflight
"$WIGATR_PROBE" validate
[[ "$(scientific_pyc_count)" == 0 ]] || refuse BYTECODE_CREATED_BEFORE_LAUNCH
launch_supervisor
log ACCEPTANCE_COORDINATOR=COMPLETE
