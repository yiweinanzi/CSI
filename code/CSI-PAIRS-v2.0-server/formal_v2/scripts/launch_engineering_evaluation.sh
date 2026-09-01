#!/usr/bin/env bash
# Portable engineering NON-CLAIM evaluation launcher.
# This is not a formal publication start. formal_cli run-evaluation now
# writes ENGINEERING_UNAPPROVED and requires LIVE data verification.
# Frozen Autodl 8cafdf4 scripts are archives. Formal prepare/run entry:
#   formal_v2/scripts/run_formal_v2.sh
# This wrapper splits P0 (identity + equivalence + inventory) from P1
# (control reports + optional second LLM judge). Evaluation may start after
# P0. The second judge is publication review, never a start barrier.
# Downstream dirs such as risk/ do not permanently refuse resume.
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CSI_PAIRS_ROOT="${CSI_PAIRS_ROOT:-${DEFAULT_ROOT}}"
CSI_PAIRS_PYTHON="${CSI_PAIRS_PYTHON:-python3}"
CSI_PAIRS_FORMAL_CONFIG="${CSI_PAIRS_FORMAL_CONFIG:-${CSI_PAIRS_ROOT}/formal_v2/configs/formal_v2.json}"
CSI_PAIRS_EVIDENCE_ROOT="${CSI_PAIRS_EVIDENCE_ROOT:-}"
CSI_PAIRS_LAUNCH_TIER="${CSI_PAIRS_LAUNCH_TIER:-p0}"
CSI_PAIRS_REQUIRE_P1="${CSI_PAIRS_REQUIRE_P1:-0}"
CSI_PAIRS_REQUIRE_SECOND_JUDGE="${CSI_PAIRS_REQUIRE_SECOND_JUDGE:-0}"
CSI_PAIRS_SUPERVISION_ROOT="${CSI_PAIRS_SUPERVISION_ROOT:-}"
MAX_EVALUATION_ATTEMPTS="${CSI_PAIRS_MAX_EVALUATION_ATTEMPTS:-3}"
RETRY_SECONDS="${CSI_PAIRS_RETRY_SECONDS:-60}"

usage() {
  cat <<'EOF'
Usage: launch_engineering_evaluation.sh <check-p0|check-p1|launch|supervise|help>

Environment:
  CSI_PAIRS_ROOT              project root (default: two levels above this script)
  CSI_PAIRS_DATASET           dataset.npz (required for launch/supervise)
  CSI_PAIRS_PYTHON            interpreter (default: python3)
  CSI_PAIRS_FORMAL_OUTPUT     run output directory (required for launch/supervise)
  CSI_PAIRS_FORMAL_CONFIG     formal config JSON
  CSI_PAIRS_EVIDENCE_ROOT     optional receipt directory for P0/P1 checks
  CSI_PAIRS_LAUNCH_TIER       p0 (default) or p1
  CSI_PAIRS_REQUIRE_P1        1 to require control reports before launch
  CSI_PAIRS_REQUIRE_SECOND_JUDGE
                              1 to require a publication judge file; never default
  CSI_PAIRS_SECOND_JUDGE_APPROVAL
                              optional judge JSON; ignored unless required
  CSI_PAIRS_P0_IDENTITY / CSI_PAIRS_P0_EQUIVALENCE / CSI_PAIRS_P0_INVENTORY
  CSI_PAIRS_FROZEN_LEGACY_COMMIT
                              optional; if unset, read from CSI_PAIRS_MIGRATION_RECEIPT

Formal prepare/run remains formal_v2/scripts/run_formal_v2.sh.
EOF
}

require_regular_file() {
  local path=$1
  local label=$2
  if [[ -z "${path}" || ! -f "${path}" || -L "${path}" ]]; then
    printf 'P_RECEIPT_MISSING=%s path=%s\n' "${label}" "${path:-unset}" >&2
    return 1
  fi
}

p0_identity() {
  if [[ -n "${CSI_PAIRS_P0_IDENTITY:-}" ]]; then
    printf '%s\n' "${CSI_PAIRS_P0_IDENTITY}"
    return 0
  fi
  [[ -n "${CSI_PAIRS_EVIDENCE_ROOT}" ]] || return 1
  printf '%s\n' "${CSI_PAIRS_EVIDENCE_ROOT}/expected_identity.json"
}

p0_equivalence() {
  if [[ -n "${CSI_PAIRS_P0_EQUIVALENCE:-}" ]]; then
    printf '%s\n' "${CSI_PAIRS_P0_EQUIVALENCE}"
    return 0
  fi
  [[ -n "${CSI_PAIRS_EVIDENCE_ROOT}" ]] || return 1
  printf '%s\n' "${CSI_PAIRS_EVIDENCE_ROOT}/equivalence.json"
}

p0_inventory() {
  if [[ -n "${CSI_PAIRS_P0_INVENTORY:-}" ]]; then
    printf '%s\n' "${CSI_PAIRS_P0_INVENTORY}"
    return 0
  fi
  [[ -n "${CSI_PAIRS_EVIDENCE_ROOT}" ]] || return 1
  printf '%s\n' "${CSI_PAIRS_EVIDENCE_ROOT}/post_exit_freeze/legacy_evaluation_inventory.json"
}

check_p0() {
  if [[ -z "${CSI_PAIRS_EVIDENCE_ROOT}" && -z "${CSI_PAIRS_P0_IDENTITY:-}" && -z "${CSI_PAIRS_P0_EQUIVALENCE:-}" && -z "${CSI_PAIRS_P0_INVENTORY:-}" ]]; then
    printf 'P0_SKIPPED=no evidence root; engineering launch is allowed without theatrical receipts\n'
    return 0
  fi
  require_regular_file "$(p0_identity)" identity
  require_regular_file "$(p0_equivalence)" equivalence
  require_regular_file "$(p0_inventory)" inventory
  printf 'P0_PASS=identity+equivalence+inventory\n'
}

p1_paths() {
  local root=${CSI_PAIRS_EVIDENCE_ROOT:-}
  [[ -n "${root}" ]] || return 1
  printf '%s\n' \
    "${root}/performance/performance.json" \
    "${root}/controls/resume.json" \
    "${root}/controls/corrupt_checkpoint.json" \
    "${root}/controls/stale_checkpoint.json" \
    "${root}/controls/lock.json" \
    "${root}/controls/progress.json" \
    "${root}/base_to_new_diff.json" \
    "${root}/post_exit_freeze/legacy_evaluation_post_exit_freeze.json"
}

check_p1() {
  if [[ -z "${CSI_PAIRS_EVIDENCE_ROOT}" ]]; then
    printf 'P1_SKIPPED=no evidence root; control reports are optional publication review\n'
    return 0
  fi
  local path
  local missing=0
  while IFS= read -r path; do
    if [[ ! -f "${path}" || -L "${path}" ]]; then
      printf 'P1_RECEIPT_MISSING=%s\n' "${path}" >&2
      missing=1
    fi
  done < <(p1_paths)
  if [[ "${missing}" != 0 ]]; then
    return 1
  fi
  printf 'P1_PASS=control_reports\n'
}

check_second_judge() {
  if [[ "${CSI_PAIRS_REQUIRE_SECOND_JUDGE}" != "1" ]]; then
    printf 'SECOND_JUDGE=optional_publication_review skipped\n'
    return 0
  fi
  require_regular_file "${CSI_PAIRS_SECOND_JUDGE_APPROVAL:-}" second_judge_approval
}

frozen_legacy_commit() {
  if [[ -n "${CSI_PAIRS_FROZEN_LEGACY_COMMIT:-}" ]]; then
    printf '%s\n' "${CSI_PAIRS_FROZEN_LEGACY_COMMIT}"
    return 0
  fi
  if [[ -n "${CSI_PAIRS_MIGRATION_RECEIPT:-}" && -f "${CSI_PAIRS_MIGRATION_RECEIPT}" && ! -L "${CSI_PAIRS_MIGRATION_RECEIPT}" ]]; then
    "${CSI_PAIRS_PYTHON}" -B - "${CSI_PAIRS_MIGRATION_RECEIPT}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in ("frozen_legacy_commit", "legacy_commit", "legacy_git_commit", "git_commit"):
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
        print(value.strip())
        raise SystemExit(0)
legacy = payload.get("legacy")
if isinstance(legacy, dict):
    for key in ("git_commit", "commit", "frozen_legacy_commit"):
        value = legacy.get(key)
        if isinstance(value, str) and value.strip():
            print(value.strip())
            raise SystemExit(0)
raise SystemExit("migration receipt does not expose a legacy commit")
PY
    return 0
  fi
  return 0
}

launch_evaluation() {
  local output=${CSI_PAIRS_FORMAL_OUTPUT:?set CSI_PAIRS_FORMAL_OUTPUT}
  local dataset=${CSI_PAIRS_DATASET:?set CSI_PAIRS_DATASET}
  check_p0
  if [[ "${CSI_PAIRS_LAUNCH_TIER}" == "p1" || "${CSI_PAIRS_REQUIRE_P1}" == "1" ]]; then
    check_p1
  else
    printf 'P1=optional_publication_review not required for engineering launch\n'
  fi
  check_second_judge
  local commit
  commit="$(frozen_legacy_commit || true)"
  if [[ -n "${commit}" ]]; then
    printf 'FROZEN_LEGACY_COMMIT=%s\n' "${commit}"
  fi
  cd "${CSI_PAIRS_ROOT}"
  exec "${CSI_PAIRS_PYTHON}" -B -m formal_v2.formal_cli run-evaluation \
    --config "${CSI_PAIRS_FORMAL_CONFIG}" \
    --dataset "${dataset}" \
    --output "${output}"
}

downstream_present() {
  local output=$1
  local relative
  for relative in risk path external_baselines representation_baselines controls scene_id evaluation/retention claims; do
    if [[ -e "${output}/${relative}" || -L "${output}/${relative}" ]]; then
      printf 'DOWNSTREAM_PRESENT=%s (idempotent re-run allowed)\n' "${relative}"
    fi
  done
  return 0
}

supervise_evaluation() {
  local output=${CSI_PAIRS_FORMAL_OUTPUT:?set CSI_PAIRS_FORMAL_OUTPUT}
  local supervision="${CSI_PAIRS_SUPERVISION_ROOT:-${output}/supervision}"
  mkdir -p "${supervision}"
  exec 9>"${supervision}/supervisor.lock"
  if ! command -v flock >/dev/null 2>&1; then
    printf 'SUPERVISOR_REFUSAL=flock_required_for_single_writer_lock\n' >&2
    exit 97
  fi
  if ! flock -n 9; then
    printf 'SUPERVISOR_ALREADY_ACTIVE\n'
    exit 0
  fi
  downstream_present "${output}"
  local attempt=0
  local code=0
  while (( attempt < MAX_EVALUATION_ATTEMPTS )); do
    attempt=$((attempt + 1))
    printf 'AUTO_RESUME=EVALUATION_ATTEMPT %s/%s\n' "${attempt}" "${MAX_EVALUATION_ATTEMPTS}"
    set +e
    "${CSI_PAIRS_PYTHON}" -B -m formal_v2.formal_cli run-evaluation \
      --config "${CSI_PAIRS_FORMAL_CONFIG}" \
      --dataset "${CSI_PAIRS_DATASET:?set CSI_PAIRS_DATASET}" \
      --output "${output}"
    code=$?
    set -e
    if (( code == 0 )); then
      printf 'AUTO_RESUME=COMPLETE\n'
      exit 0
    fi
    downstream_present "${output}"
    printf 'AUTO_RESUME=RETRY_AFTER_EVALUATION_EXIT code=%s\n' "${code}"
    if (( attempt < MAX_EVALUATION_ATTEMPTS )); then
      sleep "${RETRY_SECONDS}"
    fi
  done
  printf 'AUTO_RESUME=EXHAUSTED code=%s\n' "${code}"
  exit "${code}"
}

command="${1:-help}"
case "${command}" in
  check-p0)
    check_p0
    ;;
  check-p1)
    check_p1
    ;;
  launch)
    launch_evaluation
    ;;
  supervise)
    check_p0
    if [[ "${CSI_PAIRS_LAUNCH_TIER}" == "p1" || "${CSI_PAIRS_REQUIRE_P1}" == "1" ]]; then
      check_p1
    fi
    check_second_judge
    supervise_evaluation
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
