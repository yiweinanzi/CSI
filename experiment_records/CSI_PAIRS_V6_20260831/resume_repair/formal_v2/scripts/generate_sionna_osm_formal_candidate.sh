#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${CSI_PAIRS_SIONNA_PYTHON:-${PROJECT_ROOT}/formal_v2/external_adapters/.runtime-sionna/venv/bin/python}"
RUNTIME_ROOT="${PROJECT_ROOT}/formal_v2/external_adapters/.runtime-sionna"
OUTPUT_ROOT_INPUT="${1:?usage: generate_sionna_osm_formal_candidate.sh OUTPUT_ROOT [RAW_OSM_CACHE]}"
OUTPUT_PARENT="$(cd "$(dirname "${OUTPUT_ROOT_INPUT}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_PARENT}/$(basename "${OUTPUT_ROOT_INPUT}")"
RAW_CACHE="${2:-}"
CONFIG="${CSI_PAIRS_SIONNA_CONFIG:-${PROJECT_ROOT}/formal_v2/configs/sionna_osm_formal_candidate_v5.json}"
RENDER_WORKERS="${CSI_PAIRS_RENDER_WORKERS:-8}"
RENDER_RETRIES="${CSI_PAIRS_RENDER_RETRIES:-2}"
RESUME="${CSI_PAIRS_RESUME:-0}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"

if [[ ! -f "${CONFIG}" || -L "${CONFIG}" ]]; then
  echo "Sionna candidate config must be a regular non-symlink file: ${CONFIG}" >&2
  exit 7
fi
SCENE_COUNT="$(PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" -B -m formal_v2.sionna_osm_candidate scene-count --config "${CONFIG}")"
if [[ ! "${SCENE_COUNT}" =~ ^[0-9]+$ ]] || (( SCENE_COUNT < 1 )); then
  echo "Sionna candidate config produced an invalid scene count" >&2
  exit 8
fi
if [[ ! "${RENDER_WORKERS}" =~ ^[0-9]+$ ]] || (( RENDER_WORKERS < 1 || RENDER_WORKERS > SCENE_COUNT )); then
  echo "CSI_PAIRS_RENDER_WORKERS must be an integer from 1 through ${SCENE_COUNT}" >&2
  exit 4
fi
if [[ ! "${RENDER_RETRIES}" =~ ^[0-9]+$ ]] || (( RENDER_RETRIES > 5 )); then
  echo "CSI_PAIRS_RENDER_RETRIES must be an integer from 0 through 5" >&2
  exit 10
fi

if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "CSI_PAIRS_RESUME must be 0 or 1" >&2
  exit 9
fi
if [[ -e "${OUTPUT_ROOT}" && "${RESUME}" != "1" ]]; then
  echo "refusing to overwrite Sionna candidate root: ${OUTPUT_ROOT}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Sionna Python is unavailable; run formal_v2/external_adapters/setup_sionna.sh first" >&2
  exit 3
fi
DRJIT_LIBLLVM_PATH="$(
  PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" -B -m formal_v2.sionna_runtime_lock \
    --project-root "${PROJECT_ROOT}" \
    --runtime-root "${RUNTIME_ROOT}" \
    environment
)"
export DRJIT_LIBLLVM_PATH

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/shards" \
  "${OUTPUT_ROOT}/runtime-cache/drjit"
ASSET_ARGS=(
  -m formal_v2.sionna_osm_candidate prepare-assets
  --config "${CONFIG}"
  --output "${OUTPUT_ROOT}/assets"
)
if [[ -n "${RAW_CACHE}" ]]; then
  if [[ ! -d "${RAW_CACHE}" ]]; then
    echo "raw OSM cache is not a directory: ${RAW_CACHE}" >&2
    exit 5
  fi
  ASSET_ARGS+=(--raw-cache "${RAW_CACHE}")
fi

cd "${PROJECT_ROOT}"
if [[ "${RESUME}" == "1" && -d "${OUTPUT_ROOT}/assets" ]] && \
  "${PYTHON_BIN}" -B -m formal_v2.sionna_osm_candidate validate-assets \
    --asset-root "${OUTPUT_ROOT}/assets" --config "${CONFIG}" \
    >"${OUTPUT_ROOT}/logs/validate-assets-${RUN_ID}.log" 2>&1; then
  printf 'resume_assets=validated\n'
else
  if [[ -e "${OUTPUT_ROOT}/assets" ]]; then
    recovery="${OUTPUT_ROOT}/recovery/${RUN_ID}"
    mkdir -p "${recovery}"
    mv "${OUTPUT_ROOT}/assets" "${recovery}/assets-incomplete"
    recovered_raw_cache="${recovery}/assets-incomplete/raw_osm"
    if [[ -z "${RAW_CACHE}" && -d "${recovered_raw_cache}" ]]; then
      ASSET_ARGS+=(--raw-cache "${recovered_raw_cache}")
      printf 'resume_raw_osm_cache=%s\n' "${recovered_raw_cache}"
    fi
  fi
  "${PYTHON_BIN}" -B "${ASSET_ARGS[@]}" \
    >"${OUTPUT_ROOT}/logs/prepare-assets-${RUN_ID}.log" 2>&1
fi

SHARDS=()
PENDING_INDICES=()
recovery=""
for ((scene_index = 0; scene_index < SCENE_COUNT; scene_index++)); do
  printf -v shard_name 'bank-%03d' "${scene_index}"
  shard="${OUTPUT_ROOT}/shards/${shard_name}.npz"
  SHARDS+=("${shard}")
  shard_manifest="${shard%.npz}.manifest.json"
  if [[ "${RESUME}" == "1" && -f "${shard}" && ! -L "${shard}" \
      && -f "${shard_manifest}" && ! -L "${shard_manifest}" ]] && \
    "${PYTHON_BIN}" -B -m formal_v2.sionna_osm_candidate validate-shard \
      --asset-root "${OUTPUT_ROOT}/assets" \
      --shard "${shard}" \
      --scene-start "${scene_index}" \
      --scene-end "$((scene_index + 1))" \
      --shard-index "${scene_index}" \
      >"${OUTPUT_ROOT}/logs/validate-${shard_name}-${RUN_ID}.log" 2>&1; then
    printf 'resume_bank_%03d=validated\n' "${scene_index}"
    continue
  fi
  if [[ -e "${shard}" || -e "${shard%.npz}.manifest.json" ]]; then
    if [[ -z "${recovery}" ]]; then
      recovery="${OUTPUT_ROOT}/recovery/${RUN_ID}"
      mkdir -p "${recovery}"
    fi
    if [[ -e "${shard}" ]]; then
      mv "${shard}" "${recovery}/${shard_name}-incomplete.npz"
    fi
    if [[ -e "${shard%.npz}.manifest.json" ]]; then
      mv "${shard%.npz}.manifest.json" \
        "${recovery}/${shard_name}-incomplete.manifest.json"
    fi
  fi
  PENDING_INDICES+=("${scene_index}")
done

launch_bank() {
  local scene_index="$1"
  local attempt="$2"
  local shard_name shard log_path cache_home pid
  printf -v shard_name 'bank-%03d' "${scene_index}"
  shard="${OUTPUT_ROOT}/shards/${shard_name}.npz"
  printf -v log_path '%s/logs/render-%s-%s-attempt-%02d.log' \
    "${OUTPUT_ROOT}" "${shard_name}" "${RUN_ID}" "${attempt}"
  cache_home="${OUTPUT_ROOT}/runtime-cache/drjit/${shard_name}"
  mkdir -p "${cache_home}"
  HOME="${cache_home}" "${PYTHON_BIN}" -B -m formal_v2.sionna_osm_candidate render-shard \
    --asset-root "${OUTPUT_ROOT}/assets" \
    --output "${shard}" \
    --scene-start "${scene_index}" \
    --scene-end "$((scene_index + 1))" \
    --shard-index "${scene_index}" \
    >"${log_path}" 2>&1 &
  pid=$!
  ACTIVE_BANK_BY_PID["${pid}"]="${scene_index}"
  ACTIVE_ATTEMPT_BY_PID["${pid}"]="${attempt}"
  ACTIVE_LOG_BY_PID["${pid}"]="${log_path}"
}

# Every child writes one bank through an atomic NPZ plus an authenticated
# manifest. At most RENDER_WORKERS banks are in flight, so an interruption
# loses only unfinished banks and CSI_PAIRS_RESUME can validate the rest.
pending_offset=0
failed=0
declare -A ATTEMPTS_BY_BANK=()
declare -A ACTIVE_BANK_BY_PID=()
declare -A ACTIVE_ATTEMPT_BY_PID=()
declare -A ACTIVE_LOG_BY_PID=()
while (( pending_offset < ${#PENDING_INDICES[@]} \
    || ${#ACTIVE_BANK_BY_PID[@]} > 0 )); do
  while (( failed == 0 && ${#ACTIVE_BANK_BY_PID[@]} < RENDER_WORKERS \
      && pending_offset < ${#PENDING_INDICES[@]} )); do
    scene_index="${PENDING_INDICES[${pending_offset}]}"
    pending_offset=$((pending_offset + 1))
    attempt=$(( ${ATTEMPTS_BY_BANK[${scene_index}]:-0} + 1 ))
    ATTEMPTS_BY_BANK["${scene_index}"]="${attempt}"
    launch_bank "${scene_index}" "${attempt}"
  done

  reaped=0
  for pid in "${!ACTIVE_BANK_BY_PID[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      continue
    fi
    scene_index="${ACTIVE_BANK_BY_PID[${pid}]}"
    attempt="${ACTIVE_ATTEMPT_BY_PID[${pid}]}"
    log_path="${ACTIVE_LOG_BY_PID[${pid}]}"
    if wait "${pid}"; then
      rc=0
    else
      rc=$?
    fi
    unset 'ACTIVE_BANK_BY_PID['"${pid}"']'
    unset 'ACTIVE_ATTEMPT_BY_PID['"${pid}"']'
    unset 'ACTIVE_LOG_BY_PID['"${pid}"']'
    reaped=1
    if (( rc == 0 )); then
      continue
    fi
    if (( attempt <= RENDER_RETRIES )); then
      printf 'retry_bank_%03d_attempt_%02d_exit_%d_log=%s\n' \
        "${scene_index}" "${attempt}" "${rc}" "${log_path}" >&2
      PENDING_INDICES+=("${scene_index}")
    else
      printf 'Sionna atomic bank %03d failed after %d attempts; last_exit=%d log=%s\n' \
        "${scene_index}" "${attempt}" "${rc}" "${log_path}" >&2
      failed=1
    fi
  done
  if (( failed != 0 && ${#ACTIVE_BANK_BY_PID[@]} == 0 )); then
    break
  fi
  if (( reaped == 0 && ${#ACTIVE_BANK_BY_PID[@]} > 0 )); then
    sleep 0.2
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 6
fi

MERGE_ARGS=(
  -m formal_v2.sionna_osm_candidate merge
  --config "${CONFIG}"
  --asset-root "${OUTPUT_ROOT}/assets"
  --output "${OUTPUT_ROOT}/dataset.npz"
)
for shard in "${SHARDS[@]}"; do
  MERGE_ARGS+=(--shard "${shard}")
done
if [[ "${RESUME}" == "1" \
    && -f "${OUTPUT_ROOT}/dataset.npz" \
    && ! -L "${OUTPUT_ROOT}/dataset.npz" \
    && -f "${OUTPUT_ROOT}/dataset.generation.json" \
    && ! -L "${OUTPUT_ROOT}/dataset.generation.json" ]] && \
  "${PYTHON_BIN}" -B -m formal_v2.sionna_osm_candidate validate-generation \
    --config "${CONFIG}" \
    --asset-root "${OUTPUT_ROOT}/assets" \
    --dataset "${OUTPUT_ROOT}/dataset.npz" \
    >"${OUTPUT_ROOT}/logs/validate-generation-${RUN_ID}.log" 2>&1; then
  printf 'resume_dataset=validated\n'
  printf 'candidate=%s\n' "${OUTPUT_ROOT}/dataset.npz"
  printf 'render_workers=%s\n' "${RENDER_WORKERS}"
  printf 'scene_count=%s\n' "${SCENE_COUNT}"
  exit 0
fi
if [[ -e "${OUTPUT_ROOT}/dataset.npz" || -e "${OUTPUT_ROOT}/dataset.generation.json" ]]; then
  recovery="${OUTPUT_ROOT}/recovery/${RUN_ID}"
  mkdir -p "${recovery}"
  if [[ -e "${OUTPUT_ROOT}/dataset.npz" ]]; then
    mv "${OUTPUT_ROOT}/dataset.npz" "${recovery}/dataset-incomplete.npz"
  fi
  if [[ -e "${OUTPUT_ROOT}/dataset.generation.json" ]]; then
    mv "${OUTPUT_ROOT}/dataset.generation.json" \
      "${recovery}/dataset-incomplete.generation.json"
  fi
fi
"${PYTHON_BIN}" -B "${MERGE_ARGS[@]}" \
  >"${OUTPUT_ROOT}/logs/merge-${RUN_ID}.log" 2>&1

printf 'candidate=%s\n' "${OUTPUT_ROOT}/dataset.npz"
printf 'render_workers=%s\n' "${RENDER_WORKERS}"
printf 'scene_count=%s\n' "${SCENE_COUNT}"
printf 'next=run inspect-data, then zero-tolerance verify-data before qualification\n'
