#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${CSI_PAIRS_PYTHON:-python3}"
PHASE="${CSI_PAIRS_FULL_RUN_PHASE:-}"
if [[ "${PHASE}" != "prepare" && "${PHASE}" != "run" ]]; then
  echo "set CSI_PAIRS_FULL_RUN_PHASE=prepare or CSI_PAIRS_FULL_RUN_PHASE=run" >&2
  exit 5
fi
CONFIG="${CSI_PAIRS_FORMAL_CONFIG:-${PROJECT_ROOT}/formal_v2/configs/formal_v2.json}"
OUTPUT="${CSI_PAIRS_FORMAL_OUTPUT:?set CSI_PAIRS_FORMAL_OUTPUT to an unused output directory}"
ADAPTER_MANIFEST="${CSI_PAIRS_EXTERNAL_ADAPTER_MANIFEST:?set CSI_PAIRS_EXTERNAL_ADAPTER_MANIFEST}"
CONTROL_MANIFEST="${CSI_PAIRS_RESOURCE_CONTROL_MANIFEST:-${PROJECT_ROOT}/formal_v2/external_adapters/resource_controls_v3.json}"
SCENE_ID_MANIFEST="${CSI_PAIRS_SCENE_ID_MANIFEST:-builtin:sigmap-scene-id-v1}"
SHUFFLED_PAIR_MANIFEST="${CSI_PAIRS_SHUFFLED_PAIR_MANIFEST:-${PROJECT_ROOT}/formal_v2/external_adapters/shuffled_pair_control_v3.json}"
RETENTION_MANIFEST="${CSI_PAIRS_RETENTION_MANIFEST:-${PROJECT_ROOT}/formal_v2/external_adapters/retention_control_v3.json}"
DATASET_ARGS=()
if [[ -n "${CSI_PAIRS_FORMAL_DATASET:-}" ]]; then
  DATASET_ARGS=(--dataset "${CSI_PAIRS_FORMAL_DATASET}")
fi
OPTIONAL_ARGS=()
if [[ -n "${CSI_PAIRS_COMPUTE_PLAN:-}" ]]; then
  OPTIONAL_ARGS+=(--compute-plan "${CSI_PAIRS_COMPUTE_PLAN}")
fi
if [[ -n "${CSI_PAIRS_VERIFIER_MANIFEST:-}" ]]; then
  OPTIONAL_ARGS+=(--verifier-manifest "${CSI_PAIRS_VERIFIER_MANIFEST}")
fi
if [[ -n "${CSI_PAIRS_EXTERNAL_VALIDITY_MANIFEST:-}" ]]; then
  OPTIONAL_ARGS+=(--external-validity-manifest "${CSI_PAIRS_EXTERNAL_VALIDITY_MANIFEST}")
fi
if [[ -n "${CSI_PAIRS_LITERATURE_RESOURCE_MANIFEST:-}" ]]; then
  OPTIONAL_ARGS+=(--literature-resource-manifest "${CSI_PAIRS_LITERATURE_RESOURCE_MANIFEST}")
fi
if [[ -n "${CSI_PAIRS_RT_CALIBRATION_MANIFEST:-}" ]]; then
  OPTIONAL_ARGS+=(--rt-calibration-manifest "${CSI_PAIRS_RT_CALIBRATION_MANIFEST}")
fi
APPROVAL_ARGS=()
COMMAND="prepare-full-run"
if [[ "${PHASE}" == "run" ]]; then
  COMMAND="all"
  APPROVAL_MANIFEST="${CSI_PAIRS_LLM_JUDGE_APPROVAL_MANIFEST:?set CSI_PAIRS_LLM_JUDGE_APPROVAL_MANIFEST to the external llm-judge approval JSON}"
  APPROVAL_ARGS=(--approval-manifest "${APPROVAL_MANIFEST}")
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -B -m formal_v2.formal_cli "${COMMAND}" \
  --config "${CONFIG}" \
  --output "${OUTPUT}" \
  --adapter-manifest "${ADAPTER_MANIFEST}" \
  --control-manifest "${CONTROL_MANIFEST}" \
  --scene-id-manifest "${SCENE_ID_MANIFEST}" \
  --shuffled-pair-manifest "${SHUFFLED_PAIR_MANIFEST}" \
  --retention-manifest "${RETENTION_MANIFEST}" \
  "${OPTIONAL_ARGS[@]}" \
  "${APPROVAL_ARGS[@]}" \
  "${DATASET_ARGS[@]}"
