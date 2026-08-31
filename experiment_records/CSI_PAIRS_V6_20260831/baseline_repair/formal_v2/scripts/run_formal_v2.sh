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
COMPUTE_PLAN="${CSI_PAIRS_COMPUTE_PLAN:?set CSI_PAIRS_COMPUTE_PLAN to the reviewed compute-plan JSON}"
VERIFIER_MANIFEST="${CSI_PAIRS_VERIFIER_MANIFEST:?set CSI_PAIRS_VERIFIER_MANIFEST to an independent RT verifier manifest}"
ADAPTER_MANIFEST="${CSI_PAIRS_EXTERNAL_ADAPTER_MANIFEST:?set CSI_PAIRS_EXTERNAL_ADAPTER_MANIFEST}"
CONTROL_MANIFEST="${CSI_PAIRS_RESOURCE_CONTROL_MANIFEST:-${PROJECT_ROOT}/formal_v2/external_adapters/resource_controls_v3.json}"
SCENE_ID_MANIFEST="${CSI_PAIRS_SCENE_ID_MANIFEST:-builtin:sigmap-scene-id-v1}"
EXTERNAL_VALIDITY_MANIFEST="${CSI_PAIRS_EXTERNAL_VALIDITY_MANIFEST:?set CSI_PAIRS_EXTERNAL_VALIDITY_MANIFEST}"
LITERATURE_MANIFEST="${CSI_PAIRS_LITERATURE_RESOURCE_MANIFEST:?set CSI_PAIRS_LITERATURE_RESOURCE_MANIFEST}"
RT_CALIBRATION_MANIFEST="${CSI_PAIRS_RT_CALIBRATION_MANIFEST:?set CSI_PAIRS_RT_CALIBRATION_MANIFEST}"
SHUFFLED_PAIR_MANIFEST="${CSI_PAIRS_SHUFFLED_PAIR_MANIFEST:-${PROJECT_ROOT}/formal_v2/external_adapters/shuffled_pair_control_v3.json}"
RETENTION_MANIFEST="${CSI_PAIRS_RETENTION_MANIFEST:-${PROJECT_ROOT}/formal_v2/external_adapters/retention_control_v3.json}"
DATASET_ARGS=()
if [[ -n "${CSI_PAIRS_FORMAL_DATASET:-}" ]]; then
  DATASET_ARGS=(--dataset "${CSI_PAIRS_FORMAL_DATASET}")
fi
APPROVAL_ARGS=()
COMMAND="prepare-full-run"
if [[ "${PHASE}" == "run" ]]; then
  COMMAND="all"
  APPROVAL_MANIFEST="${CSI_PAIRS_HUMAN_APPROVAL_MANIFEST:?set CSI_PAIRS_HUMAN_APPROVAL_MANIFEST to the external approval JSON}"
  APPROVAL_ARGS=(--approval-manifest "${APPROVAL_MANIFEST}")
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -B -m formal_v2.formal_cli "${COMMAND}" \
  --config "${CONFIG}" \
  --output "${OUTPUT}" \
  --compute-plan "${COMPUTE_PLAN}" \
  --verifier-manifest "${VERIFIER_MANIFEST}" \
  --adapter-manifest "${ADAPTER_MANIFEST}" \
  --control-manifest "${CONTROL_MANIFEST}" \
  --scene-id-manifest "${SCENE_ID_MANIFEST}" \
  --external-validity-manifest "${EXTERNAL_VALIDITY_MANIFEST}" \
  --literature-resource-manifest "${LITERATURE_MANIFEST}" \
  --rt-calibration-manifest "${RT_CALIBRATION_MANIFEST}" \
  --shuffled-pair-manifest "${SHUFFLED_PAIR_MANIFEST}" \
  --retention-manifest "${RETENTION_MANIFEST}" \
  "${APPROVAL_ARGS[@]}" \
  "${DATASET_ARGS[@]}"
