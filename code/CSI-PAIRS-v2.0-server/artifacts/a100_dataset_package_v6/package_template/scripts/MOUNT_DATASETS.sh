#!/usr/bin/env bash

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  CSI_PAIRS_DATA_SUITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
  echo "MOUNT_DATASETS.sh must be sourced from Bash." >&2
  return 2 2>/dev/null || exit 2
fi

export CSI_PAIRS_DATA_SUITE_ROOT
export CSI_PAIRS_RAW_OSM_ROOT="$CSI_PAIRS_DATA_SUITE_ROOT/primary_raw_inputs/CSI-PAIRS-A100-input-v2/raw_osm"
export CSI_PAIRS_CPU_CANDIDATE="$CSI_PAIRS_DATA_SUITE_ROOT/cpu_llvm22_34bank_candidate/dataset.npz"
export CSI_PAIRS_FORMAL_SPLIT_LEDGER="$CSI_PAIRS_DATA_SUITE_ROOT/docs/FORMAL_MAIN_SPLIT_LEDGER.json"
export CSI_PAIRS_A100_FIXTURE="$CSI_PAIRS_DATA_SUITE_ROOT/a100_sionna_fixture/csi_pairs_v2_1_v6_sionna_rt_dual_a100/csi_pairs_v2_1_v6_sionna_rt_dual_a100.npz"
export DEEP_MIMO_ROOT="$CSI_PAIRS_DATA_SUITE_ROOT/external_public_datasets/external_wireless/DeepMIMO"
export URBAN_MIMO_ROOT="$CSI_PAIRS_DATA_SUITE_ROOT/external_public_datasets/external_wireless/UrbanMIMOMap"
export RADIO_MAP_ROOT="$CSI_PAIRS_DATA_SUITE_ROOT/external_public_datasets/external_wireless/RadioMapSeer"
export DEEP_SENSE_ROOT="$CSI_PAIRS_DATA_SUITE_ROOT/external_public_datasets/external_wireless/WWM/alternatives/DeepSense6G"

echo "CSI_PAIRS_DATA_SUITE_ROOT=$CSI_PAIRS_DATA_SUITE_ROOT"
echo "CSI_PAIRS_RAW_OSM_ROOT=$CSI_PAIRS_RAW_OSM_ROOT"
echo "CSI_PAIRS_CPU_CANDIDATE=$CSI_PAIRS_CPU_CANDIDATE"
echo "CSI_PAIRS_FORMAL_SPLIT_LEDGER=$CSI_PAIRS_FORMAL_SPLIT_LEDGER"
echo "CSI_PAIRS_A100_FIXTURE=$CSI_PAIRS_A100_FIXTURE"
echo "DEEP_MIMO_ROOT=$DEEP_MIMO_ROOT"
echo "URBAN_MIMO_ROOT=$URBAN_MIMO_ROOT"
echo "RADIO_MAP_ROOT=$RADIO_MAP_ROOT"
echo "DEEP_SENSE_ROOT=$DEEP_SENSE_ROOT"
