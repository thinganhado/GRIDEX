#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PAIR_CSV="${PAIR_CSV:?Set PAIR_CSV to a CSV with real_path and fake_path columns}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
EXTRA_ARGS=()
if [[ "${ASR_VAD_FILTER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--vad-filter)
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

python "${REPO_ROOT}/training/tools/prepare_assets.py" \
  --pairs-csv "${PAIR_CSV}" \
  --data-root "${DATA_ROOT}" \
  --asr-model "${ASR_MODEL:-small}" \
  --language "${ASR_LANGUAGE:-en}" \
  --asr-device "${ASR_DEVICE:-auto}" \
  --asr-compute-type "${ASR_COMPUTE_TYPE:-auto}" \
  "${EXTRA_ARGS[@]}"

python "${REPO_ROOT}/training/tools/build_region_stats.py" \
  --pairs-csv "${PAIR_CSV}" \
  --spectrogram-root "${DATA_ROOT}/spectrograms/grid" \
  --output-csv "${DATA_ROOT}/metadata/region_diff_stats.csv"
