#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CSV_IN="${CSV_IN:-${GRIDEX_ROOT}/dataset.csv}"
SFT_JSON_SWIFT="${SFT_JSON_SWIFT:-${DATA_ROOT}/processed/stage1_query1_train_swift.json}"
SFT_VAL_JSON_SWIFT="${SFT_VAL_JSON_SWIFT:-${DATA_ROOT}/processed/stage1_query1_val_swift.json}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SPLIT_BY_PATH="${SPLIT_BY_PATH:-1}"
STRICT_PATH_SPLIT="${STRICT_PATH_SPLIT:-1}"
EXPECTED_TRAIN_SAMPLES="${EXPECTED_TRAIN_SAMPLES:-10320}"
EXPECTED_VAL_SAMPLES="${EXPECTED_VAL_SAMPLES:-10192}"
USER_PROMPT="Select the top 3 regions that most likely contain spoof artifacts, ordered from most to least prominent spoof artifact evidence."
REGIONS_COL="${REGIONS_COL:-}"
PRESERVE_INPUT_REGION_ORDER="${PRESERVE_INPUT_REGION_ORDER:-0}"
IMAGE_COL="${IMAGE_COL:-img_path}"
SAMPLE_ID_COL="${SAMPLE_ID_COL:-sample_id}"
REGION_ID_COL="${REGION_ID_COL:-region_id}"
ORDER_CSV="${ORDER_CSV:-${DATA_ROOT}/metadata/region_diff_stats.csv}"
ORDER_SAMPLE_ID_COL="${ORDER_SAMPLE_ID_COL:-image}"
ORDER_REGION_ID_COL="${ORDER_REGION_ID_COL:-region_id}"
ORDER_SCORE_COL="${ORDER_SCORE_COL:-overlap_pixels}"
ORDER_METHOD_COL="${ORDER_METHOD_COL:-method}"
ORDER_METHOD="${ORDER_METHOD:-grid}"
IMAGE_DIR="${IMAGE_DIR:-${DATA_ROOT}/spectrograms/grid}"
IMAGE_SUFFIX="${IMAGE_SUFFIX:-_grid_img_edge_number_axes.png}"

EXTRA_ARGS=()
if [ -n "${REGIONS_COL}" ]; then
  EXTRA_ARGS+=(--regions-col "${REGIONS_COL}")
fi
if [[ "${PRESERVE_INPUT_REGION_ORDER}" == "1" ]]; then
  EXTRA_ARGS+=(--preserve-input-region-order)
fi
if [[ "${SPLIT_BY_PATH}" == "1" ]]; then
  EXTRA_ARGS+=(--split-by-path)
fi
if [[ "${STRICT_PATH_SPLIT}" == "1" ]]; then
  EXTRA_ARGS+=(--strict-path-split)
fi

if [[ ! -f "${ORDER_CSV}" ]]; then
  echo "ERROR: ORDER_CSV is required and was not found: ${ORDER_CSV}" >&2
  exit 1
fi

mkdir -p "$(dirname "${SFT_JSON_SWIFT}")"

python "${TRAINING_ROOT}/tools/build_swift_sft_query1_dataset.py" \
  --input-csv "${CSV_IN}" \
  --image-col "${IMAGE_COL}" \
  --sample-id-col "${SAMPLE_ID_COL}" \
  --region-id-col "${REGION_ID_COL}" \
  --order-csv "${ORDER_CSV}" \
  --order-sample-id-col "${ORDER_SAMPLE_ID_COL}" \
  --order-region-id-col "${ORDER_REGION_ID_COL}" \
  --order-score-col "${ORDER_SCORE_COL}" \
  --order-method-col "${ORDER_METHOD_COL}" \
  --order-method "${ORDER_METHOD}" \
  --require-order-scores \
  --verify-order \
  --require-images \
  --image-dir "${IMAGE_DIR}" \
  --image-suffix "${IMAGE_SUFFIX}" \
  --user-prompt "${USER_PROMPT}" \
  --json-array-target \
  --train-json "${SFT_JSON_SWIFT}" \
  --val-json "${SFT_VAL_JSON_SWIFT}" \
  --val-ratio "${VAL_RATIO}" \
  --expected-train-samples "${EXPECTED_TRAIN_SAMPLES}" \
  --expected-val-samples "${EXPECTED_VAL_SAMPLES}" \
  "${EXTRA_ARGS[@]}"

echo "Built dataset: ${SFT_JSON_SWIFT}"
echo "Built validation dataset: ${SFT_VAL_JSON_SWIFT}"
