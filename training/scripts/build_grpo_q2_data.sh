#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

FACTUAL_CSV="${FACTUAL_CSV:?Set FACTUAL_CSV to the reviewed dataset_none.csv with supported P labels}"
ORDER_CSV="${ORDER_CSV:-${DATA_ROOT}/metadata/region_diff_stats.csv}"
SPEC_ROOT="${SPEC_ROOT:-${DATA_ROOT}/spectrograms/grid}"
MFA_ROOT="${MFA_ROOT:-${DATA_ROOT}/transcripts/mfa}"
P1_PRED_JSON="${P1_PRED_JSON:?Set P1_PRED_JSON to unfiltered raw Query-1 rollout/inference outputs}"
GRPO2_GT_JSON="${GRPO2_GT_JSON:-${DATA_ROOT}/processed/grpo2_gt_train_swift.json}"
GRPO2_GT_VAL_JSON="${GRPO2_GT_VAL_JSON:-${DATA_ROOT}/processed/grpo2_gt_val_swift.json}"
GRPO2_PRED_JSON="${GRPO2_PRED_JSON:-${DATA_ROOT}/processed/grpo2_pred_aligned.json}"
EXPECTED_TRAIN_SAMPLES="${EXPECTED_TRAIN_SAMPLES:-10320}"
EXPECTED_VAL_SAMPLES="${EXPECTED_VAL_SAMPLES:-10192}"
BUILD_PARALLEL="${BUILD_PARALLEL:-1}"

mkdir -p "$(dirname "${GRPO2_GT_JSON}")" "$(dirname "${GRPO2_GT_VAL_JSON}")" "$(dirname "${GRPO2_PRED_JSON}")"

python - "${FACTUAL_CSV}" <<'PY'
import csv
import sys
from collections import Counter

path = sys.argv[1]
with open(path, "r", encoding="utf-8-sig", newline="") as f:
    labels = Counter(str(row.get("P", "")).strip().lower() for row in csv.DictReader(f))
unsupported = sorted(label for label in labels if label not in {"consonant", "vowel", "none"})
if unsupported:
    raise SystemExit(
        f"Unsupported P values in {path}: {unsupported}. "
        "Use the reviewed P=none factual CSV; do not train Q2 from the legacy unvoiced annotations."
    )
print(f"Validated factual P labels: {dict(labels)}")
PY

python - "${FACTUAL_CSV}" "${MFA_ROOT}" <<'PY'
import csv
import sys
from pathlib import Path

csv_path, mfa_root = Path(sys.argv[1]), Path(sys.argv[2])
with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
    sample_ids = {
        str(row.get("sample_id", "")).strip()
        for row in csv.DictReader(f)
        if str(row.get("sample_id", "")).strip()
    }
missing = sorted(sid for sid in sample_ids if not (mfa_root / f"{sid}.json").is_file())
print(f"Validated MFA transcripts: samples={len(sample_ids)} missing={len(missing)}")
if missing:
    raise SystemExit(f"Missing MFA JSON files under {mfa_root}: {missing[:20]}")
PY

build_gt_train() {
  python "${TRAINING_ROOT}/tools/build_q2_grpo_data.py" \
    --input-json "${FACTUAL_CSV}" \
    --input-format csv \
    --gt-csv "${FACTUAL_CSV}" \
    --diff-csv "${ORDER_CSV}" \
    --spec-root "${SPEC_ROOT}" \
    --mfa-root "${MFA_ROOT}" \
    --sample-id-contains "_LA_T_" \
    --expected-samples "${EXPECTED_TRAIN_SAMPLES}" \
    --output-json "${GRPO2_GT_JSON}"
}

build_gt_val() {
  python "${TRAINING_ROOT}/tools/build_q2_grpo_data.py" \
    --input-json "${FACTUAL_CSV}" \
    --input-format csv \
    --gt-csv "${FACTUAL_CSV}" \
    --diff-csv "${ORDER_CSV}" \
    --spec-root "${SPEC_ROOT}" \
    --mfa-root "${MFA_ROOT}" \
    --sample-id-contains "_LA_D_" \
    --expected-samples "${EXPECTED_VAL_SAMPLES}" \
    --output-json "${GRPO2_GT_VAL_JSON}"
}

build_pred() {
  python "${TRAINING_ROOT}/tools/build_q2_pred_data.py" \
    --input-json "${P1_PRED_JSON}" \
    --factual-csv "${FACTUAL_CSV}" \
    --gt-csv "${FACTUAL_CSV}" \
    --diff-csv "${ORDER_CSV}" \
    --spec-root "${SPEC_ROOT}" \
    --mfa-root "${MFA_ROOT}" \
    --sample-id-contains "${SAMPLE_ID_CONTAINS:-_LA_T_}" \
    --output-json "${GRPO2_PRED_JSON}"
}

if [[ "${BUILD_PARALLEL}" == "1" ]]; then
  echo "Building train, validation, and predicted-ID Q2 datasets in parallel."
  build_gt_train & train_pid=$!
  build_gt_val & val_pid=$!
  build_pred & pred_pid=$!
  status=0
  wait "${train_pid}" || status=1
  wait "${val_pid}" || status=1
  wait "${pred_pid}" || status=1
  if (( status != 0 )); then
    echo "ERROR: one or more Q2 dataset builds failed." >&2
    exit "${status}"
  fi
else
  build_gt_train
  build_gt_val
  build_pred
fi

echo "Built GT-ID Q2 dataset: ${GRPO2_GT_JSON}"
echo "Built GT-ID Q2 validation dataset: ${GRPO2_GT_VAL_JSON}"
echo "Built predicted-ID aligned Q2 dataset: ${GRPO2_PRED_JSON}"
