#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

FACTUAL_CSV="${FACTUAL_CSV:-${GRIDEX_ROOT}/dataset.csv}"
ORDER_CSV="${ORDER_CSV:-${DATA_ROOT}/metadata/region_diff_stats.csv}"
SPEC_ROOT="${SPEC_ROOT:-${DATA_ROOT}/spectrograms/grid}"
MFA_ROOT="${MFA_ROOT:-${DATA_ROOT}/transcripts/mfa}"
P1_PRED_JSON="${P1_PRED_JSON:?Set P1_PRED_JSON to unfiltered raw Query-1 rollout/inference outputs}"
GRPO2_GT_JSON="${GRPO2_GT_JSON:-${DATA_ROOT}/processed/grpo2_gt_swift.json}"
GRPO2_PRED_JSON="${GRPO2_PRED_JSON:-${DATA_ROOT}/processed/grpo2_pred_aligned.json}"

mkdir -p "$(dirname "${GRPO2_GT_JSON}")" "$(dirname "${GRPO2_PRED_JSON}")"

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

python "${TRAINING_ROOT}/tools/build_swift_grpo_prompt2_dataset.py" \
  --input-json "${FACTUAL_CSV}" \
  --input-format csv \
  --gt-csv "${FACTUAL_CSV}" \
  --diff-csv "${ORDER_CSV}" \
  --spec-root "${SPEC_ROOT}" \
  --mfa-root "${MFA_ROOT}" \
  --output-json "${GRPO2_GT_JSON}"

python "${TRAINING_ROOT}/tools/build_swift_grpo_prompt2_dataset_from_p1_pred.py" \
  --input-json "${P1_PRED_JSON}" \
  --factual-csv "${FACTUAL_CSV}" \
  --gt-csv "${FACTUAL_CSV}" \
  --diff-csv "${ORDER_CSV}" \
  --spec-root "${SPEC_ROOT}" \
  --mfa-root "${MFA_ROOT}" \
  --sample-id-contains "${SAMPLE_ID_CONTAINS:-_LA_T_}" \
  --output-json "${GRPO2_PRED_JSON}"

echo "Built GT-ID Q2 dataset: ${GRPO2_GT_JSON}"
echo "Built predicted-ID aligned Q2 dataset: ${GRPO2_PRED_JSON}"
