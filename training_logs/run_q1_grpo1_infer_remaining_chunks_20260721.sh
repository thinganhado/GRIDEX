#!/usr/bin/env bash
set -euo pipefail

source /home/gridex/.venvs/gridex/bin/activate
cd /mnt/c/Users/Admin/Documents/GitHub/GRIDEX

export CUDA_VISIBLE_DEVICES=0

CHUNK_DIR="/mnt/c/Users/Admin/Documents/GitHub/GRIDEX/data/processed/q1_grpo1_chunks"
OUT_DIR="/mnt/c/Users/Admin/Documents/GitHub/GRIDEX/data/processed/q1_grpo1_parts"
RUN_LOG="/mnt/c/Users/Admin/Documents/GitHub/GRIDEX/training_logs/q1_grpo1_infer_remaining_chunks_20260721.log"
MODEL="/home/gridex/training_outputs/grpo1/grpo1_q1_5090_safe_v2/v0-20260717-140034/checkpoint-200-merged"
SYSTEM_PROMPT="As an expert in deepfake speech spectrogram forensics, you can detect regions containing deepfake artifacts by analysing spectrogram segments. Return only the JSON array of your three chosen region IDs."

mkdir -p "${OUT_DIR}"
exec >>"${RUN_LOG}" 2>&1

date

if (($#)); then
  chunks=("$@")
else
  chunks=("${CHUNK_DIR}"/stage1_query1_train_*.json)
fi

for chunk in "${chunks[@]}"; do
  base="$(basename "${chunk}" .json)"
  result="${OUT_DIR}/${base}.jsonl"
  expected="$(python3 - "${chunk}" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))))
PY
)"

  if [[ -e "${result}" ]]; then
    actual="$(wc -l < "${result}")"
    if [[ "${actual}" == "${expected}" ]]; then
      echo "SKIP complete ${base}: ${actual}/${expected}"
      continue
    fi
    echo "ERROR partial existing result ${result}: ${actual}/${expected}" >&2
    exit 1
  fi

  echo "START ${base}: expected=${expected}"
  swift infer \
    --model "${MODEL}" \
    --infer_backend vllm \
    --val_dataset "${chunk}" \
    --val_dataset_shuffle false \
    --result_path "${result}" \
    --system "${SYSTEM_PROMPT}" \
    --max_length 24576 \
    --temperature 0.8 \
    --top_p 0.8 \
    --max_new_tokens 128 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 24576

  actual="$(wc -l < "${result}")"
  echo "DONE ${base}: ${actual}/${expected}"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ERROR incomplete result ${result}: ${actual}/${expected}" >&2
    exit 1
  fi
done

date
echo "ALL_CHUNKS_DONE"
