#!/usr/bin/env bash
set -euo pipefail

source /home/gridex/.venvs/gridex/bin/activate
cd /mnt/c/Users/Admin/Documents/GitHub/GRIDEX

export CUDA_VISIBLE_DEVICES=0

P1_JSONL="/mnt/c/Users/Admin/Documents/GitHub/GRIDEX/data/processed/q1_grpo1_raw_predictions.jsonl"
LOG_PATH="/mnt/c/Users/Admin/Documents/GitHub/GRIDEX/training_logs/q1_grpo1_infer_20260721.log"

if [[ -e "${P1_JSONL}" ]]; then
  echo "Output already exists; MS-Swift appends to result_path: ${P1_JSONL}" >&2
  exit 1
fi

exec >"${LOG_PATH}" 2>&1

date
swift infer \
  --model /home/gridex/training_outputs/grpo1/grpo1_q1_5090_safe_v2/v0-20260717-140034/checkpoint-200-merged \
  --infer_backend vllm \
  --val_dataset /mnt/c/Users/Admin/Documents/GitHub/GRIDEX/data/processed/stage1_query1_train_swift.json \
  --val_dataset_shuffle false \
  --result_path "${P1_JSONL}" \
  --system "As an expert in deepfake speech spectrogram forensics, you can detect regions containing deepfake artifacts by analysing spectrogram segments. Return only the JSON array of your three chosen region IDs." \
  --max_length 24576 \
  --temperature 0.8 \
  --top_p 0.8 \
  --max_new_tokens 128 \
  --vllm_gpu_memory_utilization 0.9 \
  --vllm_max_model_len 24576
date
