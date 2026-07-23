#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
  echo "Usage: $0 /path/to/chunk.json" >&2
  exit 2
fi

source /home/gridex/.venvs/gridex/bin/activate
cd /mnt/c/Users/Admin/Documents/GitHub/GRIDEX

export CUDA_VISIBLE_DEVICES=0

chunk="$1"
base="$(basename "${chunk}" .json)"
result="/mnt/c/Users/Admin/Documents/GitHub/GRIDEX/data/processed/q1_grpo1_parts/${base}.jsonl"
log="/mnt/c/Users/Admin/Documents/GitHub/GRIDEX/training_logs/q1_grpo1_${base}.pt.log"
model="/home/gridex/training_outputs/grpo1/grpo1_q1_5090_safe_v2/v0-20260717-140034/checkpoint-200-merged"
system_prompt="As an expert in deepfake speech spectrogram forensics, you can detect regions containing deepfake artifacts by analysing spectrogram segments. Return only the JSON array of your three chosen region IDs."

rm -f "${result}"

swift infer \
  --model "${model}" \
  --infer_backend pt \
  --val_dataset "${chunk}" \
  --val_dataset_shuffle false \
  --result_path "${result}" \
  --system "${system_prompt}" \
  --max_length 24576 \
  --temperature 0.8 \
  --top_p 0.8 \
  --max_new_tokens 128 \
  --write_batch_size 100 \
  >"${log}" 2>&1

wc -l "${result}"
