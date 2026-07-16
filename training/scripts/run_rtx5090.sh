#!/usr/bin/env bash
set -euo pipefail

# Conservative single-GPU profile for an RTX 5090 (32 GB).
# The original stage scripts keep their multi-GPU/paper defaults; this wrapper
# supplies memory-safe overrides without duplicating the training commands.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-}"
if [[ -z "${STAGE}" ]]; then
  echo "Usage: bash training/scripts/run_rtx5090.sh {sft1|grpo1|sft2|grpo2} [stage arguments...]" >&2
  exit 2
fi
shift

export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
export DEEPSPEED="${DEEPSPEED:-none}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

case "${STAGE}" in
  sft1)
    # Original effective batch: 4 GPUs * 10 samples. Preserve it through
    # accumulation while keeping the instantaneous batch small.
    export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
    export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-40}"
    export MAX_LENGTH="${MAX_LENGTH:-768}"
    export MAX_PIXELS="${MAX_PIXELS:-524288}"
    exec bash "${SCRIPT_DIR}/01_stage1_sft_query1.sh" "$@"
    ;;
  grpo1)
    # Two rollouts is the practical minimum group size. A low vLLM reservation
    # leaves room for the policy/reference model and backward pass.
    export NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
    export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
    export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
    export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
    export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-128}"
    export MAX_LENGTH="${MAX_LENGTH:-512}"
    exec bash "${SCRIPT_DIR}/02_stage2_grpo1.sh" "$@"
    ;;
  sft2)
    # Original effective batch: 2 GPUs * 4 samples * 8 accumulation steps.
    export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
    export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-64}"
    export MAX_PIXELS="${MAX_PIXELS:-524288}"
    exec bash "${SCRIPT_DIR}/03_stage2_sft_query2.sh" "$@"
    ;;
  grpo2)
    export NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
    export INTERSPEECH_GROUP_SIZE="${INTERSPEECH_GROUP_SIZE:-2}"
    export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
    export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
    export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
    export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-384}"
    export MAX_LENGTH="${MAX_LENGTH:-512}"
    exec bash "${SCRIPT_DIR}/04_stage3_grpo2_curriculum.sh" "$@"
    ;;
  *)
    echo "Unknown stage: ${STAGE}. Expected sft1, grpo1, sft2, or grpo2." >&2
    exit 2
    ;;
esac
