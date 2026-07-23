#!/usr/bin/env bash
set -euo pipefail

# Single-GPU profiles for an RTX 5090 (32 GB).
# The original stage scripts keep their multi-GPU/paper defaults; this wrapper
# supplies balanced or memory-safe overrides without duplicating the commands.

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
export DEEPSPEED="${DEEPSPEED:-none}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Hugging Face is substantially faster than ModelScope from this host.
export USE_HF="${USE_HF:-1}"
# Xet is severely throttled on the current network; regular HTTP is faster.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

RTX5090_PROFILE="${RTX5090_PROFILE:-balanced}"
if [[ "${RTX5090_PROFILE}" != "balanced" && "${RTX5090_PROFILE}" != "safe" ]]; then
  echo "Unknown RTX5090_PROFILE=${RTX5090_PROFILE}; expected balanced or safe." >&2
  exit 2
fi
echo "RTX 5090 profile: ${RTX5090_PROFILE}"

case "${STAGE}" in
  sft1)
    # Preserve the paper effective batch of 40. Batch 2 improves tensor-core
    # occupancy; the safe profile is available if another process uses VRAM.
    if [[ "${RTX5090_PROFILE}" == "balanced" ]]; then
      export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
      export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-20}"
      export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-6}"
      export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
      export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
    else
      export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
      export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-40}"
      export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
      export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
      export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
    fi
    export DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
    export LOGGING_STEPS="${LOGGING_STEPS:-10}"
    export MAX_LENGTH="${MAX_LENGTH:-768}"
    export MAX_PIXELS="${MAX_PIXELS:-524288}"
    exec bash "${SCRIPT_DIR}/01_stage1_sft_query1.sh" "$@"
    ;;
  grpo1)
    # Four rollouts give a materially better within-prompt GRPO comparison than
    # two. Keep the global batch at 32 and reserve most VRAM for training.
    if [[ "${RTX5090_PROFILE}" == "balanced" ]]; then
      export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
      export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
      export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
      export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.28}"
      export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
    else
      export NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
      export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
      export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
      export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
      export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
    fi
    export DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
    export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-64}"
    export MAX_LENGTH="${MAX_LENGTH:-768}"
    export MAX_PIXELS="${MAX_PIXELS:-524288}"
    exec bash "${SCRIPT_DIR}/02_stage2_grpo1.sh" "$@"
    ;;
  sft2)
    # Original effective batch: 2 GPUs * 4 samples * 8 accumulation steps.
    # Batch 2 without activation checkpointing is the measured throughput sweet
    # spot on this host and still leaves ample headroom on the 32 GB card.
    if [[ "${RTX5090_PROFILE}" == "balanced" ]]; then
      export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
      export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
      export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
      export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
      export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
    else
      export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
      export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-64}"
      export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
      export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
      export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
    fi
    export DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
    export DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
    export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
    export EVAL_STRATEGY="${EVAL_STRATEGY:-epoch}"
    export SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
    export MAX_LENGTH="${MAX_LENGTH:-1408}"
    export MAX_PIXELS="${MAX_PIXELS:-524288}"
    exec bash "${SCRIPT_DIR}/03_stage2_sft_query2.sh" "$@"
    ;;
  grpo2)
    # Phase-B is GT-ID conditioned, so four rollouts give a better GRPO signal
    # than the Phase-A memory-safe two-rollout profile. On a single 32 GB RTX
    # 5090, keep vLLM disabled unless explicitly overridden; colocated vLLM
    # competes with the trainer for VRAM.
    if [[ "${RTX5090_PROFILE}" == "balanced" ]]; then
      export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
      export INTERSPEECH_GROUP_SIZE="${INTERSPEECH_GROUP_SIZE:-4}"
      export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
      export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
      export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
      export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-384}"
      export MAX_LENGTH="${MAX_LENGTH:-1024}"
      export MAX_PIXELS="${MAX_PIXELS:-524288}"
      export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
      export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
      export DATASET_NUM_PROC="${DATASET_NUM_PROC:-2}"
    else
      export NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
      export INTERSPEECH_GROUP_SIZE="${INTERSPEECH_GROUP_SIZE:-2}"
      export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
      export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
      export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
      export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-384}"
      export MAX_LENGTH="${MAX_LENGTH:-768}"
      export MAX_PIXELS="${MAX_PIXELS:-524288}"
      export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
      export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
      export DATASET_NUM_PROC="${DATASET_NUM_PROC:-2}"
    fi
    export USE_VLLM="${USE_VLLM:-false}"
    export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
    export LOGGING_STEPS="${LOGGING_STEPS:-20}"
    export SAVE_STEPS="${SAVE_STEPS:-100}"
    export EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
    export EVAL_STEPS="${EVAL_STEPS:-1000000}"
    export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
    exec bash "${SCRIPT_DIR}/04_stage3_grpo2_curriculum.sh" "$@"
    ;;
  *)
    echo "Unknown stage: ${STAGE}. Expected sft1, grpo1, sft2, or grpo2." >&2
    exit 2
    ;;
esac
