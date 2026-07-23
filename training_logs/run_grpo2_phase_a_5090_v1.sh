#!/usr/bin/env bash
set -euo pipefail

source /home/gridex/.venvs/gridex/bin/activate
cd /mnt/c/Users/Admin/Documents/GitHub/GRIDEX

export GRIDEX_ROOT=/mnt/c/Users/Admin/Documents/GitHub/GRIDEX
export DATA_ROOT="${GRIDEX_ROOT}/data"
export OUTPUT_ROOT=/home/gridex/training_outputs
export CACHE_ROOT=/home/gridex/training_cache
export SFT2_MERGED=/home/gridex/training_outputs/sft2_query2/sft2_q2_5090_balanced_v1/sft2_checkpoint324_merged

BASE_MODEL_ID="${SFT2_MERGED}" \
GRPO2_GT_JSON_IN="${DATA_ROOT}/processed/grpo2_gt_train_swift.json" \
GRPO2_GT_VAL_JSON_IN="${DATA_ROOT}/processed/grpo2_gt_val_swift.json" \
GRPO2_PRED_JSON_IN="${DATA_ROOT}/processed/grpo2_pred_aligned.json" \
OUTPUT_DIR_BASE="${OUTPUT_ROOT}/grpo2_curriculum" \
RUN_TAG=grpo2_phase_a_5090_v1 \
USE_PRED_PHASE=1 \
PRED_ONLY=1 \
WARMUP_EPOCHS=1 \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
DEEPSPEED=none \
USE_VLLM=false \
NUM_GENERATIONS=2 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=16 \
MAX_LENGTH=1408 \
MAX_COMPLETION_LENGTH=512 \
GRPO_BETA=0.1 \
GRPO_P2_RETRY_MISSING_FIELDS=0 \
GRPO_P2_MAX_RETRIES=0 \
INTERSPEECH_DEBUG_REWARD=1 \
INTERSPEECH_DEBUG_SAMPLES=8 \
bash "${GRIDEX_ROOT}/training/scripts/04_stage3_grpo2_curriculum.sh"
