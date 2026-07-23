#!/usr/bin/env bash
set -euo pipefail

source /home/gridex/.venvs/gridex/bin/activate
cd /mnt/c/Users/Admin/Documents/GitHub/GRIDEX

export GRIDEX_ROOT="$PWD"
export DATA_ROOT="$GRIDEX_ROOT/data"
export OUTPUT_ROOT=/home/gridex/training_outputs
export CACHE_ROOT=/home/gridex/training_cache

export GRPO2_RUN="$OUTPUT_ROOT/grpo2_curriculum/grpo2_phase_a_5090_v1"
export PHASE_A_MERGED="$GRPO2_RUN/phase_a_merged"

export Q2_GT_TRAIN="$DATA_ROOT/processed/grpo2_gt_train_swift.json"
export Q2_GT_VAL="$DATA_ROOT/processed/grpo2_gt_val_swift.json"

export RTX5090_PROFILE=balanced
export BASE_MODEL_ID="$PHASE_A_MERGED"
export USE_PRED_PHASE=0
export PRED_ONLY=0
export OUTPUT_DIR="$GRPO2_RUN"
export PHASE_B_OUTPUT_DIR="$GRPO2_RUN/phase_b"
export PHASE_B_RESUME_CHECKPOINT="${PHASE_B_RESUME_CHECKPOINT:-}"
export GRPO2_GT_JSON_IN="$Q2_GT_TRAIN"
export GRPO2_GT_VAL_JSON_IN="$Q2_GT_VAL"
export GRPO2_GT_PREBUILT=1
export EXPECTED_TRAIN_SAMPLES=10320
export EXPECTED_VAL_SAMPLES=10192
export PHASE_B_EPOCHS=5
export TOTAL_MAX_STEPS=-1
export PER_DEVICE_EVAL_BATCH_SIZE=4
export EVAL_STRATEGY=no
export EVAL_STEPS=1000000
export GRPO_P2_RETRY_MISSING_FIELDS=0
export GRPO_P2_MAX_RETRIES=0
export INTERSPEECH_DEBUG_REWARD=1

bash training/scripts/run_rtx5090.sh grpo2
