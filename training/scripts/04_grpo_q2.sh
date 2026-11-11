#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_swift

# ===== User-settable =====
# Base model is the SFT-2 merged checkpoint.
# GRPO-2 trains a new LoRA adapter on top of this fixed base model.
BASE_MODEL_ID="${BASE_MODEL_ID:-${OUTPUT_ROOT}/sft2_merged}"
GRPO2_GT_JSON_IN="${GRPO2_GT_JSON_IN:-${DATA_ROOT}/processed/grpo2_gt_train_swift.json}"
GRPO2_GT_VAL_JSON_IN="${GRPO2_GT_VAL_JSON_IN:-${DATA_ROOT}/processed/grpo2_gt_val_swift.json}"
GRPO2_PRED_JSON_IN="${GRPO2_PRED_JSON_IN:-${DATA_ROOT}/processed/grpo2_pred_aligned.json}"
GRPO2_GT_JSON_SWIFT="${GRPO2_GT_JSON_SWIFT:-${DATA_ROOT}/processed/grpo2_gt_train_swift.json}"
GRPO2_GT_VAL_JSON_SWIFT="${GRPO2_GT_VAL_JSON_SWIFT:-${DATA_ROOT}/processed/grpo2_gt_val_swift.json}"
GRPO2_PRED_JSON_SWIFT="${GRPO2_PRED_JSON_SWIFT:-${DATA_ROOT}/processed/grpo2_pred_aligned_swift.json}"
GRPO2_GT_PREBUILT="${GRPO2_GT_PREBUILT:-1}"
GRPO2_PRED_PREBUILT="${GRPO2_PRED_PREBUILT:-1}"
PHASE_A_RESUME_CHECKPOINT="${PHASE_A_RESUME_CHECKPOINT:-}"
PHASE_B_RESUME_CHECKPOINT="${PHASE_B_RESUME_CHECKPOINT:-}"
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-${OUTPUT_ROOT}/grpo2_curriculum}"
RUN_TAG="${RUN_TAG:-v0-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_DIR_BASE%/}/${RUN_TAG}}"
PHASE_A_OUTPUT_DIR="${PHASE_A_OUTPUT_DIR:-${OUTPUT_DIR}/phase_a}"
PHASE_A_MERGED_MODEL="${PHASE_A_MERGED_MODEL:-${OUTPUT_DIR}/phase_a_merged}"
PHASE_B_OUTPUT_DIR="${PHASE_B_OUTPUT_DIR:-${OUTPUT_DIR}/phase_b}"
EXPECTED_TRAIN_SAMPLES="${EXPECTED_TRAIN_SAMPLES:-10320}"
EXPECTED_VAL_SAMPLES="${EXPECTED_VAL_SAMPLES:-10192}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are an expert in deepfake speech spectrogram forensics.

You are given a spectrogram and transcript. You have already selected exactly 3 region IDs, in order: ID1, ID2, ID3.
For each ID, infer timing information (T), frequency band (F), phonetic category (P), and a visual description of the artifact and the likely audio impact implied by the artificial signs (En).

OUTPUT FORMAT (must follow exactly):
(T1=..., F1=..., P1=..., En1=\"...\"); (T2=..., F2=..., P2=..., En2=\"...\"); (T3=..., F3=..., P3=..., En3=\"...\")

Field definitions:
- Fields ending in 1, 2, and 3 correspond to ID1, ID2, and ID3 respectively.
- T: one of {speech, non-speech\}
- F: one of {low, mid, high\}
- P: one of {consonant, vowel, none\}. Use none when the region is outside aligned speech and no phone can be assigned.
- En: textual description, must be enclosed in double quotes.

Do not output any other text outside the three tuples.}"

WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
PHASE_B_EPOCHS="${PHASE_B_EPOCHS:-5}"
WARMUP_MAX_STEPS="${WARMUP_MAX_STEPS:--1}"
TOTAL_MAX_STEPS="${TOTAL_MAX_STEPS:--1}"
USE_PRED_PHASE="${USE_PRED_PHASE:-1}"
PRED_ONLY="${PRED_ONLY:-0}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
USE_VLLM="${USE_VLLM:-true}"
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
SLEEP_LEVEL="${SLEEP_LEVEL:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
GRPO_BETA="${GRPO_BETA:-0.1}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-768}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_PIXELS="${MAX_PIXELS:-}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-${PER_DEVICE_TRAIN_BATCH_SIZE}}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
DEEPSPEED="${DEEPSPEED:-zero2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
INTERSPEECH_DEBUG_REWARD="${INTERSPEECH_DEBUG_REWARD:-0}"
INTERSPEECH_DEBUG_SAMPLES="${INTERSPEECH_DEBUG_SAMPLES:-8}"
INTERSPEECH_LOG_EVERY_STEPS="${INTERSPEECH_LOG_EVERY_STEPS:-1}"
INTERSPEECH_GROUP_SIZE="${INTERSPEECH_GROUP_SIZE:-8}"
GRPO_P2_RETRY_MISSING_FIELDS="${GRPO_P2_RETRY_MISSING_FIELDS:-0}"
GRPO_P2_MAX_RETRIES="${GRPO_P2_MAX_RETRIES:-0}"
TMPDIR_BASE="${TMPDIR_BASE:-/tmp/${USER:-user}_mswift_q2}"

mkdir -p "${CACHE_ROOT}/triton" "${CACHE_ROOT}/torch_extensions" "${CACHE_ROOT}/hf" \
  "${CACHE_ROOT}/xdg_cache" "${CACHE_ROOT}/modelscope" "${CACHE_ROOT}/datasets" \
  "${CACHE_ROOT}/vllm" "${CACHE_ROOT}/flashinfer" "${TMPDIR_BASE}"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export HF_HOME="${CACHE_ROOT}/hf"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export DATASETS_CACHE="${CACHE_ROOT}/datasets"
export MODELSCOPE_CACHE="${CACHE_ROOT}/modelscope"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg_cache"
export VLLM_CONFIG_ROOT="${CACHE_ROOT}/vllm"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export FLASHINFER_WORKSPACE_BASE="${CACHE_ROOT}"
export FLASHINFER_WORKSPACE_DIR="${CACHE_ROOT}/flashinfer"
export FLASHINFER_JIT_CACHE_DIR="${FLASHINFER_WORKSPACE_DIR}"
export TMPDIR="${TMPDIR_BASE}"

mkdir -p "${OUTPUT_DIR}"
echo "[run] BASE_MODEL_ID=${BASE_MODEL_ID}"
echo "[run] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[run] RUN_TAG=${RUN_TAG}"
echo "[run] USE_PRED_PHASE=${USE_PRED_PHASE}"
echo "[run] PRED_ONLY=${PRED_ONLY}"
echo "[run] VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}"
echo "[run] GRPO2_GT_PREBUILT=${GRPO2_GT_PREBUILT}"
echo "[run] GRPO2_PRED_PREBUILT=${GRPO2_PRED_PREBUILT}"
echo "[run] PHASE_A_RESUME_CHECKPOINT=${PHASE_A_RESUME_CHECKPOINT:-<none>}"
echo "[run] PHASE_B_RESUME_CHECKPOINT=${PHASE_B_RESUME_CHECKPOINT:-<none>}"
echo "[run] NUM_GENERATIONS=${NUM_GENERATIONS} INTERSPEECH_GROUP_SIZE=${INTERSPEECH_GROUP_SIZE}"
echo "[run] MAX_LENGTH=${MAX_LENGTH} MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH} MAX_PIXELS=${MAX_PIXELS:-<default>}"
echo "[run] PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE} PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE} GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "[run] DEEPSPEED=${DEEPSPEED} USE_VLLM=${USE_VLLM} GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}"
echo "[run] EVAL_STRATEGY=${EVAL_STRATEGY} EVAL_STEPS=${EVAL_STEPS} SAVE_STEPS=${SAVE_STEPS}"

get_latest_output_checkpoint() {
  local search_root="$1"
  local checkpoint_dir
  checkpoint_dir="$(
    find "${search_root}" -type d -name 'checkpoint-*' 2>/dev/null \
      | sort -V \
      | tail -n 1
  )"
  if [[ -n "${checkpoint_dir}" ]]; then
    printf '%s\n' "${checkpoint_dir}"
  fi
}

DEEPSPEED_ARGS=()
if [[ "${DEEPSPEED}" != "none" && -n "${DEEPSPEED}" ]]; then
  DEEPSPEED_ARGS=(--deepspeed "${DEEPSPEED}")
fi

MAX_PIXELS_ARGS=()
if [[ -n "${MAX_PIXELS}" ]]; then
  MAX_PIXELS_ARGS=(--max_pixels "${MAX_PIXELS}")
fi

VLLM_ARGS=()
if [[ "${USE_VLLM}" == "true" || "${USE_VLLM}" == "1" ]]; then
  VLLM_ARGS=(
    --vllm_mode "${VLLM_MODE}"
    --vllm_tensor_parallel_size "${VLLM_TP}"
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --sleep_level "${SLEEP_LEVEL}"
  )
fi

COMMON_ARGS=(
  --rlhf_type grpo
  --system "${SYSTEM_PROMPT}"
  --external_plugins "${TRAINING_ROOT}/plugins/rewards.py"
  --reward_funcs external_interspeech_p2
  --use_vllm "${USE_VLLM}"
  "${VLLM_ARGS[@]}"
  --beta "${GRPO_BETA}"
  --num_generations "${NUM_GENERATIONS}"
  --temperature 0.9
  --tuner_type lora
  --lora_rank 8
  --lora_alpha 32
  --target_modules all-linear
  --freeze_llm false
  --freeze_vit true
  --freeze_aligner true
  --torch_dtype bfloat16
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
  --max_length "${MAX_LENGTH}"
  "${MAX_PIXELS_ARGS[@]}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_ratio "${WARMUP_RATIO}"
  --lr_scheduler_type cosine
  --logging_steps "${LOGGING_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --dataset_num_proc "${DATASET_NUM_PROC}"
  "${DEEPSPEED_ARGS[@]}"
  --report_to tensorboard
  --log_completions true
)

# ===== Build GRPO2 datasets =====
if [[ "${USE_PRED_PHASE}" == "1" ]]; then
  if [[ "${GRPO2_PRED_PREBUILT}" == "1" ]]; then
    mkdir -p "$(dirname "${GRPO2_PRED_JSON_SWIFT}")"
    if [[ "${GRPO2_PRED_JSON_IN}" != "${GRPO2_PRED_JSON_SWIFT}" ]]; then
      cp -f "${GRPO2_PRED_JSON_IN}" "${GRPO2_PRED_JSON_SWIFT}"
    fi
  else
    python "${TRAINING_ROOT}/tools/build_q2_grpo_data.py" \
      --input-json "${GRPO2_PRED_JSON_IN}" \
      --output-json "${GRPO2_PRED_JSON_SWIFT}"
  fi
fi

if [[ "${GRPO2_GT_PREBUILT}" == "1" ]]; then
  mkdir -p "$(dirname "${GRPO2_GT_JSON_SWIFT}")" "$(dirname "${GRPO2_GT_VAL_JSON_SWIFT}")"
  if [[ "${GRPO2_GT_JSON_IN}" != "${GRPO2_GT_JSON_SWIFT}" ]]; then
    cp -f "${GRPO2_GT_JSON_IN}" "${GRPO2_GT_JSON_SWIFT}"
  fi
  if [[ "${GRPO2_GT_VAL_JSON_IN}" != "${GRPO2_GT_VAL_JSON_SWIFT}" ]]; then
    cp -f "${GRPO2_GT_VAL_JSON_IN}" "${GRPO2_GT_VAL_JSON_SWIFT}"
  fi
else
  python "${TRAINING_ROOT}/tools/build_q2_grpo_data.py" \
    --input-json "${GRPO2_GT_JSON_IN}" \
    --output-json "${GRPO2_GT_JSON_SWIFT}"
  python "${TRAINING_ROOT}/tools/build_q2_grpo_data.py" \
    --input-json "${GRPO2_GT_VAL_JSON_IN}" \
    --output-json "${GRPO2_GT_VAL_JSON_SWIFT}"
fi

python - "${GRPO2_GT_JSON_SWIFT}" "${GRPO2_GT_VAL_JSON_SWIFT}" "${GRPO2_PRED_JSON_SWIFT}" \
  "${USE_PRED_PHASE}" "${EXPECTED_TRAIN_SAMPLES}" "${EXPECTED_VAL_SAMPLES}" <<'PY'
import json
import sys
from pathlib import Path

train_path, val_path, pred_path = map(Path, sys.argv[1:4])
use_pred = sys.argv[4] == "1"
expected_train, expected_val = map(int, sys.argv[5:7])

def load_and_check(path, split, expected=0):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON list")
    bad_split = [str(r.get("sample_id", "")) for r in data if split not in str(r.get("sample_id", ""))]
    missing_transcript = [str(r.get("sample_id", "")) for r in data if not str(r.get("transcript", "")).strip()]
    if bad_split:
        raise ValueError(f"{path}: records outside {split}: {bad_split[:20]}")
    if missing_transcript:
        raise ValueError(f"{path}: records missing transcript: {missing_transcript[:20]}")
    if expected > 0 and len(data) != expected:
        raise ValueError(f"{path}: got {len(data)} records, expected {expected}")
    return data

train = load_and_check(train_path, "_LA_T_", expected_train)
val = load_and_check(val_path, "_LA_D_", expected_val)
train_ids = {r["sample_id"] for r in train}
val_ids = {r["sample_id"] for r in val}
if train_ids & val_ids:
    raise ValueError("Q2 train/development sample overlap detected")
if use_pred:
    load_and_check(pred_path, "_LA_T_")
print(f"[q2_audit] train={len(train)} val={len(val)} split_leakage=0")
PY

PHASE_B_MODEL="${BASE_MODEL_ID}"

if [[ "${USE_PRED_PHASE}" == "1" ]]; then
  PHASE_A_RESUME_ARGS=()
  if [[ -n "${PHASE_A_RESUME_CHECKPOINT}" ]]; then
    PHASE_A_RESUME_ARGS=(--resume_from_checkpoint "${PHASE_A_RESUME_CHECKPOINT}")
  fi

  # ===== Phase A: one epoch with raw Query-1 predicted IDs =====
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  INTERSPEECH_DEBUG_REWARD="${INTERSPEECH_DEBUG_REWARD}" \
  INTERSPEECH_DEBUG_SAMPLES="${INTERSPEECH_DEBUG_SAMPLES}" \
  INTERSPEECH_LOG_EVERY_STEPS="${INTERSPEECH_LOG_EVERY_STEPS}" \
  INTERSPEECH_GROUP_SIZE="${INTERSPEECH_GROUP_SIZE}" \
  GRPO_P2_RETRY_MISSING_FIELDS="${GRPO_P2_RETRY_MISSING_FIELDS}" \
  GRPO_P2_MAX_RETRIES="${GRPO_P2_MAX_RETRIES}" \
  swift rlhf \
    "${COMMON_ARGS[@]}" \
    --model "${BASE_MODEL_ID}" \
    --dataset "${GRPO2_PRED_JSON_SWIFT}" \
    --output_dir "${PHASE_A_OUTPUT_DIR}" \
    "${PHASE_A_RESUME_ARGS[@]}" \
    --num_train_epochs "${WARMUP_EPOCHS}" \
    --max_steps "${WARMUP_MAX_STEPS}"

  PHASE_A_CHECKPOINT="$(get_latest_output_checkpoint "${PHASE_A_OUTPUT_DIR}")"
  if [[ -z "${PHASE_A_CHECKPOINT}" ]]; then
    echo "[error] No Phase-A checkpoint found under ${PHASE_A_OUTPUT_DIR}" >&2
    exit 1
  fi
  echo "[merge] Phase-A checkpoint=${PHASE_A_CHECKPOINT}"
  swift export \
    --model "${BASE_MODEL_ID}" \
    --adapters "${PHASE_A_CHECKPOINT}" \
    --merge_lora true \
    --output_dir "${PHASE_A_MERGED_MODEL}"
  PHASE_B_MODEL="${PHASE_A_MERGED_MODEL}"
fi

if [[ "${PRED_ONLY}" == "1" ]]; then
  if [[ "${USE_PRED_PHASE}" != "1" ]]; then
    echo "[error] PRED_ONLY=1 requires USE_PRED_PHASE=1" >&2
    exit 1
  fi
  echo "[run] PRED_ONLY=1; Phase A was merged at ${PHASE_A_MERGED_MODEL}."
  exit 0
fi

PHASE_B_RESUME_ARGS=()
if [[ -n "${PHASE_B_RESUME_CHECKPOINT}" ]]; then
  PHASE_B_RESUME_ARGS=(--resume_from_checkpoint "${PHASE_B_RESUME_CHECKPOINT}")
fi

# ===== Phase B: fresh GRPO run initialized from the merged Phase-A model =====
NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
INTERSPEECH_DEBUG_REWARD="${INTERSPEECH_DEBUG_REWARD}" \
INTERSPEECH_DEBUG_SAMPLES="${INTERSPEECH_DEBUG_SAMPLES}" \
INTERSPEECH_LOG_EVERY_STEPS="${INTERSPEECH_LOG_EVERY_STEPS}" \
INTERSPEECH_GROUP_SIZE="${INTERSPEECH_GROUP_SIZE}" \
GRPO_P2_RETRY_MISSING_FIELDS="${GRPO_P2_RETRY_MISSING_FIELDS}" \
GRPO_P2_MAX_RETRIES="${GRPO_P2_MAX_RETRIES}" \
swift rlhf \
  "${COMMON_ARGS[@]}" \
  --model "${PHASE_B_MODEL}" \
  --dataset "${GRPO2_GT_JSON_SWIFT}" \
  --val_dataset "${GRPO2_GT_VAL_JSON_SWIFT}" \
  --eval_strategy "${EVAL_STRATEGY}" \
  --eval_steps "${EVAL_STEPS}" \
  --output_dir "${PHASE_B_OUTPUT_DIR}" \
  "${PHASE_B_RESUME_ARGS[@]}" \
  --num_train_epochs "${PHASE_B_EPOCHS}" \
  --max_steps "${TOTAL_MAX_STEPS}"
