#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_swift

# ===== User-settable =====
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-3B-Instruct}"
TRAIN_JSON_SWIFT="${TRAIN_JSON_SWIFT:-${DATA_ROOT}/processed/stage1_query1_train_swift.json}"
VAL_JSON_SWIFT="${VAL_JSON_SWIFT:-${DATA_ROOT}/processed/stage1_query1_val_swift.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/grpo1}"
POLICY_ADAPTERS="${POLICY_ADAPTERS:-}"
REF_ADAPTERS="${REF_ADAPTERS:-}"
SFT1_MERGED_MODEL="${SFT1_MERGED_MODEL:-0}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-As an expert in deepfake speech spectrogram forensics, you can detect regions containing deepfake artifacts by analysing spectrogram segments. Return only the JSON array of your three chosen region IDs.}"
TMPDIR_BASE="${TMPDIR_BASE:-/tmp/${USER:-user}_mswift_q1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
USE_VLLM="${USE_VLLM:-true}"
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
SLEEP_LEVEL="${SLEEP_LEVEL:-1}"
GRPO_BETA="${GRPO_BETA:-0.01}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
NUM_GENERATIONS_EVAL="${NUM_GENERATIONS_EVAL:-${NUM_GENERATIONS}}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-128}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_PIXELS="${MAX_PIXELS:-524288}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-${PER_DEVICE_TRAIN_BATCH_SIZE}}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
DEEPSPEED="${DEEPSPEED:-zero2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
LOGGING_STEPS="${LOGGING_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-200}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
EVAL_STEPS="${EVAL_STEPS:-${SAVE_STEPS}}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
AUTO_MERGE_AFTER_TRAIN="${AUTO_MERGE_AFTER_TRAIN:-1}"
MERGE_SOURCE="${MERGE_SOURCE:-best}" # best | last
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.8}"
INTERSPEECH_DEBUG_REWARD="${INTERSPEECH_DEBUG_REWARD:-1}"
INTERSPEECH_DEBUG_SAMPLES="${INTERSPEECH_DEBUG_SAMPLES:-8}"
GRPO_DEDUPE_ROLLOUTS="${GRPO_DEDUPE_ROLLOUTS:-0}"
GRPO_DEDUPE_MAX_RETRIES="${GRPO_DEDUPE_MAX_RETRIES:-0}"
GRPO_DEDUPE_REQUIRE_VALID="${GRPO_DEDUPE_REQUIRE_VALID:-0}"
GRPO_DEDUPE_OVERLAP_REJECT="${GRPO_DEDUPE_OVERLAP_REJECT:-2}"
GRPO_DEDUPE_ID_CAP="${GRPO_DEDUPE_ID_CAP:-3}"
GRPO_DEDUPE_NON_HIT_ID_CAP="${GRPO_DEDUPE_NON_HIT_ID_CAP:-1}"
GRPO_DEDUPE_HIT_ORDER_CAP="${GRPO_DEDUPE_HIT_ORDER_CAP:-1}"
GRPO_DEDUPE_EXACT_UNIQUE="${GRPO_DEDUPE_EXACT_UNIQUE:-1}"
GRPO_RETRY_USE_GT_FEEDBACK="${GRPO_RETRY_USE_GT_FEEDBACK:-0}"
GRPO_RETRY_REQUIRE_IMPROVEMENT="${GRPO_RETRY_REQUIRE_IMPROVEMENT:-0}"
GRPO_DEDUPE_STRICT="${GRPO_DEDUPE_STRICT:-1}"
GRPO_DEDUPE_ONLY_WRONG_IDS="${GRPO_DEDUPE_ONLY_WRONG_IDS:-1}"

# Optional positional override: first arg as MODEL_ID
if [[ $# -ge 1 && -n "${1:-}" ]]; then
  MODEL_ID="$1"
fi

if [[ ! -f "${TRAIN_JSON_SWIFT}" ]]; then
  echo "ERROR: training dataset not found: ${TRAIN_JSON_SWIFT}"
  exit 1
fi
if [[ ! -f "${VAL_JSON_SWIFT}" ]]; then
  echo "ERROR: validation dataset not found: ${VAL_JSON_SWIFT}"
  exit 1
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  if [[ -n "${POLICY_ADAPTERS}" ]]; then
    echo "ERROR: when resuming GRPO-1, omit POLICY_ADAPTERS and use RESUME_FROM_CHECKPOINT for the policy." >&2
    exit 1
  fi
  if [[ -z "${REF_ADAPTERS}" ]]; then
    echo "ERROR: REF_ADAPTERS must point to the original SFT-1 checkpoint when resuming GRPO-1." >&2
    exit 1
  fi
elif [[ "${SFT1_MERGED_MODEL}" == "1" ]]; then
  if [[ -n "${POLICY_ADAPTERS}" || -n "${REF_ADAPTERS}" ]]; then
    echo "ERROR: SFT1_MERGED_MODEL=1 requires POLICY_ADAPTERS and REF_ADAPTERS to be empty." >&2
    exit 1
  fi
else
  if [[ -z "${POLICY_ADAPTERS}" || -z "${REF_ADAPTERS}" ]]; then
    echo "ERROR: fresh GRPO-1 requires both POLICY_ADAPTERS and REF_ADAPTERS." >&2
    echo "Set both to the same SFT-1 checkpoint, or set SFT1_MERGED_MODEL=1 with MODEL_ID=<merged-SFT1>." >&2
    exit 1
  fi
  if [[ "${POLICY_ADAPTERS}" != "${REF_ADAPTERS}" ]]; then
    echo "ERROR: POLICY_ADAPTERS and REF_ADAPTERS must be the same SFT-1 checkpoint for a fresh run." >&2
    exit 1
  fi
fi

ADAPTER_ARGS=()
if [[ -n "${POLICY_ADAPTERS}" ]]; then
  ADAPTER_ARGS+=(--adapters "${POLICY_ADAPTERS}")
fi
if [[ -n "${REF_ADAPTERS}" ]]; then
  ADAPTER_ARGS+=(--ref_adapters "${REF_ADAPTERS}")
fi

VAL_DATASET="${VAL_JSON_SWIFT}"
if [[ -n "${EVAL_MAX_SAMPLES}" ]]; then
  VAL_DATASET="${VAL_DATASET}#${EVAL_MAX_SAMPLES}"
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
  echo "[run] Resuming from checkpoint: ${RESUME_FROM_CHECKPOINT}"
fi

mkdir -p "${CACHE_ROOT}/triton" "${CACHE_ROOT}/torch_extensions" "${CACHE_ROOT}/hf" \
  "${CACHE_ROOT}/xdg_cache" "${CACHE_ROOT}/modelscope" "${CACHE_ROOT}/datasets" "${CACHE_ROOT}/vllm" \
  "${CACHE_ROOT}/flashinfer" "${TMPDIR_BASE}"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export HF_HOME="${CACHE_ROOT}/hf"
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
export INTERSPEECH_DEBUG_REWARD="${INTERSPEECH_DEBUG_REWARD}"
export INTERSPEECH_DEBUG_SAMPLES="${INTERSPEECH_DEBUG_SAMPLES}"
export GRPO_DEDUPE_ROLLOUTS="${GRPO_DEDUPE_ROLLOUTS}"
export GRPO_DEDUPE_MAX_RETRIES="${GRPO_DEDUPE_MAX_RETRIES}"
export GRPO_DEDUPE_REQUIRE_VALID="${GRPO_DEDUPE_REQUIRE_VALID}"
export GRPO_DEDUPE_OVERLAP_REJECT="${GRPO_DEDUPE_OVERLAP_REJECT}"
export GRPO_DEDUPE_ID_CAP="${GRPO_DEDUPE_ID_CAP}"
export GRPO_DEDUPE_NON_HIT_ID_CAP="${GRPO_DEDUPE_NON_HIT_ID_CAP}"
export GRPO_DEDUPE_HIT_ORDER_CAP="${GRPO_DEDUPE_HIT_ORDER_CAP}"
export GRPO_DEDUPE_EXACT_UNIQUE="${GRPO_DEDUPE_EXACT_UNIQUE}"
export GRPO_RETRY_USE_GT_FEEDBACK="${GRPO_RETRY_USE_GT_FEEDBACK}"
export GRPO_RETRY_REQUIRE_IMPROVEMENT="${GRPO_RETRY_REQUIRE_IMPROVEMENT}"
export GRPO_DEDUPE_STRICT="${GRPO_DEDUPE_STRICT}"
export GRPO_DEDUPE_ONLY_WRONG_IDS="${GRPO_DEDUPE_ONLY_WRONG_IDS}"

DEEPSPEED_ARGS=()
if [[ "${DEEPSPEED}" != "none" && -n "${DEEPSPEED}" ]]; then
  DEEPSPEED_ARGS=(--deepspeed "${DEEPSPEED}")
fi

VLLM_ARGS=(--use_vllm "${USE_VLLM}")
if [[ "${USE_VLLM,,}" == "true" || "${USE_VLLM}" == "1" ]]; then
  VLLM_ARGS+=(
    --vllm_mode "${VLLM_MODE}"
    --vllm_tensor_parallel_size "${VLLM_TP}"
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --sleep_level "${SLEEP_LEVEL}"
  )
fi

# ===== Train GRPO-1 =====
NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL_ID}" \
  "${ADAPTER_ARGS[@]}" \
  --dataset "${TRAIN_JSON_SWIFT}" \
  --val_dataset "${VAL_DATASET}" \
  --system "${SYSTEM_PROMPT}" \
  --external_plugins "${TRAINING_ROOT}/plugins/rewards.py" \
  --reward_funcs external_interspeech_p1 \
  "${VLLM_ARGS[@]}" \
  --beta "${GRPO_BETA}" \
  --num_generations "${NUM_GENERATIONS}" \
  --num_generations_eval "${NUM_GENERATIONS_EVAL}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_llm false \
  --freeze_vit true \
  --freeze_aligner true \
  --torch_dtype "${TORCH_DTYPE}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_steps -1 \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --max_length "${MAX_LENGTH}" \
  --max_pixels "${MAX_PIXELS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate 1e-5 \
  --weight_decay 0.1 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --logging_steps "${LOGGING_STEPS}" \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 10 \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  "${DEEPSPEED_ARGS[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  --report_to tensorboard \
  --log_completions true \
  "${RESUME_ARGS[@]}"

if [[ "${AUTO_MERGE_AFTER_TRAIN}" == "1" ]]; then
  if ! command -v swift >/dev/null 2>&1; then
    echo "WARN: swift command not found in PATH; skipping merge export."
    exit 0
  fi

  MERGE_CKPT="$(
    python - "${OUTPUT_DIR}" "${MERGE_SOURCE}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
merge_source = sys.argv[2].strip().lower()

states = sorted(root.rglob("trainer_state.json"), key=lambda p: p.stat().st_mtime)
if not states:
    print("")
    raise SystemExit(0)

state = json.loads(states[-1].read_text(encoding="utf-8"))
best_ckpt = str(state.get("best_model_checkpoint", "")).strip()
last_ckpt = str(state.get("last_model_checkpoint", "")).strip()

if merge_source == "last":
    print(last_ckpt or best_ckpt)
else:
    print(best_ckpt or last_ckpt)
PY
  )"

  if [[ -z "${MERGE_CKPT}" ]]; then
    echo "WARN: Could not find best/last checkpoint under ${OUTPUT_DIR}; skipping merge export."
    exit 0
  fi
  if [[ ! -d "${MERGE_CKPT}" ]]; then
    echo "WARN: Selected merge checkpoint does not exist as directory: ${MERGE_CKPT}"
    exit 0
  fi

  MERGED_OUT="${MERGE_CKPT%/}-merged"
  echo "[merge] source=${MERGE_CKPT}"
  echo "[merge] output=${MERGED_OUT}"
  swift export \
    --adapters "${MERGE_CKPT}" \
    --merge_lora true \
    --output_dir "${MERGED_OUT}"
fi
