#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIO_DIR="${AUDIO_DIR:?Set AUDIO_DIR to a directory containing .wav/.flac files}"
WORK_DIR="${WORK_DIR:?Set WORK_DIR to an output/work directory}"
Q1_MODEL_ID="${Q1_MODEL_ID:?Set Q1_MODEL_ID to the merged GRPO-1 model}"
Q2_MODEL_ID="${Q2_MODEL_ID:?Set Q2_MODEL_ID to the merged GRPO-2 Phase-B model}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
GPU_ID="${GPU_ID:-0}"
RUN_TAG="${RUN_TAG:-gridex_$(date +%Y%m%d_%H%M%S)}"
OVERWRITE="${OVERWRITE:-0}"

ASR_BACKEND="${ASR_BACKEND:-faster-whisper}"
ASR_MODEL="${ASR_MODEL:-small}"
ASR_LANGUAGE="${ASR_LANGUAGE:-en}"
ASR_DEVICE="${ASR_DEVICE:-auto}"
ASR_COMPUTE_TYPE="${ASR_COMPUTE_TYPE:-auto}"
ASR_VAD_FILTER="${ASR_VAD_FILTER:-0}"

DTYPE="${DTYPE:-auto}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-524288}"
Q1_BACKEND="${Q1_BACKEND:-transformers}"
Q1_MAX_NEW_TOKENS="${Q1_MAX_NEW_TOKENS:-64}"
Q1_IMAGE_MIN_PIXELS="${Q1_IMAGE_MIN_PIXELS:-100352}"
Q1_VLLM_MAX_MODEL_LEN="${Q1_VLLM_MAX_MODEL_LEN:-1400}"
Q1_VLLM_GPU_MEMORY_UTILIZATION="${Q1_VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
Q1_VLLM_ENFORCE_EAGER="${Q1_VLLM_ENFORCE_EAGER:-0}"
Q2_MAX_NEW_TOKENS="${Q2_MAX_NEW_TOKENS:-384}"
Q2_BATCH_SIZE="${Q2_BATCH_SIZE:-1}"
Q2_ATTN_IMPLEMENTATION="${Q2_ATTN_IMPLEMENTATION:-eager}"

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[error] CONDA_ENV_NAME is set but conda is unavailable." >&2
    exit 1
  fi
  conda activate "${CONDA_ENV_NAME}"
fi

WORK_DIR="$(mkdir -p "${WORK_DIR}" && cd "${WORK_DIR}" && pwd)"
if [[ ! -d "${AUDIO_DIR}" ]]; then
  echo "[error] AUDIO_DIR does not exist or is not a directory: ${AUDIO_DIR}" >&2
  exit 1
fi
if [[ ! -f "${Q1_MODEL_ID}/config.json" ]]; then
  echo "[error] Q1_MODEL_ID must be a merged model directory containing config.json: ${Q1_MODEL_ID}" >&2
  exit 1
fi
if [[ ! -f "${Q2_MODEL_ID}/config.json" ]]; then
  echo "[error] Q2_MODEL_ID must be a merged model directory containing config.json: ${Q2_MODEL_ID}" >&2
  exit 1
fi

Q1_META_JSON="${WORK_DIR}/q1_meta.json"
Q2_META_CSV="${WORK_DIR}/q2_meta.csv"
Q1_OUTPUT_ROOT="${WORK_DIR}/q1_predictions/${RUN_TAG}_q1"
Q1_OUTPUT_JSONL="${WORK_DIR}/q1_predictions/${RUN_TAG}_q1.jsonl"
Q2_OUTPUT_ROOT="${WORK_DIR}/q2_results/${RUN_TAG}_q2"
Q2_OUTPUT_JSONL="${WORK_DIR}/q2_results/${RUN_TAG}_q2.jsonl"

CACHE_BASE_DIR="${CACHE_BASE_DIR:-${WORK_DIR}/cache}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${CACHE_BASE_DIR}/flashinfer}"
export FLASHINFER_JIT_CACHE_DIR="${FLASHINFER_JIT_CACHE_DIR:-${FLASHINFER_WORKSPACE_DIR}}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${CACHE_BASE_DIR}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_BASE_DIR}}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${CACHE_BASE_DIR}/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_BASE_DIR}/triton}"
export HF_HOME="${HF_HOME:-${CACHE_BASE_DIR}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_BASE_DIR}/torch}"
export TMPDIR="${TMPDIR:-${CACHE_BASE_DIR}/tmp}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

mkdir -p \
  "${Q1_OUTPUT_ROOT}" \
  "${Q2_OUTPUT_ROOT}" \
  "${FLASHINFER_WORKSPACE_DIR}" \
  "${VLLM_CONFIG_ROOT}" \
  "${TRITON_CACHE_DIR}" \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${TMPDIR}"

echo "[run] gpu=${GPU_ID}"
echo "[run] q1_model=${Q1_MODEL_ID}"
echo "[run] q2_model=${Q2_MODEL_ID}"
echo "[run] work_dir=${WORK_DIR}"

prepare_cmd=(python -u "${SCRIPT_DIR}/util/prepare.py"
  --audio-dir "${AUDIO_DIR}"
  --work-dir "${WORK_DIR}"
  --asr-backend "${ASR_BACKEND}"
  --asr-model "${ASR_MODEL}"
  --language "${ASR_LANGUAGE}"
  --asr-device "${ASR_DEVICE}"
  --asr-compute-type "${ASR_COMPUTE_TYPE}")
if [[ "${ASR_VAD_FILTER}" == "1" ]]; then
  prepare_cmd+=(--vad-filter)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  prepare_cmd+=(--overwrite)
fi
"${prepare_cmd[@]}"

q1_cmd=(python -u "${SCRIPT_DIR}/q1.py"
  --backend "${Q1_BACKEND}"
  --model-id "${Q1_MODEL_ID}"
  --meta-json "${Q1_META_JSON}"
  --system-file "${SCRIPT_DIR}/q1_prompts/system.txt"
  --user-template-file "${SCRIPT_DIR}/q1_prompts/user.txt"
  --dtype "${DTYPE}"
  --max-new-tokens "${Q1_MAX_NEW_TOKENS}"
  --image-min-pixels "${Q1_IMAGE_MIN_PIXELS}"
  --image-max-pixels "${IMAGE_MAX_PIXELS}"
  --tensor-parallel-size 1
  --vllm-max-model-len "${Q1_VLLM_MAX_MODEL_LEN}"
  --vllm-gpu-memory-utilization "${Q1_VLLM_GPU_MEMORY_UTILIZATION}"
  --vllm-max-images-per-prompt 1
  --output-dir "${Q1_OUTPUT_ROOT}"
  --output-jsonl "${Q1_OUTPUT_JSONL}")
if [[ "${Q1_VLLM_ENFORCE_EAGER}" == "1" ]]; then
  q1_cmd+=(--vllm-enforce-eager)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  q1_cmd+=(--overwrite)
fi
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${q1_cmd[@]}"

q2_cmd=(python -u "${SCRIPT_DIR}/q2.py"
  --model-id "${Q2_MODEL_ID}"
  --meta-csv "${Q2_META_CSV}"
  --prompt1-json-root "${Q1_OUTPUT_ROOT}"
  --system-file "${SCRIPT_DIR}/q2_prompts/system.txt"
  --user-template-file "${SCRIPT_DIR}/q2_prompts/user.txt"
  --device-map auto
  --dtype "${DTYPE}"
  --attn-implementation "${Q2_ATTN_IMPLEMENTATION}"
  --max-new-tokens "${Q2_MAX_NEW_TOKENS}"
  --image-max-pixels "${IMAGE_MAX_PIXELS}"
  --batch-size "${Q2_BATCH_SIZE}"
  --output-dir "${Q2_OUTPUT_ROOT}"
  --output-jsonl "${Q2_OUTPUT_JSONL}")
if [[ "${OVERWRITE}" == "1" ]]; then
  q2_cmd+=(--overwrite)
fi
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${q2_cmd[@]}"

echo "[done] work_dir=${WORK_DIR}"
echo "[done] q1_predictions=${Q1_OUTPUT_ROOT}"
echo "[done] q2_results=${Q2_OUTPUT_ROOT}"
echo "[done] q2_jsonl=${Q2_OUTPUT_JSONL}"
