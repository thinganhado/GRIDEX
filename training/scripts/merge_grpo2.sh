#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_swift

# Merge the Phase-B LoRA into the Phase-A merged base model used by Phase B.

BASE_MODEL_ID="${BASE_MODEL_ID:?Set BASE_MODEL_ID to the Phase-A merged checkpoint used to initialize Phase B}"
ADAPTER_PATH="${ADAPTER_PATH:?Set ADAPTER_PATH to the final GRPO-2 LoRA checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/gridex_final_merged}"

mkdir -p "${CACHE_ROOT}/triton" "${CACHE_ROOT}/torch_extensions" "${CACHE_ROOT}/hf" \
  "${CACHE_ROOT}/xdg_cache" "${CACHE_ROOT}/modelscope" "${CACHE_ROOT}/datasets" \
  "${CACHE_ROOT}/vllm" "${CACHE_ROOT}/flashinfer"

export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export HF_HOME="${CACHE_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export DATASETS_CACHE="${CACHE_ROOT}/datasets"
export MODELSCOPE_CACHE="${CACHE_ROOT}/modelscope"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg_cache"
export VLLM_CONFIG_ROOT="${CACHE_ROOT}/vllm"
export VLLM_NO_USAGE_STATS=1
export FLASHINFER_WORKSPACE_BASE="${CACHE_ROOT}"
export FLASHINFER_WORKSPACE_DIR="${CACHE_ROOT}/flashinfer"
export FLASHINFER_JIT_CACHE_DIR="${FLASHINFER_WORKSPACE_DIR}"
unset TRANSFORMERS_CACHE

echo "[merge] BASE_MODEL_ID=${BASE_MODEL_ID}"
echo "[merge] ADAPTER_PATH=${ADAPTER_PATH}"
echo "[merge] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[merge] CACHE_ROOT=${CACHE_ROOT}"

swift export \
  --model "${BASE_MODEL_ID}" \
  --adapters "${ADAPTER_PATH}" \
  --load_args false \
  --merge_lora true \
  --output_dir "${OUTPUT_DIR}"
