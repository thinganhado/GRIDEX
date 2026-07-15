#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_swift

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-3B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:?Set ADAPTER_PATH to the SFT-2 LoRA checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/sft2_merged}"

swift export \
  --model "${MODEL_ID}" \
  --adapters "${ADAPTER_PATH}" \
  --load_args false \
  --merge_lora true \
  --output_dir "${OUTPUT_DIR}"
