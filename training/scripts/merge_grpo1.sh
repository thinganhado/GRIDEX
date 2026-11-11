#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_swift

# ADAPTER_PATH can be an output checkpoint dir or output root with args.json/checkpoints
ADAPTER_PATH="${ADAPTER_PATH:?Set ADAPTER_PATH to the GRPO-1 LoRA checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/grpo1_merged}"

swift export \
  --adapters "${ADAPTER_PATH}" \
  --merge_lora true \
  --output_dir "${OUTPUT_DIR}"

