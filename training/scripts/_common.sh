#!/usr/bin/env bash

# Shared path setup for GRIDEX training entry points.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GRIDEX_ROOT="$(cd "${TRAINING_ROOT}/.." && pwd)"

MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-$(cd "${GRIDEX_ROOT}/.." && pwd)/ms-swift}"
DATA_ROOT="${DATA_ROOT:-${GRIDEX_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${GRIDEX_ROOT}/outputs}"
CACHE_ROOT="${CACHE_ROOT:-${GRIDEX_ROOT}/.cache/ms-swift}"

export GRIDEX_ROOT TRAINING_ROOT MS_SWIFT_ROOT DATA_ROOT OUTPUT_ROOT CACHE_ROOT

if [[ -d "${MS_SWIFT_ROOT}/swift" ]]; then
  export PYTHONPATH="${MS_SWIFT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

require_swift() {
  if ! command -v swift >/dev/null 2>&1; then
    echo "ERROR: the 'swift' command is not available." >&2
    echo "Clone MS-Swift at ${MS_SWIFT_ROOT} and install it with: pip install -e \"${MS_SWIFT_ROOT}\"" >&2
    return 1
  fi
}
