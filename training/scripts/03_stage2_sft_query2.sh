#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_swift

# ===== User-settable =====
# SFT-2 is initialized independently from the same base model as SFT-1.
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-3B-Instruct}"
Q2_JSON_IN="${Q2_JSON_IN:-${DATA_ROOT}/processed/grpo2_gt_train_swift.json}"
Q2_VAL_JSON_IN="${Q2_VAL_JSON_IN:-${DATA_ROOT}/processed/grpo2_gt_val_swift.json}"
Q2_JSON_SWIFT="${Q2_JSON_SWIFT:-${DATA_ROOT}/processed/grpo2_train_sft_q2_swift.json}"
Q2_VAL_JSON_SWIFT="${Q2_VAL_JSON_SWIFT:-${DATA_ROOT}/processed/grpo2_val_sft_q2_swift.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/sft2_query2}"
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
SFT_DEBUG_DATA="${SFT_DEBUG_DATA:-1}"
SFT_DEBUG_SAMPLES="${SFT_DEBUG_SAMPLES:-3}"
CONVERT_ONLY="${CONVERT_ONLY:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"

TMPDIR_BASE="${TMPDIR_BASE:-/tmp/${USER:-user}_mswift_q2}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
MAX_PIXELS="${MAX_PIXELS:-524288}"
DEEPSPEED="${DEEPSPEED:-zero2}"
EXPECTED_TRAIN_SAMPLES="${EXPECTED_TRAIN_SAMPLES:-10320}"
EXPECTED_VAL_SAMPLES="${EXPECTED_VAL_SAMPLES:-10192}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

mkdir -p "$(dirname "${Q2_JSON_SWIFT}")" \
  "${CACHE_ROOT}/triton" "${CACHE_ROOT}/torch_extensions" "${CACHE_ROOT}/hf" \
  "${CACHE_ROOT}/xdg_cache" "${CACHE_ROOT}/modelscope" "${CACHE_ROOT}/datasets" "${TMPDIR_BASE}"

export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export HF_HOME="${CACHE_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export DATASETS_CACHE="${CACHE_ROOT}/datasets"
export MODELSCOPE_CACHE="${CACHE_ROOT}/modelscope"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg_cache"
export TMPDIR="${TMPDIR_BASE}"
unset TRANSFORMERS_CACHE

if [[ ! -f "${Q2_JSON_IN}" ]]; then
  echo "ERROR: Q2 dataset not found: ${Q2_JSON_IN}"
  exit 1
fi
if [[ ! -f "${Q2_VAL_JSON_IN}" ]]; then
  echo "ERROR: Q2 validation dataset not found: ${Q2_VAL_JSON_IN}"
  exit 1
fi

# ===== Convert Q2-only GT data into a 2-turn SFT dataset =====
if [[ "${SKIP_CONVERT}" != "1" ]]; then
python - "${Q2_JSON_IN}" "${Q2_JSON_SWIFT}" "${Q2_VAL_JSON_IN}" "${Q2_VAL_JSON_SWIFT}" \
  "${SYSTEM_PROMPT}" "${EXPECTED_TRAIN_SAMPLES}" "${EXPECTED_VAL_SAMPLES}" <<'PY'
import json
import sys
from pathlib import Path

train_src = Path(sys.argv[1]).expanduser().resolve()
train_dst = Path(sys.argv[2]).expanduser().resolve()
val_src = Path(sys.argv[3]).expanduser().resolve()
val_dst = Path(sys.argv[4]).expanduser().resolve()
system_prompt = str(sys.argv[5] or "").strip()
expected_train = int(sys.argv[6])
expected_val = int(sys.argv[7])

def text_blocks_from_content(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content.strip()}] if content.strip() else []
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    txt = str(item.get("text", "")).strip()
                    if txt:
                        out.append({"type": "text", "text": txt})
                elif item.get("type") == "image":
                    out.append({"type": "image", "image": item.get("image")})
        return out
    return []

def normalize_user_message(msg):
    content = msg.get("content", [])
    blocks = text_blocks_from_content(content)
    return {"role": "user", "content": blocks}

def convert(src, dst, required_split, expected):
    data = json.loads(src.read_text(encoding="utf-8"))
    out = []
    wrong_split = []
    invalid = []
    for row in data:
        sid = str(row.get("sample_id", "")).strip()
        if required_split not in sid:
            wrong_split.append(sid)
            continue
        msgs = row.get("messages") or row.get("conversations") or []
        gt = str(row.get("gt_prompt2", "")).strip()
        transcript = str(row.get("transcript", "")).strip()
        if not isinstance(msgs, list) or not msgs or not gt or not transcript:
            invalid.append(sid)
            continue
        user = normalize_user_message(msgs[0] if isinstance(msgs[0], dict) else {"role": "user", "content": []})
        assistant = {"role": "assistant", "content": [{"type": "text", "text": gt}]}
        out.append({
            "messages": [user, assistant],
            "sample_id": sid,
            "prompt1_output": str(row.get("prompt1_output", "")).strip(),
            "gt_prompt2": gt,
            "transcript": transcript,
        })
    if wrong_split:
        raise ValueError(f"{src}: records outside {required_split}: {wrong_split[:20]}")
    if invalid:
        raise ValueError(f"{src}: records missing messages, transcript, or gt_prompt2: {invalid[:20]}")
    if not out:
        raise ValueError(f"{src}: conversion produced zero records")
    if expected > 0 and len(out) != expected:
        raise ValueError(f"{src}: got {len(out)} records, expected {expected}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {dst} (n={len(out)}, split={required_split})")

convert(train_src, train_dst, "_LA_T_", expected_train)
convert(val_src, val_dst, "_LA_D_", expected_val)
PY
else
  if [[ ! -f "${Q2_JSON_SWIFT}" ]]; then
    echo "ERROR: SKIP_CONVERT=1 but Q2_JSON_SWIFT not found: ${Q2_JSON_SWIFT}" >&2
    exit 1
  fi
  if [[ ! -f "${Q2_VAL_JSON_SWIFT}" ]]; then
    echo "ERROR: SKIP_CONVERT=1 but Q2_VAL_JSON_SWIFT not found: ${Q2_VAL_JSON_SWIFT}" >&2
    exit 1
  fi
  echo "[done] SKIP_CONVERT=1; using existing dataset: ${Q2_JSON_SWIFT}"
fi

if [[ "${SFT_DEBUG_DATA}" == "1" ]]; then
  python - "${Q2_JSON_SWIFT}" "${SFT_DEBUG_SAMPLES}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1]).expanduser().resolve()
n_show = max(1, int(sys.argv[2]))
data = json.loads(src.read_text(encoding="utf-8"))
print(f"[q2_dbg] rows={len(data)}")

def text_only(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            str(item.get("text", "")).strip()
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
    return str(content or "").strip()

def has_image(content):
    return isinstance(content, list) and any(
        isinstance(item, dict) and item.get("type") == "image" for item in content
    )

for i, row in enumerate(data[:n_show]):
    print(f"[q2_dbg] sample={i} sample_id={row.get('sample_id', f'row_{i}')}")
    for j, msg in enumerate(row.get("messages", [])):
        txt = text_only(msg.get("content", "")).replace("\n", " ")
        if len(txt) > 220:
            txt = txt[:220] + "..."
        print(f"[q2_dbg]   turn={j} role={msg.get('role')} has_image={has_image(msg.get('content'))} text={txt!r}")
PY
fi

if [[ "${CONVERT_ONLY}" == "1" ]]; then
  echo "[done] conversion only; skipping training."
  echo "[done] q2_json_swift=${Q2_JSON_SWIFT}"
  exit 0
fi

DEEPSPEED_ARGS=()
if [[ "${DEEPSPEED}" != "none" && -n "${DEEPSPEED}" ]]; then
  DEEPSPEED_ARGS=(--deepspeed "${DEEPSPEED}")
fi

# ===== Train Q2-only SFT adapter =====
NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift sft \
  --model "${MODEL_ID}" \
  --dataset "${Q2_JSON_SWIFT}" \
  --val_dataset "${Q2_VAL_JSON_SWIFT}" \
  --system "${SYSTEM_PROMPT}" \
  --tuner_type lora \
  --torch_dtype "${TORCH_DTYPE}" \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_llm false \
  --freeze_vit true \
  --freeze_aligner true \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --max_length 768 \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type cosine \
  --max_pixels "${MAX_PIXELS}" \
  "${DEEPSPEED_ARGS[@]}" \
  --eval_strategy steps \
  --eval_steps "${SAVE_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --output_dir "${OUTPUT_DIR}" \
  --report_to tensorboard
