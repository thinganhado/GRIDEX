#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="grpo2_phase_a_5090_v1"
GRIDEX_ROOT="/mnt/c/Users/Admin/Documents/GitHub/GRIDEX"
LOG="$GRIDEX_ROOT/training_logs/${RUN_TAG}.log"
STDERR_LOG="$GRIDEX_ROOT/training_logs/${RUN_TAG}.stderr.log"
MONITOR_LOG="$GRIDEX_ROOT/training_logs/${RUN_TAG}.monitor.log"
RUN_BASE="/home/gridex/training_outputs/grpo2_curriculum/${RUN_TAG}"
PHASE_A_ROOT="$RUN_BASE/phase_a"
MERGED_DIR="$RUN_BASE/phase_a_merged"

emit_status() {
  python3 - <<'PY'
import json, re, time
from pathlib import Path

run_tag = "grpo2_phase_a_5090_v1"
log = Path("/mnt/c/Users/Admin/Documents/GitHub/GRIDEX/training_logs/grpo2_phase_a_5090_v1.log")
stderr_log = Path("/mnt/c/Users/Admin/Documents/GitHub/GRIDEX/training_logs/grpo2_phase_a_5090_v1.stderr.log")
run_base = Path("/home/gridex/training_outputs/grpo2_curriculum/grpo2_phase_a_5090_v1")
phase_a = run_base / "phase_a"
merged = run_base / "phase_a_merged"

text = log.read_text(errors="ignore")[-8000000:] if log.exists() else ""
steps = re.findall(r"\[p2_metrics\] step=(\d+).*?valid_format_rate=([0-9.]+%).*?reward_var_within_group=([0-9.]+)", text)
last_step = int(steps[-1][0]) if steps else None

logging_rows = []
for candidate in sorted(phase_a.glob("*/logging.jsonl")):
    try:
        logging_rows = [json.loads(line) for line in candidate.read_text(errors="ignore").splitlines() if line.strip()]
    except Exception:
        logging_rows = []

checkpoints = sorted(phase_a.glob("**/checkpoint-*"), key=lambda p: str(p))
err_text = ""
for p in (log, stderr_log):
    if p.exists():
        err_text += p.read_text(errors="ignore")[-2000000:]
bad = []
for pat in ("Traceback", "RuntimeError", "CUDA out of memory", "OutOfMemory", "Killed", "missing GT", "ERROR", "ValueError"):
    if pat.lower() in err_text.lower():
        for line in err_text.splitlines():
            low = line.lower()
            if pat.lower() in low and "ignore_args_error" not in low and "vllm_mode is not supported when use_vllm is false" not in low:
                bad.append(line[-500:])
                break

print("=" * 80)
print(time.strftime("%F %T %Z"))
print("run_tag:", run_tag)
print("latest_p2_metrics:", steps[-5:] if steps else "none")
print("progress_percent:", round(100 * last_step / 645, 2) if last_step is not None else "unknown")
if logging_rows:
    row = logging_rows[-1]
    print("latest_logging:", {
        "global_step/max_steps": row.get("global_step/max_steps"),
        "loss": row.get("loss"),
        "reward": row.get("reward"),
        "reward_std": row.get("reward_std"),
        "elapsed_time": row.get("elapsed_time"),
        "remaining_time": row.get("remaining_time"),
        "memory(GiB)": row.get("memory(GiB)"),
        "train_speed(s/it)": row.get("train_speed(s/it)"),
    })
print("latest_checkpoints:", [p.name for p in checkpoints[-5:]])
print("merged_exists:", merged.exists())
print("error_hits:", bad[:5])
PY
  nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader || true
  pgrep -af 'swift rlhf|swift/cli/rlhf|run_grpo2' || true
}

echo "Starting monitor for $RUN_TAG at $(date)" >> "$MONITOR_LOG"
while true; do
  {
    emit_status
  } >> "$MONITOR_LOG" 2>&1

  if [ -d "$MERGED_DIR" ]; then
    {
      echo "Merged Phase-A directory detected at $(date): $MERGED_DIR"
      find "$MERGED_DIR" -maxdepth 1 -type f | sort
    } >> "$MONITOR_LOG" 2>&1
    break
  fi

  if ! pgrep -af 'swift rlhf|swift/cli/rlhf|run_grpo2' >/dev/null; then
    echo "Training process no longer active at $(date); monitor stopping." >> "$MONITOR_LOG"
    break
  fi

  sleep 300
done

echo "Monitor finished at $(date)" >> "$MONITOR_LOG"
