#!/usr/bin/env python3
"""Record lightweight health telemetry for a running MS-Swift job."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def trainer_pids(run_dir: Path) -> list[int]:
    marker = f"--output_dir\x00{run_dir}"
    found = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            text = cmdline.read_bytes().decode(errors="replace")
            if marker in text and any(
                entrypoint in text
                for entrypoint in ("swift/cli/sft.py", "swift/cli/rlhf.py")
            ):
                found.append(int(cmdline.parent.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return sorted(found)


def gpu_state() -> dict:
    nvidia_smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    command = [
        nvidia_smi,
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.check_output(command, text=True, timeout=10).strip().split(",")
        values = [value.strip() for value in raw]
        return {
            "gpu_util_pct": int(values[0]),
            "gpu_memory_mib": int(values[1]),
            "gpu_memory_total_mib": int(values[2]),
            "gpu_power_w": float(values[3]),
            "gpu_temp_c": int(values[4]),
        }
    except Exception as exc:
        return {"gpu_error": f"{type(exc).__name__}: {exc}"}


def latest_metrics(run_dir: Path) -> tuple[dict, float | None]:
    logs = list(run_dir.rglob("logging.jsonl"))
    if not logs:
        return {}, None
    log = max(logs, key=lambda path: path.stat().st_mtime)
    age = time.time() - log.stat().st_mtime
    lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}, age
    try:
        return json.loads(lines[-1]), age
    except json.JSONDecodeError:
        return {"last_line": lines[-1][-1000:]}, age


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--startup-grace", type=float, default=300)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "monitor.jsonl"
    start = time.time()

    while True:
        pids = trainer_pids(run_dir)
        metrics, log_age = latest_metrics(run_dir)
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "trainer_pids": pids,
            "log_age_seconds": None if log_age is None else round(log_age, 1),
            **gpu_state(),
            "metrics": metrics,
        }
        warnings = []
        if log_age is not None and log_age > max(600, args.interval * 5):
            warnings.append("training_log_stale")
        for key in ("loss", "grad_norm", "eval_loss"):
            value = metrics.get(key)
            if isinstance(value, float) and not math.isfinite(value):
                warnings.append(f"non_finite_{key}")
        if record.get("gpu_temp_c", 0) >= 85:
            warnings.append("gpu_temperature_high")
        if warnings:
            record["warnings"] = warnings

        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        if not pids and time.time() - start >= args.startup_grace:
            final = {
                "time": datetime.now(timezone.utc).isoformat(),
                "monitor_exit": "trainer_process_not_found",
            }
            with output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(final) + "\n")
            break
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    main()
