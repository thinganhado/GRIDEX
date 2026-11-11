#!/usr/bin/env python3
"""Create only GRIDEX training grid images and word-tier transcript JSON.

Uses GRIDEX's training-matched renderer; faster-whisper supplies timestamps.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

UTIL_DIR = Path(__file__).resolve().parents[2] / "inference" / "util"
sys.path.insert(0, str(UTIL_DIR))

from spectrogram import (  # noqa: E402
    audio_to_spec_image,
    draw_numbered_grid,
    save_with_training_axes,
)
from transcribe import transcribe_audio  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the training PNG and word-tier JSON for paired real/fake audio."
    )
    parser.add_argument("--pairs-csv", required=True, help="CSV with real_path and fake_path columns.")
    parser.add_argument("--data-root", required=True, help="GRIDEX data directory for training assets.")
    parser.add_argument("--real-audio-col", default="real_path")
    parser.add_argument("--fake-audio-col", default="fake_path")
    parser.add_argument("--asr-model", default="small")
    parser.add_argument("--language", default="en")
    parser.add_argument("--asr-device", default="auto")
    parser.add_argument("--asr-compute-type", default="auto")
    parser.add_argument("--vad-filter", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_pairs(path: Path, real_col: str, fake_col: str):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {real_col, fake_col}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Pairs CSV must include columns {sorted(required)}: {path}")
        rows = []
        for line, row in enumerate(reader, start=2):
            real = Path(str(row.get(real_col, "")).strip()).expanduser().resolve()
            fake = Path(str(row.get(fake_col, "")).strip()).expanduser().resolve()
            if not real.is_file() or not fake.is_file():
                raise FileNotFoundError(f"Missing paired audio at CSV row {line}: {real}, {fake}")
            rows.append((real, fake))
    if not rows:
        raise ValueError(f"No audio pairs found in {path}")
    return rows


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    image_root = data_root / "spectrograms" / "grid"
    transcript_root = data_root / "transcripts" / "mfa"
    image_root.mkdir(parents=True, exist_ok=True)
    transcript_root.mkdir(parents=True, exist_ok=True)

    pairs = read_pairs(Path(args.pairs_csv).expanduser().resolve(), args.real_audio_col, args.fake_audio_col)
    seen = set()
    for index, (real_audio, fake_audio) in enumerate(pairs, start=1):
        sample_id = fake_audio.stem
        if sample_id in seen:
            raise ValueError(f"Duplicate fake-audio sample ID: {sample_id}")
        seen.add(sample_id)
        image_path = image_root / f"{sample_id}_grid_img_edge_number_axes.png"
        transcript_path = transcript_root / f"{sample_id}.json"
        if not args.overwrite and image_path.is_file() and transcript_path.is_file():
            print(f"[{index}/{len(pairs)}] exists {sample_id}")
            continue

        if args.overwrite or not image_path.is_file():
            grid = draw_numbered_grid(audio_to_spec_image(fake_audio))
            save_with_training_axes(grid, image_path)
        if args.overwrite or not transcript_path.is_file():
            words, _text, _backend = transcribe_audio(
                real_audio,
                backend="faster-whisper",
                model=args.asr_model,
                language=args.language,
                device=args.asr_device,
                compute_type=args.asr_compute_type,
                vad_filter=args.vad_filter,
            )
            if not words:
                raise ValueError(f"No word timestamps returned for paired real audio: {real_audio}")
            transcript_path.write_text(
                json.dumps(
                    {"tiers": {"words": {"entries": [
                        [float(word["start"]), float(word["end"]), str(word["word"])]
                        for word in words
                    ]}}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(f"[{index}/{len(pairs)}] prepared {sample_id}")


if __name__ == "__main__":
    main()
