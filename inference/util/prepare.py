#!/usr/bin/env python3
"""Prepare training-matched GRIDEX images and both inference metadata files."""

import argparse
import csv
import json
from pathlib import Path

from spectrogram import collect_from_in_dir, output_paths, save_spec_and_grid
from transcribe import transcribe_audio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an audio directory into GRIDEX Q1 JSON and Q2 CSV inputs."
    )
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--asr-backend", default="faster-whisper", choices=("faster-whisper",))
    parser.add_argument("--asr-model", default="small")
    parser.add_argument("--language", default=None)
    parser.add_argument("--asr-device", default="auto")
    parser.add_argument("--asr-compute-type", default="auto")
    parser.add_argument("--vad-filter", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    audio_dir = Path(args.audio_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    if not audio_dir.is_dir():
        raise NotADirectoryError(f"--audio-dir does not exist: {audio_dir}")

    image_dir = work_dir / "images"
    transcript_dir = work_dir / "transcripts"
    q1_meta_path = work_dir / "q1_meta.json"
    q2_meta_path = work_dir / "q2_meta.csv"
    image_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    q1_records = []
    q2_records = []
    seen_sample_ids = set()
    pairs = collect_from_in_dir(audio_dir)

    for index, (audio_path, relative_png) in enumerate(pairs, start=1):
        # Output folders may retain relative structure, but IDs must remain unique
        # because Q1 persists records under <sample_id>/json.
        sample_id = audio_path.stem
        if sample_id in seen_sample_ids:
            raise ValueError(
                f"Duplicate audio stem '{sample_id}'. GRIDEX requires globally unique sample IDs."
            )
        seen_sample_ids.add(sample_id)

        base_png = image_dir / relative_png
        _, model_image = output_paths(base_png)
        if args.overwrite or not model_image.is_file():
            model_image = save_spec_and_grid(audio_path, base_png)

        transcript_txt = transcript_dir / f"{sample_id}.word_times.txt"
        transcript_json = transcript_dir / f"{sample_id}.word_times.json"
        if not args.overwrite and transcript_txt.is_file() and transcript_json.is_file():
            transcript = transcript_txt.read_text(encoding="utf-8-sig").strip()
            if not transcript:
                raise ValueError(f"Cached transcript is empty: {transcript_txt}")
        else:
            words, transcript, backend = transcribe_audio(
                audio_path,
                backend=args.asr_backend,
                model=args.asr_model,
                language=args.language,
                device=args.asr_device,
                compute_type=args.asr_compute_type,
                vad_filter=args.vad_filter,
            )
            transcript_txt.write_text(transcript + "\n", encoding="utf-8")
            transcript_json.write_text(
                json.dumps(
                    {
                        "audio": str(audio_path),
                        "backend": backend,
                        "model": args.asr_model,
                        "language": args.language,
                        "words": words,
                        "text": transcript,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        q1_records.append(
            {
                "sample_id": sample_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": str(model_image)}],
                    }
                ],
            }
        )
        q2_records.append(
            {
                "sample_id": sample_id,
                "img_path": str(model_image),
                "transcript": transcript,
            }
        )
        print(f"[{index}/{len(pairs)}] prepared {sample_id}")

    q1_meta_path.write_text(
        json.dumps(q1_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with q2_meta_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "img_path", "transcript"))
        writer.writeheader()
        writer.writerows(q2_records)

    print(f"q1_meta_json={q1_meta_path}")
    print(f"q2_meta_csv={q2_meta_path}")
    print(f"samples={len(q1_records)}")


if __name__ == "__main__":
    main()
