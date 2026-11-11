#!/usr/bin/env python3
import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create word-level timestamps from an audio file and print '[start-end] word' text."
    )
    parser.add_argument("--audio", required=True, help="Input audio file.")
    parser.add_argument(
        "--backend",
        default="faster-whisper",
        choices=("faster-whisper",),
        help="Word-timestamp backend.",
    )
    parser.add_argument("--model", default="small", help="ASR model name or local path.")
    parser.add_argument("--language", default=None, help="Optional language code, e.g. en.")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="auto, float16, int8, float32, etc. Backend-specific.",
    )
    parser.add_argument("--vad-filter", action="store_true", help="Use VAD filter for faster-whisper backend.")
    parser.add_argument("--output-txt", default=None, help="Output text path. Default: <audio>.word_times.txt")
    parser.add_argument("--output-json", default=None, help="Output JSON path. Default: <audio>.word_times.json")
    parser.add_argument(
        "--output-mfa-json",
        default=None,
        help="Optional MFA-like JSON path with tiers.words.entries = [[start, end, word], ...].",
    )
    return parser.parse_args()


def resolve_device(device):
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_compute_type(compute_type, device):
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def normalize_words(words):
    out = []
    for row in words:
        try:
            start = float(row["start"])
            end = float(row["end"])
        except Exception:
            continue
        word = str(row.get("word", "")).strip()
        if not word:
            continue
        out.append({"start": start, "end": end, "word": word})
    return out


def run_faster_whisper(args, device, compute_type):
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    transcribe_kwargs = {"word_timestamps": True, "vad_filter": bool(args.vad_filter)}
    if args.language:
        transcribe_kwargs["language"] = args.language

    segments, _info = model.transcribe(str(Path(args.audio).expanduser().resolve()), **transcribe_kwargs)
    words = []
    for segment in segments:
        for word in segment.words or []:
            words.append({"start": word.start, "end": word.end, "word": word.word})
    return normalize_words(words)


def get_words(args, device, compute_type):
    if args.backend != "faster-whisper":
        raise ValueError(f"Unsupported transcription backend: {args.backend}")
    return run_faster_whisper(args, device, compute_type), args.backend


def format_word_times(words):
    return " ".join(f"[{w['start']:.2f}-{w['end']:.2f}] {w['word']}" for w in words)


def transcribe_audio(
    audio: Path,
    backend="faster-whisper",
    model="small",
    language=None,
    device="auto",
    compute_type="auto",
    vad_filter=False,
):
    """Return word records, formatted text, and the selected backend."""
    audio = Path(audio).expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"Audio does not exist: {audio}")
    args = Namespace(
        audio=str(audio),
        backend=backend,
        model=model,
        language=language,
        device=device,
        compute_type=compute_type,
        vad_filter=vad_filter,
    )
    resolved_device = resolve_device(device)
    resolved_compute_type = resolve_compute_type(compute_type, resolved_device)
    words, selected_backend = get_words(args, resolved_device, resolved_compute_type)
    if not words:
        raise ValueError(f"Transcription produced no word timestamps: {audio}")
    return words, format_word_times(words), selected_backend


def main():
    args = parse_args()
    audio = Path(args.audio).expanduser().resolve()
    if not audio.exists():
        raise FileNotFoundError(f"--audio does not exist: {audio}")

    words, text, backend = transcribe_audio(
        audio,
        backend=args.backend,
        model=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        vad_filter=args.vad_filter,
    )

    out_txt = Path(args.output_txt).expanduser().resolve() if args.output_txt else audio.with_suffix(audio.suffix + ".word_times.txt")
    out_json = Path(args.output_json).expanduser().resolve() if args.output_json else audio.with_suffix(audio.suffix + ".word_times.json")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(text + "\n", encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "audio": str(audio),
                "backend": backend,
                "model": args.model,
                "language": args.language,
                "words": words,
                "text": text,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.output_mfa_json:
        out_mfa = Path(args.output_mfa_json).expanduser().resolve()
        out_mfa.parent.mkdir(parents=True, exist_ok=True)
        entries = [[w["start"], w["end"], w["word"]] for w in words]
        out_mfa.write_text(
            json.dumps({"tiers": {"words": {"entries": entries}}}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"saved_mfa_json: {out_mfa}")

    print(text)
    print(f"saved_txt: {out_txt}")
    print(f"saved_json: {out_json}")
    print(f"backend: {backend}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
