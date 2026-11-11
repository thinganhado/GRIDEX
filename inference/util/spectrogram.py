#!/usr/bin/env python3
"""Create the spectrogram/grid image representation used by GRIDEX training."""

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SR = 16000
N_MELS = 128
N_FFT = 1024
HOP = 256
TOP_DB = 60.0
IMAGE_SIZE = 768
GRID_SIZE = 4
AXIS_DURATION_SEC = 4.0
AXIS_DPI = 120
REGION_FONT_SIZE = 10

TRAINING_GRID_SUFFIX = "_grid_img_edge_number.png"
TRAINING_MODEL_SUFFIX = "_grid_img_edge_number_axes.png"


def audio_to_spec_image(audio_path: Path) -> Image.Image:
    """Match GPT-4V-AD/make_specs_768.py."""
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=1.0, top_db=TOP_DB)
    mel_norm = (mel_db + TOP_DB) / TOP_DB
    mel_color = matplotlib.colormaps["magma"](mel_norm)
    mel_img = (mel_color[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)
    mel_img = np.flipud(mel_img)
    return Image.fromarray(mel_img, mode="RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC
    )


def _region_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(os.environ["GRIDEX_ARIAL_FONT"]).expanduser()
        if os.environ.get("GRIDEX_ARIAL_FONT")
        else Path("__unset_gridex_font__"),
        Path(__file__).resolve().parents[1] / "assets" / "arial.ttf",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), REGION_FONT_SIZE)
    try:
        return ImageFont.truetype("Arial.ttf", REGION_FONT_SIZE)
    except OSError:
        return ImageFont.load_default()


def draw_numbered_grid(spec_img: Image.Image) -> Image.Image:
    """Match the fixed-grid branch of GPT-4V-AD/region_division.py.

    Each 192x192 cell receives its own one-pixel inner boundary, making shared
    boundaries two pixels wide, and IDs use the training renderer's placement.
    """
    rgb = np.asarray(spec_img.convert("RGB")).copy()
    cell = IMAGE_SIZE // GRID_SIZE

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            y1, y2 = row * cell, (row + 1) * cell
            x1, x2 = col * cell, (col + 1) * cell
            rgb[y1, x1:x2] = 255
            rgb[y2 - 1, x1:x2] = 255
            rgb[y1:y2, x1] = 255
            rgb[y1:y2, x2 - 1] = 255

    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = _region_font()
    region_id = 1
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            xc = col * cell + (cell - 1) / 2.0
            yc = row * cell + (cell - 1) / 2.0
            text = str(region_id)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x1 = int(xc - text_width // 2)
            y1 = int(yc - text_height // 2)
            draw.rectangle(
                [(x1, y1 + 2), (x1 + text_width, y1 + text_height)],
                fill=(128, 128, 128),
            )
            draw.text((x1, y1), text, font=font, fill=(255, 255, 255))
            region_id += 1
    return image


def save_with_training_axes(grid_img: Image.Image, out_path: Path) -> None:
    """Match region_division._save_with_axes_image and its 987x908 output."""
    rgb = np.asarray(grid_img.convert("RGB"))
    height, width = rgb.shape[:2]
    fig, ax = plt.subplots(
        figsize=((width + 220) / AXIS_DPI, (height + 140) / AXIS_DPI),
        dpi=AXIS_DPI,
    )
    ax.imshow(
        np.flipud(rgb),
        origin="lower",
        aspect="auto",
        extent=[0.0, AXIS_DURATION_SEC, 0, N_MELS - 1],
    )
    mel_hz = librosa.mel_frequencies(
        n_mels=N_MELS, fmin=0.0, fmax=SR / 2.0
    )
    y_tick_bins = np.linspace(0, N_MELS - 1, 6).round().astype(int)
    ax.set_xticks(np.linspace(0.0, AXIS_DURATION_SEC, 6))
    ax.set_yticks(y_tick_bins)
    ax.set_yticklabels([f"{mel_hz[b] / 1000.0:.1f}" for b in y_tick_bins])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title("")
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.13, top=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=AXIS_DPI)
    plt.close(fig)


def output_paths(out_path: Path) -> tuple[Path, Path]:
    grid_path = out_path.with_name(f"{out_path.stem}{TRAINING_GRID_SUFFIX}")
    model_path = out_path.with_name(f"{out_path.stem}{TRAINING_MODEL_SUFFIX}")
    return grid_path, model_path


def save_spec_and_grid(audio_path: Path, out_path: Path) -> Path:
    spec_img = audio_to_spec_image(audio_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spec_img.save(out_path)
    grid_path, model_path = output_paths(out_path)
    grid_img = draw_numbered_grid(spec_img)
    grid_img.save(grid_path)
    save_with_training_axes(grid_img, model_path)
    return model_path


def collect_from_in_dir(in_dir: Path) -> List[Tuple[Path, Path]]:
    audios = sorted(list(in_dir.rglob("*.wav")) + list(in_dir.rglob("*.flac")))
    if not audios:
        raise SystemExit(f"No .wav/.flac files found under {in_dir}")
    return [(audio, audio.relative_to(in_dir).with_suffix(".png")) for audio in audios]


def collect_from_pairs_csv(pairs_csv: Path, audio_col: str) -> List[Tuple[Path, Path]]:
    seen = set()
    pairs: List[Tuple[Path, Path]] = []
    with pairs_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or audio_col not in reader.fieldnames:
            raise SystemExit(f"CSV must contain column '{audio_col}'")
        for row in reader:
            audio_str = str(row[audio_col]).strip()
            if not audio_str:
                continue
            audio_path = Path(audio_str).expanduser().resolve()
            key = str(audio_path)
            if key not in seen:
                seen.add(key)
                pairs.append((audio_path, Path(f"{audio_path.stem}.png")))
    if not pairs:
        raise SystemExit(
            f"No valid audio paths found in column '{audio_col}' from {pairs_csv}"
        )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GRIDEX training-matched spectrogram/grid/axes PNGs."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--in-dir", help="Input root scanned recursively for .wav/.flac")
    src.add_argument("--pairs-csv", help="CSV containing audio paths")
    parser.add_argument("--pairs-audio-col", default="real_path")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    pairs = (
        collect_from_in_dir(Path(args.in_dir).expanduser().resolve())
        if args.in_dir
        else collect_from_pairs_csv(Path(args.pairs_csv).expanduser().resolve(), args.pairs_audio_col)
    )
    for index, (audio_path, out_rel) in enumerate(pairs, start=1):
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio path does not exist: {audio_path}")
        out_path = out_dir / out_rel
        grid_path, model_path = output_paths(out_path)
        if not args.overwrite and out_path.is_file() and grid_path.is_file() and model_path.is_file():
            print(f"[{index}/{len(pairs)}] exists {model_path}")
            continue
        generated = save_spec_and_grid(audio_path, out_path)
        print(f"[{index}/{len(pairs)}] generated {generated}")


if __name__ == "__main__":
    main()
