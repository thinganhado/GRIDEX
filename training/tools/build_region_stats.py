#!/usr/bin/env python3
"""Write GRIDEX region overlap statistics without temporary mask/overlay files.

The smoothed relative-magnitude difference follows GPT-4V-AD's
make_masks_overlay.py; regions are the fixed GRIDEX 4x4 grid.
"""

import argparse
import csv
import math
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter1d

EPS = 1e-8


def read_pairs(path: Path, real_col: str, fake_col: str):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {real_col, fake_col}.issubset(reader.fieldnames):
            raise ValueError(f"Pairs CSV must include columns {real_col!r} and {fake_col!r}")
        pairs = []
        for row in reader:
            real = Path(str(row.get(real_col, "")).strip()).expanduser().resolve()
            fake = Path(str(row.get(fake_col, "")).strip()).expanduser().resolve()
            if not real.is_file() or not fake.is_file():
                raise FileNotFoundError(f"Missing audio pair: {real}, {fake}")
            pairs.append((real, fake))
    if not pairs:
        raise ValueError(f"No audio pairs in {path}")
    return pairs


def load_mono_16k(path: Path, target_sr: int):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return np.asarray(audio, dtype=np.float32)


def magnitude(audio, sr: int, n_fft: int, hop: int, win_length: int):
    tensor = torch.from_numpy(audio)
    window = torch.hann_window(win_length, periodic=True)
    stft = torch.stft(
        tensor, n_fft=n_fft, hop_length=hop, win_length=win_length,
        window=window, center=True, return_complex=True,
    )
    return stft.abs().numpy().astype(np.float32, copy=False)


def smooth(mag, time_size, freq_size, time_var, freq_var):
    result = mag
    for axis, size, variance in ((1, time_size, time_var), (0, freq_size, freq_var)):
        if size % 2 == 0:
            raise ValueError("Smoothing sizes must be odd")
        if variance > 0:
            sigma = math.sqrt(variance)
            truncate = max(((size - 1) / 2.0) / sigma, 1e-6)
            result = gaussian_filter1d(result, sigma=sigma, axis=axis, mode="nearest", truncate=truncate)
    return result


def pair_difference_mask(real, fake, args):
    real_mag = magnitude(real, args.sample_rate, args.n_fft, args.hop_length, args.win_length)
    fake_mag = magnitude(fake, args.sample_rate, args.n_fft, args.hop_length, args.win_length)
    frames = min(real_mag.shape[1], fake_mag.shape[1])
    real_mag, fake_mag = real_mag[:, :frames], fake_mag[:, :frames]
    real_smooth = smooth(real_mag, args.gauss_size_t, args.gauss_size_f, args.gauss_var_t, args.gauss_var_f)
    fake_smooth = smooth(fake_mag, args.gauss_size_t, args.gauss_size_f, args.gauss_var_t, args.gauss_var_f)
    normalized = np.abs(fake_smooth - real_smooth) / (real_smooth + EPS)
    finite = normalized[np.isfinite(normalized)]
    if finite.size == 0:
        raise ValueError("No finite values in normalized real/fake magnitude difference")
    tau = float(np.quantile(finite, args.threshold_quantile))
    mask = normalized > tau
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    mask = np.asarray(image.resize((args.image_size, args.image_size), Image.Resampling.NEAREST)) > 0
    return mask


def grid_masks(size: int, divisions: int):
    if size % divisions:
        raise ValueError("Image size must be divisible by grid size")
    cell = size // divisions
    masks = []
    # Matches region_division.py grid order: rows top-to-bottom, columns left-to-right.
    for row in range(divisions):
        for col in range(divisions):
            mask = np.zeros((size, size), dtype=bool)
            mask[row * cell:(row + 1) * cell, col * cell:(col + 1) * cell] = True
            # The source overlap pipeline flips region masks into STFT orientation.
            masks.append(np.flipud(mask))
    return masks


def parse_args():
    parser = argparse.ArgumentParser(description="Generate only the GRIDEX region overlap CSV.")
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--spectrogram-root", required=True, help="Directory containing the matching GRIDEX training PNGs.")
    parser.add_argument("--real-audio-col", default="real_path")
    parser.add_argument("--fake-audio-col", default="fake_path")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--gauss-size-t", type=int, default=3)
    parser.add_argument("--gauss-size-f", type=int, default=11)
    parser.add_argument("--gauss-var-t", type=float, default=3.0)
    parser.add_argument("--gauss-var-f", type=float, default=5.0)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--grid-size", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output_csv).expanduser().resolve()
    spectrogram_root = Path(args.spectrogram_root).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    regions = grid_masks(args.image_size, args.grid_size)
    rows = []
    pairs = read_pairs(Path(args.pairs_csv).expanduser().resolve(), args.real_audio_col, args.fake_audio_col)
    for index, (real_path, fake_path) in enumerate(pairs, start=1):
        expected_image = spectrogram_root / f"{fake_path.stem}_grid_img_edge_number_axes.png"
        if not expected_image.is_file():
            raise FileNotFoundError(f"Missing training grid image for {fake_path.stem}: {expected_image}")
        diff_mask = pair_difference_mask(load_mono_16k(real_path, args.sample_rate), load_mono_16k(fake_path, args.sample_rate), args)
        for region_id, region in enumerate(regions, start=1):
            region_pixels = int(region.sum())
            overlap = int(np.logical_and(region, diff_mask).sum())
            rows.append({
                "image": fake_path.stem,
                "method": "grid",
                "region_id": region_id,
                "region_pixels": region_pixels,
                "overlap_pixels": overlap,
                "coverage": overlap / region_pixels if region_pixels else 0.0,
                "present": int(overlap > 0),
            })
        print(f"[{index}/{len(pairs)}] scored {fake_path.stem}")

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "image", "method", "region_id", "region_pixels", "overlap_pixels", "coverage", "present"
        ))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} region rows -> {output}")


if __name__ == "__main__":
    main()
