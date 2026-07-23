#!/usr/bin/env python3
"""Rewrite verified Q2 SFT JSON image references for native Linux training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def image_field(row: dict) -> dict:
    for message in row.get("messages", []):
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                return item
    raise ValueError(f"No image block for sample {row.get('sample_id')!r}")


def stage(source: Path, destination: Path, image_dir: Path, split: str, expected: int) -> None:
    rows = json.loads(source.read_text(encoding="utf-8"))
    if len(rows) != expected:
        raise ValueError(f"{source}: records={len(rows)}, expected={expected}")

    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or split not in sample_id:
            raise ValueError(f"{source}: invalid {split} sample_id={sample_id!r}")
        if sample_id in seen:
            raise ValueError(f"{source}: duplicate sample_id={sample_id}")
        seen.add(sample_id)

        image = image_dir / f"{sample_id}_grid_img_edge_number_axes.png"
        if not image.is_file():
            raise FileNotFoundError(image)
        image_field(row)["image"] = str(image)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(destination)
    print(f"WROTE {destination} records={len(rows)} split={split}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    image_dir = args.image_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    stage(args.train_json, output_dir / args.train_json.name, image_dir, "_LA_T_", 10320)
    stage(args.val_json, output_dir / args.val_json.name, image_dir, "_LA_D_", 10192)


if __name__ == "__main__":
    main()
