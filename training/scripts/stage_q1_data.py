#!/usr/bin/env python3
"""Stage Q1 JSON and referenced images onto native Linux storage."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def image_field(row: dict) -> dict:
    for message in row.get("messages", []):
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("image"):
                return item
    raise ValueError(f"No image field for sample {row.get('source_sample_id')!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    image_dir = output_root / "images"
    processed_dir = output_root / "processed"
    image_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[tuple[Path, list[dict]]] = []
    copies: dict[Path, Path] = {}
    destinations: dict[Path, Path] = {}
    for json_path in (args.train_json, args.val_json):
        rows = json.loads(json_path.read_text(encoding="utf-8"))
        datasets.append((json_path, rows))
        for row in rows:
            field = image_field(row)
            source = Path(field["image"]).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = image_dir / source.name
            previous = destinations.get(destination)
            if previous is not None and previous != source:
                raise ValueError(f"Image basename collision: {previous} and {source}")
            destinations[destination] = source
            copies[source] = destination
            field["image"] = str(destination)

    def copy_one(pair: tuple[Path, Path]) -> tuple[bool, int]:
        source, destination = pair
        size = source.stat().st_size
        if destination.is_file() and destination.stat().st_size == size:
            return False, size
        temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
        return True, size

    copied = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for was_copied, size in executor.map(copy_one, copies.items()):
            copied += int(was_copied)
            total_bytes += size

    for source_json, rows in datasets:
        destination_json = processed_dir / source_json.name
        temporary_json = destination_json.with_suffix(".json.tmp")
        temporary_json.write_text(
            json.dumps(rows, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary_json.replace(destination_json)
        print(f"WROTE {destination_json} records={len(rows)}")

    print(
        f"IMAGES unique={len(copies)} copied={copied} "
        f"referenced_gib={total_bytes / 2**30:.2f} directory={image_dir}"
    )


if __name__ == "__main__":
    main()
