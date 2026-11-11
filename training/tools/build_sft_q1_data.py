#!/usr/bin/env python3
import argparse
import csv
import json
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_PROMPT = "Select the top 3 regions that most likely contain spoof artifacts, ordered from most to least prominent spoof artifact evidence."
DEFAULT_INPUT_CSV = ""
DEFAULT_ORDER_CSV = ""
DEFAULT_IMAGE_DIR = ""
DEFAULT_IMAGE_SUFFIX = "_grid_img_edge_number_axes.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MS-Swift SFT dataset for query1 directly from a CSV."
    )
    parser.add_argument(
        "--input-csv",
        default=DEFAULT_INPUT_CSV,
        help=f"Input CSV path. Default: {DEFAULT_INPUT_CSV}",
    )
    parser.add_argument("--output-json", default=None, help="Output JSON path (single-file mode).")
    parser.add_argument("--train-json", default=None, help="Train JSON output path (split mode).")
    parser.add_argument("--val-json", default=None, help="Validation JSON output path (split mode).")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio for random split mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random split mode.")
    parser.add_argument(
        "--expected-train-samples",
        type=int,
        default=0,
        help="If positive, fail unless the constructed training partition has exactly this many samples.",
    )
    parser.add_argument(
        "--expected-val-samples",
        type=int,
        default=0,
        help="If positive, fail unless the constructed development partition has exactly this many samples.",
    )
    parser.add_argument("--split-by-path", action="store_true", help="Split train/val by path keys.")
    parser.add_argument("--train-key", default="_LA_T_", help="Path substring routed to train.")
    parser.add_argument("--val-key", default="_LA_D_", help="Path substring routed to val.")
    parser.add_argument(
        "--strict-path-split",
        action="store_true",
        help="Drop unmatched samples when --split-by-path is set.",
    )
    parser.add_argument(
        "--image-col",
        default="img_path",
        help="CSV column containing full image path. If missing/empty, sample_id+region_id mode is used.",
    )
    parser.add_argument("--sample-id-col", default="sample_id", help="CSV column containing sample id.")
    parser.add_argument("--region-id-col", default="region_id", help="CSV column containing region id.")
    parser.add_argument(
        "--order-csv",
        required=True,
        help="Required overlap-score CSV used to select and rank Query-1 targets.",
    )
    parser.add_argument(
        "--order-sample-id-col",
        default="image",
        help="Sample-id column in --order-csv.",
    )
    parser.add_argument(
        "--order-region-id-col",
        default="region_id",
        help="Region-id column in --order-csv.",
    )
    parser.add_argument(
        "--order-score-col",
        default="overlap_pixels",
        help="Score column in --order-csv. Higher means earlier rank.",
    )
    parser.add_argument("--order-method-col", default="method", help="Method column in --order-csv.")
    parser.add_argument("--order-method", default="grid", help="Only load this region method when the column exists.")
    parser.add_argument(
        "--require-order-scores",
        action="store_true",
        help="Fail if the order CSV is absent, empty, or lacks a score for any constructed target.",
    )
    parser.add_argument(
        "--verify-order",
        action="store_true",
        help="Verify every constructed top-3 target exactly matches descending order-CSV scores.",
    )
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Fail if any constructed spectrogram image path does not exist.",
    )
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, help="Image directory for sample_id mode.")
    parser.add_argument(
        "--image-suffix",
        default=DEFAULT_IMAGE_SUFFIX,
        help="Image filename suffix for sample_id mode.",
    )
    parser.add_argument(
        "--regions-col",
        default="",
        help="CSV column containing region target text, e.g. '13,1,2'. Auto-detect if omitted.",
    )
    parser.add_argument(
        "--preserve-input-region-order",
        action="store_true",
        help=(
            "In sample_id+region_id mode, use the first three distinct region rows for each sample "
            "exactly as they appear in the input CSV."
        ),
    )
    parser.add_argument("--user-prompt", default=DEFAULT_PROMPT, help="Query1 user prompt.")
    parser.add_argument(
        "--json-array-target",
        action="store_true",
        help="Write assistant target as JSON array string, e.g. [13, 1, 2].",
    )
    return parser.parse_args()


def _to_json_array_string(raw: str) -> str:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    vals = []
    for p in parts:
        vals.append(int(p) if p.isdigit() else p)
    return json.dumps(vals)


def _extract_target(row: Dict[str, str], regions_col: str) -> Optional[str]:
    if regions_col:
        val = str(row.get(regions_col, "")).strip()
        return val or None

    for col in ("regions", "topk3", "top3", "pred_top3", "region_topk3"):
        val = str(row.get(col, "")).strip()
        if val:
            return val

    multi_col_candidates = [
        ("topk1", "topk2", "topk3"),
        ("top1", "top2", "top3"),
        ("region1", "region2", "region3"),
        ("r1", "r2", "r3"),
    ]
    for cols in multi_col_candidates:
        vals: List[str] = []
        for c in cols:
            v = str(row.get(c, "")).strip()
            if not v:
                vals = []
                break
            vals.append(v)
        if vals:
            return ",".join(vals)

    return None


def _region_sort_key(x: str) -> Tuple[int, object]:
    if x.isdigit():
        return (0, int(x))
    return (1, x)


def _build_records_from_img_rows(
    reader: csv.DictReader,
    image_col: str,
    regions_col: str,
    sample_ids: set[str],
    image_suffix: str,
    user_prompt: str,
    json_array_target: bool,
) -> Tuple[List[Dict], List[Tuple[str, Dict]], int]:
    records: List[Dict] = []
    path_aware_rows: List[Tuple[str, Dict]] = []
    skipped = 0

    for idx, row in enumerate(reader):
        image_path = str(row.get(image_col, "")).strip()
        target_raw = _extract_target(row, regions_col)
        if not image_path or not target_raw:
            skipped += 1
            continue
        target = _to_json_array_string(target_raw) if json_array_target else target_raw
        source_sample_id = _infer_sample_id_from_image_path(image_path, image_suffix, sample_ids)
        rec = _build_record(
            idx=idx,
            image_path=image_path,
            target=target,
            user_prompt=user_prompt,
            source_sample_id=source_sample_id or "",
        )
        records.append(rec)
        path_aware_rows.append((image_path, rec))
    return records, path_aware_rows, skipped


def _infer_sample_id_from_image_path(
    image_path: str,
    image_suffix: str,
    sample_ids: set[str],
) -> Optional[str]:
    stem = Path(image_path).stem
    candidates: List[str] = [stem]

    suffix_stem = Path(image_suffix).stem if image_suffix else ""
    if suffix_stem and stem.endswith(suffix_stem):
        stripped = stem[: -len(suffix_stem)]
        if stripped.endswith("_"):
            stripped = stripped[:-1]
        if stripped:
            candidates.append(stripped)

    for candidate in list(candidates):
        stripped = re.sub(r"_(img(_edge(_number(_axes)?)?)?)$", "", candidate)
        if stripped and stripped != candidate:
            candidates.append(stripped)

    for candidate in list(candidates):
        for method in ("grid", "superpixel", "sam"):
            token = f"_{method}"
            pos = candidate.find(token)
            if pos > 0:
                prefix = candidate[:pos]
                if prefix:
                    candidates.append(prefix)

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in sample_ids:
            return candidate
    return None


def _build_records_from_sample_rows(
    reader: csv.DictReader,
    sample_id_col: str,
    region_id_col: str,
    order_scores: Dict[Tuple[str, str], float],
    image_dir: str,
    image_suffix: str,
    user_prompt: str,
    json_array_target: bool,
    preserve_input_region_order: bool,
) -> Tuple[List[Dict], List[Tuple[str, Dict]], int, Dict[str, List[str]]]:
    sample_to_regions: "OrderedDict[str, List[str]]" = OrderedDict()
    skipped = 0

    for row in reader:
        sample_id = str(row.get(sample_id_col, "")).strip()
        region_id = str(row.get(region_id_col, "")).strip()
        if not sample_id or not region_id:
            skipped += 1
            continue
        if sample_id not in sample_to_regions:
            sample_to_regions[sample_id] = []
        if region_id not in sample_to_regions[sample_id]:
            sample_to_regions[sample_id].append(region_id)

    records: List[Dict] = []
    path_aware_rows: List[Tuple[str, Dict]] = []
    expected_targets: Dict[str, List[str]] = {}
    idx = 0
    for sample_id, region_list in sample_to_regions.items():
        if len(region_list) < 3:
            skipped += 1
            continue
        if preserve_input_region_order:
            top3 = region_list[:3]
        else:
            ranked_regions = sorted(
                region_list,
                key=lambda rid: (-order_scores.get((sample_id, rid), float("-inf")), _region_sort_key(rid)),
            )
            top3 = ranked_regions[:3]
        expected_targets[sample_id] = list(top3)
        target_raw = ",".join(top3)
        target = _to_json_array_string(target_raw) if json_array_target else target_raw
        image_path = str(Path(image_dir) / f"{sample_id}{image_suffix}")
        rec = _build_record(
            idx=idx,
            image_path=image_path,
            target=target,
            user_prompt=user_prompt,
            source_sample_id=sample_id,
        )
        records.append(rec)
        path_aware_rows.append((image_path, rec))
        idx += 1

    return records, path_aware_rows, skipped, expected_targets


def _load_order_scores(
    order_csv: str,
    sample_id_col: str,
    region_id_col: str,
    score_col: str,
    method_col: str,
    method: str,
) -> Dict[Tuple[str, str], float]:
    if not order_csv:
        return {}

    order_path = Path(order_csv).expanduser()
    if not order_path.exists():
        print(f"warning: order-csv not found, fallback to id tie-break only: {order_path}")
        return {}

    out: Dict[Tuple[str, str], float] = {}
    with order_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        has_method_col = bool(method_col) and method_col in set(reader.fieldnames or [])
        for row in reader:
            if has_method_col and str(row.get(method_col, "")).strip() != method:
                continue
            sample_id = Path(str(row.get(sample_id_col, "")).strip()).stem
            region_id = str(row.get(region_id_col, "")).strip()
            score_raw = str(row.get(score_col, "")).strip()
            if not sample_id or not region_id or not score_raw:
                continue
            try:
                score = float(score_raw)
            except ValueError:
                continue
            key = (sample_id, region_id)
            old = out.get(key, float("-inf"))
            if score > old:
                out[key] = score

    print(f"loaded order scores: {len(out)} pairs from {order_path}")
    return out


def _parse_target_ids(target: str) -> List[str]:
    return [str(int(x)) for x in re.findall(r"\d+", str(target or ""))]


def _verify_records_against_order(
    records: List[Dict], order_scores: Dict[Tuple[str, str], float]
) -> None:
    errors: List[str] = []
    checked = 0
    for rec in records:
        source_sample_id = str(rec.get("source_sample_id", "")).strip()
        actual = _parse_target_ids(rec.get("gt_regions", ""))
        expected = sorted(
            actual,
            key=lambda rid: (-order_scores.get((source_sample_id, rid), float("-inf")), _region_sort_key(rid)),
        )
        if not source_sample_id:
            errors.append(f"missing source_sample_id for record {rec.get('sample_id')!r}")
        elif len(actual) != 3 or len(set(actual)) != 3 or any(not (1 <= int(x) <= 16) for x in actual):
            errors.append(f"{source_sample_id}: invalid Q1 target IDs {actual}")
        elif actual != expected:
            errors.append(f"{source_sample_id}: constructed={actual}, order_csv={expected}")
        checked += 1
        if len(errors) >= 20:
            break
    if errors:
        raise ValueError("Q1 target order verification failed:\n  " + "\n  ".join(errors))
    if checked == 0:
        raise ValueError("Q1 target order verification had no records to check.")
    print(f"verified target order against order CSV: {checked} records")


def _verify_records_against_input_targets(
    records: List[Dict], expected_targets: Dict[str, List[str]]
) -> None:
    errors: List[str] = []
    checked = 0
    for rec in records:
        source_sample_id = str(rec.get("source_sample_id", "")).strip()
        actual = _parse_target_ids(rec.get("gt_regions", ""))
        expected = expected_targets.get(source_sample_id)
        if expected is None:
            errors.append(f"{source_sample_id}: missing authoritative input target")
        elif actual != expected:
            errors.append(f"{source_sample_id}: constructed={actual}, input_csv={expected}")
        checked += 1
        if len(errors) >= 20:
            break
    if errors:
        raise ValueError("Q1 input-order verification failed:\n  " + "\n  ".join(errors))
    if checked == 0:
        raise ValueError("Q1 input-order verification had no records to check.")
    print(f"verified target IDs and order against input CSV: {checked} records")


def _verify_record_images(records: List[Dict]) -> None:
    missing: List[str] = []
    for rec in records:
        messages = rec.get("messages", [])
        image_path = ""
        if messages and isinstance(messages[0], dict):
            for part in messages[0].get("content", []):
                if isinstance(part, dict) and part.get("type") == "image":
                    image_path = str(part.get("image", "")).strip()
                    break
        if not image_path or not Path(image_path).expanduser().is_file():
            missing.append(image_path or f"<missing path for {rec.get('sample_id')}>")
            if len(missing) >= 20:
                break
    if missing:
        raise FileNotFoundError(f"Missing Q1 spectrogram images; examples={missing}")
    print(f"verified spectrogram images: {len(records)} files")


def _build_record(
    idx: int,
    image_path: str,
    target: str,
    user_prompt: str,
    source_sample_id: str,
) -> Dict:
    return {
        "sample_id": source_sample_id or f"stage1_q1_{idx:06d}",
        "source_sample_id": source_sample_id,
        # Keep factual targets outside messages so GRPO reward computation does
        # not depend on MS-Swift preserving the SFT assistant turn.
        "gt_regions": target,
        "solution": target,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_prompt},
                ],
            },
            # Keep content schema consistent for Arrow conversion in HF datasets.
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ],
    }


def _write_json(path: Path, data: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _verify_written_json(
    paths: List[Path],
    order_scores: Dict[Tuple[str, str], float],
    expected_targets: Optional[Dict[str, List[str]]] = None,
) -> None:
    persisted: List[Dict] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Constructed dataset is not a JSON list: {path}")
        persisted.extend(data)
    if expected_targets is not None:
        _verify_records_against_input_targets(persisted, expected_targets)
        label = "input-order"
    else:
        _verify_records_against_order(persisted, order_scores)
        label = "order-CSV"
    print(f"post-write {label} audit passed: {', '.join(str(p) for p in paths)}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input_csv).expanduser()
    order_scores = _load_order_scores(
        order_csv=args.order_csv,
        sample_id_col=args.order_sample_id_col,
        region_id_col=args.order_region_id_col,
        score_col=args.order_score_col,
        method_col=args.order_method_col,
        method=args.order_method,
    )
    if args.require_order_scores and not order_scores:
        raise ValueError("--require-order-scores was set, but no valid ordering scores were loaded.")
    sample_ids = {sample_id for sample_id, _ in order_scores.keys()}

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        use_img_mode = bool(args.image_col) and args.image_col in fieldnames
        if use_img_mode:
            if args.preserve_input_region_order:
                raise ValueError("--preserve-input-region-order requires sample_id+region_id input mode.")
            records, path_aware_rows, skipped = _build_records_from_img_rows(
                reader=reader,
                image_col=args.image_col,
                regions_col=args.regions_col,
                sample_ids=sample_ids,
                image_suffix=args.image_suffix,
                user_prompt=args.user_prompt,
                json_array_target=args.json_array_target,
            )
            expected_targets: Dict[str, List[str]] = {}
            print(f"mode: img_path (image_col='{args.image_col}')")
        else:
            records, path_aware_rows, skipped, expected_targets = _build_records_from_sample_rows(
                reader=reader,
                sample_id_col=args.sample_id_col,
                region_id_col=args.region_id_col,
                order_scores=order_scores,
                image_dir=args.image_dir,
                image_suffix=args.image_suffix,
                user_prompt=args.user_prompt,
                json_array_target=args.json_array_target,
                preserve_input_region_order=args.preserve_input_region_order,
            )
            print(
                "mode: sample_id+region_id "
                f"(sample_id_col='{args.sample_id_col}', region_id_col='{args.region_id_col}')"
            )

    if args.require_order_scores:
        missing = []
        for rec in records:
            sid = str(rec.get("source_sample_id", "")).strip()
            for rid in _parse_target_ids(rec.get("gt_regions", "")):
                if (sid, rid) not in order_scores:
                    missing.append((sid, rid))
                    if len(missing) >= 20:
                        break
            if len(missing) >= 20:
                break
        if missing:
            raise ValueError(f"Missing order scores for constructed Q1 targets; examples={missing}")
    if args.verify_order:
        if args.preserve_input_region_order:
            _verify_records_against_input_targets(records, expected_targets)
        else:
            _verify_records_against_order(records, order_scores)
    if args.require_images:
        _verify_record_images(records)

    split_mode = args.train_json is not None and args.val_json is not None
    if split_mode:
        if args.split_by_path:
            train_data: List[Dict] = []
            val_data: List[Dict] = []
            unmatched: List[Dict] = []
            for image_path, rec in path_aware_rows:
                if args.train_key in image_path:
                    train_data.append(rec)
                elif args.val_key in image_path:
                    val_data.append(rec)
                else:
                    unmatched.append(rec)
            if unmatched and not args.strict_path_split:
                train_data.extend(unmatched)

            print(f"path split: train_key='{args.train_key}', val_key='{args.val_key}'")
            print(f"unmatched: {len(unmatched)}")
            if args.strict_path_split and unmatched:
                print("dropped unmatched due to --strict-path-split")
        else:
            random.seed(args.seed)
            random.shuffle(records)
            n_val = int(len(records) * args.val_ratio)
            val_data = records[:n_val]
            train_data = records[n_val:]

        if not train_data or not val_data:
            raise ValueError(
                f"Protocol split produced an empty partition: train={len(train_data)}, val={len(val_data)}"
            )
        if args.expected_train_samples > 0 and len(train_data) != args.expected_train_samples:
            raise ValueError(
                f"Training partition count mismatch: got {len(train_data)}, "
                f"expected {args.expected_train_samples} from the PDF protocol."
            )
        if args.expected_val_samples > 0 and len(val_data) != args.expected_val_samples:
            raise ValueError(
                f"Development partition count mismatch: got {len(val_data)}, "
                f"expected {args.expected_val_samples} from the PDF protocol."
            )

        train_path = Path(args.train_json).expanduser()
        val_path = Path(args.val_json).expanduser()
        _write_json(train_path, train_data)
        _write_json(val_path, val_data)
        if args.verify_order:
            _verify_written_json(
                [train_path, val_path],
                order_scores,
                expected_targets if args.preserve_input_region_order else None,
            )
        print(f"saved train: {len(train_data)} -> {train_path}")
        print(f"saved val:   {len(val_data)} -> {val_path}")
        print(f"skipped rows: {skipped}")
        return

    if args.output_json is None:
        raise ValueError("Provide --output-json, or provide both --train-json and --val-json.")

    output_path = Path(args.output_json).expanduser()
    _write_json(output_path, records)
    if args.verify_order:
        _verify_written_json(
            [output_path],
            order_scores,
            expected_targets if args.preserve_input_region_order else None,
        )
    print(f"saved: {output_path} (n={len(records)})")
    print(f"skipped rows: {skipped}")


if __name__ == "__main__":
    main()
