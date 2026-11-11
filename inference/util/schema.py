#!/usr/bin/env python3
"""GRIDEX inference output parsing and validation.

The schemas in this module mirror the contracts used by the GRIDEX training
builders and reward functions. Q1 selects three ordered grid IDs. Q2 describes
those IDs positionally; it does not emit the IDs again.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence


TIME_LABELS = {"speech", "non-speech"}
FREQ_LABELS = {"low", "mid", "high"}
PHON_LABELS = {"consonant", "vowel", "none"}


def parse_q1_region_ids(raw_response: str) -> List[int]:
    """Parse exactly one JSON array containing exactly three integer IDs."""
    text = str(raw_response or "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Q1 response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError("Q1 response must be a JSON array.")
    if len(parsed) != 3:
        raise ValueError(f"Q1 must contain exactly three region IDs, got: {parsed}")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in parsed):
        raise ValueError(f"Q1 region IDs must be JSON integers, got: {parsed}")
    ids = list(parsed)
    if len(set(ids)) != 3:
        raise ValueError(f"Q1 region IDs must be distinct, got: {ids}")
    if any(region_id < 1 or region_id > 16 for region_id in ids):
        raise ValueError(f"Q1 region IDs must be between 1 and 16, got: {ids}")
    return ids


def format_q1_region_ids(raw_response: str) -> str:
    return "[" + ", ".join(str(region_id) for region_id in parse_q1_region_ids(raw_response)) + "]"


def _split_top_level_tuples(text: str) -> List[str]:
    tuples: List[str] = []
    start: Optional[int] = None
    depth = 0
    in_quote = False
    escaped = False
    for index, char in enumerate(str(text or "")):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == "(":
            if depth == 0:
                start = index
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                tuples.append(str(text)[start : index + 1].strip())
                start = None
    return tuples


def _normalize_time(value: str) -> Optional[str]:
    normalized = " ".join(str(value or "").strip().lower().split())
    return normalized if normalized in TIME_LABELS else None


def _normalize_freq(value: str) -> Optional[str]:
    normalized = " ".join(str(value or "").strip().lower().split())
    return normalized if normalized in FREQ_LABELS else None


def _normalize_phon(value: str) -> Optional[str]:
    normalized = " ".join(str(value or "").strip().lower().split())
    return normalized if normalized in PHON_LABELS else None


def _strip_wrapped_quotes(value: str) -> tuple[str, bool]:
    text = str(value or "").strip()
    quoted = len(text) >= 2 and text[0] == '"' and text[-1] == '"'
    return (text[1:-1] if quoted else text), quoted


def parse_q2_position_aligned(raw_response: str, region_ids: Sequence[int]) -> List[Dict[str, object]]:
    """Parse the three position-aligned Q2 tuples and bind them to Q1 IDs."""

    ids = [int(region_id) for region_id in region_ids]
    if len(ids) != 3 or len(set(ids)) != 3 or any(region_id < 1 or region_id > 16 for region_id in ids):
        raise ValueError(f"Expected three distinct Q1 region IDs in [1, 16], got: {ids}")

    response_text = str(raw_response or "").strip()
    raw_tuples = _split_top_level_tuples(response_text)
    if len(raw_tuples) != 3:
        raise ValueError(f"Q2 must contain exactly three tuples, got: {len(raw_tuples)}")

    cursor = 0
    for tuple_index, raw_tuple in enumerate(raw_tuples):
        start = response_text.find(raw_tuple, cursor)
        separator = response_text[cursor:start]
        if tuple_index == 0:
            if separator.strip():
                raise ValueError("Q2 contains text before the first tuple.")
        elif not re.fullmatch(r"\s*;\s*", separator):
            raise ValueError("Q2 tuples must be separated by exactly one semicolon.")
        cursor = start + len(raw_tuple)
    if response_text[cursor:].strip():
        raise ValueError("Q2 contains text after the third tuple.")

    regions: List[Dict[str, object]] = []
    for slot, (region_id, raw_tuple) in enumerate(zip(ids, raw_tuples), start=1):
        body = raw_tuple[1:-1].strip()
        patterns = {
            "T": rf"\bT{slot}\s*=\s*([^,]+?)\s*(?:,|$)",
            "F": rf"\bF{slot}\s*=\s*([^,]+?)\s*(?:,|$)",
            "P": rf"\bP{slot}\s*=\s*([^,]+?)\s*(?:,|$)",
            "En": rf"\bEn{slot}\s*=\s*(.+)\s*$",
        }
        matches = {
            key: re.search(pattern, body, flags=re.IGNORECASE | (re.DOTALL if key == "En" else 0))
            for key, pattern in patterns.items()
        }
        if not all(matches.values()):
            raise ValueError(f"Q2 tuple {slot} does not follow the indexed GRIDEX schema: {raw_tuple}")

        time_label = _normalize_time(matches["T"].group(1))
        frequency_band = _normalize_freq(matches["F"].group(1))
        phonetic_category = _normalize_phon(matches["P"].group(1))
        explanation, explanation_quoted = _strip_wrapped_quotes(matches["En"].group(1))
        if time_label is None or frequency_band is None or phonetic_category is None:
            raise ValueError(f"Q2 tuple {slot} contains an unsupported categorical label: {raw_tuple}")
        if not explanation or not explanation_quoted:
            raise ValueError(f"Q2 tuple {slot} must contain a non-empty quoted En{slot}: {raw_tuple}")

        regions.append(
            {
                "rank": slot,
                "region_id": region_id,
                "T": time_label,
                "F": frequency_band,
                "P": phonetic_category,
                "En": explanation,
            }
        )
    return regions
