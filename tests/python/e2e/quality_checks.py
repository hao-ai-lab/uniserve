"""Content-quality checks for the GPU e2e test (NOT part of the perf harness).

These heuristics previously lived in ``benchmarks/serving/uniserve_bench/quality.py``.
The serving benchmark is performance-only, so they were moved here next to the
one test that uses them (``test_sensenova_gpu_native``): generate first during the
perf run, judge content separately.
"""
from __future__ import annotations

import base64
import io
import re
from typing import Any, cast

from PIL import Image, ImageStat

LOCATION_PATTERNS = {
    "sonoma": re.compile(r"\bsonoma\b", re.I),
    "sequoia": re.compile(r"\bsequoia\b", re.I),
    "tahoe": re.compile(r"\btahoe\b|\blake tahoe\b", re.I),
    "golden_gate": re.compile(r"\bgolden gate\b|\bgolden gate bridge\b", re.I),
}

HIDDEN_REASONING_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in (
        r"\bthinking process\b",
        r"\banalyze the request\b",
        r"\bdrafting content\b",
        r"\blet'?s think step by step\b",
        r"\bchain[- ]of[- ]thought\b",
        r"\bthe user'?s question\b",
        r"\bthis request involves\b",
        r"\bmy approach\b",
        r"\bthe overall planning\b",
        r"\bimage generation prompt\b",
        r"\banalyze the request\b",
        r"\binitial observation\b",
        r"\bscene analysis\b",
        r"\boverlay analysis\b",
        r"\btextual content\b",
    )
]


def text_quality(text: str, required_locations: bool = False) -> tuple[bool, dict[str, Any]]:
    stripped = text.strip()
    checks = {
        "non_empty": len(stripped) >= 80,
        "has_sentence_punctuation": any(mark in stripped for mark in ".!?"),
        "replacement_ratio": (stripped.count("\ufffd") / max(1, len(stripped))) < 0.01,
        "no_hidden_reasoning_leak": not any(
            pattern.search(stripped) for pattern in HIDDEN_REASONING_PATTERNS
        ),
        "no_excessive_repetition": not has_excessive_repetition(stripped),
    }
    if required_locations:
        checks["mentions_locations"] = all(
            pattern.search(stripped) for pattern in LOCATION_PATTERNS.values()
        )
    return all(checks.values()), checks


def has_excessive_repetition(text: str) -> bool:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    if len(words) < 16:
        return False
    for ngram_len, repeat_limit in ((1, 16), (2, 8), (3, 5), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4)):
        i = 0
        while i + ngram_len * repeat_limit <= len(words):
            ngram = words[i : i + ngram_len]
            repeats = 1
            j = i + ngram_len
            while j + ngram_len <= len(words) and words[j : j + ngram_len] == ngram:
                repeats += 1
                if repeats >= repeat_limit:
                    return True
                j += ngram_len
            i += max(1, ngram_len if repeats > 1 else 1)
    return False


def png_quality(pixels_png_b64: str, width: int, height: int) -> tuple[bool, dict[str, Any]]:
    raw = base64.b64decode(pixels_png_b64)
    image = Image.open(io.BytesIO(raw))
    image.load()
    rgb = image.convert("RGB")
    stat = ImageStat.Stat(rgb.resize((min(width, 256), min(height, 144))))
    extrema = cast(tuple[tuple[int, int], tuple[int, int], tuple[int, int]], rgb.getextrema())
    variances = stat.var if isinstance(stat.var, list) else [stat.var]
    variance = sum(variances) / len(variances)
    checks = {
        "decodable": True,
        "width": image.width,
        "height": image.height,
        "matches_dimensions": image.width == width and image.height == height,
        "not_preview_sized": image.width >= 1024 and image.height >= 768,
        "non_blank_variance": variance > 1.0,
        "has_range": any((hi - lo) > 8 for lo, hi in extrema),
    }
    return bool(
        checks["matches_dimensions"]
        and checks["not_preview_sized"]
        and checks["non_blank_variance"]
        and checks["has_range"]
    ), checks
