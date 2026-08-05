#!/usr/bin/env python3
"""Validate runtime document links and the decode-runtime benchmark suite."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uniserve_eval.profiles import load_config  # noqa: E402

DOCUMENTS = (
    Path("specs/decode-runtime.md"),
    Path("specs/serving-surface.md"),
    Path("specs/decode-runtime-construction.md"),
    Path("specs/generation-runtime-qualification.md"),
    Path("specs/benchmark-validation.md"),
    Path("docs/benchmark-protocol.md"),
    Path("docs/serving-evaluation.md"),
)
PROFILE = Path("uniserve_eval/profiles.toml")
QUALIFICATION_POINTS = (
    "qwen-uniserve-sharegpt-r16",
    "sensenova-uniserve-i2t-c32",
    "sensenova-uniserve-t2i-c32",
    "sensenova-uniserve-interleave-c4",
)
MARKDOWN_LINK = re.compile(
    r"(?<!!)\[[^\]]+\]\((?P<target><[^>]+>|[^)\s]+)(?:\s+[^)]*)?\)"
)


class ContractError(ValueError):
    pass


def validate() -> dict[str, Any]:
    for document in DOCUMENTS:
        _validate_links(document)
    config = load_config(ROOT / PROFILE)
    suite = config.suites.get("decode-runtime")
    if suite is None or suite.points != QUALIFICATION_POINTS:
        actual = None if suite is None else suite.points
        raise ContractError(f"decode-runtime suite has unexpected points: {actual!r}")
    if suite.max_regression != 0.05:
        raise ContractError("decode-runtime suite must declare max_regression = 0.05")
    document_points = _qualification_document_points()
    if document_points != QUALIFICATION_POINTS:
        raise ContractError(
            f"qualification document has unexpected points: {document_points!r}"
        )
    return {
        "valid": True,
        "profile": str(PROFILE),
        "qualification_points": list(QUALIFICATION_POINTS),
        "max_regression": suite.max_regression,
    }


def _validate_links(path: Path) -> None:
    source = ROOT / path
    text = source.read_text(encoding="utf-8")
    for match in MARKDOWN_LINK.finditer(text):
        target = match.group("target").strip("<>")
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        relative = target.split("#", 1)[0]
        if relative and not (source.parent / relative).exists():
            line = text.count("\n", 0, match.start()) + 1
            raise ContractError(f"{path}:{line} references missing path {target!r}")


def _qualification_document_points() -> tuple[str, ...]:
    text = (ROOT / "specs/generation-runtime-qualification.md").read_text(encoding="utf-8")
    section = text.split("## Performance points", 1)[-1].split("## Work acceptance", 1)[0]
    return tuple(re.findall(r"^\| `([^`]+)` \|", section, flags=re.MULTILINE))


def main() -> int:
    try:
        report = validate()
    except (ContractError, OSError, TypeError, ValueError) as error:
        print(f"runtime contract invalid: {error}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
