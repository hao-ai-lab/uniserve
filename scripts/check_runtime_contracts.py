#!/usr/bin/env python3
"""Validate the canonical runtime documents and executable benchmark matrix."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uniserve_eval.profiles import benchmark_matrix_definition_contract, load_config

DOCUMENTS = (
    Path("specs/decode-runtime.md"),
    Path("specs/serving-surface.md"),
    Path("specs/decode-runtime-construction.md"),
    Path("specs/generation-runtime-qualification.md"),
    Path("docs/benchmark-protocol.md"),
    Path("docs/serving-evaluation.md"),
)
PROFILE = Path("uniserve_eval/profiles.json")
QUALIFICATION_PERFORMANCE_POINTS = (
    "sensenova-uniserve-t2i-c32",
    "sensenova-uniserve-i2t-c32",
    "sensenova-uniserve-interleave-c4",
    "sensenova-uniserve-stochastic-interleave-c4",
)
MARKDOWN_LINK = re.compile(
    r"(?<!!)\[[^\]]+\]\((?P<target><[^>]+>|[^)\s]+)(?:\s+[^)]*)?\)"
)


class ContractError(ValueError):
    """The canonical source contract is invalid."""


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_links(path: Path) -> None:
    source = ROOT / path
    text = source.read_text(encoding="utf-8")
    for match in MARKDOWN_LINK.finditer(text):
        target = match.group("target").strip("<>")
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target_path = target.split("#", 1)[0]
        if not target_path:
            continue
        resolved = source.parent / target_path
        if not resolved.exists():
            line = text.count("\n", 0, match.start()) + 1
            raise ContractError(f"{path}:{line} references missing path {target!r}")


def _expanded_matrix(config: dict[str, Any]) -> list[dict[str, str]]:
    benchmark = config["benchmarks"]["main"]
    expanded: list[dict[str, str]] = []
    names: set[str] = set()
    for group_name, group in benchmark["groups"].items():
        for point_name in group["points"]:
            point = benchmark["points"][point_name]
            load_case_set = point["load_case_set"]
            for load_case in benchmark["load_cases"][load_case_set]:
                name = point["name"].format(load_id=load_case["id"])
                if name in names:
                    raise ContractError(f"benchmark point name is duplicated: {name}")
                names.add(name)
                benchmark_matrix_definition_contract(
                    config,
                    "main",
                    group_name=group_name,
                    point_name=point_name,
                    load_case_set=load_case_set,
                    load_case_id=load_case["id"],
                )
                expanded.append(
                    {
                        "name": name,
                        "group": group_name,
                        "point": point_name,
                        "load_case": load_case["id"],
                    }
                )
    return expanded


def _qualification_performance_points() -> tuple[str, ...]:
    text = (ROOT / "specs/generation-runtime-qualification.md").read_text(encoding="utf-8")
    try:
        table = text.split("The complete performance protection set is:", 1)[1].split(
            "The stochastic architecture point is", 1
        )[0]
    except IndexError as error:
        raise ContractError("qualification performance table is missing") from error
    points = tuple(re.findall(r"^\| [^|]+ \| `([^`]+)` \|", table, flags=re.MULTILINE))
    if points != QUALIFICATION_PERFORMANCE_POINTS:
        raise ContractError(
            "qualification performance table and executable contract disagree: "
            f"{points!r}"
        )
    return points


def validate() -> dict[str, Any]:
    for document in DOCUMENTS:
        _validate_links(document)
    config = load_config(ROOT / PROFILE)
    expanded = _expanded_matrix(config)
    qualification_points = _qualification_performance_points()
    if len(expanded) != 46:
        raise ContractError(f"main benchmark expands to {len(expanded)} points, expected 46")
    expanded_names = {point["name"] for point in expanded}
    missing_qualification_points = set(qualification_points) - expanded_names
    if missing_qualification_points:
        raise ContractError(
            "qualification performance points are absent from the executable matrix: "
            f"{sorted(missing_qualification_points)}"
        )
    payload = {
        "schema_version": 1,
        "valid": True,
        "expanded_point_count": len(expanded),
        "qualification_performance_points": list(qualification_points),
        "documents": {str(path): _digest(ROOT / path) for path in DOCUMENTS},
        "profile_sha256": _digest(ROOT / PROFILE),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "fingerprint": hashlib.sha256(canonical).hexdigest()}


def main() -> int:
    try:
        report = validate()
    except (ContractError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"runtime contract invalid: {error}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
