#!/usr/bin/env python3
"""Draw a reproducible H3 prompt set for packed-checkpoint media review."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from uniserve_eval.artifacts import ArtifactWriter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "artifacts/h3_t2va_prompts_10k.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts/h3-nvfp4-review"
DRAW_SEED = "h3-nvfp4-review-2026-09-16-v1"
INVALID_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Repair only invalid JSON escape boundaries; prompt text is
                # otherwise preserved byte-for-byte after JSON decoding.
                record = json.loads(INVALID_ESCAPE.sub(r"\\\\", line))
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} is not an object")
            record["_source_line"] = line_number
            records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--frames", type=int, default=124)
    args = parser.parse_args()

    records = _records(args.source)
    eligible = [
        record
        for record in records
        if record.get("runtime_config", {}).get("num_frames") == args.frames
    ]
    ranked = sorted(
        eligible,
        key=lambda record: hashlib.sha256(
            f"{DRAW_SEED}:{record['id']}".encode()
        ).digest(),
    )
    selected = ranked[: args.count]
    if len(selected) != args.count:
        raise ValueError(
            f"requested {args.count} prompts but found {len(eligible)} eligible"
        )

    rows = []
    for record in selected:
        sampling = record["sampling"]
        rows.append(
            {
                "id": record["id"],
                "prompt": record["prompt_compiled"],
                "seed": int(sampling["seed"]),
                "seconds": 5.0,
                "source_line": record["_source_line"],
                "source_runtime_config": record["runtime_config"],
                "source_dimensions": sampling.get("dimensions", {}),
            }
        )

    writer = ArtifactWriter(args.output)
    writer.write_jsonl("selected-prompts.jsonl", rows)
    writer.write_json(
        "selection.json",
        {
            "source": str(args.source),
            "source_sha256": _sha256(args.source),
            "source_records": len(records),
            "eligible_records": len(eligible),
            "draw_seed": DRAW_SEED,
            "draw_method": "ascending sha256(draw_seed + ':' + record_id)",
            "required_source_frames": args.frames,
            "review_runtime": {
                "seconds": 5.0,
                "width": 1344,
                "height": 768,
                "fps": 24,
                "generate_audio": True,
            },
            "samples": [
                {
                    "id": row["id"],
                    "source_line": row["source_line"],
                    "seed": row["seed"],
                    "prompt_sha256": hashlib.sha256(
                        row["prompt"].encode()
                    ).hexdigest(),
                }
                for row in rows
            ],
            "interpretation": (
                "These prompts overlap the PTQ calibration source and are "
                "for generation validation and human review, not independent "
                "held-out quality acceptance."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
