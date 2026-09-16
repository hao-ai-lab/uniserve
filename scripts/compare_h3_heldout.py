#!/usr/bin/env python3
"""Measure and blind the independent H3 held-out video comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import lpips

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_h3_media import compare_audio, compare_video  # noqa: E402

POINTS = ("h3-heldout-5s", "h3-heldout-10s", "h3-heldout-15s")
CRITERIA = (
    "pairwise_preference",
    "prompt_adherence",
    "entity_scene_consistency",
    "temporal_consistency",
    "motion_camera_completion",
    "speech_lyrics_intelligibility",
    "audio_video_semantic_sync",
    "critical_failure",
)


def _candidate(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("candidate must be LABEL=ROOT")
    return label, Path(path)


def _records(root: Path) -> dict[str, dict]:
    result = {}
    for point in POINTS:
        directory = root / point
        for line in (directory / "requests.jsonl").read_text().splitlines():
            record = json.loads(line)
            if not record.get("success") or record.get("video_output") is None:
                raise RuntimeError(
                    f"{root}:{record['request_id']} is not a valid video"
                )
            identifier = record["request_id"]
            if identifier in result:
                raise RuntimeError(
                    f"duplicate held-out request id {identifier}"
                )
            result[identifier] = {
                "point": point,
                "record": record,
                "path": directory
                / "samples"
                / record["video_output"]["sample_filename"],
            }
    return result


def _prompt_metadata(path: Path) -> dict[str, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        result[record["id"]] = {
            "prompt": record["prompt_compiled"],
            "tags": record["tags"],
            "runtime_config": record["runtime_config"],
            "seed": record["sampling"]["seed"],
        }
    return result


def _means(rows: list[dict]) -> dict[str, float]:
    groups = defaultdict(list)
    for row in rows:
        for modality in ("video", "audio"):
            for name, value in row[modality].items():
                if isinstance(value, (int, float)) and not isinstance(
                    value, bool
                ):
                    if value is not None and math.isfinite(value):
                        groups[f"{modality}.{name}"].append(float(value))
    return {
        name: statistics.mean(values) for name, values in sorted(groups.items())
    }


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    os.link(source, destination)


def _blind(
    root: Path,
    prompts: dict[str, dict],
    sources: dict[str, dict[str, dict]],
    comparisons: tuple[tuple[str, str], ...],
) -> None:
    review = root / "review"
    key = []
    rows = []
    for comparison, (baseline, candidate) in enumerate(comparisons, 1):
        for identifier in sorted(prompts):
            digest = hashlib.sha256(
                f"h3-modelopt-blind-v1:{comparison}:{identifier}".encode()
            ).digest()
            left, right = (
                (baseline, candidate)
                if digest[0] & 1
                else (candidate, baseline)
            )
            pair_id = f"comparison-{comparison:02d}-{identifier}"
            _link(
                sources[left][identifier]["path"], review / pair_id / "left.mp4"
            )
            _link(
                sources[right][identifier]["path"],
                review / pair_id / "right.mp4",
            )
            metadata = prompts[identifier]
            (review / pair_id / "prompt.txt").write_text(
                metadata["prompt"] + "\n", encoding="utf-8"
            )
            key.append(
                {
                    "pair_id": pair_id,
                    "baseline": baseline,
                    "candidate": candidate,
                    "left": left,
                    "right": right,
                }
            )
            rows.append(
                {
                    "rater_id": "",
                    "pair_id": pair_id,
                    **dict.fromkeys(CRITERIA, ""),
                    "notes": "",
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    (root / "blinding-key.json").write_text(
        json.dumps(key, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (review / "ratings.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (review / "README.md").write_text(
        "# H3 blinded held-out review\n\n"
        "Each pair has the same prompt, seed, duration and production "
        "geometry. "
        "Use at least three independent raters. Record pairwise preference as "
        "left, tie, or right; score the six quality dimensions from 1 to 5; "
        "and mark critical_failure as 0 or 1. Raters must not receive the "
        "sibling blinding-key.json.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", type=_candidate, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blind-root", type=Path)
    args = parser.parse_args()

    prompts = _prompt_metadata(args.prompts)
    sources = {args.reference_label: _records(args.reference_root)}
    sources.update({label: _records(path) for label, path in args.candidate})
    expected = set(prompts)
    for label, records in sources.items():
        if set(records) != expected:
            missing = sorted(expected - set(records))
            extra = sorted(set(records) - expected)
            raise RuntimeError(
                f"{label} held-out ids differ: missing={missing}, extra={extra}"
            )

    perceptual = lpips.LPIPS(net="alex").eval().to("cuda:0")
    result = {
        "protocol": {
            "pairing": (
                "same independent prompt, seed, duration, and production "
                "geometry"
            ),
            "video": (
                "all decoded RGB24 frames: PSNR, NRMSE, SSIM, AlexNet LPIPS"
            ),
            "audio": (
                "decoded stereo 32-kHz PCM: SNR, correlation, NRMSE, and "
                "log-mel error"
            ),
            "psnr_role": (
                "descriptive same-seed trajectory drift only; never a "
                "pass/fail gate"
            ),
        },
        "reference": args.reference_label,
        "comparisons": {},
    }
    for label, _ in args.candidate:
        rows = []
        for identifier in sorted(prompts):
            reference = sources[args.reference_label][identifier]
            candidate = sources[label][identifier]
            rows.append(
                {
                    "id": identifier,
                    "tags": prompts[identifier]["tags"],
                    "seed": prompts[identifier]["seed"],
                    "point": reference["point"],
                    "reference_sample": str(reference["path"]),
                    "candidate_sample": str(candidate["path"]),
                    "reference_latency_ms": reference["record"]["e2e_ms"],
                    "candidate_latency_ms": candidate["record"]["e2e_ms"],
                    "video": compare_video(
                        reference["path"], candidate["path"], perceptual
                    ),
                    "audio": compare_audio(
                        reference["path"], candidate["path"]
                    ),
                }
            )
        result["comparisons"][label] = {"prompts": rows, "mean": _means(rows)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.blind_root is not None:
        if "candidate-a" not in sources or "candidate-b" not in sources:
            raise ValueError(
                "blinding requires candidate-a and candidate-b labels"
            )
        _blind(
            args.blind_root,
            prompts,
            sources,
            (
                (args.reference_label, "candidate-a"),
                ("candidate-a", "candidate-b"),
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
