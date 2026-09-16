#!/usr/bin/env python3
"""Pair quality and packed-NVFP4 H3 review videos and measure drift."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import lpips

from uniserve_eval.artifacts import ArtifactWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_h3_media import compare_audio, compare_video  # noqa: E402

PAIRS = {
    "four-step": ("four-step-quality", "four-step-nvfp4"),
    "eight-step": ("eight-step-quality", "eight-step-nvfp4"),
}


def _prompts(path: Path) -> dict[str, dict]:
    return {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _records(path: Path) -> dict[str, dict]:
    records = {}
    for line in (path / "requests.jsonl").read_text().splitlines():
        record = json.loads(line)
        output = record.get("video_output")
        if not record.get("success") or not isinstance(output, dict):
            raise RuntimeError(f"invalid generated video in {path}")
        records[record["request_id"]] = {
            "record": record,
            "path": path / "samples" / output["sample_filename"],
        }
    return records


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not os.path.samefile(source, destination):
            raise FileExistsError(destination)
        return
    os.link(source, destination)


def _means(rows: list[dict]) -> dict[str, float]:
    values = defaultdict(list)
    for row in rows:
        for modality in ("video", "audio"):
            for name, value in row[modality].items():
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and value is not None
                    and math.isfinite(value)
                ):
                    values[f"{modality}.{name}"].append(float(value))
    return {
        name: statistics.mean(items) for name, items in sorted(values.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=ROOT / "artifacts/h3-nvfp4-review/selected-prompts.jsonl",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "artifacts/h3-nvfp4-review/results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/h3-nvfp4-review/comparison",
    )
    args = parser.parse_args()

    prompts = _prompts(args.prompts)
    perceptual = lpips.LPIPS(net="alex").eval().to("cuda:0")
    report = {
        "protocol": {
            "pairing": "same prompt, source seed, 5 seconds, and H3 geometry",
            "video": (
                "all decoded RGB24 frames: PSNR, NRMSE, SSIM, AlexNet LPIPS"
            ),
            "audio": (
                "decoded stereo 32-kHz PCM: SNR, correlation, NRMSE, "
                "log-mel error"
            ),
            "interpretation": (
                "PSNR and related metrics describe same-seed trajectory drift; "
                "the calibration-source sample is not an independent quality "
                "gate."
            ),
        },
        "models": {},
    }
    index = ["# H3 quality versus packed NVFP4 review", ""]
    for model, (quality_name, nvfp4_name) in PAIRS.items():
        quality = _records(args.results / quality_name)
        nvfp4 = _records(args.results / nvfp4_name)
        if set(quality) != set(prompts) or set(nvfp4) != set(prompts):
            raise RuntimeError(
                f"{model} generated request ids do not match prompts"
            )

        rows = []
        review_links = []
        for identifier, prompt in prompts.items():
            quality_sample = quality[identifier]
            nvfp4_sample = nvfp4[identifier]
            directory = args.output / "review" / model / identifier
            _link(quality_sample["path"], directory / "quality.mp4")
            _link(nvfp4_sample["path"], directory / "nvfp4.mp4")
            (directory / "prompt.txt").write_text(
                prompt["prompt"] + "\n", encoding="utf-8"
            )
            rows.append(
                {
                    "id": identifier,
                    "seed": prompt["seed"],
                    "quality_sample": str(quality_sample["path"]),
                    "nvfp4_sample": str(nvfp4_sample["path"]),
                    "quality_latency_ms": quality_sample["record"]["e2e_ms"],
                    "nvfp4_latency_ms": nvfp4_sample["record"]["e2e_ms"],
                    "video": compare_video(
                        quality_sample["path"],
                        nvfp4_sample["path"],
                        perceptual,
                    ),
                    "audio": compare_audio(
                        quality_sample["path"], nvfp4_sample["path"]
                    ),
                }
            )
            relative = Path("review") / model / identifier
            review_links.append(
                f"- `{identifier}`: [quality]({relative / 'quality.mp4'}) | "
                f"[NVFP4]({relative / 'nvfp4.mp4'}) | "
                f"[prompt]({relative / 'prompt.txt'})"
            )
        mean = _means(rows)
        index.extend(
            (
                f"## {model}",
                "",
                "| Measurement | Mean |",
                "| --- | ---: |",
                f"| RGB PSNR | {mean['video.psnr_db']:.2f} dB |",
                f"| RGB SSIM | {mean['video.ssim_mean']:.4f} |",
                f"| AlexNet LPIPS | {mean['video.lpips_alex_mean']:.4f} |",
                "| Audio log-mel cosine | "
                f"{mean['audio.log_mel_cosine']:.4f} |",
                "",
                *review_links,
                "",
            )
        )
        report["models"][model] = {"samples": rows, "mean": mean}

    writer = ArtifactWriter(args.output)
    writer.write_json("metrics.json", report)
    index.extend(
        (
            "## Interpretation",
            "",
            (
                "These paired metrics measure same-seed trajectory drift, not "
                "absolute perceptual quality. The prompts were sampled from "
                "the "
                "calibration-source dataset, so this review proves executable "
                "packed checkpoints and supplies human-review material; it is "
                "not an independent held-out acceptance test."
            ),
            "",
        )
    )
    (args.output / "README.md").write_text("\n".join(index), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
