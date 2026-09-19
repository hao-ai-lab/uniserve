"""Benchmarks Sol-H3 (NVlabs Sana sol-engine) on the fast_h3 protocol.

Replicates the canonical suite semantics from uniserve_eval/profiles.toml:
one warmup request (identical to row 0), then three measured serial requests,
seed 1000, durations 5 s/15 s at 124/362 frames, 1344x768 @ 24 fps with stereo
32 kHz audio, and the exact synthesized prompt texts dumped under
artifacts/benchmark/h3-sol-comparison/prompts/.

Per request two latencies are recorded: ``generate_s`` (text encoding + DiT
denoising + VAE decoding, Sol-H3's own timing scope) and ``save_s`` (MP4
encoding), so the sum approximates UniServe's client-side video_latency_ms
scope, which additionally carries HTTP transfer.

The paths below are the reference engine's own model and adapter, which it
loads in its own format; they are not UniServe checkpoints and are not part
of the supported set in docs/fast_h3/fast_h3.md.

Launch (4 GPUs):
    torchrun --standalone --nproc_per_node=4 scripts/bench_sol_h3.py \
        --compute-quant none \
        --output-root artifacts/benchmark/h3-sol-comparison/sol-h3-bf16
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

SOL_H3_ROOT = (
    Path(__file__).resolve().parents[1] / "refs/Sana/models/minimax_h3/Sol-H3"
)
sys.path.insert(0, str(SOL_H3_ROOT))

from h3_runtime import MiniMaxH3Inference  # noqa: E402

MODEL_PATH = "/workspace/models/sol-h3/MiniMax-H3"
ADAPTER_PATH = (
    "/workspace/models/FastH3-LoRA/dense-datafree/adapter_model.safetensors"
)
PROMPT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "artifacts/benchmark/h3-sol-comparison/prompts"
)

SEED = 1000
WARMUP_REQUESTS = 1
MEASURED_REQUESTS = 3

# The four canonical fast_h3 points: (name, duration seconds, prompt tokens).
POINTS = [
    ("minimax-h3-5s-1k", 5, 1000),
    ("minimax-h3-5s-10k", 5, 10000),
    ("minimax-h3-15s-1k", 15, 1000),
    ("minimax-h3-15s-10k", 15, 10000),
]

EXPECTED_FRAMES = {5: 124, 10: 243, 15: 362}


def validate_mp4(path: Path, seconds: int) -> dict:
    """Mirror the uniserve_eval media contract on the encoded MP4."""
    import av

    checks = {}
    with av.open(str(path)) as container:
        video_streams = [s for s in container.streams if s.type == "video"]
        audio_streams = [s for s in container.streams if s.type == "audio"]
        checks["one_video_stream"] = len(video_streams) == 1
        checks["one_audio_stream"] = len(audio_streams) == 1
        if video_streams:
            stream = video_streams[0]
            checks["codec_h264"] = stream.codec_context.name == "h264"
            checks["resolution_1344x768"] = (
                stream.codec_context.width == 1344
                and stream.codec_context.height == 768
            )
            fps = stream.average_rate
            checks["fps_24"] = fps.numerator == 24 * fps.denominator
            frames = sum(1 for _ in container.decode(video_streams[0]))
            checks["frame_count"] = frames
            checks["frames_match_duration"] = frames == EXPECTED_FRAMES[seconds]
        if audio_streams:
            stream = audio_streams[0]
            checks["audio_stereo_32k"] = (
                stream.codec_context.channels == 2
                and stream.codec_context.sample_rate == 32000
            )
    checks["valid"] = all(
        v
        for k, v in checks.items()
        if k != "frame_count" and isinstance(v, bool)
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compute-quant", choices=("none", "mxfp8"), default="none"
    )
    parser.add_argument(
        "--attention-backend",
        choices=("sol_bsa", "sol", "dense"),
        default="sol_bsa",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--points", nargs="*", default=[p[0] for p in POINTS])
    args = parser.parse_args()

    rank = int(__import__("os").environ.get("RANK", "0"))

    with MiniMaxH3Inference(
        MODEL_PATH,
        ADAPTER_PATH,
        attention_backend=args.attention_backend,
        task="t2v",
        compute_quant=args.compute_quant,
    ) as engine:
        for point_name, seconds, prompt_tokens in POINTS:
            if point_name not in args.points:
                continue
            prompt = (
                (PROMPT_ROOT / f"prompt-{prompt_tokens}.txt")
                .read_text(encoding="utf-8")
                .strip()
            )
            point_dir = args.output_root / point_name
            point_dir.mkdir(parents=True, exist_ok=True)
            sample_dir = point_dir / "samples"
            sample_dir.mkdir(exist_ok=True)

            for _ in range(WARMUP_REQUESTS):
                engine.generate(prompt, duration=seconds, seed=SEED)

            records = []
            for index in range(MEASURED_REQUESTS):
                started = time.perf_counter()
                result = engine.generate(prompt, duration=seconds, seed=SEED)
                if result is None:
                    continue  # nonzero ranks
                mp4_path = sample_dir / f"request-{index}.mp4"
                save_started = time.perf_counter()
                result.save(mp4_path)
                save_s = time.perf_counter() - save_started
                total_s = time.perf_counter() - started
                checks = validate_mp4(mp4_path, seconds)
                records.append(
                    {
                        "request_index": index,
                        "generate_s": result.elapsed_s,
                        "save_s": save_s,
                        "total_s": total_s,
                        "validation": checks,
                    }
                )
                print(
                    f"[{point_name}] request {index}: "
                    f"generate={result.elapsed_s:.3f}s save={save_s:.3f}s "
                    f"total={total_s:.3f}s valid={checks['valid']}",
                    flush=True,
                )

            if rank == 0:
                generate = [r["generate_s"] for r in records]
                total = [r["total_s"] for r in records]
                summary = {
                    "point": point_name,
                    "engine": "sol-h3",
                    "engine_revision": (
                        "cee25847afdd34bc656abcca126262200b088dc8"
                    ),
                    "compute_quant": args.compute_quant,
                    "attention_backend": args.attention_backend,
                    "gpus": engine.world_size,
                    "seconds": seconds,
                    "prompt_tokens": prompt_tokens,
                    "seed": SEED,
                    "warmup_requests": WARMUP_REQUESTS,
                    "measured_requests": MEASURED_REQUESTS,
                    "requests": records,
                    "generate_s_mean": statistics.mean(generate),
                    "generate_s_median": statistics.median(generate),
                    "total_s_mean": statistics.mean(total),
                    "total_s_median": statistics.median(total),
                    "all_valid": all(r["validation"]["valid"] for r in records),
                }
                with open(
                    point_dir / "summary.json", "w", encoding="utf-8"
                ) as fh:
                    json.dump(summary, fh, indent=2)
                print(
                    f"[{point_name}] wrote {point_dir / 'summary.json'}",
                    flush=True,
                )

            if engine.world_size > 1:
                torch.distributed.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
