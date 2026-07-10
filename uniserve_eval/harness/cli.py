"""Command-line entry point for the UniServe serving performance benchmark.

One invocation runs one operating point per (request-rate, max-concurrency) combo.
Pass comma-separated ``--request-rates`` / ``--max-concurrencies`` to sweep; each
point is written to its own subdirectory and a ``combined_summary.json`` indexes
them.

Examples:

  # LLM serving (ShareGPT), sglang-comparable
  python -m uniserve_eval.harness.cli \
      --base-url http://127.0.0.1:18080 --task text --model Qwen3-32B \
      --num-prompts 1000 --request-rate inf --output-dir results/text

  # text-to-image (MJHQ-30K) latency/throughput at a few concurrencies
  python -m uniserve_eval.harness.cli \
      --base-url http://127.0.0.1:18080 --task t2i --model SenseNova-U1 \
      --num-prompts 200 --max-concurrencies 1,2,4 --output-dir results/t2i
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .runner import BenchmarkRunner
from .spec import DEFAULT_DATASETS, BenchmarkSpec, TaskName


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uniserve-eval-harness", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--task", required=True, choices=[task.value for task in TaskName])
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--endpoint", help="override the per-task default endpoint")
    parser.add_argument("--dataset", help="dataset name (default per task); 'trace' reads --dataset-path JSONL")
    parser.add_argument("--dataset-path", help="local dataset file/dir override (else HF auto-download)")
    parser.add_argument("--tokenizer", help="tokenizer path/name (ShareGPT shaping; defaults to --model)")

    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--request-rate", default="inf", help="req/s or 'inf' (single point)")
    parser.add_argument("--request-rates", help="comma-separated req/s sweep (overrides --request-rate)")
    parser.add_argument("--max-concurrency", type=int, help="in-flight cap (single point)")
    parser.add_argument("--max-concurrencies", help="comma-separated concurrency sweep")
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-tokens", type=int, help="output token cap")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--disable-ignore-eos", action="store_true", help="respect EOS (default ignores EOS)")

    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--image-guidance-scale", type=float)
    parser.add_argument("--cfg-norm")
    parser.add_argument("--cfg-interval", help="comma-separated inclusive CFG interval")
    parser.add_argument("--timestep-shift", type=float)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="default-task image cap; omit to leave image count uncapped",
    )
    parser.add_argument(
        "--wire",
        default=None,
        help=(
            "request/response shape for the task (default: task's first wire): "
            "openai_chat | openai_chat_json | images_generations"
        ),
    )
    parser.add_argument("--i2t-question", default="Describe this image in detail.")
    parser.add_argument(
        "--no-gpu-memory",
        action="store_true",
        help="disable the harness-side nvidia-smi peak-memory sampler",
    )

    parser.add_argument("--sharegpt-output-len", type=int)
    parser.add_argument("--sharegpt-context-len", type=int)

    parser.add_argument("--runtime-profile-id", default="unspecified")
    parser.add_argument("--measurement-interface", default="public_protocol_adapter")
    parser.add_argument("--cache-read-policy", default="enabled")
    parser.add_argument("--cache-write-policy", default="enabled")
    parser.add_argument("--adapter-selection", default="base")
    parser.add_argument("--structured-output-policy", default="none")
    parser.add_argument("--output-constraint", default="default")
    parser.add_argument("--preprocessing", default="dataset_default")
    parser.add_argument("--measured-runs", type=int, default=1)
    parser.add_argument("--server-topology", default="single_server")
    parser.add_argument(
        "--plan-evidence-policy",
        default="declared_contract",
        choices=["declared_contract", "runtime_inspection", "reference_protocol"],
    )
    parser.add_argument("--acceptance-min-success", type=int, default=1)
    parser.add_argument("--acceptance-max-failed", type=int, default=0)
    parser.add_argument("--acceptance-min-images-per-success", type=float, default=0.0)

    parser.add_argument("--smoke", action="store_true", help="1 prompt, no warmup (CI smoke)")
    return parser


def parse_rate(value: str) -> float:
    value = value.strip().lower()
    if value in {"inf", "infinity"}:
        return float("inf")
    rate = float(value)
    if rate <= 0:
        raise ValueError("request rate must be positive or 'inf'")
    return rate


def _rate_slug(rate: float) -> str:
    if math.isinf(rate):
        return "inf"
    return ("%g" % rate).replace(".", "p")


def _make_spec(args: argparse.Namespace, rate: float, concurrency: int | None) -> BenchmarkSpec:
    task = TaskName(args.task)
    cfg_interval: tuple[float, float] | None = None
    if args.cfg_interval:
        interval_values = tuple(float(value.strip()) for value in args.cfg_interval.split(","))
        if len(interval_values) != 2:
            raise ValueError("cfg interval must contain exactly two values")
        cfg_interval = (interval_values[0], interval_values[1])
    return BenchmarkSpec(
        task=task,
        model=args.model,
        endpoint=args.endpoint or "",
        dataset=args.dataset or DEFAULT_DATASETS[task],
        num_prompts=1 if args.smoke else args.num_prompts,
        request_rate=rate,
        max_concurrency=concurrency,
        warmup_requests=0 if args.smoke else args.warmup_requests,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        ignore_eos=not args.disable_ignore_eos,
        max_tokens=args.max_tokens,
        width=args.width,
        height=args.height,
        steps=args.steps,
        max_images=args.max_images,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        cfg_norm=args.cfg_norm,
        cfg_interval=cfg_interval,
        timestep_shift=args.timestep_shift,
        wire=args.wire or "",
        i2t_question=args.i2t_question,
        sample_gpu_memory=not args.no_gpu_memory,
        tokenizer=args.tokenizer,
        sharegpt_output_len=args.sharegpt_output_len,
        sharegpt_context_len=args.sharegpt_context_len,
        dataset_path=args.dataset_path,
        runtime_profile_id=args.runtime_profile_id,
        measurement_interface=args.measurement_interface,
        cache_read_policy=args.cache_read_policy,
        cache_write_policy=args.cache_write_policy,
        adapter_selection=args.adapter_selection,
        structured_output_policy=args.structured_output_policy,
        output_constraint=args.output_constraint,
        preprocessing=args.preprocessing,
        measured_runs=args.measured_runs,
        server_topology=args.server_topology,
        plan_evidence_policy=args.plan_evidence_policy,
        acceptance_min_success=args.acceptance_min_success,
        acceptance_max_failed=args.acceptance_max_failed,
        acceptance_min_images_per_success=args.acceptance_min_images_per_success,
    )


def spec_from_cli_args(argv: Sequence[str]) -> BenchmarkSpec:
    args = build_parser().parse_args(list(argv))
    if args.request_rates or args.max_concurrencies:
        raise ValueError("a canonical artifact contract describes exactly one operating point")
    return _make_spec(args, parse_rate(args.request_rate), args.max_concurrency)


def spec_from_harness_command(command: Sequence[str]) -> BenchmarkSpec:
    try:
        module_index = list(command).index("uniserve_eval.harness.cli")
    except ValueError as error:
        raise ValueError("command does not invoke uniserve_eval.harness.cli") from error
    return spec_from_cli_args(command[module_index + 1 :])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rates = (
        [parse_rate(part) for part in args.request_rates.split(",") if part.strip()]
        if args.request_rates
        else [parse_rate(args.request_rate)]
    )
    concurrencies: list[int | None] = (
        [int(part) for part in args.max_concurrencies.split(",") if part.strip()]
        if args.max_concurrencies
        else [args.max_concurrency]
    )

    output_root = Path(args.output_dir)
    points = [(rate, conc) for rate in rates for conc in concurrencies]
    is_sweep = len(points) > 1
    index: list[dict[str, Any]] = []
    failures = 0

    for rate, concurrency in points:
        spec = _make_spec(args, rate, concurrency)
        if is_sweep:
            slug = f"{spec.task.value}_r{_rate_slug(rate)}_c{concurrency if concurrency else 'none'}"
            out_dir = output_root / slug
        else:
            out_dir = output_root
        result = asyncio.run(BenchmarkRunner(args.base_url, spec, out_dir).run())
        artifact = result.summary.get("artifact", {})
        checks = artifact.get("checks", {})
        ok = bool(
            artifact.get("schema_version") == 2
            and artifact.get("valid") is True
            and artifact.get("valid_marker") == "canonical-valid-v2"
            and isinstance(checks, dict)
            and checks
            and all(value is True for value in checks.values())
        )
        failures += 0 if ok else 1
        index.append(
            {
                "request_rate": "inf" if math.isinf(rate) else rate,
                "max_concurrency": concurrency,
                "output_dir": str(out_dir),
                "ok_count": result.summary["ok_count"],
                "failed_count": result.summary["failed_count"],
                "metrics": result.summary["metrics"],
            }
        )
        print(result.summary["metric_family"], json.dumps(result.summary["metrics"], indent=2, sort_keys=True))

    if is_sweep:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "combined_summary.json").write_text(
            json.dumps({"task": args.task, "points": index}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
