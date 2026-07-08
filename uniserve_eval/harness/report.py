"""Summary assembly + human-readable report for a single benchmark run."""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from .metrics import RequestRecord, summarize_image, summarize_stream
from .spec import BenchmarkSpec


def spec_to_dict(spec: BenchmarkSpec) -> dict[str, Any]:
    data = asdict(spec)
    data["task"] = spec.task.value
    rate = data.get("request_rate")
    if isinstance(rate, float) and math.isinf(rate):
        data["request_rate"] = "inf"
    return data


def load_mode(spec: BenchmarkSpec) -> str:
    if spec.request_rate != float("inf"):
        return "open_loop_poisson"
    if spec.max_concurrency:
        return "closed_loop"
    return "saturation"


def build_summary(
    spec: BenchmarkSpec,
    base_url: str,
    records: list[RequestRecord],
    dur_s: float,
    *,
    tokenizer: Any | None = None,
    server_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ok = [r for r in records if r.success]
    classifiers: dict[str, int] = {}
    for record in records:
        classifiers[record.classifier] = classifiers.get(record.classifier, 0) + 1

    if spec.is_stream_task:
        family = "stream"
        metrics = summarize_stream(records, dur_s, tokenizer=tokenizer)
    else:
        family = "image"
        metrics = summarize_image(records, dur_s)
    endpoint = _observed_endpoint(records) or spec.endpoint

    return {
        "harness_status": "completed",
        "task": spec.task.value,
        "dataset": spec.dataset,
        "endpoint": endpoint,
        "wire": spec.wire,
        "model": spec.model,
        "base_url": base_url,
        "server_info": server_info,
        "spec": spec_to_dict(spec),
        "load": {
            "mode": load_mode(spec),
            "request_rate": "inf" if spec.request_rate == float("inf") else spec.request_rate,
            "max_concurrency": spec.max_concurrency,
            "warmup_requests": spec.warmup_requests,
            "num_prompts": spec.num_prompts,
            "seed": spec.seed,
        },
        "elapsed_s": dur_s,
        "request_count": len(records),
        "ok_count": len(ok),
        "failed_count": len(records) - len(ok),
        "classifiers": classifiers,
        "metric_family": family,
        "metrics": metrics,
    }


def _observed_endpoint(records: list[RequestRecord]) -> str | None:
    endpoints = {record.endpoint for record in records if record.endpoint}
    return next(iter(endpoints)) if len(endpoints) == 1 else None


def render_markdown(summary: dict[str, Any]) -> str:
    load = summary["load"]
    lines = [
        f"# {summary['task']} benchmark ({summary['dataset']})",
        "",
        f"- model: `{summary['model']}`  endpoint: `{summary['endpoint']}`",
        f"- load: {load['mode']}  rate={load['request_rate']}  "
        f"max_concurrency={load['max_concurrency']}  num_prompts={load['num_prompts']}",
        f"- requests: {summary['ok_count']}/{summary['request_count']} ok  "
        f"elapsed={summary['elapsed_s']:.2f}s",
        "",
    ]
    metrics = summary["metrics"]
    if summary["metric_family"] == "stream":
        lines += _stream_markdown(metrics)
    else:
        lines += _image_markdown(metrics)
    return "\n".join(lines) + "\n"


def _row(label: str, *values: Any) -> str:
    cells = " | ".join(_fmt(value) for value in values)
    return f"| {label} | {cells} |"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return "" if value is None else str(value)


def _stream_markdown(metrics: dict[str, Any]) -> list[str]:
    lines = [
        "| metric | p50 | p90 | p95 | p99 | mean |",
        "|---|---|---|---|---|---|",
        _row("TTFT (ms)", metrics["p50_ttft_ms"], metrics["p90_ttft_ms"], metrics["p95_ttft_ms"], metrics["p99_ttft_ms"], metrics["mean_ttft_ms"]),
        _row("TPOT (ms)", metrics["p50_tpot_ms"], metrics["p90_tpot_ms"], metrics["p95_tpot_ms"], metrics["p99_tpot_ms"], metrics["mean_tpot_ms"]),
        _row("ITL (ms)", metrics["p50_itl_ms"], metrics["p90_itl_ms"], metrics["p95_itl_ms"], metrics["p99_itl_ms"], metrics["mean_itl_ms"]),
        _row("E2E (ms)", metrics["p50_e2e_latency_ms"], metrics["p90_e2e_latency_ms"], metrics["p95_e2e_latency_ms"], metrics["p99_e2e_latency_ms"], metrics["mean_e2e_latency_ms"]),
        "",
        f"- output throughput: {_fmt(metrics['output_throughput'])} tok/s",
        f"- request throughput: {_fmt(metrics['request_throughput'])} req/s",
        f"- total throughput: {_fmt(metrics['total_throughput'])} tok/s",
        f"- concurrency: {_fmt(metrics['concurrency'])}  "
        f"peak tok/s: {_fmt(metrics['max_output_tokens_per_s'])}  "
        f"peak concurrent: {metrics['max_concurrent_requests']}",
    ]
    if "images" in metrics:
        img = metrics["images"]["image_latency_ms"]
        lines += [
            "",
            f"- default-task images: {metrics['images']['total_images']} total, "
            f"{_fmt(metrics['images']['images_per_second'])} img/s, "
            f"image E2E p50/p99 = {_fmt(img['p50'])}/{_fmt(img['p99'])} ms",
        ]
    return lines


def _image_markdown(metrics: dict[str, Any]) -> list[str]:
    lat = metrics["image_latency_ms"]
    lines = [
        "| metric | p50 | p90 | p95 | p99 | mean |",
        "|---|---|---|---|---|---|",
        _row("image latency (ms)", lat["p50"], lat["p90"], lat["p95"], lat["p99"], lat["mean"]),
        "",
        f"- images/s: {_fmt(metrics['images_per_second'])}  "
        f"images/min: {_fmt(metrics['images_per_minute'])}",
        f"- request throughput: {_fmt(metrics['request_throughput'])} req/s",
        f"- completed: {metrics['completed_requests']} requests, {metrics['completed_images']} images",
    ]
    if "time_to_first_image_ms" in metrics:
        ttfi = metrics["time_to_first_image_ms"]
        lines.append(f"- time-to-first-image p50/p99 = {_fmt(ttfi['p50'])}/{_fmt(ttfi['p99'])} ms")
    if "steps_per_second" in metrics:
        sps = metrics["steps_per_second"]
        lines.append(f"- steps/s p50 = {_fmt(sps['p50'])}")
    return lines
