"""Family A metrics: streaming token metrics (LLM serving + default task).

This summarizer is intentionally a faithful re-implementation of
``refs/sglang/python/sglang/benchmark/serving.py`` ``calculate_metrics`` so the
LLM-serving numbers are directly comparable to SGLang's. The key contracts:

* TTFT / ITL / E2E come from the per-request live measurements.
* ``TPOT = (latency - ttft) / (output_len - 1)`` per request (only ``output_len
  > 1``) using the *server-reported* ``output_len`` -- this is NOT the mean of
  ITLs.
* percentiles use ``np.percentile``; every latency is reported in milliseconds.
* throughput uses the wall-clock ``dur_s`` of the timed region.
* peak ``max_output_tokens_per_s`` / ``max_concurrent_requests`` use the same
  1-second bucketing (token times reconstructed from ``start_time + ttft`` then
  cumulative ITLs).

The parity test in ``tests/python/unit/eval_driver/test_stream_parity.py`` pins
this against the exact SGLang formulas.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .common import RequestRecord, _max, _mean, _std, distribution, percentile


def summarize_stream(
    records: list[RequestRecord],
    dur_s: float,
    *,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    successful = [r for r in records if r.success]
    timed_successful = [r for r in successful if r.token_timing_available is not False]
    completed = len(successful)

    output_lens: list[int] = []
    retokenized_output_lens: list[int] = []
    total_input = 0
    ttfts: list[float] = []
    tpots: list[float] = []
    itls: list[float] = []
    e2e_latencies: list[float] = []

    for r in successful:
        output_lens.append(r.output_len)
        if tokenizer is not None:
            retokenized_output_lens.append(
                len(tokenizer.encode(r.generated_text, add_special_tokens=False))
            )
        total_input += r.prompt_len
        if r.token_timing_available is not False:
            if r.output_len > 1:
                tpots.append((r.latency - r.ttft) / (r.output_len - 1))
            itls += r.itl
            ttfts.append(r.ttft)
        e2e_latencies.append(r.latency)

    total_output = sum(output_lens)
    total_output_retokenized = sum(retokenized_output_lens)

    max_output_tokens_per_s, _ = _peak_per_second(timed_successful)
    _, max_concurrent_requests = _peak_per_second(successful)

    dur_s = dur_s if dur_s > 0 else 1e-9

    summary: dict[str, Any] = {
        "completed": completed,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput": completed / dur_s,
        "input_throughput": total_input / dur_s,
        "output_throughput": total_output / dur_s,
        "total_throughput": (total_input + total_output) / dur_s,
        # E2E
        "mean_e2e_latency_ms": _mean(e2e_latencies) * 1000,
        "median_e2e_latency_ms": percentile(e2e_latencies, 50) * 1000,
        "std_e2e_latency_ms": _std(e2e_latencies) * 1000,
        "p50_e2e_latency_ms": percentile(e2e_latencies, 50) * 1000,
        "p90_e2e_latency_ms": percentile(e2e_latencies, 90) * 1000,
        "p95_e2e_latency_ms": percentile(e2e_latencies, 95) * 1000,
        "p99_e2e_latency_ms": percentile(e2e_latencies, 99) * 1000,
        # TTFT
        "mean_ttft_ms": _mean(ttfts) * 1000,
        "median_ttft_ms": percentile(ttfts, 50) * 1000,
        "std_ttft_ms": _std(ttfts) * 1000,
        "p50_ttft_ms": percentile(ttfts, 50) * 1000,
        "p90_ttft_ms": percentile(ttfts, 90) * 1000,
        "p95_ttft_ms": percentile(ttfts, 95) * 1000,
        "p99_ttft_ms": percentile(ttfts, 99) * 1000,
        # TPOT
        "mean_tpot_ms": _mean(tpots) * 1000,
        "median_tpot_ms": percentile(tpots, 50) * 1000,
        "std_tpot_ms": _std(tpots) * 1000,
        "p50_tpot_ms": percentile(tpots, 50) * 1000,
        "p90_tpot_ms": percentile(tpots, 90) * 1000,
        "p95_tpot_ms": percentile(tpots, 95) * 1000,
        "p99_tpot_ms": percentile(tpots, 99) * 1000,
        # ITL
        "mean_itl_ms": _mean(itls) * 1000,
        "median_itl_ms": percentile(itls, 50) * 1000,
        "std_itl_ms": _std(itls) * 1000,
        "p50_itl_ms": percentile(itls, 50) * 1000,
        "p90_itl_ms": percentile(itls, 90) * 1000,
        "p95_itl_ms": percentile(itls, 95) * 1000,
        "p99_itl_ms": percentile(itls, 99) * 1000,
        "max_itl_ms": _max(itls) * 1000,
        # Concurrency / peak
        "concurrency": float(np.sum(e2e_latencies)) / dur_s,
        "max_output_tokens_per_s": max_output_tokens_per_s,
        "max_concurrent_requests": max_concurrent_requests,
        "token_timing_available": bool(timed_successful) and len(timed_successful) == completed,
        "token_timing_request_count": len(timed_successful),
    }

    if not timed_successful:
        for key in (
            "mean_ttft_ms",
            "median_ttft_ms",
            "std_ttft_ms",
            "p50_ttft_ms",
            "p90_ttft_ms",
            "p95_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "median_tpot_ms",
            "std_tpot_ms",
            "p50_tpot_ms",
            "p90_tpot_ms",
            "p95_tpot_ms",
            "p99_tpot_ms",
            "mean_itl_ms",
            "median_itl_ms",
            "std_itl_ms",
            "p50_itl_ms",
            "p90_itl_ms",
            "p95_itl_ms",
            "p99_itl_ms",
            "max_itl_ms",
            "max_output_tokens_per_s",
        ):
            summary[key] = None
        summary["token_timing_unavailable_reason"] = "non_streaming_response"

    if tokenizer is not None:
        summary["total_output_tokens_retokenized"] = total_output_retokenized
        summary["output_throughput_retokenized"] = total_output_retokenized / dur_s
        summary["total_throughput_retokenized"] = (total_input + total_output_retokenized) / dur_s

    # Default generation can emit images alongside text; attach an image block so the same
    # run reports per-image latency without polluting the token metrics.
    image_block = _default_image_block(successful, dur_s)
    if image_block is not None:
        summary["images"] = image_block
    timing_block = _timing_attribution_block(successful)
    if timing_block is not None:
        summary["timing_attribution"] = timing_block

    return summary


def _peak_per_second(successful: list[RequestRecord]) -> tuple[float, int]:
    """Replicate sglang's 1-second peak bucketing for tokens and concurrency."""
    if not successful:
        return 0.0, 0

    min_start_time = min(r.start_time for r in successful)
    max_end_time = max(r.start_time + r.latency for r in successful)
    duration_seconds = int(np.ceil(max_end_time - min_start_time)) + 1
    tokens_per_second = np.zeros(duration_seconds)
    concurrent_requests_per_second = np.zeros(duration_seconds)

    for r in successful:
        token_times = [r.start_time + r.ttft]
        current_time = token_times[0]
        for itl_value in r.itl:
            current_time += itl_value
            token_times.append(current_time)

        for token_time in token_times:
            second_bucket = int(token_time - min_start_time)
            if 0 <= second_bucket < duration_seconds:
                tokens_per_second[second_bucket] += 1

        request_start_second = int(r.start_time - min_start_time)
        request_end_second = int((r.start_time + r.latency) - min_start_time)
        for second in range(request_start_second, min(request_end_second + 1, duration_seconds)):
            concurrent_requests_per_second[second] += 1

    return float(np.max(tokens_per_second)), int(np.max(concurrent_requests_per_second))


def _default_image_block(successful: list[RequestRecord], dur_s: float) -> dict[str, Any] | None:
    image_latencies = [value for r in successful for value in r.image_latencies]
    total_images = sum(r.images for r in successful)
    if total_images == 0 and not image_latencies:
        return None
    ttfi = [r.first_image_latency for r in successful if r.first_image_latency is not None]
    gen = [value for r in successful for value in r.image_gen_seconds]
    block: dict[str, Any] = {
        "total_images": total_images,
        "images_per_second": total_images / dur_s,
        "image_latency_ms": distribution(image_latencies, scale=1000),
    }
    if ttfi:
        block["time_to_first_image_ms"] = distribution(ttfi, scale=1000)
    if gen:
        block["image_generation_ms"] = distribution(gen, scale=1000)
    steps = [float(step) for r in successful for step in r.image_steps]
    if steps:
        block["image_steps"] = distribution(steps)
    return block


def _timing_attribution_block(successful: list[RequestRecord]) -> dict[str, Any] | None:
    def collect_seconds(fn: Any) -> list[float]:
        values: list[float] = []
        for record in successful:
            value = fn(record)
            if value is not None:
                values.append(float(value))
        return values

    client_dispatch_wait = collect_seconds(
        lambda r: r.start_time - r.scheduled_time if r.scheduled_time is not None else None
    )
    http_response = collect_seconds(
        lambda r: r.http_response_time - r.start_time if r.http_response_time is not None else None
    )
    stream_first_text_wait = collect_seconds(
        lambda r: (
            r.first_text_time - r.http_response_time
            if r.first_text_time is not None and r.http_response_time is not None
            else None
        )
    )
    server_queue_wait = collect_seconds(
        lambda r: (
            r.server_scheduled_at - r.server_queued_at
            if r.server_scheduled_at is not None and r.server_queued_at is not None
            else None
        )
    )
    residual_after_server_queue = collect_seconds(
        lambda r: (
            max(
                0.0,
                r.ttft - (r.server_scheduled_at - r.server_queued_at),
            )
            if r.ttft
            and r.server_scheduled_at is not None
            and r.server_queued_at is not None
            else None
        )
    )

    block: dict[str, Any] = {}
    if client_dispatch_wait:
        block["client_dispatch_wait_ms"] = distribution(client_dispatch_wait, scale=1000)
    if http_response:
        block["http_response_ms"] = distribution(http_response, scale=1000)
    if stream_first_text_wait:
        block["stream_first_text_wait_ms"] = distribution(stream_first_text_wait, scale=1000)
    if server_queue_wait:
        block["server_queue_wait_ms"] = distribution(server_queue_wait, scale=1000)
    if residual_after_server_queue:
        block["ttft_residual_after_server_queue_ms"] = distribution(
            residual_after_server_queue,
            scale=1000,
        )
    return block or None
