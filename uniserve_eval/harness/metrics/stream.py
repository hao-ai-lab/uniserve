"""Streaming token and realized interleave metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .common import RequestRecord, _max, _mean, distribution


def summarize_stream(
    records: list[RequestRecord],
    dur_s: float,
    *,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    successful = [record for record in records if record.success]
    timed = [record for record in successful if record.token_timing_available is not False]
    output_lens = [record.output_len for record in successful]
    total_input = sum(record.prompt_len for record in successful)
    ttfts = [record.ttft for record in timed]
    tpots = [
        (record.latency - record.ttft) / (record.output_len - 1)
        for record in timed
        if record.output_len > 1
    ]
    itls = [value for record in timed for value in record.itl]
    e2e = [record.latency for record in successful]
    total_output = sum(output_lens)
    duration = max(dur_s, 1e-9)
    peak_tokens, peak_requests = _peak_per_second(successful)

    summary: dict[str, Any] = {
        "completed": len(successful),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput": len(successful) / duration,
        "input_throughput": total_input / duration,
        "output_throughput": total_output / duration,
        "total_throughput": (total_input + total_output) / duration,
        "mean_e2e_latency_ms": _mean(e2e) * 1000,
        "e2e_latency_ms": distribution(e2e, scale=1000),
        "mean_ttft_ms": _mean(ttfts) * 1000 if timed else None,
        "ttft_ms": distribution(ttfts, scale=1000),
        "mean_tpot_ms": _mean(tpots) * 1000 if tpots else None,
        "tpot_ms": distribution(tpots, scale=1000),
        "mean_itl_ms": _mean(itls) * 1000 if itls else None,
        "itl_ms": distribution(itls, scale=1000),
        "max_itl_ms": _max(itls) * 1000 if itls else None,
        "concurrency": float(np.sum(e2e)) / duration,
        "max_output_tokens_per_s": peak_tokens,
        "max_concurrent_requests": peak_requests,
        "token_timing_available": bool(timed) and len(timed) == len(successful),
        "token_timing_request_count": len(timed),
    }

    if tokenizer is not None:
        retokenized = sum(
            len(tokenizer.encode(record.generated_text, add_special_tokens=False))
            for record in successful
        )
        summary["total_output_tokens_retokenized"] = retokenized
        summary["output_throughput_retokenized"] = retokenized / duration

    image_block = _image_block(successful, duration)
    if image_block is not None:
        summary["images"] = image_block
    return summary


def _image_block(successful: list[RequestRecord], dur_s: float) -> dict[str, Any] | None:
    image_latencies = [value for record in successful for value in record.image_latencies]
    total_images = sum(record.images for record in successful)
    if total_images == 0:
        return None
    return {
        "total_images": total_images,
        "images_per_second": total_images / dur_s,
        "image_latency_ms": distribution(image_latencies, scale=1000),
    }


def _peak_per_second(successful: list[RequestRecord]) -> tuple[float, int]:
    if not successful:
        return 0.0, 0
    start = min(record.start_time for record in successful)
    end = max(record.start_time + record.latency for record in successful)
    buckets = int(np.ceil(end - start)) + 1
    tokens = np.zeros(buckets)
    requests = np.zeros(buckets)
    for record in successful:
        token_times = [record.start_time + record.ttft]
        for latency in record.itl:
            token_times.append(token_times[-1] + latency)
        for timestamp in token_times:
            index = int(timestamp - start)
            if 0 <= index < buckets:
                tokens[index] += 1
        first = int(record.start_time - start)
        last = int(record.start_time + record.latency - start)
        for index in range(first, min(last + 1, buckets)):
            requests[index] += 1
    return float(np.max(tokens)), int(np.max(requests))
