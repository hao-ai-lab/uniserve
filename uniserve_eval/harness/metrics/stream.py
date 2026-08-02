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

import hashlib
import json
import math
from typing import Any

import numpy as np

from .common import RequestRecord, _max, _mean, _std, distribution, percentile

UEVAL_LATENCY_DEFINITION = {
    "clock": "time.perf_counter",
    "event_timestamp": "client_sse_receive_before_parse",
    "server_public_commit_clock": "server_process_monotonic",
    "public_commit_correlation": "scheduler_event_sequence_and_fixed_semantic_root",
    "visible_text": "nonempty_content_or_reasoning",
    "visible_image": "received_image_part_that_decodes_and_conforms",
    "segment": "maximal_consecutive_visible_events_of_one_modality",
    "transition": "destination_segment_first_event_minus_source_segment_last_event",
    "transition_directions": ["text_to_image", "image_to_text"],
    "ttft": "first_visible_text_event_minus_request_send",
    "tpot": "request_e2e_minus_ttft_divided_by_server_completion_tokens_minus_one",
    "image_latency": "decoded_image_event_minus_request_send",
    "aggregation": "arithmetic_mean_with_complete_distributions",
    "missing_sample": "invalidate_point",
}
UEVAL_LATENCY_DEFINITION_DIGEST = hashlib.sha256(
    json.dumps(
        UEVAL_LATENCY_DEFINITION,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()


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
        "e2e_latency_ms": distribution(e2e_latencies, scale=1000),
        # TTFT
        "mean_ttft_ms": _mean(ttfts) * 1000,
        "median_ttft_ms": percentile(ttfts, 50) * 1000,
        "std_ttft_ms": _std(ttfts) * 1000,
        "p50_ttft_ms": percentile(ttfts, 50) * 1000,
        "p90_ttft_ms": percentile(ttfts, 90) * 1000,
        "p95_ttft_ms": percentile(ttfts, 95) * 1000,
        "p99_ttft_ms": percentile(ttfts, 99) * 1000,
        "ttft_ms": distribution(ttfts, scale=1000),
        # TPOT
        "mean_tpot_ms": _mean(tpots) * 1000,
        "median_tpot_ms": percentile(tpots, 50) * 1000,
        "std_tpot_ms": _std(tpots) * 1000,
        "p50_tpot_ms": percentile(tpots, 50) * 1000,
        "p90_tpot_ms": percentile(tpots, 90) * 1000,
        "p95_tpot_ms": percentile(tpots, 95) * 1000,
        "p99_tpot_ms": percentile(tpots, 99) * 1000,
        "tpot_ms": distribution(tpots, scale=1000),
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
    interleave_block = _interleave_block(successful)
    if interleave_block is not None:
        summary["modality_interleave"] = interleave_block
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


def _interleave_block(successful: list[RequestRecord]) -> dict[str, Any] | None:
    multimodal = [record for record in successful if record.images > 0]
    if not multimodal:
        return None
    timings = [_request_transition_timing(record) for record in multimodal]
    transitions = [float(timing["expected_transition_count"]) for timing in timings]
    patterns: dict[str, int] = {}
    for record, timing in zip(multimodal, timings, strict=True):
        pattern = str(timing["signature"] or "->".join(record.output_modalities))
        patterns[pattern] = patterns.get(pattern, 0) + 1
    complete_timings = [timing for timing in timings if timing["valid"] is True]
    transition_latencies = [
        float(value) for timing in complete_timings for value in timing["transition_latencies"]
    ]
    text_to_image = [
        float(value)
        for timing in complete_timings
        for value in timing["directional_latencies"]["text_to_image"]
    ]
    image_to_text = [
        float(value)
        for timing in complete_timings
        for value in timing["directional_latencies"]["image_to_text"]
    ]
    server_transition_latencies = [
        float(value)
        for timing in complete_timings
        for value in timing["server_transition_latencies"]
    ]
    server_text_to_image = [
        float(value)
        for timing in complete_timings
        for value in timing["server_directional_latencies"]["text_to_image"]
    ]
    server_image_to_text = [
        float(value)
        for timing in complete_timings
        for value in timing["server_directional_latencies"]["image_to_text"]
    ]
    delivery_transition_deltas = [
        float(value)
        for timing in complete_timings
        for value in timing["delivery_transition_deltas"]
    ]
    visible_event_count = sum(int(timing["visible_event_count"]) for timing in timings)
    timestamped_event_count = sum(int(timing["timestamped_event_count"]) for timing in timings)
    expected_transition_count = sum(int(timing["expected_transition_count"]) for timing in timings)
    measured_transition_count = len(transition_latencies)
    return {
        "requests_with_text_and_image": sum(
            "text" in record.output_modalities and "image" in record.output_modalities
            for record in multimodal
        ),
        "modality_transitions": distribution(transitions),
        "patterns": patterns,
        "transition_timing": {
            "valid": bool(timings)
            and len(complete_timings) == len(timings)
            and expected_transition_count > 0
            and measured_transition_count == expected_transition_count,
            "latency_definition": UEVAL_LATENCY_DEFINITION,
            "latency_definition_digest": UEVAL_LATENCY_DEFINITION_DIGEST,
            "request_count": len(timings),
            "complete_request_count": len(complete_timings),
            "visible_event_count": visible_event_count,
            "timestamped_event_count": timestamped_event_count,
            "timestamp_coverage": (
                timestamped_event_count / visible_event_count if visible_event_count else 0.0
            ),
            "ambiguous_event_count": sum(
                int(timing["ambiguous_event_count"]) for timing in timings
            ),
            "non_monotonic_event_count": sum(
                int(timing["non_monotonic_event_count"]) for timing in timings
            ),
            "public_commit_event_count": sum(
                int(timing["public_commit_event_count"]) for timing in timings
            ),
            "public_commit_coverage": (
                sum(int(timing["public_commit_event_count"]) for timing in timings)
                / visible_event_count
                if visible_event_count
                else 0.0
            ),
            "public_commit_invalid_count": sum(
                int(timing["public_commit_invalid_count"]) for timing in timings
            ),
            "public_commit_non_monotonic_count": sum(
                int(timing["public_commit_non_monotonic_count"]) for timing in timings
            ),
            "expected_transition_count": expected_transition_count,
            "measured_transition_count": measured_transition_count,
            "transition_sample_coverage": (
                measured_transition_count / expected_transition_count
                if expected_transition_count
                else 0.0
            ),
            "request_signatures": {
                record.request_id: timing["signature"]
                for record, timing in zip(multimodal, timings, strict=True)
            },
            "transition_latency_ms": distribution(transition_latencies, scale=1000),
            "text_to_image_transition_latency_ms": distribution(text_to_image, scale=1000),
            "image_to_text_transition_latency_ms": distribution(image_to_text, scale=1000),
            "server_transition_latency_ms": distribution(
                server_transition_latencies,
                scale=1000,
            ),
            "server_text_to_image_transition_latency_ms": distribution(
                server_text_to_image,
                scale=1000,
            ),
            "server_image_to_text_transition_latency_ms": distribution(
                server_image_to_text,
                scale=1000,
            ),
            "client_delivery_transition_delta_ms": distribution(
                delivery_transition_deltas,
                scale=1000,
            ),
            "boundary_correlations": {
                record.request_id: timing["boundary_correlations"]
                for record, timing in zip(multimodal, timings, strict=True)
            },
        },
    }


def _request_transition_timing(record: RequestRecord) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    visible_event_count = len(record.modality_events)
    timestamped_event_count = 0
    ambiguous_event_count = 0
    non_monotonic_event_count = 0
    public_commit_event_count = 0
    public_commit_invalid_count = 0
    public_commit_non_monotonic_count = 0
    previous_timestamp: float | None = None
    previous_public_seq: int | None = None
    previous_public_timestamp: float | None = None

    for event in record.modality_events:
        modalities = event.get("modalities")
        timestamp = event.get("client_time")
        if not isinstance(modalities, list) or len(modalities) != 1:
            ambiguous_event_count += 1
            continue
        modality = modalities[0]
        if modality not in {"text", "image"}:
            ambiguous_event_count += 1
            continue
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
        ):
            continue
        timestamp_f = float(timestamp)
        timestamped_event_count += 1
        if previous_timestamp is not None and timestamp_f < previous_timestamp:
            non_monotonic_event_count += 1
        previous_timestamp = timestamp_f
        raw_public_commit = event.get("public_commit")
        public_commit = _validated_public_commit(raw_public_commit, modality)
        if public_commit is None:
            if raw_public_commit is not None:
                public_commit_invalid_count += 1
        else:
            public_commit_event_count += 1
            event_seq = int(public_commit["event_seq"])
            committed_at = float(public_commit["committed_at"])
            if previous_public_seq is not None and event_seq <= previous_public_seq:
                public_commit_non_monotonic_count += 1
            if (
                previous_public_timestamp is not None
                and committed_at < previous_public_timestamp
            ):
                public_commit_non_monotonic_count += 1
            previous_public_seq = event_seq
            previous_public_timestamp = committed_at
        if segments and segments[-1]["modality"] == modality:
            segments[-1]["last_timestamp"] = timestamp_f
            segments[-1]["event_count"] = int(segments[-1]["event_count"]) + 1
            segments[-1]["last_public_commit"] = public_commit
        else:
            segments.append(
                {
                    "modality": modality,
                    "first_timestamp": timestamp_f,
                    "last_timestamp": timestamp_f,
                    "event_count": 1,
                    "first_public_commit": public_commit,
                    "last_public_commit": public_commit,
                }
            )

    signature = "->".join(str(segment["modality"]) for segment in segments)
    expected_transition_count = max(0, len(record.output_modalities) - 1)
    transition_latencies: list[float] = []
    directional_latencies: dict[str, list[float]] = {
        "text_to_image": [],
        "image_to_text": [],
    }
    server_transition_latencies: list[float] = []
    server_directional_latencies: dict[str, list[float]] = {
        "text_to_image": [],
        "image_to_text": [],
    }
    delivery_transition_deltas: list[float] = []
    boundary_correlations: list[dict[str, Any]] = []
    for source, destination in zip(segments, segments[1:]):
        latency = float(destination["first_timestamp"]) - float(source["last_timestamp"])
        if latency < 0:
            non_monotonic_event_count += 1
            continue
        direction = f"{source['modality']}_to_{destination['modality']}"
        transition_latencies.append(latency)
        directional_latencies[direction].append(latency)
        source_commit = source["last_public_commit"]
        destination_commit = destination["first_public_commit"]
        if source_commit is None or destination_commit is None:
            continue
        server_latency = float(destination_commit["committed_at"]) - float(
            source_commit["committed_at"]
        )
        if server_latency < 0:
            public_commit_non_monotonic_count += 1
            continue
        server_transition_latencies.append(server_latency)
        server_directional_latencies[direction].append(server_latency)
        delivery_transition_deltas.append(latency - server_latency)
        boundary_correlations.append(
            {
                "direction": direction,
                "source_event_seq": int(source_commit["event_seq"]),
                "destination_event_seq": int(destination_commit["event_seq"]),
                "source_semantic_root": source_commit["semantic_root"],
                "destination_semantic_root": destination_commit["semantic_root"],
                "client_latency_ms": latency * 1000.0,
                "server_commit_latency_ms": server_latency * 1000.0,
                "client_delivery_delta_ms": (latency - server_latency) * 1000.0,
            }
        )

    valid = bool(record.modality_events)
    valid = valid and timestamped_event_count == visible_event_count
    valid = valid and ambiguous_event_count == 0
    valid = valid and non_monotonic_event_count == 0
    valid = valid and len(segments) >= 2
    valid = valid and signature == "->".join(record.output_modalities)
    valid = valid and len(transition_latencies) == expected_transition_count
    valid = valid and public_commit_event_count == visible_event_count
    valid = valid and public_commit_invalid_count == 0
    valid = valid and public_commit_non_monotonic_count == 0
    valid = valid and len(server_transition_latencies) == expected_transition_count
    return {
        "valid": valid,
        "signature": signature,
        "visible_event_count": visible_event_count,
        "timestamped_event_count": timestamped_event_count,
        "ambiguous_event_count": ambiguous_event_count,
        "non_monotonic_event_count": non_monotonic_event_count,
        "public_commit_event_count": public_commit_event_count,
        "public_commit_invalid_count": public_commit_invalid_count,
        "public_commit_non_monotonic_count": public_commit_non_monotonic_count,
        "expected_transition_count": expected_transition_count,
        "transition_latencies": transition_latencies if valid else [],
        "server_transition_latencies": server_transition_latencies if valid else [],
        "delivery_transition_deltas": delivery_transition_deltas if valid else [],
        "boundary_correlations": boundary_correlations if valid else [],
        "directional_latencies": directional_latencies
        if valid
        else {
            "text_to_image": [],
            "image_to_text": [],
        },
        "server_directional_latencies": server_directional_latencies
        if valid
        else {
            "text_to_image": [],
            "image_to_text": [],
        },
    }


def _validated_public_commit(value: Any, modality: str) -> dict[str, Any] | None:
    if not isinstance(value, dict) or value.get("modality") != modality:
        return None
    event_seq = value.get("event_seq")
    committed_at = value.get("committed_at")
    semantic_root = value.get("semantic_root")
    if (
        isinstance(event_seq, bool)
        or not isinstance(event_seq, int)
        or event_seq <= 0
        or isinstance(committed_at, bool)
        or not isinstance(committed_at, (int, float))
        or not math.isfinite(float(committed_at))
        or not isinstance(semantic_root, dict)
    ):
        return None
    producer_op_id = semantic_root.get("producer_op_id")
    point_index = semantic_root.get("point_index")
    semantic_digest = semantic_root.get("semantic_digest")
    if (
        isinstance(producer_op_id, bool)
        or not isinstance(producer_op_id, int)
        or producer_op_id < 0
        or isinstance(point_index, bool)
        or not isinstance(point_index, int)
        or not 0 <= point_index <= 0xFFFF_FFFF
        or not isinstance(semantic_digest, str)
        or len(semantic_digest) != 64
        or any(character not in "0123456789abcdef" for character in semantic_digest)
    ):
        return None
    return {
        "event_seq": event_seq,
        "committed_at": float(committed_at),
        "semantic_root": {
            "producer_op_id": producer_op_id,
            "point_index": point_index,
            "semantic_digest": semantic_digest,
        },
    }


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
            if r.ttft and r.server_scheduled_at is not None and r.server_queued_at is not None
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
