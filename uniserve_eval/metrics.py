"""Computes request, token, image, video, and decision benchmark metrics.

``build_summary`` calls ``summarize`` on the measured ``RequestRecord`` list
and the measured duration from ``run_load``. Record latencies (end-to-end,
TTFT, image) are seconds measured from each request's client send time,
which follows any client-side concurrency queueing. Reported latency
distributions are in milliseconds. Throughputs and per-second rates divide by
the measured window; the one-second peaks from ``_peak_per_second`` do not.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import RequestRecord


def percentile(values: list[float], p: float) -> float:
    """Return a percentile, using zero for an empty population.

    Uses NumPy's default linear interpolation between order statistics.
    """
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, p))


def _mean(values: list[float]) -> float:
    """Return the population mean, using zero for an empty population."""
    return float(np.mean(values)) if len(values) else 0.0


def _std(values: list[float]) -> float:
    """Return population standard deviation, using zero when empty."""
    return float(np.std(values)) if len(values) else 0.0


def _max(values: list[float]) -> float:
    """Return the maximum, using zero for an empty population."""
    return float(np.max(values)) if len(values) else 0.0


def _min(values: list[float]) -> float:
    """Return the minimum, using zero for an empty population."""
    return float(np.min(values)) if len(values) else 0.0


def distribution(
    values: list[float], *, scale: float = 1.0
) -> dict[str, float | int]:
    """Summarize a population with count, moments, extrema, and percentiles.

    Every statistic except ``count`` is multiplied by ``scale``, which callers
    use to convert seconds to milliseconds. An empty population reports zero
    for every statistic.
    """
    return {
        "count": len(values),
        "mean": _mean(values) * scale,
        "std": _std(values) * scale,
        "min": _min(values) * scale,
        "p50": percentile(values, 50) * scale,
        "p90": percentile(values, 90) * scale,
        "p95": percentile(values, 95) * scale,
        "p99": percentile(values, 99) * scale,
        "max": _max(values) * scale,
    }


def summarize(
    records: list[RequestRecord],
    dur_s: float,
    *,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Build aggregate throughput and latency metrics from successful records.

    Args:
        records: Measured request records; warmup records are excluded by the
            caller.
        dur_s: Measured wall-clock window in seconds, the denominator of
            every throughput and of ``concurrency``.
        tokenizer: Optional tokenizer for retokenized output counts. The
            benchmark runner passes the dataset tokenizer, which is loaded
            only for datasets that require one.

    Returns:
        A JSON-compatible metric mapping. Media keys are present only when
        successful records produced that modality, decision keys only when
        they answered decision readouts, and retokenized keys only when
        ``tokenizer`` is given.
    """  # noqa: E501
    # Failed requests remain validation inputs but do not contribute
    # service-rate or latency populations. Token metrics additionally
    # require stream timing.
    successful = [record for record in records if record.success]
    timed = [
        record
        for record in successful
        if record.token_timing_available is not False
    ]
    output_lens = [record.output_len for record in successful]
    total_input = sum(record.prompt_len for record in successful)
    ttfts = [record.ttft for record in timed if record.token_timing_available]
    # TPOT spreads the time after the first token over the remaining output
    # tokens, so it includes any delay between the last text event and the
    # final stream event that closes ``latency``.
    tpots = [
        (record.latency - record.ttft) / (record.output_len - 1)
        for record in timed
        if record.token_timing_available and record.output_len > 1
    ]
    itls = [
        value
        for record in timed
        if record.token_timing_available
        for value in record.itl
    ]
    e2e = [record.latency for record in successful]
    total_output = sum(output_lens)
    duration = max(dur_s, 1e-9)

    # Both peaks consider only records with stream timing.
    timed_success = [
        record for record in timed if record.token_timing_available
    ]
    peak_tokens, peak_requests = (
        _peak_per_second(timed_success) if timed_success else (0.0, 0)
    )

    # The measured wall-clock window is the shared denominator for
    # throughput and time-integrated concurrency; its floor only protects
    # empty synthetic inputs. ``concurrency`` is the time-averaged number of
    # in-flight successful requests: summed e2e latency over the window.
    summary: dict[str, Any] = {
        "completed": len(successful),
        "completed_requests": len(successful),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput": len(successful) / duration,
        "input_throughput": total_input / duration,
        "output_throughput": total_output / duration,
        "total_throughput": (total_input + total_output) / duration,
        "mean_e2e_latency_ms": _mean(e2e) * 1000,
        "e2e_latency_ms": distribution(e2e, scale=1000),
        "mean_ttft_ms": _mean(ttfts) * 1000 if ttfts else None,
        "ttft_ms": distribution(ttfts, scale=1000),
        "mean_tpot_ms": _mean(tpots) * 1000 if tpots else None,
        "tpot_ms": distribution(tpots, scale=1000),
        "mean_itl_ms": _mean(itls) * 1000 if itls else None,
        "itl_ms": distribution(itls, scale=1000),
        "max_itl_ms": _max(itls) * 1000 if itls else None,
        "concurrency": float(np.sum(e2e)) / duration if e2e else 0.0,
        "max_output_tokens_per_s": peak_tokens,
        "max_concurrent_requests": peak_requests,
        "token_timing_available": bool(timed_success)
        and len(timed_success) == len(successful),
        "token_timing_request_count": len(timed_success),
    }

    # Retokenization provides a tokenizer-derived count alongside
    # server-reported output lengths without replacing the primary protocol
    # metric.
    if tokenizer is not None:
        retokenized = sum(
            len(
                tokenizer.encode(
                    record.generated_text, add_special_tokens=False
                )
            )
            for record in successful
        )
        summary["total_output_tokens_retokenized"] = retokenized
        summary["output_throughput_retokenized"] = retokenized / duration

    # Media metrics are emitted only for modalities present in successful
    # results.
    image_latencies = [
        value for record in successful for value in record.image_latencies
    ]
    total_images = sum(record.images for record in successful)
    if total_images:
        # When no successful record carries per-image latencies, each image
        # is assigned its request's end-to-end latency.
        if not image_latencies:
            image_latencies = [
                record.latency
                for record in successful
                for _ in range(record.images)
            ]
        summary["completed_images"] = total_images
        summary["images_per_second"] = total_images / duration
        summary["images_per_minute"] = 60.0 * total_images / duration
        summary["image_latency_ms"] = distribution(image_latencies, scale=1000)
        ttfi = [
            record.first_image_latency
            for record in successful
            if record.first_image_latency is not None
        ]
        if ttfi:
            summary["time_to_first_image_ms"] = distribution(ttfi, scale=1000)
    videos = [
        record.decoded_video
        for record in successful
        if record.decoded_video is not None
    ]
    if any(record.task == "video" for record in records):
        summary["completed_videos"] = len(videos)
        summary["videos_per_second"] = len(videos) / duration
        # Only MP4 bytes of returned videos count; images do not contribute.
        summary["media_bytes_per_second"] = (
            sum(video.byte_size for video in videos) / duration
        )
        summary["video_latency_ms"] = distribution(
            [
                record.latency
                for record in successful
                if record.decoded_video is not None
            ],
            scale=1000,
        )
        summary.update(_server_timings(successful))

    # Decision readouts answer states and questions rather than generating
    # tokens; their rates share the measured-window denominator.
    decision_states = sum(record.decision_states for record in successful)
    if decision_states:
        decision_questions = sum(
            record.decision_questions for record in successful
        )
        summary["completed_states"] = decision_states
        summary["states_per_second"] = decision_states / duration
        summary["completed_questions"] = decision_questions
        summary["questions_per_second"] = decision_questions / duration
    return summary


def _server_timings(successful: list[RequestRecord]) -> dict[str, Any]:
    """Summarize reported video timings using each server's definitions."""
    summary: dict[str, Any] = {}
    inference = [
        record.server_inference_s
        for record in successful
        if record.server_inference_s is not None
    ]
    if inference:
        summary["server_inference_ms"] = distribution(inference, scale=1000)

    stages: dict[str, list[float]] = {}
    for record in successful:
        for name, seconds in record.server_stage_s.items():
            stages.setdefault(name, []).append(seconds)
    if stages:
        summary["server_stage_ms"] = {
            name: distribution(values, scale=1000)
            for name, values in stages.items()
        }

    peaks = [
        record.server_peak_memory_mib
        for record in successful
        if record.server_peak_memory_mib is not None
    ]
    if peaks:
        summary["server_peak_memory_mib"] = max(peaks)
    return summary


def _peak_per_second(successful: list[RequestRecord]) -> tuple[float, int]:
    """Return peak one-second token completions and overlapping requests.

    Time is divided into one-second buckets from the earliest request start.
    Token arrivals are the recorded ``text_times`` of each request, so each
    stamped text event counts as one token at the time it arrived, including
    an event that follows an image and so has no ``itl`` gap. A request
    counts toward every bucket its start-to-end interval touches, so the
    request peak can exceed the instantaneous maximum concurrency. The two
    peaks are maximized independently and may come from different buckets.
    """
    if not successful:
        return 0.0, 0
    start = min(record.start_time for record in successful)
    end = max(record.start_time + record.latency for record in successful)
    buckets = int(np.ceil(end - start)) + 1
    tokens = np.zeros(buckets)
    requests = np.zeros(buckets)

    for record in successful:
        for timestamp in record.text_times:
            index = int(timestamp - start)
            if 0 <= index < buckets:
                tokens[index] += 1
        first = int(record.start_time - start)
        last = int(record.start_time + record.latency - start)
        for index in range(first, min(last + 1, buckets)):
            requests[index] += 1

    return float(np.max(tokens)), int(np.max(requests))
