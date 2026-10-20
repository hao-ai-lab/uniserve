"""Computes request and video benchmark metrics.

``build_summary`` calls ``summarize`` on the measured ``RequestRecord`` list
and the measured duration from ``run_load``. Record latencies are seconds
measured from each request's client send time, which follows any client-side
concurrency queueing. Reported latency distributions are in milliseconds.
Throughputs and per-second rates divide by the measured window.
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


def summarize(records: list[RequestRecord], dur_s: float) -> dict[str, Any]:
    """Build aggregate throughput and latency metrics from successful records.

    Args:
        records: Measured request records; warmup records are excluded by the
            caller.
        dur_s: Measured wall-clock window in seconds, the denominator of
            every throughput and of ``concurrency``.

    Returns:
        A JSON-compatible metric mapping. The server timing keys are present
        only when successful records carry reported timings.
    """
    # Failed requests remain validation inputs but do not contribute
    # service-rate or latency populations.
    successful = [record for record in records if record.success]
    total_input = sum(record.prompt_len for record in successful)
    e2e = [record.latency for record in successful]
    duration = max(dur_s, 1e-9)

    # The measured wall-clock window is the shared denominator for
    # throughput and time-integrated concurrency; its floor only protects
    # empty synthetic inputs. ``concurrency`` is the time-averaged number of
    # in-flight successful requests: summed e2e latency over the window.
    summary: dict[str, Any] = {
        "completed": len(successful),
        "completed_requests": len(successful),
        "total_input_tokens": total_input,
        "request_throughput": len(successful) / duration,
        "input_throughput": total_input / duration,
        "mean_e2e_latency_ms": _mean(e2e) * 1000,
        "e2e_latency_ms": distribution(e2e, scale=1000),
        "concurrency": float(np.sum(e2e)) / duration if e2e else 0.0,
    }

    videos = [
        record.decoded_video
        for record in successful
        if record.decoded_video is not None
    ]
    summary["completed_videos"] = len(videos)
    summary["videos_per_second"] = len(videos) / duration
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
    return summary


def _server_timings(successful: list[RequestRecord]) -> dict[str, Any]:
    """Summarize reported video timings using the server's definitions."""
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
    return summary
