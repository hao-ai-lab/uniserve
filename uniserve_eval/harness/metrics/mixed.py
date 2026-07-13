"""Metrics for fixed-composition image-generation and image-understanding loads."""

from __future__ import annotations

from typing import Any

from .common import RequestRecord
from .image import summarize_image
from .stream import summarize_stream


def summarize_mixed(records: list[RequestRecord], dur_s: float) -> dict[str, Any]:
    successful = [record for record in records if record.success]
    t2i = [record for record in records if record.task == "t2i"]
    i2t = [record for record in records if record.task == "i2t"]
    safe_duration = dur_s if dur_s > 0 else 1e-9
    overlap = _cross_task_overlap(
        [record for record in t2i if record.success],
        [record for record in i2t if record.success],
        safe_duration,
    )
    return {
        "completed_requests": len(successful),
        "mixed_request_throughput": len(successful) / safe_duration,
        "task_counts": {
            "t2i": {"requested": len(t2i), "completed": sum(record.success for record in t2i)},
            "i2t": {"requested": len(i2t), "completed": sum(record.success for record in i2t)},
        },
        "t2i": summarize_image(t2i, safe_duration),
        "i2t": summarize_stream(i2t, safe_duration),
        "client_cross_task_overlap": overlap,
    }


def _cross_task_overlap(
    t2i: list[RequestRecord],
    i2t: list[RequestRecord],
    duration: float,
) -> dict[str, float]:
    t2i_intervals = _merge_intervals(t2i)
    i2t_intervals = _merge_intervals(i2t)
    overlap_s = _intersection_duration(t2i_intervals, i2t_intervals)
    t2i_active_s = sum(end - start for start, end in t2i_intervals)
    i2t_active_s = sum(end - start for start, end in i2t_intervals)
    common_active_s = min(t2i_active_s, i2t_active_s)
    return {
        "duration_s": overlap_s,
        "timed_region_fraction": overlap_s / duration,
        "common_active_time_fraction": (
            overlap_s / common_active_s if common_active_s > 0 else 0.0
        ),
    }


def _merge_intervals(records: list[RequestRecord]) -> list[tuple[float, float]]:
    intervals = sorted(
        (record.start_time, record.start_time + record.latency)
        for record in records
        if record.latency > 0
    )
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        prior_start, prior_end = merged[-1]
        merged[-1] = (prior_start, max(prior_end, end))
    return merged


def _intersection_duration(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    total = 0.0
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        total += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total
