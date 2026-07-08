"""Shared per-request record and percentile helpers for the perf harness.

``RequestRecord`` is the single raw measurement produced for every request,
regardless of task. The streaming families (LLM serving / interleave) fill the
token-timing fields (``ttft``/``itl``/``output_len``); the image families
(t2i / i2i) fill the image fields (``image_latencies``/``image_gen_seconds``).
A request can fill both (interleave emits text *and* images).

Percentiles go through :func:`np.percentile` (linear interpolation) so the
LLM-serving summary is numerically identical to ``refs/sglang``'s
``calculate_metrics`` (which also uses ``np.percentile`` and ``np.mean`` and
multiplies seconds by 1000 to report milliseconds).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RequestRecord:
    """One request's raw measurements (seconds for all timing fields)."""

    request_id: str
    task: str
    success: bool = False
    classifier: str = ""
    error: str | None = None
    endpoint: str = ""

    # Timing (seconds). ``start_time`` is a ``perf_counter()`` taken right before
    # the request is sent; ``latency`` is the request E2E; ``ttft`` is time to the
    # first text token; ``itl`` is the per-token inter-token-latency list.
    start_time: float = 0.0
    latency: float = 0.0
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)

    # Token accounting (server-reported where available).
    prompt_len: int = 0
    output_len: int = 0
    generated_text: str = ""
    # Visible content chunks (OpenAI streaming), retained only for the optional
    # retokenized-ITL cross-check that mirrors sglang.
    text_chunks: list[str] = field(default_factory=list)

    # Image accounting. ``image_latencies`` is per-image E2E (image_done - start);
    # ``first_image_latency`` is time to the first ``image_begin``;
    # ``image_gen_seconds`` is per-image (image_done - image_begin);
    # ``image_steps`` is the diffusion step count reported per image.
    images: int = 0
    image_latencies: list[float] = field(default_factory=list)
    first_image_latency: float | None = None
    image_gen_seconds: list[float] = field(default_factory=list)
    image_steps: list[int] = field(default_factory=list)

    status_code: int | None = None
    finish_reason: str | None = None
    stop_reason: str | None = None

    def record_dict(self) -> dict[str, Any]:
        """Compact per-request row for ``requests.jsonl`` (no large blobs)."""
        return {
            "request_id": self.request_id,
            "task": self.task,
            "success": self.success,
            "classifier": self.classifier,
            "error": self.error,
            "endpoint": self.endpoint,
            "e2e_ms": self.latency * 1000.0,
            "ttft_ms": self.ttft * 1000.0 if self.ttft else None,
            "tpot_ms": (
                (self.latency - self.ttft) / (self.output_len - 1) * 1000.0
                if self.output_len > 1
                else None
            ),
            "itl_count": len(self.itl),
            "prompt_len": self.prompt_len,
            "output_len": self.output_len,
            "images": self.images,
            "first_image_latency_ms": (
                self.first_image_latency * 1000.0 if self.first_image_latency is not None else None
            ),
            "image_latencies_ms": [value * 1000.0 for value in self.image_latencies],
            "image_generation_ms": [value * 1000.0 for value in self.image_gen_seconds],
            "status_code": self.status_code,
            "finish_reason": self.finish_reason,
            "stop_reason": self.stop_reason,
        }


def percentile(values: list[float], p: float) -> float:
    """``np.percentile`` with an empty-list fallback of ``0.0``.

    Mirrors sglang's ``np.percentile(x or 0, p)`` idiom so an empty sample (e.g.
    a non-streaming backend yielding no ITLs) reports ``0.0`` rather than raising.
    """
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, p))


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values)) if len(values) else 0.0


def _max(values: list[float]) -> float:
    return float(np.max(values)) if len(values) else 0.0


def _min(values: list[float]) -> float:
    return float(np.min(values)) if len(values) else 0.0


def distribution(values: list[float], *, scale: float = 1.0) -> dict[str, float | int]:
    """Full distribution (count/mean/std/min/p50/p90/p95/p99/max).

    ``scale`` multiplies every reported statistic; pass ``scale=1000`` to turn a
    seconds sample into milliseconds. ``p50`` is ``np.percentile(.., 50)`` which
    equals ``np.median``.
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
