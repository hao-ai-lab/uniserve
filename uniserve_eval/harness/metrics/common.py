"""Shared per-request record and percentile helpers for the perf harness.

``RequestRecord`` is the single raw measurement produced for every request,
regardless of task. The streaming families (LLM serving / default task) fill the
token-timing fields (``ttft``/``itl``/``output_len``); the image families
(t2i / i2i) fill the image fields (``image_latencies``/``image_gen_seconds``).
A request can fill both when it emits text and images.

Percentiles go through :func:`np.percentile` (linear interpolation) so the
LLM-serving summary is numerically identical to ``refs/sglang``'s
``calculate_metrics`` (which also uses ``np.percentile`` and ``np.mean`` and
multiplies seconds by 1000 to report milliseconds).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..image_outputs import DecodedImage


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
    scheduled_time: float | None = None
    start_time: float = 0.0
    http_response_time: float | None = None
    first_text_time: float | None = None
    first_image_begin_time: float | None = None
    first_image_done_time: float | None = None
    final_event_time: float | None = None
    latency: float = 0.0
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    token_timing_available: bool | None = None

    # Server-side scheduling timestamps when a stream exposes them. They are wall-clock seconds from the server process; only the duration between them is mixed with client-side durations.
    server_queued_at: float | None = None
    server_scheduled_at: float | None = None

    # Token accounting (server-reported where available).
    prompt_len: int = 0
    output_len: int = 0
    requested_output_len: int = 0
    prompt_len_source: str = "request_fallback"
    output_len_source: str = "requested_fallback"
    cached_prompt_tokens: int | None = None
    cached_prompt_tokens_source: str = "unavailable"
    generated_text: str = ""
    # Visible content chunks (OpenAI streaming), retained only for the optional
    # retokenized-ITL cross-check that mirrors sglang.
    text_chunks: list[str] = field(default_factory=list)
    output_modalities: list[str] = field(default_factory=list)
    modality_events: list[dict[str, Any]] = field(default_factory=list)

    # Image accounting. ``image_latencies`` is per-image E2E from request start to image availability; ``first_image_latency`` is time to the first image signal; ``image_gen_seconds`` is populated only when a backend exposes a per-image generation span; ``image_steps`` is populated only when a backend reports per-image diffusion steps.
    images: int = 0
    image_latencies: list[float] = field(default_factory=list)
    first_image_latency: float | None = None
    image_gen_seconds: list[float] = field(default_factory=list)
    image_steps: list[int] = field(default_factory=list)
    image_spans: list[dict[str, Any]] = field(default_factory=list)
    generated_images_expected: bool = False
    requested_image_count: int | None = None
    requested_image_count_is_cap: bool = False
    requested_image_width: int | None = None
    requested_image_height: int | None = None
    # Exact response bytes live only for the duration of the harness process.
    # ``record_dict`` emits compact metadata and the runner writes the bytes to
    # content-addressed files under ``samples/``.
    decoded_images: list[DecodedImage] = field(default_factory=list, repr=False)

    status_code: int | None = None
    finish_reason: str | None = None
    stop_reason: str | None = None

    def record_dict(self) -> dict[str, Any]:
        """Compact per-request row for ``requests.jsonl`` (no large blobs)."""
        generated_text_bytes = self.generated_text.encode("utf-8")
        server_queue_wait = (
            self.server_scheduled_at - self.server_queued_at
            if self.server_queued_at is not None and self.server_scheduled_at is not None
            else None
        )
        http_response = (
            self.http_response_time - self.start_time
            if self.http_response_time is not None
            else None
        )
        dispatch_wait = (
            self.start_time - self.scheduled_time if self.scheduled_time is not None else None
        )
        stream_first_text_wait = (
            self.first_text_time - self.http_response_time
            if self.first_text_time is not None and self.http_response_time is not None
            else None
        )
        ttft_residual = (
            max(0.0, self.ttft - server_queue_wait)
            if self.ttft and server_queue_wait is not None
            else None
        )
        return {
            "request_id": self.request_id,
            "task": self.task,
            "success": self.success,
            "classifier": self.classifier,
            "error": self.error,
            "endpoint": self.endpoint,
            "scheduled_time": self.scheduled_time,
            "client_send_time": self.start_time,
            "http_response_time": self.http_response_time,
            "first_text_time": self.first_text_time,
            "first_image_begin_time": self.first_image_begin_time,
            "first_image_done_time": self.first_image_done_time,
            "final_event_time": self.final_event_time,
            "server_queued_at": self.server_queued_at,
            "server_scheduled_at": self.server_scheduled_at,
            "client_dispatch_wait_ms": (
                dispatch_wait * 1000.0 if dispatch_wait is not None else None
            ),
            "http_response_ms": (http_response * 1000.0 if http_response is not None else None),
            "stream_first_text_wait_ms": (
                stream_first_text_wait * 1000.0 if stream_first_text_wait is not None else None
            ),
            "server_queue_wait_ms": (
                server_queue_wait * 1000.0 if server_queue_wait is not None else None
            ),
            "ttft_residual_after_server_queue_ms": (
                ttft_residual * 1000.0 if ttft_residual is not None else None
            ),
            "e2e_ms": self.latency * 1000.0,
            "token_timing_available": self.token_timing_available is not False,
            "ttft_ms": (
                self.ttft * 1000.0
                if self.token_timing_available is not False and self.ttft
                else None
            ),
            "tpot_ms": (
                (self.latency - self.ttft) / (self.output_len - 1) * 1000.0
                if self.token_timing_available is not False and self.output_len > 1
                else None
            ),
            "itl_count": len(self.itl),
            "prompt_len": self.prompt_len,
            "output_len": self.output_len,
            "requested_output_len": self.requested_output_len,
            "prompt_len_source": self.prompt_len_source,
            "output_len_source": self.output_len_source,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "cached_prompt_tokens_source": self.cached_prompt_tokens_source,
            "generated_text_bytes": len(generated_text_bytes),
            "generated_text_sha256": (
                hashlib.sha256(generated_text_bytes).hexdigest() if generated_text_bytes else None
            ),
            "output_modalities": list(self.output_modalities),
            "modality_events": [
                {
                    "modalities": list(event.get("modalities", [])),
                    "text_bytes": int(event.get("text_bytes", 0)),
                    "image_count": int(event.get("image_count", 0)),
                    "client_offset_ms": (
                        (float(event["client_time"]) - self.start_time) * 1000.0
                        if isinstance(event.get("client_time"), (int, float))
                        and not isinstance(event.get("client_time"), bool)
                        else None
                    ),
                }
                for event in self.modality_events
            ],
            "images": self.images,
            "generated_images_expected": self.generated_images_expected,
            "requested_image_count": self.requested_image_count,
            "requested_image_count_is_cap": self.requested_image_count_is_cap,
            "requested_image_width": self.requested_image_width,
            "requested_image_height": self.requested_image_height,
            "image_outputs": [image.metadata_dict() for image in self.decoded_images],
            "first_image_latency_ms": (
                self.first_image_latency * 1000.0 if self.first_image_latency is not None else None
            ),
            "image_latencies_ms": [value * 1000.0 for value in self.image_latencies],
            "image_generation_ms": [value * 1000.0 for value in self.image_gen_seconds],
            "image_steps": list(self.image_steps),
            "image_spans": list(self.image_spans),
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
