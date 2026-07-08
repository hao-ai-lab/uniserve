"""Family B metrics: image-speed metrics (t2i + i2i).

Follows the academic/industry convention for diffusion-model serving (Baseten /
GigaGPU / NVIDIA Cosmos / Lambda): the headline is per-image latency percentiles
plus image throughput, reported under a stated config and after warmup.

* per-image latency = request E2E unless the response stream exposes per-image completion timing.
* throughput = images/s and images/min over the wall-clock timed region.
* streamed image responses additionally report time-to-first-image, per-image generation time, and steps/s when those fields are available.
"""
from __future__ import annotations

from typing import Any

from .common import RequestRecord, distribution


def summarize_image(records: list[RequestRecord], dur_s: float) -> dict[str, Any]:
    successful = [r for r in records if r.success]
    completed = len(successful)

    image_latencies: list[float] = []
    total_images = 0
    for r in successful:
        if r.image_latencies:
            image_latencies.extend(r.image_latencies)
            total_images += len(r.image_latencies)
        else:
            # Non-streaming t2i: the request E2E is the image latency.
            image_latencies.append(r.latency)
            total_images += max(1, r.images)

    dur_s = dur_s if dur_s > 0 else 1e-9

    summary: dict[str, Any] = {
        "completed_requests": completed,
        "completed_images": total_images,
        "request_throughput": completed / dur_s,
        "images_per_second": total_images / dur_s,
        "images_per_minute": 60.0 * total_images / dur_s,
        "image_latency_ms": distribution(image_latencies, scale=1000),
    }

    # Stream extras: only present when the stream exposed image timing events.
    ttfi = [r.first_image_latency for r in successful if r.first_image_latency is not None]
    if ttfi:
        summary["time_to_first_image_ms"] = distribution(ttfi, scale=1000)

    gen = [value for r in successful for value in r.image_gen_seconds]
    if gen:
        summary["image_generation_ms"] = distribution(gen, scale=1000)

    steps_per_second: list[float] = []
    for r in successful:
        for steps, gen_s in zip(r.image_steps, r.image_gen_seconds):
            if steps and gen_s > 0:
                steps_per_second.append(steps / gen_s)
    if steps_per_second:
        summary["steps_per_second"] = distribution(steps_per_second)

    return summary
