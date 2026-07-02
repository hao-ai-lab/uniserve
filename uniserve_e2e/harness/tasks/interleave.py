from __future__ import annotations

from typing import Any

from .base import BenchmarkTask, TaskRequest


class InterleaveTask(BenchmarkTask):
    """Interleaved text+image generation via the native ``/generate`` SSE endpoint."""

    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        image: dict[str, Any] = {
            "max_images": item.get("max_images", self.spec.max_images),
        }
        width = item.get("width", self.spec.width)
        height = item.get("height", self.spec.height)
        if width is not None and height is not None:
            image["width"] = int(width)
            image["height"] = int(height)
        if self.spec.steps is not None:
            image["steps"] = int(self.spec.steps)
        max_tokens = item.get("max_tokens", self.spec.max_tokens or 512)
        payload: dict[str, Any] = {
            "prompt": item["prompt"],
            "mode": "interleave",
            "max_tokens": int(max_tokens),
            "image": image,
        }
        if self.spec.extra_request_body:
            payload.update(self.spec.extra_request_body)
        return TaskRequest(endpoint=self.spec.endpoint, payload=payload, kind="native_generate")
