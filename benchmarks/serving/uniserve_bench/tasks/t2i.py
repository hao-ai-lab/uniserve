from __future__ import annotations

from typing import Any

from .base import BenchmarkTask, TaskRequest


class T2ITask(BenchmarkTask):
    """Text-to-image via OpenAI-style ``/v1/images/generations`` (non-streaming).

    Request body matches the UniServe handler: ``prompt`` plus optional
    ``size="WxH"``, ``steps`` and ``seed``. The response is a single JSON
    ``{"data": [{"b64_json", ...}]}`` object.
    """

    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        payload: dict[str, Any] = {"prompt": item["prompt"]}
        width = item.get("width", self.spec.width)
        height = item.get("height", self.spec.height)
        if width is not None and height is not None:
            payload["size"] = f"{width}x{height}"
        steps = item.get("steps", self.spec.steps)
        if steps is not None:
            payload["steps"] = int(steps)
        if item.get("seed") is not None:
            payload["seed"] = int(item["seed"])
        if self.spec.extra_request_body:
            payload.update(self.spec.extra_request_body)
        return TaskRequest(endpoint=self.spec.endpoint, payload=payload, kind="images_generations")
