from __future__ import annotations

from typing import Any

from .base import BenchmarkTask, TaskRequest


class InterleaveTask(BenchmarkTask):
    """Interleaved text+image generation."""

    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        image: dict[str, Any] = {}
        max_images = item.get("max_images", self.spec.max_images)
        if max_images is not None:
            image["max_images"] = int(max_images)
        width = item.get("width", self.spec.width)
        height = item.get("height", self.spec.height)
        if width is not None and height is not None:
            image["width"] = int(width)
            image["height"] = int(height)
        if self.spec.steps is not None:
            image["steps"] = int(self.spec.steps)
        max_tokens = item.get("max_tokens", self.spec.max_tokens or 512)
        if self.spec.interleave_wire == "openai_chat":
            image_config = dict(image)
            if item.get("aspect_ratio") is not None:
                image_config["aspect_ratio"] = str(item["aspect_ratio"])
            payload: dict[str, Any] = {
                "model": self.spec.model,
                "stream": True,
                "stream_options": {"include_usage": True},
                "modalities": ["text", "image"],
                "messages": [{"role": "user", "content": item["prompt"]}],
                "max_completion_tokens": int(max_tokens),
                "temperature": self.spec.temperature,
                "top_p": self.spec.top_p,
                "image_config": image_config,
            }
            endpoint = (
                "/v1/chat/completions"
                if self.spec.endpoint == "/generate"
                else self.spec.endpoint
            )
            if self.spec.extra_request_body:
                extra = dict(self.spec.extra_request_body)
                extra_image = extra.pop("image_config", None)
                if isinstance(extra_image, dict):
                    image_config.update(extra_image)
                payload.update(extra)
            return TaskRequest(endpoint=endpoint, payload=payload, kind="openai_chat")
        if self.spec.interleave_wire != "native":
            raise ValueError(f"unsupported interleave_wire: {self.spec.interleave_wire}")
        payload: dict[str, Any] = {
            "prompt": item["prompt"],
            "mode": "interleave",
            "max_tokens": int(max_tokens),
            "image": image,
        }
        if self.spec.extra_request_body:
            payload.update(self.spec.extra_request_body)
        return TaskRequest(endpoint=self.spec.endpoint, payload=payload, kind="native_generate")
