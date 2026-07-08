from __future__ import annotations

from typing import Any

from .base import BenchmarkTask, TaskRequest


class DefaultTask(BenchmarkTask):
    """Default text+image generation, dual-wire.

    * ``"native"`` — UniServe ``/generate`` SSE with the default generation constraint.
    * ``"openai_chat"`` — OpenAI chat completions SSE with
      ``modalities: ["text", "image"]`` and ``image_config`` (the official
      LightLLM V2 chat shape); text arrives as ``delta.content`` and
      generated images as ``delta.images`` data URLs.
    """

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

        if self.spec.wire == "openai_chat":
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
            if self.spec.extra_request_body:
                extra = dict(self.spec.extra_request_body)
                extra_image = extra.pop("image_config", None)
                if isinstance(extra_image, dict):
                    image_config.update(extra_image)
                payload.update(extra)
            return TaskRequest(endpoint=self.spec.endpoint, payload=payload, kind="openai_chat")

        payload = {
            "prompt": item["prompt"],
            "constraint": "default",
            "max_tokens": int(max_tokens),
            "temperature": self.spec.temperature,
            "image": image,
        }
        if self.spec.top_p < 1.0:
            payload["top_p"] = self.spec.top_p
        if self.spec.extra_request_body:
            payload.update(self.spec.extra_request_body)
        return TaskRequest(endpoint=self.spec.endpoint, payload=payload, kind="native_generate")
