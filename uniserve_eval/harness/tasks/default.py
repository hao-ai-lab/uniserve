from __future__ import annotations

from typing import Any

from .base import BenchmarkTask, TaskRequest, apply_text_sampling_contract


class DefaultTask(BenchmarkTask):
    """Default text+image generation through streamed chat completions."""

    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        image: dict[str, Any] = {}
        max_images = item.get("max_images", self.spec.max_images)
        if max_images is not None:
            image["num_images"] = int(max_images)
        width = item.get("width", self.spec.width)
        height = item.get("height", self.spec.height)
        if width is not None and height is not None:
            image["width"] = int(width)
            image["height"] = int(height)
        if self.spec.steps is not None:
            image["steps"] = int(self.spec.steps)
        image["seed"] = int(item.get("seed", self.spec.seed))
        if self.spec.guidance_scale is not None:
            image["guidance_scale"] = self.spec.guidance_scale
        if self.spec.image_guidance_scale is not None:
            image["image_guidance_scale"] = self.spec.image_guidance_scale
        if self.spec.cfg_norm is not None:
            image["cfg_norm"] = self.spec.cfg_norm
        if self.spec.cfg_interval is not None:
            image["cfg_interval"] = list(self.spec.cfg_interval)
        if self.spec.timestep_shift is not None:
            image["timestep_shift"] = self.spec.timestep_shift
        max_tokens = item.get("max_tokens", self.spec.max_tokens or 512)

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
            "ignore_eos": self.spec.ignore_eos,
            "image_config": image_config,
        }
        if self.spec.extra_request_body:
            extra = dict(self.spec.extra_request_body)
            extra_image = extra.pop("image_config", None)
            if isinstance(extra_image, dict):
                image_config.update(extra_image)
            payload.update(extra)
        apply_text_sampling_contract(payload, self.spec)
        return TaskRequest(endpoint=self.spec.endpoint, payload=payload, kind="openai_chat")
