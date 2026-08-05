from __future__ import annotations

from typing import Any

from .base import (
    BenchmarkTask,
    TaskRequest,
    apply_chat_image_contract,
    input_image_data_url,
    uses_external_request_schema,
)


class I2ITask(BenchmarkTask):
    """Image-to-image editing through non-streamed chat completions."""

    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        image_config: dict[str, Any] = {}
        max_images = item.get("max_images", self.spec.max_images)
        if max_images is not None:
            image_config["num_images"] = int(max_images)
        width = item.get("width", self.spec.width)
        height = item.get("height", self.spec.height)
        if width is not None and height is not None:
            image_config["width"] = int(width)
            image_config["height"] = int(height)
        if self.spec.steps is not None:
            image_config["steps"] = int(self.spec.steps)
        image_config["seed"] = int(item.get("seed", self.spec.seed))
        if self.spec.guidance_scale is not None:
            image_config["guidance_scale"] = self.spec.guidance_scale
        if self.spec.image_guidance_scale is not None:
            image_config["image_guidance_scale"] = self.spec.image_guidance_scale
        if self.spec.cfg_norm is not None:
            image_config["cfg_norm"] = self.spec.cfg_norm
        if self.spec.cfg_interval is not None:
            image_config["cfg_interval"] = list(self.spec.cfg_interval)
        if self.spec.timestep_shift is not None:
            image_config["timestep_shift"] = self.spec.timestep_shift
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "modalities": ["image"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": item["prompt"]},
                        {
                            "type": "image_url",
                            "image_url": {"url": input_image_data_url(item)},
                        },
                    ],
                }
            ],
            "temperature": self.spec.temperature,
            "top_p": self.spec.top_p,
            "ignore_eos": self.spec.ignore_eos,
        }
        apply_chat_image_contract(
            payload,
            image_config,
            include_reference_aliases=uses_external_request_schema(self.spec),
        )
        if self.spec.extra_request_body:
            payload.update(self.spec.extra_request_body)
        return TaskRequest(endpoint=self.spec.endpoint, payload=payload, kind="openai_chat_json")
