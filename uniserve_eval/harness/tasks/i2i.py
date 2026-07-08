from __future__ import annotations

from typing import Any

from .base import BenchmarkTask, TaskRequest


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
                            "image_url": {
                                "url": f"data:image/png;base64,{item.get('input_image_b64')}"
                            },
                        },
                    ],
                }
            ],
            "image_config": image_config,
        }
        if self.spec.extra_request_body:
            payload.update(self.spec.extra_request_body)
        return TaskRequest(endpoint=self.spec.endpoint, payload=payload, kind="openai_chat_json")
