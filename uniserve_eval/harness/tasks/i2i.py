from __future__ import annotations

from typing import Any

from .base import BenchmarkTask, TaskRequest


class I2ITask(BenchmarkTask):
    """Image-to-image (editing) via the native ``/generate`` SSE endpoint.

    The source image is sent as base64 under ``input_image_b64`` and the edit
    instruction as ``prompt``. ``mode`` defaults to ``"image"`` to match the
    documented API contract.

    NOTE: this assumes only the API endpoint, not the model. The current server
    ``build()`` path drops input images for ``mode:"image"`` (only
    ``mode:"understand"`` consumes them); ``spec.i2i_mode`` lets a caller switch
    to ``"understand"`` if the server is changed to condition image generation on
    the input there. Either way the harness sends a well-formed i2i request.
    """

    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        image: dict[str, Any] = {"max_images": item.get("max_images", self.spec.max_images)}
        width = item.get("width", self.spec.width)
        height = item.get("height", self.spec.height)
        if width is not None and height is not None:
            image["width"] = int(width)
            image["height"] = int(height)
        if self.spec.steps is not None:
            image["steps"] = int(self.spec.steps)
        payload: dict[str, Any] = {
            "prompt": item["prompt"],
            "mode": self.spec.i2i_mode,
            "input_image_b64": item.get("input_image_b64"),
            "image": image,
        }
        if self.spec.extra_request_body:
            payload.update(self.spec.extra_request_body)
        return TaskRequest(endpoint=self.spec.endpoint, payload=payload, kind="native_generate")
