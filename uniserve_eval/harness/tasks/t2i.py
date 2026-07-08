from __future__ import annotations

from typing import Any

from .base import BenchmarkTask, TaskRequest


class T2ITask(BenchmarkTask):
    """Text-to-image, dual-wire.

    * ``"images_generations"`` — OpenAI-style ``/v1/images/generations``
      (non-streaming). Request body carries ``prompt`` plus optional
      ``size="WxH"`` and a seed, with the step count under both accepted
      spellings — ``steps`` (UniServe) and ``num_inference_steps``
      (vLLM-Omni) — since each server ignores the other's field. The
      response is a single JSON ``{"data": [{"b64_json", ...}]}`` object
      either way.
    * ``"openai_chat_json"`` — OpenAI chat completions with
      ``modalities: ["image"]`` and ``image_config`` (the official LightLLM
      V2 chat shape), one non-streamed JSON response; generated images
      arrive in ``message.images``.
    """

    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        width = item.get("width", self.spec.width)
        height = item.get("height", self.spec.height)
        steps = item.get("steps", self.spec.steps)
        seed = item.get("seed", self.spec.seed)

        if self.spec.wire == "openai_chat_json":
            image_config: dict[str, Any] = {}
            if width is not None and height is not None:
                image_config["width"] = int(width)
                image_config["height"] = int(height)
            if steps is not None:
                image_config["steps"] = int(steps)
            if seed is not None:
                image_config["seed"] = int(seed)
            payload: dict[str, Any] = {
                "model": self.spec.model,
                "modalities": ["image"],
                "messages": [{"role": "user", "content": item["prompt"]}],
                "image_config": image_config,
            }
            if self.spec.extra_request_body:
                payload.update(self.spec.extra_request_body)
            return TaskRequest(
                endpoint=self.spec.endpoint, payload=payload, kind="openai_chat_json"
            )

        payload = {"prompt": item["prompt"]}
        if width is not None and height is not None:
            payload["size"] = f"{width}x{height}"
        if steps is not None:
            payload["steps"] = int(steps)
            payload["num_inference_steps"] = int(steps)
        if seed is not None:
            payload["seed"] = int(seed)
        if self.spec.extra_request_body:
            payload.update(self.spec.extra_request_body)
        return TaskRequest(endpoint=self.spec.endpoint, payload=payload, kind="images_generations")
