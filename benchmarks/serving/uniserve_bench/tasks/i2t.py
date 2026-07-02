from __future__ import annotations

from typing import Any

from .base import BenchmarkTask, TaskRequest


class I2TTask(BenchmarkTask):
    """Image understanding (image → text), dual-wire.

    ``spec.i2t_wire`` selects the request shape so the same dataset, arrival
    engine, and stream metrics compare backends over each system's public API:

    * ``"native"`` — UniServe ``/generate`` SSE with ``mode:"understand"`` and
      the image as ``input_image_b64``.
    * ``"openai_chat"`` — OpenAI chat completions SSE with the image as an
      ``image_url`` data URI content part (vLLM-Omni's SenseNova i2t shape,
      ``modalities: ["text"]``).
    """

    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        max_tokens = int(item.get("max_tokens", self.spec.max_tokens or 512))
        image_b64 = item.get("input_image_b64")
        if self.spec.i2t_wire == "openai_chat":
            # Diffusion-pipeline chat backends (vLLM-Omni) answer with one
            # non-streamed chat.completion JSON regardless of ``stream``, so
            # this wire measures E2E + tokens (no TTFT/ITL decomposition).
            payload: dict[str, Any] = {
                "model": self.spec.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": item["prompt"]},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
                "modalities": ["text"],
                "temperature": self.spec.temperature,
                "max_tokens": max_tokens,
            }
            if self.spec.extra_request_body:
                payload.update(self.spec.extra_request_body)
            endpoint = (
                self.spec.endpoint
                if self.spec.endpoint != "/generate"
                else "/v1/chat/completions"
            )
            return TaskRequest(endpoint=endpoint, payload=payload, kind="openai_chat_json")

        payload = {
            "prompt": item["prompt"],
            "mode": "understand",
            "max_tokens": max_tokens,
            "temperature": self.spec.temperature,
            "input_image_b64": image_b64,
        }
        if self.spec.extra_request_body:
            payload.update(self.spec.extra_request_body)
        return TaskRequest(endpoint=self.spec.endpoint, payload=payload, kind="native_generate")
