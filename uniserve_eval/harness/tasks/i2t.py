from __future__ import annotations

from typing import Any

from .base import BenchmarkTask, TaskRequest


class I2TTask(BenchmarkTask):
    """Image-to-text, tri-wire.

    ``spec.wire`` selects the request shape so the same dataset, arrival
    engine, and stream metrics compare backends over each system's public API:

    * ``"native"`` — UniServe ``/generate`` SSE with ``constraint:"und_only"``
      and the image as ``input_image_b64``.
    * ``"openai_chat"`` — OpenAI chat completions SSE with the image as an
      ``image_url`` data-URI content part, streamed; measures true TTFT/ITL
      through the public chat endpoint.
    * ``"openai_chat_json"`` — the same chat payload, one non-streamed JSON
      response; E2E + token counts only. Diffusion-pipeline chat backends
      (vLLM-Omni) answer with one chat.completion JSON regardless of
      ``stream``, so this is their honest measurement.
    """

    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        max_tokens = int(item.get("max_tokens", self.spec.max_tokens or 512))
        image_b64 = item.get("input_image_b64")
        if self.spec.wire in ("openai_chat", "openai_chat_json"):
            streamed = self.spec.wire == "openai_chat"
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
            if streamed:
                payload["stream"] = True
                payload["stream_options"] = {"include_usage": True}
            if self.spec.extra_request_body:
                payload.update(self.spec.extra_request_body)
            return TaskRequest(endpoint=self.spec.endpoint, payload=payload, kind=self.spec.wire)

        payload = {
            "prompt": item["prompt"],
            "constraint": "und_only",
            "max_tokens": max_tokens,
            "temperature": self.spec.temperature,
            "input_image_b64": image_b64,
        }
        if self.spec.extra_request_body:
            payload.update(self.spec.extra_request_body)
        return TaskRequest(endpoint=self.spec.endpoint, payload=payload, kind="native_generate")
