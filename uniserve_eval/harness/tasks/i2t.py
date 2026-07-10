from __future__ import annotations

from typing import Any

from .base import BenchmarkTask, RequestKind, TaskRequest


class I2TTask(BenchmarkTask):
    """Image-to-text through chat completions."""

    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        max_tokens = int(item.get("max_tokens", self.spec.max_tokens or 512))
        image_b64 = item.get("input_image_b64")
        streamed = self.spec.wire == "openai_chat"
        kind: RequestKind = "openai_chat" if streamed else "openai_chat_json"
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
            "top_p": self.spec.top_p,
            "ignore_eos": self.spec.ignore_eos,
            "max_completion_tokens": max_tokens,
        }
        if streamed:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        if self.spec.extra_request_body:
            payload.update(self.spec.extra_request_body)
        return TaskRequest(endpoint=self.spec.endpoint, payload=payload, kind=kind)
