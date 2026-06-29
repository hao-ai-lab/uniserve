from __future__ import annotations

from typing import Any

from .base import BenchmarkTask, TaskRequest


class TextTask(BenchmarkTask):
    """LLM serving via OpenAI ``/v1/chat/completions``.

    Mirrors ``refs/sglang`` ``async_request_openai_chat_completions``: a single
    user message, ``max_completion_tokens`` from the dataset row, ``temperature``
    and ``ignore_eos`` defaults, streaming on. ``stream_options.include_usage`` is
    requested so the server reports ``completion_tokens`` for accurate
    output-token accounting (UniServe honours it; SGLang's metrics fall back to
    the requested length the same way when usage is absent).
    """

    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        messages = item.get("messages") or [{"role": "user", "content": item["prompt"]}]
        output_len = item.get("output_len")
        if output_len is None:
            output_len = self.spec.max_tokens
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": self.spec.temperature,
            "ignore_eos": self.spec.ignore_eos,
        }
        if output_len is not None:
            payload["max_completion_tokens"] = int(output_len)
        if self.spec.top_p < 1.0:
            payload["top_p"] = self.spec.top_p
        if self.spec.extra_request_body:
            payload.update(self.spec.extra_request_body)
        return TaskRequest(endpoint=self.spec.endpoint, payload=payload, kind="openai_chat")
