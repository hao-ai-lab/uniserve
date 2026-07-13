from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..spec import BenchmarkSpec

RequestKind = Literal["openai_chat", "openai_chat_json", "images_generations"]


@dataclass(frozen=True)
class TaskRequest:
    endpoint: str
    payload: dict[str, Any]
    # Wire shape, dispatched by ``core.client.send_request``.
    kind: RequestKind
    semantic_task: str | None = None


class BenchmarkTask:
    def __init__(self, spec: BenchmarkSpec) -> None:
        self.spec = spec

    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        raise NotImplementedError


def apply_text_sampling_contract(payload: dict[str, Any], spec: BenchmarkSpec) -> None:
    payload["temperature"] = spec.temperature
    payload["top_p"] = spec.top_p
    payload["ignore_eos"] = spec.ignore_eos
    for key in (
        "top_k",
        "min_p",
        "repetition_penalty",
        "frequency_penalty",
        "presence_penalty",
    ):
        value = getattr(spec, key)
        if value is not None:
            payload[key] = value
    if spec.sampling_seed is not None:
        payload["seed"] = spec.sampling_seed
    if spec.chat_template_kwargs:
        payload["chat_template_kwargs"] = dict(spec.chat_template_kwargs)


def input_image_data_url(item: dict[str, Any]) -> str:
    image_b64 = item.get("input_image_b64")
    if not isinstance(image_b64, str) or not image_b64:
        raise ValueError("input image row has no base64 payload")
    mime = item.get("input_image_mime", "image/png")
    if not isinstance(mime, str) or not mime.startswith("image/"):
        raise ValueError("input image row has an invalid MIME type")
    return f"data:{mime};base64,{image_b64}"
