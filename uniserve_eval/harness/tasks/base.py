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


def uses_external_request_schema(spec: BenchmarkSpec) -> bool:
    return spec.request_schema != "uniserve"


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
    if spec.chat_template_kwargs and spec.request_schema == "sglang":
        payload["chat_template_kwargs"] = dict(spec.chat_template_kwargs)


# Chat image parameters that served backends read from the request root, keyed
# by their declared name. Width and height also carry the combined ``size``
# spelling and are handled alongside these.
_CHAT_IMAGE_ROOT_NAMES: dict[str, tuple[str, ...]] = {
    "steps": ("num_inference_steps",),
    "seed": ("seed",),
    "num_images": ("num_outputs_per_prompt",),
    "guidance_scale": ("guidance_scale", "cfg_scale", "cfg_text_scale"),
    "image_guidance_scale": ("image_guidance_scale", "img_cfg_scale", "cfg_img_scale"),
    "cfg_norm": ("cfg_norm", "cfg_renorm_type"),
    "cfg_interval": ("cfg_interval",),
    "timestep_shift": ("timestep_shift",),
    "think": ("think",),
    "t_eps": ("t_eps",),
}


def apply_chat_image_contract(
    payload: dict[str, Any],
    image_config: dict[str, Any],
    *,
    root_parameters: dict[str, Any] | None = None,
    include_reference_aliases: bool = False,
) -> None:
    """Carry one declared image parameter set through the canonical chat field.

    Reference runtimes that lack the canonical object may additionally receive
    their request-root aliases. UniServe requests never carry those aliases.
    """

    payload["image_config"] = image_config
    if not include_reference_aliases:
        return
    declared = {**image_config, **(root_parameters or {})}
    width = declared.get("width")
    height = declared.get("height")
    if width is not None and height is not None:
        payload["width"] = int(width)
        payload["height"] = int(height)
        payload["size"] = f"{int(width)}x{int(height)}"
    for name, root_names in _CHAT_IMAGE_ROOT_NAMES.items():
        if name not in declared:
            continue
        value = declared[name]
        for root_name in root_names:
            payload[root_name] = list(value) if isinstance(value, list) else value


def input_image_data_url(item: dict[str, Any]) -> str:
    image_b64 = item.get("input_image_b64")
    if not isinstance(image_b64, str) or not image_b64:
        raise ValueError("input image row has no base64 payload")
    mime = item.get("input_image_mime", "image/png")
    if not isinstance(mime, str) or not mime.startswith("image/"):
        raise ValueError("input image row has an invalid MIME type")
    return f"data:{mime};base64,{image_b64}"
