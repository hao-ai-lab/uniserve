"""Image-decode egress driver for ``commit_gen`` ops.

``step()`` decodes a generated image (``model.decode_image``), runs the sampler
on any returned logits, and encodes the result for the wire. It is the diffusion
egress path; it holds no KV-cache logic.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch

from ..contracts.outputs import CommitOutput
from ..foundation.errors import invalid_descriptor
from ..runtime.image_utils import pil_image_to_png_b64, to_uint8_image
from ..runtime.request_state import RequestState
from .text_driver import sample_logits_result

__all__ = [
    'ImageDecodeDriver',
]


class ImageDecodeDriver:
    @torch.inference_mode()
    def step(self, req_id: int, state: RequestState, model: Any, op: Mapping[str, Any]) -> CommitOutput:
        result = model.decode_image(state.latent, req_id=int(req_id), state=state, op=op)
        out = dict(result) if isinstance(result, Mapping) else _image_to_result(int(req_id), result)
        logits = out.pop("logits", None)
        if logits is not None:
            sampled = sample_logits_result(req_id=int(req_id), state=state, logits=logits, op=op)
            sampled.pop("req_id", None)
            out.update(sampled)
        return _commit_output_from_dict(int(req_id), out)


def _commit_output_from_dict(req_id: int, out: Mapping[str, Any]) -> CommitOutput:
    image_hw = out.get("image_hw")
    return CommitOutput(
        req_id=req_id,
        image_png_b64=out.get("image_png_b64"),
        image_hw=(int(image_hw[0]), int(image_hw[1])) if image_hw is not None else None,
        sampled_token_id=out.get("sampled_token_id"),
        sampled_logprob=out.get("sampled_logprob"),
        top_logprobs=out.get("top_logprobs"),
        num_tokens=out.get("num_tokens"),
        locator=out.get("locator"),
    )


def _image_to_result(req_id: int, image: Any) -> dict[str, Any]:
    save = getattr(image, "save", None)
    if callable(save):
        width, height = getattr(image, "size", (None, None))
        out = {"req_id": req_id, "image_png_b64": pil_image_to_png_b64(image)}
        if width is not None and height is not None:
            out["image_hw"] = [int(height), int(width)]
        return out
    if isinstance(image, torch.Tensor):
        if image.ndim not in (3, 4):
            raise invalid_descriptor("decode_image tensor output must be CHW or NCHW")
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - dependency failure is environment-specific.
            raise invalid_descriptor("PIL is required to encode tensor image outputs") from exc
        # decode_image returns already-normalized [0, 1] image space (not the
        # diffusion [-1, 1] latent convention), so decode against that range.
        pil = Image.fromarray(to_uint8_image(image, value_range=(0.0, 1.0)))
        return _image_to_result(req_id, pil)
    raise invalid_descriptor("decode_image must return a mapping, PIL image, or image tensor")
