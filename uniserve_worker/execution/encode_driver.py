"""Shared encode dispatch for multimodal input encoders."""
from __future__ import annotations

from typing import Any, Mapping

import torch

from ..contracts.batches import UniForwardBatch
from ..contracts.forward_mode import ForwardMode, mode_for_op
from ..contracts.op_kinds import VAE_ENCODE, VIT_ENCODE
from ..contracts.outputs import EncodeOutput
from ..foundation.errors import invalid_descriptor
from .forward.result import ForwardResult

__all__ = [
    'EncodeDriver',
]


class EncodeDriver:
    """Run model-provided image/text encoder primitives for ENCODE ops."""

    @torch.inference_mode()
    def step(self, fb: UniForwardBatch, model: Any) -> list[EncodeOutput]:
        # The encode view validates req_id/kind/mm_hash once; ``ops`` is the raw
        # op mapping retained so the model encode hooks still receive their
        # unparsed pixel/grid payloads.
        ops = fb.as_encode().ops
        return self._run_many(model, ops)

    @torch.inference_mode()
    def forward_result(
        self,
        fb: UniForwardBatch,
        model: Any,
        *,
        row_indices: tuple[int, ...] | list[int] | None = None,
    ) -> ForwardResult:
        ops = fb.as_encode().ops
        rows = tuple(range(len(ops))) if row_indices is None else tuple(int(row) for row in row_indices)
        if len(rows) != len(ops):
            raise invalid_descriptor("encode row_indices must align with encode ops")
        outputs = dict(zip(rows, self._run_many(model, ops), strict=True))
        return ForwardResult(encode_outputs=outputs)

    def _run_many(self, model: Any, ops: tuple[Mapping[str, Any], ...]) -> list[EncodeOutput]:
        encode_many = getattr(model, "encode_many", None)
        if callable(encode_many):
            outputs = list(encode_many(ops))
        else:
            outputs = [self._run_one(model, op) for op in ops]
        if len(outputs) != len(ops):
            raise invalid_descriptor(
                f"model returned {len(outputs)} encode outputs for {len(ops)} ops"
            )
        return [_coerce_encode_output(output) for output in outputs]

    def _run_one(self, model: Any, op: Mapping[str, Any]) -> Any:
        kind = str(op.get("kind"))
        if mode_for_op(kind) != ForwardMode.ENCODE:
            raise invalid_descriptor(f"unsupported encode op {op.get('kind')!r}")
        if kind == VIT_ENCODE:
            return model.encode_image(op.get("pixels"), op.get("grid"), op=op)
        if kind == VAE_ENCODE:
            return model.encode_latents(op.get("pixels"), op.get("grid"), op=op)
        raise invalid_descriptor(f"unsupported encode op {op.get('kind')!r}")


def _coerce_encode_output(output: Mapping[str, Any] | EncodeOutput) -> EncodeOutput:
    # The model->driver boundary stays flexible: encode adapters may return either
    # a typed ``EncodeOutput`` or a wire mapping. The driver normalizes both to
    # ``EncodeOutput`` so the runner consumes one uniform ``ForwardOutput`` list.
    if isinstance(output, EncodeOutput):
        return output
    if not isinstance(output, Mapping):
        raise invalid_descriptor("encode adapter outputs must be mappings")
    req_id = output.get("req_id")
    handle = output.get("encoder_handle")
    if not isinstance(req_id, int) or isinstance(req_id, bool):
        raise invalid_descriptor("encode output req_id must be an integer")
    if not isinstance(handle, int) or isinstance(handle, bool):
        raise invalid_descriptor("encode output encoder_handle must be an integer")
    return EncodeOutput(
        req_id=int(req_id),
        encoder_handle=int(handle),
        num_tokens=_optional_int(output.get("num_tokens")),
        image_hw=_image_hw(output.get("image_hw")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise invalid_descriptor("encode output num_tokens must be an integer")
    return int(value)


def _image_hw(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 2
    ):
        raise invalid_descriptor("encode output image_hw must be [height, width]")
    return (int(value[0]), int(value[1]))
