"""Image encode and decode internals for ``ModelExecutor``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch

from uniserve_worker.contracts.forward_batch import (
    EncodeContext,
    EncodeRow,
    ForwardBatch,
    ForwardResult,
)
from uniserve_worker.contracts.forward_mode import ForwardMode, mode_for_op
from uniserve_worker.contracts.op_kinds import VAE_ENCODE, VIT_ENCODE
from uniserve_worker.contracts.outputs import (
    CommitOutput,
    EncodeOutput,
)
from uniserve_worker.foundation.errors import invalid_descriptor
from uniserve_worker.runtime.image_utils import pil_image_to_png_b64, to_uint8_image
from uniserve_worker.runtime.request_state import RequestState
from uniserve_worker.runtime.residency import encoder_handle_from_mm_hash

if TYPE_CHECKING:
    from uniserve_worker.contracts.forward_batch import ForwardBatch


# Decode token/position relays (device-resident sequence feedback)
# ---------------------
# Encode-step execution
# ---------------------


@torch.inference_mode()
def encode_result(
    fb: ForwardBatch,
    model: Any,
    *,
    image_stage: Any = None,
    row_indices: tuple[int, ...] | list[int] | None = None,
) -> ForwardResult:
    ops = fb.as_encode().ops
    rows = tuple(range(len(ops))) if row_indices is None else tuple(int(row) for row in row_indices)
    if len(rows) != len(ops):
        raise invalid_descriptor("encode row_indices must align with encode ops")
    outputs = [
        _coerce_encode_output(output)
        for output in run_encode_ops(model, ops, image_stage=image_stage)
    ]
    return ForwardResult(encode_outputs=dict(zip(rows, outputs, strict=True)))


def run_encode_ops(
    model: Any,
    ops: Sequence[Mapping[str, Any]],
    *,
    image_stage: Any = None,
) -> list[Any]:
    """Stage every encode op's inputs, then dispatch the model's neural encode.

    Op parsing, media decode, declared transforms, and device staging happen
    here — before the model is invoked — so encode entry points receive typed
    :class:`EncodeRow` values only.
    """
    rows = [prepare_encode_row(op, image_stage) for op in ops]
    encode_many = getattr(model, "encode_many", None)
    outputs = (
        list(encode_many(tuple(rows)))
        if callable(encode_many)
        else [_encode_one(model, row) for row in rows]
    )
    if len(outputs) != len(ops):
        raise invalid_descriptor(f"model returned {len(outputs)} encode outputs for {len(ops)} ops")
    return outputs


def prepare_encode_row(op: Mapping[str, Any], image_stage: Any) -> EncodeRow:
    """Parse one encode op and run the model's declared image input stage."""
    kind = str(op.get("kind"))
    if mode_for_op(kind) != ForwardMode.ENCODE:
        raise invalid_descriptor(f"unsupported encode op {op.get('kind')!r}")
    req_id = int(op["req_id"])
    cond_pos = op.get("cond_pos")
    temporal_index = None if cond_pos is None else int(cond_pos)
    new_block_ids = tuple(int(block) for block in (op.get("new_block_ids") or ()))
    raw_pos_range = op.get("pos_range")
    pos_range = (
        None if not raw_pos_range else (int(raw_pos_range[0]), int(raw_pos_range[1]))
    )
    image_b64 = op.get("image_b64")
    if image_b64:
        if image_stage is None:
            raise invalid_descriptor("model declares no image input transforms")
        prepared = image_stage.prepare(kind, str(image_b64))
        return EncodeRow(
            ctx=EncodeContext(
                req_id=req_id,
                kind=kind,
                handle=encoder_handle_from_mm_hash(op.get("mm_hash")),
                temporal_index=temporal_index,
                image_hw=prepared.image_hw,
                new_block_ids=new_block_ids,
                pos_range=pos_range,
            ),
            pixels=prepared.pixels,
            grid=prepared.grid,
        )
    cached_handle = op.get("image_in")
    if not isinstance(cached_handle, int) or isinstance(cached_handle, bool):
        raise invalid_descriptor("cached image encode requires an encoder handle")
    return EncodeRow(
        ctx=EncodeContext(
            req_id=req_id,
            kind=kind,
            handle=int(cached_handle),
            temporal_index=temporal_index,
            new_block_ids=new_block_ids,
            pos_range=pos_range,
        )
    )


def _encode_one(model: Any, row: EncodeRow) -> Any:
    if row.ctx.kind == VIT_ENCODE:
        return model.encode_image(row.pixels, row.grid, ctx=row.ctx)
    if row.ctx.kind == VAE_ENCODE:
        return model.encode_latents(row.pixels, row.grid, ctx=row.ctx)
    raise invalid_descriptor(f"unsupported encode op {row.ctx.kind!r}")


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


# ---------------------
# Materialize-step execution (image decode)
# ---------------------


@torch.inference_mode()
def commit_result(
    items: list[tuple[int, RequestState, Mapping[str, Any]]]
    | tuple[tuple[int, RequestState, Mapping[str, Any]], ...],
    model: Any,
    *,
    row_indices: tuple[int, ...] | list[int] | None = None,
) -> ForwardResult:
    rows = (
        tuple(range(len(items))) if row_indices is None else tuple(int(row) for row in row_indices)
    )
    if len(rows) != len(items):
        raise invalid_descriptor("commit row_indices must align with commit items")
    outputs = {
        int(row): model.decode_image(state.latent, req_id=int(req_id), state=state, op=op)
        for row, (req_id, state, op) in zip(rows, items, strict=True)
    }
    return ForwardResult(commit_outputs=outputs)


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
