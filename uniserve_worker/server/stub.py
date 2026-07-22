from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
from PIL import Image

from ..contracts.forward_batch import BatchPolicy, ForwardBatch
from ..contracts.forward_context import get_forward_context
from ..contracts.forward_mode import ForwardMode
from ..contracts.op_kinds import COMMIT_GEN, COMMIT_WRITEBACK, DECODE_UND, DENOISE_GEN, PREFILL_UND
from ..contracts.outputs import (
    CommitOutput,
    EncodeOutput,
    FlowOutput,
    ForwardOutput,
    TextTokenOutput,
)
from ..contracts.resource_plan import (
    CapsDescriptor,
    KvBlockResourcePolicy,
    LatentTokens,
    PerBranch,
    ResourcePlan,
)
from ..foundation.env import env_int
from ..foundation.sizing import DEFAULT_BLOCK_SIZE, DEFAULT_MAX_BATCH_OPS
from ..models.catalog import UniModelBase
from ..runtime.image_params import required_image_height, required_image_width
from ..runtime.image_utils import pil_image_to_png_b64
from ..worker.model import ModelWorker

STUB_EOS_TOKEN_ID = 151645
STUB_IMG_START_TOKEN_ID = 151670
STUB_IMAGE_TRIGGER_STEP = 2
STUB_TEXT_EOS_STEP = 8
STUB_NUM_BLOCKS = 4096
STUB_NUM_LAYERS = 28
STUB_SCRATCH_TOKENS = 1 << 20
STUB_MAX_LATENT_SIZE = 1024
STUB_LATENT_DOWNSAMPLE = 16
STUB_BYTES_PER_TOKEN = 57344
STUB_MAX_BATCH_OPS = DEFAULT_MAX_BATCH_OPS

__all__ = [
    "StubUniModel",
    "StubWorker",
]


def _synthetic_png_b64(width: int, height: int) -> str:
    """Placeholder gradient PNG for stub image commit outputs."""
    width = max(1, int(width))
    height = max(1, int(height))
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    image: np.ndarray = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = x
    image[:, :, 1] = y
    image[:, :, 2] = ((x.astype(np.uint16) + y.astype(np.uint16)) // 2).astype(np.uint8)
    return pil_image_to_png_b64(Image.fromarray(image))


class StubUniModel(UniModelBase):
    """Deterministic fake model for sim/stub worker paths."""

    architectures: tuple[str, ...] = ("UniServeStubForUnifiedGeneration",)
    supported_ops: tuple[str, ...] = (
        PREFILL_UND,
        DECODE_UND,
        DENOISE_GEN,
        COMMIT_GEN,
        COMMIT_WRITEBACK,
    )
    supported_controls: tuple[str, ...] = (
        "copy_blocks",
        "load_lora",
        "unload_lora",
        "free_encoder",
        "reset_prefix_cache",
    )
    adapter_mode = "none"
    resource_plan = ResourcePlan(
        kv_block=KvBlockResourcePolicy.PER_BLOCK,
        image_latent=LatentTokens(downsample=16),
        scratch=PerBranch(),
    )
    num_layers = STUB_NUM_LAYERS
    max_latent_size = STUB_MAX_LATENT_SIZE
    latent_downsample = STUB_LATENT_DOWNSAMPLE
    bytes_per_token = STUB_BYTES_PER_TOKEN
    max_batch_ops = STUB_MAX_BATCH_OPS

    def __init__(self, config=None) -> None:
        self.config = config
        self._die_after = env_int("UNISERVE_STUB_DIE_AFTER", default=0)
        self._executes = 0

    def load_weights(self, weights: Iterable[tuple[str, Any]]) -> set[str]:
        return {str(name) for name, _tensor in weights}

    def batch_policy(self) -> BatchPolicy:
        return BatchPolicy(max_batch_ops=self.max_batch_ops, supports_mixed_modes=True)

    def _caps_descriptor(
        self,
        *,
        block_size: int | None = None,
        kv_token_capacity: int | None = None,
    ) -> CapsDescriptor:
        resolved_block_size = DEFAULT_BLOCK_SIZE if block_size is None else int(block_size)
        num_blocks = (
            max(1, int(kv_token_capacity) // resolved_block_size)
            if kv_token_capacity
            else STUB_NUM_BLOCKS
        )
        return CapsDescriptor(
            block_size=resolved_block_size,
            num_blocks=num_blocks,
            num_layers=self.num_layers,
            scratch_capacity_tokens=STUB_SCRATCH_TOKENS,
            max_latent_size=self.max_latent_size,
            latent_downsample=self.latent_downsample,
            max_vae_grid_tokens=self.max_latent_size,
            commit_marker_tokens=2,
            gen_rope_advance=2,
            max_cfg_branches=3,
            bytes_per_token=self.bytes_per_token,
            max_batch_ops=self.max_batch_ops,
        )

    def forward(self, batch: ForwardBatch) -> list[ForwardOutput]:
        self._executes += 1
        if self._die_after and self._executes > self._die_after:
            print("[worker] fault injection: dying abruptly", flush=True)
            os._exit(1)
        handler = self._FORWARD_BY_MODE.get(batch.mode)
        if handler is None:
            return self._mixed(batch)
        return cast(list[ForwardOutput], handler(self, batch))

    def _mixed(self, batch: ForwardBatch) -> list[ForwardOutput]:
        outputs: list[ForwardOutput] = []
        for op in batch.ops:
            row = ForwardBatch.from_ops([op])
            handler = self._FORWARD_BY_MODE.get(row.mode)
            if handler is None:
                raise RuntimeError(f"unsupported stub forward mode {row.mode}")
            outputs.extend(cast(list[ForwardOutput], handler(self, row)))
        return outputs

    def _text(self, batch: ForwardBatch) -> list[TextTokenOutput]:
        out = []
        for req_id in batch.as_text().req_ids:
            state = get_forward_context().request_states.get(int(req_id))
            n = state.kv_length("stub_emitted")
            if state.image and n == STUB_IMAGE_TRIGGER_STEP:
                tok = STUB_IMG_START_TOKEN_ID
            elif n >= STUB_TEXT_EOS_STEP:
                tok = STUB_EOS_TOKEN_ID
            else:
                tok = 1000 + (req_id * 7 + n) % 5000
            state.set_kv_length(n + 1, "stub_emitted")
            out.append(TextTokenOutput(req_id=req_id, sampled_token_id=tok))
        return out

    def _denoise(self, batch: ForwardBatch) -> list[FlowOutput]:
        view = batch.as_denoise()
        out = []
        for req_id in view.req_ids:
            state = get_forward_context().request_states.get(int(req_id))
            step = int(state.schedule_cursor) + 1
            state.schedule_cursor = step
            total = int((state.image or {}).get("steps", 50) or 50)
            out.append(FlowOutput(req_id=req_id, denoise_done=step >= total, num_steps_done=step))
        return out

    def _commit(self, batch: ForwardBatch) -> list[CommitOutput]:
        out = []
        for req_id in batch.as_commit().req_ids:
            state = get_forward_context().request_states.get(int(req_id))
            state.schedule_cursor = 0
            image = state.image or {}
            hw = (required_image_height(image), required_image_width(image))
            png = _synthetic_png_b64(hw[1], hw[0])
            out.append(CommitOutput(req_id=req_id, image_png_b64=png, image_hw=hw))
        return out

    def _encode(self, batch: ForwardBatch) -> list[EncodeOutput]:
        view = batch.as_encode()
        out = []
        for req_id, mm_hash in zip(view.req_ids, view.mm_hashes):
            handle = int(mm_hash or 0) * 0x9E3779B1 | 1
            out.append(EncodeOutput(req_id=req_id, encoder_handle=handle))
        return out

    # Forward-mode dispatch table. EXTEND/DECODE/VERIFY_DRAFT share ``_text``.
    _FORWARD_BY_MODE = {
        ForwardMode.EXTEND: _text,
        ForwardMode.DECODE: _text,
        ForwardMode.VERIFY_DRAFT: _text,
        ForwardMode.DENOISE: _denoise,
        ForwardMode.COMMIT: _commit,
        ForwardMode.ENCODE: _encode,
    }


class StubWorker(ModelWorker):
    """GPU-free model worker for protocol and serving tests."""

    supported_ops = StubUniModel.supported_ops
    supported_controls = StubUniModel.supported_controls
    adapter_mode = StubUniModel.adapter_mode
    resource_plan = StubUniModel.resource_plan

    def __init__(
        self,
        block_size: int = DEFAULT_BLOCK_SIZE,
        *,
        pipeline_depth: int = 1,
    ) -> None:
        model = StubUniModel()
        super().__init__(
            model,
            allowed_ops=frozenset(model.supported_ops),
            pipeline_depth=pipeline_depth,
            block_size=block_size,
            simulation=True,
        )
