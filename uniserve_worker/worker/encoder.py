"""Worker implementation for model encoding operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts.batches import UniForwardBatch
from ..contracts.model_protocols import ModelHooks
from ..contracts.op_kinds import VAE_ENCODE, VIT_ENCODE
from ..execution.encode_driver import EncodeDriver
from ..foundation.errors import capability_mismatch
from ..foundation.sizing import DEFAULT_BLOCK_SIZE
from ..runtime.request_state import RequestStateTable
from .protocol import (
    BaseWorker,
    ResultPolicy,
    model_free_capabilities,
)


class EncoderWorker(BaseWorker):
    """Runs the encode capabilities of one loaded multimodal model."""

    def __init__(
        self,
        model: Any,
        *,
        allowed_ops: frozenset[str] | None = None,
        pipeline_depth: int = 1,
        result_policy: ResultPolicy = ResultPolicy.DEFER_WHEN_AVAILABLE,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        super().__init__(block_size=block_size)
        if not isinstance(model, ModelHooks):
            raise capability_mismatch("encoder worker model must inherit ModelHooks")
        self.model = model
        self.encode_driver = EncodeDriver()
        self.request_states = RequestStateTable()
        supported_ops = self._supported_encode_ops()
        if not supported_ops:
            raise capability_mismatch("encoder worker requires encode_image() or encode_latents()")
        declared_capabilities = model_free_capabilities(
            block_size=self.block_size,
            supported_ops=supported_ops,
            supported_controls=("free_encoder",),
            num_layers=int(getattr(model, "num_layers", 1) or 1),
            max_latent_size=int(getattr(model, "max_latent_size", 0) or 0),
            latent_downsample=int(getattr(model, "latent_downsample", 1) or 1),
            encoder_cache_budget=int(getattr(model, "encoder_cache_budget", 0) or 0),
        )
        self._compile_contract(
            declared_capabilities,
            allowed_ops=(allowed_ops if allowed_ops is not None else frozenset(supported_ops)),
            pipeline_depth=pipeline_depth,
            result_policy=result_policy,
        )

    def execute(
        self,
        batch: Mapping[str, Any],
        *,
        defer_text_cpu_results: bool = False,
    ) -> dict[str, Any]:
        del defer_text_cpu_results
        for new_request in batch.get("new_reqs") or ():
            self._register_request(new_request)
        operations = list(batch.get("ops") or [])
        if not operations:
            return {"step_id": batch.get("step_id"), "per_seq": []}
        forward_batch = UniForwardBatch.from_ops(operations)
        outputs = self.encode_driver.step(forward_batch, self.model)
        return {
            "step_id": batch.get("step_id"),
            "per_seq": [output.to_seq_result() for output in outputs],
        }

    def drop_request(self, request_id: int) -> None:
        self.model.drop_request(int(request_id))
        self.request_states.drop(int(request_id))

    def free_encoder(self, handles: Any) -> None:
        self.model.free_encoder(handles)

    def _register_request(self, new_request: Mapping[str, Any]) -> None:
        request_id = int(new_request["req_id"])
        state = self.request_states.create_or_update(
            request_id,
            dict(new_request),
        )
        self.model.on_new_request(request_id, state)

    def _supported_encode_ops(self) -> tuple[str, ...]:
        supported_ops: list[str] = []
        encode_image = getattr(type(self.model), "encode_image", None)
        if encode_image is not None and encode_image is not ModelHooks.encode_image:
            supported_ops.append(VIT_ENCODE)
        encode_latents = getattr(type(self.model), "encode_latents", None)
        if encode_latents is not None and encode_latents is not ModelHooks.encode_latents:
            supported_ops.append(VAE_ENCODE)
        return tuple(supported_ops)
