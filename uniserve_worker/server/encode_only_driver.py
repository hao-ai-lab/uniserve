"""Encoder worker driver.

A peeled stage that runs only the vision encoders (``vit_encode`` / ``vae_encode``)
and hands embeddings to a downstream Prefill/Full worker by handle. Reuses the
shared :class:`~uniserve_worker.execution.encode_driver.EncodeDriver` (a
self-contained sub-driver decoupled from the LM forward); this module is the
worker shell: encode-only caps + encode dispatch.

Embeddings stay worker-resident (the model's encoder cache mints the
``encoder_handle``); only the scalar handle + ``num_tokens`` cross the control
plane, and the downstream worker pulls the embedding by handle on the data plane.
``free_encoder`` reclaims a cached embedding.

``build_encode_only_driver`` loads the model through the shared loader and wraps
it. Loading vision-only weights (skipping the LM) is a model-specific memory
optimization; correctness comes from the encode-only op subset.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from ..contracts.batches import UniForwardBatch
from ..contracts.model_protocols import ModelHooks
from ..contracts.op_kinds import VAE_ENCODE, VIT_ENCODE
from ..execution.encode_driver import EncodeDriver
from ..foundation.errors import capability_mismatch
from ..foundation.sizing import DEFAULT_BLOCK_SIZE
from ..runtime.request_state import RequestStateTable
from .base_driver import BaseWorkerDriver

__all__ = ["EncodeOnlyDriver", "build_encode_only_driver"]

logger = logging.getLogger(__name__)


class EncodeOnlyDriver(BaseWorkerDriver):
    """WorkerDriver running only the model's vision encoders."""

    def __init__(self, model: Any, *, block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        super().__init__(block_size=block_size)
        if not isinstance(model, ModelHooks):
            raise capability_mismatch("encoder worker model must inherit ModelHooks")
        self.model = model
        self.encode_driver = EncodeDriver()
        self.request_states = RequestStateTable()
        supported = self._encode_ops()
        if not supported:
            raise capability_mismatch(
                "encoder worker requires a model implementing encode_image() and/or "
                "encode_latents()"
            )
        self._caps = self._build_model_free_caps(
            block_size=self.block_size,
            supported_ops=supported,
            supported_controls=("free_encoder",),
            num_layers=int(getattr(model, "num_layers", 1) or 1),
            max_latent_size=int(getattr(model, "max_latent_size", 0) or 0),
            latent_downsample=int(getattr(model, "latent_downsample", 1) or 1),
            encoder_cache_budget=int(getattr(model, "encoder_cache_budget", 0) or 0),
        )

    def execute(
        self,
        batch: Mapping[str, Any],
        *,
        defer_text_cpu_results: bool = False,
    ) -> dict[str, Any]:
        del defer_text_cpu_results
        for nr in batch.get("new_reqs") or ():
            self._register(nr)
        ops = list(batch.get("ops") or [])
        if not ops:
            return {"step_id": batch.get("step_id"), "per_seq": []}
        fb = UniForwardBatch.from_ops(ops)
        outputs = self.encode_driver.step(fb, self.model)
        return {
            "step_id": batch.get("step_id"),
            "per_seq": [out.to_seq_result() for out in outputs],
        }

    def drop_request(self, req_id: int) -> None:
        self.model.drop_request(int(req_id))
        self.request_states.drop(int(req_id))

    def free_encoder(self, handles: Any) -> None:
        self.model.free_encoder(handles)

    def _register(self, nr: Mapping[str, Any]) -> None:
        req_id = int(nr["req_id"])
        state = self.request_states.create_or_update(req_id, dict(nr))
        self.model.on_new_request(req_id, state)

    def _encode_ops(self) -> tuple[str, ...]:
        ops: list[str] = []
        encode_image = getattr(type(self.model), "encode_image", None)
        if encode_image is not None and encode_image is not ModelHooks.encode_image:
            ops.append(VIT_ENCODE)
        encode_latents = getattr(type(self.model), "encode_latents", None)
        if encode_latents is not None and encode_latents is not ModelHooks.encode_latents:
            ops.append(VAE_ENCODE)
        return tuple(ops)


def build_encode_only_driver(args: Any) -> EncodeOnlyDriver:
    # Reuse the shared loader (LM + vision); the encode-only op subset is what
    # makes this an encoder worker, not which weights are resident.
    from .runner_driver import load_runner_engine

    runner = load_runner_engine(
        args.model,
        device=args.device,
        attention_backend=getattr(args, "attention_backend", None),
        kv_token_capacity=getattr(args, "kv_token_capacity", None),
        block_size=int(getattr(args, "block_size", DEFAULT_BLOCK_SIZE)),
    )
    return EncodeOnlyDriver(runner.model, block_size=int(getattr(args, "block_size", DEFAULT_BLOCK_SIZE)))
