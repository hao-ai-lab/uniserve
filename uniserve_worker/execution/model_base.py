"""Concrete contract glue shared by the registered model entries.

Provides the impl-side ``UniModelBase`` mixin that assembles a
:class:`~uniserve_worker.contracts.caps.Caps` / ``BatchPolicy`` from a
model-supplied :class:`~uniserve_worker.contracts.resource_plan.CapsDescriptor`
and owns the KV store-dtype glue. Lives in the execution layer, separate from
the pure ``contracts.model_protocols`` abstractions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..contracts.batch_policy import BatchPolicy
from ..contracts.caps import Caps, ExecutionConstraints
from ..contracts.model_protocols import ModelHooks
from ..foundation.runtime_config import get_execution_config
from ..nn.quant import get_current_kv_cache_dtype, kv_store_dtype_name, resolve_kv_store_dtype
from ..nn.quant.kv_cache import KV_CACHE_NO_OVERRIDE_SENTINELS
from ..runtime.compile import TorchCompileConfig, compile_model_pieces

if TYPE_CHECKING:
    import torch

    from ..contracts.resource_plan import CapsDescriptor

__all__ = ["UniModelBase"]

logger = logging.getLogger(__name__)


class UniModelBase(ModelHooks):
    """Concrete glue shared by the registered model entries.

    This is a composition-friendly *mixin* (it carries no ``__init__`` and no
    real superclass), so a model keeps its own base — ``nn.Module``,
    ``TextImageDenoiseOps``, or plain ``object`` — and lists ``UniModelBase``
    first to pick up the shared implementations. It complements the
    :class:`~uniserve_worker.contracts.model_protocols.UniModel` structural
    Protocol rather than duplicating it.

    It owns the duplicated contract glue:

    * the KV store-dtype helpers ``_kv_store_dtype_for`` / ``_kv_dtype_name_for``
      (require an instance ``kv_cache_dtype`` attribute);
    * a single ``batch_policy()`` and ``caps()`` driven by ``_caps_descriptor``,
      both reading ``max_batch_ops`` from the same descriptor so the two stay aligned.

    The model-specific ``__init__`` bodies (cuda-graph runners, LoRA/enc_store,
    dual-device pools) stay in the subclasses; only the caps/batch_policy/
    kv-dtype glue is shared.
    """

    kv_cache_dtype: Any

    # Config-gated piecewise torch.compile is applied at most once per model
    # instance; subclasses call this after weights/residency are ready.
    _torch_compile_applied: bool = False

    def _kv_store_dtype_for(self, compute_dtype: "torch.dtype") -> "torch.dtype":
        return resolve_kv_store_dtype(compute_dtype, self.kv_cache_dtype)

    def _kv_dtype_name_for(self, compute_dtype: "torch.dtype") -> str:
        return kv_store_dtype_name(self._kv_store_dtype_for(compute_dtype))

    def _requested_kv_cache_dtype_for(self, config: Any | None) -> str | None:
        value = get_execution_config().kv_cache_dtype
        if value is not None and str(value).lower() not in KV_CACHE_NO_OVERRIDE_SENTINELS | {
            "null"
        }:
            return value
        return get_current_kv_cache_dtype(config)

    def _text_decode_graph_query_geometry_from(self, attn: Any) -> tuple[int, float, "torch.dtype"]:
        """Query-side decode-graph geometry read off one attention module.

        The shared extraction behind each model's
        ``text_decode_graph_query_geometry`` hook: the module's (tensor-parallel
        local) head count, its softmax scale, and the query dtype. ``q_norm``'s
        weight stays in compute dtype even under weight quantization, so it is
        the faithful dtype of the query tensor the decode kernel sees.
        """
        scale = getattr(attn, "scale", None)
        if scale is None:
            scale = attn.scaling
        return int(attn.num_heads), float(scale), attn.q_norm.weight.dtype

    def _maybe_compile_piecewise(self) -> None:
        if self._torch_compile_applied:
            return
        cfg = TorchCompileConfig.from_runtime_config()
        if not cfg.enabled:
            return
        report = compile_model_pieces(self, config=cfg)
        self._torch_compile_applied = True
        if report.compiled:
            logger.info(
                "enabled %s model-stack torch.compile pieces count=%s",
                type(self).__name__,
                report.compiled,
            )

    def _caps_descriptor(
        self,
        *,
        block_size: int | None = None,
        kv_token_capacity: int | None = None,
    ) -> "CapsDescriptor":
        raise NotImplementedError

    def batch_policy(self) -> BatchPolicy:
        descriptor = self._caps_descriptor()
        # A runner-backed model lets the system group a heterogeneous (und+gen)
        # batch by mode and drive each mode through its system driver; thin text
        # additionally fuses an all-text extend+decode window. No per-model opt-out.
        return BatchPolicy(
            max_batch_ops=descriptor.max_batch_ops,
            supports_mixed_modes=True,
        )

    def caps(
        self,
        *,
        block_size: int | None = None,
        kv_token_capacity: int | None = None,
    ) -> Caps:
        descriptor = self._caps_descriptor(
            block_size=block_size,
            kv_token_capacity=kv_token_capacity,
        )
        return Caps(
            block_size=descriptor.block_size,
            num_blocks=descriptor.num_blocks,
            num_layers=descriptor.num_layers,
            scratch_capacity_tokens=descriptor.scratch_capacity_tokens,
            supported_ops=tuple(self.supported_ops),
            max_latent_size=descriptor.max_latent_size,
            latent_downsample=descriptor.latent_downsample,
            bytes_per_token=descriptor.bytes_per_token,
            supported_controls=tuple(self.supported_controls),
            adapter_mode=self.adapter_mode,
            execution_constraints=ExecutionConstraints(
                max_batch_ops=descriptor.max_batch_ops,
            ),
            resource_classes=self.resource_plan.classes(),
            attention_backend=descriptor.attention_backend,
            kv_dtype=descriptor.kv_dtype,
            encoder_cache_budget=descriptor.encoder_cache_budget,
            max_vae_grid_tokens=descriptor.max_vae_grid_tokens or descriptor.max_latent_size,
            max_vit_grid_tokens=descriptor.max_vit_grid_tokens,
            commit_marker_tokens=descriptor.commit_marker_tokens,
            gen_rope_advance=descriptor.gen_rope_advance,
            max_cfg_branches=descriptor.max_cfg_branches,
        )
