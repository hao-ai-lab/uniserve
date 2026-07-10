"""Generic Hugging Face causal-LM fallback.

This is the day-zero text path: an unported HF causal language model can serve
prefill/decode through UniServe's shared runner, text driver, and sampler while
native ports continue to provide the optimized multimodal/diffusion paths.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from ..contracts.resource_plan import CapsDescriptor, KvBlockResourcePolicy, ResourcePlan
from ..execution.interleaved_text_stepper import resolve_op_token_ids
from ..execution.model_base import UniModelBase
from ..foundation.errors import capability_mismatch, invalid_descriptor, resource_lease_violation
from ..foundation.runtime_config import get_worker_config
from ..foundation.sizing import DEFAULT_BLOCK_SIZE, DEFAULT_MAX_BATCH_OPS
from ..loader.transformers import dtype_from_name, infer_input_device
from ..nn import RadixAttention
from ..nn.logits import forced_eos_logits

__all__ = [
    'TransformersForCausalLM',
    'EntryClass',
]

logger = logging.getLogger(__name__)

_ATTN_IMPL = "uniserve"


def _trust_remote_code() -> bool:
    return get_worker_config().transformers_trust_remote_code


def _attention_implementation() -> str:
    return get_worker_config().transformers_attn_implementation


def _uniserve_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask=None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **_kwargs: Any,
) -> tuple[torch.Tensor, None]:
    if isinstance(attention_mask, dict):
        sliding_window = getattr(module, "sliding_window", 1)
        layer_type = "full_attention" if sliding_window in {None, 1} else "sliding_attention"
        attention_mask = attention_mask[layer_type]
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise invalid_descriptor("UniServe transformers attention expects [batch, heads, tokens, dim]")
    if dropout and bool(getattr(module, "training", False)):
        raise capability_mismatch("UniServe transformers fallback attention is inference-only")
    num_heads = int(query.shape[1])
    num_kv_heads = int(key.shape[1])
    head_dim = int(query.shape[-1])
    layer_id = int(getattr(module, "layer_idx", 0) or 0)
    attn = getattr(module, "_uniserve_attention", None)
    if (
        not isinstance(attn, RadixAttention)
        or attn.num_heads != num_heads
        or attn.num_kv_heads != num_kv_heads
        or attn.head_dim != head_dim
    ):
        attn = RadixAttention(num_heads, num_kv_heads, head_dim, layer_id=layer_id)
        module._uniserve_attention = attn
    attn.layer_id = layer_id
    if is_causal is None:
        is_causal = bool(attention_mask is None and getattr(module, "is_causal", True))
    out = attn(
        query,
        key,
        value,
        causal=bool(is_causal),
        attn_mask=attention_mask,
        scale=scaling,
    )
    return out.transpose(1, 2).contiguous(), None


def _register_uniserve_attention() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(_ATTN_IMPL, _uniserve_attention_forward)


def _num_layers(config: Any) -> int:
    for name in (
        "num_hidden_layers",
        "n_layer",
        "num_layers",
        "n_layers",
        "decoder_layers",
    ):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        return _num_layers(text_cfg)
    return 1


def _bytes_per_token(config: Any) -> int:
    layers = _num_layers(config)
    hidden = int(getattr(config, "hidden_size", getattr(config, "n_embd", 4096)) or 4096)
    heads = int(getattr(config, "num_attention_heads", getattr(config, "n_head", 1)) or 1)
    kv_heads = int(getattr(config, "num_key_value_heads", heads) or heads)
    head_dim = int(getattr(config, "head_dim", hidden // max(1, heads)) or hidden // max(1, heads))
    # K + V, bytes for bf16/fp16 by default, per decoder layer.
    return kv_heads * head_dim * 2 * layers * 2


@dataclass
class _HFTextState:
    past_key_values: Any = None
    last_token_id: int | None = None
    cached_tokens: int = 0


class _HFTextPath:
    def __init__(self, owner: "TransformersForCausalLM") -> None:
        self.owner = owner
        self.states: dict[int, _HFTextState] = {}

    def drop_request(self, req_id: int) -> None:
        self.states.pop(int(req_id), None)

    @torch.inference_mode()
    def run_text_logits_batch(self, ops: list[dict[str, Any]]) -> list[torch.Tensor]:
        return [self.run_text_logits(op) for op in ops]

    @torch.inference_mode()
    def run_text_logits(self, op: dict[str, Any]) -> torch.Tensor:
        req_id = int(op["req_id"])
        tokens = [int(token) for token in resolve_op_token_ids(op)]
        if not tokens:
            return forced_eos_logits(int(self.owner.eos_id), device=self.owner.device)

        state = self.states.setdefault(req_id, _HFTextState())
        # The fallback grows HF's contiguous past_key_values per request and does
        # not allocate from the paged KV pool, so enforce the advertised
        # kv_token_capacity here rather than letting the cache grow until OOM.
        capacity = self.owner.kv_token_capacity
        projected = state.cached_tokens + len(tokens)
        if capacity is not None and int(capacity) > 0 and projected > int(capacity):
            raise resource_lease_violation(
                "Transformers fallback KV capacity exceeded for request "
                f"{req_id}: {projected} tokens > kv_token_capacity={int(capacity)}"
            )
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.owner.device)
        outputs = self.owner.model(
            input_ids=input_ids,
            past_key_values=state.past_key_values,
            use_cache=True,
        )
        state.past_key_values = getattr(outputs, "past_key_values", None)
        state.last_token_id = int(tokens[-1])
        state.cached_tokens = projected
        if bool(op.get("return_all_logits")):
            return outputs.logits
        return outputs.logits[:, -1, :]


class TransformersForCausalLM(UniModelBase):
    """Last-resort text-only wrapper around ``AutoModelForCausalLM``."""

    architectures = ("TransformersForCausalLM", "AutoModelForCausalLM", "transformers")
    fallback = True
    supported_ops = ("prefill_und", "decode_und")
    supported_controls: tuple[str, ...] = ()
    adapter_mode = "none"
    resource_plan = ResourcePlan(kv_block=KvBlockResourcePolicy.PER_BLOCK)

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        config: Any,
        device: str,
        block_size: int = DEFAULT_BLOCK_SIZE,
        kv_token_capacity: int | None = None,
        attention_backend: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = str(device)
        self.block_size = int(block_size)
        self.kv_token_capacity = kv_token_capacity
        self.attention_backend = attention_backend or "auto"
        self.num_layers = _num_layers(config)
        self.bytes_per_token = _bytes_per_token(config)
        self.eos_id = int(
            getattr(tokenizer, "eos_token_id", None)
            or getattr(config, "eos_token_id", None)
            or 0
        )
        self._text = _HFTextPath(self)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device: str,
        block_size: int = DEFAULT_BLOCK_SIZE,
        kv_token_capacity: int | None = None,
        attention_backend: str | None = None,
        **_kwargs: Any,
    ) -> "TransformersForCausalLM":
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        trust_remote_code = _trust_remote_code()
        dtype = dtype_from_name(get_worker_config().model_dtype)
        attn_impl = _attention_implementation()
        if attn_impl == _ATTN_IMPL:
            _register_uniserve_attention()
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=trust_remote_code,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_impl,
            ).eval()
        except ValueError as exc:
            msg = str(exc).lower()
            if attn_impl != _ATTN_IMPL or "attn_implementation" not in msg:
                raise
            logger.warning(
                "model rejected the %r attention implementation; downgrading to "
                "'sdpa' (paged attention disabled for this fallback model)",
                attn_impl,
                extra={"model_path": str(model_path), "error": str(exc)},
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                attn_implementation="sdpa",
            ).eval()
        if device is not None:
            model = model.to(device)
        real_device = str(infer_input_device(model, fallback=device))
        return cls(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=real_device,
            block_size=block_size,
            kv_token_capacity=kv_token_capacity,
            attention_backend=attention_backend,
        )

    def _caps_descriptor(
        self,
        *,
        block_size: int | None = None,
        kv_token_capacity: int | None = None,
    ) -> CapsDescriptor:
        block = int(block_size or self.block_size)
        cap = kv_token_capacity if kv_token_capacity is not None else self.kv_token_capacity
        num_blocks = max(1, int(cap) // block) if cap else 4096
        return CapsDescriptor(
            block_size=block,
            num_blocks=num_blocks,
            num_layers=int(self.num_layers),
            scratch_capacity_tokens=0,
            max_latent_size=0,
            latent_downsample=1,
            bytes_per_token=int(self.bytes_per_token),
            max_batch_ops=DEFAULT_MAX_BATCH_OPS,
            attention_backend=self.attention_backend,
        )

    def drop_request(self, req_id: int) -> None:
        self._text.drop_request(req_id)

    def forward(
        self,
        input_ids: Any,
        positions: Any | None = None,
        forward_batch: Any | None = None,
        *,
        op: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        """HF day-zero text forward: per-op logits via HF's own KV cache.

        Unlike the system-managed text models, this fallback has no paged KV
        pool — it grows HF ``past_key_values`` per request — so the
        :class:`~uniserve_worker.execution.text_driver.TextDriver` routes it
        through ``run_text_logits_batch`` rather than the system forward.
        """
        del positions, forward_batch
        if op is None:
            op = {"req_id": 0, "token_ids": input_ids.detach().cpu().tolist()}
        return self._text.run_text_logits(dict(op))


EntryClass = TransformersForCausalLM
