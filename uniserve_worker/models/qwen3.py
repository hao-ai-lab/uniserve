"""Qwen3 causal-LM model — a thin ``nn.Module`` over the system-managed forward.

The model is a pure function of ``(input_ids, positions, forward_batch)``: it
embeds tokens, runs the decoder stack (single-axis RoPE via
``RotaryEmbedding.cos_sin_1d``, full ``head_dim`` QK-norm, the fused
QK-norm+RoPE kernel), and calls :class:`RadixAttention` per layer. It owns **no**
KV pool, builds **no** attention metadata, captures **no** CUDA graphs, and never
advances KV length — the worker runtime (``ResidencyManager`` /
``ForwardBatchBuilder`` / the system attention plan / the CUDA-graph runner /
the sampler) owns all of that. The model only *declares* its KV geometry via
:meth:`kv_cache_spec` so the system can own the pool.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import torch

import uniserve_worker.ops as ops

__all__ = [
    "Qwen3Attention",
    "Qwen3MLP",
    "Qwen3MoE",
    "Qwen3DecoderLayer",
    "Qwen3Model",
    "Qwen3ForCausalLM",
    "EntryClass",
]

if TYPE_CHECKING:
    from ..contracts.forward_batch import ForwardBatch
import torch.nn as nn

from ..contracts.resource_plan import CapsDescriptor, KvBlockResourcePolicy, ResourcePlan
from ..foundation.runtime_config import get_execution_config
from ..foundation.sizing import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_BATCH_OPS,
    derive_num_blocks,
    derive_runtime_kv_capacity,
)
from ..loader.weight_utils import WeightLoadReport, stacked_params_mapping_loop
from ..nn import (
    FusedMoE,
    LinearBase,
    ParallelLMHead,
    QKVParallelLinear,
    RadixAttention,
    RMSNorm,
    RowParallelLinear,
    VocabParallelEmbedding,
    get_current_mesh,
    get_rope,
    local_kv_head_count,
)
from ..nn.decoder import Qwen3MLP
from ..nn.logits import LogitsProcessor
from ..nn.quant import (
    QuantizationConfig,
    kv_cache_bytes_per_token,
    use_quantization_config,
)
from ..runtime.compile import CompileTarget
from ..runtime.residency import KvCacheSpec
from .cache_registrations import qwen3_cache_registration
from .registry import UniModelBase

logger = logging.getLogger(__name__)


def _cfg(config: Any | None) -> SimpleNamespace:
    if isinstance(config, SimpleNamespace):
        cfg = config
    elif isinstance(config, dict):
        cfg = SimpleNamespace(**config)
    else:
        values = {
            name: getattr(config, name)
            for name in dir(config or object())
            if not name.startswith("_") and not callable(getattr(config, name, None))
        }
        cfg = SimpleNamespace(**values)
    cfg.vocab_size = int(getattr(cfg, "vocab_size", 0))
    cfg.hidden_size = int(getattr(cfg, "hidden_size", 4096))
    cfg.intermediate_size = int(getattr(cfg, "intermediate_size", cfg.hidden_size * 4))
    cfg.num_hidden_layers = int(getattr(cfg, "num_hidden_layers", 1))
    cfg.num_attention_heads = int(getattr(cfg, "num_attention_heads", 1))
    cfg.num_key_value_heads = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
    cfg.head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // max(1, cfg.num_attention_heads)))
    cfg.hidden_act = str(getattr(cfg, "hidden_act", "silu"))
    cfg.rms_norm_eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
    cfg.rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
    cfg.max_position_embeddings = int(getattr(cfg, "max_position_embeddings", 4096))
    cfg.attention_bias = bool(getattr(cfg, "attention_bias", False))
    cfg.tie_word_embeddings = bool(getattr(cfg, "tie_word_embeddings", False))
    cfg.qk_norm_output_fp32 = bool(getattr(cfg, "qk_norm_output_fp32", False))
    return cfg


def _expert_cfg(cfg: SimpleNamespace, intermediate_size: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=cfg.hidden_size,
        intermediate_size=int(
            intermediate_size or getattr(cfg, "moe_intermediate_size", cfg.intermediate_size)
        ),
    )


class Qwen3Attention(nn.Module):
    """Multi-head self-attention with QK-norm, RoPE, and paged KV via ``RadixAttention``."""

    def __init__(self, cfg: SimpleNamespace, layer_id: int) -> None:
        super().__init__()
        self.total_num_heads = cfg.num_attention_heads
        self.total_num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.total_q_size = self.total_num_heads * self.head_dim
        self.total_kv_size = self.total_num_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            cfg.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=cfg.attention_bias,
        )
        self.q_size = int(self.qkv_proj.output_sizes[0])
        self.kv_size = int(self.qkv_proj.output_sizes[1])
        self.num_heads = self.q_size // self.head_dim
        self.num_kv_heads = self.kv_size // self.head_dim
        if self.num_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("Qwen3 local attention heads must be positive")
        self.o_proj = RowParallelLinear(self.total_q_size, cfg.hidden_size, bias=cfg.attention_bias)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.rope_theta = float(getattr(cfg, "rope_theta", 1000000.0))
        self.attn = RadixAttention(
            self.num_heads, self.num_kv_heads, self.head_dim, layer_id=layer_id
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: "ForwardBatch",
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state_shape = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)
        batched = len(state_shape) == 2
        batched_decode = batched and int(state_shape[1]) == 1
        fused_prefill = self._try_fused_prefill(
            qkv, state_shape, forward_batch, batched, cos, sin, positions
        )
        if fused_prefill is not None:
            return fused_prefill

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._prepare_qk(q, k, v, batched_decode, cos, sin)
        q_attn, k_attn, v_attn = self._attention_inputs(
            q, k, v, state_shape, batched_decode, batched
        )
        out = self.attn(
            q_attn, k_attn, v_attn, forward_batch, save_kv_cache=True, causal=True, scale=self.scale
        )
        return self.o_proj(
            self._restore_attention_output(out, state_shape, batched_decode, batched)
        )

    def _try_fused_prefill(
        self,
        qkv: torch.Tensor,
        state_shape: torch.Size,
        forward_batch: "ForwardBatch",
        batched: bool,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if batched or positions is None:
            return None
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_attn = q.reshape(-1, self.num_heads, self.head_dim)
        k_attn = k.reshape(-1, self.num_kv_heads, self.head_dim)
        q_attn, k_attn = ops.qk_norm_rope(
            q_attn,
            k_attn,
            self.q_norm.weight,
            self.k_norm.weight,
            cos,
            sin,
            self.q_norm.eps,
            override=None,
        )
        v_attn = v.reshape(-1, self.num_kv_heads, self.head_dim)
        q_attn = q_attn.to(dtype=v_attn.dtype)
        k_attn = k_attn.to(dtype=v_attn.dtype)
        out = self.attn(
            q_attn, k_attn, v_attn, forward_batch, save_kv_cache=True, causal=True, scale=self.scale
        )
        return self.o_proj(out.reshape(*state_shape, self.q_size))

    def _prepare_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batched_decode: bool,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_heads = q.reshape(-1, self.num_heads, self.head_dim)
        k_heads = k.reshape(-1, self.num_kv_heads, self.head_dim)
        q, k = ops.qk_norm_rope(
            q_heads,
            k_heads,
            self.q_norm.weight,
            self.k_norm.weight,
            cos,
            sin,
            self.q_norm.eps,
        )
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        return q.to(dtype=v.dtype), k.to(dtype=v.dtype)

    def _attention_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state_shape: torch.Size,
        batched_decode: bool,
        batched: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        if batched_decode:
            batch = int(state_shape[0])
            return (
                q.reshape(batch, self.num_heads, self.head_dim).contiguous(),
                k.reshape(batch, self.num_kv_heads, self.head_dim),
                v.reshape(batch, self.num_kv_heads, self.head_dim),
            )
        if batched:
            batch, seq = int(state_shape[0]), int(state_shape[1])
            return (
                q.reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2).contiguous(),
                k.reshape(batch, seq, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous(),
                v.reshape(batch, seq, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous(),
            )
        return q, k, v

    def _restore_attention_output(
        self,
        out: torch.Tensor,
        state_shape: torch.Size,
        batched_decode: bool,
        batched: bool,
    ) -> torch.Tensor:
        if batched_decode:
            return out.reshape(int(state_shape[0]), 1, self.q_size)
        if batched:
            return out.transpose(1, 2).reshape(*state_shape, self.q_size)
        return out.reshape(*state_shape, self.q_size)


class Qwen3MoE(nn.Module):
    """Mixture-of-experts feed-forward routed by a learned gate."""

    def __init__(self, cfg: SimpleNamespace) -> None:
        super().__init__()
        num_experts = int(getattr(cfg, "num_experts", 0) or 0)
        top_k = int(getattr(cfg, "num_experts_per_tok", 1) or 1)
        self.gate = LinearBase(cfg.hidden_size, num_experts, bias=False)
        expert_intermediate = int(
            getattr(cfg, "moe_intermediate_size", cfg.intermediate_size) or cfg.intermediate_size
        )
        self.experts = FusedMoE(
            [Qwen3MLP(_expert_cfg(cfg, expert_intermediate)) for _ in range(num_experts)],
            top_k=top_k,
            norm_topk_prob=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.experts(hidden_states, self.gate(hidden_states))


class Qwen3DecoderLayer(nn.Module):
    """One transformer decoder layer (attention + MLP or MoE)."""

    def __init__(self, cfg: SimpleNamespace, layer_id: int) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(cfg, layer_id)
        self.mlp = Qwen3MoE(cfg) if int(getattr(cfg, "num_experts", 0) or 0) > 0 else Qwen3MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        forward_batch: "ForwardBatch",
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            attn_in = self.input_layernorm(hidden_states)
        else:
            attn_in, residual = self.input_layernorm.forward_with_residual(
                hidden_states,
                residual,
                in_place=True,
            )
        attn_out = self.self_attn(attn_in, forward_batch, cos=cos, sin=sin, positions=positions)
        mlp_in, residual = self.post_attention_layernorm.forward_with_residual(
            attn_out,
            residual,
            in_place=True,
        )
        return self.mlp(mlp_in), residual


class Qwen3Model(nn.Module):
    """Stack of Qwen3 decoder layers with token embeddings and final RMSNorm."""

    def __init__(self, cfg: SimpleNamespace) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(cfg, idx) for idx in range(cfg.num_hidden_layers)
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = get_rope(
            cfg.head_dim,
            theta=cfg.rope_theta,
            max_position_embeddings=cfg.max_position_embeddings,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: "ForwardBatch",
        *,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = input_embeds if input_embeds is not None else self.embed_tokens(input_ids)
        cos, sin = self.rotary.cos_sin_1d(positions.reshape(-1))
        residual = None
        for layer_module in self.layers:
            layer = cast(Qwen3DecoderLayer, layer_module)
            hidden_states, residual = layer(
                hidden_states,
                residual,
                forward_batch,
                cos=cos,
                sin=sin,
                positions=positions,
            )
        if residual is None:
            return self.norm(hidden_states)
        hidden_states, _ = self.norm.forward_with_residual(hidden_states, residual, in_place=True)
        return hidden_states


class Qwen3ForCausalLM(UniModelBase, nn.Module):
    """Qwen3 serving model — thin: forward(input_ids, positions, forward_batch)."""

    family = "qwen3"
    architectures = ("Qwen3ForCausalLM", "Qwen3MoeForCausalLM")
    supported_ops = ("prefill_und", "decode_und", "target_verify_und")
    cache_registration_factory = staticmethod(qwen3_cache_registration)
    supported_controls: tuple[str, ...] = ()
    adapter_mode = "none"
    resource_plan = ResourcePlan(kv_block=KvBlockResourcePolicy.PER_BLOCK)

    def __init__(self, config: Any | None = None) -> None:
        super().__init__()
        self.mesh = get_current_mesh()
        self.config = _cfg(config)
        if self.config.vocab_size <= 0:
            raise ValueError("Qwen3 config must provide vocab_size")
        # Derive the checkpoint quantization policy from the original HF config
        # and enter it for layer construction so the model is self-contained.
        self._quant_config = QuantizationConfig.from_model_config(config)
        with use_quantization_config(self._quant_config):
            self.model = Qwen3Model(self.config)
            self.lm_head = ParallelLMHead(
                self.config.hidden_size, self.config.vocab_size, bias=False
            )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits = LogitsProcessor()
        self.num_layers = self.config.num_hidden_layers
        self.head_dim = int(self.config.head_dim)
        self.block_size = DEFAULT_BLOCK_SIZE
        self.num_blocks = derive_num_blocks(self.block_size, None)
        self.output_vocab_size: int | None = None
        self.kv_cache_dtype = self._requested_kv_cache_dtype_for(self.config)
        self.bytes_per_token = self._kv_bytes_per_token(torch.bfloat16)

    @property
    def device(self) -> str:
        return str(next(self.parameters()).device)

    # -- system-managed residency: the model only *declares* its KV geometry --
    def kv_cache_spec(self) -> KvCacheSpec:
        param = next(self.parameters())
        return KvCacheSpec(
            num_layers=int(self.num_layers),
            num_kv_heads=local_kv_head_count(int(self.config.num_key_value_heads)),
            head_dim=int(self.config.head_dim),
            dtype=param.dtype,
            store_dtype=self._kv_store_dtype_for(param.dtype),
        )

    def _caps_descriptor(
        self,
        *,
        block_size: int | None = None,
        kv_token_capacity: int | None = None,
    ) -> CapsDescriptor:
        block_size = DEFAULT_BLOCK_SIZE if block_size is None else int(block_size)
        num_blocks = self._runtime_num_blocks(
            block_size=block_size,
            kv_token_capacity=kv_token_capacity,
            compute_dtype=torch.bfloat16,
        )
        return CapsDescriptor(
            block_size=block_size,
            num_blocks=num_blocks,
            num_layers=int(self.num_layers),
            scratch_capacity_tokens=0,
            max_latent_size=0,
            latent_downsample=1,
            bytes_per_token=int(self._kv_bytes_per_token(torch.bfloat16)),
            max_batch_ops=DEFAULT_MAX_BATCH_OPS,
            kv_dtype=self._kv_dtype_name_for(torch.bfloat16),
        )

    def compile_targets(self) -> tuple[CompileTarget, ...]:
        return (
            CompileTarget(
                label="qwen3.model",
                module=self.model,
                owner=self,
                attr_name="model",
            ),
        )

    def configure_runtime(
        self,
        *,
        block_size: int = DEFAULT_BLOCK_SIZE,
        kv_token_capacity: int | None = None,
        caps: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.block_size = int(block_size)
        self.num_blocks = int(
            (caps or {}).get("num_blocks")
            or self._runtime_num_blocks(
                block_size=block_size,
                kv_token_capacity=kv_token_capacity,
                compute_dtype=torch.bfloat16,
            )
        )
        self.prepare_serving_dtype()
        self._maybe_compile_piecewise()

    def configure_tokenizer(
        self,
        *,
        model_path: str | None = None,
        tokenizer_vocab_size: int | None = None,
    ) -> None:
        del model_path
        if tokenizer_vocab_size is None:
            self.output_vocab_size = None
            return
        size = int(tokenizer_vocab_size)
        if 0 < size <= int(self.config.vocab_size):
            self.output_vocab_size = size

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: "ForwardBatch",
        *,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the decoder and return sampling logits for ``forward_batch``.

        The attention plan + paged residency are resolved from the published
        context (system-built); the model threads only ``forward_batch`` down to
        :class:`RadixAttention`. Returns ``[batch, vocab]`` reduced to the last
        token per row for extend/decode, or per-position logits for verify.
        """

        hidden = self.model(input_ids, positions, forward_batch, input_embeds=input_embeds)
        return self.compute_logits(hidden, forward_batch)

    def compute_logits(self, hidden: torch.Tensor, forward_batch: "ForwardBatch") -> torch.Tensor:
        """Reduce decoder hidden states to logits for ``forward_batch``.

        ``target_verify`` keeps every position's logits (the verifier slices per
        row); extend/decode reduce to the last token per request — flat token
        streams gather via ``last_token_indices``, rectangular batches take the
        final column.
        """

        from ..contracts.forward_mode import ForwardMode

        if (
            forward_batch.forward_mode == ForwardMode.VERIFY_DRAFT
            or forward_batch.return_all_logits
        ):
            return self.logits(hidden, self.lm_head, valid_vocab_size=self.output_vocab_size)
        if hidden.ndim == 3:
            last_hidden = hidden[:, -1, :]
        else:
            if forward_batch.last_token_indices is None:
                raise RuntimeError("flat Qwen3 logits require last-token indices")
            last_hidden = hidden.index_select(0, forward_batch.last_token_indices)
        return self.logits(last_hidden, self.lm_head, valid_vocab_size=self.output_vocab_size)

    def prepare_serving_dtype(self) -> None:
        """Fold checkpoint-loaded float32 weights down to the bf16 serving dtype."""
        try:
            param = next(self.parameters())
        except StopIteration:
            return
        if param.dtype == torch.float32:
            if self._has_quantized_linear_modules():
                self._cast_non_quantized_float32_parameters(torch.bfloat16)
                return
            self.to(dtype=torch.bfloat16)

    def _has_quantized_linear_modules(self) -> bool:
        for module in self.modules():
            method = getattr(module, "quant_method", None)
            if method is not None and method.is_quantized:
                return True
        return False

    def _cast_non_quantized_float32_parameters(self, dtype: torch.dtype) -> None:
        for param in self.parameters():
            if param.dtype == torch.float32 and not bool(
                getattr(param, "_uniserve_skip_serving_cast", False)
            ):
                param.data = param.data.to(dtype=dtype)

    def _kv_bytes_per_token(self, compute_dtype: torch.dtype) -> int:
        return kv_cache_bytes_per_token(
            num_kv_heads=local_kv_head_count(int(self.config.num_key_value_heads)),
            head_dim=self.config.head_dim,
            num_layers=self.num_layers,
            compute_dtype=compute_dtype,
            store_dtype=self.kv_cache_dtype,
        )

    def query_geometry(self) -> tuple[int, float, "torch.dtype"]:
        return self._query_geometry_from(self.model.layers[0].self_attn)

    def _runtime_num_blocks(
        self,
        *,
        block_size: int,
        kv_token_capacity: int | None,
        compute_dtype: torch.dtype,
    ) -> int:
        try:
            param = next(self.parameters())
            device = getattr(param, "device", None)
        except StopIteration:
            device = None
        capacity = derive_runtime_kv_capacity(
            block_size=block_size,
            kv_token_capacity=kv_token_capacity,
            bytes_per_token=self._kv_bytes_per_token(compute_dtype),
            device=device,
            memory_fraction=get_execution_config().kv_memory_fraction,
        )
        if capacity.cuda is not None:
            logger.info(
                "auto-sized Qwen3 KV pool",
                extra={
                    "device": capacity.cuda.device,
                    "free_bytes": capacity.cuda.free_bytes,
                    "total_bytes": capacity.cuda.total_bytes,
                    "fraction": capacity.cuda.memory_fraction,
                    "bytes_per_token": capacity.cuda.bytes_per_token,
                    "block_size": capacity.cuda.block_size,
                    "num_blocks": capacity.cuda.num_blocks,
                    "token_capacity": capacity.cuda.token_capacity,
                },
            )
        return capacity.num_blocks

    def load_weights(self, weights) -> WeightLoadReport:
        stacked: list[tuple[str, str, str | int]] = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        loaded, ignored = stacked_params_mapping_loop(
            self,
            list(weights),
            stacked,
            name_mapper=lambda name: name,
        )
        return WeightLoadReport(loaded=loaded, ignored=tuple(ignored))


EntryClass = Qwen3ForCausalLM
