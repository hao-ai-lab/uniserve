"""Qwen3 causal-LM model — a thin ``nn.Module`` over the system-managed forward.

The model is a pure function of ``(input_ids, positions, forward_batch)``: it
embeds tokens, runs the decoder stack (single-axis RoPE via
``RotaryEmbedding.cos_sin_1d``, full ``head_dim`` QK-norm, the fused
QK-norm+RoPE kernel), and calls :class:`RadixAttention` per layer. It owns **no**
KV pool, builds **no** attention metadata, captures **no** CUDA graphs, and never
advances KV length. The executor, runner, and stores own those behaviors.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch

from ..forward import (
    ForwardBatch,
    ForwardContext,
    ForwardOutput,
    ForwardRow,
    PagedVarlenPlan,
    RouteId,
    TokenEmbeddings,
    TokenHidden,
    TokenIds,
    TokenLogits,
    TokenOutput,
    TokenRow,
    TokenSegments,
    TokenSelection,
    packed_token_ids,
    packed_token_positions,
)

__all__ = [
    "Qwen3Attention",
    "Qwen3MLP",
    "Qwen3MoE",
    "Qwen3DecoderLayer",
    "Qwen3Model",
    "Qwen3ForCausalLM",
]

import torch.nn as nn

from ..batch import WorkVariant
from ..loader.schema import Stack, WeightSpec
from ..nn import (
    FusedMoE,
    LayerSpec,
    LinearBase,
    ParallelLMHead,
    QKVParallelLinear,
    RadixAttention,
    RMSNorm,
    RowParallelLinear,
    VocabParallelEmbedding,
    get_rope,
    local_attention_head_count,
    local_kv_head_count,
    qk_norm_rope,
)
from ..nn.decoder import Qwen3MLP
from ..nn.logits import LogitsProcessor
from .runtime import (
    CacheGeometry,
    DeviceRole,
    ExecutionModel,
    LoweredStage,
    ResourceGeometry,
    RowKind,
)


def _required_int(config: Mapping[str, object], name: str) -> int:
    raw = config.get(name)
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"Qwen3 config requires integer field {name!r}")
    if raw <= 0:
        raise ValueError(f"Qwen3 config field {name!r} must be positive")
    return raw


def _optional_int(
    config: Mapping[str, object],
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    raw = config.get(name, default)
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < minimum:
        raise ValueError(f"Qwen3 config field {name!r} must be an integer >= {minimum}")
    return raw


def _number(config: Mapping[str, object], name: str, default: float) -> float:
    raw = config.get(name, default)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ValueError(f"Qwen3 config field {name!r} must be numeric")
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Qwen3 config field {name!r} must be finite and positive")
    return value


def _boolean(config: Mapping[str, object], name: str, default: bool) -> bool:
    raw = config.get(name, default)
    if not isinstance(raw, bool):
        raise ValueError(f"Qwen3 config field {name!r} must be boolean")
    return raw


def _string(config: Mapping[str, object], name: str, default: str) -> str:
    raw = config.get(name, default)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"Qwen3 config field {name!r} must be a non-empty string")
    return raw


@dataclass(frozen=True, slots=True)
class _QwenConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_act: str
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    attention_bias: bool
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> "_QwenConfig":
        hidden_size = _required_int(config, "hidden_size")
        num_attention_heads = _required_int(config, "num_attention_heads")
        if hidden_size % num_attention_heads:
            raise ValueError("Qwen3 hidden_size must be divisible by num_attention_heads")
        head_dim = _optional_int(
            config,
            "head_dim",
            hidden_size // num_attention_heads,
            minimum=1,
        )
        num_experts = _optional_int(config, "num_experts", 0, minimum=0)
        num_experts_per_tok = _optional_int(config, "num_experts_per_tok", 1, minimum=1)
        intermediate_size = _required_int(config, "intermediate_size")
        cfg = cls(
            vocab_size=_required_int(config, "vocab_size"),
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=_required_int(config, "num_hidden_layers"),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=_required_int(config, "num_key_value_heads"),
            head_dim=head_dim,
            hidden_act=_string(config, "hidden_act", "silu"),
            rms_norm_eps=_number(config, "rms_norm_eps", 1e-6),
            rope_theta=_number(config, "rope_theta", 1_000_000.0),
            max_position_embeddings=_optional_int(
                config, "max_position_embeddings", 4096, minimum=1
            ),
            attention_bias=_boolean(config, "attention_bias", False),
            tie_word_embeddings=_boolean(config, "tie_word_embeddings", False),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=_optional_int(
                config,
                "moe_intermediate_size",
                intermediate_size,
                minimum=1,
            ),
        )
        if cfg.head_dim <= 0 or cfg.max_position_embeddings <= 0:
            raise ValueError("Qwen3 head_dim and max_position_embeddings must be positive")
        if cfg.num_attention_heads % cfg.num_key_value_heads:
            raise ValueError("Qwen3 attention heads must be divisible by KV heads")
        if cfg.num_experts < 0 or cfg.num_experts_per_tok <= 0:
            raise ValueError("Qwen3 expert counts must be non-negative and top-k must be positive")
        if cfg.num_experts and cfg.num_experts_per_tok > cfg.num_experts:
            raise ValueError("Qwen3 num_experts_per_tok must not exceed num_experts")
        return cfg


@dataclass(frozen=True, slots=True)
class _MlpConfig:
    hidden_size: int
    intermediate_size: int


def _expert_cfg(cfg: _QwenConfig, intermediate_size: int | None = None) -> _MlpConfig:
    return _MlpConfig(
        hidden_size=cfg.hidden_size,
        intermediate_size=int(intermediate_size or cfg.moe_intermediate_size),
    )


class Qwen3Attention(nn.Module):
    """Multi-head self-attention with QK-norm, RoPE, and paged KV via ``RadixAttention``."""

    def __init__(self, cfg: _QwenConfig, layer_id: int, *, spec: LayerSpec) -> None:
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
            spec=spec,
            bias=cfg.attention_bias,
        )
        self.q_size = int(self.qkv_proj.output_sizes[0])
        self.kv_size = int(self.qkv_proj.output_sizes[1])
        self.num_heads = self.q_size // self.head_dim
        self.num_kv_heads = self.kv_size // self.head_dim
        if self.num_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("Qwen3 local attention heads must be positive")
        self.o_proj = RowParallelLinear(
            self.total_q_size,
            cfg.hidden_size,
            spec=spec,
            bias=cfg.attention_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.rope_theta = float(getattr(cfg, "rope_theta", 1000000.0))
        self.attn = RadixAttention(
            self.num_heads, self.num_kv_heads, self.head_dim, layer_id=layer_id
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: ForwardContext,
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
            qkv, state_shape, context, batched, cos, sin, positions
        )
        if fused_prefill is not None:
            return fused_prefill

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._prepare_qk(q, k, v, batched_decode, cos, sin)
        q_attn, k_attn, v_attn = self._attention_inputs(
            q, k, v, state_shape, batched_decode, batched
        )
        out = self.attn(q_attn, k_attn, v_attn, context, causal=True, scale=self.scale)
        return self.o_proj(
            self._restore_attention_output(out, state_shape, batched_decode, batched),
            context.mesh,
        )

    def _try_fused_prefill(
        self,
        qkv: torch.Tensor,
        state_shape: torch.Size,
        context: ForwardContext,
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
        q_attn, k_attn = qk_norm_rope(
            q_attn,
            k_attn,
            self.q_norm.weight,
            self.k_norm.weight,
            cos,
            sin,
            self.q_norm.eps,
        )
        v_attn = v.reshape(-1, self.num_kv_heads, self.head_dim)
        q_attn = q_attn.to(dtype=v_attn.dtype)
        k_attn = k_attn.to(dtype=v_attn.dtype)
        out = self.attn(q_attn, k_attn, v_attn, context, causal=True, scale=self.scale)
        return self.o_proj(out.reshape(*state_shape, self.q_size), context.mesh)

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
        q, k = qk_norm_rope(
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

    def __init__(self, cfg: _QwenConfig, *, spec: LayerSpec) -> None:
        super().__init__()
        num_experts = cfg.num_experts
        top_k = cfg.num_experts_per_tok
        self.gate = LinearBase(cfg.hidden_size, num_experts, spec=spec, bias=False)
        expert_intermediate = int(cfg.moe_intermediate_size or cfg.intermediate_size)
        self.experts = FusedMoE(
            [
                Qwen3MLP(_expert_cfg(cfg, expert_intermediate), spec=spec)
                for _ in range(num_experts)
            ],
            top_k=top_k,
            norm_topk_prob=True,
        )

    def forward(self, hidden_states: torch.Tensor, context: ForwardContext) -> torch.Tensor:
        return self.experts(hidden_states, self.gate(hidden_states), context.mesh)


class Qwen3DecoderLayer(nn.Module):
    """One transformer decoder layer (attention + MLP or MoE)."""

    def __init__(self, cfg: _QwenConfig, layer_id: int, *, spec: LayerSpec) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(cfg, layer_id, spec=spec)
        self.mlp = Qwen3MoE(cfg, spec=spec) if cfg.num_experts > 0 else Qwen3MLP(cfg, spec=spec)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        context: ForwardContext,
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
        attn_out = self.self_attn(attn_in, context, cos=cos, sin=sin, positions=positions)
        mlp_in, residual = self.post_attention_layernorm.forward_with_residual(
            attn_out,
            residual,
            in_place=True,
        )
        if isinstance(self.mlp, Qwen3MoE):
            return self.mlp(mlp_in, context), residual
        return self.mlp(mlp_in, context.mesh), residual


class Qwen3Model(nn.Module):
    """Stack of Qwen3 decoder layers with token embeddings and final RMSNorm."""

    def __init__(self, cfg: _QwenConfig, *, spec: LayerSpec) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size, spec=spec)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(cfg, idx, spec=spec) for idx in range(cfg.num_hidden_layers)
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
        context: ForwardContext,
        *,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            input_embeds if input_embeds is not None else self.embed_tokens(input_ids, context.mesh)
        )
        cos, sin = self.rotary.cos_sin_1d(positions.reshape(-1))
        residual = None
        for layer_module in self.layers:
            layer = cast(Qwen3DecoderLayer, layer_module)
            hidden_states, residual = layer(
                hidden_states,
                residual,
                context,
                cos=cos,
                sin=sin,
                positions=positions,
            )
        if residual is None:
            return self.norm(hidden_states)
        hidden_states, _ = self.norm.forward_with_residual(hidden_states, residual, in_place=True)
        return hidden_states


class Qwen3ForCausalLM(ExecutionModel):
    """Qwen3 serving model with a thin tensor-level text core."""

    weight_spec = WeightSpec(
        transforms=(
            Stack("qkv_proj", "q_proj", "q"),
            Stack("qkv_proj", "k_proj", "k"),
            Stack("qkv_proj", "v_proj", "v"),
            Stack("gate_up_proj", "gate_proj", 0),
            Stack("gate_up_proj", "up_proj", 1),
        ),
    )

    def __init__(self, config: Mapping[str, object], *, layer_spec: LayerSpec) -> None:
        super().__init__()
        if not isinstance(config, Mapping):
            raise TypeError("Qwen3 config must be a mapping")
        cfg = _QwenConfig.from_mapping(config)
        self._parallel = layer_spec.parallel
        self.model = Qwen3Model(cfg, spec=layer_spec)
        self.lm_head = ParallelLMHead(
            cfg.hidden_size,
            cfg.vocab_size,
            spec=layer_spec,
            bias=False,
        )
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits = LogitsProcessor()
        self.num_layers = cfg.num_hidden_layers
        self.head_dim = cfg.head_dim
        self.architecture = "Qwen3ForCausalLM"
        self.supported_work = frozenset(
            {
                WorkVariant.TOKEN_EXTEND,
                WorkVariant.TOKEN_DECODE,
                WorkVariant.TOKEN_VERIFY,
            }
        )
        self.cache_geometry = CacheGeometry(
            num_layers=int(self.num_layers),
            num_attention_heads=local_attention_head_count(
                cfg.num_attention_heads,
                parallel=self._parallel,
            ),
            num_kv_heads=local_kv_head_count(
                cfg.num_key_value_heads,
                parallel=self._parallel,
            ),
            head_dim=cfg.head_dim,
            dtype="bfloat16",
        )
        self.resource_geometry = ResourceGeometry()
        self._max_sequence_tokens = int(cfg.max_position_embeddings)

    def lower(
        self,
        variant: WorkVariant,
        *,
        retain_image: bool = False,
    ) -> tuple[LoweredStage, ...]:
        del retain_image
        if variant in self.supported_work:
            return (LoweredStage(RouteId("text"), RowKind.TOKEN),)
        return ()

    def route_dtype(self, route: RouteId) -> str:
        self._require_text_route(route)
        return "bfloat16"

    def route_device_role(self, route: RouteId) -> DeviceRole:
        self._require_text_route(route)
        return DeviceRole.PRIMARY

    def route_topology(self, route: RouteId) -> tuple[str, ...]:
        self._require_text_route(route)
        return ("tp",)

    def route_graph_eligible(self, route: RouteId) -> bool:
        self._require_text_route(route)
        return True

    def route_max_tokens(self, route: RouteId) -> int:
        self._require_text_route(route)
        return self._max_sequence_tokens

    def route_shape_key(self, route: RouteId, row: ForwardRow) -> tuple[int, ...]:
        del row
        self._require_text_route(route)
        return ()

    @staticmethod
    def _require_text_route(route: RouteId) -> None:
        if route != "text":
            raise ValueError(f"Qwen3 received unknown route {route!s}")

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch) -> ForwardOutput:
        if any(not isinstance(row, TokenRow) for row in batch.rows):
            raise TypeError("Qwen3 text route accepts TokenRow values only")
        rows = cast(tuple[TokenRow, ...], batch.rows)
        positions = packed_token_positions(rows)
        if positions is None:
            positions = torch.cat(tuple(row.positions.reshape(-1) for row in rows), dim=0)
        if all(isinstance(row.inputs, TokenIds) for row in rows):
            input_ids = packed_token_ids(rows)
            if input_ids is None:
                input_ids = torch.cat(
                    tuple(cast(TokenIds, row.inputs).values.reshape(-1) for row in rows),
                    dim=0,
                )
            hidden = self.model(input_ids, positions, batch.context)
        else:
            embeddings = tuple(self._row_embeddings(row, batch.context) for row in rows)
            inputs = torch.cat(embeddings, dim=0)
            placeholder = torch.zeros(inputs.shape[0], dtype=torch.long, device=inputs.device)
            hidden = self.model(
                placeholder,
                positions,
                batch.context,
                input_embeds=inputs,
            )
        return self._outputs(hidden, rows, batch.context)

    def _outputs(
        self,
        hidden: torch.Tensor,
        rows: tuple[TokenRow, ...],
        context: ForwardContext,
    ) -> ForwardOutput:
        attention = getattr(context, "attention", None)
        dynamic_last = (
            attention.output_indices
            if isinstance(attention, PagedVarlenPlan)
            and all(row.selection is TokenSelection.LAST_LOGITS for row in rows)
            else None
        )
        if dynamic_last is not None:
            selected = hidden.index_select(0, dynamic_last.to(dtype=torch.long))
            dynamic_projected = self.logits(self.lm_head(selected, context.mesh))
            return ForwardOutput(
                tuple(
                    TokenOutput(
                        row.row_id,
                        row.output_slot,
                        TokenLogits(dynamic_projected[index : index + 1]),
                    )
                    for index, row in enumerate(rows)
                )
            )
        row_hidden: list[torch.Tensor] = []
        begin = 0
        for row in rows:
            count = int(row.positions.numel())
            row_hidden.append(hidden[begin : begin + count])
            begin += count
        projected_rows = tuple(
            index for index, row in enumerate(rows) if row.selection is not TokenSelection.HIDDEN
        )
        projected: torch.Tensor | None = None
        if projected_rows:
            if (
                len(projected_rows) == len(rows)
                and all(row.selection is TokenSelection.LAST_LOGITS for row in rows)
                and all(int(value.shape[0]) == 1 for value in row_hidden)
            ):
                selected = hidden
            else:
                selected_rows = tuple(
                    row_hidden[index]
                    if rows[index].selection is TokenSelection.ALL_LOGITS
                    else row_hidden[index][-1:]
                    for index in projected_rows
                )
                selected = (
                    selected_rows[0] if len(selected_rows) == 1 else torch.cat(selected_rows, dim=0)
                )
            projected = self.logits(self.lm_head(selected, context.mesh))

        outputs: list[TokenOutput] = []
        projected_offset = 0
        for index, row in enumerate(rows):
            value: TokenHidden | TokenLogits
            if row.selection is TokenSelection.HIDDEN:
                value = TokenHidden(row_hidden[index])
            else:
                if projected is None:
                    raise RuntimeError("Qwen3 projected output buffer is missing")
                count = (
                    int(row_hidden[index].shape[0])
                    if row.selection is TokenSelection.ALL_LOGITS
                    else 1
                )
                value = TokenLogits(projected[projected_offset : projected_offset + count])
                projected_offset += count
            outputs.append(TokenOutput(row.row_id, row.output_slot, value))
        return ForwardOutput(tuple(outputs))

    def _row_embeddings(self, row: TokenRow, context: ForwardContext) -> torch.Tensor:
        if isinstance(row.inputs, TokenEmbeddings):
            return row.inputs.values.reshape(-1, row.inputs.values.shape[-1])
        if isinstance(row.inputs, TokenIds):
            return self.model.embed_tokens(row.inputs.values.reshape(-1), context.mesh)
        if not isinstance(row.inputs, TokenSegments):
            raise TypeError("Qwen3 token row has an unknown input variant")
        return torch.cat(
            tuple(
                self.model.embed_tokens(segment.values.reshape(-1), context.mesh)
                if isinstance(segment, TokenIds)
                else segment.values.reshape(-1, segment.values.shape[-1])
                for segment in row.inputs.values
            ),
            dim=0,
        )
