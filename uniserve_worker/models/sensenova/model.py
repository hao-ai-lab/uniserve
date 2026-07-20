"""SenseNova-U1 UniModel entry.

This is the registry/runner-facing model port. Execution is driven by
``ModelRunner`` and selected by ``models.registry``; text and denoise attention
use the worker-owned paged KV pool through the shared ``RadixAttention`` seam.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

import uniserve_worker.ops as ops
from uniserve_worker.execution.flow import (
    FlowExecution,
    FlowState,
    PreparedFlowStep,
    ProgramState,
)
from uniserve_worker.execution.products import (
    ImageEncoder,
    ImageMaterializer,
    ProductTransferSession,
)
from uniserve_worker.execution.segment import SegmentExecutor
from uniserve_worker.execution.sequence import SequenceCache, SequenceExecutor
from uniserve_worker.runtime.forward_stream import ForwardPagedKVView, ForwardStream

from ...contracts.forward_batch import ForwardBatch
from ...contracts.forward_context import get_forward_context
from ...contracts.resource_plan import (
    CapsDescriptor,
    EncoderResourcePolicy,
    KvBlockResourcePolicy,
    LatentTokens,
    PerBranch,
    ResourcePlan,
    active_latent_capacity_tokens,
)
from ...foundation.errors import capability_mismatch, invalid_descriptor
from ...foundation.runtime_config import decode_graph_padding_block_count
from ...foundation.sizing import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_BATCH_OPS,
    ceil_div,
    derive_num_blocks,
)
from ...loader.checkpoint_layout import CheckpointLayout
from ...loader.transformers import NativeLoadSpec
from ...nn import (
    LinearBase,
    ParallelLMHead,
    QKVParallelLinear,
    RadixAttention,
    RMSNorm,
    RowParallelLinear,
    VocabParallelEmbedding,
    WeightMode,
    get_current_mesh,
    get_rope,
    get_tower_coord,
    place_towers,
    set_tower_coord,
)
from ...nn.decoder import Modality, Qwen3MLP, route_by_modality, tower_modality_coords
from ...nn.diffusion import (
    ConvDecoder,
    FlowMatchingHead,
    ScheduleDirection,
    ScheduleShiftDomain,
    TimestepEmbedder,
)
from ...nn.diffusion.cfg import CfgRecipe
from ...nn.linear import local_kv_head_count as _local_kv_head_count
from ...nn.quant import (
    QuantizationConfig,
    kv_cache_bytes_per_token,
    use_quantization_config,
)
from ...nn.vision import NeoVitConfig, NeoVitEncoder, build_abs_positions_from_grid_hw
from ...processors.registry import get_processor_for_model
from ...runtime.compile import CompileTarget
from ...runtime.kv_pool import PagedKVPool
from ...runtime.request_state import RequestState as RunnerRequestState
from ...runtime.residency import (
    DEFAULT_ENCODER_CACHE_BUDGET,
    GenResidencySpec,
    KvCacheSpec,
    ResidencyManager,
    encoder_handle_from_mm_hash,
)
from ...runtime.tower_handoff import (
    ConditioningSnapshot,
    LocalP2PTowerHandoff,
    TowerBinding,
    TowerHandoff,
)
from ..registry import UniModelBase
from .config import NeoChatConfig

__all__ = [
    "IMG_START_TOKEN",
    "IMG_END_TOKEN",
    "SENSENOVA_MODEL_CODE_VERSION",
    "MAX_BATCH_OPS",
    "NeoVisionModel",
    "NEOChatModel",
    "check_checkpoint_compatibility",
    "SenseNovaU1ForUnifiedGeneration",
    "EntryClass",
]

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"

# Version of this SenseNova-U1 model port. A checkpoint may declare a minimum
# required model-code version via ``uniserve_sensenova_min_version``; bump this
# constant when the served graph/loader contract changes.
SENSENOVA_MODEL_CODE_VERSION = "0.1.0"

# Worker batching limit for capability reporting.
MAX_BATCH_OPS = DEFAULT_MAX_BATCH_OPS
MAX_VIT_GRID_TOKENS = 70 * 70
COMMIT_MARKER_TOKENS = 2
GEN_ROPE_ADVANCE = 2
_UNDERSTANDING_IMAGE_MEAN = (0.485, 0.456, 0.406)
_UNDERSTANDING_IMAGE_STD = (0.229, 0.224, 0.225)
MAX_CFG_BRANCHES = 3
GENERATION_T_EPS = 0.02

# Resolution-aware modes apply sqrt sequence-length scaling; the others leave the
# base noise scale unchanged. Unknown string modes are treated like fixed scale.
_NOISE_RESOLUTION_EXPONENT = 0.5
_NOISE_DYNAMIC_SQRT_EXPONENT = 0.5
_NOISE_RESOLUTION_MODES = frozenset({"resolution", "dynamic", "dynamic_sqrt"})

logger = logging.getLogger(__name__)


def _cat_token_slices(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    tensors = [part for part in parts if int(part.shape[0]) > 0]
    if not tensors:
        raise invalid_descriptor("route slicing requires at least one tensor")
    return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)


def _contiguous_route_split(forward_stream: ForwardStream, total_tokens: int) -> int | None:
    first_route_tokens = 0
    second_route_tokens = 0
    seen_second_route = False
    for seg in forward_stream.segments:
        q_len = int(seg.q_len)
        if seg.modality == "und":
            if seen_second_route:
                return None
            first_route_tokens += q_len
        elif seg.modality == "gen":
            seen_second_route = True
            second_route_tokens += q_len
        else:
            return None
    if first_route_tokens <= 0 or second_route_tokens <= 0:
        return None
    if first_route_tokens + second_route_tokens != int(total_tokens):
        return None
    return int(first_route_tokens)


@dataclass(frozen=True)
class _SenseNovaTowerLayout:
    def tag_generation_modules(self, model: Any, gen: int) -> None:
        set_tower_coord(model.fm_modules, gen)
        decoder = model.language_model.model
        set_tower_coord(decoder.norm_mot_gen, gen)
        for layer in decoder.layers:
            set_tower_coord(layer.input_layernorm_mot_gen, gen)
            set_tower_coord(layer.post_attention_layernorm_mot_gen, gen)
            set_tower_coord(layer.mlp_mot_gen, gen)
            attn = layer.self_attn
            for module in (
                attn.qkv_proj_mot_gen,
                attn.o_proj_mot_gen,
                attn.q_norm_mot_gen,
                attn.k_norm_mot_gen,
                attn.q_norm_hw_mot_gen,
                attn.k_norm_hw_mot_gen,
            ):
                set_tower_coord(module, gen)

    def filter_from_model(
        self,
        model: nn.Module,
        tower_role: str | None,
    ) -> Callable[[str], bool] | None:
        if tower_role is None:
            return None
        if tower_role not in {"gen", "und"}:
            raise ValueError(f"unknown tower_role {tower_role!r}")
        self.tag_generation_modules(model, 1)
        gen_names = self._tagged_param_names(model)
        if tower_role == "gen":
            return gen_names.__contains__
        return lambda name: name not in gen_names

    @staticmethod
    def _tagged_param_names(model: nn.Module) -> set[str]:
        names: set[str] = set()
        for module_name, module in model.named_modules():
            if get_tower_coord(module) is None:
                continue
            for param_name, _ in module.named_parameters(recurse=True):
                names.add(f"{module_name}.{param_name}" if module_name else param_name)
        return names


_TOWER_LAYOUT = _SenseNovaTowerLayout()
_SENSENOVA_STACKED_PARAMS = (
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("qkv_proj_mot_gen", "q_proj_mot_gen", "q"),
    ("qkv_proj_mot_gen", "k_proj_mot_gen", "k"),
    ("qkv_proj_mot_gen", "v_proj_mot_gen", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)


def _config_int(config: Any | None, key: str, default: int) -> int:
    if isinstance(config, dict):
        return int(config.get(key, default) or default)
    return int(getattr(config, key, default) or default)


def _axis_position_ids(indexes: torch.Tensor, axis: int) -> torch.Tensor:
    positions = indexes[int(axis)]
    if positions.ndim == 1:
        return positions.unsqueeze(0)
    if positions.ndim == 2:
        return positions
    raise ValueError("SenseNova 3D RoPE indexes must be shaped [3, L] or [3, B, L]")


def _flatten_3d_indexes(indexes: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
    if indexes.ndim == 2:
        if indexes.shape != (3, seq_len):
            raise ValueError("SenseNova 3D RoPE indexes must be shaped [3, L]")
        expanded = indexes[:, None, :].expand(3, batch, seq_len)
    elif indexes.ndim == 3:
        if indexes.shape != (3, batch, seq_len):
            raise ValueError("SenseNova batched 3D RoPE indexes must be shaped [3, B, L]")
        expanded = indexes
    else:
        raise ValueError("SenseNova 3D RoPE indexes must be shaped [3, L] or [3, B, L]")
    return expanded.reshape(3, batch * seq_len)


@dataclass(frozen=True)
class SenseNovaPackedRope:
    cos: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    sin: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    # Structural caller guarantee that every h/w index in this forward is zero
    # (pure text tokens), so the spatial rotations are the identity and the
    # attention path may use the fused single-launch norm+RoPE kernel.
    hw_identity: bool = False

    def select(self, mask: torch.Tensor) -> "SenseNovaPackedRope":
        positions = mask.nonzero(as_tuple=False).flatten()
        return self.select_indices(positions)

    def select_indices(self, positions: torch.Tensor) -> "SenseNovaPackedRope":
        return SenseNovaPackedRope(
            (
                self.cos[0].index_select(0, positions),
                self.cos[1].index_select(0, positions),
                self.cos[2].index_select(0, positions),
            ),
            (
                self.sin[0].index_select(0, positions),
                self.sin[1].index_select(0, positions),
                self.sin[2].index_select(0, positions),
            ),
            hw_identity=self.hw_identity,
        )

    def slice(self, start: int, end: int) -> "SenseNovaPackedRope":
        start = int(start)
        end = int(end)
        return SenseNovaPackedRope(
            (self.cos[0][start:end], self.cos[1][start:end], self.cos[2][start:end]),
            (self.sin[0][start:end], self.sin[1][start:end], self.sin[2][start:end]),
            hw_identity=self.hw_identity,
        )


@dataclass(frozen=True)
class _ChatTemplateRenderer:
    name: str

    def render(
        self,
        prompt_text: str,
        *,
        system_message: str = "",
        append_text: str | None = None,
    ) -> str:
        return _render_chatml_prompt(
            prompt_text,
            system_message=system_message,
            append_text=append_text,
        )


def _render_chatml_prompt(
    prompt_text: str,
    *,
    system_message: str = "",
    append_text: str | None = None,
) -> str:
    out = ""
    if system_message:
        out += f"<|im_start|>system\n{system_message}<|im_end|>\n"
    out += f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
    if append_text is not None:
        out += append_text
    return out


_PROMPT_RENDERERS = {
    "neo1_0": _ChatTemplateRenderer("neo1_0"),
}


def _resolve_prompt_renderer(template: str | None) -> _ChatTemplateRenderer:
    return _PROMPT_RENDERERS[template or "neo1_0"]


class NeoVisionModel(nn.Module):
    """NEO ViT tower for image understanding embeddings."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.embeddings = NeoVitEncoder(
            NeoVitConfig(
                hidden_size=int(config.hidden_size),
                llm_hidden_size=int(config.llm_hidden_size),
                downsample_ratio=float(config.downsample_ratio),
                patch_size=int(config.patch_size),
                num_channels=int(config.num_channels),
                rope_theta_vision=float(config.rope_theta_vision),
            )
        )

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        *,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        pixel_embeds: torch.Tensor | None = None,
        grid_hw: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPast:
        del output_hidden_states, return_dict
        if pixel_values is None and pixel_embeds is None:
            raise ValueError("pixel_values or pixel_embeds is required")
        hidden = (
            pixel_embeds if pixel_embeds is not None else self.embeddings(pixel_values, grid_hw)
        )
        return BaseModelOutputWithPast(last_hidden_state=cast(Any, hidden))


def _resolve_tower(mesh: Any | None = None) -> tuple[Any | None, dict[Modality, int] | None]:
    """Resolve the ``tower`` axis transport + per-modality coordinates from the mesh.

    Returns ``(None, None)`` for a trivial/absent tower, which makes every
    :func:`route_by_modality` call below take the in-place single-device path
    (byte-identical). With a tower, mixed-modality batches route each modality's
    tokens to its coordinate's device via the transport (handled by the shared
    Router instead of model-private routing)."""
    mesh = mesh if mesh is not None else get_current_mesh()
    coords = tower_modality_coords(mesh)
    tower_axis = mesh.axis("tower")
    transport = tower_axis.transport if coords is not None and tower_axis is not None else None
    return transport, coords


def _SenseNovaMLP(config: Any) -> Qwen3MLP:
    return Qwen3MLP(config, weight_mode=WeightMode.FUSED_GATE_UP_LINEAR)


class _SenseNovaAttention(nn.Module):
    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.head_dim = int(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        )
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        self.scaling = self.head_dim**-0.5

        q_out = self.total_num_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
        )
        self.qkv_proj_mot_gen = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
        )
        # Local (per-tensor-parallel-rank) head counts, derived from the sharded
        # projection exactly like Qwen3: activations downstream of qkv_proj carry
        # these local counts, and both towers share one geometry.
        self.num_heads = int(self.qkv_proj.output_sizes[0]) // self.head_dim
        self.num_kv_heads = int(self.qkv_proj.output_sizes[1]) // self.head_dim
        if self.num_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("SenseNova local attention heads must be positive")
        self.attn = RadixAttention(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            layer_id=self.layer_idx,
        )
        self.o_proj = RowParallelLinear(q_out, config.hidden_size, bias=config.attention_bias)
        self.o_proj_mot_gen = RowParallelLinear(
            q_out, config.hidden_size, bias=config.attention_bias
        )

        self.q_norm = RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.q_norm_mot_gen = RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.q_norm_hw = RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.q_norm_hw_mot_gen = RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.k_norm_mot_gen = RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.k_norm_hw = RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.k_norm_hw_mot_gen = RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        # SenseNova-U1 is served as full attention: RadixAttention has no per-layer
        # sliding-window channel and the worker emits an empty caps groups list,
        # so the host always builds a single full-attention BlockManager. The
        # config's per-layer sliding_window is therefore intentionally not honored
        # here. If sliding-window support is ever added it must be wired end to end
        # (caps groups + per-layer window in attention + host block trimming), not
        # reintroduced as a dead per-layer attribute. config.layer_types is still
        # consulted by the decoder layer for the causal-mask mapping only.

        # get_rope / HFRotaryEmbedding only reads scalar fields from the config,
        # so a shallow copy with overridden scalars is sufficient and avoids a
        # full deep-copy of the (large, nested) config in every one of the
        # num_layers attention modules. The shallow copy gives each rope table
        # its own __dict__, so these scalar overrides do not mutate ``config``.
        t_config = copy.copy(config)
        t_config.head_dim = self.head_dim // 2
        self.rotary_emb = get_rope(config=t_config, keep_freq_range=True)
        hw_config = copy.copy(config)
        hw_config.head_dim = self.head_dim // 4
        hw_config.rope_theta = config.rope_theta_hw
        hw_config.max_position_embeddings = config.max_position_embeddings_hw
        self.rotary_emb_hw = get_rope(config=hw_config, keep_freq_range=True)
        # Tower-axis routing for the o_proj scatter (no-op on a trivial tower).
        self._tower_transport, self._tower_coords = _resolve_tower()

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        *,
        gen_branch: bool,
        packed_rope: SenseNovaPackedRope | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        if gen_branch:
            q_norm, k_norm = self.q_norm_mot_gen, self.k_norm_mot_gen
            q_norm_hw, k_norm_hw = self.q_norm_hw_mot_gen, self.k_norm_hw_mot_gen
            qkv_proj = self.qkv_proj_mot_gen
        else:
            q_norm, k_norm = self.q_norm, self.k_norm
            q_norm_hw, k_norm_hw = self.q_norm_hw, self.k_norm_hw
            qkv_proj = self.qkv_proj

        split_sizes = [int(size) for size in qkv_proj.output_sizes]
        if gen_branch and int(hidden_states.shape[-2]) == 1:
            q_w, k_w, v_w = qkv_proj.weight.split(split_sizes, dim=0)
            bias = qkv_proj.bias
            if bias is not None:
                q_b, k_b, v_b = bias.split(split_sizes, dim=0)
            else:
                q_b = k_b = v_b = None
            # One-token autoregressive image decode is numerically sensitive to
            # GEMM reassociation; keep the generation tower on the per-projection
            # accumulation order while text decode uses the packed QKV path that
            # text prefill already uses.
            q_flat = F.linear(hidden_states, q_w, q_b)
            k_flat = F.linear(hidden_states, k_w, k_b)
            v_flat = F.linear(hidden_states, v_w, v_b)
        else:
            q_flat, k_flat, v_flat = qkv_proj(hidden_states).split(split_sizes, dim=-1)
        query_states = q_flat.view(hidden_shape)
        local_kv_heads = int(k_flat.shape[-1]) // self.head_dim
        key_states = k_flat.view(*input_shape, local_kv_heads, self.head_dim)
        value_states = v_flat.view(*input_shape, local_kv_heads, self.head_dim).transpose(1, 2)

        query_states, key_states = self._qk_norm_rope_3d(
            query_states,
            key_states,
            indexes,
            q_norm=q_norm,
            k_norm=k_norm,
            q_norm_hw=q_norm_hw,
            k_norm_hw=k_norm_hw,
            packed_rope=packed_rope,
        )
        return query_states, key_states, value_states

    def _project_qkv_packed_slices(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        text_tokens: int,
        *,
        packed_rope: SenseNovaPackedRope | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_tokens = int(text_tokens)
        total_tokens = int(hidden_states.shape[0])
        if text_tokens <= 0 or text_tokens >= total_tokens:
            raise invalid_descriptor("packed slice routing requires both text and gen tokens")
        pieces: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        text_rope = packed_rope.slice(0, text_tokens) if packed_rope is not None else None
        q_text, k_text, v_text = self._project_qkv(
            hidden_states[:text_tokens].unsqueeze(0),
            indexes[:, :text_tokens],
            gen_branch=False,
            packed_rope=text_rope,
        )
        pieces.append(
            (
                q_text.squeeze(0).transpose(0, 1).contiguous(),
                k_text.squeeze(0).transpose(0, 1).contiguous(),
                v_text.squeeze(0).transpose(0, 1).contiguous(),
            )
        )
        gen_rope = packed_rope.slice(text_tokens, total_tokens) if packed_rope is not None else None
        q_gen, k_gen, v_gen = self._project_qkv(
            hidden_states[text_tokens:].unsqueeze(0),
            indexes[:, text_tokens:],
            gen_branch=True,
            packed_rope=gen_rope,
        )
        pieces.append(
            (
                q_gen.squeeze(0).transpose(0, 1).contiguous(),
                k_gen.squeeze(0).transpose(0, 1).contiguous(),
                v_gen.squeeze(0).transpose(0, 1).contiguous(),
            )
        )
        return (
            _cat_token_slices([piece[0] for piece in pieces]),
            _cat_token_slices([piece[1] for piece in pieces]),
            _cat_token_slices([piece[2] for piece in pieces]),
        )

    def _qk_norm_rope_3d(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        indexes: torch.Tensor,
        *,
        q_norm: RMSNorm,
        k_norm: RMSNorm,
        q_norm_hw: RMSNorm,
        k_norm_hw: RMSNorm,
        packed_rope: SenseNovaPackedRope | None = None,
        override: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if packed_rope is None:
            packed_rope = self._packed_rope(indexes)
        axis_dims = (self.head_dim // 2, self.head_dim // 4, self.head_dim // 4)
        if query_states.ndim == 4:
            # Flatten [B, L, H, D] -> [B*L, H, D]: a free view of the projection
            # output that keeps the canonical token-major rope layout. For the
            # pure-text forward (hw_identity) the provider takes its fused
            # single-launch path; for spatial (image) tokens the multi-axis
            # provider consumes the token-major rows directly, so it skips the
            # per-group transpose->contiguous flatten/unflatten copies of the
            # 4-D path. The norm/rope kernels are stride-aware and see the same
            # shapes and values either way, so results are bit-identical; only
            # the storage layout behind the returned [B, H, L, D] views changes
            # (token-major, which is also what the varlen attention flattener
            # wants, making its materialization a no-op).
            batch, seq_len, q_heads, dim = query_states.shape
            k_heads = int(key_states.shape[2])
            q_flat, k_flat = ops.qk_norm_rope(
                query_states.reshape(batch * seq_len, q_heads, dim),
                key_states.reshape(batch * seq_len, k_heads, dim),
                (q_norm.weight, q_norm_hw.weight, q_norm_hw.weight),
                (k_norm.weight, k_norm_hw.weight, k_norm_hw.weight),
                packed_rope.cos,
                packed_rope.sin,
                q_norm.eps,
                axis_dims=axis_dims,
                identity_axes=(1, 2) if packed_rope.hw_identity else None,
                override=override,
            )
            return (
                q_flat.view(batch, seq_len, q_heads, dim).transpose(1, 2),
                k_flat.view(batch, seq_len, k_heads, dim).transpose(1, 2),
            )
        query_states, key_states = ops.qk_norm_rope(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            (q_norm.weight, q_norm_hw.weight, q_norm_hw.weight),
            (k_norm.weight, k_norm_hw.weight, k_norm_hw.weight),
            packed_rope.cos,
            packed_rope.sin,
            q_norm.eps,
            axis_dims=axis_dims,
            override=override,
        )
        return query_states, key_states

    def _packed_rope(
        self, indexes: torch.Tensor, *, hw_identity: bool = False
    ) -> SenseNovaPackedRope:
        if indexes.ndim != 2 or indexes.shape[0] != 3:
            raise ValueError("SenseNova packed RoPE expects flat indexes [3, N]")
        device = indexes.device
        t_index = indexes[0]
        h_index = indexes[1]
        w_index = indexes[2]
        if t_index.device != self.rotary_emb.inv_freq.device:
            t_index = t_index.to(device=self.rotary_emb.inv_freq.device)
        if h_index.device != self.rotary_emb_hw.inv_freq.device:
            h_index = h_index.to(device=self.rotary_emb_hw.inv_freq.device)
        if w_index.device != self.rotary_emb_hw.inv_freq.device:
            w_index = w_index.to(device=self.rotary_emb_hw.inv_freq.device)
        cos_t, sin_t = self.rotary_emb.cos_sin_1d(t_index)
        cos_h, sin_h = self.rotary_emb_hw.cos_sin_1d(h_index)
        cos_w, sin_w = self.rotary_emb_hw.cos_sin_1d(w_index)

        def _on_index_device(tensor: torch.Tensor) -> torch.Tensor:
            return tensor if tensor.device == device else tensor.to(device=device)

        return SenseNovaPackedRope(
            (_on_index_device(cos_t), _on_index_device(cos_h), _on_index_device(cos_w)),
            (_on_index_device(sin_t), _on_index_device(sin_h), _on_index_device(sin_w)),
            hw_identity=hw_identity,
        )

    def _project_qkv_routed(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        route_indicators: torch.Tensor,
        *,
        exist_non_image_gen_tokens: bool | None = None,
        exist_image_gen_tokens: bool | None = None,
        packed_rope: SenseNovaPackedRope | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError("SenseNova routed QKV expects hidden_states [B, L, C]")
        batch, seq_len, _ = hidden_states.shape
        flat_hidden = hidden_states.reshape(batch * seq_len, -1)
        flat_indexes = _flatten_3d_indexes(indexes, batch, seq_len)
        flat_gen = route_indicators.reshape(batch * seq_len).to(dtype=torch.bool)

        # Callers on the packed forward path know the und/gen split for the whole
        # batch up front; reuse it to skip the per-branch device->host sync that
        # bool(mask.any()) would otherwise force every layer.
        branch_exists = {
            False: exist_non_image_gen_tokens,
            True: exist_image_gen_tokens,
        }
        projected = []
        for gen_branch, mask in ((False, ~flat_gen), (True, flat_gen)):
            exists = branch_exists[gen_branch]
            if exists is None:
                exists = bool(mask.any())
            if not exists:
                continue
            branch_rope = packed_rope.select(mask) if packed_rope is not None else None
            q, k, v = self._project_qkv(
                flat_hidden[mask].unsqueeze(0),
                flat_indexes[:, mask],
                gen_branch=gen_branch,
                packed_rope=branch_rope,
            )
            projected.append(
                (
                    mask,
                    q.squeeze(0).transpose(0, 1).contiguous(),
                    k.squeeze(0).transpose(0, 1).contiguous(),
                    v.squeeze(0).transpose(0, 1).contiguous(),
                )
            )

        if not projected:
            raise ValueError("SenseNova routed QKV requires at least one token")
        q_dim = int(projected[0][1].shape[-1])
        k_dim = int(projected[0][2].shape[-1])
        v_dim = int(projected[0][3].shape[-1])
        q_flat = hidden_states.new_empty(batch * seq_len, self.num_heads, q_dim)
        k_flat = hidden_states.new_empty(batch * seq_len, self.num_kv_heads, k_dim)
        v_flat = hidden_states.new_empty(batch * seq_len, self.num_kv_heads, v_dim)
        for mask, q_part, k_part, v_part in projected:
            if q_part.shape[-1] != q_dim or k_part.shape[-1] != k_dim or v_part.shape[-1] != v_dim:
                raise ValueError(
                    "SenseNova routed QKV expert projections produced mismatched head dims"
                )
            q_flat[mask] = q_part
            k_flat[mask] = k_part
            v_flat[mask] = v_part

        q = q_flat.view(batch, seq_len, self.num_heads, q_dim).transpose(1, 2).contiguous()
        k = k_flat.view(batch, seq_len, self.num_kv_heads, k_dim).transpose(1, 2).contiguous()
        v = v_flat.view(batch, seq_len, self.num_kv_heads, v_dim).transpose(1, 2).contiguous()
        return q, k, v

    def _project_qkv_flat_routed(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        route_indicators: torch.Tensor,
        *,
        exist_non_image_gen_tokens: bool | None = None,
        exist_image_gen_tokens: bool | None = None,
        und: torch.Tensor | None = None,
        packed_rope: SenseNovaPackedRope | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del und  # accepted for call-site symmetry; routing uses the gen mask
        if hidden_states.ndim != 2:
            raise ValueError("SenseNova packed routed QKV expects hidden_states [N, C]")
        if indexes.shape != (3, hidden_states.shape[0]):
            raise ValueError("SenseNova packed routed QKV expects indexes [3, N]")
        if route_indicators.shape != (hidden_states.shape[0],):
            raise ValueError("SenseNova packed routed QKV expects route_indicators [N]")
        q, k, v = self._project_qkv_routed(
            hidden_states.unsqueeze(0),
            indexes,
            route_indicators.unsqueeze(0),
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            packed_rope=packed_rope,
        )
        return (
            q.squeeze(0).transpose(0, 1).contiguous(),
            k.squeeze(0).transpose(0, 1).contiguous(),
            v.squeeze(0).transpose(0, 1).contiguous(),
        )

    def _attend_bhld(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:
        out = self.attn(q, k, v, causal=False, scale=self.scaling, attn_mask=attention_mask)
        return out.transpose(1, 2).contiguous(), None

    def _attend_bshd(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out = self.attn(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            causal=False,
            scale=self.scaling,
        )
        return out.transpose(1, 2).contiguous()

    def _attend_packed_visible(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        forward_stream: ForwardStream,
        kv_view: ForwardPagedKVView,
    ) -> torch.Tensor:
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("SenseNova packed visible attention expects [N, H, D] tensors")
        kv_view.append_packed(self.layer_idx, k, v)
        k_cache, v_cache = kv_view.pool.layer_cache(self.layer_idx)
        cache_after = kv_view.cache_seqlens_after(device=q.device)
        # max_seqlen_k is identical for every layer of the packed forward; derive
        # it on the host from the segment metadata (Python ints) instead of
        # forcing a per-layer device->host sync via cache_after.max().item().
        max_seqlen_k_hook = getattr(kv_view, "max_seqlen_k", None)
        max_seqlen_k = (
            int(max_seqlen_k_hook())
            if callable(max_seqlen_k_hook)
            else max((seg.base_len + seg.q_len for seg in kv_view.segments), default=0)
        )
        if forward_stream.fully_visible:
            out = self.attn.forward_visible_end(
                q,
                k_cache,
                v_cache,
                visible_end=forward_stream.visible_end,
                cu_seqlens_q=forward_stream.cu_seqlens_q,
                page_table=kv_view.block_table(device=q.device),
                seqused_k=cache_after,
                max_seqlen_q=int(forward_stream.visible_end.shape[1]),
                max_seqlen_k=int(max_seqlen_k),
                scale=self.scaling,
                use_prefix_bounds=True,
                fully_visible=True,
            )
            return out.contiguous()
        # Route visible_end attention through RadixAttention (backend resolution + stats).
        out = self.attn.forward_visible_end(
            q,
            k_cache,
            v_cache,
            visible_end=forward_stream.visible_end,
            cu_seqlens_q=forward_stream.cu_seqlens_q,
            page_table=kv_view.block_table(device=q.device),
            seqused_k=cache_after,
            max_seqlen_q=int(forward_stream.visible_end.shape[1]),
            max_seqlen_k=int(max_seqlen_k),
            scale=self.scaling,
            use_prefix_bounds=True,
            fully_visible=forward_stream.fully_visible,
        )
        return out.contiguous()

    def forward_packed_visible(
        self,
        hidden_states: torch.Tensor,
        route_indicators: torch.Tensor,
        indexes: torch.Tensor,
        *,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        und: torch.Tensor,
        und_indices: torch.Tensor | None = None,
        forward_stream: ForwardStream,
        kv_view: ForwardPagedKVView,
        packed_rope: SenseNovaPackedRope | None = None,
        dense_gen_route: bool = False,
        modality_split: int | None = None,
    ) -> torch.Tensor:
        # gen mask (route_indicators) and its complement (und) plus the exist
        # flags are precomputed once by the model; reuse them so the o_proj
        # routing avoids a per-layer device->host sync.
        gen = route_indicators
        ctx = get_forward_context()
        qkv_start = ctx.component_timer_start()
        if exist_non_image_gen_tokens != exist_image_gen_tokens:
            gen_branch = bool(exist_image_gen_tokens)
            q_bhld, k_bhld, v_bhld = self._project_qkv(
                hidden_states.unsqueeze(0),
                indexes,
                gen_branch=gen_branch,
                packed_rope=packed_rope,
            )
            q = q_bhld.squeeze(0).transpose(0, 1).contiguous()
            k = k_bhld.squeeze(0).transpose(0, 1).contiguous()
            v = v_bhld.squeeze(0).transpose(0, 1).contiguous()
        elif modality_split is not None:
            q, k, v = self._project_qkv_packed_slices(
                hidden_states,
                indexes,
                int(modality_split),
                packed_rope=packed_rope,
            )
        elif dense_gen_route:
            q_bhld, k_bhld, v_bhld = self._project_qkv(
                hidden_states.unsqueeze(0),
                indexes,
                gen_branch=True,
                packed_rope=packed_rope,
            )
            q = q_bhld.squeeze(0).transpose(0, 1).contiguous()
            k = k_bhld.squeeze(0).transpose(0, 1).contiguous()
            v = v_bhld.squeeze(0).transpose(0, 1).contiguous()
            if exist_non_image_gen_tokens:
                if und_indices is None:
                    und_indices = und.nonzero(as_tuple=False).flatten()
                text_hidden = hidden_states.index_select(0, und_indices)
                text_indexes = indexes.index_select(1, und_indices)
                text_rope = (
                    packed_rope.select_indices(und_indices) if packed_rope is not None else None
                )
                q_text, k_text, v_text = self._project_qkv(
                    text_hidden.unsqueeze(0),
                    text_indexes,
                    gen_branch=False,
                    packed_rope=text_rope,
                )
                q.index_copy_(0, und_indices, q_text.squeeze(0).transpose(0, 1).contiguous())
                k.index_copy_(0, und_indices, k_text.squeeze(0).transpose(0, 1).contiguous())
                v.index_copy_(0, und_indices, v_text.squeeze(0).transpose(0, 1).contiguous())
        else:
            q, k, v = self._project_qkv_flat_routed(
                hidden_states,
                indexes,
                gen,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                und=und,
                packed_rope=packed_rope,
            )
        ctx.record_component_elapsed("packed_decoder_qkv", qkv_start)
        attn_start = ctx.component_timer_start()
        out = self._attend_packed_visible(
            q,
            k,
            v,
            forward_stream=forward_stream,
            kv_view=kv_view,
        )
        ctx.record_component_elapsed("packed_decoder_attention", attn_start)
        out = out.reshape(hidden_states.shape[0], -1).contiguous()
        o_proj_start = ctx.component_timer_start()
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            projected = self.o_proj(out)
            ctx.record_component_elapsed("packed_decoder_o_proj", o_proj_start)
            return projected
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            projected = self.o_proj_mot_gen(out)
            ctx.record_component_elapsed("packed_decoder_o_proj", o_proj_start)
            return projected
        if modality_split is not None:
            routed = self._route_o_proj_slices(out, int(modality_split))
            ctx.record_component_elapsed("packed_decoder_o_proj", o_proj_start)
            return routed
        if dense_gen_route:
            routed = self.o_proj_mot_gen(out)
            if und_indices is None:
                und_indices = und.nonzero(as_tuple=False).flatten()
            routed.index_copy_(0, und_indices, self.o_proj(out.index_select(0, und_indices)))
            ctx.record_component_elapsed("packed_decoder_o_proj", o_proj_start)
            return routed
        projected = self._route_o_proj(
            out,
            text_mask=und,
            gen_mask=gen,
            exist_text=exist_non_image_gen_tokens,
            exist_gen=exist_image_gen_tokens,
            out=out.new_empty((out.shape[0], self.config.hidden_size)),
        )
        ctx.record_component_elapsed("packed_decoder_o_proj", o_proj_start)
        return projected

    def _attend_paged_update(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_values: Any,
        *,
        attention_mask: torch.Tensor | None,
        causal: bool = False,
    ) -> tuple[torch.Tensor, None] | None:
        if attention_mask is not None or past_key_values is None or q.ndim != 4:
            return None
        if q.shape[0] != 1 and not getattr(past_key_values, "supports_batched_paged", False):
            return None
        request_cache = getattr(past_key_values, "request_cache_for_update", None)
        finish = getattr(past_key_values, "finish_layer_update", None)
        cancel = getattr(past_key_values, "cancel_layer_update", None)
        if not callable(request_cache) or not callable(finish):
            return None
        n_tokens = int(q.shape[2])
        plan = get_forward_context().attention_plan
        residency_cache = getattr(plan, "residency_cache", None)
        plan_rows = len(tuple(getattr(residency_cache, "base_lens", ()) or ()))
        packed_varlen = (
            causal
            and residency_cache is not None
            and getattr(past_key_values, "cache", None) is residency_cache
            and plan_rows > 1
            and int(q.shape[0]) != plan_rows
        )
        if packed_varlen or not self.attn.can_run_paged_attention(q, None):
            if (
                causal
                and residency_cache is not None
                and getattr(past_key_values, "cache", None) is residency_cache
            ):
                cache = request_cache(self.layer_idx, n_tokens)
                try:
                    batch, heads, q_len, head_dim = q.shape
                    q_run = q.transpose(1, 2).reshape(batch * q_len, heads, head_dim).contiguous()
                    k_run = (
                        k.transpose(1, 2)
                        .reshape(batch * q_len, k.shape[1], k.shape[3])
                        .contiguous()
                    )
                    v_run = (
                        v.transpose(1, 2)
                        .reshape(batch * q_len, v.shape[1], v.shape[3])
                        .contiguous()
                    )
                    out = self.attn(
                        q_run,
                        k_run,
                        v_run,
                        kv_cache=cache,
                        update_cache=True,
                        causal=True,
                        scale=self.scaling,
                    )
                except Exception:
                    if callable(cancel):
                        cancel(self.layer_idx)
                    raise
                finish(self.layer_idx, n_tokens)
                return out.view(batch, q_len, heads, head_dim).contiguous(), None
            return None
        cache = request_cache(self.layer_idx, n_tokens)
        try:
            out = self.attn(
                q,
                k,
                v,
                kv_cache=cache,
                update_cache=True,
                causal=causal,
                scale=self.scaling,
            )
        except Exception:
            if callable(cancel):
                cancel(self.layer_idx)
            raise
        finish(self.layer_idx, n_tokens)
        return out.transpose(1, 2).contiguous(), None

    def _attend_paged_transient(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_values: Any,
        *,
        attention_mask: torch.Tensor | None,
        causal: bool = False,
    ) -> tuple[torch.Tensor, None] | None:
        if attention_mask is not None or past_key_values is None or q.ndim != 4:
            return None
        if q.shape[0] != 1 and not getattr(past_key_values, "supports_batched_paged", False):
            return None
        request_cache = getattr(past_key_values, "request_cache_for_transient", None)
        if not callable(request_cache):
            return None
        n_tokens = int(q.shape[2])
        cache = request_cache(self.layer_idx, n_tokens)
        pool = getattr(cache, "pool", None)
        cache_dtype = getattr(pool, "dtype", None)
        if cache_dtype is not None:
            q = q.to(dtype=cache_dtype)
            k = k.to(dtype=cache_dtype)
            v = v.to(dtype=cache_dtype)
        if not self.attn.can_run_transient_paged_varlen(q, k, v, kv_cache=cache):
            if getattr(past_key_values, "supports_batched_paged", False):
                raise capability_mismatch(
                    "batched transient denoise attention requires a paged-varlen attention backend"
                )
            return None
        out = self.attn(
            q,
            k,
            v,
            kv_cache=cache,
            update_cache=True,
            causal=causal,
            scale=self.scaling,
        )
        return out.transpose(1, 2).contiguous(), None

    def forward_und(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Any = None,
        packed_rope: SenseNovaPackedRope | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        causal_paged_update = bool(kwargs.pop("causal_paged_update", False))
        q, k, v = self._project_qkv(
            hidden_states, indexes, gen_branch=False, packed_rope=packed_rope
        )
        if past_key_values is not None:
            update_cache = kwargs.get("update_cache", True)
            if update_cache:
                paged = self._attend_paged_update(
                    q,
                    k,
                    v,
                    past_key_values,
                    attention_mask=attention_mask,
                    causal=causal_paged_update,
                )
                if paged is not None:
                    out, weights = paged
                    return self.o_proj(out.reshape(*input_shape, -1).contiguous()), weights
                k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs=None)
            else:
                layer = past_key_values.layers[self.layer_idx]
                past_k, past_v = layer.keys, layer.values
                if past_k is not None:
                    k = torch.cat([past_k, k], dim=2)
                    v = torch.cat([past_v, v], dim=2)
        out, weights = self._attend_bhld(q, k, v, attention_mask)
        return self.o_proj(out.reshape(*input_shape, -1).contiguous()), weights

    def forward_gen(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Any = None,
        packed_rope: SenseNovaPackedRope | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        q, k, v = self._project_qkv(
            hidden_states, indexes, gen_branch=True, packed_rope=packed_rope
        )
        update_cache = kwargs.get("update_cache", True)
        if attention_mask is None:
            if past_key_values is not None:
                if update_cache:
                    paged = self._attend_paged_update(q, k, v, past_key_values, attention_mask=None)
                    if paged is not None:
                        out, _ = paged
                        return self.o_proj_mot_gen(out.reshape(*input_shape, -1).contiguous()), None
                    k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs=None)
                    q_bshd = q.transpose(1, 2).contiguous()
                    k_attn = k.transpose(1, 2).contiguous()
                    v_attn = v.transpose(1, 2).contiguous()
                else:
                    paged = self._attend_paged_transient(
                        q, k, v, past_key_values, attention_mask=None
                    )
                    if paged is not None:
                        out, _ = paged
                        return self.o_proj_mot_gen(out.reshape(*input_shape, -1).contiguous()), None
                    layer = past_key_values.layers[self.layer_idx]
                    past_k, past_v = layer.keys, layer.values
                    q_bshd = q.transpose(1, 2).contiguous()
                    k_cur = k.transpose(1, 2).contiguous()
                    v_cur = v.transpose(1, 2).contiguous()
                    if past_k is not None:
                        k_attn = torch.cat([past_k.transpose(1, 2).contiguous(), k_cur], dim=1)
                        v_attn = torch.cat([past_v.transpose(1, 2).contiguous(), v_cur], dim=1)
                    else:
                        k_attn, v_attn = k_cur, v_cur
            else:
                q_bshd = q.transpose(1, 2).contiguous()
                k_cur = k.transpose(1, 2).contiguous()
                v_cur = v.transpose(1, 2).contiguous()
                k_attn, v_attn = k_cur, v_cur
            out = self._attend_bshd(q_bshd, k_attn, v_attn)
            return self.o_proj_mot_gen(out.reshape(*input_shape, -1).contiguous()), None

        if past_key_values is not None:
            if update_cache:
                k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs=None)
            else:
                layer = past_key_values.layers[self.layer_idx]
                past_k, past_v = layer.keys, layer.values
                if past_k is not None:
                    k = torch.cat([past_k, k], dim=2)
                    v = torch.cat([past_v, v], dim=2)
        out, weights = self._attend_bhld(q, k, v, attention_mask)
        return self.o_proj_mot_gen(out.reshape(*input_shape, -1).contiguous()), weights

    def _route_o_proj(
        self,
        attn_out: torch.Tensor,
        *,
        text_mask: torch.Tensor,
        gen_mask: torch.Tensor,
        exist_text: bool,
        exist_gen: bool,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter the per-modality output projection via the shared Router.

        ``self.o_proj``/``self.o_proj_mot_gen`` are this attention's per-modality
        experts; routing through :func:`route_by_modality` makes the modality
        dispatch the shared primitive (and tower-aware for mixed batches) instead
        of model-private mask indexing. Only present modalities are routed."""
        routes: dict[Modality, tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]] = {}
        if exist_text:
            routes[Modality.TEXT] = (text_mask, self.o_proj)
        if exist_gen:
            routes[Modality.GEN] = (gen_mask, self.o_proj_mot_gen)
        return route_by_modality(
            attn_out,
            routes,
            out=out,
            transport=self._tower_transport,
            coords=self._tower_coords,
        )

    def _route_o_proj_slices(self, attn_out: torch.Tensor, text_tokens: int) -> torch.Tensor:
        text_tokens = int(text_tokens)
        partial = _cat_token_slices(
            (
                self.o_proj(attn_out[:text_tokens], reduce=False),
                self.o_proj_mot_gen(attn_out[text_tokens:], reduce=False),
            )
        )
        return self.o_proj.reduce_output(partial)

    def forward(
        self,
        hidden_states: torch.Tensor,
        route_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Any = None,
        packed_rope: SenseNovaPackedRope | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            return self.forward_und(
                hidden_states,
                indexes,
                attention_mask,
                past_key_values,
                packed_rope=packed_rope,
                **kwargs,
            )
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            return self.forward_gen(
                hidden_states,
                indexes,
                attention_mask,
                past_key_values,
                packed_rope=packed_rope,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        text_mask = ~route_indicators
        if packed_rope is None:
            packed_rope = self._packed_rope(
                _flatten_3d_indexes(indexes, hidden_states.shape[0], hidden_states.shape[1])
            )
        q, k, v = self._project_qkv_routed(
            hidden_states, indexes, route_indicators, packed_rope=packed_rope
        )

        if past_key_values is not None:
            update_cache = kwargs.get("update_cache", True)
            if update_cache:
                paged = self._attend_paged_update(
                    q, k, v, past_key_values, attention_mask=attention_mask
                )
                if paged is not None:
                    out, weights = paged
                    out = out.reshape(*input_shape, -1).contiguous()
                    routed = self._route_o_proj(
                        out,
                        text_mask=text_mask,
                        gen_mask=route_indicators,
                        exist_text=exist_non_image_gen_tokens,
                        exist_gen=exist_image_gen_tokens,
                        out=out.new_zeros((*input_shape, self.config.hidden_size)),
                    )
                    return routed, weights
                k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs=None)
            else:
                layer = past_key_values.layers[self.layer_idx]
                past_k, past_v = layer.keys, layer.values
                if past_k is not None:
                    k = torch.cat([past_k, k], dim=2)
                    v = torch.cat([past_v, v], dim=2)
        out, weights = self._attend_bhld(q, k, v, attention_mask)
        out = out.reshape(*input_shape, -1).contiguous()
        routed = self._route_o_proj(
            out,
            text_mask=text_mask,
            gen_mask=route_indicators,
            exist_text=exist_non_image_gen_tokens,
            exist_gen=exist_image_gen_tokens,
            out=out.new_zeros((*input_shape, self.config.hidden_size)),
        )
        return routed, weights


class _SenseNovaDecoderLayer(nn.Module):
    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = _SenseNovaAttention(config, layer_idx)
        self.mlp = _SenseNovaMLP(config)
        self.mlp_mot_gen = _SenseNovaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_mot_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_mot_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]
        # Per-modality sublayers as the SenseNova expert flavor, routed by the
        # shared route_by_modality primitive; tower routing is a no-op when trivial.
        self._tower_transport, self._tower_coords = _resolve_tower()
        self._input_norm_by_modality = {
            Modality.TEXT: self.input_layernorm,
            Modality.GEN: self.input_layernorm_mot_gen,
        }
        self._mlp_by_modality = {
            Modality.TEXT: (self.post_attention_layernorm, self.mlp),
            Modality.GEN: (self.post_attention_layernorm_mot_gen, self.mlp_mot_gen),
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        route_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Any = None,
        use_cache: bool | None = False,
        packed_rope: SenseNovaPackedRope | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del use_cache
        if exist_non_image_gen_tokens != exist_image_gen_tokens:
            hidden_states, residual = self.forward_with_residual(
                hidden_states,
                None,
                route_indicators=route_indicators,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                indexes=indexes,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                packed_rope=packed_rope,
                **kwargs,
            )
            if residual is None:
                raise RuntimeError("single-modality layer did not return a residual")
            return hidden_states + residual

        text_mask = ~route_indicators
        residual = hidden_states
        routed = route_by_modality(
            hidden_states,
            {
                Modality.TEXT: (text_mask, self._input_norm_by_modality[Modality.TEXT]),
                Modality.GEN: (route_indicators, self._input_norm_by_modality[Modality.GEN]),
            },
            out=hidden_states.new_zeros(hidden_states.shape),
            transport=self._tower_transport,
            coords=self._tower_coords,
        )
        hidden_states, _ = self.self_attn(
            routed,
            route_indicators,
            exist_non_image_gen_tokens,
            exist_image_gen_tokens,
            indexes,
            attention_mask,
            past_key_values=past_key_values,
            packed_rope=packed_rope,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states

        def _mlp(modality: Modality) -> Callable[[torch.Tensor], torch.Tensor]:
            post_norm, mlp = self._mlp_by_modality[modality]
            return lambda x: mlp(post_norm(x))

        routed = route_by_modality(
            hidden_states,
            {
                Modality.TEXT: (text_mask, _mlp(Modality.TEXT)),
                Modality.GEN: (route_indicators, _mlp(Modality.GEN)),
            },
            out=hidden_states.new_zeros(hidden_states.shape),
            transport=self._tower_transport,
            coords=self._tower_coords,
        )
        return residual + routed

    def forward_with_residual(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        *,
        route_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Any = None,
        packed_rope: SenseNovaPackedRope | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            return self._forward_single_modality_with_residual(
                hidden_states,
                residual,
                route_indicators=route_indicators,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                indexes=indexes,
                attention_mask=attention_mask,
                input_norm=self.input_layernorm,
                post_norm=self.post_attention_layernorm,
                mlp=self.mlp,
                past_key_values=past_key_values,
                packed_rope=packed_rope,
                **kwargs,
            )
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            return self._forward_single_modality_with_residual(
                hidden_states,
                residual,
                route_indicators=route_indicators,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                indexes=indexes,
                attention_mask=attention_mask,
                input_norm=self.input_layernorm_mot_gen,
                post_norm=self.post_attention_layernorm_mot_gen,
                mlp=self.mlp_mot_gen,
                past_key_values=past_key_values,
                packed_rope=packed_rope,
                **kwargs,
            )
        return self(
            hidden_states,
            route_indicators=route_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            packed_rope=packed_rope,
            **kwargs,
        ), None

    def _forward_single_modality_with_residual(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        *,
        route_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: torch.Tensor,
        attention_mask: torch.Tensor | None,
        input_norm: RMSNorm,
        post_norm: RMSNorm,
        mlp: nn.Module,
        past_key_values: Any,
        packed_rope: SenseNovaPackedRope | None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            attn_in = input_norm(hidden_states)
        else:
            attn_in, residual = input_norm.forward_with_residual(
                hidden_states,
                residual,
                in_place=True,
            )
        attn_out, _ = self.self_attn(
            attn_in,
            route_indicators,
            exist_non_image_gen_tokens,
            exist_image_gen_tokens,
            indexes,
            attention_mask,
            past_key_values=past_key_values,
            packed_rope=packed_rope,
            **kwargs,
        )
        mlp_in, residual = post_norm.forward_with_residual(
            attn_out,
            residual,
            in_place=True,
        )
        return mlp(mlp_in), residual

    def forward_packed_visible(
        self,
        hidden_states: torch.Tensor,
        *,
        gen: torch.Tensor,
        und: torch.Tensor,
        und_indices: torch.Tensor | None = None,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: torch.Tensor,
        forward_stream: ForwardStream,
        kv_view: ForwardPagedKVView,
        packed_rope: SenseNovaPackedRope | None = None,
        dense_gen_route: bool = False,
        modality_split: int | None = None,
    ) -> torch.Tensor:
        # gen/und masks and the exist flags are precomputed once by the model
        # and threaded down so each layer avoids per-layer device->host syncs.
        if hidden_states.ndim != 2:
            raise ValueError("SenseNova packed layer expects hidden_states [N, C]")

        residual = hidden_states
        ctx = get_forward_context()
        norm_start = ctx.component_timer_start()
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            routed = self.input_layernorm(hidden_states)
        elif not exist_non_image_gen_tokens and exist_image_gen_tokens:
            routed = self.input_layernorm_mot_gen(hidden_states)
        elif modality_split is not None:
            split = int(modality_split)
            routed = _cat_token_slices(
                (
                    self.input_layernorm(hidden_states[:split]),
                    self.input_layernorm_mot_gen(hidden_states[split:]),
                )
            )
        elif dense_gen_route:
            routed = self.input_layernorm_mot_gen(hidden_states)
            if und_indices is None:
                und_indices = und.nonzero(as_tuple=False).flatten()
            routed.index_copy_(
                0,
                und_indices,
                self.input_layernorm(hidden_states.index_select(0, und_indices)),
            )
        else:
            routed = route_by_modality(
                hidden_states,
                {
                    Modality.TEXT: (und, self._input_norm_by_modality[Modality.TEXT]),
                    Modality.GEN: (gen, self._input_norm_by_modality[Modality.GEN]),
                },
                out=hidden_states.new_empty(hidden_states.shape),
                transport=self._tower_transport,
                coords=self._tower_coords,
            )
        ctx.record_component_elapsed("packed_decoder_input_norm", norm_start)

        attn_block_start = ctx.component_timer_start()
        attn_out = self.self_attn.forward_packed_visible(
            routed,
            gen,
            indexes,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            und=und,
            und_indices=und_indices,
            forward_stream=forward_stream,
            kv_view=kv_view,
            packed_rope=packed_rope,
            dense_gen_route=dense_gen_route,
            modality_split=modality_split,
        )
        ctx.record_component_elapsed("packed_decoder_attn_block", attn_block_start)
        if modality_split is None:
            hidden_states = residual + attn_out
            residual = hidden_states

        mlp_start = ctx.component_timer_start()
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            mlp_out = self.mlp(self.post_attention_layernorm(hidden_states))
        elif not exist_non_image_gen_tokens and exist_image_gen_tokens:
            mlp_out = self.mlp_mot_gen(self.post_attention_layernorm_mot_gen(hidden_states))
        elif modality_split is not None:
            split = int(modality_split)
            text_mlp_in, text_residual = self.post_attention_layernorm.forward_with_residual(
                attn_out[:split],
                residual[:split],
            )
            gen_mlp_in, gen_residual = self.post_attention_layernorm_mot_gen.forward_with_residual(
                attn_out[split:],
                residual[split:],
            )
            residual = _cat_token_slices((text_residual, gen_residual))
            partial = _cat_token_slices(
                (
                    self.mlp.down_proj(
                        self.mlp.act(self.mlp.gate_up_proj(text_mlp_in)),
                        reduce=False,
                    ),
                    self.mlp_mot_gen.down_proj(
                        self.mlp_mot_gen.act(self.mlp_mot_gen.gate_up_proj(gen_mlp_in)),
                        reduce=False,
                    ),
                )
            )
            mlp_out = self.mlp.down_proj.reduce_output(partial)
        elif dense_gen_route:
            mlp_out = self.mlp_mot_gen(self.post_attention_layernorm_mot_gen(hidden_states))
            if und_indices is None:
                und_indices = und.nonzero(as_tuple=False).flatten()
            text_hidden = hidden_states.index_select(0, und_indices)
            text_mlp = self.mlp(self.post_attention_layernorm(text_hidden))
            mlp_out.index_copy_(0, und_indices, text_mlp)
        else:

            def _mlp(modality: Modality) -> Callable[[torch.Tensor], torch.Tensor]:
                post_norm, mlp = self._mlp_by_modality[modality]
                return lambda x: mlp(post_norm(x))

            mlp_out = route_by_modality(
                hidden_states,
                {
                    Modality.TEXT: (und, _mlp(Modality.TEXT)),
                    Modality.GEN: (gen, _mlp(Modality.GEN)),
                },
                out=hidden_states.new_empty(hidden_states.shape),
                transport=self._tower_transport,
                coords=self._tower_coords,
            )
        ctx.record_component_elapsed("packed_decoder_mlp", mlp_start)
        return residual + mlp_out


class _SenseNovaDecoderModel(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [_SenseNovaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_mot_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Final-norm routing uses the shared primitive; tower-aware when split.
        self._tower_transport, self._tower_coords = _resolve_tower()
        self._final_norm_by_modality = {
            Modality.TEXT: self.norm,
            Modality.GEN: self.norm_mot_gen,
        }

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        route_indicators: torch.Tensor | None = None,
        indexes: torch.Tensor | None = None,
        attention_mask: Any = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        exist_non_image_gen_tokens: bool | None = None,
        exist_image_gen_tokens: bool | None = None,
        text_only_rope: bool = False,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        del position_ids
        inputs_embeds = self._resolve_inputs_embeds(input_ids, inputs_embeds)
        if route_indicators is None:
            exist_non_image_gen_tokens = (
                True if exist_non_image_gen_tokens is None else bool(exist_non_image_gen_tokens)
            )
            exist_image_gen_tokens = (
                False if exist_image_gen_tokens is None else bool(exist_image_gen_tokens)
            )
            if exist_non_image_gen_tokens != exist_image_gen_tokens:
                route_indicators = torch.empty(0, dtype=torch.bool, device=inputs_embeds.device)
            else:
                route_indicators = self._resolve_route_indicators(None, inputs_embeds)
        else:
            route_indicators = self._resolve_route_indicators(route_indicators, inputs_embeds)
            if exist_non_image_gen_tokens is None:
                exist_non_image_gen_tokens = bool((~route_indicators).any())
            if exist_image_gen_tokens is None:
                exist_image_gen_tokens = bool(route_indicators.any())
        if use_cache and past_key_values is None:
            raise RuntimeError("native decoder serving requires an explicit paged cache")
        if indexes is None:
            cache_position = self._resolve_cache_position(
                cache_position, past_key_values, inputs_embeds
            )
        elif cache_position is None:
            cache_position = torch.empty(0, dtype=torch.long, device=inputs_embeds.device)
        indexes, causal_mask_mapping = self._resolve_indexes_and_masks(
            indexes,
            attention_mask,
            inputs_embeds,
            cache_position,
        )
        flat_indexes = _flatten_3d_indexes(indexes, inputs_embeds.shape[0], inputs_embeds.shape[1])
        packed_rope = (
            cast(_SenseNovaDecoderLayer, self.layers[0]).self_attn._packed_rope(
                flat_indexes, hw_identity=bool(text_only_rope)
            )
            if self.layers
            else None
        )

        if exist_non_image_gen_tokens != exist_image_gen_tokens:
            hidden_states = self._forward_single_modality_layers(
                inputs_embeds,
                route_indicators=route_indicators,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                indexes=indexes,
                causal_mask_mapping=causal_mask_mapping,
                past_key_values=past_key_values,
                cache_position=cache_position,
                packed_rope=packed_rope,
                **kwargs,
            )
        else:
            hidden_states = self._forward_mixed_modality_layers(
                inputs_embeds,
                route_indicators=route_indicators,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                indexes=indexes,
                causal_mask_mapping=causal_mask_mapping,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                packed_rope=packed_rope,
                **kwargs,
            )
        return BaseModelOutputWithPast(
            last_hidden_state=cast(Any, hidden_states),
            past_key_values=past_key_values if use_cache else None,
        )

    def _resolve_inputs_embeds(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("exactly one of input_ids or inputs_embeds is required")
        return self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds

    def _resolve_route_indicators(
        self,
        route_indicators: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if route_indicators is not None:
            return route_indicators
        return torch.zeros(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)

    def _resolve_cache_position(
        self,
        cache_position: torch.Tensor | None,
        past_key_values: Any,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if cache_position is not None:
            return cache_position
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        return torch.arange(
            past_seen,
            past_seen + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    def _resolve_indexes_and_masks(
        self,
        indexes: torch.Tensor | None,
        attention_mask: Any,
        inputs_embeds: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if indexes is None:
            indexes = self._indexes_from_cache_position(cache_position, inputs_embeds)
        if isinstance(attention_mask, dict):
            return indexes, attention_mask
        return indexes, {"full_attention": None}

    @staticmethod
    def _indexes_from_cache_position(
        cache_position: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        batch = int(inputs_embeds.shape[0])
        seq_len = int(inputs_embeds.shape[1])
        positions = cache_position.to(device=inputs_embeds.device, dtype=torch.long).reshape(-1)
        if positions.numel() == seq_len:
            zeros = torch.zeros_like(positions)
            return torch.stack((positions, zeros, zeros), dim=0)
        if positions.numel() != batch * seq_len:
            raise ValueError(
                "cache_position must provide one position per token when indexes are omitted"
            )
        positions = positions.view(batch, seq_len)
        zeros = torch.zeros_like(positions)
        return torch.stack((positions, zeros, zeros), dim=0)

    def _forward_single_modality_layers(
        self,
        hidden_states: torch.Tensor,
        *,
        route_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: torch.Tensor,
        causal_mask_mapping: dict[str, Any],
        past_key_values: Any,
        cache_position: torch.Tensor,
        packed_rope: SenseNovaPackedRope | None,
        pre_norm_out: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = None
        for layer_module in self.layers:
            layer = cast(_SenseNovaDecoderLayer, layer_module)
            hidden_states, residual = layer.forward_with_residual(
                hidden_states,
                residual,
                route_indicators=route_indicators,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                indexes=indexes,
                attention_mask=causal_mask_mapping[layer.attention_type],
                past_key_values=past_key_values,
                cache_position=cache_position,
                packed_rope=packed_rope,
                **kwargs,
            )
        norm = self.norm_mot_gen if exist_image_gen_tokens else self.norm
        if pre_norm_out is not None:
            # Flow residual-reuse capture: expose the pre-final-norm stream
            # (block-stack output) alongside the normal normalized output.
            pre_norm = hidden_states if residual is None else hidden_states + residual
            pre_norm_out.append(pre_norm)
            return norm(pre_norm)
        if residual is None:
            return norm(hidden_states)
        hidden_states, _ = norm.forward_with_residual(hidden_states, residual, in_place=True)
        return hidden_states

    def _forward_mixed_modality_layers(
        self,
        hidden_states: torch.Tensor,
        *,
        route_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: torch.Tensor,
        causal_mask_mapping: dict[str, Any],
        past_key_values: Any,
        use_cache: bool | None,
        cache_position: torch.Tensor,
        packed_rope: SenseNovaPackedRope | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        for layer_module in self.layers:
            layer = cast(_SenseNovaDecoderLayer, layer_module)
            hidden_states = layer(
                hidden_states,
                route_indicators=route_indicators,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                indexes=indexes,
                attention_mask=causal_mask_mapping[layer.attention_type],
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                packed_rope=packed_rope,
                **kwargs,
            )
        return route_by_modality(
            hidden_states,
            {
                Modality.TEXT: (~route_indicators, self._final_norm_by_modality[Modality.TEXT]),
                Modality.GEN: (route_indicators, self._final_norm_by_modality[Modality.GEN]),
            },
            out=hidden_states.new_zeros(hidden_states.shape),
            transport=self._tower_transport,
            coords=self._tower_coords,
        )

    def forward_packed_visible(
        self,
        inputs_embeds: torch.Tensor,
        *,
        route_indicators: torch.Tensor,
        indexes: torch.Tensor,
        forward_stream: ForwardStream,
        kv_view: ForwardPagedKVView,
    ) -> torch.Tensor:
        if inputs_embeds.ndim != 2:
            raise ValueError("SenseNova packed model expects inputs_embeds [N, C]")
        if route_indicators.shape != (inputs_embeds.shape[0],):
            raise ValueError("SenseNova packed model expects route_indicators [N]")
        if indexes.shape != (3, inputs_embeds.shape[0]):
            raise ValueError("SenseNova packed model expects indexes [3, N]")
        gen = route_indicators.to(dtype=torch.bool)
        und = ~gen
        und_indices = forward_stream.und_indices
        exist_non_image_gen_tokens = any(
            seg.modality == "und" and int(seg.q_len) > 0 for seg in forward_stream.segments
        )
        exist_image_gen_tokens = any(
            seg.modality == "gen" and int(seg.q_len) > 0 for seg in forward_stream.segments
        )
        packed_rope = (
            cast(_SenseNovaDecoderLayer, self.layers[0]).self_attn._packed_rope(indexes)
            if self.layers
            else None
        )
        und_tokens = sum(seg.q_len for seg in forward_stream.segments if seg.modality == "und")
        gen_tokens = sum(seg.q_len for seg in forward_stream.segments if seg.modality == "gen")
        modality_split = (
            _contiguous_route_split(forward_stream, inputs_embeds.shape[0])
            if self._tower_transport is None
            else None
        )
        dense_gen_route = (
            exist_non_image_gen_tokens
            and exist_image_gen_tokens
            and gen_tokens >= und_tokens
            and self._tower_transport is None
            and modality_split is None
        )

        hidden_states = inputs_embeds
        # The und/gen split is constant for the whole packed batch, so compute
        # the masks/exist flags once here and thread them into every layer
        # instead of re-deriving them per layer (each bool(.any()) on a CUDA
        # tensor forces a host sync, serializing the kernel chain).
        for layer_module in self.layers:
            layer = cast(_SenseNovaDecoderLayer, layer_module)
            hidden_states = layer.forward_packed_visible(
                hidden_states,
                gen=gen,
                und=und,
                und_indices=und_indices,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                indexes=indexes,
                forward_stream=forward_stream,
                kv_view=kv_view,
                packed_rope=packed_rope,
                dense_gen_route=dense_gen_route,
                modality_split=modality_split,
            )
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            return self.norm(hidden_states)
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            return self.norm_mot_gen(hidden_states)
        if dense_gen_route:
            out = self.norm_mot_gen(hidden_states)
            if und_indices is None:
                und_indices = und.nonzero(as_tuple=False).flatten()
            out.index_copy_(
                0,
                und_indices,
                self.norm(hidden_states.index_select(0, und_indices)),
            )
            return out
        if modality_split is not None:
            split = int(modality_split)
            return _cat_token_slices(
                (
                    self.norm(hidden_states[:split]),
                    self.norm_mot_gen(hidden_states[split:]),
                )
            )
        return route_by_modality(
            hidden_states,
            {
                Modality.TEXT: (und, self._final_norm_by_modality[Modality.TEXT]),
                Modality.GEN: (gen, self._final_norm_by_modality[Modality.GEN]),
            },
            out=hidden_states.new_empty(hidden_states.shape),
            transport=self._tower_transport,
            coords=self._tower_coords,
        )


class _SenseNovaLanguageModel(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.model = _SenseNovaDecoderModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = ParallelLMHead(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, VocabParallelEmbedding):
            raise TypeError("SenseNova input embeddings must be VocabParallelEmbedding")
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, ParallelLMHead):
            raise TypeError("SenseNova output embeddings must be ParallelLMHead")
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        indexes: torch.Tensor | None = None,
        attention_mask: Any = None,
        past_key_values: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            indexes=indexes,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :])
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.last_hidden_state,
        )


class NEOChatModel(nn.Module):
    """Native SenseNova language model with vision and flow-matching heads."""

    config_class = NeoChatConfig

    def __init__(self, config: NeoChatConfig) -> None:
        super().__init__()
        self.config = config
        patch_size = int(config.vision_config.patch_size)
        self.patch_size = patch_size
        self.template = config.template
        self.prompt_renderer = _resolve_prompt_renderer(config.template)
        self.downsample_ratio = config.downsample_ratio
        # Enter the checkpoint quantization context for all layer construction so
        # the model is self-contained instead of relying on an ambient context.
        self._quant_config = QuantizationConfig.from_model_config(config)
        merge_size = int(1 / self.downsample_ratio)
        output_dim = 3 * (patch_size * merge_size) ** 2
        hidden = int(config.llm_config.hidden_size)
        self.use_deep_fm_head = bool(config.fm_head_layers > 2)
        self.use_pixel_head = bool(config.use_pixel_head)
        with use_quantization_config(self._quant_config):
            self._build_backbone(config, hidden=hidden, output_dim=output_dim)

        self._init_flow_params(config, hidden=hidden)

        self.img_context_token_id = None
        self.system_message = ""

    def _build_fm_head(self, config: NeoChatConfig, *, hidden: int, output_dim: int) -> nn.Module:
        """Construct the flow-matching head module (deep vs. shallow variant).

        Called inside the checkpoint quantization context so its ``LinearBase``
        layers pick up the active quant config.
        """
        if self.use_deep_fm_head:
            return FlowMatchingHead(
                hidden,
                output_dim,
                dim=config.fm_head_dim,
                layers=config.fm_head_layers,
                mlp_ratio=config.fm_head_mlp_ratio,
            )
        return nn.Sequential(
            LinearBase(hidden, 4096, bias=True),
            nn.GELU(),
            LinearBase(4096, output_dim, bias=True),
        )

    def _build_backbone(self, config: NeoChatConfig, *, hidden: int, output_dim: int) -> None:
        """Build and register the vision towers, language model, and fm modules.

        Must run inside the ``use_quantization_config`` context so every
        constructed layer is materialized against the checkpoint quant config.
        """
        self.vision_model = NeoVisionModel(config.vision_config)
        vision_model_mot_gen = NeoVisionModel(config.vision_config)
        self.language_model = _SenseNovaLanguageModel(config.llm_config)

        fm_head = self._build_fm_head(config, hidden=hidden, output_dim=output_dim)

        self.fm_modules = nn.ModuleDict(
            {
                "vision_model_mot_gen": vision_model_mot_gen,
                "timestep_embedder": TimestepEmbedder(hidden),
                "fm_head": fm_head,
            }
        )
        if self.use_pixel_head:
            self.fm_modules["fm_head"] = ConvDecoder(hidden)

    def _init_flow_params(self, config: NeoChatConfig, *, hidden: int) -> None:
        """Copy the flow-matching scalar config and add the optional embedder."""
        self.concat_time_token_num = config.concat_time_token_num
        self.noise_scale = config.noise_scale
        self.noise_scale_mode = config.noise_scale_mode
        self.noise_scale_base_image_seq_len = config.noise_scale_base_image_seq_len
        self.add_noise_scale_embedding = config.add_noise_scale_embedding
        self.noise_scale_max_value = config.noise_scale_max_value
        self.time_schedule = config.time_schedule
        self.time_shift_type = config.time_shift_type
        self.base_shift = config.base_shift
        self.max_shift = config.max_shift
        self.base_image_seq_len = config.base_image_seq_len
        self.max_image_seq_len = config.max_image_seq_len
        if self.add_noise_scale_embedding:
            self.fm_modules["noise_scale_embedder"] = TimestepEmbedder(hidden)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def extract_feature(
        self,
        pixel_values: torch.Tensor,
        gen_model: bool = False,
        grid_hw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tower = self.fm_modules["vision_model_mot_gen"] if gen_model else self.vision_model
        return tower(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
            grid_hw=grid_hw,
        ).last_hidden_state

    def _build_t2i_query(
        self,
        prompt_text: str,
        system_message: str | None = None,
        append_text: str | None = None,
    ) -> str:
        return self.prompt_renderer.render(
            prompt_text,
            system_message=self.system_message if system_message is None else system_message,
            append_text=append_text,
        )

    def _build_t2i_text_inputs(self, tokenizer: Any, query: str):
        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)
        t_idx = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        h_idx = torch.zeros_like(t_idx)
        w_idx = torch.zeros_like(t_idx)
        indexes = torch.stack([t_idx, h_idx, w_idx], dim=0)
        from ...runtime.masks import create_block_causal_mask

        return input_ids, indexes, {"full_attention": create_block_causal_mask(indexes[0])}

    def _build_t2i_image_indexes(
        self,
        token_h: int,
        token_w: int,
        text_len: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        t_image = torch.full((token_h * token_w,), text_len, dtype=torch.long, device=device)
        idx = torch.arange(token_h * token_w, device=device, dtype=torch.long)
        h_image = idx // token_w
        w_image = idx % token_w
        return torch.stack([t_image, h_image, w_image], dim=0)

    def _t2i_predict_v(
        self,
        input_embeds: torch.Tensor,
        indexes_image: torch.Tensor,
        attn_mask: Any,
        past_key_values: Any,
        t: torch.Tensor,
        z: torch.Tensor,
        image_token_num: int,
        timestep_embeddings: torch.Tensor | None = None,
        image_size: tuple[int, int] | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # timestep_embeddings is part of the cross-file _t2i_predict_v call
        # contract (flow execution and reference paths pass it) but
        # this native path conditions on t inside _t2i_hidden_to_x_pred, so the
        # precomputed embedding is unused here.
        del timestep_embeddings
        pre_norm_out: list[torch.Tensor] | None = [] if return_hidden else None
        outputs = self.language_model.model(
            inputs_embeds=input_embeds,
            route_indicators=torch.ones(
                input_embeds.shape[:2],
                dtype=torch.bool,
                device=input_embeds.device,
            ),
            indexes=indexes_image,
            attention_mask=attn_mask,
            past_key_values=past_key_values,
            update_cache=False,
            use_cache=True,
            pre_norm_out=pre_norm_out,
            # The indicators above are all-ones by construction; passing the
            # exist flags explicitly skips the decoder's ``bool(mask.any())``
            # derivation — a per-step device->host sync that is also illegal
            # inside denoise-step CUDA graph capture.
            exist_non_image_gen_tokens=False,
            exist_image_gen_tokens=True,
        )
        x_pred = self._t2i_hidden_to_x_pred(
            outputs.last_hidden_state,
            t,
            z,
            image_token_num=image_token_num,
            image_size=image_size,
        )
        velocity = (x_pred - z) / (1 - t).clamp_min(self.config.t_eps)
        if return_hidden:
            if not pre_norm_out:
                raise RuntimeError("denoise forward did not capture the pre-norm stream")
            # The residual-reuse contract caches the *pre-final-norm* stream;
            # the replay path re-applies the final norm via the adapter.
            return velocity, pre_norm_out[0]
        return velocity

    def _t2i_hidden_to_x_pred(
        self,
        hidden_states: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        *,
        image_token_num: int,
        image_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        batch, latent_len = z.shape[0], z.shape[1]
        if self.use_pixel_head:
            if image_size is None:
                raise RuntimeError("pixel head requires image_size")
            merge_size = int(1 / self.downsample_ratio)
            token_h = image_size[1] // (self.patch_size * merge_size)
            token_w = image_size[0] // (self.patch_size * merge_size)
            img = hidden_states[:, -image_token_num:].view(batch, token_h, token_w, -1)
            img = torch.einsum("b h w c -> b c h w", img).contiguous()
            smoothed = self.fm_modules["fm_head"](img.view(batch, -1, token_h, token_w))
            smoothed = smoothed.view(
                batch,
                3,
                token_h,
                self.patch_size * merge_size,
                token_w,
                self.patch_size * merge_size,
            )
            smoothed = torch.einsum("b c h p w q -> b h w p q c", smoothed)
            return smoothed.contiguous().view(
                batch,
                latent_len,
                self.patch_size * merge_size * self.patch_size * merge_size * 3,
            )
        if self.use_deep_fm_head:
            return self.fm_modules["fm_head"](
                hidden_states[:, -image_token_num:].view(batch * latent_len, -1),
                t.repeat(batch * latent_len),
            ).view(batch, latent_len, -1)
        return self.fm_modules["fm_head"](
            hidden_states[:, -image_token_num:].view(batch, latent_len, -1)
        ).view(batch, latent_len, -1)

    def _t2i_hidden_to_velocity(
        self,
        hidden_states: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        *,
        image_token_num: int,
        image_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        x_pred = self._t2i_hidden_to_x_pred(
            hidden_states,
            t,
            z,
            image_token_num=image_token_num,
            image_size=image_size,
        )
        return (x_pred - z) / (1 - t).clamp_min(self.config.t_eps)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()


def check_checkpoint_compatibility(config_or_dict: Any) -> None:
    try:
        from packaging.version import Version
    except ImportError:  # pragma: no cover
        return
    cfg = config_or_dict.to_dict() if hasattr(config_or_dict, "to_dict") else config_or_dict
    if not isinstance(cfg, dict):
        return
    required = cfg.get("uniserve_sensenova_min_version")
    if required and Version(SENSENOVA_MODEL_CODE_VERSION) < Version(str(required)):
        raise RuntimeError(f"checkpoint requires UniServe model code >= {required}")


class SenseNovaU1ForUnifiedGeneration(
    UniModelBase,
):
    """SenseNova-U1 serving model: text prefill/decode, image denoise, and commit."""

    family = "sensenova"
    architectures = ("NEOChatModel", "neo_chat", "neo-unify", "neo_unify")
    supported_model_load_scopes = ("understanding", "generation")
    supported_ops = (
        "prefill_und",
        "decode_und",
        "denoise_gen",
        "commit_gen",
        "commit_writeback",
        "vit_encode",
    )
    supported_controls: tuple[str, ...] = ("free_encoder",)
    adapter_mode = "none"
    resource_plan = ResourcePlan(
        kv_block=KvBlockResourcePolicy.PER_BLOCK,
        encoder_output=EncoderResourcePolicy.PER_HANDLE,
        image_latent=LatentTokens(downsample=16),
        scratch=PerBranch(),
    )
    ENCODER_CACHE_BUDGET = DEFAULT_ENCODER_CACHE_BUDGET
    checkpoint_layout = CheckpointLayout(stacked=_SENSENOVA_STACKED_PARAMS)

    def velocity_parameterization(self) -> str:
        return "velocity"

    # Flow configuration consumed by the system FlowExecution engine.
    denoise_schedule_direction = ScheduleDirection.ASCENDING
    denoise_schedule_shift_domain = ScheduleShiftDomain.SIGMA
    denoise_cfg_recipe = CfgRecipe.ADDITIVE_DELTAS

    def __init__(
        self,
        config: Any | None = None,
        *,
        model=None,
        tokenizer=None,
        device: str = "cpu",
        gen_snapshot_kv_capacity: int | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        kv_token_capacity: int | None = None,
        attention_backend: str | None = None,
        tower_role: str | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = str(device)
        if self.config is not None and hasattr(self.config, "t_eps"):
            self.config.t_eps = GENERATION_T_EPS
        # Tower device profile: when set ("und"/"gen"), only this tower's modules
        # were materialized by the loader; the other tower's params stay on
        # ``meta`` (no memory, never read by this worker's ops).
        # ``None`` (default / single-device) materializes the whole model.
        self.tower_role = tower_role
        self._init_tower_profile()
        self.gen_snapshot_kv_capacity = gen_snapshot_kv_capacity
        self.block_size = int(block_size)
        self.kv_token_capacity = kv_token_capacity
        self.attention_backend = attention_backend or "auto"
        self.runner_states: dict[int, RunnerRequestState] = {}
        # Per-request program state, keyed by req_id and
        # cleared in drop_request (the authoritative owner).
        self.reqs: dict[int, ProgramState] = {}

        llm_cfg = self._init_token_geometry(config)
        self.resource_plan = ResourcePlan(
            kv_block=KvBlockResourcePolicy.PER_BLOCK,
            encoder_output=EncoderResourcePolicy.PER_HANDLE,
            image_latent=LatentTokens(downsample=int(self.latent_downsample)),
            scratch=PerBranch(),
        )

        n_kv, head_dim = self._init_kv_geometry(config, llm_cfg, kv_token_capacity)
        self._init_empty_residency_state()
        if self.model is not None:
            self._init_loaded_model_residency(n_kv, head_dim, gen_snapshot_kv_capacity)
            self._maybe_compile_piecewise()
        # The und↔gen crossing is one model-facing object over the tower axis.
        # In-process transport binds NVLink peer copy; a trivial tower binds None
        # and degrades to a same-device scratch copy. The binding resolves live
        # so destination residency tracks the tower-vs-trivial choice.
        self._tower_handoff: TowerHandoff = LocalP2PTowerHandoff(self._resolve_tower_binding)
        self.tower_session = ProductTransferSession(self, states=self.reqs)
        self.flow_execution = FlowExecution(self, transfer=self.tower_session)
        self.segment_executor = SegmentExecutor(self)
        self._img_start_token = IMG_START_TOKEN

    def _init_tower_profile(self) -> None:
        # The generation tower is the ``tower`` mesh axis. A trivial/absent tower
        # leaves coords/transport unset and ``gen_device == device``.
        self.mesh = get_current_mesh()
        self._tower_coords = tower_modality_coords(self.mesh)
        tower_axis = self.mesh.axis("tower")
        self._tower_transport = (
            tower_axis.transport
            if self._tower_coords is not None and tower_axis is not None
            else None
        )
        tower_devices = (
            getattr(self._tower_transport, "devices", None)
            if self._tower_transport is not None
            else None
        )
        self.gen_device = (
            str(tower_devices[self._tower_coords[Modality.GEN]])
            if tower_devices is not None and self._tower_coords is not None
            else self.device
        )

    def _init_token_geometry(self, config: Any | None) -> Any:
        if self.model is not None and self.tokenizer is not None:
            self.img_start_id = self.tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
            self.img_end_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
            self.eos_id = self.tokenizer.eos_token_id
            self.merge_size = int(1 / self.model.downsample_ratio)
            self.latent_downsample = int(self.model.patch_size * self.merge_size)
            self.max_latent_size = int(self.model.config.max_image_seq_len)
            return getattr(self.model.config, "llm_config", None)
        self.img_start_id = 0
        self.img_end_id = 0
        self.eos_id = 0
        self.merge_size = 2
        self.latent_downsample = 16
        self.max_latent_size = _config_int(config, "max_image_seq_len", 0)
        if isinstance(config, dict):
            return config.get("llm_config") or {}
        return getattr(config, "llm_config", None) if config is not None else None

    def _init_kv_geometry(
        self,
        config: Any | None,
        llm_cfg: Any,
        kv_token_capacity: int | None,
    ) -> tuple[int, int]:
        if isinstance(llm_cfg, dict):
            self.num_layers = int(llm_cfg["num_hidden_layers"])
            n_kv = int(llm_cfg["num_key_value_heads"])
            head_dim = int(llm_cfg["head_dim"])
        else:
            self.num_layers = int(llm_cfg.num_hidden_layers)
            n_kv = int(llm_cfg.num_key_value_heads)
            head_dim = int(llm_cfg.head_dim)
        # KV pools store this rank's shard: divide by tp_size with the same
        # "a KV group too small to split stays whole" rule QKVParallelLinear
        # applies, so pool geometry always matches what sharded attention writes.
        n_kv = _local_kv_head_count(n_kv)
        self.kv_cache_dtype = self._requested_kv_cache_dtype_for(config)
        if self.kv_cache_dtype in {None, "auto", "native", "compute"}:
            self.kv_cache_dtype = "bf16"
        self._kv_num_heads = n_kv
        self._kv_head_dim = head_dim
        self.bytes_per_token = self._kv_bytes_per_token(torch.bfloat16)
        self.num_blocks = derive_num_blocks(self.block_size, kv_token_capacity)
        return n_kv, head_dim

    def _init_empty_residency_state(self) -> None:
        self.kv_pool: PagedKVPool | None = None
        self.scratch_pool: PagedKVPool | None = None
        self.gen_scratch_pool: PagedKVPool | None = None
        self._scratch_blocks = 0
        self.residency = ResidencyManager(encoder_cache_budget=self.ENCODER_CACHE_BUDGET)

    def _init_loaded_model_residency(
        self,
        n_kv: int,
        head_dim: int,
        gen_snapshot_kv_capacity: int | None,
    ) -> None:
        dtype = torch.bfloat16
        self._annotate_towers()
        place_towers(self.model, self.mesh)
        self._ensure_rope_buffers_on_device(torch.device(str(self.device)))
        self.bytes_per_token = self._kv_bytes_per_token(dtype)
        scratch_blocks, gen_blocks = self._scratch_block_counts(gen_snapshot_kv_capacity)
        self.residency = ResidencyManager.build_gen(
            GenResidencySpec(
                kv=KvCacheSpec(
                    num_layers=self.num_layers,
                    num_kv_heads=n_kv,
                    head_dim=head_dim,
                    dtype=dtype,
                    store_dtype=self._kv_store_dtype_for(dtype),
                ),
                num_blocks=self.num_blocks,
                block_size=self.block_size,
                device=self.device,
                scratch_num_blocks=scratch_blocks,
                reserved_tail_blocks=decode_graph_padding_block_count(self.block_size),
                gen_scratch_num_blocks=gen_blocks,
                gen_device=self.gen_device if gen_blocks is not None else None,
                gen_tower_coord=(
                    self._tower_coords[Modality.GEN] if self._tower_coords is not None else None
                ),
                encoder_cache_budget=self.ENCODER_CACHE_BUDGET,
            )
        )
        self.kv_pool = self.residency.kv
        self.scratch_pool = self.residency.scratch
        self.gen_scratch_pool = self.residency.gen_scratch
        self._scratch_blocks = scratch_blocks

    def _ensure_rope_buffers_on_device(self, device: torch.device | str) -> None:
        if self.model is None:
            return
        language = getattr(self.model, "language_model", None)
        decoder = getattr(language, "model", None)
        layers = getattr(decoder, "layers", None)
        if layers is None:
            return
        target = torch.device(device)
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            for name in ("rotary_emb", "rotary_emb_hw"):
                rope = getattr(attn, name, None)
                inv_freq = getattr(rope, "inv_freq", None)
                if (
                    isinstance(rope, nn.Module)
                    and isinstance(inv_freq, torch.Tensor)
                    and inv_freq.device != target
                ):
                    rope.to(target)

    def _scratch_block_counts(self, gen_snapshot_kv_capacity: int | None) -> tuple[int, int | None]:
        image_scratch_blocks = max(
            1,
            ceil_div(self._active_latent_capacity_tokens(self.kv_token_capacity), self.block_size),
        )
        scratch_blocks = max(8, self.num_blocks + image_scratch_blocks * 4)
        gen_blocks: int | None = None
        if self._tower_coords is not None:
            gen_blocks = scratch_blocks
            if gen_snapshot_kv_capacity is not None:
                gen_blocks = max(1, int(gen_snapshot_kv_capacity) // self.block_size)
        return scratch_blocks, gen_blocks

    def _active_latent_capacity_tokens(self, kv_token_capacity: int | None) -> int:
        """Total concurrently resident image-latent tokens this worker advertises."""
        return active_latent_capacity_tokens(self.max_latent_size, kv_token_capacity)

    @classmethod
    def native_load_spec(cls) -> NativeLoadSpec:
        """Declare how the native HF checkpoint is materialized.

        ``NativeTransformersLoader`` drives the meta-init + per-tensor streaming
        from this spec; ``from_native`` then builds the serving wrapper.
        """
        from transformers import AutoTokenizer

        return NativeLoadSpec(
            config_cls=NeoChatConfig,
            model_cls=NEOChatModel,
            tokenizer_cls=AutoTokenizer,
            config_patch=None,
            compatibility_check=check_checkpoint_compatibility,
            param_filter_from_model=cls.tower_role_param_filter_from_model,
            checkpoint_layout=cls.checkpoint_layout,
        )

    @classmethod
    def tower_role_param_filter_from_model(cls, model: nn.Module, tower_role: str | None) -> Any:
        """Map a tower role to a checkpoint-param predicate for partial load.

        ``"gen"`` keeps only the generation-tower params; ``"und"`` keeps the
        complement (embed/lm_head/und attn-mlp-norm/model norm/und ViT). ``None``
        returns ``None`` (no filter — the whole model loads).
        """
        return _TOWER_LAYOUT.filter_from_model(model, tower_role)

    @classmethod
    def from_native(
        cls,
        inner: nn.Module,
        *,
        tokenizer: Any,
        device: str,
        gen_snapshot_kv_capacity: int | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        kv_token_capacity: int | None = None,
        attention_backend: str | None = None,
        tower_role: str | None = None,
        **_kwargs: Any,
    ) -> "SenseNovaU1ForUnifiedGeneration":
        """Wrap the natively-materialized inner model into the serving model."""
        return cls(
            inner.config,
            model=inner,
            tokenizer=tokenizer,
            device=device,
            gen_snapshot_kv_capacity=gen_snapshot_kv_capacity,
            block_size=block_size,
            kv_token_capacity=kv_token_capacity,
            attention_backend=attention_backend,
            tower_role=tower_role,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device: str,
        gen_snapshot_kv_capacity: int | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        kv_token_capacity: int | None = None,
        attention_backend: str | None = None,
        tower_role: str | None = None,
        **_kwargs: Any,
    ) -> "SenseNovaU1ForUnifiedGeneration":
        # Route the heavy materialization through the registered native loader so
        # it is governed by the BaseModelLoader contract; ``from_native`` builds
        # the serving wrapper from the loader's result. ``tower_role`` selects the
        # loader's partial-load filter and is threaded to the wrapper.
        from ...loader import get_loader

        loaded = (
            get_loader("native")
            .load_model(
                cast(Any, cls),
                None,
                device=device,
                model_path=model_path,
                gen_snapshot_kv_capacity=gen_snapshot_kv_capacity,
                block_size=block_size,
                kv_token_capacity=kv_token_capacity,
                attention_backend=attention_backend,
                tower_role=tower_role,
            )
            .model
        )
        if not isinstance(loaded, cls):
            raise TypeError("native loader returned the wrong SenseNova model type")
        return loaded

    def _caps_descriptor(
        self,
        *,
        block_size: int | None = None,
        kv_token_capacity: int | None = None,
    ) -> CapsDescriptor:
        block = int(block_size or self.block_size)
        token_capacity = (
            kv_token_capacity if kv_token_capacity is not None else self.kv_token_capacity
        )
        physical_blocks = (
            max(1, int(token_capacity) // block) if token_capacity else int(self.num_blocks)
        )
        padding_blocks = decode_graph_padding_block_count(block)
        num_blocks = max(1, physical_blocks - padding_blocks)
        # Report scratch pool capacity in tokens; use a large sentinel when no pool exists.
        scratch_capacity_tokens = (
            int(self._scratch_blocks) * block if self._scratch_blocks > 0 else 1 << 24
        )
        max_image_tokens = int(self.max_latent_size)
        return CapsDescriptor(
            block_size=block,
            num_blocks=num_blocks,
            num_layers=self.num_layers,
            scratch_capacity_tokens=scratch_capacity_tokens,
            max_latent_size=int(self._active_latent_capacity_tokens(token_capacity)),
            latent_downsample=int(self.latent_downsample),
            max_vae_grid_tokens=max_image_tokens,
            max_vit_grid_tokens=MAX_VIT_GRID_TOKENS,
            commit_marker_tokens=COMMIT_MARKER_TOKENS,
            gen_rope_advance=GEN_ROPE_ADVANCE,
            max_cfg_branches=MAX_CFG_BRANCHES,
            bytes_per_token=int(self.bytes_per_token),
            max_batch_ops=MAX_BATCH_OPS,
            attention_backend=self.attention_backend,
            kv_dtype=self._kv_dtype_name_for(torch.bfloat16),
            encoder_cache_budget=self.ENCODER_CACHE_BUDGET,
        )

    def compile_targets(self) -> tuple[CompileTarget, ...]:
        language_model = getattr(self.model, "language_model", None)
        decoder = getattr(language_model, "model", None)
        if not isinstance(decoder, nn.Module):
            return ()
        return (
            CompileTarget(
                label="sensenova.language_model.model",
                module=decoder,
                owner=language_model if isinstance(language_model, nn.Module) else None,
                attr_name="model" if isinstance(language_model, nn.Module) else None,
            ),
        )

    def _kv_bytes_per_token(self, compute_dtype: torch.dtype) -> int:
        return kv_cache_bytes_per_token(
            num_kv_heads=self._kv_num_heads,
            head_dim=self._kv_head_dim,
            num_layers=self.num_layers,
            compute_dtype=compute_dtype,
            store_dtype=self.kv_cache_dtype,
        )

    def _annotate_towers(self) -> None:
        """Tag the generation-tower modules ``Pinned(tower, gen)`` for placement.

        Declares which modules belong to the gen tower; the generic
        :func:`place_towers` pass realizes device placement from these tags.
        A trivial/absent tower leaves the model untagged (a no-op).
        """
        if self.model is None or self._tower_coords is None:
            return
        _TOWER_LAYOUT.tag_generation_modules(self.model, self._tower_coords[Modality.GEN])

    def _wait_gen_cache_ready(self, cache: Any) -> None:
        """Gen tower waits until the staged snapshot is fully written."""
        self.tower_session.wait_gen_cache_ready(cache)

    def _resolve_tower_binding(self) -> TowerBinding:
        """Resolve the live destination residency + coordinates for a crossing.

        A tower split stages into the gen-tower KV residency over the tower
        transport with copy barriers; a trivial tower stages a same-device copy into
        the per-branch scratch pool as a writable replica, byte-identical to the
        single-device path."""
        if self._tower_coords is not None:
            allocate_blocks = self.residency.require_allocator_for_pool(
                self.gen_scratch_pool,
                label="SenseNova gen snapshot KV pool",
            )
            return TowerBinding(
                transport=self._tower_transport,
                primary_coord=self._tower_coords[Modality.TEXT],
                gen_coord=self._tower_coords[Modality.GEN],
                num_layers=self.num_layers,
                block_size=self.block_size,
                target_pool=self.gen_scratch_pool,
                target_device=self.gen_device,
                allocate_blocks=allocate_blocks,
            )
        allocate_blocks = self.residency.require_allocator_for_pool(
            self.scratch_pool,
            label="SenseNova scratch KV pool",
        )
        return TowerBinding(
            transport=None,
            primary_coord=0,
            gen_coord=0,
            num_layers=self.num_layers,
            block_size=self.block_size,
            target_pool=self.scratch_pool,
            target_device=self.device,
            allocate_blocks=allocate_blocks,
        )

    def _denoise_cache(self, cache: Any) -> Any:
        """Snapshot the cond-KV into a writable replica for denoising.

        The whole und->gen KV crossing is owned by :class:`TowerHandoff`."""
        return self.tower_session.denoise_cache(cache)

    def bind_data_plane_handoff(self, transport: Any) -> None:
        """Bind the Mode-A cross-process und<->gen handoff to a data-plane transport.

        Called by the runner driver on a tower-disaggregated (und/gen) worker. The
        per-branch :attr:`_tower_handoff` (local, same-device) is unchanged; this
        adds the cross-process publish (und) / fetch (gen) of the conditioning KV
        over the registered ``cuda_ipc`` / ``mooncake`` transport."""
        self.tower_session.bind_data_plane_handoff(transport)

    def maybe_publish_conditioning(self, req_id: int, sampled_token_id: int) -> str | None:
        """und side: when text decode emits ``img_start``, publish ``st.cond``.

        Returns the wire locator (for ``SeqResult.locator``) the gen pool will fetch
        and rebuild ``st.cond`` from, or ``None`` outside Mode A / a non-image token.
        A no-op unless a data-plane handoff is bound (Mode A)."""
        return self.tower_session.publish_conditioning(req_id, sampled_token_id)

    def _stage_text_cache_from_snapshot(
        self,
        target: SequenceCache,
        snapshot: ConditioningSnapshot,
        *,
        locators: tuple[Any, ...],
        length: int,
        t_index: int,
        last_token_id: int | None,
    ) -> None:
        self.tower_session.stage_text_cache_from_snapshot(
            target,
            snapshot,
            locators=locators,
            length=length,
            t_index=t_index,
            last_token_id=last_token_id,
        )

    def _maybe_stage_conditioning_from_op(self, st: Any, op: dict[str, Any]) -> None:
        """gen side: rebuild ``st.cond`` from the fetched conditioning snapshot.

        When a ``denoise_gen`` op carries the und pool's conditioning locator and
        this request has no local ``st.cond`` (the gen pool never ran the und text),
        fetch the published KV into the gen replica and populate the decode-state
        scalars the denoise setup reads. A no-op in Mode C / single-device."""
        self.tower_session.stage_conditioning_from_op(st, op)

    def _release_image_state_caches(self, image_state: FlowState | None) -> None:
        if image_state is None:
            return
        # The latent trajectory lives in the system LatentPool; free its handle
        # so the buffer is reclaimed at commit/drop.
        self.residency.latent.free(image_state.latent_handle)
        live_cache_ids: set[int] = set()
        for state in self.reqs.values():
            for text_cache in (state.cond, state.tu, state.iu):
                cache = getattr(text_cache, "past", None)
                if cache is not None:
                    live_cache_ids.add(id(cache))
        seen: set[int] = set()
        for cache in (image_state.cond_cache, image_state.tu_cache, image_state.iu_cache):
            cache_id = id(cache)
            if cache is None or cache_id in seen or cache_id in live_cache_ids:
                continue
            seen.add(cache_id)
            self.segment_executor.release_staging(cache)
            self.residency.release_scratch_cache(cache)

    def _text_driver(self) -> SequenceExecutor:
        driver = getattr(self, "_shared_text_driver", None)
        if driver is None:
            driver = SequenceExecutor(
                self,
                request_state_factory=ProgramState,
                image_start_token=self._img_start_token,
            )
            self._shared_text_driver = driver
        return driver

    def _commit_driver(self) -> ImageMaterializer:
        driver = getattr(self, "_shared_commit_driver", None)
        if driver is None:
            driver = ImageMaterializer(self, self.tower_session)
            self._shared_commit_driver = driver
        return driver

    def _state(self, op: dict[str, Any]) -> ProgramState:
        return self._text_driver().state(op)

    def run_text_logits_batch(self, ops: list[Mapping[str, Any]]) -> list[torch.Tensor]:
        # Batching and sequence CUDA graphs are system-owned by the sequence
        # driver; the model only supplies the neural forward.
        return self._text_driver().run_text_logits_batch(ops)

    def try_run_graph_logits_batch(self, ops: list[Mapping[str, Any]]) -> list[torch.Tensor] | None:
        return self._text_driver().try_run_graph_logits_batch(ops)

    def run_text_logits(self, op: Mapping[str, Any]) -> torch.Tensor:
        return self._text_driver().run_text_logits(dict(op))

    def prompt_predecessor_logits(self, req_id: int) -> torch.Tensor | None:
        return self.program_state(int(req_id)).cond.last_logits

    def sequence_forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        indexes: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        attention_mask: Any = None,
        past_key_values: Any = None,
        use_cache: bool = True,
        text_only_rope: bool = False,
        causal_paged_update: bool = False,
        return_all_logits: bool = False,
    ) -> CausalLMOutputWithPast:
        return self.model.language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            indexes=indexes,
            cache_position=cache_position,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            text_only_rope=text_only_rope,
            causal_paged_update=causal_paged_update,
            logits_to_keep=0 if return_all_logits else 1,
        )

    def sequence_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.language_model.get_input_embeddings()(input_ids)

    def sequence_inputs(self, query: str) -> tuple[torch.Tensor, torch.Tensor, Any]:
        return self.model._build_t2i_text_inputs(self.tokenizer, query)

    def empty_image_start_query(self, image_start_token: str) -> str:
        return self.model._build_t2i_query("", append_text=image_start_token)

    def flow_query(self, text: str, *, append_text: str) -> str:
        return self.model._build_t2i_query(text, append_text=append_text)

    def flow_indexes(
        self,
        token_h: int,
        token_w: int,
        text_len: int,
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        return self.model._build_t2i_image_indexes(token_h, token_w, text_len, device=device)

    def flow_predict_velocity(
        self,
        image_embeds: torch.Tensor,
        indexes: torch.Tensor,
        attention_mask: Any,
        cache: Any,
        t: torch.Tensor,
        z: torch.Tensor,
        *,
        image_token_num: int,
        image_size: tuple[int, int],
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.model._t2i_predict_v(
            image_embeds,
            indexes,
            attention_mask,
            cache,
            t,
            z,
            image_token_num=image_token_num,
            image_size=image_size,
            return_hidden=return_hidden,
        )

    def image_patch_size(self) -> int:
        return int(self.model.patch_size)

    def image_downsample_ratio(self) -> float:
        return float(self.model.downsample_ratio)

    def product_transfer_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def normalize_materialized_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize a generated image for the understanding vision encoder."""

        raw = image * 0.5 + 0.5
        mean = raw.new_tensor(_UNDERSTANDING_IMAGE_MEAN).view(1, 3, 1, 1)
        std = raw.new_tensor(_UNDERSTANDING_IMAGE_STD).view(1, 3, 1, 1)
        return (raw - mean) / std

    def sequence_position_indexes(
        self,
        grid_hw: torch.Tensor,
        temporal_indexes: torch.Tensor,
    ) -> torch.Tensor:
        """Build SenseNova's temporal-height-width position axes."""

        merge = int(1 / self.image_downsample_ratio())
        abs_w, abs_h = build_abs_positions_from_grid_hw(
            grid_hw[:1].to(temporal_indexes.device) // merge,
            device=temporal_indexes.device,
        )
        if int(temporal_indexes.numel()) == int(abs_h.numel()) + 1:
            abs_h = torch.cat((abs_h, abs_h.new_zeros(1)))
            abs_w = torch.cat((abs_w, abs_w.new_zeros(1)))
        if int(temporal_indexes.numel()) != int(abs_h.numel()):
            raise invalid_descriptor("image position axes do not match the sequence length")
        return torch.stack(
            (temporal_indexes.to(torch.long), abs_h.to(torch.long), abs_w.to(torch.long)),
            dim=0,
        )

    def image_features(
        self,
        image_input: torch.Tensor,
        *,
        grid_hw: torch.Tensor,
        gen_model: bool = False,
    ) -> torch.Tensor:
        return self.model.extract_feature(image_input, gen_model=gen_model, grid_hw=grid_hw)

    def flow_feature_dtype(self) -> torch.dtype:
        gen_vit = self.model.fm_modules["vision_model_mot_gen"]
        return next(gen_vit.parameters()).dtype

    def flow_noise_scale(self, grid_h: int, grid_w: int) -> float:
        noise_scale = self.model.noise_scale
        mode = getattr(self.model.noise_scale_mode, "value", self.model.noise_scale_mode)
        if str(mode) in _NOISE_RESOLUTION_MODES:
            base = float(self.model.noise_scale_base_image_seq_len)
            seq_len_ratio = float(grid_h * grid_w) / (self.merge_size**2) / base
            noise_scale = seq_len_ratio**_NOISE_RESOLUTION_EXPONENT * float(noise_scale)
            if str(mode) == "dynamic_sqrt":
                noise_scale = noise_scale**_NOISE_DYNAMIC_SQRT_EXPONENT
        return min(float(noise_scale), float(self.model.noise_scale_max_value))

    def flow_noise_scale_embedding(
        self,
        noise_scale: float,
        token_count: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor | None:
        if not bool(self.model.add_noise_scale_embedding):
            return None
        ns = torch.full(
            (int(token_count),),
            float(noise_scale) / float(self.model.noise_scale_max_value),
            device=device,
            dtype=dtype,
        )
        return self.model.fm_modules["noise_scale_embedder"](ns).view(1, int(token_count), -1)

    def flow_timestep_embeddings(self, t_values: torch.Tensor) -> torch.Tensor:
        return self.model.fm_modules["timestep_embedder"](t_values)

    def packed_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.language_model.get_input_embeddings()(input_ids)

    def packed_decoder_forward(
        self,
        input_embeds: torch.Tensor,
        *,
        route_indicators: torch.Tensor,
        indexes: torch.Tensor,
        forward_stream: Any,
        kv_view: Any,
    ) -> torch.Tensor:
        return self.model.language_model.model.forward_packed_visible(
            input_embeds,
            route_indicators=route_indicators,
            indexes=indexes,
            forward_stream=forward_stream,
            kv_view=kv_view,
        )

    def packed_text_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.language_model.lm_head(hidden_states)

    def packed_hidden_to_velocity(
        self,
        hidden_states: torch.Tensor,
        t: torch.Tensor,
        latent: torch.Tensor,
        *,
        image_token_num: int,
        image_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        return self.model._t2i_hidden_to_velocity(
            hidden_states,
            t,
            latent,
            image_token_num=image_token_num,
            image_size=image_size,
        )

    def segment_graph_attention(self) -> Any:
        if self.model is None or not self.model.language_model.model.layers:
            raise capability_mismatch("SenseNova packed graph requires at least one decoder layer")
        return self.model.language_model.model.layers[0].self_attn

    def query_geometry(self) -> tuple[int, float, torch.dtype]:
        """Query-side geometry for the system decode-graph FlashInfer planner.

        Returns ``(num_query_heads, attention_scale, query_dtype)``. The KV-side
        geometry (heads / head dim / page size / dtype) is read off the shared KV
        pool by the system adapter; only these query-side values are model-specific.
        """

        return self._query_geometry_from(self.model.language_model.model.layers[0].self_attn)

    def _text_indexes(
        self, start: int, seq_len: int, *, device: torch.device | str | None = None
    ) -> torch.Tensor:
        target = device if device is not None else self.device
        return self._text_driver().text_indexes(int(start), int(seq_len)).to(target)

    def _extend_cache_blocks(self, cache: SequenceCache, op: dict[str, Any]) -> None:
        self._text_driver().extend_cache_blocks(cache, op)

    def _ensure_host_cache(self, cache: SequenceCache) -> None:
        self._text_driver().ensure_host_cache(cache)

    def _prefix_from_query(self, query: str) -> SequenceCache:
        return self._text_driver().prefix_from_query(query)

    def _ensure_img_start(self, cache: SequenceCache | None) -> None:
        self._text_driver().ensure_img_start(cache)

    def _empty_img_start_prefix(self) -> SequenceCache:
        return self._text_driver().empty_img_start_prefix()

    def commit_generated_image(self, req_id: int, state: Any, op: dict[str, Any]) -> dict[str, Any]:
        op = dict(op)
        if op.get("kind") == "commit_writeback":
            return self._commit_driver().commit_writeback(op)
        return self._commit_driver().commit_generated_image(op)

    def _ingest_driver(self) -> ImageEncoder:
        driver = getattr(self, "_shared_ingest_driver", None)
        if driver is None:
            driver = ImageEncoder(self)
            self._shared_ingest_driver = driver
        return driver

    def _understanding_processor(self) -> Any:
        processor = getattr(self, "_shared_understanding_processor", None)
        if processor is None:
            processor = get_processor_for_model(type(self))
            if processor is None:
                raise capability_mismatch(
                    "no multimodal processor is registered for SenseNova understanding inputs"
                )
            self._shared_understanding_processor = processor
        return processor

    def encode_image(
        self,
        pixels: Any = None,
        grid: Any = None,
        *,
        op: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ingest an external understanding image (``vit_encode``).

        The engine hands the image bytes plus the shared temporal RoPE index
        (``cond_pos``); the begin/end markers are ordinary prompt tokens, so
        this op appends only the patch block into the conditional text cache
        and reports how many vision tokens it added.
        """
        del pixels, grid
        if op is None:
            raise invalid_descriptor("SenseNova image encode requires an op descriptor")
        if self.model is None:
            raise RuntimeError("SenseNova model weights are not loaded")
        op = dict(op)
        req_id = int(op["req_id"])
        image_b64 = op.get("image_b64")
        cond_pos = op.get("cond_pos")
        if cond_pos is None:
            raise invalid_descriptor("vit_encode requires the shared temporal index (cond_pos)")

        st = self._state(op)
        text_driver = self._text_driver()
        text_driver.extend_cache_blocks(st.cond, op)
        text_driver.ensure_host_cache(st.cond)

        driver = self._ingest_driver()
        if image_b64:
            processor = self._understanding_processor()
            image = processor.decode_image_b64(str(image_b64))
            image_hw = [int(image.height), int(image.width)]
            flattened, grid_hw = processor.understanding_patches(image)
            flattened = flattened.to(device=self.device, dtype=self.model.dtype)
            vit_embeds = driver.encode_understanding_image(flattened, grid_hw).detach()
            handle = encoder_handle_from_mm_hash(op.get("mm_hash"))
            self.residency.encoder.put(
                handle,
                {
                    "kind": "vit_encode",
                    "vit_embeds": vit_embeds,
                    "grid_hw": grid_hw.detach(),
                    "image_hw": image_hw,
                },
            )
        else:
            cached_handle = op.get("image_in")
            if not isinstance(cached_handle, int) or isinstance(cached_handle, bool):
                raise invalid_descriptor("cached vit_encode requires an encoder handle")
            cached = self.residency.encoder.get(cached_handle)
            if not isinstance(cached, Mapping) or cached.get("kind") != "vit_encode":
                raise invalid_descriptor("cached vit_encode handle is not resident")
            cached_vit_embeds = cached.get("vit_embeds")
            cached_grid_hw = cached.get("grid_hw")
            cached_image_hw = cached.get("image_hw")
            if not isinstance(cached_vit_embeds, torch.Tensor) or not isinstance(
                cached_grid_hw, torch.Tensor
            ):
                raise invalid_descriptor("cached vit_encode payload is incomplete")
            if (
                not isinstance(cached_image_hw, list)
                or len(cached_image_hw) != 2
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in cached_image_hw
                )
            ):
                raise invalid_descriptor("cached vit_encode dimensions are invalid")
            vit_embeds = cached_vit_embeds
            grid_hw = cached_grid_hw
            image_hw = [int(value) for value in cached_image_hw]
            handle = cached_handle

        num_tokens = driver.ingest_understanding_embeddings(
            st.cond,
            vit_embeds,
            grid_hw,
            t_index=int(cond_pos),
        )
        return {
            "req_id": req_id,
            "encoder_handle": handle,
            "num_tokens": num_tokens,
            "image_hw": image_hw,
        }

    def free_encoder(self, handles: Any) -> None:
        # Encoder-output residency is system-owned: the handle store lives on
        # the ResidencyManager, not the model.
        for handle in handles or []:
            self.residency.encoder.pop(int(handle))

    def on_new_request(self, req_id: int, state: RunnerRequestState) -> None:
        req_id = int(req_id)
        self.runner_states[req_id] = state
        existing = self.reqs.get(req_id)
        if isinstance(existing, ProgramState):
            existing.sampling = dict(state.sampling or existing.sampling or {})
            existing.image = dict(state.image or existing.image or {})
            existing.neg_token_ids = list(state.neg_token_ids or existing.neg_token_ids or [])
            if not existing.cond.block_ids and state.block_ids:
                existing.cond.block_ids = list(state.block_ids)
                set_blocks = getattr(existing.cond.past, "set_blocks", None)
                if callable(set_blocks):
                    set_blocks(existing.cond.block_ids)
            return
        image_state = self._new_program_state(state)
        self.reqs[req_id] = image_state

    def _new_program_state(self, state: RunnerRequestState) -> ProgramState:
        image_state = ProgramState(
            sampling=dict(state.sampling or {}),
            image=dict(state.image or {}),
            neg_token_ids=list(state.neg_token_ids or []),
        )
        image_state.cond.block_ids = list(state.block_ids or [])
        image_state.rng = state.device_rng(self.gen_device)
        return image_state

    def program_state(self, req_id: int) -> ProgramState:
        req_id = int(req_id)
        existing = self.reqs.get(req_id)
        if isinstance(existing, ProgramState):
            return existing
        state = self.runner_states.get(req_id)
        if state is None:
            return self.reqs.setdefault(req_id, ProgramState())
        created = self._new_program_state(state)
        self.reqs[req_id] = created
        return created

    transformers_image_state = program_state

    def drop_request(self, req_id: int) -> None:
        req_id = int(req_id)
        self.runner_states.pop(req_id, None)
        st = self.reqs.pop(req_id, None)
        if st is not None:
            self._release_image_state_caches(st.image_state)
            self.segment_executor.release_staging(st.cond.past)
            self.segment_executor.release_staging(st.tu.past)
            self.segment_executor.release_staging(st.iu.past)
            self.residency.release_scratch_cache(st.tu.past)
            self.residency.release_scratch_cache(st.iu.past)

    def prepare_flow(self, state: RunnerRequestState, op: dict[str, Any] | Any) -> PreparedFlowStep:
        req_id = int(op["req_id"])
        return self.flow_execution.prepare_flow_step(req_id, state, dict(op))

    def predict_velocity(
        self,
        ctx: PreparedFlowStep,
        t: torch.Tensor,
        latent: torch.Tensor,
        branch: str,
    ) -> torch.Tensor:
        del t, latent
        velocity = self.flow_execution.predict_flow_velocity(ctx, branch)
        if not isinstance(velocity, torch.Tensor):
            raise invalid_descriptor(
                "SenseNova velocity prediction unexpectedly returned hidden state"
            )
        return velocity

    def predict_flow_velocity_batch(self, steps, branches_by_step, *, graph_mode: str = "auto"):
        return self.flow_execution.predict_flow_velocity_batch(
            steps,
            branches_by_step,
            graph_mode=graph_mode,
        )

    def accept_flow_update(self, ctx: PreparedFlowStep, latent: torch.Tensor) -> None:
        self.flow_execution.apply_flow_update(ctx, latent)

    def _denoise_branch_inputs(self, image: FlowState, branch: str) -> tuple[torch.Tensor, Any]:
        return self.flow_execution._denoise_branch_inputs(image, branch)

    def decode_image(
        self,
        latent: Any,
        *,
        req_id: int | None = None,
        state: Any = None,
        op: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        del latent, state
        if req_id is None or op is None:
            raise invalid_descriptor("SenseNova image commit requires req_id and op")
        return self.commit_generated_image(int(req_id), None, dict(op))

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch) -> Any:
        if self.model is None:
            raise RuntimeError("SenseNova model weights are not loaded")
        options = get_forward_context().execution_options
        return self.segment_executor.execute(
            batch,
            request_states=self.runner_states,
            defer_text_cpu_results=bool(getattr(options, "defer_text_cpu_results", False)),
        )

    @torch.inference_mode()
    def forward_text(self, batch: ForwardBatch) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("SenseNova model weights are not loaded")
        if len(batch.ops) != 1:
            raise invalid_descriptor("SenseNova tensor text forward requires one operation")
        logits = self.run_text_logits(dict(batch.ops[0]))
        if not isinstance(logits, torch.Tensor):
            raise invalid_descriptor("SenseNova text forward must return logits")
        return logits


EntryClass = SenseNovaU1ForUnifiedGeneration
