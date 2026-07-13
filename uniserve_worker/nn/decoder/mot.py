"""Batched Mixture-of-Transformers decoder primitives.

Processes a batch of independent segments (requests and/or CFG branches):
modality-routed linears/norms/MLP run batched over all segment tokens (two GEMMs
per op: understanding slice + generation slice), while attention runs per segment
against that segment's own KV cache so segments stay isolated.
"""
from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn as nn

from ..attention import RadixAttention
from ..linear import (
    LinearBase,
    QKVParallelLinear,
    RowParallelLinear,
    local_attention_head_count,
    local_kv_head_count,
)
from ..mesh import get_current_mesh
from ..norm import RMSNorm
from ..placement import set_tower_coord
from ..rope import apply_rotary_emb, get_rope
from ..vocab_parallel_embedding import VocabParallelEmbedding
from .qwen import Qwen3MLP

__all__ = [
    'Modality',
    'route_by_modality',
    'tower_modality_coords',
    'KVCache',
    'Segment',
    'MoTMLP',
    'ModalityExpert',
    'MoTDecoderLayer',
    'MoTLayer',
    'MoTModel',
]


class Modality(enum.Enum):
    """The two transformer modalities MoT routes per token.

    ``TEXT`` is the understanding branch (checkpoint params without a suffix);
    ``GEN`` is the generation/image-latent branch (the ``*_moe_gen`` twins)."""
    TEXT = "text"
    GEN = "gen"


def route_by_modality(
    src: torch.Tensor,
    routes: dict[Modality, tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]],
    *,
    out: torch.Tensor | None = None,
    transport: Any | None = None,
    coords: dict[Modality, int] | None = None,
) -> torch.Tensor:
    """Scatter per-modality sublayer outputs into one batched tensor.

    ``routes`` maps each *present* modality to ``(mask, fn)``; the rows of
    ``src`` selected by ``mask`` are transformed by ``fn`` and written to the
    same rows of ``out``.  ``out`` defaults to an empty tensor shaped like
    ``src`` (use it when the destination shape differs).  The caller passes only
    the modalities it knows are present, so empty branches are skipped without an
    extra device sync.  Returns ``out``.

    Tower-aware dispatch (the ``Router`` for the ``tower`` axis): when
    ``transport`` (a tower ``AxisTransport``) and ``coords`` (modality -> tower
    coordinate) are supplied and a modality's expert lives on a *different*
    device than ``src`` (the in-process Option-C tower split, the rare
    mixed-modality batch), the selected rows are moved to that device via
    ``transport.copy_to`` (dispatch), the expert runs there, and the result is
    moved back to ``out``'s device (combine). A modality already resident on
    ``src``'s device -- the steady state where a whole forward is single-modality
    on one coordinate, and the trivial-tower default -- takes the in-place path
    with no copy, so behavior is byte-identical when no tower is configured."""
    dst = torch.empty_like(src) if out is None else out
    for modality, (mask, fn) in routes.items():
        target_device = _tower_target_device(transport, coords, modality)
        if target_device is not None and target_device != src.device:
            if transport is None or coords is None:
                raise RuntimeError("tower-routed modality is missing transport coordinates")
            piece = transport.copy_to(src[mask], coord=int(coords[modality]))
            dst[mask] = fn(piece).to(dst.device)
        else:
            dst[mask] = fn(src[mask])
    return dst


def _tower_target_device(
    transport: Any | None,
    coords: dict[Modality, int] | None,
    modality: Modality,
) -> Any | None:
    """Device a modality's expert lives on for an in-process tower, else ``None``.

    Only the local-peer transport exposes a per-coordinate device, so this returns
    ``None`` for a trivial/absent tower or a cross-process transport (which routes
    by worker, not by an in-process device hop).
    """
    if transport is None or coords is None or modality not in coords:
        return None
    device_for = getattr(transport, "device", None)
    if not callable(device_for):
        return None
    return device_for(int(coords[modality]))


def tower_modality_coords(mesh: Any) -> dict[Modality, int] | None:
    """Map each modality to its ``tower``-axis coordinate, or ``None`` if trivial.

    The mapping is a *global* convention, independent of which coordinate this
    worker occupies (identical on und and gen workers in cross-process setups):
    coordinate 0 is the primary/understanding (``TEXT``) tower — the home for
    shared modules and the authoritative KV — and coordinate 1 is the ``GEN``
    tower. Returns ``None`` when there is no
    non-trivial tower axis, which makes every tower-aware call site fall back to
    the single-device path.
    """
    axis = mesh.axis("tower") if mesh is not None else None
    if axis is None or int(axis.size) <= 1:
        return None
    return {Modality.TEXT: 0, Modality.GEN: 1}


class KVCache:
    """Per-request growable KV cache: one (K, V) per layer, each ``(length, heads, dim)``.

    Uses doubling buffers per layer so ``append`` is amortized O(1). ``get`` returns
    a view of the populated prefix; copy before the next ``append`` if retaining it.
    """

    def __init__(self, num_layers: int):
        # Backing buffers (over-allocated capacity) and the logical length used.
        self._k_buf: list[torch.Tensor | None] = [None] * num_layers
        self._v_buf: list[torch.Tensor | None] = [None] * num_layers
        self._len: list[int] = [0] * num_layers

    def length(self) -> int:
        return self._len[0]

    def get(self, layer: int):
        n = self._len[layer]
        if n == 0:
            return None, None
        k_buffer = self._k_buf[layer]
        v_buffer = self._v_buf[layer]
        if k_buffer is None or v_buffer is None:
            raise RuntimeError("non-empty KV cache layer is missing its backing buffers")
        return k_buffer[:n], v_buffer[:n]

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        add = k.shape[0]
        if add == 0:
            return
        n = self._len[layer]
        buf = self._k_buf[layer]
        cap = 0 if buf is None else buf.shape[0]
        if n + add > cap:
            # Grow geometrically (at least double, but enough for this append).
            new_cap = max(cap * 2, n + add, 1)
            new_k = k.new_empty((new_cap, *k.shape[1:]))
            new_v = v.new_empty((new_cap, *v.shape[1:]))
            if n:
                old_k = self._k_buf[layer]
                old_v = self._v_buf[layer]
                if old_k is None or old_v is None:
                    raise RuntimeError("non-empty KV cache layer is missing its backing buffers")
                new_k[:n] = old_k[:n]
                new_v[:n] = old_v[:n]
            self._k_buf[layer] = new_k
            self._v_buf[layer] = new_v
        k_buffer = self._k_buf[layer]
        v_buffer = self._v_buf[layer]
        if k_buffer is None or v_buffer is None:
            raise RuntimeError("KV cache allocation did not create backing buffers")
        k_buffer[n : n + add] = k
        v_buffer[n : n + add] = v
        self._len[layer] = n + add


@dataclass
class Segment:
    """One lane of work inside a batched forward."""
    embeds: torch.Tensor       # (n, hidden) input embeddings (already built)
    positions: torch.Tensor    # (n,) long rope position ids
    is_gen: torch.Tensor       # (n,) bool — generation-modality tokens routed through _moe_gen params
    cache: KVCache             # context KV for this lane
    causal: bool               # attention regime (True=text, False=latents/full)
    update_cache: bool         # append new K/V into cache after attention


def MoTMLP(hidden: int, inter: int) -> Qwen3MLP:
    """SiLU-gated feed-forward block for one MoT modality.

    The block itself is the shared :class:`Qwen3MLP` (fused merged gate/up
    projection + ``silu_and_mul``); this factory only adapts the MoT
    ``(hidden, inter)`` construction signature onto its config surface.
    """
    return Qwen3MLP(
        SimpleNamespace(hidden_size=hidden, intermediate_size=inter, hidden_act="silu")
    )


@dataclass(frozen=True)
class ModalityExpert:
    """One modality's per-layer sublayers plus the attention *flavor* driving them.

    The module fields are *references* to modules already registered on the
    owning ``MoTDecoderLayer`` (so the layer keeps the checkpoint-compatible
    attribute names ``<x>``/``<x>_moe_gen``); grouping them lets the forward
    route by ``Modality`` rather than dispatching on suffixed attribute names.

    ``project_qkv`` and ``attend`` are the *flavor*: this base implements the
    fused-QKV, single-qk-norm, single-RoPE Qwen-style attention. A model whose
    attention differs (separate q/k/v projections, dual qk-norm, multi-axis
    RoPE, or a packed visible-end kernel) supplies its own expert exposing the
    same surface -- the four sublayer modules plus ``project_qkv``/``attend`` --
    which is all the layer skeleton and :func:`route_by_modality` consume. The
    expert is the single unit of per-modality flavor, so the routing/placement
    skeleton stays model-agnostic.
    """

    input_norm: nn.Module
    qkv_proj: nn.Module
    o_proj: nn.Module
    q_norm: nn.Module
    k_norm: nn.Module
    attn: RadixAttention
    post_norm: nn.Module
    mlp: nn.Module
    n_heads: int
    n_kv: int
    head_dim: int
    scale: float

    def project_qkv(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states into per-head q/k/v for this modality."""
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv * self.head_dim
        qkv_out = self.qkv_proj(x)
        q, k, v = qkv_out.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(-1, self.n_heads, self.head_dim)
        k = k.view(-1, self.n_kv, self.head_dim)
        v = v.view(-1, self.n_kv, self.head_dim)
        # qk-norm in fp32 before RoPE (bf16 RMSNorm drift can collapse text quality).
        q = self.q_norm(q.float())
        k = self.k_norm(k.float())
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        return q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

    def attend(
        self,
        layer_idx: int,
        qi: torch.Tensor,
        ki: torch.Tensor,
        vi: torch.Tensor,
        cache: Any,
        causal: bool,
        update_cache: bool,
    ) -> torch.Tensor:
        """Run this modality's attention over its own segment KV cache."""
        # RadixAttention owns backend selection, paged-cache reads, and append
        # writes.  This keeps the model graph on the shared attention seam.
        self.attn.layer_id = int(layer_idx)
        return self.attn(
            qi,
            ki,
            vi,
            kv_cache=cache,
            update_cache=update_cache,
            causal=causal,
            scale=self.scale,
        )


class MoTDecoderLayer(nn.Module):
    """One MoT decoder layer with separate understanding and generation parameter sets."""

    def __init__(self, cfg: Any):
        super().__init__()
        h, hd = cfg.hidden_size, cfg.head_dim
        self.total_n_heads = int(cfg.num_attention_heads)
        self.total_n_kv = int(cfg.num_key_value_heads)
        # Local (per-tensor-parallel-rank) head counts: activations downstream
        # of the sharded qkv projections carry these, and the shared helpers
        # keep the split rule identical to QKVParallelLinear's shard sizes.
        self.n_heads = local_attention_head_count(self.total_n_heads)
        self.n_kv = local_kv_head_count(self.total_n_kv)
        self.head_dim = hd
        self.total_q_size = self.total_n_heads * hd
        self.q_size = self.n_heads * hd
        self.kv_size = self.n_kv * hd
        self.scale = 1.0 / (hd**0.5)
        self.rep = self.n_heads // self.n_kv

        self._init_text_expert_modules(cfg, h, hd)
        self._init_gen_expert_modules(cfg, h, hd)

        # ``experts`` bundles each modality's sublayers behind one handle so the
        # forward routes by ``Modality`` instead of branching on suffixed names.
        # It is a plain dict of *references* to the modules already registered
        # above (not an ``nn.Module``), so it adds no entries to
        # ``named_parameters``/``state_dict``: checkpoint keys stay
        # ``<x>``/``<x>_moe_gen`` exactly as the loaders expect.
        self.experts = self._build_expert_table()
        self._configure_tower_routing()

    def _init_text_expert_modules(self, cfg: Any, hidden_size: int, head_dim: int) -> None:
        self.input_layernorm = RMSNorm(hidden_size, cfg.rms_norm_eps)
        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim, self.total_n_heads, self.total_n_kv, bias=True
        )
        self.o_proj = RowParallelLinear(self.total_q_size, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, cfg.rms_norm_eps)
        self.attn = RadixAttention(self.n_heads, self.n_kv, head_dim)
        self.post_attention_layernorm = RMSNorm(hidden_size, cfg.rms_norm_eps)
        self.mlp = MoTMLP(hidden_size, cfg.intermediate_size)

    def _init_gen_expert_modules(self, cfg: Any, hidden_size: int, head_dim: int) -> None:
        self.input_layernorm_moe_gen = RMSNorm(hidden_size, cfg.rms_norm_eps)
        self.qkv_proj_moe_gen = QKVParallelLinear(
            hidden_size, head_dim, self.total_n_heads, self.total_n_kv, bias=True
        )
        self.o_proj_moe_gen = RowParallelLinear(self.total_q_size, hidden_size, bias=False)
        self.q_norm_moe_gen = RMSNorm(head_dim, cfg.rms_norm_eps)
        self.k_norm_moe_gen = RMSNorm(head_dim, cfg.rms_norm_eps)
        self.attn_moe_gen = RadixAttention(self.n_heads, self.n_kv, head_dim)
        self.post_attention_layernorm_moe_gen = RMSNorm(hidden_size, cfg.rms_norm_eps)
        self.mlp_moe_gen = MoTMLP(hidden_size, cfg.intermediate_size)

    def _build_expert_table(self) -> dict[Modality, ModalityExpert]:
        return {
            Modality.TEXT: self._modality_expert(
                self.input_layernorm,
                self.qkv_proj,
                self.o_proj,
                self.q_norm,
                self.k_norm,
                self.attn,
                self.post_attention_layernorm,
                self.mlp,
            ),
            Modality.GEN: self._modality_expert(
                self.input_layernorm_moe_gen,
                self.qkv_proj_moe_gen,
                self.o_proj_moe_gen,
                self.q_norm_moe_gen,
                self.k_norm_moe_gen,
                self.attn_moe_gen,
                self.post_attention_layernorm_moe_gen,
                self.mlp_moe_gen,
            ),
        }

    def _modality_expert(
        self,
        input_norm: RMSNorm,
        qkv_proj: QKVParallelLinear,
        o_proj: LinearBase,
        q_norm: RMSNorm,
        k_norm: RMSNorm,
        attn: RadixAttention,
        post_norm: RMSNorm,
        mlp: Qwen3MLP,
    ) -> ModalityExpert:
        return ModalityExpert(
            input_norm=input_norm,
            qkv_proj=qkv_proj,
            o_proj=o_proj,
            q_norm=q_norm,
            k_norm=k_norm,
            attn=attn,
            post_norm=post_norm,
            mlp=mlp,
            n_heads=self.n_heads,
            n_kv=self.n_kv,
            head_dim=self.head_dim,
            scale=self.scale,
        )

    def _configure_tower_routing(self) -> None:
        # Tower axis: tag the generation sublayers Pinned(tower, gen) so the
        # generic place_towers pass lands them on the gen coordinate's device, and
        # resolve the routing transport/coords once. Trivial/absent tower -> both
        # are None and every route_by_modality below takes the in-place path
        # (byte-identical to a single-device model).
        mesh = get_current_mesh()
        self._tower_coords = tower_modality_coords(mesh)
        tower_axis = mesh.axis("tower")
        self._tower_transport = (
            tower_axis.transport
            if self._tower_coords is not None and tower_axis is not None
            else None
        )
        gen_coord = self._tower_coords[Modality.GEN] if self._tower_coords is not None else 1
        for module in (
            self.input_layernorm_moe_gen,
            self.qkv_proj_moe_gen,
            self.o_proj_moe_gen,
            self.q_norm_moe_gen,
            self.k_norm_moe_gen,
            self.attn_moe_gen,
            self.post_attention_layernorm_moe_gen,
            self.mlp_moe_gen,
        ):
            set_tower_coord(module, gen_coord)

    def forward_paged_text(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_values: Any,
    ) -> torch.Tensor:
        """One text-modality (und) layer step over a shared paged text cache.

        The interleaved text driver's serving path: token-major
        ``hidden_states`` ``[tokens, hidden]`` attend causally against the
        request's paged KV through the duck-typed layer-update protocol
        (``request_cache_for_update`` / ``finish_layer_update`` /
        ``cancel_layer_update``). The protocol hands RadixAttention the same
        per-request page view the eager ``Segment(causal=True)`` path passes
        (PAGED_EXTEND for a multi-token span, PAGED_DECODE for one token), so
        the numerics match :meth:`forward`'s text lane exactly.
        """
        expert = self.experts[Modality.TEXT]
        n_tokens = int(hidden_states.shape[0])
        normed = expert.input_norm(hidden_states)
        q, k, v = expert.project_qkv(normed, cos, sin)
        cache = past_key_values.request_cache_for_update(layer_idx, n_tokens)
        try:
            attn_values = expert.attend(layer_idx, q, k, v, cache, True, True)
        except BaseException:
            cancel = getattr(past_key_values, "cancel_layer_update", None)
            if callable(cancel):
                cancel(layer_idx)
            raise
        past_key_values.finish_layer_update(layer_idx, n_tokens)
        hidden_states = hidden_states + expert.o_proj(
            attn_values.reshape(n_tokens, self.q_size)
        )
        normed = expert.post_norm(hidden_states)
        return hidden_states + expert.mlp(normed.to(torch.bfloat16))

    def forward_paged_text_batch(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_values: Any,
    ) -> torch.Tensor:
        """Run one text decode token for each row of a batched paged cache."""

        expert = self.experts[Modality.TEXT]
        batch = int(hidden_states.shape[0])
        normed = expert.input_norm(hidden_states)
        q, k, v = expert.project_qkv(normed, cos, sin)
        q = q.view(batch, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, 1, self.n_kv, self.head_dim).transpose(1, 2)
        v = v.view(batch, 1, self.n_kv, self.head_dim).transpose(1, 2)
        cache = past_key_values.request_cache_for_update(layer_idx, 1)
        try:
            attn_values = expert.attend(layer_idx, q, k, v, cache, True, True)
        except BaseException:
            cancel = getattr(past_key_values, "cancel_layer_update", None)
            if callable(cancel):
                cancel(layer_idx)
            raise
        past_key_values.finish_layer_update(layer_idx, 1)
        hidden_states = hidden_states + expert.o_proj(
            attn_values.transpose(1, 2).reshape(batch, self.q_size)
        )
        normed = expert.post_norm(hidden_states)
        return hidden_states + expert.mlp(normed.to(torch.bfloat16))

    def forward_paged_gen_batch(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        text_idx: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_values: Any,
        batch: int,
        n_tokens: int,
    ) -> torch.Tensor:
        """One layer step over ``batch`` transient CFG-branch rows of one gen segment.

        The batched-rows twin of the eager ``Segment(causal=False,
        update_cache=False)`` gen lane in :meth:`forward`: per-token math is
        identical, and attention runs the GEN expert for every row (marker
        tokens included, exactly like the eager path's per-segment expert
        selection) over ``[row prefix + row tokens]`` through the batched
        transient paged protocol (``request_cache_for_transient``): row K/V
        land in scratch pages past each row's fixed base length — overwritten
        next step, never persisted — and the paged varlen kernel reads the
        prefix in place of the eager path's per-layer ``cache.get`` gather +
        concat.

        Modality dispatch is *indexed*, not boolean-masked: a gen segment is
        almost entirely gen-modality (only the two image markers per row are
        text), so the GEN expert computes every token and the TEXT expert
        overwrites the ``text_idx`` rows. Row-wise ops (norms) are exact
        either way; for the GEMMs this only changes the batch a row sits in.
        Boolean ``mask``-indexing here would issue hundreds of ``nonzero``
        device syncs + masked scatters per step — the measured dominant cost.
        Single-device only: this path performs no tower-transport routing
        (models with a non-trivial tower axis must keep the routed
        :meth:`forward` segment path).
        """
        gen = self.experts[Modality.GEN]
        text = self.experts[Modality.TEXT]
        normed = gen.input_norm(hidden_states)
        normed_text = None
        if text_idx is not None:
            normed_text = text.input_norm(hidden_states.index_select(0, text_idx))
            normed.index_copy_(0, text_idx, normed_text)
        q, k, v = gen.project_qkv(normed, cos, sin)
        if text_idx is not None:
            if normed_text is None:
                raise RuntimeError("text expert rows are missing normalized inputs")
            tq, tk, tv = text.project_qkv(
                normed_text,
                cos.index_select(0, text_idx),
                sin.index_select(0, text_idx),
            )
            q.index_copy_(0, text_idx, tq)
            k.index_copy_(0, text_idx, tk)
            v.index_copy_(0, text_idx, tv)
        # Token-major [batch*n_tokens, heads, dim] -> [batch, heads, n_tokens, dim]
        # views (the transient varlen path flattens back to token-major for free).
        q = q.view(batch, n_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, n_tokens, self.n_kv, self.head_dim).transpose(1, 2)
        v = v.view(batch, n_tokens, self.n_kv, self.head_dim).transpose(1, 2)
        cache = past_key_values.request_cache_for_transient(layer_idx, n_tokens)
        attn = gen.attend(layer_idx, q, k, v, cache, False, True)
        attn_values = attn.transpose(1, 2).reshape(batch * n_tokens, self.q_size)
        attn_out = gen.o_proj(attn_values)
        if text_idx is not None:
            attn_out.index_copy_(
                0, text_idx, text.o_proj(attn_values.index_select(0, text_idx))
            )
        hidden_states = hidden_states + attn_out
        normed = gen.post_norm(hidden_states)
        normed_text = None
        if text_idx is not None:
            normed_text = text.post_norm(hidden_states.index_select(0, text_idx))
        mlp_out = gen.mlp(normed.to(torch.bfloat16))
        if text_idx is not None:
            if normed_text is None:
                raise RuntimeError("text expert rows are missing post-attention inputs")
            mlp_out.index_copy_(0, text_idx, text.mlp(normed_text.to(torch.bfloat16)))
        return hidden_states + mlp_out

    def _present_routes(
        self, text_mask, gen_mask, any_text, any_gen,
    ) -> dict[Modality, tuple[torch.Tensor, ModalityExpert]]:
        """Map each present modality to ``(token_mask, expert)``.

        Empty branches are dropped so the modality-routed ops below run only the
        GEMMs they need (skipping them also avoids a redundant device sync)."""
        present: dict[Modality, tuple[torch.Tensor, ModalityExpert]] = {}
        if any_text:
            present[Modality.TEXT] = (text_mask, self.experts[Modality.TEXT])
        if any_gen:
            present[Modality.GEN] = (gen_mask, self.experts[Modality.GEN])
        return present

    def forward(
        self, layer_idx, H, text_mask, gen_mask, cos, sin, segs, slices,
        *, any_text=None, any_gen=None, seg_is_gen=None,
    ):
        # ``forward_segments`` may pass precomputed modality flags to skip GPU syncs.
        if any_text is None:
            any_text = bool(text_mask.any())
        if any_gen is None:
            any_gen = bool(gen_mask.any())
        routes = self._present_routes(text_mask, gen_mask, any_text, any_gen)

        # ---- input norm (modality-routed) ----
        Hn = self._route_input_norm(H, routes)
        q, k, v = self._project_qkv_by_modality(Hn, routes, cos, sin)

        # ---- per-segment attention (isolated) ----
        # Per-segment attention: pick the RadixAttention instance for the segment modality.
        attn_values = self._attend_segments(layer_idx, q, k, v, segs, slices, seg_is_gen)

        # ---- output projection (modality-routed) ----
        attn_out = self._route_output_projection(attn_values, H, routes)
        H = H + attn_out

        # ---- MLP (modality-routed) ----
        Hn2 = self._route_post_norm(H, routes)
        mlp_out = self._route_mlp(Hn2, H, routes)
        return H + mlp_out

    def _route_input_norm(
        self,
        hidden: torch.Tensor,
        routes: dict[Modality, tuple[torch.Tensor, ModalityExpert]],
    ) -> torch.Tensor:
        return route_by_modality(
            hidden,
            {m: (mask, e.input_norm) for m, (mask, e) in routes.items()},
            transport=self._tower_transport,
            coords=self._tower_coords,
        )

    def _project_qkv_by_modality(
        self,
        hidden: torch.Tensor,
        routes: dict[Modality, tuple[torch.Tensor, ModalityExpert]],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_tokens = hidden.shape[0]
        q = hidden.new_zeros(n_tokens, self.n_heads, self.head_dim)
        k = hidden.new_zeros(n_tokens, self.n_kv, self.head_dim)
        v = hidden.new_zeros(n_tokens, self.n_kv, self.head_dim)
        for mask, expert in routes.values():
            q[mask], k[mask], v[mask] = expert.project_qkv(hidden[mask], cos[mask], sin[mask])
        return q, k, v

    def _attend_segments(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        segs: Any,
        slices: Any,
        seg_is_gen: Any,
    ) -> torch.Tensor:
        if seg_is_gen is None:
            seg_is_gen = [bool(seg.is_gen.any()) for seg in segs]
        attn_values = q.new_zeros(q.shape[0], self.n_heads, self.head_dim)
        for seg, sl, seg_gen in zip(segs, slices, seg_is_gen):
            expert = self.experts[Modality.GEN if seg_gen else Modality.TEXT]
            attn_values[sl] = expert.attend(
                layer_idx,
                q[sl],
                k[sl],
                v[sl],
                seg.cache,
                seg.causal,
                seg.update_cache,
            )
        return attn_values.reshape(q.shape[0], self.q_size)

    def _route_output_projection(
        self,
        attn_values: torch.Tensor,
        residual: torch.Tensor,
        routes: dict[Modality, tuple[torch.Tensor, ModalityExpert]],
    ) -> torch.Tensor:
        return route_by_modality(
            attn_values,
            {m: (mask, e.o_proj) for m, (mask, e) in routes.items()},
            out=residual.new_zeros(residual.shape[0], residual.shape[1]),
            transport=self._tower_transport,
            coords=self._tower_coords,
        )

    def _route_post_norm(
        self,
        hidden: torch.Tensor,
        routes: dict[Modality, tuple[torch.Tensor, ModalityExpert]],
    ) -> torch.Tensor:
        return route_by_modality(
            hidden,
            {m: (mask, e.post_norm) for m, (mask, e) in routes.items()},
            transport=self._tower_transport,
            coords=self._tower_coords,
        )

    def _route_mlp(
        self,
        normalized: torch.Tensor,
        residual: torch.Tensor,
        routes: dict[Modality, tuple[torch.Tensor, ModalityExpert]],
    ) -> torch.Tensor:
        def mlp_fn(expert: ModalityExpert) -> Callable[[torch.Tensor], torch.Tensor]:
            return lambda x: expert.mlp(x.to(torch.bfloat16))

        return route_by_modality(
            normalized,
            {m: (mask, mlp_fn(e)) for m, (mask, e) in routes.items()},
            out=residual.new_zeros(residual.shape[0], residual.shape[1]),
            transport=self._tower_transport,
            coords=self._tower_coords,
        )


MoTLayer = MoTDecoderLayer


class MoTModel(nn.Module):
    """Unified MoT language model: shared attention over one KV cache,
    two modality parameter sets per layer."""

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([MoTDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.norm_moe_gen = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = get_rope(cfg.head_dim, theta=cfg.rope_theta)
        # References to the registered final norms, keyed by modality (no new
        # params; checkpoint keys stay ``norm``/``norm_moe_gen``).
        self.final_norm: dict[Modality, RMSNorm] = {
            Modality.TEXT: self.norm,
            Modality.GEN: self.norm_moe_gen,
        }
        # Tower axis: the gen final norm is Pinned(tower, gen); the final-norm
        # routing uses the same transport/coords as the layers (None when trivial).
        mesh = get_current_mesh()
        self._tower_coords = tower_modality_coords(mesh)
        tower_axis = mesh.axis("tower")
        self._tower_transport = (
            tower_axis.transport
            if self._tower_coords is not None and tower_axis is not None
            else None
        )
        if self._tower_coords is not None:
            set_tower_coord(self.norm_moe_gen, self._tower_coords[Modality.GEN])

    @torch.no_grad()
    def forward_paged_text(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        past_key_values: Any,
    ) -> torch.Tensor:
        """Run the text (und) expert stack over one shared paged text cache.

        Token-major ``inputs_embeds`` ``[tokens, hidden]`` with 1-D rope
        ``positions`` ``[tokens]``; ``past_key_values`` implements the paged
        layer-update protocol (``PagedTextCache`` eagerly, the system decode
        graph's past adapter under capture/replay). Returns final-norm hidden
        states ``[tokens, hidden]``.
        """
        cos, sin = self.rotary.cos_sin_1d(positions)
        hidden_states = inputs_embeds
        for layer_idx, layer_module in enumerate(self.layers):
            layer = cast(MoTDecoderLayer, layer_module)
            hidden_states = layer.forward_paged_text(
                layer_idx, hidden_states, cos, sin, past_key_values
            )
        return self.norm(hidden_states)

    @torch.no_grad()
    def forward_paged_text_batch(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        past_key_values: Any,
    ) -> torch.Tensor:
        """Run a batched one-token text decode over independent paged rows."""

        if inputs_embeds.ndim != 3 or int(inputs_embeds.shape[1]) != 1:
            raise ValueError("paged text batch expects [batch, 1, hidden] inputs")
        batch = int(inputs_embeds.shape[0])
        positions = positions.reshape(-1)
        if int(positions.numel()) != batch:
            raise ValueError("paged text batch positions must align with rows")
        cos, sin = self.rotary.cos_sin_1d(positions)
        hidden_states = inputs_embeds[:, 0, :]
        for layer_idx, layer_module in enumerate(self.layers):
            layer = cast(MoTDecoderLayer, layer_module)
            hidden_states = layer.forward_paged_text_batch(
                layer_idx,
                hidden_states,
                cos,
                sin,
                past_key_values,
            )
        return self.norm(hidden_states).unsqueeze(1)

    @torch.no_grad()
    def forward_paged_gen_batch(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        is_gen: torch.Tensor,
        past_key_values: Any,
        *,
        text_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run all layers over ``B`` CFG-branch rows of one shared gen segment.

        ``inputs_embeds`` is ``[B, T, hidden]`` (rows may be expanded views of
        one segment: CFG branches share the latent/timestep embedding),
        ``positions`` is ``[B, T]`` 1-D rope positions (branches differ only by
        their scalar position offset), and ``is_gen`` is the shared per-token
        modality pattern ``[T]`` (markers are text-modality, VAE latents gen).
        ``past_key_values`` implements the batched transient paged protocol
        (``request_cache_for_transient``) over per-row prefix caches sharing
        one pool. Returns modality-routed final-norm hidden ``[B, T, hidden]``.
        CUDA-graph callers pass the static flattened text-row ``text_idx``;
        eager callers may omit it and derive the indexes from ``is_gen``.
        """
        batch, n_tokens, hidden_size = inputs_embeds.shape
        hidden_states = inputs_embeds.reshape(batch * n_tokens, hidden_size)
        cos, sin = self.rotary.cos_sin_1d(positions.reshape(-1))
        # Flat indexes of the (few) text-modality rows — one nonzero for the
        # whole step; the per-layer dispatch is index-based (see the layer's
        # ``forward_paged_gen_batch`` docstring).
        if text_idx is None:
            text_base = (~is_gen.reshape(-1)).nonzero(as_tuple=False).reshape(-1)
            if int(text_base.numel()):
                row_offsets = (
                    torch.arange(batch, device=text_base.device, dtype=text_base.dtype)
                    * n_tokens
                )
                text_idx = (row_offsets.unsqueeze(1) + text_base.unsqueeze(0)).reshape(-1)
        for layer_idx, layer_module in enumerate(self.layers):
            layer = cast(MoTDecoderLayer, layer_module)
            hidden_states = layer.forward_paged_gen_batch(
                layer_idx,
                hidden_states,
                text_idx,
                cos,
                sin,
                past_key_values,
                batch,
                n_tokens,
            )
        out = self.final_norm[Modality.GEN](hidden_states)
        if text_idx is not None:
            out.index_copy_(
                0,
                text_idx,
                self.final_norm[Modality.TEXT](hidden_states.index_select(0, text_idx)),
            )
        return out.view(batch, n_tokens, hidden_size)

    @torch.no_grad()
    def forward_segments(self, segs: list[Segment]) -> list[torch.Tensor]:
        """Run all layers over a batch of segments; return final-norm hidden
        per segment (in original token order)."""
        if not segs:
            return []
        sizes = [s.embeds.shape[0] for s in segs]
        slices, off = [], 0
        for n in sizes:
            slices.append(slice(off, off + n))
            off += n

        H = torch.cat([s.embeds for s in segs], dim=0)
        positions = torch.cat([s.positions for s in segs], dim=0)
        gen_mask = torch.cat([s.is_gen for s in segs], dim=0)
        text_mask = ~gen_mask
        cos, sin = self.rotary.cos_sin_1d(positions)          # (N, head_dim/2)

        # Resolve modality presence once per forward to avoid per-layer GPU syncs.
        any_text = bool(text_mask.any())
        any_gen = bool(gen_mask.any())
        seg_is_gen = [bool(seg.is_gen.any()) for seg in segs]

        for li, layer_module in enumerate(self.layers):
            layer = cast(MoTDecoderLayer, layer_module)
            H = layer.forward(
                li, H, text_mask, gen_mask, cos, sin, segs, slices,
                any_text=any_text, any_gen=any_gen, seg_is_gen=seg_is_gen,
            )

        final_routes: dict[Modality, tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]] = {}
        if any_text:
            final_routes[Modality.TEXT] = (text_mask, self.final_norm[Modality.TEXT])
        if any_gen:
            final_routes[Modality.GEN] = (gen_mask, self.final_norm[Modality.GEN])
        out = route_by_modality(
            H,
            final_routes,
            transport=self._tower_transport,
            coords=self._tower_coords,
        )
        return [out[sl] for sl in slices]
