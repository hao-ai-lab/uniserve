"""Rotary embedding modules for packed and HF-shaped decoders."""
from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from uniserve_worker.ops import apply_rotary_emb, apply_rotary_pos_emb, qk_norm_rope, rotate_half

__all__ = [
    "rotate_half",
    "apply_rotary_emb",
    "qk_norm_rope",
    "apply_rotary_pos_emb",
    "RotaryEmbedding",
    "HFRotaryEmbedding",
    "get_rope",
]

try:  # transformers is present in production, but keep shared layers importable in light envs.
    from transformers.modeling_rope_utils import (
        ROPE_INIT_FUNCTIONS as _HF_ROPE_INIT_FUNCTIONS,
    )
    from transformers.modeling_rope_utils import (
        dynamic_rope_update as _hf_dynamic_rope_update,
    )
except Exception:  # pragma: no cover - exercised only in minimal dependency environments.
    ROPE_INIT_FUNCTIONS: dict[str, Callable[..., tuple[torch.Tensor, float]]] = {}

    def dynamic_rope_update(fn):
        return fn
else:
    ROPE_INIT_FUNCTIONS = _HF_ROPE_INIT_FUNCTIONS
    dynamic_rope_update = _hf_dynamic_rope_update


def _cos_sin_bshd(
    inv_freq: torch.Tensor,
    attention_scaling: float,
    x: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return duplicated cos/sin shaped ``[batch, seq, dim]`` for HF-shaped decoders."""

    inv_freq_expanded = inv_freq[None, :, None].float().expand(
        position_ids.shape[0], -1, 1
    ).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()
    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    """Default rotary embedding with an optional Qwen frequency-range mode."""

    inv_freq: torch.Tensor
    def __init__(
        self,
        dim: int,
        *,
        theta: float = 10000.0,
        attention_scaling: float = 1.0,
        keep_freq_range: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        # ``keep_freq_range`` selects the dual-resolution rope recipe: a closed
        # model-recipe choice, not a generic toggle. When set, inv_freq is built
        # over ``dim * 2`` and decimated by two so the kept frequencies span the same
        # range as the full-dim model while the per-axis rope uses only ``dim`` entries.
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.attention_scaling = attention_scaling
        self.keep_freq_range = bool(keep_freq_range)
        inv_dim = dim * 2 if keep_freq_range else dim
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, inv_dim, 2, dtype=torch.float32, device=device) / inv_dim)
        )
        if keep_freq_range:
            inv_freq = inv_freq[::2]
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def materialize_load_buffers(self, device: torch.device | str) -> None:
        if not self.inv_freq.is_meta:
            return
        inv_dim = self.dim * 2 if self.keep_freq_range else self.dim
        inv_freq = 1.0 / (
            self.theta
            ** (
                torch.arange(0, inv_dim, 2, dtype=torch.float32, device=device)
                / inv_dim
            )
        )
        self.inv_freq = inv_freq[::2] if self.keep_freq_range else inv_freq

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return duplicated cos/sin shaped ``[batch, seq, dim]``."""

        return _cos_sin_bshd(self.inv_freq, self.attention_scaling, x, position_ids)

    @torch.no_grad()
    def cos_sin_1d(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unduplicated packed-decoder cos/sin shaped ``[seq, dim/2]``."""

        freqs = position_ids.float()[:, None] * self.inv_freq[None, :].to(position_ids.device)
        return freqs.cos(), freqs.sin()


def _compute_default_rope_parameters(config: Any, device=None, **_kwargs) -> tuple[torch.Tensor, float]:
    """Default HF/Qwen-style RoPE frequencies with stable transformers-version behavior."""

    base = _rope_config_float(config, "rope_theta", default=10000.0)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, attention_factor


def _rope_config_float(config: Any, name: str, *, default: float) -> float:
    for key in (name, f"{name}_hw"):
        try:
            value = getattr(config, key)
        except AttributeError:
            value = None
        if value is not None:
            return float(value)
    for mapping_name in ("rope_parameters", "rope_scaling"):
        mapping = getattr(config, mapping_name, None)
        if isinstance(mapping, dict) and mapping.get(name) is not None:
            return float(mapping[name])
    return float(default)


class HFRotaryEmbedding(nn.Module):
    """HF-shaped rotary embedding with optional frequency-range preservation."""

    inv_freq: torch.Tensor

    def __init__(self, config: Any, *, device=None, keep_freq_range: bool = False):
        # ``keep_freq_range`` selects the dual-resolution rope recipe: a closed
        # model-recipe choice, not a generic toggle. When set, the rope init fn
        # is wrapped (see ``_keep_freq_range``) so inv_freq is computed over a doubled
        # head_dim and decimated by two, preserving the full-dim frequency range.
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.keep_freq_range = bool(keep_freq_range)
        if self.rope_type == "default" or self.rope_type is None:
            base_rope_init_fn = _compute_default_rope_parameters
        else:
            try:
                base_rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            except KeyError as exc:
                raise ValueError(f"unknown rope type {self.rope_type!r}") from exc

        if self.keep_freq_range:
            self.rope_init_fn = self._keep_freq_range(base_rope_init_fn)
        else:
            self.rope_init_fn = base_rope_init_fn

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def materialize_load_buffers(self, device: torch.device | str) -> None:
        if not self.inv_freq.is_meta:
            return
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.inv_freq = inv_freq
        self.original_inv_freq = inv_freq

    def _keep_freq_range(self, base_rope_init_fn):
        def _rope_init_fn_keep_freq_range(cfg: Any, dev=None):
            inv_freq, attention_scaling = base_rope_init_fn(cfg, dev)
            del inv_freq

            # The base init functions only read scalar fields from the config, so a
            # shallow copy with an overridden head_dim is sufficient and avoids the
            # cost of deep-copying the entire (potentially large/nested) config.
            cfg2 = copy.copy(cfg)
            head_dim = getattr(cfg2, "head_dim", None)
            if head_dim is None:
                head_dim = cfg2.hidden_size // cfg2.num_attention_heads
            cfg2.head_dim = int(head_dim) * 2

            inv_freq_full, _ = base_rope_init_fn(cfg2, dev)
            return inv_freq_full[::2], attention_scaling

        return _rope_init_fn_keep_freq_range

    @torch.no_grad()
    @dynamic_rope_update
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _cos_sin_bshd(self.inv_freq, self.attention_scaling, x, position_ids)

    @torch.no_grad()
    def cos_sin_1d(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids.float()[:, None] * self.inv_freq[None, :].to(position_ids.device)
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling
        return cos, sin


def get_rope(
    dim: int | None = None,
    *,
    theta: float = 10000.0,
    attention_scaling: float = 1.0,
    keep_freq_range: bool = False,
    config: Any | None = None,
    device: torch.device | str | None = None,
) -> RotaryEmbedding | HFRotaryEmbedding:
    """Build the shared RoPE implementation for packed or HF-shaped decoders.

    ``keep_freq_range`` selects the dual-resolution rope recipe: a closed
    model-recipe choice, not a generic toggle. It is forwarded to the underlying
    ``RotaryEmbedding`` / ``HFRotaryEmbedding`` to preserve the full-dim frequency range.
    """

    if config is not None:
        return HFRotaryEmbedding(config, device=device, keep_freq_range=keep_freq_range)
    if dim is None:
        raise ValueError("dim is required when config is not provided")
    return RotaryEmbedding(
        dim,
        theta=theta,
        attention_scaling=attention_scaling,
        keep_freq_range=keep_freq_range,
        device=device,
    )
