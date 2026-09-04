"""Rotary-position factor construction for packed and HF-shaped decoders.

The implementations keep inverse frequencies in nonpersistent buffers so meta
models can materialize them directly on the execution device. Callers can
request either duplicated batch/sequence factors or compact one-dimensional
factors, matching the two attention layouts used by the worker.
"""

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

# Transformers supplies the scaled and dynamically updated RoPE recipes. The
# local default remains usable when that optional integration is unavailable.
try:
    from transformers.modeling_rope_utils import (
        ROPE_INIT_FUNCTIONS as _HF_ROPE_INIT_FUNCTIONS,
    )
    from transformers.modeling_rope_utils import (
        dynamic_rope_update as _hf_dynamic_rope_update,
    )
except Exception:  # pragma: no cover - depends on the installed model stack.
    ROPE_INIT_FUNCTIONS: dict[str, Callable[..., tuple[torch.Tensor, float]]] = {}

    def dynamic_rope_update(fn):
        """Use a RoPE forward function without dynamic frequency updates."""

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
    """Return duplicated cosine and sine factors shaped ``[batch, seq, dim]``."""

    # Form every batch/position phase as an outer product with the frequency
    # vector. Duplicating the phases matches the two halves consumed by
    # ``rotate_half``-style rotary application.
    inv_freq_expanded = inv_freq[None, :, None].float().expand(
        position_ids.shape[0], -1, 1
    ).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()
    device_type = (
        x.device.type
        if isinstance(x.device.type, str) and x.device.type != "mps"
        else "cpu"
    )

    # Phase construction stays in fp32 with autocast disabled; the final
    # factors return to the activation dtype at the operator boundary.
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    """Default rotary-factor generator parameterized by an explicit dimension.

    ``keep_freq_range`` constructs frequencies over twice the rotary dimension
    and keeps alternating entries. This gives a smaller per-axis factor the
    frequency span of the corresponding full-width embedding.
    """

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
        """Initialize inverse frequencies for the requested rotary geometry."""

        super().__init__()
        self.dim = dim
        self.theta = theta
        self.attention_scaling = attention_scaling
        self.keep_freq_range = bool(keep_freq_range)

        # Frequency-range preservation derives the geometric progression at
        # double width, then decimates it to the requested number of factors.
        inv_dim = dim * 2 if keep_freq_range else dim
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, inv_dim, 2, dtype=torch.float32, device=device) / inv_dim)
        )
        if keep_freq_range:
            inv_freq = inv_freq[::2]
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def materialize_load_buffers(self, device: torch.device | str) -> None:
        """Materialize a meta-initialized frequency buffer on ``device``."""

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
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return duplicated factors shaped ``[batch, seq, dim]`` in ``x``'s dtype."""

        return _cos_sin_bshd(self.inv_freq, self.attention_scaling, x, position_ids)

    @torch.no_grad()
    def cos_sin_1d(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return compact packed-decoder factors shaped ``[seq, dim/2]``."""

        freqs = position_ids.float()[:, None] * self.inv_freq[None, :].to(position_ids.device)
        return freqs.cos(), freqs.sin()


def _compute_default_rope_parameters(
    config: Any,
    device=None,
    **_kwargs,
) -> tuple[torch.Tensor, float]:
    """Build default HF/Qwen inverse frequencies from a model configuration."""

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
    """Read a scalar RoPE option from direct fields or nested RoPE mappings."""

    # Direct configuration fields define the active model geometry. The
    # hardware-qualified alias supports configurations that carry both forms.
    for key in (name, f"{name}_hw"):
        try:
            value = getattr(config, key)
        except AttributeError:
            value = None
        if value is not None:
            return float(value)

    # Mapping-based configurations use the same key in either supported
    # Transformers schema.
    for mapping_name in ("rope_parameters", "rope_scaling"):
        mapping = getattr(config, mapping_name, None)
        if isinstance(mapping, dict) and mapping.get(name) is not None:
            return float(mapping[name])

    return float(default)


class HFRotaryEmbedding(nn.Module):
    """Rotary-factor generator driven by a model configuration's HF recipe.

    The selected recipe owns attention scaling and may update frequencies as
    sequence lengths change. Frequency-range preservation applies the recipe at
    double head width and retains alternating frequencies.
    """

    inv_freq: torch.Tensor

    def __init__(self, config: Any, *, device=None, keep_freq_range: bool = False):
        """Resolve and initialize the RoPE recipe declared by ``config``."""

        super().__init__()

        # Resolve the serialized recipe name while accepting both HF schema
        # spellings used by model configurations.
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

        # Wrap the selected recipe before materialization so meta-buffer reloads
        # reproduce the same frequency geometry.
        if self.keep_freq_range:
            self.rope_init_fn = self._keep_freq_range(base_rope_init_fn)
        else:
            self.rope_init_fn = base_rope_init_fn

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Dynamic HF recipes restore this baseline when their cache contracts.
        self.original_inv_freq = self.inv_freq

    def materialize_load_buffers(self, device: torch.device | str) -> None:
        """Materialize meta frequency state on ``device`` using the selected recipe."""

        if not self.inv_freq.is_meta:
            return

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.inv_freq = inv_freq
        self.original_inv_freq = inv_freq

    def _keep_freq_range(self, base_rope_init_fn):
        """Wrap a RoPE initializer with double-width frequency construction."""

        def _rope_init_fn_keep_freq_range(cfg: Any, dev=None):
            """Return decimated double-width frequencies and the recipe's scaling."""

            # Attention scaling belongs to the selected recipe and is unchanged
            # by the frequency-width transformation.
            inv_freq, attention_scaling = base_rope_init_fn(cfg, dev)
            del inv_freq

            # RoPE initializers read scalar geometry from the configuration, so
            # a shallow copy isolates the temporary head-width override.
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
        """Return duplicated factors shaped ``[batch, seq, dim]`` in ``x``'s dtype."""

        return _cos_sin_bshd(self.inv_freq, self.attention_scaling, x, position_ids)

    @torch.no_grad()
    def cos_sin_1d(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return scaled compact factors for packed one-dimensional attention."""

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
    """Build RoPE from an explicit dimension or an HF model configuration.

    ``keep_freq_range`` retains the full-width frequency span when a model uses
    reduced per-axis rotary dimensions.
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
