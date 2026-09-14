"""Rotary-position factor construction for packed and HF-shaped decoders.

The implementations keep inverse frequencies in nonpersistent buffers so meta
models can materialize them directly on the execution device. Callers can
request either duplicated batch/sequence factors or compact one-dimensional
factors, matching the two attention layouts used by the worker.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from uniserve.ops import apply_rotary_emb, apply_rotary_pos_emb, qk_norm_rope, rotate_half

__all__ = [
    "rotate_half",
    "apply_rotary_emb",
    "qk_norm_rope",
    "apply_rotary_pos_emb",
    "RopeScaling",
    "RotaryEmbedding",
    "HFRotaryEmbedding",
    "get_rope",
]


@dataclass(frozen=True, slots=True)
class RopeScaling:
    """Numerical scaling recipe consumed by Transformers' rotary initializers.

    Fixed factor sequences are immutable. Dimensions and the frequency base
    belong to the rotary module, so one recipe can serve temporal and spatial
    partitions without copying or mutating the model's configuration.
    """

    rope_type: str
    factor: float | None = 1.0
    original_max_position_embeddings: int | None = None
    attention_factor: float | None = None
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    short_factor: tuple[float, ...] = ()
    long_factor: tuple[float, ...] = ()
    mscale: float | None = None
    mscale_all_dim: float | None = None
    truncate: bool = True

    def __post_init__(self) -> None:
        if self.rope_type not in {
            "linear",
            "dynamic",
            "yarn",
            "longrope",
            "llama3",
            "proportional",
        }:
            raise ValueError(f"unsupported rotary scaling {self.rope_type!r}")
        for name in (
            "factor",
            "attention_factor",
            "beta_fast",
            "beta_slow",
            "low_freq_factor",
            "high_freq_factor",
            "mscale",
            "mscale_all_dim",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"rotary {name} must be finite and positive")
        if self.factor is None and self.rope_type not in {"yarn", "longrope"}:
            raise ValueError(f"{self.rope_type} requires a scaling factor")
        original = self.original_max_position_embeddings
        if original is not None and (
            not isinstance(original, int) or isinstance(original, bool) or original <= 0
        ):
            raise ValueError("rotary original_max_position_embeddings must be a positive integer")
        if self.rope_type in {"yarn", "longrope", "llama3"} and original is None:
            raise ValueError(f"{self.rope_type} requires original_max_position_embeddings")
        if self.rope_type == "llama3" and self.high_freq_factor <= self.low_freq_factor:
            raise ValueError("Llama rotary high_freq_factor must exceed low_freq_factor")
        for name in ("short_factor", "long_factor"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
                for value in values
            ):
                raise ValueError(f"rotary {name} must be a tuple of positive finite factors")
            if self.rope_type == "longrope" and not values:
                raise ValueError(f"longrope requires {name}")
        if not isinstance(self.truncate, bool):
            raise ValueError("rotary truncate must be boolean")


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
    inv_freq_expanded = (
        inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    )
    position_ids_expanded = position_ids[:, None, :].float()
    device_type = (
        x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
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
            ** (torch.arange(0, inv_dim, 2, dtype=torch.float32, device=device) / inv_dim)
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


class HFRotaryEmbedding(nn.Module):
    """Rotary-factor generator driven by a model configuration's HF recipe.

    The selected recipe owns attention scaling and may update frequencies as
    sequence lengths change. Frequency-range preservation applies the recipe at
    double head width and retains alternating frequencies.
    """

    inv_freq: torch.Tensor

    def __init__(
        self,
        dim: int,
        *,
        theta: float,
        max_position_embeddings: int,
        scaling: RopeScaling,
        partial_rotary_factor: float = 1.0,
        device=None,
        keep_freq_range: bool = False,
    ):
        """Bind a typed recipe at the third-party rotary API boundary."""

        from transformers import PretrainedConfig

        super().__init__()
        self.rope_type = scaling.rope_type
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings
        self.keep_freq_range = keep_freq_range
        # Transformers' numerical recipes and update decorator require their
        # native mutable config. It stays inside this API binding; models only
        # supply the immutable scaling recipe and explicit numerical dimensions.
        parameters = {key: value for key, value in asdict(scaling).items() if value is not None}
        parameters.update(
            rope_theta=theta, partial_rotary_factor=partial_rotary_factor, factor=scaling.factor
        )
        self.config = PretrainedConfig.from_dict(
            {
                "head_dim": dim,
                "hidden_size": dim,
                "num_attention_heads": 1,
                "max_position_embeddings": max_position_embeddings,
                "rope_parameters": parameters,
            }
        )
        base_rope_init_fn = ROPE_INIT_FUNCTIONS[scaling.rope_type]

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

        def _rope_init_fn_keep_freq_range(cfg: Any, device=None, **kwargs):
            """Return decimated double-width frequencies and the recipe's scaling."""

            # Attention scaling belongs to the selected recipe and is unchanged
            # by the frequency-width transformation.
            inv_freq, attention_scaling = base_rope_init_fn(cfg, device, **kwargs)
            del inv_freq

            # RoPE initializers read scalar geometry from the configuration, so
            # a shallow copy isolates the temporary head-width override.
            cfg2 = copy.copy(cfg)
            cfg2.head_dim = cfg.head_dim * 2

            inv_freq_full, _ = base_rope_init_fn(cfg2, device, **kwargs)
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
    scaling: RopeScaling | None = None,
    max_position_embeddings: int = 10000,
    partial_rotary_factor: float = 1.0,
    device: torch.device | str | None = None,
) -> RotaryEmbedding | HFRotaryEmbedding:
    """Build RoPE from explicit dimensions and an optional typed scaling recipe.

    ``keep_freq_range`` retains the full-width frequency span when a model uses
    reduced per-axis rotary dimensions.
    """

    if dim is None:
        raise ValueError("rotary dimension is required")
    if scaling is not None:
        return HFRotaryEmbedding(
            dim,
            theta=theta,
            max_position_embeddings=max_position_embeddings,
            scaling=scaling,
            partial_rotary_factor=partial_rotary_factor,
            device=device,
            keep_freq_range=keep_freq_range,
        )
    return RotaryEmbedding(
        int(dim * partial_rotary_factor),
        theta=theta,
        attention_scaling=attention_scaling,
        keep_freq_range=keep_freq_range,
        device=device,
    )
