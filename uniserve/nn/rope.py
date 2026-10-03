"""Typed rotary recipes and stateless numerical position factors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import TypeAlias

import torch
from torch import nn

from uniserve.tensors import BufferConfig


def _positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"rotary {name} must be finite and positive")


def _recipe_values(recipe):
    for name, value in asdict(recipe).items():
        if value is None or name == "truncate":
            continue
        if isinstance(value, tuple):
            if not value:
                raise ValueError(f"rotary {name} cannot be empty")
            for factor in value:
                _positive(factor, name)
        else:
            _positive(value, name)
    original = getattr(recipe, "original_max_position_embeddings", None)
    if original is not None and type(original) is not int:
        raise ValueError(
            "the original rotary context length must be an integer"
        )


@dataclass(frozen=True, slots=True)
class LinearScaling:
    factor: float

    def __post_init__(self):
        _recipe_values(self)


@dataclass(frozen=True, slots=True)
class DynamicScaling:
    factor: float

    def __post_init__(self):
        _recipe_values(self)


@dataclass(frozen=True, slots=True)
class YaRNScaling:
    factor: float | None
    original_max_position_embeddings: int
    attention_factor: float | None = None
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float | None = None
    mscale_all_dim: float | None = None
    truncate: bool = True

    def __post_init__(self):
        _recipe_values(self)
        if type(self.truncate) is not bool:
            raise ValueError("YaRN truncation must be boolean")


@dataclass(frozen=True, slots=True)
class LongRoPEScaling:
    factor: float | None
    original_max_position_embeddings: int
    attention_factor: float | None
    short_factor: tuple[float, ...]
    long_factor: tuple[float, ...]

    def __post_init__(self):
        _recipe_values(self)
        if not isinstance(self.short_factor, tuple) or not isinstance(
            self.long_factor, tuple
        ):
            raise ValueError("LongRoPE factors must be immutable tuples")


@dataclass(frozen=True, slots=True)
class LlamaScaling:
    factor: float
    original_max_position_embeddings: int
    low_freq_factor: float
    high_freq_factor: float

    def __post_init__(self):
        _recipe_values(self)
        if self.high_freq_factor <= self.low_freq_factor:
            raise ValueError(
                "Llama's high-frequency boundary must exceed its low boundary"
            )


@dataclass(frozen=True, slots=True)
class ProportionalScaling:
    factor: float

    def __post_init__(self):
        _recipe_values(self)


RoPEScaling: TypeAlias = (
    LinearScaling
    | DynamicScaling
    | YaRNScaling
    | LongRoPEScaling
    | LlamaScaling
    | ProportionalScaling
)

_RECIPE_NAMES = {
    LinearScaling: "linear",
    DynamicScaling: "dynamic",
    YaRNScaling: "yarn",
    LongRoPEScaling: "longrope",
    LlamaScaling: "llama3",
    ProportionalScaling: "proportional",
}


def _interleaved_axes(sections: tuple[int, ...]) -> tuple[int, ...]:
    """Assign each compact frequency to its multimodal position axis.

    Interleaved M-RoPE (Qwen3-VL) cycles the axes over the frequencies:
    frequency ``j`` rotates by axis ``a = j % len(sections)`` while
    ``j < sections[a] * len(sections)``; axis 0 takes every remaining
    frequency. Each axis therefore receives exactly its section width, and
    every axis keeps frequencies from the whole spectrum.
    """
    count = len(sections)
    axes = []
    for frequency in range(sum(sections)):
        axis = frequency % count
        axes.append(axis if frequency < sections[axis] * count else 0)
    return tuple(axes)


class RotaryEmbedding(nn.Module):
    """Produce compact [..., dimension / 2] cosine and sine factors.

    Positions select coordinates; sequence_length independently selects a
    dynamic recipe's frequency domain. Calls never mutate model frequencies,
    so separate contexts can evaluate different lengths on shared weights.

    ``sections`` selects multimodal rotary positions (M-RoPE): positions
    then carry a leading axis of ``len(sections)`` coordinates, such as
    (temporal, height, width), and each frequency rotates by the coordinate
    of the axis it is interleaved onto (see ``_interleaved_axes``). The
    widths must sum to the compact factor width. A token whose axes share
    one coordinate, such as text, receives exactly the factors of the
    one-dimensional recipe.
    """

    def __init__(
        self,
        dim: int,
        *,
        theta: float = 10000.0,
        scaling: RoPEScaling | None = None,
        attention_scale: float = 1.0,
        keep_freq_range: bool = False,
        max_position_embeddings: int = 10000,
        partial_rotary_factor: float = 1.0,
        sections: tuple[int, ...] | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        if type(dim) is not int or dim < 2 or dim % 2:
            raise ValueError("rotary dimension must be a positive even width")
        _positive(theta, "theta")
        _positive(attention_scale, "attention_scale")
        _positive(partial_rotary_factor, "partial_rotary_factor")
        if partial_rotary_factor > 1 or max_position_embeddings < 1:
            raise ValueError(
                "rotary context and partial width must define a valid domain"
            )
        if scaling is not None and type(scaling) not in _RECIPE_NAMES:
            raise TypeError("rotary scaling must use a typed numerical recipe")

        self.dim = (
            dim
            if isinstance(scaling, ProportionalScaling)
            else int(dim * partial_rotary_factor)
        )
        if self.dim < 2 or self.dim % 2:
            raise ValueError(
                "the partial rotary width must remain positive and even"
            )
        if isinstance(scaling, DynamicScaling) and self.dim == 2:
            raise ValueError(
                "dynamic NTK scaling requires a width greater than two"
            )
        if sections is not None and (
            not isinstance(sections, tuple)
            or len(sections) < 2
            or any(type(width) is not int or width < 1 for width in sections)
            or sum(sections) != self.dim // 2
            or tuple(
                _interleaved_axes(sections).count(axis)
                for axis in range(len(sections))
            )
            != sections
        ):
            raise ValueError(
                "M-RoPE sections must be positive widths partitioning every "
                "interleaved rotary frequency"
            )

        self.theta = theta
        self.sections = sections
        self.scaling = scaling
        self.attention_scale = attention_scale
        self.keep_freq_range = keep_freq_range
        self._maximum = max_position_embeddings
        self._partial = partial_rotary_factor
        self._head_dim = dim

        if isinstance(scaling, LongRoPEScaling):
            expected = self.dim if keep_freq_range else self.dim // 2
            if (
                len(scaling.short_factor) != expected
                or len(scaling.long_factor) != expected
            ):
                raise ValueError(
                    "LongRoPE factors must cover every constructed frequency"
                )

        # Small derived model constants remain real tensors under meta model
        # construction. Loading moves each registered buffer with its module.
        actual_device = (
            torch.device("cpu") if device is None else torch.device(device)
        )
        if actual_device.type == "meta":
            actual_device = torch.device("cpu")
        inv_freq, self._frequency_scale = self._frequencies(
            actual_device, sequence_length=0
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        if sections is not None:
            # [dim / 2] position axis of each compact frequency.
            self.register_buffer(
                "frequency_axes",
                torch.tensor(
                    _interleaved_axes(sections),
                    dtype=torch.int64,
                    device=actual_device,
                ),
                persistent=False,
            )

    def _frequencies(self, device, *, sequence_length):
        if self.scaling is None:
            # keep_freq_range computes over the doubled width, then decimates
            # to retain the original frequency range at compact width.
            width = self.dim * 2 if self.keep_freq_range else self.dim
            inverse = 1.0 / (
                self.theta
                ** (
                    torch.arange(
                        0, width, 2, dtype=torch.float32, device=device
                    )
                    / width
                )
            )
            return inverse[::2] if self.keep_freq_range else inverse, 1.0

        from transformers import PretrainedConfig
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        # These are the upstream recipe's numerical arguments, assembled from
        # typed fields. Checkpoint dictionaries never enter a numerical layer.
        parameters = asdict(self.scaling)
        parameters.update(
            rope_type=_RECIPE_NAMES[type(self.scaling)],
            rope_theta=self.theta,
            partial_rotary_factor=self._partial,
        )
        # The base config declares none of these fields; like its keyword
        # constructor, attribute assignment stores them as plain attributes.
        config = PretrainedConfig()
        config.head_dim = self._head_dim * (2 if self.keep_freq_range else 1)
        config.hidden_size = self._head_dim
        config.num_attention_heads = 1
        config.max_position_embeddings = self._maximum
        config.rope_parameters = parameters
        # Upstream recipes allocate some intermediates on the default device;
        # pin it so meta model construction still yields real frequencies.
        with torch.device(device):
            frequencies, scale = ROPE_INIT_FUNCTIONS[parameters["rope_type"]](
                config, device, seq_len=sequence_length
            )
        return frequencies[::2] if self.keep_freq_range else frequencies, scale

    @torch.no_grad()
    def forward(self, positions, *, dtype: torch.dtype, sequence_length: int):
        """Return ``(cos, sin)`` factors of shape [..., dim / 2] for
        ``positions``.

        With ``sections``, ``positions`` is ``[len(sections), ...]``: one
        coordinate per position axis for each token, and the factors cover
        the trailing token axes.
        """  # noqa: D205
        if self.sections is None:
            return self._factors(
                positions, dtype=dtype, sequence_length=sequence_length
            )
        if positions.ndim < 2 or positions.shape[0] != len(self.sections):
            raise ValueError(
                "M-RoPE positions require one leading coordinate per section"
            )

        # Factors of every axis's coordinates, [axes, ..., dim / 2]; each
        # frequency then selects the factors of the axis it rotates by.
        # Selection copies values exactly, so equal coordinates on every axis
        # reproduce the one-dimensional factors bit for bit.
        cosine, sine = self._factors(
            positions, dtype=dtype, sequence_length=sequence_length
        )
        axes = self.frequency_axes.to(device=positions.device)
        selected_cosine, selected_sine = cosine[0], sine[0]
        for axis in range(1, len(self.sections)):
            chosen = axes == axis
            selected_cosine = torch.where(chosen, cosine[axis], selected_cosine)
            selected_sine = torch.where(chosen, sine[axis], selected_sine)
        return selected_cosine, selected_sine

    @torch.no_grad()
    def _factors(self, positions, *, dtype: torch.dtype, sequence_length: int):
        """Evaluate the one-dimensional recipe at every position coordinate."""
        if type(sequence_length) is not int or sequence_length < 0:
            raise ValueError(
                "rotary sequence length must be a nonnegative host integer"
            )
        if not dtype.is_floating_point or (
            positions.dtype not in {torch.int32, torch.int64}
            and not positions.is_floating_point()
        ):
            raise ValueError(
                "rotary factors require numerical positions and a floating "
                "output dtype"
            )

        dynamic = isinstance(self.scaling, (DynamicScaling, LongRoPEScaling))
        if dynamic:
            frequencies, scale = self._frequencies(
                positions.device, sequence_length=sequence_length
            )
        else:
            frequencies = self.inv_freq.to(device=positions.device)
            scale = self._frequency_scale

        from uniserve_kernels.triton import require_kernel

        from uniserve_kernels import rope

        if positions.is_cuda:
            require_kernel(
                "RotaryEmbedding",
                rope.unsupported_rotary_factors(positions, frequencies, dtype),
                positions=positions,
                frequencies=frequencies,
            )
            shape = (*positions.shape, frequencies.numel())
            cosine = torch.empty(shape, device=positions.device, dtype=dtype)
            sine = torch.empty_like(cosine)
            rope.rotary_factors(
                positions,
                frequencies,
                scale * self.attention_scale,
                cosine,
                sine,
            )
            return cosine, sine

        device_type = (
            positions.device.type if positions.device.type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            # [..., dim / 2] phases: positions broadcast against frequencies.
            phases = positions.float().unsqueeze(-1) * frequencies.float()
            cosine = phases.cos() * (scale * self.attention_scale)
            sine = phases.sin() * (scale * self.attention_scale)
        return cosine.to(dtype=dtype), sine.to(dtype=dtype)

    def constant_buffers(self, max_position: int):
        """Describe the FP32 cosine/sine table storage for ``max_position``
        rows.
        """  # noqa: D205
        if type(max_position) is not int or max_position < 0:
            raise ValueError(
                "rotary table extent must be a nonnegative integer"
            )
        return {
            name: BufferConfig((max_position, self.dim // 2), torch.float32)
            for name in ("cos", "sin")
        }

    def prepare_constants(self, max_position: int, *, out):
        """Fill caller-owned factors for one complete sequence-length domain.

        Rows are indexed by one coordinate. With ``sections`` they are the
        per-axis table every position axis looks its coordinate up in.
        """
        from .functional._tensors import result as _result

        expected = self.constant_buffers(max_position)
        if set(out) != set(expected):
            raise ValueError(
                "rotary constant storage must supply cosine and sine tables"
            )
        positions = torch.arange(max_position, device=out["cos"].device)
        values = self._factors(
            positions, dtype=torch.float32, sequence_length=max_position
        )
        for name, value in zip(("cos", "sin"), values, strict=True):
            _result(value, out[name])
