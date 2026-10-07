"""Checkpoint-exact MiniMax H3 audio encoder and decoder.

The decoder is a 1x1 input convolution followed by the BigVGAN network from
``diffusers``; ``receptive_field`` bounds the latent context a windowed
decode needs. The encoder is the DAC waveform network, the causal attention
projection that narrows its width to the latent channels, and the posterior
mean head. Both treat the two stereo channels as a batch of two mono
timelines.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrizations

from uniserve.nn.attention import Attention, AttentionBatch, DenseInput
from uniserve.nn.vae import ChannelStatistics, LatentDecoder, LatentEncoder

__all__ = [
    "Config",
    "Encoder",
    "Model",
    "encoder_assignments",
    "receptive_field",
]


# The BigVGAN alias-free activations resample by this ratio with this filter
# length, fixed by the decoder module rather than by the checkpoint.
_ACTIVATION_RATIO = 2
_ACTIVATION_KERNEL = 12


def receptive_field(config: Config) -> int:
    """Latent frames of context one decoded sample depends on, per side.

    Walks the decoder in forward order accumulating how far one dependency
    spreads. ``rate`` is the number of output samples one latent frame has
    become at the current point, so a span of ``n`` samples there is ``n /
    rate`` latent frames. The residual branches of a stage run in parallel on
    the same input, so a stage contributes the widest of them rather than their
    sum. The result is the context a media unit must carry on each side for its
    decode to equal the whole-track decode of the same samples; it rounds up
    and may exceed the exact field, which costs context and never correctness.
    """
    spread, rate = 0.0, 1.0

    def convolution(kernel: int, dilation: int = 1) -> float:
        return (kernel - 1) * dilation / rate

    def activation() -> float:
        # An alias-free activation upsamples through a sinc filter, applies
        # SnakeBeta, and low-passes back down at the same ratio.
        upsampled = (
            math.ceil(_ACTIVATION_KERNEL / _ACTIVATION_RATIO) - 1
        ) / rate
        return upsampled + (_ACTIVATION_KERNEL - 1) / (rate * _ACTIVATION_RATIO)

    spread += convolution(7)
    for upsample, kernel in zip(
        config.decoder_rates, config.decoder_kernel_sizes, strict=True
    ):
        spread += (math.ceil(kernel / upsample) - 1) / rate
        rate *= upsample
        widest = 0.0
        for block_kernel, dilations in zip(
            config.resblock_kernel_sizes,
            config.resblock_dilation_sizes,
            strict=True,
        ):
            branch = 0.0
            for dilation in dilations:
                branch += activation() + convolution(block_kernel, dilation)
                branch += activation() + convolution(block_kernel)
            widest = max(widest, branch)
        spread += widest
    spread += activation() + convolution(7)

    # The spread covers both sides of one dependency, so one side is half.
    return math.ceil(spread / 2.0)


@dataclass(frozen=True, slots=True)
class Config:
    """Native audio network and channel normalization.

    The product of ``encoder_rates`` is the number of samples per latent
    frame: the encoder's input hop and the decoder's output rate
    (``AudioDecoder.latent_rate``).
    """

    encoder_dim: int = 64
    encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5)
    latent_dim: int = 2048
    latent_channels: int = 32
    decoder_dim: int = 1024
    decoder_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2)
    decoder_kernel_sizes: tuple[int, ...] = (9, 9, 4, 4, 4, 4, 4)
    num_attention_heads: int = 8
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    latents_mean: tuple[float, ...] = (
        -0.020211687488382354,
        0.3876466479950502,
        -0.04398279799186767,
        -0.28591514936373,
        0.08179686214561671,
        -0.35782641352446604,
        0.040623809960919084,
        -0.01552534501956604,
        -0.223362481667332,
        0.1821006842509091,
        0.2941778783780663,
        -0.07901167601970885,
        -0.056815072777201,
        -0.3699028221860095,
        -0.31616315591624855,
        0.5905951377425391,
        -0.052139568068853864,
        0.013673160263486295,
        -0.03691647864630577,
        0.09732660653298163,
        -0.3394662328788498,
        -0.30685677538541667,
        -0.24504598907458763,
        -0.034698524462007344,
        0.02868032184767538,
        -0.21217779266454084,
        -0.1678263169941987,
        0.3221287889040614,
        -0.1223055851554907,
        0.4356604928128464,
        -0.0502599202236253,
        0.3979258376211797,
    )
    latents_std: tuple[float, ...] = (
        1.6895524230479284,
        2.76263727217653,
        1.7945344281264435,
        1.6801681847309828,
        1.6390226546605453,
        2.7788298348882177,
        1.7659090095747236,
        1.6199757612137327,
        2.6336525640336896,
        1.8539356672817833,
        2.5056497896915633,
        1.811019237886178,
        1.9579657790720237,
        1.6685498243529284,
        1.4922469314453364,
        3.298670198067373,
        1.9491804496832168,
        1.8720003270431442,
        1.8334080103291832,
        1.6488070416529093,
        1.6176957696319716,
        1.9131449234774398,
        1.5695245398428617,
        1.6943659940415912,
        1.8318420762504692,
        1.5540637421583379,
        1.9344930328968526,
        1.599198216109855,
        1.718045989838149,
        1.6307219190837705,
        1.8661226051202384,
        1.5613768203168363,
    )

    def __post_init__(self) -> None:
        for name in ("latents_mean", "latents_std"):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or len(values) != self.latent_channels
            ):
                raise ValueError(
                    f"H3 {name} must describe every latent channel"
                )
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in values
            ):
                raise ValueError(f"H3 {name} must be finite")
        if any(value <= 0 for value in self.latents_std):
            raise ValueError("H3 latent standard deviations must be positive")

        for name in (
            "encoder_dim",
            "latent_dim",
            "latent_channels",
            "decoder_dim",
            "num_attention_heads",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"H3 audio {name} must be a positive integer")
        for name in (
            "encoder_rates",
            "decoder_rates",
            "decoder_kernel_sizes",
            "resblock_kernel_sizes",
        ):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or not values
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value <= 0
                    for value in values
                )
            ):
                raise ValueError(
                    f"H3 audio {name} must contain positive integers"
                )

        if len(self.decoder_rates) != len(self.decoder_kernel_sizes):
            raise ValueError(
                "H3 audio decoder rates and kernels must have matching stages"
            )
        if not isinstance(self.resblock_dilation_sizes, tuple) or len(
            self.resblock_dilation_sizes
        ) != len(self.resblock_kernel_sizes):
            raise ValueError(
                "H3 audio residual kernels and dilations "
                "must have matching stages"
            )
        if any(
            not isinstance(row, tuple)
            or not row
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in row
            )
            for row in self.resblock_dilation_sizes
        ):
            raise ValueError(
                "H3 audio residual dilations must contain positive integers"
            )
        if self.latent_dim % self.num_attention_heads:
            raise ValueError(
                "H3 audio latent width must be divisible by attention heads"
            )


class Model(LatentDecoder):
    """Invert audio latent statistics and reconstruct interleaved stereo PCM."""

    def __init__(self, config: Config):
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (  # noqa: E501
            MiniMaxH3AudioBigVGANDecoder,
        )

        decoder = nn.Sequential(
            OrderedDict(
                (
                    (
                        "input",
                        nn.Conv1d(config.latent_channels, config.latent_dim, 1),
                    ),
                    (
                        "network",
                        MiniMaxH3AudioBigVGANDecoder(
                            in_channels=config.latent_dim,
                            upsample_initial_channel=config.decoder_dim,
                            upsample_rates=config.decoder_rates,
                            upsample_kernel_sizes=config.decoder_kernel_sizes,
                            resblock_kernel_sizes=config.resblock_kernel_sizes,
                            resblock_dilation_sizes=config.resblock_dilation_sizes,
                        ),
                    ),
                )
            )
        )
        # The checkpoint supplies the exact anti-alias filters as numerical
        # weights. Register them as non-trainable Parameters so the ordinary
        # assignment path loads their values alongside the convolutions.
        for module in decoder.modules():
            for name, value in tuple(module.named_buffers(recurse=False)):
                if name not in module._non_persistent_buffers_set:
                    delattr(module, name)
                    module.register_parameter(
                        name, nn.Parameter(value, requires_grad=False)
                    )
        # [stereo, channels, frames]: each stereo channel is one batch row
        # of a variable-length latent timeline.
        super().__init__(
            decoder,
            latent_shape=(2, config.latent_channels, None),
            normalization=ChannelStatistics(
                mean=torch.tensor(
                    config.latents_mean, dtype=torch.float32, device="cpu"
                ).view(1, config.latent_channels, 1),
                std=torch.tensor(
                    config.latents_std, dtype=torch.float32, device="cpu"
                ).view(1, config.latent_channels, 1),
            ),
        )
        self.config = config

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        decoded = super().forward(latents).float()
        if decoded.ndim != 3 or decoded.shape[:2] != (2, 1):
            raise RuntimeError(
                "audio decoder must produce two mono channel timelines"
            )
        # [stereo, samples] float waveforms become [samples, 2] int16 PCM.
        return (
            decoded[:, 0]
            .transpose(0, 1)
            .clamp(-1.0, 1.0)
            .mul(32767.0)
            .round()
            .to(torch.int16)
            .contiguous()
        )


def assignments(model: Model, reader):
    """Map native decoder fields without retaining the checkpoint encoder."""
    from uniserve.loading import weights

    available = frozenset(reader.names())
    values = []
    # Translate the module's sequential container names to native field names.
    for name, parameter in model.named_parameters():
        source = name.replace("decoder.input.", "dec_in_proj.").replace(
            "decoder.network.", "decoder."
        )
        if source in available:
            values.append(weights.Assignment(parameter, reader.get(source)))
    return tuple(values)


def _normalized(convolution: nn.Conv1d) -> nn.Conv1d:
    """Parameterize a convolution's weight by magnitude and direction.

    The checkpoint stores the encoder's convolutions in this weight-norm form
    (``weight_g``, ``weight_v``) and the weight is recomputed from both, as
    the reference does on every call.
    """
    return parametrizations.weight_norm(convolution)


class Snake(nn.Module):
    """Apply ``x + sin(alpha * x)^2 / alpha`` with one frequency per channel."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # The 1e-9 offset keeps the reciprocal finite for a zero frequency.
        return values + (self.alpha + 1e-9).reciprocal() * torch.sin(
            self.alpha * values
        ).pow(2)


class ResidualUnit(nn.Module):
    """Add a dilated Snake convolution branch that preserves the length."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake(channels),
            _normalized(
                nn.Conv1d(
                    channels,
                    channels,
                    7,
                    dilation=dilation,
                    padding=3 * dilation,
                )
            ),
            Snake(channels),
            _normalized(nn.Conv1d(channels, channels, 1)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.block(values)


class EncoderBlock(nn.Module):
    """Apply three dilated residual units, then a strided widening step.

    The strided convolution doubles the channels and divides the length by
    ``stride``.
    """

    def __init__(self, channels: int, stride: int):
        super().__init__()
        width = channels // 2
        self.block = nn.Sequential(
            ResidualUnit(width, 1),
            ResidualUnit(width, 3),
            ResidualUnit(width, 9),
            Snake(width),
            _normalized(
                nn.Conv1d(
                    width,
                    channels,
                    2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                )
            ),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class GeGLU(nn.Module):
    """Normalize, then apply a tanh-GELU gated linear unit and its output."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.gate = nn.Linear(channels, hidden)
        self.value = nn.Linear(channels, hidden)
        self.output = nn.Linear(hidden, channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.norm(values)
        return self.output(
            F.gelu(self.gate(values), approximate="tanh") * self.value(values)
        )


class AttentionProjection(nn.Module):
    """Narrow ``[batch, frames, width]`` features to the latent channels.

    A linear shortcut of the normalized input adds to causal self-attention
    over the latent frames, whose heads are averaged instead of concatenated
    and whose head width is average-pooled down to the latent channels; a
    normalized GeGLU residual follows. Every frame therefore depends on all
    earlier frames of the track. The key bias is the checkpoint's stored zero
    vector.
    """

    def __init__(self, config: Config):
        super().__init__()
        width, channels = config.latent_dim, config.latent_channels
        heads = config.num_attention_heads
        self.shortcut_norm = nn.LayerNorm(width)
        self.shortcut = nn.Linear(width, channels)
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention = Attention(heads, heads, width // heads)
        self.attention_output = nn.Linear(channels, channels)
        self.mlp_norm = nn.LayerNorm(channels)
        self.mlp = GeGLU(channels, 2 * channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, frames, _ = values.shape
        heads, head_dim = self.attention.num_heads, self.attention.head_dim

        # [batch, frames, 3 * width] -> three [batch, heads, frames, head_dim].
        query, key, value = (
            self.qkv(self.attention_norm(values))
            .reshape(batch, frames, 3, heads, head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attended = self.attention(
            query,
            key,
            value,
            AttentionBatch.single(DenseInput(causal=True, mask=None)),
        )
        # Average the heads of every frame, then pool the head width down to
        # the latent channels: [batch, frames, channels].
        pooled = F.adaptive_avg_pool1d(
            attended.transpose(1, 2).mean(dim=2),
            self.attention_output.in_features,
        )
        hidden = self.shortcut(
            self.shortcut_norm(values)
        ) + self.attention_output(pooled)
        return hidden + self.mlp(self.mlp_norm(hidden))


class Encoder(nn.Module):
    """Encode mono ``[batch, 1, samples]`` waveforms into posterior means.

    The DAC network downsamples by the product of ``encoder_rates`` into
    ``[batch, latent_dim, frames]`` features; the attention projection
    narrows them to the latent channels, and the mean head returns the
    posterior mode ``[batch, latent_channels, frames]``, the latent that
    conditions H3. The log-scale head is never evaluated. ``samples`` must be
    a whole number of latent frames.
    """

    def __init__(self, config: Config):
        super().__init__()
        width = config.encoder_dim
        layers: list[nn.Module] = [
            _normalized(nn.Conv1d(1, width, 7, padding=3))
        ]
        for stride in config.encoder_rates:
            width *= 2
            layers.append(EncoderBlock(width, stride))
        layers += [
            Snake(width),
            _normalized(nn.Conv1d(width, config.latent_dim, 3, padding=1)),
        ]
        self.network = nn.Sequential(*layers)
        self.projection = AttentionProjection(config)
        self.mean_projection = nn.Conv1d(
            config.latent_channels, config.latent_channels, 1
        )

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        hidden = self.network(samples)
        # The projection attends over frames: [batch, frames, latent_dim].
        hidden = self.projection(hidden.transpose(1, 2)).transpose(1, 2)
        return self.mean_projection(hidden)


# Module paths of ``Encoder`` and the native names they load from.
_ENCODER_SOURCES = (
    ("network.", "encoder.block."),
    ("projection.shortcut_norm.", "pre_block.norm3."),
    ("projection.shortcut.", "pre_block.proj."),
    ("projection.attention_norm.", "pre_block.norm1."),
    ("projection.qkv.", "pre_block.attn.qkv."),
    ("projection.attention_output.", "pre_block.attn.proj."),
    ("projection.mlp_norm.", "pre_block.norm2."),
    ("projection.mlp.norm.", "pre_block.mlp.norm."),
    ("projection.mlp.gate.", "pre_block.mlp.w0."),
    ("projection.mlp.value.", "pre_block.mlp.w1."),
    ("projection.mlp.output.", "pre_block.mlp.w2."),
    ("mean_projection.", "mean_proj."),
)


def encoder_assignments(model: Encoder | LatentEncoder, reader):
    """Map the native DAC encoder, ``pre_block`` and ``mean_proj``.

    The checkpoint's ``encoder.*``, ``pre_block.*`` and ``mean_proj.*``
    tensors become resident. Weight-normalized convolutions load their
    ``weight_g`` and ``weight_v`` fields, and the merged attention bias
    loads the query bias, the stored zero key bias and the value bias.
    ``logs_proj`` and the decoder half stay nonresident.
    """
    from uniserve.loading import weights

    encoder = model.encoder if isinstance(model, LatentEncoder) else model
    available = frozenset(reader.names())
    values = []
    for name, parameter in encoder.named_parameters():
        if name == "projection.qkv.bias":
            width = parameter.shape[0] // 3
            for index, field in enumerate(("q_bias", "zero_k_bias", "v_bias")):
                source = f"pre_block.attn.{field}"
                if source in available:
                    values.append(
                        weights.Assignment(
                            parameter,
                            reader.get(source),
                            target_slice=(
                                slice(index * width, (index + 1) * width),
                            ),
                        )
                    )
            continue

        match = next(
            (
                (prefix, native)
                for prefix, native in _ENCODER_SOURCES
                if name.startswith(prefix)
            ),
            None,
        )
        if match is None:
            raise ValueError(f"unmapped audio encoder weight {name!r}")
        prefix, native = match
        source = native + name.removeprefix(prefix)
        # A weight-norm parametrization holds the magnitude as original0 and
        # the direction as original1.
        source = source.replace(
            ".parametrizations.weight.original0", ".weight_g"
        ).replace(".parametrizations.weight.original1", ".weight_v")
        if source in available:
            values.append(weights.Assignment(parameter, reader.get(source)))
    return tuple(values)
