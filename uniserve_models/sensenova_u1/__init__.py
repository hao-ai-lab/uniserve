"""Immutable SenseNova architecture and checkpoint metadata normalization."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from uniserve import loading
from uniserve.diffusion import AdditiveGuidance, EulerSolver, NoiseScale, Renorm, make_schedule
from uniserve.loading import checkpoint, weights
from uniserve.media import image
from uniserve.model import (
    CausalLM,
    EntryPoint,
    ImageDecoder,
    ImageDenoiser,
    PatchEncoder,
    TransformerDecoder,
)
from uniserve.model import (
    DenoiserInput as NumericalDenoiserInput,
)
from uniserve.nn.attention import (
    Attention,
    AttentionInput,
    AxialQKVProjection,
    DenseInput,
    PagedInput,
    SegmentedInput,
)
from uniserve.nn.linear import (
    Linear,
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelHead,
)
from uniserve.nn.mlp import GatedMLP
from uniserve.nn.norm import RMSNorm
from uniserve.nn.rope import (
    DynamicScaling,
    LinearScaling,
    LlamaScaling,
    LongRoPEScaling,
    ProportionalScaling,
    RoPEScaling,
    RotaryEmbedding,
    YaRNScaling,
)
from uniserve.nn.routing import RoutedTensor, RouteSpan
from uniserve.nn.timestep import TimestepEmbedding
from uniserve.nn.vae.patch import RGBDecoder
from uniserve.tensors import OutputLayout, TensorOutput

from . import flow, vision


def _positive(value: object, name: str, *, integer: bool = False) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or (integer and not isinstance(value, int))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            f"SenseNova {name} must be a positive {'integer' if integer else 'finite number'}"
        )


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """Text/flow decoder dimensions, normalization, and multi-axis rotary math."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    layer_types: tuple[str, ...]
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    max_position_embeddings: int = 32768
    max_position_embeddings_hw: int = 10000
    rope_theta: float = 10000.0
    rope_theta_hw: float = 10000.0
    rope_scaling: RoPEScaling | None = None
    partial_rotary_factor: float = 1.0
    sliding_window: int | None = None
    pad_token_id: int | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "max_position_embeddings_hw",
        ):
            _positive(getattr(self, name), name, integer=True)
        for name in ("rms_norm_eps", "rope_theta", "rope_theta_hw", "partial_rotary_factor"):
            _positive(getattr(self, name), name)
        if self.head_dim % 4:
            raise ValueError("SenseNova head_dim must divide into temporal/height/width partitions")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("SenseNova attention heads must be divisible by KV heads")
        if self.partial_rotary_factor > 1:
            raise ValueError("SenseNova partial_rotary_factor must not exceed one")
        if (
            not isinstance(self.layer_types, tuple)
            or len(self.layer_types) != self.num_hidden_layers
        ):
            raise ValueError("SenseNova layer_types must describe every decoder layer")
        if any(value not in {"full_attention", "sliding_attention"} for value in self.layer_types):
            raise ValueError("SenseNova layer_types contains an unsupported attention type")
        if self.sliding_window is not None:
            _positive(self.sliding_window, "sliding_window", integer=True)
        if "sliding_attention" in self.layer_types:
            if self.sliding_window is None:
                raise ValueError("SenseNova sliding_attention requires sliding_window")
            # The packed multimodal mask implements full attention within image
            # blocks and causal text visibility, without a local-window cutoff.
            raise ValueError(
                "SenseNova layer_types: sliding_attention is not supported by the MoT decoder"
            )
        for name in ("attention_bias", "tie_word_embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"SenseNova {name} must be boolean")
        if self.pad_token_id is not None and (
            not isinstance(self.pad_token_id, int)
            or isinstance(self.pad_token_id, bool)
            or not 0 <= self.pad_token_id < self.vocab_size
        ):
            raise ValueError("SenseNova pad_token_id must be inside the vocabulary")
        if self.hidden_act not in {
            "silu",
            "swish",
            "silu_and_mul",
            "swiglu",
            "gelu",
            "gelu_and_mul",
            "geglu",
            "gelu_pytorch_tanh",
            "gelu_tanh",
        }:
            raise ValueError(f"unsupported SenseNova hidden_act {self.hidden_act!r}")


@dataclass(frozen=True, slots=True)
class Config:
    """Compose the language, patch embedding, and image prediction networks."""

    text: TransformerConfig
    vision: vision.Config
    flow: flow.Config
    max_image_seq_len: int = 4096

    def __post_init__(self) -> None:
        _positive(self.max_image_seq_len, "max_image_seq_len", integer=True)
        if self.vision.output_size != self.text.hidden_size:
            raise ValueError("SenseNova vision output must match text hidden_size")
        if self.vision.num_channels != 3:
            raise ValueError("SenseNova image prediction requires RGB vision inputs")


def _stage_scalar(value: Any, name: str) -> Any:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"SenseNova vision_config.{name} must not be empty")
        return value[0]
    return value


def _alias(
    primary: Mapping[str, Any], name: str, aliases: tuple[Mapping[str, Any], ...], default: Any
) -> Any:
    values = [source[name] for source in (primary, *aliases) if source.get(name) is not None]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"SenseNova checkpoint has conflicting aliases for {name}")
    return values[0] if values else default


def _normalize(raw: Mapping[str, Any]) -> Config:
    """Resolve checkpoint aliases once, without allocating numerical resources."""

    from packaging.version import Version

    required = raw.get("uniserve_sensenova_min_version")
    if required and Version("0.1.0") < Version(str(required)):
        raise RuntimeError(f"checkpoint requires UniServe model code >= {required}")
    text, image = raw["llm_config"], raw["vision_config"]
    if not isinstance(text, Mapping) or not isinstance(image, Mapping):
        raise ValueError("SenseNova llm_config and vision_config must be objects")
    if text.get("num_experts", 0) != 0:
        raise ValueError(
            "SenseNova llm_config.num_experts: sparse MoE is not supported by the dual-route MoT decoder"
        )
    scaling = text.get("rope_scaling") or {}
    parameters = text.get("rope_parameters") or {}
    if not isinstance(scaling, Mapping) or not isinstance(parameters, Mapping):
        raise ValueError("SenseNova rotary metadata must be an object")
    kind = _alias(
        parameters, "rope_type", (scaling,), parameters.get("type", scaling.get("type", "default"))
    )
    if any(source.get("type", kind) != kind for source in (parameters, scaling)):
        raise ValueError("SenseNova checkpoint has conflicting rotary type aliases")
    recipe = None
    if kind != "default":

        def option(name, default=None):
            return _alias(parameters, name, (scaling,), default)

        factor = option("factor")
        original = option(
            "original_max_position_embeddings", text.get("max_position_embeddings", 32768)
        )
        match kind:
            case "linear":
                recipe = LinearScaling(factor)
            case "dynamic":
                recipe = DynamicScaling(factor)
            case "proportional":
                recipe = ProportionalScaling(factor)
            case "yarn":
                recipe = YaRNScaling(
                    factor,
                    original,
                    option("attention_factor"),
                    option("beta_fast", 32.0),
                    option("beta_slow", 1.0),
                    option("mscale"),
                    option("mscale_all_dim"),
                    option("truncate", True),
                )
            case "longrope":
                recipe = LongRoPEScaling(
                    factor,
                    original,
                    option("attention_factor"),
                    tuple(option("short_factor", ())),
                    tuple(option("long_factor", ())),
                )
            case "llama3":
                recipe = LlamaScaling(
                    factor,
                    original,
                    option("low_freq_factor", 1.0),
                    option("high_freq_factor", 4.0),
                )
            case _:
                raise ValueError(f"unsupported SenseNova rotary recipe {kind!r}")
    layers, heads = text["num_hidden_layers"], text["num_attention_heads"]
    _positive(layers, "llm_config.num_hidden_layers", integer=True)
    _positive(heads, "llm_config.num_attention_heads", integer=True)
    hidden = text["hidden_size"]
    _positive(hidden, "llm_config.hidden_size", integer=True)
    if "head_dim" not in text and hidden % heads:
        raise ValueError("SenseNova requires explicit head_dim for this hidden width")
    use_window = text.get("use_sliding_window", False)
    if not isinstance(use_window, bool):
        raise ValueError("SenseNova use_sliding_window must be boolean")
    window = text.get("sliding_window")
    boundary = text.get("max_window_layers", 0)
    if not isinstance(boundary, int) or isinstance(boundary, bool) or boundary < 0:
        raise ValueError("SenseNova llm_config.max_window_layers must be a non-negative integer")
    layer_types = text.get("layer_types")
    if layer_types is None:
        layer_types = tuple(
            "sliding_attention"
            if use_window and window is not None and index >= boundary
            else "full_attention"
            for index in range(layers)
        )
    if not isinstance(layer_types, (list, tuple)):
        raise ValueError("SenseNova layer_types must be a sequence")
    vision_ratio = _stage_scalar(image.get("downsample_ratio", 0.5), "downsample_ratio")
    if raw.get("downsample_ratio", vision_ratio) != vision_ratio:
        raise ValueError("SenseNova root and vision downsample_ratio must agree")
    head_layers = raw.get("fm_head_layers", 2)
    _positive(head_layers, "fm_head_layers", integer=True)
    return Config(
        text=TransformerConfig(
            vocab_size=text["vocab_size"],
            hidden_size=hidden,
            intermediate_size=text["intermediate_size"],
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text.get("head_dim", hidden // heads),
            layer_types=tuple(layer_types),
            hidden_act=text.get("hidden_act", "silu"),
            rms_norm_eps=text.get("rms_norm_eps", 1e-6),
            attention_bias=text.get("attention_bias", False),
            max_position_embeddings=text.get("max_position_embeddings", 32768),
            max_position_embeddings_hw=text.get("max_position_embeddings_hw", 10000),
            rope_theta=_alias(text, "rope_theta", (parameters, scaling), 10000.0),
            rope_theta_hw=text.get("rope_theta_hw", 10000.0),
            rope_scaling=recipe,
            partial_rotary_factor=_alias(text, "partial_rotary_factor", (parameters, scaling), 1.0),
            sliding_window=window,
            pad_token_id=text.get("pad_token_id")
            if text.get("pad_token_id") is not None
            else raw.get("pad_token_id"),
            tie_word_embeddings=_alias(text, "tie_word_embeddings", (raw,), False),
        ),
        vision=vision.Config(
            hidden_size=image.get("hidden_size", 1024),
            output_size=_stage_scalar(image.get("llm_hidden_size", 2048), "llm_hidden_size"),
            downsample_ratio=vision_ratio,
            patch_size=image.get("patch_size", 16),
            num_channels=image.get("num_channels", 3),
            rope_theta=image.get("rope_theta_vision", 10000.0),
        ),
        flow=flow.Config(
            head=flow.HeadConfig(
                hidden_size=raw["fm_head_dim"] if head_layers > 2 else 4096,
                num_layers=head_layers,
                mlp_ratio=raw["fm_head_mlp_ratio"] if head_layers > 2 else 1.0,
            ),
            use_pixel_head=raw.get("use_pixel_head", False),
            add_noise_scale_embedding=raw.get("add_noise_scale_embedding", False),
            noise=NoiseScale(
                raw.get("noise_scale", 1.0),
                "resolution"
                if raw.get("noise_scale_mode") == "dynamic"
                else raw.get("noise_scale_mode", "constant"),
                raw.get("noise_scale_base_image_seq_len", 1.0),
                raw.get("noise_scale_max_value", 1.0),
            ),
        ),
        max_image_seq_len=raw.get("max_image_seq_len", 4096),
    )


def read_config(root: Path, io: loading.Config) -> Config:
    """Normalize checkpoint aliases and typed rotary recipes before construction."""
    return _normalize(json.loads((root / "config.json").read_text()))


class TransformerLayer(nn.Module):
    """Compose route-specific experts around three-axis token attention."""

    def __init__(self, config: TransformerConfig, index: int):
        super().__init__()
        hidden, dim, eps = config.hidden_size, config.head_dim, config.rms_norm_eps
        self.input_norms = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.outputs = nn.ModuleDict()
        self.post_attention_norms = nn.ModuleDict()
        self.mlps = nn.ModuleDict()
        activation = (
            "silu"
            if config.hidden_act in {"silu", "swish", "silu_and_mul", "swiglu"}
            else "gelu"
            if config.hidden_act in {"gelu", "gelu_and_mul", "geglu"}
            else "gelu_pytorch_tanh"
        )
        for route in ("text", "flow"):
            self.input_norms[route] = RMSNorm(hidden, eps)
            self.projections[route] = AxialQKVProjection(
                QKVParallelLinear(
                    hidden,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    dim,
                    bias=config.attention_bias,
                ),
                nn.ModuleList((RMSNorm(dim // 2, eps), RMSNorm(dim // 2, eps))),
                nn.ModuleList((RMSNorm(dim // 2, eps), RMSNorm(dim // 2, eps))),
                axis_dims=(dim // 2, dim // 4, dim // 4),
                rotations=("split",) * 3,
            )
            self.outputs[route] = RowParallelLinear(
                config.num_attention_heads * dim, hidden, bias=config.attention_bias
            )
            self.post_attention_norms[route] = RMSNorm(hidden, eps)
            self.mlps[route] = GatedMLP(hidden, config.intermediate_size, activation=activation)
        self.attention = Attention(
            config.num_attention_heads,
            config.num_key_value_heads,
            dim,
            cache_name=f"text.backbone.layers.{index}.attention",
        )
        self.window = (
            config.sliding_window if config.layer_types[index] == "sliding_attention" else None
        )
        self.temporal_rotary = RotaryEmbedding(
            dim // 2,
            theta=config.rope_theta,
            scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            partial_rotary_factor=config.partial_rotary_factor,
            keep_freq_range=True,
        )
        self.spatial_rotary = RotaryEmbedding(
            dim // 4,
            theta=config.rope_theta_hw,
            scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings_hw,
            partial_rotary_factor=config.partial_rotary_factor,
            keep_freq_range=True,
        )

    def forward(
        self,
        hidden: RoutedTensor,
        residual: RoutedTensor | None,
        positions: torch.Tensor,
        attention: AttentionInput,
        *,
        routes: tuple[RouteSpan, ...],
    ):
        hidden = hidden if residual is None else hidden.add(residual)
        normalized = hidden.apply(self.input_norms)
        if positions.ndim == 1:
            positions = torch.stack(
                (positions, torch.zeros_like(positions), torch.zeros_like(positions))
            )
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError("SenseNova positions require temporal, height and width axes")
        dynamic = isinstance(self.temporal_rotary.scaling, (DynamicScaling, LongRoPEScaling))
        if not dynamic:
            length = positions.shape[1]
        elif isinstance(attention, (PagedInput, SegmentedInput)):
            if attention.queries.host is None or attention.prefixes.host is None:
                raise ValueError("dynamic rotary scaling requires exact host sequence lengths")
            length = max(
                (
                    query + prefix
                    for query, prefix in zip(
                        attention.queries.host, attention.prefixes.host, strict=True
                    )
                ),
                default=0,
            )
        else:
            length = (
                positions.shape[1]
                if isinstance(attention, DenseInput)
                else attention.queries.maximum
            )
        if length is None:
            raise ValueError("dynamic rotary scaling requires exact host sequence lengths")
        names = frozenset(hidden.values)
        # Height and width use the same frequency recipe. Evaluate their
        # independent coordinates in one call, then restore the two axes.
        spatial = tuple(
            table.reshape(2, positions.shape[1], table.shape[-1])
            for table in self.spatial_rotary(
                positions[1:].reshape(-1), dtype=torch.float32, sequence_length=length
            )
        )
        pairs = (
            self.temporal_rotary(positions[0], dtype=torch.float32, sequence_length=length),
            (spatial[0][0], spatial[1][0]),
            (spatial[0][1], spatial[1][1]),
        )
        cos, sin = (
            tuple(RoutedTensor.from_packed(pair[index], routes, routes=names) for pair in pairs)
            for index in (0, 1)
        )
        projected = {
            route: self.projections[route](
                value,
                tuple(axis.values[route] for axis in cos),
                tuple(axis.values[route] for axis in sin),
            )
            for route, value in normalized.values.items()
        }
        query, key, value = (
            RoutedTensor({route: values[index] for route, values in projected.items()}).packed(
                routes
            )
            for index in range(3)
        )
        attended = self.attention(query, key, value, attention).flatten(1)
        update = RoutedTensor.from_packed(attended, routes, routes=names).apply(self.outputs)
        residual = hidden.add(update)
        return residual.apply(self.post_attention_norms).apply(self.mlps), residual


class Transformer(TransformerDecoder):
    def __init__(self, config: TransformerConfig):
        super().__init__(
            VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
            ),
            nn.ModuleDict(
                (str(index), TransformerLayer(config, index))
                for index in range(config.num_hidden_layers)
            ),
            nn.ModuleDict(
                (route, RMSNorm(config.hidden_size, config.rms_norm_eps))
                for route in ("text", "flow")
            ),
            default_route="text",
        )
        self.config = config


@dataclass(frozen=True)
class ImageConditioning:
    """Borrow the complete noisy NCHW image, input patch grid and noise scale."""

    pixels: torch.Tensor
    grid: torch.Tensor
    noise_scale: torch.Tensor

    def __post_init__(self):
        if (
            self.pixels.ndim != 4
            or self.pixels.shape[:2] != (1, 3)
            or self.grid.shape != (1, 2)
            or self.noise_scale.numel() != 1
        ):
            raise ValueError(
                "SenseNova image conditioning requires one NCHW RGB image, grid and scalar noise scale"
            )


@dataclass(frozen=True)
class DenoiserInput(NumericalDenoiserInput[image.Config]):
    images: tuple[ImageConditioning, ...]
    positions: tuple[torch.Tensor, ...]
    sequence_lengths: tuple[int, ...]
    attention: AttentionInput

    def __post_init__(self):
        super().__post_init__()
        if any(
            len(values) != self.batch_size
            for values in (self.images, self.positions, self.sequence_lengths)
        ):
            raise ValueError("SenseNova image conditioning and positions must align with samples")
        if any(
            position.shape != (3, count)
            for position, count in zip(self.positions, self.sequence_lengths, strict=True)
        ):
            raise ValueError("SenseNova image positions must cover three axes per image token")
        if (
            not isinstance(self.attention, DenseInput)
            and self.attention.queries.host is not None
            and self.attention.queries.host != self.sequence_lengths
        ):
            raise ValueError("SenseNova attention lengths must match its image sequences")


class Denoiser(ImageDenoiser[DenoiserInput]):
    def __init__(self, config: Config, backbone: Transformer):
        stride = config.vision.patch_size * round(1 / config.vision.downsample_ratio)
        super().__init__(
            patch_size=stride,
            latent_channels=3,
            downsample=stride,
            noise_scale=config.flow.noise,
            prediction_dtype=torch.float32,
            solver=EulerSolver("velocity"),
        )
        self.config, self.backbone = config, backbone
        self.input = vision.Encoder(config.vision)
        self.time_embedding = TimestepEmbedding(config.text.hidden_size)
        self.noise_embedding = (
            TimestepEmbedding(config.text.hidden_size)
            if config.flow.add_noise_scale_embedding
            else None
        )
        if config.flow.use_pixel_head:
            head = nn.Identity()
            decoder = flow.Decoder(config.text.hidden_size, final_upscale=stride // 4)
        elif config.flow.head.num_layers > 2:
            head = flow.Head(
                config.flow.head, input_size=config.text.hidden_size, output_size=3 * stride**2
            )
            decoder = nn.Identity()
        else:
            head = nn.Sequential(
                Linear(config.text.hidden_size, config.flow.head.hidden_size),
                nn.GELU(),
                Linear(config.flow.head.hidden_size, 3 * stride**2),
            )
            decoder = nn.Identity()
        self.prediction = flow.Velocity(head, decoder, patch_size=stride)

    def make_schedules(self, steps, *, shift, device):
        return {
            "image": make_schedule(
                steps,
                shift=1.0 if shift is None else shift,
                direction="ascending",
                shift_domain="sigma",
                device=device,
            )
        }

    def make_guidance(
        self,
        *,
        text_scale: float,
        image_scale: float,
        interval: tuple[float, float],
        renorm: Renorm,
        renorm_min: float,
    ):
        return AdditiveGuidance(text_scale, image_scale, interval, renorm, renorm_min)

    def forward(self, inputs: DenoiserInput, *, state, constants, workspace):
        if set(inputs.latents) != {"image"}:
            raise ValueError("SenseNova predicts the image latent modality")
        if not inputs.batch_size:
            return {"image": ()}
        for latent, size, conditioning, count in zip(
            inputs.latents["image"],
            inputs.sizes,
            inputs.images,
            inputs.sequence_lengths,
            strict=True,
        ):
            shape = self.latent_shape("image", size)
            if (
                latent.tensor.shape != shape
                or shape[0] != count
                or conditioning.pixels.shape[-2:] != (size.height, size.width)
            ):
                raise ValueError(
                    "SenseNova samples and conditioning must cover their declared image dimensions"
                )
        pipeline = self.mesh.get_group("pp" if "pp" in self.mesh.axes else ())
        hidden = None
        if pipeline.rank == 0:
            patch = self.config.vision.patch_size
            shapes = tuple((size.height // patch, size.width // patch) for size in inputs.sizes)
            pixels = torch.cat(
                tuple(
                    value.pixels.reshape(1, 3, height, patch, width, patch)
                    .permute(0, 2, 4, 1, 3, 5)
                    .reshape(-1, 3 * patch**2)
                    for value, (height, width) in zip(inputs.images, shapes, strict=True)
                )
            )
            hidden = self.input(
                pixels, torch.cat(tuple(value.grid for value in inputs.images)), shapes
            )
            times = torch.cat(
                tuple(
                    latent.timestep.reshape(1).expand(count)
                    for latent, count in zip(
                        inputs.latents["image"], inputs.sequence_lengths, strict=True
                    )
                )
            )
            hidden = hidden + self.time_embedding(times)
            if self.noise_embedding is not None:
                scales = torch.cat(
                    tuple(
                        value.noise_scale.reshape(1).expand(count)
                        for value, count in zip(inputs.images, inputs.sequence_lengths, strict=True)
                    )
                )
                hidden = hidden + self.noise_embedding(scales / self.noise_scale.maximum)
        hidden = self.backbone(
            hidden,
            torch.cat(inputs.positions, dim=1),
            inputs.attention,
            routes=(RouteSpan("flow", 0, sum(inputs.sequence_lengths)),),
        )
        if pipeline.rank != pipeline.size - 1:
            return {"image": (None,) * inputs.batch_size}
        outputs = []
        for features, latent, size, conditioning in zip(
            hidden.split(inputs.sequence_lengths),
            inputs.latents["image"],
            inputs.sizes,
            inputs.images,
            strict=True,
        ):
            from uniserve.nn.functional import unpatchify

            # Conditioning is the network input; the solver's current sample
            # is the origin of the velocity equation, even when they differ.
            sample = unpatchify(
                latent.tensor, size, patch_size=self.patch_size, channels=3
            ).unsqueeze(0)
            velocity = self.prediction(
                features,
                latent.timestep,
                ImageConditioning(sample, conditioning.grid, conditioning.noise_scale),
            )
            shape = self.latent_shape("image", size)
            outputs.append(
                TensorOutput(
                    velocity, OutputLayout(shape, velocity.dtype, tuple(slice(0, n) for n in shape))
                )
            )
        return {"image": tuple(outputs)}


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        backbone = Transformer(config.text)
        self.text = CausalLM(
            backbone, VocabParallelHead(config.text.hidden_size, config.text.vocab_size)
        )
        if config.text.tie_word_embeddings:
            self.text.lm_head.weight = backbone.embedding.weight
        self.denoiser = Denoiser(config, backbone)
        self.vision_encoder = PatchEncoder(
            vision.Encoder(config.vision),
            nn.Identity(),
            patch_size=config.vision.patch_size,
            downsample=round(1 / config.vision.downsample_ratio),
            output_size=config.text.hidden_size,
            output_dtype=torch.bfloat16,
        )
        self.image_decoder = ImageDecoder(RGBDecoder(self.denoiser.patch_size))


checkpoint_sources = (checkpoint.Config("primary"),)


def _backbone_names(backbone: Transformer):
    names = {
        "embedding.weight": "embed_tokens.weight",
        "norm.text.weight": "norm.weight",
        "norm.flow.weight": "norm_mot_gen.weight",
    }
    for path, _ in backbone.named_parameters():
        if not path.startswith("layers."):
            continue
        _, index, kind, route, *parts = path.split(".")
        suffix = "" if route == "text" else "_mot_gen"
        tail = ".".join(parts)
        if kind in ("input_norms", "post_attention_norms"):
            target = (
                ("input_layernorm" if kind == "input_norms" else "post_attention_layernorm")
                + suffix
                + "."
                + tail
            )
        elif kind == "outputs":
            target = f"self_attn.o_proj{suffix}.{tail}"
        elif kind == "projections":
            if parts[0] == "projection":
                target = f"self_attn.{parts[2]}_proj{suffix}.{parts[3]}"
            else:
                spatial = "_hw" if parts[1] == "1" else ""
                target = f"self_attn.{'q' if parts[0] == 'query_norm' else 'k'}_norm{spatial}{suffix}.{parts[2]}"
        elif kind == "mlps":
            tail = (
                tail.replace("gate_up.projections.gate", "gate_proj")
                .replace("gate_up.projections.up", "up_proj")
                .replace("down.", "down_proj.")
            )
            target = f"mlp{suffix}.{tail}"
        else:
            raise ValueError(f"unmapped SenseNova backbone parameter {path}")
        names[path] = f"layers.{index}.{target}"
    return {target: "language_model.model." + source for target, source in names.items()}


def _mapped(module, names, *, nonresident=frozenset()):
    def map_weights(reader):
        available = frozenset(reader.names())
        return tuple(
            weights.Assignment(parameter, reader.get(names[name]))
            for name, parameter in module.named_parameters()
            if name in names and names[name] in available
        )

    return weights.ModuleMapping(
        module,
        "primary",
        map_weights,
        frozenset(name for name, _ in module.named_parameters() if name in names),
        nonresident=nonresident,
    )


def checkpoint_mappings(model: Model):
    backbone = model.text.backbone
    names = _backbone_names(backbone)
    template = tuple(
        source.split(".", 4)[-1]
        for target, source in names.items()
        if target.startswith(f"layers.{next(iter(backbone.layers))}.")
    )
    nonresident = {
        f"language_model.model.layers.{index}.{tail}"
        for index in range(model.config.text.num_hidden_layers)
        if str(index) not in backbone.layers
        for tail in template
    }
    if backbone.embedding is None and not model.config.text.tie_word_embeddings:
        nonresident.add("language_model.model.embed_tokens.weight")
    if backbone.norm is None:
        nonresident.update(
            ("language_model.model.norm.weight", "language_model.model.norm_mot_gen.weight")
        )
    head_name = (
        "language_model.model.embed_tokens.weight"
        if model.config.text.tie_word_embeddings
        else "language_model.lm_head.weight"
    )
    if model.text.lm_head is None or model.config.text.tie_word_embeddings:
        nonresident.add("language_model.lm_head.weight")
    components = [_mapped(backbone, names, nonresident=frozenset(nonresident))]
    if model.text.lm_head is not None:
        components.append(_mapped(model.text, {"lm_head.weight": head_name}))
    denoiser_names = {}
    for path, prefix in (
        ("input", "fm_modules.vision_model_mot_gen.embeddings."),
        ("time_embedding.projection", "fm_modules.timestep_embedder.mlp."),
    ):
        denoiser_names.update(
            {
                f"{path}.{name}": prefix + name
                for name, _ in model.denoiser.get_submodule(path).named_parameters()
            }
        )
    if model.denoiser.noise_embedding is not None:
        denoiser_names.update(
            {
                "noise_embedding.projection." + name: "fm_modules.noise_scale_embedder.mlp." + name
                for name, _ in model.denoiser.noise_embedding.projection.named_parameters()
            }
        )
    for name, _ in model.denoiser.prediction.named_parameters():
        if model.config.flow.use_pixel_head:
            source = name.replace("decoder.blocks.1.", "conv1.").replace(
                "decoder.output.", "conv2."
            )
        elif model.config.flow.head.num_layers <= 2:
            source = name.removeprefix("head.")
        else:
            source = name.removeprefix("head.")
            source = source.replace("time_embedding.projection.", "time_embed.mlp.")
            source = source.replace("input.", "input_proj.")
            source = source.replace("blocks.", "res_blocks.")
            source = (
                source.replace(".norm.", ".in_ln.") if source.startswith("res_blocks.") else source
            )
            source = source.replace(".modulation.", ".adaLN_modulation.")
            source = source.replace("output.projection.", "final_layer.linear.").replace(
                "output.", "final_layer."
            )
            source = "net." + source
        denoiser_names["prediction." + name] = "fm_modules.fm_head." + source
    components.append(_mapped(model.denoiser, denoiser_names))
    components.append(
        _mapped(
            model.vision_encoder,
            {
                "network." + name: "vision_model.embeddings." + name
                for name, _ in model.vision_encoder.network.named_parameters()
            },
        )
    )
    return tuple(components)


def entry_points(config: Config):
    return MappingProxyType(
        {
            "": (
                EntryPoint("text.forward", groups=("tp", "sp", "pp")),
                EntryPoint("text.embed_input_ids", "first", ("tp",)),
                EntryPoint("text.compute_logits", "last", ("tp",)),
                EntryPoint("denoiser.forward", groups=("tp", "sp", "pp")),
                EntryPoint("vision_encoder.encode"),
                EntryPoint("image_decoder.decode"),
            ),
        }
    )


entry_paths = MappingProxyType({"model": "text.forward"})

precisions = MappingProxyType({"bf16": weights.Config()})
