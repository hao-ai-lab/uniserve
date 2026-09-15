"""BAGEL language, image and latent computation with a shared MoT backbone."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch
from torch import nn

from uniserve import loading
from uniserve.diffusion import EulerSolver, NestedGuidance, NoiseScale, Renorm, make_schedule
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
from uniserve.nn.attention import Attention, AttentionInput, DenseInput, RotaryQKVProjection
from uniserve.nn.linear import (
    Linear,
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelHead,
)
from uniserve.nn.mlp import GatedMLP
from uniserve.nn.norm import RMSNorm
from uniserve.nn.rope import RotaryEmbedding
from uniserve.nn.routing import RoutedTensor, RouteSpan
from uniserve.nn.timestep import TimestepEmbedding
from uniserve.nn.vae.patch import PatchAutoencoder
from uniserve.nn.vision import MLPConnector, PositionEmbedding
from uniserve.nn.vision.patching import build_abs_positions_from_grid_hw
from uniserve.tensors import OutputLayout, TensorOutput
from uniserve_models import siglip

from . import vae


@dataclass(frozen=True)
class TransformerConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    head_dim: int
    qk_norm: bool
    max_position_embeddings: int

    def __post_init__(self):
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "head_dim",
            "max_position_embeddings",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"BAGEL {name} must be a positive integer")
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError("BAGEL requires compatible GQA heads and even rotary dimensions")
        if type(self.qk_norm) is not bool:
            raise ValueError("BAGEL QK normalization must be boolean")
        if any(
            not math.isfinite(value) or value <= 0 for value in (self.rms_norm_eps, self.rope_theta)
        ):
            raise ValueError(
                "BAGEL normalization epsilon and rotary theta must be finite and positive"
            )


@dataclass(frozen=True)
class Config:
    text: TransformerConfig
    vision: siglip.Config
    vae: vae.Config
    start_of_image_id: int
    end_of_image_id: int
    latent_patch_size: int
    max_latent_size: int
    timestep_shift: float
    connector_act: str

    def __post_init__(self):
        for value in (self.start_of_image_id, self.end_of_image_id):
            if type(value) is not int or not 0 <= value < self.text.vocab_size:
                raise ValueError("BAGEL image marker IDs must lie within its vocabulary")
        if self.start_of_image_id == self.end_of_image_id:
            raise ValueError("BAGEL image markers must be distinct")
        if any(
            type(value) is not int or value < 1
            for value in (self.latent_patch_size, self.max_latent_size)
        ):
            raise ValueError("BAGEL latent patches and learned grid size must be positive")
        if not math.isfinite(self.timestep_shift) or self.timestep_shift <= 0:
            raise ValueError("BAGEL timestep shift must be finite and positive")


class TransformerLayer(nn.Module):
    """Route projection/MLP experts around one shared token-attention domain."""

    def __init__(self, config: TransformerConfig, index: int):
        super().__init__()
        hidden, dim = config.hidden_size, config.head_dim
        self.input_norms = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.outputs = nn.ModuleDict()
        self.post_attention_norms = nn.ModuleDict()
        self.mlps = nn.ModuleDict()
        for route in ("text", "flow"):
            self.input_norms[route] = RMSNorm(hidden, config.rms_norm_eps)
            self.projections[route] = RotaryQKVProjection(
                QKVParallelLinear(
                    hidden, config.num_attention_heads, config.num_key_value_heads, dim, bias=True
                ),
                RMSNorm(dim, config.rms_norm_eps) if config.qk_norm else nn.Identity(),
                RMSNorm(dim, config.rms_norm_eps) if config.qk_norm else nn.Identity(),
            )
            self.outputs[route] = RowParallelLinear(
                config.num_attention_heads * dim, hidden, bias=False
            )
            self.post_attention_norms[route] = RMSNorm(hidden, config.rms_norm_eps)
            self.mlps[route] = GatedMLP(hidden, config.intermediate_size)
        self.attention = Attention(
            config.num_attention_heads,
            config.num_key_value_heads,
            dim,
            cache_name=f"text.backbone.layers.{index}.attention",
        )
        self.rotary = RotaryEmbedding(dim, theta=config.rope_theta)

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
        temporal = positions if positions.ndim == 1 else positions[0]
        cosine, sine = self.rotary(temporal, dtype=torch.float32, sequence_length=temporal.numel())
        route_names = frozenset(hidden.values)
        cos = RoutedTensor.from_packed(cosine, routes, routes=route_names)
        sin = RoutedTensor.from_packed(sine, routes, routes=route_names)
        projected = {
            route: self.projections[route](value, (cos.values[route],), (sin.values[route],))
            for route, value in normalized.values.items()
        }
        query, key, value = (
            RoutedTensor({route: values[index] for route, values in projected.items()}).packed(
                routes
            )
            for index in range(3)
        )
        attended = self.attention(query, key, value, attention).flatten(1)
        update = RoutedTensor.from_packed(attended, routes, routes=route_names).apply(self.outputs)
        residual = hidden.add(update)
        normalized = residual.apply(self.post_attention_norms)
        # The checkpoint's expert FFNs consume BF16 normalization results,
        # including when their surrounding accumulation is higher precision.
        normalized = RoutedTensor(
            {route: value.to(torch.bfloat16) for route, value in normalized.values.items()}
        )
        return normalized.apply(self.mlps), residual


class Transformer(TransformerDecoder):
    def __init__(self, config: TransformerConfig):
        super().__init__(
            VocabParallelEmbedding(config.vocab_size, config.hidden_size),
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


class _Vision(nn.Module):
    """BAGEL's SigLIP features, connector and learned language-width grid."""

    def __init__(self, config: Config):
        super().__init__()
        self.encoder = siglip.Encoder(config.vision)
        self.connector = MLPConnector(
            config.vision.encoder.hidden_size, config.text.hidden_size, config.connector_act
        )
        side = config.vision.image_size // config.vision.patch_size
        self.position = PositionEmbedding((side, side), config.text.hidden_size)

    def forward(self, pixels, grids, grid_shapes):
        features = self.connector(self.encoder(pixels, grids, grid_shapes))
        columns, rows = build_abs_positions_from_grid_hw(
            grids, total=sum(height * width for height, width in grid_shapes)
        )
        return features + self.position(rows * self.position.grid_size[1] + columns)


@dataclass(frozen=True)
class DenoiserInput(NumericalDenoiserInput[image.Config]):
    """Framed image sequences with temporal/row/column positions [3, tokens].

    Each positions tensor includes both marker rows. Its spatial coordinates
    on interior rows index the learned latent grid; marker spatial coordinates
    are unused. Temporal coordinates apply to every row through shared RoPE.
    """

    positions: tuple[torch.Tensor, ...]
    sequence_lengths: tuple[int, ...]
    attention: AttentionInput

    def __post_init__(self):
        super().__post_init__()
        if len(self.positions) != self.batch_size or len(self.sequence_lengths) != self.batch_size:
            raise ValueError("BAGEL coordinates must align with image samples")
        if any(
            position.shape != (3, count) or count < 3
            for position, count in zip(self.positions, self.sequence_lengths, strict=True)
        ):
            raise ValueError("BAGEL positions require three axes and two framing markers")
        if (
            not isinstance(self.attention, DenseInput)
            and self.attention.queries.host is not None
            and self.attention.queries.host != self.sequence_lengths
        ):
            raise ValueError("BAGEL attention lengths must match its framed image sequences")


class Denoiser(ImageDenoiser[DenoiserInput]):
    def __init__(self, config: Config, backbone: Transformer):
        super().__init__(
            patch_size=config.latent_patch_size,
            latent_channels=config.vae.latent_channels,
            downsample=config.vae.downsample * config.latent_patch_size,
            noise_scale=NoiseScale(1.0, "constant", 1.0, 1.0),
            prediction_dtype=torch.bfloat16,
            solver=EulerSolver("velocity"),
        )
        self.config, self.backbone = config, backbone
        width = config.latent_patch_size**2 * config.vae.latent_channels
        self.input = Linear(width, config.text.hidden_size)
        self.time_embedding = TimestepEmbedding(config.text.hidden_size)
        self.position = PositionEmbedding((config.max_latent_size,) * 2, config.text.hidden_size)
        self.prediction = Linear(config.text.hidden_size, width)
        # Derived marker IDs are numerical constants. Explicit CPU construction
        # survives meta initialization; loading places their borrowed buffer.
        self.register_buffer(
            "markers",
            torch.tensor(
                (config.start_of_image_id, config.end_of_image_id), dtype=torch.long, device="cpu"
            ),
            persistent=False,
        )

    def noise_shape(self, modality: str, size: image.Config):
        # BAGEL draws directly in canonical patch/channel order.
        return self.latent_shape(modality, size)

    def make_schedules(self, steps, *, shift, device):
        return {
            "image": make_schedule(
                steps,
                shift=self.config.timestep_shift if shift is None else shift,
                direction="descending",
                shift_domain="time",
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
        return NestedGuidance(text_scale, image_scale, interval, renorm, renorm_min)

    def forward(self, inputs: DenoiserInput, *, state, constants, workspace):
        if set(inputs.latents) != {"image"}:
            raise ValueError("BAGEL predicts the image latent modality")
        group = self.mesh.get_group("pp" if "pp" in self.mesh.axes else ())
        chunks, routes = [], []
        cursor = 0
        for latent, size, positions, count in zip(
            inputs.latents["image"],
            inputs.sizes,
            inputs.positions,
            inputs.sequence_lengths,
            strict=True,
        ):
            shape = self.latent_shape("image", size)
            if latent.tensor.shape != shape or count != shape[0] + 2:
                raise ValueError("BAGEL latents must cover their framed image sequence")
            if group.rank == 0:
                marker = self.backbone.embed_input_ids(self.markers).to(torch.bfloat16)
                coordinates = positions[1, 1:-1] * self.config.max_latent_size + positions[2, 1:-1]
                features = self.input(latent.tensor.to(torch.bfloat16))
                features = (
                    features
                    + self.time_embedding(latent.timestep.reshape(1).expand(shape[0]))
                    + self.position(coordinates)
                ).to(torch.bfloat16)
                chunks.append(torch.cat((marker[:1], features, marker[1:])))
            routes.extend(
                (
                    RouteSpan("text", cursor, 1),
                    RouteSpan("flow", cursor + 1, shape[0]),
                    RouteSpan("text", cursor + count - 1, 1),
                )
            )
            cursor += count
        if not inputs.batch_size:
            return {"image": ()}
        hidden = self.backbone(
            torch.cat(chunks) if group.rank == 0 else None,
            torch.cat(inputs.positions, dim=1),
            inputs.attention,
            routes=tuple(routes),
        )
        if group.rank != group.size - 1:
            return {"image": (None,) * inputs.batch_size}
        outputs = []
        for hidden_row, size in zip(
            hidden.split(inputs.sequence_lengths), inputs.sizes, strict=True
        ):
            prediction = self.prediction(hidden_row[1:-1].to(torch.bfloat16))
            shape = self.latent_shape("image", size)
            outputs.append(
                TensorOutput(
                    prediction,
                    OutputLayout(shape, prediction.dtype, tuple(slice(0, n) for n in shape)),
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
        self.denoiser = Denoiser(config, backbone)
        self.vision_encoder = PatchEncoder(
            _Vision(config),
            nn.Identity(),
            patch_size=config.vision.patch_size,
            downsample=1,
            output_size=config.text.hidden_size,
            output_dtype=torch.bfloat16,
        )
        autoencoder = vae.Model(config.vae)
        self.latent_encoder = PatchAutoencoder(
            autoencoder.encoder,
            autoencoder.decoder,
            autoencoder.posterior,
            patch_size=config.latent_patch_size,
            latent_channels=config.vae.latent_channels,
            latent_dtype=torch.bfloat16,
            downsample=config.vae.downsample * config.latent_patch_size,
            scale=config.vae.scale_factor,
            shift=config.vae.shift_factor,
        )
        self.image_decoder = ImageDecoder(self.latent_encoder)


checkpoint_sources = (
    checkpoint.Config("primary", filenames=("ema.safetensors", "model.safetensors")),
    checkpoint.Config("autoencoder", filenames=("ae.safetensors",)),
)


def read_config(root: Path, io: loading.Config) -> Config:
    raw = json.loads((root / "config.json").read_text())
    towers = []
    for name in ("llm", "vit", "vae"):
        towers.append(
            raw[f"{name}_config"]
            if f"{name}_config" in raw
            else json.loads((root / f"{name}_config.json").read_text())
        )
    text, vision, latent = towers
    heads, hidden = text["num_attention_heads"], text["hidden_size"]
    if type(heads) is not int or heads < 1 or ("head_dim" not in text and hidden % heads):
        raise ValueError("BAGEL checkpoint requires compatible text width and heads")
    with checkpoint_sources[0].resolve(root, io=io).open(io=io) as reader:
        shape = reader.get("latent_pos_embed.pos_embed").shape
    side = math.isqrt(shape[0])
    if len(shape) != 2 or shape[1] != hidden or side * side != shape[0]:
        raise ValueError("BAGEL latent position table must be a square grid at text width")
    return Config(
        TransformerConfig(
            hidden,
            text["intermediate_size"],
            text["num_hidden_layers"],
            heads,
            text["num_key_value_heads"],
            text["vocab_size"],
            text.get("rms_norm_eps", 1e-6),
            text.get("rope_theta", 1_000_000.0),
            text.get("head_dim", hidden // heads),
            text.get("qk_norm", True),
            text.get("max_position_embeddings", 32768),
        ),
        siglip.Config(
            vision.get("patch_size", 14),
            vision.get("image_size", 980),
            vision.get("num_channels", 3),
            siglip.TransformerConfig(
                vision.get("hidden_size", 1152),
                vision.get("num_attention_heads", 16),
                vision.get("intermediate_size", 4304),
                vision.get("num_hidden_layers", 27) - 1,
                vision.get("layer_norm_eps", 1e-6),
            ),
        ),
        vae.Config(
            latent.get("resolution", 256),
            latent.get("in_channels", 3),
            latent.get("downsample", 8),
            latent.get("ch", 128),
            latent.get("out_ch", 3),
            tuple(latent.get("ch_mult", (1, 2, 4, 4))),
            latent.get("num_res_blocks", 2),
            latent.get("z_channels", 16),
            latent.get("scale_factor", 0.3611),
            latent.get("shift_factor", 0.1159),
        ),
        raw.get("start_of_image_id", 151652),
        raw.get("end_of_image_id", 151653),
        raw.get("latent_patch_size", 2),
        side,
        raw.get("timestep_shift", 1.0),
        raw.get("connector_act", "gelu_pytorch_tanh"),
    )


def entry_points(config: Config):
    return MappingProxyType(
        {
            "": (
                EntryPoint("text.forward", groups=("tp", "sp", "pp")),
                EntryPoint("text.embed_input_ids", "first", ("tp",)),
                EntryPoint("text.compute_logits", "last", ("tp",)),
                EntryPoint("denoiser.forward", groups=("tp", "sp", "pp")),
                EntryPoint("vision_encoder.encode"),
                EntryPoint("latent_encoder.encode"),
                EntryPoint("image_decoder.decode"),
            ),
        }
    )


entry_paths = MappingProxyType({"model": "text.forward"})

precisions = MappingProxyType(
    {
        "bf16": weights.Config(
            dtypes={"latent_encoder": torch.float32, "image_decoder.decoder": torch.float32}
        )
    }
)


def _backbone_names(backbone: Transformer) -> dict[str, str]:
    names = {
        "embedding.weight": "embed_tokens.weight",
        "norm.text.weight": "norm.weight",
        "norm.flow.weight": "norm_moe_gen.weight",
    }
    for path, _ in backbone.named_parameters():
        if not path.startswith("layers."):
            continue
        _, index, kind, route, *parts = path.split(".")
        suffix = "" if route == "text" else "_moe_gen"
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
                target = (
                    f"self_attn.{'q' if parts[0] == 'query_norm' else 'k'}_norm{suffix}.{parts[1]}"
                )
        elif kind == "mlps":
            tail = (
                tail.replace("gate_up.projections.gate", "gate_proj")
                .replace("gate_up.projections.up", "up_proj")
                .replace("down.", "down_proj.")
            )
            target = f"mlp{suffix}.{tail}"
        else:
            raise ValueError(f"unmapped BAGEL backbone parameter {path}")
        names[path] = f"layers.{index}.{target}"
    return {target: "language_model.model." + source for target, source in names.items()}


def _mapped(module, source_names, *, nonresident=frozenset()):
    def map_weights(reader):
        available = frozenset(reader.names())
        return tuple(
            weights.Assignment(parameter, reader.get(source_names[name]))
            for name, parameter in module.named_parameters()
            if name in source_names and source_names[name] in available
        )

    return weights.ModuleMapping(
        module,
        "primary",
        map_weights,
        frozenset(name for name, _ in module.named_parameters() if name in source_names),
        nonresident=nonresident,
    )


def checkpoint_mappings(model: Model) -> tuple[weights.ModuleMapping, ...]:
    backbone = model.text.backbone
    source_names = _backbone_names(backbone)
    # PP omits only the source layers and terminal modules assigned elsewhere.
    template = tuple(
        source.split(".", 4)[-1]
        for target, source in source_names.items()
        if target.startswith(f"layers.{next(iter(backbone.layers))}.")
    )
    nonresident = {
        f"language_model.model.layers.{index}.{tail}"
        for index in range(model.config.text.num_hidden_layers)
        if str(index) not in backbone.layers
        for tail in template
    }
    if backbone.embedding is None:
        nonresident.add("language_model.model.embed_tokens.weight")
    if backbone.norm is None:
        nonresident.update(
            ("language_model.model.norm.weight", "language_model.model.norm_moe_gen.weight")
        )
    components = [_mapped(backbone, source_names, nonresident=frozenset(nonresident))]
    if model.text.lm_head is not None:
        components.append(_mapped(model.text, {"lm_head.weight": "language_model.lm_head.weight"}))
    else:
        first = components[0]
        components[0] = weights.ModuleMapping(
            first.module,
            first.source,
            first.map_weights,
            first.required,
            nonresident=first.nonresident | {"language_model.lm_head.weight"},
        )
    denoiser_names = {}
    for path, prefix in (
        ("input", "vae2llm."),
        ("prediction", "llm2vae."),
        ("time_embedding.projection", "time_embedder.mlp."),
    ):
        denoiser_names.update(
            {
                f"{path}.{name}": prefix + name
                for name, _ in model.denoiser.get_submodule(path).named_parameters()
            }
        )
    denoiser_names["position.weight"] = "latent_pos_embed.pos_embed"
    components.append(_mapped(model.denoiser, denoiser_names))
    vision = model.vision_encoder.network
    vision_names = {
        "network.connector." + name: "connector."
        + name.replace("projection.0.", "fc1.").replace("projection.2.", "fc2.")
        for name, _ in vision.connector.named_parameters()
    }
    vision_names["network.position.weight"] = "vit_pos_embed.pos_embed"
    projected = _mapped(model.vision_encoder, vision_names)
    components.append(
        weights.ModuleMapping(
            model.vision_encoder,
            "primary",
            lambda reader: (
                projected.map_weights(reader)
                + siglip.assignments(vision.encoder, reader, prefix="vit_model.vision_model.")
            ),
            frozenset(dict(model.vision_encoder.named_parameters())),
        )
    )
    components.append(
        weights.ModuleMapping(
            model.latent_encoder,
            "autoencoder",
            lambda reader: vae.assignments(model.latent_encoder, reader),
            frozenset(dict(model.latent_encoder.named_parameters())),
        )
    )
    return tuple(components)
