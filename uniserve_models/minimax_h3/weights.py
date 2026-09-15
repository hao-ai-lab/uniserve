"""H3 checkpoint assignments and fixed-ladder modulation preparation."""

from __future__ import annotations

from dataclasses import asdict
from functools import cache

import torch
from torch import nn
from torch.nn import functional as F

from uniserve.loading import weights
from uniserve.nn import Modulation
from uniserve_models import qwen3

from . import audio_vae, video_vae
from .conditioning import assignments as conditioning_assignments
from .config import TEXT_FIELDS, TRANSFORMER_FIELDS, TransformerConfig
from .diffusion import schedules
from .modulation import TimestepEmbedding


@cache
def _transformer_names(config: TransformerConfig) -> frozenset[str]:
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

    with torch.device("meta"):
        native = MiniMaxH3Transformer3DModel(
            **{source: getattr(config, target) for source, target in TRANSFORMER_FIELDS.items()},
            patch_size=(1, 2, 2),
            final_norm_eps=config.norm_eps,
        )
    return frozenset(native.state_dict()) | {
        f"transformer_blocks.{index}.attn.to_gate_compress.weight"
        for index in range(config.num_hidden_layers)
    }


@cache
def _text_names(config) -> frozenset[str]:
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    native_config = Qwen3VLConfig(
        text_config={source: getattr(config, target) for source, target in TEXT_FIELDS.items()},
        vision_config={"depth": 27, "deepstack_visual_indexes": [8, 16, 24]},
    )
    with torch.device("meta"):
        native = Qwen3VLForConditionalGeneration(native_config)
    return frozenset(native.state_dict())


@cache
def _video_names(config) -> frozenset[str]:
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

    with torch.device("meta"):
        native = AutoencoderKLMiniMaxH3(**asdict(config))
    return frozenset(native.state_dict())


@cache
def _audio_names(config) -> frozenset[str]:
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
        AutoencoderKLMiniMaxH3Audio,
    )

    with torch.device("meta"):
        native = AutoencoderKLMiniMaxH3Audio(**asdict(config))
    return frozenset(native.state_dict())


def _resident_layers(model):
    """Keep checkpoint layers and endpoint heads at their mathematical PP stage."""

    pipeline = model.mesh.get_group("pp" if "pp" in model.mesh.axes else ())
    if pipeline.size > model.config.num_hidden_layers:
        raise ValueError("every H3 pipeline stage requires at least one transformer layer")
    start = model.config.num_hidden_layers * pipeline.rank // pipeline.size
    stop = model.config.num_hidden_layers * (pipeline.rank + 1) // pipeline.size
    model.layers = nn.ModuleDict(
        {str(index): model.layers[str(index)] for index in range(start, stop)}
    )
    if pipeline.rank:
        model.video_input = model.audio_input = None
    if pipeline.rank + 1 != pipeline.size:
        model.output_norm = model.video_output = model.audio_output = None
    return pipeline


def transformer_assignments(model, reader):
    """Map independent attention branches and value-first SwiGLU source rows."""

    available = frozenset(reader.names())
    for name, parameter in model.named_parameters():
        branch = None
        if name.startswith("layers."):
            _, index, kind, *parts = name.split(".")
            prefix = f"transformer_blocks.{index}."
            if kind == "norm":
                source = prefix + f"norm{int(parts[0]) + 1}.weight"
            elif kind == "attention":
                if parts[0] == "projection":
                    field = "gate_compress" if parts[2] == "gate" else parts[2]
                    source = prefix + f"attn.to_{field}.weight"
                elif parts[0] == "output":
                    source = prefix + "attn.to_out.0.weight"
                else:
                    field = {"query_norm": "q", "key_norm": "k"}[parts[0]]
                    source = prefix + f"attn.norm_{field}.weight"
            elif kind == "mlp":
                if parts[0] == "gate_up":
                    branch = 0 if parts[2] == "up" else 1
                    source = prefix + "ff.net.0.proj.weight"
                else:
                    source = prefix + "ff.net.2.weight"
            else:
                raise ValueError(f"unmapped H3 transformer parameter {name!r}")
        else:
            kind, *parts = name.split(".")
            source = (
                {
                    "video_input": "proj_in",
                    "audio_input": "audio_proj_in",
                    "video_output": "proj_out",
                    "audio_output": "audio_proj_out",
                    "output_norm": "norm_out",
                }[kind]
                + "."
                + ".".join(parts)
            )
        if source not in available:
            continue
        value = reader.get(source)
        region = None
        if branch is not None:
            width = value.shape[0] // 2
            region = (slice(branch * width, (branch + 1) * width), slice(0, value.shape[1]))
        yield weights.Assignment(parameter, value, source_slice=region)


@torch.inference_mode()
def _prepare_modulation(model, diffusion, reader):
    # These temporary learned projections belong to loading. Only their fixed
    # step products remain with the numerical transformer after this function.
    device = next(model.parameters()).device
    with torch.device("meta"):
        embedding = TimestepEmbedding(model.config)
    for index, source in ((0, "linear_1"), (2, "linear_2")):
        for field in ("weight", "bias"):
            value = (
                reader.get(f"time_embedder.{source}.{field}")
                .read()
                .to(device=device, dtype=torch.float32)
            )
            setattr(
                embedding.video_projection[index], field, nn.Parameter(value, requires_grad=False)
            )
    ladder = schedules(diffusion, device=device)
    activated = torch.stack(
        [
            F.silu(embedding(torch.stack((video, audio))))
            for video, audio in zip(
                ladder["video"].timesteps[:-1], ladder["audio"].timesteps[:-1], strict=True
            )
        ]
    )
    del embedding

    def projection(prefix):
        return tuple(
            reader.get(f"{prefix}.{field}").read().to(device=device, dtype=torch.bfloat16)
            for field in ("weight", "bias")
        )

    prepared = Modulation.from_projections(
        activated,
        (projection(f"transformer_blocks.{index}.adaln_proj.linear") for index in model.layers),
        projection("norm_out.linear") if model.output_norm is not None else None,
        layer_count=len(model.layers),
    )
    # Keep the existing module identity so the loader can retain its buffers
    # along the same selected-component and device bindings as the model.
    model.modulation.products = prepared.products
    model.modulation.output_products = prepared.output_products


def transformer_component(model, diffusion):
    """Declare resident transformer matrices and streamed modulation sources."""

    _resident_layers(model)
    all_names = _transformer_names(model.config)
    nonresident = set()
    for name in all_names:
        parts = name.split(".")
        if name.startswith(
            ("context_embedder.", "token_refiner.", "time_embedder.", "norm_out.linear.")
        ):
            nonresident.add(name)
        elif parts[0] == "transformer_blocks" and (
            parts[1] not in model.layers or parts[2] == "adaln_proj"
        ):
            nonresident.add(name)
        elif model.video_input is None and parts[0] in {"proj_in", "audio_proj_in"}:
            nonresident.add(name)
        elif model.output_norm is None and parts[0] in {"norm_out", "proj_out", "audio_proj_out"}:
            nonresident.add(name)
    return weights.ModuleMapping(
        model,
        "denoiser",
        lambda reader: tuple(transformer_assignments(model, reader)),
        frozenset(name for name, _ in model.named_parameters()),
        nonresident=frozenset(nonresident),
        post_load=lambda reader: _prepare_modulation(model, diffusion, reader),
    )


def _text_component(model):
    names = {
        "network." + target.removeprefix("backbone."): source.replace(
            "model.", "model.language_model.", 1
        )
        for target, source in qwen3._parameter_sources(model.network.config).items()
        if target.startswith("backbone.")
    }
    parameters = dict(model.named_parameters())
    used = {names[name] for name in parameters}

    def assign(reader):
        available = frozenset(reader.names())
        return tuple(
            weights.Assignment(parameter, reader.get(names[name]))
            for name, parameter in parameters.items()
            if names[name] in available
        )

    return weights.ModuleMapping(
        model,
        "text_encoder",
        assign,
        frozenset(parameters),
        nonresident=_text_names(model.config) - used,
    )


def checkpoint_mappings(model) -> tuple[weights.ModuleMapping, ...]:
    """Account for every native source using the complete model architecture."""

    denoiser = model.denoiser
    transformer = transformer_component(denoiser.transformer, denoiser.diffusion)
    pipeline = denoiser.transformer.mesh.get_group(
        "pp" if "pp" in denoiser.transformer.mesh.axes else ()
    )
    # Conditioner participates only in the input stage of the denoiser. It is
    # otherwise an ordinary encoder with its own tensor-parallel layer binding.
    components = [transformer]
    if pipeline.rank == 0:
        conditioner = denoiser.conditioner
        components.append(
            weights.ModuleMapping(
                conditioner,
                "denoiser",
                lambda reader: tuple(conditioning_assignments(conditioner, reader)),
                frozenset(name for name, _ in conditioner.named_parameters()),
                nonresident=frozenset(
                    name
                    for name in _transformer_names(denoiser.config)
                    if not name.startswith(("context_embedder.", "token_refiner."))
                ),
            )
        )
    else:
        denoiser.conditioner = None
    components.append(_text_component(model.text_encoder))
    video, audio = model.video_decoder.decoder, model.audio_decoder.decoder
    components.extend(
        (
            weights.ModuleMapping(
                video,
                "video_decoder",
                lambda reader: video_vae.assignments(video, reader),
                frozenset(name for name, _ in video.named_parameters()),
                nonresident=frozenset(
                    name
                    for name in _video_names(model.config.video_decoder)
                    if name.startswith(("encoder.", "quant_conv."))
                ),
            ),
            weights.ModuleMapping(
                audio,
                "audio_decoder",
                lambda reader: audio_vae.assignments(audio, reader),
                frozenset(name for name, _ in audio.named_parameters()),
                nonresident=frozenset(
                    name
                    for name in _audio_names(audio.config)
                    if not name.startswith(("decoder.", "dec_in_proj."))
                ),
            ),
        )
    )
    return tuple(components)
