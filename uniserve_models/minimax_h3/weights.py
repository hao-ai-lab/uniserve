"""H3 checkpoint assignments and fixed-ladder modulation preparation.

An H3 checkpoint is a diffusers pipeline directory; ``checkpoint_sources``
names its four weight subdirectories. ``checkpoint_mappings`` returns a
``weights.ModuleMapping`` for each component module. The loader fails on a
checkpoint tensor that no assignment, derived constant, or ``nonresident``
entry accounts for, so each mapping enumerates the tensors it intentionally
leaves unloaded from a meta-device instance of the native diffusers or
transformers model (``_transformer_names`` and its siblings).
"""

from __future__ import annotations

from dataclasses import asdict
from functools import cache

import torch
from torch import nn
from torch.nn import functional as F

from uniserve.loading import checkpoint, weights
from uniserve.nn import Modulation
from uniserve_models import qwen3_vl

from . import audio_vae, video_vae
from .conditioning import assignments as conditioning_assignments
from .config import TEXT_FIELDS, TRANSFORMER_FIELDS, TransformerConfig
from .denoiser import schedules
from .modulation import TimestepEmbedding

checkpoint_sources = (
    checkpoint.Config("denoiser", "transformer", module_path="denoiser"),
    checkpoint.Config(
        "text_encoder", "text_encoder", module_path="text_encoder"
    ),
    checkpoint.Config("video_decoder", "vae", module_path="video_decoder"),
    checkpoint.Config(
        "audio_decoder", "audio_vae", module_path="audio_decoder"
    ),
)


@cache
def _transformer_names(config: TransformerConfig) -> frozenset[str]:
    # The native diffusers model has no ``to_gate_compress`` projection, but
    # the checkpoint stores one per block for the attention projection's
    # ``gate`` branch. Listing it lets the nonresident sets of other pipeline
    # stages and of the conditioner mapping account for it.
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )

    with torch.device("meta"):
        native = MiniMaxH3Transformer3DModel(
            **{
                source: getattr(config, target)
                for source, target in TRANSFORMER_FIELDS.items()
            },
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

    # Only tensor names matter here. The vision depth and deepstack indexes
    # determine the vision tower's tensor names and match the H3 checkpoint's
    # text_encoder config, so every vision tensor is declared nonresident.
    native_config = Qwen3VLConfig(
        text_config={
            source: getattr(config, target)
            for source, target in TEXT_FIELDS.items()
        },
        vision_config={"depth": 27, "deepstack_visual_indexes": [8, 16, 24]},
    )
    with torch.device("meta"):
        native = Qwen3VLForConditionalGeneration(native_config)
    return frozenset(native.state_dict())


@cache
def _video_names(config) -> frozenset[str]:
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
        AutoencoderKLMiniMaxH3,
    )

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
    """Keep checkpoint layers and endpoint heads at their mathematical PP stage.

    Mutates ``model``: of ``L`` layers, stage ``r`` of ``P`` keeps the
    contiguous layers ``[L * r // P, L * (r + 1) // P)`` under their global
    keys, only the first stage keeps the latent input heads, and only the
    last keeps the output norm and heads. Returns the pipeline group.

    Raises:
        ValueError: The pipeline has more stages than transformer layers.
    """  # noqa: E501
    pipeline = model.mesh.get_group("pp" if "pp" in model.mesh.axes else ())
    if pipeline.size > model.config.num_hidden_layers:
        raise ValueError(
            "every H3 pipeline stage requires at least one transformer layer"
        )
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
    """Map independent attention branches and value-first SwiGLU source rows.

    Yields one assignment per resident parameter whose checkpoint source
    exists. An absent source leaves its parameter unassigned, which the
    loader reports as missing.

    Raises:
        ValueError: A layer parameter lies outside the norm, attention and
            mlp submodules.
    """
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

        # SwiGLU branches share one fused checkpoint tensor with value rows
        # first: the ``up`` branch reads the leading half, ``gate`` the rest.
        region = None
        if branch is not None:
            width = value.shape[0] // 2
            region = (
                slice(branch * width, (branch + 1) * width),
                slice(0, value.shape[1]),
            )
        yield weights.Assignment(parameter, value, source_slice=region)


@torch.inference_mode()
def _prepare_modulation(model, diffusion, reader):
    """Precompute the resident layers' modulation products for the ladder.

    Runs as the transformer mapping's post-load hook, after the resident
    parameters are materialized. Evaluates the checkpoint's FP32 timestep
    embedding and SiLU at every evaluated rung, projects the result through
    each resident layer's ``adaln_proj`` and, on the last stage, ``norm_out``
    in BF16, and stores the products on ``model.modulation``.
    """
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
                embedding.video_projection[index],
                field,
                nn.Parameter(value, requires_grad=False),
            )
    # The clean endpoint closes each schedule but is never evaluated, so the
    # step count equals the ladder length the transformer was built with.
    # ``activated`` is [steps, 2 timesteps (video, audio), time_dim].
    ladder = schedules(diffusion, device=device)
    activated = torch.stack(
        [
            F.silu(embedding(torch.stack((video, audio))))
            for video, audio in zip(
                ladder["video"].timesteps[:-1],
                ladder["audio"].timesteps[:-1],
                strict=True,
            )
        ]
    )
    del embedding

    def projection(prefix):
        return tuple(
            reader.get(f"{prefix}.{field}")
            .read()
            .to(device=device, dtype=torch.bfloat16)
            for field in ("weight", "bias")
        )

    prepared = Modulation.from_projections(
        activated,
        (
            projection(f"transformer_blocks.{index}.adaln_proj.linear")
            for index in model.layers
        ),
        projection("norm_out.linear")
        if model.output_norm is not None
        else None,
        layer_count=len(model.layers),
    )
    # Keep the existing module identity so the loader can retain its buffers
    # along the same selected-component and device bindings as the model.
    model.modulation.products = prepared.products
    model.modulation.output_products = prepared.output_products


def transformer_component(model, diffusion):
    """Declare resident transformer matrices and streamed modulation sources.

    Prunes ``model`` to this pipeline stage first (see ``_resident_layers``).
    Declares nonresident the conditioner's tensors, other stages' layers and
    heads, and the timestep and modulation projections that only the
    post-load hook reads.
    """
    _resident_layers(model)
    all_names = _transformer_names(model.config)
    nonresident = set()
    for name in all_names:
        parts = name.split(".")
        if name.startswith(
            (
                "context_embedder.",
                "token_refiner.",
                "time_embedder.",
                "norm_out.linear.",
            )
        ):
            nonresident.add(name)
        elif parts[0] == "transformer_blocks" and (
            parts[1] not in model.layers or parts[2] == "adaln_proj"
        ):
            nonresident.add(name)
        elif model.video_input is None and parts[0] in {
            "proj_in",
            "audio_proj_in",
        }:
            nonresident.add(name)
        elif model.output_norm is None and parts[0] in {
            "norm_out",
            "proj_out",
            "audio_proj_out",
        }:
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
    # The text encoder checkpoint is a whole Qwen3-VL model: the resident
    # language-model layers and the vision tower load from it, and every
    # other native tensor (the language head, layers past the retained
    # depth) is nonresident.
    return weights.ModuleMapping(
        model,
        "text_encoder",
        lambda reader: qwen3_vl.weights.assignments(model, reader),
        frozenset(name for name, _ in model.named_parameters()),
        nonresident=_text_names(model.config) - qwen3_vl.weights.sources(model),
    )


def checkpoint_mappings(model) -> tuple[weights.ModuleMapping, ...]:
    """Account for every native source using the complete model architecture.

    Builds the mappings for the pipeline stage bound on the denoiser's
    transformer mesh, pruning that transformer to its resident layers and
    setting ``denoiser.conditioner`` to None on every stage after the first.
    """
    denoiser = model.denoiser
    transformer = transformer_component(
        denoiser.transformer, denoiser.diffusion
    )
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
                lambda reader: tuple(
                    conditioning_assignments(conditioner, reader)
                ),
                frozenset(name for name, _ in conditioner.named_parameters()),
                nonresident=frozenset(
                    name
                    for name in _transformer_names(denoiser.config)
                    if not name.startswith(
                        ("context_embedder.", "token_refiner.")
                    )
                ),
            )
        )
    else:
        denoiser.conditioner = None
    components.append(_text_component(model.text_encoder))

    # Only the decoder halves of both VAEs are resident.
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
