"""H3 component geometry, checkpoint name mapping and numerical precomputation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch import nn

from ...loader.component import (
    CheckpointComponent,
    construct_owned_module,
    construction_dtype,
)
from ...loader.handles import WeightHandle
from ...loader.mapping import LoadReport, stacked_weight_name
from ...loader.weight_loaders import attach_parameter_loaders, load_parameter_weight
from ...modeling.context import BuildContext
from ...nn.diffusion.schedule import DiffusionSchedule
from ...nn.layer import LayerConfig
from ...nn.linear import LinearBase
from ...nn.mesh import Communicator
from ...nn.parallel import ParallelConfig
from ...nn.quant.config import QuantizationConfig
from .audio_vae import MiniMaxH3AudioVAE
from .encoder import MiniMaxH3TextEncoder
from .layout import MIN_H3_FRAMES, H3Layout
from .packing import audio_latent_frames
from .transformer import (
    H3TimestepEmbedding,
    MiniMaxH3Transformer,
    build_conditioner,
)
from .video_vae import MiniMaxH3VideoDecoder, MiniMaxH3VideoVAE


@dataclass(slots=True)
class H3Components:
    """Resident numerical modules composed before checkpoint materialization."""

    transformer: MiniMaxH3Transformer | None
    conditioner: nn.Sequential | None
    encoder: MiniMaxH3TextEncoder | None
    video_vae: MiniMaxH3VideoVAE | None
    audio_vae: MiniMaxH3AudioVAE | None


@torch.inference_mode()
def _prepare_modulation(
    model: MiniMaxH3Transformer,
    sources: Mapping[str, WeightHandle],
    device: torch.device,
    schedule: DiffusionSchedule,
) -> None:
    """Stream fixed-timestep checkpoint projections into model-owned products."""

    from ...nn.diffusion.modulation import ModulationPlan

    embedding = H3TimestepEmbedding(model.config, device="meta", buffer_device=device)
    attach_parameter_loaders(embedding, device=device, dtype=torch.float32)
    for name, parameter in tuple(embedding.named_parameters()):
        load_parameter_weight(parameter, sources[f"time_embedder.{name}"])
    activated = torch.stack(
        tuple(
            torch.nn.functional.silu(embedding(torch.stack((video, audio))))
            for video, audio in zip(schedule.timesteps[0], schedule.timesteps[1], strict=True)
        )
    )
    del embedding

    def projection(prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            sources[f"{prefix}.weight"].full().to(device=device, dtype=torch.bfloat16),
            sources[f"{prefix}.bias"].full().to(device=device, dtype=torch.bfloat16),
        )

    model.modulation_plan = ModulationPlan.materialize(
        activated,
        (
            projection(f"transformer_blocks.{index}.adaln_proj.linear")
            for index in model.pipeline.layers
        ),
        projection("norm_out.linear") if model.pipeline.last else None,
        layer_count=len(model.pipeline.layers),
    )


def _map_transformer(
    model: nn.Module, handles: Iterable[WeightHandle], *, omitted: frozenset[str]
) -> LoadReport:
    """Map checkpoint attention projections into the rank's packed resident layers."""

    mapping = tuple(
        ("attn.to_qkvg", f"attn.{name}", index)
        for index, name in enumerate(("to_q", "to_k", "to_v", "to_gate_compress"))
    )
    parameters = dict(model.named_parameters())
    report = LoadReport()
    for handle in handles:
        checkpoint_name = handle.name.replace(".ff.net.0.proj.", ".ff.gate_up_proj.").replace(
            ".ff.net.2.", ".ff.down_proj."
        )
        # The dense conditioning refiner retains separate Q/K/V projections.
        name, shard = (
            (checkpoint_name, None)
            if checkpoint_name in parameters
            else stacked_weight_name(checkpoint_name, mapping)
        )
        parameter = parameters.get(name)
        if parameter is None:
            if handle.name in omitted:
                report.skipped.append(handle.name)
            else:
                report.unexpected.append(handle.name)
            continue
        load_parameter_weight(parameter, handle, shard)
        report.loaded.add(name)
    return report


def _map_encoder(
    model: MiniMaxH3TextEncoder, handles: Iterable[WeightHandle], *, omitted: frozenset[str]
) -> LoadReport:
    """Select retained Qwen text layers and express their packed checkpoint projection names."""

    mapping = (
        ("self_attn.qkv_proj", "self_attn.q_proj", "q"),
        ("self_attn.qkv_proj", "self_attn.k_proj", "k"),
        ("self_attn.qkv_proj", "self_attn.v_proj", "v"),
        ("mlp.gate_up_proj", "mlp.gate_proj", 0),
        ("mlp.gate_up_proj", "mlp.up_proj", 1),
    )
    parameters = dict(model.named_parameters())
    report = LoadReport()
    for handle in handles:
        name, shard = stacked_weight_name(handle.name.removeprefix("model."), mapping)
        parameter = parameters.get(name)
        if parameter is None:
            if handle.name in omitted:
                report.skipped.append(handle.name)
            else:
                report.unexpected.append(handle.name)
            continue
        load_parameter_weight(parameter, handle, shard)
        report.loaded.add(name)
    return report


def _map_video_decoder(
    model: MiniMaxH3VideoDecoder, handles: Iterable[WeightHandle], *, omitted: frozenset[str]
) -> LoadReport:
    """Translate the checkpoint's value-first feed-forward projection names."""

    parameters = dict(model.named_parameters())
    report = LoadReport()
    for handle in handles:
        name = handle.name.replace(".ff.net.0.proj.", ".ff.gate_up_proj.").replace(
            ".ff.net.2.", ".ff.down_proj."
        )
        parameter = parameters.get(name)
        if parameter is None:
            if handle.name in omitted:
                report.skipped.append(handle.name)
            else:
                report.unexpected.append(handle.name)
            continue
        load_parameter_weight(parameter, handle)
        report.loaded.add(name)
    return report


def _transformer_omissions(
    model: MiniMaxH3Transformer, config: dict[str, Any]
) -> tuple[frozenset[str], frozenset[str]]:
    """Declare records used by other components, PP stages, or modulation.

    The native architecture supplies exact checkpoint names through metadata-only
    construction. A record is omitted only for a known mathematical role; unknown
    names are never inferred to be nonresident merely because lookup failed.
    """

    from diffusers import MiniMaxH3Transformer3DModel

    _, names = construct_owned_module(
        lambda: MiniMaxH3Transformer3DModel.from_config(config), resident=False
    )
    # FastH3 adds one learned VSA compression gate to each native attention
    # block. Diffusers' dense architecture does not enumerate those records.
    names |= frozenset(
        f"transformer_blocks.{index}.attn.to_gate_compress.weight"
        for index in range(model.config.layers)
    )
    conditioning = frozenset(
        name for name in names if name.startswith(("context_embedder.", "token_refiner."))
    )
    omitted = set(conditioning)
    for name in names:
        parts = name.split(".")
        if name.startswith("time_embedder.") or name.startswith("norm_out.linear."):
            omitted.add(name)
        elif parts[0] == "transformer_blocks":
            if int(parts[1]) not in model.pipeline.layers or parts[2] == "adaln_proj":
                omitted.add(name)
        elif not model.pipeline.first and parts[0] in {"proj_in", "audio_proj_in"}:
            omitted.add(name)
        elif not model.pipeline.last and parts[0] in {"norm_out", "proj_out", "audio_proj_out"}:
            omitted.add(name)
    return names - conditioning, frozenset(omitted)


def _encoder_omissions(model: MiniMaxH3TextEncoder, config: dict[str, Any]) -> frozenset[str]:
    """Describe the unused visual tower, vocabulary head, and later text layers."""

    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    native_config = Qwen3VLConfig(
        **{
            **config,
            "text_config": {
                "num_hidden_layers": model.config.checkpoint_layers,
                **config.get("text_config", {}),
            },
        }
    )
    _, names = construct_owned_module(
        lambda: Qwen3VLForConditionalGeneration(native_config), resident=False
    )
    return frozenset(
        name
        for name in names
        if name.startswith("model.visual.")
        or name in {"lm_head.weight", "model.language_model.norm.weight"}
        or (
            name.startswith("model.language_model.layers.")
            and int(name.split(".")[3]) >= model.config.retained_layers
        )
    )


def _video_omissions(config: dict[str, Any]) -> frozenset[str]:
    """Declare the native encoder and posterior projection unused by decoding."""

    from diffusers import AutoencoderKLMiniMaxH3

    _, names = construct_owned_module(
        lambda: AutoencoderKLMiniMaxH3.from_config(config), resident=False
    )
    return frozenset(name for name in names if name.startswith(("encoder.", "quant_conv.")))


def build_components(
    config: dict[str, Any], context: BuildContext
) -> tuple[H3Components, H3Layout, tuple[CheckpointComponent, ...]]:
    """Declare resident H3 components; the shared loader owns their materialization."""

    meshes, schedule = context.meshes, context.schedule
    if schedule is None:
        raise ValueError("H3 construction requires its diffusion schedule")
    device = schedule.sigmas[0].device
    precisions = context.component_precisions
    text_capacity = ((int(context.limits["text_tokens"]) + 63) // 64) * 64
    raw_frames = math.floor(float(context.limits["video_seconds"]) * 24.0 + 0.5)
    max_frames = int(raw_frames + (5 - raw_frames) % 17)
    if text_capacity < 64 or max_frames < MIN_H3_FRAMES:
        raise ValueError("H3 numerical limits are smaller than a legal shape")
    layout = H3Layout.build(
        context.parallel.get("denoiser", ParallelConfig()),
        meshes.get("denoiser"),
        frames=max_frames,
        text_rows=text_capacity,
        audio_frames=audio_latent_frames(max_frames),
        postprocess="output" in meshes,
    )
    components = []
    transformer = encoder = video_decoder = audio_decoder = None
    conditioner = None
    mesh = meshes.get("denoiser")
    if mesh is not None:
        transformer = MiniMaxH3Transformer(
            mesh,
            parameter_device="meta",
            attention_linear_precision=precisions["transformer.attention"],
            mlp_linear_precision=precisions["transformer.mlp"],
        )
        conditioning_omissions, transformer_omissions = _transformer_omissions(
            transformer, config.get("transformer", {})
        )
        if transformer.pipeline.first:
            conditioner = build_conditioner(mesh, "meta")
            components.append(
                CheckpointComponent(
                    conditioner,
                    source="denoiser",
                    map_weights=partial(
                        _map_transformer, conditioner, omitted=conditioning_omissions
                    ),
                    dtype=torch.bfloat16,
                )
            )
        components.append(
            CheckpointComponent(
                transformer,
                source="denoiser",
                map_weights=partial(_map_transformer, transformer, omitted=transformer_omissions),
                dtype=torch.bfloat16,
                parameter_dtypes=tuple(
                    (name, torch.float32)
                    for name in ("proj_in", "audio_proj_in", "proj_out", "audio_proj_out")
                    if getattr(transformer, name) is not None
                ),
                post_load=partial(
                    _prepare_modulation, transformer, device=device, schedule=schedule
                ),
            )
        )
    mesh = meshes.get("text_encoder")
    if mesh is not None:
        encoder = MiniMaxH3TextEncoder(
            mesh,
            max_text_rows=text_capacity,
            parameter_device="meta",
            linear_precision=precisions["text_encoder"],
        )
        components.append(
            CheckpointComponent(
                encoder,
                source="text_encoder",
                map_weights=partial(
                    _map_encoder,
                    encoder,
                    omitted=_encoder_omissions(encoder, config.get("text_encoder", {})),
                ),
                dtype=torch.bfloat16,
            )
        )
    if "video_decoder" in meshes:
        precision = precisions["video_vae"]
        dense = precision in {"fp16", "bf16"}
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        # The 24-channel input projection is not aligned for NVFP4 packing.
        quantization = (
            None
            if dense
            else QuantizationConfig(method=precision, ignored_layers=("decoder.proj_in",))
        )
        video_decoder = MiniMaxH3VideoDecoder(
            layer_config=LayerConfig(Communicator(), quantization),
            parameter_device="meta",
            buffer_device=device,
        )
        components.append(
            CheckpointComponent(
                video_decoder,
                source="video_decoder",
                map_weights=partial(
                    _map_video_decoder,
                    video_decoder,
                    omitted=_video_omissions(config.get("video_vae", {})),
                ),
                dtype=torch.float32,
                parameter_dtypes=tuple(
                    (name, dtype)
                    for name, module in video_decoder.named_modules()
                    if (
                        isinstance(module, LinearBase)
                        and (dense or module.quant_method.is_quantized)
                    )
                    or (dense and isinstance(module, torch.nn.Conv3d))
                ),
            )
        )
    if "audio_decoder" in meshes:
        from diffusers import AutoencoderKLMiniMaxH3Audio

        audio_config = config["audio_vae"]
        with construction_dtype(torch.float32), torch.device("meta"):
            audio_decoder = AutoencoderKLMiniMaxH3Audio.from_config(audio_config)
        components.append(
            CheckpointComponent(
                audio_decoder,
                source="audio_decoder",
                dtype=torch.float32,
            )
        )

    return (
        H3Components(
            transformer=transformer,
            conditioner=conditioner,
            encoder=encoder,
            video_vae=MiniMaxH3VideoVAE(video_decoder, linear_precision=precisions["video_vae"])
            if video_decoder is not None
            else None,
            audio_vae=MiniMaxH3AudioVAE(audio_decoder) if audio_decoder is not None else None,
        ),
        layout,
        tuple(components),
    )
