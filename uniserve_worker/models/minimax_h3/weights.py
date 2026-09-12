"""H3 component geometry, checkpoint name mapping and numerical precomputation."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ...loader.component import (
    CheckpointComponent,
    ModelBuildContext,
    ModelConstruction,
    construction_dtype,
)
from ...loader.handles import WeightHandle
from ...loader.mapping import LoadReport, stacked_weight_name
from ...loader.weight_loaders import attach_parameter_loaders, load_parameter_weight
from ...nn.diffusion.schedule import DiffusionSchedule
from ...nn.layer import LayerConfig
from ...nn.linear import LinearBase
from ...nn.mesh import Communicator, EntryBindings
from ...nn.quant.config import QuantizationConfig
from .audio_vae import MiniMaxH3AudioVAE
from .config import H3TransformerConfig, resolve_h3_contract
from .encoder import H3TextEncoderConfig, MiniMaxH3TextEncoder
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
    """Resident computation modules assembled after their public loader finalizes them."""

    transformer: MiniMaxH3Transformer | None
    conditioner: nn.Sequential | None
    encoder: MiniMaxH3TextEncoder | None
    video_vae: MiniMaxH3VideoVAE | None
    audio_vae: MiniMaxH3AudioVAE | None


def validate_h3_entries(bindings: EntryBindings) -> None:
    """Validate component placement before constructing checkpoint modules."""

    expected = {"denoiser", "text_encoder", "video_decoder", "audio_decoder", "output"}
    if not bindings.entries or not set(bindings.entries) <= expected:
        raise ValueError(f"H3 entries must belong to {sorted(expected)}")
    for name, component in bindings.entries.items():
        config = component.parallel_config
        if name == "video_decoder":
            if component.distribution != "temporal_units" or component.units_per_rank != 1:
                raise ValueError("H3 video decoder requires temporal_units with native batch one")
            continue
        if component.distribution is not None:
            raise ValueError(f"H3 {name} requires model-parallel membership")
        if name in {"audio_decoder", "output"}:
            if len(component.ranks) != 1 or config.world_size != 1:
                raise ValueError(f"H3 {name} requires one local owner")
        elif name == "text_encoder":
            if config.pipeline_parallel_size != 1 or config.sequence_parallel_size != 1:
                raise ValueError("H3 text encoder supports direct tensor parallelism")
            if any(width % config.tensor_parallel_size for width in (64, 25600)):
                raise ValueError("H3 encoder TP must divide query heads and MLP width")
        elif name == "denoiser":
            if config.pipeline_parallel_size > 50:
                raise ValueError("H3 pipeline stages cannot exceed its 50 transformer layers")
            if config.sequence_parallel.kind not in {
                "local",
                "ulysses",
                "allgather",
                "ring",
                "hybrid",
                "attention2d",
            }:
                raise ValueError("H3 sequence attention requires global sparse selection")
            tensor = config.tensor_parallel_size
            ulysses = dict(config.dimensions)["ulysses"]
            if 56 % (tensor * ulysses) or 5376 % tensor or 14336 % tensor:
                raise ValueError(
                    "H3 TP × Ulysses must divide heads; TP must divide hidden and MLP widths"
                )


def require_h3_checkpoint(root: Path) -> None:
    """Validate checkpoint component files and tensor dimensions against the H3 architecture."""

    resolve_h3_contract(root)
    transformer = json.loads((root / "transformer" / "config.json").read_text(encoding="utf-8"))
    transformer_config = H3TransformerConfig()
    expected_transformer = {
        "num_attention_heads": transformer_config.heads,
        "attention_head_dim": transformer_config.head_dim,
        "hidden_size": transformer_config.hidden_size,
        "num_layers": transformer_config.layers,
        "num_refiner_layers": transformer_config.refiner_layers,
        "ffn_dim": transformer_config.ffn_dim,
        "in_channels": transformer_config.video_channels,
        "audio_in_channels": transformer_config.audio_channels,
        "patch_size": [1, 2, 2],
        "text_dim": transformer_config.text_dim,
        "freq_dim": transformer_config.frequency_dim,
        "time_embed_hidden_dim": transformer_config.time_hidden_dim,
        "time_embed_dim": transformer_config.time_dim,
        "rope_freq_dim": transformer_config.rope_frequency_dim,
        "rope_theta": transformer_config.rope_theta,
        "norm_eps": transformer_config.norm_eps,
        "qk_norm_eps": transformer_config.qk_norm_eps,
        "final_norm_eps": transformer_config.norm_eps,
    }
    for field, expected in expected_transformer.items():
        if transformer.get(field) != expected:
            raise ValueError(
                f"FastH3 transformer {field} must be {expected!r}, got {transformer.get(field)!r}"
            )

    encoder = json.loads((root / "text_encoder" / "config.json").read_text(encoding="utf-8")).get(
        "text_config"
    )
    if not isinstance(encoder, dict):
        raise ValueError("FastH3 text encoder has no Qwen3-VL text configuration")
    encoder_config = H3TextEncoderConfig()
    expected_encoder = {
        "vocab_size": encoder_config.vocab_size,
        "hidden_size": encoder_config.hidden_size,
        "intermediate_size": encoder_config.intermediate_size,
        "num_hidden_layers": encoder_config.checkpoint_layers,
        "num_attention_heads": encoder_config.heads,
        "num_key_value_heads": encoder_config.kv_heads,
        "head_dim": encoder_config.head_dim,
        "rope_theta": encoder_config.rope_theta,
        "rms_norm_eps": encoder_config.norm_eps,
    }
    for field, expected in expected_encoder.items():
        if encoder.get(field) != expected:
            raise ValueError(
                f"FastH3 text encoder {field} must be {expected!r}, got {encoder.get(field)!r}"
            )

    for component, expected_shift in (("scheduler", 12.0), ("audio_scheduler", 3.0)):
        scheduler = json.loads(
            (root / component / "scheduler_config.json").read_text(encoding="utf-8")
        )
        if scheduler.get("shift") != expected_shift:
            raise ValueError(
                f"FastH3 {component} shift must be {expected_shift:g}, got {scheduler.get('shift')!r}"
            )


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


def _map_transformer(model: nn.Module, handles: Iterable[WeightHandle]) -> LoadReport:
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
            report.skipped.append(handle.name)
            continue
        load_parameter_weight(parameter, handle, shard)
        report.loaded.add(name)
    return report


def _map_encoder(model: MiniMaxH3TextEncoder, handles: Iterable[WeightHandle]) -> LoadReport:
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
            report.skipped.append(handle.name)
            continue
        load_parameter_weight(parameter, handle, shard)
        report.loaded.add(name)
    return report


def _map_video_decoder(model: MiniMaxH3VideoDecoder, handles: Iterable[WeightHandle]) -> LoadReport:
    """Translate the checkpoint's value-first feed-forward projection names."""

    parameters = dict(model.named_parameters())
    report = LoadReport()
    for handle in handles:
        name = handle.name.replace(".ff.net.0.proj.", ".ff.gate_up_proj.").replace(
            ".ff.net.2.", ".ff.down_proj."
        )
        parameter = parameters.get(name)
        if parameter is None:
            report.skipped.append(handle.name)
            continue
        load_parameter_weight(parameter, handle)
        report.loaded.add(name)
    return report


def build_h3_checkpoint(config: dict[str, Any], context: ModelBuildContext) -> ModelConstruction:
    """Declare resident H3 components; the shared loader owns their materialization."""

    from .model import MiniMaxH3Model

    request = context.request
    bindings, schedule = request.bindings, context.schedule
    if schedule is None:
        raise ValueError("H3 construction requires its diffusion schedule")
    validate_h3_entries(bindings)
    require_h3_checkpoint(context.root)
    contract = resolve_h3_contract(context.root)
    device = bindings.process_group.device
    precisions = context.component_precisions
    text_capacity = ((int(request.max_text_rows) + 63) // 64) * 64
    raw_frames = math.floor(float(request.max_video_seconds) * 24.0 + 0.5)
    max_frames = int(raw_frames + (5 - raw_frames) % 17)
    if text_capacity < 64 or max_frames < MIN_H3_FRAMES:
        raise ValueError("H3 worker_config capacity is smaller than a legal request")
    layout = H3Layout.build(
        bindings,
        frames=max_frames,
        text_rows=text_capacity,
        audio_frames=audio_latent_frames(max_frames),
        attention=str(contract["attention"]),
        video_dtype=torch.float32 if precisions["video_vae"] == "fp32" else torch.float16,
    )
    components = []
    transformer = encoder = video_decoder = audio_decoder = None
    conditioner = None
    mesh = bindings.meshes.get("denoiser")
    if mesh is not None:
        transformer = MiniMaxH3Transformer(
            mesh,
            parameter_device="meta",
            attention_linear_precision=precisions["transformer.attention"],
            mlp_linear_precision=precisions["transformer.mlp"],
            attention=str(contract["attention"]),
        )
        if transformer.pipeline.first:
            conditioner = build_conditioner(mesh, "meta")
            components.append(
                CheckpointComponent(
                    conditioner,
                    source="denoiser",
                    map_weights=partial(_map_transformer, conditioner),
                    dtype=torch.bfloat16,
                )
            )
        components.append(
            CheckpointComponent(
                transformer,
                source="denoiser",
                map_weights=partial(_map_transformer, transformer),
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
    mesh = bindings.meshes.get("text_encoder")
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
                map_weights=partial(_map_encoder, encoder),
                dtype=torch.bfloat16,
            )
        )
    if bindings.owns("video_decoder"):
        precision = precisions["video_vae"]
        dense = precision in {"fp32", "fp16", "bf16"}
        dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(precision, torch.bfloat16)
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
                map_weights=partial(_map_video_decoder, video_decoder),
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
                strict=False,
            )
        )
    if bindings.owns("audio_decoder"):
        from diffusers import AutoencoderKLMiniMaxH3Audio

        audio_config = json.loads((context.root / "audio_vae" / "config.json").read_text())
        with construction_dtype(torch.float32), torch.device("meta"):
            audio_decoder = AutoencoderKLMiniMaxH3Audio.from_config(audio_config)
        components.append(
            CheckpointComponent(
                audio_decoder,
                source="audio_decoder",
                dtype=torch.float32,
                persistent_buffers=True,
            )
        )

    def assemble() -> MiniMaxH3Model:
        return MiniMaxH3Model(
            bindings,
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
            denoise_steps=int(contract["denoise_steps"]),
        )

    return ModelConstruction(tuple(components), assemble, config)
