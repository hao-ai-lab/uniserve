"""H3 component geometry, checkpoint name mapping and numerical precomputation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from functools import partial

import torch
from torch import nn

from uniserve.distributed.mesh import DeviceMesh
from uniserve.distributed.parallel import ParallelConfig
from uniserve.loading.component import (
    CheckpointComponent,
    construct_owned_module,
    construction_dtype,
)
from uniserve.loading.handles import WeightHandle
from uniserve.loading.mapping import LoadReport, stacked_weight_name
from uniserve.loading.weight_loaders import attach_parameter_loaders, load_parameter_weight
from uniserve.model.limits import ModelLimits
from uniserve.nn.diffusion.schedule import DiffusionSchedule
from uniserve.nn.layer import LayerConfig
from uniserve.nn.linear import LinearBase
from uniserve.nn.quant.config import LinearPrecision, QuantizationConfig
from uniserve_models.minimax_h3.audio_vae import MiniMaxH3AudioVAE
from uniserve_models.minimax_h3.config import (
    TEXT_FIELDS,
    TRANSFORMER_FIELDS,
    H3Config,
    H3TransformerConfig,
)
from uniserve_models.minimax_h3.encoder import MiniMaxH3TextEncoder
from uniserve_models.minimax_h3.layout import MIN_H3_FRAMES, H3Layout
from uniserve_models.minimax_h3.packing import audio_latent_frames
from uniserve_models.minimax_h3.transformer import (
    H3TimestepEmbedding,
    MiniMaxH3Transformer,
    build_conditioner,
)
from uniserve_models.minimax_h3.video_vae import (
    MiniMaxH3VideoDecoder,
    MiniMaxH3VideoVAE,
    VideoDecoderConfig,
)


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

    from uniserve.nn.diffusion.modulation import ModulationPlan

    embedding = H3TimestepEmbedding(model.config, device="meta", buffer_device=device)
    attach_parameter_loaders(embedding, device=device, dtype=torch.float32)
    for name, parameter in tuple(embedding.named_parameters()):
        load_parameter_weight(parameter, sources[f"time_embedder.{name}"])
    activated = torch.stack(
        tuple(
            torch.nn.functional.silu(embedding(torch.stack((video, audio))))
            for video, audio in zip(
                schedule.timesteps[0][:-1], schedule.timesteps[1][:-1], strict=True
            )
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
    model: MiniMaxH3Transformer, config: H3TransformerConfig
) -> tuple[frozenset[str], frozenset[str]]:
    """Declare records used by other components, PP stages, or modulation.

    The native architecture supplies exact checkpoint names through metadata-only
    construction. A record is omitted only for a known mathematical role; unknown
    names are never inferred to be nonresident merely because lookup failed.
    """

    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

    _, names = construct_owned_module(
        lambda: MiniMaxH3Transformer3DModel(
            **{
                **{
                    source: getattr(config, target) for source, target in TRANSFORMER_FIELDS.items()
                },
                "patch_size": (1, 2, 2),
                "final_norm_eps": config.norm_eps,
            }
        ),
        resident=False,
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


def _encoder_omissions(model: MiniMaxH3TextEncoder) -> frozenset[str]:
    """Describe the unused visual tower, vocabulary head, and later text layers."""

    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    native_config = Qwen3VLConfig(
        text_config={
            source: getattr(model.config, target) for source, target in TEXT_FIELDS.items()
        },
        vision_config={"depth": 27, "deepstack_visual_indexes": [8, 16, 24]},
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


def _video_omissions(config: VideoDecoderConfig) -> frozenset[str]:
    """Declare the native encoder and posterior projection unused by decoding."""

    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

    _, names = construct_owned_module(
        lambda: AutoencoderKLMiniMaxH3(
            **{
                name: value
                for name, value in asdict(config).items()
                if name not in {"width", "height", "fps"}
            }
        ),
        resident=False,
    )
    return frozenset(name for name in names if name.startswith(("encoder.", "quant_conv.")))


def configure_layers(
    layers: Mapping[str, LayerConfig], precisions: Mapping[str, LinearPrecision]
) -> dict[str, LayerConfig]:
    """Bind numerical precision to the H3 projection and reconstruction owners."""

    result = dict(layers)
    for owner, name, selection in (
        ("denoiser", "denoiser.attention", "transformer.attention"),
        ("denoiser", "denoiser.mlp", "transformer.mlp"),
        ("text_encoder", "text_encoder", "text_encoder"),
        ("video_decoder", "video_decoder", "video_vae"),
    ):
        if owner not in layers:
            continue
        precision = precisions[selection]
        # The native video input projection has 24 channels and cannot use
        # the quantized backend's aligned packing. Other projections retain it.
        quantization = (
            None
            if precision in {"fp16", "bf16"}
            else QuantizationConfig(
                method=precision,
                ignored_layers=("decoder.proj_in",) if owner == "video_decoder" else (),
            )
        )
        result[name] = replace(
            layers[owner],
            quantization=quantization,
            dense_dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
        )
    return result


def build_components(
    config: H3Config,
    *,
    parallel: Mapping[str, ParallelConfig],
    meshes: Mapping[str, DeviceMesh],
    layers: Mapping[str, LayerConfig],
    limits: ModelLimits,
) -> tuple[H3Components, H3Layout, tuple[CheckpointComponent, ...]]:
    """Declare resident H3 components; the shared loader owns their materialization."""

    text_capacity = ((limits.text_tokens + 63) // 64) * 64
    raw_frames = limits.video_frames
    max_frames = int(raw_frames + (5 - raw_frames) % 17)
    if text_capacity < 64 or max_frames < MIN_H3_FRAMES:
        raise ValueError("H3 numerical limits are smaller than a legal shape")
    layout = H3Layout.build(
        parallel.get("denoiser", ParallelConfig()),
        meshes.get("denoiser"),
        frames=max_frames,
        text_rows=text_capacity,
        audio_frames=audio_latent_frames(max_frames),
    )
    components = []
    transformer = encoder = video_decoder = audio_decoder = None
    conditioner = None
    mesh = meshes.get("denoiser")
    if mesh is not None:
        device = mesh.local_device
        diffusion = config.diffusion
        schedule = DiffusionSchedule.build(
            diffusion.ladder,
            (diffusion.video_shift, diffusion.audio_shift),
            scale=diffusion.time_scale,
            device=device,
        )
        transformer = MiniMaxH3Transformer(
            config.denoiser,
            mesh,
            parameter_device="meta",
            attention_linear_precision=layers["denoiser.attention"].linear_precision,
            mlp_linear_precision=layers["denoiser.mlp"].linear_precision,
        )
        conditioning_omissions, transformer_omissions = _transformer_omissions(
            transformer, config.denoiser
        )
        if transformer.pipeline.first:
            conditioner = build_conditioner(config.denoiser, mesh, "meta")
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
            config.text_encoder,
            mesh,
            max_text_rows=text_capacity,
            parameter_device="meta",
            linear_precision=layers["text_encoder"].linear_precision,
        )
        components.append(
            CheckpointComponent(
                encoder,
                source="text_encoder",
                map_weights=partial(
                    _map_encoder,
                    encoder,
                    omitted=_encoder_omissions(encoder),
                ),
                dtype=torch.bfloat16,
            )
        )
    if "video_decoder" in meshes:
        precision = layers["video_decoder"].linear_precision
        dense = precision in {"fp16", "bf16"}
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        video_decoder = MiniMaxH3VideoDecoder(
            config.video_decoder,
            layer_config=layers["video_decoder"],
            parameter_device="meta",
            buffer_device=meshes["video_decoder"].local_device,
        )
        components.append(
            CheckpointComponent(
                video_decoder,
                source="video_decoder",
                map_weights=partial(
                    _map_video_decoder,
                    video_decoder,
                    omitted=_video_omissions(config.video_decoder),
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
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
            AutoencoderKLMiniMaxH3Audio,
        )

        audio_config = asdict(config.audio_decoder)
        with construction_dtype(torch.float32), torch.device("meta"):
            audio_decoder = AutoencoderKLMiniMaxH3Audio(**audio_config)
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
            video_vae=MiniMaxH3VideoVAE(
                video_decoder, linear_precision=layers["video_decoder"].linear_precision
            )
            if video_decoder is not None
            else None,
            audio_vae=MiniMaxH3AudioVAE(audio_decoder, config.audio_decoder)
            if audio_decoder is not None
            else None,
        ),
        layout,
        tuple(components),
    )
