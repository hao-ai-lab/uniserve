"""Self-contained FastH3 v0.2 component discovery and rank-aware loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .audio_vae import MiniMaxH3AudioVAE
from .encoder import H3TextEncoderConfig, MiniMaxH3TextEncoder
from .placement import H3Placement
from .precision import H3LinearPrecisionPolicy
from .transformer import H3TimestepEmbedding, H3TransformerConfig, MiniMaxH3Transformer
from .video_vae import MiniMaxH3VideoVAE

__all__ = ["H3Checkpoint", "H3Components", "load_h3_components", "resolve_h3_checkpoint"]

CHECKPOINT_ID = "FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2"


@dataclass(frozen=True, slots=True)
class H3Checkpoint:
    """Canonical component paths rooted at one H3 checkpoint directory."""

    root: Path


@dataclass(slots=True)
class H3Components:
    """Bundles the H3 transformer, text encoder, video VAE, audio VAE, and their checkpoint root."""

    checkpoint: H3Checkpoint
    transformer: MiniMaxH3Transformer | None
    encoder: MiniMaxH3TextEncoder | None
    video_vae: MiniMaxH3VideoVAE | None
    audio_vae: MiniMaxH3AudioVAE | None


def resolve_h3_checkpoint(
    model_path: str,
    *,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> H3Checkpoint:
    """Resolve a local or Hugging Face checkpoint and verify its component manifest."""

    candidate = Path(model_path).expanduser()
    if candidate.is_dir():
        root = candidate.resolve()
    else:
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                repo_id=model_path,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=[
                    "modular_model_index.json",
                    "transformer/*.json",
                    "transformer/*.safetensors",
                    "text_encoder/*.json",
                    "text_encoder/*.safetensors",
                    "tokenizer/*",
                    "processor/*",
                    "scheduler/*",
                    "audio_scheduler/*",
                    "vae/*",
                    "audio_vae/*",
                ],
            )
        ).resolve()

    # Every execution component has an independent config and weight namespace.
    required = (
        "modular_model_index.json",
        "transformer/config.json",
        "text_encoder/config.json",
        "vae/config.json",
        "audio_vae/config.json",
        "scheduler/scheduler_config.json",
        "audio_scheduler/scheduler_config.json",
    )
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            f"FastH3 checkpoint {root} is missing required components {missing!r}"
        )
    return H3Checkpoint(root=root)


def _require_checkpoint_geometry(checkpoint: H3Checkpoint) -> None:
    """Validate checkpoint component files and tensor dimensions against the H3 architecture."""

    transformer = json.loads(
        (checkpoint.root / "transformer" / "config.json").read_text(encoding="utf-8")
    )
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

    encoder = json.loads(
        (checkpoint.root / "text_encoder" / "config.json").read_text(encoding="utf-8")
    ).get("text_config")
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
            (checkpoint.root / component / "scheduler_config.json").read_text(encoding="utf-8")
        )
        if scheduler.get("shift") != expected_shift:
            raise ValueError(
                f"FastH3 {component} shift must be {expected_shift:g}, got {scheduler.get('shift')!r}"
            )


def _weight_map(component: Path) -> dict[str, Any]:
    """Read and validate the safetensors index that maps parameter names to shard files."""

    from ...loader.handles import SafetensorFileWeightHandle, safetensor_dtype

    indexes = sorted(component.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise RuntimeError(f"component {component} contains multiple safetensor indexes")
    if indexes:
        payload = json.loads(indexes[0].read_text(encoding="utf-8"))
        mapping = payload.get("weight_map")
        if not isinstance(mapping, dict):
            raise RuntimeError(f"checkpoint index {indexes[0]} has no weight_map")
        locations = {str(name): component / str(filename) for name, filename in mapping.items()}
    else:
        files = sorted(component.glob("*.safetensors"))
        if len(files) != 1:
            raise RuntimeError(f"component {component} has no unambiguous safetensor source")
        from safetensors.torch import safe_open

        with safe_open(files[0], framework="pt", device="cpu") as source:
            locations = {str(name): files[0] for name in source.keys()}
    missing = sorted({path for path in locations.values() if not path.is_file()})
    if missing:
        raise FileNotFoundError(f"component {component} is missing weight shards {missing!r}")
    from safetensors.torch import safe_open

    handles: dict[str, Any] = {}
    by_path: dict[Path, list[str]] = {}
    for name, path in locations.items():
        by_path.setdefault(path, []).append(name)
    for path, names in sorted(by_path.items()):
        with safe_open(path, framework="pt", device="cpu") as source:
            available = set(source.keys())
            absent = sorted(set(names) - available)
            if absent:
                raise KeyError(f"checkpoint shard {path} is missing indexed tensor {absent[0]!r}")
            for name in names:
                value = source.get_slice(name)
                handles[name] = SafetensorFileWeightHandle(
                    name=name,
                    path=path,
                    shape=tuple(int(size) for size in value.get_shape()),
                    dtype=safetensor_dtype(str(value.get_dtype())),
                )
    return handles


def _set_parameter(module: nn.Module, name: str, value: torch.Tensor) -> None:
    """Replace a nested parameter by dotted checkpoint name without enabling gradients."""

    path, _, field = name.rpartition(".")
    owner = module.get_submodule(path)
    parameter = getattr(owner, field)
    if not isinstance(parameter, nn.Parameter):
        raise TypeError(f"target {name!r} is not a parameter")
    if tuple(parameter.shape) != tuple(value.shape):
        raise ValueError(
            f"checkpoint tensor {name!r} shape {tuple(value.shape)} does not match {tuple(parameter.shape)}"
        )
    setattr(owner, field, nn.Parameter(value, requires_grad=False))


def _transformer_dtype(name: str) -> torch.dtype:
    """Map a transformer parameter name to its checkpoint storage dtype."""

    fp32_prefixes = (
        "proj_in.",
        "audio_proj_in.",
        "proj_out.",
        "audio_proj_out.",
    )
    return torch.float32 if name.startswith(fp32_prefixes) else torch.bfloat16


@torch.inference_mode()
def _prepare_modulation(
    model: MiniMaxH3Transformer, sources: dict[str, Any], device: torch.device
) -> None:
    """Stream fixed-timestep checkpoint projections into model-owned products."""

    from ...nn.diffusion.modulation import ModulationPlan

    embedding = H3TimestepEmbedding(model.config, device="meta", buffer_device=device)
    for name, _parameter in tuple(embedding.named_parameters()):
        value = sources[f"time_embedder.{name}"].full().to(device=device, dtype=torch.float32)
        _set_parameter(embedding, name, value)
    schedule = model.layout.schedule
    activated = torch.stack(
        tuple(
            torch.nn.functional.silu(embedding(torch.stack((video, audio))))
            for video, audio in zip(schedule.video_timesteps, schedule.audio_timesteps, strict=True)
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


def _load_transformer(
    model: MiniMaxH3Transformer,
    component: Path,
    device: torch.device,
) -> None:
    """Stream transformer shards into resident parameters and finalize quantized projections."""

    from ...loader.handles import weight_handle_materialization
    from ...loader.weight_loaders import attach_parameter_loaders, load_parameter_weight
    from ...nn.placement import get_shard_plan
    from ...nn.quant import process_quantized_modules

    sources = _weight_map(component)
    _prepare_modulation(model, sources, device)
    attach_parameter_loaders(model, device=device, dtype=torch.bfloat16)
    targets = dict(model.named_parameters())
    with weight_handle_materialization():
        for name, target in targets.items():
            packed_suffix = ".attn.to_qkvg.weight"
            if name.startswith("transformer_blocks.") and name.endswith(packed_suffix):
                prefix = name[: -len("to_qkvg.weight")]
                source_names = tuple(
                    prefix + suffix
                    for suffix in (
                        "to_q.weight",
                        "to_k.weight",
                        "to_v.weight",
                        "to_gate_compress.weight",
                    )
                )
                for projection, source_name in enumerate(source_names):
                    try:
                        handle = sources[source_name]
                    except KeyError as error:
                        raise KeyError(
                            f"FastH3 transformer is missing checkpoint tensor {source_name!r}"
                        ) from error
                    load_parameter_weight(target, handle, projection)
                continue
            try:
                handle = sources[name]
            except KeyError as error:
                raise KeyError(
                    f"FastH3 transformer is missing checkpoint tensor {name!r}"
                ) from error
            if get_shard_plan(target) is not None:
                load_parameter_weight(target, handle)
                continue
            tensor = handle.full().to(
                device=device,
                dtype=_transformer_dtype(name),
                non_blocking=False,
            )
            _set_parameter(model, name, tensor)
    packed_names = {
        f"transformer_blocks.{index}.attn.to_qkvg.weight" for index in model.pipeline.layers
    }
    if not packed_names <= targets.keys():
        raise RuntimeError("FastH3 transformer did not materialize all attention projections")
    process_quantized_modules(model.modules())


def _load_encoder(
    model: MiniMaxH3TextEncoder,
    component: Path,
    device: torch.device,
) -> None:
    """Stream retained text-encoder layers and embeddings into resident parameters."""

    from ...loader.handles import weight_handle_materialization
    from ...loader.weight_loaders import attach_parameter_loaders, load_parameter_weight
    from ...nn.quant import process_quantized_modules

    sources = _weight_map(component)
    attach_parameter_loaders(model, device=device, dtype=torch.bfloat16)
    targets = dict(model.named_parameters())
    with weight_handle_materialization():
        for name, target in targets.items():
            if name.endswith(".self_attn.qkv_proj.weight"):
                prefix = name[: -len("qkv_proj.weight")]
                for shard, projection in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")):
                    source_name = f"model.{prefix}{projection}.weight"
                    try:
                        handle = sources[source_name]
                    except KeyError as error:
                        raise KeyError(
                            f"H3 text encoder is missing checkpoint tensor {source_name!r}"
                        ) from error
                    load_parameter_weight(target, handle, shard)
                continue
            if name.endswith(".mlp.gate_up_proj.weight"):
                prefix = name[: -len("gate_up_proj.weight")]
                for shard, projection in (("gate", "gate_proj"), ("up", "up_proj")):
                    source_name = f"model.{prefix}{projection}.weight"
                    try:
                        handle = sources[source_name]
                    except KeyError as error:
                        raise KeyError(
                            f"H3 text encoder is missing checkpoint tensor {source_name!r}"
                        ) from error
                    load_parameter_weight(target, handle, shard)
                continue
            source_name = f"model.{name}"
            try:
                handle = sources[source_name]
            except KeyError as error:
                raise KeyError(
                    f"H3 text encoder is missing checkpoint tensor {source_name!r}"
                ) from error
            load_parameter_weight(target, handle)
    process_quantized_modules(model.modules())


def load_h3_components(
    checkpoint_path: str,
    placement: H3Placement,
    layout: Any,
    *,
    cache_dir: str | None = None,
    revision: str | None = None,
    precision_policy: H3LinearPrecisionPolicy,
) -> H3Components:
    """Validate and materialize the rank-local H3 transformer, encoder, and VAE modules."""

    checkpoint = resolve_h3_checkpoint(
        checkpoint_path,
        cache_dir=cache_dir,
        revision=revision,
    )
    _require_checkpoint_geometry(checkpoint)
    device = placement.process_group.device
    transformer = None
    mesh = placement.denoiser_mesh
    if mesh is not None:
        transformer = MiniMaxH3Transformer(
            mesh,
            layout,
            parameter_device="meta",
            attention_linear_precision=precision_policy.transformer_attention,
            mlp_linear_precision=precision_policy.transformer_mlp,
        )
        _load_transformer(transformer, checkpoint.root / "transformer", device)
    encoder = None
    encoder_mesh = placement.encoder_mesh
    if encoder_mesh is not None:
        encoder = MiniMaxH3TextEncoder(
            encoder_mesh,
            max_text_rows=int(layout.packed.text_indices.numel()),
            parameter_device="meta",
            linear_precision=precision_policy.text_encoder,
        )
        _load_encoder(encoder, checkpoint.root / "text_encoder", device)
    video_vae = (
        MiniMaxH3VideoVAE.from_pretrained(
            str(checkpoint.root),
            device=device,
            local_files_only=True,
            linear_precision=precision_policy.video_vae,
        )
        if placement.owns("video_decoder")
        else None
    )
    audio_vae = (
        MiniMaxH3AudioVAE.from_pretrained(
            str(checkpoint.root), device=device, local_files_only=True
        )
        if placement.owns("audio_decoder")
        else None
    )
    return H3Components(
        checkpoint=checkpoint,
        transformer=transformer,
        encoder=encoder,
        video_vae=video_vae,
        audio_vae=audio_vae,
    )
