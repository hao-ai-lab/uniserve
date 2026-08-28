"""Self-contained FastH3 v0.2 component discovery and rank-aware loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from ...nn.mesh import DeviceMesh
from .audio_vae import MiniMaxH3AudioVAE
from .encoder import H3TextEncoderConfig, MiniMaxH3TextEncoder
from .transformer import H3TransformerConfig, MiniMaxH3Transformer
from .video_vae import MiniMaxH3VideoVAE

__all__ = ["H3Checkpoint", "H3Components", "load_h3_components", "resolve_h3_checkpoint"]

CHECKPOINT_ID = "FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2"


@dataclass(frozen=True, slots=True)
class H3Checkpoint:
    root: Path


@dataclass(slots=True)
class H3Components:
    checkpoint: H3Checkpoint
    transformer: MiniMaxH3Transformer
    encoder: MiniMaxH3TextEncoder
    video_vae: MiniMaxH3VideoVAE
    audio_vae: MiniMaxH3AudioVAE | None
    tokenizer: Any | None


def resolve_h3_checkpoint(
    model_path: str,
    *,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> H3Checkpoint:
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
                allow_patterns=(
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
                ),
            )
        ).resolve()
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
    owner: nn.Module = module
    fields = name.split(".")
    for field in fields[:-1]:
        owner = owner[int(field)] if field.isdigit() else getattr(owner, field)
    parameter = getattr(owner, fields[-1])
    if not isinstance(parameter, nn.Parameter):
        raise TypeError(f"target {name!r} is not a parameter")
    if tuple(parameter.shape) != tuple(value.shape):
        raise ValueError(
            f"checkpoint tensor {name!r} shape {tuple(value.shape)} does not match {tuple(parameter.shape)}"
        )
    setattr(owner, fields[-1], nn.Parameter(value, requires_grad=False))


def _transformer_dtype(name: str) -> torch.dtype:
    fp32_prefixes = (
        "proj_in.",
        "audio_proj_in.",
        "time_embedder.",
        "proj_out.",
        "audio_proj_out.",
    )
    return torch.float32 if name.startswith(fp32_prefixes) else torch.bfloat16


def _load_transformer(
    model: MiniMaxH3Transformer,
    component: Path,
    device: torch.device,
) -> None:
    from ...loader.handles import weight_handle_materialization

    sources = _weight_map(component)
    targets = dict(model.named_parameters())
    with weight_handle_materialization():
        for name in targets:
            try:
                handle = sources[name]
            except KeyError as error:
                raise KeyError(
                    f"FastH3 transformer is missing checkpoint tensor {name!r}"
                ) from error
            tensor = handle.full().to(
                device=device,
                dtype=_transformer_dtype(name),
                non_blocking=False,
            )
            _set_parameter(model, name, tensor)
    gate_names = {
        f"transformer_blocks.{index}.attn.to_gate_compress.weight"
        for index in range(model.config.layers)
    }
    if not gate_names <= targets.keys():
        raise RuntimeError("FastH3 transformer did not materialize all trained VSA gates")


def _encoder_shard(
    name: str,
    handle: Any,
    mesh: DeviceMesh,
) -> torch.Tensor:
    size, rank = mesh.size("tp"), mesh.coord("tp")
    if name == "language_model.embed_tokens.weight":
        rows = handle.shape[0] // size
        return handle.narrow(0, rank * rows, rows)
    if name.endswith(
        (
            ".q_proj.weight",
            ".k_proj.weight",
            ".v_proj.weight",
            ".gate_proj.weight",
            ".up_proj.weight",
        )
    ):
        rows = handle.shape[0] // size
        return handle.narrow(0, rank * rows, rows)
    if name.endswith((".o_proj.weight", ".down_proj.weight")):
        columns = handle.shape[1] // size
        return handle.narrow(1, rank * columns, columns)
    return handle.full()


def _load_encoder(
    model: MiniMaxH3TextEncoder,
    component: Path,
    device: torch.device,
) -> None:
    from ...loader.handles import weight_handle_materialization

    sources = _weight_map(component)
    targets = dict(model.named_parameters())
    with weight_handle_materialization():
        for name in targets:
            source_name = f"model.{name}"
            try:
                handle = sources[source_name]
            except KeyError as error:
                raise KeyError(
                    f"H3 text encoder is missing checkpoint tensor {source_name!r}"
                ) from error
            tensor = _encoder_shard(source_name[6:], handle, model.mesh)
            _set_parameter(
                model,
                name,
                tensor.to(device=device, dtype=torch.bfloat16, non_blocking=False),
            )


def load_h3_components(
    checkpoint_path: str,
    mesh: DeviceMesh,
    layout: Any,
    *,
    cache_dir: str | None = None,
    revision: str | None = None,
    attention_mode: Literal["sparse_kernel", "sparse_oracle", "dense_oracle"] = "sparse_kernel",
) -> H3Components:
    checkpoint = resolve_h3_checkpoint(
        checkpoint_path,
        cache_dir=cache_dir,
        revision=revision,
    )
    _require_checkpoint_geometry(checkpoint)
    transformer = MiniMaxH3Transformer(
        mesh,
        layout,
        parameter_device="meta",
        attention_mode=attention_mode,
    )
    encoder = MiniMaxH3TextEncoder(mesh, parameter_device="meta")
    _load_transformer(transformer, checkpoint.root / "transformer", mesh.local_device)
    _load_encoder(encoder, checkpoint.root / "text_encoder", mesh.local_device)

    if mesh.coord("tp") == 0:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint.root,
            subfolder="tokenizer",
            local_files_only=True,
            trust_remote_code=False,
        )
    else:
        tokenizer = None
    video_vae = MiniMaxH3VideoVAE.from_pretrained(
        str(checkpoint.root), device=mesh.local_device, local_files_only=True
    )
    audio_vae = (
        MiniMaxH3AudioVAE.from_pretrained(
            str(checkpoint.root), device=mesh.local_device, local_files_only=True
        )
        if mesh.coord("sp") == 0
        else None
    )
    return H3Components(
        checkpoint=checkpoint,
        transformer=transformer,
        encoder=encoder,
        video_vae=video_vae,
        audio_vae=audio_vae,
        tokenizer=tokenizer,
    )
