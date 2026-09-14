"""Concrete architecture readers and checkpoint sources for model loading."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from uniserve.loading.source import WeightSourceConfig, WeightSourceSet
from uniserve.model.model import Model
from uniserve.nn.diffusion.schedule import DiffusionSchedule
from uniserve.nn.layer import LayerConfig
from uniserve.nn.quant.config import LinearPrecision, resolve_component_precisions
from uniserve_models.bagel import BagelForConditionalGeneration
from uniserve_models.metadata import bagel_config, h3_config, sensenova_config
from uniserve_models.minimax_h3 import MiniMaxH3Model
from uniserve_models.minimax_h3.config import (
    PRECISION_PRESETS,
    PRECISION_SHORTHANDS,
    SUPPORTED_PRECISIONS,
)
from uniserve_models.minimax_h3.weights import configure_layers as h3_layers
from uniserve_models.processing import (
    SENSENOVA_PROMPT,
    FlowPrompt,
    ImageProcessor,
    bagel_processor,
    sensenova_processor,
)
from uniserve_models.qwen3 import Qwen3ForCausalLM
from uniserve_models.qwen3 import read_config as read_qwen_config
from uniserve_models.sensenova.model import NEOChatModel

__all__ = ["CatalogEntry", "resolve_catalog_entry"]


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """Architecture discovery, checkpoint sources, and resolved precision policy."""

    architecture: str
    model_class: type[Model]
    prepare_config: Callable[[dict[str, Any], Path, tuple[WeightSourceSet, ...]], Any]
    sidecars: tuple[str, ...] = ("config.json",)
    sources: tuple[WeightSourceConfig, ...] = (WeightSourceConfig(),)
    component_precisions: Callable[[Mapping[str, object]], Mapping[str, LinearPrecision]] | None = (
        None
    )
    configure_layers: (
        Callable[
            [Mapping[str, LayerConfig], Mapping[str, LinearPrecision]], Mapping[str, LayerConfig]
        ]
        | None
    ) = None
    create_schedule: Callable[[Any, torch.device], DiffusionSchedule] | None = None
    image_processor: Callable[[Any], ImageProcessor] | None = None
    flow_prompt: FlowPrompt | None = None
    tokenizer: bool = False
    component_paths: tuple[tuple[str, str], ...] = (("model", ""),)

    def __post_init__(self) -> None:
        """Validate architecture identity and the required source manifest."""

        if not self.architecture:
            raise ValueError("catalog entries require a stable architecture name")
        if not issubclass(self.model_class, Model):
            raise ValueError("catalog model classes must implement Model")
        if not self.sidecars or any(not value for value in self.sidecars):
            raise ValueError("catalog entries require a non-empty sidecar manifest")


QWEN3_ENTRY = CatalogEntry(
    architecture="Qwen3ForCausalLM",
    model_class=Qwen3ForCausalLM,
    prepare_config=lambda config, root, sources: read_qwen_config(config),
)
BAGEL_ENTRY = CatalogEntry(
    architecture="BagelForConditionalGeneration",
    model_class=BagelForConditionalGeneration,
    prepare_config=bagel_config,
    image_processor=bagel_processor,
    sidecars=("config.json", "llm_config.json", "vit_config.json", "vae_config.json"),
    sources=(
        WeightSourceConfig(filenames=("ema.safetensors", "model.safetensors")),
        WeightSourceConfig("autoencoder", filenames=("ae.safetensors",)),
    ),
)
SENSENOVA_ENTRY = CatalogEntry(
    architecture="NEOChatModel",
    model_class=NEOChatModel,
    prepare_config=sensenova_config,
    image_processor=sensenova_processor,
    flow_prompt=SENSENOVA_PROMPT,
    tokenizer=True,
    sidecars=(
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
        "chat_template.json",
        "chat_template.jinja",
    ),
)
MINIMAX_H3_ENTRY = CatalogEntry(
    component_paths=(
        ("text_encoder", "text_encoder"),
        ("denoiser", "denoiser"),
        ("video_decoder", "video_decoder"),
        ("audio_decoder", "audio_decoder"),
        ("output", "video_output"),
    ),
    architecture="MiniMaxH3Transformer3DModel",
    model_class=MiniMaxH3Model,
    configure_layers=h3_layers,
    prepare_config=h3_config,
    component_precisions=partial(
        resolve_component_precisions,
        supported=SUPPORTED_PRECISIONS,
        presets=PRECISION_PRESETS,
        shorthands=PRECISION_SHORTHANDS,
        default_mode="balanced",
    ),
    create_schedule=lambda config, device: DiffusionSchedule.build(
        config.diffusion.ladder,
        (config.diffusion.video_shift, config.diffusion.audio_shift),
        scale=config.diffusion.time_scale,
        device=device,
    ),
    sources=(
        WeightSourceConfig("denoiser", "transformer", component="denoiser"),
        WeightSourceConfig("text_encoder", "text_encoder", component="text_encoder"),
        WeightSourceConfig("video_decoder", "vae", component="video_decoder"),
        WeightSourceConfig("audio_decoder", "audio_vae", component="audio_decoder"),
    ),
    sidecars=(
        "modular_model_index.json",
        "fastvideo_inference.json",
        "transformer/config.json",
        "text_encoder/config.json",
        "vae/config.json",
        "audio_vae/config.json",
        "scheduler/scheduler_config.json",
        "audio_scheduler/scheduler_config.json",
    ),
)


def resolve_catalog_entry(architectures: list[str] | tuple[str, ...]) -> CatalogEntry:
    """Resolve one exact configured checkpoint architecture."""

    match tuple(str(architecture) for architecture in architectures):
        case ("Qwen3ForCausalLM",):
            return QWEN3_ENTRY
        case ("BagelForConditionalGeneration",):
            return BAGEL_ENTRY
        case ("NEOChatModel",):
            return SENSENOVA_ENTRY
        case ("MiniMaxH3Transformer3DModel",):
            return MINIMAX_H3_ENTRY
    raise ValueError(
        "configured checkpoint must declare exactly one architecture from "
        "Qwen3ForCausalLM, BagelForConditionalGeneration, NEOChatModel, or "
        "MiniMaxH3Transformer3DModel; "
        f"found {architectures!r}"
    )


def entry_paths(model_class: type[Model], config: Any) -> dict[str, str]:
    """Map catalog entry names to actual numerical module paths.

    Unregistered Python models expose their declared module paths directly;
    the root module uses the ordinary serving entry name ``model``.
    """

    for entry in (QWEN3_ENTRY, BAGEL_ENTRY, SENSENOVA_ENTRY, MINIMAX_H3_ENTRY):
        if entry.model_class is model_class:
            return dict(entry.component_paths)
    return {
        call.component or "model": call.component for call in model_class.component_calls(config)
    }
