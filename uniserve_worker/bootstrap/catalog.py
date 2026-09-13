"""Architecture registry for worker model construction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from ..foundation.errors import invalid_descriptor, unsupported_setup
from ..loader.source import WeightSourceConfig, WeightSourceSet
from ..modeling.model import Model
from ..models.bagel import BagelForConditionalGeneration
from ..models.minimax_h3 import MiniMaxH3Model
from ..models.minimax_h3.config import (
    FASTH3_LADDER,
    FASTH3_SHIFTS,
    FASTH3_TIME_SCALE,
    PRECISION_PRESETS,
    PRECISION_SHORTHANDS,
    SUPPORTED_PRECISIONS,
)
from ..models.qwen3 import Qwen3ForCausalLM
from ..models.sensenova.model import NEOChatModel
from ..nn.diffusion.schedule import DiffusionSchedule
from ..nn.quant.config import LinearPrecision, resolve_component_precisions
from .metadata import bagel_config, h3_config, sensenova_config

__all__ = ["CatalogEntry", "resolve_catalog_entry"]


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """Architecture discovery, checkpoint sources, and resolved precision policy."""

    architecture: str
    model_class: type[Model]
    sidecars: tuple[str, ...] = ("config.json",)
    sources: tuple[WeightSourceConfig, ...] = (WeightSourceConfig(),)
    minimum_cuda_capability: tuple[int, int] | None = None
    component_precisions: Callable[[Mapping[str, object]], Mapping[str, LinearPrecision]] | None = (
        None
    )
    create_schedule: Callable[[torch.device], DiffusionSchedule] | None = None
    prepare_config: Callable[[dict[str, Any], Path, tuple[WeightSourceSet, ...]], Any] | None = None
    tokenizer: bool = False

    def __post_init__(self) -> None:
        """Validate architecture identity and the required source manifest."""

        if not self.architecture:
            raise invalid_descriptor("catalog entries require a stable architecture name")
        if not issubclass(self.model_class, Model):
            raise invalid_descriptor("catalog model classes must implement Model")
        if not self.sidecars or any(not value for value in self.sidecars):
            raise invalid_descriptor("catalog entries require a non-empty sidecar manifest")


QWEN3_ENTRY = CatalogEntry(
    architecture="Qwen3ForCausalLM",
    model_class=Qwen3ForCausalLM,
)
BAGEL_ENTRY = CatalogEntry(
    architecture="BagelForConditionalGeneration",
    model_class=BagelForConditionalGeneration,
    prepare_config=bagel_config,
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
    architecture="MiniMaxH3Transformer3DModel",
    model_class=MiniMaxH3Model,
    prepare_config=h3_config,
    minimum_cuda_capability=(9, 0),
    component_precisions=partial(
        resolve_component_precisions,
        supported=SUPPORTED_PRECISIONS,
        presets=PRECISION_PRESETS,
        shorthands=PRECISION_SHORTHANDS,
        default_mode="balanced",
    ),
    create_schedule=lambda device: DiffusionSchedule.build(
        FASTH3_LADDER,
        FASTH3_SHIFTS,
        scale=FASTH3_TIME_SCALE,
        device=device,
    ),
    sources=(
        WeightSourceConfig("denoiser", "transformer", entry="denoiser"),
        WeightSourceConfig("text_encoder", "text_encoder", entry="text_encoder"),
        WeightSourceConfig("video_decoder", "vae", entry="video_decoder"),
        WeightSourceConfig("audio_decoder", "audio_vae", entry="audio_decoder"),
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
    raise unsupported_setup(
        "configured checkpoint must declare exactly one architecture from "
        "Qwen3ForCausalLM, BagelForConditionalGeneration, NEOChatModel, or "
        "MiniMaxH3Transformer3DModel; "
        f"found {architectures!r}"
    )
