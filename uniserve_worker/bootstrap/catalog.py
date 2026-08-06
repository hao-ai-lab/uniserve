"""Explicit model construction data owned by the worker composition root."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from torch import nn
from transformers import AutoTokenizer

from ..foundation.errors import capability_mismatch, invalid_descriptor
from ..models.bagel import BagelConfig, BagelForUnifiedGeneration, _BagelGraph
from ..models.qwen3 import Qwen3ForCausalLM
from ..models.sensenova.config import NeoChatConfig
from ..models.sensenova.model import _MODEL_CODE_VERSION, NEOChatModel
from ..spec import (
    EncoderResourcePolicy,
    KvBlockResourcePolicy,
    LatentTokens,
    ModelLoadScope,
    PerBranch,
    ResourcePlan,
)

__all__ = ["Catalog", "CatalogEntry", "CheckpointFormat", "MODEL_CATALOG"]


class CheckpointFormat(StrEnum):
    STREAM = "stream"
    COMPOSITE = "composite"
    NATIVE = "native"


class DimensionTransform(StrEnum):
    IDENTITY = "identity"
    SQUARE_ROOT = "square_root"


@dataclass(frozen=True, slots=True)
class ConfigDimension:
    tensor: str
    axis: int
    field: str
    transform: DimensionTransform = DimensionTransform.IDENTITY

    def __post_init__(self) -> None:
        if not self.tensor or not self.field:
            raise invalid_descriptor("configuration dimension bindings must be named")


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One startup-only architecture binding and deployment declaration."""

    architecture: str
    model_class: type[nn.Module]
    checkpoint: CheckpointFormat
    resources: ResourcePlan
    config_class: type[Any] | None = None
    graph_class: type[nn.Module] | None = None
    tokenizer_class: type[Any] | None = None
    tokenizer_use_fast: bool = False
    tokenizer_special_tokens: tuple[tuple[str, str], ...] = ()
    minimum_code_version_key: str | None = None
    code_version: str | None = None
    serving_dtype: str = "bfloat16"
    scopes: tuple[ModelLoadScope, ...] = (ModelLoadScope.WHOLE,)
    required_files: tuple[str, ...] = ()
    alternative_files: tuple[str, ...] = ()
    config_files: tuple[tuple[str, str], ...] = ()
    config_dimensions: tuple[ConfigDimension, ...] = ()

    def __post_init__(self) -> None:
        if not self.architecture:
            raise invalid_descriptor("catalog entries require a stable architecture name")
        if not issubclass(self.model_class, nn.Module):
            raise invalid_descriptor("catalog model classes must inherit torch.nn.Module")
        if self.checkpoint is CheckpointFormat.COMPOSITE and (
            self.config_class is None or self.graph_class is None
        ):
            raise invalid_descriptor("composite catalog entries require config and graph classes")
        if self.checkpoint is CheckpointFormat.NATIVE and (
            self.config_class is None or self.tokenizer_class is None
        ):
            raise invalid_descriptor("native catalog entries require config and tokenizer classes")
        if not self.scopes:
            raise invalid_descriptor("catalog entries require at least one materialization scope")
        config_fields = tuple(field for field, _file in self.config_files)
        if len(set(config_fields)) != len(config_fields) or any(
            not field or not file for field, file in self.config_files
        ):
            raise invalid_descriptor("catalog config file declarations must be unique and named")

    def recognizes(self, root: Path) -> bool:
        if self.required_files and any(not (root / name).exists() for name in self.required_files):
            return False
        if self.alternative_files and not any(
            (root / name).exists() for name in self.alternative_files
        ):
            return False
        return bool(self.required_files or self.alternative_files)


class Catalog:
    """Immutable architecture-to-entry mapping used only during startup."""

    def __init__(self, entries: tuple[CatalogEntry, ...]) -> None:
        self._entries = entries
        by_name: dict[str, CatalogEntry] = {}
        for entry in entries:
            if entry.architecture in by_name:
                raise invalid_descriptor(
                    f"duplicate catalog architecture {entry.architecture!r}"
                )
            by_name[entry.architecture] = entry
        self._by_name = by_name

    def resolve(self, architectures: list[str] | tuple[str, ...]) -> CatalogEntry:
        for architecture in architectures:
            entry = self._by_name.get(str(architecture))
            if entry is not None:
                return entry
        known = ", ".join(sorted(self._by_name)) or "<none>"
        raise capability_mismatch(
            f"no model catalog entry for architectures {architectures!r}; known: {known}"
        )

    def detect_architectures(self, model_path: str | Path) -> list[str]:
        root = Path(model_path)
        return [entry.architecture for entry in self._entries if entry.recognizes(root)]


MODEL_CATALOG = Catalog(
    (
        CatalogEntry(
            architecture="Qwen3ForCausalLM",
            model_class=Qwen3ForCausalLM,
            checkpoint=CheckpointFormat.STREAM,
            resources=ResourcePlan(kv_block=KvBlockResourcePolicy.PER_BLOCK),
        ),
        CatalogEntry(
            architecture="BagelForUnifiedGeneration",
            model_class=BagelForUnifiedGeneration,
            checkpoint=CheckpointFormat.COMPOSITE,
            resources=ResourcePlan(
                kv_block=KvBlockResourcePolicy.PER_BLOCK,
                encoder_output=EncoderResourcePolicy.PER_HANDLE,
                image_latent=LatentTokens(downsample=16),
                scratch=PerBranch(fixed_tokens=65536, mirror_kv=True),
            ),
            config_class=BagelConfig,
            graph_class=_BagelGraph,
            required_files=("ae.safetensors",),
            alternative_files=("ema.safetensors", "model.safetensors"),
            config_files=(
                ("llm_config", "llm_config.json"),
                ("vit_config", "vit_config.json"),
                ("vae_config", "vae_config.json"),
            ),
            config_dimensions=(
                ConfigDimension(
                    tensor="latent_pos_embed.pos_embed",
                    axis=0,
                    field="max_latent_size",
                    transform=DimensionTransform.SQUARE_ROOT,
                ),
            ),
        ),
        CatalogEntry(
            architecture="NEOChatModel",
            model_class=NEOChatModel,
            checkpoint=CheckpointFormat.NATIVE,
            resources=ResourcePlan(
                kv_block=KvBlockResourcePolicy.PER_BLOCK,
                encoder_output=EncoderResourcePolicy.PER_HANDLE,
                image_latent=LatentTokens(downsample=16),
                scratch=PerBranch(
                    minimum_blocks=8,
                    mirror_kv=True,
                    latent_copies=4,
                    tower_copy=True,
                ),
            ),
            scopes=(
                ModelLoadScope.WHOLE,
                ModelLoadScope.UNDERSTANDING,
                ModelLoadScope.GENERATION,
            ),
            config_class=NeoChatConfig,
            tokenizer_class=AutoTokenizer,
            minimum_code_version_key="uniserve_sensenova_min_version",
            code_version=_MODEL_CODE_VERSION,
        ),
    )
)
