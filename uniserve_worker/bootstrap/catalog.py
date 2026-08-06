"""Explicit model construction data owned by the worker composition root."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from torch import nn

from ..foundation.errors import capability_mismatch, invalid_descriptor
from ..models.bagel import BagelConfig, BagelForConditionalGeneration, _BagelGraph
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

__all__ = ["CatalogEntry", "CheckpointFormat", "resolve_catalog_entry"]


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
    minimum_code_version_key: str | None = None
    code_version: str | None = None
    serving_dtype: str = "bfloat16"
    scopes: tuple[ModelLoadScope, ...] = (ModelLoadScope.WHOLE,)
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
        if self.checkpoint is CheckpointFormat.NATIVE and self.config_class is None:
            raise invalid_descriptor("native catalog entries require a config class")
        if not self.scopes:
            raise invalid_descriptor("catalog entries require at least one materialization scope")
        config_fields = tuple(field for field, _file in self.config_files)
        if len(set(config_fields)) != len(config_fields) or any(
            not field or not file for field, file in self.config_files
        ):
            raise invalid_descriptor("catalog config file declarations must be unique and named")


QWEN3_ENTRY = CatalogEntry(
    architecture="Qwen3ForCausalLM",
    model_class=Qwen3ForCausalLM,
    checkpoint=CheckpointFormat.STREAM,
    resources=ResourcePlan(kv_block=KvBlockResourcePolicy.PER_BLOCK),
)
BAGEL_ENTRY = CatalogEntry(
    architecture="BagelForConditionalGeneration",
    model_class=BagelForConditionalGeneration,
    checkpoint=CheckpointFormat.COMPOSITE,
    resources=ResourcePlan(
        kv_block=KvBlockResourcePolicy.PER_BLOCK,
        encoder_output=EncoderResourcePolicy.PER_HANDLE,
        image_latent=LatentTokens(downsample=16),
        scratch=PerBranch(fixed_tokens=65536, mirror_kv=True),
    ),
    config_class=BagelConfig,
    graph_class=_BagelGraph,
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
)
SENSENOVA_ENTRY = CatalogEntry(
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
    minimum_code_version_key="uniserve_sensenova_min_version",
    code_version=_MODEL_CODE_VERSION,
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
    raise capability_mismatch(
        "configured checkpoint must declare exactly one architecture from "
        "Qwen3ForCausalLM, BagelForConditionalGeneration, or NEOChatModel; "
        f"found {architectures!r}"
    )
