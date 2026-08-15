"""Architecture registry for worker model construction."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from ..foundation.errors import capability_mismatch, invalid_descriptor
from ..models.bagel import BagelForConditionalGeneration
from ..models.qwen3 import Qwen3ForCausalLM
from ..models.sensenova.model import NEOChatModel
from .plan import ModelLoadScope

__all__ = ["CatalogEntry", "resolve_catalog_entry"]


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One configured architecture and its supported deployment scopes."""

    architecture: str
    model_class: type[nn.Module]
    scopes: tuple[ModelLoadScope, ...] = (ModelLoadScope.WHOLE,)

    def __post_init__(self) -> None:
        if not self.architecture:
            raise invalid_descriptor("catalog entries require a stable architecture name")
        if not issubclass(self.model_class, nn.Module):
            raise invalid_descriptor("catalog model classes must inherit torch.nn.Module")
        if not self.scopes:
            raise invalid_descriptor("catalog entries require at least one materialization scope")


QWEN3_ENTRY = CatalogEntry(
    architecture="Qwen3ForCausalLM",
    model_class=Qwen3ForCausalLM,
)
BAGEL_ENTRY = CatalogEntry(
    architecture="BagelForConditionalGeneration",
    model_class=BagelForConditionalGeneration,
)
SENSENOVA_ENTRY = CatalogEntry(
    architecture="NEOChatModel",
    model_class=NEOChatModel,
    scopes=(
        ModelLoadScope.WHOLE,
        ModelLoadScope.UNDERSTANDING,
        ModelLoadScope.GENERATION,
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
    raise capability_mismatch(
        "configured checkpoint must declare exactly one architecture from "
        "Qwen3ForCausalLM, BagelForConditionalGeneration, or NEOChatModel; "
        f"found {architectures!r}"
    )
