"""Static model-declared resource-residency rules (torch-free dataclasses).

The runner evaluates these rules against each wire op; models declare what
resource classes they need but never account units themselves. Extracted from
the former ``core.model`` so the pure resource-rule data lives apart from the
protocol abstractions and the concrete ``UniModelBase`` glue.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "AdapterResourcePolicy",
    "EncoderResourcePolicy",
    "KvBlockResourcePolicy",
    "LatentTokens",
    "PerBranch",
    "ResourcePlan",
    "CapsDescriptor",
]


class KvBlockResourcePolicy(StrEnum):
    PER_BLOCK = "per_block"


class EncoderResourcePolicy(StrEnum):
    PER_HANDLE = "per_handle"


class AdapterResourcePolicy(StrEnum):
    PER_ADAPTER = "per_adapter"


@dataclass(frozen=True)
class LatentTokens:
    """Resource rule for image latent residency."""

    downsample: int = 16


@dataclass(frozen=True)
class PerBranch:
    """Resource rule for one scratch reservation per active CFG branch."""

    minimum: int = 1


@dataclass(frozen=True)
class ResourcePlan:
    """Static model-declared resource residency plan.

    The runner evaluates the rules against each wire op; models declare what
    resource classes they need but never account units themselves.
    """

    kv_block: KvBlockResourcePolicy | None = KvBlockResourcePolicy.PER_BLOCK
    image_latent: LatentTokens | None = None
    scratch: PerBranch | None = None
    encoder_output: EncoderResourcePolicy | None = None
    adapter: AdapterResourcePolicy | None = None

    def classes(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.kv_block:
            out.append("kv_block")
        if self.encoder_output:
            out.append("encoder_output")
        if self.image_latent is not None:
            out.append("image_latent")
        if self.scratch is not None:
            out.append("scratch")
        if self.adapter:
            out.append("adapter")
        return tuple(out)

@dataclass(frozen=True)
class CapsDescriptor:
    """Per-model scalars the shared :class:`UniModelBase.caps` assembles into a
    :class:`~uniserve_worker.contracts.caps.Caps`.

    Each model fills in only the numbers that vary between models (sizing,
    latent geometry, KV bytes, execution constraints, optional fields); the
    ``Caps`` assembly and the contract glue (``supported_ops`` /
    ``supported_controls`` / ``adapter_mode`` / ``resource_plan``) live once in
    the base.
    """

    block_size: int
    num_blocks: int
    num_layers: int
    scratch_capacity_tokens: int
    max_latent_size: int
    latent_downsample: int
    bytes_per_token: int
    max_batch_ops: int
    attention_backend: str | None = None
    kv_dtype: str | None = None
    encoder_cache_budget: int | None = None
