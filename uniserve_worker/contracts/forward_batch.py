"""The unified GPU-snapshot ``ForwardBatch`` for every forward mode.

This is the UniServe analog of SGLang's ``ForwardBatch.init_new``: the single
device snapshot the *system* builds for text (extend/decode/verify), denoise,
encode, and commit. It carries **indices and small typed sub-blocks, never the
pools themselves** — the KV/latent/scratch/encoder residency is addressed by the
host-leased handle (``block_table``/``latent_handle``/``out_handle``), resolved
to physical buffers by the worker-owned ``ResidencyManager`` behind the published
``ForwardContext``. Forward batches therefore carry descriptor indices only; pool
storage stays in the residency layer.

One builder (``runtime.forward_batch_builder``) stages every mode's tensors
into this type.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    import torch

    from .forward_mode import ForwardMode

__all__ = [
    "SegmentSpec",
    "BranchSpec",
    "CfgPlan",
    "VisiblePolicy",
    "KvSource",
    "CfgRenorm",
    "CfgRecipeKind",
    "DenoiseInputs",
    "EncodeInputs",
    "CommitInputs",
    "ForwardBatch",
]


class VisiblePolicy(StrEnum):
    CAUSAL = "causal"
    BIDIRECTIONAL = "bidirectional"
    PREFIX = "prefix"


class KvSource(StrEnum):
    PAGED = "paged"
    SCRATCH = "scratch"


class CfgRenorm(StrEnum):
    NONE = "none"
    GLOBAL = "global"
    PER_SAMPLE = "per_sample"
    PER_CHANNEL = "per_channel"
    MATCH_CFG = "match_cfg"


class CfgRecipeKind(StrEnum):
    DEFAULT = "default"
    IMAGE_OVER_TEXT = "image_over_text"


def _coerce_enum(enum_type: type[StrEnum], value: Any, field: str) -> StrEnum:
    try:
        return value if isinstance(value, enum_type) else enum_type(str(value))
    except ValueError as exc:
        raise ValueError(f"unknown {field} {value!r}") from exc


@dataclass(frozen=True)
class SegmentSpec:
    """A contiguous token span with a homogeneous attention regime.

    Carries the per-segment attention policy (causal text vs bidirectional
    image-latent span) and the CFG branch id so the system metadata builder can
    drive segment-batched / mixed und+gen attention from indices the model never
    has to reconstruct.
    """

    start: int
    length: int
    visible_policy: VisiblePolicy = VisiblePolicy.CAUSAL
    branch_id: int = 0
    is_gen: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "visible_policy",
            _coerce_enum(VisiblePolicy, self.visible_policy, "visible_policy"),
        )


@dataclass(frozen=True)
class BranchSpec:
    """One CFG branch and which system pool sources its KV.

    ``kv_source`` selects the residency pool (``"paged"`` = the shared
    ``KvPool``; ``"scratch"`` = the per-branch uncond ``ScratchKvPool``);
    ``kv_handle`` is the block-table id (paged) or scratch handle the system
    allocated for the branch. CFG KV is system-managed: each branch reads the pool
    named in ``kv_source`` via ``kv_handle`` rather than choosing scratch vs paged
    inside the model.
    """

    name: str  # "cond" | "text_uncond" | "img_uncond"
    kv_source: KvSource
    kv_handle: int
    kv_len: int
    position: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "kv_source", _coerce_enum(KvSource, self.kv_source, "kv_source"))


@dataclass(frozen=True)
class CfgPlan:
    """Guidance recipe for one denoise step (scales, interval, renorm)."""

    branches: tuple[BranchSpec, ...]
    text_scale: float = 1.0
    image_scale: float = 1.0
    interval: tuple[float, float] = (0.0, 1.0)
    renorm: CfgRenorm = CfgRenorm.NONE
    recipe: CfgRecipeKind = CfgRecipeKind.DEFAULT

    def __post_init__(self) -> None:
        object.__setattr__(self, "renorm", _coerce_enum(CfgRenorm, self.renorm, "renorm"))
        object.__setattr__(self, "recipe", _coerce_enum(CfgRecipeKind, self.recipe, "recipe"))


@dataclass(frozen=True)
class DenoiseInputs:
    """Generation sub-block for a ``DENOISE`` forward.

    The latent trajectory ``x_t`` is named by ``latent_handle`` (a
    ``LatentPool`` buffer the system owns); the schedule cursor is the
    ``step_index``/``total_steps`` pair; the per-step ODE endpoints are ``t`` /
    ``t_next``. The model never holds ``x_t`` or the schedule.
    """

    latent_handle: int
    step_index: int
    total_steps: int
    t: "torch.Tensor | None" = None
    t_next: "torch.Tensor | None" = None
    grid_hw: tuple[int, int] = (0, 0)
    branches: tuple[BranchSpec, ...] = ()
    cfg: CfgPlan | None = None
    rng_handle: int = 0


@dataclass(frozen=True)
class EncodeInputs:
    """Understanding sub-block for an ``ENCODE`` forward (vit/vae encode).

    The embeddings land in the ``EncoderCache`` slot named by ``out_handle``,
    keyed by ``mm_hash`` for cross-request reuse; the system, not the model,
    owns the budget and eviction.
    """

    kind: str  # "vit_encode" | "vae_encode"
    out_handle: int
    mm_hash: int | None = None
    cond_pos: int = 0


@dataclass(frozen=True)
class CommitInputs:
    """Commit sub-block for a ``COMMIT`` forward (VAE-decode the trajectory).

    ``fold_back`` requests re-embedding the generated image into the text KV
    (interleaved generation); the system owns the residency on both sides.
    """

    latent_handle: int
    fold_back: bool = False


@dataclass
class ForwardBatch:
    """The system-built GPU snapshot of one scheduler op group.

    Holds indices and small typed sub-blocks for every mode; never holds pools
    or KV/latent/logit bytes. Mutable (not frozen) so the metadata builder and
    the CUDA-graph runner can attach/refresh per-forward plan state in place, the
    way the captured graph rewrites its dynamic summary fields each replay.
    """

    # --- identity / shape ---
    forward_mode: "ForwardMode"
    req_ids: tuple[int, ...]
    op_modes: tuple["ForwardMode", ...] = ()  # per-op mode (MIXED carries het. modes)
    ops: tuple[Mapping[str, Any], ...] = ()
    device: "torch.device | None" = None  # resolves gen_device / tower placement

    # --- text core (SGLang-parity; None for pure-gen groups) ---
    input_ids: "torch.Tensor | None" = None
    positions: "torch.Tensor | None" = None
    seq_lens: "torch.Tensor | None" = None
    extend_seq_lens: "torch.Tensor | None" = None
    extend_start_loc: "torch.Tensor | None" = None
    extend_prefix_lens: "torch.Tensor | None" = None
    last_token_indices: "torch.Tensor | None" = None
    num_token_non_padded: int = 0
    padded_num_tokens: int = 0
    spec_token_ids: tuple[tuple[int, ...], ...] = ()

    # --- KV residency, by index (the system fills these from block_ids) ---
    block_table: "torch.Tensor | None" = None  # [batch, max_blocks] logical->physical
    out_cache_loc: "torch.Tensor | None" = None  # per-token destination slot (write side)
    cache_seqlens: "torch.Tensor | None" = None

    # --- modality / attention regime (per token / per segment) ---
    is_gen: "torch.Tensor | None" = None  # per-token modality mask (MoT routing)
    segments: tuple[SegmentSpec, ...] = ()  # causal vs bidirectional spans, branch ids

    # --- generation sub-blocks (None unless the mode needs them) ---
    denoise: DenoiseInputs | None = None
    encode: EncodeInputs | None = None
    commit: CommitInputs | None = None

    # --- sampling / spec (deferred-handle-friendly) ---
    sampling: Any = None

    # --- per-forward attention plan (attached by AttentionBackend) ---
    # The system metadata builder stashes the per-forward kernel plan here (the
    # paged request-cache view + cu_seqlens / decode write-locations); the model
    # never builds or threads it. ``None`` for dense (vision) attention.
    attn_metadata: Any = None

    @property
    def mode(self) -> "ForwardMode":
        """Back-compat alias: the single forward mode of this snapshot."""
        return self.forward_mode

    @property
    def batch_size(self) -> int:
        return len(self.req_ids)

    @property
    def has_padding(self) -> bool:
        return int(self.padded_num_tokens) > int(self.num_token_non_padded)
