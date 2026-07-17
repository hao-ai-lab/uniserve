"""Unified device snapshot for one scheduler forward.

:class:`ForwardBatch` is the single GPU-facing batch type for every
:class:`~.forward_mode.ForwardMode`. The runtime builder
(``runtime.forward_batch_builder``) stages text, denoise, encode, and commit
groups into this type.

The batch carries indices, geometry tensors, and small typed mode sub-blocks —
never the residency pools themselves. Physical KV, latent, scratch, and encoder
buffers are resolved through the published
:class:`~.forward_context.ForwardContext`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    import torch

    from .attention_plan import AttentionPlanBase
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
    """System-built device snapshot of one scheduler op group.

    The explicit argument into model and graph-runner entry points. Carries:

    - identity and shape (``forward_mode``, ``req_ids``, per-op modes for
      ``MIXED``)
    - text-core tensors when the group is text-shaped (``input_ids``,
      ``positions``, extend/decode geometry); ``None`` for pure-gen groups
    - residency *indices* only (``block_table``, ``out_cache_loc``, and the
      handles inside :class:`DenoiseInputs` / :class:`EncodeInputs` /
      :class:`CommitInputs`) — never the pools or KV/latent bytes
    - modality and attention-regime metadata (``is_gen``, ``segments``)
    - the per-forward :class:`~.attention_plan.AttentionPlanBase` attached by the
      system plan builder (``None`` for dense vision attention)

    Pool storage stays in the residency layer; shared layers resolve physical
    buffers from :class:`~.forward_context.ForwardContext`. Mutable so the
    metadata builder and CUDA-graph runner can attach or refresh plan state in
    place, including rewriting dynamic summary fields on each graph replay.
    """

    # identity / shape
    forward_mode: "ForwardMode"
    req_ids: tuple[int, ...]
    op_modes: tuple["ForwardMode", ...] = ()  # per-op mode (MIXED carries het. modes)
    ops: tuple[Mapping[str, Any], ...] = ()
    device: "torch.device | None" = None  # resolves gen_device / tower placement

    # text core (None for pure-gen groups)
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

    # KV residency, by index (filled from host-leased block_ids)
    block_table: "torch.Tensor | None" = None  # [batch, max_blocks] logical->physical
    out_cache_loc: "torch.Tensor | None" = None  # per-token destination slot (write side)
    cache_seqlens: "torch.Tensor | None" = None

    # modality / attention regime
    is_gen: "torch.Tensor | None" = None  # per-token modality mask (MoT routing)
    segments: tuple[SegmentSpec, ...] = ()  # causal vs bidirectional spans, branch ids

    # mode sub-blocks (None unless the mode needs them)
    denoise: DenoiseInputs | None = None
    encode: EncodeInputs | None = None
    commit: CommitInputs | None = None

    # sampling / spec
    sampling: Any = None
    return_all_logits: bool = False

    # per-forward attention plan (system-attached; None for dense vision)
    attn_plan: "AttentionPlanBase | None" = None

    @property
    def mode(self) -> "ForwardMode":
        """Alias for :attr:`forward_mode`."""
        return self.forward_mode

    @property
    def batch_size(self) -> int:
        return len(self.req_ids)

    @property
    def has_padding(self) -> bool:
        return int(self.padded_num_tokens) > int(self.num_token_non_padded)
