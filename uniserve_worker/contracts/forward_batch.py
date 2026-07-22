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

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from ..foundation.errors import invalid_descriptor
from .forward_mode import ForwardMode, mode_for_op

if TYPE_CHECKING:
    from .attention_plan import AttnPlan
    from .batches import CommitBatch, DenoiseBatch, EncodeBatch, MixedBatch, TextBatch

__all__ = [
    "BatchPolicy",
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
    "ForwardModality",
    "ForwardSegmentClass",
    "KvWritePolicy",
    "ForwardOutputKind",
    "ForwardResultProjection",
    "ForwardPostprocessPolicy",
    "TextTokenSpanPlan",
    "CacheSpanPlan",
    "DenoiseRowPlan",
    "CommitRowPlan",
    "EncodeRowPlan",
    "ForwardRowPlan",
    "ForwardSegmentPlan",
    "ForwardOutputSlot",
    "ForwardShapeSummary",
    "ForwardExecutionOptions",
    "ForwardPlan",
    "EagerFallbackReason",
    "ForwardGraphPolicy",
    "EagerFallbackWarning",
    "StrictForwardGraphError",
    "CommitNeuralResult",
    "DenoiseBranchKey",
    "DenoisePostprocessEntry",
    "DenoiseVelocityResult",
    "EncodeResult",
    "GraphInfo",
    "ForwardResult",
    "TextLogitsResult",
    "TextPostprocessEntry",
    "coerce_forward_result",
]


@dataclass(frozen=True)
class BatchPolicy:
    """Limits and mode-order rules for runner-owned batch grouping."""

    max_batch_ops: int = 1
    supports_mixed_modes: bool = False
    mode_order: tuple[ForwardMode, ...] = (
        ForwardMode.ENCODE,
        ForwardMode.EXTEND,
        ForwardMode.DECODE,
        ForwardMode.VERIFY_DRAFT,
        ForwardMode.DENOISE,
        ForwardMode.COMMIT,
    )

    def __post_init__(self) -> None:
        if self.max_batch_ops < 1:
            raise invalid_descriptor("BatchPolicy.max_batch_ops must be at least 1")
        seen: set[ForwardMode] = set()
        for mode in self.mode_order:
            if not isinstance(mode, ForwardMode):
                raise invalid_descriptor("BatchPolicy.mode_order entries must be ForwardMode values")
            if mode in seen:
                raise invalid_descriptor("BatchPolicy.mode_order must not contain duplicates")
            seen.add(mode)

    def allows_group(self, modes: list[ForwardMode]) -> bool:
        if not modes or len(modes) > self.max_batch_ops:
            return False
        return self.supports_mixed_modes or all(mode == modes[0] for mode in modes)


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
    ``LatentStore`` buffer the system owns); the schedule cursor is the
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

    ``fold_back`` requests re-embedding the generated image for subsequent
    sequence operations; the system owns the residency on both sides.
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
    - the per-forward :data:`~.attention_plan.AttnPlan` attached by the
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
    attn_plan: "AttnPlan | None" = None

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

    @classmethod
    def from_ops(cls, ops: Sequence[Mapping[str, Any]]) -> "ForwardBatch":
        """Parse an op group into the canonical batch before device staging."""

        from .batches import BatchBase

        if not ops:
            raise invalid_descriptor("forward batch group must contain at least one op")
        parsed_ops: list[Mapping[str, Any]] = []
        modes: list[ForwardMode] = []
        for index, op in enumerate(ops):
            if not isinstance(op, Mapping):
                raise invalid_descriptor(f"forward batch op {index} must be a map")
            kind = op.get("kind")
            if not isinstance(kind, str):
                raise invalid_descriptor(f"forward batch op {index}.kind must be a string")
            parsed_ops.append(op)
            modes.append(mode_for_op(kind))
        op_tuple = tuple(parsed_ops)
        first = modes[0]
        mode = first if all(item is first for item in modes) else ForwardMode.MIXED
        return cls(
            forward_mode=mode,
            req_ids=BatchBase.req_ids_from_ops(op_tuple),
            op_modes=tuple(modes),
            ops=op_tuple,
        )

    def as_text(self, *, allow_mixed_text: bool = False) -> "TextBatch":
        from .batches import TextBatch

        return TextBatch.from_ops(
            self.mode,
            self.ops,
            op_modes=self.op_modes,
            allow_mixed_text=allow_mixed_text,
        )

    def as_denoise(self) -> "DenoiseBatch":
        from .batches import DenoiseBatch

        if self.mode is not ForwardMode.DENOISE:
            raise invalid_descriptor(f"batch mode {self.mode.value} is not denoise")
        return DenoiseBatch.from_ops(self.ops)

    def as_commit(self) -> "CommitBatch":
        from .batches import CommitBatch

        if self.mode is not ForwardMode.COMMIT:
            raise invalid_descriptor(f"batch mode {self.mode.value} is not commit")
        return CommitBatch.from_ops(self.ops)

    def as_encode(self) -> "EncodeBatch":
        from .batches import EncodeBatch

        if self.mode is not ForwardMode.ENCODE:
            raise invalid_descriptor(f"batch mode {self.mode.value} is not encode")
        return EncodeBatch.from_ops(self.ops)

    def as_mixed(self) -> "MixedBatch":
        from .batches import MixedBatch

        if self.mode is not ForwardMode.MIXED:
            raise invalid_descriptor(f"batch mode {self.mode.value} is not mixed")
        return MixedBatch.from_ops(self.ops, op_modes=self.op_modes)


# --- Host forward-plan values (canonical row/segment/output planning) ---

_TEXT_MODES = frozenset({ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.VERIFY_DRAFT})


class ForwardModality(StrEnum):
    TEXT = "text"
    GENERATION = "generation"


class ForwardSegmentClass(StrEnum):
    EXTEND = "extend"
    DECODE = "decode"
    DENOISE = "denoise"
    REENCODE = "reencode"
    COMMIT_INPUT = "commit_input"
    ENCODE_INPUT = "encode_input"


class KvWritePolicy(StrEnum):
    PERSISTENT = "persistent"
    TRANSIENT = "transient"
    NONE = "none"
    STAGED_PROMOTION = "staged_promotion"


class ForwardOutputKind(StrEnum):
    TEXT_TOKEN = "text_token"
    DENOISE_STEP = "denoise_step"
    COMMIT = "commit"
    ENCODE = "encode"
    COMBINED = "combined"


class ForwardResultProjection(StrEnum):
    LAST_TEXT_ROW = "last_text_row"
    DENOISE_BRANCHES = "denoise_branches"
    ENCODE_ROW = "encode_row"
    COMMIT_ROW = "commit_row"
    RUNTIME_OUTPUT = "runtime_output"


class ForwardPostprocessPolicy(StrEnum):
    SAMPLE = "sample"
    CFG_COMBINE = "cfg_combine"
    LATENT_UPDATE = "latent_update"
    COMMIT_DECODE = "commit_decode"
    ENCODE_PUBLISH = "encode_publish"
    NONE = "none"


@dataclass(frozen=True)
class TextTokenSpanPlan:
    token_ids: tuple[int, ...]
    position_start: int
    position_end: int
    token_source: str = "wire"
    last_token_only: bool = True

    @property
    def q_len(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class CacheSpanPlan:
    block_ids: tuple[int, ...] = ()
    base_len: int = 0
    append_len: int = 0
    pool_identity: str | None = None
    persistent: bool = True


@dataclass(frozen=True)
class DenoiseRowPlan:
    step_index: int
    total_steps: int
    branch_count: int
    branch_ids: tuple[str, ...]
    image_token_count: int
    latent_handle: int | None = None
    grid_hw: tuple[int, int] = (0, 0)
    cfg: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitRowPlan:
    latent_handle: int | None = None
    fold_back: bool = False
    image_token_count: int = 0


@dataclass(frozen=True)
class EncodeRowPlan:
    kind: str
    out_handle: int | None = None
    mm_hash: int | None = None
    num_tokens: int = 0


@dataclass(frozen=True)
class ForwardRowPlan:
    row_index: int
    original_index: int
    req_id: int
    op: Mapping[str, Any]
    mode: ForwardMode
    token_span: TextTokenSpanPlan | None = None
    cache_span: CacheSpanPlan | None = None
    denoise: DenoiseRowPlan | None = None
    commit: CommitRowPlan | None = None
    encode: EncodeRowPlan | None = None


@dataclass(frozen=True)
class ForwardSegmentPlan:
    segment_index: int
    row_index: int
    mode: ForwardMode
    modality: ForwardModality
    segment_class: ForwardSegmentClass
    q_len: int
    prefix_len: int = 0
    visible_policy: VisiblePolicy = VisiblePolicy.CAUSAL
    branch_id: int = 0
    position_source: str = "positions"
    kv_write_policy: KvWritePolicy = KvWritePolicy.PERSISTENT


@dataclass(frozen=True)
class ForwardOutputSlot:
    row_index: int
    req_id: int
    kind: ForwardOutputKind
    result_projection: ForwardResultProjection
    postprocess_policy: ForwardPostprocessPolicy


@dataclass(frozen=True)
class ForwardShapeSummary:
    forward_mode: ForwardMode
    op_modes: tuple[ForwardMode, ...]
    row_count: int
    token_count: int
    segment_count: int
    branch_count: int
    text_row_count: int
    denoise_row_count: int
    commit_row_count: int
    encode_row_count: int
    padded_token_count: int = 0
    padded_row_count: int = 0

    @classmethod
    def from_parts(
        cls,
        *,
        forward_mode: ForwardMode,
        rows: Sequence[ForwardRowPlan],
        segments: Sequence[ForwardSegmentPlan],
    ) -> "ForwardShapeSummary":
        token_count = sum(int(segment.q_len) for segment in segments)
        branch_count = sum(int(row.denoise.branch_count) for row in rows if row.denoise is not None)
        return cls(
            forward_mode=forward_mode,
            op_modes=tuple(row.mode for row in rows),
            row_count=len(rows),
            token_count=token_count,
            segment_count=len(segments),
            branch_count=branch_count,
            text_row_count=sum(1 for row in rows if row.mode in _TEXT_MODES),
            denoise_row_count=sum(1 for row in rows if row.mode is ForwardMode.DENOISE),
            commit_row_count=sum(1 for row in rows if row.mode is ForwardMode.COMMIT),
            encode_row_count=sum(1 for row in rows if row.mode is ForwardMode.ENCODE),
            padded_token_count=token_count,
            padded_row_count=len(rows),
        )


@dataclass(frozen=True, slots=True)
class ForwardExecutionOptions:
    """Per-call execution controls that are not part of a replayable plan."""

    defer_text_cpu_results: bool = False
    defer_sampling: bool = False


@dataclass(frozen=True)
class ForwardPlan:
    step_id: int | None
    rows: tuple[ForwardRowPlan, ...]
    segments: tuple[ForwardSegmentPlan, ...]
    output_slots: tuple[ForwardOutputSlot, ...]
    shape: ForwardShapeSummary
    graph_policy: "ForwardGraphPolicy | None" = None

    @property
    def forward_mode(self) -> ForwardMode:
        return self.shape.forward_mode

    @property
    def req_ids(self) -> tuple[int, ...]:
        return tuple(row.req_id for row in self.rows)

    @property
    def op_modes(self) -> tuple[ForwardMode, ...]:
        return self.shape.op_modes

    @property
    def ops(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(row.op for row in self.rows)

    def validate(self) -> None:
        if not self.rows:
            raise invalid_descriptor("forward plan must contain at least one row")
        if len(self.output_slots) != len(self.rows):
            raise invalid_descriptor("forward plan must contain one output slot per row")
        row_indices = tuple(row.row_index for row in self.rows)
        if row_indices != tuple(range(len(self.rows))):
            raise invalid_descriptor("forward plan row indices must be contiguous")
        for segment_index, segment in enumerate(self.segments):
            if segment.segment_index != segment_index:
                raise invalid_descriptor("forward plan segment indices must be contiguous")
            if segment.row_index < 0 or segment.row_index >= len(self.rows):
                raise invalid_descriptor("forward segment references an unknown row")
            if segment.q_len <= 0:
                raise invalid_descriptor("forward segment q_len must be positive")
        for slot in self.output_slots:
            if slot.row_index < 0 or slot.row_index >= len(self.rows):
                raise invalid_descriptor("forward output slot references an unknown row")


# --- Forward graph policy values ---


class EagerFallbackReason(StrEnum):
    GRAPH_DISABLED = "graph_disabled"
    GRAPH_MISS = "graph_miss"
    GRAPH_INELIGIBLE = "graph_ineligible"
    CAPTURE_FAILURE = "capture_failure"
    REPLAY_FAILURE = "replay_failure"
    BACKEND_INELIGIBLE = "backend_ineligible"
    SHAPE_UNSUPPORTED = "shape_unsupported"
    STRICT_MODE_DISABLED = "strict_mode_disabled"
    CUDA_UNAVAILABLE = "cuda_unavailable"


@dataclass(frozen=True)
class ForwardGraphPolicy:
    prefer_graph: bool = True
    strict: bool = True
    allow_capture: bool = True
    graph_selection_delegated: bool = False


@dataclass(frozen=True)
class EagerFallbackWarning:
    reason: EagerFallbackReason
    mode: ForwardMode
    op_modes: tuple[ForwardMode, ...]
    tokens: int
    rows: int
    padded_tokens: int
    padded_rows: int
    topology_id: str | None = None
    capacity_key: Any | None = None
    backend: str | None = None


class StrictForwardGraphError(RuntimeError):
    def __init__(self, warning: EagerFallbackWarning) -> None:
        super().__init__(
            "strict forward graph policy rejected eager execution: "
            f"reason={warning.reason.value} mode={warning.mode.value}"
        )
        self.warning = warning


# --- Typed neural forward results ---


@dataclass(frozen=True)
class DenoiseBranchKey:
    row_index: int
    branch_id: int


@dataclass(frozen=True)
class DenoisePostprocessEntry:
    row_index: int
    req_id: int
    step_index: int
    total_steps: int
    branch_names: tuple[Any, ...]
    latent: torch.Tensor
    t: torch.Tensor
    t_next: torch.Tensor
    combine_velocity: Callable[[Mapping[Any, torch.Tensor]], torch.Tensor]
    accept_update: Callable[[torch.Tensor], None]


@dataclass(frozen=True)
class TextPostprocessEntry:
    row_index: int
    req_id: int
    logits_index: int
    position_id: int
    kv_new_length: int
    last_input_token: int
    program_state: Any | None = None
    persistent_cache: Any | None = None
    staged_cache: Any | None = None
    kv_promotion: Any | None = None
    num_layers: int = 0
    mark_staging_advanced: Callable[[Any, Any, int], None] | None = None


@dataclass(frozen=True)
class GraphInfo:
    path: str
    capacity: Any


@dataclass(frozen=True)
class TextLogitsResult:
    logits: torch.Tensor
    row_indices: tuple[int, ...]


@dataclass(frozen=True)
class DenoiseVelocityResult:
    velocities: Mapping[DenoiseBranchKey, torch.Tensor]


@dataclass(frozen=True)
class EncodeResult:
    outputs: Mapping[int, torch.Tensor]


@dataclass(frozen=True)
class CommitNeuralResult:
    outputs: Mapping[int, torch.Tensor]


@dataclass
class ForwardResult:
    text_logits: torch.Tensor | None = None
    text_postprocess: tuple[TextPostprocessEntry, ...] | None = None
    denoise_velocities: Mapping[DenoiseBranchKey, torch.Tensor] | None = None
    denoise_updates: Mapping[int, DenoisePostprocessEntry] | None = None
    encode_outputs: Mapping[int, Any] | None = None
    commit_outputs: Mapping[int, Any] | None = None
    hidden: torch.Tensor | None = None
    graph: GraphInfo | None = None
    text_cuda_ready_start_event: Any | None = None
    runtime_outputs: tuple[Any, ...] | None = None

    def validate_for_plan(self, plan: "ForwardPlan") -> None:
        plan.validate()
        if self.runtime_outputs is not None:
            if len(self.runtime_outputs) != len(plan.output_slots):
                raise invalid_descriptor("forward runtime output count must match output slots")
            return
        text_slots = [
            slot for slot in plan.output_slots if slot.kind is ForwardOutputKind.TEXT_TOKEN
        ]
        if text_slots:
            if not isinstance(self.text_logits, torch.Tensor):
                raise invalid_descriptor("forward result is missing text logits")
            logits_rows = self.text_logits.reshape(-1, self.text_logits.shape[-1])
            if self.text_logits.ndim == 1:
                if len(text_slots) != 1:
                    raise invalid_descriptor(
                        "single text logits row cannot satisfy multiple output slots"
                    )
            elif self.text_logits.ndim < 2:
                raise invalid_descriptor("text logits must include a vocabulary dimension")
            elif int(logits_rows.shape[0]) < len(text_slots):
                raise invalid_descriptor("text logits row count is smaller than text output slots")
            if self.text_postprocess is not None:
                if len(self.text_postprocess) != len(text_slots):
                    raise invalid_descriptor(
                        "text postprocess entry count must match text output slots"
                    )
                text_rows = {int(slot.row_index): slot for slot in text_slots}
                for entry in self.text_postprocess:
                    slot = text_rows.get(int(entry.row_index))
                    if slot is None or int(slot.req_id) != int(entry.req_id):
                        raise invalid_descriptor(
                            "text postprocess entry does not align with output slot"
                        )
                    if int(entry.logits_index) < 0 or int(entry.logits_index) >= int(
                        logits_rows.shape[0]
                    ):
                        raise invalid_descriptor(
                            "text postprocess entry references an unknown logits row"
                        )
        denoise_slots = [
            slot for slot in plan.output_slots if slot.kind is ForwardOutputKind.DENOISE_STEP
        ]
        if denoise_slots:
            velocities = self.denoise_velocities or {}
            updates = self.denoise_updates or {}
            for slot in denoise_slots:
                update = updates.get(int(slot.row_index))
                if update is not None:
                    if int(update.row_index) != int(slot.row_index) or int(update.req_id) != int(
                        slot.req_id
                    ):
                        raise invalid_descriptor(
                            "denoise update entry does not align with output slot"
                        )
                    if not update.branch_names:
                        raise invalid_descriptor(
                            "denoise update entry must name at least one branch"
                        )
                    branch_count = len(update.branch_names)
                else:
                    row = plan.rows[slot.row_index]
                    branch_count = int(row.denoise.branch_count if row.denoise is not None else 0)
                for branch_id in range(branch_count):
                    key = DenoiseBranchKey(slot.row_index, branch_id)
                    if key not in velocities:
                        raise invalid_descriptor(
                            "forward result is missing a denoise branch velocity"
                        )
        encode_slots = [slot for slot in plan.output_slots if slot.kind is ForwardOutputKind.ENCODE]
        if encode_slots and self.encode_outputs is None:
            raise invalid_descriptor("forward result is missing encode outputs")
        for slot in encode_slots:
            if self.encode_outputs is not None and int(slot.row_index) not in self.encode_outputs:
                raise invalid_descriptor("forward result is missing an encode row output")
        commit_slots = [slot for slot in plan.output_slots if slot.kind is ForwardOutputKind.COMMIT]
        if commit_slots and self.commit_outputs is None:
            raise invalid_descriptor("forward result is missing commit outputs")
        for slot in commit_slots:
            if self.commit_outputs is not None and int(slot.row_index) not in self.commit_outputs:
                raise invalid_descriptor("forward result is missing a commit row output")


def coerce_forward_result(value: Any) -> ForwardResult:
    if isinstance(value, ForwardResult):
        return value
    if isinstance(value, torch.Tensor):
        return ForwardResult(text_logits=value)
    if isinstance(value, (list, tuple)):
        return ForwardResult(runtime_outputs=tuple(value))
    if isinstance(value, Mapping):
        if "text_logits" in value or "hidden" in value:
            return ForwardResult(
                text_logits=value.get("text_logits"),
                hidden=value.get("hidden"),
            )
        return ForwardResult(runtime_outputs=(dict(value),))
    raise invalid_descriptor(f"unsupported forward result type {type(value).__name__}")
