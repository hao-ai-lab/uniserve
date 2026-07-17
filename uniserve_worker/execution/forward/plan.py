"""Control-side forward plans for the unified execution stack."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ...contracts.forward_batch import VisiblePolicy
from ...contracts.forward_mode import ForwardMode, mode_for_op
from ...foundation.errors import invalid_descriptor

if TYPE_CHECKING:
    from .fallback import ForwardGraphPolicy

__all__ = [
    "CacheSpanPlan",
    "ForwardAdmissionDecision",
    "ForwardAdmissionRouter",
    "ForwardModality",
    "ForwardOutputKind",
    "ForwardOutputSlot",
    "ForwardPlan",
    "ForwardPlanBuilder",
    "ForwardPostprocessPolicy",
    "ForwardResultProjection",
    "ForwardRowPlan",
    "ForwardRuntimeHandles",
    "ForwardSegmentClass",
    "ForwardSegmentPlan",
    "ForwardShapeSummary",
    "KvWritePolicy",
    "Route",
    "TextTokenSpanPlan",
]

_TEXT_MODES = frozenset(
    {ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.VERIFY_DRAFT}
)
_PACKED_FORWARD_MODES = frozenset(
    {ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.DENOISE, ForwardMode.COMMIT}
)


class Route(str, Enum):
    """Planning route for a resource-admitted worker op group."""

    PER_MODE = "per_mode"
    FORWARD = "forward"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ForwardAdmissionDecision:
    route: Route
    reason: str
    modes: tuple[ForwardMode, ...]

    @property
    def use_forward(self) -> bool:
        return self.route is Route.FORWARD


@dataclass(frozen=True)
class ForwardAdmissionRouter:
    @classmethod
    def from_runtime_config(cls) -> "ForwardAdmissionRouter":
        return cls()

    def decide(self, ops: Sequence[Mapping[str, object]]) -> ForwardAdmissionDecision:
        modes = tuple(mode_for_op(str(op.get("kind"))) for op in ops)
        if not ops:
            return ForwardAdmissionDecision(Route.PER_MODE, "empty batch", modes)
        if any(mode not in _PACKED_FORWARD_MODES for mode in modes):
            return ForwardAdmissionDecision(
                Route.PER_MODE,
                "mixed forward supports only text and gen ops",
                modes,
            )
        text_modes = {ForwardMode.EXTEND, ForwardMode.DECODE}
        if set(modes).issubset(text_modes) and all(mode in modes for mode in text_modes):
            if any(_has_values(op.get("spec_token_ids")) for op in ops):
                return ForwardAdmissionDecision(
                    Route.PER_MODE,
                    "text mixed forward does not route speculative rows",
                    modes,
                )
            return ForwardAdmissionDecision(
                Route.FORWARD, "text extend+decode mixed forward", modes
            )
        has_text = any(mode in text_modes for mode in modes)
        gen_modes = {ForwardMode.DENOISE, ForwardMode.COMMIT}
        has_gen = any(mode in gen_modes for mode in modes)
        if has_text and has_gen:
            return ForwardAdmissionDecision(Route.FORWARD, "und/gen mixed forward", modes)
        return ForwardAdmissionDecision(Route.PER_MODE, "requires concurrent und and gen ops", modes)

    def partition_supported(
        self,
        ops: Sequence[Mapping[str, object]],
    ) -> tuple[list[tuple[int, Mapping[str, object]]], list[tuple[int, Mapping[str, object]]]]:
        supported: list[tuple[int, Mapping[str, object]]] = []
        delegated: list[tuple[int, Mapping[str, object]]] = []
        for index, op in enumerate(ops):
            mode = mode_for_op(str(op.get("kind")))
            target = supported if mode in _PACKED_FORWARD_MODES else delegated
            target.append((index, op))
        return supported, delegated


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
        branch_count = sum(
            int(row.denoise.branch_count) for row in rows if row.denoise is not None
        )
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


@dataclass(frozen=True)
class ForwardRuntimeHandles:
    request_states: Any = None
    residency: Any = None
    scratch: Any = None
    tensor_store: Any = None
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


@dataclass(frozen=True)
class ForwardPlan:
    step_id: int | None
    rows: tuple[ForwardRowPlan, ...]
    segments: tuple[ForwardSegmentPlan, ...]
    output_slots: tuple[ForwardOutputSlot, ...]
    shape: ForwardShapeSummary
    graph_policy: "ForwardGraphPolicy | None" = None
    runtime_handles: ForwardRuntimeHandles = field(default_factory=ForwardRuntimeHandles)

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


class ForwardPlanBuilder:
    """Build immutable control plans from admitted worker op groups."""

    def build(
        self,
        group: Sequence[Mapping[str, Any] | tuple[int, Mapping[str, Any]]],
        *,
        request_states: Any = None,
        step_id: int | None = None,
        graph_policy: "ForwardGraphPolicy | None" = None,
        runtime_handles: ForwardRuntimeHandles | None = None,
    ) -> ForwardPlan:
        if not group:
            raise invalid_descriptor("forward plan group must not be empty")
        rows: list[ForwardRowPlan] = []
        segments: list[ForwardSegmentPlan] = []
        output_slots: list[ForwardOutputSlot] = []
        for row_index, item in enumerate(group):
            original_index, op = _group_item(row_index, item)
            row = self._row_plan(row_index, original_index, op, request_states)
            rows.append(row)
            segments.extend(self._segments_for_row(row, len(segments)))
            output_slots.append(self._output_slot(row))
        forward_mode = _summary_mode(tuple(row.mode for row in rows))
        shape = ForwardShapeSummary.from_parts(
            forward_mode=forward_mode,
            rows=rows,
            segments=segments,
        )
        handles = runtime_handles or ForwardRuntimeHandles(request_states=request_states)
        plan = ForwardPlan(
            step_id=step_id,
            rows=tuple(rows),
            segments=tuple(segments),
            output_slots=tuple(output_slots),
            shape=shape,
            graph_policy=graph_policy,
            runtime_handles=handles,
        )
        plan.validate()
        return plan

    def _row_plan(
        self,
        row_index: int,
        original_index: int,
        op: Mapping[str, Any],
        request_states: Any,
    ) -> ForwardRowPlan:
        kind = op.get("kind")
        if not isinstance(kind, str):
            raise invalid_descriptor("forward op kind must be a string")
        req_id = _int_field(op, "req_id")
        mode = mode_for_op(kind)
        token_span = self._text_span(op, mode)
        cache_span = self._cache_span(op, request_states, req_id, token_span)
        return ForwardRowPlan(
            row_index=row_index,
            original_index=original_index,
            req_id=req_id,
            op=MappingProxyType(dict(op)),
            mode=mode,
            token_span=token_span,
            cache_span=cache_span,
            denoise=self._denoise_plan(op, mode),
            commit=self._commit_plan(op, mode),
            encode=self._encode_plan(op, mode),
        )

    @staticmethod
    def _text_span(op: Mapping[str, Any], mode: ForwardMode) -> TextTokenSpanPlan | None:
        if mode not in _TEXT_MODES:
            return None
        tokens = tuple(int(token) for token in (op.get("token_ids") or ()))
        start, end = _pos_range(op, len(tokens))
        return TextTokenSpanPlan(
            token_ids=tokens,
            position_start=start,
            position_end=end,
            token_source=str(op.get("token_source") or "wire"),
            last_token_only=mode in {ForwardMode.DECODE, ForwardMode.EXTEND, ForwardMode.VERIFY_DRAFT},
        )

    @staticmethod
    def _cache_span(
        op: Mapping[str, Any],
        request_states: Any,
        req_id: int,
        token_span: TextTokenSpanPlan | None,
    ) -> CacheSpanPlan | None:
        if token_span is None:
            return None
        state = None
        if request_states is not None:
            get = getattr(request_states, "get", None)
            if callable(get):
                try:
                    state = get(req_id)
                except Exception:
                    state = None
        state_blocks = tuple(int(block) for block in getattr(state, "block_ids", ()) or ())
        new_blocks = tuple(int(block) for block in (op.get("new_block_ids") or ()))
        block_ids = state_blocks
        if new_blocks and not block_ids[-len(new_blocks):] == new_blocks:
            block_ids = (*block_ids, *new_blocks)
        return CacheSpanPlan(
            block_ids=block_ids,
            base_len=int(token_span.position_start),
            append_len=token_span.q_len,
            pool_identity=str(op.get("kv_pool") or "text"),
            persistent=True,
        )

    @staticmethod
    def _denoise_plan(op: Mapping[str, Any], mode: ForwardMode) -> DenoiseRowPlan | None:
        if mode is not ForwardMode.DENOISE:
            return None
        cfg = dict(op.get("cfg") or {})
        branch_count = int(cfg.get("branch_count") or op.get("branch_count") or 1)
        if branch_count < 1:
            raise invalid_descriptor("denoise branch count must be positive")
        return DenoiseRowPlan(
            step_index=int(op.get("timestep_idx") or 0),
            total_steps=max(1, int(op.get("num_steps") or op.get("total_steps") or 1)),
            branch_count=branch_count,
            branch_ids=tuple(_branch_name(i, branch_count) for i in range(branch_count)),
            image_token_count=max(1, _image_token_count(op)),
            latent_handle=_optional_int(op.get("latent_handle")),
            grid_hw=_grid_hw(op.get("grid_hw")),
            cfg=MappingProxyType(cfg),
        )

    @staticmethod
    def _commit_plan(op: Mapping[str, Any], mode: ForwardMode) -> CommitRowPlan | None:
        if mode is not ForwardMode.COMMIT:
            return None
        return CommitRowPlan(
            latent_handle=_optional_int(op.get("latent_handle")),
            fold_back=bool(op.get("fold_back", False)),
            image_token_count=max(1, _image_token_count(op)),
        )

    @staticmethod
    def _encode_plan(op: Mapping[str, Any], mode: ForwardMode) -> EncodeRowPlan | None:
        if mode is not ForwardMode.ENCODE:
            return None
        return EncodeRowPlan(
            kind=str(op.get("kind")),
            out_handle=_optional_int(op.get("out_handle") or op.get("encoder_handle")),
            mm_hash=_optional_int(op.get("mm_hash")),
            num_tokens=max(1, int(op.get("num_tokens") or op.get("image_token_count") or 1)),
        )

    @staticmethod
    def _segments_for_row(
        row: ForwardRowPlan,
        next_segment_index: int,
    ) -> list[ForwardSegmentPlan]:
        segments: list[ForwardSegmentPlan] = []
        if row.token_span is not None and row.token_span.q_len > 0:
            segment_class = (
                ForwardSegmentClass.DECODE
                if row.mode is ForwardMode.DECODE
                else ForwardSegmentClass.EXTEND
            )
            segments.append(
                ForwardSegmentPlan(
                    segment_index=next_segment_index,
                    row_index=row.row_index,
                    mode=row.mode,
                    modality=ForwardModality.TEXT,
                    segment_class=segment_class,
                    q_len=row.token_span.q_len,
                    prefix_len=row.token_span.position_start,
                    visible_policy=VisiblePolicy.CAUSAL,
                    branch_id=0,
                    position_source=row.token_span.token_source,
                    kv_write_policy=KvWritePolicy.PERSISTENT,
                )
            )
            next_segment_index += 1
        if row.denoise is not None:
            for branch_index in range(row.denoise.branch_count):
                segments.append(
                    ForwardSegmentPlan(
                        segment_index=next_segment_index,
                        row_index=row.row_index,
                        mode=row.mode,
                        modality=ForwardModality.GENERATION,
                        segment_class=ForwardSegmentClass.DENOISE,
                        q_len=row.denoise.image_token_count,
                        prefix_len=0,
                        visible_policy=VisiblePolicy.BIDIRECTIONAL,
                        branch_id=branch_index,
                        position_source="generation_grid",
                        kv_write_policy=KvWritePolicy.TRANSIENT,
                    )
                )
                next_segment_index += 1
        if row.commit is not None:
            segments.append(
                ForwardSegmentPlan(
                    segment_index=next_segment_index,
                    row_index=row.row_index,
                    mode=row.mode,
                    modality=ForwardModality.GENERATION,
                    segment_class=ForwardSegmentClass.COMMIT_INPUT,
                    q_len=row.commit.image_token_count,
                    visible_policy=VisiblePolicy.BIDIRECTIONAL,
                    kv_write_policy=KvWritePolicy.NONE,
                )
            )
            next_segment_index += 1
        if row.encode is not None:
            segments.append(
                ForwardSegmentPlan(
                    segment_index=next_segment_index,
                    row_index=row.row_index,
                    mode=row.mode,
                    modality=ForwardModality.GENERATION,
                    segment_class=ForwardSegmentClass.ENCODE_INPUT,
                    q_len=row.encode.num_tokens,
                    visible_policy=VisiblePolicy.BIDIRECTIONAL,
                    kv_write_policy=KvWritePolicy.NONE,
                )
            )
        return segments

    @staticmethod
    def _output_slot(row: ForwardRowPlan) -> ForwardOutputSlot:
        if row.mode in _TEXT_MODES:
            return ForwardOutputSlot(
                row_index=row.row_index,
                req_id=row.req_id,
                kind=ForwardOutputKind.TEXT_TOKEN,
                result_projection=ForwardResultProjection.LAST_TEXT_ROW,
                postprocess_policy=ForwardPostprocessPolicy.SAMPLE,
            )
        if row.mode is ForwardMode.DENOISE:
            return ForwardOutputSlot(
                row_index=row.row_index,
                req_id=row.req_id,
                kind=ForwardOutputKind.DENOISE_STEP,
                result_projection=ForwardResultProjection.DENOISE_BRANCHES,
                postprocess_policy=ForwardPostprocessPolicy.LATENT_UPDATE,
            )
        if row.mode is ForwardMode.COMMIT:
            return ForwardOutputSlot(
                row_index=row.row_index,
                req_id=row.req_id,
                kind=ForwardOutputKind.COMMIT,
                result_projection=ForwardResultProjection.COMMIT_ROW,
                postprocess_policy=ForwardPostprocessPolicy.COMMIT_DECODE,
            )
        if row.mode is ForwardMode.ENCODE:
            return ForwardOutputSlot(
                row_index=row.row_index,
                req_id=row.req_id,
                kind=ForwardOutputKind.ENCODE,
                result_projection=ForwardResultProjection.ENCODE_ROW,
                postprocess_policy=ForwardPostprocessPolicy.ENCODE_PUBLISH,
            )
        return ForwardOutputSlot(
            row_index=row.row_index,
            req_id=row.req_id,
            kind=ForwardOutputKind.COMBINED,
            result_projection=ForwardResultProjection.RUNTIME_OUTPUT,
            postprocess_policy=ForwardPostprocessPolicy.NONE,
        )


def _summary_mode(modes: tuple[ForwardMode, ...]) -> ForwardMode:
    first = modes[0]
    if any(mode is not first for mode in modes):
        return ForwardMode.MIXED
    return first


def _group_item(
    fallback_index: int,
    item: Mapping[str, Any] | tuple[int, Mapping[str, Any]],
) -> tuple[int, Mapping[str, Any]]:
    if isinstance(item, tuple) and len(item) == 2:
        index, op = item
        if not isinstance(op, Mapping):
            raise invalid_descriptor("forward group item op must be a mapping")
        return int(index), op
    if not isinstance(item, Mapping):
        raise invalid_descriptor("forward group item must be an op mapping")
    return fallback_index, item


def _int_field(op: Mapping[str, Any], field_name: str) -> int:
    value = op.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise invalid_descriptor(f"forward op {field_name} must be an integer")
    return int(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _pos_range(op: Mapping[str, Any], token_count: int) -> tuple[int, int]:
    raw = op.get("pos_range")
    if raw is None:
        return 0, int(token_count)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise invalid_descriptor("text op pos_range must be [start, end]")
    start, end = int(raw[0]), int(raw[1])
    if end < start:
        raise invalid_descriptor("text op pos_range end must be >= start")
    return start, end


def _image_token_count(op: Mapping[str, Any]) -> int:
    for key in ("image_token_count", "latent_tokens", "num_tokens"):
        if op.get(key) is not None:
            return int(op[key])
    shape = op.get("latent_shape")
    if isinstance(shape, (list, tuple)) and shape:
        total = 1
        for value in shape:
            total *= max(1, int(value))
        return total
    return 1


def _grid_hw(raw: Any) -> tuple[int, int]:
    if raw is None:
        return (0, 0)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise invalid_descriptor("grid_hw must be [height, width]")
    return (int(raw[0]), int(raw[1]))


def _branch_name(index: int, count: int) -> str:
    if count == 1:
        return "cond"
    if index == 0:
        return "cond"
    return f"branch_{index}"


def _has_values(raw: object) -> bool:
    if raw is None:
        return False
    try:
        return len(raw) > 0  # type: ignore[arg-type]
    except TypeError:
        return bool(raw)
