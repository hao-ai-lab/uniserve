"""Bounded CUDA graph executables over fixed and exact-shape forward batches."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Hashable, Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import partial
from typing import Any, cast

import torch

from uniserve_worker.execution.batch import ForwardMode, PipelineStage
from uniserve_worker.execution.forward_batch import (
    AttentionMode,
    AttentionSelection,
    ForwardBatch,
    ForwardOutput,
    TokenSelection,
    packed_tensor_views,
)
from uniserve_worker.execution.graph.backend import CudaGraphBackend, GraphExecutionError
from uniserve_worker.execution.input_buffers import InputBuffers
from uniserve_worker.execution.lane import ExecutionLaneRuntime
from uniserve_worker.foundation.math import bucketed_length
from uniserve_worker.foundation.resources import close_resources
from uniserve_worker.models.runtime import CacheGeometry
from uniserve_worker.nn.collective import StreamCollectives, stream_collective_scope
from uniserve_worker.runtime.cache_pool import CachePool

logger = logging.getLogger(__name__)
TOKEN_CONTINUATION_BIT = 1 << 31
_GRAPH_BINDINGS = itertools.count(1)


class _GraphMiss(RuntimeError):
    """Signals that a requested CUDA graph signature has no captured executable."""

    pass


@dataclass(frozen=True, slots=True)
class FlowCapture:
    """A denoise request shape, including its physical guidance branch count."""

    rows: int
    height: int
    width: int
    cfg_branches: int

    @property
    def executable_key(self) -> tuple[object, ...]:
        return "flow", self.rows * self.cfg_branches, self.height, self.width


@dataclass(frozen=True, slots=True)
class MixedCapture:
    """A compatible token-decode and denoise capture configuration."""

    decode_rows: int
    flow_rows: int
    height: int
    width: int
    cfg_branches: int

    def __post_init__(self) -> None:
        if min(self.decode_rows, self.flow_rows, self.height, self.width, self.cfg_branches) < 1:
            raise ValueError("mixed capture requires positive token, flow, and media extents")

    @property
    def executable_key(self) -> tuple[object, ...]:
        return (
            "decode_flow",
            self.decode_rows,
            self.flow_rows * self.cfg_branches,
            self.height,
            self.width,
        )


@dataclass(frozen=True, slots=True)
class PrefixCapture:
    """Repeated request prefixes that share one physical packed executable."""

    rows: int
    prefix_lengths: tuple[int, ...]

    @property
    def executable_key(self) -> tuple[object, ...]:
        return "flow_prefix", self.prefix_lengths * self.rows


@dataclass(frozen=True, slots=True)
class PrefillCapture:
    """A padded prefill shape and the minimum live rows exercising its bucket."""

    token_bucket: int
    row_bucket: int
    live_rows: int


def select_flow_captures(
    shapes: Sequence[tuple[int, int]],
    request_counts: Sequence[int],
    cfg_branches: Sequence[int],
    *,
    max_operations: int,
    max_tokens: int,
    per_image_capacity: int,
    latent_capacity: int,
    physical_tokens: Callable[[int, int], int],
    image_tokens: Callable[[int, int], int],
) -> tuple[FlowCapture, ...]:
    """Intersect requested capture shapes with actual staging and latent bounds."""

    return tuple(
        FlowCapture(rows, height, width, branches)
        for height, width in shapes
        for rows in request_counts
        for branches in cfg_branches
        if 0 < rows <= max_operations
        and rows * physical_tokens(height, width) * branches <= max_tokens
        and image_tokens(height, width) <= per_image_capacity
        and rows * image_tokens(height, width) <= latent_capacity
    )


def select_mixed_captures(
    flow: Sequence[FlowCapture], decode_rows: Sequence[int]
) -> tuple[MixedCapture, ...]:
    """Enumerate the supported single-trajectory mixed execution shapes."""

    return tuple(
        MixedCapture(rows, 1, item.height, item.width, item.cfg_branches)
        for item in flow
        if item.rows == 1
        for rows in decode_rows
    )


def select_prefill_captures(
    token_sizes: Sequence[int],
    row_sizes: Sequence[int],
    *,
    max_rows: int,
    max_tokens: int,
) -> tuple[PrefillCapture, ...]:
    """Enumerate exactly the padded prefill buckets within the execution bounds."""

    buckets: list[PrefillCapture] = []
    minimum_rows = 1
    for row_bucket in sorted({int(value) for value in row_sizes if int(value) > 1}):
        if minimum_rows > int(max_rows):
            break
        minimum_tokens = minimum_rows if minimum_rows == 1 else minimum_rows + 1
        for token_bucket in sorted(
            {int(value) for value in token_sizes if minimum_tokens <= int(value) <= int(max_tokens)}
        ):
            buckets.append(PrefillCapture(token_bucket, row_bucket, minimum_rows))
        minimum_rows = row_bucket
    return tuple(buckets)


@dataclass(frozen=True, slots=True)
class GraphGreedyOutput:
    """Carries graph-produced hidden states with device-resident greedy token and validity vectors."""

    request_pool_indices: torch.Tensor
    tokens: torch.Tensor
    valid: torch.Tensor
    active: torch.Tensor
    finish: torch.Tensor
    continuation: torch.Tensor
    tagged_tokens: torch.Tensor
    completion: torch.Tensor


@dataclass(frozen=True, slots=True)
class GraphRun:
    """Returns a model output with graph-path identity, live rows, padded rows, and optional greedy samples."""

    output: ForwardOutput
    path: str
    row_count: int
    padded_row_count: int
    greedy: GraphGreedyOutput | None = None


@dataclass(slots=True)
class _PackedInputs:
    """Static input addresses and attention resources for one physical key."""

    batch: ForwardBatch
    releases: tuple[Callable[[], None], ...]
    bucketed: bool
    batch_leaves: tuple[torch.Tensor, ...] = ()
    plan_leaves: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True, slots=True)
class _DecodeGeometry:
    """Captures row, head, page, and cache bounds that determine a decode graph signature."""

    bucket: int
    width: int
    padding: int


@dataclass(frozen=True, slots=True)
class _PrefillGeometry:
    """Captures token, row, head, page, and cache bounds that determine a prefill graph signature."""

    token_bucket: int
    row_bucket: int
    width: int
    padding: int
    maximum_padding: int
    max_query_len: int
    max_key_len: int


class PackedRunner:
    """Own one lane's packed inputs, attention metadata, and output publication."""

    def __init__(
        self,
        *,
        backend: CudaGraphBackend[tuple[ForwardOutput, GraphGreedyOutput | None]] | None,
        enabled: bool,
        prefill_enabled: bool,
        cache: CacheGeometry,
        cache_pool: CachePool,
        attention: AttentionSelection,
        block_size: int,
        memory_budget_bytes: int,
        decode_batch_sizes: tuple[int, ...] = (),
        decode_predicates: torch.Tensor | None = None,
        decode_context_blocks: int = 0,
        packed_context_blocks: int = 0,
        prefill_token_sizes: tuple[int, ...] = (),
        prefill_row_sizes: tuple[int, ...] = (8, 16),
        stream: torch.cuda.Stream | None = None,
        prefill_shapes: tuple[PrefillCapture, ...] = (),
        inputs: InputBuffers | None = None,
        lane: ExecutionLaneRuntime | None = None,
    ) -> None:
        """Configure one lane's bounded graph catalog, workspaces, and capture identity."""

        if block_size < 1 or memory_budget_bytes < 0:
            raise ValueError("graph-store identity and geometry are invalid")
        if decode_predicates is not None and (
            decode_predicates.ndim != 1 or decode_predicates.dtype is not torch.bool
        ):
            raise ValueError("decode predicate state must be a boolean row vector")
        self.collectives: dict[str, StreamCollectives] = {}
        self.inputs = inputs
        self.lane = lane
        self.backend = backend
        self.enabled = bool(enabled)
        self.prefill_enabled = bool(prefill_enabled)
        self.cache = cache
        self.cache_pool = cache_pool
        self.attention = attention
        self.block_size = int(block_size)
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.decode_batch_sizes = tuple(
            sorted({int(value) for value in decode_batch_sizes if int(value) > 0})
        )
        self.decode_predicates = decode_predicates
        self.decode_context_blocks = max(0, int(decode_context_blocks))
        self.packed_context_blocks = max(0, int(packed_context_blocks))
        self.prefill_token_sizes = tuple(
            sorted({int(value) for value in prefill_token_sizes if int(value) > 0})
        )
        self.prefill_row_sizes = tuple(
            sorted({int(value) for value in prefill_row_sizes if int(value) > 1})
        )
        self.captures = 0
        self._inputs: dict[tuple[object, ...], _PackedInputs] = {}
        self._capture_keys: dict[tuple[object, ...], None] = {}
        self._equivalence_checks: list[tuple[str, torch.Tensor]] = []
        self._device: torch.device | None = None
        self._sealed = False
        self._stream = stream
        self.prefill_shapes = prefill_shapes

    @property
    def resident_bytes(self) -> int:
        """Measure memory retained by the graph runner's private CUDA pool."""

        return _private_pool_bytes(self._device)

    def complete_startup(self) -> None:
        """Verify and seal the configured bucket set before request admission."""

        if self._sealed:
            return
        if self.backend is not None and any(
            not self.backend.contains(key) for key in self._capture_keys
        ):
            raise GraphExecutionError("startup left configured graph buckets uncaptured")
        self._complete_equivalence_checks()
        if self.resident_bytes > self.memory_budget_bytes:
            raise GraphExecutionError("captured graph residency exceeds its startup budget")
        self._sealed = True

    @property
    def startup_signature(self) -> tuple[object, ...]:
        """Return immutable graph configuration used to detect startup-shape changes."""

        return (
            self.decode_batch_sizes,
            self.prefill_token_sizes,
            self.prefill_row_sizes,
            tuple(sorted((repr(key) for key in self._inputs))),
            self.captures,
            self._sealed,
        )

    def select_shape(
        self,
        batch: ForwardBatch,
        *,
        eligible: bool,
    ) -> tuple[tuple[object, ...], ForwardBatch, int, bool] | None:
        """Select physical geometry while preserving each path's eager policy."""

        rows = batch.row_count
        if not eligible or not self.enabled or not _cuda_batch(batch):
            return None
        if batch.attention_mode is AttentionMode.PAGED_VARLEN and (
            not self.prefill_enabled
            or any(
                selection is not TokenSelection.LAST_LOGITS for selection in batch.token_selections
            )
        ):
            return None
        if batch.attention_mode is AttentionMode.PACKED and not self.prefill_enabled:
            return None
        if batch.attention_mode is AttentionMode.PACKED and _quantized_kv(self):
            return None
        try:
            _graph_provider(
                self.attention,
                batch.attention_mode,
                head_dim=self.cache.head_dim,
                block_size=self.block_size,
                device=batch.request_pool_indices.device,
            )
        except _GraphMiss:
            return None

        decode = _decode_geometry(
            batch,
            self.decode_batch_sizes,
            self.block_size,
            self.decode_context_blocks,
        )
        prefill = (
            None
            if decode is not None
            else _prefill_geometry(
                batch,
                self.prefill_token_sizes,
                self.block_size,
                self.prefill_row_sizes,
                self.decode_context_blocks,
            )
        )
        if decode is not None:
            execution = _pad_decode_batch(batch, decode, self.block_size)
            state_key = _decode_signature(batch, decode)
            signature = state_key
            padded_rows = decode.bucket
            startup_resident = True
        elif prefill is not None:
            execution = _pad_prefill_batch(batch, prefill)
            state_key = _prefill_signature(batch, prefill)
            signature = state_key
            padded_rows = prefill.row_bucket
            startup_resident = True
        else:
            execution = _normalize_exact_batch(
                batch,
                context_blocks=self.packed_context_blocks,
                block_size=self.block_size,
            )
            signature = _exact_signature(execution)
            state_key = ("exact", signature)
            padded_rows = rows
            startup_resident = False

        return state_key, execution, padded_rows, startup_resident

    @torch.inference_mode()
    def capture(
        self,
        batch: ForwardBatch,
        forward: Callable[[ForwardBatch], ForwardOutput],
    ) -> None:
        """Explicitly warm and capture one selected startup shape, then check replay."""

        if self._sealed:
            raise GraphExecutionError("packed capture is outside startup preparation")
        selected = self.select_shape(batch, eligible=True)
        if selected is None:
            self.warmup(batch, forward)
            return
        state_key, execution, _, bucketed = selected
        self._capture_keys[state_key] = None
        assert self.backend is not None
        if self.backend.contains(state_key):
            return
        if self.memory_budget_bytes == 0:
            raise GraphExecutionError("configured CUDA graph residency has no memory budget")
        self._device = _batch_device(batch)
        static = _graph_batch(execution, next(_GRAPH_BINDINGS), own_inputs=not bucketed)
        releases: tuple[Callable[[], None], ...] = ()
        context = nullcontext() if self._stream is None else torch.cuda.stream(self._stream)
        with context, stream_collective_scope(self.collectives):
            restore = self._capture_restore(static)
            try:
                # Compare the same physical input and state with the eager provider.
                eager = self._snapshot_output(forward(execution))
                restore()
                releases = self._prepare_attention(static, execution, capture=True)

                def compute() -> tuple[ForwardOutput, GraphGreedyOutput | None]:
                    output = forward(static)
                    return output, _greedy_decode(static, output, self.decode_predicates)

                self.backend.capture_one(state_key, compute, keepalive=(static,), restore=restore)
                inputs = _PackedInputs(
                    static,
                    releases,
                    bucketed,
                    tuple(_tensor_leaves(static)),
                    tuple(_attention_tensor_leaves(static)),
                )
                output, greedy = self._replay(state_key, inputs, execution)
                self._queue_equivalence_check(eager, output, label=repr(state_key))
                force_finish = static.decode_force_finish
                restore()
                expected = _greedy_decode_values(
                    static, output, self.decode_predicates, force_finish, clear_force_finish=False
                )
                self._queue_greedy_equivalence_check(expected, greedy, label=repr(state_key))
                restore()
                self._inputs[state_key] = inputs
                self.captures += 1
            except BaseException as error:
                if self._stream is not None:
                    self._stream.synchronize()
                else:
                    torch.cuda.current_stream(self._device).synchronize()
                self.backend.discard(state_key)
                for release in reversed(releases):
                    release()
                error.add_note(f"packed capture device={self._device} shape={state_key!r}")
                raise
            finally:
                restore()
                torch.cuda.current_stream(self._device).synchronize()

    def _capture_restore(self, batch: ForwardBatch) -> Callable[[], None]:
        """Retain the bounded KV write set and graph-greedy mutable input."""

        tensors: list[tuple[torch.Tensor, torch.Tensor]] = []
        finish = batch.decode_force_finish
        if finish is not None:
            tensors.append((finish, finish.clone()))
        pages = torch.unique(batch.out_cache_loc // self.block_size)
        pages = pages[pages != 0].long()
        cache = self.cache_pool
        saved = (cache.k.index_select(1, pages), cache.v.index_select(1, pages))

        def restore() -> None:
            for tensor, snapshot in tensors:
                tensor.copy_(snapshot)
            cache.k.index_copy_(1, pages, saved[0])
            cache.v.index_copy_(1, pages, saved[1])

        return restore

    def warmup(self, batch: ForwardBatch, forward: Callable[[ForwardBatch], ForwardOutput]) -> None:
        """Prepare eager-only numerical inputs independently of graph capture."""

        self._eager(batch, forward)

    def run(
        self,
        batch: ForwardBatch,
        forward: Callable[[ForwardBatch], ForwardOutput],
        *,
        eligible: bool,
        borrow_output: bool = False,
    ) -> GraphRun:
        """Stage metadata and replay a resident key, or use the established eager path."""

        rows = batch.row_count
        selected = self.select_shape(batch, eligible=eligible)
        if selected is None:
            return GraphRun(self._eager(batch, forward), "eager", rows, rows)
        state_key, execution, padded_rows, bucketed = selected
        captured = False
        if state_key not in self._capture_keys:
            if not bucketed:
                return GraphRun(self._eager(execution, forward), "eager", rows, padded_rows)
            if self._sealed:
                raise GraphExecutionError("configured CUDA graph bucket is not resident")
            # Direct execution may precede explicit startup. Materialize the
            # configured bucket on its binding, preserving the same full policy.
            self.capture(batch, forward)
            captured = True
        assert self.backend is not None
        inputs = self._inputs.get(state_key)
        if inputs is None:
            raise GraphExecutionError("packed graph has no input owner")
        output, greedy = self._replay(state_key, inputs, execution)
        output = _trim_output(output, rows) if borrow_output else self._publish_output(output, rows)
        return GraphRun(
            output,
            "graph_capture" if captured else "graph_replay",
            rows,
            padded_rows,
            _trim_greedy(greedy, rows),
        )

    def _discard_inputs(self) -> None:
        inputs, self._inputs = self._inputs, {}
        self._equivalence_checks.clear()
        actions = []
        for key, state in inputs.items():
            if self.backend is not None:
                actions.append(partial(self.backend.discard, key))
            actions.extend(reversed(state.releases))
        close_resources(*actions)

    def close(self) -> None:
        """Drain graph accesses and release owned inputs before the borrowed lane."""

        actions = [self._discard_inputs]
        if self.backend is not None:
            actions.append(self.backend.close)
        actions.extend(binding.close for binding in self.collectives.values())
        if self.inputs is not None:
            actions.append(self.inputs.close)
        try:
            close_resources(*actions)
        finally:
            self._capture_keys.clear()
            self.collectives.clear()
            self.inputs = None
            self.lane = None
            self.backend = None

    def _eager(
        self,
        batch: ForwardBatch,
        forward: Callable[[ForwardBatch], ForwardOutput],
    ) -> ForwardOutput:
        """Execute a numerical batch on the lane's eager path."""

        context = nullcontext() if self._stream is None else torch.cuda.stream(self._stream)
        with context, stream_collective_scope(self.collectives):
            return forward(batch)

    def _replay(
        self, key: Hashable, inputs: _PackedInputs, execution: ForwardBatch
    ) -> tuple[ForwardOutput, GraphGreedyOutput | None]:
        context = nullcontext() if self._stream is None else torch.cuda.stream(self._stream)
        with context, stream_collective_scope(self.collectives):
            if not inputs.bucketed:
                _copy_into_leaves(inputs.batch_leaves, execution, "forward")
            else:
                _copy_into_leaves(
                    inputs.plan_leaves, tuple(_attention_tensor_leaves(execution)), "attention"
                )
            self._prepare_attention(inputs.batch, execution, capture=False)
            assert self.backend is not None
            return self.backend.replay(key)

    def _publish_output(self, output: ForwardOutput, rows: int) -> ForwardOutput:
        """Publish live rows before another graph reuses capture storage."""

        consumer = None if self._stream is None else torch.cuda.current_stream(self._stream.device)
        context = nullcontext() if self._stream is None else torch.cuda.stream(self._stream)
        with context, stream_collective_scope(self.collectives):
            published = _trim_output(output, rows).clone()
        if consumer is not None and consumer != self._stream:
            for value in published.values:
                value.record_stream(consumer)
        return published

    def _snapshot_output(self, output: ForwardOutput) -> ForwardOutput:
        """Clone a forward output for later direct-versus-graph comparison."""

        context = nullcontext() if self._stream is None else torch.cuda.stream(self._stream)
        with context, stream_collective_scope(self.collectives):
            return output.clone()

    def _queue_equivalence_check(
        self,
        reference: ForwardOutput,
        candidate: ForwardOutput,
        *,
        label: str,
    ) -> None:
        """Queue device-side equality checks for every forward-output tensor."""

        context = nullcontext() if self._stream is None else torch.cuda.stream(self._stream)
        with context, stream_collective_scope(self.collectives):
            check = _equivalence_check(reference, candidate)
        self._equivalence_checks.append((label, check))

    def _queue_greedy_equivalence_check(
        self,
        reference: GraphGreedyOutput | None,
        candidate: GraphGreedyOutput | None,
        *,
        label: str,
    ) -> None:
        """Queue device-side equality checks between direct and captured greedy outputs."""

        if reference is None or candidate is None:
            if reference is not candidate:
                raise GraphExecutionError("CUDA graph greedy output availability changed")
            return
        for field in fields(GraphGreedyOutput):
            expected = getattr(reference, field.name)
            actual = getattr(candidate, field.name)
            if expected.shape != actual.shape or expected.dtype != actual.dtype:
                raise GraphExecutionError("CUDA graph greedy output geometry changed")
            self._equivalence_checks.append(
                (f"{label}:{field.name}", torch.eq(expected, actual).all())
            )

    def _complete_equivalence_checks(self) -> None:
        """Synchronize queued equality results and report any divergent graph fields."""

        if not self._equivalence_checks:
            return
        context = nullcontext() if self._stream is None else torch.cuda.stream(self._stream)
        with context, stream_collective_scope(self.collectives):
            complete = torch.stack(tuple(check for _, check in self._equivalence_checks)).all()
        if not bool(complete.item()):
            failed = tuple(
                label for label, check in self._equivalence_checks if not bool(check.item())
            )
            raise GraphExecutionError(
                "CUDA graph output differs from direct execution for buckets: " + ", ".join(failed)
            )
        self._equivalence_checks.clear()

    def _prepare_attention(
        self,
        static_batch: ForwardBatch,
        live_batch: ForwardBatch,
        *,
        capture: bool,
    ) -> tuple[Callable[[], None], ...]:
        """Bind static attention wrappers to live page metadata for capture or replay."""

        # Dense and packed attention carry no persistent backend plan. Paged
        # modes first copy live table views into the static graph batch.
        static = static_batch
        live = live_batch
        if static.attention_mode is not live.attention_mode:
            raise _GraphMiss("attention form changed for a graph bucket")
        if static.attention_mode in {AttentionMode.DENSE, AttentionMode.PACKED}:
            return ()
        if static.attention_mode not in {AttentionMode.PAGED_DECODE, AttentionMode.PAGED_VARLEN}:
            return ()
        prepared = _live_attention(static, live)
        key_cache, _value_cache = self.cache_pool.layer_cache(0, static.group_id)
        backend = _graph_provider(
            self.attention,
            static.attention_mode,
            head_dim=self.cache.head_dim,
            block_size=self.block_size,
            device=key_cache.device,
        )
        q_dtype = key_cache.dtype
        kv_dtype = key_cache.dtype
        releases: list[Callable[[], None]] = []

        if capture:
            release_name = (
                "release_paged_decode_graph_binding"
                if static.attention_mode is AttentionMode.PAGED_DECODE
                else "release_paged_prefill_graph_wrapper"
            )
            release = getattr(backend, release_name, None)
            if callable(release):
                releases.append(_release_call(release, static.binding))
        try:
            # Decode wrappers are keyed by graph binding and can be replanned for
            # each live table while retaining fixed tensor addresses.
            if static.attention_mode is AttentionMode.PAGED_DECODE:
                prepare = getattr(backend, "prepare_paged_decode_cuda_graph", None)
                if callable(prepare):
                    prepare(
                        static.binding,
                        prepared,
                        batch_size=int(cast(torch.Tensor, static.block_table).shape[0]),
                        max_indices=max(1, int(cast(torch.Tensor, static.block_table).numel())),
                        num_q_heads=int(self.cache.num_attention_heads),
                        num_kv_heads=int(self.cache.num_kv_heads),
                        head_dim=int(self.cache.head_dim),
                        page_size=self.block_size,
                        q_dtype=q_dtype,
                        kv_dtype=kv_dtype,
                    )
                return tuple(releases)

            # Prefill capture owns a graph-bound wrapper until graph eviction;
            # replay updates only its caller-owned metadata buffers.
            if static.attention_mode is not AttentionMode.PAGED_VARLEN:
                return ()
            bind = getattr(backend, "bind_paged_prefill_graph_wrapper", None)
            prepare = getattr(backend, "prepare_paged_prefill_cuda_graph", None)
            if callable(bind) and callable(prepare):
                if capture:
                    bind(
                        static.binding,
                        static,
                        device=cast(torch.Tensor, static.block_table).device,
                    )
                prepare(
                    static.binding,
                    prepared,
                    num_q_heads=int(self.cache.num_attention_heads),
                    num_kv_heads=int(self.cache.num_kv_heads),
                    head_dim=int(self.cache.head_dim),
                    page_size=self.block_size,
                    q_dtype=q_dtype,
                    kv_dtype=kv_dtype,
                    causal=static.causal,
                )
            return tuple(releases)
        except BaseException as error:
            for release in reversed(releases):
                try:
                    release()
                except BaseException as cleanup:
                    error.add_note(f"attention binding cleanup failed: {cleanup!r}")
            raise


def _decode_geometry(
    batch: ForwardBatch,
    batch_sizes: tuple[int, ...],
    block_size: int,
    context_blocks: int,
) -> _DecodeGeometry | None:
    """Select a graph decode bucket and derive padded rows, blocks, and token counts."""

    if batch.attention_mode is not AttentionMode.PAGED_DECODE or not batch_sizes:
        return None
    rows = batch.row_count
    bucket = next((value for value in batch_sizes if value >= rows), None)
    if bucket is None or batch.input_ids is None or batch.positions is None:
        return None
    if (
        batch.token_row_indices != tuple(range(rows))
        or batch.query_lens_cpu != (1,) * rows
        or int(batch.input_ids.numel()) != rows
        or int(batch.positions.shape[-1]) != rows
        or len(set(batch.token_selections)) != 1
    ):
        return None
    if batch.block_table is None:
        return None
    live_width = int(batch.block_table.shape[1])
    if context_blocks > 0 and live_width > context_blocks:
        return None
    reserved_width = max(1, (int(bucket) + int(block_size) - 1) // int(block_size))
    return _DecodeGeometry(
        bucket=int(bucket),
        width=max(live_width, int(context_blocks), reserved_width),
        padding=int(bucket) - rows,
    )


def _prefill_geometry(
    batch: ForwardBatch,
    token_sizes: tuple[int, ...],
    block_size: int,
    row_sizes: tuple[int, ...],
    context_blocks: int,
) -> _PrefillGeometry | None:
    """Select a graph prefill bucket and derive padded query and cache geometry."""

    if batch.attention_mode is not AttentionMode.PAGED_VARLEN or not token_sizes:
        return None
    rows = batch.row_count
    row_bucket = next((value for value in row_sizes if value > rows), None)
    query_lens = tuple(int(value) for value in batch.query_lens_cpu)
    if (
        row_bucket is None
        or (rows >= row_sizes[0] and all(value == 1 for value in query_lens))
        or batch.input_ids is None
        or batch.positions is None
        or batch.token_row_indices != tuple(range(rows))
        or len(query_lens) != rows
        or any(value < 1 for value in query_lens)
        or int(batch.input_ids.numel()) != sum(query_lens)
        or int(batch.positions.shape[-1]) != sum(query_lens)
    ):
        return None
    live_tokens = sum(query_lens)
    token_bucket = next((value for value in token_sizes if value >= live_tokens), None)
    if token_bucket is None:
        return None
    if batch.block_table is None:
        return None
    live_width = int(batch.block_table.shape[1])
    if context_blocks > 0 and live_width > context_blocks:
        return None
    previous = max((value for value in token_sizes if value < token_bucket), default=0)
    maximum_padding = int(token_bucket) - int(previous)
    reserved_width = max(1, (maximum_padding + int(block_size) - 1) // int(block_size))
    width = max(live_width, int(context_blocks), reserved_width)
    return _PrefillGeometry(
        token_bucket=int(token_bucket),
        row_bucket=int(row_bucket),
        width=width,
        padding=int(token_bucket) - live_tokens,
        maximum_padding=maximum_padding,
        max_query_len=bucketed_length(int(token_bucket)),
        max_key_len=width * int(block_size),
    )


def _pad_decode_batch(
    batch: ForwardBatch,
    geometry: _DecodeGeometry,
    block_size: int,
) -> ForwardBatch:
    """Copy a live decode batch into fixed-row graph buffers and synthesize padding rows."""

    bucket = geometry.bucket
    input_ids = _fixed_view(cast(torch.Tensor, batch.input_ids), (bucket,))
    positions = _expand_token_axis(cast(torch.Tensor, batch.positions), bucket)
    input_embeddings = (
        None
        if batch.input_embeddings is None
        else _fixed_view(
            batch.input_embeddings,
            (bucket, int(batch.input_embeddings.shape[1])),
        )
    )
    embedding_mask = (
        None if batch.embedding_mask is None else _fixed_view(batch.embedding_mask, (bucket,))
    )
    if batch.block_table is None or batch.seq_lens is None:
        raise _GraphMiss("paged decode has incomplete tensors")
    return replace(
        batch,
        row_count=bucket,
        request_pool_indices=_fixed_view(batch.request_pool_indices, (bucket,)),
        prefix_lens=_fixed_view(batch.prefix_lens, (bucket,)),
        query_lens=_fixed_view(batch.query_lens, (bucket,)),
        out_cache_loc=_fixed_view(batch.out_cache_loc, (bucket,)),
        block_table=_fixed_view(batch.block_table, (bucket, geometry.width)),
        seq_lens=_fixed_view(batch.seq_lens, (bucket,)),
        prefix_lens_cpu=(*batch.prefix_lens_cpu, *(0 for _ in range(geometry.padding))),
        seq_lens_cpu=(*batch.seq_lens_cpu, *(1 for _ in range(geometry.padding))),
        query_lens_cpu=(*batch.query_lens_cpu, *(1 for _ in range(geometry.padding))),
        max_seqlen_k=geometry.width * int(block_size),
        decode_force_finish=(
            None
            if batch.decode_force_finish is None
            else _fixed_view(batch.decode_force_finish, (bucket,))
        ),
        token_row_indices=tuple(range(bucket)),
        input_ids=input_ids,
        input_embeddings=input_embeddings,
        embedding_mask=embedding_mask,
        positions=positions,
        token_selections=(batch.token_selections[0],) * bucket,
    )


def _pad_prefill_batch(batch: ForwardBatch, geometry: _PrefillGeometry) -> ForwardBatch:
    """Extend a paged-prefill batch into fixed graph buckets using inert rows and tokens."""

    live_rows = batch.row_count
    dummy_rows = geometry.row_bucket - live_rows

    # Fixed-address views expose the capture bucket without reallocating the
    # staging tensors that back the live prefix.
    input_ids = _fixed_view(cast(torch.Tensor, batch.input_ids), (geometry.token_bucket,))
    positions = _expand_token_axis(cast(torch.Tensor, batch.positions), geometry.token_bucket)
    input_embeddings = (
        None
        if batch.input_embeddings is None
        else _fixed_view(
            batch.input_embeddings,
            (geometry.token_bucket, int(batch.input_embeddings.shape[1])),
        )
    )
    embedding_mask = (
        None
        if batch.embedding_mask is None
        else _fixed_view(batch.embedding_mask, (geometry.token_bucket,))
    )
    if (
        batch.block_table is None
        or batch.seq_lens is None
        or batch.cu_seqlens_q is None
        or batch.cu_seqlens_k is None
        or batch.output_indices is None
    ):
        raise _GraphMiss("paged prefill has incomplete tensors")
    block_table = _fixed_view(
        batch.block_table,
        (geometry.row_bucket, geometry.width),
    )
    cache_seqlens = _fixed_view(batch.prefix_lens, (geometry.row_bucket,))
    query_lens = _fixed_view(batch.query_lens, (geometry.row_bucket,))
    kv_seqlens = _fixed_view(batch.seq_lens, (geometry.row_bucket,))
    cu_seqlens_q = _fixed_view(batch.cu_seqlens_q, (geometry.row_bucket + 1,))
    cu_seqlens_k = _fixed_view(batch.cu_seqlens_k, (geometry.row_bucket + 1,))
    output_indices = _fixed_view(batch.output_indices, (geometry.row_bucket,))

    # Dummy rows advertise no cached or query tokens. Any token-axis padding is
    # assigned to the first dummy row so cumulative lengths still terminate at
    # the graph's fixed token bucket.
    cache_seqlens[live_rows:].zero_()
    query_lens[live_rows:].zero_()
    kv_seqlens[live_rows:].zero_()
    if geometry.padding:
        query_lens[live_rows : live_rows + 1].fill_(geometry.padding)
        kv_seqlens[live_rows : live_rows + 1].fill_(geometry.padding)
    cu_seqlens_q[live_rows + 1 :].fill_(geometry.token_bucket)
    padded_kv_tokens = sum(int(value) for value in batch.seq_lens_cpu) + geometry.padding
    cu_seqlens_k[live_rows + 1 :].fill_(padded_kv_tokens)
    output_indices[live_rows:].zero_()
    if geometry.padding:
        output_indices[live_rows : live_rows + 1].fill_(geometry.token_bucket - 1)
    dummy_query_lens = (geometry.padding, *(0 for _ in range(dummy_rows - 1)))

    # CPU mirrors must encode the same geometry because backend planning reads
    # them independently of the device-side cumulative arrays.
    return replace(
        batch,
        row_count=geometry.row_bucket,
        request_pool_indices=_fixed_view(
            batch.request_pool_indices,
            (geometry.row_bucket,),
        ),
        prefix_lens=cache_seqlens,
        query_lens=query_lens,
        out_cache_loc=_fixed_view(batch.out_cache_loc, (geometry.token_bucket,)),
        block_table=block_table,
        seq_lens=kv_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        output_indices=output_indices,
        prefix_lens_cpu=(*batch.prefix_lens_cpu, *(0 for _ in range(dummy_rows))),
        query_lens_cpu=(*batch.query_lens_cpu, *dummy_query_lens),
        seq_lens_cpu=(*batch.seq_lens_cpu, *dummy_query_lens),
        max_seqlen_q=geometry.max_query_len,
        max_seqlen_k=geometry.max_key_len,
        token_row_indices=tuple(range(geometry.row_bucket)),
        input_ids=input_ids,
        input_embeddings=input_embeddings,
        embedding_mask=embedding_mask,
        positions=positions,
        token_selections=(batch.token_selections[0],) * geometry.row_bucket,
    )


def _decode_signature(batch: ForwardBatch, geometry: _DecodeGeometry) -> tuple[object, ...]:
    """Build a decode graph signature from padded geometry and tensor contracts."""

    return (
        "paged_decode_bucket",
        geometry.bucket,
        geometry.width,
        batch.token_selections[0].value,
        _batch_tensor_signature(batch),
        bool(batch.causal),
        bool(batch.has_cache_writes),
    )


def _prefill_signature(batch: ForwardBatch, geometry: _PrefillGeometry) -> tuple[object, ...]:
    """Build a prefill graph signature from padded rows, tokens, and cache geometry."""

    return (
        "paged_prefill_bucket",
        geometry.row_bucket,
        geometry.token_bucket,
        geometry.width,
        geometry.max_query_len,
        geometry.max_key_len,
        batch.token_selections[0].value,
        _batch_tensor_signature(batch),
        bool(batch.causal),
        bool(batch.has_cache_writes),
    )


def _batch_tensor_signature(batch: ForwardBatch) -> tuple[object, ...]:
    """Describe all batch tensor leaves by dtype, shape, stride, and device."""

    assert batch.input_ids is not None and batch.positions is not None
    return (
        str(batch.input_ids.dtype),
        batch.input_ids.device.type,
        batch.positions.ndim,
        int(batch.positions.shape[0]) if batch.positions.ndim == 2 else 1,
        str(batch.positions.dtype),
        batch.input_embeddings is not None,
        None if batch.input_embeddings is None else str(batch.input_embeddings.dtype),
    )


def _exact_signature(batch: ForwardBatch) -> tuple[object, ...]:
    """Build a hashable signature for all graph-observable batch geometry."""

    # AR and denoising invoke the same model.forward entry. Row geometry and
    # selections below distinguish their executed math; encoders and the VAE
    # decoder use separate neural entry points even when tensor shapes match.
    entry = (
        "forward"
        if isinstance(batch.forward_mode, ForwardMode)
        or batch.forward_mode is PipelineStage.DENOISING
        else batch.forward_mode.value
    )
    return (
        entry,
        batch.row_count,
        batch.token_row_indices,
        batch.flow_row_indices,
        batch.attention_mode.value,
        batch.query_lens_cpu,
        tuple(value.value for value in batch.token_selections),
        batch.flow_image_tokens,
        batch.flow_heights,
        batch.flow_widths,
        tuple(value is not None for value in batch.flow_conditioning),
        tuple(_tensor_signature(value) for value in _tensor_leaves(batch)),
        batch.route_spans,
        batch.causal_rows_cpu,
        bool(batch.has_cache_writes),
        batch.max_seqlen_q,
        batch.max_seqlen_k,
    )


def _normalize_exact_batch(
    batch: ForwardBatch,
    *,
    context_blocks: int,
    block_size: int,
) -> ForwardBatch:
    """Give exact packed graphs their startup-fixed KV table geometry.

    Packed flow and mixed calls use request-variable KV prefix lengths, but the
    lane input buffer already owns a maximum-width, zero-scrubbed block
    table.  Capturing the active request-width view makes otherwise identical
    startup and serving calls different graph shapes.  Widening that view here
    keeps the physical kernel launch fixed while ``seqused_k`` and the other
    staged tensors carry the live lengths on every replay.
    """

    if batch.attention_mode is not AttentionMode.PACKED or context_blocks <= 0:
        return batch
    if batch.block_table is None:
        raise _GraphMiss("packed attention has no block table")
    if int(batch.block_table.shape[1]) > context_blocks:
        raise _GraphMiss("packed attention exceeds the configured context width")
    return replace(
        batch,
        block_table=_fixed_view(
            batch.block_table,
            (int(batch.block_table.shape[0]), int(context_blocks)),
        ),
        max_seqlen_k=int(context_blocks) * int(block_size),
    )


def _tensor_signature(value: torch.Tensor) -> tuple[object, ...]:
    """Describe one tensor's dtype, shape, stride, and device for graph identity."""

    return (
        tuple(int(extent) for extent in value.shape),
        str(value.dtype),
        value.device.type,
        tuple(value.stride()),
    )


def _graph_batch(
    batch: ForwardBatch,
    binding: int,
    *,
    own_inputs: bool,
) -> ForwardBatch:
    """Clone a batch into graph-owned inputs or bind its existing static tensors."""

    graph_batch = _clone_value(batch) if own_inputs else batch
    if not isinstance(graph_batch, ForwardBatch):
        raise TypeError("graph input cloning did not preserve ForwardBatch")
    if graph_batch.attention_mode is AttentionMode.REQUEST_INDEXED_DECODE:
        raise _GraphMiss("request-indexed decode metadata was not staged")
    return replace(
        graph_batch,
        binding=int(binding),
        cuda_graph_capture=True,
    )


def _clone_value(value: Any) -> Any:
    """Clone tensors recursively while retaining immutable structural values."""

    if isinstance(value, torch.Tensor):
        return value.clone(memory_format=torch.preserve_format)
    if isinstance(value, AttentionSelection):
        return value
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{field.name: _clone_value(getattr(value, field.name)) for field in fields(value)},
        )
    return value


_LEAF_FIELD_NAMES: dict[type, tuple[str, ...] | None] = {}


def _leaf_field_names(value: Any) -> tuple[str, ...] | None:
    """List structural paths to every tensor leaf in a nested batch value."""

    kind = type(value)
    names = _LEAF_FIELD_NAMES.get(kind, ())
    if names == ():
        names = (
            tuple(field.name for field in fields(value))
            if is_dataclass(value) and not isinstance(value, type)
            else None
        )
        _LEAF_FIELD_NAMES[kind] = names
    return names


def _tensor_leaves(value: Any) -> Iterator[torch.Tensor]:
    """Collect tensor leaves from nested tuples, lists, mappings, and dataclasses."""

    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, AttentionSelection):
        return
    if isinstance(value, tuple):
        for item in value:
            yield from _tensor_leaves(item)
        return
    names = _leaf_field_names(value)
    if names is not None:
        for name in names:
            yield from _tensor_leaves(getattr(value, name))


def _copy_into_leaves(
    target_tensors: tuple[torch.Tensor, ...],
    source: object,
    structure: str,
) -> None:
    """Copy source tensor leaves into a previously captured structure."""

    index = 0
    limit = len(target_tensors)
    for value in _tensor_leaves(source):
        if index >= limit:
            raise _GraphMiss(f"{structure} tensor structure changed")
        destination = target_tensors[index]
        index += 1
        if (
            destination.shape != value.shape
            or destination.dtype != value.dtype
            or destination.device != value.device
        ):
            raise _GraphMiss(f"{structure} tensor geometry changed")
        destination.copy_(value, non_blocking=True)
    if index != limit:
        raise _GraphMiss(f"{structure} tensor structure changed")


def _graph_provider(
    selection: AttentionSelection,
    mode: AttentionMode,
    *,
    head_dim: int,
    block_size: int,
    device: torch.device,
):
    """Resolve the geometry-bound paged-attention backend for a graph mode."""

    for provider in selection.providers:
        if not provider.can_bind(
            mode,
            head_dim=head_dim,
            block_size=block_size,
            device=device,
        ):
            continue
        if provider.can_bind(
            mode,
            head_dim=head_dim,
            block_size=block_size,
            device=device,
            cuda_graph=True,
        ):
            return provider
        break
    raise _GraphMiss("no provisioned attention provider is graph-safe")


_PLAN_LENGTH_BOUNDS = frozenset({"max_seqlen_q", "max_seqlen_k"})


def _live_attention(static: object, live: object) -> object:
    """Project live attention metadata into the static batch's bounded shapes."""

    if not is_dataclass(static) or not is_dataclass(live):
        raise _GraphMiss("attention plan is not immutable data")
    updates: dict[str, object] = {}
    for field in fields(static):
        static_value = getattr(static, field.name)
        if (
            isinstance(static_value, torch.Tensor)
            or field.name == "binding"
            or field.name in _PLAN_LENGTH_BOUNDS
        ):
            updates[field.name] = static_value
        else:
            updates[field.name] = getattr(live, field.name)
    return replace(cast(Any, live), **updates)


def _fixed_view(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Return a bounded tensor view after validating the requested element count."""

    if tensor.ndim != len(shape) or any(value < 0 for value in shape):
        raise _GraphMiss("fixed graph view rank changed")
    strides = tuple(int(value) for value in tensor.stride())
    if any(value < 0 for value in strides):
        raise _GraphMiss("fixed graph view has a negative stride")
    maximum = int(tensor.storage_offset())
    for extent, stride in zip(shape, strides, strict=True):
        if extent:
            maximum += (int(extent) - 1) * stride
    storage_elements = tensor.untyped_storage().nbytes() // tensor.element_size()
    if maximum >= storage_elements:
        raise _GraphMiss("graph bucket exceeds its fixed input storage")
    return tensor.as_strided(shape, strides, storage_offset=int(tensor.storage_offset()))


def _expand_token_axis(tensor: torch.Tensor, tokens: int) -> torch.Tensor:
    """Expand a single-token tensor view to a fixed graph token extent."""

    if tensor.ndim == 1:
        return _fixed_view(tensor, (tokens,))
    if tensor.ndim == 2:
        return _fixed_view(tensor, (int(tensor.shape[0]), tokens))
    raise _GraphMiss("graph token positions have an invalid rank")


def _cuda_batch(batch: ForwardBatch) -> bool:
    """Move all tensor leaves in a forward batch onto its execution device."""

    tensors = tuple(_tensor_leaves(batch))
    return bool(
        tensors
        and torch.cuda.is_available()
        and all(value.device.type == "cuda" for value in tensors)
        and len({value.device for value in tensors}) == 1
    )


def _batch_device(batch: ForwardBatch) -> torch.device:
    """Return the unique device that owns a forward batch's tensor leaves."""

    for tensor in _tensor_leaves(batch):
        return tensor.device
    raise _GraphMiss("forward batch carries no device tensors")


def _attention_tensor_leaves(batch: ForwardBatch) -> Iterator[torch.Tensor]:
    """Collect every tensor whose address participates in attention graph capture."""

    for name in (
        "request_pool_indices",
        "prefix_lens",
        "query_lens",
        "out_cache_loc",
        "block_table",
        "seq_lens",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "output_indices",
        "attention_indexes",
        "visible_end",
    ):
        value = getattr(batch, name)
        if isinstance(value, torch.Tensor):
            yield value


def _quantized_kv(runner: PackedRunner) -> bool:
    """Return whether the graph runner reads scale-aware quantized KV storage."""

    return bool(runner.cache_pool.is_quantized)


def _private_pool_bytes(device: torch.device | None) -> int:
    """Return CUDA allocator bytes held outside active and reserved pools."""

    if device is None or not torch.cuda.is_available():
        return 0
    index = device.index if device.index is not None else torch.cuda.current_device()
    total = 0
    for segment in torch.cuda.memory_snapshot():
        if segment.get("device") != index:
            continue
        pool_id = segment.get("segment_pool_id")
        if isinstance(pool_id, tuple) and any(pool_id):
            total += int(segment.get("total_size", 0))
    return total


def _trim_output(output: ForwardOutput, rows: int) -> ForwardOutput:
    """Slice every forward-output row tensor to the live batch extent."""

    return ForwardOutput(tuple(output.values[:rows]), output.vocabularies[:rows])


def _greedy_decode(
    batch: ForwardBatch,
    output: ForwardOutput,
    predicate_state: torch.Tensor | None,
) -> GraphGreedyOutput | None:
    """Return graph-capturable greedy output for eligible decode logits."""

    return _greedy_decode_values(
        batch,
        output,
        predicate_state,
        batch.decode_force_finish,
        clear_force_finish=True,
    )


def _greedy_decode_values(
    batch: ForwardBatch,
    output: ForwardOutput,
    predicate_state: torch.Tensor | None,
    force_finish: torch.Tensor | None,
    *,
    clear_force_finish: bool,
) -> GraphGreedyOutput | None:
    """Derive graph-capturable greedy tokens and continuation state from model logits."""

    if (
        batch.attention_mode is not AttentionMode.PAGED_DECODE
        or predicate_state is None
        or force_finish is None
        or len(output.values) != batch.row_count
        or batch.token_row_indices != tuple(range(batch.row_count))
        or any(selection is not TokenSelection.LAST_LOGITS for selection in batch.token_selections)
        or batch.flow_row_indices
    ):
        return None
    rows = tuple(value.reshape(-1) for value in output.values)
    logits = packed_tensor_views(rows)
    if logits is None:
        raise _GraphMiss("decode logits are not one contiguous graph output")
    logits = logits.reshape(batch.row_count, -1)
    from ...nn.logits import greedy_vocabulary

    partitions = output.vocabularies
    if any(partition != partitions[0] for partition in partitions):
        return None
    max_values, tokens = greedy_vocabulary(logits, partitions[0])
    valid = torch.isfinite(max_values)
    active = predicate_state.index_select(0, batch.request_pool_indices.reshape(-1))
    finish = force_finish.reshape(-1) & valid & active
    continuation = valid & active & ~finish
    tags = torch.where(continuation, TOKEN_CONTINUATION_BIT, 0)
    tagged_tokens = tokens.bitwise_or(tags)
    completion = torch.cat(
        (
            valid,
            active,
            tokens,
            torch.zeros_like(tokens),
        )
    )
    if clear_force_finish:
        force_finish.zero_()
    return GraphGreedyOutput(
        request_pool_indices=batch.request_pool_indices,
        tokens=tokens,
        valid=valid,
        active=active,
        finish=finish,
        continuation=continuation,
        tagged_tokens=tagged_tokens,
        completion=completion,
    )


def _trim_greedy(
    output: GraphGreedyOutput | None,
    rows: int,
) -> GraphGreedyOutput | None:
    """Slice padded graph-greedy output tensors back to the live row count."""

    if output is None:
        return None
    total = int(output.tokens.numel())
    if rows < 0 or rows > total or int(output.completion.numel()) != 4 * total:
        raise GraphExecutionError("CUDA graph greedy output has invalid row geometry")
    completion = torch.cat(
        tuple(output.completion[index * total : index * total + rows] for index in range(4))
    )
    return GraphGreedyOutput(
        request_pool_indices=output.request_pool_indices[:rows],
        tokens=output.tokens[:rows],
        valid=output.valid[:rows],
        active=output.active[:rows],
        finish=output.finish[:rows],
        continuation=output.continuation[:rows],
        tagged_tokens=output.tagged_tokens[:rows],
        completion=completion,
    )


def _equivalence_check(reference: ForwardOutput, candidate: ForwardOutput) -> torch.Tensor:
    """Compare direct and captured forward outputs across every tensor field."""

    if len(reference.values) != len(candidate.values):
        raise GraphExecutionError("CUDA graph output count differs from eager execution")
    checks: list[torch.Tensor] = []
    for expected, actual in zip(reference.values, candidate.values, strict=True):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise GraphExecutionError("CUDA graph output geometry differs from eager execution")
        if not (expected.is_floating_point() or expected.is_complex()):
            checks.append(torch.eq(expected, actual).all())
            continue
        tolerance = 0.01 if expected.element_size() <= 2 else 1e-5
        checks.append(
            torch.isclose(
                actual,
                expected,
                rtol=tolerance,
                atol=tolerance,
                equal_nan=False,
            ).all()
        )
    if not checks:
        device = reference.values[0].device if reference.values else torch.device("cpu")
        return torch.ones((), dtype=torch.bool, device=device)
    return torch.stack(tuple(checks)).all()


def _release_call(method: Callable[[int], object], binding: int) -> Callable[[], None]:
    """Bind a graph resource release method to one capture identity."""

    def release() -> None:
        """Release the captured backend binding identified by this closure."""

        method(binding)

    return release
