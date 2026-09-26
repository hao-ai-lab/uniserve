"""Per-call completion records and their materialization into results.

`reserve_outputs` in `uniserve_worker.execution.prepare` calls
`create_outputs` to create one `PendingOutput` per call of a batch, each bound
to one row of the batch's pinned `OutputBuffer`. Execution then stages the
call's domain results into it: sampling captures, canvas readouts and
speculative verification coordinates (`TokenResult`), latent trajectory updates
(`LatentResult`), host tasks and media (`HostResult`), and the projected
`RequestProgress`.

After the batch commits, `uniserve_worker.execution.commit` drops the
borrowed device references with `release_execution_references`. The executor
polls `ready`, calls `materialize` once to resolve the wire `RequestOutput`,
and passes `request_result` to `RequestPool.apply_result`. A discarded batch
calls `abandon` instead. No method here installs request progress itself.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Final

import torch

from uniserve.runtime.resources import close_resources
from uniserve_worker.execution.host import HostTask
from uniserve_worker.execution.request import (
    RequestPool,
    RequestProgress,
    RequestResult,
    RequestState,
)
from uniserve_worker.media.storage import publish_media_bytes
from uniserve_worker.protocol.batch import LatentParams, TensorPublication
from uniserve_worker.protocol.call import Call, CallKind, CallStatus, ErrorCode
from uniserve_worker.protocol.identity import BufferId, CallId, RequestKey
from uniserve_worker.protocol.output import (
    FinishFlags,
    MediaOutput,
    PosixShmArtifact,
    RequestOutput,
    TimingCounters,
)
from uniserve_worker.protocol.transfer import KvTransfer, Locator
from uniserve_worker.sampling.result import LogprobValues, SamplerRow
from uniserve_worker.storage.latent_pool import LatentStaging, LatentUpdate
from uniserve_worker.storage.output import OutputBuffer
from uniserve_worker.storage.tensor_store import TensorRead, TensorRecord
from uniserve_worker.transport.exports import ExportLocations

logger = logging.getLogger(__name__)

__all__ = [
    "PendingOutput",
]

# The sampler packs one sampled batch's completion column as four consecutive
# fields, [valid | active | token | accepted], each `count` rows wide
# (`sampling_columns` in `uniserve_worker.sampling.sampler`). `valid` marks a
# usable sampling distribution, `active` the resolved device predicate, `token`
# the selected token, and `accepted` the number of accepted draft tokens (zero
# without speculation). This must equal `SAMPLING_COMPLETION_FIELDS`, which
# `reserve_outputs` uses to size completion storage.
_SAMPLING_FIELDS_PER_CALL: Final[int] = 4


# Control-flow signals raised by `sampled_tokens`; `PendingOutput.materialize`
# maps an invalid distribution to `ErrorCode.INVALID_CALL` and a predicated
# call to `CallStatus.PREDICATED`. Their names describe the condition instead
# of ending in "Error".
class _InvalidSamplingDistribution(RuntimeError):  # noqa: N818
    """Marks a sampling row whose filtered probability mass is unusable."""

    pass


class _PredicatedCall(RuntimeError):  # noqa: N818  # deliberate taxonomy name
    """Marks a call suppressed by its resolved device predicate."""

    pass


def capture_logprobs(
    details: LogprobValues | None,
    output: OutputBuffer,
) -> dict[int, tuple[int, int, int]]:
    """Capture one packed logprob column into completion storage.

    The column's row layout is recorded in `output.logprob_layouts` under the
    capture span so `OutputBuffer.logprob_values` can decode it after the copy
    completes.

    Returns:
        A map from each sampler row index that carries logprobs to its
        `(offset, count, row)` span; empty when `details` is None.
    """
    if details is None:
        return {}
    packed, rows, counts, requested_ids, max_count, max_requested = details
    capture = output.capture(packed)
    key = capture
    output.logprob_layouts[key] = (
        rows,
        counts,
        requested_ids,
        max_count,
        max_requested,
    )
    return {index: (*key, index) for index in rows}


def capture_samples(
    samples: Sequence[SamplerRow],
    requests: Sequence[PendingOutput],
    output: OutputBuffer,
) -> None:
    """Attach row ranges while copying each shared sampling column only once.

    Call on the producer stream before sealing the output buffer. Its fence
    protects all sampling and score ranges until their PendingOutput retires.

    Raises:
        RuntimeError: A completion column is not a whole number of
            `_SAMPLING_FIELDS_PER_CALL` fields or a row index lies outside
            it. Errors from `OutputBuffer.capture` also propagate.
        ValueError: `samples` and `requests` differ in length.
    """
    spans: dict[int, tuple[int, int]] = {}
    details: dict[int, dict[int, tuple[int, int, int]]] = {}
    for sample, request in zip(samples, requests, strict=True):
        metadata = sample.batch.completion
        count = int(metadata.numel()) // _SAMPLING_FIELDS_PER_CALL
        if metadata.numel() != count * _SAMPLING_FIELDS_PER_CALL or not (
            0 <= sample.index < count
        ):
            raise RuntimeError("sampling completion vectors do not align")

        # Rows of a shared batch reference one completion column; capture it
        # on first encounter and hand each request its row span. Columns are
        # keyed by object identity, which stays unique because `samples`
        # keeps every column alive for the duration of this loop.
        key = id(metadata)
        span = spans.get(key)
        if span is None:
            capture = output.capture(metadata)
            span = capture
            spans[key] = span
        request.token.sampling_range = (*span, sample.index)

        if sample.batch.logprobs is not None:
            key = id(sample.batch.logprobs)
            if key not in details:
                details[key] = capture_logprobs(sample.batch.logprobs, output)
            request.token.logprob_range = details[key].get(sample.index)


def sampled_tokens(record: PendingOutput) -> tuple[int, ...]:
    """Resolve validity and acceptance from one captured sampling row.

    Requires the output buffer's copy to be complete. The whole captured
    column is read once and cached in `record.token.sampling_values`.

    Returns:
        `record.token.committed_tokens` when the call captured no sampling
        row. Otherwise the sampled token alone, or for speculative
        verification the accepted draft prefix followed by the sampled token.
        When acceptance reaches `terminal_prefix`, the accepted drafts already
        end in the terminal token and no sampled token is appended.

    Raises:
        _PredicatedCall: The row's resolved predicate is inactive. This is
            checked before validity.
        _InvalidSamplingDistribution: The row's distribution is invalid.
        RuntimeError: The record lost its output buffer before the values
            were cached, or the acceptance count lies outside the draft span.
            Errors from `OutputBuffer.read_tokens` also propagate.
    """
    if record.token.sampling_range is None:
        return record.token.committed_tokens
    offset, extent, index = record.token.sampling_range
    values = record.token.sampling_values
    if values is None:
        if record._buffer is None:
            raise RuntimeError("sampling output lost its pinned range")
        values = record._buffer.read_tokens(offset, extent)
        record.token.sampling_values = values
    # Field layout of the packed sampling column is documented at
    # _SAMPLING_FIELDS_PER_CALL.
    count = extent // _SAMPLING_FIELDS_PER_CALL
    if not bool(values[count + index]):
        raise _PredicatedCall("call predicate selected no state")
    if not bool(values[index]):
        raise _InvalidSamplingDistribution(
            "sampling policy produced an invalid distribution"
        )
    token = values[count * 2 + index]
    accepted = values[count * 3 + index]

    draft = record.token.draft_tokens
    if not draft:
        return (token,)
    if accepted < 0 or accepted > len(draft):
        raise RuntimeError(
            "speculative acceptance count is outside the draft span"
        )
    if (
        record.token.terminal_prefix is not None
        and accepted >= record.token.terminal_prefix
    ):
        return draft[:accepted]
    return (*draft[:accepted], token)


def logprob_entries(record: PendingOutput, span: tuple[int, int, int]) -> int:
    """Return the number of score entries one logprob row can report.

    The count is the sampled entry plus the row's top-k count plus its
    explicitly requested token ids. `uniserve_worker.execution.commit` uses it
    to bound the logprob payload against `max_completion_bytes`.
    """
    if record._buffer is None:
        raise RuntimeError("logprob output lost its pinned range")
    offset, count, index = span
    rows, counts, requested_ids, _max_count, _max_requested = (
        record._buffer.logprob_layouts[(offset, count)]
    )
    local = rows.index(index)
    return 1 + counts[local] + len(requested_ids[local])


def create_outputs(
    requests: RequestPool,
    calls: Sequence[Call],
    request_pool_indices: Sequence[int],
    buffer: OutputBuffer,
) -> tuple[PendingOutput, ...]:
    """Bind output rows to validated, admitted requests for one batch.

    Row `i` of `buffer` belongs to `calls[i]`. Slot and identity validation is
    done by `RequestPool.bind_calls`, whose errors propagate.
    """
    bindings = requests.bind_calls(calls, request_pool_indices)
    return tuple(
        PendingOutput(call, request, buffer, index)
        for index, (call, request) in enumerate(
            zip(calls, bindings, strict=True)
        )
    )


@dataclass(slots=True)
class TokenResult:
    """Sampling captures and numerical token-state updates for one call."""

    # Tokens reported when no sampling row was captured.
    committed_tokens: tuple[int, ...] = ()
    # Spans are `(offset, count, row)` into the call's `OutputBuffer`: the
    # capture's element offset and length, and the row within the captured
    # column (the call's sampler row for `sampling_range` and `logprob_range`,
    # the scored token's index within its prompt chunk for each
    # `prompt_logprob_ranges` entry).
    sampling_range: tuple[int, int, int] | None = None
    sampling_values: tuple[int, ...] | None = None
    logprob_range: tuple[int, int, int] | None = None
    prompt_logprob_ranges: tuple[tuple[int, int, int], ...] = ()
    logprobs: tuple[float, tuple[tuple[int, float, int], ...]] | None = None
    prompt_logprobs: tuple[tuple[tuple[int, float, int], ...], ...] = ()
    sampled: SamplerRow | None = None
    # A canvas readout's `(offset, count)` span: one word per candidate,
    # holding its FP32 log-probability's bits sign-extended from int32.
    candidate_range: tuple[int, int] | None = None
    candidate_logprobs: tuple[float, ...] = ()
    # A canvas step's `(offset, count)` span: its stop flag, then its canvas
    # tokens, which become the committed tokens when the flag is set.
    canvas_range: tuple[int, int] | None = None

    # These borrowed numerical views survive until DecodeState accepts them.
    runtime_logical_position: int | torch.Tensor = 0
    runtime_sampling_position: int | torch.Tensor = 0
    runtime_penalty_base: torch.Tensor | None = None
    runtime_decode_increment: bool = False
    runtime_cache_length: int | torch.Tensor | None = None
    runtime_prompt_logits: torch.Tensor | None = None

    # Speculative verification (set when `draft_tokens` is not None). The
    # device selects the accepted span; host completion resolves only the
    # accepted count, without retaining logits. The `base_*` fields are the
    # coordinates before the draft span, to which `PendingOutput.materialize`
    # adds the accepted token count. `initialized_kv` is the KV extent the
    # verification forward wrote; rejected drafts stay initialized but beyond
    # the visible length.
    draft_tokens: tuple[int, ...] | None = None
    terminal_prefix: int | None = None
    base_logical_position: int = 0
    base_rng_counter: int = 0
    base_kv_visible: int = 0
    initialized_kv: int = 0


@dataclass(slots=True)
class LatentResult:
    """Borrowed trajectory inputs, prepared storage, and visibility updates."""

    update: LatentUpdate
    input_params: LatentParams | None = None
    staging: LatentStaging | None = None
    imported: bool = False
    exports: dict[BufferId, ExportLocations] = field(default_factory=dict)


@dataclass(slots=True)
class HostResult:
    """Host tasks and media results following numerical execution."""

    tasks: tuple[HostTask, ...] = ()
    # When set, `PendingOutput.materialize` passes the task results to this
    # callback (for example to publish deferred product bytes) instead of
    # interpreting them as media output.
    finish: Callable[[tuple[object, ...]], None] | None = None
    media: MediaOutput | None = None


class PendingOutput:
    """One call's projected progress and domain-specific completion records.

    The output row and CPU tasks retain their actual storage until materialized
    or abandoned. No request progress is installed by this object: RequestPool
    accepts explicit result updates after materialization.

    Execution code in `uniserve_worker.execution` writes the public fields
    directly while staging the call. `uniserve_worker.execution.commit` sets
    `_reports_output` to whether this rank is the component's output rank;
    only that rank reports scores and publishes host-produced media bytes.
    """

    def __init__(
        self,
        call: Call,
        request: RequestState,
        buffer: OutputBuffer,
        row: int,
    ) -> None:
        self.call = call
        self.request = request
        self.token = TokenResult()
        self.latent = LatentResult(LatentUpdate(request.request_pool_idx))
        self.host = HostResult()
        # The call states the coordinates it runs at, so the rank reads them
        # rather than deriving them from a predecessor's record. The device
        # state a previous call left behind stays with the request.
        self.progress: RequestProgress = RequestProgress(
            logical_position=call.coordinates.logical_position,
            rng_counter=request.rng_counter,
            flow_step=call.coordinates.flow_step,
            kv_visible_len=call.coordinates.kv_visible_len,
            kv_computed_len=call.coordinates.kv_computed_len,
            prompt_logits_ready=request.prompt_logits_ready,
        )
        self.accepted_progress: RequestProgress | None = None

        self.status = CallStatus.OK
        self.finish_flags = FinishFlags()
        self.product_generations: tuple[int, ...] = ()
        self.error_code: ErrorCode | None = None

        # Exports, store reads and writes, and the predicate are borrowed from
        # their owning stores until the batch commits or is discarded. On
        # both paths `uniserve_worker.execution.commit` completes and clears
        # the reads, then `release_execution_references` drops the rest. Host
        # acceptance uses only `progress` and the completion ranges,
        # independently of these device references.
        self.tensor_exports: dict[BufferId, ExportLocations] = {}
        self.cache_exports: dict[BufferId, ExportLocations] = {}
        self.exported_locators: list[Locator] = []
        self.cache_publication: tuple[BufferId, KvTransfer] | None = None
        self.cache_installation: (
            tuple[BufferId, BufferId, KvTransfer] | None
        ) = None
        self.device_reads: list[TensorRead] = []
        self.feature_reads: list[TensorRead] = []
        self.writes: list[TensorRecord] = []
        self.predicate: tuple[torch.Tensor, bool] | None = None
        self.token_write: TensorRecord | None = None
        self.transition_write: TensorRecord | None = None
        self.completion_write: TensorRecord | None = None
        self.producer_write: TensorRecord | None = None

        self.kv_output: KvTransfer | None = None
        # When host tasks produce a product's bytes, `products` is empty at
        # commit; the `host.finish` callback publishes the product once the
        # tasks complete and sets it here. Its consumer is scheduled only
        # after this call completes, so the bytes are in place before any
        # rank can read them.
        self.products: tuple[TensorPublication, ...] = ()

        # `_row` and `_generation` identify this call's row in the leased
        # `OutputBuffer`. A generation from an earlier lease of the same
        # buffer makes `OutputBuffer.observe` raise and `discard` a no-op.
        self._buffer: OutputBuffer | None = buffer
        self._row = int(row)
        self._generation = int(buffer.generation)
        self._completion_timing: tuple[int, int, int, int] | None = None
        self._observed = False

        self._reports_output = True
        self.value: RequestOutput | None = None

    def release_execution_references(self) -> None:
        """Drop borrowed device views once their stores own the lifetimes.

        Call only after the stores have committed or abandoned this call's
        writes and fenced its reads; `uniserve_worker.execution.commit` does
        so on both the commit and the discard path. Host completion may
        outlive every product, so this record must not keep these numerical
        allocations alive after their owners free them. `progress` and the
        completion ranges are kept for materialization.
        """
        self.writes.clear()
        self.tensor_exports.clear()
        self.cache_exports.clear()
        self.latent.exports.clear()
        self.exported_locators.clear()
        self.cache_publication = None
        self.cache_installation = None

        self.latent.input_params = None
        self.latent.staging = None
        self.latent.imported = False

        self.predicate = None
        self.token_write = None
        self.transition_write = None
        self.completion_write = None
        self.producer_write = None
        self.token.sampled = None

        self.token.runtime_penalty_base = None
        self.token.runtime_prompt_logits = None
        self.token.runtime_cache_length = None
        self.token.runtime_logical_position = 0
        self.token.runtime_sampling_position = 0

    @property
    def request_key(self) -> RequestKey:
        return self.call.request_key

    @property
    def call_id(self) -> CallId:
        return self.call.call_id

    @property
    def kind(self) -> CallKind:
        return self.call.kind

    def ready(self) -> bool:
        """Query host output readiness without changing request acceptance."""
        if self.value is not None:
            return True
        if self._buffer is None or not self._buffer.ready():
            return False
        for task in self.host.tasks:
            if not task.ready():
                return False
        return True

    def materialize(self) -> RequestOutput:
        """Resolve output fields once.

        A call that did not run keeps the request's committed coordinates, so
        its completion reports where the request still stands.

        Outcomes other than success:

        - Any exception raised while resolving host task results (including
          `host.finish` and media publication) or decoding logprobs is
          logged and reported as `CallStatus.ERROR` with
          `ErrorCode.COMPUTE_ERROR`.
        - An inactive sampled predicate (or a call already marked predicated)
          reports `CallStatus.PREDICATED` at the request's accepted progress.
        - An invalid sampling distribution reports `CallStatus.ERROR` with
          `ErrorCode.INVALID_CALL`.

        These suppressed outcomes report no tokens, scores, product
        generations, finish flags, or KV output; `media_output` is reported
        whenever it was resolved, whatever the status. For any `PREDICATED` or
        `ERROR` status, `accepted_progress` stays the request's accepted
        progress; otherwise it is the resolved projection.

        Side effects: observes this call's `OutputBuffer` row, records
        `accepted_progress`, releases the buffer and host task references,
        and caches the result in `value`.

        Raises:
            RuntimeError: The output is not ready (which includes an
                abandoned output), or a speculative sampling row disagrees
                with its draft span or initialized KV extent. Errors from
                `OutputBuffer.read_tokens`, `OutputBuffer.observe`, and
                `RequestOutput.validate` also propagate.
        """
        if self.value is not None:
            return self.value
        if not self.ready():
            raise RuntimeError("completion was resolved before query-ready")
        accepted_parent = self.request.accepted_progress

        status = self.status
        error_code = self.error_code
        runtime = self.progress
        tokens = self.token.committed_tokens
        suppressed = status is CallStatus.PREDICATED
        if not suppressed:
            try:
                # A `finish` callback consumes every host task result. Without
                # one, bytes become a POSIX shared-memory media artifact on the
                # output rank, and at most one result may be media output.
                results = tuple(task.result() for task in self.host.tasks)
                finish, self.host.finish = self.host.finish, None
                if finish is not None:
                    finish(results)
                    results = ()
                for result in results:
                    if isinstance(result, bytes) and self._reports_output:
                        result = MediaOutput(
                            handle=PosixShmArtifact(
                                name=publish_media_bytes(result)
                            ),
                            bytes=len(result),
                        )
                    if isinstance(result, MediaOutput):
                        if self.host.media is not None:
                            raise RuntimeError(
                                "completion produced more than one media output"
                            )
                        self.host.media = result

                if self._buffer is None:
                    raise RuntimeError(
                        "completion lost its pinned output buffer"
                    )
                if self.token.logprob_range is not None:
                    self.token.logprobs = self._buffer.logprob_values(
                        self.token.logprob_range
                    )
                self.token.prompt_logprobs = tuple(
                    self._buffer.logprob_values(span)[1]
                    for span in self.token.prompt_logprob_ranges
                )
                if self.token.candidate_range is not None:
                    words = self._buffer.read_tokens(
                        *self.token.candidate_range
                    )
                    self.token.candidate_logprobs = struct.unpack(
                        f"<{len(words)}f",
                        struct.pack(f"<{len(words)}i", *words),
                    )
                if self.token.canvas_range is not None:
                    stopped, *canvas = self._buffer.read_tokens(
                        *self.token.canvas_range
                    )
                    self.token.committed_tokens = (
                        tuple(canvas) if stopped else ()
                    )
            except Exception:
                logger.exception(
                    "completion materialization failed: request=%s "
                    "call=%s computation=%s",
                    self.request_key,
                    self.call_id,
                    self.kind,
                )
                status = CallStatus.ERROR
                error_code = ErrorCode.COMPUTE_ERROR
                suppressed = True
            else:
                try:
                    tokens = sampled_tokens(self)
                except _PredicatedCall:
                    status = CallStatus.PREDICATED
                    suppressed = True
                except _InvalidSamplingDistribution:
                    status = CallStatus.ERROR
                    error_code = ErrorCode.INVALID_CALL
                    suppressed = True
                else:
                    if self.token.sampling_range is not None:
                        if runtime is None:
                            raise RuntimeError(
                                "sampling output has no request progress"
                            )
                        if self.token.draft_tokens is not None:
                            # Acceptance adds at most one sampled token to the
                            # drafts and must stay within the KV extent the
                            # verification forward initialized.
                            accepted = len(tokens)
                            visible = self.token.base_kv_visible + accepted
                            if (
                                accepted > len(self.token.draft_tokens) + 1
                                or runtime.kv_computed_len
                                != self.token.initialized_kv
                                or visible > runtime.kv_computed_len
                            ):
                                raise RuntimeError(
                                    "speculative acceptance exceeds "
                                    "initialized KV state"
                                )
                            # Rejected drafts remain initialized but invisible.
                            # Resolve logical, RNG and visible KV coordinates
                            # together once.
                            runtime = replace(
                                runtime,
                                logical_position=self.token.base_logical_position
                                + accepted,
                                rng_counter=self.token.base_rng_counter
                                + accepted,
                                kv_visible_len=visible,
                            )

        # A predicated call did not run, so it reports the request's accepted
        # coordinates and carries no error code.
        if status is CallStatus.PREDICATED:
            runtime = accepted_parent
            error_code = None
        if suppressed:
            tokens = ()
        scores = (
            None
            if suppressed or not self._reports_output
            else self.token.logprobs
        )

        buffer = self._buffer
        if buffer is None:
            raise RuntimeError("completion lost its pinned output buffer")
        buffer.observe(self._row, self._generation)
        self._completion_timing = buffer.timing()
        self._observed = True
        self._buffer = None

        timing = TimingCounters(
            queued_us=self._completion_timing[0],
            device_us=self._completion_timing[1],
            copy_us=self._completion_timing[2],
            host_us=self._completion_timing[3],
        )
        concrete = RequestOutput(
            request_key=self.request_key,
            call_id=self.call_id,
            status=status,
            product_generations=() if suppressed else self.product_generations,
            error_code=error_code,
            timing_counters=timing,
            kind=self.kind,
            position=0 if runtime is None else int(runtime.logical_position),
            kv_visible_len=0
            if runtime is None
            else int(runtime.kv_visible_len),
            kv_computed_len=0
            if runtime is None
            else int(runtime.kv_computed_len),
            num_completed_steps=0
            if runtime is None
            else int(runtime.flow_step),
            committed_tokens=tokens,
            sampled_logprob=None if scores is None else scores[0],
            top_logprobs=() if scores is None else scores[1],
            prompt_logprobs=(
                ()
                if suppressed or not self._reports_output
                else self.token.prompt_logprobs
            ),
            candidate_logprobs=(
                ()
                if suppressed or not self._reports_output
                else self.token.candidate_logprobs
            ),
            finish_flags=FinishFlags() if suppressed else self.finish_flags,
            media_output=self.host.media,
            kv_output=None if suppressed else self.kv_output,
        )
        # Successful calls accept their projection, resolved above for
        # speculative verification; predicated and failed calls keep the
        # request's accepted progress.
        self.accepted_progress = (
            accepted_parent
            if status in (CallStatus.PREDICATED, CallStatus.ERROR)
            else runtime
        )

        concrete.validate()
        self.value = concrete
        self.host.tasks = ()
        return concrete

    def request_result(self) -> RequestResult:
        """Return the explicit acceptance update after materialization."""
        if self.value is None:
            raise RuntimeError("request output has not been materialized")
        return RequestResult(
            self.request_key,
            self.call_id,
            self.value.status,
            self.accepted_progress,
        )

    def abandon(self) -> None:
        """Stop result delivery without waiting for in-flight readers.

        Host tasks not yet submitted are cancelled; submitted ones run to
        completion. The output row is discarded if it was never observed;
        when it is the buffer's last row, `OutputBuffer.discard` defers
        releasing the buffer's events until in-flight copies complete. Every
        release is attempted and the first failure is raised with later ones
        noted. Calling it again is a no-op.
        """
        tasks, self.host.tasks = self.host.tasks, ()
        self.host.finish = None
        actions = [task.abandon for task in tasks]

        buffer, self._buffer = self._buffer, None
        if buffer is not None and not self._observed:
            self._observed = True
            actions.append(partial(buffer.discard, self._row, self._generation))

        close_resources(*actions)
