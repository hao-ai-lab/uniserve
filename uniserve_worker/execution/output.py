"""Numerical captures and borrowed staging for native PendingOutput results.

Python stages sampling, canvas, latent and host work into these per-call views.
Rust owns readiness, result decoding, request acceptance and output retirement.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from uniserve_worker._uniserve_ipc import PendingOutput
from uniserve_worker.execution.host import HostTask
from uniserve_worker.protocol.batch import LatentParams
from uniserve_worker.protocol.identity import BufferId
from uniserve_worker.protocol.output import MediaOutput
from uniserve_worker.sampling.result import LogprobValues, SamplerRow
from uniserve_worker.storage.latent_pool import LatentStaging, LatentUpdate
from uniserve_worker.storage.output import OutputBuffer
from uniserve_worker.transport.exports import ExportLocations

__all__ = [
    "PendingOutput",
]


def capture_logprobs(
    details: LogprobValues | None,
    output: OutputBuffer,
) -> dict[int, tuple[int, int, int]]:
    """Capture one packed logprob column into completion storage.

    Rust retains the row layout and decodes the column after its copy finishes.

    Returns:
        A map from each sampler row index that carries logprobs to its
        `(offset, count, row)` span; empty when `details` is None.
    """
    if details is None:
        return {}
    packed, rows, counts, requested_ids, max_count, max_requested = details
    capture = output.capture(packed)
    output.register_logprobs(
        capture,
        rows,
        counts,
        requested_ids,
        max_count,
        max_requested,
    )
    return {index: (*capture, index) for index in rows}


def capture_samples(
    samples: Sequence[SamplerRow],
    requests: Sequence[PendingOutput],
    output: OutputBuffer,
) -> None:
    """Attach row ranges while copying each shared sampling column only once.

    Call on the producer stream before sealing the output buffer. Its fence
    protects all sampling and score ranges until their PendingOutput retires.

    Raises:
        ValueError: `samples` and `requests` differ in length.
    """
    spans: dict[int, tuple[int, int]] = {}
    details: dict[int, dict[int, tuple[int, int, int]]] = {}
    for sample, request in zip(samples, requests, strict=True):
        metadata = sample.batch.completion
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
    logprob_range: tuple[int, int, int] | None = None
    prompt_logprob_ranges: tuple[tuple[int, int, int], ...] = ()
    sampled: SamplerRow | None = None
    # A canvas readout's `(offset, count)` span: one word per candidate,
    # holding its FP32 log-probability's bits sign-extended from int32.
    candidate_range: tuple[int, int] | None = None
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
