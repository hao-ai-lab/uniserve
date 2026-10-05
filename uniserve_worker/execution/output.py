"""Numerical captures and borrowed views for native PendingOutput results.

Python supplies sampling, canvas and latent tensors through these views.
Rust owns readiness, result decoding, request acceptance and output retirement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from uniserve_worker._uniserve_ipc import PendingOutput
from uniserve_worker.protocol.batch import LatentParams
from uniserve_worker.protocol.identity import BufferId
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
        scores = None

        if sample.batch.logprobs is not None:
            key = id(sample.batch.logprobs)
            if key not in details:
                details[key] = capture_logprobs(sample.batch.logprobs, output)
            scores = details[key].get(sample.index)
        request.set_sampling((*span, sample.index), scores)


@dataclass(slots=True)
class LatentResult:
    """Borrowed trajectory inputs, prepared storage, and visibility updates."""

    update: LatentUpdate
    input_params: LatentParams | None = None
    staging: LatentStaging | None = None
    exports: dict[BufferId, ExportLocations] = field(default_factory=dict)
