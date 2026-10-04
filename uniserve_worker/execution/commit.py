"""Numerical output checks, completion predicates and device state updates."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from uniserve.tensors import concatenate_views
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.execution.output import logprob_entries
from uniserve_worker.protocol.batch import TensorPublication
from uniserve_worker.protocol.call import Call, CallStatus
from uniserve_worker.sampling.result import sample_columns

if TYPE_CHECKING:
    from uniserve_worker.execution.batch import BatchState
    from uniserve_worker.storage.decode_state import DecodeState
    from uniserve_worker.storage.tensor_store import TensorStore


def validate_outputs(state: BatchState) -> None:
    """Check numerical result descriptors and captured score capacity."""
    for call, outcome in zip(
        state.batch.calls, state.pending_outputs(), strict=True
    ):
        _validate_completion_products(call, outcome.products)

        if outcome.kv_output is not None:
            if outcome.kv_output.source != call.kv_output:
                raise invalid_descriptor(
                    "KV publication differs from its declared output"
                )
            if (
                sum(tensor.nbytes for tensor in outcome.kv_output.tensors)
                > call.bounds.max_transfer_bytes
            ):
                raise invalid_descriptor(
                    "KV publication exceeds its transfer-byte bound"
                )
            # Called only for its check, which raises when the descriptor
            # exceeds ``MAX_TRANSFER_HANDLE_BYTES`` or names an unknown
            # transport.
            outcome.kv_output.encoded_size_bound()

        # The bound covers score values and prompt-position counts; framing
        # is owned by the single IPC result message, not by stored products.
        # Each entry is one 12-byte ``TokenLogprob`` of the IPC schema and
        # each reported row adds 4 bytes, the same costs the engine's
        # ``logprob_result_bytes`` uses to size ``max_completion_bytes``.
        logprob_bytes = (
            0
            if outcome.token.logprob_range is None
            else 4 + 12 * logprob_entries(outcome, outcome.token.logprob_range)
        ) + sum(
            4 + 12 * logprob_entries(outcome, span)
            for span in outcome.token.prompt_logprob_ranges
        )
        if logprob_bytes > call.bounds.max_completion_bytes:
            raise invalid_descriptor(
                "logprob result exceeds its registered completion capacity"
            )


def apply_decode_state(
    *,
    state: BatchState,
    decode_state: DecodeState | None,
) -> None:
    """Apply the batch's token-state updates to ``DecodeState``.

    The updates are the ``runtime_*`` fields and sampled rows that execution
    left on each pending output. Without a ``DecodeState``, a sampled row,
    runtime prompt logits or a runtime cache length raises ``RuntimeError``.
    """
    requests = state.pending_outputs()
    states = decode_state
    if states is None:
        if any(
            request.token.sampled is not None
            or request.token.runtime_prompt_logits is not None
            or request.token.runtime_cache_length is not None
            for request in requests
        ):
            raise RuntimeError(
                "runtime state publication has no backing storage"
            )
        return

    # Install lengths before advancing tokens. Decode rows share one update;
    # prefill/verification retain their explicit logical and RNG coordinates.
    for request in requests:
        if request.token.runtime_cache_length is not None:
            states.set_cache_length(
                int(request.request.request_pool_idx),
                request.token.runtime_cache_length,
            )

    decode = tuple(
        request
        for request in requests
        if request.token.sampled is not None
        and request.token.runtime_decode_increment
    )
    if decode:
        samples = tuple(
            request.token.sampled
            for request in decode
            if request.token.sampled is not None
        )
        if any(sample.request_pool_index is None for sample in samples):
            raise RuntimeError("decode samples have no device request slots")
        tokens, continuation, valid, active = sample_columns(
            samples, ("tokens", "continuation", "valid", "active")
        )
        states.apply_tokens(
            tuple(int(request.request.request_pool_idx) for request in decode),
            device_slots=concatenate_views(
                tuple(
                    cast(torch.Tensor, sample.request_pool_index)
                    for sample in samples
                )
            ),
            tokens=tokens,
            predicates=continuation,
            penalty_bases=tuple(
                request.token.runtime_penalty_base for request in decode
            ),
            valid=valid,
            active=active,
        )

    for request in requests:
        sampled = request.token.sampled
        if sampled is not None and not request.token.runtime_decode_increment:
            states.apply_tokens(
                (int(request.request.request_pool_idx),),
                tokens=sampled.tokens,
                predicates=sampled.continuation,
                logical_position=request.token.runtime_logical_position,
                sampling_position=request.token.runtime_sampling_position,
                penalty_bases=(request.token.runtime_penalty_base,),
                valid=sampled.valid,
                active=sampled.active,
            )

    for request in requests:
        if request.token.runtime_prompt_logits is not None:
            states.set_prompt_logits(
                int(request.request.request_pool_idx),
                request.token.runtime_prompt_logits,
            )


def _validate_completion_products(
    call: Call,
    products: tuple[TensorPublication, ...],
) -> None:
    """Require each completion product to be declared by its call.

    Raises ``invalid_descriptor`` for an undeclared product, and for a
    product whose locators exceed ``MAX_TRANSFER_HANDLE_BYTES`` or name an
    unknown transport.
    """
    declared = {output: output for output in call.tensor_outputs()}
    for product in products:
        reference = declared.get(product.product)
        if reference is None:
            raise invalid_descriptor(
                "completion carries a product not declared by its call"
            )
        product.encoded_size_bound()


def publish_predicates(*, state: BatchState, tensor_store: TensorStore) -> None:
    """Publish true into the completion outputs of calls that ran.

    Predicated calls already published false through
    ``prepare._publish_predicated_outputs``, and a call whose execution wrote
    its own completion output is skipped.
    """
    writes = tuple(
        request.completion_write
        for request in state.pending_outputs()
        if request.status is not CallStatus.PREDICATED
        and request.completion_write is not None
        and not request.completion_write.producer_recorded
    )
    if not writes:
        return

    views = tensor_store.producer_write_views(writes)
    first = views[0]

    # Resolved predicates publish as 1, one scalar per producer view.
    tensor_store.publish_writes(
        writes,
        torch.ones(
            (len(writes),),
            dtype=first.dtype,
            device=first.device,
        ),
    )
