"""Decode sampled log probabilities from their packed host representation.

The sampler (``logprob_details`` in ``uniserve_worker.sampling.sampler``)
packs every logprob field of a sampled batch into one int64 device column.
After the output buffer copies that column to the host,
``OutputBuffer.logprob_values`` (``uniserve_worker.storage.output``) decodes
it here once per column.
"""

from __future__ import annotations

import struct


def decode_logprobs(
    values: tuple[int, ...],
    layout: tuple[
        tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...], int, int
    ],
) -> dict[int, tuple[float, tuple[tuple[int, float, int], ...]]]:
    """Decode the sampler's packed float bits and row-major rank columns.

    Args:
        values: The packed column read back from completion storage.
        layout: ``(rows, counts, requested_ids, max_count, max_requested)``
            from the ``LogprobValues`` the sampler produced: the sampler row
            index of each packed row, its top-logprob count, its
            de-duplicated in-vocabulary requested token ids, and the padded
            widths of the top and requested fields.

    Returns:
        A map from sampler row index to ``(selected_logprob, entries)``.
        Each entry is ``(token_id, logprob, rank)`` with a 1-based
        competition rank; entries list the selected token first, then the
        top tokens, then the requested tokens, each token at most once.

    Raises:
        RuntimeError: ``values`` is shorter or longer than ``layout``
            describes.
    """
    rows, counts, requested_ids, max_count, max_requested = layout

    # Score fields carry float32 bit patterns in the low 32 bits of int64
    # words. The sampler widens them from int32, which sign-extends, so the
    # high bits are masked off before reinterpreting.
    def float_value(value: int) -> float:
        return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]

    row_count = len(rows)
    cursor = 0

    def vector(width: int) -> tuple[tuple[int, ...], ...]:
        """Consume the next row-major field of fixed width per row."""
        nonlocal cursor
        total = row_count * width
        part = values[cursor : cursor + total]
        if len(part) != total:
            raise RuntimeError("logprob completion metadata is truncated")
        cursor += total
        return tuple(
            tuple(part[row * width : (row + 1) * width])
            for row in range(row_count)
        )

    # The packed column is a fixed sequence of row-major fields; each
    # vector(width) call consumes the next one in the order ``logprob_details``
    # concatenates them. Top and requested fields are padded to the widest
    # row; ``counts`` and ``requested_ids`` bound the real columns.
    selected_tokens = vector(1)
    selected_values = vector(1)
    selected_ranks = vector(1)
    top_indexes = vector(max_count)
    top_values = vector(max_count)
    top_ranks = vector(max_count)
    candidate_values = vector(max_requested)
    candidate_ranks = vector(max_requested)
    if cursor != len(values):
        raise RuntimeError("logprob completion metadata has trailing values")

    details: dict[int, tuple[float, tuple[tuple[int, float, int], ...]]] = {}
    for local, result_index in enumerate(rows):
        selected = selected_tokens[local][0]
        selected_value = float_value(selected_values[local][0])
        entries: list[tuple[int, float, int]] = [
            (selected, selected_value, selected_ranks[local][0])
        ]
        seen = {selected}

        for index in range(counts[local]):
            candidate = top_indexes[local][index]
            if candidate not in seen:
                entries.append(
                    (
                        candidate,
                        float_value(top_values[local][index]),
                        top_ranks[local][index],
                    )
                )
                seen.add(candidate)

        for index, candidate in enumerate(requested_ids[local]):
            if candidate not in seen:
                entries.append(
                    (
                        candidate,
                        float_value(candidate_values[local][index]),
                        candidate_ranks[local][index],
                    )
                )
                seen.add(candidate)

        details[result_index] = (selected_value, tuple(entries))
    return details
