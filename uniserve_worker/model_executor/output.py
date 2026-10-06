"""Numerical copies and vocabulary gathers for native execution results."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import torch

from uniserve.model.logits import Logits, VocabShard
from uniserve.tensors import adjacent_view
from uniserve_worker._uniserve_ipc import ExecutionOutput as ExecutionOutput


def _validate_vocabularies(
    values: tuple[torch.Tensor, ...],
    vocabularies: tuple[VocabShard | None, ...],
) -> None:
    """Require local logit columns to cover their declared vocabulary shard."""
    for value, vocab in zip(values, vocabularies, strict=True):
        if vocab is not None and (
            value.ndim != 2
            or value.shape[-1]
            != vocab.local_slice.stop - vocab.local_slice.start
        ):
            raise ValueError("vocabulary output rows disagree with their shard")


def _combine_greedy(parts):
    """Concatenate each column and each of the four completion sections."""
    return replace(
        parts[0],
        **{
            name: torch.cat([getattr(part, name) for part in parts])
            for name in (
                "tokens",
                "valid",
                "active",
                "finish",
                "continuation",
                "tagged_tokens",
            )
        },
        completion=torch.cat(
            [part.completion.reshape(4, -1) for part in parts], dim=1
        ).reshape(-1),
    )


def _materialize(output: ExecutionOutput) -> ExecutionOutput:
    """Gather global vocabulary rows, preserving their row counts.

    The native result has joined its completion event. Each gather is
    an all-gather collective over the shard's tensor group, so every rank
    of that group must make the matching call. The result has no
    vocabulary metadata; it is ``output`` when no row is sharded.
    """
    if not any(output.vocabularies):
        return output

    # Bucket rows that share one vocabulary shard so they gather together.
    groups: dict[
        tuple[int, int, int, int, int, torch.device, torch.dtype],
        tuple[VocabShard, list[int]],
    ] = {}
    for index, vocab in enumerate(output.vocabularies):
        if vocab is not None:
            key = (
                vocab.size,
                vocab.padded_size,
                vocab.local_slice.start,
                vocab.local_slice.stop,
                id(vocab.group),
                output.values[index].device,
                output.values[index].dtype,
            )
            groups.setdefault(key, (vocab, []))[1].append(index)

    values = list(output.values)
    for vocab, indexes in groups.values():
        sources = tuple(output.values[index] for index in indexes)
        # Adjacent request rows already share backing; gather them together
        # without adding another allocation or collective per request.
        view = adjacent_view(sources)
        rows = (
            torch.cat(sources, dim=0)
            if view is None
            else view.reshape(
                -1, vocab.local_slice.stop - vocab.local_slice.start
            )
        )
        gathered = Logits(rows, vocab).gather()
        for index, value in zip(
            indexes,
            gathered.split(tuple(source.shape[0] for source in sources)),
            strict=True,
        ):
            values[index] = value

    return output.replace(values=tuple(values), vocabularies=())


def _clone(output: ExecutionOutput) -> ExecutionOutput:
    """Own detached copies that survive reuse of the producer's storage.

    The native result has joined its completion event; the copy carries
    no event. Outputs on one device with one dtype share a
    contiguous allocation. Shapes and logical tensor values are preserved
    independently of source strides; storage remains live for as long as
    any returned tensor is retained. ``greedy`` and
    ``request_pool_indices`` are cloned as well.
    """
    # Tensors sharing a device and dtype copy through one flat allocation.
    groups: dict[tuple[torch.device, torch.dtype], list[int]] = defaultdict(
        list
    )
    for index, value in enumerate(output.values):
        groups[(value.device, value.dtype)].append(index)

    copied = list(output.values)
    for indexes in groups.values():
        if len(indexes) == 1:
            index = indexes[0]
            copied[index] = output.values[index].detach().clone()
            continue
        sources = [output.values[index] for index in indexes]
        storage = torch.cat(
            tuple(value.detach().reshape(-1) for value in sources)
        )
        views = storage.split(tuple(value.numel() for value in sources))
        for index, view in zip(indexes, views, strict=True):
            copied[index] = view.reshape(output.values[index].shape)

    greedy = output.greedy
    if greedy is not None:
        greedy = greedy.clone()
    return output.replace(
        values=tuple(copied),
        output_event=None,
        greedy=greedy,
        request_pool_indices=None
        if output.request_pool_indices is None
        else output.request_pool_indices.clone(),
    )
