"""Token-row selection and vocabulary masking for projected logits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from uniserve.model.batch import TextBatch, TextOutput
    from uniserve.model.tensors import VocabularyPartition
    from uniserve.nn.parallel_pipeline import LayerPipeline
    from uniserve.nn.vocab_parallel_embedding import ParallelLMHead

__all__ = [
    "LogitsProcessor",
    "forced_eos_logits",
    "gather_vocabulary",
    "greedy_vocabulary",
    "project_outputs",
]


def project_outputs(
    hidden: torch.Tensor,
    batch: TextBatch,
    head: ParallelLMHead | None,
    *,
    pipeline: LayerPipeline | None = None,
    vocabulary: VocabularyPartition | None = None,
) -> TextOutput:
    """Project selected text rows with their logical vocabulary partition.

    Final-stage weights produce pipeline outputs, which are broadcast to other
    mathematical participants. Nonresident heads supply vocabulary geometry.
    """

    from uniserve.attention.metadata import AttentionMode
    from uniserve.model.batch import TextOutput
    from uniserve.model.tensors import TokenSelection

    selections = dict(enumerate(batch.selections))
    partition = head.vocabulary_partition() if head is not None else vocabulary
    if partition is None:
        raise ValueError("vocabulary projection requires explicit partition geometry")
    owner = pipeline is None or pipeline.last
    if owner and head is None:
        raise ValueError("the output pipeline stage must own its vocabulary projection")

    def finish(output: TextOutput) -> TextOutput:
        if pipeline is not None and pipeline.group.world_size > 1:
            for value in output.values:
                pipeline.group.broadcast(value, src=pipeline.group.world_size - 1)
        return output

    if (
        batch.attention.attention_mode is AttentionMode.PAGED_VARLEN
        and batch.attention.output_indices is not None
        and len(selections) == batch.row_count
        and all(selection is TokenSelection.LAST_LOGITS for selection in selections.values())
    ):
        if owner:
            assert head is not None
            selected = hidden.index_select(0, batch.attention.output_indices.to(dtype=torch.long))
            selected_logits = head.forward_local(selected)
        else:
            selected_logits = hidden.new_empty((batch.row_count, partition.width))
        return finish(
            TextOutput(
                tuple(selected_logits[index : index + 1] for index in range(batch.row_count)),
                (partition,) * batch.row_count,
            )
        )

    lengths = list(batch.attention.query_lens_cpu)
    # Captured token buckets can include storage-only rows after the live spans.
    visible = hidden[: sum(lengths)]
    rows = (
        tuple(visible.split(lengths, dim=0))
        if owner
        else tuple(hidden.new_empty((length, hidden.shape[-1])) for length in lengths)
    )
    selected_rows = tuple(
        index for index, selection in selections.items() if selection is not TokenSelection.HIDDEN
    )
    projected: torch.Tensor | None = None
    if selected_rows and owner:
        assert head is not None
        if len(selected_rows) == batch.row_count and all(length == 1 for length in lengths):
            selected = visible
        else:
            selected_values = tuple(
                rows[index] if selections[index] is TokenSelection.ALL_LOGITS else rows[index][-1:]
                for index in selected_rows
            )
            selected = (
                selected_values[0]
                if len(selected_values) == 1
                else torch.cat(selected_values, dim=0)
            )
        projected = head.forward_local(selected)
    elif selected_rows:
        projected = hidden.new_empty(
            (
                sum(
                    lengths[index] if selections[index] is TokenSelection.ALL_LOGITS else 1
                    for index in selected_rows
                ),
                partition.width,
            )
        )

    outputs = list(rows)
    vocabularies: list[VocabularyPartition | None] = [None] * batch.row_count
    offset = 0
    for index in selected_rows:
        assert projected is not None
        count = lengths[index] if selections[index] is TokenSelection.ALL_LOGITS else 1
        outputs[index] = projected[offset : offset + count]
        vocabularies[index] = partition
        offset += count
    return finish(TextOutput(tuple(outputs), tuple(vocabularies)))


def _gather_partitions(
    value: torch.Tensor, partition: VocabularyPartition
) -> tuple[torch.Tensor, ...]:
    """Gather matrix rows and expose shards in logical vocabulary order."""

    from uniserve.distributed.mesh import _all_gather_into_tensor

    size = len(partition.backend_order)
    if size == 1:
        return (value,)
    gathered = value.new_empty((size * value.shape[0], value.shape[1]))
    assert partition.group_name is not None
    _all_gather_into_tensor(gathered, value.contiguous(), partition.group_name)
    physical = gathered.view(size, *value.shape)
    return tuple(physical[partition.backend_order.index(rank)] for rank in range(size))


def gather_vocabulary(logits: torch.Tensor, partition: VocabularyPartition) -> torch.Tensor:
    """Materialize unpadded global logits from local vocabulary columns."""

    shards = _gather_partitions(logits, partition)
    return (shards[0] if len(shards) == 1 else torch.cat(shards, dim=-1))[
        ..., : partition.vocab_size
    ]


def greedy_vocabulary(
    logits: torch.Tensor, partition: VocabularyPartition | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select global maxima and token IDs with the dense ``torch.max`` contract.

    Each rank exchanges one value/index pair per row. FP64 transports both the
    original floating score and vocabulary IDs exactly, including FP32 scores
    and IDs beyond the FP32 integer range. Logical rank ordering preserves the
    lowest-token tie rule and the first-NaN rule of dense selection.
    """

    if partition is None:
        return torch.max(logits, dim=-1)
    begin = partition.rank * partition.width
    valid_columns = max(0, min(partition.width, partition.vocab_size - begin))
    if valid_columns:
        values, tokens = torch.max(logits[:, :valid_columns], dim=-1)
        tokens = tokens + begin
    else:
        values = logits.new_full((logits.shape[0],), float("-inf"))
        tokens = torch.full_like(values, partition.vocab_size, dtype=torch.long)
    if len(partition.backend_order) == 1:
        return values, tokens
    candidates = torch.stack((values.to(torch.float64), tokens.to(torch.float64)), dim=-1)
    gathered = torch.stack(_gather_partitions(candidates, partition))
    maxima, owners = torch.max(gathered[..., 0], dim=0)
    selected = gathered[..., 1].gather(0, owners.unsqueeze(0)).squeeze(0)
    return maxima.to(logits.dtype), selected.to(torch.long)


def forced_eos_logits(
    eos_id: int,
    *,
    device: torch.device | str,
    batch_shape: tuple[int, ...] = (),
) -> torch.Tensor:
    """Create a minimal logits tensor whose only finite token is ``eos_id``.

    Empty-token text operations still require a sampling input even though no
    model forward is run. Any categorical sampler over the returned tensor
    therefore selects EOS. ``batch_shape`` is prepended to the vocabulary axis.
    """

    eos = int(eos_id)

    # The vocabulary axis needs only enough entries to address EOS. Negative
    # infinity excludes every preceding token under ordinary logit sampling.
    logits = torch.full((*batch_shape, eos + 1), float("-inf"), device=device)
    logits[..., eos] = 0.0
    return logits


class LogitsProcessor(nn.Module):
    """Token-row selector and padded-vocabulary masker for projected logits."""

    def forward(
        self,
        logits: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        valid_vocab_size: int | None = None,
    ) -> torch.Tensor:
        """Post-process projected logits for sampling.

        ``positions`` indexes a flattened token axis, typically selecting the
        final token of each sequence. ``valid_vocab_size`` masks any padded
        projection columns in place on the resulting tensor.
        """

        # Selection precedes vocabulary masking so only sampling rows are
        # materialized when callers provide packed sequence positions.
        if positions is not None:
            logits = logits.reshape(-1, logits.shape[-1]).index_select(
                0,
                positions,
            )

        # Tensor-parallel projections may pad their vocabulary width; prevent
        # those storage-only columns from participating in sampling.
        if valid_vocab_size is not None and 0 < int(valid_vocab_size) < int(logits.shape[-1]):
            logits[..., int(valid_vocab_size) :] = float("-inf")

        return logits
