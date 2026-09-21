"""Sequence and block representations expose explicit attention addresses."""

from dataclasses import replace

import pytest
import torch

from uniserve.nn.attention import (
    AttentionParallelConfig,
    ContextParallelConfig,
    PagedInput,
    SequenceLengths,
    Ulysses,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_sequence_queries_do_not_materialize_missing_host_values(device):
    lengths = SequenceLengths(
        torch.empty(2, dtype=torch.int32, device=device),
        torch.empty(3, dtype=torch.int32, device=device),
    )
    assert lengths.batch_size == 2
    assert lengths.num_tokens is None
    assert lengths.maximum is None
    with pytest.raises(ValueError, match="host lengths must match"):
        replace(lengths, host=(3,))


def test_paged_input_maps_append_positions_across_reordered_blocks():
    batch = PagedInput.from_blocks(
        blocks=((2, 0), (3,), ()),
        query_lengths=(3, 1, 0),
        prefix_lengths=(2, 0, 0),
        block_size=4,
        causal=(True, False, True),
        device="cpu",
    )
    assert batch.queries.batch_size == 3
    assert batch.queries.num_tokens == 4
    assert batch.queries.maximum == 3
    assert batch.queries.offsets.tolist() == [0, 3, 4, 4]
    assert batch.prefixes.values.tolist() == [2, 0, 0]
    assert batch.write_indices.tolist() == [10, 11, 0, 12]
    assert batch.block_table.indices.tolist() == [[2, 0], [3, 0], [0, 0]]
    assert replace(batch, write_indices=None).write_indices is None
    with pytest.raises(ValueError, match="cover the prefix and query"):
        PagedInput.from_blocks(
            blocks=((2,),),
            query_lengths=(3,),
            prefix_lengths=(2,),
            block_size=4,
            causal=True,
            device="cpu",
        )


def test_empty_attention_lengths_and_batches_remain_well_defined():
    lengths = SequenceLengths.from_lengths((), device="cpu")
    assert (lengths.batch_size, lengths.num_tokens, lengths.maximum) == (
        0,
        0,
        0,
    )
    assert lengths.offsets.tolist() == [0]
    batch = PagedInput.from_blocks(
        blocks=(),
        query_lengths=(),
        prefix_lengths=(),
        block_size=4,
        causal=True,
        device="cpu",
    )
    assert batch.block_table.indices.shape == (0, 0)
    assert batch.write_indices.numel() == 0


def test_attention_axes_are_distinct_and_context_names_its_axis():
    config = AttentionParallelConfig(
        heads=Ulysses("heads"),
        context=ContextParallelConfig(gather_axis="columns"),
    )
    assert config.heads.axis == "heads"
    with pytest.raises(ValueError, match="independent"):
        AttentionParallelConfig(
            heads=Ulysses("columns"), context=config.context
        )
    with pytest.raises(TypeError):
        ContextParallelConfig()
    with pytest.raises(ValueError, match="nonempty"):
        ContextParallelConfig(gather_axis="")
