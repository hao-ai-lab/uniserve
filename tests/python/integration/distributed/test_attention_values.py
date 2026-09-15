"""Public attention preserves token order, GQA replication and empty shards."""

import pytest
import torch
import torch.multiprocessing as mp
from torch.nn import functional as F

from uniserve.cache import Config, mha
from uniserve.distributed import DeviceMesh, parallelize_
from uniserve.model import TextSize
from uniserve.nn.attention import (
    Attention,
    AttentionParallelConfig,
    PagedInput,
    Ulysses,
)
from uniserve.runtime import (
    ExecutionContext,
    PrefixCache,
    initialize_process_groups,
)

pytestmark = pytest.mark.integration


def _run(rank, rendezvous):
    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=4,
        device="cpu",
        init_method=rendezvous,
    ) as owner:
        mesh = owner.bind(
            DeviceMesh(
                ranks=(3, 1, 0, 2), shape=(4,), axes=("tokens",), rank=rank
            ),
            device="cpu",
        )
        group = mesh.get_group("tokens")
        generator = torch.Generator().manual_seed(71)
        for count in (0, 2, 7):
            q = torch.randn(
                count, 8, 4, dtype=torch.float32, generator=generator
            )
            k, v = (
                torch.randn(
                    count, 2, 4, dtype=torch.float32, generator=generator
                )
                for _ in range(2)
            )
            layer = Attention(8, 2, 4, cache_name="attention")
            parallelize_(
                layer,
                mesh,
                attention=AttentionParallelConfig(heads=Ulysses("tokens")),
            )
            width = (count + 3) // 4
            start = min(count, group.rank * width)
            stop = min(count, start + width)
            config = Config(
                {
                    "attention": mha.Config(
                        2, 4, (group.rank // 2,), torch.float32
                    )
                }
            )
            with PrefixCache(
                config, num_blocks=1, block_size=16, device="cpu"
            ) as cache:
                with ExecutionContext(
                    layer, cache=cache, attention="torch"
                ) as context:
                    context.prepare(TextSize(count, 1))
                    batch = PagedInput.from_blocks(
                        blocks=((0,),),
                        query_lengths=(count,),
                        prefix_lengths=(0,),
                        block_size=16,
                        causal=True,
                        device="cpu",
                    )
                    context.bind_attention(batch)
                    out = torch.empty_like(q[start:stop])
                    actual = layer(
                        q[start:stop],
                        k[start:stop],
                        v[start:stop],
                        batch,
                        out=out,
                    )
                    expected = (
                        F.scaled_dot_product_attention(
                            q.transpose(0, 1).unsqueeze(0),
                            k.transpose(0, 1).unsqueeze(0),
                            v.transpose(0, 1).unsqueeze(0),
                            is_causal=True,
                            enable_gqa=True,
                        )
                        .squeeze(0)
                        .transpose(0, 1)
                    )
                    assert actual is out
                    torch.testing.assert_close(actual, expected[start:stop])
                    key, value = cache.state("attention").read(
                        (0,), start=0, length=count
                    )
                    head = group.rank // 2
                    torch.testing.assert_close(key, k[:, head : head + 1])
                    torch.testing.assert_close(value, v[:, head : head + 1])


def test_ulysses_paged_attention_values(tmp_path):
    mp.spawn(
        _run, args=((tmp_path / "rendezvous").as_uri(),), nprocs=4, join=True
    )


def _context(rank, rendezvous, gpu):
    from uniserve.nn.attention import (
        ContextParallelConfig,
        SegmentedInput,
        SequenceLengths,
        VarlenInput,
    )
    from uniserve.runtime import CUDAGraph

    device = torch.device("cuda", rank) if gpu else torch.device("cpu")
    if gpu:
        torch.cuda.set_device(device)
    dtype = torch.bfloat16 if gpu else torch.float32
    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=4,
        device=device,
        init_method=rendezvous,
    ) as owner:
        cases = (
            (
                (4,),
                ("context",),
                AttentionParallelConfig(
                    context=ContextParallelConfig(gather_axis="context")
                ),
            ),
            (
                (2, 2),
                ("heads", "context"),
                AttentionParallelConfig(
                    heads=Ulysses("heads"),
                    context=ContextParallelConfig(gather_axis="context"),
                ),
            ),
        )
        if gpu:
            cases += (
                (
                    (4,),
                    ("context",),
                    AttentionParallelConfig(
                        context=ContextParallelConfig(peer_axis="context")
                    ),
                ),
                (
                    (2, 2),
                    ("columns", "rows"),
                    AttentionParallelConfig(
                        context=ContextParallelConfig(
                            gather_axis="columns", peer_axis="rows"
                        )
                    ),
                ),
            )
        for shape, axes, parallel in cases:
            mesh = owner.bind(
                DeviceMesh(
                    ranks=(3, 1, 0, 2), shape=shape, axes=axes, rank=rank
                ),
                device=device,
            )
            tokens = mesh.get_group(axes)
            layer = Attention(8, 2, 64, cache_name="attention")
            parallelize_(layer, mesh, attention=parallel)
            heads = mesh.get_group(
                () if parallel.heads is None else parallel.heads.axis
            )
            first_head = heads.rank * (2 // heads.size)
            local_heads = tuple(range(first_head, first_head + 2 // heads.size))
            config = Config(
                {"attention": mha.Config(2, 64, local_heads, dtype)}
            )
            stream = torch.cuda.Stream(device=device) if gpu else None
            if stream is not None:
                stream.wait_stream(torch.cuda.current_stream(device))
            with PrefixCache(
                config, num_blocks=3, block_size=16, device=device
            ) as cache:
                with ExecutionContext(
                    layer,
                    cache=cache,
                    stream=stream,
                    attention="auto" if gpu else "torch",
                ) as context:
                    context.prepare(TextSize(7, 3))
                    for count in (0, 2, 7):
                        generator = torch.Generator().manual_seed(41 + count)
                        q = torch.randn(count, 8, 64, generator=generator).to(
                            device, dtype
                        )
                        k, v = (
                            torch.randn(count, 2, 64, generator=generator).to(
                                device, dtype
                            )
                            for _ in range(2)
                        )
                        width = (count + 3) // 4
                        interval = slice(
                            min(count, tokens.rank * width),
                            min(count, (tokens.rank + 1) * width),
                        )
                        counts = (count // 2, 0, count - count // 2)
                        lengths = SequenceLengths.from_lengths(
                            counts, device=device
                        )
                        batch = VarlenInput(
                            lengths, lengths, (True, False, False)
                        )
                        expected = []
                        for index, (query, key, value) in enumerate(
                            zip(
                                q.split(counts),
                                k.split(counts),
                                v.split(counts),
                                strict=True,
                            )
                        ):
                            expected.append(
                                F.scaled_dot_product_attention(
                                    query.transpose(0, 1).unsqueeze(0),
                                    key.transpose(0, 1).unsqueeze(0),
                                    value.transpose(0, 1).unsqueeze(0),
                                    is_causal=index == 0,
                                    enable_gqa=True,
                                )
                                .squeeze(0)
                                .transpose(0, 1)
                            )
                        expected = torch.cat(expected)[interval]
                        actual = layer(
                            q[interval], k[interval], v[interval], batch
                        )
                        torch.testing.assert_close(actual, expected)
                        cached = PagedInput.from_blocks(
                            blocks=((0,), (1,), (2,)),
                            query_lengths=counts,
                            prefix_lengths=(0, 0, 0),
                            block_size=16,
                            causal=(True, False, False),
                            device=device,
                        )
                        actual = layer(
                            q[interval], k[interval], v[interval], cached
                        )
                        torch.testing.assert_close(actual, expected)
                        offset = 0
                        for block, length in enumerate(counts):
                            key, value = cache.state("attention").read(
                                (block,), start=0, length=length
                            )
                            torch.testing.assert_close(
                                key, k[offset : offset + length, local_heads]
                            )
                            torch.testing.assert_close(
                                value, v[offset : offset + length, local_heads]
                            )
                            offset += length

                        if count != 7:
                            continue
                        prefix_counts = (2, 0, 1)
                        source_prefix_counts = prefix_counts
                        prefix_k, prefix_v = (
                            torch.randn(3, 2, 64, generator=generator).to(
                                device, dtype
                            )
                            for _ in range(2)
                        )
                        prefix_interval = slice(
                            min(3, tokens.rank), min(3, tokens.rank + 1)
                        )
                        layer.update_cache(
                            prefix_k[prefix_interval],
                            prefix_v[prefix_interval],
                            indices=torch.tensor(
                                [0, 1, 32], dtype=torch.long, device=device
                            ),
                        )
                        ends = torch.zeros(
                            (3, max(counts)), dtype=torch.int32, device=device
                        )
                        for row, length in enumerate(counts):
                            ends[row, :length] = (
                                torch.arange(length, device=device) + 1
                            )
                        segmented = SegmentedInput(
                            lengths,
                            SequenceLengths.from_lengths(
                                prefix_counts, device=device
                            ),
                            cached.block_table,
                            None,
                            ends,
                            False,
                        )

                        def reference():
                            outputs = []
                            start = prefix_start = 0
                            for row, (length, prefix_count) in enumerate(
                                zip(counts, prefix_counts, strict=True)
                            ):
                                key = torch.cat(
                                    (
                                        prefix_k[
                                            prefix_start : prefix_start
                                            + prefix_count
                                        ],
                                        k[start : start + length],
                                    )
                                )
                                value = torch.cat(
                                    (
                                        prefix_v[
                                            prefix_start : prefix_start
                                            + prefix_count
                                        ],
                                        v[start : start + length],
                                    )
                                )
                                allowed = torch.arange(
                                    key.shape[0], device=device
                                )[None] < (
                                    prefix_count + ends[row, :length, None]
                                )
                                outputs.append(
                                    F.scaled_dot_product_attention(
                                        q[start : start + length]
                                        .transpose(0, 1)
                                        .unsqueeze(0),
                                        key.transpose(0, 1).unsqueeze(0),
                                        value.transpose(0, 1).unsqueeze(0),
                                        attn_mask=allowed,
                                        enable_gqa=True,
                                    )
                                    .squeeze(0)
                                    .transpose(0, 1)
                                )
                                start += length
                                prefix_start += source_prefix_counts[row]
                            return torch.cat(outputs)[interval]

                        def invoke():
                            return layer(
                                q[interval], k[interval], v[interval], segmented
                            )

                        torch.testing.assert_close(
                            invoke(),
                            reference(),
                            rtol=2e-2 if gpu else 1e-5,
                            atol=2e-2 if gpu else 1e-6,
                        )
                        if gpu:
                            with CUDAGraph(context=context) as graph:
                                graph.capture(invoke)
                                q.mul_(0.7)
                                k.add_(0.25)
                                ends.sub_(1).clamp_min_(0)
                                prefix_counts = (1, 0, 1)
                                segmented.prefixes.values.copy_(
                                    torch.tensor(
                                        prefix_counts,
                                        dtype=torch.int32,
                                        device=device,
                                    )
                                )
                                segmented.prefixes.offsets.copy_(
                                    torch.tensor(
                                        [0, 1, 1, 2],
                                        dtype=torch.int32,
                                        device=device,
                                    )
                                )
                                from dataclasses import replace

                                context.bind_attention(
                                    replace(
                                        segmented,
                                        prefixes=SequenceLengths(
                                            host=prefix_counts,
                                            values=segmented.prefixes.values,
                                            offsets=segmented.prefixes.offsets,
                                        ),
                                    )
                                )
                                torch.testing.assert_close(
                                    graph.replay(),
                                    reference(),
                                    rtol=2e-2,
                                    atol=2e-2,
                                )
                                torch.cuda.synchronize(device)
                        start = 0
                        for block, length in enumerate(source_prefix_counts):
                            key, value = cache.state("attention").read(
                                (block,), start=0, length=length
                            )
                            torch.testing.assert_close(
                                key,
                                prefix_k[start : start + length, local_heads],
                            )
                            torch.testing.assert_close(
                                value,
                                prefix_v[start : start + length, local_heads],
                            )
                            start += length


def test_context_gather_preserves_sequences_and_cache_values(tmp_path):
    mp.spawn(
        _context,
        args=((tmp_path / "rendezvous").as_uri(), False),
        nprocs=4,
        join=True,
    )


@pytest.mark.gpu
def test_context_peer_and_column_gather_preserve_sequences_and_cache_values(
    tmp_path,
):
    mp.spawn(
        _context,
        args=((tmp_path / "rendezvous").as_uri(), True),
        nprocs=4,
        join=True,
    )
