"""Sparse head extents and fused head-to-row output composition on CUDA."""

import pytest
import torch
import torch.nn.functional as F

from uniserve.attention import video_sparse_cute, video_sparse_flashinfer, video_sparse_triton
from uniserve.attention.video_sparse_provider import resolve_sparse_provider
from uniserve.nn.parallel_attention import AttentionOutputTargets
from uniserve.nn.sparse_attention import PreparedVideoSparseInputs
from uniserve.ops.video_sparse import compose_to_head_shards
from uniserve.ops.video_sparse_rows import SparseAttentionPattern

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _available_sparse_attention(
    query,
    key,
    value,
    output,
    indices,
    counts,
    valid_sizes,
):
    """Call the architecture's production sparse-attention provider."""

    if video_sparse_cute.available(query.device):
        return video_sparse_cute.block_sparse_attention(
            query, key, value, output, indices, counts, valid_sizes
        )
    return video_sparse_triton.block_sparse_attention(
        query, key, value, output, indices, counts, valid_sizes
    )


@pytest.mark.parametrize("heads", [14, 28, 56, 7])
def test_sparse_attention_heads_preserve_block_mask_and_partial_tiles(heads):
    torch.manual_seed(123)
    rows, width, tile = 256, 128, 64
    # Interleaved projections exercise the actual Q/K/V ingress strides.
    projected = torch.randn(rows, heads, 4, width, device="cuda", dtype=torch.bfloat16)
    query, key, value, _ = projected.unbind(2)
    valid_sizes = torch.tensor([64, 17, 64, 0], device="cuda", dtype=torch.int32)
    indices = torch.zeros((heads, 4, 4), device="cuda", dtype=torch.int32)
    for head in range(heads):
        indices[head, :, 1] = 2 if head % 2 == 0 else 1
    counts = torch.full((heads, 4), 2, device="cuda", dtype=torch.int32)
    output = torch.empty((rows, heads, width), device="cuda", dtype=torch.bfloat16)
    actual = _available_sparse_attention(query, key, value, output, indices, counts, valid_sizes)
    mask = torch.zeros((heads, rows, rows), device="cuda", dtype=torch.bool)
    for head in range(heads):
        mask[head, :, :64] = True
        if head % 2 == 0:
            mask[head, :, 128:192] = True
        else:
            mask[head, :, 64:81] = True
    reference = F.scaled_dot_product_attention(
        query.transpose(0, 1).double(),
        key.transpose(0, 1).double(),
        value.transpose(0, 1).double(),
        attn_mask=mask,
    ).to(torch.bfloat16)
    valid_queries = torch.arange(rows, device="cuda") % tile < valid_sizes.repeat_interleave(tile)
    # The established BF16 attention tolerance covers softmax/MMA rounding;
    # padded query outputs are not part of the model's valid-token contract.
    torch.testing.assert_close(
        actual[0, :, valid_queries], reference[:, valid_queries], rtol=2e-2, atol=2e-2
    )


def _sparse_provider(provider_name):
    from functools import partial

    from uniserve.attention.video_sparse_provider import SparseAttentionProvider
    from uniserve.ops.video_sparse_rows import SparseRowExecution

    if provider_name == "native":
        provider = resolve_sparse_provider(torch.device("cuda"))
    elif provider_name == "flashinfer":
        state = video_sparse_flashinfer.SparseExecutionState()
        provider = SparseAttentionProvider(
            "flashinfer",
            partial(video_sparse_flashinfer.execute_sparse_attention, state),
            partial(video_sparse_flashinfer.prepare_sparse_attention_rows, state),
            video_sparse_flashinfer.uses_row_major_inputs(torch.device("cuda")),
        )
    else:
        rows = SparseRowExecution(video_sparse_triton.block_sparse_attention)
        provider = SparseAttentionProvider(
            "triton", video_sparse_triton.execute_sparse_attention, rows.prepare, True
        )

    return provider


@pytest.mark.parametrize(
    ("members", "chunk_rows", "prefix_tiles"),
    [(1, None, 1), (2, None, 1), (1, 64, 1), (2, 64, 1), (1, 192, 1), (2, 192, 1), (2, 64, 3)],
)
@pytest.mark.parametrize("provider_name", ["native", "flashinfer", "triton"])
def test_sparse_attention_replays_dynamic_maps_across_output_shards(
    members, chunk_rows, prefix_tiles, provider_name
):
    provider = _sparse_provider(provider_name)

    torch.manual_seed(518)
    rows, heads, width, tile = 256, 28, 128, 64
    projected = torch.randn(rows, heads, 4, width, device="cuda", dtype=torch.bfloat16)
    query, key, value, gate = projected.unbind(2)
    valid_sizes = torch.tensor([64, 17, 64, 0], device="cuda", dtype=torch.int32)
    indices = torch.zeros((heads, 4, 3), device="cuda", dtype=torch.int32)
    indices[:, :prefix_tiles] = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    if prefix_tiles < 3:
        indices[:, prefix_tiles:3, prefix_tiles] = 1
    pattern = SparseAttentionPattern(
        (tuple([3] * prefix_tiles + [prefix_tiles + 1] * (3 - prefix_tiles) + [1]),),
        dense_prefix_tiles=prefix_tiles,
        dense_key_tiles=3,
    )
    counts = torch.tensor(
        pattern.row_counts[0],
        device="cuda",
        dtype=torch.int32,
    )
    counts = counts.view(1, -1).expand(heads, -1).contiguous()
    compressed = torch.randn((heads, 4, width), device="cuda")
    attention_output = torch.empty_like(query)
    outputs = tuple(
        torch.zeros(
            (rows // members, heads * (members if chunk_rows is None else 1), width),
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(members)
    )

    def invoke() -> None:
        if chunk_rows is not None:
            prepared = None
            if chunk_rows == 64:
                pooled = [
                    torch.empty((rows // tile, heads, width), device="cuda") for _ in range(3)
                ]
                prepared = PreparedVideoSparseInputs(
                    (rows, heads, width),
                    query.dtype,
                    valid_sizes,
                    members,
                    chunk_rows,
                    *pooled,
                    provider.row_major,
                )
                for start, end in ((128, 256), (0, 64), (64, 128)):
                    prepared.append(
                        slice(start, end), query[start:end], key[start:end], value[start:end]
                    )
            produce = provider.prepare_rows(
                query,
                key,
                value,
                mask_block_indices=indices,
                mask_block_count=counts,
                valid_sizes=valid_sizes,
                pattern=pattern,
                gate=gate,
                compressed=compressed,
                attention_output=attention_output,
                owners=members,
                chunk_rows=chunk_rows,
                packed=None if prepared is None else prepared.packed,
            )
            for start in range(0, rows // members, chunk_rows):
                interval = slice(start, min(start + chunk_rows, rows // members))
                produce(interval, tuple(output[interval] for output in outputs))
            return
        provider.execute(
            query,
            key,
            value,
            mask_block_count=counts,
            mask_block_indices=indices,
            valid_sizes=valid_sizes,
            tile_size=tile,
            pattern=pattern,
            gate=gate,
            compressed=compressed,
            attention_output=attention_output,
            targets=AttentionOutputTargets(outputs, 0),
        )

    def reference(selected_video_tile: int) -> torch.Tensor:
        mask = torch.zeros((heads, rows, rows), device="cuda", dtype=torch.bool)
        mask[:, : prefix_tiles * tile, : 3 * tile] = True
        mask[:, prefix_tiles * tile : 3 * tile, : prefix_tiles * tile] = True
        mask[
            :,
            prefix_tiles * tile : 3 * tile,
            selected_video_tile * tile : (selected_video_tile + 1) * tile,
        ] = True
        mask[:, 3 * tile :, :tile] = True
        key_valid = torch.arange(rows, device="cuda") % tile < valid_sizes.repeat_interleave(tile)
        mask &= key_valid.view(1, 1, -1)
        attended = F.scaled_dot_product_attention(
            query.transpose(0, 1).double(),
            key.transpose(0, 1).double(),
            value.transpose(0, 1).double(),
            attn_mask=mask,
        )
        return (
            attended
            + gate.transpose(0, 1).double() * compressed.double().repeat_interleave(tile, dim=1)
        ).to(torch.bfloat16)

    def assert_matches(expected: torch.Tensor) -> None:
        valid_queries = torch.arange(rows, device="cuda") % tile < valid_sizes.repeat_interleave(
            tile
        )
        for destination, shard in enumerate(outputs):
            begin = destination * (rows // members)
            end = begin + rows // members
            live = valid_queries[begin:end]
            torch.testing.assert_close(
                shard[live, :heads],
                expected[:, begin:end].transpose(0, 1)[live],
                rtol=2e-2,
                atol=2e-2,
            )

    invoke()
    torch.cuda.synchronize()
    assert_matches(reference(1))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    if prefix_tiles < 3:
        indices[:, prefix_tiles:3, prefix_tiles] = 2
    else:
        valid_sizes[1] = 31
    graph.replay()
    assert_matches(reference(2))
    graph.reset()


@pytest.mark.parametrize("heads", [7, 14, 28, 56])
def test_sparse_query_partitions_preserve_complete_key_attention(heads):
    from uniserve_kernel.sparse_attention import block_sparse_attention as native_attention

    torch.manual_seed(972)
    rows, width = 512, 128
    query = torch.randn(1, heads, rows, width, device="cuda", dtype=torch.bfloat16)
    key, value = torch.randn_like(query), torch.randn_like(query)
    valid = torch.tensor([64, 17, 64, 64, 0, 64, 31, 64], device="cuda", dtype=torch.int32)
    indices = torch.tensor([0, 1, 3, 6], device="cuda", dtype=torch.int32)
    indices = indices.view(1, 1, 4).expand(heads, rows // 64, 4).contiguous()
    counts = (
        torch.arange(heads, device="cuda", dtype=torch.int32).view(-1, 1) * 3
        + torch.arange(rows // 64, device="cuda", dtype=torch.int32).view(1, -1)
    ).remainder(4) + 1
    full = native_attention(query, key, value, indices, counts, valid)
    strided_key = key.transpose(1, 2).contiguous().transpose(1, 2)
    strided_value = torch.empty(1, rows, heads, 2, width, device="cuda", dtype=torch.bfloat16)
    strided_value = strided_value[:, :, :, 1].transpose(1, 2)
    strided_value.copy_(value)
    strided = native_attention(query, strided_key, strided_value, indices, counts, valid)
    torch.testing.assert_close(strided, full, rtol=2e-2, atol=2e-2)
    partitions = [
        native_attention(
            query[:, :, start:end].contiguous(),
            key,
            value,
            indices[:, start // 64 : end // 64].contiguous(),
            counts[:, start // 64 : end // 64].contiguous(),
            valid,
        )
        for start, end in ((0, 64), (64, 256), (256, 512))
    ]
    torch.testing.assert_close(torch.cat(partitions, dim=2), full, rtol=2e-2, atol=2e-2)
    tail_query = query[:, :, :64].contiguous()
    tail_indices, tail_counts = indices[:, :1].contiguous(), counts[:, :1].contiguous()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = native_attention(
            tail_query, strided_key, strided_value, tail_indices, tail_counts, valid
        )
    for _ in range(2):
        graph.replay()
        torch.testing.assert_close(captured, full[:, :, :64], rtol=2e-2, atol=2e-2)
    graph.reset()
    selected = torch.arange(4, device="cuda").view(1, 1, -1) < counts.unsqueeze(-1)
    mask = torch.zeros(heads, rows // 64, rows // 64, device="cuda", dtype=torch.bool)
    mask.scatter_(2, indices.long(), selected)
    mask = mask.repeat_interleave(64, 1).repeat_interleave(64, 2)
    mask &= (torch.arange(rows, device="cuda") % 64 < valid.repeat_interleave(64)).view(1, 1, -1)
    reference = F.scaled_dot_product_attention(
        query.double(), key.double(), value.double(), attn_mask=mask.unsqueeze(0)
    )
    torch.testing.assert_close(full.double(), reference, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("members", [2, 4, 8])
def test_fused_composition_restores_each_rows_head_interval(members):
    torch.manual_seed(456)
    local_rows, local_heads, width = 64, 7, 128
    rows = local_rows * members
    attended = torch.randn((1, local_heads, rows, width), device="cuda", dtype=torch.bfloat16)
    gate = torch.randn((rows, local_heads, width), device="cuda", dtype=torch.bfloat16)
    compressed = torch.randn((local_heads, rows // 64, width), device="cuda")
    outputs = tuple(
        torch.zeros((local_rows, members * local_heads, width), device="cuda", dtype=torch.bfloat16)
        for _ in range(members)
    )
    # The independent formula permits fused or separate floating-point
    # arithmetic within the BF16 output error bound.
    expected = attended[0].transpose(0, 1).double() + gate.double() * compressed.transpose(
        0, 1
    ).double().repeat_interleave(64, dim=0)
    for source in range(members):
        compose_to_head_shards(attended, gate, compressed, outputs, source)
    for member, actual in enumerate(outputs):
        row_interval = expected[member * local_rows : (member + 1) * local_rows]
        torch.testing.assert_close(
            actual.double(), row_interval.repeat(1, members, 1), rtol=2e-2, atol=2e-2
        )


def test_distinct_query_key_extents_and_empty_partition_merge():
    from uniserve.attention.base import merge_attention_states
    from uniserve.attention.video_sparse_cute import block_sparse_attention

    torch.manual_seed(891)
    heads, query_rows, key_rows, width = 7, 128, 256, 128
    query = torch.randn(query_rows, heads, width, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(key_rows, heads, width, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    valid = torch.tensor([64, 13, 64, 0], device="cuda", dtype=torch.int32)
    indices = torch.zeros((heads, 2, 4), device="cuda", dtype=torch.int32)
    indices[:, 0, :2] = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    indices[:, 1, :2] = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    counts = torch.full((heads, 2), 2, device="cuda", dtype=torch.int32)
    output = torch.empty_like(query)
    lse = torch.empty((heads, query_rows), device="cuda", dtype=torch.float32)
    full = block_sparse_attention(query, key, value, output, indices, counts, valid, lse=lse)
    mask = torch.zeros((heads, query_rows, key_rows), device="cuda", dtype=torch.bool)
    mask[:, :64, :64] = True
    mask[:, 64:, 64:77] = True
    mask[:, :, 128:192] = True
    logits = query.transpose(0, 1).double() @ key.transpose(0, 1).double().transpose(-1, -2)
    logits.mul_(width**-0.5).masked_fill_(~mask, -torch.inf)
    reference_lse = logits.logsumexp(-1)
    reference = logits.softmax(-1) @ value.transpose(0, 1).double()
    torch.testing.assert_close(full[0].double(), reference, rtol=2e-2, atol=2e-2)
    # FP32 statistics preserve enough precision for stable partition weighting;
    # this bound is tighter than BF16's relative unit roundoff (2**-8).
    torch.testing.assert_close(lse.double(), reference_lse, rtol=1e-4, atol=1e-4)

    merged = None
    for owner in range(2):
        offset = owner * 2
        local_indices = indices - offset
        selected = (local_indices >= 0) & (local_indices < 2)
        selected &= torch.arange(4, device="cuda").view(1, 1, -1) < counts.unsqueeze(-1)
        # One entirely empty head exercises the zero-mass contract on both
        # partitions. Its reference output and LSE are defined explicitly.
        selected[0] = False
        local_counts = selected.sum(-1, dtype=torch.int32)
        local_indices = torch.where(selected, local_indices, 2).sort(-1).values
        partial_output = torch.empty_like(query)
        partial_lse = torch.empty_like(lse)
        partial = block_sparse_attention(
            query,
            key[owner * 128 : (owner + 1) * 128],
            value[owner * 128 : (owner + 1) * 128],
            partial_output,
            local_indices,
            local_counts,
            valid[offset : offset + 2].clone(),
            lse=partial_lse,
            allow_empty_blocks=True,
        )[0].contiguous()
        state = partial, partial_lse
        merged = state if merged is None else merge_attention_states(*merged, *state)
    assert merged is not None
    assert torch.count_nonzero(merged[0][0]) == 0
    assert torch.isneginf(merged[1][0]).all()
    torch.testing.assert_close(merged[0][1:].double(), reference[1:], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(merged[1][1:].double(), reference_lse[1:], rtol=1e-4, atol=1e-4)


def test_partial_attention_preserves_cancellation_until_all_keys_are_merged():
    from uniserve.attention.base import merge_attention_states
    from uniserve.attention.video_sparse_cute import block_sparse_attention

    query = torch.zeros(128, 7, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.zeros_like(query)
    value = torch.full_like(query, -256)
    value[:32] = 256
    value[32:64] = 258
    indices = torch.zeros(7, 2, 1, device="cuda", dtype=torch.int32)
    counts = torch.ones(7, 2, device="cuda", dtype=torch.int32)
    valid = torch.full((1,), 64, device="cuda", dtype=torch.int32)
    states = []
    for start in (0, 64):
        output = torch.empty_like(query, dtype=torch.float32)
        lse = torch.empty(7, 128, device="cuda", dtype=torch.float32)
        attended = block_sparse_attention(
            query,
            key[start : start + 64],
            value[start : start + 64],
            output,
            indices,
            counts,
            valid,
            lse=lse,
            allow_empty_blocks=True,
        )
        states.append((attended[0].contiguous(), lse))
    merged, _ = merge_attention_states(*states[0], *states[1])
    # Equal logits define the exact mean: (32*256 + 32*258 - 64*256)/128.
    # Rounding the first partition's mean 257 to BF16 before merging loses it.
    torch.testing.assert_close(merged, torch.full_like(merged, 0.5), rtol=0, atol=0)


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_sparse_softmax_underflow_and_empty_key_partitions(output_dtype):
    from uniserve.attention.video_sparse_cute import block_sparse_attention

    torch.manual_seed(772)
    rows, heads, width = 4096, 7, 128
    # Alternating zero and large logits require zero and nonzero softmax
    # rescaling within the same warp. Empty tiles additionally require zero
    # mass when the persistent kernel reuses its accumulator storage.
    query = torch.zeros(rows, heads, width, device="cuda", dtype=torch.bfloat16)
    query[::2, :, 0] = 32
    key = torch.zeros(1024, heads, width, device="cuda", dtype=torch.bfloat16)
    key[:512, :, 0] = -32
    key[512:, :, 0] = 32
    value = torch.randn_like(key)
    counts = (
        (
            torch.arange(rows // 64, device="cuda").view(1, -1)
            + torch.arange(heads, device="cuda").view(-1, 1)
        )
        % 3
        != 0
    ).to(torch.int32) * 16
    indices = torch.arange(16, device="cuda", dtype=torch.int32)
    indices = indices.expand(heads, rows // 64, -1).contiguous()
    valid_sizes = torch.full((16,), 64, device="cuda", dtype=torch.int32)
    valid_sizes[-1] = 37
    output = torch.empty_like(query, dtype=output_dtype)
    lse = torch.empty(heads, rows, device="cuda")
    actual = block_sparse_attention(
        query,
        key,
        value,
        output,
        indices,
        counts,
        valid_sizes,
        lse=lse,
        allow_empty_blocks=True,
    )[0]
    live = counts.repeat_interleave(64, 1).bool()
    logits = query.transpose(0, 1).double() @ key[:997].transpose(0, 1).double().transpose(-1, -2)
    logits.mul_(width**-0.5)
    reference = logits.softmax(-1) @ value[:997].transpose(0, 1).double()
    torch.testing.assert_close(actual[live].double(), reference[live], rtol=2e-2, atol=2e-2)
    assert torch.count_nonzero(actual[~live]) == 0
    assert torch.isneginf(lse[~live]).all()
    torch.testing.assert_close(lse[live].double(), logits.logsumexp(-1)[live], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("provider_name", ["native", "flashinfer", "triton"])
@torch.inference_mode()
@pytest.mark.parametrize("chunk_rows", [None, 64])
def test_sparse_attention_uses_declared_per_head_cardinalities(provider_name, chunk_rows):
    """Nonuniform sparsity remains correct across replanning and device-map replay."""

    provider = _sparse_provider(provider_name)
    torch.manual_seed(537)
    rows, heads, width, tile = 256, 4, 128, 64
    query, key, value, gate = torch.randn(
        4, rows, heads, width, device="cuda", dtype=torch.bfloat16
    ).unbind(0)
    # A selected empty tile can precede live keys or be the entire visible domain.
    # Neither case may contaminate the online softmax state for later tiles.
    valid = torch.tensor([0, 23, 64, 51], device="cuda", dtype=torch.int32)
    # Valid logits can carry much less mass than zero-padded keys. Masking
    # must preserve their distribution instead of subtracting two rounded sums.
    query[:, 0].zero_()
    key[:, 0].zero_()
    query[:, 0, 0] = 16
    key[:, 0, 0] = -16
    indices = torch.arange(4, device="cuda", dtype=torch.int32).expand(heads, 4, 4).clone()
    counts = torch.empty(heads, 4, device="cuda", dtype=torch.int32)
    compressed = torch.randn(heads, 4, width, device="cuda")
    scratch, output = torch.empty_like(query), torch.empty_like(query)
    live = torch.arange(rows, device="cuda") % tile < valid.repeat_interleave(tile)

    for shift in range(2):
        pattern = SparseAttentionPattern(
            tuple(tuple(1 + (head + row + shift) % 4 for row in range(4)) for head in range(heads))
        )
        counts.copy_(torch.tensor(pattern.row_counts, device="cuda", dtype=torch.int32))

        def invoke():
            arguments = dict(
                mask_block_indices=indices,
                mask_block_count=counts,
                valid_sizes=valid,
                pattern=pattern,
                gate=gate,
                compressed=compressed,
                attention_output=scratch,
            )
            if chunk_rows is None:
                provider.execute(
                    query,
                    key,
                    value,
                    **arguments,
                    tile_size=tile,
                    targets=AttentionOutputTargets((output,), 0),
                )
            else:
                produce = provider.prepare_rows(
                    query, key, value, **arguments, owners=1, chunk_rows=chunk_rows
                )
                for start in range(0, rows, chunk_rows):
                    produce(slice(start, start + chunk_rows), (output[start : start + chunk_rows],))

        invoke()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            invoke()
        for selected_shift in range(2):
            indices.copy_(
                (torch.arange(4, device="cuda", dtype=torch.int32) + selected_shift)
                .remainder(4)
                .expand_as(indices)
            )
            selected = torch.arange(4, device="cuda").view(1, 1, 4) < counts.unsqueeze(-1)
            blocks = torch.zeros(heads, 4, 4, device="cuda", dtype=torch.bool)
            blocks.scatter_(2, indices.long(), selected)
            mask = (
                blocks.repeat_interleave(tile, 1).repeat_interleave(tile, 2) & live[None, None, :]
            )
            attended = F.scaled_dot_product_attention(
                query.transpose(0, 1).double(),
                key.transpose(0, 1).double(),
                value.transpose(0, 1).double(),
                attn_mask=mask,
            )
            expected = (
                attended
                + gate.transpose(0, 1).double() * compressed.double().repeat_interleave(tile, 1)
            ).transpose(0, 1)
            graph.replay()
            torch.testing.assert_close(output[live].double(), expected[live], rtol=2e-2, atol=2e-2)
        del graph


@pytest.mark.parametrize("provider_name", ["native", "flashinfer", "triton"])
@torch.inference_mode()
def test_sparse_execution_domains_preserve_independent_replay_inputs(provider_name):
    """Equal-geometry domains keep their own maps and scratch on concurrent streams."""

    device = torch.device("cuda", 0)
    torch.manual_seed(529)
    rows, heads, width, tile = 256, 4, 128, 64
    domains = []
    executions = []
    for domain in range(2):
        provider = _sparse_provider(provider_name)
        query, key, value, gate = torch.randn(
            4, rows, heads, width, device=device, dtype=torch.bfloat16
        ).unbind(0)
        valid = torch.tensor([64, 17 + domain * 14, 64, 0], device=device, dtype=torch.int32)
        indices = torch.zeros(heads, 4, 3, device=device, dtype=torch.int32)
        indices[:, 0] = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
        indices[:, 1:3, 1] = domain + 1
        counts = torch.tensor([3, 2, 2, 1], device=device, dtype=torch.int32)
        counts = counts.repeat(heads, 1)
        compressed = torch.randn(heads, 4, width, device=device)
        scratch, output = torch.empty_like(query), torch.empty_like(query)
        stream = torch.cuda.Stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))

        def invoke(
            provider=provider,
            query=query,
            key=key,
            value=value,
            gate=gate,
            indices=indices,
            counts=counts,
            valid=valid,
            compressed=compressed,
            scratch=scratch,
            output=output,
        ):
            produce = provider.prepare_rows(
                query,
                key,
                value,
                mask_block_indices=indices,
                mask_block_count=counts,
                valid_sizes=valid,
                pattern=SparseAttentionPattern(((3, 2, 2, 1),), 1, 3),
                gate=gate,
                compressed=compressed,
                attention_output=scratch,
                owners=1,
                chunk_rows=64,
            )
            for start in range(0, rows, tile):
                produce(slice(start, start + tile), (output[start : start + tile],))

        with torch.cuda.stream(stream):
            invoke()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            invoke()
        # Graph replay borrows external input and plan storage. Retain each
        # execution owner, including its count buffers and attention scratch.
        executions.append(invoke)
        domains.append((graph, stream, query, key, value, gate, valid, indices, compressed, output))

    try:
        for replay in range(2):
            expected = []
            for domain, (
                _,
                stream,
                query,
                key,
                value,
                gate,
                valid,
                indices,
                compressed,
                _,
            ) in enumerate(domains):
                selected = 1 + (domain + replay) % 2
                indices[:, 1:3, 1] = selected
                query.mul_(0.75)
                live = torch.arange(rows, device=device) % tile < valid.repeat_interleave(tile)
                mask = torch.zeros(rows, rows, device=device, dtype=torch.bool)
                mask[:tile, : 3 * tile] = True
                mask[tile : 3 * tile, :tile] = True
                mask[tile : 3 * tile, selected * tile : (selected + 1) * tile] = True
                mask &= live[None, :]
                attention = F.scaled_dot_product_attention(
                    query.transpose(0, 1).double(),
                    key.transpose(0, 1).double(),
                    value.transpose(0, 1).double(),
                    attn_mask=mask,
                ).transpose(0, 1)
                reference = attention + gate.double() * compressed.double().repeat_interleave(
                    tile, dim=1
                ).transpose(0, 1)
                expected.append((live, reference.to(torch.bfloat16)))
                stream.wait_stream(torch.cuda.current_stream(device))
            for graph, stream, *_ in domains:
                with torch.cuda.stream(stream):
                    graph.replay()
            for (_, stream, *_, output), (live, reference) in zip(domains, expected, strict=True):
                stream.synchronize()
                torch.testing.assert_close(output[live], reference[live], rtol=2e-2, atol=2e-2)
    finally:
        for graph, stream, *_ in domains:
            stream.synchronize()
            graph.reset()
