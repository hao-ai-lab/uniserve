"""Dense and grouped-query attention through public sequence communication."""

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from uniserve.distributed.parallel import ParallelConfig, SequenceParallel
from uniserve.distributed.process_groups import initialize_model_parallel, initialize_process_groups
from uniserve.nn.parallel_attention import ParallelAttention, context_scope
from uniserve.runtime.attention_storage import allocate_context_storage

pytestmark = pytest.mark.integration


def _run_attention(rank: int, rendezvous: str) -> None:
    environment = initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=4,
        device="cpu",
        backend="gloo",
        init_method=rendezvous,
    )
    meshes = initialize_model_parallel(
        environment,
        {
            kind: (
                (3, 1, 2, 0),
                ParallelConfig(sequence_parallel=SequenceParallel(kind, (4,))),
            )
            for kind in ("ulysses", "allgather")
        },
    )
    try:
        generator = torch.Generator().manual_seed(714)
        rows, query_heads, width = 20, 8, 32
        for kv_heads in (1, 2, 4, 8):
            query = torch.randn(rows, query_heads, width, dtype=torch.float64, generator=generator)
            key = torch.randn(rows, kv_heads, width, dtype=torch.float64, generator=generator)
            value = torch.randn(rows, kv_heads, width, dtype=torch.float64, generator=generator)
            for causal in (False, True):
                reference = F.scaled_dot_product_attention(
                    query.transpose(0, 1),
                    key.transpose(0, 1),
                    value.transpose(0, 1),
                    is_causal=causal,
                    enable_gqa=True,
                ).transpose(0, 1)
                for kind, mesh in meshes.items():
                    attention = ParallelAttention(mesh=mesh)
                    local_rows = rows // 4
                    begin = mesh.coord("sp") * local_rows
                    local_query = attention.exchange_heads(query[begin : begin + local_rows])
                    local_key = attention.exchange_heads(key[begin : begin + local_rows])
                    local_value = attention.exchange_heads(value[begin : begin + local_rows])
                    contexts = allocate_context_storage(
                        (attention,),
                        rows=local_rows,
                        heads=kv_heads,
                        head_dim=width,
                        dtype=torch.float64,
                        block_size=1,
                    )
                    with context_scope(contexts):
                        local_key, local_value = attention.distribute_key_value(
                            local_key, local_value
                        )
                        if kind == "allgather" and kv_heads == 1 and not causal:
                            other = allocate_context_storage(
                                (attention,),
                                rows=local_rows,
                                heads=kv_heads,
                                head_dim=width,
                                dtype=torch.float64,
                                block_size=1,
                            )
                            with pytest.raises(ValueError, match="caller exit"):
                                with context_scope(other):
                                    other_key, other_value = attention.distribute_key_value(
                                        key[begin : begin + local_rows] + 3,
                                        value[begin : begin + local_rows] - 2,
                                    )
                                    raise ValueError("caller exit")
                            torch.testing.assert_close(local_key, key)
                            torch.testing.assert_close(local_value, value)
                            # The enclosing caller is restored after an error;
                            # publishing there cannot overwrite the other caller.
                            local_key, local_value = attention.distribute_key_value(
                                key[begin : begin + local_rows],
                                value[begin : begin + local_rows],
                            )
                            torch.testing.assert_close(other_key, key + 3)
                            torch.testing.assert_close(other_value, value - 2)
                        query_begin = begin if kind == "allgather" else 0
                        mask = None
                        if causal:
                            mask = (
                                torch.arange(local_query.shape[0])[:, None] + query_begin
                                >= torch.arange(rows)[None, :]
                            )
                        output = F.scaled_dot_product_attention(
                            local_query.transpose(0, 1),
                            local_key.transpose(0, 1),
                            local_value.transpose(0, 1),
                            attn_mask=mask,
                            enable_gqa=True,
                        ).transpose(0, 1)
                        attention.finish_context()
                        actual = attention.restore_rows(output)
                        torch.testing.assert_close(actual, reference[begin : begin + local_rows])
                    if kind == "allgather":
                        with pytest.raises(RuntimeError, match="bound numerical buffers"):
                            attention.distribute_key_value(
                                key[begin : begin + local_rows], value[begin : begin + local_rows]
                            )
    finally:
        environment.close()


def test_sequence_communication_preserves_dense_and_gqa_attention(tmp_path):
    mp.spawn(_run_attention, ((tmp_path / "rendezvous").as_uri(),), nprocs=4, join=True)
