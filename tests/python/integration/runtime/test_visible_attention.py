"""Visible attention reads live sequence boundaries.

It also reads partially visible tiles.
"""

from contextlib import ExitStack

import pytest
import torch
import torch.nn.functional as F

from uniserve.model import TextSize
from uniserve.nn.attention import SequenceLengths, VisibleInput
from uniserve.runtime import TensorBuffers
from uniserve.runtime.backends.attention import resolve

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("provider", ["flash_attn_4", "flashinfer"])
@pytest.mark.parametrize("variable_length", [False, True])
@torch.inference_mode()
def test_prefix_tiles_preserve_live_visibility_and_sequence_boundaries(
    variable_length, provider
):
    """Block skipping and mask evaluation preserve empty and full tiles.

    Partial tiles are preserved as well.
    """
    device = torch.device("cuda", 0)
    torch.manual_seed(793)
    query_lengths = [193, 301] if variable_length else [301, 301]
    key_lengths = [129, 385] if variable_length else [385, 385]
    maximum_query = 301
    query = torch.randn(
        sum(query_lengths), 4, 128, device=device, dtype=torch.bfloat16
    )
    key = torch.randn(
        sum(key_lengths), 2, 128, device=device, dtype=torch.bfloat16
    )
    value = torch.randn_like(key)
    visible = torch.empty(2, maximum_query, device=device, dtype=torch.int32)
    cu_query = torch.zeros(3, device=device, dtype=torch.int32)
    cu_key = torch.zeros_like(cu_query)
    backend = resolve(provider, device=device)
    query_values = torch.empty(2, device=device, dtype=torch.int32)
    key_values = torch.empty_like(query_values)

    def update(iteration):
        cu_query.copy_(
            torch.tensor(
                [0, query_lengths[0], sum(query_lengths)], device=device
            )
        )
        cu_key.copy_(
            torch.tensor([0, key_lengths[0], sum(key_lengths)], device=device)
        )
        for index, (q_len, k_len) in enumerate(
            zip(query_lengths, key_lengths, strict=True)
        ):
            positions = torch.arange(maximum_query, device=device)
            # Inactive padding must never enlarge the tile's visible KV range.
            limits = (positions * 7 + iteration * 137).remainder(k_len + 1)
            if iteration == 0 and index == 0:
                limits.zero_()
            else:
                limits[positions < min(256, q_len)] = k_len
            visible[index].copy_(torch.where(positions < q_len, limits, 12345))

    def inputs():
        query_values.copy_(
            torch.tensor(query_lengths, device=device, dtype=torch.int32)
        )
        key_values.copy_(
            torch.tensor(key_lengths, device=device, dtype=torch.int32)
        )
        return VisibleInput(
            SequenceLengths(
                host=tuple(query_lengths), values=query_values, offsets=cu_query
            ),
            SequenceLengths(
                host=tuple(key_lengths), values=key_values, offsets=cu_key
            ),
            visible,
            None,
            True,
            False,
        )

    def reference():
        values = []
        q_begin = k_begin = 0
        for index, (q_len, k_len) in enumerate(
            zip(query_lengths, key_lengths, strict=True)
        ):
            mask = (
                torch.arange(k_len, device=device)[None]
                < visible[index, :q_len, None]
            )
            values.append(
                F.scaled_dot_product_attention(
                    query[q_begin : q_begin + q_len].double().transpose(0, 1),
                    key[k_begin : k_begin + k_len].double().transpose(0, 1),
                    value[k_begin : k_begin + k_len].double().transpose(0, 1),
                    attn_mask=mask,
                    enable_gqa=True,
                )
                .transpose(0, 1)
                .to(query.dtype)
            )
            q_begin += q_len
            k_begin += k_len
        return torch.cat(values)

    arguments = {
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 128,
        "dtype": query.dtype,
        "size": TextSize(sum(query_lengths), 2),
        "cache": None,
    }
    requirements = backend.workspace_buffers(**arguments)
    with ExitStack() as scope:
        buffers = scope.enter_context(
            TensorBuffers.allocate(requirements, device=device)
        )
        operator = backend.prepare(
            **arguments, workspace=buffers.view(requirements)
        )
        scope.callback(operator.close)
        update(0)
        batch = inputs()
        operator.bind(batch)
        actual = torch.empty_like(query)
        operator(query, key, value, batch, scale=128**-0.5, out=actual)
        torch.testing.assert_close(actual, reference(), atol=0.01, rtol=0.01)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        scope.callback(graph.reset)
        with torch.cuda.graph(graph):
            operator(query, key, value, batch, scale=128**-0.5, out=actual)
        if variable_length:
            query_lengths[:] = [194, 300]
            key_lengths[:] = [257, 257]
        update(1)
        value.mul_(0.75)
        operator.bind(inputs())
        graph.replay()
        torch.testing.assert_close(actual, reference(), atol=0.01, rtol=0.01)
