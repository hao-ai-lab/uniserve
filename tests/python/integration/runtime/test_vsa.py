"""Public VSA providers preserve selected keys and compression.

They also preserve mutable graph inputs.
"""

import pytest
import torch

from tests.python.fixtures.vsa import reference_attention
from uniserve.nn.attention import vsa
from uniserve.runtime import CUDAGraph, ExecutionContext

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _input(valid):
    return vsa.Input(
        padded_tokens=256,
        prefix_tiles=1,
        video_tiles=2,
        valid_tiles=3,
        valid_sizes=valid,
        prefix_key_indices=torch.tensor(
            [0], device=valid.device, dtype=torch.int32
        ),
        dense_key_indices=torch.arange(
            3, device=valid.device, dtype=torch.int32
        ),
        prefix_count=torch.tensor(1, device=valid.device, dtype=torch.int32),
    )


def _workspace(q):
    rows, heads, width = q.shape

    def tensor(shape, dtype=torch.float32):
        return torch.empty(shape, device=q.device, dtype=dtype)

    return vsa.Workspace(
        attention_output=tensor(q.shape, q.dtype),
        tile_scores=tensor((heads, 4, 4)),
        block_counts=tensor((heads, 4), torch.int32),
        block_indices=tensor((heads, 4, 3), torch.int32),
        pooled_query=tensor((4, heads, width)),
        pooled_key=tensor((4, heads, width)),
        pooled_value=tensor((4, heads, width)),
        compressed_tiles=tensor((heads, 4, width)),
        topk_indices=tensor((heads, 2, 1), torch.int32),
    )


@pytest.mark.parametrize("provider", ["cute", "flashinfer", "triton"])
@torch.inference_mode()
def test_selection_compression_and_projected_chunks(provider):
    torch.manual_seed(518)
    projections = torch.randn(
        256, 7, 4, 128, device="cuda", dtype=torch.bfloat16
    )
    q, k, v, gate = projections.unbind(2)
    valid = torch.tensor([64, 17, 64, 0], device="cuda", dtype=torch.int32)
    inputs, workspace = _input(valid), _workspace(q)
    module = vsa.Attention(vsa.BlockAttention(128**-0.5))
    expected, live = reference_attention(q, k, v, gate, valid)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with ExecutionContext(module, stream=stream, vsa=provider) as context:
        context.prepare(None)
        batch = module.select(
            q, k, inputs, selected_tiles=1, workspace=workspace
        )
        actual = module(q, k, v, gate, batch, workspace=workspace)
        torch.testing.assert_close(
            actual[live], expected[live], rtol=2e-2, atol=2e-2
        )
        result = torch.empty_like(actual)

        def chunks():
            for start, stop in ((0, 64), (64, 192), (192, 256)):
                yield (
                    slice(start, stop),
                    tuple(value[start:stop] for value in (q, k, v, gate)),
                )

        def invoke():
            for interval, output in module.forward_chunks(
                chunks(), inputs, selected_tiles=1, workspace=workspace
            ):
                result[interval].copy_(output)
            return result

        invoke()
        torch.testing.assert_close(
            result[live], expected[live], rtol=2e-2, atol=2e-2
        )
        with CUDAGraph(context=context) as graph:
            graph.capture(invoke)
            # Numerical inputs change without changing the prepared tile shape.
            projections.mul_(0.5)
            valid[1] = 9
            graph.replay()
            expected, live = reference_attention(q, k, v, gate, valid)
            torch.testing.assert_close(
                result[live], expected[live], rtol=2e-2, atol=2e-2
            )
