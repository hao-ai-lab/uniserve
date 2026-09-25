"""Public VSA providers preserve selected keys and compression.

They also preserve mutable graph inputs.
"""

from importlib import import_module

import pytest
import torch

from tests.python.fixtures.vsa import reference_attention
from uniserve.nn.attention import vsa
from uniserve.runtime import CUDAGraph, CUDAStream, ExecutionContext

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.fixture
def stream():
    owner = CUDAStream.external(torch.cuda.Stream())
    yield owner
    owner.close()


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
    )


@pytest.mark.parametrize("provider", ["cute", "flashinfer", "triton"])
@torch.inference_mode()
def test_selection_compression_and_projected_chunks(provider, stream):
    torch.manual_seed(518)
    projections = torch.randn(
        256, 7, 4, 128, device="cuda", dtype=torch.bfloat16
    )
    q, k, v, gate = projections.unbind(2)
    valid = torch.tensor([64, 17, 64, 0], device="cuda", dtype=torch.int32)
    inputs, workspace = _input(valid), _workspace(q)
    module = vsa.Attention(vsa.BlockAttention(128**-0.5))
    expected, live = reference_attention(q, k, v, gate, valid)
    stream.wait(torch.cuda.current_stream())
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


@pytest.mark.parametrize("provider", ["sm100", "cute", "flashinfer", "triton"])
@torch.inference_mode()
def test_projected_chunks_apply_the_layer_softmax_scale(provider, stream):
    from uniserve.runtime.backends.attention import vsa as providers

    if not import_module(f"{providers.__name__}.{provider}").available(
        torch.device("cuda")
    ):
        pytest.skip(f"VSA provider {provider} is unavailable on this device")

    torch.manual_seed(733)
    q, k, v, gate = torch.randn(
        256, 3, 4, 128, device="cuda", dtype=torch.bfloat16
    ).unbind(2)
    valid = torch.tensor([64, 23, 64, 0], device="cuda", dtype=torch.int32)
    inputs, workspace = _input(valid), _workspace(q)
    scale = 0.021
    module = vsa.Attention(vsa.BlockAttention(scale))
    expected, live = reference_attention(q, k, v, gate, valid, scale=scale)

    stream.wait(torch.cuda.current_stream())
    with ExecutionContext(module, stream=stream, vsa=provider) as context:
        context.prepare(None)
        result = torch.empty_like(q)
        chunks = (
            (
                slice(start, stop),
                tuple(value[start:stop] for value in (q, k, v, gate)),
            )
            for start, stop in ((0, 128), (128, 256))
        )
        for interval, output in module.forward_chunks(
            chunks, inputs, selected_tiles=1, workspace=workspace
        ):
            result[interval].copy_(output)
        torch.testing.assert_close(
            result[live], expected[live], rtol=2e-2, atol=2e-2
        )


@torch.inference_mode()
def test_norm_rope_prepared_chunks_match_normalized_projections(stream):
    """Raw chunks with a norm-rope attend like pre-normalized chunks."""
    torch.manual_seed(1119)
    projections = torch.randn(
        256, 7, 4, 128, device="cuda", dtype=torch.bfloat16
    )
    q, k, v, gate = projections.unbind(2)
    valid = torch.tensor([64, 64, 25, 0], device="cuda", dtype=torch.int32)
    query_weight = torch.rand(128, device="cuda", dtype=torch.bfloat16) + 0.5
    key_weight = torch.rand(128, device="cuda", dtype=torch.bfloat16) + 0.5
    angles = torch.rand(256, 48, device="cuda") * 6.0
    cos, sin = angles.cos(), angles.sin()
    inputs, workspace = _input(valid), _workspace(q)
    module = vsa.Attention(vsa.BlockAttention(128**-0.5))

    # The reference normalizes and rotates the projections up front with the
    # public functional, then attends the prepared chunks.
    from uniserve.nn.functional import qk_norm_rope

    normalized_q, normalized_k = qk_norm_rope(
        q,
        k,
        (query_weight,),
        (key_weight,),
        (cos,),
        (sin,),
        eps=1e-6,
        axis_dims=(128,),
    )

    def chunks(query, key):
        for start, stop in ((0, 64), (64, 192), (192, 256)):
            yield (
                slice(start, stop),
                tuple(value[start:stop] for value in (query, key, v, gate)),
            )

    def attend(query, key, norm_rope):
        result = torch.empty_like(q)
        for interval, output in module.forward_chunks(
            chunks(query, key),
            inputs,
            selected_tiles=1,
            workspace=workspace,
            norm_rope=norm_rope,
        ):
            result[interval].copy_(output)
        return result

    stream.wait(torch.cuda.current_stream())
    with ExecutionContext(module, stream=stream, vsa="cute") as context:
        context.prepare(None)
        expected = attend(normalized_q, normalized_k, None).clone()
        actual = attend(
            q,
            k,
            vsa.NormRope(query_weight, key_weight, 1e-6, cos, sin),
        )
        live = torch.arange(256, device="cuda") < 64 * 2 + 25
        # Both paths round the prepared rows to bf16 after fp32 normalization
        # and rotation; the tolerance is one bf16 ulp on inputs and outputs.
        torch.testing.assert_close(
            actual[live], expected[live], rtol=2**-7, atol=2**-7
        )
        torch.cuda.synchronize()


@torch.inference_mode()
def test_row_production_over_many_intervals_matches_one_call(stream):
    """Rows produced interval by interval equal the single-call result.

    Small exchange intervals give several packed segments; the fine
    attention is launched once for the whole domain and every interval
    composes its own rows.
    """
    torch.manual_seed(2207)
    projections = torch.randn(
        256, 5, 4, 128, device="cuda", dtype=torch.bfloat16
    )
    q, k, v, gate = projections.unbind(2)
    valid = torch.tensor([64, 64, 40, 0], device="cuda", dtype=torch.int32)
    inputs, workspace = _input(valid), _workspace(q)
    module = vsa.Attention(vsa.BlockAttention(128**-0.5))
    live = torch.arange(256, device="cuda") < 64 * 2 + 40

    stream.wait(torch.cuda.current_stream())
    with ExecutionContext(module, stream=stream, vsa="cute") as context:
        context.prepare(None)
        batch = module.select(
            q, k, inputs, selected_tiles=1, workspace=workspace
        )
        expected = module(q, k, v, gate, batch, workspace=workspace).clone()

        produced = torch.zeros_like(q)
        with module.attention._operator(q, batch) as operator:
            producer = operator.rows(
                q,
                k,
                v,
                batch,
                gate=gate,
                compressed=workspace.compressed_tiles,
                out=workspace.attention_output,
                owners=1,
                chunk_tokens=64,
                packed=None,
                scale=module.attention.scale,
            )
            for start in range(0, 256, 64):
                producer(
                    slice(start, start + 64), (produced[start : start + 64],)
                )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            produced[live], expected[live], rtol=2e-2, atol=2e-2
        )


@pytest.mark.parametrize("provider", ["sm100", "cute", "flashinfer", "triton"])
@torch.inference_mode()
def test_empty_prefix_tiles_leave_live_rows_unchanged(provider, stream):
    """A prefix tile without valid rows changes no live row's attention.

    The capacity domain holds a text region of two tiles for a prompt that
    fills one: its second text tile has no valid rows and carries large
    values, and the live prefix lists name only the text and audio tiles.
    Every live row must attend exactly as in the domain without that tile.
    """
    from uniserve.runtime.backends.attention import vsa as providers

    if not import_module(f"{providers.__name__}.{provider}").available(
        torch.device("cuda")
    ):
        pytest.skip(f"VSA provider {provider} is unavailable on this device")

    torch.manual_seed(4211)
    device = torch.device("cuda")
    heads = 3
    # Exact domain: text, audio, four video tiles, two padding tiles.
    exact_valid = [40, 30, 64, 64, 64, 50, 0, 0]
    # Capacity domain: text, empty text, audio, four video tiles, padding.
    capacity_valid = [40, 0, 30, 64, 64, 64, 50, 0]
    live_rows = torch.cat(
        [
            torch.arange(tile * 64, tile * 64 + count)
            for tile, count in enumerate(exact_valid)
        ]
    ).to(device)
    capacity_tiles = torch.tensor([0, 2, 3, 4, 5, 6], device=device)
    capacity_rows = torch.cat(
        [
            torch.arange(tile * 64, tile * 64 + capacity_valid[tile])
            for tile in capacity_tiles.tolist()
        ]
    ).to(device)

    exact = torch.randn(512, heads, 4, 128, device=device, dtype=torch.bfloat16)
    capacity = torch.full_like(exact, 100.0)
    capacity[capacity_rows] = exact[live_rows]

    def domain(valid, prefix_tiles, live_prefix):
        valid = torch.tensor(valid, device=device, dtype=torch.int32)
        prefix = torch.zeros(prefix_tiles, device=device, dtype=torch.int32)
        prefix[: len(live_prefix)] = torch.tensor(live_prefix)
        dense = torch.zeros(prefix_tiles + 4, device=device, dtype=torch.int32)
        dense[: len(live_prefix) + 4] = torch.tensor(
            [*live_prefix, *range(prefix_tiles, prefix_tiles + 4)]
        )
        inputs = vsa.Input(
            padded_tokens=512,
            prefix_tiles=prefix_tiles,
            video_tiles=4,
            valid_tiles=prefix_tiles + 4,
            valid_sizes=valid,
            prefix_key_indices=prefix,
            dense_key_indices=dense,
            prefix_count=torch.tensor(
                [len(live_prefix)], device=device, dtype=torch.int32
            ),
        )

        def tensor(shape, dtype=torch.float32):
            return torch.empty(shape, device=device, dtype=dtype)

        workspace = vsa.Workspace(
            attention_output=tensor((512, heads, 128), torch.bfloat16),
            tile_scores=tensor((heads, 8, 8)),
            block_counts=tensor((heads, 8), torch.int32),
            block_indices=tensor((heads, 8, 8), torch.int32),
            pooled_query=tensor((8, heads, 128)),
            pooled_key=tensor((8, heads, 128)),
            pooled_value=tensor((8, heads, 128)),
            compressed_tiles=tensor((heads, 8, 128)),
        )
        return inputs, workspace

    module = vsa.Attention(vsa.BlockAttention(128**-0.5))
    stream.wait(torch.cuda.current_stream())
    with ExecutionContext(module, stream=stream, vsa=provider) as context:
        context.prepare(None)
        results = []
        for projections, (inputs, workspace) in (
            (exact, domain(exact_valid, 2, (0, 1))),
            (capacity, domain(capacity_valid, 3, (0, 2))),
        ):
            q, k, v, gate = projections.unbind(2)
            result = torch.empty_like(q)
            chunks = (
                (
                    slice(start, stop),
                    tuple(value[start:stop] for value in (q, k, v, gate)),
                )
                for start, stop in ((0, 256), (256, 512))
            )
            for interval, output in module.forward_chunks(
                chunks, inputs, selected_tiles=2, workspace=workspace
            ):
                result[interval].copy_(output)
            results.append(result)
        torch.testing.assert_close(
            results[1][capacity_rows],
            results[0][live_rows],
            rtol=2e-2,
            atol=2e-2,
        )
