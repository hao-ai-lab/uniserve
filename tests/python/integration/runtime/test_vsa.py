"""Public VSA providers preserve selected keys and compression.

They also preserve mutable graph inputs.
"""

import gc
from importlib import import_module

import pytest
import torch

from tests.python.fixtures.vsa import layout_tables, segment_reference
from uniserve.nn.attention import vsa
from uniserve.runtime import CUDAGraph, CUDAStream, ExecutionContext

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.fixture
def provider(request):
    from uniserve.runtime.backends.attention import vsa as providers

    name = request.param
    if not import_module(f"{providers.__name__}.{name}").available(
        torch.device("cuda")
    ):
        pytest.skip(f"VSA provider {name} is unavailable on this device")
    return name


@pytest.fixture
def stream():
    owner = CUDAStream.external(torch.cuda.Stream())
    yield owner
    owner.close()


@pytest.fixture
def cuda_cache():
    """Retire large cached allocations before another GPU consumer starts."""
    yield
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.usefixtures("cuda_cache")
@torch.inference_mode()
def test_a_plan_beyond_the_provider_mask_range_is_refused(stream):
    """The FlashInfer provider refuses a plan its mask offsets cannot address.

    1600 query tiles of 56 heads selecting 384 tiles each need 34 million
    selected tiles, beyond the 4.19 million that 32-bit packed-mask byte
    offsets address; the plan is refused before any launch.
    """
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the FlashInfer provider is validated on SM100 only")

    tiles, heads, selected, width = 1600, 56, 384, 128
    rows = tiles * 64
    q = torch.zeros(rows, heads, width, device="cuda", dtype=torch.bfloat16)
    batch = vsa.BlockInput(
        vsa.Pattern(((selected,) * tiles,), 0, 0, 64),
        torch.arange(selected, device="cuda", dtype=torch.int32)
        .expand(heads, tiles, -1)
        .contiguous(),
        torch.full((heads, tiles), selected, device="cuda", dtype=torch.int32),
        torch.full((tiles,), 64, device="cuda", dtype=torch.int32),
        0,
    )
    module = vsa.BlockAttention(128**-0.5)

    stream.wait(torch.cuda.current_stream())
    with ExecutionContext(module, stream=stream, vsa="flashinfer") as context:
        context.prepare(None)
        with pytest.raises(RuntimeError, match="selected tiles per plan"):
            module(q, q, q, batch, out=torch.empty_like(q))


# Tiles of 64 rows: a dense tile, a segment of three tiles, an empty dense
# tile and a segment of two tiles, the odd count a layout may have.
TILE = 64
INDICES = (-1, 0, 0, 0, -1, 1, 1)
KEEP = (2, 1)
VALID = (64, 17, 64, 40, 0, 64, 23)
PROVIDERS = ["sm100", "flashinfer", "triton"]


def _segments(device, valid=VALID):
    """The layout's tables, which a captured call reads at fixed addresses."""
    return vsa.Segments(
        TILE,
        len(valid) * TILE,
        *layout_tables(valid, INDICES, KEEP, device),
        dense_prefix_tiles=1,
        # A video query attends at most the dense tile and 2 + 1 kept tiles;
        # backends plan from the bound.
        key_bounds=(len(valid),) + (4,) * (len(valid) - 1),
    )


@pytest.mark.parametrize("provider", PROVIDERS, indirect=True)
@torch.inference_mode()
def test_selection_compression_and_graph_replay(provider, stream):
    torch.manual_seed(518)
    device = torch.device("cuda")
    projections = torch.randn(
        len(VALID) * TILE, 7, 4, 128, device=device, dtype=torch.bfloat16
    )
    valid = list(VALID)
    segments = _segments(device)
    module = vsa.SegmentAttention(vsa.BlockAttention(128**-0.5))

    def attend():
        return module(*projections.unbind(2), segments)

    def expected():
        # Attention zeroes padding rows of its inputs in place, so the dense
        # equations read a copy.
        return segment_reference(
            *projections.clone().unbind(2),
            valid=valid,
            indices=INDICES,
            keep=KEEP,
            tile=TILE,
        )

    stream.wait(torch.cuda.current_stream())
    with ExecutionContext(module, stream=stream, vsa=provider) as context:
        context.prepare(None)
        want, attending = expected()
        actual = attend().clone()
        torch.testing.assert_close(
            actual[attending], want[attending], rtol=2e-2, atol=2e-2
        )

        replayed = torch.empty_like(actual)
        with CUDAGraph(context=context) as graph:
            graph.capture(lambda: replayed.copy_(attend()))
            # Numerical inputs and the tables change without changing the
            # prepared shapes.
            projections.mul_(0.5)
            valid[1] = 9
            segments.variable_block_sizes[1] = 9
            want, attending = expected()
            graph.replay()
            torch.testing.assert_close(
                replayed[attending], want[attending], rtol=2e-2, atol=2e-2
            )


@pytest.mark.parametrize("provider", PROVIDERS, indirect=True)
@torch.inference_mode()
def test_fine_attention_applies_the_layer_softmax_scale(provider, stream):
    torch.manual_seed(733)
    device = torch.device("cuda")
    tiles, heads, scale = 3, 3, 0.021
    q, k, v = torch.randn(
        tiles * TILE, 3, heads, 128, device=device, dtype=torch.bfloat16
    ).unbind(1)
    valid = torch.tensor([64, 23, 0], device=device, dtype=torch.int32)
    # Every query tile attends both live key tiles; the empty tile attends
    # nothing.
    indices = torch.tensor([[0, 1, 2]] * tiles, device=device).int()
    batch = vsa.BlockInput(
        vsa.Pattern(((2, 2, 2),), 0, 0, TILE),
        indices.expand(heads, tiles, 3).contiguous(),
        torch.tensor([2, 2, 0], device=device, dtype=torch.int32)
        .expand(heads, tiles)
        .contiguous(),
        valid,
        0,
    )
    module = vsa.BlockAttention(scale)
    stream.wait(torch.cuda.current_stream())
    with ExecutionContext(module, stream=stream, vsa=provider) as context:
        context.prepare(None)
        actual = module(q, k, v, batch)

    live = torch.arange(tiles * TILE, device=device) % TILE < (
        valid.repeat_interleave(TILE)
    )
    keys = live.clone()
    keys[2 * TILE :] = False
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).double(),
        k.transpose(0, 1).double(),
        v.transpose(0, 1).double(),
        attn_mask=keys.view(1, 1, -1),
        scale=scale,
    ).transpose(0, 1)
    attending = live.clone()
    attending[2 * TILE :] = False
    torch.testing.assert_close(
        actual[attending].double(),
        expected[attending],
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("provider", PROVIDERS, indirect=True)
@torch.inference_mode()
def test_empty_dense_tiles_leave_live_rows_unchanged(provider, stream):
    """A dense tile without valid rows changes no live row's attention.

    The capacity layout holds a text region of two tiles for a prompt that
    fills one: its second text tile has no valid rows and carries large
    values. Every live row must attend exactly as in the layout without that
    tile.
    """
    torch.manual_seed(4211)
    device = torch.device("cuda")
    heads = 3
    # Exact layout: text, audio, four video tiles, two padding tiles.
    exact = {
        "valid": (40, 30, 64, 64, 64, 50, 0, 0),
        "indices": (-1, -1, 0, 0, 0, 0, -1, -1),
    }
    # Capacity layout: text, empty text, audio, four video tiles, padding.
    capacity = {
        "valid": (40, 0, 30, 64, 64, 64, 50, 0),
        "indices": (-1, -1, -1, 0, 0, 0, 0, -1),
    }
    live_rows = torch.cat(
        [
            torch.arange(tile * TILE, tile * TILE + count)
            for tile, count in enumerate(exact["valid"])
        ]
    ).to(device)
    capacity_rows = torch.cat(
        [
            torch.arange(tile * TILE, tile * TILE + count)
            for tile, count in enumerate(capacity["valid"])
            if count
        ]
    ).to(device)

    values = torch.randn(
        8 * TILE, heads, 4, 128, device=device, dtype=torch.bfloat16
    )
    padded = torch.full_like(values, 100.0)
    padded[capacity_rows] = values[live_rows]

    module = vsa.SegmentAttention(vsa.BlockAttention(128**-0.5))
    stream.wait(torch.cuda.current_stream())
    with ExecutionContext(module, stream=stream, vsa=provider) as context:
        context.prepare(None)
        results = []
        for projections, layout in ((values, exact), (padded, capacity)):
            q, k, v, gate = projections.unbind(2)
            segments = vsa.Segments(
                TILE,
                8 * TILE,
                *layout_tables(
                    layout["valid"], layout["indices"], (2,), device
                ),
                dense_prefix_tiles=1,
            )
            results.append(module(q, k, v, gate, segments).clone())
        torch.testing.assert_close(
            results[1][capacity_rows],
            results[0][live_rows],
            rtol=2e-2,
            atol=2e-2,
        )
