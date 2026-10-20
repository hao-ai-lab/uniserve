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
PROVIDERS = ["sm100", "sm120", "flashinfer", "triton"]


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


# Providers of 128-row tiles. The window covers sequence tiles 2-6, whose
# valid rows are 128, 100, 64, 30 and 0; the key blocks add whole blocks,
# blocks with more and with at most 64 valid rows, an empty block, and a
# block of 7 rows.
WIDE_PROVIDERS = ["sm100", "sm90", "sm120"]
WIDE_TILE = 128
WIDE_VALID = (64, 128, 128, 100, 64, 30, 0, 7)
WIDE_WINDOW = range(2, 7)


def _wide_selection(heads, generator):
    """Per-head key blocks of each window tile, in no particular order.

    The first tile selects every block; the others select several partial
    blocks together, an odd number of them on some heads; the empty tile
    selects none.
    """
    choices = (
        tuple(range(len(WIDE_VALID))),
        (0, 2, 4, 5, 7),
        (1, 5, 6),
        (3, 7),
        (),
    )
    indices = torch.zeros(
        heads, len(WIDE_WINDOW), len(WIDE_VALID), dtype=torch.int32
    )
    counts = torch.zeros(heads, len(WIDE_WINDOW), dtype=torch.int32)
    for head in range(heads):
        for tile, chosen in enumerate(choices):
            chosen = list(chosen)
            if head % 2 and len(chosen) > 1:
                chosen = chosen[1:]
            order = torch.randperm(len(chosen), generator=generator).tolist()
            for slot, position in enumerate(order):
                indices[head, tile, slot] = chosen[position]
            counts[head, tile] = len(chosen)
    return indices, counts


def _wide_reference(q, k, v, indices, counts, valid):
    """FP64 attention of each window row over its selected valid keys."""
    tiles, heads = len(valid), q.shape[1]
    keys = torch.arange(tiles * WIDE_TILE, device=q.device)
    live_keys = keys % WIDE_TILE < torch.tensor(
        valid, device=q.device
    ).repeat_interleave(WIDE_TILE)
    selected = torch.zeros(
        heads, len(WIDE_WINDOW), tiles, dtype=torch.bool, device=q.device
    )
    for head in range(heads):
        for tile in range(len(WIDE_WINDOW)):
            count = int(counts[head, tile])
            selected[head, tile, indices[head, tile, :count].long()] = True
    mask = selected.repeat_interleave(WIDE_TILE, 1).repeat_interleave(
        WIDE_TILE, 2
    ) & live_keys.view(1, 1, -1)
    # A row attends when its tile is valid there and selects a valid key.
    rows = torch.arange(len(WIDE_WINDOW) * WIDE_TILE, device=q.device)
    window_valid = torch.tensor(
        [valid[tile] for tile in WIDE_WINDOW], device=q.device
    ).repeat_interleave(WIDE_TILE)
    attending = (rows % WIDE_TILE < window_valid).view(1, -1) & mask.any(-1)
    expected = torch.zeros(
        heads, rows.numel(), q.shape[2], dtype=torch.float64, device=q.device
    )
    for head in range(heads):
        live = attending[head]
        expected[head, live] = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1).double()[head, live],
            k.transpose(0, 1).double()[head],
            v.transpose(0, 1).double()[head],
            attn_mask=mask[head, live],
            scale=WIDE_TILE**-0.5,
        )
    return expected, attending


@pytest.mark.parametrize("provider", WIDE_PROVIDERS, indirect=True)
@torch.inference_mode()
def test_wide_tiles_attend_their_selected_valid_keys(provider, stream):
    """128-row tiles attend exactly the valid keys of their selected blocks.

    The query tiles are a window of the sequence, as a context partition
    holds, and read row-major views of merged projections. Keys and values
    past every valid size carry large finite payloads that must not
    contribute. A captured call replays the dense equations after the
    projections, a valid size and the selection change.
    """
    generator = torch.Generator().manual_seed(1287)
    device = torch.device("cuda")
    heads = 3
    tiles = len(WIDE_VALID)
    valid = list(WIDE_VALID)
    projections = torch.randn(
        tiles * WIDE_TILE,
        heads,
        3,
        128,
        generator=generator,
        dtype=torch.bfloat16,
    ).to(device)
    window = slice(WIDE_WINDOW.start * WIDE_TILE, WIDE_WINDOW.stop * WIDE_TILE)
    q = projections[window, :, 0]
    k, v = projections[:, :, 1], projections[:, :, 2]
    sizes = torch.tensor(valid, dtype=torch.int32, device=device)
    indices, counts = (
        value.to(device) for value in _wide_selection(heads, generator)
    )

    def pad():
        live = torch.arange(tiles * WIDE_TILE, device=device) % WIDE_TILE < (
            sizes.repeat_interleave(WIDE_TILE)
        )
        k[~live], v[~live] = 300.0, -300.0

    pad()
    batch = vsa.BlockInput(
        vsa.Pattern(((tiles,) * len(WIDE_WINDOW),), 0, 0, WIDE_TILE),
        indices,
        counts,
        sizes,
        WIDE_WINDOW.start,
    )
    module = vsa.BlockAttention(WIDE_TILE**-0.5)

    def check(actual):
        expected, attending = _wide_reference(q, k, v, indices, counts, valid)
        actual = actual.transpose(0, 1).double()
        torch.testing.assert_close(
            actual[attending], expected[attending], rtol=2e-2, atol=2e-2
        )

    stream.wait(torch.cuda.current_stream())
    with ExecutionContext(module, stream=stream, vsa=provider) as context:
        context.prepare(None)
        check(module(q, k, v, batch).clone())

        replayed = torch.empty_like(q)
        with CUDAGraph(context=context) as graph:
            graph.capture(lambda: replayed.copy_(module(q, k, v, batch)))
            # The 100-row block drops to 60 rows, below half a block, and the
            # third tile selects the first block in place of the sixth.
            projections.mul_(0.5)
            valid[3] = 60
            sizes[3] = 60
            pad()
            indices[:, 2][indices[:, 2] == 5] = 0
            graph.replay()
            check(replayed)


@torch.inference_mode()
def test_sm120_attends_strided_projections_in_place():
    """SM120 reads strided projection views exactly as contiguous copies.

    Segment attention passes the query, key and value as views of one
    ``[rows, q | k | v | gate]`` matrix. The provider launches FlashInfer's
    kernel on those views, which must give the output and log-sum-exp the
    public wrapper gives on contiguous copies, bit for bit.
    """
    from uniserve.runtime.backends.attention.vsa import sm120

    device = torch.device("cuda")
    if not sm120.available(device):
        pytest.skip("the SM120 provider is unavailable on this device")
    torch.manual_seed(1201)
    tiles, heads = 9, 4
    rows = tiles * TILE
    projections = torch.randn(
        rows, 4, heads, 128, device=device, dtype=torch.bfloat16
    )
    q, k, v, _ = projections.unbind(1)
    assert all(sm120._in_place(value) for value in (q, k, v))
    sizes = torch.tensor(
        (64, 17, 64, 40, 0, 64, 23, 64, 1), device=device, dtype=torch.int32
    )
    # Every query tile of every head keeps a random leading share of a
    # random order of the key tiles.
    indices = torch.stack(
        [
            torch.stack([torch.randperm(tiles) for _ in range(tiles)])
            for _ in range(heads)
        ]
    ).to(device=device, dtype=torch.int32)
    counts = torch.randint(
        1, tiles + 1, (heads, tiles), device=device, dtype=torch.int32
    )
    scale = 128**-0.5

    out = torch.empty_like(q)
    lse = torch.empty((1, heads, rows), device=device, dtype=torch.float32)
    sm120._launch(
        q, k, v, indices, counts, sizes, scale=scale, out=out, lse=lse
    )
    want = torch.empty_like(out)
    want_lse = torch.empty_like(lse)
    sm120.bsa_attn_sm120_blk64_fwd(
        q.contiguous().unsqueeze(0),
        k.contiguous().unsqueeze(0),
        v.contiguous().unsqueeze(0),
        indices.unsqueeze(0),
        tiles,
        block_sizes=sizes,
        q2k_block_nums=counts.unsqueeze(0),
        softmax_scale=scale,
        out=want.unsqueeze(0),
        lse=want_lse,
    )
    assert torch.equal(out, want)
    assert torch.equal(lse, want_lse)
