"""Region-wise VSA reproduces its dense equations over 128-row tiles.

Dense tiles attend every live tile; every video query tile attends the live
dense tiles and, of each region, the key tiles of highest pooled score that
the region keeps; empty tiles neither attend nor are attended; and each row
adds its gate times the softmax over live tiles of their mean values.
"""

import pytest
import torch
import torch.nn.functional as F

from uniserve.nn.attention import vsa
from uniserve.runtime import ExecutionContext

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

TILE, HEADS, WIDTH = 128, 4, 128
# Two dense tiles, an empty one, a region of three tiles, another dense
# tile, a region of four tiles and an empty alignment tile.
VALID = (128, 40, 0, 128, 128, 96, 77, 128, 128, 128, 64, 0)
REGIONS = (-1, -1, -1, 0, 0, 0, -1, 1, 1, 1, 1, -1)
KEEP = (2, 1)


def _regions(device) -> vsa.Regions:
    regions = torch.tensor(REGIONS, dtype=torch.int32)
    starts = torch.zeros(len(VALID), dtype=torch.int32)
    keep = torch.zeros(len(VALID), dtype=torch.int32)
    for region, kept in enumerate(KEEP):
        starts[region] = int((regions < region).sum())
        keep[region] = kept
    return vsa.Regions(
        TILE,
        len(VALID) * TILE,
        torch.tensor(VALID, dtype=torch.int32, device=device),
        regions.to(device),
        starts.to(device),
        keep.to(device),
    )


def _reference(q, k, v, gate):
    """Evaluate selection, fine attention and compression in FP64."""
    tiles = len(VALID)
    valid = torch.tensor(VALID, device=q.device)
    regions = torch.tensor(REGIONS, device=q.device)
    live = torch.arange(tiles * TILE, device=q.device) % TILE < (
        valid.repeat_interleave(TILE)
    )
    means = []
    for value in (q, k, v):
        rows = value.double().view(tiles, TILE, HEADS, WIDTH)
        means.append(
            (
                rows.masked_fill(~live.view(tiles, TILE, 1, 1), 0).sum(1)
                / valid.clamp_min(1).view(-1, 1, 1)
            ).transpose(0, 1)
        )
    scores = means[0] @ means[1].transpose(-1, -2) / WIDTH**0.5
    occupied = valid > 0
    dense, video = occupied & (regions < 0), occupied & (regions >= 0)
    mask = torch.zeros(HEADS, tiles, tiles, dtype=torch.bool, device=q.device)
    mask[:, dense] = occupied
    mask[:, video] = dense
    for region, kept in enumerate(KEEP):
        members = torch.nonzero(regions == region).flatten()
        best = scores[:, :, members].topk(kept, dim=-1).indices
        chosen = torch.zeros_like(mask)
        chosen.scatter_(-1, members[best], True)
        mask |= chosen & video.view(1, -1, 1)
    rows = mask.repeat_interleave(TILE, 1).repeat_interleave(TILE, 2)
    rows &= live.view(1, 1, -1)
    attending = live & (occupied.repeat_interleave(TILE))
    fine = torch.zeros(HEADS, tiles * TILE, WIDTH, dtype=torch.float64)
    fine = fine.to(q.device)
    fine[:, attending] = F.scaled_dot_product_attention(
        q.transpose(0, 1).double()[:, attending],
        k.transpose(0, 1).double(),
        v.transpose(0, 1).double(),
        attn_mask=rows[:, attending],
    )
    compressed = (
        scores.masked_fill(~occupied.view(1, 1, -1), -torch.inf).softmax(-1)
        @ means[2]
    )
    result = fine.transpose(0, 1) + gate.double() * compressed.transpose(
        0, 1
    ).repeat_interleave(TILE, dim=0)
    return result.to(q.dtype), attending


@torch.inference_mode()
def test_region_attention_matches_its_dense_equations():
    torch.manual_seed(2024)
    # Interleaved projections exercise the strided rows a merged projection
    # produces.
    projected = torch.randn(
        len(VALID) * TILE, HEADS, 4, WIDTH, device="cuda", dtype=torch.bfloat16
    )
    q, k, v, gate = projected.unbind(2)
    module = vsa.RegionAttention(
        vsa.BlockAttention(WIDTH**-0.5, tile_size=TILE)
    )
    regions = _regions(q.device)

    with ExecutionContext(module) as context:
        context.prepare(None)
        with context.activate():
            actual = module(q, k, v, gate, regions)

    expected, attending = _reference(q, k, v, gate)
    # The established BF16 attention tolerance of the VSA tests covers the
    # kernel's softmax and MMA rounding and the BF16 compression terms.
    torch.testing.assert_close(
        actual[attending], expected[attending], rtol=2e-2, atol=2e-2
    )
