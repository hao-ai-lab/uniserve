"""Multi-segment sparse layouts reproduce FastVideo's OmniRef packing.

FastVideo's FastH3 OmniRef path packs ``[text | references | target audio |
target video]`` and tiles it per segment: text, every
audio track, images and the target audio form dense tiles of their own
segment, and each reference video and the target video form a segment of
4x4x8 token tiles whose tiles every video query selects independently. The
segment layout holds the same rows in another physical order with spare text
and condition tiles. Every row's rotary coordinates, tag and timestep group
must equal the reference's, every tile must hold the same rows in the same
order, and selection must keep the reference's key tiles. The vectors,
``minimax_h3_segments.json``, are FastVideo's packing of these cases.
"""

import json
import math
from pathlib import Path

import pytest
import torch

from uniserve.media import image, video
from uniserve.model import Condition, ConditionRole
from uniserve.nn.attention import vsa
from uniserve.nn.attention.vsa.segments import select
from uniserve_models.minimax_h3.packing import (
    AUDIO_CONDITION_GROUP,
    AUDIO_GROUP,
    VIDEO_GROUP,
    VISUAL_CONDITION_GROUP,
    condition_segments,
    segment_packing,
    segment_tables,
    segment_tiles,
)

pytestmark = pytest.mark.unit

FIXTURE = json.loads(
    (
        Path(__file__).parents[2] / "fixtures" / "minimax_h3_segments.json"
    ).read_text()
)
CASES = {case["name"]: case for case in FIXTURE["cases"]}
GROUP_TIMESTEPS = {
    VIDEO_GROUP: FIXTURE["group_timesteps"]["video"],
    AUDIO_GROUP: FIXTURE["group_timesteps"]["audio"],
    VISUAL_CONDITION_GROUP: FIXTURE["group_timesteps"]["visual_condition"],
    AUDIO_CONDITION_GROUP: FIXTURE["group_timesteps"]["audio_condition"],
}


def _conditions(case) -> tuple[Condition, ...]:
    return tuple(
        Condition(
            ConditionRole.REFERENCE,
            None
            if not condition["frames"]
            else video.Config(
                condition["frames"],
                image.Config(condition["height"], condition["width"]),
            ),
            condition["samples"],
        )
        for condition in case["conditions"]
    )


def _layout(case):
    """The case's segment layout, with one spare text and condition tile."""
    tile = FIXTURE["tile"]
    canvas = image.Config(case["height"], case["width"])
    segments = condition_segments(_conditions(case), canvas)
    packing = segment_packing(
        num_frames=case["frames"],
        canvas=canvas,
        text_rows=(math.ceil(case["text_tokens"] / tile) + 1) * tile,
        condition_rows=(
            sum(segment_tiles(segment, tile) for segment in segments) + 1
        )
        * tile,
        tile=tile,
        token_multiple=4 * tile,
    )
    tables = segment_tables(
        packing,
        num_frames=case["frames"],
        canvas=canvas,
        num_text_tokens=case["text_tokens"],
        segments=segments,
        vision_spans=tuple(tuple(span) for span in case["vision_spans"]),
        sparsity=FIXTURE["sparsity"],
        reference_keep=FIXTURE["reference_keep"],
    )
    return packing, tables, segments


def _reference_rows(case, packing, tables, segments) -> torch.Tensor:
    """Our packed row of every row of the reference's packed order."""
    taken = tables.prefix_index != packing.zero_row
    source = torch.full((packing.zero_row,), -1, dtype=torch.int64)
    source[tables.prefix_index[taken]] = torch.nonzero(taken).flatten()
    conditions = sum(s.video_rows + s.audio_rows for s in segments)
    order = torch.cat(
        (
            source[: case["text_tokens"]],
            source[packing.text_rows : packing.text_rows + conditions],
            packing.audio_indices,
            packing.video_indices[packing.video_raster_indices.argsort()],
        )
    )
    assert order.numel() == len(case["position_ids"])
    assert bool((order >= 0).all())
    return order


def _tile_pairs(case, packing, order) -> dict[int, int]:
    """Map each reference tile to ours, checking every row's slot."""
    tile = FIXTURE["tile"]
    slots = torch.tensor(case["row_slots"])
    pairs: dict[int, int] = {}
    for reference, ours in zip(
        (slots // tile).tolist(), (order // tile).tolist(), strict=True
    ):
        assert pairs.setdefault(reference, ours) == ours
    # Each row also sits at its reference offset within the tile.
    assert torch.equal(slots % tile, order % tile)
    assert len(set(pairs.values())) == len(pairs)
    return pairs


@pytest.mark.parametrize("name", sorted(CASES))
def test_rows_follow_the_reference_packing(name):
    case = CASES[name]
    packing, tables, segments = _layout(case)
    order = _reference_rows(case, packing, tables, segments)

    expected = torch.tensor(case["position_ids"], dtype=torch.float64)
    assert torch.equal(tables.position_ids.index_select(0, order), expected)
    assert torch.equal(
        tables.token_tags.index_select(0, order),
        torch.tensor(case["token_tags"]),
    )
    timesteps = torch.tensor(
        [GROUP_TIMESTEPS[int(group)] for group in tables.groups[order]],
        dtype=torch.float32,
    )
    assert torch.equal(
        timesteps, torch.tensor(case["row_timesteps"], dtype=torch.float32)
    )
    # Rows outside the request gather the zero row of the prefix source.
    outside = torch.ones(packing.padded_tokens, dtype=torch.bool)
    outside[order] = False
    assert bool((tables.prefix_index[outside] == packing.zero_row).all())


@pytest.mark.parametrize("name", sorted(CASES))
def test_tiles_follow_the_reference_segments(name):
    case = CASES[name]
    packing, tables, segments = _layout(case)
    pairs = _tile_pairs(
        case, packing, _reference_rows(case, packing, tables, segments)
    )
    reference_sizes = case["valid_sizes"]
    for reference, ours in pairs.items():
        assert int(tables.valid_sizes[ours]) == reference_sizes[reference]
        if reference < case["prefix_tiles"]:
            assert int(tables.tile_segments[ours]) == vsa.DENSE_TILE
        else:
            segment = next(
                index
                for index, (start, stop) in enumerate(case["segment_spans"])
                if start <= reference < stop
            )
            assert int(tables.tile_segments[ours]) == segment
    # Every other tile, spare capacity and alignment alike, is empty.
    empty = torch.ones(tables.valid_sizes.numel(), dtype=torch.bool)
    empty[list(pairs.values())] = False
    assert not tables.valid_sizes[empty].any()


@pytest.mark.parametrize("name", sorted(CASES))
def test_selection_keeps_the_reference_key_tiles(name):
    case = CASES[name]
    packing, tables, segments = _layout(case)
    pairs = _tile_pairs(
        case, packing, _reference_rows(case, packing, tables, segments)
    )
    ours = torch.tensor([pairs[index] for index in range(len(pairs))])
    reference_scores = torch.tensor(case["scores"])
    heads, tiles = reference_scores.shape[0], tables.valid_sizes.numel()
    # Empty tiles score highest, so selection must exclude them by state.
    scores = torch.full((heads, tiles, tiles), 1e4)
    scores[:, ours[:, None], ours[None, :]] = reference_scores
    segments = vsa.Segments(
        packing.tile,
        packing.padded_tokens,
        tables.valid_sizes,
        tables.tile_segments,
        tables.segment_starts,
        tables.segment_keep,
    )

    indices, counts = select(scores, segments)

    expected = torch.zeros((heads, tiles, tiles), dtype=torch.bool)
    for head, rows in enumerate(case["mask"]):
        for query, keys in enumerate(rows):
            expected[head, ours[query], ours[keys]] = True
    kept = torch.zeros_like(expected)
    for head in range(heads):
        for query in range(tiles):
            listed = indices[head, query, : counts[head, query]].long()
            assert bool((listed[1:] > listed[:-1]).all())
            kept[head, query, listed] = True
    assert torch.equal(kept, expected)
