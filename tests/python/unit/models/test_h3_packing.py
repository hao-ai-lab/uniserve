"""Conditioned layouts reproduce the released H3 packed sequences.

The released diffusers pipeline packs ``[text | conditions | target audio |
target video]``; the dense and segment layouts hold the same rows in other
physical orders. Every row's rotary coordinates and modulation tag must equal
the reference's bit for bit, and the condition rows must read the timesteps
the reference holds them at.
"""

from types import SimpleNamespace

import pytest
import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3Ref2VAPrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
)

from uniserve.media import image, video
from uniserve.model import Condition, ConditionRole
from uniserve.nn.attention import vsa
from uniserve_models.minimax_h3.packing import (
    AUDIO_CONDITION_GROUP,
    AUDIO_GROUP,
    AUDIO_TAG,
    VIDEO_GROUP,
    VIDEO_TAG,
    VISUAL_CONDITION_GROUP,
    audio_latent_frames,
    condition_segments,
    dense_packing,
    dense_tables,
    latent_raster,
    segment_packing,
    segment_tables,
    segment_tiles,
    video_latent_frames,
)

pytestmark = pytest.mark.unit

WIDE = image.Config(768, 1344)
TALL = image.Config(1344, 768)


def _text_tags(length: int, spans: tuple[tuple[int, int], ...]):
    tags = torch.ones(length, dtype=torch.long)
    for start, stop in spans:
        tags[start:stop] = VIDEO_TAG
    return tags


def _reference_order(tables, packing, text_rows: int, condition_rows: int):
    """Our rows in the reference's ``[text | conditions | audio | video]``."""
    prefix = packing.prefix_start
    order = torch.cat(
        (
            torch.arange(prefix, prefix + text_rows + condition_rows),
            torch.arange(packing.video_rows, prefix),
            torch.arange(0, packing.video_rows),
        )
    )
    return (
        tables.position_ids.index_select(0, order),
        tables.token_tags.index_select(0, order),
        tables.groups.index_select(0, order),
    )


def _tables(frames, canvas, text_rows, spans, conditions):
    segments = condition_segments(conditions, canvas)
    condition_rows = sum(s.video_rows + s.audio_rows for s in segments)
    packing = dense_packing(
        num_frames=frames,
        canvas=canvas,
        text_rows=text_rows + 64,
        condition_rows=condition_rows + 64,
        token_multiple=512,
    )
    tables = dense_tables(
        packing,
        num_frames=frames,
        canvas=canvas,
        num_text_tokens=text_rows,
        segments=segments,
        vision_spans=spans,
    )
    return packing, tables, condition_rows


def _assert_condition_groups(
    groups, video_indices, audio_indices, conditions, text_rows
):
    """Each row reads the timestep group the reference holds it at.

    The reference's row timesteps, built from distinct stand-in values for
    the generated video and audio and the two condition levels, name each
    row's group.
    """
    visual, audio = conditions
    timesteps, index = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_indices,
        audio_indices,
        visual,
        audio,
        text_rows,
        0.25,
        0.5,
        0.999,
        1.0,
    )
    held = {
        0.25: VIDEO_GROUP,
        0.5: AUDIO_GROUP,
        0.999: VISUAL_CONDITION_GROUP,
        1.0: AUDIO_CONDITION_GROUP,
    }
    by_time = torch.tensor(
        [held[round(float(value), 3)] for value in timesteps]
    )
    assert torch.equal(groups, by_time[index])


# 8 s is 192 frames, 57 latent frames: past 16, so the last keyframe's
# pairwise-summed anchor differs from a sequential sum in the last place.
KEYFRAME_FRAMES, KEYFRAME_TEXT = 192, 1935
KEYFRAME_SPANS = ((11, 1027), (1040, 1100))
ANCHORS = [("first",), ("last",), ("first", "last")]


def _keyframes(canvas, anchors) -> tuple[Condition, ...]:
    roles = {
        "first": ConditionRole.FIRST_FRAME,
        "last": ConditionRole.LAST_FRAME,
    }
    return tuple(
        Condition(roles[anchor], video.Config(1, canvas)) for anchor in anchors
    )


def _released_fl2va(canvas, anchors):
    """The released fl2va packed sequence of the keyframe cases."""
    height, width = latent_raster(canvas)
    return MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        _text_tags(KEYFRAME_TEXT, KEYFRAME_SPANS),
        video_latent_frames(KEYFRAME_FRAMES),
        height,
        width,
        audio_latent_frames(KEYFRAME_FRAMES),
        (1, 2, 2),
        2,
        AUDIO_TAG,
        VIDEO_TAG,
        anchors,
    )


@pytest.mark.parametrize("canvas", [WIDE, TALL])
@pytest.mark.parametrize("anchors", ANCHORS)
def test_keyframes_follow_the_released_fl2va_layout(canvas, anchors):
    frames, text_rows, spans = KEYFRAME_FRAMES, KEYFRAME_TEXT, KEYFRAME_SPANS
    packing, tables, condition_rows = _tables(
        frames, canvas, text_rows, spans, _keyframes(canvas, anchors)
    )
    (
        positions,
        tags,
        video_indices,
        audio_indices,
        _,
        visual,
        audio,
    ) = _released_fl2va(canvas, anchors)
    ours, our_tags, our_groups = _reference_order(
        tables, packing, text_rows, condition_rows
    )
    assert torch.equal(ours, positions)
    assert torch.equal(our_tags, tags)
    assert (visual, audio) == (condition_rows, 0)
    _assert_condition_groups(
        our_groups, video_indices, audio_indices, (visual, audio), text_rows
    )
    # Attention sees the generated rows, the prompt and the conditions.
    assert tables.used == packing.prefix_start + text_rows + condition_rows


@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("canvas", [WIDE, TALL])
@pytest.mark.parametrize("anchors", ANCHORS)
def test_keyframes_follow_the_released_fl2va_layout_in_tiles(
    tile, canvas, anchors
):
    """Sparse students hold the released keyframe rows in dense tiles."""
    frames, text_rows, spans = KEYFRAME_FRAMES, KEYFRAME_TEXT, KEYFRAME_SPANS
    segments = condition_segments(_keyframes(canvas, anchors), canvas)
    packing = segment_packing(
        num_frames=frames,
        canvas=canvas,
        text_rows=-(-text_rows // tile) * tile,
        condition_rows=sum(segment_tiles(s, tile) for s in segments) * tile,
        tile=tile,
        token_multiple=4 * tile,
    )
    tables = segment_tables(
        packing,
        num_frames=frames,
        canvas=canvas,
        num_text_tokens=text_rows,
        segments=segments,
        vision_spans=spans,
        sparsity=0.9,
        reference_keep=None,
    )
    (
        positions,
        tags,
        video_indices,
        audio_indices,
        _,
        visual,
        audio,
    ) = _released_fl2va(canvas, anchors)

    # Our packed row of every row of the reference's packed order: the
    # prompt and the keyframes by the prefix source rows they gather, then
    # the generated audio and the generated video in raster order.
    taken = tables.prefix_index != packing.zero_row
    source = torch.full((packing.zero_row,), -1, dtype=torch.int64)
    source[tables.prefix_index[taken]] = torch.nonzero(taken).flatten()
    order = torch.cat(
        (
            source[:text_rows],
            source[packing.text_rows : packing.text_rows + visual],
            packing.audio_indices,
            packing.video_indices[packing.video_raster_indices.argsort()],
        )
    )
    assert bool((order >= 0).all())
    assert torch.equal(tables.position_ids.index_select(0, order), positions)
    assert torch.equal(tables.token_tags.index_select(0, order), tags)
    assert audio == 0
    _assert_condition_groups(
        tables.groups.index_select(0, order),
        video_indices,
        audio_indices,
        (visual, audio),
        text_rows,
    )

    # Every keyframe starts a tile of its own after the prompt's tiles, and
    # every query attends its tiles densely.
    keyframe_rows = order[text_rows : text_rows + visual]
    per_keyframe = visual // len(anchors)
    tiles_per_keyframe = segment_tiles(segments[0], tile)
    assert torch.equal(
        keyframe_rows[::per_keyframe],
        packing.text_rows
        + torch.arange(len(anchors)) * tiles_per_keyframe * tile,
    )
    used = torch.unique(keyframe_rows // tile)
    assert used.numel() == len(anchors) * tiles_per_keyframe
    assert bool((tables.tile_segments[used] == vsa.DENSE_TILE).all())
    assert int(tables.variable_block_sizes[used].sum()) == visual


def _reference_latents(condition: Condition):
    """Shapes the reference encoders produce for one condition."""
    visual = []
    if condition.video is not None:
        height, width = latent_raster(condition.video.frame)
        frames = (
            1
            if condition.video.num_frames == 1
            else video_latent_frames(condition.video.num_frames)
        )
        visual.append(torch.empty(1, 24, frames, height, width))
    audio = []
    if condition.audio_samples:
        rows = 2 * -(-condition.audio_samples // 800)
        audio.append(torch.empty(rows, 32))
    return visual, audio


@pytest.mark.parametrize(
    "references",
    [
        # A 2048-short-edge image reference.
        (("image", image.Config(1152, 2048), 1, 0),),
        # An image, then an audio track.
        (
            ("image", image.Config(2048, 2048), 1, 0),
            ("audio", None, 0, 165_600),
        ),
        # A video with its soundtrack: 22 latent frames span less time than
        # its 240 audio latents.
        (("video", WIDE, 107, 192_000),),
        # Two videos, one silent; 57 latent frames exceed the soundtrack.
        (
            ("video", image.Config(768, 1024), 192, 64_000),
            ("video", TALL, 107, 0),
            ("audio", None, 0, 32_000),
        ),
    ],
)
def test_references_follow_the_released_ref2va_layout(references):
    frames, text_rows = 124, 700
    spans = ((5, 300), (320, 600))
    conditions = tuple(
        Condition(
            ConditionRole.REFERENCE,
            None if kind == "audio" else video.Config(count, raster),
            samples,
        )
        for kind, raster, count, samples in references
    )
    packing, tables, condition_rows = _tables(
        frames, WIDE, text_rows, spans, conditions
    )
    visual_latents, audio_latents = [], []
    for condition in conditions:
        visual, audio = _reference_latents(condition)
        visual_latents += visual
        audio_latents += audio
    height, width = latent_raster(WIDE)
    (
        positions,
        tags,
        video_indices,
        audio_indices,
        _,
        visual,
        audio,
    ) = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        _text_tags(text_rows, spans),
        [
            SimpleNamespace(kind=kind, has_audio=bool(samples))
            for kind, _, _, samples in references
        ],
        visual_latents,
        audio_latents,
        video_latent_frames(frames),
        height,
        width,
        audio_latent_frames(frames),
        (1, 2, 2),
        2,
        AUDIO_TAG,
        VIDEO_TAG,
    )
    ours, our_tags, our_groups = _reference_order(
        tables, packing, text_rows, condition_rows
    )
    assert torch.equal(ours, positions)
    assert torch.equal(our_tags, tags)
    assert visual + audio == condition_rows
    _assert_condition_groups(
        our_groups, video_indices, audio_indices, (visual, audio), text_rows
    )


def test_prefix_rows_gather_the_prompt_then_the_conditions():
    conditions = (Condition(ConditionRole.FIRST_FRAME, video.Config(1, WIDE)),)
    packing, tables, condition_rows = _tables(124, WIDE, 100, (), conditions)
    prefix = packing.prefix_start
    gathered = tables.prefix_index
    assert torch.equal(gathered[prefix : prefix + 100], torch.arange(100))
    assert torch.equal(
        gathered[prefix + 100 : prefix + 100 + condition_rows],
        torch.arange(packing.text_rows, packing.text_rows + condition_rows),
    )
    # Generated and padding rows gather the source's zero row.
    rest = torch.cat(
        (gathered[:prefix], gathered[prefix + 100 + condition_rows :])
    )
    assert torch.all(rest == packing.zero_row)


def test_conditions_must_fit_the_layout():
    conditions = (Condition(ConditionRole.FIRST_FRAME, video.Config(1, WIDE)),)
    segments = condition_segments(conditions, WIDE)
    packing = dense_packing(
        num_frames=124,
        canvas=WIDE,
        text_rows=128,
        condition_rows=64,
        token_multiple=512,
    )
    with pytest.raises(ValueError, match="prefix segment"):
        dense_tables(
            packing,
            num_frames=124,
            canvas=WIDE,
            num_text_tokens=100,
            segments=segments,
        )
    # Keyframes are fitted to the generated canvas.
    with pytest.raises(ValueError, match="target canvas"):
        condition_segments(
            (Condition(ConditionRole.LAST_FRAME, video.Config(1, TALL)),), WIDE
        )
