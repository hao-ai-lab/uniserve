"""H3 continuation clips reproduce FastVideo's packing and size their rows.

A ``ref2va`` request may continue an earlier video from its last 5, 22 or
39 frames. FastVideo's ``continuation_video`` (FastVideo#2001) packs the
clip after the references: it advances the rotary clock as a video
reference with its soundtrack does, then its latent frames take the
coordinates of the generated video's first latent frames and its audio
latents those of the generated audio's first latents, so the clip
conditions the start of the generation in place. Its rows are anchored and
held at the visual-condition and audio-reference timesteps like every
reference's, and VSA-H3 keeps them dense.

Every row's rotary coordinates, tag and timestep group, in the dense and in
the multi-segment layout, must equal FastVideo's bit for bit, and every
tile must hold the same rows. The vectors, ``minimax_h3_continuation.json``,
are FastVideo 36dc971dc's ``build_ref2va_packed_sequence``,
``build_row_timesteps`` and VSA-H3 tile geometry of these cases; the case
with no reference besides its clip packs through the same loop over no
references, past FastVideo's check that requires one.
"""

import json
import math
from pathlib import Path

import pytest
import torch

from tests.python.fixtures.h3 import (
    WIDE,
    base_denoiser,
    dmd_denoiser,
    omni_denoiser,
)
from uniserve.media import image, video
from uniserve.model import Condition, ConditionRole
from uniserve.nn.attention import vsa
from uniserve_models.minimax_h3.denoiser import Denoiser
from uniserve_models.minimax_h3.packing import (
    AUDIO_CONDITION_GROUP,
    AUDIO_GROUP,
    CONTINUATION_FRAMES,
    VIDEO_GROUP,
    VISUAL_CONDITION_GROUP,
    SegmentKind,
    condition_segments,
    continuation_latent_frames,
    dense_packing,
    dense_tables,
    segment_packing,
    segment_tables,
    segment_tiles,
    video_latent_frames,
)

pytestmark = pytest.mark.unit

FIXTURE = json.loads(
    (
        Path(__file__).parents[2] / "fixtures" / "minimax_h3_continuation.json"
    ).read_text()
)
CASES = {case["name"]: case for case in FIXTURE["cases"]}
GROUP_TIMESTEPS = {
    VIDEO_GROUP: FIXTURE["group_timesteps"]["video"],
    AUDIO_GROUP: FIXTURE["group_timesteps"]["audio"],
    VISUAL_CONDITION_GROUP: FIXTURE["group_timesteps"]["visual_condition"],
    AUDIO_CONDITION_GROUP: FIXTURE["group_timesteps"]["audio_condition"],
}
ROLES = {
    "reference": ConditionRole.REFERENCE,
    "continuation": ConditionRole.CONTINUATION,
}


def _conditions(case) -> tuple[Condition, ...]:
    return tuple(
        Condition(
            ROLES[condition["role"]],
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


def _canvas(case) -> image.Config:
    return image.Config(case["height"], case["width"])


def _spans(case) -> tuple[tuple[int, int], ...]:
    return tuple(tuple(span) for span in case["vision_spans"])


def _timesteps(groups: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [GROUP_TIMESTEPS[int(group)] for group in groups], dtype=torch.float32
    )


def _dense(case):
    canvas = _canvas(case)
    segments = condition_segments(_conditions(case), canvas)
    condition_rows = sum(s.video_rows + s.audio_rows for s in segments)
    packing = dense_packing(
        num_frames=case["frames"],
        canvas=canvas,
        text_rows=case["text_tokens"] + 64,
        condition_rows=condition_rows + 64,
        token_multiple=512,
    )
    tables = dense_tables(
        packing,
        num_frames=case["frames"],
        canvas=canvas,
        num_text_tokens=case["text_tokens"],
        segments=segments,
        vision_spans=_spans(case),
    )
    return packing, tables, condition_rows


@pytest.mark.parametrize("name", sorted(CASES))
def test_dense_rows_follow_the_reference_packing(name):
    case = CASES[name]
    packing, tables, condition_rows = _dense(case)
    assert condition_rows == (
        case["condition_video_rows"] + case["condition_audio_rows"]
    )
    # The reference's [text | conditions | target audio | target video].
    prefix = packing.prefix_start
    order = torch.cat(
        (
            torch.arange(prefix, prefix + case["text_tokens"] + condition_rows),
            torch.arange(packing.video_rows, prefix),
            torch.arange(0, packing.video_rows),
        )
    )
    expected = torch.tensor(case["position_ids"], dtype=torch.float64)
    assert torch.equal(tables.position_ids.index_select(0, order), expected)
    assert torch.equal(
        tables.token_tags.index_select(0, order),
        torch.tensor(case["token_tags"]),
    )
    assert torch.equal(
        _timesteps(tables.groups.index_select(0, order)),
        torch.tensor(case["row_timesteps"], dtype=torch.float32),
    )
    assert tables.used == prefix + case["text_tokens"] + condition_rows


def _segments(case):
    tile = FIXTURE["tile"]
    canvas = _canvas(case)
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
        vision_spans=_spans(case),
        sparsity=0.9,
        reference_keep=0.1,
    )
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
    assert bool((order >= 0).all())
    return packing, tables, order


@pytest.mark.parametrize("name", sorted(CASES))
def test_segment_rows_and_tiles_follow_the_reference_packing(name):
    """FastH3 Omni holds the same rows; the clip's tiles stay dense."""
    case = CASES[name]
    packing, tables, order = _segments(case)
    expected = torch.tensor(case["position_ids"], dtype=torch.float64)
    assert torch.equal(tables.position_ids.index_select(0, order), expected)
    assert torch.equal(
        tables.token_tags.index_select(0, order),
        torch.tensor(case["token_tags"]),
    )
    assert torch.equal(
        _timesteps(tables.groups.index_select(0, order)),
        torch.tensor(case["row_timesteps"], dtype=torch.float32),
    )

    # Each reference tile maps to one of ours, every row at the same offset.
    tile = FIXTURE["tile"]
    slots = torch.tensor(case["row_slots"])
    pairs: dict[int, int] = {}
    for reference, ours in zip(
        (slots // tile).tolist(), (order // tile).tolist(), strict=True
    ):
        assert pairs.setdefault(reference, ours) == ours
    assert torch.equal(slots % tile, order % tile)
    assert len(set(pairs.values())) == len(pairs)
    for reference, ours in pairs.items():
        assert (
            int(tables.variable_block_sizes[ours])
            == case["valid_sizes"][reference]
        )
        if reference < case["prefix_tiles"]:
            assert int(tables.tile_segments[ours]) == vsa.DENSE_TILE
        else:
            segment = next(
                index
                for index, (start, stop) in enumerate(case["segment_spans"])
                if start <= reference < stop
            )
            assert int(tables.tile_segments[ours]) == segment


def _clip_case(frames: int, samples: int, *, canvas=WIDE, num_frames=158):
    clip = Condition(
        ConditionRole.CONTINUATION, video.Config(frames, canvas), samples
    )
    still = Condition(
        ConditionRole.REFERENCE, video.Config(1, image.Config(1024, 1024))
    )
    # Listed first, the clip still packs after the references.
    return (clip, still), num_frames


@pytest.mark.parametrize("frames", CONTINUATION_FRAMES)
@pytest.mark.parametrize("samples", [0, 29_334])
def test_a_clip_aliases_the_start_of_the_generation(frames, samples):
    """Dense and tiled: the clip's rows sit on the generated rows' start."""
    conditions, num_frames = _clip_case(frames, samples)
    segments = condition_segments(conditions, WIDE)
    assert [segment.kind for segment in segments] == [
        SegmentKind.IMAGE,
        SegmentKind.CONTINUATION,
    ]
    clip = segments[-1]
    assert clip.condition == 0
    assert clip.latent_frames == continuation_latent_frames(frames)
    assert clip.audio_frames == math.ceil(samples / 800)

    text = 80
    rows = sum(s.video_rows + s.audio_rows for s in segments)
    packing = dense_packing(
        num_frames=num_frames,
        canvas=WIDE,
        text_rows=128,
        condition_rows=rows,
        token_multiple=128,
    )
    tables = dense_tables(
        packing,
        num_frames=num_frames,
        canvas=WIDE,
        num_text_tokens=text,
        segments=segments,
    )
    clip_start = packing.prefix_start + text + segments[0].video_rows
    clip_audio = slice(clip_start, clip_start + clip.audio_rows)
    clip_video = slice(clip_audio.stop, clip_audio.stop + clip.video_rows)
    positions = tables.position_ids
    # Raster order: the clip's frames are the generated video's first ones.
    assert torch.equal(positions[clip_video], positions[0 : clip.video_rows])
    # Each stereo channel's latents are the generated audio's first ones.
    generated = packing.audio_rows // 2
    for channel in range(2):
        ours = positions[
            clip_audio.start + channel * clip.audio_frames : clip_audio.start
            + (channel + 1) * clip.audio_frames
        ]
        theirs = positions[
            packing.video_rows + channel * generated : packing.video_rows
            + channel * generated
            + clip.audio_frames
        ]
        assert torch.equal(ours, theirs)
    # The generated timeline starts past the text, the image's slot and the
    # clip's span.
    span = max(
        float(clip.audio_frames),
        sum(
            (5 / 3) * (1, 4, 4, 4, 4)[k % 5] for k in range(clip.latent_frames)
        ),
    )
    assert float(positions[0, 0]) == text + 1.0 + span
    assert bool((tables.groups[clip_video] == VISUAL_CONDITION_GROUP).all())
    assert bool((tables.groups[clip_audio] == AUDIO_CONDITION_GROUP).all())

    for tile in (64, 128):
        tiled = segment_packing(
            num_frames=num_frames,
            canvas=WIDE,
            text_rows=128,
            condition_rows=sum(segment_tiles(s, tile) for s in segments) * tile,
            tile=tile,
            token_multiple=tile,
        )
        tables = segment_tables(
            tiled,
            num_frames=num_frames,
            canvas=WIDE,
            num_text_tokens=text,
            segments=segments,
            vision_spans=(),
            sparsity=0.9,
            reference_keep=None,
        )
        # The clip's visual rows gather the prefix source after the image's
        # and its audio rows.
        start = tiled.text_rows + segments[0].video_rows + clip.audio_rows
        held = (tables.prefix_index >= start) & (
            tables.prefix_index < start + clip.video_rows
        )
        rows = torch.nonzero(held).flatten()
        rows = rows[tables.prefix_index[rows].argsort()]
        generated_rows = tiled.video_indices[
            tiled.video_raster_indices.argsort()
        ][: clip.video_rows]
        assert torch.equal(
            tables.position_ids[rows], tables.position_ids[generated_rows]
        )
        assert bool(
            (
                tables.tile_segments[torch.unique(rows // tile)]
                == vsa.DENSE_TILE
            ).all()
        )


def test_clip_frame_counts_end_on_vae_clip_boundaries():
    assert CONTINUATION_FRAMES == (5, 22, 39)
    assert [continuation_latent_frames(n) for n in CONTINUATION_FRAMES] == [
        2,
        7,
        12,
    ]
    # The longer counts encode as videos of the same length do.
    assert continuation_latent_frames(22) == video_latent_frames(22)
    assert continuation_latent_frames(39) == video_latent_frames(39)
    for frames in (4, 6, 21, 23, 56):
        with pytest.raises(ValueError, match="5, 22, 39 frames"):
            continuation_latent_frames(frames)


def test_clips_are_refused_off_the_contract():
    still = Condition(ConditionRole.REFERENCE, video.Config(1, WIDE))
    clip = Condition(ConditionRole.CONTINUATION, video.Config(22, WIDE))
    first = Condition(ConditionRole.FIRST_FRAME, video.Config(1, WIDE))
    with pytest.raises(ValueError, match="5, 22, 39 frames"):
        condition_segments(
            (Condition(ConditionRole.CONTINUATION, video.Config(23, WIDE)),),
            WIDE,
        )
    with pytest.raises(ValueError, match="target canvas"):
        condition_segments(
            (
                Condition(
                    ConditionRole.CONTINUATION,
                    video.Config(22, image.Config(480, 832)),
                ),
            ),
            WIDE,
        )
    with pytest.raises(ValueError, match="at most one clip"):
        condition_segments((clip, still, clip), WIDE)
    with pytest.raises(ValueError, match="no keyframes"):
        condition_segments((first, clip), WIDE)
    with pytest.raises(ValueError, match="two or more frames"):
        Condition(ConditionRole.CONTINUATION, video.Config(1, WIDE))
    with pytest.raises(ValueError, match="two or more frames"):
        Condition(ConditionRole.CONTINUATION, None, 32_000)
    # A continuation is no keyframe: it may carry a soundtrack.
    assert not ConditionRole.CONTINUATION.is_keyframe
    assert (
        Condition(
            ConditionRole.CONTINUATION, video.Config(5, WIDE), 6_667
        ).audio_samples
        == 6_667
    )


def _meta(config) -> Denoiser:
    with torch.device("meta"):
        return Denoiser(config)


@pytest.mark.parametrize(
    ("canvas", "rows_per_frame"),
    [(WIDE, 1008), (image.Config(480, 832), 390)],
)
def test_denoisers_count_a_clip_toward_their_condition_rows(
    canvas, rows_per_frame
):
    """A clip's rows: its latent frames' patches and its stereo latents."""
    # 22 frames with 22 / 24 s of 32 kHz sound: 7 latent frames and 37
    # audio latents a channel; 39 frames: 12 and 65; 5 frames: 2 and 9.
    for frames, samples, latents, audio in (
        (22, 29_334, 7, 37),
        (39, 52_000, 12, 65),
        (5, 6_667, 2, 9),
    ):
        clip = Condition(
            ConditionRole.CONTINUATION, video.Config(frames, canvas), samples
        )
        dense = _meta(base_denoiser(tasks=("ref2va",)))
        size = dense.make_size(158, 64, canvas=canvas, conditions=(clip,))
        assert size.condition_rows == latents * rows_per_frame + 2 * audio
        # The clip draws one noise latent of its own geometry.
        assert dense.condition_noise_shapes(size) == (
            (1, 24, latents, canvas.height // 16, canvas.width // 16),
        )
        # FastH3 Omni rounds the clip's audio and frames to whole tiles.
        omni = _meta(omni_denoiser())
        tile = omni._tile
        size = omni.make_size(158, 64, canvas=canvas, conditions=(clip,))
        assert size.condition_rows == tile * (
            math.ceil(2 * audio / tile)
            + math.ceil(latents * rows_per_frame / tile)
        )

    clip = Condition(ConditionRole.CONTINUATION, video.Config(22, WIDE))
    # Only a reference-to-video network takes a clip.
    for config in (base_denoiser(), dmd_denoiser()):
        with pytest.raises(ValueError, match="no references"):
            _meta(config).make_size(158, 64, canvas=WIDE, conditions=(clip,))


def test_a_clip_is_anchored_like_a_reference():
    """A clip's rows follow the references' and are anchored as theirs are.

    Its frames mix in their draw at the visual-condition time and its
    soundtrack is projected unchanged.
    """
    from tests.python.unit.models.test_h3_conditions import _small
    from uniserve_models.minimax_h3.packing import patchify_video

    model = _small()
    clip = Condition(ConditionRole.CONTINUATION, video.Config(22, WIDE), 29_334)
    still = Condition(
        ConditionRole.REFERENCE, video.Config(1, image.Config(1024, 1024))
    )
    size = model.make_size(158, 50, canvas=WIDE, conditions=(clip, still))
    layout = model.layout_size(size)
    generator = torch.Generator().manual_seed(7)
    clip_rows = torch.randn(7 * 1008, 96, generator=generator)
    clip_audio = torch.randn(74, 32, generator=generator)
    still_rows = torch.randn(1024, 96, generator=generator)
    shapes = model.condition_noise_shapes(size)
    # The image draws first, then the clip, in packed order.
    assert shapes == ((1, 24, 1, 64, 64), (1, 24, 7, 48, 84))
    noise = tuple(torch.randn(shape, generator=generator) for shape in shapes)
    out = torch.zeros(
        (model.text_condition_rows(layout), model.text_condition_width),
        dtype=torch.bfloat16,
    )
    with torch.inference_mode():
        model.encode_conditions(
            size,
            layout,
            latents=(clip_rows, clip_audio, still_rows),
            noise=noise,
            out=out,
        )
        level = torch.tensor(0.999)

        def visual(rows, draw):
            anchored = level * rows + (1.0 - level) * patchify_video(draw)[0]
            return model.transformer.video_input(
                anchored, output_dtype=torch.float32
            ).bfloat16()

        expected = torch.cat(
            (
                visual(still_rows, noise[0]),
                model.transformer.audio_input(
                    clip_audio, output_dtype=torch.float32
                ).bfloat16(),
                visual(clip_rows, noise[1]),
            )
        )
    start = layout.num_text_tokens
    assert torch.equal(out[start : start + size.condition_rows], expected)
