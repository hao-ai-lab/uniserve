"""H3 denoisers size, draw and encode their requests' conditions.

A deployment's denoiser takes the conditions its task family serves, packs
keyframes before references, draws one native noise latent per visual
condition ahead of the generated rows, and writes each condition's projected
rows into the request's retained conditioning past the prompt capacity.
"""

import pytest
import torch

from tests.python.fixtures.h3 import (
    WIDE,
    base_denoiser,
    dmd_denoiser,
    omniref_denoiser,
)
from uniserve.media import image, video
from uniserve.model import Condition, ConditionRole
from uniserve_models.minimax_h3 import TransformerConfig
from uniserve_models.minimax_h3.denoiser import Denoiser
from uniserve_models.minimax_h3.inputs import DenoiserSize
from uniserve_models.minimax_h3.packing import patchify_video

pytestmark = pytest.mark.unit

FIRST = Condition(ConditionRole.FIRST_FRAME, video.Config(1, WIDE))
IMAGE = Condition(
    ConditionRole.REFERENCE, video.Config(1, image.Config(2048, 1152))
)
# A 22-frame video with 1.5 s of 32 kHz audio: 7 latent frames, 60 latents.
CLIP = Condition(
    ConditionRole.REFERENCE, video.Config(22, image.Config(768, 768)), 48_000
)


def _meta(config) -> Denoiser:
    with torch.device("meta"):
        return Denoiser(config)


def test_each_denoiser_takes_the_conditions_of_its_tasks():
    keyframes = _meta(base_denoiser())
    size = keyframes.make_size(124, 64, canvas=WIDE, conditions=(FIRST,))
    # One latent frame of the 16:9 canvas: 24 x 42 patches.
    assert (size.condition_rows, size.conditions) == (1008, (FIRST,))
    # A layout holds capacities alone.
    layout = keyframes.layout_size(size)
    assert (layout.condition_rows, layout.conditions) == (1024, ())
    assert keyframes.holds(layout, size)
    with pytest.raises(ValueError, match="no references"):
        keyframes.make_size(124, 64, canvas=WIDE, conditions=(IMAGE,))

    references = _meta(base_denoiser(tasks=("ref2va",)))
    size = references.make_size(
        124, 64, canvas=WIDE, conditions=(CLIP, IMAGE, FIRST)
    )
    # The clip's 7 x 24 x 24 patches and 2 x 60 audio rows, the image's
    # 64 x 36 patches and the keyframe's 1008.
    assert size.condition_rows == 4032 + 120 + 2304 + 1008

    with pytest.raises(ValueError, match="sparse"):
        _meta(dmd_denoiser()).make_size(
            124, 64, canvas=WIDE, conditions=(FIRST,)
        )


def test_visual_condition_draws_lead_in_packed_order():
    model = _meta(base_denoiser(tasks=("ref2va",)))
    size = model.make_size(
        124, 64, canvas=WIDE, conditions=(CLIP, IMAGE, FIRST)
    )
    # The keyframe packs first; the audio track draws nothing.
    assert model.condition_noise_shapes(size) == (
        (1, 24, 1, 48, 84),
        (1, 24, 7, 48, 48),
        (1, 24, 1, 128, 72),
    )
    layout = model.layout_size(size)
    elements = sum(
        torch.Size(shape).numel()
        for shape in model.condition_noise_shapes(size)
    )
    assert elements <= model.condition_noise_capacity(layout)


def _small() -> Denoiser:
    transformer = TransformerConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_refiner_layers=1,
        intermediate_size=128,
        text_dim=40,
        frequency_dim=16,
        time_hidden_dim=64,
        time_dim=32,
        rope_frequency_dim=4,
    )
    model = Denoiser(base_denoiser(transformer, tasks=("ref2va",)))
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for projection in (
            model.transformer.video_input,
            model.transformer.audio_input,
        ):
            for parameter in projection.parameters():
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator) * 0.1
                )
    return model


def test_encoded_conditions_fill_the_rows_past_the_prompt():
    model = _small()
    size = model.make_size(124, 50, canvas=WIDE, conditions=(CLIP, FIRST))
    layout = model.layout_size(size)
    generator = torch.Generator().manual_seed(5)
    clip_rows = torch.randn(4032, 96, generator=generator)
    clip_audio = torch.randn(120, 32, generator=generator)
    keyframe_rows = torch.randn(1008, 96, generator=generator)
    noise = tuple(
        torch.randn(shape, generator=generator)
        for shape in model.condition_noise_shapes(size)
    )
    out = torch.full(
        (model.text_condition_rows(layout), model.text_condition_width),
        7.0,
        dtype=torch.bfloat16,
    )
    with torch.inference_mode():
        model.encode_conditions(
            size,
            layout,
            latents=(clip_rows, clip_audio, keyframe_rows),
            noise=noise,
            out=out,
        )

        level = torch.tensor(0.999)

        def visual(rows, draw):
            anchored = level * rows + (1.0 - level) * patchify_video(draw)[0]
            return model.transformer.video_input(
                anchored, output_dtype=torch.float32
            ).bfloat16()

        # Keyframe first, then the clip's soundtrack ahead of its frames.
        expected = torch.cat(
            (
                visual(keyframe_rows, noise[0]),
                model.transformer.audio_input(
                    clip_audio, output_dtype=torch.float32
                ).bfloat16(),
                visual(clip_rows, noise[1]),
            )
        )
    start = layout.num_text_tokens
    assert torch.equal(out[start : start + size.condition_rows], expected)
    # The prompt rows and the rest of the source are left to their writers.
    assert torch.all(out[:start] == 7.0)
    assert torch.all(out[start + size.condition_rows :] == 7.0)

    with pytest.raises(ValueError, match="latents"):
        model.encode_conditions(
            size, layout, latents=(clip_rows,), noise=noise, out=out
        )


def test_reference_regions_take_whole_tiles():
    model = _meta(omniref_denoiser())
    voice = Condition(ConditionRole.REFERENCE, None, 32_000)
    size = model.make_size(
        124, 300, canvas=WIDE, conditions=(CLIP, IMAGE, voice)
    )
    # The clip's 120 soundtrack rows take one tile and its 7 x 24 x 24
    # patches 2 x 6 x 3 tiles of 4 x 4 x 8; the image's 2304 rows take 18
    # tiles and the voice's 80 rows one.
    assert size.condition_rows == 128 * (1 + 36 + 18 + 1)
    layout = model.layout_size(size)
    assert (layout.num_text_tokens, layout.condition_rows) == (384, 7168)
    assert model.holds(layout, size)
    # The retained conditioning holds the prompt capacity, the conditions'
    # rows and one zero row.
    assert model.text_condition_rows(layout) == 384 + 7168 + 1
    # Region packing holds no keyframe rows.
    with pytest.raises(ValueError, match="keyframe"):
        model.make_size(124, 300, canvas=WIDE, conditions=(CLIP, FIRST))


def test_reference_region_state_follows_its_layout():
    model = Denoiser(
        omniref_denoiser(
            TransformerConfig(
                hidden_size=64,
                num_attention_heads=4,
                num_hidden_layers=1,
                num_refiner_layers=1,
                intermediate_size=128,
                text_dim=40,
                frequency_dim=16,
                time_hidden_dim=64,
                time_dim=32,
                rope_frequency_dim=4,
            )
        )
    )
    size = model.make_size(124, 300, canvas=WIDE, conditions=(CLIP,))
    # A layout with spare text and condition tiles holds the request.
    layout = model.layout_size(
        DenoiserSize(124, WIDE, 512, size.condition_rows + 256)
    )
    assert model.holds(layout, size)
    buffers = model.state_buffers(layout)
    out = {
        name: torch.empty(config.shape, dtype=config.dtype)
        for name, config in buffers.items()
        if name not in model.modalities
    }
    model.prepare_state((size,), layouts=(layout,), out=out)

    # The prompt, the clip's soundtrack and patches, and the generated 2 x
    # 207 audio rows and 37 x 24 x 42 video patches fill the tiles.
    prefix = 300 + 120 + 4032
    assert int(out["valid_sizes"].sum()) == prefix + 2 * 207 + 37 * 24 * 42
    # The clip's video forms region 0 and the generated video region 1.
    assert set(out["tile_regions"].tolist()) == {-1, 0, 1}
    # Only the request's prefix rows gather the prefix source; every other
    # row takes its zero row.
    zero = model.text_condition_rows(layout) - 1
    assert int((out["prefix_index"] != zero).sum()) == prefix


def test_a_condition_layout_holds_its_condition_rows():
    model = _meta(base_denoiser(tasks=("ref2va",)))
    layout = model.layout_size(model.make_size(124, 64, canvas=WIDE))
    # The clip's 4032 + 120 rows and the image's 2304 take 101 whole 64-row
    # tiles; the rest of the layout is kept.
    size = model.make_size(124, 64, canvas=WIDE, conditions=(CLIP, IMAGE))
    widened = model.condition_layout(layout, size.condition_rows)
    assert widened.condition_rows == 101 * 64
    assert (
        widened.num_frames,
        widened.canvas,
        widened.num_text_tokens,
    ) == (layout.num_frames, layout.canvas, layout.num_text_tokens)
    # The widened layout holds the request; the text-only one does not.
    assert model.holds(widened, size)
    assert not model.holds(layout, size)
    # A layout whose region already holds the rows is kept as it is.
    assert model.condition_layout(widened, 64) == widened


def test_a_condition_layout_respects_the_attention_kind():
    with pytest.raises(ValueError, match="single-region sparse"):
        model = _meta(dmd_denoiser())
        model.condition_layout(
            model.layout_size(model.make_size(124, 64, canvas=WIDE)), 64
        )

    # Region packing takes conditions in its own 128-row tiles.
    model = _meta(omniref_denoiser())
    layout = model.layout_size(model.make_size(124, 300, canvas=WIDE))
    widened = model.condition_layout(layout, 7000)
    assert widened.condition_rows == 55 * 128
    size = model.make_size(124, 300, canvas=WIDE, conditions=(CLIP, IMAGE))
    widened = model.condition_layout(layout, size.condition_rows)
    assert model.holds(widened, size)
