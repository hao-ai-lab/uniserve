"""Ordered semantic reference rows and fixed-conditioning clocks."""

import pytest
import torch

from uniserve_worker.models.minimax_h3.reference import (
    H3ReferenceGeometry,
    build_reference_layout,
    validate_reference_geometry,
)

pytestmark = pytest.mark.unit


def test_ordered_image_soundtrack_video_audio_and_target_layout():
    references = (
        H3ReferenceGeometry("image", 1, 4, 4),
        H3ReferenceGeometry("video", 2, 4, 8, 5),
        H3ReferenceGeometry("audio", audio_frames=2),
    )
    layout = build_reference_layout(
        text_token_tags=torch.tensor([1, 0, 1]),
        references=references,
        video_frames=2,
        latent_height=4,
        latent_width=4,
        audio_frames=3,
    )
    # Three presentation rows, image, video's stereo soundtrack, video,
    # standalone stereo audio, target stereo audio, target video.
    assert layout.sequence_length == 51
    assert layout.condition_video_rows == 20
    assert layout.condition_audio_rows == 14
    torch.testing.assert_close(
        layout.video_indices,
        torch.tensor(list(range(3, 7)) + list(range(17, 33)) + list(range(43, 51))),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        layout.audio_indices,
        torch.tensor(list(range(7, 17)) + list(range(33, 43))),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(layout.target_video_indices, torch.arange(43, 51))
    torch.testing.assert_close(layout.target_audio_indices, torch.arange(37, 43))
    assert layout.video_regions == ((17, 2, 2, 4), (43, 2, 2, 2))
    assert (
        layout.token_tags.tolist() == [1, 0, 1] + [0] * 4 + [2] * 10 + [0] * 16 + [2] * 10 + [0] * 8
    )
    times = layout.position_ids[:, 0]
    assert times[:7].tolist() == [0, 1, 2, 3, 3, 3, 3]
    assert times[7:17].tolist() == [4, 5, 6, 7, 8] * 2
    assert times[17:33].tolist() == [4] * 8 + [4 + 5 / 3] * 8
    # A video's two latent spans total 5/3 + 20/3; unlike an image it
    # advances the shared clock by duration, not by its spatial row count.
    audio_origin = 4 + sum((5 / 3, (5 / 3) * 4))
    target_origin = audio_origin + 2
    assert times[33:37].tolist() == [audio_origin, audio_origin + 1] * 2
    assert times[37:43].tolist() == [target_origin + i for i in range(3)] * 2
    assert times[43:51].tolist() == [target_origin] * 4 + [target_origin + 5 / 3] * 4
    assert layout.position_ids.dtype == torch.float64

    # Qwen vision presentation tokens still use the target clock; only the
    # external VAE reference rows stay fixed across solver evaluations.
    for video_time, audio_time in ((0.0, 0.0), (0.5, 0.25), (1.0, 1.0)):
        clean = layout.row_clean_times(video_time, audio_time)
        torch.testing.assert_close(clean[:3], torch.full((3,), video_time))
        torch.testing.assert_close(
            clean[layout.video_indices[:20]], torch.full((20,), max(video_time, 0.999))
        )
        torch.testing.assert_close(clean[layout.audio_indices[:14]], torch.ones(14))
        torch.testing.assert_close(clean[layout.target_video_indices], torch.full((8,), video_time))
        torch.testing.assert_close(clean[layout.target_audio_indices], torch.full((6,), audio_time))


@pytest.mark.parametrize(
    "references",
    [
        [],
        [H3ReferenceGeometry("audio", audio_frames=2)],
        [H3ReferenceGeometry("image", 1, 2, 2)] * 10,
        [H3ReferenceGeometry("video", 1, 2, 2)] * 4,
        [H3ReferenceGeometry("image", 1, 2, 2)]
        + [H3ReferenceGeometry("audio", audio_frames=2)] * 4,
        [H3ReferenceGeometry("image", 1, 2, 2)] * 9
        + [H3ReferenceGeometry("video", 1, 2, 2)] * 3
        + [H3ReferenceGeometry("audio", audio_frames=2)],
    ],
)
def test_reference_bundle_limits(references):
    with pytest.raises(ValueError):
        validate_reference_geometry(references)


@pytest.mark.parametrize(
    "args",
    [
        ("image", 2, 4, 4, 0),
        ("image", 1, 4, 4, 1),
        ("video", 1, 3, 4, 0),
        ("video", 0, 4, 4, 0),
        ("audio", 0, 0, 0, 0),
        ("audio", 1, 4, 4, 2),
        ("video", True, 4, 4, 0),
    ],
)
def test_invalid_reference_geometry(args):
    with pytest.raises(ValueError):
        H3ReferenceGeometry(*args)


def test_presentation_padding_and_invalid_target_rejected():
    kwargs = dict(
        references=[H3ReferenceGeometry("image", 1, 2, 2)],
        video_frames=2,
        latent_height=4,
        latent_width=4,
        audio_frames=3,
    )
    with pytest.raises(ValueError, match="presentation"):
        build_reference_layout(text_token_tags=torch.tensor([1, -1]), **kwargs)
    kwargs["audio_frames"] = 0
    with pytest.raises(ValueError, match="target audio"):
        build_reference_layout(text_token_tags=torch.tensor([1]), **kwargs)
