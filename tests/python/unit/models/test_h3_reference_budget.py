"""Packed storage admission includes alignment, video boundary tiles, and audio."""

import pytest

from uniserve_worker.models.minimax_h3.reference import H3ReferenceGeometry, reference_packed_rows


def test_exact_packed_budget_boundary():
    reference = H3ReferenceGeometry("image", 1, 2, 2)
    target = H3ReferenceGeometry("video", 7, 8, 8, 1)
    # Three 64-row dense blocks plus two 64-row target-video tiles, rounded
    # to the 256-row rank allocation: 512, despite only 179 semantic rows.
    arguments = dict(presentation_rows=64, target=target)
    assert reference_packed_rows((reference,), max_rows=512, **arguments) == 512
    with pytest.raises(ValueError, match="requires 512 rows; budget is 511"):
        reference_packed_rows((reference,), max_rows=511, **arguments)


def test_embedded_audio_consumes_reference_budget():
    target = H3ReferenceGeometry("video", 7, 8, 8, 1)
    silent = H3ReferenceGeometry("video", 1, 2, 2)
    audible = H3ReferenceGeometry("video", 1, 2, 2, 64)
    arguments = dict(presentation_rows=64, target=target, row_multiple=128, max_rows=384)
    assert reference_packed_rows((silent,), **arguments) == 384
    with pytest.raises(ValueError, match="requires 512"):
        reference_packed_rows((audible,), **arguments)


def test_all_ordered_images_consume_budget():
    image = H3ReferenceGeometry("image", 1, 128, 128)
    target = H3ReferenceGeometry("video", 7, 8, 8, 1)
    # Five 2048-square image canvases contain 20480 reference patch rows.
    assert (
        reference_packed_rows((image,) * 5, presentation_rows=64, target=target, max_rows=20736)
        == 20736
    )
    with pytest.raises(ValueError, match="requires 20736"):
        reference_packed_rows((image,) * 5, presentation_rows=64, target=target, max_rows=20735)
