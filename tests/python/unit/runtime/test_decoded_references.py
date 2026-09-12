"""Decoded reference admission preserves media order and bounded tensor contracts."""

from dataclasses import replace

import pytest

from uniserve_worker.execution.batch import (
    DecodedReference,
    DiffusionRequestParams,
    DType,
    MediaGeometry,
    NewRequest,
    PointRange,
    ProductKind,
    ProductRef,
    RequestKey,
    ShapeBound,
    StaticDim,
    StorageClass,
)
from uniserve_worker.foundation.errors import WorkerError

OWNER = RequestKey(4, 7, 2)


def product(index, shape, dtype):
    return ProductRef(
        OWNER,
        0,
        index,
        1,
        ProductKind.TENSOR,
        StorageClass.DEVICE_TENSOR,
        dtype,
        ShapeBound(tuple(StaticDim(n) for n in shape)),
        PointRange(0, 1),
    )


def references():
    return (
        DecodedReference(
            "image",
            "first_last_frame",
            "last_frame",
            False,
            product(0, (1, 16, 24, 3), DType.U8),
            None,
            0,
            1,
        ),
        DecodedReference(
            "video",
            "continue_scene",
            "preceding",
            True,
            product(1, (24, 16, 24, 3), DType.U8),
            product(2, (2, 32000), DType.F32),
            24000,
            1001,
        ),
        DecodedReference(
            "audio", "reference", "reference", False, None, product(3, (2, 64000), DType.F32), 0, 1
        ),
    )


def test_ordered_reference_admission_round_trip():
    params = DiffusionRequestParams((1, 2), 17, MediaGeometry(22, 3, 2, 4), references())
    admission = NewRequest.create(OWNER, request_pool_idx=1, diffusion=params)
    restored = NewRequest.from_mapping(admission.to_mapping())
    assert restored == admission
    assert tuple(item.kind for item in restored.diffusion.references) == ("image", "video", "audio")
    assert restored.diffusion.references[1].fps_num == 24000
    assert restored.diffusion.references[1].fps_den == 1001


@pytest.mark.parametrize(
    "changes",
    [
        {"include_audio": True},
        {"fps_den": 0},
        {"role": "preceding"},
        {"pixels": product(0, (2, 16, 24, 3), DType.U8)},
        {"pixels": product(0, (1, 4097, 24, 3), DType.U8)},
        {"pixels": product(0, (1, 16, 24, 3), DType.F32)},
    ],
)
def test_invalid_decoded_geometry_and_policy(changes):
    with pytest.raises(WorkerError):
        replace(references()[0], **changes)


def test_foreign_request_product_rejected():
    image = references()[0]
    foreign = replace(image, pixels=replace(image.pixels, request_key=RequestKey(4, 7, 3)))
    params = DiffusionRequestParams((1,), 0, MediaGeometry(1, 1, 1, 1), (foreign,))
    with pytest.raises(WorkerError, match="belong"):
        NewRequest.create(OWNER, request_pool_idx=1, diffusion=params)


def test_empty_and_omitted_bundle_are_equal():
    params = DiffusionRequestParams((1,), 0, MediaGeometry(1, 1, 1, 1))
    mapping = params.to_mapping()
    del mapping["references"]
    assert DiffusionRequestParams.from_mapping(mapping) == params


def test_soundtrack_and_audio_bounds():
    video, audio = references()[1:]
    with pytest.raises(WorkerError, match="soundtrack"):
        replace(video, include_audio=False)
    with pytest.raises(WorkerError, match="30 seconds"):
        replace(audio, audio=product(3, (2, 960001), DType.F32))
    with pytest.raises(WorkerError, match="visual"):
        DiffusionRequestParams((1,), 0, MediaGeometry(1, 1, 1, 1), (audio,))
