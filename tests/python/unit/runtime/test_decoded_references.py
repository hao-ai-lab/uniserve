"""Worker admission accepts one bounded image reference and preserves plain requests."""

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
            "reference",
            "reference",
            False,
            product(0, (1, 32, 64, 3), DType.U8),
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
    params = DiffusionRequestParams((1, 2), 17, MediaGeometry(22, 3, 2, 4), references()[:1])
    admission = NewRequest.create(OWNER, request_pool_idx=1, diffusion=params)
    restored = NewRequest.from_mapping(admission.to_mapping())
    assert restored == admission
    assert len(restored.diffusion.references) == 1
    assert (
        restored.diffusion.references[0].pixels.shape_bound
        == product(0, (1, 32, 64, 3), DType.U8).shape_bound
    )


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


@pytest.mark.parametrize(
    ("bundle", "rule"),
    [
        (references()[:1] * 2, "at most 1"),
        (references()[1:2], "requires image"),
        (references()[2:], "requires image"),
        ((replace(references()[0], task="first_frame", role="first_frame"),), "task=reference"),
        (
            (replace(references()[0], pixels=product(0, (1, 32, 33, 3), DType.U8)),),
            "multiples of 32",
        ),
    ],
)
def test_unsupported_reference_bundles_name_the_admission_rule(bundle, rule):
    with pytest.raises(WorkerError, match=rule):
        DiffusionRequestParams((1,), 0, MediaGeometry(1, 1, 1, 1), bundle)


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
    with pytest.raises(WorkerError, match="requires image"):
        DiffusionRequestParams((1,), 0, MediaGeometry(1, 1, 1, 1), (audio,))
