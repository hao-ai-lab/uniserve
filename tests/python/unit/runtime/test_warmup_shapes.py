"""The shapes a worker asks the engine to warm before it serves."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from uniserve_worker.bootstrap.warmup import canvas_step_rows, image_warmups
from uniserve_worker.model_executor.graph_inputs import DiffusionShape
from uniserve_worker.protocol.call import MediaCall


def _runner(captures, *, max_tokens=4096, framing=2, downsample=16):
    builder = SimpleNamespace(
        max_tokens=max_tokens,
        framing=framing,
        denoiser=SimpleNamespace(downsample=downsample),
    )
    return SimpleNamespace(
        image_builder=builder,
        flow_cfg_branches=(1, 2, 3),
        flow_captures=tuple(captures),
    )


def _info(*, request_slots=8, latent_units=1024, calls=None):
    return SimpleNamespace(
        request_slots=request_slots,
        latent_capacity_units=latent_units,
        supported_calls=frozenset(
            (MediaCall.LATENT_PREPARATION, MediaCall.DENOISING)
            if calls is None
            else calls
        ),
    )


def test_each_captured_branch_count_warms_its_last_captured_shape() -> None:
    runner = _runner(
        [
            DiffusionShape(1, 512, 512, 1),
            DiffusionShape(1, 512, 512, 2),
            DiffusionShape(2, 768, 1024, 2),
            DiffusionShape(1, 512, 512, 3),
        ]
    )

    shapes = image_warmups(runner, _info())

    assert [(s.height, s.width) for s in shapes] == [
        (512, 512),
        (768, 1024),
        (512, 512),
    ]
    # One, two and three branches: no, text and text-and-image guidance.
    assert [(s.cfg_text_scale, s.cfg_img_scale) for s in shapes] == [
        (1.0, 1.0),
        (4.0, 1.0),
        (4.0, 2.0),
    ]


def test_an_uncaptured_branch_count_warms_a_square_of_the_latent_capacity() -> (
    None
):
    # The side is the integer square root of the latent capacity, 32 patches
    # of 16 pixels, within the builder's token bound.
    shapes = image_warmups(_runner([]), _info(latent_units=1024))

    assert [(s.height, s.width) for s in shapes] == [(512, 512)] * 3

    bounded = image_warmups(_runner([], max_tokens=100), _info())
    assert [(s.height, s.width) for s in bounded] == [(160, 160)] * 3


def test_guided_shapes_need_a_slot_for_the_alternative_prefix() -> None:
    shapes = image_warmups(_runner([]), _info(request_slots=1))

    # A single slot serves unguided generation only.
    assert [(s.cfg_text_scale, s.cfg_img_scale) for s in shapes] == [(1.0, 1.0)]


@pytest.mark.parametrize(
    "calls", [(), (MediaCall.DENOISING,), (MediaCall.LATENT_PREPARATION,)]
)
def test_a_worker_without_the_flow_calls_states_no_image_shape(calls) -> None:
    assert image_warmups(_runner([]), _info(calls=calls)) == ()


def test_a_worker_without_an_image_builder_states_no_image_shape() -> None:
    runner = _runner([])
    runner.image_builder = None

    assert image_warmups(runner, _info()) == ()


def test_canvas_rows_are_the_sampler_chunk_only_when_canvases_generate() -> (
    None
):
    runner = SimpleNamespace(canvas_runner=SimpleNamespace(step_rows=6))

    assert canvas_step_rows(runner, True) == 6
    assert canvas_step_rows(runner, False) == 0
    assert canvas_step_rows(SimpleNamespace(canvas_runner=None), True) == 0
