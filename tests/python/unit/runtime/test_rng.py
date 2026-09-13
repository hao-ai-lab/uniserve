from __future__ import annotations

import pytest
import torch

from uniserve_worker.nn.diffusion.schedule import ScheduleDirection, ScheduleShiftDomain
from uniserve_worker.nn.diffusion.spec import DiffusionSpec, ModalitySpec, ScheduleRule
from uniserve_worker.nn.rng import (
    DRAW_LAYOUT_PROPOSAL,
    DRAW_LAYOUT_TARGET,
    diffusion_noise,
    sampling_key,
    sampling_uniform,
)

pytestmark = pytest.mark.unit


def test_multimodal_noise_uses_one_cpu_generator_in_declared_order():
    schedule = ScheduleRule(ScheduleDirection.DESCENDING, ScheduleShiftDomain.SIGMA, 1.0)
    spec = DiffusionSpec(
        modalities=tuple(
            ModalitySpec(name, shape, shape, schedule, "velocity", torch.float32)
            for name, shape in (("video", (1, 3, 7, 2, 4)), ("audio", (16, 2)))
        ),
        steps=4,
        cfg=None,
        max_cfg_branches=1,
        solver="clean_sample_euler",
        noise_device="cpu",
        seed_transform="identity",
    )
    outputs = {
        modality.name: torch.empty((2, *modality.noise_shape)) for modality in spec.modalities
    }
    diffusion_noise(
        spec, seeds=(17, 29), device=torch.device("cuda", 0), dtype=torch.float32, out=outputs
    )
    for row, seed in enumerate((17, 29)):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        video = torch.empty((1, 3, 7, 2, 4)).normal_(generator=generator)
        audio = torch.empty((16, 2)).normal_(generator=generator)
        torch.testing.assert_close(outputs["video"][row], video, rtol=0, atol=0)
        torch.testing.assert_close(outputs["audio"][row], audio, rtol=0, atol=0)


def test_sampling_draw_is_stable_and_coordinate_scoped() -> None:
    key = sampling_key(
        41, engine_id=3, request_id=7, request_epoch=2, draw_layout=DRAW_LAYOUT_TARGET
    )
    draw = sampling_uniform(key, 19, processor_stage=1, draw_index=5)

    assert draw == sampling_uniform(key, 19, processor_stage=1, draw_index=5)
    assert 0.0 <= draw < 1.0
    assert (
        len(
            {
                draw,
                sampling_uniform(
                    sampling_key(
                        41,
                        engine_id=3,
                        request_id=8,
                        request_epoch=2,
                        draw_layout=DRAW_LAYOUT_TARGET,
                    ),
                    19,
                    processor_stage=1,
                    draw_index=5,
                ),
                sampling_uniform(key, 20, processor_stage=1, draw_index=5),
                sampling_uniform(key, 19, processor_stage=2, draw_index=5),
                sampling_uniform(key, 19, processor_stage=1, draw_index=6),
            }
        )
        == 5
    )


def test_draw_layout_separates_proposal_and_target_spaces() -> None:
    target = sampling_key(
        41, engine_id=3, request_id=7, request_epoch=2, draw_layout=DRAW_LAYOUT_TARGET
    )
    proposal = sampling_key(
        41, engine_id=3, request_id=7, request_epoch=2, draw_layout=DRAW_LAYOUT_PROPOSAL
    )

    assert target != proposal
    assert sampling_uniform(target, 19) != sampling_uniform(proposal, 19)
