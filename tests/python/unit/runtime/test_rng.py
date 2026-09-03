from __future__ import annotations

import pytest

from uniserve_worker.execution.rng import (
    DRAW_LAYOUT_PROPOSAL,
    DRAW_LAYOUT_TARGET,
    sampling_key,
    sampling_uniform,
)

pytestmark = pytest.mark.unit


def test_sampling_draw_is_stable_and_coordinate_scoped() -> None:
    key = sampling_key(41, authority_id=3, request_id=7, epoch=2, draw_layout=DRAW_LAYOUT_TARGET)
    draw = sampling_uniform(key, 19, processor_stage=1, draw_index=5)

    assert draw == sampling_uniform(key, 19, processor_stage=1, draw_index=5)
    assert 0.0 <= draw < 1.0
    assert (
        len(
            {
                draw,
                sampling_uniform(
                    sampling_key(
                        41, authority_id=3, request_id=8, epoch=2, draw_layout=DRAW_LAYOUT_TARGET
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
    target = sampling_key(41, authority_id=3, request_id=7, epoch=2, draw_layout=DRAW_LAYOUT_TARGET)
    proposal = sampling_key(
        41, authority_id=3, request_id=7, epoch=2, draw_layout=DRAW_LAYOUT_PROPOSAL
    )

    assert target != proposal
    assert sampling_uniform(target, 19) != sampling_uniform(proposal, 19)
