from __future__ import annotations

import pytest

from uniserve_worker.batch import RequestKey
from uniserve_worker.runtime.rng import semantic_sampling_seed

pytestmark = pytest.mark.unit


def test_semantic_sampling_coordinates_are_stable_and_identity_scoped() -> None:
    request = RequestKey(authority_id=3, session_id=7, epoch=2)
    coordinate = semantic_sampling_seed(request, 41, 19, processor_stage=1, draw_index=5)

    assert coordinate == semantic_sampling_seed(
        request,
        41,
        19,
        processor_stage=1,
        draw_index=5,
    )
    assert len(
        {
            coordinate,
            semantic_sampling_seed(
                RequestKey(authority_id=3, session_id=8, epoch=2),
                41,
                19,
                processor_stage=1,
                draw_index=5,
            ),
            semantic_sampling_seed(request, 41, 20, processor_stage=1, draw_index=5),
            semantic_sampling_seed(request, 41, 19, processor_stage=2, draw_index=5),
            semantic_sampling_seed(request, 41, 19, processor_stage=1, draw_index=6),
        }
    ) == 5
