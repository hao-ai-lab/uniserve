"""The prompt capacity is the largest text capacity of a video deployment."""

from __future__ import annotations

import pytest
import torch

from tests.python.fixtures.h3 import fasth3_config
from uniserve_models.minimax_h3 import Model
from uniserve_worker.model_executor.media_inputs import MediaBuilder

pytestmark = pytest.mark.unit

# Prompt capacity of the builders below. Capacities are whole 64-row text
# tiles, so it is provisioned as the 5056 rows that hold it.
PROMPT_TOKENS = 5000
PROMPT_ROWS = 5056


@pytest.fixture(scope="module")
def denoiser():
    with torch.device("meta"):
        return Model(fasth3_config()).transformer


def _capacities(denoiser, listed: tuple[int, ...]) -> tuple[int, ...]:
    return MediaBuilder(
        denoiser,
        max_frames=124,
        max_text_tokens=PROMPT_TOKENS,
        text_capacities=listed,
    ).text_capacities


def test_the_listed_capacities_sit_below_the_prompt_capacity(denoiser):
    assert _capacities(denoiser, (1024, 2048)) == (1024, 2048, PROMPT_ROWS)


def test_the_prompt_capacity_may_end_the_list(denoiser):
    assert _capacities(denoiser, (1024, PROMPT_TOKENS)) == (1024, PROMPT_ROWS)


def test_the_prompt_capacity_alone_is_the_only_capacity(denoiser):
    assert _capacities(denoiser, (PROMPT_TOKENS,)) == (PROMPT_ROWS,)


@pytest.mark.parametrize("listed", [(1024, PROMPT_TOKENS + 1), (0, 2048)])
def test_a_capacity_the_prompt_cannot_use_is_refused(denoiser, listed):
    with pytest.raises(ValueError, match="positive and not exceed"):
        _capacities(denoiser, listed)
