"""A Ulysses component's head projection mode reaches its attention config."""

import pytest

from uniserve.nn.attention import Ulysses
from uniserve_worker.bootstrap.model_loader import attention_parallel
from uniserve_worker.config.deployment import ComponentConfig, ParallelConfig

pytestmark = pytest.mark.unit


def _component(sequence):
    parallel = ParallelConfig.from_dict({"sequence_parallel": sequence})
    return ComponentConfig(
        ranks=tuple(range(parallel.world_size)), parallel_config=parallel
    )


@pytest.mark.parametrize(
    ("sequence", "heads"),
    [
        ({"kind": "ulysses", "ulysses_degree": 8}, Ulysses()),
        (
            {"kind": "ulysses", "ulysses_degree": 8, "projection": "gather"},
            Ulysses(),
        ),
        (
            {"kind": "ulysses", "ulysses_degree": 8, "projection": "exchange"},
            Ulysses(projection="exchange"),
        ),
        (
            {
                "kind": "ulysses",
                "ulysses_degree": 8,
                "projection": "exchange",
                "projection_rows": 4096,
            },
            Ulysses(projection="exchange", projection_rows=4096),
        ),
        (
            {"kind": "hybrid", "ulysses_degree": 2, "allgather_degree": 2},
            Ulysses(),
        ),
    ],
)
def test_the_deployment_selects_the_ulysses_projection(sequence, heads):
    component = _component(sequence)
    assert attention_parallel(component).heads == heads
    parsed = ParallelConfig.from_dict(component.parallel_config.to_dict())
    assert parsed == component.parallel_config


@pytest.mark.parametrize(
    "sequence",
    [
        {"kind": "ulysses", "projection": "scatter"},
        {"kind": "ulysses", "projection_rows": 4096},
        {"kind": "ulysses", "projection": "exchange", "projection_rows": 0},
        {"kind": "ulysses", "projection": "exchange", "projection_rows": True},
        {"kind": "hybrid", "projection": "exchange"},
        {"kind": "allgather", "projection": "gather"},
    ],
)
def test_invalid_projection_settings_are_rejected(sequence):
    with pytest.raises(ValueError):
        _component(sequence)
