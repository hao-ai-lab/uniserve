"""Invalid capability and placement declarations fail at worker startup."""

import pytest
import torch

from uniserve.distributed import Communicator
from uniserve.model import ComponentEntry, Encoder, EntryPoint
from uniserve_worker.bootstrap.components import validate_components
from uniserve_worker.config.deployment import ComponentConfig
from uniserve_worker.errors import WorkerError
from uniserve_worker.model_executor.component_binding import ComponentBinding

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("rank", (0, 1))
def test_component_media_assignment_uses_worker_local_ranks(rank):
    binding = ComponentBinding(
        "decoder",
        ComponentConfig(
            (1, 0), distribution="temporal_units", units_per_rank=2
        ),
        Communicator(ranks=(4, 5), rank=rank),
        mesh=None,
        device=torch.device("cpu"),
    )
    assert binding.owns
    assert tuple(binding.media_units(7, 3)) == ((9,) if rank == 0 else (7, 8))

    # A round that names its ranks deals the units in that order, and a
    # member it leaves out holds none.
    assert tuple(binding.media_units(7, 3, (0, 1))) == (
        (7, 8) if rank == 0 else (9,)
    )
    assert tuple(binding.media_units(7, 2, (0,))) == (
        (7, 8) if rank == 0 else ()
    )

    nonmember = ComponentBinding(
        "decoder",
        ComponentConfig((1 - rank,), distribution="temporal_units"),
        Communicator(ranks=(4, 5), rank=rank),
        mesh=None,
        device=torch.device("cpu"),
    )
    assert not nonmember.owns


def test_binding_rejects_a_missing_numerical_method():
    model = torch.nn.Module()
    model.encoder = torch.nn.Identity()
    with pytest.raises(WorkerError, match="no callable 'encode'"):
        validate_components(
            model,
            {"encoder": ComponentConfig((0,))},
            entries={
                "encoder": ComponentEntry("encoder", (EntryPoint("encode"),))
            },
        )


def test_source_validation_rejects_conflicting_stage_declarations():
    model = Encoder(torch.nn.Identity())
    with pytest.raises(WorkerError, match="repeats numerical method"):
        validate_components(
            model,
            {"encoder": ComponentConfig((0,))},
            entries={
                "encoder": ComponentEntry(
                    "",
                    (
                        EntryPoint("encode"),
                        EntryPoint("encode", stage="last"),
                    ),
                )
            },
        )


def test_placement_rejects_an_undeclared_computation_entry():
    network = torch.nn.Linear(4, 4)
    model = torch.nn.Module()
    model.first = Encoder(network)
    model.second = Encoder(network)
    with pytest.raises(WorkerError, match="unknown components"):
        validate_components(
            model,
            {"first": ComponentConfig((0,)), "second": ComponentConfig((0,))},
            entries={"first": ComponentEntry("first", (EntryPoint("encode"),))},
        )
