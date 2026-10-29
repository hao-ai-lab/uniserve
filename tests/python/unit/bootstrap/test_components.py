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


def test_component_weights_are_written_only_off_the_device():
    host = ComponentConfig.from_dict({"ranks": [0], "weights": "host"})
    assert host.weights == "host"
    assert host.to_dict()["weights"] == "host"
    assert "weights" not in ComponentConfig.from_dict({"ranks": [0]}).to_dict()
    with pytest.raises(ValueError, match="unsupported component weights"):
        ComponentConfig.from_dict({"ranks": [0], "weights": "disk"})


def test_a_component_accelerator_names_its_ranks_gpus():
    shared = ComponentConfig.from_dict(
        {
            "ranks": [0, 1],
            "distribution": "temporal_units",
            "accelerator": "cuda:3",
        }
    )
    assert (shared.accelerator_index(0), shared.accelerator_index(1)) == (3, 3)
    assert shared.to_dict()["accelerator"] == "cuda:3"
    # A list follows the order of ``ranks``, not the rank numbers.
    each = ComponentConfig.from_dict(
        {
            "ranks": [2, 0],
            "distribution": "temporal_units",
            "accelerator": ["cuda:5", "cuda:4"],
        }
    )
    assert (each.accelerator_index(2), each.accelerator_index(0)) == (5, 4)
    assert each.to_dict()["accelerator"] == ["cuda:5", "cuda:4"]
    plain = ComponentConfig.from_dict({"ranks": [0]})
    assert plain.accelerator_index(0) is None
    assert "accelerator" not in plain.to_dict()
    pair = {"ranks": [0, 1], "distribution": "temporal_units"}
    for invalid in ("cuda", "cuda:-1", "cuda:01", "cpu", ["cuda:0", "cpu"]):
        with pytest.raises(ValueError, match="is not cuda:<index>"):
            ComponentConfig.from_dict({**pair, "accelerator": invalid})
    with pytest.raises(ValueError, match="one device per rank"):
        ComponentConfig.from_dict({**pair, "accelerator": ["cuda:0"]})


def test_only_host_media_components_take_an_accelerator():
    model = Encoder(torch.nn.Identity())
    with pytest.raises(WorkerError, match="take an accelerator"):
        validate_components(
            model,
            {"encoder": ComponentConfig((0,), accelerator="cuda:0")},
            entries={"encoder": ComponentEntry("", (EntryPoint("encode"),))},
        )
