"""Invalid capability and placement declarations fail at worker startup."""

import pytest
import torch

from uniserve.model import Encoder, EntryPoint
from uniserve_worker.bootstrap.components import validate_components
from uniserve_worker.bootstrap.config import ComponentConfig
from uniserve_worker.foundation.errors import WorkerError

pytestmark = pytest.mark.unit


def test_binding_rejects_a_missing_numerical_method():
    model = torch.nn.Module()
    model.encoder = torch.nn.Identity()
    with pytest.raises(WorkerError, match="no callable 'encode'"):
        validate_components(
            model,
            {"encoder": ComponentConfig((0,))},
            entries={"encoder": (EntryPoint("encode"),)},
            paths={"encoder": "encoder.encode"},
        )


def test_source_validation_rejects_conflicting_stage_declarations():
    model = Encoder(torch.nn.Identity())
    with pytest.raises(WorkerError, match="repeats numerical method"):
        validate_components(
            model,
            {"encoder": ComponentConfig((0,))},
            entries={
                "": (EntryPoint("encode"), EntryPoint("encode", stage="last"))
            },
            paths={"encoder": "encode"},
        )


def test_component_requires_explicit_placement():
    network = torch.nn.Linear(4, 4)
    model = torch.nn.Module()
    model.first = Encoder(network)
    model.second = Encoder(network)
    model.conditioner = Encoder(network)
    with pytest.raises(WorkerError, match="require an explicit IPC entry"):
        validate_components(
            model,
            {"first": ComponentConfig((0,)), "second": ComponentConfig((0,))},
            entries={
                path: (EntryPoint("encode"),)
                for path in ("first", "second", "conditioner")
            },
            paths={"first": "first.encode", "second": "second.encode"},
        )
