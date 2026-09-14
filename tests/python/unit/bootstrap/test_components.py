"""Component startup preserves callable and participation constraints."""

import pytest
import torch

from uniserve.distributed.mesh import Communicator, DeviceMesh
from uniserve.distributed.parallel import ParallelConfig
from uniserve.model.components import ComponentCall
from uniserve.model.model import Model
from uniserve_worker.bootstrap.components import bind_components, validate_components
from uniserve_worker.config import ComponentConfig
from uniserve_worker.execution.model_entry import ModelEntry
from uniserve_worker.foundation.errors import WorkerError

pytestmark = pytest.mark.unit


def test_binding_rejects_a_missing_numerical_method():
    class DeclaredModel(Model):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Identity()

        @classmethod
        def component_calls(cls, config):
            return (ComponentCall("encoder", "encode:text"),)

    group = Communicator(device=torch.device("cpu"))
    binding = ModelEntry(
        "encoder",
        ComponentConfig((0,)),
        group,
        DeviceMesh((0,), 0, ParallelConfig(), group.device),
        group.device,
    )
    with pytest.raises(WorkerError, match="no callable 'encode'"):
        bind_components(DeclaredModel(), {"encoder": binding})


def test_source_validation_rejects_conflicting_stage_declarations():
    class DeclaredModel(Model):
        @classmethod
        def component_calls(cls, config):
            return (
                ComponentCall("", "forward"),
                ComponentCall("", "forward", stage="last"),
            )

    with pytest.raises(WorkerError, match="repeats numerical method"):
        validate_components(DeclaredModel, {}, {})
