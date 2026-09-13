"""Startup rejects component declarations that cannot supply their numerical calls."""

from pathlib import Path

import pytest

from uniserve_worker.bootstrap.catalog import CatalogEntry
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.loader.source import ModelSource
from uniserve_worker.modeling.components import Call, CallSpec, ComponentSpec
from uniserve_worker.models.qwen3 import Qwen3ForCausalLM
from uniserve_worker.models.stub import StubModel

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("base", "call"),
    [(Qwen3ForCausalLM, call) for call in Call if call is not Call.TEXT]
    + [
        (StubModel, Call.ENCODE_TEXT),
        (StubModel, Call.ENCODE_CONDITIONING),
        (StubModel, Call.DECODE_VIDEO),
        (StubModel, Call.DECODE_AUDIO),
    ],
)
def test_source_validation_rejects_unsupported_calls_even_without_local_components(base, call):
    class DeclaredModel(base):
        @classmethod
        def components(cls, config):
            return (ComponentSpec("encoder", (CallSpec(call),)),)

    source = ModelSource(Path("."), None, {}, CatalogEntry("DeclaredModel", DeclaredModel))

    with pytest.raises(WorkerError, match=f"unsupported {call.value} computation"):
        source.validate({})


def test_source_validation_rejects_duplicate_component_roles():
    class DeclaredModel(Qwen3ForCausalLM):
        @classmethod
        def components(cls, config):
            return (
                ComponentSpec("model", (CallSpec(Call.TEXT),)),
                ComponentSpec("model", (CallSpec(Call.TEXT, stage="last"),)),
            )

    source = ModelSource(Path("."), None, {}, CatalogEntry("DeclaredModel", DeclaredModel))

    with pytest.raises(WorkerError, match="repeats component role"):
        source.validate({})
