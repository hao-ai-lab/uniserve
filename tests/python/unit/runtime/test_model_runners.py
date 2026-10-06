"""Component operations have one unambiguous numerical runner."""

import pytest

from uniserve_worker._uniserve_ipc import ModelRunners
from uniserve_worker.errors import WorkerError
from uniserve_worker.protocol.call import ForwardMode

pytestmark = pytest.mark.unit


def test_ambiguous_binding_does_not_install_part_of_a_runner():
    runners = ModelRunners()
    runners.bind("model", (ForwardMode.PREFILL,), "first")
    with pytest.raises(WorkerError, match="multiple lane bindings"):
        runners.bind(
            "model", (ForwardMode.DECODE, ForwardMode.PREFILL), "second"
        )
    assert runners.get("model", ForwardMode.PREFILL) == "first"
    assert runners.get("model", ForwardMode.DECODE) is None
    runners.clear()
    assert runners.get("model", ForwardMode.PREFILL) is None
