"""The runtime-visible denoising plan follows the checkpoint forward count."""

import pytest

from uniserve_worker.models.minimax_h3.layout import H3Layout
from uniserve_worker.models.minimax_h3.model import MiniMaxH3Model
from uniserve_worker.models.minimax_h3.weights import H3Components
from uniserve_worker.nn.mesh import Communicator, DeviceMesh, EntryBindings
from uniserve_worker.nn.parallel import EntryConfig, ParallelConfig

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("forwards", [4, 8])
def test_output_owner_publishes_checkpoint_denoising_plan(forwards):
    # Output ranks carry no transformer, yet must advertise the same complete
    # media dependency plan as the denoiser ranks to the shared scheduler.
    bindings = EntryBindings(
        {"output": EntryConfig((0,), ParallelConfig())},
        {"output": DeviceMesh((0,), 0, ParallelConfig())},
        Communicator(),
    )
    layout = H3Layout.build(bindings, frames=22, text_rows=64, audio_frames=8)
    model = MiniMaxH3Model(
        bindings,
        H3Components(None, None, None, None, None),
        layout,
        denoise_steps=forwards,
    )
    assert model.denoise_steps == forwards
    assert model.media_plan.denoise_steps == forwards
