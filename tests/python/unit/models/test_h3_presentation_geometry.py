"""Public execution planning reserves both Qwen and VAE reference spans."""

from dataclasses import replace

import pytest
import torch

from uniserve_worker.execution.batch import MediaGeometry
from uniserve_worker.execution.bounded_storage import BoundedTensorStorage
from uniserve_worker.models.minimax_h3.layout import H3Layout
from uniserve_worker.models.minimax_h3.model import MiniMaxH3Model
from uniserve_worker.models.minimax_h3.presentation import H3MediaGeometry
from uniserve_worker.models.minimax_h3.weights import H3Components
from uniserve_worker.nn.mesh import Communicator, DeviceMesh, EntryBindings
from uniserve_worker.nn.parallel import EntryConfig, ParallelConfig

pytestmark = pytest.mark.unit


def test_execution_reserves_expanded_presentation_and_image_pages():
    config = ParallelConfig()
    bindings = EntryBindings(
        {"output": EntryConfig((0,), config)},
        {"output": DeviceMesh((0,), 0, config)},
        Communicator(),
    )
    capacity = H3Layout.build(
        bindings,
        frames=124,
        text_rows=1024,
        audio_frames=207,
        height=480,
        width=832,
        attention="dense",
        reference_shape=(480, 832),
        presentation_tags=torch.ones(1024, dtype=torch.long),
    )
    model = MiniMaxH3Model(bindings, H3Components(None, None, None, None, None), capacity)
    # Qwen: six label tokens, 390 merged image patches plus boundaries, three prompt tokens.
    tags = (1,) * 6 + (0,) * 392 + (1,) * 3
    geometry = H3MediaGeometry(124, 7, 3, model.denoise_steps, (480, 832), tags)
    storage = BoundedTensorStorage(
        {
            name: torch.empty(spec.shape, dtype=spec.dtype, device="meta")
            for name, spec in model.scratch_schema.items()
        }
    )
    execution = model.build_execution(geometry, storage, None)
    packed = execution.layout.packed
    assert packed.text_indices.numel() == 448
    assert packed.reference_indices.numel() == 390
    assert packed.audio_indices[0] == 896
    assert packed.video_indices[0] == 1344
    assert model.execution_key(geometry) == execution.layout.shape_key
    assert packed.presentation_tags.tolist() == list(tags)
    with pytest.raises(ValueError, match="reference geometry exceeds"):
        model.execution_key(replace(geometry, reference_shape=(640, 832)))

    plain = MediaGeometry(124, 7, 3, model.denoise_steps)
    plain_execution = model.build_execution(plain, storage, None)
    assert model.execution_key(plain) == (124, 64, 207, 480, 832)
    assert plain_execution.layout.packed.reference_indices.numel() == 0
    assert plain_execution.layout.packed.text_indices.numel() == 64
