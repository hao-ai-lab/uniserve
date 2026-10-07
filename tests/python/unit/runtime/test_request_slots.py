"""Request views share their slot's bank and retain values through closure."""

import pytest
import torch

from uniserve.tensors import BufferConfig
from uniserve_worker.storage.request_slots import RequestSlots

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu))
)
def test_slot_views_preserve_independent_values_and_borrowed_lifetimes(device):
    fields = {
        "latent": BufferConfig((2, 3), torch.float32, capacity_shape=(3, 4)),
        "schedule": BufferConfig((2,), torch.int64, host=True),
    }
    storage = RequestSlots(2, state_buffers=fields, device=device)
    try:
        first = storage.tensors(1).view(fields)
        second = storage.tensors(2).view(fields)
        bank = storage.bank["latent"]
        expected = torch.arange(6, dtype=torch.float32, device=device)
        first["latent"].copy_(expected.reshape(2, 3))
        second["latent"].fill_(17)
        first["schedule"].fill_(3)
        second["schedule"].fill_(9)

        # Graphs index the full bank by slot; compact views use its leading
        # elements even when the current dimensions are below capacity.
        torch.testing.assert_close(bank[0].reshape(-1)[:6], expected)
        torch.testing.assert_close(
            bank[1].reshape(-1)[:6], torch.full_like(expected, 17)
        )
        assert first["schedule"].tolist() == [3, 3]
        assert second["schedule"].tolist() == [9, 9]
        assert first["schedule"].device.type == "cpu"
        assert first["schedule"].is_pinned() == (device != "cpu")

        storage.close()
        torch.testing.assert_close(first["latent"].reshape(-1), expected)
        assert second["schedule"].tolist() == [9, 9]
        with pytest.raises(RuntimeError, match="closed"):
            storage.tensors(1)
    finally:
        storage.close()
