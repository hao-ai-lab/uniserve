"""Captured inputs preserve their backing across live tensor copies."""

from dataclasses import dataclass

import pytest
import torch

from uniserve_worker.model_executor.cuda_graph import GraphInputs

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


@dataclass(frozen=True)
class Values:
    tokens: torch.Tensor
    coordinates: torch.Tensor
    label: str


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_copy_updates_nested_strided_and_broadcast_inputs(device):
    tokens = torch.zeros(8, dtype=torch.int64, device=device)[::2]
    coordinates = torch.zeros(1, 4, device=device).expand(3, 4)
    captured = {"rows": [Values(tokens, coordinates, "fixed"), tokens]}
    inputs = GraphInputs(captured)

    for offset in (3, 9):
        live_tokens = torch.arange(offset, offset + 4, device=device)
        live_coordinates = live_tokens.float().expand(3, 4)
        inputs.copy(
            {
                "rows": [
                    Values(live_tokens, live_coordinates, "live"),
                    live_tokens,
                ]
            }
        )
        torch.testing.assert_close(captured["rows"][0].tokens, live_tokens)
        torch.testing.assert_close(
            captured["rows"][0].coordinates, live_coordinates
        )
        torch.testing.assert_close(captured["rows"][1], live_tokens)
        assert captured["rows"][0].label == "fixed"

    # A batch may already borrow the captured backing on a later replay.
    inputs.copy(captured)
    torch.testing.assert_close(
        captured["rows"][0].coordinates, live_coordinates
    )


@pytest.mark.parametrize("mismatch", ("shape", "dtype", "device"))
def test_copy_rejects_a_changed_tensor_representation(mismatch):
    captured = torch.ones(4)
    inputs = GraphInputs((captured,))
    live = {
        "shape": lambda: torch.zeros(5),
        "dtype": lambda: torch.zeros(4, dtype=torch.int64),
        "device": lambda: torch.zeros(4, device="cuda:0"),
    }[mismatch]()

    with pytest.raises(
        ValueError, match="graph tensor shape or representation changed"
    ):
        inputs.copy((live,))
    torch.testing.assert_close(captured, torch.ones(4))
