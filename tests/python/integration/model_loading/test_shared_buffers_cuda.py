"""Buffers shared by several module paths load once onto the chosen device."""

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from uniserve import loading
from uniserve.loading import checkpoint, weights
from uniserve.nn.linear import Linear

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


class _Shared(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(4, 4, bias=False)
        self.register_buffer("table", torch.arange(4.0), persistent=False)


class _Owner(nn.Module):
    """Two capabilities reaching one module, as shared backbones do."""

    def __init__(self):
        super().__init__()
        shared = _Shared()
        self.first, self.second = shared, shared


def test_shared_buffers_follow_an_unindexed_cuda_device(tmp_path):
    save_file({"linear.weight": torch.eye(4)}, tmp_path / "model.safetensors")
    io = loading.Config()
    source = checkpoint.Config(name="primary").resolve(tmp_path, io=io)
    model = _Owner()

    def assign(reader):
        return (
            weights.Assignment(
                model.first.linear.weight, reader.get("linear.weight")
            ),
        )

    loading.load_weights(
        model,
        (source,),
        mapping=lambda owner: (
            weights.ModuleMapping(
                owner, "primary", assign, frozenset({"first.linear.weight"})
            ),
        ),
        device="cuda",
    )
    assert model.first.table is model.second.table
    assert model.first.table.device == torch.device(
        "cuda", torch.cuda.current_device()
    )
    torch.testing.assert_close(model.first.table.cpu(), torch.arange(4.0))
