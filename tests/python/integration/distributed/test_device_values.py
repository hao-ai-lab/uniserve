"""Public numerical calls follow tensor devices.

They also preserve caller state.
"""

import pytest
import torch

from uniserve.cache import Config, mha
from uniserve.nn import RotaryEmbedding
from uniserve.quantization import Quantizer
from uniserve.runtime import PrefixCache

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_rotary_factors_follow_positions_on_another_device():
    with torch.cuda.device(0):
        rotary = RotaryEmbedding(8, theta=100).to("cuda:1")
        positions = torch.tensor([0, 1, 9], device="cuda:1")
        actual = rotary(positions, dtype=torch.float32, sequence_length=10)
        frequency = 1 / (100 ** (torch.arange(0, 8, 2).float() / 8))
        phase = torch.tensor([0, 1, 9]).float()[:, None] * frequency
        for value, reference in zip(
            actual, (phase.cos(), phase.sin()), strict=True
        ):
            assert value.device == positions.device
            torch.testing.assert_close(value.cpu(), reference)
        assert torch.cuda.current_device() == 0


@pytest.mark.parametrize("quantized", (False, True))
def test_cache_reset_follows_backing_on_another_device(quantized):
    with (
        torch.cuda.device(0),
        PrefixCache(
            Config({"attention": mha.Config(1, 16, (0,), torch.float32)}),
            num_blocks=3,
            block_size=4,
            device="cuda:1",
            quantization={
                "attention": Quantizer("fp8", axis=0) if quantized else None
            },
        ) as cache,
    ):
        state = cache.state("attention")
        source = torch.full((12, 1, 16), 896.0, device="cuda:1")
        state.write((0, 1, 2), start=0, key=source, value=-source)
        cache.zero_blocks("attention", (0, 2))
        expected = torch.zeros(12, 1, 16)
        expected[4:8] = 896
        key, value = state.read((0, 1, 2), start=0, length=12)
        torch.testing.assert_close(key.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(value.cpu(), -expected, rtol=0, atol=0)
        assert state.initialized["key"].tolist() == [False, True, False]
        if quantized:
            assert state.key.buffers()["scale"].flatten().tolist() == [1, 2, 1]
        assert torch.cuda.current_device() == 0


@torch.inference_mode()
@pytest.mark.parametrize("quantized", (False, True))
def test_cache_updates_follow_backing_and_preserve_existing_values_on_another_device(  # noqa: E501
    quantized,
):
    with (
        torch.cuda.device(0),
        PrefixCache(
            Config({"attention": mha.Config(1, 16, (0,), torch.float32)}),
            num_blocks=3,
            block_size=4,
            device="cuda:1",
            quantization={
                "attention": Quantizer("fp8", axis=0) if quantized else None
            },
        ) as cache,
    ):
        state = cache.state("attention")
        first = torch.full((1, 1, 16), 448.0, device="cuda:1")
        state.write((0,), start=0, key=first, value=-first)
        current = torch.full((3, 1, 16), 896.0, device="cuda:1")
        indices = torch.tensor([1, 4, -1], device="cuda:1")
        state.update(current, -current, indices=indices)
        expected = torch.zeros(12, 1, 16)
        expected[0] = 448
        expected[[1, 4]] = 896
        key, value = state.read((0, 1, 2), start=0, length=12)
        torch.testing.assert_close(key.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(value.cpu(), -expected, rtol=0, atol=0)
        assert all(
            flags.tolist() == [True, True, False]
            for flags in state.initialized.values()
        )
        if quantized:
            assert state.key.buffers()["scale"].flatten().tolist() == [2, 2, 1]
        assert torch.cuda.current_device() == 0
