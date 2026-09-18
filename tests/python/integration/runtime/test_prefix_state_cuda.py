"""Prefix state updates preserve masks and scales through CUDA graph replay."""

import pytest
import torch
import torch.multiprocessing as mp

from uniserve.cache import Config, mha
from uniserve.quantization import Quantizer
from uniserve.runtime import PrefixCache

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
def _reject_invalid_index(rank, index):
    # Device assertions poison their CUDA context, so each rejection runs in
    # its own process. Replay must validate live addresses, not capture values.
    cache = PrefixCache(
        Config({"attention": mha.Config(1, 16, (0,), torch.float32)}),
        num_blocks=3,
        block_size=4,
        device="cuda",
    )
    state = cache.state("attention")
    key = torch.ones(1, 1, 16, device="cuda")
    indices = torch.tensor([0], device="cuda")
    state.update(key, key, indices=indices)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.update(key, key, indices=indices)
    indices.fill_(index)
    graph.replay()
    with pytest.raises(RuntimeError, match="device-side assert"):
        torch.cuda.synchronize()


@pytest.mark.parametrize("index", (-2, 12))
def test_cache_replay_rejects_out_of_range_addresses(index):
    mp.spawn(_reject_invalid_index, args=(index,), nprocs=1, join=True)


@torch.inference_mode()
def _reject_invalid_copy(rank, targets):
    cache = PrefixCache(
        Config({"attention": mha.Config(1, 16, (0,), torch.float32)}),
        num_blocks=3,
        block_size=4,
        device="cuda",
    )
    state = cache.state("attention")
    source = torch.tensor([0, 1], device="cuda")
    target = torch.tensor([1, 2], device="cuda")
    state.copy_blocks(source, target)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.copy_blocks(source, target)
    target.copy_(torch.tensor(targets, device="cuda"))
    with pytest.raises(RuntimeError, match="device-side assert"):
        graph.replay()
        torch.cuda.synchronize()


@pytest.mark.parametrize("targets", ((1, 3), (1, 1)))
def test_block_copy_replay_rejects_invalid_or_repeated_targets(targets):
    # CUDA assertions have process-wide consequences. Isolate malformed
    # replay inputs while exercising the actual public device call.
    mp.spawn(_reject_invalid_copy, args=(targets,), nprocs=1, join=True)


@pytest.mark.parametrize("quantized", (False, True))
@torch.inference_mode()
def test_state_updates_replay_addresses_and_first_write_scales(quantized):
    cache = PrefixCache(
        Config({"attention": mha.Config(2, 2, (0, 1), torch.float32)}),
        num_blocks=3,
        block_size=4,
        device="cuda",
        quantization={
            "attention": Quantizer("fp8", axis=0) if quantized else None
        },
    )
    state = cache.state("attention")
    key = torch.tensor(
        [
            [[448.0, -448.0], [224.0, -224.0]],
            [[896.0, -896.0], [448.0, -448.0]],
            [[1792.0, -1792.0], [896.0, -896.0]],
        ],
        device="cuda",
    )
    value = -key.clone()
    indices = torch.tensor([0, 1, -1], device="cuda")
    state.update(key, value, indices=indices)
    cache.zero_blocks("attention", (0, 1, 2))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.update(key, value, indices=indices)
    for slots in ((4, 5, -1), (-1, 8, 0)):
        cache.zero_blocks("attention", (0, 1, 2))
        indices.copy_(torch.tensor(slots, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.zeros(12, 2, 2)
        scales = {
            block: max(
                float(key[row].abs().max())
                for row, slot in enumerate(slots)
                if slot >= 0 and slot // 4 == block
            )
            / 448.0
            for block in range(3)
            if any(slot >= 0 and slot // 4 == block for slot in slots)
        }
        for row, slot in enumerate(slots):
            if slot == -1:
                continue
            source = key[row].cpu()
            if quantized:
                scale = scales[slot // 4]
                source = (source / scale).clamp(-448.0, 448.0).to(
                    torch.float8_e4m3fn
                ).float() * scale
            expected[slot] = source
        actual_key, actual_value = state.read((0, 1, 2), start=0, length=12)
        torch.testing.assert_close(actual_key.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            actual_value.cpu(), -expected, rtol=0, atol=0
        )
        flags = [
            any(slot >= 0 and slot // 4 == block for slot in slots)
            for block in range(3)
        ]
        assert state.initialized["key"].tolist() == flags
        assert state.initialized["value"].tolist() == flags
    if quantized:
        # Reuse an initialized block and grow its scale on the captured path.
        original = state.read((0,), start=0, length=1)[0].clone()
        indices.copy_(torch.tensor([1, -1, -1], device="cuda"))
        key[0].fill_(3584.0)
        value[0].fill_(-3584.0)
        graph.replay()
        torch.cuda.synchronize()
        assert state.key.buffers()["scale"][0].item() == 8.0
        torch.testing.assert_close(
            state.read((0,), start=0, length=1)[0], original, rtol=0, atol=0
        )
    graph.reset()
    cache.close()
