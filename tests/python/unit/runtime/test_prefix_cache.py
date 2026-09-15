"""Prefix state copying, addressing and fixed-scale partial writes."""

from contextlib import ExitStack
from dataclasses import dataclass
from typing import ClassVar, Literal

import pytest
import torch

from uniserve.cache import Config, State, StateConfig, mha
from uniserve.quantization import Quantizer
from uniserve.runtime import PrefixCache
from uniserve.tensors import BufferConfig

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class Snapshot(StateConfig):
    indexing: ClassVar[Literal["states"]] = "states"

    def buffers(self, *, num_blocks, block_size, dtype, quantizer):
        return {
            "hidden": BufferConfig((num_blocks, 2), torch.float32),
            "initialized": BufferConfig((num_blocks,), torch.bool),
        }

    def bind(self, tensors, *, block_size, dtype, quantizer):
        return State(
            {"hidden": tensors["hidden"]},
            {"hidden": tensors["initialized"]},
            block_size,
        )


def test_heterogeneous_state_preserves_overlapping_block_sources():
    config = Config(
        {
            "recurrent": Snapshot(),
            "attention": mha.Config(4, 2, (1, 3), torch.float32),
        }
    )
    cache = PrefixCache(
        config,
        num_blocks={"recurrent": 3, "attention": 2},
        block_size={"recurrent": 1, "attention": 4},
        device="cpu",
    )
    state = cache.state("recurrent")
    state.tensors["hidden"].copy_(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    )
    cache.mark_initialized("recurrent", (0, 2), fields=("hidden",))
    state.copy_blocks((0, 1), (1, 0))
    torch.testing.assert_close(
        state.tensors["hidden"],
        torch.tensor([[3.0, 4.0], [1.0, 2.0], [5.0, 6.0]]),
        rtol=0,
        atol=0,
    )
    assert state.initialized["hidden"].tolist() == [False, True, True]
    views = state.transfer_views((2, 0))
    views["hidden.values"][0].fill_(7)
    assert state.tensors["hidden"][2].tolist() == [7.0, 7.0]
    assert views["hidden.initialized"][0].item()
    cache.zero_blocks("recurrent", (1,))
    assert state.tensors["hidden"][1].tolist() == [0.0, 0.0]
    assert state.initialized["hidden"].tolist() == [False, False, True]
    assert cache.state("attention").key.shape == (2, 4, 2, 2)
    cache.close()
    # Borrowing alone does not invalidate tensors. Releasing an owner requires
    # the caller to have retired numerical uses, rather than mutating them.
    assert state.tensors["hidden"][2].tolist() == [7.0, 7.0]
    with pytest.raises(KeyError):
        cache.state("recurrent")


def _cache(quantized=False, device="cpu"):
    config = Config({"attention": mha.Config(2, 2, (0, 1), torch.float32)})
    return PrefixCache(
        config,
        num_blocks=3,
        block_size=4,
        device=device,
        quantization={
            "attention": Quantizer("fp8", axis=0) if quantized else None
        },
    )


@torch.inference_mode()
@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("representation", ("dense", "fp8", "strided"))
def test_device_block_copies_preserve_snapshots_masks_and_backing_fields(
    device, representation
):
    with ExitStack() as owners:
        if representation == "strided":
            hidden = torch.arange(
                18, dtype=torch.float32, device=device
            ).reshape(3, 2, 3)
            state = State(
                {"hidden": hidden.transpose(1, 2)},
                {"hidden": torch.tensor([True, False, True], device=device)},
                1,
            )
        else:
            cache = owners.enter_context(
                _cache(representation == "fp8", device)
            )
            state = cache.state("attention")
            for block in (0, 2):
                values = torch.full(
                    (4, 2, 2), (block + 1) * 448.0, device=device
                )
                state.write((block,), start=0, key=values, value=-values)
        # Both selection columns may be strided. Unselected entries in their
        # surrounding allocation have no numerical meaning.
        source = torch.tensor([0, 99, -1, 99, 1, 99], device=device)[::2]
        target = torch.tensor([1, 99, -1, 99, 0, 99], device=device)[::2]
        state.copy_blocks(source, target)
        graph = None
        if device == "cuda":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                state.copy_blocks(source, target)

        before = {
            name: tuple(view.clone() for view in views)
            for name, views in state.transfer_views((0, 1, 2)).items()
        }
        source.copy_(torch.tensor([2, -1, 0], device=device))
        target.copy_(torch.tensor([0, -1, 2], device=device))
        if graph is None:
            state.copy_blocks(source, target)
        else:
            graph.replay()
        for name, views in state.transfer_views((0, 1, 2)).items():
            for value, previous in zip(views, (2, 1, 0), strict=True):
                expected = before[name][previous]
                if value.dtype.is_floating_point and value.element_size() == 1:
                    value, expected = (
                        value.view(torch.uint8),
                        expected.view(torch.uint8),
                    )
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
        if graph is not None:
            graph.reset()


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ((-1,), (0,)),
        ((0,), (-1,)),
        ((-2,), (-2,)),
        ((3,), (0,)),
        ((0,), (3,)),
        ((0, 1), (2, 2)),
    ],
)
def test_device_block_copies_reject_invalid_indices(source, target):
    with _cache() as cache:
        state = cache.state("attention")
        with pytest.raises(ValueError):
            state.copy_blocks(torch.tensor(source), torch.tensor(target))


def test_empty_state_accepts_masked_device_copies():
    state = State(
        {"hidden": torch.empty(0, 2)},
        {"hidden": torch.empty(0, dtype=torch.bool)},
        1,
    )
    state.copy_blocks(torch.tensor([-1, -1]), torch.tensor([-1, -1]))
    state.copy_blocks(
        torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)
    )
    with pytest.raises(ValueError):
        state.copy_blocks(torch.tensor([0]), torch.tensor([0]))


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (torch.tensor([0], dtype=torch.int32), torch.tensor([1])),
        (torch.tensor([[0]]), torch.tensor([[1]])),
        (torch.tensor([0]), torch.tensor([1, 2])),
        ((0,), torch.tensor([1])),
    ],
)
def test_block_copy_rejects_incompatible_index_representations(source, target):
    with _cache() as cache, pytest.raises(ValueError):
        cache.state("attention").copy_blocks(source, target)


def test_aliases_copy_from_original_block_values():
    values = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    flags = torch.tensor([True, False, True])
    state = State({"x": values, "y": values}, {"x": flags, "y": flags}, 1)
    state.copy_blocks((0, 1), (1, 0))
    torch.testing.assert_close(
        values, torch.tensor([[2.0, 3.0], [0.0, 1.0], [4.0, 5.0]])
    )
    assert flags.tolist() == [False, True, True]


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("quantized", (False, True))
def test_zero_blocks_preserves_other_pages_and_layer_state(device, quantized):
    config = Config(
        {
            "attention": mha.Config(2, 16, (0, 1), torch.float32),
            "other": Snapshot(),
        }
    )
    with PrefixCache(
        config,
        num_blocks={"attention": 5, "other": 2},
        block_size={"attention": 3, "other": 1},
        device=device,
        quantization={
            "attention": Quantizer("fp8", axis=0) if quantized else None
        },
    ) as cache:
        state, other = cache.state("attention"), cache.state("other")
        key = (
            torch.arange(1, 6, device=device).float() * 448
        ).repeat_interleave(3)
        key = key[:, None, None].expand(15, 2, 16).contiguous()
        state.write((0, 1, 2, 3, 4), start=0, key=key, value=-key)
        other.tensors["hidden"].fill_(7)
        cache.mark_initialized("other", (0, 1), fields=("hidden",))
        # Caller order and duplicate indices do not change a reset; holes and
        # independent layer storage must retain their values and initialization.
        cache.zero_blocks("attention", (4, 0, 2, 2))
        expected = key.clone().reshape(5, 3, 2, 16)
        expected[[0, 2, 4]] = 0
        actual = state.read((0, 1, 2, 3, 4), start=0, length=15)
        for value, reference in zip(actual, (expected, -expected), strict=True):
            torch.testing.assert_close(
                value, reference.reshape(15, 2, 16), rtol=0, atol=0
            )
        assert state.initialized["key"].tolist() == [
            False,
            True,
            False,
            True,
            False,
        ]
        assert state.initialized["value"].tolist() == [
            False,
            True,
            False,
            True,
            False,
        ]
        assert other.tensors["hidden"].tolist() == [[7, 7], [7, 7]]
        assert other.initialized["hidden"].tolist() == [True, True]
        if quantized:
            for tensor in (state.key, state.value):
                assert tensor.buffers()["scale"].flatten().tolist() == [
                    1,
                    2,
                    1,
                    4,
                    1,
                ]
        # A reused encoded page must derive its first scale from new values.
        replacement = torch.full((1, 2, 16), 1792.0, device=device)
        state.write((2,), start=1, key=replacement, value=-replacement)
        updated = state.read((2,), start=0, length=3)[0]
        torch.testing.assert_close(updated[1:2], replacement, rtol=0, atol=0)
        assert torch.count_nonzero(updated[[0, 2]]).item() == 0
        if quantized:
            assert state.key.buffers()["scale"][2].item() == 4


@pytest.mark.parametrize("quantized", (False, True))
def test_slot_zero_is_writable_and_minus_one_does_not_initialize(quantized):
    state = _cache(quantized).state("attention")
    key = torch.tensor(
        [
            [[448.0, -448.0], [224.0, -224.0]],
            [[896.0, -896.0], [448.0, -448.0]],
            [[10000.0, 10000.0], [10000.0, 10000.0]],
        ]
    )
    state.update(key, -key, indices=torch.tensor([0, 1, -1]))
    expected = key[:2].clone()
    actual = state.read((0,), start=0, length=2)
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], -expected, rtol=0, atol=0)
    assert state.initialized["key"].tolist() == [True, False, False]
    assert state.initialized["value"].tolist() == [True, False, False]
    for index in (-2, 12):
        with pytest.raises(ValueError, match="out of bounds"):
            state.update(key[:1], key[:1], indices=torch.tensor([index]))


def test_fp8_partial_copy_preserves_uncovered_values_and_transfers_encoding():
    cache = _cache(True)
    state = cache.state("attention")
    source = torch.tensor(
        [[[896.0, -896.0], [448.0, -448.0]], [[448.0, -448.0], [224.0, -224.0]]]
    )
    state.write((2, 0), start=3, key=source, value=-source)
    assert state.key.buffers()["scale"].flatten().tolist() == [1.0, 1.0, 2.0]
    views = state.transfer_blocks((2, 0), start=3, length=2)
    assert views["key.scale"][0].item() == 2.0
    assert views["key.scale"][1].item() == 1.0
    assert views["key.initialized"][0].item()
    torch.testing.assert_close(
        state.read((2, 0), start=3, length=2)[0], source, rtol=0, atol=0
    )
    state.copy_blocks((2, 0), (0, 2))
    assert state.key.buffers()["scale"].flatten().tolist() == [2.0, 1.0, 1.0]
    # Copy a separately encoded head rectangle through the public tensor value.
    encoded = Quantizer("fp8").from_tensors(
        {
            "values": torch.tensor([[[112.0], [-112.0]]]).to(
                torch.float8_e4m3fn
            ),
            "scale": torch.tensor(4.0),
        },
        shape=(1, 2, 1),
        dtype=torch.bfloat16,
    )
    state.copy_region(
        encoded,
        field="key",
        block=0,
        source_slice=(slice(0, 1), slice(0, 2), slice(0, 1)),
        target_slice=(slice(1, 2), slice(0, 2), slice(1, 2)),
        workspace={
            "values": torch.empty(2),
            "rounded": torch.empty(2, dtype=torch.bfloat16),
        },
    )
    expected = torch.tensor([[[0.0, 448.0], [0.0, -448.0]]])
    torch.testing.assert_close(
        state.read((0,), start=1, length=1)[0], expected, rtol=0, atol=0
    )
    assert state.key.buffers()["scale"][0].item() == 2.0
    growth = torch.full((1, 2, 2), 1792.0)
    state.write((0,), start=2, key=growth, value=-growth)
    assert state.key.buffers()["scale"][0].item() == 4.0
    # Earlier head updates and the final token remain after a larger write
    # causes every encoded value in the resident block to be rescaled.
    torch.testing.assert_close(
        state.read((0,), start=1, length=1)[0], expected, rtol=0, atol=0
    )
    torch.testing.assert_close(
        state.read((0,), start=3, length=1)[0], source[:1], rtol=0, atol=0
    )
    cache.zero_blocks("attention", (0,))
    assert not state.initialized["key"][0]
    assert state.key.buffers()["scale"][0].item() == 1.0
    assert state.read((0,), start=0, length=0)[0].shape == (0, 2, 2)


def test_mha_binding_borrows_buffers_and_validates_physical_layout():
    config = mha.Config(4, 2, (3, 1), torch.bfloat16)
    quantizer = Quantizer("fp8", axis=0)
    specifications = config.buffers(
        num_blocks=2, block_size=4, dtype=None, quantizer=quantizer
    )
    backing = {
        name: torch.zeros(spec.shape, dtype=spec.dtype)
        for name, spec in specifications.items()
    }
    state = config.bind(backing, block_size=4, dtype=None, quantizer=quantizer)
    assert state.key.buffers()["values"] is backing["key.values"]
    assert state.initialized["value"] is backing["value.initialized"]
    with pytest.raises(ValueError, match="one scale per block"):
        config.buffers(
            num_blocks=2, block_size=4, dtype=None, quantizer=Quantizer("fp8")
        )
    backing["key.scale"] = torch.ones(())
    with pytest.raises(ValueError, match="disagrees with its layout"):
        config.bind(backing, block_size=4, dtype=None, quantizer=quantizer)
