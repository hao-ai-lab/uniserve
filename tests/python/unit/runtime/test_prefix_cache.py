"""Unit-pool planes, prefix state addressing and fixed-scale writes."""

from contextlib import ExitStack

import pytest
import torch

from uniserve.cache import Config, State, mha
from uniserve.quantization import Quantizer
from uniserve.runtime import PrefixCache
from uniserve.runtime.prefix_cache import CacheTable, plan_units

pytestmark = pytest.mark.unit


def _layer(head_dim, window=None, heads=2, dtype=torch.float32):
    return mha.Config(heads, head_dim, tuple(range(heads)), dtype, window)


def _hybrid_config():
    """Five windowed layers per full layer, the full rows twice as wide."""
    layers = {}
    for index in range(12):
        full = index % 6 == 5
        layers[f"layers.{index}"] = _layer(16) if full else _layer(8, window=16)
    return Config(layers)


def test_homogeneous_pool_keeps_the_per_layer_stack_layout():
    config = Config({f"layers.{index}": _layer(8) for index in range(3)})
    with PrefixCache(config, num_units=5, block_size=4, device="cpu") as cache:
        planes = cache.planes
        # One group: one unit per page holding every layer as a column, one
        # table, and pages of exactly ``block_size`` tokens.
        assert planes.columns == 3
        assert [
            (group.page_tokens, group.units_per_page) for group in planes.groups
        ] == [(4, 1)]
        assert cache.tables == (CacheTable(0, 0),)
        stack = cache.planes_of(0, "key")
        assert stack.shape == (3, 5, 4, 2, 8)
        for index in range(3):
            name = f"layers.{index}"
            state = cache.state(name)
            assert cache.table(name) == 0
            assert state.key.shape == (5, 4, 2, 8)
            # Layer ``k`` of the stack is the layer's own storage.
            values = torch.full((4, 2, 8), float(index + 1))
            state.write((3,), start=0, key=values, value=-values)
            torch.testing.assert_close(stack[index, 3], values, rtol=0, atol=0)
        # Only the written unit changed.
        assert torch.count_nonzero(stack[:, (0, 1, 2, 4)]).item() == 0


def test_hybrid_groups_read_column_planes_of_one_unit_pool():
    with PrefixCache(
        _hybrid_config(), num_units=6, block_size=4, device="cpu"
    ) as cache:
        planes = cache.planes
        windowed, full = planes.groups
        # gcd(10, 2) columns; the widest rows (full attention) fill a plane
        # with ``block_size`` tokens, the half-width windowed rows with twice
        # as many.
        assert planes.columns == 2
        assert planes.plane_bytes == 4 * 2 * 16 * 4
        assert (
            windowed.window,
            windowed.page_tokens,
            windowed.units_per_page,
        ) == (16, 8, 5)
        assert (full.window, full.page_tokens, full.units_per_page) == (
            None,
            4,
            1,
        )
        # Two fields per column, each with one initialization flag per unit.
        assert planes.unit_bytes == 2 * 2 * (planes.plane_bytes + 1)
        assert cache.tables == tuple(CacheTable(0, row) for row in range(5)) + (
            CacheTable(1, 0),
        )

        # The windowed group's k-th layer reads column k % 2 of its page's
        # unit k // 2; the full group's layers read one unit's two columns.
        windowed_names = [
            f"layers.{index}" for index in range(12) if index % 6 != 5
        ]
        full_names = [f"layers.{index}" for index in (5, 11)]
        for position, name in enumerate(windowed_names):
            assert cache.table(name) == position // 2
            assert cache.state(name).key.shape == (6, 8, 2, 8)
        for name in full_names:
            assert cache.table(name) == 5
            assert cache.state(name).key.shape == (6, 4, 2, 16)

        # A layer's writes land in its own column of the addressed unit.
        values = torch.arange(8 * 2 * 8, dtype=torch.float32).reshape(8, 2, 8)
        cache.state(windowed_names[3]).write(
            (2,), start=0, key=values, value=-values
        )
        torch.testing.assert_close(
            cache.planes_of(0, "key")[1, 2], values, rtol=0, atol=0
        )
        assert torch.count_nonzero(cache.planes_of(0, "key")[0]).item() == 0

        # Both groups address the same bytes: the full group reads the unit's
        # column as its own page shape, which the unit allocator keeps
        # exclusive between groups.
        assert torch.equal(
            cache.planes_of(1, "key")[1, 2].reshape(-1).view(torch.uint8),
            values.reshape(-1).view(torch.uint8),
        )

        # Resetting a unit clears every column, field and flag of it alone.
        ones = torch.ones(4, 2, 16)
        cache.state(full_names[0]).write((4,), start=0, key=ones, value=ones)
        cache.zero_units((2,))
        assert torch.count_nonzero(cache.planes_of(0, "key")[:, 2]).item() == 0
        assert not cache.state(windowed_names[3]).initialized["key"][2]
        assert cache.state(full_names[0]).initialized["value"][4]
        torch.testing.assert_close(
            cache.state(full_names[0]).read((4,), start=0, length=4)[0], ones
        )


def test_quantized_units_carry_one_scale_per_column_and_field():
    config = Config({f"layers.{index}": _layer(8) for index in range(2)})
    planes = plan_units(
        config,
        block_size=4,
        quantization={name: Quantizer("fp8", axis=0) for name in config.layers},
    )
    # FP8 rows are a quarter of the FP32 rows; each column and field adds
    # one flag byte and one FP32 scale.
    assert planes.plane_bytes == 4 * 2 * 8
    assert planes.unit_bytes == 2 * 2 * (planes.plane_bytes + 1 + 4)


@pytest.mark.parametrize(
    ("layers", "options"),
    [
        # A windowed row that does not divide the plane into a power-of-two
        # page.
        ({"a": _layer(16), "b": _layer(12, window=8)}, {"block_size": 4}),
        # A page size that is not a power of two.
        ({"a": _layer(8)}, {"block_size": 3}),
        # Layers that do not share one storage quantizer.
        (
            {"a": _layer(8), "b": _layer(8)},
            {"block_size": 4, "quantization": {"a": Quantizer("fp8", axis=0)}},
        ),
    ],
)
def test_unit_planning_rejects_unrepresentable_layouts(layers, options):
    with pytest.raises(ValueError):
        plan_units(Config(layers), **options)


def _cache(quantized=False, device="cpu"):
    config = Config({"attention": mha.Config(2, 2, (0, 1), torch.float32)})
    return PrefixCache(
        config,
        num_units=3,
        block_size=4,
        device=device,
        quantization={
            "attention": Quantizer("fp8", axis=0) if quantized else None
        },
    )


@torch.inference_mode()
@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda", marks=pytest.mark.gpu))
)
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


@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda", marks=pytest.mark.gpu))
)
@pytest.mark.parametrize("quantized", (False, True))
def test_zero_units_preserves_other_units_and_layers(device, quantized):
    config = Config({name: _layer(16) for name in ("attention", "other")})
    with PrefixCache(
        config,
        num_units=5,
        block_size=4,
        device=device,
        quantization={
            name: Quantizer("fp8", axis=0) if quantized else None
            for name in config.layers
        },
    ) as cache:
        state, other = cache.state("attention"), cache.state("other")
        key = (
            torch.arange(1, 6, device=device).float() * 448
        ).repeat_interleave(4)
        key = key[:, None, None].expand(20, 2, 16).contiguous()
        state.write((0, 1, 2, 3, 4), start=0, key=key, value=-key)
        other.write((1, 3), start=0, key=key[:8], value=key[:8])
        # Caller order and duplicate indices do not change a reset; other
        # units of every column retain their values and initialization.
        cache.zero_units((4, 0, 2, 2))
        expected = key.clone().reshape(5, 4, 2, 16)
        expected[[0, 2, 4]] = 0
        actual = state.read((0, 1, 2, 3, 4), start=0, length=20)
        for value, reference in zip(actual, (expected, -expected), strict=True):
            torch.testing.assert_close(
                value, reference.reshape(20, 2, 16), rtol=0, atol=0
            )
        for layer in (state, other):
            for field in ("key", "value"):
                assert layer.initialized[field].tolist() == [
                    False,
                    True,
                    False,
                    True,
                    False,
                ]
        torch.testing.assert_close(
            other.read((1, 3), start=0, length=8)[0], key[:8], rtol=0, atol=0
        )
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
        updated = state.read((2,), start=0, length=4)[0]
        torch.testing.assert_close(updated[1:2], replacement, rtol=0, atol=0)
        assert torch.count_nonzero(updated[[0, 2, 3]]).item() == 0
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
    cache.zero_units((0,))
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
