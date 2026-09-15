"""Public loading preserves values, shared identities and scope.

It also preserves completeness.
"""

from dataclasses import dataclass

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

from uniserve import loading
from uniserve.distributed import DeviceMesh
from uniserve.loading import checkpoint, weights
from uniserve.nn.linear import (
    ColumnParallelLinear,
    Linear,
    MergedColumnParallelLinear,
)
from uniserve.quantization import QuantizationConfig, QuantizedTensor, Quantizer

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class NetworkConfig:
    width: int = 4
    tied: bool = False


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.first = ColumnParallelLinear(
            config.width, config.width, bias=False
        )
        self.second = Linear(config.width, config.width, bias=False)
        if config.tied:
            self.second.weight = self.first.weight

    def forward(self, x):
        return self.second(self.first(x))


def _mapping(model):
    def assign(reader):
        return tuple(
            weights.Assignment(
                parameter,
                reader.get("first" if name.startswith("first") else "second"),
            )
            for name, parameter in model.named_parameters()
        )

    return (
        weights.ModuleMapping(
            model,
            "primary",
            assign,
            frozenset(name for name, _ in model.named_parameters()),
        ),
    )


@pytest.mark.parametrize("mode", ["eager", "layered"])
def test_public_model_and_weight_loading_agree(tmp_path, mode):
    first = torch.arange(16).view(4, 4).float() / 8
    second = torch.eye(4) * 2
    save_file(
        {"first": first, "second": second}, tmp_path / "model.safetensors"
    )
    io = loading.Config(mode=mode)
    sources = (checkpoint.Config().resolve(tmp_path, io=io),)
    result = loading.load_model(
        Network,
        NetworkConfig(),
        checkpoint=sources,
        mapping=_mapping,
        device="cpu",
        weights=weights.Config(dtype=torch.float32),
        io=io,
    )
    source = torch.arange(12).view(3, 4).float()
    expected = F.linear(F.linear(source, first), second)
    torch.testing.assert_close(result.model(source), expected, rtol=0, atol=0)
    existing = Network(NetworkConfig())
    loading.load_weights(
        existing,
        sources,
        mapping=_mapping,
        device="cpu",
        weights=weights.Config(dtype=torch.float32),
        io=io,
    )
    torch.testing.assert_close(existing(source), expected, rtol=0, atol=0)
    assert result.reports[0].loaded == frozenset(
        {"first.weight", "second.weight"}
    )


@pytest.mark.parametrize("extra", (False, True))
@pytest.mark.parametrize("mode", ("layered", "dummy"))
@pytest.mark.parametrize("format", ("safetensors", "pt"))
def test_post_load_can_derive_constants_without_hiding_unused_sources(
    tmp_path, extra, mode, format
):
    class Shifted(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.projection = Linear(width, width, bias=False)
            self.register_buffer("offset", torch.empty(width))

        def forward(self, value):
            return self.projection(value) + self.offset

    def mapping(model):
        def post_load(reader):
            model.offset = reader.get("offset").read().square()

        return (
            weights.ModuleMapping(
                model,
                "primary",
                lambda reader: (
                    weights.Assignment(
                        model.projection.weight, reader.get("weight")
                    ),
                ),
                frozenset({"projection.weight"}),
                post_load=post_load,
            ),
        )

    matrix = torch.eye(4) * 2
    offset = torch.tensor([-1.0, 2.0, -3.0, 4.0])
    tensors = {"weight": matrix, "offset": offset}
    if extra:
        tensors["unmapped"] = torch.ones(4)
    if mode == "dummy":
        # NaN payloads distinguish metadata-based initialization from reading
        # source values, including the auxiliary input to the derived offset.
        tensors = {
            name: torch.full_like(value, float("nan"))
            for name, value in tensors.items()
        }
    if format == "safetensors":
        save_file(tensors, tmp_path / "model.safetensors")
    else:
        torch.save(tensors, tmp_path / "model.pt")
    io = loading.Config(mode=mode)
    source = checkpoint.Config().resolve(tmp_path, io=io)

    def load():
        return loading.load_model(
            Shifted,
            4,
            checkpoint=(source,),
            mapping=mapping,
            device="cpu",
            weights=weights.Config(dtype=torch.float32),
            io=io,
        )

    if extra:
        with pytest.raises(RuntimeError, match="unexpected=.*unmapped"):
            load()
    else:
        result = load()
        inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        with source.open(io=io) as reader:
            expected = (
                F.linear(inputs, reader.get("weight").read())
                + reader.get("offset").read().square()
            )
        actual = result.model(inputs)
        assert bool(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected)
        assert result.reports[0].unexpected == ()


@pytest.mark.parametrize("quantizer", (None, Quantizer("fp8")))
def test_merged_parameter_updates_are_visible_across_execution_contexts(
    tmp_path, quantizer
):
    from uniserve.model import TextSize
    from uniserve.runtime import ExecutionContext

    first = torch.arange(32).reshape(8, 4).float().sin()
    second = torch.arange(32).reshape(8, 4).float().cos() * 20
    save_file({"gate": first, "up": second}, tmp_path / "model.safetensors")

    def mapping(model):
        return (
            weights.ModuleMapping(
                model,
                "primary",
                lambda reader: tuple(
                    weights.Assignment(branch.weight, reader.get(name))
                    for name, branch in model.projections.items()
                ),
                frozenset(name for name, _ in model.named_parameters()),
            ),
        )

    result = loading.load_model(
        lambda width: MergedColumnParallelLinear(
            width, {"gate": 8, "up": 8}, bias=False
        ),
        4,
        checkpoint=(
            checkpoint.Config().resolve(tmp_path, io=loading.Config()),
        ),
        mapping=mapping,
        device="cpu",
        weights=weights.Config(
            dtype=torch.float32,
            quantization={}
            if quantizer is None
            else {"": QuantizationConfig(quantizer, quantizer)},
        ),
    )
    model = result.model
    source = torch.arange(12).reshape(3, 4).float() / 3

    def encoded(value):
        return (
            value
            if quantizer is None
            else quantizer.quantize(value).dequantize()
        )

    contexts = [ExecutionContext(model, matmul="torch") for _ in range(2)]
    try:
        for context in contexts:
            context.prepare(TextSize(3, 1))
        with contexts[0].activate():
            retained = model(source)
        model.projections["up"].weight.copy_(second * 7)
        for context in contexts:
            with context.activate():
                actual = model(source)
            torch.testing.assert_close(
                actual["gate"], F.linear(encoded(source), encoded(first))
            )
            torch.testing.assert_close(
                actual["up"], F.linear(encoded(source), encoded(second * 7))
            )
        torch.testing.assert_close(
            retained["up"], F.linear(encoded(source), encoded(second))
        )
    finally:
        for context in contexts:
            context.close()


def test_parallel_loading_reads_the_logical_channel_shard(tmp_path):
    full = torch.arange(16).view(4, 4).float()
    save_file({"weight": full}, tmp_path / "model.safetensors")

    def mapping(model):
        return (
            weights.ModuleMapping(
                model,
                "primary",
                lambda reader: (
                    weights.Assignment(model.weight, reader.get("weight")),
                ),
                frozenset({"weight"}),
            ),
        )

    for rank in (0, 1):
        mesh = DeviceMesh(ranks=(1, 0), shape=(2,), axes=("tp",), rank=rank)
        result = loading.load_model(
            lambda width: ColumnParallelLinear(width, width, bias=False),
            4,
            checkpoint=(
                checkpoint.Config().resolve(tmp_path, io=loading.Config()),
            ),
            mapping=mapping,
            device="cpu",
            weights=weights.Config(dtype=torch.float32),
            meshes={"": mesh},
        )
        torch.testing.assert_close(
            result.model(torch.eye(4)),
            full.chunk(2)[1 - rank].T,
            rtol=0,
            atol=0,
        )


def test_prequantized_tied_parameters_keep_source_scales(tmp_path):
    encoded = torch.arange(16).view(4, 4).to(torch.float8_e4m3fn)
    scale = torch.tensor(8.0)
    save_file(
        {"first.weight": encoded, "first.weight_scale": scale.reshape(1)},
        tmp_path / "model.safetensors",
    )

    def mapping(model):
        def assign(reader):
            source = reader.get("first.weight")
            return (
                weights.Assignment(model.first.weight, source),
                weights.Assignment(model.second.weight, source),
            )

        return (
            weights.ModuleMapping(
                model,
                "primary",
                assign,
                frozenset({"first.weight", "second.weight"}),
            ),
        )

    result = loading.load_model(
        Network,
        NetworkConfig(tied=True),
        checkpoint=(
            checkpoint.Config().resolve(tmp_path, io=loading.Config()),
        ),
        mapping=mapping,
        device="cpu",
        weights=weights.Config(
            dtype=torch.float32,
            quantization={
                "": QuantizationConfig(
                    Quantizer("fp8", axis=0), Quantizer("fp8", axis=0)
                )
            },
        ),
    )
    assert result.model.first.weight is result.model.second.weight
    assert isinstance(result.model.first.weight, QuantizedTensor)
    torch.testing.assert_close(
        result.model.first.weight.buffers()["scale"], scale, rtol=0, atol=0
    )
    source = torch.eye(4)

    def activation(value):
        step = value.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
        return (value / step).clamp(-448, 448).to(
            torch.float8_e4m3fn
        ).float() * step

    intermediate = F.linear(activation(source), encoded.float() * scale)
    expected = F.linear(activation(intermediate), encoded.float() * scale)
    torch.testing.assert_close(result.model(source), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="aliases.*conflicting"):
        loading.load_model(
            Network,
            NetworkConfig(tied=True),
            checkpoint=(),
            mapping=mapping,
            device="cpu",
            weights=weights.Config(
                dtype=torch.float32, dtypes={"second": torch.bfloat16}
            ),
        )


def test_fragment_assignments_require_complete_nonoverlapping_values(tmp_path):
    full = torch.arange(16).reshape(4, 4).float()
    save_file({"weight": full}, tmp_path / "model.safetensors")
    sources = (checkpoint.Config().resolve(tmp_path, io=loading.Config()),)

    def mapping(model):
        def assign(reader):
            source = reader.get("weight")
            return tuple(
                weights.Assignment(
                    model.weight,
                    source,
                    (slice(start, start + 2), slice(0, 4)),
                    (slice(start, start + 2), slice(0, 4)),
                )
                for start in (0, 2)
            )

        return (
            weights.ModuleMapping(
                model, "primary", assign, frozenset({"weight"})
            ),
        )

    model = Linear(4, 4, bias=False)
    loading.load_weights(
        model,
        sources,
        mapping=mapping,
        device="cpu",
        weights=weights.Config(dtype=torch.float32),
    )
    torch.testing.assert_close(model(torch.eye(4)), full.T, rtol=0, atol=0)

    def incomplete(model):
        component = mapping(model)[0]
        return (
            weights.ModuleMapping(
                model,
                "primary",
                lambda reader: component.map_weights(reader)[:1],
                frozenset({"weight"}),
            ),
        )

    with pytest.raises(RuntimeError, match="incomplete=.*weight"):
        loading.load_weights(model, sources, mapping=incomplete, device="cpu")


def test_longest_prefix_can_exclude_quantization(tmp_path):
    save_file(
        {"first": torch.eye(4), "second": torch.eye(4)},
        tmp_path / "model.safetensors",
    )
    quantization = QuantizationConfig(
        Quantizer("fp8"), Quantizer("fp8", axis=0)
    )
    result = loading.load_model(
        Network,
        NetworkConfig(),
        checkpoint=(
            checkpoint.Config().resolve(tmp_path, io=loading.Config()),
        ),
        mapping=_mapping,
        device="cpu",
        weights=weights.Config(
            dtype=torch.float32, quantization={"": quantization, "second": None}
        ),
    )
    source = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    torch.testing.assert_close(result.model(source), source, rtol=0, atol=0)
    assert isinstance(result.model.first.weight, QuantizedTensor)
    assert not isinstance(result.model.second.weight, QuantizedTensor)


def test_fp8_fragments_keep_independent_source_scale_domains(tmp_path):
    save_file(
        {
            "first": torch.ones(2, 4, dtype=torch.float8_e4m3fn),
            "second": torch.ones(2, 4, dtype=torch.float8_e4m3fn),
            "first_scale": torch.tensor(0.5),
            "second_scale": torch.tensor(8.0),
        },
        tmp_path / "model.safetensors",
    )

    def mapping(model):
        def assign(reader):
            return tuple(
                weights.Assignment(
                    model.weight,
                    checkpoint.FP8Weight(
                        reader.get(name),
                        reader.get(name + "_scale"),
                        None,
                        torch.float32,
                    ),
                    target_slice=(slice(index * 2, index * 2 + 2), slice(0, 4)),
                )
                for index, name in enumerate(("first", "second"))
            )

        return (
            weights.ModuleMapping(
                model, "primary", assign, frozenset({"weight"})
            ),
        )

    model = Linear(4, 4, bias=False)
    loading.load_weights(
        model,
        (checkpoint.Config().resolve(tmp_path, io=loading.Config()),),
        mapping=mapping,
        device="cpu",
        weights=weights.Config(dtype=torch.float32),
    )
    torch.testing.assert_close(
        model(torch.eye(4)),
        torch.tensor([[0.5, 0.5, 8.0, 8.0]]).expand(4, 4),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        model.weight.buffers()["scale"],
        torch.tensor([[0.5], [0.5], [8.0], [8.0]]),
        rtol=0,
        atol=0,
    )


def test_component_selection_keeps_nonresident_modules_on_meta(tmp_path):
    save_file(
        {"first": torch.eye(4), "second": torch.eye(4) * 2},
        tmp_path / "model.safetensors",
    )

    def mapping(model):
        def component(name):
            layer = getattr(model, name)
            return weights.ModuleMapping(
                layer,
                "primary",
                lambda reader: (
                    weights.Assignment(layer.weight, reader.get(name)),
                ),
                frozenset({"weight"}),
            )

        return component("first"), component("second")

    # A remote component retains its declared global dimensions. Its weights
    # must not consume local storage when every component was requested.
    result = loading.load_model(
        Network,
        NetworkConfig(),
        checkpoint=(
            checkpoint.Config().resolve(tmp_path, io=loading.Config()),
        ),
        mapping=mapping,
        device="cpu",
        weights=weights.Config(dtype=torch.float32),
        meshes={
            "second": DeviceMesh(ranks=(1,), shape=(1,), axes=("tp",), rank=0)
        },
    )
    torch.testing.assert_close(
        result.model.first(torch.eye(4)), torch.eye(4), rtol=0, atol=0
    )
    assert result.model.second.weight.is_meta
    assert result.model.second.weight.shape == (4, 4)
