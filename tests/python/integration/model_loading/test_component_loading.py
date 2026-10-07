"""Independent checkpoint modules retain their values, buffers and placement."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from threading import Barrier

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from uniserve import loading
from uniserve.loading import checkpoint, weights
from uniserve.nn import Linear

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class Config:
    width: int = 2
    tied: bool = False


class NormalizedProjection(Linear):
    def __init__(self, width):
        super().__init__(width, width, bias=False)
        self.register_buffer("scale", torch.empty(width))

    def forward(self, value, *, out=None):
        result = super().forward(value) * self.scale
        return result if out is None else out.copy_(result)


class ProjectionPair(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Linear(config.width, config.width, bias=False)
        self.decoder = NormalizedProjection(config.width)
        if config.tied:
            self.decoder.weight = self.encoder.weight

    def forward(self, value, *, out=None):
        return self.decoder(self.encoder(value), out=out)


def _mapping(model):
    def assign(module, reader):
        return (weights.Assignment(module.weight, reader.get("weight")),)

    def normalize(reader):
        value = reader.get("scale").read()
        if value.shape != model.decoder.scale.shape:
            raise ValueError("normalization scales must cover output channels")
        model.decoder.scale = value

    return (
        weights.ModuleMapping(
            model.encoder,
            "encode",
            lambda reader: assign(model.encoder, reader),
            frozenset({"weight"}),
        ),
        weights.ModuleMapping(
            model.decoder,
            "decode",
            lambda reader: assign(model.decoder, reader),
            frozenset({"weight"}),
            post_load=normalize,
        ),
    )


@pytest.fixture
def component_checkpoint(tmp_path):
    tensors = {
        "encoder": {"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        "decoder": {
            "weight": torch.diag(torch.tensor([2.0, 3.0])),
            "scale": torch.tensor([0.5, 2.0]),
        },
    }
    hashes = {}
    for name, values in tensors.items():
        directory = tmp_path / name
        directory.mkdir()
        path = directory / "model.safetensors"
        save_file(values, path)
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": dict.fromkeys(values, path.name)})
        )
        hashes[path.relative_to(tmp_path).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    manifest = tmp_path / "sha256.json"
    manifest.write_text(json.dumps({"files": hashes}))
    return tmp_path, loading.Config(checksum_manifest=manifest)


def _load(
    root, io, *, device="cpu", devices=None, modules=None, config=Config()
):
    declarations = (
        checkpoint.Config("encode", directory="encoder", module_path="encoder"),
        checkpoint.Config("decode", directory="decoder", module_path="decoder"),
    )
    sources = tuple(
        source.resolve(root, io=io)
        for source in declarations
        if modules is None or source.module_path in modules
    )
    return loading.load_model(
        ProjectionPair,
        config,
        checkpoint=sources,
        mapping=_mapping,
        device=device,
        devices=devices,
        modules=modules,
        weights=weights.Config(dtype=torch.float32),
        io=io,
    )


@pytest.mark.parametrize("mode", ["eager", "layered"])
def test_component_directories_preserve_namespaces_and_persistent_buffers(
    component_checkpoint, mode
):
    root, io = component_checkpoint
    result = _load(root, replace(io, mode=mode))
    # [1, 2] -> [5, 11] -> [10, 33] -> [5, 66].
    torch.testing.assert_close(
        result.model(torch.tensor([[1.0, 2.0]])), torch.tensor([[5.0, 66.0]])
    )


def test_checksum_covers_each_component_source(component_checkpoint):
    root, io = component_checkpoint
    save_file(
        {"weight": torch.eye(2), "scale": torch.ones(2)},
        root / "decoder" / "model.safetensors",
    )
    with pytest.raises(ValueError, match="checksum mismatch.*decoder/"):
        _load(root, io)


def test_missing_serialized_buffer_rejects_incomplete_component(
    component_checkpoint,
):
    root, _ = component_checkpoint
    save_file({"weight": torch.eye(2)}, root / "decoder" / "model.safetensors")
    with pytest.raises(KeyError, match="scale"):
        _load(root, loading.Config())


def test_component_selection_needs_only_resident_source_files(
    component_checkpoint,
):
    root, io = component_checkpoint
    (root / "decoder" / "model.safetensors").unlink()
    result = _load(root, io, modules=frozenset({"encoder"}))
    torch.testing.assert_close(
        result.model.encoder(torch.tensor([[1.0, 2.0]])),
        torch.tensor([[5.0, 11.0]]),
    )
    assert result.model.decoder.weight.is_meta


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["eager", "layered"])
def test_each_component_accepts_values_on_its_declared_device(
    component_checkpoint, mode
):
    root, io = component_checkpoint
    result = _load(
        root,
        replace(io, mode=mode),
        device="cuda:0",
        devices={"decoder": "cuda:1"},
    )
    encoded = result.model.encoder(torch.tensor([[1.0, 2.0]], device="cuda:0"))
    torch.testing.assert_close(
        encoded, torch.tensor([[5.0, 11.0]], device="cuda:0")
    )
    decoded = result.model.decoder(encoded.to("cuda:1"))
    torch.testing.assert_close(
        decoded, torch.tensor([[5.0, 66.0]], device="cuda:1")
    )


def test_shared_parameter_rejects_conflicting_placement_before_materialization(
    component_checkpoint,
):
    root, io = component_checkpoint
    with pytest.raises(
        ValueError, match="shared parameter aliases have conflicting"
    ):
        _load(
            root,
            io,
            device="cpu",
            devices={"decoder": "cuda:1"},
            config=Config(tied=True),
        )


@pytest.mark.gpu
@pytest.mark.parametrize("stream_kind", ("default", "external", "partitioned"))
@torch.inference_mode()
def test_contexts_deliver_cross_device_components_with_independent_graphs(
    component_checkpoint, stream_kind
):
    from uniserve.model import TextSize
    from uniserve.runtime import (
        CUDAGraph,
        CUDAStream,
        ExecutionContext,
        partition_streams,
    )

    root, io = component_checkpoint
    model = _load(
        root, io, device="cuda:0", devices={"decoder": "cuda:1"}
    ).model
    first = torch.tensor([[1.0, 2.0]], device="cuda:0")
    second = torch.tensor([[3.0, 4.0]], device="cuda:0")
    if stream_kind == "partitioned":
        owners = partition_streams(torch.device("cuda:0"), (64, 64))
    else:
        owners = tuple(
            CUDAStream.external(torch.cuda.Stream(device="cuda:0"))
            if stream_kind == "external"
            else None
            for _ in range(2)
        )
    streams = tuple(
        owner.stream
        if owner is not None
        else torch.cuda.current_stream("cuda:0")
        for owner in owners
    )
    # MemPool ownership is associated with the current device at construction.
    # Each graph keeps its secondary-device allocation domain alive for replay.
    with torch.cuda.device("cuda:1"):
        pools = tuple(
            {torch.device("cuda:1"): torch.cuda.MemPool()} for _ in range(2)
        )
    with ExecutionContext(model, stream=owners[0]) as a:
        a.prepare(TextSize(1, 1))
        streams[0].wait_stream(torch.cuda.default_stream("cuda:0"))
        torch.testing.assert_close(
            model(first), first.new_tensor([[5.0, 66.0]])
        )
        with pytest.raises(ValueError, match="contraction dimension"):
            model.decoder(first.new_zeros((1, 3)))
        torch.testing.assert_close(
            model(first), first.new_tensor([[5.0, 66.0]])
        )
        with CUDAGraph(context=a, pools=pools[0]) as ga:
            ga.capture(lambda: model(first))
            with ExecutionContext(model, stream=owners[1]) as b:
                b.prepare(TextSize(1, 1))
                streams[1].wait_stream(torch.cuda.default_stream("cuda:0"))
                torch.testing.assert_close(
                    model(second), second.new_tensor([[11.0, 150.0]])
                )
                with CUDAGraph(context=b, pools=pools[1]) as gb:
                    gb.capture(lambda: model(second))
                    with torch.cuda.stream(streams[0]):
                        first.copy_(first.new_tensor([[2.0, 3.0]]))
                    actual_a = ga.replay()
                    actual_b = gb.replay()
                    for stream in streams:
                        stream.synchronize()
                    torch.testing.assert_close(
                        actual_a, first.new_tensor([[8.0, 108.0]])
                    )
                    torch.testing.assert_close(
                        actual_b, second.new_tensor([[11.0, 150.0]])
                    )
            # Returning to the first context restores its stream bindings.
            out = torch.empty_like(first)
            assert model(first, out=out) is out
            torch.testing.assert_close(out, first.new_tensor([[8.0, 108.0]]))
        streams[0].synchronize()
    for owner in owners:
        if owner is not None:
            owner.close()


@pytest.mark.gpu
def test_shared_components_deliver_independently_on_concurrent_contexts(
    component_checkpoint,
):
    from uniserve.model import TextSize
    from uniserve.runtime import CUDAStream, ExecutionContext

    root, io = component_checkpoint
    model = _load(
        root, io, device="cuda:0", devices={"decoder": "cuda:1"}
    ).model
    barrier = Barrier(2, timeout=30)

    @torch.inference_mode()
    def execute(index):
        with torch.cuda.device(0):
            value = torch.tensor([[1.0, 2.0]], device="cuda:0") + index * 2
            with (
                CUDAStream.external(
                    torch.cuda.Stream(device="cuda:0")
                ) as stream,
                ExecutionContext(model, stream=stream) as context,
            ):
                context.prepare(TextSize(1, 1))
                stream.stream.wait_stream(torch.cuda.default_stream("cuda:0"))
                # Both contexts use the same numerical modules concurrently.
                # One failing call must leave the other's delivery intact.
                barrier.wait()
                if index == 0:
                    with pytest.raises(
                        ValueError, match="contraction dimension"
                    ):
                        model.decoder(value.new_zeros((1, 3)))
                result = model(value).cpu()
                barrier.wait()
                return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(execute, range(2)))
    for result, expected in zip(
        results, ([[5.0, 66.0]], [[11.0, 150.0]]), strict=True
    ):
        torch.testing.assert_close(result, torch.tensor(expected))


@pytest.mark.gpu
@pytest.mark.timeout(30)
@torch.inference_mode()
def test_cpu_context_unwinds_a_pending_gpu_call_without_waiting(
    component_checkpoint,
):
    from tests.python.fixtures.cuda_stream import blocked_stream
    from uniserve.model import TextSize
    from uniserve.runtime import ExecutionContext

    root, io = component_checkpoint
    model = _load(root, io, device="cpu", devices={"decoder": "cuda:1"}).model
    value = torch.tensor([[1.0, 2.0]], device="cuda:1")
    invalid = value.new_zeros((1, 3))
    context = ExecutionContext(model)
    context.prepare(TextSize(1, 1))
    with context.activate():
        torch.testing.assert_close(
            model.decoder(value), value.new_tensor([[1.0, 12.0]])
        )

    # The module's delivery stream waits for this producer. Exceptional context
    # cleanup must return while that work is still pending on the device.
    with blocked_stream("cuda:0") as producer, torch.cuda.stream(producer):
        with pytest.raises(ValueError, match="contraction dimension"):
            with context:
                model.decoder(invalid)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("source", "destination"),
    (("cuda:0", "cpu"), ("cuda:0", "cuda:1"), ("cpu", "cuda:1")),
)
@torch.inference_mode()
def test_component_delivery_preserves_nested_inputs_and_named_output_storage(
    source,
    destination,
):
    from uniserve.model import TextSize
    from uniserve.runtime import ExecutionContext

    @dataclass(frozen=True)
    class Residual:
        hidden: torch.Tensor
        skip: tuple[torch.Tensor, float]

    class Component(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.tensor(3.0, device=destination))

        def forward(self, value, *, out=None):
            result = {
                "hidden": value.hidden * self.scale,
                "skip": value.skip[0] + value.skip[1] * self.scale,
            }
            if out is None:
                return result
            for name, tensor in result.items():
                out[name].copy_(tensor)
            return out

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("offset", torch.tensor(2.0, device=source))
            self.component = Component()

        def forward(self, value, *, out=None):
            value = replace(value, hidden=value.hidden + self.offset)
            return self.component(value, out=out)

    value = Residual(
        torch.tensor([[1.0, 2.0]], device=source),
        (torch.tensor([[4.0, 5.0]], device=source), 2.0),
    )
    model = Model()
    with torch.cuda.device(0), ExecutionContext(model) as context:
        context.prepare(TextSize(1, 1))
        expected = {
            "hidden": value.hidden.new_tensor([[9.0, 12.0]]),
            "skip": value.hidden.new_tensor([[10.0, 11.0]]),
        }
        for name, result in model(value).items():
            torch.testing.assert_close(result, expected[name])

        output = {
            name: torch.empty_like(tensor) for name, tensor in expected.items()
        }
        assert model(value, out=output) is output
        for name, result in output.items():
            torch.testing.assert_close(result, expected[name])
        torch.testing.assert_close(
            value.hidden, value.hidden.new_tensor([[1.0, 2.0]])
        )
