"""Independent checkpoint modules retain their values, buffers and placement."""

import hashlib
import json
from dataclasses import dataclass, replace

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
            json.dumps({"weight_map": {name: path.name for name in values}})
        )
        hashes[path.relative_to(tmp_path).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    manifest = tmp_path / "sha256.json"
    manifest.write_text(json.dumps({"files": hashes}))
    return tmp_path, loading.Config(checksum_manifest=manifest)


def _load(root, io, *, device="cpu", devices=None, modules=None, config=Config()):
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
        {"weight": torch.eye(2), "scale": torch.ones(2)}, root / "decoder" / "model.safetensors"
    )
    with pytest.raises(ValueError, match="checksum mismatch.*decoder/"):
        _load(root, io)


def test_missing_serialized_buffer_rejects_incomplete_component(component_checkpoint):
    root, _ = component_checkpoint
    save_file({"weight": torch.eye(2)}, root / "decoder" / "model.safetensors")
    with pytest.raises(KeyError, match="scale"):
        _load(root, loading.Config())


def test_component_selection_needs_only_resident_source_files(component_checkpoint):
    root, io = component_checkpoint
    (root / "decoder" / "model.safetensors").unlink()
    result = _load(root, io, modules=frozenset({"encoder"}))
    torch.testing.assert_close(
        result.model.encoder(torch.tensor([[1.0, 2.0]])), torch.tensor([[5.0, 11.0]])
    )
    assert result.model.decoder.weight.is_meta


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["eager", "layered"])
def test_each_component_accepts_values_on_its_declared_device(component_checkpoint, mode):
    root, io = component_checkpoint
    result = _load(root, replace(io, mode=mode), device="cuda:0", devices={"decoder": "cuda:1"})
    encoded = result.model.encoder(torch.tensor([[1.0, 2.0]], device="cuda:0"))
    torch.testing.assert_close(encoded, torch.tensor([[5.0, 11.0]], device="cuda:0"))
    decoded = result.model.decoder(encoded.to("cuda:1"))
    torch.testing.assert_close(decoded, torch.tensor([[5.0, 66.0]], device="cuda:1"))


def test_shared_parameter_rejects_conflicting_placement_before_materialization(
    component_checkpoint,
):
    root, io = component_checkpoint
    with pytest.raises(ValueError, match="shared parameter aliases have conflicting"):
        _load(root, io, device="cpu", devices={"decoder": "cuda:1"}, config=Config(tied=True))


@pytest.mark.gpu
@pytest.mark.parametrize("explicit_stream", (False, True))
@torch.inference_mode()
def test_contexts_deliver_cross_device_components_with_independent_graphs(
    component_checkpoint, explicit_stream
):
    from uniserve.model import TextSize
    from uniserve.runtime import CUDAGraph, ExecutionContext

    root, io = component_checkpoint
    model = _load(root, io, device="cuda:0", devices={"decoder": "cuda:1"}).model
    first = torch.tensor([[1.0, 2.0]], device="cuda:0")
    second = torch.tensor([[3.0, 4.0]], device="cuda:0")
    streams = tuple(
        torch.cuda.Stream(device="cuda:0")
        if explicit_stream
        else torch.cuda.current_stream("cuda:0")
        for _ in range(2)
    )
    # MemPool ownership is associated with the current device at construction.
    # Each graph keeps its secondary-device allocation domain alive for replay.
    with torch.cuda.device("cuda:1"):
        pools = tuple({torch.device("cuda:1"): torch.cuda.MemPool()} for _ in range(2))
    with ExecutionContext(model, stream=streams[0] if explicit_stream else None) as a:
        a.prepare(TextSize(1, 1))
        streams[0].wait_stream(torch.cuda.default_stream("cuda:0"))
        torch.testing.assert_close(model(first), first.new_tensor([[5.0, 66.0]]))
        with pytest.raises(ValueError, match="contraction dimension"):
            model.decoder(first.new_zeros((1, 3)))
        torch.testing.assert_close(model(first), first.new_tensor([[5.0, 66.0]]))
        with CUDAGraph(context=a, pools=pools[0]) as ga:
            ga.capture(lambda: model(first))
            with ExecutionContext(model, stream=streams[1] if explicit_stream else None) as b:
                b.prepare(TextSize(1, 1))
                streams[1].wait_stream(torch.cuda.default_stream("cuda:0"))
                torch.testing.assert_close(model(second), second.new_tensor([[11.0, 150.0]]))
                with CUDAGraph(context=b, pools=pools[1]) as gb:
                    gb.capture(lambda: model(second))
                    with torch.cuda.stream(streams[0]):
                        first.copy_(first.new_tensor([[2.0, 3.0]]))
                    actual_a = ga.replay()
                    actual_b = gb.replay()
                    for stream in streams:
                        stream.synchronize()
                    torch.testing.assert_close(actual_a, first.new_tensor([[8.0, 108.0]]))
                    torch.testing.assert_close(actual_b, second.new_tensor([[11.0, 150.0]]))
            # Returning to the first context restores its stream bindings.
            out = torch.empty_like(first)
            assert model(first, out=out) is out
            torch.testing.assert_close(out, first.new_tensor([[8.0, 108.0]]))
        streams[0].synchronize()
