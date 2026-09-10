"""Component source isolation and complete serialized-state loading through the public loader."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from tests.python.fixtures.model_execution import tensor_parallel_bindings
from uniserve_worker.bootstrap.catalog import CatalogEntry
from uniserve_worker.config import WorkerConfig
from uniserve_worker.loader import LoadConfig, LoadFormat, LoadRequest, get_model_loader
from uniserve_worker.loader.component import CheckpointComponent, ModelConstruction
from uniserve_worker.loader.source import WeightSourceConfig
from uniserve_worker.models.runtime import ExecutionModel
from uniserve_worker.nn.mesh import Communicator

pytestmark = pytest.mark.integration


class NormalizedProjection(nn.Linear):
    def __init__(self):
        super().__init__(2, 2, bias=False)
        self.register_buffer("scale", torch.empty(2))

    def forward(self, value):
        return super().forward(value) * self.scale


class ProjectionPair(ExecutionModel):
    """A two-source graph with repeated weight names and a serialized normalization buffer."""

    def __init__(self, encoder_device=None, decoder_device=None):
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        self.decoder = NormalizedProjection()
        self.encoder_device = encoder_device
        self.decoder_device = decoder_device

    @classmethod
    def build_checkpoint(cls, config, context):
        model = cls(config.get("encoder_device"), config.get("decoder_device"))
        return ModelConstruction(model.checkpoint_components(), lambda: model, config)

    def checkpoint_components(self):
        return (
            CheckpointComponent(
                self.encoder,
                source="encode",
                module_devices=(
                    ()
                    if self.encoder_device is None
                    else (("", torch.device(self.encoder_device)),)
                ),
            ),
            CheckpointComponent(
                self.decoder,
                source="decode",
                persistent_buffers=True,
                module_devices=(
                    ()
                    if self.decoder_device is None
                    else (("", torch.device(self.decoder_device)),)
                ),
            ),
        )

    def forward(self, value):
        target = value.device
        encoded = self.encoder(value.to(self.encoder.weight.device))
        return self.decoder(encoded.to(self.decoder.weight.device)).to(target)


@pytest.fixture
def component_checkpoint(tmp_path):
    entry = CatalogEntry(
        "ProjectionPair",
        ProjectionPair,
        sources=(WeightSourceConfig("encode", "encoder"), WeightSourceConfig("decode", "decoder")),
    )
    weights = {
        "encoder": {"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        "decoder": {
            "weight": torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
            "scale": torch.tensor([0.5, 2.0]),
        },
    }
    manifest = {}
    for directory, state in weights.items():
        folder = tmp_path / directory
        folder.mkdir()
        path = folder / "diffusion_pytorch_model.safetensors"
        save_file(state, path)
        (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {name: path.name for name in state},
                }
            )
        )
        manifest[f"{directory}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    checksum = tmp_path / "sha256.json"
    checksum.write_text(json.dumps({"files": manifest}))
    request = LoadRequest(
        model_path=str(tmp_path),
        execution=WorkerConfig(model_dtype="float32"),
        bindings=tensor_parallel_bindings(),
        load=LoadConfig(checksum_manifest=str(checksum)),
    )
    return entry, request, tmp_path


@pytest.mark.parametrize(
    "load_format", [LoadFormat.AUTO, LoadFormat.SAFETENSORS, LoadFormat.LAYERED]
)
def test_component_directories_preserve_namespaces_and_persistent_buffers(
    component_checkpoint, load_format
):
    entry, request, root = component_checkpoint
    request = replace(
        request,
        load=LoadConfig(
            load_format=load_format,
            checksum_manifest=request.load.checksum_manifest,
        ),
    )
    loaded = get_model_loader(load_format).load(entry, {}, request, root=root, repository_id=None)
    # [1, 2] -> [5, 11] -> [10, 33] -> [5, 66].
    torch.testing.assert_close(
        loaded.model(torch.tensor([[1.0, 2.0]])), torch.tensor([[5.0, 66.0]])
    )


def test_checksum_covers_each_component_source(component_checkpoint):
    entry, request, root = component_checkpoint
    path = root / "decoder" / "diffusion_pytorch_model.safetensors"
    save_file({"weight": torch.eye(2), "scale": torch.ones(2)}, path)
    with pytest.raises(ValueError, match="checksum mismatch.*decoder/"):
        get_model_loader(request.load.load_format).load(
            entry, {}, request, root=root, repository_id=None
        )


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices are required")
@pytest.mark.parametrize("load_format", [LoadFormat.AUTO, LoadFormat.LAYERED])
def test_components_execute_on_their_declared_devices(component_checkpoint, load_format):
    entry, request, root = component_checkpoint
    request = replace(request, load=LoadConfig(load_format=load_format))
    loaded = get_model_loader(load_format).load(
        entry,
        {"encoder_device": "cuda:0", "decoder_device": "cuda:1"},
        request,
        root=root,
        repository_id=None,
    )
    inputs = torch.tensor([[1.0, 2.0]])
    torch.testing.assert_close(loaded.model(inputs), torch.tensor([[5.0, 66.0]]), rtol=0, atol=0)
    # Each public neural component accepts tensors on its configured device.
    torch.testing.assert_close(
        loaded.model.encoder(inputs.cuda(0)), torch.tensor([[5.0, 11.0]], device="cuda:0")
    )
    torch.testing.assert_close(
        loaded.model.decoder(inputs.cuda(1)), torch.tensor([[1.0, 12.0]], device="cuda:1")
    )


def test_missing_serialized_buffer_rejects_incomplete_component(component_checkpoint):
    entry, request, root = component_checkpoint
    path = root / "decoder" / "diffusion_pytorch_model.safetensors"
    save_file({"weight": torch.eye(2)}, path)
    request = replace(request, load=LoadConfig())
    with pytest.raises(RuntimeError, match="missing=1.*scale"):
        get_model_loader(request.load.load_format).load(
            entry, {}, request, root=root, repository_id=None
        )


def test_component_resolution_requires_only_resident_sources(component_checkpoint):
    from uniserve_worker.loader.source import resolve_weight_sources
    from uniserve_worker.nn.mesh import DeviceMesh, EntryBindings
    from uniserve_worker.nn.parallel import EntryConfig, ParallelConfig

    entry, request, root = component_checkpoint
    bindings = EntryBindings(
        {"encode": EntryConfig((0,)), "decode": EntryConfig((1,))},
        {"encode": DeviceMesh((0,), 0, ParallelConfig(), torch.device("cpu"))},
        Communicator(ranks=(0, 1), rank=0),
    )
    request = replace(request, bindings=bindings)
    # A rank need not have the checkpoint bytes for another rank's component.
    (root / "decoder" / "diffusion_pytorch_model.safetensors").unlink()
    sources = resolve_weight_sources(
        request,
        sources=tuple(replace(source, entry=source.name) for source in entry.sources),
        sidecars=entry.sidecars,
        root=root,
        repository_id=None,
    )
    assert len(sources) == 1
    assert sources[0].source_name == "encode"
    assert sources[0].preview_shape("weight") == (2, 2)


class PackedProjection(ExecutionModel):
    """A checkpoint with two concatenated output branches sharded across ranks."""

    def __init__(self, layer_config):
        super().__init__()
        from uniserve_worker.nn.linear import MergedColumnParallelLinear

        self.projection = MergedColumnParallelLinear(
            2, (4, 4), layer_config=layer_config, bias=False
        )

    @classmethod
    def build_checkpoint(cls, config, context):
        model = cls(context.packed_decoder_layers("model"))
        return ModelConstruction((CheckpointComponent(model),), lambda: model, config)

    def forward(self, value):
        return self.projection(value)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("quantized_checkpoint", [False, True])
@pytest.mark.parametrize("checkpoint_layout", ["global", "rank_local"])
def test_fp8_merged_checkpoint_preserves_each_rank_branch_and_scale(
    tmp_path, rank, quantized_checkpoint, checkpoint_layout
):
    # E4M3 extrema and power-of-two row scales make the projection exact.
    dense = torch.tensor(
        [[448, 0], [0, 448], [224, 0], [0, 224], [112, 0], [0, 112], [56, 0], [0, 56]],
        dtype=torch.bfloat16,
    )
    if quantized_checkpoint:
        scales = dense.float().abs().amax(dim=1, keepdim=True) / 448
        state = {
            "projection.weight": (dense.float() / scales).to(torch.float8_e4m3fn),
            "projection.weight_scale": scales,
        }
    else:
        state = {"projection.weight": dense}
    output_rows = [2 * rank, 2 * rank + 1, 4 + 2 * rank, 5 + 2 * rank]
    if checkpoint_layout == "rank_local":
        state = {name: tensor[output_rows] for name, tensor in state.items()}
    save_file(state, tmp_path / "model.safetensors")
    request = LoadRequest(
        model_path=str(tmp_path),
        execution=WorkerConfig(model_dtype="bfloat16"),
        bindings=tensor_parallel_bindings(Communicator(ranks=(0, 1), rank=rank)),
        quantization_config={"quant_method": "fp8"},
    )
    loaded = get_model_loader(LoadFormat.AUTO).load(
        CatalogEntry("PackedProjection", PackedProjection),
        {},
        request,
        root=tmp_path,
        repository_id=None,
    )
    with torch.inference_mode():
        actual = loaded.model(torch.eye(2, dtype=torch.bfloat16))
    torch.testing.assert_close(actual, dense[output_rows].T, rtol=0, atol=0)
