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
from uniserve_worker.loader.component import CheckpointComponent
from uniserve_worker.loader.source import WeightSourceConfig
from uniserve_worker.modeling.batch import EncodeBatch, TensorOutput
from uniserve_worker.modeling.components import Call, CallSpec, ComponentSpec
from uniserve_worker.modeling.encoder import EncoderMixin
from uniserve_worker.modeling.model import Model
from uniserve_worker.modeling.tensors import ExpertRoute
from uniserve_worker.nn.branch import branch
from uniserve_worker.nn.mesh import Communicator

pytestmark = pytest.mark.integration


class NormalizedProjection(nn.Linear):
    def __init__(self):
        super().__init__(2, 2, bias=False)
        self.register_buffer("scale", torch.empty(2))

    def forward(self, value):
        return super().forward(value) * self.scale


class ProjectionPair(EncoderMixin, Model):
    """A two-source graph with repeated weight names and a serialized normalization buffer."""

    encoder_kinds = frozenset({"conditioning"})

    @classmethod
    def components(cls, config):
        return (ComponentSpec("model", (CallSpec(Call.ENCODE_CONDITIONING),)),)

    def __init__(self, config, context):
        super().__init__(config, context)
        self.encoder = nn.Linear(2, 2, bias=False)
        self.decoder = branch(NormalizedProjection(), ExpertRoute.FLOW)

    def checkpoint_components(self):
        return (
            CheckpointComponent(
                self.encoder,
                source="encode",
            ),
            CheckpointComponent(
                self.decoder,
                source="decode",
            ),
        )

    def encode(self, kind, batch, *, constants, scratch):
        if kind != "conditioning":
            raise ValueError("projection pair encodes conditioning features")
        outputs = []
        for value in batch.values:
            outputs.append(self.decoder(self.encoder(value)))
        return TensorOutput({"conditioning": tuple(outputs)})


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
        loaded.model.encode(
            "conditioning", EncodeBatch((torch.tensor([[1.0, 2.0]]),)), constants={}, scratch={}
        ).values["conditioning"][0],
        torch.tensor([[5.0, 66.0]]),
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
    request = replace(
        request,
        execution=replace(request.execution, device="cuda:0", generation_device="cuda:1"),
        load=LoadConfig(load_format=load_format),
    )
    loaded = get_model_loader(load_format).load(
        entry,
        {},
        request,
        root=root,
        repository_id=None,
    )
    inputs = torch.tensor([[1.0, 2.0]], device="cuda:0")
    result = loaded.model.encode("conditioning", EncodeBatch((inputs,)), constants={}, scratch={})
    torch.testing.assert_close(
        result.values["conditioning"][0],
        torch.tensor([[5.0, 66.0]], device="cuda:0"),
        rtol=0,
        atol=0,
    )
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
    from uniserve_worker.execution.model_entry import ModelEntry
    from uniserve_worker.loader.source import resolve_weight_sources
    from uniserve_worker.nn.mesh import DeviceMesh
    from uniserve_worker.nn.parallel import ComponentConfig, ParallelConfig

    entry, request, root = component_checkpoint
    group = Communicator(ranks=(0, 1), rank=0)
    bindings = {
        "encode": ModelEntry(
            "encode",
            ComponentConfig((0,)),
            group,
            DeviceMesh((0,), 0, ParallelConfig(), group.device),
            group.device,
        ),
        "decode": ModelEntry("decode", ComponentConfig((1,)), group, None, group.device),
    }
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


class PackedProjection(EncoderMixin, Model):
    """A checkpoint with two concatenated output branches sharded across ranks."""

    encoder_kinds = frozenset({"conditioning"})

    @classmethod
    def components(cls, config):
        return (ComponentSpec("model", (CallSpec(Call.ENCODE_CONDITIONING, groups=("tp",)),)),)

    def __init__(self, config, context):
        super().__init__()
        from uniserve_worker.nn.linear import MergedColumnParallelLinear

        self.projection = MergedColumnParallelLinear(
            2, (4, 4), layer_config=context.layers["model"], bias=False
        )

    def checkpoint_components(self):
        return (CheckpointComponent(self),)

    def encode(self, kind, batch, *, constants, scratch):
        if kind != "conditioning":
            raise ValueError("packed projection encodes conditioning features")
        return TensorOutput(
            {"conditioning": tuple(self.projection(value) for value in batch.values)}
        )


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
        actual = loaded.model.encode(
            "conditioning",
            EncodeBatch((torch.eye(2, dtype=torch.bfloat16),)),
            constants={},
            scratch={},
        ).values["conditioning"][0]
    torch.testing.assert_close(actual, dense[output_rows].T, rtol=0, atol=0)


class SharedBranches(ProjectionPair):
    """Two mathematical paths sharing one learned linear transformation."""

    def __init__(self, config, context):
        Model.__init__(self, config, context)
        shared = nn.Linear(2, 2, bias=False)
        self.encoder = branch(nn.Sequential(shared), ExpertRoute.TEXT)
        self.decoder = branch(nn.Sequential(shared), ExpertRoute.FLOW)

    def checkpoint_components(self):
        return (CheckpointComponent(self),)


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_shared_branch_parameter_rejects_conflicting_placement(tmp_path):
    save_file({"encoder.0.weight": torch.eye(2)}, tmp_path / "model.safetensors")
    request = LoadRequest(
        model_path=str(tmp_path),
        execution=WorkerConfig(device="cuda:0", generation_device="cuda:1", model_dtype="float32"),
        bindings=tensor_parallel_bindings(),
        load=LoadConfig(load_format=LoadFormat.LAYERED),
    )
    with pytest.raises(ValueError, match="shared parameter cannot belong to different devices"):
        get_model_loader(request.load.load_format).load(
            CatalogEntry("SharedBranches", SharedBranches),
            {},
            request,
            root=tmp_path,
            repository_id=None,
        )


def test_synthetic_loading_initializes_serialized_buffers(component_checkpoint):
    entry, request, root = component_checkpoint
    request = replace(request, load=LoadConfig(load_format=LoadFormat.DUMMY))
    results = []
    for _ in range(2):
        loaded = get_model_loader(request.load.load_format).load(
            entry, {}, request, root=root, repository_id=None
        )
        result = loaded.model.encode(
            "conditioning", EncodeBatch((torch.tensor([[1.0, 2.0]]),)), constants={}, scratch={}
        ).values["conditioning"][0]
        assert torch.isfinite(result).all() and torch.count_nonzero(result) == 2
        results.append(result)
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
