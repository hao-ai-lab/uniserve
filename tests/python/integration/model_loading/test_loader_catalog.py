"""Catalog resolution and checkpoint loading conformance."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from uniserve_worker.bootstrap.catalog import (
    CatalogEntry,
    CheckpointFormat,
    resolve_catalog_entry,
)
from uniserve_worker.bootstrap.model_loader import (
    WorkerModelLoadRequest,
    load_worker_model,
)
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.foundation.runtime_config import ExecutionConfig
from uniserve_worker.loader import Loader
from uniserve_worker.loader.schema import (
    ModelLoadScope,
    Quantize,
    Rename,
    Reshape,
    Shard,
    Sidecar,
    Slice,
    Split,
    Stack,
    Tie,
    Transpose,
    WeightSpec,
)
from uniserve_worker.loader.transformers import (
    dtype_from_name,
)
from uniserve_worker.loader.weight_utils import transform_weight
from uniserve_worker.models.qwen3 import Qwen3ForCausalLM
from uniserve_worker.nn.layer import LayerSpec
from uniserve_worker.nn.linear import ColumnParallelLinear, LinearBase, QKVParallelLinear
from uniserve_worker.nn.mesh import TensorParallelSpec
from uniserve_worker.nn.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    zero_vocab_padding,
)

pytestmark = pytest.mark.integration


def _parallel() -> TensorParallelSpec:
    return TensorParallelSpec(rank=0, size=1)


def _execution(dtype: str = "float32") -> ExecutionConfig:
    return ExecutionConfig(
        model_dtype=dtype,
        cuda_graph=False,
        prefill_cuda_graph=False,
    )


def _qwen_config() -> dict[str, object]:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "attention_bias": False,
        "max_position_embeddings": 128,
    }


def test_catalog_resolves_only_explicit_architecture_identifiers():
    assert resolve_catalog_entry(("Qwen3ForCausalLM",)).model_class is Qwen3ForCausalLM
    assert (
        resolve_catalog_entry(("BagelForConditionalGeneration",)).checkpoint
        is CheckpointFormat.COMPOSITE
    )
    assert resolve_catalog_entry(("NEOChatModel",)).checkpoint is CheckpointFormat.NATIVE

    with pytest.raises(WorkerError, match="declare exactly one architecture"):
        resolve_catalog_entry(("QwenForCausalLM", "SenseNovaU1ForUnifiedGeneration"))
    with pytest.raises(WorkerError, match="declare exactly one architecture"):
        resolve_catalog_entry(("Qwen3ForCausalLM", "BagelForConditionalGeneration"))


@pytest.mark.parametrize("alias", ["bf16", "fp16", "fp32", "half"])
def test_transformer_dtype_parser_rejects_aliases(alias: str):
    with pytest.raises(ValueError, match="unknown transformer dtype"):
        dtype_from_name(alias)


def test_qwen_checkpoint_load_resolves_architecture_and_weight_identity(tmp_path):
    config = _qwen_config()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    reference = Qwen3ForCausalLM(
        config,
        layer_spec=LayerSpec(parallel=_parallel(), quantization=None),
    )
    with torch.no_grad():
        for index, parameter in enumerate(reference.parameters(), start=1):
            parameter.fill_(index / 16)
        for module in reference.modules():
            if isinstance(module, VocabParallelEmbedding):
                zero_vocab_padding(
                    module.num_embeddings,
                    module.vocab_start_index,
                    module.num_embeddings_per_partition,
                    module.weight,
                )
            elif isinstance(module, ParallelLMHead):
                zero_vocab_padding(
                    module.vocab_size,
                    module.vocab_start_index,
                    module.output_size,
                    module.weight,
                )
    checkpoint = {
        name: value.detach().contiguous() for name, value in reference.state_dict().items()
    }
    save_file(checkpoint, tmp_path / "model.safetensors")
    request = WorkerModelLoadRequest(
        model_path=str(tmp_path),
        device="cpu",
        block_size=16,
        max_batch_tokens=4096,
        kv_token_capacity=64,
        attention_backend="torch_sdpa",
        execution=_execution("bfloat16"),
        parallel=_parallel(),
    )

    loaded = load_worker_model(request)

    assert type(loaded.model) is Qwen3ForCausalLM
    assert loaded.identity.architecture == "Qwen3ForCausalLM"
    assert len(loaded.identity.architecture_digest) == 64
    assert len(loaded.identity.weight_digest) == 64
    assert loaded.tokenizer is None
    assert loaded.model.image_processor is None
    assert loaded.model.weight_spec.targets
    assert loaded.deployment.max_batch_tokens == 4096
    for name, parameter in loaded.model.named_parameters():
        torch.testing.assert_close(parameter, checkpoint[name].to(torch.bfloat16))

    changed = dict(checkpoint)
    first = next(iter(changed))
    changed[first] = changed[first].clone()
    changed[first].view(-1)[0] += 1
    save_file(changed, tmp_path / "model.safetensors")
    assert load_worker_model(request).identity.weight_digest != loaded.identity.weight_digest


def test_partial_scope_is_rejected_before_model_materialization(tmp_path):
    config = _qwen_config()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(WorkerError, match="does not support 'generation'"):
        load_worker_model(
            WorkerModelLoadRequest(
                model_path=str(tmp_path),
                device="cpu",
                block_size=16,
                max_batch_tokens=4096,
                kv_token_capacity=64,
                attention_backend="torch_sdpa",
                execution=_execution(),
                parallel=_parallel(),
                scope=ModelLoadScope.GENERATION,
            )
        )


@dataclass(frozen=True, slots=True)
class _CompositeConfig:
    @classmethod
    def from_mapping(cls, _raw: dict[str, object]) -> "_CompositeConfig":
        return cls()


class _CompositeGraph(nn.Module):
    def __init__(self, config: _CompositeConfig, *, layer_spec: LayerSpec) -> None:
        super().__init__()
        del config, layer_spec
        self.core = nn.Linear(2, 2, bias=False)
        self.vae = nn.Linear(2, 2, bias=False)


class _CompositeRoot(nn.Module):
    weight_spec = WeightSpec(
        files=("model.safetensors",),
        transforms=(Rename("inner.", ""),),
        sidecars=(Sidecar("ae.safetensors", "vae"),),
    )

    def __init__(
        self,
        config: _CompositeConfig,
        *,
        layer_spec: LayerSpec,
        graph: _CompositeGraph,
    ) -> None:
        super().__init__()
        del config, layer_spec
        self.graph = graph

    def forward(self, batch):
        raise AssertionError(f"loader conformance does not execute {batch!r}")


def test_composite_loader_streams_root_and_sidecar_before_ready(tmp_path):
    core = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    vae = torch.arange(4, dtype=torch.float32).reshape(2, 2) + 10
    save_file({"inner.core.weight": core}, tmp_path / "model.safetensors")
    save_file({"weight": vae}, tmp_path / "ae.safetensors")
    entry = CatalogEntry(
        architecture="CompositeConformanceModel",
        model_class=_CompositeRoot,
        checkpoint=CheckpointFormat.COMPOSITE,
        config_class=_CompositeConfig,
        graph_class=_CompositeGraph,
        serving_dtype="float32",
    )

    loaded = Loader().load(
        entry,
        {},
        model_path=str(tmp_path),
        device="cpu",
        attention_backend=None,
        model_scope="whole",
        execution=_execution(),
        parallel=_parallel(),
    )

    torch.testing.assert_close(loaded.model.graph.core.weight, core)
    torch.testing.assert_close(loaded.model.graph.vae.weight, vae)
    assert tuple(target.name for target in loaded.model.weight_spec.targets) == (
        "graph.core.weight",
        "graph.vae.weight",
    )


class _StackRoot(nn.Module):
    weight_spec = WeightSpec(
        transforms=(
            Stack("qkv", "q_proj", "q"),
            Stack("qkv", "k_proj", "k"),
            Stack("qkv", "v_proj", "v"),
        )
    )

    def __init__(self, config: object, *, layer_spec: LayerSpec) -> None:
        super().__init__()
        del config
        self.qkv = QKVParallelLinear(
            2,
            1,
            2,
            1,
            spec=layer_spec,
            bias=False,
        )


def test_stream_loader_stacks_declared_checkpoint_parts(tmp_path):
    query = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    key = torch.tensor([[5.0, 6.0]])
    value = torch.tensor([[7.0, 8.0]])
    save_file(
        {
            "q_proj.weight": query,
            "k_proj.weight": key,
            "v_proj.weight": value,
        },
        tmp_path / "model.safetensors",
    )
    entry = CatalogEntry(
        architecture="StackConformanceModel",
        model_class=_StackRoot,
        checkpoint=CheckpointFormat.STREAM,
    )

    loaded = Loader().load(
        entry,
        {},
        model_path=str(tmp_path),
        device="cpu",
        attention_backend=None,
        model_scope="whole",
        execution=_execution(),
        parallel=_parallel(),
    )

    torch.testing.assert_close(loaded.model.qkv.weight, torch.cat((query, key, value)))


def test_stack_transform_matches_exact_parameter_path_segments_once():
    spec = WeightSpec(
        transforms=(
            Stack("qkv_proj", "q_proj", "q"),
            Stack("qkv_proj", "k_proj", "k"),
            Stack("qkv_proj", "v_proj", "v"),
            Stack("gate_up_proj", "gate_proj", 0),
            Stack("gate_up_proj", "up_proj", 1),
        )
    )
    tensor = torch.empty(1)

    expected = {
        "model.q_proj.weight": ("model.qkv_proj.weight", "q"),
        "model.k_proj.weight": ("model.qkv_proj.weight", "k"),
        "model.v_proj.weight": ("model.qkv_proj.weight", "v"),
        "model.gate_proj.weight": ("model.gate_up_proj.weight", 0),
        "model.up_proj.weight": ("model.gate_up_proj.weight", 1),
    }
    for source_name, (target_name, part) in expected.items():
        fragment = transform_weight(spec, source_name, tensor)[0]
        assert fragment.name == target_name
        assert fragment.shard_id == part

    for fused_name in ("model.qkv_proj.weight", "model.gate_up_proj.weight"):
        fragment = transform_weight(spec, fused_name, tensor)[0]
        assert fragment.name == fused_name
        assert fragment.shard_id is None


class _AlgebraRoot(nn.Module):
    weight_spec = WeightSpec(
        transforms=(
            Slice("slice_source", "sliced", 0, 1, 3),
            Split("split_source", ("split_left", "split_right"), 0, (1, 2)),
            Transpose("transpose_source", "transposed", (1, 0)),
            Reshape("reshape_source", "reshaped", (2, 2)),
            Shard("global_weight", "sharded.weight", 0),
            Tie("tied_source", "tied_target"),
        )
    )

    def __init__(self, config: object, *, layer_spec: LayerSpec) -> None:
        super().__init__()
        del config
        self.sliced = nn.Parameter(torch.empty(2))
        self.split_left = nn.Parameter(torch.empty(1))
        self.split_right = nn.Parameter(torch.empty(2))
        self.transposed = nn.Parameter(torch.empty(3, 2))
        self.reshaped = nn.Parameter(torch.empty(2, 2))
        self.sharded = ColumnParallelLinear(4, 4, spec=layer_spec, bias=False)
        self.tied_source = nn.Parameter(torch.empty(2))
        self.tied_target = nn.Parameter(torch.empty(2))


def test_stream_loader_applies_tensor_algebra_sharding_and_ties(tmp_path):
    source = {
        "slice_source": torch.tensor([0.0, 1.0, 2.0, 3.0]),
        "split_source": torch.tensor([4.0, 5.0, 6.0]),
        "transpose_source": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "reshape_source": torch.arange(4, dtype=torch.float32),
        "global_weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "tied_source": torch.tensor([9.0, 10.0]),
    }
    save_file(source, tmp_path / "model.safetensors")
    entry = CatalogEntry(
        architecture="WeightAlgebraConformanceModel",
        model_class=_AlgebraRoot,
        checkpoint=CheckpointFormat.STREAM,
    )

    loaded = (
        Loader()
        .load(
            entry,
            {},
            model_path=str(tmp_path),
            device="cpu",
            attention_backend=None,
            model_scope="whole",
            execution=_execution(),
            parallel=TensorParallelSpec(rank=1, size=2),
        )
        .model
    )

    torch.testing.assert_close(loaded.sliced, source["slice_source"][1:3])
    torch.testing.assert_close(loaded.split_left, source["split_source"][:1])
    torch.testing.assert_close(loaded.split_right, source["split_source"][1:])
    torch.testing.assert_close(loaded.transposed, source["transpose_source"].t())
    torch.testing.assert_close(loaded.reshaped, source["reshape_source"].reshape(2, 2))
    torch.testing.assert_close(loaded.sharded.weight, source["global_weight"][2:])
    assert loaded.tied_target is loaded.tied_source
    torch.testing.assert_close(loaded.tied_source, source["tied_source"])


class _QuantizedRoot(nn.Module):
    weight_spec = WeightSpec(transforms=(Quantize("dense.weight", "projection.weight", "fp8"),))

    def __init__(self, config: object, *, layer_spec: LayerSpec) -> None:
        super().__init__()
        del config
        self.projection = LinearBase(2, 2, spec=layer_spec, bias=False)


def test_stream_loader_applies_declared_quantization(tmp_path):
    weight = torch.tensor([[1.0, -0.5], [0.25, 2.0]])
    save_file({"dense.weight": weight}, tmp_path / "model.safetensors")
    entry = CatalogEntry(
        architecture="QuantizedWeightConformanceModel",
        model_class=_QuantizedRoot,
        checkpoint=CheckpointFormat.STREAM,
    )

    loaded = (
        Loader()
        .load(
            entry,
            {"quantization_config": {"quant_method": "fp8"}},
            model_path=str(tmp_path),
            device="cpu",
            attention_backend=None,
            model_scope="whole",
            execution=_execution(),
            parallel=_parallel(),
        )
        .model
    )

    assert loaded.projection.weight.dtype == torch.float8_e4m3fn
    actual = loaded.projection(torch.tensor([[2.0, -1.0]]))
    expected = torch.nn.functional.linear(torch.tensor([[2.0, -1.0]]), weight)
    torch.testing.assert_close(actual, expected, rtol=0.08, atol=0.08)
