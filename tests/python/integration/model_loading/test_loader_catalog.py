"""Conformance for catalog-driven model resolution and dummy loading."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

import uniserve_worker.foundation.runtime_config as runtime_config
from uniserve_worker.bootstrap.catalog import MODEL_CATALOG
from uniserve_worker.bootstrap.model_loader import (
    WorkerModelLoadRequest,
    load_worker_model,
    model_architecture_candidates,
)
from uniserve_worker.contracts import ModelLoadScope, UniModel
from uniserve_worker.contracts.batches import seal_batch
from uniserve_worker.contracts.caps import Caps, ExecutionConstraints
from uniserve_worker.contracts.resource_plan import ResourcePlan
from uniserve_worker.foundation.errors import ErrorCode, WorkerError
from uniserve_worker.foundation.runtime_config import TorchCompileRuntimeConfig
from uniserve_worker.loader import get_loader
from uniserve_worker.loader.transformers import (
    dtype_from_name,
    load_native_transformers_checkpoint,
)
from uniserve_worker.loader.weight_utils import tensor_shape
from uniserve_worker.models.catalog import Catalog
from uniserve_worker.nn import LinearBase
from uniserve_worker.server.app import dispatch
from uniserve_worker.worker.model import ModelWorker

pytestmark = pytest.mark.integration


def _set_worker_runtime(monkeypatch, **kwargs):
    monkeypatch.setattr(
        runtime_config,
        "_CURRENT_EXECUTION_CONFIG",
        replace(runtime_config.get_execution_config(), **kwargs),
    )


def test_catalog_resolves_real_model_entries():
    assert MODEL_CATALOG.resolve(("NEOChatModel",)).__name__ == "SenseNovaU1ForUnifiedGeneration"
    assert (
        MODEL_CATALOG.resolve(("BagelForUnifiedGeneration",)).__name__
        == "BagelForUnifiedGeneration"
    )
    assert MODEL_CATALOG.resolve(("Qwen3ForCausalLM",)).__name__ == "Qwen3ForCausalLM"


def test_catalog_rejects_unknown_architecture(monkeypatch):
    _set_worker_runtime(monkeypatch)

    with pytest.raises(WorkerError) as excinfo:
        MODEL_CATALOG.resolve(("CompletelyNewCausalLM",))

    assert excinfo.value.code == ErrorCode.CAPABILITY_MISMATCH
    assert "no UniModel catalog entry" in str(excinfo.value)




def test_bagel_detection_hook_supplies_architecture_for_stub_config(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"")
    (tmp_path / "ae.safetensors").write_bytes(b"")

    archs = model_architecture_candidates(str(tmp_path))

    assert "BagelForUnifiedGeneration" in archs


def test_model_worker_rejects_declared_missing_control():
    class BadControlModel(UniModel):
        # Minimal valid caps so the worker reaches control validation (which is
        # what this test exercises), rather than tripping the caps check first.
        supported_ops = ("prefill_und", "decode_und")
        num_layers = 1
        bytes_per_token = 1
        supported_controls = ("missing_control",)

        def forward(self, batch):  # pragma: no cover - constructor should fail first.
            raise AssertionError("unreachable")

    try:
        ModelWorker(BadControlModel(), block_size=256)
    except WorkerError as exc:
        assert exc.code == ErrorCode.CAPABILITY_MISMATCH
        assert "declares control 'missing_control'" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing control declaration should fail at startup")


def test_catalog_rejects_classes_outside_the_model_contract():
    class BadDenoiseModel:
        supported_ops = ("denoise_gen",)

    with pytest.raises(WorkerError) as exc:
        Catalog((BadDenoiseModel,))
    assert exc.value.code == ErrorCode.CAPABILITY_MISMATCH
    assert "inherit UniModel" in exc.value.message


def test_dummy_loader_constructs_registered_model():
    from uniserve_worker.server.stub import StubUniModel

    config = {"model_type": "stub", "architectures": ("UniServeStubForUnifiedGeneration",)}
    result = get_loader("dummy").load_model(StubUniModel, config, device="cpu")
    assert result.model.config is config


def test_no_arg_loader_is_default():
    from uniserve_worker.loader.default import DefaultModelLoader

    assert isinstance(get_loader(), DefaultModelLoader)


def test_descriptor_loader_name_follows_the_declared_weight_spec():
    # Each family's WeightSpec names the loader that serves its checkpoint
    # layout; a family without one uses the default loader.
    from uniserve_worker.contracts.model_family import ModelFamilyDescriptor
    from uniserve_worker.models.bagel import BagelForUnifiedGeneration
    from uniserve_worker.models.qwen3 import Qwen3ForCausalLM
    from uniserve_worker.models.sensenova.model import SenseNovaU1ForUnifiedGeneration
    from uniserve_worker.server.stub import StubUniModel

    def loader_name(cls):
        return ModelFamilyDescriptor.from_model_class(cls).loader_name

    assert loader_name(SenseNovaU1ForUnifiedGeneration) == "native"
    assert loader_name(BagelForUnifiedGeneration) == "composite"
    assert loader_name(Qwen3ForCausalLM) == "default"
    assert loader_name(StubUniModel) == "default"


def test_partial_model_scope_requires_explicit_model_support(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"architectures": ["Qwen3ForCausalLM"]}',
        encoding="utf-8",
    )

    with pytest.raises(WorkerError) as error:
        load_worker_model(
            WorkerModelLoadRequest(
                model_path=str(tmp_path),
                device="cpu",
                block_size=16,
                kv_token_capacity=64,
                attention_backend="auto",
                scope=ModelLoadScope.GENERATION,
            )
        )

    assert error.value.code is ErrorCode.CAPABILITY_MISMATCH


def test_load_worker_model_resolves_spec_overlay_and_digest(tmp_path):
    from uniserve_worker.contracts.model_spec import resolved_digest
    from uniserve_worker.models.qwen3 import Qwen3ForCausalLM

    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "attention_bias": False,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    torch.manual_seed(3)
    checkpoint = {
        name: tensor.detach().clone()
        for name, tensor in Qwen3ForCausalLM(config=config).state_dict().items()
    }
    save_file(checkpoint, tmp_path / "model.safetensors")

    loaded = load_worker_model(
        WorkerModelLoadRequest(
            model_path=str(tmp_path),
            device="cpu",
            block_size=16,
            kv_token_capacity=64,
            attention_backend="auto",
        )
    )

    assert loaded.spec.architecture == "Qwen3ForCausalLM"
    assert loaded.spec.revision
    assert loaded.spec.op_kinds() == frozenset(loaded.model.supported_ops)
    assert loaded.overlay.device == "cpu"
    assert loaded.overlay.block_size == 16
    assert loaded.resolved_digest == resolved_digest(loaded.spec, loaded.overlay)

    # The same checkpoint under a different deployment overlay is a different
    # resolved identity; the same request resolves the same identity again.
    assert (
        resolved_digest(loaded.spec, replace(loaded.overlay, block_size=32))
        != loaded.resolved_digest
    )
    reloaded = load_worker_model(
        WorkerModelLoadRequest(
            model_path=str(tmp_path),
            device="cpu",
            block_size=16,
            kv_token_capacity=64,
            attention_backend="auto",
        )
    )
    assert reloaded.resolved_digest == loaded.resolved_digest


def test_transformers_dtype_typos_fail_loudly():
    assert dtype_from_name("bf16") is torch.bfloat16
    with pytest.raises(ValueError, match="unknown transformer dtype"):
        dtype_from_name("bflaot16")


def test_model_worker_computes_caps_once_and_serves_cached_copy():
    class CountingCapsModel(UniModel):
        supported_ops = ("prefill_und", "decode_und")
        supported_controls: tuple[str, ...] = ()
        adapter_mode = "none"
        resource_plan = ResourcePlan(kv_block="per_block")

        def __init__(self) -> None:
            self.calls = 0

        def caps(self, *, block_size=256, kv_token_capacity=None):
            self.calls += 1
            return Caps(
                block_size=block_size,
                num_blocks=4,
                num_layers=1,
                scratch_capacity_tokens=0,
                supported_ops=self.supported_ops,
                max_latent_size=0,
                latent_downsample=1,
                bytes_per_token=1,
                supported_controls=self.supported_controls,
                adapter_mode=self.adapter_mode,
                execution_constraints=ExecutionConstraints(
                    max_batch_ops=8,
                ),
                resource_classes=self.resource_plan.classes(),
            )

        def forward(self, *args, **kwargs):  # pragma: no cover - not executed.
            raise AssertionError("unreachable")

    model = CountingCapsModel()
    worker = ModelWorker(model, block_size=16, kv_token_capacity=64)

    assert model.calls == 1
    assert dispatch(worker, set(), {"kind": "get_caps"})["caps"] == worker.caps().to_wire()
    assert model.calls == 1


def test_model_worker_executes_registered_stub_model():
    from uniserve_worker.server.stub import StubUniModel

    model = StubUniModel()
    worker = ModelWorker(model, block_size=256, simulation=True)
    result = worker.execute(
        seal_batch(
            7,
            [
                {
                    "kind": "prefill_und",
                    "req_id": 1,
                    "token_ids": [10, 11],
                    "pos_range": [0, 2],
                }
            ],
            new_reqs=[{"req_id": 1, "sampling": {}, "image": {"steps": 1}}],
        )
    )
    assert result["step_id"] == 7
    assert result["per_seq"][0]["req_id"] == 1
    assert "sampled_token_id" in result["per_seq"][0]


def test_qwen3_entry_executes_through_model_worker():
    from uniserve_worker.models.qwen3 import Qwen3ForCausalLM
    from uniserve_worker.nn import ParallelLMHead, VocabParallelEmbedding

    torch.manual_seed(10)
    model = Qwen3ForCausalLM(
        config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "attention_bias": False,
        }
    )
    assert isinstance(model.model.embed_tokens, VocabParallelEmbedding)
    assert isinstance(model.lm_head, ParallelLMHead)
    worker = ModelWorker(
        model,
        block_size=16,
        kv_token_capacity=64,
        simulation=True,
    )
    result = worker.execute(
        seal_batch(
            8,
            [
                {
                    "kind": "prefill_und",
                    "req_id": 1,
                    "token_ids": [1, 2, 3],
                    "pos_range": [0, 3],
                    "new_block_ids": [1],
                }
            ],
            new_reqs=[{"req_id": 1, "sampling": {"temperature": 0.0}, "block_ids": []}],
        )
    )

    assert result["step_id"] == 8
    assert result["per_seq"][0]["req_id"] == 1
    assert isinstance(result["per_seq"][0]["sampled_token_id"], int)


def test_qwen3_tied_vocab_parallel_weights_share_parameter():
    from uniserve_worker.models.qwen3 import Qwen3ForCausalLM
    from uniserve_worker.nn import ParallelLMHead, VocabParallelEmbedding

    model = Qwen3ForCausalLM(
        config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "attention_bias": False,
            "tie_word_embeddings": True,
        }
    )

    assert isinstance(model.model.embed_tokens, VocabParallelEmbedding)
    assert isinstance(model.lm_head, ParallelLMHead)
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_qwen3_always_advertises_mixed_batch(monkeypatch):
    # Mixed-mode grouping is independent of the fused-kernel token budget.
    from uniserve_worker.models.qwen3 import Qwen3ForCausalLM

    model = Qwen3ForCausalLM(
        config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "attention_bias": False,
        },
    )

    assert model.batch_policy().supports_mixed_modes

    _set_worker_runtime(monkeypatch, mixed_text_max_tokens=256)
    assert model.batch_policy().supports_mixed_modes


def test_compile_runtime_applies_qwen3_model_stack_once(monkeypatch):
    from uniserve_worker.models.qwen3 import Qwen3ForCausalLM
    from uniserve_worker.runtime.compile import compile_model_pieces

    calls = []

    def fake_compile(module, **kwargs):
        calls.append((module, kwargs))
        setattr(module, "_compiled_by_test", True)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)
    _set_worker_runtime(
        monkeypatch,
        torch_compile=TorchCompileRuntimeConfig(enabled=True, backend="eager", mode=None),
        cuda_graph_warmup=False,
    )
    model = Qwen3ForCausalLM(
        config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "attention_bias": False,
        },
    )

    compile_model_pieces(model)
    compile_model_pieces(model)

    assert len(calls) == 1
    assert all(kwargs["backend"] == "eager" for _, kwargs in calls)
    assert all(kwargs["fullgraph"] is False for _, kwargs in calls)
    assert getattr(model.model, "_compiled_by_test", False)


def test_sensenova_applies_opt_in_native_model_stack_compile(monkeypatch):
    from uniserve_worker.models.sensenova.model import (
        SenseNovaU1ForUnifiedGeneration,
        _SenseNovaDecoderLayer,
    )

    calls = []

    def fake_compile(module, **kwargs):
        calls.append((module, kwargs))
        setattr(module, "_compiled_by_test", True)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)
    _set_worker_runtime(
        monkeypatch,
        torch_compile=TorchCompileRuntimeConfig(enabled=True, backend="eager", mode=None),
    )

    layer_cfg = SimpleNamespace(
        hidden_size=16,
        intermediate_size=32,
        hidden_act="silu",
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        attention_bias=False,
        rms_norm_eps=1e-6,
        layer_types=["full_attention"],
        sliding_window=None,
        rope_theta=10000.0,
        max_position_embeddings=128,
        rope_theta_hw=10000.0,
        max_position_embeddings_hw=128,
    )
    layer = _SenseNovaDecoderLayer(layer_cfg, layer_idx=0)
    decoder = nn.Module()
    decoder.layers = nn.ModuleList([layer])
    language_model = nn.Module()
    language_model.model = decoder
    native_model = nn.Module()
    native_model.language_model = language_model

    model = SenseNovaU1ForUnifiedGeneration(
        config={
            "llm_config": {
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
                "head_dim": 4,
            }
        }
    )
    model.model = native_model
    from uniserve_worker.runtime.compile import compile_model_pieces

    compile_model_pieces(model)

    assert len(calls) == 1
    assert all(kwargs["backend"] == "eager" for _, kwargs in calls)
    assert getattr(language_model.model, "_compiled_by_test", False)


def test_zero_day_diffusion_model_uses_shared_cfg_zero_star_path():
    from tests.python.fixtures.zero_day import UniServeZeroDayCfgZeroStarModel

    # A test-only model enters serving through an explicit catalog entry.
    catalog = Catalog((UniServeZeroDayCfgZeroStarModel,))
    cls = catalog.resolve(("UniServeZeroDayCfgZeroStarModel",))
    worker = ModelWorker(cls(), block_size=256, simulation=True)
    result = worker.execute(
        seal_batch(
            9,
            [
                {
                    "kind": "denoise_gen",
                    "req_id": 11,
                    "timestep_idx": 0,
                    "num_steps": 1,
                    "latent_shape": [1, 2, 2],
                    "cfg": {
                        "branch_count": 2,
                        "text_scale": 2.0,
                        "img_scale": 1.0,
                        "renorm_type": "cfg_zero_star",
                        "renorm_min": 0.0,
                        "interval": [0.0, 1.0],
                    },
                }
            ],
            new_reqs=[{"req_id": 11, "sampling": {}, "image": {"steps": 1}}],
        )
    )
    assert result["step_id"] == 9
    [row] = result["per_seq"]
    assert row["req_id"] == 11
    assert row["denoise_done"] is True
    assert row["num_steps_done"] == 1


class TinyLoadableModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = LinearBase(3, 2, bias=True)


def test_default_loader_streams_safetensors_into_weight_hooks(tmp_path):
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    bias = torch.tensor([0.5, -0.5])
    save_file({"proj.weight": weight, "proj.bias": bias}, tmp_path / "model.safetensors")

    config = {"model_type": "tiny"}
    result = get_loader("default").load_model(
        TinyLoadableModel,
        config,
        device="cpu",
        model_path=str(tmp_path),
    )
    model = result.model
    torch.testing.assert_close(model.proj.weight, weight)
    torch.testing.assert_close(model.proj.bias, bias)


def test_default_loader_fails_loudly_when_a_parameter_has_no_checkpoint_tensor(tmp_path):
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    save_file({"proj.weight": weight}, tmp_path / "model.safetensors")

    with pytest.raises(RuntimeError, match="missing"):
        get_loader("default").load_model(
            TinyLoadableModel,
            {"model_type": "tiny"},
            device="cpu",
            model_path=str(tmp_path),
        )


def test_native_transformers_loader_streams_to_plain_module(tmp_path):
    class TinyConfig(SimpleNamespace):
        @classmethod
        def from_pretrained(cls, model_dir):
            return cls(model_dir=model_dir)

    class TinyTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class TinyNativeModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.proj = nn.Linear(3, 2)

    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    bias = torch.tensor([0.25, -0.75], dtype=torch.float32)
    save_file({"proj.weight": weight, "proj.bias": bias}, tmp_path / "model.safetensors")

    model, tokenizer, real_device = load_native_transformers_checkpoint(
        str(tmp_path),
        "cpu",
        config_cls=TinyConfig,
        model_cls=TinyNativeModel,
        tokenizer_cls=TinyTokenizer,
    )

    assert isinstance(tokenizer, TinyTokenizer)
    assert real_device == "cpu"
    assert not any(param.is_meta for param in model.parameters())
    torch.testing.assert_close(model.proj.weight, weight.to(torch.bfloat16))
    torch.testing.assert_close(model.proj.bias, bias.to(torch.bfloat16))


def test_native_transformers_loader_maps_separate_projection_weights_to_fused_params(tmp_path):
    from uniserve_worker.nn import MergedColumnParallelLinear, QKVParallelLinear
    from uniserve_worker.nn.placement import WeightMode

    class TinyConfig(SimpleNamespace):
        @classmethod
        def from_pretrained(cls, model_dir):
            return cls(model_dir=model_dir)

    class TinyTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class TinySelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = QKVParallelLinear(
                hidden_size=3,
                head_size=2,
                total_num_heads=2,
                total_num_kv_heads=1,
                bias=False,
            )

    class TinyMlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = MergedColumnParallelLinear(
                3,
                (4, 4),
                bias=False,
                weight_mode=WeightMode.FUSED_GATE_UP_LINEAR,
            )

    class TinyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = TinySelfAttention()
            self.mlp = TinyMlp()

    class TinyNativeModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.layers = nn.ModuleList([TinyLayer()])

    q = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    k = torch.arange(6, dtype=torch.float32).reshape(2, 3) + 100
    v = torch.arange(6, dtype=torch.float32).reshape(2, 3) + 200
    gate = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 300
    up = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 400
    save_file(
        {
            "layers.0.self_attn.q_proj.weight": q,
            "layers.0.self_attn.k_proj.weight": k,
            "layers.0.self_attn.v_proj.weight": v,
            "layers.0.mlp.gate_proj.weight": gate,
            "layers.0.mlp.up_proj.weight": up,
        },
        tmp_path / "model.safetensors",
    )

    model, tokenizer, real_device = load_native_transformers_checkpoint(
        str(tmp_path),
        "cpu",
        config_cls=TinyConfig,
        model_cls=TinyNativeModel,
        tokenizer_cls=TinyTokenizer,
        stacked_params_mapping=(
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ),
    )

    assert isinstance(tokenizer, TinyTokenizer)
    assert real_device == "cpu"
    layer = model.layers[0]
    expected_qkv = torch.cat([q, k, v], dim=0).to(torch.bfloat16)
    expected_gate_up = torch.cat([gate, up], dim=0).to(torch.bfloat16)
    torch.testing.assert_close(layer.self_attn.qkv_proj.weight, expected_qkv)
    torch.testing.assert_close(layer.mlp.gate_up_proj.weight, expected_gate_up)

    x = torch.randn(5, 3, dtype=torch.bfloat16)
    torch.testing.assert_close(
        layer.self_attn.qkv_proj(x),
        torch.nn.functional.linear(x, expected_qkv),
    )
    torch.testing.assert_close(
        layer.mlp.gate_up_proj(x),
        torch.nn.functional.linear(x, expected_gate_up),
    )


def test_native_loader_builds_the_serving_wrapper_from_its_weight_spec(tmp_path):
    # The registered "native" BaseModelLoader reads the wrapper's declared
    # WeightSpec, materializes the inner module, and constructs the serving
    # wrapper around it, returning a LoadResult(model + tokenizer + device).
    from uniserve_worker.loader import LoadResult, NativeSource, WeightSpec, get_loader

    class TinyConfig(SimpleNamespace):
        @classmethod
        def from_pretrained(cls, model_dir):
            return cls(model_dir=model_dir)

    class TinyTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class TinyNativeModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.proj = nn.Linear(3, 2)

    class TinyWrapper:
        weight_spec = WeightSpec(
            native=NativeSource(
                config_cls=TinyConfig,
                module_cls=TinyNativeModel,
                tokenizer_cls=TinyTokenizer,
            ),
        )

        def __init__(self, config, *, model, tokenizer, device, **serving):
            del serving
            self.config = config
            self.model = model
            self.tokenizer = tokenizer
            self.device = device

    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    bias = torch.tensor([0.25, -0.75], dtype=torch.float32)
    save_file({"proj.weight": weight, "proj.bias": bias}, tmp_path / "model.safetensors")

    result = get_loader("native").load_model(
        TinyWrapper, None, device="cpu", model_path=str(tmp_path)
    )

    assert isinstance(result, LoadResult)
    assert isinstance(result.model, TinyWrapper)
    assert isinstance(result.model.tokenizer, TinyTokenizer)
    assert result.device == "cpu"
    assert isinstance(result.model.model, TinyNativeModel)
    torch.testing.assert_close(result.model.model.proj.weight, weight.to(torch.bfloat16))


def test_native_loader_materializes_only_the_declared_tower_role(tmp_path):
    from uniserve_worker.loader import NativeSource, TowerSplit, WeightSpec

    class TinyConfig(SimpleNamespace):
        @classmethod
        def from_pretrained(cls, model_dir):
            return cls(model_dir=model_dir)

    class TinyTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class TinyNativeModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.shared = nn.Linear(2, 2, bias=False)
            self.gen = nn.Linear(2, 2, bias=False)

    class TinyWrapper:
        weight_spec = WeightSpec(
            native=NativeSource(
                config_cls=TinyConfig,
                module_cls=TinyNativeModel,
                tokenizer_cls=TinyTokenizer,
            ),
            tower=TowerSplit(gen_prefixes=("gen.",)),
        )

        def __init__(self, config, *, model, tokenizer, device, **serving):
            del config, tokenizer, device, serving
            self.model = model

    shared = torch.ones(2, 2)
    gen = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    save_file({"shared.weight": shared, "gen.weight": gen}, tmp_path / "model.safetensors")

    result = get_loader("native").load_model(
        TinyWrapper,
        None,
        device="cpu",
        model_path=str(tmp_path),
        tower_role="gen",
    )

    assert result.model.model.shared.weight.is_meta
    assert not result.model.model.gen.weight.is_meta
    torch.testing.assert_close(result.model.model.gen.weight, gen.to(torch.bfloat16))


def test_composite_loader_streams_root_file_sidecar_and_wraps_the_graph(tmp_path):
    # The registered "composite" BaseModelLoader builds the inner graph from
    # its declared config, streams the first existing root file through the
    # declared rename rules, loads sidecar submodule files, casts to the
    # declared serving dtype, and constructs the serving wrapper.
    from uniserve_worker.foundation.errors import WorkerError
    from uniserve_worker.loader import GraphSource, Rename, Sidecar, WeightSpec, get_loader

    class TinyGraphConfig:
        @classmethod
        def from_pretrained(cls, model_dir):
            del model_dir
            return cls()

    class TinyGraph(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.core = nn.Linear(2, 2, bias=False)
            self.vae = nn.Linear(2, 2, bias=False)

    class TinyComposite:
        weight_spec = WeightSpec(
            graph=GraphSource(config_cls=TinyGraphConfig, module_cls=TinyGraph),
            checkpoint_files=("ema.safetensors", "model.safetensors"),
            renames=(Rename("inner.core.", "core."),),
            unmatched="skip",
            sidecars=(Sidecar(file="ae.safetensors", module="vae"),),
        )

        def __init__(self, config, *, model, device, **serving):
            del serving
            self.config = config
            self.model = model
            self.device = device

    core = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    vae = torch.arange(4, dtype=torch.float32).reshape(2, 2) + 10
    save_file(
        {"inner.core.weight": core, "ignored.tensor": torch.zeros(1)},
        tmp_path / "ema.safetensors",
    )
    save_file({"weight": vae}, tmp_path / "ae.safetensors")

    result = get_loader("composite").load_model(
        TinyComposite, None, device="cpu", model_path=str(tmp_path)
    )

    assert isinstance(result.model, TinyComposite)
    graph = result.model.model
    assert graph.core.weight.dtype == torch.bfloat16
    torch.testing.assert_close(graph.core.weight, core.to(torch.bfloat16))
    torch.testing.assert_close(graph.vae.weight, vae.to(torch.bfloat16))

    save_file({"unrelated.weight": core}, tmp_path / "ema.safetensors")
    with pytest.raises(WorkerError, match="missing"):
        get_loader("composite").load_model(
            TinyComposite, None, device="cpu", model_path=str(tmp_path)
        )


def test_native_transformers_loader_preserves_fp8_linear_checkpoint_tensors(tmp_path):
    class TinyConfig(SimpleNamespace):
        @classmethod
        def from_pretrained(cls, model_dir):
            return cls(
                model_dir=model_dir,
                quantization_config={"quant_method": "fp8"},
            )

    class TinyTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class TinyNativeModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.proj = LinearBase(16, 8, bias=True, prefix="proj")

    dense_weight = torch.linspace(-1.0, 1.0, 128, dtype=torch.float32).reshape(8, 16)
    scale = dense_weight.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-12) / 448.0
    fp8_weight = (dense_weight / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    bias = torch.linspace(-0.25, 0.25, 8, dtype=torch.float32)
    save_file(
        {
            "proj.weight": fp8_weight,
            "proj.weight_scale": scale,
            "proj.bias": bias,
        },
        tmp_path / "model.safetensors",
    )

    model, _, _ = load_native_transformers_checkpoint(
        str(tmp_path),
        "cpu",
        config_cls=TinyConfig,
        model_cls=TinyNativeModel,
        tokenizer_cls=TinyTokenizer,
    )

    assert model.proj.weight.dtype == torch.float8_e4m3fn
    assert model.proj.weight_scale.dtype == torch.float32
    assert bool(getattr(model.proj.weight_scale, "_uniserve_skip_serving_cast", False))
    torch.testing.assert_close(model.proj.weight_scale, scale)

    x = torch.linspace(-0.4, 0.4, 32, dtype=torch.float32).reshape(2, 16)
    got = model.proj(x)
    expected = F.linear(x, fp8_weight.float() * scale, model.proj.bias.float())
    torch.testing.assert_close(got, expected)


def test_native_transformers_loader_uses_weight_loader_for_padded_vocab_tensors(tmp_path):
    from uniserve_worker.nn import ParallelLMHead, VocabParallelEmbedding

    class TinyConfig(SimpleNamespace):
        @classmethod
        def from_pretrained(cls, model_dir):
            return cls(model_dir=model_dir)

    class TinyTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class TinyNativeModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.embed_tokens = VocabParallelEmbedding(5, 3, pad_vocab_size_to=8)
            self.lm_head = ParallelLMHead(3, 5, bias=False, pad_vocab_size_to=8)

    table = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    save_file(
        {
            "embed_tokens.weight": table,
            "lm_head.weight": table + 10,
        },
        tmp_path / "model.safetensors",
    )

    model, _, _ = load_native_transformers_checkpoint(
        str(tmp_path),
        "cpu",
        config_cls=TinyConfig,
        model_cls=TinyNativeModel,
        tokenizer_cls=TinyTokenizer,
    )

    assert model.embed_tokens.weight.shape == (8, 3)
    assert model.lm_head.weight.shape == (8, 3)
    torch.testing.assert_close(model.embed_tokens.weight[:5], table.to(torch.bfloat16))
    torch.testing.assert_close(
        model.embed_tokens.weight[5:], torch.zeros(3, 3, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(model.lm_head.weight[:5], (table + 10).to(torch.bfloat16))
    torch.testing.assert_close(model.lm_head.weight[5:], torch.zeros(3, 3, dtype=torch.bfloat16))

    ids = torch.tensor([[0, 4, 2]], dtype=torch.long)
    hidden = torch.randn(2, 3, dtype=torch.bfloat16)
    torch.testing.assert_close(model.embed_tokens(ids), F.embedding(ids, table.to(torch.bfloat16)))
    torch.testing.assert_close(
        model.lm_head(hidden), F.linear(hidden, (table + 10).to(torch.bfloat16))
    )


def test_weight_utils_reads_safetensors_tensor_shape_without_loading_tensor(tmp_path):
    save_file({"latent_pos_embed.pos_embed": torch.zeros(4096, 8)}, tmp_path / "model.safetensors")
    assert tensor_shape(tmp_path / "model.safetensors", "latent_pos_embed.pos_embed") == (4096, 8)
