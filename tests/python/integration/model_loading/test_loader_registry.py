"""Conformance for new-style model discovery and dummy loading."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

import uniserve_worker.foundation.runtime_config as runtime_config
from uniserve_worker.contracts.caps import Caps, ExecutionConstraints, validate_caps
from uniserve_worker.contracts.model_protocols import ModelHooks
from uniserve_worker.contracts.resource_plan import ResourcePlan
from uniserve_worker.foundation.errors import ErrorCode, WorkerError
from uniserve_worker.foundation.runtime_config import TorchCompileRuntimeConfig
from uniserve_worker.loader import get_loader
from uniserve_worker.loader.transformers import (
    dtype_from_name,
    load_native_transformers_checkpoint,
)
from uniserve_worker.loader.weight_utils import tensor_shape
from uniserve_worker.models.registry import resolve_model_cls
from uniserve_worker.nn import LinearBase
from uniserve_worker.runtime.lora import MergeOnLoadLoRA
from uniserve_worker.server.app import dispatch
from uniserve_worker.server.runner_driver import RunnerDriver, _architectures

pytestmark = pytest.mark.integration


def _set_worker_runtime(monkeypatch, **kwargs):
    monkeypatch.setattr(
        runtime_config,
        "_CURRENT_CONFIG",
        replace(runtime_config.get_worker_config(), **kwargs),
    )


def test_stub_fixture_is_runtime_owned_not_model_discovered():
    from uniserve_worker.server.stub import StubUniModel

    assert StubUniModel.__name__ == "StubUniModel"


def test_registry_resolves_real_model_entries():
    assert resolve_model_cls(("NEOChatModel",)).__name__ == "SenseNovaU1ForUnifiedGeneration"
    assert resolve_model_cls(("BagelForUnifiedGeneration",)).__name__ == "BagelForUnifiedGeneration"
    assert resolve_model_cls(("Qwen3ForCausalLM",)).__name__ == "Qwen3ForCausalLM"


def test_registry_falls_back_to_generic_transformers_for_unknown_architecture():
    cls = resolve_model_cls(("CompletelyNewCausalLM",))
    assert cls.__name__ == "TransformersForCausalLM"
    assert cls.fallback is True


def test_transformers_fallback_caps_are_text_only_and_validated():
    from uniserve_worker.models.transformers_fallback import TransformersForCausalLM

    cfg = SimpleNamespace(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=3,
        eos_token_id=2,
    )
    model = TransformersForCausalLM(
        model=nn.Linear(1, 1),
        tokenizer=SimpleNamespace(eos_token_id=2),
        config=cfg,
        device="cpu",
        block_size=16,
        kv_token_capacity=64,
    )

    caps = validate_caps(model.caps().to_wire(), owner="TransformersForCausalLM")

    assert caps.supported_ops == ("prefill_und", "decode_und")
    assert caps.num_blocks == 4
    assert caps.bytes_per_token == 2 * 8 * 2 * 3 * 2


def test_transformers_fallback_registers_uniserve_attention_function():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from uniserve_worker.models.transformers_fallback import (
        _ATTN_IMPL,
        _register_uniserve_attention,
    )

    _register_uniserve_attention()
    fn = ALL_ATTENTION_FUNCTIONS[_ATTN_IMPL]
    module = nn.Module()
    module.layer_idx = 0
    module.is_causal = False
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)

    out, weights = fn(module, q, k, v, attention_mask=None, scaling=4**-0.5)

    assert weights is None
    assert out.shape == (1, 3, 2, 4)


def test_bagel_detection_hook_supplies_architecture_for_stub_config(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"")
    (tmp_path / "ae.safetensors").write_bytes(b"")

    archs = _architectures(str(tmp_path))

    assert "BagelForUnifiedGeneration" in archs


def test_runner_driver_rejects_declared_missing_control():
    class BadControlModel(ModelHooks):
        # Minimal valid caps so the driver reaches control validation (which is
        # what this test exercises), rather than tripping the caps check first.
        supported_ops = ("prefill_und", "decode_und")
        num_layers = 1
        bytes_per_token = 1
        supported_controls = ("missing_control",)

        def forward(self, batch):  # pragma: no cover - constructor should fail first.
            raise AssertionError("unreachable")

    try:
        RunnerDriver(BadControlModel(), block_size=256)
    except WorkerError as exc:
        assert exc.code == ErrorCode.CAPABILITY_MISMATCH
        assert "declares control 'missing_control'" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing control declaration should fail at startup")


def test_registry_rejects_declared_op_without_runner_capability():
    from uniserve_worker.models.registry import ModelRegistry

    class BadDenoiseModel:
        supported_ops = ("denoise_gen",)

    with pytest.raises(WorkerError) as exc:
        ModelRegistry().register(BadDenoiseModel, names=("bad",))
    assert exc.value.code == ErrorCode.CAPABILITY_MISMATCH
    assert "predict_velocity" in exc.value.message


def test_model_import_isolation_warns_and_continues(monkeypatch, caplog):
    import importlib
    from types import SimpleNamespace

    import uniserve_worker.foundation.plugins as plugins
    import uniserve_worker.models.registry as registry

    # Plugin discovery (import isolation) lives in foundation.plugins; the registry
    # delegates to discover_package_plugins. Patch the plugins-layer iteration/import.
    registry.import_model_classes.cache_clear()
    real_import = importlib.import_module

    def fake_iter_modules(_paths):
        return [SimpleNamespace(name="broken"), SimpleNamespace(name="empty")]

    def fake_import_module(name):
        if name.endswith(".broken"):
            raise RuntimeError("broken import")
        if name.endswith(".empty"):
            return SimpleNamespace(EntryClass=None)
        return real_import(name)

    monkeypatch.setattr(plugins.pkgutil, "iter_modules", fake_iter_modules)
    monkeypatch.setattr(plugins.importlib, "import_module", fake_import_module)

    with caplog.at_level("WARNING"):
        registry.import_model_classes(strict=False)
    assert "skipping plugin module" in caplog.text

    registry.import_model_classes.cache_clear()
    with pytest.raises(RuntimeError, match="broken import"):
        registry.import_model_classes(strict=True)
    registry.import_model_classes.cache_clear()


def test_dummy_loader_constructs_registered_model():
    from uniserve_worker.server.stub import StubUniModel

    config = {"model_type": "stub", "architectures": ("UniServeStubForUnifiedGeneration",)}
    result = get_loader("dummy").load_model(StubUniModel, config, device="cpu")
    assert result.model.config is config


def test_no_arg_loader_is_default():
    from uniserve_worker.loader.default import DefaultModelLoader

    assert isinstance(get_loader(), DefaultModelLoader)


def test_model_bring_up_contract_selects_the_from_pretrained_path():
    # load_runner_engine dispatches on the ModelBringUp contract instead of
    # reflecting an incidental ``from_pretrained`` attribute: model-owned
    # bring-up classes satisfy it, config-driven (registry-loaded) ones do not.
    from uniserve_worker.loader import ModelBringUp
    from uniserve_worker.models.bagel import BagelForUnifiedGeneration
    from uniserve_worker.models.qwen3 import Qwen3ForCausalLM
    from uniserve_worker.models.sensenova.model import SenseNovaU1ForUnifiedGeneration
    from uniserve_worker.models.transformers_fallback import TransformersForCausalLM
    from uniserve_worker.server.stub import StubUniModel

    assert issubclass(SenseNovaU1ForUnifiedGeneration, ModelBringUp)
    assert issubclass(BagelForUnifiedGeneration, ModelBringUp)
    assert issubclass(TransformersForCausalLM, ModelBringUp)
    assert not issubclass(Qwen3ForCausalLM, ModelBringUp)
    assert not issubclass(StubUniModel, ModelBringUp)


def test_transformers_dtype_typos_fail_loudly():
    assert dtype_from_name("bf16") is torch.bfloat16
    with pytest.raises(ValueError, match="unknown transformer dtype"):
        dtype_from_name("bflaot16")


def test_runner_driver_computes_caps_once_and_serves_cached_copy():
    class CountingCapsModel(ModelHooks):
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
    driver = RunnerDriver(model, block_size=16, kv_token_capacity=64)

    assert model.calls == 1
    assert dispatch(driver, set(), {"kind": "get_caps"})["caps"] == driver.caps().to_wire()
    assert model.calls == 1


def test_runner_driver_executes_registered_stub_model():
    from uniserve_worker.server.stub import StubUniModel

    model = StubUniModel()
    driver = RunnerDriver(model, block_size=256)
    result = driver.execute(
        {
            "step_id": 7,
            "new_reqs": [{"req_id": 1, "sampling": {}, "image": {"steps": 1}}],
            "ops": [
                {
                    "kind": "prefill_und",
                    "req_id": 1,
                    "token_ids": [10, 11],
                    "pos_range": [0, 2],
                }
            ],
        }
    )
    assert result["step_id"] == 7
    assert result["per_seq"][0]["req_id"] == 1
    assert "sampled_token_id" in result["per_seq"][0]


def test_qwen3_entry_executes_through_text_driver():
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
    driver = RunnerDriver(model, block_size=16, kv_token_capacity=64)
    result = driver.execute(
        {
            "step_id": 8,
            "new_reqs": [{"req_id": 1, "sampling": {"temperature": 0.0}, "block_ids": []}],
            "ops": [
                {
                    "kind": "prefill_und",
                    "req_id": 1,
                    "token_ids": [1, 2, 3],
                    "pos_range": [0, 3],
                    "new_block_ids": [1],
                }
            ],
        }
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


def test_qwen3_prefill_projects_only_last_token_logits(monkeypatch):
    from uniserve_worker.models.qwen3 import Qwen3ForCausalLM

    torch.manual_seed(11)
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
    seen: dict[str, tuple[int, ...]] = {}
    original_forward = model.lm_head.forward

    def record_lm_head(hidden):
        seen["shape"] = tuple(hidden.shape)
        return original_forward(hidden)

    monkeypatch.setattr(model.lm_head, "forward", record_lm_head)
    driver = RunnerDriver(model, block_size=16, kv_token_capacity=64)
    driver.execute(
        {
            "step_id": 88,
            "new_reqs": [{"req_id": 1, "sampling": {"temperature": 0.0}, "block_ids": []}],
            "ops": [
                {
                    "kind": "prefill_und",
                    "req_id": 1,
                    "token_ids": [1, 2, 3],
                    "pos_range": [0, 3],
                    "new_block_ids": [1],
                }
            ],
        }
    )

    assert seen["shape"] == (1, 16)


def test_qwen3_qk_norm_defaults_to_native_output_dtype_with_fp32_escape_hatch():
    from uniserve_worker.models.qwen3 import Qwen3Attention

    cfg = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        attention_bias=False,
        rms_norm_eps=1e-6,
        qk_norm_output_fp32=False,
    )
    native = Qwen3Attention(cfg, layer_id=0).to(dtype=torch.bfloat16)
    x = torch.randn(3, 2, 4, dtype=torch.bfloat16)

    assert native._apply_qk_norm(native.q_norm, x).dtype == torch.bfloat16

    cfg.qk_norm_output_fp32 = True
    fp32 = Qwen3Attention(cfg, layer_id=0).to(dtype=torch.bfloat16)
    assert fp32._apply_qk_norm(fp32.q_norm, x).dtype == torch.float32


def test_qwen3_always_advertises_mixed_batch(monkeypatch):
    # und/gen (here prefill/decode) mixed-batch single-forward is a
    # non-negotiable invariant (§9.6): qwen3 advertises mixed-mode grouping
    # unconditionally, regardless of the worker-side fused-kernel token budget.
    # The runtime mixed-token budget only narrows fused-kernel eligibility, not the
    # advertised capability.
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

    assert not hasattr(model.caps().execution_constraints, "supports_mixed_op_kinds")
    assert model.batch_policy().supports_mixed_modes

    _set_worker_runtime(monkeypatch, mixed_text_max_tokens=256)
    assert model.batch_policy().supports_mixed_modes


def test_qwen3_runtime_applies_opt_in_model_stack_compile(monkeypatch):
    from uniserve_worker.models.qwen3 import Qwen3ForCausalLM

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

    model.configure_runtime(block_size=16, kv_token_capacity=64)
    model.configure_runtime(block_size=16, kv_token_capacity=64)

    assert len(calls) == 1
    assert all(kwargs["backend"] == "eager" for _, kwargs in calls)
    assert all(kwargs["fullgraph"] is False for _, kwargs in calls)
    assert getattr(model.model, "_compiled_by_test", False)
    assert model._torch_compile_applied


def test_sensenova_applies_opt_in_native_model_stack_compile(monkeypatch):
    from uniserve_worker.models.sensenova.model import (
        SenseNovaU1ForUnifiedGeneration,
        _NativeQwen3DecoderLayer,
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
    layer = _NativeQwen3DecoderLayer(layer_cfg, layer_idx=0)
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
    model._maybe_compile_piecewise()

    assert len(calls) == 1
    assert all(kwargs["backend"] == "eager" for _, kwargs in calls)
    assert getattr(language_model.model, "_compiled_by_test", False)
    assert model._torch_compile_applied


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen3_mixed_prefill_decode_runs_flashinfer_paged_varlen_cuda(monkeypatch):
    pytest.importorskip("flashinfer")
    from uniserve_worker.models.qwen3 import Qwen3ForCausalLM

    _set_worker_runtime(
        monkeypatch,
        varlen_prefill=True,
        cuda_graph_warmup=False,
        mixed_text_max_tokens=256,
    )
    monkeypatch.setenv("UNISERVE_FORWARD_METRICS", "1")
    torch.manual_seed(125)
    model = Qwen3ForCausalLM(
        config={
            "vocab_size": 64,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "attention_bias": False,
        },
    ).cuda()
    driver = RunnerDriver(model, block_size=16, kv_token_capacity=128)
    assert model.kv_pool is not None
    model.kv_pool.k.zero_()
    model.kv_pool.v.zero_()
    ops = [
        {
            "kind": "prefill_und",
            "req_id": 1,
            "token_ids": [7, 8, 9],
            "pos_range": [4, 7],
        },
        {
            "kind": "decode_und",
            "req_id": 2,
            "token_ids": [10],
            "pos_range": [2, 3],
        },
    ]
    assert model.can_run_mixed_batch(ops, attention_backend_name="auto")

    result = driver.execute(
        {
            "step_id": 92,
            "new_reqs": [
                {"req_id": 1, "sampling": {"temperature": 0.0}, "block_ids": [1]},
                {"req_id": 2, "sampling": {"temperature": 0.0}, "block_ids": [2]},
            ],
            "ops": ops,
        }
    )

    assert result["step_id"] == 92
    assert [row["req_id"] for row in result["per_seq"]] == [1, 2]
    assert all(isinstance(row["sampled_token_id"], int) for row in result["per_seq"])
    assert driver.runner.request_states.get(1).kv_length("qwen3") == 7
    assert driver.runner.request_states.get(2).kv_length("qwen3") == 3
    stats = result["forward_stats"]
    assert stats["mode_counts"] == {"mixed": 2}
    assert stats["attention_backend_counts"] == {"flashinfer_paged_varlen": 1}
    assert stats["flashinfer_prefill_plan_calls"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen3_initial_prefill_cuda_graph_replay_preserves_raw_kv_lengths(monkeypatch):
    pytest.importorskip("sgl_kernel")
    from uniserve_worker.models.qwen3 import Qwen3ForCausalLM

    cfg = {
        "vocab_size": 64,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "attention_bias": False,
    }
    _set_worker_runtime(
        monkeypatch,
        cuda_graph=False,
        cuda_graph_warmup=False,
        prefill_cuda_graph_warmup=False,
        prefill_cuda_graph_warmup_tokens=(4, 8),
    )
    monkeypatch.setenv("UNISERVE_FORWARD_METRICS", "1")

    torch.manual_seed(777)
    eager_model = Qwen3ForCausalLM(config=cfg).cuda()
    graph_model = Qwen3ForCausalLM(config=cfg).cuda()
    graph_model.load_state_dict(eager_model.state_dict())

    def run_initial_prefills(model, *, graph_enabled: bool):
        _set_worker_runtime(monkeypatch, prefill_cuda_graph=graph_enabled)
        driver = RunnerDriver(model, block_size=256, kv_token_capacity=2048)
        outputs = []
        for step, tokens, blocks in (
            (1, [1, 2, 3, 4, 5], [1]),
            (2, [6, 7, 8, 9, 10, 11, 12], [2]),
        ):
            result = driver.execute(
                {
                    "step_id": step,
                    "new_reqs": [{"req_id": step, "sampling": {"temperature": 0.0}, "block_ids": []}],
                    "ops": [
                        {
                            "kind": "prefill_und",
                            "req_id": step,
                            "token_ids": tokens,
                            "pos_range": [0, len(tokens)],
                            "new_block_ids": blocks,
                        }
                    ],
                }
            )
            assert driver.runner.request_states.get(step).kv_length("qwen3") == len(tokens)
            outputs.append((result["per_seq"][0]["sampled_token_id"], result["forward_stats"]))
        torch.cuda.synchronize()
        return outputs

    eager = run_initial_prefills(eager_model, graph_enabled=False)
    graph = run_initial_prefills(graph_model, graph_enabled=True)

    assert [row[0] for row in graph] == [row[0] for row in eager]
    assert sorted(graph_model._prefill_graphs) == [(8, 1)]
    assert graph_model._prefill_graph_disabled == set()
    stats = graph[-1][1]
    assert stats["cuda_graph_replays"] >= 1
    assert stats["cuda_graph_runtime_mode_counts"].get("extend", 0) >= 1
    assert stats["cuda_graph_unpadded_tokens"] == 7
    assert stats["cuda_graph_padded_tokens"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen3_initial_prefill_cuda_graph_buckets_do_not_share_static_metadata(monkeypatch):
    pytest.importorskip("sgl_kernel")
    from uniserve_worker.models.qwen3 import Qwen3ForCausalLM

    cfg = {
        "vocab_size": 64,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "attention_bias": False,
    }
    _set_worker_runtime(
        monkeypatch,
        cuda_graph=False,
        cuda_graph_warmup=False,
        prefill_cuda_graph=True,
        prefill_cuda_graph_warmup=False,
        prefill_cuda_graph_warmup_tokens=(8, 16),
    )
    monkeypatch.setenv("UNISERVE_FORWARD_METRICS", "1")

    torch.manual_seed(778)
    eager_model = Qwen3ForCausalLM(config=cfg).cuda()
    graph_model = Qwen3ForCausalLM(config=cfg).cuda()
    graph_model.load_state_dict(eager_model.state_dict())
    requests = (
        (1, [1, 2, 3, 4, 5], [1]),
        (2, [6, 7, 8, 9, 10, 11, 12, 13, 14], [2]),
        (3, [15, 16, 17, 18, 19, 20, 21], [3]),
    )

    def run(model, *, graph_enabled: bool):
        _set_worker_runtime(monkeypatch, prefill_cuda_graph=graph_enabled)
        driver = RunnerDriver(model, block_size=256, kv_token_capacity=2048)
        outputs = []
        for step, tokens, blocks in requests:
            result = driver.execute(
                {
                    "step_id": step,
                    "new_reqs": [{"req_id": step, "sampling": {"temperature": 0.0}, "block_ids": []}],
                    "ops": [
                        {
                            "kind": "prefill_und",
                            "req_id": step,
                            "token_ids": tokens,
                            "pos_range": [0, len(tokens)],
                            "new_block_ids": blocks,
                        }
                    ],
                }
            )
            assert driver.runner.request_states.get(step).kv_length("qwen3") == len(tokens)
            outputs.append((result["per_seq"][0]["sampled_token_id"], result["forward_stats"]))
        torch.cuda.synchronize()
        return outputs

    eager = run(eager_model, graph_enabled=False)
    graph = run(graph_model, graph_enabled=True)

    assert [row[0] for row in graph] == [row[0] for row in eager]
    assert sorted(graph_model._prefill_graphs) == [(8, 1), (16, 1)]
    assert (
        graph_model._prefill_graphs[(8, 1)].query_lens.data_ptr()
        != graph_model._prefill_graphs[(16, 1)].query_lens.data_ptr()
    )
    assert graph[-1][1]["cuda_graph_captures"] == 0
    assert graph[-1][1]["cuda_graph_replays"] == 1
    assert graph[-1][1]["cuda_graph_unpadded_tokens"] == 7
    assert graph[-1][1]["cuda_graph_padded_tokens"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen3_initial_prefill_cuda_graph_replays_multi_row_initial_prefill(monkeypatch):
    pytest.importorskip("sgl_kernel")
    from uniserve_worker.models.qwen3 import Qwen3ForCausalLM

    cfg = {
        "vocab_size": 64,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "attention_bias": False,
    }
    _set_worker_runtime(
        monkeypatch,
        cuda_graph=False,
        cuda_graph_warmup=False,
        prefill_cuda_graph=True,
        prefill_cuda_graph_warmup=False,
        prefill_cuda_graph_warmup_tokens=(8,),
    )
    monkeypatch.setenv("UNISERVE_FORWARD_METRICS", "1")

    torch.manual_seed(779)
    eager_model = Qwen3ForCausalLM(config=cfg).cuda()
    graph_model = Qwen3ForCausalLM(config=cfg).cuda()
    graph_model.load_state_dict(eager_model.state_dict())
    batches = (
        (
            (1, [1, 2, 3], [1]),
            (2, [4, 5], [2]),
        ),
        (
            (3, [6, 7], [3]),
            (4, [8, 9, 10], [4]),
        ),
    )

    def run(model, *, graph_enabled: bool):
        _set_worker_runtime(monkeypatch, prefill_cuda_graph=graph_enabled)
        driver = RunnerDriver(model, block_size=256, kv_token_capacity=2048)
        rows = []
        for step, batch_rows in enumerate(batches, start=1):
            result = driver.execute(
                {
                    "step_id": step,
                    "new_reqs": [
                        {"req_id": req_id, "sampling": {"temperature": 0.0}, "block_ids": []}
                        for req_id, _tokens, _blocks in batch_rows
                    ],
                    "ops": [
                        {
                            "kind": "prefill_und",
                            "req_id": req_id,
                            "token_ids": tokens,
                            "pos_range": [0, len(tokens)],
                            "new_block_ids": blocks,
                        }
                        for req_id, tokens, blocks in batch_rows
                    ],
                }
            )
            for req_id, tokens, _blocks in batch_rows:
                assert driver.runner.request_states.get(req_id).kv_length("qwen3") == len(tokens)
            rows.extend((row["req_id"], row["sampled_token_id"]) for row in result["per_seq"])
            stats = result["forward_stats"]
        torch.cuda.synchronize()
        return rows, stats

    eager_rows, _ = run(eager_model, graph_enabled=False)
    graph_rows, stats = run(graph_model, graph_enabled=True)

    assert graph_rows == eager_rows
    assert sorted(graph_model._prefill_graphs) == [(8, 2)]
    assert graph_model._prefill_graph_disabled == set()
    assert stats["cuda_graph_captures"] == 0
    assert stats["cuda_graph_replays"] == 1
    assert stats["cuda_graph_unpadded_tokens"] == 5
    assert stats["cuda_graph_padded_tokens"] == 3
    assert graph_model._prefill_graphs[(8, 2)].query_lens.tolist() == [2, 6]


def test_zero_day_diffusion_model_uses_shared_cfg_zero_star_path():
    from tests.python.fixtures.zero_day import UniServeZeroDayCfgZeroStarModel

    cls = UniServeZeroDayCfgZeroStarModel
    driver = RunnerDriver(cls(), block_size=256)
    result = driver.execute(
        {
            "step_id": 9,
            "new_reqs": [{"req_id": 11, "sampling": {}, "image": {"steps": 1}}],
            "ops": [
                {
                    "kind": "denoise_gen",
                    "req_id": 11,
                    "timestep_idx": 0,
                    "num_steps": 1,
                    "latent_shape": [1, 2, 2],
                    "cfg": {
                        "branch_count": 2,
                        "text_scale": 2.0,
                        "renorm_type": "cfg_zero_star",
                    },
                }
            ],
        }
    )
    assert result["step_id"] == 9
    assert result["per_seq"] == [
        {"req_id": 11, "denoise_done": True, "num_steps_done": 1}
    ]


class TinyLoadableModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = LinearBase(3, 2, bias=True)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        seen = set()
        for name, tensor in weights:
            param = params[name]
            param.weight_loader(param, tensor)
            seen.add(name)
        missing = set(params) - seen
        unexpected = seen - set(params)
        if missing or unexpected:
            return type("LoadResult", (), {"missing": missing, "unexpected": unexpected})()
        return None


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


def test_native_loader_class_orchestrates_spec_and_from_native(tmp_path):
    # The registered "native" BaseModelLoader reads the wrapper's native_load_spec,
    # materializes the inner module, and builds the serving model via from_native,
    # returning a LoadResult(model + tokenizer + device).
    from uniserve_worker.loader import LoadResult, NativeLoadSpec, get_loader

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
        def __init__(self, inner, tokenizer, device):
            self.model = inner
            self.tokenizer = tokenizer
            self.device = device

        @classmethod
        def native_load_spec(cls):
            return NativeLoadSpec(
                config_cls=TinyConfig,
                model_cls=TinyNativeModel,
                tokenizer_cls=TinyTokenizer,
            )

        @classmethod
        def from_native(cls, inner, *, tokenizer, device, **kwargs):
            del kwargs
            return cls(inner, tokenizer, device)

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
    torch.testing.assert_close(model.embed_tokens.weight[5:], torch.zeros(3, 3, dtype=torch.bfloat16))
    torch.testing.assert_close(model.lm_head.weight[:5], (table + 10).to(torch.bfloat16))
    torch.testing.assert_close(model.lm_head.weight[5:], torch.zeros(3, 3, dtype=torch.bfloat16))

    ids = torch.tensor([[0, 4, 2]], dtype=torch.long)
    hidden = torch.randn(2, 3, dtype=torch.bfloat16)
    torch.testing.assert_close(model.embed_tokens(ids), F.embedding(ids, table.to(torch.bfloat16)))
    torch.testing.assert_close(model.lm_head(hidden), F.linear(hidden, (table + 10).to(torch.bfloat16)))


def test_weight_utils_reads_safetensors_tensor_shape_without_loading_tensor(tmp_path):
    save_file({"latent_pos_embed.pos_embed": torch.zeros(4096, 8)}, tmp_path / "model.safetensors")
    assert tensor_shape(tmp_path / "model.safetensors", "latent_pos_embed.pos_embed") == (4096, 8)


def test_merge_on_load_lora_round_trips_exact_weight_delta(tmp_path):
    model = nn.Sequential()
    model.add_module("proj", nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model.proj.weight.zero_()
    adapter = tmp_path / "adapter_model.safetensors"
    save_file(
        {
            "base_model.model.proj.lora_A.weight": torch.tensor([[1.0, 2.0]]),
            "base_model.model.proj.lora_B.weight": torch.tensor([[3.0], [4.0]]),
        },
        adapter,
    )
    (tmp_path / "adapter_config.json").write_text('{"r": 1, "lora_alpha": 2}', encoding="utf-8")

    lora = MergeOnLoadLoRA(model)
    assert lora.load(7, str(tmp_path)) == 1
    expected = torch.tensor([[6.0, 12.0], [8.0, 16.0]])
    torch.testing.assert_close(model.proj.weight, expected)
    assert lora.unload(7) == 1
    torch.testing.assert_close(model.proj.weight, torch.zeros_like(model.proj.weight))
