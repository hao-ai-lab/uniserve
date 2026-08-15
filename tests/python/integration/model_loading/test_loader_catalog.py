"""Configured architecture and checkpoint loading behavior."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from uniserve_worker.bootstrap.model_loader import WorkerModelLoadRequest, load_worker_model
from uniserve_worker.bootstrap.plan import ModelLoadScope
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.foundation.runtime_config import ExecutionConfig
from uniserve_worker.models.qwen3 import Qwen3ForCausalLM
from uniserve_worker.nn.layer import LayerSpec
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
    return ExecutionConfig(model_dtype=dtype, cuda_graph=False, prefill_cuda_graph=False)


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


def _qwen_hugging_face_weights(model: Qwen3ForCausalLM) -> dict[str, torch.Tensor]:
    checkpoint: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        tensor = value.detach().contiguous()
        if ".qkv_proj." in name:
            query, key, value_part = tensor.split((8, 4, 4), dim=0)
            checkpoint[name.replace("qkv_proj", "q_proj")] = query.contiguous()
            checkpoint[name.replace("qkv_proj", "k_proj")] = key.contiguous()
            checkpoint[name.replace("qkv_proj", "v_proj")] = value_part.contiguous()
        elif ".gate_up_proj." in name:
            gate, up = tensor.chunk(2, dim=0)
            checkpoint[name.replace("gate_up_proj", "gate_proj")] = gate.contiguous()
            checkpoint[name.replace("gate_up_proj", "up_proj")] = up.contiguous()
        elif name in {"model.embed_tokens.weight", "lm_head.weight"}:
            checkpoint[name] = tensor[:32].contiguous()
        else:
            checkpoint[name] = tensor
    return checkpoint


def _qwen_request(path: str, *, scope: ModelLoadScope = ModelLoadScope.WHOLE) -> WorkerModelLoadRequest:
    return WorkerModelLoadRequest(
        model_path=path,
        device="cpu",
        block_size=16,
        max_batch_tokens=4096,
        kv_token_capacity=64,
        attention_backend="torch_sdpa",
        execution=_execution("bfloat16"),
        parallel=_parallel(),
        scope=scope,
    )


def test_qwen_hugging_face_checkpoint_materializes_a_ready_worker_model(tmp_path):
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
    checkpoint = _qwen_hugging_face_weights(reference)
    save_file(checkpoint, tmp_path / "model.safetensors")

    loaded = load_worker_model(_qwen_request(str(tmp_path)))

    assert loaded.identity.architecture == "Qwen3ForCausalLM"
    assert len(loaded.identity.architecture_digest) == 64
    assert len(loaded.identity.weight_digest) == 64
    assert loaded.tokenizer is None
    assert loaded.model.image_processor is None
    assert loaded.deployment.max_batch_tokens == 4096
    expected = reference.state_dict()
    for name, parameter in loaded.model.state_dict().items():
        torch.testing.assert_close(parameter, expected[name].to(torch.bfloat16))

    changed = dict(checkpoint)
    first = next(iter(changed))
    changed[first] = changed[first].clone()
    changed[first].view(-1)[0] += 1
    save_file(changed, tmp_path / "model.safetensors")
    assert (
        load_worker_model(_qwen_request(str(tmp_path))).identity.weight_digest
        != loaded.identity.weight_digest
    )


def test_qwen_checkpoint_requires_every_model_parameter(tmp_path):
    config = _qwen_config()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file({"model.norm.weight": torch.ones(8)}, tmp_path / "model.safetensors")

    with pytest.raises(ValueError, match="checkpoint load mismatch"):
        load_worker_model(_qwen_request(str(tmp_path)))


def test_qwen_rejects_partial_model_materialization(tmp_path):
    config = _qwen_config()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(WorkerError, match="does not support 'generation'"):
        load_worker_model(
            _qwen_request(str(tmp_path), scope=ModelLoadScope.GENERATION)
        )
