"""Observable checkpoint loading, placement, identity, and update behavior."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from uniserve_worker.bootstrap.catalog import resolve_catalog_entry
from uniserve_worker.bootstrap.execution_config import ExecutionConfig
from uniserve_worker.bootstrap.model_loader import WorkerModelLoadRequest, load_worker_model
from uniserve_worker.bootstrap.plan import ModelLoadScope
from uniserve_worker.loader import LoadConfig, LoadFormat, LoadRequest, WeightSet, get_model_loader
from uniserve_worker.loader.handles import TensorWeightHandle
from uniserve_worker.loader.source import resolve_model_root, resolve_weight_sources
from uniserve_worker.loader.update import BucketTensor, WeightUpdater
from uniserve_worker.loader.weight_loaders import attach_parameter_loaders, load_parameter_weight
from uniserve_worker.models.qwen3 import Qwen3ForCausalLM
from uniserve_worker.nn.layer import LayerSpec
from uniserve_worker.nn.linear import ColumnParallelLinear, QKVParallelLinear
from uniserve_worker.nn.mesh import TensorParallelSpec
from uniserve_worker.nn.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    zero_vocab_padding,
)

pytestmark = pytest.mark.integration


def _parallel(rank: int = 0, size: int = 1) -> TensorParallelSpec:
    return TensorParallelSpec(rank=rank, size=size)


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


def _sense_config() -> dict[str, object]:
    return {
        "architectures": ["NEOChatModel"],
        "vision_config": {
            "hidden_size": 8,
            "llm_hidden_size": 8,
            "downsample_ratio": 0.5,
            "patch_size": 2,
            "num_channels": 3,
            "rope_theta_vision": 10_000.0,
            "max_position_embeddings_vision": 128,
        },
        "llm_config": {
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "attention_bias": False,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000.0,
            "max_position_embeddings": 128,
            "rope_theta_hw": 10_000.0,
            "max_position_embeddings_hw": 128,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        "downsample_ratio": 0.5,
        "max_image_seq_len": 16,
        "fm_head_layers": 2,
    }


def _qwen_reference(config: dict[str, object]) -> Qwen3ForCausalLM:
    model = Qwen3ForCausalLM(
        config,
        layer_spec=LayerSpec(parallel=_parallel(), quantization=None),
    )
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters(), start=1):
            parameter.fill_(index / 16)
        for module in model.modules():
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
    return model


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


def _qwen_request(
    path: str,
    *,
    load: LoadConfig = LoadConfig(),
    scope: ModelLoadScope = ModelLoadScope.WHOLE,
) -> WorkerModelLoadRequest:
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
        load=load,
    )


def _write_qwen_checkpoint(root: Path, *, indexed: bool = False) -> Qwen3ForCausalLM:
    config = _qwen_config()
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    reference = _qwen_reference(config)
    checkpoint = _qwen_hugging_face_weights(reference)
    filename = "model-00001-of-00001.safetensors" if indexed else "model.safetensors"
    save_file(checkpoint, root / filename)
    if indexed:
        index = {"weight_map": {name: filename for name in checkpoint}}
        (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return reference


def test_indexed_qwen_checkpoint_installs_packed_weights_on_the_requested_device(tmp_path):
    reference = _write_qwen_checkpoint(tmp_path, indexed=True)

    loaded = load_worker_model(_qwen_request(str(tmp_path)))

    assert loaded.identity.architecture == "Qwen3ForCausalLM"
    assert loaded.weights.version == 0
    assert loaded.weights.digest == loaded.identity.weight_digest
    assert {parameter.device.type for parameter in loaded.model.parameters()} == {"cpu"}
    for name, parameter in loaded.model.state_dict().items():
        torch.testing.assert_close(parameter, reference.state_dict()[name].to(torch.bfloat16))


def test_index_is_the_closed_weight_set_for_failures_and_identity(tmp_path):
    _write_qwen_checkpoint(tmp_path, indexed=True)
    first = load_worker_model(_qwen_request(str(tmp_path))).identity.weight_digest
    save_file({"unused": torch.tensor([1.0])}, tmp_path / "extra.safetensors")
    (tmp_path / "unused.pth").write_bytes(b"not a checkpoint")
    second = load_worker_model(_qwen_request(str(tmp_path))).identity.weight_digest
    assert second == first

    shard = tmp_path / "model-00001-of-00001.safetensors"
    changed = _qwen_hugging_face_weights(_qwen_reference(_qwen_config()))
    changed["model.norm.weight"] = changed["model.norm.weight"] + 1
    save_file(changed, shard)
    third = load_worker_model(_qwen_request(str(tmp_path))).identity.weight_digest
    assert third != first

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    missing_name = "model-00002-of-00002.safetensors"
    index["weight_map"]["model.norm.weight"] = missing_name
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=missing_name):
        load_worker_model(_qwen_request(str(tmp_path)))


def test_checksum_manifest_gates_checkpoint_materialization(tmp_path):
    _write_qwen_checkpoint(tmp_path)
    checkpoint = tmp_path / "model.safetensors"
    manifest = tmp_path / "checksums.json"
    manifest.write_text(
        json.dumps({checkpoint.name: hashlib.sha256(checkpoint.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    load = LoadConfig(checksum_manifest=str(manifest))

    load_worker_model(_qwen_request(str(tmp_path), load=load))

    manifest.write_text(json.dumps({checkpoint.name: "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_worker_model(_qwen_request(str(tmp_path), load=load))


def test_pt_index_selects_only_its_declared_shards(tmp_path):
    config = _qwen_config()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    reference = _qwen_reference(config)
    checkpoint = _qwen_hugging_face_weights(reference)
    names = sorted(checkpoint)
    midpoint = len(names) // 2
    shards = (
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model-00002-of-00002.bin",
    )
    torch.save({name: checkpoint[name] for name in names[:midpoint]}, tmp_path / shards[0])
    torch.save({name: checkpoint[name] for name in names[midpoint:]}, tmp_path / shards[1])
    weight_map = {
        name: shards[0] if index < midpoint else shards[1]
        for index, name in enumerate(names)
    }
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )
    (tmp_path / "extra.bin").write_bytes(b"unread")

    loaded = load_worker_model(
        _qwen_request(str(tmp_path), load=LoadConfig(load_format=LoadFormat.PT))
    )

    for name, value in loaded.model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name].to(torch.bfloat16))


def test_remote_resolution_fetches_only_weights_index_and_architecture_sidecars(
    tmp_path, monkeypatch
):
    remote = tmp_path / "remote"
    cache = tmp_path / "cache"
    remote.mkdir()
    config = _qwen_config()
    (remote / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (remote / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.norm.weight": "weights.safetensors"}}),
        encoding="utf-8",
    )
    save_file({"model.norm.weight": torch.ones(8)}, remote / "weights.safetensors")
    (remote / "README.md").write_text("unrelated", encoding="utf-8")
    requested: list[str] = []

    def download(*, repo_id, filename, cache_dir, revision):
        del repo_id, cache_dir, revision
        requested.append(filename)
        target = cache / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote / filename, target)
        return str(target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr(
        "huggingface_hub.HfApi.list_repo_files",
        lambda self, repo_id, revision: [
            "README.md",
            "config.json",
            "model.safetensors.index.json",
            "weights.safetensors",
        ],
    )
    load = LoadConfig(download_dir=str(cache))
    root, repository_id = resolve_model_root("owner/model", load)
    request = LoadRequest(
        model_path="owner/model",
        device="cpu",
        execution=_execution(),
        parallel=_parallel(),
        scope=ModelLoadScope.WHOLE,
        load=load,
    )

    sources = resolve_weight_sources(
        request,
        architecture="Qwen3ForCausalLM",
        sidecars=resolve_catalog_entry(("Qwen3ForCausalLM",)).sidecars,
        root=root,
        repository_id=repository_id,
    )

    assert sources[0].relative_paths == ("weights.safetensors",)
    assert set(requested) == {
        "config.json",
        "model.safetensors.index.json",
        "weights.safetensors",
    }


def test_partition_loaders_copy_rank_slices_packed_slots_and_vocab_overlap():
    spec = LayerSpec(parallel=_parallel(rank=1, size=2), quantization=None)
    column = ColumnParallelLinear(4, 6, spec=spec, bias=False)
    qkv = QKVParallelLinear(4, 2, 2, 2, spec=spec, bias=False)
    vocab = VocabParallelEmbedding(65, 2, spec=spec, init_weights=False)
    graph = torch.nn.ModuleDict({"column": column, "qkv": qkv, "vocab": vocab})
    attach_parameter_loaders(graph, device="cpu", dtype=torch.float32)

    global_column = torch.arange(24, dtype=torch.float32).view(6, 4)
    load_parameter_weight(column.weight, TensorWeightHandle("column.weight", global_column))
    torch.testing.assert_close(column.weight, global_column[3:])

    projections = {
        "q": torch.arange(16, dtype=torch.float32).view(4, 4),
        "k": torch.arange(16, 32, dtype=torch.float32).view(4, 4),
        "v": torch.arange(32, 48, dtype=torch.float32).view(4, 4),
    }
    for shard_id, tensor in projections.items():
        load_parameter_weight(qkv.weight, TensorWeightHandle(shard_id, tensor), shard_id)
    torch.testing.assert_close(qkv.weight, torch.cat([value[2:] for value in projections.values()]))

    global_vocab = torch.arange(130, dtype=torch.float32).view(65, 2)
    load_parameter_weight(vocab.weight, TensorWeightHandle("vocab.weight", global_vocab))
    torch.testing.assert_close(vocab.weight[0], global_vocab[64])
    assert torch.count_nonzero(vocab.weight[1:]) == 0


def test_missing_packed_shard_and_unexpected_tensor_fail_completeness(tmp_path):
    config = _qwen_config()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    checkpoint = _qwen_hugging_face_weights(_qwen_reference(config))
    checkpoint.pop("model.layers.0.self_attn.k_proj.weight")
    checkpoint["unknown.weight"] = torch.ones(1)
    save_file(checkpoint, tmp_path / "model.safetensors")

    with pytest.raises(RuntimeError, match="packed shards.*unexpected=1"):
        load_worker_model(_qwen_request(str(tmp_path)))


def test_dummy_load_is_deterministic_and_does_not_read_weight_bytes(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_qwen_config()), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"invalid checkpoint bytes")
    load = LoadConfig(load_format=LoadFormat.DUMMY)

    first = load_worker_model(_qwen_request(str(tmp_path), load=load))
    second = load_worker_model(_qwen_request(str(tmp_path), load=load))

    assert first.weights.digest == second.weights.digest
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[name])


def test_sharded_state_loads_installed_names_without_hugging_face_remapping(tmp_path):
    config = _qwen_config()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    reference = _qwen_reference(config)
    save_file(
        {name: parameter.detach().contiguous() for name, parameter in reference.named_parameters()},
        tmp_path / "rank-00000-of-00001.safetensors",
    )
    load = LoadConfig(load_format=LoadFormat.SHARDED_STATE)

    loaded = load_worker_model(_qwen_request(str(tmp_path), load=load))

    for name, value in loaded.model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name].to(torch.bfloat16))


def test_layered_load_materializes_the_complete_graph_from_file_backed_handles(tmp_path):
    reference = _write_qwen_checkpoint(tmp_path, indexed=True)
    load = LoadConfig(load_format=LoadFormat.LAYERED)

    loaded = load_worker_model(_qwen_request(str(tmp_path), load=load))

    assert {parameter.device.type for parameter in loaded.model.parameters()} == {"cpu"}
    for name, value in loaded.model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name].to(torch.bfloat16))


def test_sensenova_dummy_scopes_materialize_only_selected_parameters(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    config = _sense_config()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    entry = resolve_catalog_entry(("NEOChatModel",))
    load = LoadConfig(load_format=LoadFormat.DUMMY)
    for scope in (ModelLoadScope.UNDERSTANDING, ModelLoadScope.GENERATION):
        request = LoadRequest(
            model_path=str(tmp_path),
            device="cpu",
            execution=_execution(),
            parallel=_parallel(),
            scope=scope,
            load=load,
        )
        loaded = get_model_loader(load.load_format).load(
            entry,
            config,
            request,
            root=tmp_path,
            repository_id=None,
        )
        for name, parameter in loaded.model.named_parameters():
            generation = name.startswith("fm_modules.") or "_mot_gen." in name
            selected = generation if scope is ModelLoadScope.GENERATION else not generation
            assert parameter.device.type == ("cpu" if selected else "meta")
        for name, buffer in loaded.model.named_buffers():
            generation_buffer = name.startswith("fm_modules.")
            shared_language_buffer = name.startswith("language_model.")
            selected = (
                generation_buffer or shared_language_buffer
                if scope is ModelLoadScope.GENERATION
                else not generation_buffer
            )
            assert buffer.device.type == ("cpu" if selected else "meta")


def test_weight_update_publishes_identity_and_rolls_back_partial_failure():
    model = _qwen_reference(_qwen_config())
    attach_parameter_loaders(model, device="cpu", dtype=torch.float32)
    initial = WeightSet.from_module(model)
    updater = WeightUpdater(
        model,
        architecture="Qwen3ForCausalLM",
        scope="whole",
        weights=initial,
    )
    replacement = torch.full_like(model.model.norm.weight, 3)

    current = updater.update_named(
        {"model.norm.weight": replacement},
        expected_parameters={"model.norm.weight"},
    )

    assert current.version == 1
    assert current.digest != initial.digest
    assert current.tensors["model.norm.weight"].data_ptr() == model.model.norm.weight.data_ptr()
    torch.testing.assert_close(model.model.norm.weight, replacement)

    with pytest.raises(RuntimeError, match="missing=1"):
        updater.update_named(
            {"model.norm.weight": torch.full_like(replacement, 9)},
            expected_parameters={"model.norm.weight", "lm_head.weight"},
        )
    assert updater.weights is current
    torch.testing.assert_close(model.model.norm.weight, replacement)

    flattened = torch.arange(replacement.numel(), dtype=replacement.dtype)
    installed = updater.update_flattened(
        flattened,
        (
            BucketTensor(
                name="model.norm.weight",
                shape=tuple(replacement.shape),
                offset=0,
                length=replacement.numel(),
            ),
        ),
        expected_parameters={"model.norm.weight"},
    )
    assert installed.version == 2
    torch.testing.assert_close(model.model.norm.weight, flattened.view_as(replacement))

    with pytest.raises(ValueError, match="repeats tensor"):
        updater.update_distributed(
            (
                ("model.norm.weight", replacement),
                ("model.norm.weight", replacement),
            ),
            expected_parameters={"model.norm.weight"},
        )


def test_online_fp8_weights_remain_replaceable_after_post_load_finalize(tmp_path):
    config = _qwen_config()
    config["quantization_config"] = {"quant_method": "fp8"}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    reference = _qwen_reference(config)
    save_file(_qwen_hugging_face_weights(reference), tmp_path / "model.safetensors")
    loaded = load_worker_model(
        _qwen_request(str(tmp_path), load=LoadConfig(load_format=LoadFormat.LAYERED))
    )
    updater = WeightUpdater(
        loaded.model,
        architecture="Qwen3ForCausalLM",
        scope="whole",
        weights=loaded.weights,
    )
    name = "model.layers.0.self_attn.o_proj.weight"
    shape = tuple(dict(loaded.model.named_parameters())[name].shape)

    first = updater.update_named(
        {name: torch.full(shape, 0.25, dtype=torch.bfloat16)},
        expected_parameters={name},
    )
    second = updater.update_named(
        {name: torch.full(shape, 0.5, dtype=torch.bfloat16)},
        expected_parameters={name},
    )

    assert first.version == 1
    assert second.version == 2
    assert dict(loaded.model.named_parameters())[name].dtype == torch.float8_e4m3fn


def test_qwen_rejects_partial_model_materialization(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_qwen_config()), encoding="utf-8")
    with pytest.raises(Exception, match="does not support 'generation'"):
        load_worker_model(
            _qwen_request(str(tmp_path), scope=ModelLoadScope.GENERATION)
        )
