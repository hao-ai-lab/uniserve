"""Observable checkpoint loading, params, identity, and update behavior."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests.python.fixtures.model_execution import model_context, tensor_parallel_bindings
from uniserve_worker.bootstrap.catalog import resolve_catalog_entry
from uniserve_worker.bootstrap.worker_info_builder import (
    build_worker_layout,
    configuration_identity,
)
from uniserve_worker.config import WorkerConfig
from uniserve_worker.loader import (
    LoadConfig,
    LoadFormat,
    LoadRequest,
    get_model_loader,
    load_model,
)
from uniserve_worker.loader.source import (
    read_model_config,
    resolve_model_root,
    resolve_weight_sources,
)
from uniserve_worker.loader.weight_loaders import attach_parameter_loaders
from uniserve_worker.models.qwen3 import Qwen3ForCausalLM
from uniserve_worker.nn.layer import LayerConfig
from uniserve_worker.nn.mesh import Communicator
from uniserve_worker.nn.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    zero_vocab_padding,
)

pytestmark = pytest.mark.integration


def _execution(dtype: str = "float32") -> WorkerConfig:
    return WorkerConfig(model_dtype=dtype, graph_policy="off", prefill_cuda_graph=False)


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
        context=model_context(LayerConfig(communicator=Communicator(), quantization=None)),
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
) -> LoadRequest:
    return LoadRequest(
        model_path=path,
        execution=replace(
            _execution("bfloat16"),
            attention_backend="torch_sdpa",
            block_size=16,
            kv_token_capacity=64,
            max_batch_operations=4,
            max_batch_tokens=4096,
        ),
        bindings=tensor_parallel_bindings(),
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


def test_modular_h3_manifest_resolves_through_the_model_catalog(tmp_path):
    (tmp_path / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ModularPipeline"}),
        encoding="utf-8",
    )

    config = read_model_config(tmp_path)
    entry = resolve_catalog_entry(tuple(config["architectures"]))

    assert entry.architecture == "MiniMaxH3Transformer3DModel"


def test_unknown_modular_pipeline_is_rejected_at_discovery(tmp_path):
    (tmp_path / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "UnknownPipeline"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported pipeline"):
        read_model_config(tmp_path)


def test_indexed_qwen_checkpoint_installs_packed_weights_on_the_requested_device(tmp_path):
    reference = _write_qwen_checkpoint(tmp_path, indexed=True)

    loaded = load_model(_qwen_request(str(tmp_path)))

    assert loaded.model.architecture == "Qwen3ForCausalLM"
    assert {parameter.device.type for parameter in loaded.model.parameters()} == {"cpu"}
    for name, parameter in loaded.model.state_dict().items():
        torch.testing.assert_close(parameter, reference.state_dict()[name].to(torch.bfloat16))


@pytest.mark.parametrize(
    ("ignored", "dense"),
    [
        (
            ["model.layers.0.self_attn.o_proj", "model.layers.0.mlp"],
            ["self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"],
        ),
        (
            [
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.v_proj",
                "model.layers.0.mlp.gate_proj",
                "model.layers.0.mlp.up_proj",
            ],
            ["self_attn.qkv_proj", "mlp.gate_up_proj"],
        ),
    ],
)
@pytest.mark.parametrize("configuration_source", ["checkpoint", "worker"])
def test_qwen_quantization_honors_excluded_layers(tmp_path, ignored, dense, configuration_source):
    reference = _write_qwen_checkpoint(tmp_path)
    config = _qwen_config()
    quantization = {"quant_method": "fp8", "ignored_layers": ignored}
    request = _qwen_request(str(tmp_path))
    if configuration_source == "checkpoint":
        config["quantization_config"] = quantization
        (tmp_path / "config.json").write_text(json.dumps(config))
    else:
        request = replace(request, quantization_config=quantization)
    loaded = load_model(request)
    exported = loaded.model.state_dict()
    for projection in (
        "self_attn.qkv_proj",
        "self_attn.o_proj",
        "mlp.gate_up_proj",
        "mlp.down_proj",
    ):
        name = f"model.layers.0.{projection}.weight"
        if projection in dense:
            torch.testing.assert_close(exported[name], reference.state_dict()[name].bfloat16())
        else:
            assert exported[name].dtype == torch.float8_e4m3fn


@pytest.mark.parametrize("method", ["fp8", "unquantized"])
def test_qwen_packed_projection_requires_one_precision(tmp_path, method):
    reference = _write_qwen_checkpoint(tmp_path)
    config = _qwen_config()
    config["quantization_config"] = {
        "quant_method": method,
        "ignored_layers": ["model.layers.0.self_attn.q_proj"],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    if method == "fp8":
        with pytest.raises(ValueError, match="same precision"):
            load_model(_qwen_request(str(tmp_path)))
    else:
        loaded = load_model(_qwen_request(str(tmp_path)))
        for name, parameter in loaded.model.state_dict().items():
            torch.testing.assert_close(parameter, reference.state_dict()[name].bfloat16())


def test_resolved_configuration_identity_distinguishes_loaded_precision(tmp_path):
    _write_qwen_checkpoint(tmp_path)
    identities = []
    for dtype in ("bfloat16", "float32", "bfloat16"):
        request = replace(_qwen_request(str(tmp_path)), execution=_execution(dtype))
        loaded = load_model(request)
        layout = build_worker_layout(loaded.model, loaded.worker_config, bindings=loaded.bindings)
        identities.append(
            configuration_identity(loaded.model, loaded.worker_config, layout, (), "torch_sdpa")
        )

    assert identities[0] == identities[2]
    assert identities[0] != identities[1]


def test_index_is_the_closed_weight_set_for_loading(tmp_path):
    _write_qwen_checkpoint(tmp_path, indexed=True)
    load_model(_qwen_request(str(tmp_path)))
    save_file({"unused": torch.tensor([1.0])}, tmp_path / "extra.safetensors")
    (tmp_path / "unused.pth").write_bytes(b"not a checkpoint")
    load_model(_qwen_request(str(tmp_path)))

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    missing_name = "model-00002-of-00002.safetensors"
    index["weight_map"]["model.norm.weight"] = missing_name
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=missing_name):
        load_model(_qwen_request(str(tmp_path)))


def test_checksum_manifest_gates_checkpoint_materialization(tmp_path):
    _write_qwen_checkpoint(tmp_path)
    checkpoint = tmp_path / "model.safetensors"
    manifest = tmp_path / "checksums.json"
    manifest.write_text(
        json.dumps({checkpoint.name: hashlib.sha256(checkpoint.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    load = LoadConfig(checksum_manifest=str(manifest))

    load_model(_qwen_request(str(tmp_path), load=load))

    manifest.write_text(json.dumps({checkpoint.name: "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_model(_qwen_request(str(tmp_path), load=load))


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
        name: shards[0] if index < midpoint else shards[1] for index, name in enumerate(names)
    }
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )
    (tmp_path / "extra.bin").write_bytes(b"unread")

    loaded = load_model(_qwen_request(str(tmp_path), load=LoadConfig(load_format=LoadFormat.PT)))

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
        execution=_execution(),
        bindings=tensor_parallel_bindings(),
        load=load,
    )

    sources = resolve_weight_sources(
        request,
        sources=resolve_catalog_entry(("Qwen3ForCausalLM",)).sources,
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


def test_missing_packed_shard_and_unexpected_tensor_fail_completeness(tmp_path):
    config = _qwen_config()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    checkpoint = _qwen_hugging_face_weights(_qwen_reference(config))
    checkpoint.pop("model.layers.0.self_attn.k_proj.weight")
    checkpoint["unknown.weight"] = torch.ones(1)
    save_file(checkpoint, tmp_path / "model.safetensors")

    with pytest.raises(RuntimeError, match="packed shards.*unexpected=1"):
        load_model(_qwen_request(str(tmp_path)))


def test_dummy_load_is_deterministic_and_does_not_read_weight_bytes(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_qwen_config()), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"invalid checkpoint bytes")
    load = LoadConfig(load_format=LoadFormat.DUMMY)

    first = load_model(_qwen_request(str(tmp_path), load=load))
    second = load_model(_qwen_request(str(tmp_path), load=load))

    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[name])


def test_layered_load_materializes_the_complete_graph_from_file_backed_handles(tmp_path):
    reference = _write_qwen_checkpoint(tmp_path, indexed=True)
    load = LoadConfig(load_format=LoadFormat.LAYERED)

    loaded = load_model(_qwen_request(str(tmp_path), load=load))

    assert {parameter.device.type for parameter in loaded.model.parameters()} == {"cpu"}
    for name, value in loaded.model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name].to(torch.bfloat16))


@pytest.mark.parametrize("branch", ["text", "flow"])
@pytest.mark.parametrize("load_format", [LoadFormat.AUTO, LoadFormat.LAYERED])
def test_sensenova_checkpoint_component_materializes_selected_expert_parameters(
    tmp_path, branch, load_format
):
    from uniserve_worker.loader.loader import load_component
    from uniserve_worker.loader.source import WeightSourceSet
    from uniserve_worker.models.sensenova.config import NeoChatConfig
    from uniserve_worker.models.sensenova.model import NEOChatModel

    config = NeoChatConfig.from_dict(_sense_config())
    layers = LayerConfig(Communicator(), None)
    reference = NEOChatModel(config, context=model_context(layers))
    with torch.no_grad():
        for index, parameter in enumerate(reference.parameters(), start=1):
            parameter.fill_(index / 37)
        head = reference.language_model.lm_head
        zero_vocab_padding(head.vocab_size, head.vocab_start_index, head.output_size, head.weight)
        embedding = reference.language_model.model.embed_tokens
        zero_vocab_padding(
            embedding.num_embeddings,
            embedding.vocab_start_index,
            embedding.num_embeddings_per_partition,
            embedding.weight,
        )
    expected = reference.state_dict()
    checkpoint = {}
    for name, value in expected.items():
        if ".text_qkv.projection." in name or ".flow_qkv.projection." in name:
            packed = (
                "flow_qkv.projection" if ".flow_qkv.projection." in name else "text_qkv.projection"
            )
            suffix = "_mot_gen" if packed == "flow_qkv.projection" else ""
            for part, tensor in zip(("q", "k", "v"), value.split((8, 4, 4), dim=0), strict=True):
                checkpoint[name.replace(packed, f"{part}_proj{suffix}")] = tensor.contiguous()
        elif ".gate_up_proj." in name:
            for part, tensor in zip(("gate", "up"), value.chunk(2, dim=0), strict=True):
                checkpoint[name.replace("gate_up_proj", f"{part}_proj")] = tensor.contiguous()
        else:
            checkpoint[name] = value.contiguous()
    path = tmp_path / "model.safetensors"
    save_file(checkpoint, path)
    included = frozenset(
        name
        for name, _ in reference.named_parameters()
        if (name.startswith("fm_modules.") or "_mot_gen." in name or ".flow_qkv." in name)
        == (branch == "flow")
    )
    with torch.device("meta"):
        model = NEOChatModel(config, context=model_context(layers))
    attach_parameter_loaders(model, device="cpu", dtype=torch.float32)
    component = replace(model.checkpoint_components()[0], included=included)
    request = LoadRequest(
        model_path=str(tmp_path),
        execution=_execution(),
        bindings=tensor_parallel_bindings(),
        load=LoadConfig(load_format=load_format),
    )
    load_component(component, WeightSourceSet(tmp_path, (path,), (path.name,)), request)
    for name, parameter in model.named_parameters():
        if name in included:
            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
        else:
            assert parameter.device.type == "meta"
    for name, buffer in model.named_buffers():
        flow_buffer = name.startswith("fm_modules.")
        shared_language_buffer = name.startswith("language_model.")
        active = (flow_buffer or shared_language_buffer) if branch == "flow" else not flow_buffer
        assert buffer.device.type == ("cpu" if active else "meta")


@pytest.mark.parametrize(
    "generation_device",
    [
        None,
        pytest.param(
            "cuda:1",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    torch.cuda.device_count() < 2, reason="two CUDA devices are required"
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize("deep_head", [False, True])
def test_sensenova_checkpoint_layer_exclusions_preserve_projection_weights(
    tmp_path, monkeypatch, deep_head, generation_device
):
    from transformers import AutoTokenizer

    from uniserve_worker.models.sensenova.config import NeoChatConfig
    from uniserve_worker.models.sensenova.model import NEOChatModel

    config = _sense_config()
    if deep_head:
        config.update(fm_head_layers=3, fm_head_dim=8, fm_head_mlp_ratio=2.0)
    reference = NEOChatModel(
        NeoChatConfig.from_dict(config), context=model_context(LayerConfig(Communicator(), None))
    )
    with torch.no_grad():
        for index, parameter in enumerate(reference.parameters(), start=1):
            parameter.fill_(index / 37)
        head = reference.language_model.lm_head
        zero_vocab_padding(head.vocab_size, head.vocab_start_index, head.output_size, head.weight)
    state = reference.state_dict()
    save_file(state, tmp_path / "model.safetensors")
    ignored = [
        "language_model.model.layers.0.self_attn.o_proj",
        "language_model.model.layers.0.self_attn.q_proj_mot_gen",
        "language_model.model.layers.0.self_attn.k_proj_mot_gen",
        "language_model.model.layers.0.self_attn.v_proj_mot_gen",
        "language_model.model.layers.0.mlp_mot_gen",
        "language_model.lm_head",
    ]
    dense_heads = ("net.res_blocks.0.mlp.0", "net.final_layer.linear") if deep_head else ("2",)
    ignored.extend(f"fm_modules.fm_head.{name}" for name in dense_heads)
    config["quantization_config"] = {"quant_method": "fp8", "ignored_layers": ignored}
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    request = LoadRequest(
        model_path=str(tmp_path),
        execution=replace(_execution("bfloat16"), generation_device=generation_device),
        bindings=tensor_parallel_bindings(),
        load=LoadConfig(),
    )
    loaded = get_model_loader(request.load.load_format).load(
        resolve_catalog_entry(("NEOChatModel",)),
        config,
        request,
        root=tmp_path,
        repository_id=None,
    )
    exported = loaded.model.state_dict()
    for name in (
        "language_model.model.layers.0.self_attn.o_proj.weight",
        "language_model.model.layers.0.self_attn.flow_qkv.projection.weight",
        "language_model.model.layers.0.mlp_mot_gen.gate_up_proj.weight",
        "language_model.model.layers.0.mlp_mot_gen.down_proj.weight",
        "language_model.lm_head.weight",
        *(f"fm_modules.velocity.head.{name}.weight" for name in dense_heads),
    ):
        torch.testing.assert_close(exported[name].cpu(), state[name].bfloat16(), rtol=0, atol=0)
    for name in (
        "language_model.model.layers.0.self_attn.o_proj_mot_gen.weight",
        "language_model.model.layers.0.self_attn.text_qkv.projection.weight",
        "language_model.model.layers.0.mlp.gate_up_proj.weight",
        f"fm_modules.velocity.head.{'net.input_proj' if deep_head else '0'}.weight",
    ):
        assert exported[name].dtype == torch.float8_e4m3fn
    device = torch.device(generation_device or "cpu")
    normalized = loaded.model.language_model.model.norm_mot_gen(
        torch.ones((2, 8), dtype=torch.bfloat16, device=device)
    )
    expected = state["language_model.model.norm_mot_gen.weight"].bfloat16().expand(2, 8)
    torch.testing.assert_close(normalized.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("load_format", [LoadFormat.AUTO, LoadFormat.LAYERED])
def test_merged_checkpoint_projections_load_as_complete_tensors(tmp_path, load_format):
    config = _qwen_config()
    reference = _qwen_reference(config)
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_file(
        {name: tensor.contiguous() for name, tensor in reference.state_dict().items()},
        tmp_path / "model.safetensors",
    )
    loaded = load_model(_qwen_request(str(tmp_path), load=LoadConfig(load_format=load_format)))
    for name, tensor in loaded.model.state_dict().items():
        torch.testing.assert_close(tensor, reference.state_dict()[name].bfloat16())


@pytest.mark.parametrize("method", ["mxfp8", "nvfp4"])
def test_block_quantization_requires_its_cuda_backend(tmp_path, method):
    _write_qwen_checkpoint(tmp_path)
    request = replace(_qwen_request(str(tmp_path)), quantization_config={"quant_method": method})
    with pytest.raises(ValueError, match="SM100-class CUDA device"):
        load_model(request)


def test_quantization_override_requires_declared_checkpoint_format_conversion(tmp_path):
    _write_qwen_checkpoint(tmp_path)
    config = _qwen_config()
    config["quantization_config"] = {"quant_method": "fp8"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    request = replace(_qwen_request(str(tmp_path)), quantization_config={"quant_method": "nvfp4"})
    with pytest.raises(ValueError, match="requires checkpoint format conversion"):
        load_model(request)
