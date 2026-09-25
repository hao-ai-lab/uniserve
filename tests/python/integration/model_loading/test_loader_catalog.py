"""Model discovery and checkpoint closure.

Both run through the public loading boundary.
"""

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests.python.fixtures.checkpoints import qwen_checkpoint
from uniserve import loading
from uniserve.loading import weights
from uniserve.model import TextInput, TextSize
from uniserve.nn.attention import SequenceLengths, VarlenInput
from uniserve.processing import load_tokenizer
from uniserve.quantization import QuantizationConfig, QuantizedTensor, Quantizer
from uniserve.runtime import ExecutionContext
from uniserve_models import loading as models

pytestmark = pytest.mark.integration


def _sense_config() -> dict[str, object]:
    return {
        "architectures": ["NEOChatModel"],
        "vision_config": {
            "hidden_size": 8,
            "llm_hidden_size": 16,
            "downsample_ratio": 0.5,
            "patch_size": 2,
            "num_channels": 3,
            "rope_theta_vision": 10_000.0,
            "max_position_embeddings_vision": 128,
        },
        "llm_config": {
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
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


def _bagel_config() -> dict[str, object]:
    return {
        "architectures": ["BagelForConditionalGeneration"],
        "llm_config": {
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 37,
        },
        "vit_config": {
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "patch_size": 2,
            "image_size": 8,
        },
        "vae_config": {
            "ch": 32,
            "ch_mult": [1, 1],
            "downsample": 2,
            "z_channels": 2,
        },
        "start_of_image_id": 35,
        "end_of_image_id": 36,
    }


def _write_bagel_checkpoint(root: Path, metadata: dict[str, object]) -> None:
    """Write BAGEL metadata and the primary shard's 3x3 position table."""
    (root / "config.json").write_text(json.dumps(metadata))
    save_file(
        {"latent_pos_embed.pos_embed": torch.zeros(9, 32)},
        root / "ema.safetensors",
    )


def _write_input_tokenizer(root: Path, *, has_markers: bool = True) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    vocabulary = {"[UNK]": 0}
    if has_markers:
        vocabulary.update({"<img>": 1, "</img>": 2})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(root)


def _load(root, io=loading.Config(), precision=None):
    config = models.read_config(root, io=io)
    options = (
        replace(config.weights, dtype=torch.float32)
        if precision is None
        else precision
    )
    return models.load_model(config, device="cpu", weights=options).model


@torch.inference_mode()
def _logits(model):
    tokens = torch.tensor([1, 3, 9])
    lengths = SequenceLengths.from_lengths((3,), device="cpu")
    attention = VarlenInput(lengths, lengths, (True,))
    with ExecutionContext(model, attention="torch") as context:
        context.prepare(TextSize(3, 1))
        hidden = model(TextInput(tokens, torch.arange(3), attention))
        return model.compute_logits(
            hidden, token_indices=torch.arange(tokens.numel())
        ).gather()


def _indexed(root, format):
    tensors = load_file(root / "model.safetensors")
    names = sorted(tensors)
    shards = {}
    for index, name in enumerate(names):
        file = (
            f"weights-{index % 2}."
            f"{'safetensors' if format == 'safetensors' else 'bin'}"
        )
        shards.setdefault(file, {})[name] = tensors[name]
    (root / "model.safetensors").unlink()
    for name, values in shards.items():
        (save_file if format == "safetensors" else torch.save)(
            values, root / name
        )
    filename = (
        "model.safetensors.index.json"
        if format == "safetensors"
        else "pytorch_model.bin.index.json"
    )
    path = root / filename
    path.write_text(
        json.dumps(
            {
                "weight_map": {
                    name: file
                    for file, values in shards.items()
                    for name in values
                }
            }
        )
    )
    return path


@pytest.mark.parametrize("mode", ("eager", "layered"))
@pytest.mark.parametrize("format", ("safetensors", "pt"))
@torch.inference_mode()
def test_indexed_checkpoint_defines_the_complete_numerical_source(
    tmp_path, mode, format
):
    reference = qwen_checkpoint(tmp_path)
    index = _indexed(tmp_path, format)
    # Unindexed payloads cannot alter the checkpoint selected by the index.
    (
        tmp_path
        / ("extra.safetensors" if format == "safetensors" else "extra.bin")
    ).write_bytes(b"unused")
    actual = _logits(_load(tmp_path, loading.Config(mode=mode, format=format)))
    expected = reference(torch.tensor([[1, 3, 9]])).logits[0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    mapping = json.loads(index.read_text())
    mapping["weight_map"]["model.norm.weight"] = (
        "missing.safetensors" if format == "safetensors" else "missing.bin"
    )
    index.write_text(json.dumps(mapping))
    with pytest.raises(FileNotFoundError):
        _load(tmp_path, loading.Config(mode=mode, format=format))


@pytest.mark.parametrize("has_markers", (True, False))
def test_image_processing_metadata_has_resolved_token_identities(
    tmp_path, has_markers
):
    _write_input_tokenizer(tmp_path, has_markers=has_markers)
    (tmp_path / "config.json").write_text(json.dumps(_sense_config()))
    save_file({"weight": torch.ones(1)}, tmp_path / "model.safetensors")
    if not has_markers:
        with pytest.raises(ValueError, match="does not define declared token"):
            models.read_config(tmp_path)
        return
    config = models.read_config(tmp_path)
    injection = config.image_processor.feature_injection
    assert injection.start_token_id == 1
    assert injection.end_token_id == 2
    assert config.image_processor.vit.patch_size == 2
    assert config.image_processor.vit.downsample_ratio == 0.5
    assert config.flow_prompt is not None
    assert load_tokenizer(config.tokenizer).convert_tokens_to_ids("<img>") == 1


@pytest.mark.parametrize("mode", ("eager", "dummy"))
def test_remote_image_architecture_resolves_checkpoint_dimensions_and_transforms(  # noqa: E501
    tmp_path, monkeypatch, mode
):
    remote = tmp_path / "remote"
    snapshot = tmp_path / "snapshots" / ("b" * 40)
    remote.mkdir()
    _write_bagel_checkpoint(remote, _bagel_config())
    # An unselected payload source stays on the Hub in every mode.
    save_file({"weight": torch.zeros(1)}, remote / "ae.safetensors")
    downloaded = set()

    def download(*, repo_id, filename, revision, cache_dir):
        downloaded.add(filename)
        target = snapshot / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote / filename, target)
        return str(target)

    def files(self, *, repo_id, revision):
        return [path.name for path in remote.iterdir()]

    def tree(self, *, repo_id, revision, recursive):
        return [_tree_file(path, remote) for path in remote.iterdir()]

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr("huggingface_hub.HfApi.list_repo_files", files)
    monkeypatch.setattr("huggingface_hub.HfApi.list_repo_tree", tree)
    # Architecture inspection still needs the learned position-table extent,
    # even when no numerical module is selected for loading, and dummy
    # loading cannot synthesize the header that extent comes from.
    config = models.read_config(
        "owner/bagel", io=loading.Config(mode=mode), modules=frozenset()
    )
    assert downloaded == {"config.json", "ema.safetensors"}
    assert config.model.max_latent_size == 3
    assert config.image_processor.vit.resize.stride == 2
    assert config.image_processor.vit.resize.max_size == 8
    assert config.image_processor.feature_injection.start_token_id == 35
    assert config.image_processor.feature_injection.end_token_id == 36
    assert config.flow_prompt is None


def _tree_file(path, remote):
    """Describe one repository file as the Hub tree listing reports it."""
    return SimpleNamespace(
        path=path.relative_to(remote).as_posix(), size=path.stat().st_size
    )


def _malformed_architectures(root: Path) -> None:
    (root / "config.json").write_text(json.dumps({"architectures": 5}))


def _malformed_bagel_shift(root: Path) -> None:
    _write_bagel_checkpoint(root, {**_bagel_config(), "timestep_shift": "1.0"})


def _malformed_bagel_vision_epsilon(root: Path) -> None:
    metadata = _bagel_config()
    metadata["vit_config"]["layer_norm_eps"] = "1e-6"
    _write_bagel_checkpoint(root, metadata)


def _bagel_without_vocabulary(root: Path) -> None:
    metadata = _bagel_config()
    del metadata["llm_config"]["vocab_size"]
    _write_bagel_checkpoint(root, metadata)


def _sense_without_text_tower(root: Path) -> None:
    metadata = _sense_config()
    del metadata["llm_config"]
    (root / "config.json").write_text(json.dumps(metadata))


def _sense_with_malformed_tokenizer(root: Path) -> None:
    (root / "config.json").write_text(json.dumps(_sense_config()))
    (root / "tokenizer.json").write_text(json.dumps({"model": 5}))


@pytest.mark.parametrize(
    "write",
    (
        _malformed_architectures,
        _malformed_bagel_shift,
        _malformed_bagel_vision_epsilon,
        _bagel_without_vocabulary,
        _sense_without_text_tower,
        _sense_with_malformed_tokenizer,
    ),
)
def test_malformed_checkpoint_metadata_is_rejected_as_invalid(tmp_path, write):
    # Parseable JSON whose fields have the wrong type or are missing is
    # invalid checkpoint metadata, which read_config reports as ValueError.
    write(tmp_path)
    with pytest.raises(ValueError):
        models.read_config(tmp_path, modules=frozenset())


def test_unknown_modular_pipeline_fails_at_discovery(tmp_path):
    (tmp_path / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "UnknownPipeline"})
    )
    with pytest.raises(ValueError, match="supported architecture"):
        models.read_config(tmp_path)


@pytest.mark.parametrize("origin", ("checkpoint", "caller"))
@torch.inference_mode()
def test_precision_selection_preserves_independent_projection_branches(
    tmp_path, origin
):
    reference = qwen_checkpoint(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    # Q alone stays dense. Independent K/V and gate/up statistics remain
    # meaningful even when their physical projection can be fused.
    excluded = (
        "backbone.layers.0.attention.qkv.projection.projections.q",
        "backbone.layers.0.mlp",
    )
    converter = Quantizer("fp8", axis=0)
    selected = weights.Config(
        dtype=torch.float32,
        quantization={
            "": QuantizationConfig(converter, converter),
            **dict.fromkeys(excluded),
        },
    )
    if origin == "checkpoint":
        config["quantization_config"] = {
            "quant_method": "fp8",
            "ignored_layers": [
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.mlp",
            ],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        model = _load(tmp_path)
    else:
        model = _load(tmp_path, precision=selected)
    qkv = model.backbone.layers["0"].attention.qkv.projection
    source = reference.model.layers[0].self_attn
    x = torch.arange(96).reshape(3, 32).sin()
    with ExecutionContext(qkv, matmul="torch") as context:
        context.prepare(TextSize(4, 1))
        actual = qkv(x)
    for name, layer in qkv.projections.items():
        source_weight = getattr(source, name + "_proj").weight
        source_bias = getattr(source, name + "_proj").bias
        if name == "q":
            assert not isinstance(layer.weight, QuantizedTensor)
            expected = torch.nn.functional.linear(x, source_weight, source_bias)
        else:
            assert isinstance(layer.weight, QuantizedTensor)
            expected = torch.nn.functional.linear(
                converter.quantize(x).dequantize(),
                converter.quantize(source_weight).dequantize(),
                source_bias,
            )
        torch.testing.assert_close(actual[name], expected, rtol=1e-5, atol=1e-6)


def test_remote_snapshot_loads_only_the_closed_payload_set(
    tmp_path, monkeypatch
):
    remote, snapshot = tmp_path / "remote", tmp_path / "snapshots" / ("a" * 40)
    remote.mkdir()
    reference = qwen_checkpoint(remote)
    _indexed(remote, "safetensors")
    (remote / "README.md").write_text("Model description")
    (remote / "unindexed.safetensors").write_bytes(b"unused")
    inventory = [file.name for file in remote.iterdir()]

    def download(*, repo_id, filename, cache_dir, revision):
        assert repo_id == "owner/model"
        assert revision in {"release", snapshot.name}
        if filename in {"README.md", "unindexed.safetensors"}:
            raise AssertionError("unrelated payload was requested")
        target = snapshot / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote / filename, target)
        return str(target)

    def files(self, *, repo_id, revision):
        assert repo_id == "owner/model"
        assert revision == snapshot.name
        return inventory

    def tree(self, *, repo_id, revision, recursive):
        assert repo_id == "owner/model"
        assert revision == snapshot.name
        return [_tree_file(remote / name, remote) for name in inventory]

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr("huggingface_hub.HfApi.list_repo_files", files)
    monkeypatch.setattr("huggingface_hub.HfApi.list_repo_tree", tree)
    io = loading.Config(
        revision="release", download_dir=str(tmp_path / "snapshots")
    )
    config = models.read_config("owner/model", io=io)
    result = models.load_model(
        config, device="cpu", weights=weights.Config(dtype=torch.float32)
    )
    with torch.no_grad():
        expected = reference(torch.tensor([[1, 3, 9]])).logits[0]
    torch.testing.assert_close(
        _logits(result.model), expected, rtol=1e-5, atol=1e-6
    )


def test_dummy_loading_is_deterministic_without_checkpoint_payload(tmp_path):
    qwen_checkpoint(tmp_path, tied=True)
    (tmp_path / "model.safetensors").write_bytes(b"invalid checkpoint bytes")
    io = loading.Config(mode="dummy")
    first, second = _load(tmp_path, io), _load(tmp_path, io)
    assert first.backbone.embedding.weight is first.lm_head.weight
    actual = _logits(first)
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, _logits(second), rtol=0, atol=0)
