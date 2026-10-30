"""LoRA adapters merge into checkpoint weights as a source reads them."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from uniserve import loading
from uniserve.loading import checkpoint, lora, weights
from uniserve.nn.linear import Linear
from uniserve_models import loading as model_loading
from uniserve_models import minimax_h3

pytestmark = pytest.mark.integration

WEIGHT = "blocks.0.attn.to_out.0.weight"
# A FastVideo adapter names the module ``attn.to_out``.
RENAME = ((r"^(.*)\.attn\.to_out$", r"\1.attn.to_out.0"),)


def _checkpoint(directory: Path) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    tensors = {
        WEIGHT: torch.randn(6, 8, generator=generator).bfloat16(),
        "blocks.0.norm.weight": torch.ones(8, dtype=torch.bfloat16),
    }
    directory.mkdir(exist_ok=True)
    save_file(tensors, directory / "model.safetensors")
    return tensors


def _adapter(path: Path, rank: int, *, seed: int, alpha: float | None = None):
    generator = torch.Generator().manual_seed(seed)
    down = torch.randn(rank, 8, generator=generator).bfloat16()
    up = torch.randn(6, rank, generator=generator).bfloat16()
    tensors = {
        "blocks.0.attn.to_out.lora_A.weight": down,
        "blocks.0.attn.to_out.lora_B.weight": up,
    }
    if alpha is not None:
        tensors["blocks.0.attn.to_out.alpha"] = torch.tensor(alpha)
    save_file(tensors, path)
    return down, up


def _source(directory: Path, *adapters: lora.Adapter) -> checkpoint.Source:
    io = loading.Config()
    return replace(
        checkpoint.Config().resolve(directory, io=io), adapters=adapters
    )


def test_an_adapter_merges_as_an_offline_merge_would(tmp_path):
    tensors = _checkpoint(tmp_path / "model")
    down, up = _adapter(tmp_path / "style.safetensors", 2, seed=1, alpha=4.0)
    adapter = lora.Adapter(
        tmp_path / "style.safetensors", strength=1.5, names=RENAME
    )
    io = loading.Config()

    with _source(tmp_path / "model", adapter).open(io=io) as reader:
        weight = reader.get(WEIGHT)
        merged = weight.read()
        # The fused SwiGLU and TP partitions read rectangles of the merge.
        region = weight.read((slice(3, 6), slice(2, 8)))
        norm = reader.get("blocks.0.norm.weight").read()

    # W += strength * alpha / rank * B @ A in FP32, rounded once.
    expected = (
        tensors[WEIGHT].float() + 1.5 * 4.0 / 2 * (up.float() @ down.float())
    ).bfloat16()
    assert weight.dtype == torch.bfloat16 and weight.shape == (6, 8)
    assert torch.equal(merged, expected)
    torch.testing.assert_close(region, expected[3:6, 2:8])
    assert torch.equal(norm, tensors["blocks.0.norm.weight"])


def test_adapters_add_up_and_alpha_defaults_to_rank(tmp_path):
    tensors = _checkpoint(tmp_path / "model")
    first = _adapter(tmp_path / "first.safetensors", 2, seed=1, alpha=1.0)
    second = _adapter(tmp_path / "second.safetensors", 4, seed=2)
    adapters = (
        lora.Adapter(tmp_path / "first.safetensors", 0.5, RENAME),
        lora.Adapter(tmp_path / "second.safetensors", -1.0, RENAME),
    )
    io = loading.Config()

    with _source(tmp_path / "model", *adapters).open(io=io) as reader:
        merged = reader.get(WEIGHT).read()

    expected = tensors[WEIGHT].float()
    for (down, up), scale in ((first, 0.5 * 1.0 / 2), (second, -1.0)):
        expected += scale * (up.float() @ down.float())
    assert torch.equal(merged, expected.bfloat16())


def test_the_loader_materializes_merged_parameters(tmp_path):
    tensors = _checkpoint(tmp_path / "model")
    down, up = _adapter(tmp_path / "style.safetensors", 2, seed=1)

    class Network(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.projection = Linear(8, 6, bias=False)

    def mapping(model):
        return (
            weights.ModuleMapping(
                model,
                "primary",
                lambda reader: (
                    weights.Assignment(
                        model.projection.weight, reader.get(WEIGHT)
                    ),
                ),
                frozenset({"projection.weight"}),
                nonresident_prefixes=("blocks.0.norm.",),
            ),
        )

    adapter = lora.Adapter(tmp_path / "style.safetensors", names=RENAME)
    result = loading.load_model(
        Network,
        None,
        checkpoint=(_source(tmp_path / "model", adapter),),
        mapping=mapping,
        device="cpu",
        weights=weights.Config(dtype=torch.float32),
    )

    expected = (tensors[WEIGHT].float() + up.float() @ down.float()).bfloat16()
    assert torch.equal(result.model.projection.weight, expected.float())


@pytest.mark.parametrize(
    "tensors,message",
    [
        (
            {"blocks.1.attn.to_out.0.weight": torch.zeros(6, 8)},
            "has no weight 'blocks.0.attn.to_out.0.weight'",
        ),
        ({WEIGHT: torch.zeros(6, 6)}, r"makes \(6, 8\), not \(6, 6\)"),
        (
            {
                WEIGHT: torch.zeros(6, 8).to(torch.float8_e4m3fn),
                "blocks.0.attn.to_out.0.weight_scale": torch.tensor(1.0),
            },
            "merge into dense floating matrices",
        ),
    ],
)
def test_an_adapter_must_fit_its_weight(tmp_path, tensors, message):
    (tmp_path / "model").mkdir()
    save_file(tensors, tmp_path / "model" / "model.safetensors")
    _adapter(tmp_path / "style.safetensors", 2, seed=1)
    adapter = lora.Adapter(tmp_path / "style.safetensors", names=RENAME)

    with pytest.raises(ValueError, match=message):
        _source(tmp_path / "model", adapter).open(io=loading.Config())


@pytest.mark.parametrize(
    "tensors,message",
    [
        ({"blocks.0.attn.to_out.weight": torch.zeros(2, 2)}, "unexpected"),
        (
            {"blocks.0.attn.to_out.lora_A.weight": torch.zeros(2, 8)},
            "lacks lora_A or lora_B",
        ),
        (
            {
                "blocks.0.attn.to_out.lora_A.weight": torch.zeros(2, 8),
                "blocks.0.attn.to_out.lora_B.weight": torch.zeros(6, 4),
            },
            r"lora_A \(2, 8\) and lora_B \(6, 4\)",
        ),
    ],
)
def test_a_malformed_adapter_is_refused(tmp_path, tensors, message):
    save_file(tensors, tmp_path / "style.safetensors")

    with pytest.raises(ValueError, match=message):
        lora.read(lora.Adapter(tmp_path / "style.safetensors"))


def test_h3_adapters_merge_into_the_placed_denoiser(tmp_path):
    fixture = Path(__file__).parents[2] / "fixtures/models/fasth3"
    shutil.copytree(fixture, tmp_path, dirs_exist_ok=True)
    (tmp_path / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ModularPipeline"}),
        encoding="utf-8",
    )
    adapter = lora.Adapter(tmp_path / "style.safetensors", 0.8)

    source = model_loading.read_config(
        tmp_path,
        io=loading.Config(mode="dummy"),
        modules=frozenset({"transformer", "text_encoder"}),
        adapters=(adapter,),
    )

    adapters = {item.name: item.adapters for item in source.checkpoint}
    assert adapters["transformer"] == (
        replace(adapter, names=minimax_h3.lora_names),
    )
    assert adapters["text_encoder"] == ()
