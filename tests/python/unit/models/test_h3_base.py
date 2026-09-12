"""Base recipe grids, dense padding semantics, and checkpoint provenance."""

import json
from pathlib import Path

import pytest
import torch

from uniserve_worker.backends.attention.torch_sdpa import TorchSDPAAttentionBackend
from uniserve_worker.models.minimax_h3.base_contract import (
    BASE_H3_REVISION,
    resolve_base_h3_contract,
)
from uniserve_worker.models.minimax_h3.packing import build_packed_layout, dense_key_mask
from uniserve_worker.nn.diffusion.schedule import DiffusionSchedule

pytestmark = pytest.mark.unit


def test_base_grid_matches_fastvideo_cpu_fp32_recipe():
    schedule = DiffusionSchedule.uniform_grid(50, (12.0, 3.0), device="cpu")
    for shift, sigmas, times in zip((12.0, 3.0), schedule.sigmas, schedule.timesteps, strict=True):
        # FastVideo a943220c scheduling_minimax_h3.py:set_timesteps evaluates
        # these expressions in CPU FP32, then drops the terminal clean point.
        base = torch.linspace(1.0, 0.0, 50, dtype=torch.float32)
        expected = torch.unique_consecutive(shift * base / (1 + (shift - 1) * base))
        assert times.numel() == 49
        torch.testing.assert_close(sigmas, expected, rtol=0, atol=0)
        torch.testing.assert_close(times, 1.0 - expected[:-1], rtol=0, atol=0)


@pytest.mark.parametrize("points", [0, 1, True, 2.5])
def test_uniform_grid_rejects_invalid_point_count(points):
    with pytest.raises(ValueError, match="two points"):
        DiffusionSchedule.uniform_grid(points, (12.0, 3.0), device="cpu")


def test_dense_mask_excludes_prompt_audio_video_and_transport_padding():
    packed = build_packed_layout(text_rows=128, num_frames=22, audio_frames=37)
    valid_sizes = packed.tile_valid_sizes.clone()
    valid_sizes[1] = 1  # A 65-token prompt occupies a 128-row text allocation.
    mask = dense_key_mask(valid_sizes).flatten()
    expected = torch.zeros(packed.padded_rows, dtype=torch.bool)
    expected[:65] = True
    expected[packed.audio_indices] = True
    expected[packed.video_indices] = True
    assert torch.equal(mask, expected)


def test_dense_attention_padding_does_not_change_semantic_output():
    generator = torch.Generator().manual_seed(42)
    query = torch.randn(1, 2, 3, 8, generator=generator)
    key = torch.randn(1, 2, 12, 8, generator=generator)
    value = torch.randn(1, 2, 12, 8, generator=generator)
    mask = dense_key_mask(torch.tensor([2, 0, 3]), tile_size=4)
    semantic = torch.tensor([0, 1, 8, 9, 10])
    provider = TorchSDPAAttentionBackend()
    actual = provider.forward(query, key, value, causal=False, scale=8**-0.5, attn_mask=mask)
    expected = provider.forward(
        query, key[:, :, semantic], value[:, :, semantic], causal=False, scale=8**-0.5
    )
    torch.testing.assert_close(actual, expected)


@pytest.fixture
def base_root(tmp_path: Path):
    def install(relative, content):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        receipt = tmp_path / ".cache/huggingface/download" / (relative + ".metadata")
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(BASE_H3_REVISION + "\nreceipt-etag\n0\n")

    install("modular_model_index.json", '{"_class_name":"MiniMaxH3ModularPipeline"}')
    for component in ("transformer", "text_encoder", "vae", "audio_vae"):
        install(f"{component}/config.json", "{}")
        shards = [f"model-{i}.safetensors" for i in range(14 if component == "transformer" else 1)]
        install(
            f"{component}/model.safetensors.index.json",
            json.dumps({"weight_map": {f"weight{i}": name for i, name in enumerate(shards)}}),
        )
        for shard in shards:
            install(f"{component}/{shard}", "weight-file")
    for component, shift in (("scheduler", 12.0), ("audio_scheduler", 3.0)):
        install(f"{component}/scheduler_config.json", json.dumps({"shift": shift}))
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        install(f"tokenizer/{filename}", "{}")
    return tmp_path


def test_base_contract_identifies_dense_49_forward_recipe(base_root):
    contract = resolve_base_h3_contract(base_root)
    assert contract["revision"] == BASE_H3_REVISION
    assert contract["attention"] == "dense"
    assert contract["num_inference_steps"] == 50
    assert contract["denoise_steps"] == 49
    assert contract["guidance_scale"] == 1.0


@pytest.mark.parametrize(
    "defect", ["revision", "empty_receipt", "shard", "nested", "manifest", "shift", "pipeline"]
)
def test_base_contract_rejects_incomplete_or_wrong_root(base_root, defect):
    if defect in {"revision", "empty_receipt"}:
        receipt = base_root / ".cache/huggingface/download/transformer/model-0.safetensors.metadata"
        receipt.write_text(
            "5d9b308a59ab12e67147f191e184baf704185bd1\n" if defect == "revision" else ""
        )
    elif defect == "shard":
        path = base_root / "transformer/model-0.safetensors"
        path.rename(path.with_suffix(".missing"))
    elif defect == "nested":
        base_root = base_root / "Ref2VA"
    elif defect == "manifest":
        (base_root / "fastvideo_inference.json").write_text("{}")
    elif defect == "pipeline":
        (base_root / "modular_model_index.json").write_text('{"_class_name":"OtherPipeline"}')
    else:
        (base_root / "scheduler/scheduler_config.json").write_text('{"shift":10.0}')
    with pytest.raises(ValueError, match="base H3"):
        resolve_base_h3_contract(base_root)
