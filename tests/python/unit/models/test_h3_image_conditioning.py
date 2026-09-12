"""CPU contracts for Qwen image presentation, independent of H3 VAE rows."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLVisionConfig

from uniserve_worker.backends.attention.torch_sdpa import TorchSDPAAttentionBackend
from uniserve_worker.execution.forward_batch import AttentionSelection
from uniserve_worker.models.minimax_h3.encoder import H3TextEncoderConfig, MiniMaxH3TextEncoder
from uniserve_worker.models.minimax_h3.packing import TEXT_TAG, VIDEO_TAG
from uniserve_worker.models.minimax_h3.vision import H3VisionModel
from uniserve_worker.nn.attention import bind_dense_attention_modules
from uniserve_worker.nn.mesh import DeviceMesh
from uniserve_worker.nn.parallel import ParallelConfig

CHECKPOINT = Path("/mnt/lustre/vlm-k1kong/models/MiniMax-H3/text_encoder")


@pytest.fixture
def encoder():
    torch.manual_seed(7)
    config = H3TextEncoderConfig(
        vocab_size=151936,
        hidden_size=16,
        intermediate_size=32,
        retained_layers=2,
        heads=2,
        kv_heads=2,
        head_dim=8,
        max_text_rows=1024,
    )
    model = MiniMaxH3TextEncoder(
        DeviceMesh((0,), 0, ParallelConfig(), torch.device("cpu")),
        max_text_rows=1024,
        config=config,
        parameter_device="cpu",
    ).to(torch.bfloat16)
    bind_dense_attention_modules(
        model, AttentionSelection("torch_sdpa", (TorchSDPAAttentionBackend(),))
    )
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 1:
                parameter.fill_(1)
            else:
                parameter.copy_(torch.randn(parameter.shape) / 8)
    model.processor = AutoProcessor.from_pretrained(CHECKPOINT, local_files_only=True)
    vision = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=16,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
    )
    model.visual = H3VisionModel(vision).to(torch.bfloat16)
    model.vision_config = SimpleNamespace(
        image_token_id=151655,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        vision_config=SimpleNamespace(spatial_merge_size=2, deepstack_visual_indexes=[0]),
        text_config=SimpleNamespace(rope_parameters={"mrope_section": [2, 1, 1]}),
    )
    return model.eval()


@torch.no_grad()
def test_token_only_is_byte_identical(encoder):
    tokens = torch.tensor([[42, 87, 99]])
    before = encoder.language_model(tokens, torch.arange(3))
    assert torch.equal(before.view(torch.uint8), encoder(tokens).view(torch.uint8))
    assert torch.equal(before.view(torch.uint8), encoder(tokens, []).view(torch.uint8))
    assert torch.equal(before.view(torch.uint8), encoder.numerical_entry(tokens).view(torch.uint8))


@torch.no_grad()
def test_image_presentation_and_processor_grid(encoder):
    # FastVideo supplies PIL RGB images to this exact checkpoint processor.
    image = torch.arange(480 * 832 * 3).remainder(256).to(torch.uint8).reshape(480, 832, 3)
    reference = encoder.processor.image_processor(
        images=[Image.fromarray(image.numpy())], return_tensors="pt"
    )
    actual = encoder.prepare_images([image])
    assert torch.equal(actual["image_grid_thw"], reference["image_grid_thw"])
    assert torch.equal(actual["pixel_values"], reference["pixel_values"])
    assert actual["image_grid_thw"].tolist() == [[1, 30, 52]]
    tokens = torch.tensor([[42, 87, 99]])
    states, tags = encoder.encode_presentation(tokens, [image])
    label_count = len(
        encoder.processor.tokenizer("<Picture 1>: ", add_special_tokens=False)["input_ids"]
    )
    vision_count = int(np.prod(reference["image_grid_thw"].numpy())) // 4
    assert states.shape == (1, label_count + vision_count + 2 + 3, 16)
    assert (
        tags.tolist()
        == [TEXT_TAG] * label_count + [VIDEO_TAG] * (vision_count + 2) + [TEXT_TAG] * 3
    )
    published, published_tags = encoder.numerical_entry(tokens, image.unsqueeze(0))
    assert torch.equal(states, published)
    assert torch.equal(tags, published_tags)
    from uniserve_worker.models.minimax_h3.presentation import image_presentation_tags

    assert tuple(tags.tolist()) == image_presentation_tags(encoder.processor, (480, 832), 3)
    changed, _ = encoder.encode_presentation(tokens, [torch.zeros_like(image)])
    assert not torch.equal(states, changed)


def test_invalid_decoded_raster(encoder):
    with pytest.raises(ValueError, match="HWC uint8"):
        encoder(torch.tensor([[42]]), [torch.zeros(3, 32, 32)])


@torch.no_grad()
def test_vision_tower_matches_qwen_cpu_reference():
    from transformers import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    from uniserve_worker.models.minimax_h3.vision import H3VisionModel

    torch.manual_seed(11)
    config = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=16,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
    )
    reference = Qwen3VLVisionModel(config).eval()
    tower = H3VisionModel(config).eval()
    tower.load_state_dict(reference.state_dict(), strict=True)
    pixels = torch.randn(24, 24)
    grid = torch.tensor([[1, 4, 6]])
    expected = reference(pixels, grid)
    features, deepstack = tower(pixels, grid)
    # PyTorch's default FP32 tolerances cover equivalent SDPA/interpolation math.
    torch.testing.assert_close(features, expected.pooler_output)
    assert len(deepstack) == len(expected.deepstack_features)
    for actual, wanted in zip(deepstack, expected.deepstack_features, strict=True):
        torch.testing.assert_close(actual, wanted)
