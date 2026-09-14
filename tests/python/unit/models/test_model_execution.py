"""Observable execution behavior exposed by concrete model roots."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from tests.python.fixtures.model_execution import model_arguments
from uniserve.attention.metadata import AttentionMetadata, AttentionMode
from uniserve.distributed.mesh import Communicator
from uniserve.model.batch import DecodeBatch, DiffusionBatch, EncodeBatch, TextBatch
from uniserve.model.encoder import EncoderMixin
from uniserve.model.media import ImageSize
from uniserve.model.tensors import TokenSelection
from uniserve.nn.decoder.mot import MoTConfig
from uniserve.nn.decoder.qwen import Qwen3Config
from uniserve.nn.diffusion.config import DiffusionConfig
from uniserve.nn.diffusion.fm_modules import FlowHeadConfig
from uniserve.nn.layer import LayerConfig
from uniserve.nn.rng import flow_noise_seed, normal_noise
from uniserve.nn.vision.patching import patchify_batch
from uniserve.nn.vision.siglip_navit import SiglipNavitConfig
from uniserve.runtime.kv_cache import KVCache, KVCacheConfig
from uniserve.tensors import ImageRange
from uniserve_models.bagel import BagelConfig, BagelForConditionalGeneration
from uniserve_models.qwen3 import Qwen3ForCausalLM
from uniserve_models.qwen3 import read_config as read_qwen_config
from uniserve_models.sensenova.config import NeoChatConfig
from uniserve_models.sensenova.config import read_config as read_neo_config
from uniserve_models.sensenova.model import NEOChatModel

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("architecture", ["bagel", "sensenova"])
def test_vision_encoding_preserves_image_rows_and_isolates_attention(architecture):
    if architecture == "bagel":
        config = replace(
            _bagel_config(),
            vision=SiglipNavitConfig(
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                patch_size=2,
                image_size=8,
            ),
            vit_max_num_patch_per_side=4,
        )
        # Only vision participates in this call; other checkpoint components
        # retain metadata without allocating their backing tensors.
        with torch.device("meta"):
            model = BagelForConditionalGeneration(config, **model_arguments(_layer_config()))
        model.vision_encoder.to_empty(device="cpu").to(torch.bfloat16)
        pixels = torch.linspace(-1, 1, 2 * 3 * 4 * 6).reshape(2, 3, 4, 6)
        batch = EncodeBatch(tuple(pixels.unbind(0)), grids=(None, None), grid_shapes=(None, None))
        expected_shapes = ((6, 8), (6, 8))
    else:
        model = NEOChatModel(_sensenova_config(), **model_arguments(_layer_config()))
        # Unequal patch grids exercise split offsets after 2x2 downsampling.
        pixels = tuple(torch.linspace(-1, 1, count * 12).reshape(count, 12) for count in (8, 24))
        batch = EncodeBatch(
            pixels,
            grids=(torch.tensor([[2, 4]]), torch.tensor([[4, 6]])),
            grid_shapes=((2, 4), (4, 6)),
        )
        expected_shapes = ((2, 8), (6, 8))
    from uniserve.attention.selection import AttentionSelection
    from uniserve.attention.torch_sdpa import TorchSDPAAttentionBackend
    from uniserve.nn.attention import bind_dense_attention_modules

    bind_dense_attention_modules(
        model, AttentionSelection("torch_sdpa", (TorchSDPAAttentionBackend(),))
    )
    with torch.no_grad():
        for parameter in model.vision_encoder.parameters():
            values = torch.arange(parameter.numel(), dtype=torch.float32).cos().mul_(0.1)
            parameter.copy_(values.reshape_as(parameter))
        snapshots = tuple(value.clone() for value in batch.values)
        output = model.encode("vision", batch, constants={}, scratch={})
        combined = output.values["features"]
        assert tuple(tuple(value.shape) for value in combined) == expected_shapes
        for index, value in enumerate(batch.values):
            single = EncodeBatch(
                (value,),
                grids=batch.grids[index : index + 1],
                grid_shapes=batch.grid_shapes[index : index + 1],
            )
            independent = model.encode("vision", single, constants={}, scratch={})
            torch.testing.assert_close(combined[index], independent.values["features"][0])
            torch.testing.assert_close(value, snapshots[index], rtol=0, atol=0)
            assert torch.isfinite(combined[index]).all()
            assert torch.count_nonzero(combined[index]) > 0
    if architecture == "sensenova":
        with pytest.raises(TypeError, match="requires patch grids"):
            model.encode("vision", EncodeBatch(batch.values), constants={}, scratch={})


def test_encoder_composition_preserves_variable_text_and_conditioning_rows():
    class Encoders(EncoderMixin, nn.Module):
        encoder_kinds = frozenset({"text", "conditioning"})

        def __init__(self):
            super().__init__()
            self.text_encoder = nn.Embedding.from_pretrained(torch.arange(12.0).reshape(4, 3))
            self.conditioner = nn.Linear(3, 2, bias=False)
            with torch.no_grad():
                self.conditioner.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]))

    model = Encoders()
    tokens = EncodeBatch((torch.tensor([[3, 1]]), torch.tensor([[2]])))
    encoded = model.encode("text", tokens, constants={}, scratch={})
    expected = (
        torch.tensor([[[9.0, 10.0, 11.0], [3.0, 4.0, 5.0]]]),
        torch.tensor([[[6.0, 7.0, 8.0]]]),
    )
    for actual, reference in zip(encoded.values["conditioning"], expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    conditioned = model.encode("conditioning", EncodeBatch(expected), constants={}, scratch={})
    for actual, reference in zip(
        conditioned.values["conditioning"],
        (torch.tensor([[[9.0, 22.0], [3.0, 10.0]]]), torch.tensor([[[6.0, 16.0]]])),
        strict=True,
    ):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    model.conditioner = None
    with pytest.raises(ValueError, match="does not participate"):
        model.encode("conditioning", EncodeBatch(expected), constants={}, scratch={})


@pytest.mark.parametrize("architecture", ["bagel", "sensenova"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_diffusion_preparation_preserves_native_draw_and_patch_order(architecture, dtype):
    # These numerical methods consume configuration and borrowed tensors, so
    # learned parameters need no backing for this direct library call.
    with torch.device("meta"):
        if architecture == "bagel":
            model = BagelForConditionalGeneration(
                _bagel_config(), **model_arguments(_layer_config())
            )
            shape = ImageSize(16, 32)
        else:
            config = _sensenova_config()
            config = replace(
                config, flow=replace(config.flow, noise_scale=0.7, noise_scale_max_value=2.0)
            )
            model = NEOChatModel(config, **model_arguments(_layer_config()))
            shape = ImageSize(8, 12)
    target = torch.empty((1, *model.latent_shape("image", shape)), dtype=dtype)
    noise = {"image": target.view(1, *model.noise_shape("image", shape))}
    normal_noise((flow_noise_seed(3, 2),), tuple(noise.values()))
    reference = torch.empty(model.noise_shape("image", shape), dtype=dtype).normal_(
        generator=torch.Generator().manual_seed(flow_noise_seed(3, 2))
    )
    reference.mul_(1.0 if architecture == "bagel" else 0.7)
    native = reference.clone()
    if architecture == "sensenova":
        # NCHW normal draws become row-major 4x4 RGB patches only after scaling.
        reference = reference.view(1, 3, 2, 4, 3, 4).permute(0, 2, 4, 3, 5, 1).reshape(6, 48)
    model.prepare_latents(
        DiffusionBatch({"image": (target[0],)}, (shape,)),
        noise=noise,
        state={"image": target},
        constants={},
        scratch={},
    )
    torch.testing.assert_close(target[0], reference.reshape_as(target[0]), rtol=0, atol=0)
    reconstructed = model.generation.unpatchify(target[0], shape.height, shape.width)
    torch.testing.assert_close(reconstructed, native, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rgb_decode_preserves_spatial_rows_and_signed_pixels(dtype):
    model = NEOChatModel(_sensenova_config(), **model_arguments(_layer_config()))
    pixels = torch.linspace(-2, 2, 2 * 3 * 8 * 12, dtype=dtype).reshape(2, 3, 8, 12)
    patches = patchify_batch(pixels, 4)
    shape = ImageSize(8, 12)
    output = model.decode(
        "image",
        DecodeBatch(tuple(patches.unbind(0)), (shape,) * 2),
        constants={},
        scratch={},
    )
    torch.testing.assert_close(torch.stack(output.values["image"]), pixels, rtol=0, atol=0)
    for layout in output.layouts["image"]:
        assert layout is not None
        assert layout.shape == (3, 8, 12)
        assert layout.value_range is ImageRange.SIGNED_UNIT


def test_diffusion_results_preserve_each_rows_declared_image_geometry():
    from uniserve_models.stub import StubModel

    model = StubModel()
    shapes = (ImageSize(16, 32), ImageSize(32, 48))
    # The simulation denoiser predicts zero velocity in 16x16 RGB patch rows.
    latents = (torch.ones(2, 768), torch.ones(6, 768))
    output = model.forward_diffusion(
        DiffusionBatch({"image": latents}, shapes), state={}, constants={}, scratch={}
    )
    for value, latent in zip(output.values["image"], latents, strict=True):
        torch.testing.assert_close(value, torch.zeros_like(latent, dtype=torch.bfloat16))
        torch.testing.assert_close(latent, torch.ones_like(latent))


def _layer_config() -> LayerConfig:
    return LayerConfig(Communicator(), None)


def _qwen_metadata() -> dict[str, object]:
    return {
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


def _qwen_config() -> Qwen3Config:
    return read_qwen_config(_qwen_metadata())


def _bagel_config() -> BagelConfig:
    return BagelConfig(
        text=MoTConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
            head_dim=4,
            max_position_embeddings=128,
        ),
        start_of_image_id=30,
        end_of_image_id=31,
        max_latent_size=2,
        vision=SiglipNavitConfig(image_size=224, patch_size=14),
        vit_max_num_patch_per_side=16,
    )


def _projection_weight(module: nn.Module) -> torch.Tensor:
    weight = torch.arange(module.weight.numel(), dtype=torch.float32).reshape_as(module.weight)
    with torch.no_grad():
        module.weight.copy_(weight)
    return weight[:32]


def _sensenova_config() -> NeoChatConfig:
    return read_neo_config(
        dict(
            vision_config={
                "hidden_size": 8,
                "llm_hidden_size": 8,
                "downsample_ratio": 0.5,
                "patch_size": 2,
                "num_channels": 3,
                "rope_theta_vision": 10_000.0,
                "max_position_embeddings_vision": 128,
            },
            llm_config={
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
            downsample_ratio=0.5,
            max_image_seq_len=16,
            fm_head_layers=2,
        )
    )


def _text_batch(
    query_lens: tuple[int, ...],
    *,
    attention_mode: AttentionMode,
) -> TextBatch:
    rows = len(query_lens)
    total = sum(query_lens)
    cumulative = torch.tensor(
        [0, *[sum(query_lens[: index + 1]) for index in range(rows)]],
        dtype=torch.int32,
    )
    return TextBatch(
        attention=AttentionMetadata(
            attention_mode=attention_mode,
            prefix_lens=torch.zeros(rows, dtype=torch.int32),
            query_lens=torch.tensor(query_lens, dtype=torch.int32),
            out_cache_loc=torch.zeros(total, dtype=torch.int64),
            block_table=torch.zeros((rows, 1), dtype=torch.int32),
            seq_lens=torch.tensor(query_lens, dtype=torch.int32),
            cu_seqlens_q=cumulative if attention_mode is AttentionMode.PAGED_VARLEN else None,
            cu_seqlens_k=cumulative if attention_mode is AttentionMode.PAGED_VARLEN else None,
            output_indices=torch.tensor(
                [sum(query_lens[: index + 1]) - 1 for index in range(rows)],
                dtype=torch.int64,
            )
            if attention_mode is AttentionMode.PAGED_VARLEN
            else None,
            max_seqlen_q=max(query_lens),
            max_seqlen_k=max(query_lens),
            prefix_lens_cpu=(0,) * rows,
            query_lens_cpu=query_lens,
            seq_lens_cpu=query_lens,
        ),
        input_ids=torch.zeros(total, dtype=torch.long),
        positions=torch.arange(total, dtype=torch.long),
        selections=(TokenSelection.LAST_LOGITS,) * rows,
    )


def test_qwen_configuration_is_independent_of_checkpoint_metadata_changes():
    config = _qwen_metadata()
    model = Qwen3ForCausalLM(read_qwen_config(config), **model_arguments(_layer_config()))
    config["num_hidden_layers"] = 7
    config["max_position_embeddings"] = 4096

    assert model.architecture == "Qwen3ForCausalLM"
    assert model.text_backbone.cache_config.num_layers == 1
    assert model.text_backbone.max_tokens == 128


def test_qwen_decode_projection_preserves_row_alignment():
    model = Qwen3ForCausalLM(_qwen_config(), **model_arguments(_layer_config()))
    weight = _projection_weight(model.lm_head)
    hidden = torch.arange(32, dtype=torch.float32).view(4, 8)
    batch = _text_batch((1, 1, 1, 1), attention_mode=AttentionMode.PAGED_DECODE)

    output = model.compute_logits(hidden, batch).materialize()

    assert len(output.values) == 4
    assert torch.equal(torch.cat(output.values), hidden @ weight.T)


@pytest.mark.parametrize("tied", [False, True])
def test_sensenova_decode_projection_preserves_row_alignment(tied):
    config = _sensenova_config()
    config = replace(config, text=replace(config.text, tie_word_embeddings=tied))
    model = NEOChatModel(config, **model_arguments(_layer_config()))
    weight = _projection_weight(model.language_model.lm_head)
    hidden = torch.arange(32, dtype=torch.float32).view(4, 8)
    batch = _text_batch((1, 1, 1, 1), attention_mode=AttentionMode.PAGED_DECODE)

    output = model.compute_logits(hidden, batch).materialize()

    assert len(output.values) == 4
    assert torch.equal(torch.cat(output.values), hidden @ weight.T)
    if tied:
        ids = torch.tensor([1, 7, 31])
        torch.testing.assert_close(model.embed_input_ids(ids), weight[ids], rtol=0, atol=0)


def test_qwen_prefill_selects_the_last_logit_for_each_ragged_row():
    model = Qwen3ForCausalLM(_qwen_config(), **model_arguments(_layer_config()))
    weight = _projection_weight(model.lm_head)
    hidden = torch.arange(56, dtype=torch.float32).view(7, 8)
    batch = _text_batch((2, 5), attention_mode=AttentionMode.PAGED_VARLEN)

    output = model.compute_logits(hidden, batch).materialize()

    assert torch.equal(torch.cat(output.values), hidden[[1, 6]] @ weight.T)
    for invalid in (hidden[:6], hidden[:, :7], hidden.double()):
        with pytest.raises(ValueError, match="hidden states"):
            model.compute_logits(invalid, batch)


@pytest.mark.parametrize("architecture", ["qwen", "sensenova", "bagel"])
@pytest.mark.parametrize("empty_rows", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_model_projection_preserves_mixed_token_selections(architecture, empty_rows, dtype):
    if architecture == "qwen":
        model = Qwen3ForCausalLM(_qwen_config(), **model_arguments(_layer_config()))
        head = model.lm_head
    elif architecture == "sensenova":
        model = NEOChatModel(_sensenova_config(), **model_arguments(_layer_config()))
        head = model.language_model.lm_head
    else:
        # Projection only needs checkpoint storage for the vocabulary head.
        # Build the complete model topology without allocating unused towers.
        with torch.device("meta"):
            model = BagelForConditionalGeneration(
                _bagel_config(), **model_arguments(_layer_config())
            )
        head = model.model.lm_head.to_empty(device="cpu")
    model.to(dtype)
    weight = _projection_weight(head).to(dtype)
    hidden = torch.arange(80, dtype=dtype).view(10, 8)
    lengths = (0, 5, 0) if empty_rows else (2, 5, 3)
    batch = replace(
        _text_batch(lengths, attention_mode=AttentionMode.PAGED_VARLEN),
        selections=(
            TokenSelection.HIDDEN,
            TokenSelection.ALL_LOGITS,
            TokenSelection.LAST_LOGITS,
        ),
    )
    output = model.compute_logits(hidden, batch)
    actual = output.materialize().values
    expected = (
        (hidden[:0], hidden[:5] @ weight.T, hidden[:0] @ weight.T)
        if empty_rows
        else (hidden[:2], hidden[2:7] @ weight.T, hidden[9:10] @ weight.T)
    )
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_qwen_rejects_incomplete_checkpoint_configuration():
    with pytest.raises(ValueError, match="requires integer field"):
        read_qwen_config({})


@pytest.mark.parametrize(
    "values, message",
    [
        ({"num_key_value_heads": 3}, "divisible by KV heads"),
        ({"num_experts": 2, "num_experts_per_tok": 3}, "must not exceed"),
        ({"head_dim": 3}, "even"),
        ({"rms_norm_eps": float("nan")}, "finite and positive"),
        ({"num_hidden_layers": True}, "positive integer"),
    ],
)
def test_qwen_direct_configuration_validates_math(values, message):
    with pytest.raises(ValueError, match=message):
        replace(_qwen_config(), **values)


def test_qwen_explicit_head_dimension_supports_independent_hidden_width():
    config = read_qwen_config({**_qwen_metadata(), "hidden_size": 10, "num_attention_heads": 3})
    model = Qwen3ForCausalLM(config, **model_arguments(_layer_config()))
    from uniserve.attention.selection import AttentionSelection
    from uniserve.attention.torch_sdpa import TorchSDPAAttentionBackend
    from uniserve.nn.attention import bind_dense_attention_modules

    bind_dense_attention_modules(
        model, AttentionSelection("torch_sdpa", (TorchSDPAAttentionBackend(),))
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.125)
        batch = _text_batch((2,), attention_mode=AttentionMode.DENSE)
        hidden = model(batch, constants={}, scratch={})
        output = model.compute_logits(hidden, batch).materialize().values[0]
    assert output.shape == (1, 32)
    assert torch.isfinite(output).all()


def test_bagel_declares_patch_latents_and_descending_schedule():
    config = _bagel_config()
    with torch.device("meta"):
        model = BagelForConditionalGeneration(config, **model_arguments(_layer_config()))

    size = ImageSize(16, 32)
    assert model.latent_shape("image", size) == model.noise_shape("image", size) == (2, 64)
    assert tuple(
        model.generation.create_schedule(DiffusionConfig(steps=4, timestep_shift=1.0), device="cpu")
        .timesteps[0]
        .tolist()
    ) == (1.0, 0.75, 0.5, 0.25, 0.0)


@pytest.mark.parametrize(
    "total_kv_heads,intervals",
    ((8, ((2, 0), (2, 2), (2, 4), (2, 6))), (2, ((1, 0), (1, 0), (1, 1), (1, 1)))),
)
def test_kv_partition_preserves_tp_member_order(total_kv_heads, intervals):
    ranks = (7, 3, 11, 5)
    config = replace(
        _qwen_config(),
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=8,
        num_key_value_heads=total_kv_heads,
    )
    for rank, (heads, offset) in zip(ranks, intervals, strict=True):
        model = Qwen3ForCausalLM(
            config, **model_arguments(LayerConfig(Communicator(ranks=ranks, rank=rank), None))
        )
        geometry = model.text_backbone.cache_config
        assert geometry.total_kv_heads == total_kv_heads
        assert (geometry.num_kv_heads, geometry.kv_head_offset) == (heads, offset)


def test_text_embedding_mask_matches_replacement_token_computation():
    from uniserve.attention.selection import AttentionSelection
    from uniserve.attention.torch_sdpa import TorchSDPAAttentionBackend
    from uniserve.nn.attention import bind_dense_attention_modules

    model = Qwen3ForCausalLM(_qwen_config(), **model_arguments(_layer_config()))
    bind_dense_attention_modules(
        model, AttentionSelection("torch_sdpa", (TorchSDPAAttentionBackend(),))
    )
    generator = torch.Generator().manual_seed(814)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 1:
                parameter.fill_(1)
            else:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) / 8)
    batch = replace(
        _text_batch((4,), attention_mode=AttentionMode.DENSE),
        input_ids=torch.tensor([1, 3, 5, 7]),
        selections=(TokenSelection.HIDDEN,),
    )
    replacements = torch.tensor([2, 4, 6, 8])
    mask = torch.tensor([False, True, False, True])
    with torch.inference_mode():
        embedded = replace(
            batch, inputs_embeds=model.embed_input_ids(replacements), embedding_mask=mask
        )
        explicit = replace(batch, input_ids=torch.where(mask, replacements, batch.input_ids))
        actual = model(embedded, constants={}, scratch={})
        expected = model(explicit, constants={}, scratch={})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("head", ["shallow", "deep", "pixel"])
@pytest.mark.parametrize("time", [0.25, 0.999])
def test_sensenova_velocity_preserves_fp32_latents_and_clean_endpoint(head, time):
    from uniserve.attention.metadata import ExpertRoute, RouteSpan
    from uniserve.attention.selection import AttentionSelection
    from uniserve.attention.torch_sdpa import TorchSDPAAttentionBackend
    from uniserve.model.tensors import FlowPatches
    from uniserve.nn.attention import bind_attention_modules
    from uniserve.runtime.branches import bind_branches
    from uniserve_worker.runtime.cache_manager import CacheManager

    config = _sensenova_config()
    if head == "deep":
        config = replace(
            config, flow=replace(config.flow, head=FlowHeadConfig(dim=8, layers=3, mlp_ratio=2.0))
        )
    elif head == "pixel":
        config = replace(
            config,
            flow=replace(config.flow, use_pixel_head=True),
            vision=replace(config.vision, patch_size=16),
        )
    model = NEOChatModel(config, **model_arguments(_layer_config())).bfloat16()
    # A zero checkpoint predicts the zero clean sample through each supported
    # head. Its complete diffusion call therefore has a closed-form velocity.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    geometry = model.text_backbone.cache_config
    cache = CacheManager(
        KVCache(
            KVCacheConfig(
                num_layers=geometry.num_layers,
                num_kv_heads=geometry.num_kv_heads,
                head_dim=geometry.head_dim,
                dtype=torch.bfloat16,
                total_layers=geometry.num_layers,
                total_kv_heads=geometry.num_kv_heads,
            ),
            num_pages=1,
            page_size=64,
            device="cpu",
        )
    )
    bind_attention_modules(
        model, cache.cache, AttentionSelection("torch_sdpa", (TorchSDPAAttentionBackend(),))
    )
    bind_branches(model, device="cpu")
    patch = config.vision.patch_size
    side = patch * 2
    latent = torch.linspace(-0.12345, 0.23456, side * side * 3).reshape(1, -1)
    timestep = torch.tensor(time)
    attention = AttentionMetadata(
        attention_mode=AttentionMode.PACKED,
        attention_indexes=torch.zeros((3, 1), dtype=torch.long),
        route_spans=(RouteSpan(ExpertRoute.FLOW, 0, 1),),
        prefix_lens=torch.zeros(1, dtype=torch.int32),
        query_lens=torch.ones(1, dtype=torch.int32),
        out_cache_loc=torch.empty(0, dtype=torch.int64),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        visible_end=torch.ones((1, 1), dtype=torch.int32),
        prefix_lens_cpu=(0,),
        query_lens_cpu=(1,),
        seq_lens_cpu=(1,),
        causal_rows_cpu=(False,),
        causal=False,
        has_cache_writes=False,
        fully_visible=True,
        max_seqlen_q=1,
        max_seqlen_k=64,
    )
    batch = DiffusionBatch(
        latents={"image": (latent,)},
        timesteps={"image": (timestep,)},
        positions=(torch.zeros(1, dtype=torch.long),),
        conditioning={
            "image": (
                FlowPatches(
                    pixels=torch.zeros((4, patch * patch * 3), dtype=torch.bfloat16),
                    grid=torch.tensor([[2, 2]]),
                    noise_scale=torch.tensor(0.3),
                ),
            )
        },
        sequence_lengths=(1,),
        sizes=(ImageSize(side, side),),
        attention=attention,
    )
    with torch.inference_mode():
        output = model.forward_diffusion(batch, state={}, constants={}, scratch={})
        actual = output.values["image"][0]
    expected = -latent / (1 - timestep).clamp_min(0.02)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ["bagel", "sensenova"])
def test_guidance_interval_uses_analytical_time_before_fp32_staging(architecture):
    from uniserve.nn.diffusion.cfg import Branch, RenormKind

    with torch.device("meta"):
        if architecture == "bagel":
            model = BagelForConditionalGeneration(
                _bagel_config(), **model_arguments(_layer_config())
            )
            boundary = 1.0 - 1.0 / 3.0
        else:
            model = NEOChatModel(_sensenova_config(), **model_arguments(_layer_config()))
            boundary = 1.0 / 3.0
    config = DiffusionConfig(
        steps=3,
        timestep_shift=1.0,
        cfg_text_scale=2.0,
        cfg_interval=(boundary, boundary),
        cfg_renorm=RenormKind.NONE,
    )
    schedule = model.generation.create_schedule(config, device="cpu")
    torch.testing.assert_close(schedule.timesteps[0][1], torch.tensor(boundary), rtol=0, atol=0)
    predictions = {Branch.COND: torch.tensor([4.0]), Branch.TEXT_UNCOND: torch.tensor([2.0])}
    # Twofold guidance gives unconditioned + 2 * (conditioned - unconditioned).
    guided = model.generation.guidance(config, 1).combine(predictions)
    outside = model.generation.guidance(config, 0).combine(predictions)
    torch.testing.assert_close(guided, torch.tensor([6.0]), rtol=0, atol=0)
    torch.testing.assert_close(outside, predictions[Branch.COND], rtol=0, atol=0)
