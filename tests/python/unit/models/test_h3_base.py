"""Base recipe grids, dense padding semantics, and checkpoint provenance."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from uniserve_worker.backends.attention.torch_sdpa import TorchSDPAAttentionBackend
from uniserve_worker.bootstrap.catalog import resolve_catalog_entry
from uniserve_worker.bootstrap.inspect_model import inspect_model
from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.batch import MediaGeometry, StaticDim
from uniserve_worker.execution.bounded_storage import BoundedTensorStorage
from uniserve_worker.loader.component import ModelBuildContext
from uniserve_worker.loader.config import LoadRequest
from uniserve_worker.models.minimax_h3.base_contract import (
    BASE_H3_REVISION,
    resolve_base_h3_contract,
)
from uniserve_worker.models.minimax_h3.config import resolve_h3_contract
from uniserve_worker.models.minimax_h3.model import MiniMaxH3Model
from uniserve_worker.models.minimax_h3.packing import build_packed_layout, dense_key_mask
from uniserve_worker.models.minimax_h3.weights import require_h3_checkpoint
from uniserve_worker.nn.diffusion.schedule import DiffusionSchedule
from uniserve_worker.nn.mesh import Communicator, DeviceMesh, EntryBindings
from uniserve_worker.nn.parallel import EntryConfig, ParallelConfig

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
def base_root(tmp_path: Path, request):
    def install(relative, content):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        receipt = tmp_path / ".cache/huggingface/download" / (relative + ".metadata")
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(BASE_H3_REVISION + "\nreceipt-etag\n0\n")

    install("modular_model_index.json", '{"_class_name":"MiniMaxH3ModularPipeline"}')
    components = ["transformer", "text_encoder", "vae", "audio_vae"]
    if getattr(request, "param", False):
        components.append("transformer_ref")
    for component in components:
        install(f"{component}/config.json", "{}")
        shards = [
            f"model-{i}.safetensors"
            for i in range(14 if component in {"transformer", "transformer_ref"} else 1)
        ]
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
        install(f"text_encoder/{filename}", "{}")
    install("text_encoder/preprocessor_config.json", "{}")
    transformer = {
        "num_attention_heads": 56,
        "attention_head_dim": 128,
        "hidden_size": 5376,
        "num_layers": 50,
        "num_refiner_layers": 2,
        "ffn_dim": 14336,
        "in_channels": 24,
        "audio_in_channels": 32,
        "patch_size": [1, 2, 2],
        "text_dim": 5120,
        "freq_dim": 256,
        "time_embed_hidden_dim": 5376,
        "time_embed_dim": 2688,
        "rope_freq_dim": 16,
        "rope_theta": 10000.0,
        "norm_eps": 1e-5,
        "qk_norm_eps": 1e-5,
        "final_norm_eps": 1e-5,
    }
    for component in (name for name in components if name.startswith("transformer")):
        install(f"{component}/config.json", json.dumps(transformer))
    install(
        "text_encoder/config.json",
        json.dumps(
            {
                "text_config": {
                    "vocab_size": 151936,
                    "hidden_size": 5120,
                    "intermediate_size": 25600,
                    "num_hidden_layers": 64,
                    "num_attention_heads": 64,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "rope_theta": 5000000.0,
                    "rms_norm_eps": 1e-6,
                }
            }
        ),
    )
    return tmp_path


def test_base_contract_identifies_dense_49_forward_recipe(base_root):
    contract = resolve_h3_contract(base_root)
    schedule = resolve_catalog_entry(
        ("MiniMaxH3Transformer3DModel",), root=base_root
    ).create_schedule("cpu")
    for shift, sigmas in zip((12.0, 3.0), schedule.sigmas, strict=True):
        grid = torch.linspace(1.0, 0.0, 50, dtype=torch.float32)
        torch.testing.assert_close(sigmas, shift * grid / (1 + (shift - 1) * grid), rtol=0, atol=0)
    assert contract["revision"] == BASE_H3_REVISION
    assert contract["attention"] == "dense"
    assert contract["num_inference_steps"] == 50
    assert contract["denoise_steps"] == 49
    assert contract["guidance_scale"] == 1.0


@pytest.mark.parametrize("base_root", [True], indirect=True)
def test_reference_contract_identifies_fixed_dense_image_recipe(base_root):
    contract = resolve_base_h3_contract(base_root, reference=True)
    entry = resolve_catalog_entry(("minimax-h3-ref",), root=base_root)
    schedule = entry.create_schedule("cpu")
    assert contract["references"] == {"max": 1, "kinds": ["image"]}
    assert (contract["height"], contract["width"], contract["num_frames"]) == (480, 832, 124)
    assert contract["transformer_component"] == "transformer_ref"
    assert contract["attention"] == contract["reference_attention"] == "dense"
    assert contract["revision"] == BASE_H3_REVISION
    for shift, sigmas in zip((12.0, 3.0), schedule.sigmas, strict=True):
        grid = torch.linspace(1.0, 0.0, 50, dtype=torch.float32)
        torch.testing.assert_close(sigmas, shift * grid / (1 + (shift - 1) * grid), rtol=0, atol=0)
    assert schedule.timesteps[0].numel() == 49


@pytest.mark.parametrize(
    "defect", ["missing", "revision", "nested", "unsafe_index", "missing_index", "processor"]
)
@pytest.mark.parametrize("base_root", [True], indirect=True)
def test_reference_catalog_requires_top_level_pinned_reference_weights(base_root, defect):
    if defect == "missing":
        path = base_root / "transformer_ref/model-0.safetensors"
        path.rename(path.with_suffix(".missing"))
    elif defect == "revision":
        receipt = base_root / ".cache/huggingface/download/transformer_ref/config.json.metadata"
        receipt.write_text("0" * 40 + "\n")
    elif defect == "nested":
        base_root = base_root / "Ref2VA"
    elif defect in {"missing_index", "processor"}:
        relative = (
            "transformer_ref/model.safetensors.index.json"
            if defect == "missing_index"
            else "text_encoder/preprocessor_config.json"
        )
        path = base_root / relative
        path.rename(path.with_suffix(".missing"))
    else:
        index = base_root / "transformer_ref/model.safetensors.index.json"
        data = json.loads(index.read_text())
        data["weight_map"]["weight0"] = "../transformer/model-0.safetensors"
        index.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="base H3"):
        resolve_catalog_entry(("minimax-h3-ref",), root=base_root)
    with pytest.raises(ValueError, match="H3"):
        require_h3_checkpoint(base_root)


@pytest.mark.parametrize("base_root", [False, True], indirect=True)
def test_root_inspection_and_worker_catalog_select_same_recipe(base_root):
    reference = (base_root / "transformer_ref").exists()
    # No safetensors payload is read: these files are inventory placeholders.
    inspected = inspect_model(str(base_root))
    assert inspected["contract"]["variant"] == ("ref" if reference else "base")
    entry = resolve_catalog_entry(("MiniMaxH3Transformer3DModel",), root=base_root)
    denoiser = next(source for source in entry.sources if source.name == "denoiser")
    assert denoiser.directory == ("transformer_ref" if reference else "transformer")
    assert entry.create_schedule("cpu").timesteps[0].numel() == 49


@pytest.mark.parametrize("base_root", [True], indirect=True)
def test_reference_checkpoint_validates_selected_transformer_dimensions(base_root):
    # A broken unselected base config must not affect reference inspection.
    (base_root / "transformer/config.json").write_text("{}")
    require_h3_checkpoint(base_root)
    selected = base_root / "transformer_ref/config.json"
    config = json.loads(selected.read_text())
    config["hidden_size"] = 1
    selected.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="hidden_size"):
        require_h3_checkpoint(base_root)


@pytest.mark.parametrize("base_root", [True], indirect=True)
@pytest.mark.parametrize("variant,width", [("base", 1344), ("ref", 832)])
def test_explicit_variant_selects_one_recipe_from_full_root(base_root, monkeypatch, variant, width):
    monkeypatch.setenv("UNISERVE_H3_VARIANT", variant)
    contract = inspect_model(str(base_root))["contract"]
    assert contract["variant"] == variant
    assert contract["width"] == width
    assert contract["denoise_steps"] == 49


def test_explicit_variant_rejects_unknown_recipe(base_root, monkeypatch):
    monkeypatch.setenv("UNISERVE_H3_VARIANT", "typo")
    with pytest.raises(ValueError, match="UNISERVE_H3_VARIANT"):
        resolve_h3_contract(base_root)


@pytest.mark.parametrize(
    "base_root,recipe,height,width,frames,forwards",
    [
        (False, "base", 768, 1344, 141, 49),
        (False, "eight-step", 768, 1344, 141, 8),
        (True, "ref", 480, 832, 124, 49),
    ],
    indirect=["base_root"],
)
def test_worker_products_follow_checkpoint_geometry(
    base_root, recipe, height, width, frames, forwards
):
    if recipe == "eight-step":
        fixture = (
            Path(__file__).parents[2] / "fixtures/models/fasth3-eight-step/fastvideo_inference.json"
        )
        (base_root / "fastvideo_inference.json").write_text(fixture.read_text())
        (base_root / "scheduler/scheduler_config.json").write_text('{"shift":10.0}')
    bindings = EntryBindings(
        {"output": EntryConfig((0,), ParallelConfig())},
        {"output": DeviceMesh((0,), 0, ParallelConfig())},
        Communicator(),
    )
    entry = resolve_catalog_entry(("MiniMaxH3Transformer3DModel",), root=base_root)
    context = ModelBuildContext(
        root=base_root,
        sources=(),
        request=LoadRequest(
            str(base_root), WorkerConfig(), bindings, max_text_rows=64, max_video_seconds=5.5
        ),
        quantization=None,
        component_precisions=entry.component_precisions({}),
        schedule=entry.create_schedule("cpu"),
    )
    # An output-only worker constructs no checkpoint modules; its public media
    # products must still carry the same contract as the denoiser worker.
    model = MiniMaxH3Model.build_checkpoint({}, context).assemble()
    media = MediaGeometry(
        frame_count=frames, video_units=(frames - 5) // 17, prompt_tokens=1, denoise_steps=forwards
    )
    assert (
        model.output_capacity.height,
        model.output_capacity.width,
        model.output_capacity.frame_count,
    ) == (height, width, frames)
    assert model.output_geometry(media) == model.output_capacity
    decoder_schema = model.entry_outputs["video_decoder"][0]
    assert decoder_schema.shape_bound.dims[-2:] == (StaticDim(height), StaticDim(width))
    # Meta storage exercises view geometry without allocating full RGB movies.
    storage = BoundedTensorStorage(
        {
            name: torch.empty(spec.shape, dtype=spec.dtype, device="meta")
            for name, spec in model.scratch_schema.items()
        }
    )
    execution = model.build_execution(media, storage, None)
    assert execution.media.rgb_round.shape == (frames, height, width, 3)
    request_storage = BoundedTensorStorage(
        {
            name: torch.empty(spec.shape, dtype=spec.dtype, device="meta")
            for name, spec in model.resource_geometry.request_tensors.items()
        }
    )
    slot = model.request_tensors(request_storage, media, execution)
    assert slot.video_overlap.shape == (1, 3, 5, height, width)
    if recipe == "ref":
        undersized = replace(context, request=replace(context.request, max_video_seconds=1.0))
        with pytest.raises(ValueError, match="checkpoint frame count"):
            MiniMaxH3Model.build_checkpoint({}, undersized)


def test_explicit_missing_contract_does_not_select_base_recipe(base_root, monkeypatch):
    missing = base_root / "operator-contract.json"
    with pytest.raises(ValueError, match="sidecar does not exist"):
        resolve_h3_contract(base_root, missing)
    monkeypatch.setenv("UNISERVE_H3_CONTRACT", str(missing))
    with pytest.raises(ValueError, match="sidecar does not exist"):
        resolve_catalog_entry(("MiniMaxH3Transformer3DModel",), root=base_root)


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
