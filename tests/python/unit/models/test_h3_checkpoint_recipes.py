"""Checkpoint recipes are resolved without allocating model weights or CUDA state."""

import json
from pathlib import Path

import pytest
import torch
from diffusers import MiniMaxH3Scheduler

from uniserve_worker.bootstrap.catalog import resolve_catalog_entry
from uniserve_worker.bootstrap.metadata import resolve_h3_contract

pytestmark = pytest.mark.unit
FIXTURES = Path(__file__).parents[2] / "fixtures/models"
ARCHITECTURES = ("MiniMaxH3Transformer3DModel",)


@pytest.fixture
def eight_step(tmp_path):
    manifest = json.loads((FIXTURES / "fasth3-eight-step/fastvideo_inference.json").read_text())
    (tmp_path / "fastvideo_inference.json").write_text(json.dumps(manifest))
    return tmp_path


@pytest.mark.parametrize(
    "release,forwards,shifts",
    [
        ("fasth3", 4, (12.0, 3.0)),
        ("fasth3-eight-step", 8, (10.0, 3.0)),
    ],
)
def test_catalog_release_schedule(release, forwards, shifts):
    root = FIXTURES / release
    contract = resolve_h3_contract(root)
    entry = resolve_catalog_entry(ARCHITECTURES, root=root)
    schedule = entry.create_schedule(torch.device("cpu"))
    assert contract["denoise_steps"] == forwards
    assert contract["sigma_shifts"] == list(shifts)
    assert all(t.numel() == forwards for t in schedule.timesteps)
    assert all(s[-1] == 0 for s in schedule.sigmas)


def test_eight_step_schedule_matches_upstream_scheduler(eight_step):
    # FastVideo's scheduler accepts already shifted explicit sigmas. Compute
    # those with its FP32 tensor arithmetic, independently of UniServe's
    # analytical scalar implementation. Its default linspace(9) is a distinct
    # protocol and is deliberately not used for the explicit manifest ladder.
    schedule = resolve_catalog_entry(ARCHITECTURES, root=eight_step).create_schedule("cpu")
    base = torch.tensor([999, 874, 749, 624, 500, 375, 250, 125, 0]) / 1000
    for index, shift in enumerate((10.0, 3.0)):
        reference = MiniMaxH3Scheduler(shift=shift)
        reference.set_timesteps(sigmas=shift * base / (1 + (shift - 1) * base))
        torch.testing.assert_close(schedule.sigmas[index], reference.sigmas)
        torch.testing.assert_close(schedule.timesteps[index], reference.timesteps)


def test_local_export_requires_bound_operator_sidecar(tmp_path, monkeypatch):
    root = tmp_path / "checkpoint-1600"
    root.mkdir()
    with pytest.raises(ValueError, match="explicit"):
        resolve_catalog_entry(ARCHITECTURES, root=root)

    # Synthetic test identity, not an assertion about any production export.
    manifest = json.loads((FIXTURES / "fasth3-eight-step/fastvideo_inference.json").read_text())
    manifest.update(
        model_id="test/local-export",
        checkpoint_root=str(root.resolve()),
        transformer_forwards=4,
        num_inference_steps=5,
        dmd_denoising_steps=[999, 749, 500, 250],
    )
    sidecar = tmp_path / "contract.json"
    sidecar.write_text(json.dumps(manifest))
    monkeypatch.setenv("UNISERVE_H3_CONTRACT", str(sidecar))
    contract = resolve_h3_contract(root)
    assert contract["model_id"] == "test/local-export"
    schedule = resolve_catalog_entry(ARCHITECTURES, root=root).create_schedule("cpu")
    assert schedule.timesteps[0].numel() == 4
    assert contract["ladder"] == [999, 749, 500, 250]
    with pytest.raises(ValueError, match="checkpoint_root"):
        resolve_h3_contract(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_content_sha256", "0" * 64),
        ("checkpoint_metadata_sha256", "invalid"),
        ("video_scheduler_shift", 12.0),
        ("vsa_sparsity", 0.9),
        ("dmd_denoising_steps", [999] * 8),
        ("transformer_forwards", 4),
        ("num_inference_steps", 8),
        ("guidance_scale", 2.0),
        ("sequence_parallel_size", 0),
        ("attention_backend", "FLASH_ATTN"),
    ],
)
def test_eight_step_recipe_cannot_be_silently_changed(eight_step, field, value):
    path = eight_step / "fastvideo_inference.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=field):
        resolve_h3_contract(eight_step)


def test_eight_step_numerical_recipe_and_worker_advertisement(eight_step):
    from uniserve_worker.bootstrap.worker_info_builder import build_worker_layout
    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.modeling.context import BuildContext
    from uniserve_worker.modeling.geometry import MediaShape
    from uniserve_worker.models.minimax_h3.model import MiniMaxH3Model
    from uniserve_worker.nn.mesh import DeviceMesh
    from uniserve_worker.nn.parallel import ParallelConfig

    entry = resolve_catalog_entry(ARCHITECTURES, root=eight_step)
    parallel = ParallelConfig()
    context = BuildContext(
        parallel={"output": parallel},
        meshes={"output": DeviceMesh((0,), 0, parallel)},
        layers={},
        limits={"text_tokens": 64, "video_seconds": 22 / 24},
        component_precisions=entry.component_precisions({"mode": "quality"}),
        schedule=entry.create_schedule(torch.device("cpu")),
    )
    manifest = json.loads((eight_step / "fastvideo_inference.json").read_text())
    model = MiniMaxH3Model({"inference": manifest}, context)
    shape = MediaShape(768, 1344, frames=22, prompt_tokens=64)
    spec = model.diffusion_spec(shape, 8)
    for modality, shift in zip(spec.modalities, (10.0, 3.0), strict=True):
        assert modality.schedule.ladder == (999, 874, 749, 624, 500, 375, 250, 125)
        assert modality.schedule.shift == shift
    with pytest.raises(ValueError, match="ladder"):
        model.diffusion_spec(shape, 4)
    worker = WorkerConfig(
        device="cpu", max_batch_operations=2, max_batch_tokens=2, max_request_pool_size=2
    )
    assert build_worker_layout(model, worker, queue_depth=6).info.num_inference_steps == 8


@pytest.mark.parametrize("sparsity,selected", [(0.9, 1), (0.8, 2), (0.0, 10)])
def test_checkpoint_sparse_selection_cardinality(sparsity, selected):
    from uniserve_worker.nn.sparse_attention import VideoSparseAttentionMetadata

    metadata = VideoSparseAttentionMetadata(
        padded_rows=12 * 64,
        prefix_tiles=2,
        video_tiles=10,
        valid_tiles=12,
        valid_sizes=torch.full((12,), 64, dtype=torch.int32),
        sparsity=sparsity,
    )
    pattern = metadata.pattern(12)
    expected = torch.tensor([[12, 12, *((2 + selected,) * 10)]], dtype=torch.int32)
    torch.testing.assert_close(pattern.counts(1, 12, 12), expected)


def test_sidecar_cannot_override_embedded_manifest(eight_step, tmp_path):
    manifest = json.loads((eight_step / "fastvideo_inference.json").read_text())
    manifest["model_id"] = "test/different-export"
    sidecar = tmp_path / "override.json"
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="disagrees"):
        resolve_h3_contract(eight_step, sidecar)
