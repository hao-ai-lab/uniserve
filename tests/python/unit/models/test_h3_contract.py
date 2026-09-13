"""Supported checkpoint metadata controls capabilities independently of directory names."""

import json
from pathlib import Path

import pytest

from uniserve_worker.models.minimax_h3.config import h3_contract

pytestmark = pytest.mark.unit


@pytest.fixture
def checkpoint(tmp_path):
    source = Path(__file__).parents[2] / "fixtures/models/fasth3/fastvideo_inference.json"
    (tmp_path / "fastvideo_inference.json").write_text(source.read_text())
    return tmp_path


def test_full_vsa_checkpoint_resolves_t2va_and_inference_grid(checkpoint):
    contract = h3_contract(json.loads((checkpoint / "fastvideo_inference.json").read_text()))
    assert contract["tasks"] == ["t2va"]
    assert contract["attention"] == "vsa"
    assert contract["inference_grid"] == [1, 0.75, 0.5, 0.25, 0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task", "ref2va"),
        ("attention_backend", "FLASH_ATTN"),
        ("transformer_forwards", 50),
        ("vsa_sparsity", 0.8),
        ("schema_version", "unknown"),
    ],
)
def test_incompatible_checkpoint_is_rejected(checkpoint, field, value):
    path = checkpoint / "fastvideo_inference.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=field):
        h3_contract(manifest)


def test_architecture_without_variant_metadata_is_rejected(tmp_path):
    from uniserve_worker.bootstrap.metadata import h3_metadata

    with pytest.raises(ValueError, match="fastvideo_inference.json"):
        h3_metadata(tmp_path)


@pytest.mark.parametrize(
    "component,parallel,error",
    [
        ("denoiser", {"tensor_parallel_size": 3}, "must divide heads"),
        ("denoiser", {"pipeline_parallel_size": 51}, "50 transformer layers"),
        ("text_encoder", {"pipeline_parallel_size": 2}, "direct tensor parallelism"),
        ("video_decoder", {"tensor_parallel_size": 2}, "local numerical computation"),
    ],
)
def test_h3_rejects_invalid_numerical_parallelism(component, parallel, error):
    from uniserve_worker.models.minimax_h3.model import MiniMaxH3Model
    from uniserve_worker.nn.parallel import ParallelConfig

    with pytest.raises(ValueError, match=error):
        MiniMaxH3Model.validate_parallel({}, {component: ParallelConfig(**parallel)})


def test_metadata_validation_preserves_temporal_distribution_contract(checkpoint):
    from uniserve_worker.foundation.errors import WorkerError
    from uniserve_worker.loader.config import LoadConfig
    from uniserve_worker.loader.source import ModelSource
    from uniserve_worker.nn.parallel import ComponentConfig

    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": ["MiniMaxH3Transformer3DModel"]})
    )
    source = ModelSource.resolve(str(checkpoint), LoadConfig())
    source.validate({"video_decoder": ComponentConfig((3, 1), distribution="temporal_units")})
    for config in (
        ComponentConfig((0,)),
        ComponentConfig((3, 1), distribution="temporal_units", units_per_rank=2),
    ):
        with pytest.raises(WorkerError, match="one native unit per rank"):
            source.validate({"video_decoder": config})
    with pytest.raises(WorkerError, match="model-parallel membership"):
        source.validate({"text_encoder": ComponentConfig((0, 1), distribution="temporal_units")})


def test_text_requirements_preserve_unpadded_conditioning_extents():
    import torch

    from uniserve_worker.bootstrap.catalog import MINIMAX_H3_ENTRY
    from uniserve_worker.modeling.components import Call
    from uniserve_worker.modeling.context import BuildContext
    from uniserve_worker.modeling.geometry import TextShape
    from uniserve_worker.modeling.tensors import TokenSelection
    from uniserve_worker.models.minimax_h3.model import MiniMaxH3Model
    from uniserve_worker.runtime.tensors import prepare_constants

    context = BuildContext(
        parallel={},
        meshes={},
        layers={},
        limits={"text_tokens": 128, "video_seconds": 22 / 24},
        component_precisions=MINIMAX_H3_ENTRY.component_precisions({"mode": "quality"}),
        schedule=MINIMAX_H3_ENTRY.create_schedule(torch.device("cpu")),
    )
    # Numerical declarations remain usable when this rank owns no learned
    # component. Token outputs describe valid rows, not padded denoiser pages.
    with torch.device("meta"):
        model = MiniMaxH3Model({}, context)
    for call, width in ((Call.ENCODE_TEXT, 5120), (Call.ENCODE_CONDITIONING, 5376)):
        for tokens in (63, 64, 65):
            shape = TextShape(tokens)
            output = model.tensor_specs(call, shape).outputs["conditioning"]
            assert output.shape == (1, tokens, width)
            assert output.dtype == torch.bfloat16
            assert prepare_constants(model, call, shape, device="cpu") == {}
        for shape in (
            TextShape(129),
            TextShape(64, rows=2),
            TextShape(0, selection=TokenSelection.HIDDEN),
        ):
            with pytest.raises(ValueError):
                model.tensor_specs(call, shape)


@pytest.mark.parametrize(
    "frames,seed,tokens", [(22, 0, 63), (22, 0, 64), (22, 0, 65), (39, 1000, 128)]
)
def test_h3_latent_preparation_preserves_native_noise_and_logical_shards(frames, seed, tokens):
    from dataclasses import replace

    import torch

    from uniserve_worker.bootstrap.catalog import MINIMAX_H3_ENTRY
    from uniserve_worker.execution.video import initialize_latents
    from uniserve_worker.modeling.batch import DiffusionBatch
    from uniserve_worker.modeling.components import Call
    from uniserve_worker.modeling.context import BuildContext
    from uniserve_worker.modeling.geometry import MediaShape
    from uniserve_worker.models.minimax_h3.layout import H3Layout
    from uniserve_worker.models.minimax_h3.model import MiniMaxH3Model
    from uniserve_worker.models.minimax_h3.packing import audio_latent_frames
    from uniserve_worker.nn.mesh import Communicator, DeviceMesh
    from uniserve_worker.nn.parallel import ParallelConfig, SequenceParallel
    from uniserve_worker.nn.rng import diffusion_noise
    from uniserve_worker.runtime.tensor_buffers import TensorBuffers
    from uniserve_worker.runtime.tensors import bind_state, prepare_constants

    ranks = tuple(range(7, -1, -1))
    parallel = ParallelConfig(sequence_parallel=SequenceParallel("ulysses", (8,)))
    padded_tokens = ((tokens + 63) // 64) * 64
    shape = MediaShape(768, 1344, frames=frames, prompt_tokens=tokens)
    outputs = {"video": [], "audio": []}
    rows = {"text": [], "video": [], "audio": []}
    noise = None
    for rank in ranks:
        mesh = DeviceMesh(
            ranks,
            rank,
            parallel,
            groups={
                name: Communicator(ranks=ranks, rank=rank, name=name) for name in ("ulysses", "sp")
            },
        )
        layout = H3Layout.build(
            parallel,
            mesh,
            frames=frames,
            text_rows=padded_tokens,
            audio_frames=audio_latent_frames(frames),
        )
        # The numerical graph uses real borrowed group descriptors. Native
        # initialization needs neither learned weights nor collective transport.
        context = BuildContext(
            parallel={"denoiser": parallel},
            meshes={"denoiser": mesh},
            layers={},
            limits={"text_tokens": padded_tokens, "video_seconds": frames / 24},
            component_precisions=MINIMAX_H3_ENTRY.component_precisions({"mode": "quality"}),
            schedule=MINIMAX_H3_ENTRY.create_schedule(torch.device("cpu")),
        )
        with torch.device("meta"):
            model = MiniMaxH3Model({}, context)
        constants = prepare_constants(model, Call.DIFFUSION, shape, device="cpu")
        for name in rows:
            rows[name].append(constants[f"local_{name}_indices"] + layout.local_start)
        spec = model.diffusion_spec(shape, 4)
        if noise is None:
            noise = {
                item.name: torch.empty((1, *item.noise_shape), dtype=torch.float32)
                for item in spec.modalities
            }
            diffusion_noise(
                spec, seeds=(seed,), device=torch.device("cpu"), dtype=torch.float32, out=noise
            )
            reference_generator = torch.Generator().manual_seed(seed)
            video = torch.empty(spec.modalities[0].noise_shape).normal_(
                generator=reference_generator
            )
            audio = torch.empty(spec.modalities[1].noise_shape).normal_(
                generator=reference_generator
            )
            torch.testing.assert_close(noise["video"][0], video, rtol=0, atol=0)
            torch.testing.assert_close(noise["audio"][0], audio, rtol=0, atol=0)
            reference_video = (
                video.reshape(1, 24, video.shape[2], 24, 2, 42, 2)
                .permute(0, 2, 3, 5, 1, 4, 6)
                .reshape(-1, 96)
                .index_select(0, layout.packed.video_raster_indices)
            )
        schemas = model.tensor_specs(Call.DIFFUSION, shape).state
        buffers = TensorBuffers(
            {name: torch.empty(item.shape, dtype=item.dtype) for name, item in schemas.items()}
        )
        bound = bind_state(model, Call.DIFFUSION, shape, buffers)
        sources = {name: bound[f"{name}_source"].unsqueeze(0) for name in ("video", "audio")}
        state = {**{name: value.unsqueeze(0) for name, value in bound.items()}, **sources}
        scratch_specs = model.tensor_specs(Call.DIFFUSION, shape).scratch
        scratch = {
            name: torch.empty(scratch_specs[name].shape, dtype=scratch_specs[name].dtype)
            for name in ("rotary_positions", "rotary_frequencies")
        }
        model.prepare_latents(
            DiffusionBatch({name: (value[0],) for name, value in sources.items()}, (shape,)),
            noise=noise,
            state=state,
            constants=constants,
            scratch=scratch,
        )
        # Request validity must not overwrite the reusable page constants.
        expected_valid = (tokens - torch.arange(0, padded_tokens, 64)).clamp(0, 64).int()
        torch.testing.assert_close(
            bound["tile_valid_sizes"][: padded_tokens // 64], expected_valid, rtol=0, atol=0
        )
        torch.testing.assert_close(
            constants["tile_valid_sizes"], layout.packed.tile_valid_sizes, rtol=0, atol=0
        )
        torch.testing.assert_close(
            constants["positions"], layout.packed.position_ids.float(), rtol=0, atol=0
        )
        for name, value in sources.items():
            outputs[name].append(value[0])
    # Prepared indices cover every mathematical modality exactly once in
    # logical sequence order, despite reversed physical rank membership.
    for name in rows:
        torch.testing.assert_close(
            torch.cat(rows[name]), getattr(layout.packed, f"{name}_indices"), rtol=0, atol=0
        )
    torch.testing.assert_close(torch.cat(outputs["video"]), reference_video, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(outputs["audio"]), audio, rtol=0, atol=0)
    torch.testing.assert_close(noise["video"][0], video, rtol=0, atol=0)
    torch.testing.assert_close(noise["audio"][0], audio, rtol=0, atol=0)

    # Serving supplies the same borrowed views and then delivers the returned
    # source tensors. An unrelated prior RNG draw must not change this request.
    torch.randn(17)
    expected = {name: value[0].clone() for name, value in sources.items()}
    storage = bind_state(model, Call.DIFFUSION, shape, buffers)
    for target, source in initialize_latents(
        model,
        shape,
        storage,
        constants,
        scratch,
        seed,
    ):
        target.copy_(source)
    for name in ("video", "audio"):
        torch.testing.assert_close(storage[name], expected[name], rtol=0, atol=0)
    original_validity = storage["tile_valid_sizes"].clone()
    with pytest.raises(ValueError, match="prompt length"):
        model.prepare_latents(
            DiffusionBatch(
                {name: (value[0],) for name, value in sources.items()},
                (replace(shape, prompt_tokens=0),),
            ),
            noise=noise,
            state=state,
            constants=constants,
            scratch=scratch,
        )
    torch.testing.assert_close(storage["tile_valid_sizes"], original_validity, rtol=0, atol=0)
