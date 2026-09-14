"""Supported checkpoint metadata controls capabilities independently of directory names."""

import json
import shutil
from pathlib import Path

import pytest

from uniserve.model.limits import ModelLimits
from uniserve_models.metadata import h3_metadata
from uniserve_models.minimax_h3.config import H3Config

pytestmark = pytest.mark.unit


@pytest.fixture
def checkpoint(tmp_path):
    source = Path(__file__).parents[2] / "fixtures/models/fasth3"
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    return tmp_path


def test_full_vsa_checkpoint_resolves_t2va_and_inference_grid(checkpoint):
    config = h3_metadata(checkpoint)
    assert config.diffusion.ladder == (1000, 750, 500, 250)
    assert config.diffusion.time_scale == 1000
    assert config.diffusion.video_shift == 12
    assert config.diffusion.audio_shift == 3


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
        h3_metadata(checkpoint)


def test_architecture_without_variant_metadata_is_rejected(tmp_path):
    from uniserve_models.metadata import h3_metadata

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
    from uniserve.distributed.parallel import ParallelConfig
    from uniserve_models.minimax_h3.model import MiniMaxH3Model

    with pytest.raises(ValueError, match=error):
        MiniMaxH3Model.validate_parallel({}, {component: ParallelConfig(**parallel)})


def test_metadata_validation_preserves_temporal_distribution_contract(checkpoint):
    from uniserve_models import resolve_model
    from uniserve_worker.bootstrap.components import validate_components
    from uniserve_worker.config import ComponentConfig
    from uniserve_worker.foundation.errors import WorkerError

    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": ["MiniMaxH3Transformer3DModel"]})
    )
    source = resolve_model(str(checkpoint), components=frozenset())
    validate_components(
        source.model_class,
        source.config,
        {"video_decoder": ComponentConfig((3, 1), distribution="temporal_units")},
    )
    for config in (
        ComponentConfig((0,)),
        ComponentConfig((3, 1), distribution="temporal_units", units_per_rank=2),
    ):
        with pytest.raises(WorkerError, match="one native unit per rank"):
            validate_components(source.model_class, source.config, {"video_decoder": config})
    with pytest.raises(WorkerError, match="model-parallel membership"):
        validate_components(
            source.model_class,
            source.config,
            {"text_encoder": ComponentConfig((0, 1), distribution="temporal_units")},
        )


def test_text_requirements_preserve_unpadded_conditioning_extents():
    import torch

    from tests.python.fixtures.model_execution import h3_arguments
    from uniserve.model.tensors import TokenSelection
    from uniserve.model.text import TextSize
    from uniserve_models.catalog import MINIMAX_H3_ENTRY
    from uniserve_models.minimax_h3.model import MiniMaxH3Model

    arguments = h3_arguments(
        parallel={},
        meshes={},
        limits=ModelLimits(text_tokens=128, video_frames=22),
        precisions=MINIMAX_H3_ENTRY.component_precisions({"mode": "quality"}),
    )
    # Numerical declarations remain usable when this rank owns no learned
    # component. Token outputs describe valid rows, not padded denoiser pages.
    with torch.device("meta"):
        model = MiniMaxH3Model(H3Config(), **arguments)
    for component, width in ((model.text_encoder, 5120),):
        for tokens in (63, 64, 65):
            shape = TextSize(tokens)
            output = component.output_layout(shape)["conditioning"]
            assert output.shape == (1, tokens, width)
            assert output.dtype == torch.bfloat16
        for shape in (
            TextSize(129),
            TextSize(64, rows=2),
            TextSize(0, selection=TokenSelection.HIDDEN),
        ):
            with pytest.raises(ValueError):
                component.output_layout(shape)


@pytest.mark.parametrize(
    "frames,seed,tokens", [(22, 0, 63), (22, 0, 64), (22, 0, 65), (39, 1000, 128)]
)
def test_h3_latent_preparation_preserves_native_noise_and_logical_shards(frames, seed, tokens):
    from dataclasses import replace

    import torch

    from tests.python.fixtures.model_execution import h3_arguments
    from uniserve.distributed.mesh import Communicator, DeviceMesh
    from uniserve.distributed.parallel import ParallelConfig, SequenceParallel
    from uniserve.model.batch import DiffusionBatch
    from uniserve.model.media import VideoSize
    from uniserve.nn.rng import normal_noise
    from uniserve.runtime.tensor_buffers import TensorBuffers
    from uniserve.runtime.tensors import bind_state, prepare_constants
    from uniserve_models.catalog import MINIMAX_H3_ENTRY
    from uniserve_models.minimax_h3.layout import H3Layout
    from uniserve_models.minimax_h3.model import MiniMaxH3Model
    from uniserve_models.minimax_h3.packing import audio_latent_frames
    from uniserve_worker.execution.video import initialize_latents

    ranks = tuple(range(7, -1, -1))
    parallel = ParallelConfig(sequence_parallel=SequenceParallel("ulysses", (8,)))
    padded_tokens = ((tokens + 63) // 64) * 64
    shape = VideoSize(frames, prompt_tokens=tokens)
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
        arguments = h3_arguments(
            parallel={"denoiser": parallel},
            meshes={"denoiser": mesh},
            limits=ModelLimits(text_tokens=padded_tokens, video_frames=frames),
            precisions=MINIMAX_H3_ENTRY.component_precisions({"mode": "quality"}),
        )
        with torch.device("meta"):
            model = MiniMaxH3Model(H3Config(), **arguments)
        constants = prepare_constants(
            model.denoiser, VideoSize(shape.frames, shape.prompt_tokens), device="cpu"
        )
        for name in rows:
            rows[name].append(constants[f"local_{name}_indices"] + layout.local_start)
        if noise is None:
            noise = {
                name: torch.empty((1, *model.noise_shape(name, shape)), dtype=torch.float32)
                for name in model.modalities
            }
            normal_noise((seed,), tuple(noise.values()))
            reference_generator = torch.Generator().manual_seed(seed)
            video = torch.empty(model.noise_shape("video", shape)).normal_(
                generator=reference_generator
            )
            audio = torch.empty(model.noise_shape("audio", shape)).normal_(
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
        schemas = model.denoiser.state_buffers(VideoSize(shape.frames, shape.prompt_tokens))
        buffers = TensorBuffers(
            {name: torch.empty(item.shape, dtype=item.dtype) for name, item in schemas.items()}
        )
        bound = bind_state(
            model.denoiser.state_buffers(VideoSize(shape.frames, shape.prompt_tokens)), buffers
        )
        sources = {name: bound[f"{name}_source"].unsqueeze(0) for name in ("video", "audio")}
        state = {**{name: value.unsqueeze(0) for name, value in bound.items()}, **sources}
        scratch_specs = model.denoiser.workspace_buffers(
            VideoSize(shape.frames, shape.prompt_tokens)
        )
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
    storage = bind_state(
        model.denoiser.state_buffers(VideoSize(shape.frames, shape.prompt_tokens)), buffers
    )
    for target, source in initialize_latents(
        model.denoiser,
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


@pytest.mark.parametrize(
    "sidecar, field, value, error",
    [
        ("vae/config.json", "decoder_num_layers", 35, "video_decoder.decoder_num_layers"),
        ("audio_vae/config.json", "latents_std", [0.0] * 32, "standard deviations"),
        ("vae/config.json", "spatial_downsample_factors", 16, "must be a sequence"),
    ],
)
def test_h3_reader_rejects_unsupported_decoder_math(checkpoint, sidecar, field, value, error):
    path = checkpoint / sidecar
    values = json.loads(path.read_text())
    values[field] = value
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match=error):
        h3_metadata(checkpoint)


def test_h3_reader_reports_missing_decoder_field(checkpoint):
    path = checkpoint / "vae/config.json"
    values = json.loads(path.read_text())
    del values["latent_channels"]
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="video_vae is missing field latent_channels"):
        h3_metadata(checkpoint)


def test_h3_direct_configuration_preserves_cross_component_dimensions():
    from dataclasses import replace

    from uniserve_models.minimax_h3.encoder import H3TextEncoderConfig

    with pytest.raises(ValueError, match="conditioning width"):
        H3Config(text_encoder=replace(H3TextEncoderConfig(), hidden_size=4096))


@pytest.mark.parametrize("steps", [1, 2, 3, 5])
def test_h3_rejects_a_schedule_outside_its_trained_evaluation_count(steps):
    from uniserve.distributed.parallel import ParallelConfig
    from uniserve.nn.diffusion.schedule import DiffusionSchedule
    from uniserve_models.minimax_h3.diffusion import Denoiser

    config = H3Config()
    denoiser = Denoiser(
        config.denoiser,
        config.diffusion,
        transformer=None,
        conditioner=None,
        parallel=ParallelConfig(),
        mesh=None,
        limits=ModelLimits(text_tokens=64, video_frames=22),
    )
    schedule = DiffusionSchedule.build(
        tuple(range(steps, 0, -1)), (1.0, 1.0), scale=float(steps), device="cpu"
    )
    with pytest.raises(ValueError, match="four-evaluation"):
        denoiser.validate_schedule(schedule)


def test_h3_loading_requires_supported_cuda_before_materialization():
    import torch

    from uniserve.distributed.parallel import ParallelConfig
    from uniserve.loading import load_model
    from uniserve_models.minimax_h3 import MiniMaxH3Model

    with pytest.raises(ValueError, match="requires CUDA compute capability 9.0"):
        load_model(
            MiniMaxH3Model,
            H3Config(),
            sources=(),
            device="cpu",
            dtype=torch.bfloat16,
            parallel={"denoiser": ParallelConfig()},
            meshes={},
            layers={},
            limits=ModelLimits(text_tokens=64, video_frames=22),
        )
