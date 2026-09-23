"""H3's public denoiser composes latent packing.

It also composes partitions and solver feedback.
"""

from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Transformer3DModel,
)
from safetensors.torch import save_file
from torch.nn import functional as F

from uniserve import loading
from uniserve.diffusion import DenoisingStep, normal_noise
from uniserve.distributed import DeviceMesh, communication_axes
from uniserve.loading import checkpoint, weights
from uniserve.model import LatentInput
from uniserve.nn.attention import (
    AttentionParallelConfig,
    ContextParallelConfig,
    Ulysses,
)
from uniserve.runtime import (
    CUDAGraph,
    CUDAStream,
    ExecutionContext,
    TensorBuffers,
    initialize_process_groups,
)
from uniserve_models.minimax_h3 import (
    Denoiser,
    DenoiserInput,
    DenoiserSize,
    DiffusionConfig,
    TransformerConfig,
)
from uniserve_models.minimax_h3.conditioning import (
    assignments as conditioning_assignments,
)
from uniserve_models.minimax_h3.config import TRANSFORMER_FIELDS
from uniserve_models.minimax_h3.weights import transformer_component
from uniserve_worker.execution.request import RequestPool

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _config():
    return TransformerConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_refiner_layers=1,
        intermediate_size=128,
        video_channels=2,
        audio_channels=2,
        text_dim=40,
        frequency_dim=16,
        time_hidden_dim=64,
        time_dim=32,
        rope_frequency_dim=4,
    )


def _mapping(model):
    component = transformer_component(model.transformer, model.diffusion)
    pipeline = model.mesh.get_group("pp")
    if pipeline.rank:
        model.conditioner = None
        return (component,)
    conditioner = model.conditioner
    return (
        component,
        weights.ModuleMapping(
            conditioner,
            "denoiser",
            lambda reader: tuple(conditioning_assignments(conditioner, reader)),
            frozenset(name for name, _ in conditioner.named_parameters()),
        ),
    )


def _prediction(sample, source, name):
    prefix = "" if name == "video" else "audio_"
    hidden = F.linear(
        sample,
        source[prefix + "proj_in.weight"],
        source[prefix + "proj_in.bias"],
    ).bfloat16()
    normalized = (
        hidden.float()
        * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + 1e-5)
    ).bfloat16()
    shift, scale = source["norm_out.linear.bias"].bfloat16().chunk(2)
    normalized = normalized * (1.0 + scale) + shift
    return F.linear(
        normalized.float(),
        source[prefix + "proj_out.weight"],
        source[prefix + "proj_out.bias"],
    )


@torch.inference_mode()
def _run(rank, rendezvous, directory, source):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    groups = initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=4,
        device=device,
        init_method=rendezvous,
    )
    cases = (
        (
            (1, 1, 4),
            ("pp", "tp", "heads"),
            AttentionParallelConfig(heads=Ulysses("heads")),
        ),
        (
            (1, 2, 2),
            ("pp", "tp", "heads"),
            AttentionParallelConfig(heads=Ulysses("heads")),
        ),
        (
            (2, 1, 2),
            ("pp", "tp", "heads"),
            AttentionParallelConfig(heads=Ulysses("heads")),
        ),
        (
            (1, 1, 4),
            ("pp", "tp", "context"),
            AttentionParallelConfig(
                context=ContextParallelConfig(gather_axis="context")
            ),
        ),
    )
    matrices = {
        name: value.to(device)
        for name, value in source.items()
        if name.startswith(("proj_", "audio_proj_", "norm_out.linear"))
    }
    for shape, axes, attention in cases:
        topology = DeviceMesh(
            ranks=(3, 1, 0, 2), shape=shape, axes=axes, rank=rank
        )
        with torch.device("meta"):
            description = Denoiser(_config(), diffusion=DiffusionConfig())
        mesh = groups.bind(
            topology,
            device=device,
            axes=communication_axes(description, topology, attention=attention),
        )
        model = loading.load_model(
            partial(Denoiser, diffusion=DiffusionConfig()),
            _config(),
            checkpoint=(
                checkpoint.Config("denoiser").resolve(
                    directory, io=loading.Config()
                ),
            ),
            mapping=_mapping,
            device=device,
            meshes={"": mesh},
            attention={"": attention},
            weights=weights.Config(
                dtypes={
                    f"transformer.{name}": torch.float32
                    for name in (
                        "video_input",
                        "audio_input",
                        "video_output",
                        "audio_output",
                    )
                }
            ),
        ).model
        # The prompt fills 63 rows of one text tile; the call evaluates the
        # tile's layout with the prompt's own tables.
        size = DenoiserSize(22, 63)
        layout = model.layout_size(size)
        stream = CUDAStream.external(torch.cuda.Stream(device=device))
        stream.wait(torch.cuda.current_stream(device))
        with (
            stream,
            ExecutionContext(model, stream=stream, vsa="cute") as execution,
        ):
            execution.prepare(layout)
            requirements = model.state_buffers(layout)
            with (
                TensorBuffers.allocate(requirements, device="cpu") as host,
                TensorBuffers.allocate(requirements, device=device) as resident,
            ):
                request = resident.view(requirements)
                staged = host.view(requirements)
                state = {name: request[name] for name in model.modalities}
                noise = {
                    name: torch.empty(
                        (1, *model.noise_shape(name, layout)),
                        dtype=torch.float32,
                        device="cpu",
                    )
                    for name in model.modalities
                }
                normal_noise((923,), out=tuple(noise.values()))
                model.prepare_latents(
                    (layout,),
                    noise=noise,
                    state={
                        name: staged[name].unsqueeze(0)
                        for name in model.modalities
                    },
                    constants=execution.constants,
                    workspace=execution.workspace,
                )
                model.prepare_state(
                    (size,),
                    out={
                        name: value
                        for name, value in staged.items()
                        if name not in model.modalities
                    },
                )
                for name, value in request.items():
                    value.copy_(staged[name])
                features = torch.zeros(
                    layout.num_text_tokens,
                    model.config.hidden_size,
                    device=device,
                    dtype=torch.bfloat16,
                )
                schedules = model.make_schedules(4, shift=None, device=device)
                assert model.prediction_type == "velocity"
                inputs = DenoiserInput(
                    latents={
                        name: (
                            LatentInput(value, schedules[name].timesteps[1]),
                        )
                        for name, value in state.items()
                    },
                    sizes=(layout,),
                    step_index=1,
                    text_features=(features,),
                )
                initial = {name: value.clone() for name, value in state.items()}
                predicted = {
                    name: _prediction(value, matrices, name)
                    for name, value in initial.items()
                }
                output = model(
                    inputs,
                    state=request,
                    constants=execution.constants,
                    workspace=execution.workspace,
                )
                for name in model.modalities:
                    if mesh.get_group("pp").rank + 1 == mesh.size("pp"):
                        torch.testing.assert_close(
                            output[name][0].tensor,
                            predicted[name],
                            rtol=2e-2,
                            atol=2e-2,
                        )
                        assert (
                            output[name][0].layout
                            == model.output_layout(layout)[name]
                        )
                    else:
                        assert output[name] == (None,)
                    torch.testing.assert_close(
                        state[name], initial[name], rtol=0, atol=0
                    )
                step = DenoisingStep(
                    model,
                    inputs,
                    schedules,
                    request,
                    execution.constants,
                    execution.workspace,
                )
                expected = {}
                for name, value in initial.items():
                    schedule = schedules[name]
                    ratio = schedule.sigmas[2] / schedule.sigmas[1]
                    clean = (
                        value.double()
                        + predicted[name].double()
                        * (1.0 - schedule.timesteps[1]).double()
                    )
                    expected[name] = (
                        ratio.double() * value.double()
                        + (1.0 - ratio).double() * clean
                    ).float()

                def restore():
                    for name, value in state.items():
                        value.copy_(initial[name])

                step()
                restore()
                with CUDAGraph(context=execution) as graph:
                    graph.capture(step, restore=restore)
                    result = graph.replay()
                    for name in model.modalities:
                        if mesh.get_group("pp").rank in (
                            0,
                            mesh.size("pp") - 1,
                        ):
                            torch.testing.assert_close(
                                result[name][0],
                                expected[name],
                                rtol=2e-2,
                                atol=2e-2,
                            )
                torch.cuda.synchronize(device)
    # Exceptions leave teardown to multiprocessing, so a failing rank can
    # report its numerical error without waiting for another rank's collective.
    groups.close()


def _native_weights():
    config = _config()
    torch.manual_seed(922)
    native = MiniMaxH3Transformer3DModel(
        **{
            source: getattr(config, target)
            for source, target in TRANSFORMER_FIELDS.items()
        },
        patch_size=(1, 2, 2),
        final_norm_eps=config.norm_eps,
    )
    return native.state_dict()


def _checkpoint(tmp_path):
    """Save weights whose prediction has a closed form: attention is zero."""
    config = _config()
    source = _native_weights()
    for value in source.values():
        value.zero_()
    for name, value in source.items():
        if name.endswith("weight") and "norm" in name:
            value.fill_(1.0)
        if name.startswith(("proj_", "audio_proj_")):
            value.normal_(0.0, 0.2 if name.endswith("weight") else 0.05)
    source["norm_out.linear.bias"][: config.hidden_size] = torch.linspace(
        -0.25, 0.25, config.hidden_size
    )
    source["norm_out.linear.bias"][config.hidden_size :] = 0.125
    for index in range(config.num_hidden_layers):
        source[f"transformer_blocks.{index}.attn.to_gate_compress.weight"] = (
            torch.zeros(
                config.num_attention_heads * config.head_dim, config.hidden_size
            )
        )
    save_file(source, tmp_path / "model.safetensors")
    return source


def test_partitioned_denoising_and_feedback(tmp_path):
    source = _checkpoint(tmp_path)
    mp.spawn(
        _run,
        args=((tmp_path / "rendezvous").as_uri(), tmp_path, source),
        nprocs=4,
        join=True,
    )


def _load(groups, directory, device):
    """Load the tiny denoiser on one device as a worker binds it."""
    mesh = groups.bind(
        DeviceMesh(ranks=(0,), shape=(1, 1), axes=("pp", "tp"), rank=0),
        device=device,
    )
    return loading.load_model(
        partial(Denoiser, diffusion=DiffusionConfig()),
        _config(),
        checkpoint=(
            checkpoint.Config("denoiser").resolve(
                directory, io=loading.Config()
            ),
        ),
        mapping=_mapping,
        device=device,
        meshes={"": mesh},
        weights=weights.Config(
            dtypes={
                f"transformer.{name}": torch.float32
                for name in (
                    "video_input",
                    "audio_input",
                    "video_output",
                    "audio_output",
                )
            }
        ),
    ).model


def _stage(factory, size, views, context, *, seed, features):
    """Stage one admitted request into its slot as the worker does."""
    factory.stage_request(size, views, seed=seed)
    for target, value in factory.initialize(
        size,
        views,
        constants=context.constants,
        workspace=context.workspace,
    ):
        target.copy_(value, non_blocking=True)
    factory.store_conditioning(size, views, features)


def _diffusion(model, layout, *, device, stream, bank=None):
    """Prepare the denoiser's runner for one layout over two request slots."""
    from uniserve.model import EntryPoint
    from uniserve_worker.model_executor.component_binding import Call
    from uniserve_worker.model_executor.diffusion_runner import DiffusionRunner
    from uniserve_worker.model_executor.graph_storage import GraphStorage

    return DiffusionRunner.for_layout(
        "denoiser",
        Call("denoiser", model, EntryPoint("forward")),
        layout,
        device=device,
        stream=stream,
        storage=GraphStorage(),
        devices=(device,) if stream is not None else (),
        bank=bank,
        slots=2,
    )


@torch.inference_mode()
def test_worker_owns_noise_and_replays_one_solver_update(tmp_path):
    from uniserve_worker.model_executor.media_inputs import MediaBuilder

    source = _checkpoint(tmp_path)
    device = torch.device("cuda", 0)
    with initialize_process_groups(
        rank=0, local_rank=0, world_size=1, device=device
    ) as groups:
        model = _load(groups, tmp_path, device)
        factory = MediaBuilder(model, max_frames=22, max_text_tokens=65)
        size = factory.size(22, 63)
        layout = factory.layout(size)
        matrices = {name: value.to(device) for name, value in source.items()}
        features = torch.zeros(
            size.num_text_tokens,
            model.config.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        stream = CUDAStream.external(torch.cuda.Stream(device=device))
        # Slot storage is the request pool's bank, whose rows the captured
        # ladder reaches through the device slot index.
        pool = RequestPool(
            2, state_buffers=factory.capacity_buffers(), device=device
        )
        runner = _diffusion(
            model, layout, device=device, stream=stream, bank=pool.storage.bank
        )
        try:
            context = runner.context
            with pool.storage.tensors(1) as storage:
                views = storage.view(factory.buffers(size))
                state = {name: views[name] for name in model.modalities}
                for seed in (31, 92):
                    schedules = model.make_schedules(
                        factory.num_steps, shift=None, device=device
                    )
                    _stage(
                        factory,
                        size,
                        views,
                        context,
                        seed=seed,
                        features=features,
                    )
                    initial = {
                        name: value.clone() for name, value in state.items()
                    }
                    inputs = factory.bind(size, views, schedules, 0)
                    runner.warmup(inputs, schedules, state=views)
                    for name in model.modalities:
                        torch.testing.assert_close(
                            state[name], initial[name], rtol=0, atol=0
                        )
                    ladder = runner.bind(
                        tuple(
                            factory.bind(size, views, schedules, index)
                            for index in range(4)
                        ),
                        schedules,
                        state=views,
                        slot=1,
                    )
                    # Startup captures the first request's ladder; the second
                    # request, with its own schedules, replays it.
                    if seed == 31:
                        for index in range(4):
                            runner.capture(ladder, index)
                    for index in range(4):
                        expected = {}
                        for name, value in state.items():
                            prediction = _prediction(value, matrices, name)
                            schedule = schedules[name]
                            ratio = (
                                schedule.sigmas[index + 1]
                                / schedule.sigmas[index]
                            )
                            clean = (
                                value.double()
                                + prediction.double()
                                * (1.0 - schedule.timesteps[index]).double()
                            )
                            expected[name] = (
                                ratio.double() * value.double()
                                + (1.0 - ratio).double() * clean
                            ).float()
                        result, path = runner.step(ladder, index)
                        assert path == "graph_replay"
                        for name in model.modalities:
                            torch.testing.assert_close(
                                result[name][0],
                                expected[name],
                                rtol=2e-2,
                                atol=2e-2,
                            )
                    torch.cuda.synchronize(device)
                    # Retired request schedules may be overwritten immediately.
                    # A later ladder supplies its own endpoints to replay.
                    for schedule in schedules.values():
                        schedule.timesteps.fill_(float("nan"))
                        schedule.sigmas.fill_(float("nan"))
        finally:
            runner.close()
            pool.close()
            stream.close()


@torch.inference_mode()
def test_prompt_lengths_of_one_layout_replay_its_ladder_exactly(tmp_path):
    """One captured ladder serves every prompt length within its layout.

    Startup captures the ladder with a placeholder prompt on one slot. Two
    requests whose prompts fill the same text tile to different lengths then
    replay it from both slots, reading their own tile valid counts and rotary
    tables from their slots, and each final sample equals its eager
    evaluation bit for bit.
    """
    from uniserve_worker.model_executor.media_inputs import MediaBuilder

    # Every weight is live, so attention depends on the prompt length.
    config = _config()
    source = _native_weights()
    for index in range(config.num_hidden_layers):
        source[f"transformer_blocks.{index}.attn.to_gate_compress.weight"] = (
            torch.randn(
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
            )
            * 0.1
        )
    save_file(source, tmp_path / "model.safetensors")
    device = torch.device("cuda", 0)
    with initialize_process_groups(
        rank=0, local_rank=0, world_size=1, device=device
    ) as groups:
        model = _load(groups, tmp_path, device)
        factory = MediaBuilder(model, max_frames=22, max_text_tokens=128)
        placeholder = factory.size(22, 1)
        requests = {1: factory.size(22, 40), 2: factory.size(22, 63)}
        layout = factory.layout(placeholder)
        assert {factory.layout(size) for size in requests.values()} == {layout}
        generator = torch.Generator().manual_seed(5)
        features = {
            slot: torch.randn(
                size.num_text_tokens,
                model.config.hidden_size,
                generator=generator,
            ).to(device, torch.bfloat16)
            for slot, size in requests.items()
        }

        def denoise(runner, pool, context):
            samples, paths = {}, []
            for slot, size in requests.items():
                views = pool.storage.tensors(slot).view(factory.buffers(size))
                _stage(
                    factory,
                    size,
                    views,
                    context,
                    seed=100 + slot,
                    features=features[slot],
                )
                schedules = model.make_schedules(
                    factory.num_steps, shift=None, device=device
                )
                ladder = runner.bind(
                    tuple(
                        factory.bind(size, views, schedules, index)
                        for index in range(4)
                    ),
                    schedules,
                    state=views,
                    slot=slot,
                )
                for index in range(4):
                    result, path = runner.step(ladder, index)
                    paths.append(path)
                samples[slot] = {
                    name: values[0].clone() for name, values in result.items()
                }
            return samples, paths

        pool = RequestPool(
            2, state_buffers=factory.capacity_buffers(), device=device
        )
        try:
            eager = _diffusion(model, layout, device=device, stream=None)
            try:
                expected, paths = denoise(eager, pool, eager.context)
                assert set(paths) == {"eager"}
            finally:
                torch.cuda.current_stream(device).synchronize()
                eager.close()

            stream = CUDAStream.external(torch.cuda.Stream(device=device))
            runner = _diffusion(
                model,
                layout,
                device=device,
                stream=stream,
                bank=pool.storage.bank,
            )
            try:
                context = runner.context
                views = pool.storage.tensors(1).view(
                    factory.buffers(placeholder)
                )
                _stage(
                    factory,
                    placeholder,
                    views,
                    context,
                    seed=0,
                    features=torch.zeros(
                        1,
                        model.config.hidden_size,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                )
                schedules = model.make_schedules(
                    factory.num_steps, shift=None, device=device
                )
                ladder = runner.bind(
                    tuple(
                        factory.bind(placeholder, views, schedules, index)
                        for index in range(4)
                    ),
                    schedules,
                    state=views,
                    slot=1,
                )
                for index in range(4):
                    runner.capture(ladder, index)

                actual, paths = denoise(runner, pool, context)
                assert paths == ["graph_replay"] * 8
            finally:
                torch.cuda.current_stream(device).synchronize()
                runner.close()
                stream.close()
        finally:
            pool.close()

        for slot in requests:
            for name in model.modalities:
                torch.testing.assert_close(
                    actual[slot][name], expected[slot][name], rtol=0, atol=0
                )
        # The two prompt lengths are distinct computations.
        assert not torch.equal(expected[1]["video"], expected[2]["video"])
