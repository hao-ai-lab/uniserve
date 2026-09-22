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
        size = DenoiserSize(22, 63)
        stream = CUDAStream.external(torch.cuda.Stream(device=device))
        stream.wait(torch.cuda.current_stream(device))
        with (
            stream,
            ExecutionContext(model, stream=stream, vsa="cute") as execution,
        ):
            execution.prepare(size)
            requirements = model.state_buffers(size)
            with (
                TensorBuffers.allocate(requirements, device="cpu") as host,
                TensorBuffers.allocate(requirements, device=device) as resident,
            ):
                state = resident.view(requirements)
                prepared = {
                    name: value.unsqueeze(0)
                    for name, value in host.view(requirements).items()
                }
                noise = {
                    name: torch.empty(
                        (1, *model.noise_shape(name, size)),
                        dtype=torch.float32,
                        device="cpu",
                    )
                    for name in model.modalities
                }
                normal_noise((923,), out=tuple(noise.values()))
                model.prepare_latents(
                    (size,),
                    noise=noise,
                    state=prepared,
                    constants=execution.constants,
                    workspace=execution.workspace,
                )
                for name, value in state.items():
                    value.copy_(prepared[name][0])
                features = torch.zeros(
                    size.num_text_tokens,
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
                    sizes=(size,),
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
                    state=state,
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
                            == model.output_layout(size)[name]
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
                    state,
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


def _checkpoint(tmp_path):
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
    source = native.state_dict()
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


@torch.inference_mode()
def test_worker_owns_noise_and_replays_one_solver_update(tmp_path):
    from uniserve_worker.model_executor.diffusion_runner import TrajectoryRunner
    from uniserve_worker.model_executor.media_inputs import MediaBuilder

    source = _checkpoint(tmp_path)
    device = torch.device("cuda", 0)
    with initialize_process_groups(
        rank=0, local_rank=0, world_size=1, device=device
    ) as groups:
        mesh = groups.bind(
            DeviceMesh(ranks=(0,), shape=(1, 1), axes=("pp", "tp"), rank=0),
            device=device,
        )
        model = loading.load_model(
            partial(Denoiser, diffusion=DiffusionConfig()),
            _config(),
            checkpoint=(
                checkpoint.Config("denoiser").resolve(
                    tmp_path, io=loading.Config()
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
        factory = MediaBuilder(model, max_frames=22, max_text_tokens=65)
        size = factory.size(22, 63)
        matrices = {name: value.to(device) for name, value in source.items()}
        stream = CUDAStream.external(torch.cuda.Stream(device=device))
        runner = TrajectoryRunner(
            model,
            device=device,
            stream=stream,
            groups=(),
            capacity=2,
        )
        # Slot storage is the request pool's bank, whose rows the captured
        # ladder reaches through the device slot index.
        pool = RequestPool(
            2, state_buffers=factory.capacity_buffers(), device=device
        )
        runner.bind_bank(pool.storage.bank)
        try:
            context = runner.prepare_inputs(size, size)
            with pool.storage.tensors(1) as storage:
                views = storage.view(factory.buffers(size))
                state = {name: views[name] for name in model.modalities}
                views["text_condition"].zero_()
                for seed in (31, 92):
                    schedules = factory.schedules(device=device)
                    copies = factory.initialize(
                        size,
                        views,
                        seed=seed,
                        constants=context.constants,
                        workspace=context.workspace,
                    )
                    for target, value in copies:
                        target.copy_(value, non_blocking=True)
                    initial = {
                        name: value.clone() for name, value in state.items()
                    }
                    inputs = factory.bind(size, views, schedules, 0)
                    runner.warmup(
                        inputs, schedules, state=state, input_key=size
                    )
                    for name in model.modalities:
                        torch.testing.assert_close(
                            state[name], initial[name], rtol=0, atol=0
                        )
                    trajectory = runner.bind_inputs(
                        size,
                        tuple(
                            factory.bind(size, views, schedules, index)
                            for index in range(4)
                        ),
                        schedules,
                        state=views,
                        slot=1,
                    )
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
                        result, _ = runner.step(
                            trajectory,
                            index,
                        )
                        for name in model.modalities:
                            torch.testing.assert_close(
                                result[name][0],
                                expected[name],
                                rtol=2e-2,
                                atol=2e-2,
                            )
                    torch.cuda.synchronize(device)
                    # Retired request schedules may be overwritten immediately.
                    # A later trajectory supplies its own endpoints to replay.
                    for schedule in schedules.values():
                        schedule.timesteps.fill_(float("nan"))
                        schedule.sigmas.fill_(float("nan"))
        finally:
            runner.close()
            pool.close()
            stream.close()
