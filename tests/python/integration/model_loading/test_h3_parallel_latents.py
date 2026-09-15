"""Full H3 checkpoint trajectories produce usable decoder inputs across partitions."""

import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from uniserve.diffusion import DenoisingStep, normal_noise
from uniserve.distributed import DeviceMesh
from uniserve.model import LatentInput, TextSize
from uniserve.nn.attention import AttentionParallelConfig, ContextParallelConfig, Ulysses
from uniserve.runtime import CUDAGraph, ExecutionContext, TensorBuffers, initialize_process_groups
from uniserve_eval.config import load_config
from uniserve_eval.datasets.minimax_h3 import MiniMaxH3Dataset
from uniserve_models import loading as models
from uniserve_models.minimax_h3 import DenoiserInput, DenoiserSize
from uniserve_models.minimax_h3.packing import (
    audio_latent_frames,
    build_packing,
    unpatchify_video,
    video_latent_frames,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.model("minimax_h3"),
    pytest.mark.slow,
]


_LAYOUTS = {
    "ulysses4": ((3, 1, 2, 0), (4,), ("heads",), AttentionParallelConfig(heads=Ulysses("heads"))),
    "local": ((3,), (1,), ("tp",), AttentionParallelConfig()),
    "ulysses2": ((3, 1), (2,), ("heads",), AttentionParallelConfig(heads=Ulysses("heads"))),
    "tp2": ((3, 1), (2,), ("tp",), AttentionParallelConfig()),
    "tp4": ((3, 1, 2, 0), (4,), ("tp",), AttentionParallelConfig()),
    "tp2_ulysses2": (
        (3, 1, 2, 0),
        (2, 2),
        ("tp", "heads"),
        AttentionParallelConfig(heads=Ulysses("heads")),
    ),
    "gather2": (
        (3, 1),
        (2,),
        ("context",),
        AttentionParallelConfig(context=ContextParallelConfig(gather_axis="context")),
    ),
    "gather4": (
        (3, 1, 2, 0),
        (4,),
        ("context",),
        AttentionParallelConfig(context=ContextParallelConfig(gather_axis="context")),
    ),
    "peer2": (
        (3, 1),
        (2,),
        ("context",),
        AttentionParallelConfig(context=ContextParallelConfig(peer_axis="context")),
    ),
    "peer4": (
        (3, 1, 2, 0),
        (4,),
        ("context",),
        AttentionParallelConfig(context=ContextParallelConfig(peer_axis="context")),
    ),
    "gather_ulysses": (
        (3, 1, 2, 0),
        (2, 2),
        ("context", "heads"),
        AttentionParallelConfig(
            heads=Ulysses("heads"), context=ContextParallelConfig(gather_axis="context")
        ),
    ),
    "gather_peer": (
        (3, 1, 2, 0),
        (2, 2),
        ("rows", "columns"),
        AttentionParallelConfig(
            context=ContextParallelConfig(gather_axis="columns", peer_axis="rows")
        ),
    ),
    "pp2": ((3, 1), (2,), ("pp",), AttentionParallelConfig()),
    "pp4": ((3, 1, 2, 0), (4,), ("pp",), AttentionParallelConfig()),
}


@torch.inference_mode()
def _generate(
    rank, rendezvous, checkpoint, kind, encoder_tp, precision, requests, directory, completed
):
    device = torch.device("cuda", rank)
    groups = initialize_process_groups(
        rank=rank, local_rank=rank, world_size=4, device=device, init_method=rendezvous
    )
    ranks, shape, axes, parallel = _LAYOUTS[kind]
    encoder_ranks = (0, 2, 1, 3)[:encoder_tp]
    meshes = {
        "denoiser": groups.bind(
            DeviceMesh(ranks=ranks, shape=shape, axes=axes, rank=rank), device=device
        ),
        "text_encoder": groups.bind(
            DeviceMesh(ranks=encoder_ranks, shape=(encoder_tp,), axes=("tp",), rank=rank),
            device=device,
        ),
    }
    config = models.read_config(checkpoint, modules=frozenset(meshes))
    model = models.load_model(
        config, device=device, precision=precision, meshes=meshes, attention={"denoiser": parallel}
    ).model
    world = groups.process_group
    for index, (frames, token_ids) in enumerate(requests):
        features = torch.empty(
            len(token_ids),
            config.model.text_encoder.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        if rank in encoder_ranks:
            with ExecutionContext(model.text_encoder) as context:
                context.prepare(TextSize(len(token_ids), 1))
                encoded = model.text_encoder.encode((torch.tensor(token_ids, device=device),))[0]
                features.copy_(encoded)
        world.broadcast(features, src=0, out=features)
        if rank in ranks:
            denoiser = model.denoiser
            mesh = meshes["denoiser"]
            pipeline = mesh.get_group("pp" if "pp" in axes else ())
            tensor = mesh.get_group("tp" if "tp" in axes else ())
            size = DenoiserSize(frames, len(token_ids))
            conditioning = torch.empty(
                len(token_ids), denoiser.config.hidden_size, dtype=torch.bfloat16, device=device
            )
            if pipeline.rank == 0:
                with ExecutionContext(denoiser.conditioner) as context:
                    context.prepare(TextSize(len(token_ids), 1))
                    conditioning.copy_(denoiser.conditioner.encode((features,))[0])
            requirements = denoiser.state_buffers(size)
            with (
                ExecutionContext(denoiser) as context,
                TensorBuffers.allocate(requirements, device="cpu") as host,
                TensorBuffers.allocate(requirements, device=device) as backing,
            ):
                context.prepare(size)
                state = backing.view(requirements)
                initial = {
                    name: value.unsqueeze(0) for name, value in host.view(requirements).items()
                }
                noise = {
                    name: torch.empty((1, *denoiser.noise_shape(name, size)), dtype=torch.float32)
                    for name in denoiser.modalities
                }
                normal_noise((1000 + index,), out=tuple(noise.values()))
                denoiser.prepare_latents(
                    (size,),
                    noise=noise,
                    state=initial,
                    constants=context.constants,
                    workspace=context.workspace,
                )
                for name, value in state.items():
                    value.copy_(initial[name][0])
                schedules = denoiser.make_schedules(4, shift=None, device=device)
                for step_index in range(4):
                    inputs = DenoiserInput(
                        {
                            name: (LatentInput(value, schedules[name].timesteps[step_index]),)
                            for name, value in state.items()
                        },
                        (size,),
                        step_index,
                        (conditioning,),
                    )
                    step = DenoisingStep(
                        denoiser, inputs, schedules, state, context.constants, context.workspace
                    )
                    saved = {name: value.clone() for name, value in state.items()}
                    step()
                    expected = {name: value.clone() for name, value in state.items()}

                    def restore():
                        for name, value in state.items():
                            value.copy_(saved[name])

                    restore()
                    with CUDAGraph(context=context) as graph:
                        graph.capture(step, restore=restore)
                        graph.replay()
                        for name, value in state.items():
                            torch.testing.assert_close(value, expected[name], rtol=2e-2, atol=2e-2)
                if pipeline.rank + 1 == pipeline.size and tensor.rank == 0:
                    layouts = denoiser.output_layout(size)
                    torch.save(
                        {
                            name: {
                                "start": layouts[name].local_slice[0].start,
                                "value": value.cpu(),
                            }
                            for name, value in state.items()
                        },
                        Path(directory) / f"case-{index}-rank-{rank}.pt",
                    )
            torch.cuda.synchronize(device)
        # Independent component owners may finish at different times. A host
        # join retires this trajectory before any rank starts the next one.
        completed.wait()
    groups.close()


def _collect(checkpoint, requests, kind, directory, *, encoder_tp, precision):
    directory.mkdir()
    completed = mp.get_context("spawn").Barrier(4)
    mp.spawn(
        _generate,
        args=(
            (directory / "world").as_uri(),
            checkpoint,
            kind,
            encoder_tp,
            precision,
            requests,
            directory,
            completed,
        ),
        nprocs=4,
        join=True,
    )
    results = []
    for index, (frames, token_ids) in enumerate(requests):
        shards = [
            torch.load(path, weights_only=True)
            for path in directory.glob(f"case-{index}-rank-*.pt")
        ]
        joined = {
            name: torch.cat(
                [
                    entry[name]["value"]
                    for entry in sorted(shards, key=lambda entry: entry[name]["start"])
                ]
            )
            for name in ("video", "audio")
        }
        packing = build_packing(num_text_tokens=len(token_ids), num_frames=frames)
        raster = torch.empty_like(joined["video"])
        raster[packing.video_raster_indices] = joined["video"]
        video = unpatchify_video(raster, frames=packing.video_frames, height=48, width=84)
        audio = (
            joined["audio"].reshape(2, audio_latent_frames(frames), 32).transpose(1, 2).contiguous()
        )
        results.append({"video": video, "audio": audio})
    return results


@pytest.fixture(scope="module")
def generation_requests():
    checkpoint = os.environ.get("UNISERVE_H3_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail(
            "UNISERVE_H3_MODEL must name the supported full FastH3 VSA checkpoint directory"
        )
    tokenizer = AutoTokenizer.from_pretrained(Path(checkpoint) / "tokenizer")
    point = load_config().benchmarks["minimax-h3-5s-1k"]
    requests = []
    for seconds, frames, tokens in ((5, 124, 1000), (15, 362, 16384)):
        case = replace(
            point,
            load=replace(point.load, num_prompts=1),
            video=replace(point.video, seconds=seconds, prompt_tokens=tokens),
        )
        example = MiniMaxH3Dataset(case).load(tokenizer)[0]
        requests.append((frames, tuple(tokenizer.encode(example.prompt, add_special_tokens=False))))
    tokens = requests[0][1]
    # Exact tile boundaries and a different prompt with the same signature expose
    # stale conditioning, while the minimum frame count exercises decoder geometry.
    requests.extend((22, tokens[:length]) for length in (63, 64, 65))
    requests.append((22, tokens[1:65]))
    return checkpoint, requests


@pytest.mark.parametrize(
    "precision,kind,encoder_tp",
    [("bf16", kind, 1) for kind in _LAYOUTS]
    + [("bf16", "ulysses4", 2), ("bf16", "ulysses4", 4)]
    + [(precision, "tp4", 1) for precision in ("fp8", "mxfp8", "nvfp4", "maximum")]
    + [("fp8", "ulysses2", 1)]
    + [(precision, "ulysses4", 4) for precision in ("nvfp4", "maximum")],
)
def test_parallel_layout_generates_usable_decoder_latents(
    generation_requests, tmp_path, precision, kind, encoder_tp
):
    checkpoint, requests = generation_requests
    actual = _collect(
        checkpoint, requests, kind, tmp_path / "latents", encoder_tp=encoder_tp, precision=precision
    )
    for (frames, _), observed in zip(requests, actual, strict=True):
        assert observed["video"].shape == (1, 24, video_latent_frames(frames), 48, 84)
        assert observed["audio"].shape == (2, 32, audio_latent_frames(frames))
        for modality in ("video", "audio"):
            latent = observed[modality].float()
            assert latent.isfinite().all(), f"{modality}: nonfinite denoised latents"
            energy = latent.square().mean()
            assert energy.isfinite() and energy > 0, f"{modality}: invalid denoised signal energy"
            if modality == "video" and precision in ("bf16", "fp8", "mxfp8"):
                assert latent.to(torch.float16).isfinite().all(), "video: FP16 decoder overflow"
