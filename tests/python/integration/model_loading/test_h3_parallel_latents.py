"""Checkpoint-backed numerical stability at the public H3 denoising boundary."""

import os
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from uniserve_eval.config import load_config
from uniserve_eval.datasets.minimax_h3 import MiniMaxH3Dataset
from uniserve_worker.bootstrap.capacity import product_storage_bytes
from uniserve_worker.bootstrap.distributed import (
    initialize_entries,
    initialize_process_groups,
)
from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.denoising import DenoisingStep, denoising_batch
from uniserve_worker.execution.diffusion_state import DiffusionState
from uniserve_worker.execution.model_runner import ModelRunner
from uniserve_worker.execution.video import initialize_latents, prepare_call
from uniserve_worker.loader import LoadRequest, load_model
from uniserve_worker.modeling.components import Call
from uniserve_worker.modeling.geometry import MediaShape
from uniserve_worker.models.minimax_h3.config import (
    PRECISION_PRESETS,
    PRECISION_SHORTHANDS,
    SUPPORTED_PRECISIONS,
)
from uniserve_worker.models.minimax_h3.packing import (
    audio_latent_frames,
    build_packed_layout,
    unpatchify_video,
    video_latent_frames,
)
from uniserve_worker.nn.parallel import ComponentConfig, ParallelConfig, SequenceParallel
from uniserve_worker.nn.quant.config import resolve_component_precisions
from uniserve_worker.protocol.batch import (
    TensorTransfer,
    WorkerEndpoint,
)
from uniserve_worker.runtime.device_events import EventPool
from uniserve_worker.runtime.tensor_buffers import TensorBuffers
from uniserve_worker.runtime.tensors import (
    stage_tensor,
)
from uniserve_worker.transfer.layout import fetch_tensor
from uniserve_worker.transfer.tickets import make_transport

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.model("minimax_h3"),
    pytest.mark.slow,
]

_LAYOUTS = {
    "u4": ((3, 1, 2, 0), ParallelConfig(sequence_parallel=SequenceParallel("ulysses", (4,)))),
    "local": ((3,), ParallelConfig()),
    "u2": ((3, 1), ParallelConfig(sequence_parallel=SequenceParallel("ulysses", (2,)))),
    "t2": ((3, 1), ParallelConfig(tensor_parallel_size=2)),
    "t4": ((3, 1, 2, 0), ParallelConfig(tensor_parallel_size=4)),
    "t2_u2": (
        (3, 1, 2, 0),
        ParallelConfig(tensor_parallel_size=2, sequence_parallel=SequenceParallel("ulysses", (2,))),
    ),
    "gather2": ((3, 1), ParallelConfig(sequence_parallel=SequenceParallel("allgather", (2,)))),
    "gather4": (
        (3, 1, 2, 0),
        ParallelConfig(sequence_parallel=SequenceParallel("allgather", (4,))),
    ),
    "ring2": ((3, 1), ParallelConfig(sequence_parallel=SequenceParallel("ring", (2,)))),
    "ring4": ((3, 1, 2, 0), ParallelConfig(sequence_parallel=SequenceParallel("ring", (4,)))),
    "attention2d": (
        (3, 1, 2, 0),
        ParallelConfig(sequence_parallel=SequenceParallel("attention2d", (2, 2, 1))),
    ),
    "hybrid": ((3, 1, 2, 0), ParallelConfig(sequence_parallel=SequenceParallel("hybrid", (2, 2)))),
    "pp2": ((3, 1), ParallelConfig(pipeline_parallel_size=2)),
    "pp4": ((3, 1, 2, 0), ParallelConfig(pipeline_parallel_size=4)),
}


def _generate(
    rank,
    rendezvous,
    checkpoint,
    kind,
    encoder_tp,
    component_precisions,
    requests,
    directory,
    completed,
):
    import faulthandler
    import signal

    # Spawned workers expose on-demand Python stacks for distributed stalls.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    environment = initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=4,
        device=torch.device("cuda", rank),
        backend="nccl",
        init_method=rendezvous,
    )
    _generate_requests(
        environment,
        rank,
        checkpoint,
        kind,
        encoder_tp,
        component_precisions,
        requests,
        directory,
        completed,
    )
    # Successful workers release numerical views before collective teardown.
    # On failure, mp.spawn must receive the exception and terminate peers that
    # can still be waiting at the request barrier; teardown would block it.
    environment.close()


def _generate_requests(
    environment,
    rank,
    checkpoint,
    kind,
    encoder_tp,
    component_precisions,
    requests,
    directory,
    completed,
):
    ranks, parallel = _LAYOUTS[kind]
    components = {
        "denoiser": ComponentConfig(ranks, parallel),
        "text_encoder": ComponentConfig(
            (0, 2, 1, 3)[:encoder_tp], ParallelConfig(tensor_parallel_size=encoder_tp)
        ),
        "video_decoder": ComponentConfig((2, 0), distribution="temporal_units"),
        "audio_decoder": ComponentConfig((1,)),
        "output": ComponentConfig((2,)),
    }
    bindings = initialize_entries(environment, components)
    loaded = load_model(
        LoadRequest(
            model_path=checkpoint,
            execution=replace(
                WorkerConfig(model_dtype="bfloat16"),
                attention_backend=None,
                block_size=256,
                device=str(environment.local_device),
                kv_token_capacity=None,
                max_batch_operations=2,
                max_batch_tokens=2,
                rank=rank,
                world_size=4,
            ),
            bindings=bindings,
            max_text_rows=16384,
            max_video_seconds=15,
            quantization_config={"components": component_precisions},
            pipeline_depth=8,
        ),
    )
    runner = loaded.model
    schedule = loaded.schedule
    assert schedule is not None
    execution = ModelRunner(
        runner,
        loaded.worker_config,
        bindings=loaded.bindings,
        schedule=schedule,
    )
    storage = (
        TensorBuffers.allocate(execution.tensor_resources.state, runner.device)
        if execution.tensor_resources.state
        else None
    )
    scratch = execution.scratch
    assert scratch is not None
    transfer_events = EventPool()
    transport = make_transport(
        "cuda_ipc",
        byte_capacity=product_storage_bytes(execution.output_schema),
        ticket_capacity=4,
        event_pool=transfer_events,
        source=WorkerEndpoint.local("worker", rank=rank),
    )
    with torch.inference_mode():
        for index, (frames, token_ids) in enumerate(requests):
            print(f"{kind} rank {rank} case {index}: encoding", flush=True)
            numerical_shape = MediaShape(768, 1344, frames=frames, prompt_tokens=len(token_ids))
            constants = {}
            views_scratch = {}
            slot = {}
            trajectory = DiffusionState(geometry=numerical_shape)
            if bindings["denoiser"].owns:
                assert storage is not None
                slot, constants, views_scratch = prepare_call(
                    runner, execution, trajectory, Call.DIFFUSION, numerical_shape, storage
                )
            encoded = None
            if runner.text_encoder is not None:
                tokens = execution.stage_text_tokens(token_ids)
                (encoded,) = execution.run_encoder("text", tokens).values
            owner = bindings["text_encoder"].output_ranks[0]
            publication = transport.publish(encoded) if rank == owner else None
            descriptor = [publication]
            # The test coordinator distributes only the physical descriptor;
            # numerical conditioning reaches each input owner through Tensor reads.
            dist.broadcast_object_list(descriptor, src=owner)
            location = descriptor[0]
            tickets = ()
            if rank in bindings["denoiser"].input_ranks and rank != owner:
                conditioning = torch.empty(
                    location.shape, dtype=torch.bfloat16, device=runner.device
                )
                tickets = fetch_tensor(
                    TensorTransfer(shape=location.shape, locations=(location,)),
                    conditioning,
                    bindings={(location.source, location.backend): transport},
                )
                for ticket in tickets:
                    ready = Event()
                    ticket.add_done_callback(ready.set)
                    assert ready.wait(30), "conditioning Tensor did not become consumable"
                    ticket.result()
                encoded = conditioning

            initial_latents = (
                initialize_latents(
                    runner,
                    numerical_shape,
                    slot,
                    constants,
                    views_scratch,
                    1000 + index,
                )
                if bindings["denoiser"].owns
                else ()
            )
            with execution.preparing_inputs(initial_latents):
                if "conditioning" in execution.encoder_kinds:
                    assert encoded is not None
                    (encoded,) = execution.run_encoder("conditioning", encoded).values
                    stage_tensor(encoded, slot["text_condition"])
            for ticket in tickets:
                ticket.close()
            if "denoiser" in bindings and bindings["denoiser"].owns:
                for step in range(4):
                    samples = (slot["video"], slot["audio"])
                    initial = tuple(value.clone() for value in samples)
                    batch = denoising_batch(
                        runner,
                        numerical_shape,
                        slot,
                        schedule,
                        step,
                    )
                    assert execution.diffusion is not None and views_scratch is not None
                    with execution.diffusion.attention_scope(trajectory.geometry):
                        DenoisingStep(runner, batch, slot, constants, views_scratch, schedule)()
                    eager = tuple(value.clone() for value in samples)
                    for destination, saved in zip(samples, initial, strict=True):
                        destination.copy_(saved)
                    execution.run_denoising(
                        batch,
                        1,
                        schedule,
                        state=slot,
                        constants=constants,
                        scratch=views_scratch,
                        slot=1,
                        geometry=trajectory.geometry,
                    )
                    for observed, expected in zip(samples, eager, strict=True):
                        torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)
                    print(
                        f"{kind} rank {rank} case {index} step {step}: numerical parity", flush=True
                    )
            torch.cuda.synchronize(runner.device)
            if rank in bindings["denoiser"].output_ranks:
                owner = bindings["denoiser"].output_ranks.index(rank)
                torch.save(
                    {"video": slot["video"].cpu(), "audio": slot["audio"].cpu()},
                    Path(directory) / f"case-{index}-owner-{owner}.pt",
                )
            # Idle component owners can finish long before denoiser ranks. This
            # test-only lifetime join must not enqueue a GPU collective whose
            # watchdog runs while other ranks are still doing numerical work.
            completed.wait()
            if publication is not None:
                transport.release(publication)
            transfer_events.reap()
    transport.close()
    transfer_events.close()
    execution.synchronize()
    execution.close()


def _collect(checkpoint, requests, kind, directory, *, encoder_tp=1, component_precisions):
    directory.mkdir()
    completed = mp.get_context("spawn").Barrier(4)
    mp.spawn(
        _generate,
        (
            (directory / "rendezvous").as_uri(),
            checkpoint,
            kind,
            encoder_tp,
            component_precisions,
            requests,
            str(directory),
            completed,
        ),
        nprocs=4,
        join=True,
    )
    results = []
    owners = _LAYOUTS[kind][1].sequence_parallel_size
    for index, (frames, token_ids) in enumerate(requests):
        shards = [
            torch.load(directory / f"case-{index}-owner-{owner}.pt", weights_only=True)
            for owner in range(owners)
        ]
        packed = build_packed_layout(
            text_rows=((len(token_ids) + 63) // 64) * 64,
            num_frames=frames,
            audio_frames=audio_latent_frames(frames),
            row_multiple=256,
        )
        video_rows = torch.cat([shard["video"] for shard in shards])
        raster = torch.empty_like(video_rows)
        raster[packed.video_raster_indices] = video_rows
        video = unpatchify_video(raster, frames=packed.video_frames, height=48, width=84)
        audio = (
            torch.cat([shard["audio"] for shard in shards])
            .reshape(2, audio_latent_frames(frames), 32)
            .transpose(1, 2)
            .contiguous()
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
    ("precision_name", "kind", "encoder_tp"),
    [("bf16", kind, 1) for kind in _LAYOUTS]
    + [("bf16", "u4", 2), ("bf16", "u4", 4)]
    + [
        (precision, kind, 1)
        for precision in ("fp8", "mxfp8", "nvfp4", "maximum")
        for kind in ("t2", "t4", "u2")
    ]
    + [(precision, "u4", degree) for precision in ("nvfp4", "maximum") for degree in (2, 4)],
)
def test_parallel_layout_generates_finite_latents(
    generation_requests, tmp_path, precision_name, kind, encoder_tp
):
    checkpoint, requests = generation_requests
    component_precisions = dict(
        resolve_component_precisions(
            {"mode": "maximum"}
            if precision_name == "maximum"
            else {"quant_method": precision_name},
            supported=SUPPORTED_PRECISIONS,
            presets=PRECISION_PRESETS,
            shorthands=PRECISION_SHORTHANDS,
            default_mode="balanced",
        )
    )
    actual = _collect(
        checkpoint,
        requests,
        kind,
        tmp_path / f"{precision_name}-{kind}-encoder{encoder_tp}",
        encoder_tp=encoder_tp,
        component_precisions=component_precisions,
    )
    # Floating-point reduction order can change the denoising trajectory.
    # Require usable decoder inputs, not reproduction of another topology.
    for case, ((frames, _), observed) in enumerate(zip(requests, actual, strict=True)):
        assert observed["video"].shape == (1, 24, video_latent_frames(frames), 48, 84)
        assert observed["audio"].shape == (2, 32, audio_latent_frames(frames))
        for modality in ("video", "audio"):
            latent = observed[modality].float()
            assert latent.isfinite().all(), f"{modality}: nonfinite denoised latents"
            energy = latent.square().mean()
            assert energy.isfinite() and energy > 0, f"{modality}: invalid denoised signal energy"
            if modality == "video" and component_precisions["video_vae"] == "fp16":
                assert latent.to(torch.float16).isfinite().all(), "video: FP16 decoder overflow"
            print(
                component_precisions,
                kind,
                "encoder TP",
                encoder_tp,
                case,
                modality,
                "RMS",
                float(energy.sqrt()),
                "max abs",
                float(latent.abs().max()),
                flush=True,
            )
