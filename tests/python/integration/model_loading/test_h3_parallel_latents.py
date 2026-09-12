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
from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.batch import (
    DiffusionSamplingParams,
    TensorTransfer,
    WorkerEndpoint,
)
from uniserve_worker.execution.bounded_storage import BoundedTensorStorage
from uniserve_worker.execution.model_runner import ModelRunner
from uniserve_worker.loader import LoadRequest, load_model
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
from uniserve_worker.nn.mesh import EntryBindings
from uniserve_worker.nn.parallel import ComponentConfig, ParallelConfig, SequenceParallel
from uniserve_worker.nn.quant.config import resolve_component_precisions
from uniserve_worker.runtime.device_events import DeviceEventPool
from uniserve_worker.runtime.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
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
    rank, rendezvous, checkpoint, kind, encoder_tp, component_precisions, requests, directory
):
    environment = init_distributed_environment(
        rank=rank,
        local_rank=rank,
        world_size=4,
        device=torch.device("cuda", rank),
        backend="nccl",
        init_method=rendezvous,
    )
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
    meshes = initialize_model_parallel(
        environment,
        {
            name: (component.ranks, component.parallel_config)
            for name, component in components.items()
            if component.distribution is None
        },
    )
    bindings = EntryBindings(components, meshes, environment.process_group)
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
    storage = BoundedTensorStorage.allocate(runner.resource_geometry.request_tensors, runner.device)
    execution = ModelRunner(
        runner,
        loaded.worker_config,
        environment=environment,
        schedule=schedule,
    )
    scratch = execution.scratch
    assert scratch is not None
    context_workspace = execution.context_workspace
    transfer_events = DeviceEventPool()
    transport = make_transport(
        "cuda_ipc",
        byte_capacity=runner.product_storage_bytes,
        ticket_capacity=4,
        event_pool=transfer_events,
        source=WorkerEndpoint.local("worker", rank=rank),
    )
    with torch.inference_mode():
        for index, (frames, token_ids) in enumerate(requests):
            print(f"{kind} rank {rank} case {index}: encoding", flush=True)
            media = DiffusionSamplingParams(
                num_frames=frames,
                num_decode_chunks=(frames - 5) // 17,
                seed=0,
                num_inference_steps=4,
            )
            metadata = runner.build_execution(media, len(token_ids), scratch, context_workspace)
            slot = runner.request_tensors(storage, metadata)
            encoded = None
            if runner.text_encoder is not None:
                tokens = execution.stage_text_tokens(token_ids)
                (encoded,) = execution.run_entry("text_encoder", tokens).values
            owner = bindings.output_ranks("text_encoder")[0]
            publication = transport.publish(encoded) if rank == owner else None
            descriptor = [publication]
            # The test coordinator distributes only the physical descriptor;
            # numerical conditioning reaches each input owner through Tensor reads.
            dist.broadcast_object_list(descriptor, src=owner)
            location = descriptor[0]
            tickets = ()
            if rank in bindings.input_ranks("denoiser") and rank != owner:
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

            def execute(name: str, value: torch.Tensor) -> torch.Tensor:
                (result,) = execution.run_module(name, value).values
                return result

            with execution.preparing_inputs(runner.initialize_tensors(slot, 1000 + index)):
                if runner.conditioner is not None:
                    assert encoded is not None
                    encoded = execute("conditioner", encoded)
                runner.prepare_tensors(slot, metadata, encoded, len(token_ids))
            for ticket in tickets:
                ticket.close()
            if bindings.owns("denoiser"):
                for step in range(4):
                    samples = (slot.video_rows, slot.audio_rows)
                    initial = tuple(value.clone() for value in samples)
                    runner.bind_denoising_step(slot, metadata, step, schedule)()
                    eager = tuple(value.clone() for value in samples)
                    for destination, saved in zip(samples, initial, strict=True):
                        destination.copy_(saved)
                    execution.run_denoising(
                        slot,
                        metadata,
                        step,
                        1,
                        schedule,
                        slot=1,
                        geometry=runner.execution_key(media, len(token_ids)),
                    )
                    for observed, expected in zip(samples, eager, strict=True):
                        torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)
                    print(
                        f"{kind} rank {rank} case {index} step {step}: numerical parity", flush=True
                    )
            torch.cuda.synchronize(runner.device)
            if rank in bindings.output_ranks("denoiser"):
                owner = bindings.output_ranks("denoiser").index(rank)
                torch.save(
                    {"video": slot.video_rows.cpu(), "audio": slot.audio_rows.cpu()},
                    Path(directory) / f"case-{index}-owner-{owner}.pt",
                )
            dist.barrier()
            if publication is not None:
                transport.release(publication)
            transfer_events.reap()
    transport.close()
    transfer_events.close()
    execution.synchronize()
    execution.close()
    environment.close()


def _collect(checkpoint, requests, kind, directory, *, encoder_tp=1, component_precisions):
    directory.mkdir()
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
