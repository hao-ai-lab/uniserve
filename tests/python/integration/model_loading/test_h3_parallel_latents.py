"""Checkpoint-backed numerical stability at the public H3 denoising boundary."""

import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from uniserve_eval.config import load_config
from uniserve_eval.datasets.minimax_h3 import MiniMaxH3Dataset
from uniserve_worker.bootstrap.plan import ComponentDeployConfig
from uniserve_worker.execution.batch import (
    DiffusionRequestParams,
    MediaGeometry,
    NewRequest,
    RequestKey,
)
from uniserve_worker.models.minimax_h3.model import MiniMaxH3Runner
from uniserve_worker.models.minimax_h3.packing import (
    audio_latent_frames,
    build_packed_layout,
    unpatchify_video,
    video_latent_frames,
)
from uniserve_worker.models.minimax_h3.placement import H3Placement
from uniserve_worker.models.minimax_h3.precision import H3LinearPrecisionPolicy
from uniserve_worker.nn.parallel import (
    Attention2DSequence,
    GatherSequence,
    HybridSequence,
    ParallelConfig,
    RingSequence,
    UlyssesSequence,
)
from uniserve_worker.runtime.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.model("minimax_h3"),
    pytest.mark.slow,
]

_LAYOUTS = {
    "u4": ((3, 1, 2, 0), ParallelConfig(sequence_parallel=UlyssesSequence(4))),
    "local": ((3,), ParallelConfig()),
    "u2": ((3, 1), ParallelConfig(sequence_parallel=UlyssesSequence(2))),
    "t2": ((3, 1), ParallelConfig(tensor_parallel_size=2)),
    "t4": ((3, 1, 2, 0), ParallelConfig(tensor_parallel_size=4)),
    "t2_u2": (
        (3, 1, 2, 0),
        ParallelConfig(tensor_parallel_size=2, sequence_parallel=UlyssesSequence(2)),
    ),
    "gather2": ((3, 1), ParallelConfig(sequence_parallel=GatherSequence(2))),
    "gather4": ((3, 1, 2, 0), ParallelConfig(sequence_parallel=GatherSequence(4))),
    "ring2": ((3, 1), ParallelConfig(sequence_parallel=RingSequence(2))),
    "ring4": ((3, 1, 2, 0), ParallelConfig(sequence_parallel=RingSequence(4))),
    "attention2d": ((3, 1, 2, 0), ParallelConfig(sequence_parallel=Attention2DSequence(2, 2))),
    "hybrid": ((3, 1, 2, 0), ParallelConfig(sequence_parallel=HybridSequence(2, 2))),
    "pp2": ((3, 1), ParallelConfig(pipeline_parallel_size=2)),
    "pp4": ((3, 1, 2, 0), ParallelConfig(pipeline_parallel_size=4)),
}


def _generate(
    rank, rendezvous, checkpoint, kind, encoder_tp, precision_policy, requests, directory
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
        "denoiser": ComponentDeployConfig(ranks, parallel),
        "text_encoder": ComponentDeployConfig(
            (0, 2, 1, 3)[:encoder_tp], ParallelConfig(tensor_parallel_size=encoder_tp)
        ),
        "video_decoder": ComponentDeployConfig((2, 0), distribution="temporal_units"),
        "audio_decoder": ComponentDeployConfig((1,)),
        "output": ComponentDeployConfig((2,)),
    }
    meshes = initialize_model_parallel(
        environment,
        {
            name: (component.ranks, component.parallel_config)
            for name, component in components.items()
            if component.distribution is None
        },
    )
    placement = H3Placement(components, meshes, environment.process_group)
    runner = MiniMaxH3Runner.from_pretrained(
        checkpoint,
        placement,
        max_state_slots=2,
        max_text_rows=16384,
        max_video_seconds=15,
        precision_policy=precision_policy,
    )
    pool = runner.create_request_state()
    with torch.inference_mode():
        for index, (frames, token_ids) in enumerate(requests):
            admission = NewRequest.create(
                RequestKey(1, index + 1, 0),
                request_pool_idx=1,
                diffusion=DiffusionRequestParams(
                    prompt_token_ids=token_ids,
                    seed=1000,
                    geometry=MediaGeometry(
                        frame_count=frames,
                        decode_units=((frames - 5) // 17 + 1) // 2 + 2,
                        prompt_tokens=len(token_ids),
                        denoise_steps=4,
                    ),
                ),
            )
            slot = pool.slots[0]
            runner.prepare(slot, admission)
            for step in range(4):
                runner.denoise(slot, step, 1)
            runner.synchronize_runtime()
            if rank in placement.latent_producers:
                owner = placement.latent_producers.index(rank)
                torch.save(
                    {"video": slot.video_rows.cpu(), "audio": slot.audio_rows.cpu()},
                    Path(directory) / f"case-{index}-owner-{owner}.pt",
                )
            slot.clear()
    del pool, runner
    environment.close()
    dist.destroy_process_group()


def _collect(checkpoint, requests, kind, directory, *, encoder_tp=1, precision_policy):
    directory.mkdir()
    mp.spawn(
        _generate,
        (
            (directory / "rendezvous").as_uri(),
            checkpoint,
            kind,
            encoder_tp,
            precision_policy,
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
        pytest.fail("UNISERVE_H3_MODEL must name the FastH3 Preview v0.2 checkpoint directory")
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
    precision_policy = (
        H3LinearPrecisionPolicy.from_mode("maximum")
        if precision_name == "maximum"
        else H3LinearPrecisionPolicy.resolve(precision_name)
    )
    actual = _collect(
        checkpoint,
        requests,
        kind,
        tmp_path / f"{precision_name}-{kind}-encoder{encoder_tp}",
        encoder_tp=encoder_tp,
        precision_policy=precision_policy,
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
            if modality == "video" and precision_policy.video_vae == "fp16":
                assert latent.to(torch.float16).isfinite().all(), "video: FP16 decoder overflow"
            print(
                precision_policy,
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
