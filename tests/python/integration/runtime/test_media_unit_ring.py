"""Media units reconstructed across ranks reproduce the whole-track pixels.

Post-processing blends each media unit's leading frames with the overlap its
predecessor decoded. When consecutive units are held by different ranks that
overlap crosses the ring between them, so the reconstruction has to be
indistinguishable from one rank reconstructing the whole track in order.
"""

from __future__ import annotations

import pytest
import torch

from uniserve.media import image, video
from uniserve.model import VideoPostprocessor
from uniserve.tensors import OutputLayout, TensorOutput

pytestmark = pytest.mark.integration

UNITS = 6
EXTENT = 2
BODY = 3
WINDOW = 6
HEIGHT, WIDTH = 2, 3
FRAME = image.Config(HEIGHT, WIDTH)


class _Blend(VideoPostprocessor):
    """Three body frames and a two-frame successor overlap per window."""

    def reconstruction_slices(self, frames, num_frames):
        return slice(0, BODY), slice(WINDOW - EXTENT, WINDOW)


def _postprocessor() -> _Blend:
    return _Blend(
        torch.tensor([0.0, 0.5], dtype=torch.float16),
        frame_rate=24,
    )


def _windows() -> tuple[slice, ...]:
    return tuple(
        slice(
            index * BODY,
            index * BODY + (EXTENT + BODY if index == UNITS - 1 else BODY),
        )
        for index in range(UNITS)
    )


def _segment(unit: int) -> torch.Tensor:
    values = torch.arange(WINDOW, dtype=torch.float16) / WINDOW + unit
    return (
        values.reshape(1, 1, WINDOW, 1, 1)
        .expand(1, 3, WINDOW, HEIGHT, WIDTH)
        .contiguous()
    )


def _resources():
    constants = {
        "pixel_mean": torch.zeros((1, 3, 1, 1, 1)),
        "pixel_std": torch.ones((1, 3, 1, 1, 1)),
    }
    workspace = {
        "rgb_frames": torch.empty(
            (EXTENT + BODY, HEIGHT, WIDTH, 3), dtype=torch.uint8
        ),
        "overlap_exchange": torch.empty(
            (1, 3, EXTENT, HEIGHT, WIDTH), dtype=torch.float16
        ),
    }
    state = {
        "video_overlap": torch.zeros(
            (1, 3, EXTENT, HEIGHT, WIDTH), dtype=torch.float16
        )
    }
    return constants, workspace, state


def _output(unit: int, total: int) -> TensorOutput:
    """One decoded native window, as the decoder hands it to this rank."""
    value = _segment(unit)
    return TensorOutput(
        value,
        OutputLayout(
            tuple(value.shape),
            value.dtype,
            tuple(slice(0, size) for size in value.shape),
        ),
    )


def _serial() -> list[torch.Tensor]:
    """Reconstruct every media unit on one rank, in order."""
    model = _postprocessor()
    constants, workspace, state = _resources()
    frames = _windows()
    total = frames[-1].stop
    return [
        model(
            (_output(unit, total),),
            frames=(frames[unit],),
            sizes=(video.Config(total, FRAME),),
            state=state,
            constants=constants,
            workspace=workspace,
        )[0].tensor.clone()
        for unit in range(UNITS)
    ]


def _ring(rank: int, expected_bytes: bytes, device: torch.device) -> None:
    import pickle
    from contextlib import ExitStack

    import torch.distributed as dist

    from uniserve.distributed import Communicator
    from uniserve.runtime import CUDAStream, ExecutionContext

    expected = pickle.loads(expected_bytes)
    model = _postprocessor().to(device)
    constants, workspace, state = _resources()
    constants, workspace, state = (
        {name: value.to(device) for name, value in values.items()}
        for values in (constants, workspace, state)
    )
    frames = _windows()
    total = frames[-1].stop
    ranks = tuple(range(dist.get_world_size()))
    model.units = Communicator(
        ranks=ranks,
        rank=rank,
        name="units",
        device=device,
        _group=dist.group.WORLD,
    )

    with ExitStack() as scope:
        if device.type == "cuda":
            stream = CUDAStream.external(torch.cuda.Stream(device=device))
            stream.wait(torch.cuda.current_stream(device))
            # The stream retires its communicators after the context.
            scope.callback(stream.close)
            context = ExecutionContext(
                model, stream=stream, groups=(model.units,)
            )
            scope.callback(context.close)
            context.prepare(video.Config(total, FRAME))
            scope.enter_context(context.activate())
        cursor = 0
        while cursor < UNITS:
            count = min(len(ranks), UNITS - cursor)
            if rank < count:
                unit = cursor + rank
                output = _output(unit, total)
                output = TensorOutput(output.tensor.to(device), output.layout)
                produced = model(
                    (output,),
                    frames=(frames[unit],),
                    sizes=(video.Config(total, FRAME),),
                    state=state,
                    constants=constants,
                    workspace=workspace,
                    unit_count=count,
                )[0].tensor
                assert torch.equal(produced.cpu(), expected[unit]), (
                    f"media unit {unit} on rank {rank} differs from the "
                    "whole-track reconstruction"
                )
            cursor += count


def _worker(
    rank: int, directory: str, expected_bytes: bytes, accelerator: str
) -> None:
    import torch.distributed as dist

    device = (
        torch.device(accelerator, rank)
        if accelerator == "cuda"
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if device.type == "cuda" else "gloo",
        init_method=f"file://{directory}/ring",
        rank=rank,
        world_size=4,
    )
    try:
        _ring(rank, expected_bytes, device)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "accelerator", ("cpu", pytest.param("cuda", marks=pytest.mark.gpu))
)
def test_media_units_across_a_ring_match_the_whole_track_reconstruction(
    tmp_path,
    accelerator,
):
    import pickle

    import torch.multiprocessing as mp

    mp.spawn(
        _worker,
        args=(str(tmp_path), pickle.dumps(_serial()), accelerator),
        nprocs=4,
        join=True,
    )
