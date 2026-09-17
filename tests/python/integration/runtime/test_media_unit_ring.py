"""Media units reconstructed across ranks reproduce the whole-track pixels.

Post-processing blends each media unit's leading frames with the overlap its
predecessor decoded. When consecutive units are held by different ranks that
overlap crosses the ring between them, so the reconstruction has to be
indistinguishable from one rank reconstructing the whole track in order.
"""

from __future__ import annotations

import pytest
import torch

from uniserve.media import image
from uniserve.model import VideoPostprocessor
from uniserve.tensors import OutputLayout, TensorOutput

pytestmark = pytest.mark.integration

UNITS = 6
EXTENT = 2
BODY = 3
WINDOW = 6
HEIGHT, WIDTH = 2, 3


class _Blend(VideoPostprocessor):
    """Three body frames and a two-frame successor overlap per window."""

    def reconstruction_slices(self, frames, num_frames):
        return slice(0, BODY), slice(WINDOW - EXTENT, WINDOW)


def _postprocessor() -> _Blend:
    return _Blend(
        torch.tensor([0.0, 0.5], dtype=torch.float16),
        frame_size=image.Config(HEIGHT, WIDTH),
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
            num_frames=(total,),
            state=state,
            constants=constants,
            workspace=workspace,
        )[0].tensor.clone()
        for unit in range(UNITS)
    ]


def _ring(rank: int, expected_bytes: bytes) -> None:
    import pickle

    import torch.distributed as dist

    from uniserve.distributed import Communicator

    expected = pickle.loads(expected_bytes)
    model = _postprocessor()
    constants, workspace, state = _resources()
    frames = _windows()
    total = frames[-1].stop
    ranks = tuple(range(dist.get_world_size()))
    model.units = Communicator(
        ranks=ranks,
        rank=rank,
        name="units",
        device=torch.device("cpu"),
        _group=dist.group.WORLD,
    )

    cursor = 0
    while cursor < UNITS:
        count = min(len(ranks), UNITS - cursor)
        if rank < count:
            unit = cursor + rank
            produced = model(
                (_output(unit, total),),
                frames=(frames[unit],),
                num_frames=(total,),
                state=state,
                constants=constants,
                workspace=workspace,
                unit_count=count,
            )[0].tensor
            assert torch.equal(produced, expected[unit]), (
                f"media unit {unit} on rank {rank} differs from the "
                "whole-track reconstruction"
            )
        cursor += count


def _worker(rank: int, directory: str, expected_bytes: bytes) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/ring",
        rank=rank,
        world_size=4,
    )
    try:
        _ring(rank, expected_bytes)
    finally:
        dist.destroy_process_group()


def test_media_units_across_a_ring_match_the_whole_track_reconstruction(
    tmp_path,
):
    import pickle

    import torch.multiprocessing as mp

    mp.spawn(
        _worker,
        args=(str(tmp_path), pickle.dumps(_serial())),
        nprocs=4,
        join=True,
    )
