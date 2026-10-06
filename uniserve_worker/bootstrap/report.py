"""Native worker allocation plans and numerical video model descriptions."""

from uniserve.model import VideoDenoiser
from uniserve_worker._uniserve_ipc import WorkerLayout as WorkerLayout
from uniserve_worker._uniserve_ipc import (
    build_worker_layout as build_worker_layout,
)
from uniserve_worker.bootstrap.inputs import executed_video_tasks
from uniserve_worker.errors import unsupported_setup
from uniserve_worker.protocol.worker_info import VideoDenoiserInfo


def video_denoiser_info(denoiser: VideoDenoiser) -> VideoDenoiserInfo:
    """Describe what the deployment's video denoiser serves.

    The tasks are the denoiser's tasks whose request calls the worker
    executes, in the denoiser's order. The schedule's sigma points include
    the clean endpoint the network never evaluates.

    Raises:
        WorkerError: ``UnsupportedSetup`` when the worker executes none of
            the denoiser's tasks.
    """
    tasks = executed_video_tasks(denoiser)
    if not tasks:
        raise unsupported_setup(
            f"the placed video denoiser serves {', '.join(denoiser.tasks)}, "
            "none of which this worker executes"
        )
    canvases = denoiser.fixed_canvases
    shifts = denoiser.schedule_shifts
    return VideoDenoiserInfo(
        tasks=tasks,
        schedule_points=denoiser.num_steps + 1,
        video_shift=float(shifts["video"]),
        audio_shift=float(shifts["audio"]),
        canvases=()
        if canvases is None
        else tuple((canvas.width, canvas.height) for canvas in canvases),
        max_sequence_rows=denoiser.max_sequence_rows,
        condition_tiles=denoiser.condition_tiles,
    )
