"""Host execution of a video request's media reading.

The media reader runs on a host rank of the head host, where the server
published each condition's fetched bytes to shared memory. One media reading
call per conditioned request decodes every condition exactly as the server
planned it (``uniserve_worker.media.reader``) and writes the request's
condition products (``uniserve_worker.execution.conditions``):

- ``condition_pixels``: each visual condition's RGB24 frames the video
  encoder encodes, ``[pixels, 3]`` uint8, frame-major;
- ``condition_samples``: each audio track's model-rate stereo PCM,
  ``[samples, 2]`` FP32;
- ``vision_pixels``: the vision encoder's packed patch rows of each
  condition the conditioner reads (``TubeletEncoder.pack_pixels``) from its
  decoded frames: an image's one frame, or a video's sampled frames.

Conditions follow each other in request order in every product. Decoding
runs as one task on the rank's host lane; ``execute`` reserves the products
and configures that task, whose completion publishes them as host products
for the vision and latent encoders, which read them on their own hosts.
Within the task every condition's frames and every audio track decode
concurrently, since nothing reads the products until all of them are filled.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING

import torch

from uniserve.model import AudioEncoder, TubeletEncoder
from uniserve_worker.bootstrap.inputs import capability
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.execution import calls, transfer
from uniserve_worker.execution.conditions import (
    CONDITION_PIXELS,
    CONDITION_SAMPLES,
    VISION_PIXELS,
    video_admission,
)
from uniserve_worker.execution.image import bound_device_write
from uniserve_worker.media.reader import (
    media_path,
    read_audio,
    read_bytes,
    read_image,
    read_video,
)
from uniserve_worker.protocol.call import CallStatus
from uniserve_worker.protocol.output import FinishFlags

if TYPE_CHECKING:
    from uniserve_worker.execution.batch import BatchState
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.execution.output import PendingOutput
    from uniserve_worker.protocol.call import Call
    from uniserve_worker.protocol.video import VideoCondition
    from uniserve_worker.storage.tensor_store import TensorStore
    from uniserve_worker.transport.interface import Transport

__all__ = ["execute", "read_conditions"]


def read_conditions(
    conditions: tuple[VideoCondition, ...],
    *,
    pixels: torch.Tensor | None,
    samples: torch.Tensor | None,
    patches: torch.Tensor | None,
    vision: TubeletEncoder,
    sample_rate: int,
    ffmpeg: str,
) -> None:
    """Decode a request's conditions into its condition products.

    ``pixels``, ``samples`` and ``patches`` are the request's
    ``condition_pixels``, ``condition_samples`` and ``vision_pixels``
    storage, each None when no condition contributes to it. ``vision``
    packs the conditioner's patch rows, ``sample_rate`` is the audio
    encoder's rate and ``ffmpeg`` the executable reference videos decode
    with.

    Raises:
        ValueError: A condition decodes to other extents than planned, or
            the conditions do not fill the products exactly.
    """
    rows = {"pixels": 0, "samples": 0, "patches": 0}

    def fill(name: str, target: torch.Tensor | None, value: torch.Tensor):
        # Each product is filled in request order from its first row.
        if target is None:
            raise ValueError(f"a condition's {name} have no product")
        start = rows[name]
        if start + value.shape[0] > target.shape[0]:
            raise ValueError(f"the request's conditions exceed their {name}")
        target[start : start + value.shape[0]].copy_(value)
        rows[name] = start + value.shape[0]

    def visual(condition: VideoCondition):
        # A condition's decoded frames and, for one the conditioner reads,
        # its packed patch rows.
        source = condition.source
        if condition.image is not None:
            frames = read_image(read_bytes(source), condition.image)
        else:
            assert condition.video is not None
            frames = read_video(
                media_path(source), condition.video, ffmpeg=ffmpeg
            )
        view = condition.vision
        if view is None:
            return frames, None
        # The conditioner reads an image's one frame and a video's sampled
        # frames; indexing copies them.
        sampled = (
            frames
            if condition.video is None
            else frames[list(view.frame_indices)]
        )
        return frames, vision.pack_pixels(torch.from_numpy(sampled), view.grid)

    def track(condition: VideoCondition):
        # An audio reference's track is its file's; a video's soundtrack is
        # its container's first audio stream.
        assert condition.audio is not None
        return read_audio(
            media_path(condition.source), condition.audio, rate=sample_rate
        )

    # Decoding releases the interpreter lock (image codecs and resampling,
    # the FFmpeg pipe, tensor operations), so the conditions' decodes overlap;
    # the products are then filled in request order.
    with ThreadPoolExecutor(max_workers=max(1, 2 * len(conditions))) as pool:
        decoded = [
            (
                pool.submit(visual, condition)
                if condition.image is not None or condition.video is not None
                else None,
                pool.submit(track, condition)
                if condition.audio is not None
                else None,
            )
            for condition in conditions
        ]
        for condition, (frames_task, track_task) in zip(
            conditions, decoded, strict=True
        ):
            if frames_task is not None:
                frames, packed = frames_task.result()
                # The video encoder encodes a video's leading frames.
                encoded = condition.pixels
                assert encoded is not None
                fill(
                    "pixels",
                    pixels,
                    torch.from_numpy(frames[: encoded.num_frames]).reshape(
                        -1, 3
                    ),
                )
                if packed is not None:
                    fill("patches", patches, packed)
            if track_task is not None:
                fill("samples", samples, track_task.result())

    for name, target in (
        ("pixels", pixels),
        ("samples", samples),
        ("patches", patches),
    ):
        if target is not None and rows[name] != target.shape[0]:
            raise ValueError(f"the request's conditions leave {name} unfilled")


def execute(
    call: Call,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    publication_transports: Mapping[str, Transport],
    model_runner: ModelExecutor,
) -> PendingOutput:
    """Schedule one request's media reading on the rank's host lane.

    Reserves the call's products, configures its one reserved host task to
    decode the request's conditions into them, and stages ``host.finish``,
    which publishes the products once the task completes. The returned
    output carries no products at commit.

    Raises:
        WorkerError: ``invalid_descriptor`` when the request has no
            conditions, or the call declares a product other than the
            media reader's.
        RuntimeError: The call has no reserved lane slot, or the model
            lacks the vision or audio encoder whose inputs it reads.
    """
    request = state.pending_output(call.request_key.request_id)
    video = video_admission(request)
    reservations = request.host.tasks
    if len(reservations) != 1:
        raise RuntimeError("media reading has no reserved lane slot")
    vision = capability(model_runner.model, TubeletEncoder)
    audio = capability(model_runner.model, AudioEncoder)
    if vision is None or audio is None:
        raise RuntimeError("media reading requires the condition encoders")

    # Each product's storage is reserved now and filled by the host task;
    # the products are published only once it completes.
    declared = model_runner.outputs.get(call.component, ())
    products = {}
    for output in call.outputs:
        if output.output_index >= len(declared):
            raise invalid_descriptor("media reading declares no such product")
        name = declared[output.output_index].name
        if name not in (CONDITION_PIXELS, CONDITION_SAMPLES, VISION_PIXELS):
            raise invalid_descriptor(f"media reading does not produce {name}")
        write = bound_device_write(output, state=state)
        tensor_store.defer_write(write)
        products[name] = (output, write)

    def target(name: str) -> torch.Tensor | None:
        return products[name][1].tensor if name in products else None

    task = reservations[0].configure(
        partial(
            read_conditions,
            video.conditions,
            pixels=target(CONDITION_PIXELS),
            samples=target(CONDITION_SAMPLES),
            patches=target(VISION_PIXELS),
            vision=vision,
            sample_rate=audio.sample_rate,
            ffmpeg=model_runner.worker_config.ffmpeg,
        ),
        profile_name=(
            "uniserve.host.read "
            f"request={call.request_key.request_id} "
            f"conditions={len(video.conditions)}"
        ),
    )

    def publish(results: tuple[object, ...]) -> None:
        from uniserve_worker.execution.commit import (
            _validate_completion_products,
        )

        published = tuple(
            transfer.publish_deferred_product(
                output,
                write,
                write.tensor,
                tensor_store=tensor_store,
                publication_transports=publication_transports,
                consumers=call.consumer_slots,
            )
            for output, write in products.values()
        )
        _validate_completion_products(call, published)
        # The batch recorded its products when the call committed, before
        # the media was read; these join the batch's result now.
        request.products = published
        state.products = (*state.products, *published)

    request.status = CallStatus.OK
    request.progress = calls.execution_runtime(request, None)
    request.finish_flags = FinishFlags()
    request.product_generations = calls.output_generations(call)
    request.host.tasks = (task,)
    request.host.finish = publish
    request.products = ()
    return request
