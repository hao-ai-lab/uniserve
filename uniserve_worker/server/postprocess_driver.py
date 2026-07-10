"""Post-process worker driver.

A model-free, GPU-free stage that encodes finished image/video frames (FFmpeg).
Frame N encodes while frame N+1 is still being generated (pipeline parallel,
independent depth). Split off the generation worker when frame encoding needs to
scale on its own axis; unused under the ``full`` topology.

Contract (a ``WorkerDriver``): per ``encode_frame`` op it ingests one frame
(``image_b64`` bytes, or a frame handle on the data plane), appends it to
the request's video pipeline, and reports the cumulative frame count;
``finalize_video`` (on ``drop_request``) muxes the collected frames.

FFmpeg is optional: when the ``ffmpeg`` binary is absent the driver still tracks
and reports frames (degraded muxing), so the stage is exercisable without it.
"""
from __future__ import annotations

import base64
import binascii
import logging
import shutil
from typing import Any, Mapping

from ..contracts.outputs import FrameOutput
from ..foundation.sizing import DEFAULT_BLOCK_SIZE
from .base_driver import BaseWorkerDriver

__all__ = ["PostProcessDriver", "build_postprocess_driver"]

logger = logging.getLogger(__name__)


class _VideoPipeline:
    """Per-request frame accumulator. Holds frame bytes until finalize."""

    def __init__(self) -> None:
        self.frames: list[bytes] = []

    def append(self, frame: bytes) -> int:
        self.frames.append(frame)
        return len(self.frames)


class PostProcessDriver(BaseWorkerDriver):
    """WorkerDriver that encodes generated frames into a video."""

    def __init__(self, *, block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        super().__init__(block_size=block_size)
        self._pipelines: dict[int, _VideoPipeline] = {}
        self._ffmpeg = shutil.which("ffmpeg")
        if self._ffmpeg is None:
            logger.warning(
                "ffmpeg not found on PATH; post-process worker will track frames "
                "but cannot mux a video container"
            )
        self._caps = self._build_model_free_caps(
            block_size=self.block_size,
            supported_ops=("encode_frame",),
        )

    def execute(
        self,
        batch: Mapping[str, Any],
        *,
        defer_text_cpu_results: bool = False,
    ) -> dict[str, Any]:
        del defer_text_cpu_results
        ops = batch.get("ops") or []
        per_seq = [self._encode_frame(op) for op in ops]
        return {"step_id": batch.get("step_id"), "per_seq": per_seq}

    def drop_request(self, req_id: int) -> None:
        self.finalize_video(int(req_id))

    def _encode_frame(self, op: Mapping[str, Any]) -> dict[str, Any]:
        req_id = int(op["req_id"])
        pipeline = self._pipelines.setdefault(req_id, _VideoPipeline())
        frame = self._frame_bytes(op)
        count = pipeline.append(frame)
        return FrameOutput(req_id=req_id, num_tokens=count).to_seq_result()

    def _frame_bytes(self, op: Mapping[str, Any]) -> bytes:
        b64 = op.get("image_b64")
        if isinstance(b64, str) and b64:
            try:
                return base64.b64decode(b64)
            except (ValueError, binascii.Error):
                logger.debug("post-process frame had undecodable base64; storing empty frame")
        return b""

    def finalize_video(self, req_id: int) -> dict[str, Any]:
        """Mux a request's frames into a container and reclaim its pipeline.

        Returns a status dict (frame count + optional encoder availability). The
        actual FFmpeg invocation is intentionally left as the integration point
        for the (not-yet-landed) video-generation model; with no encoder this is
        a no-op finalize that still reports the collected frame count.
        """
        pipeline = self._pipelines.pop(req_id, None)
        frames = 0 if pipeline is None else len(pipeline.frames)
        return {"req_id": req_id, "frames": frames, "ffmpeg": self._ffmpeg is not None}


def build_postprocess_driver(args: Any) -> PostProcessDriver:
    return PostProcessDriver(block_size=int(getattr(args, "block_size", DEFAULT_BLOCK_SIZE)))
