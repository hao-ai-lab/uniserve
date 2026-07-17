"""Model-free worker that retains generated frame payloads per request."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..contracts.outputs import FrameOutput
from ..foundation.errors import invalid_descriptor
from ..foundation.sizing import DEFAULT_BLOCK_SIZE
from .protocol import (
    BaseWorker,
    ResultPolicy,
    model_free_capabilities,
)


@dataclass
class _RequestFrameBuffer:
    frames: list[bytes] = field(default_factory=list)

    def append(self, frame: bytes) -> int:
        self.frames.append(frame)
        return len(self.frames)


class FrameAccumulatorWorker(BaseWorker):
    """Collects frame bytes until the request is released."""

    def __init__(
        self,
        *,
        allowed_ops: frozenset[str] = frozenset({"encode_frame"}),
        pipeline_depth: int = 1,
        result_policy: ResultPolicy = ResultPolicy.DEFER_WHEN_AVAILABLE,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        super().__init__(block_size=block_size)
        self._frame_buffers: dict[int, _RequestFrameBuffer] = {}
        self._compile_contract(
            model_free_capabilities(
                block_size=self.block_size,
                supported_ops=("encode_frame",),
            ),
            allowed_ops=allowed_ops,
            pipeline_depth=pipeline_depth,
            result_policy=result_policy,
        )

    def execute(
        self,
        batch: Mapping[str, Any],
        *,
        defer_text_cpu_results: bool = False,
    ) -> dict[str, Any]:
        del defer_text_cpu_results
        operations = batch.get("ops") or []
        return {
            "step_id": batch.get("step_id"),
            "per_seq": [self._append_frame(operation) for operation in operations],
        }

    def drop_request(self, request_id: int) -> None:
        self.release_request_frames(int(request_id))

    def release_request_frames(self, request_id: int) -> dict[str, Any]:
        frame_buffer = self._frame_buffers.pop(int(request_id), None)
        return {
            "req_id": int(request_id),
            "frames": (0 if frame_buffer is None else len(frame_buffer.frames)),
        }

    def _append_frame(
        self,
        operation: Mapping[str, Any],
    ) -> dict[str, Any]:
        request_id = int(operation["req_id"])
        frame_buffer = self._frame_buffers.setdefault(
            request_id,
            _RequestFrameBuffer(),
        )
        frame_count = frame_buffer.append(self._decode_frame(operation))
        return FrameOutput(
            req_id=request_id,
            num_tokens=frame_count,
        ).to_seq_result()

    @staticmethod
    def _decode_frame(operation: Mapping[str, Any]) -> bytes:
        encoded_frame = operation.get("image_b64")
        if not isinstance(encoded_frame, str) or not encoded_frame:
            raise invalid_descriptor("encode_frame operation requires image_b64")
        try:
            return base64.b64decode(encoded_frame, validate=True)
        except (ValueError, binascii.Error) as error:
            raise invalid_descriptor("encode_frame operation contains invalid image_b64") from error
