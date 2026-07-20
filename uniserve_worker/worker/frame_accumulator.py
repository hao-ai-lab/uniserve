"""Model-free worker that retains generated frame payloads per request."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from typing import Any

from ..contracts.batches import Batch
from ..contracts.outputs import FrameOutput
from ..execution.operation_executor import OperationExecutor
from ..foundation.errors import invalid_descriptor
from ..foundation.sizing import DEFAULT_BLOCK_SIZE
from ..runtime.product_store import ProductStore
from ..runtime.request_session import SessionStore
from .protocol import (
    BaseWorker,
    ResultPolicy,
    model_free_capabilities,
)


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
        self.sessions = SessionStore()
        self.products = ProductStore()
        self.executor = OperationExecutor(
            self.sessions,
            self._execute_once,
            admit=self._admit,
            stores=(self.products,),
        )
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
        return self.executor.execute(
            batch,
            defer_text_cpu_results=defer_text_cpu_results,
        )

    def _execute_once(
        self,
        batch: Batch,
        *,
        defer_text_cpu_results: bool = False,
    ) -> dict[str, Any]:
        del defer_text_cpu_results
        return {
            "step_id": batch.step_id,
            "per_seq": [self._append_frame(operation) for operation in batch.ops],
        }

    def drop_request(self, request_id: int) -> None:
        self.release_request_frames(int(request_id))
        self.sessions.drop(int(request_id))

    def release_request_frames(self, request_id: int) -> dict[str, Any]:
        return {
            "req_id": int(request_id),
            "frames": self.products.release_frames(int(request_id)),
        }

    def _admit(self, batch: Batch) -> None:
        operations = {operation.session_id: operation for operation in batch.ops}
        for new_request in batch.new_reqs:
            request_id = int(new_request["req_id"])
            operation = operations[request_id]
            self.sessions.admit(
                request_id,
                new_request,
                epoch=operation.epoch,
                base_version=operation.base_version,
            )

    def _append_frame(
        self,
        operation: Mapping[str, Any],
    ) -> dict[str, Any]:
        request_id = int(operation["req_id"])
        frame_count = self.products.append_frame(request_id, self._decode_frame(operation))
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
