"""Model-free worker that samples tokens from published logits."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

import torch

from ..contracts.batches import Batch
from ..contracts.outputs import SampleOutput
from ..execution.operation_executor import OperationExecutor
from ..execution.sampling import sampled_token_position, sampling_draw_generator
from ..foundation.errors import invalid_descriptor
from ..foundation.sizing import DEFAULT_BLOCK_SIZE
from ..nn.sampler import sample_one_from_logits
from ..runtime.request_session import SessionStore
from ..runtime.tensor_store import TensorStore
from .protocol import (
    BaseWorker,
    ResultPolicy,
    model_free_capabilities,
)


class SamplerWorker(BaseWorker):
    """Fetches logits and applies per-request sampling parameters."""

    def __init__(
        self,
        *,
        tensor_store: TensorStore | None = None,
        allowed_ops: frozenset[str] = frozenset({"sample"}),
        pipeline_depth: int = 1,
        result_policy: ResultPolicy = ResultPolicy.DEFER_WHEN_AVAILABLE,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        super().__init__(block_size=block_size)
        self.tensor_store = tensor_store if tensor_store is not None else TensorStore()
        self.sessions = SessionStore()
        self.executor = OperationExecutor(
            self.sessions,
            self._execute_once,
            admit=self._admit,
        )
        self._compile_contract(
            model_free_capabilities(
                block_size=self.block_size,
                supported_ops=("sample",),
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
            "per_seq": [self._sample_operation(operation) for operation in batch.ops],
        }

    def drop_request(self, request_id: int) -> None:
        self.sessions.drop(int(request_id))

    def free_logits(self, handles: list[int]) -> None:
        self.tensor_store.release_many([int(handle) for handle in handles])

    def _fetch_logits(self, operation: Mapping[str, Any]) -> torch.Tensor:
        encoded_locator = operation.get("locator")
        if encoded_locator:
            return self.tensor_store.fetch_locator(base64.b64decode(encoded_locator))
        handle = operation.get("logits_handle")
        if handle is not None:
            return self.tensor_store.fetch(int(handle))
        raise invalid_descriptor("sample operation requires a logits locator or handle")

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

    def _sample_operation(
        self,
        operation: Mapping[str, Any],
    ) -> dict[str, Any]:
        request_id = int(operation["req_id"])
        logits = self._fetch_logits(operation)
        state = self.sessions.get(request_id)
        sampling_params = state.sampling
        sample = sample_one_from_logits(
            logits,
            sampling_params,
            recent=tuple(operation.get("recent_tokens") or ()),
            allowed=operation.get("allowed_tokens"),
            suppress=operation.get("suppress_tokens"),
            n_logprobs=int(sampling_params.get("n_logprobs", 0) or 0),
            generator=sampling_draw_generator(
                state,
                logits.device,
                position=sampled_token_position(operation),
            ),
        )
        output = SampleOutput(
            req_id=request_id,
            sampled_token_id=int(sample.token_id),
            sampled_logprob=sample.logprob,
            top_logprobs=(
                [(int(item[0]), float(item[1]), int(item[2])) for item in sample.top_logprobs]
                if sample.top_logprobs is not None
                else None
            ),
        )
        return output.to_seq_result()
