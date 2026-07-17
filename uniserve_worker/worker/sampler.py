"""Model-free worker that samples tokens from published logits."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

import torch

from ..contracts.outputs import SampleOutput
from ..foundation.errors import invalid_descriptor
from ..foundation.sizing import DEFAULT_BLOCK_SIZE
from ..nn.sampler import sample_one_from_logits
from ..runtime.request_state import request_seed
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
        self._sampling_params: dict[int, dict[str, Any]] = {}
        self._request_seeds: dict[int, int] = {}
        self._sampling_generators: dict[tuple[int, str], torch.Generator] = {}
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
        del defer_text_cpu_results
        for new_request in batch.get("new_reqs") or ():
            self._register_request(new_request)
        operations = batch.get("ops") or []
        return {
            "step_id": batch.get("step_id"),
            "per_seq": [self._sample_operation(operation) for operation in operations],
        }

    def drop_request(self, request_id: int) -> None:
        target = int(request_id)
        self._sampling_params.pop(target, None)
        self._request_seeds.pop(target, None)
        for key in [key for key in self._sampling_generators if key[0] == target]:
            self._sampling_generators.pop(key, None)

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

    def _register_request(self, new_request: Mapping[str, Any]) -> None:
        request_id = int(new_request["req_id"])
        sampling_params = new_request.get("sampling")
        self._sampling_params[request_id] = (
            dict(sampling_params) if isinstance(sampling_params, Mapping) else {}
        )
        self._request_seeds[request_id] = request_seed(
            new_request,
            request_id,
        )

    def _sample_operation(
        self,
        operation: Mapping[str, Any],
    ) -> dict[str, Any]:
        request_id = int(operation["req_id"])
        logits = self._fetch_logits(operation)
        sampling_params = self._sampling_params.get(request_id, {})
        sample = sample_one_from_logits(
            logits,
            sampling_params,
            recent=tuple(operation.get("recent_tokens") or ()),
            allowed=operation.get("allowed_tokens"),
            suppress=operation.get("suppress_tokens"),
            n_logprobs=int(sampling_params.get("n_logprobs", 0) or 0),
            generator=self._sampling_generator(request_id, logits.device),
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

    def _sampling_generator(
        self,
        request_id: int,
        device: torch.device | str,
    ) -> torch.Generator:
        target_device = torch.device(device)
        key = (int(request_id), str(target_device))
        generator = self._sampling_generators.get(key)
        if generator is None:
            generator = torch.Generator(device=target_device).manual_seed(
                int(
                    self._request_seeds.get(
                        int(request_id),
                        int(request_id),
                    )
                )
            )
            self._sampling_generators[key] = generator
        return generator
