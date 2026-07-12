"""Sampler worker driver.

A model-free stage: loads only the sampler (``nn/sampler``) — no LM, no GPU
required — and turns a ``Logits`` handle into a sampled token. Split off the
decode worker when sampling becomes the bottleneck (structured generation, beam/MCTS,
very large batches); unused under the ``full`` topology when decode samples inline.

Contract (a ``WorkerDriver``: caps / execute / drop_request):
  * control input: per-request ``SamplingParams`` (once, via ``NewRequestData``),
    plus per-op ``logits_handle`` and the ``allowed``/``suppress``/``recent`` lists.
  * data input: the logits tensor, fetched by ``logits_handle`` from the
    worker-side :class:`~uniserve_worker.runtime.tensor_store.TensorStore`
    (in-process by default; cross-process via a Tier-2 ``TransferAgent`` pull).
  * output: ``sampled_token_id`` (+ optional logprobs).
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Mapping

import torch

from ..contracts.outputs import SampleOutput
from ..foundation.errors import invalid_descriptor
from ..foundation.sizing import DEFAULT_BLOCK_SIZE
from ..nn.sampler import sample_one_from_logits
from ..runtime.request_state import request_seed
from ..runtime.tensor_store import TensorStore
from ..runtime.transfer import make_transport
from .base_driver import BaseWorkerDriver

__all__ = ["SamplerDriver", "build_sampler_driver"]

logger = logging.getLogger(__name__)


class SamplerDriver(BaseWorkerDriver):
    """WorkerDriver that samples tokens from fetched logits handles."""

    def __init__(
        self,
        *,
        block_size: int = DEFAULT_BLOCK_SIZE,
        tensor_store: TensorStore | None = None,
    ) -> None:
        super().__init__(block_size=block_size)
        # The store is where the decode worker's logits land. May be injected for
        # same-process use (e.g. tests); otherwise defaults to an in-process store.
        # Explicit None-check: an empty store is falsy via __len__, so ``or`` would
        # wrongly discard a passed-in store.
        self.tensor_store = tensor_store if tensor_store is not None else TensorStore()
        self._sampling: dict[int, dict[str, Any]] = {}
        self._seeds: dict[int, int] = {}
        self._sampling_generators: dict[tuple[int, str], torch.Generator] = {}
        # Model-free worker: KV/layer fields are inert placeholders (caps
        # validation requires positive values and a kv_block class).
        self._caps = self._build_model_free_caps(
            block_size=self.block_size,
            supported_ops=("sample",),
        )

    def execute(
        self,
        batch: Mapping[str, Any],
        *,
        defer_text_cpu_results: bool = False,
    ) -> dict[str, Any]:
        del defer_text_cpu_results
        for nr in batch.get("new_reqs") or ():
            self._register(nr)
        ops = batch.get("ops") or []
        per_seq = [self._sample_one(op) for op in ops]
        return {"step_id": batch.get("step_id"), "per_seq": per_seq}

    def drop_request(self, req_id: int) -> None:
        target = int(req_id)
        self._sampling.pop(target, None)
        self._seeds.pop(target, None)
        for key in [key for key in self._sampling_generators if key[0] == target]:
            self._sampling_generators.pop(key, None)

    def free_logits(self, handles: "list[int]") -> None:
        # Release control op: reclaim published logits buffers.
        self.tensor_store.release_many([int(h) for h in handles])

    def _fetch_logits(self, op: Mapping[str, Any]):
        # Cross-process: fetch by the producer's data-plane locator. The local
        # handle is a same-process fallback (decode + sampler co-located).
        locator_b64 = op.get("locator")
        if locator_b64:
            return self.tensor_store.fetch_locator(base64.b64decode(locator_b64))
        handle = op.get("logits_handle")
        if handle is not None:
            return self.tensor_store.fetch(int(handle))
        raise invalid_descriptor("sample op missing both logits locator and handle")

    def _register(self, nr: Mapping[str, Any]) -> None:
        req_id = int(nr["req_id"])
        sampling = nr.get("sampling")
        self._sampling[req_id] = dict(sampling) if isinstance(sampling, Mapping) else {}
        self._seeds[req_id] = request_seed(nr, req_id)

    def _sample_one(self, op: Mapping[str, Any]) -> dict[str, Any]:
        req_id = int(op["req_id"])
        logits = self._fetch_logits(op)
        sp = self._sampling.get(req_id, {})
        sample = sample_one_from_logits(
            logits,
            sp,
            recent=tuple(op.get("recent_tokens") or ()),
            allowed=op.get("allowed_tokens"),
            suppress=op.get("suppress_tokens"),
            n_logprobs=int(sp.get("n_logprobs", 0) or 0),
            generator=self._sampling_generator(req_id, logits.device),
        )
        out = SampleOutput(
            req_id=req_id,
            sampled_token_id=int(sample.token_id),
            sampled_logprob=sample.logprob,
            top_logprobs=(
                [(int(item[0]), float(item[1]), int(item[2])) for item in sample.top_logprobs]
                if sample.top_logprobs is not None
                else None
            ),
        )
        return out.to_seq_result()

    def _sampling_generator(
        self,
        req_id: int,
        device: torch.device | str,
    ) -> torch.Generator:
        target = torch.device(device)
        key = (int(req_id), str(target))
        generator = self._sampling_generators.get(key)
        if generator is None:
            generator = torch.Generator(device=target).manual_seed(
                int(self._seeds.get(int(req_id), int(req_id)))
            )
            self._sampling_generators[key] = generator
        return generator


def build_sampler_driver(args: Any) -> SamplerDriver:
    # The sampler fetches logits over the data-plane transport the producing
    # decode worker publishes to (default in-process).
    transport = make_transport(str(getattr(args, "transfer_backend", "local") or "local"))
    return SamplerDriver(
        block_size=int(getattr(args, "block_size", DEFAULT_BLOCK_SIZE)),
        tensor_store=TensorStore(transport=transport),
    )
