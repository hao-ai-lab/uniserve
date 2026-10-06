"""Step coordination and token exchange for distributed expert layers.

Rust selects ready source groups, rotates call kinds and records layer
participation. Fixed CPU buffers carry rank readiness over Gloo. Numerical
backends dispatch token rows to their experts and combine the results;
each microbatch owns its communication stream and buffers.

Every rank participates in the same layer sequence, including ranks without
local tokens. The execution context joins any expert layers a forward skips
at its tail, preserving collective order and buffer reuse.
"""

from __future__ import annotations

from functools import partial
from typing import Literal

import torch
import torch.distributed as dist

from uniserve.distributed import Communicator
from uniserve.quantization import QuantizedTensor, ScaleLayout
from uniserve_worker._uniserve_ipc import ExpertExchange as _ExpertExchange
from uniserve_worker._uniserve_ipc import JoinGraphs as JoinGraphs

__all__ = ["ExpertExchange", "JoinGraphs"]


def _prepare_experts(group, ready) -> None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm the expert exchange before capture")

    # Source ranks may compile attention before reaching their first expert.
    # Wait here so that compilation cannot consume a device collective timeout.
    dist.all_reduce(ready, group=group)


class ExpertExchange:
    """One invocation domain's transport resources for an expert group.

    Construction is collective over ``group``: every rank constructs its
    exchange at the same point of its startup, because the workspace maps
    peer memory. ``max_tokens`` bounds every step's per-rank capacity. The
    workspace stays mapped until process exit; FlashInfer caches it
    process-wide. ``transport`` selects the NVLink all-to-all (``alltoall``)
    or the fused MegaMoE kernel (``megamoe``, NVFP4 experts), which also
    needs the experts' gated width ``intermediate_size`` and gate
    ``activation``. ``deepep`` uses elastic dispatch/combine and supports
    symmetric or disaggregated placement; ``attention_ranks`` counts the
    leading source-only members, or zero for symmetric expert parallelism.
    DeepEP transports BF16 or calibrated NVFP4 states without requantization.
    Each microbatch owns a separate exchange,
    including its communication stream and buffer.

    ``source_group`` lists the global ranks executing each local forward
    together (for example a tensor-parallel worker). Its members must all
    submit work before any may start that forward. Omission makes each
    source independent; expert-only ranks have no source group. All ranks
    must declare consistent, disjoint memberships at construction.
    """

    def __init__(
        self,
        group: Communicator,
        *,
        max_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        device: torch.device,
        transport: Literal["alltoall", "megamoe", "deepep"] = "alltoall",
        intermediate_size: int | None = None,
        activation: str | None = None,
        attention_ranks: int = 0,
        source_group: tuple[int, ...] | None = None,
    ) -> None:
        self.group = group
        self.max_tokens = max_tokens
        self.device = device
        self.transport = transport
        self.attention_ranks = attention_ranks
        self.fused = None
        self._elastic = None
        self.source_only = 0 <= group.rank < attention_ranks
        # A tensor/pipeline group submits one forward together. IPC delivery
        # can reach its members on different host turns; a partial group must
        # join with no tokens until every member has that forward available.
        # Membership uses global ranks, just like Communicator.ranks.
        source_count = attention_ranks or group.size
        if source_group is None:
            source_group = (
                (group.ranks[group.rank],) if group.rank < source_count else ()
            )
        memberships: list[tuple[int, ...]] = [()] * group.size
        host = group._require()
        dist.all_gather_object(memberships, tuple(source_group), group=host)
        if attention_ranks and transport not in {"deepep", "megamoe"}:
            raise ValueError("disaggregated experts require a split transport")

        local = torch.empty((1, 3), dtype=torch.int64, device="cpu")
        records = torch.empty((group.size, 3), dtype=torch.int64, device="cpu")
        self._control = _ExpertExchange(
            group.ranks,
            group.rank,
            memberships,
            attention_ranks=attention_ranks,
            max_tokens=max_tokens,
            fused=transport == "megamoe",
            local=local.numpy(),
            records=records.numpy(),
            gather=partial(
                dist.all_gather, list(records.split(1)), local, group=host
            ),
            prepare=(
                partial(
                    _prepare_experts,
                    host,
                    torch.zeros((), dtype=torch.int32, device="cpu"),
                )
                if transport == "deepep" or attention_ranks
                else None
            ),
            broadcast=(
                partial(
                    dist.broadcast,
                    local.view(-1)[0],
                    src=group.ranks[0],
                    group=host,
                )
                if attention_ranks
                else None
            ),
        )

        if transport == "megamoe":
            if intermediate_size is None or activation is None:
                raise ValueError(
                    "the MegaMoE transport needs the experts' width and gate"
                )
            if attention_ranks:
                from .backends.moe.deepgemm import Buffer
            else:
                from .backends.moe.megamoe import MegaMoEBuffer as Buffer
            self.fused = Buffer(
                group,
                max_tokens=max_tokens,
                num_experts=num_experts,
                top_k=top_k,
                hidden=hidden_size,
                intermediate=intermediate_size,
                activation=activation,
                device=device,
                **(
                    {"attention_ranks": attention_ranks}
                    if attention_ranks
                    else {}
                ),
            )
        elif transport == "deepep":
            from .backends.deepep import DeepEP

            self._elastic = DeepEP(
                group,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                max_tokens=max_tokens,
                attention_ranks=attention_ranks,
            )
        elif transport != "alltoall":
            raise ValueError(f"unknown expert transport {transport!r}")
        else:
            self._alltoall = self._build_alltoall(
                group, max_tokens, top_k, num_experts, hidden_size, device
            )

    @property
    def capacity(self) -> int:
        return self._control.capacity

    @property
    def capacities(self) -> tuple[int, ...]:
        return self._control.capacities

    @property
    def kind(self) -> int:
        return self._control.kind

    @property
    def active(self) -> bool:
        return self._control.active

    @property
    def released(self) -> bool:
        return self._control.released

    @property
    def invoked(self) -> frozenset[int]:
        return self._control.invoked

    def pending_layers(self, modules: list[int]) -> list[int]:
        """Select the forward's unvisited tail in collective order."""
        return self._control.pending_layers(modules)

    def reset_layers(self) -> None:
        self._control.reset_layers()

    def record_layers(self, modules: frozenset[int]) -> None:
        """Record the layers reached by a captured forward."""
        self._control.record_layers(modules)

    def bind_microbatches(self, exchanges):
        """Bind a shared native plan when the transport needs one."""
        if len({id(exchange) for exchange in exchanges}) != len(exchanges):
            raise ValueError("each microbatch requires its own expert exchange")
        if self.transport == "megamoe" and self.attention_ranks:
            if any(
                exchange.transport != self.transport
                or exchange.attention_ranks != self.attention_ranks
                for exchange in exchanges
            ):
                raise ValueError(
                    "microbatches must use the same expert transport"
                )
            self.fused.bind_microbatches(
                [exchange.fused for exchange in exchanges]
            )

    @staticmethod
    def _build_alltoall(group, max_tokens, top_k, num_experts, hidden, device):
        """Map the group's MNNVL all-to-all workspace, collectively."""
        from flashinfer.comm.comm_backend import TorchDistBackend
        from flashinfer.comm.mapping import Mapping
        from flashinfer.comm.mnnvl import MnnvlConfig
        from flashinfer.comm.trtllm_moe_alltoall import MoeAlltoAll

        # FlashInfer's Mapping describes expert parallelism inside a
        # TP-sized container; attention stays data-parallel on every rank.
        mapping = Mapping(
            world_size=group.size,
            rank=group.rank,
            gpus_per_node=torch.cuda.device_count(),
            tp_size=group.size,
            moe_tp_size=1,
            moe_ep_size=group.size,
            enable_attention_dp=True,
        )
        with torch.cuda.device(device):
            return MoeAlltoAll(
                mapping,
                max_tokens,
                top_k,
                num_experts,
                hidden_size=hidden,
                mnnvl_config=MnnvlConfig(
                    comm_backend=TorchDistBackend(group._require())
                ),
            )

    def warmup(self, capacity: int = 0) -> int:
        """Send one source warmup call, or receive it on expert ranks.

        All source ranks call this before each eager graph warmup. Expert
        ranks receive the capacity and execute the same layer sequence with
        no source tokens. Zero ends warmup. This host operation stays outside
        CUDA capture and serving steps.
        """
        return self._control.warmup(capacity)

    def agree(
        self, tokens: int, *, kind: int = 0, leaving: bool = False
    ) -> int:
        """Agree with the group on the next step's per-rank capacity.

        Every rank contributes the tokens its local graph or eager forward
        sends, or zero tokens when it has no forward, over
        the group's Gloo backend; every rank then computes the same result.
        Only source groups whose members all have the same call kind ready
        may submit tokens. ``active`` reports whether this rank's pending
        forward was selected; otherwise it must retain its input and join
        with zero tokens. Ready call kinds take turns in cyclic order;
        ``self.kind`` identifies the selected kind. A rank
        with another kind joins with no tokens, then agrees again
        with its still-pending input. Returns zero when no rank has tokens,
        else the smallest capacity holding the selected senders. A rank shutting
        down agrees with ``leaving`` until ``released`` reports that every
        rank is leaving without tokens, so no rank stops agreeing while
        another still waits on it.

        Raises:
            RuntimeError: A rank sends more than the configured maximum.
        """
        return self._control.agree(tokens, kind=kind, leaving=leaving)

    def begin(self, capacity: int) -> None:
        """Open a step carrying at most ``capacity`` tokens from each rank.

        Callers prepare their numerical contexts before the first step.
        The first expert call also waits for peers to reach that numerical
        boundary, so preceding attention compilation cannot consume a device
        protocol timeout. Subsequent steps require no preparation collective.
        """
        self._control.begin(capacity)

    def end(self) -> None:
        """Close the open step."""
        self._control.end()

    def enter(self, module: int) -> None:
        """Record that the open step reaches expert layer ``module``.

        Raises:
            RuntimeError: No step is open.
        """
        self._control.enter(module)

    def dispatch(
        self,
        module: int,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        invalid_expert: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Send each token to the ranks holding its experts.

        Returns this rank's received ``hidden [P * C, H]``, ``topk_ids
        [P * C, K]`` and ``topk_weights [P * C, K]``, where ``C`` is the step
        capacity. Row order is transport-owned and preserved through the
        matching ``combine``. Ids of experts this rank does not hold, and of
        unused rows,
        read ``invalid_expert``, which the local expert kernel skips. The
        views are workspace-backed and valid until ``combine``.

        With all-to-all, ``hidden`` is a dense ``[T, H]`` tensor or a calibrated
        NVFP4 ``QuantizedTensor`` with linear
        block scales. An encoded row travels as its packed E2M1 values
        (``H / 2`` bytes) and E4M3 block scales (``H / 16`` bytes), 9/32 of
        its BF16 bytes, and arrives as a ``QuantizedTensor`` of the same
        quantizer and layout; its tensor scale is the static calibration
        every rank shares, so it stays local. DeepEP carries the same encoded
        fields through byte-valued data and packed scale channels.
        Every rank of a step must send
        a layer's rows in one representation, since the all-to-all pairs
        the ranks' payload lists.

        Raises:
            RuntimeError: No step is open.
            ValueError: ``hidden`` exceeds the step capacity or is encoded
                otherwise.
        """
        self.enter(module)
        if hidden.shape[0] > self.capacity:
            raise ValueError("a rank's tokens exceed the step capacity")
        if self._elastic is not None:
            states, ids, weights = self._elastic.dispatch(
                hidden, topk_ids, topk_weights, capacity=self.capacity
            )
            # Each numerical provider declares the sentinel its kernel skips.
            return states, torch.where(ids < 0, invalid_expert, ids), weights
        encoded = isinstance(hidden, QuantizedTensor)
        if encoded:
            quantizer = hidden.quantizer
            if (
                quantizer.format != "nvfp4"
                or quantizer.calibrated_scale is None
                or hidden.scale_layout is not ScaleLayout.LINEAR
            ):
                raise ValueError(
                    "encoded hidden states are exchanged as calibrated NVFP4 "
                    "with linear block scales"
                )
            fields = hidden.buffers()
            states = [fields["values"], fields["block_scale"]]
        else:
            states = [hidden]

        # The states, then the ids the all-to-all routes by, then the
        # weights, as SGLang's NVFP4 dispatch orders them
        # (layers/moe/token_dispatcher/flashinfer.py:369-407).
        received = [
            value.flatten(0, 1)
            for value in self._alltoall.dispatch(
                topk_ids,
                [*states, topk_ids, topk_weights],
                self.capacity,
                invalid_token_expert_id=invalid_expert,
                expert_id_payload_index=len(states),
            )
        ]
        ids, weights = received[-2:]
        if not encoded:
            return received[0], ids, weights
        return (
            quantizer.from_tensors(
                {
                    "values": received[0],
                    "block_scale": received[1],
                    "tensor_scale": fields["tensor_scale"],
                },
                shape=(ids.shape[0], hidden.shape[1]),
                dtype=hidden.dtype,
            ),
            ids,
            weights,
        )

    def combine(self, partial: torch.Tensor, tokens: int) -> torch.Tensor:
        """Return the summed expert outputs of this rank's ``tokens`` tokens.

        ``partial`` is ``[P * C, H]`` in dispatch's returned row order,
        containing this rank's weighted expert sums. The transport combines
        them and restores the original source token order.
        """
        if self._elastic is not None:
            output = self._elastic.combine(partial)
        else:
            output = self._alltoall.combine(
                partial.view(self.group.size, self.capacity, partial.shape[-1]),
                self.capacity,
                payload_in_workspace=False,
                output_dtype=partial.dtype,
                use_low_precision=False,
            )
        if output.shape[0] != tokens:
            raise RuntimeError("the combine returned another token count")
        return output

    def close(self) -> None:
        """Release owned transport resources after contexts and graphs retire.

        DeepEP teardown is collective. The caller drains outstanding work
        first and leaves collective resources alive on distributed failure.
        FlashInfer's all-to-all allocation is cached by that provider.
        """
        self._control.close()
        if self._elastic is not None:
            self._elastic.close()
            self._elastic = None
        if self.fused is not None and self.attention_ranks:
            self.fused.close()
            self.fused = None
