"""Step coordination and token exchange for distributed expert layers.

An expert-parallel ``FusedMoE`` keeps a contiguous share of every layer's
routed experts on each rank of its expert group (see
``uniserve.distributed.partition_experts``). At every such layer each rank
sends each of its tokens, with its top-k ids and weights, to the ranks
holding its experts; each rank runs its local experts over the tokens it
received; the partial outputs travel back and sum on the token's rank. This
is the dispatch/combine of FlashInfer's MNNVL ``MoeAlltoAll``, the transport
SGLang (``layers/moe/token_dispatcher/flashinfer.py:235-555``) and vLLM
(``prepare_finalize/flashinfer_nvlink_one_sided.py:75-168``) use on GB200.
Hidden states of NVFP4 experts travel in the experts' calibrated input
encoding rather than BF16, as SGLang's NVFP4 dispatch sends them.

With the ``megamoe`` transport the exchange instead owns the NVSHMEM
symmetric staging of the fused MegaMoE kernel
(``uniserve.runtime.backends.moe.megamoe``), which dispatches, runs the
experts and combines in one launch per layer; the step protocol below is the
same, since that kernel pairs the ranks' launches too.

DeepEP provides elastic dispatch/combine with the same numerical contract.
Its union group can separate source-only attention ranks from expert-only
ranks. It compacts received rows, masks unused capacity and carries BF16 or
calibrated NVFP4 states. Each microbatch owns its communication stream and
buffer.

``ExpertExchange`` owns the resources for one sequence of expert calls,
shared by the layers and contexts running on that sequence's stream. Its
dispatch/combine alternation serializes buffer reuse. Participating ranks
agree on layer order and transfer capacity because the transports pair
calls by position. Every forward reaching expert layers is therefore a step:
``agree`` gathers each rank's token count and capability ordinal over the
group's host backend, selects a ready capability in cyclic order, and
returns one transfer capacity every rank computes alike. Other capabilities
retain their input and join the selected step with no tokens. All-to-all uses
powers of two plus the configured maximum, independently of the local
numerical graph shapes. MegaMoE retains maximum-sized symmetric storage
while dispatching and reducing only the local numerical extent. Ranks send
only their local graph's tokens;
``begin`` opens the step, each
expert layer exchanges at that capacity, and ``invoked`` records the layers
the forward reached, so the execution context can join the layers it
skipped with zero tokens before ``end`` closes the step. A rank without
work agrees with no tokens and joins every layer. This is the per-step
agreement vLLM makes with ``coordinate_batch_across_dp``
(``v1/worker/gpu/dp_utils.py:16-117``) and SGLang with
``prepare_mlp_sync_batch`` (``scheduler_components/dp_attn.py``).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.distributed as dist

from uniserve.distributed import Communicator
from uniserve.quantization import QuantizedTensor, ScaleLayout

__all__ = ["ExpertExchange", "JoinGraphs"]


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
        if group.size < 2 or max_tokens < 1:
            raise ValueError(
                "an expert exchange spans two or more ranks and one token"
            )
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
        memberships = [None] * group.size
        dist.all_gather_object(
            memberships, tuple(source_group), group=group._require()
        )
        sources = set(group.ranks[:source_count])
        for index, members in enumerate(memberships):
            expected_source = index < source_count
            if (
                bool(members) != expected_source
                or len(set(members)) != len(members)
                or not set(members) <= sources
                or (expected_source and group.ranks[index] not in members)
                or any(
                    memberships[group.ranks.index(member)] != members
                    for member in members
                )
            ):
                raise ValueError(
                    "expert sources must declare disjoint, consistent "
                    "groups of ranks that execute each forward together"
                )
        self._source_groups = tuple(
            tuple(group.ranks.index(member) for member in members)
            for members in dict.fromkeys(memberships)
            if members
        )
        if attention_ranks and transport not in {"deepep", "megamoe"}:
            raise ValueError("disaggregated experts require a split transport")
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

        # The host step state: the open step's per-rank capacity and the
        # expert layers it has exchanged at, by module identity.
        self.capacity = 0
        self.invoked: set[int] = set()
        # Powers of two bound the transfer graph variants logarithmically
        # while receive-buffer padding stays below twice the largest sender.
        # Local attention, dense layers and sampling keep their own shapes.
        self.capacities = (
            (max_tokens,)
            if self.fused is not None
            else tuple(
                1 << power for power in range((max_tokens - 1).bit_length())
            )
            + (max_tokens,)
        )
        self._records = torch.zeros((group.size, 3), dtype=torch.int64)
        # Capabilities take turns among the ranks that currently have work.
        # All ranks derive the same ordinal from the common runner catalog.
        self.kind = -1
        self.active = False
        # Whether the last agreement found every rank leaving the group.
        self.released = False
        self._ready = False

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
        """Publish one source warmup call, or receive it on expert ranks.

        All source ranks call this before each eager graph warmup. Expert
        ranks receive the capacity and execute the same layer sequence with
        no source tokens. Zero closes startup publication. This host control
        operation stays outside CUDA capture and the serving step protocol.
        """
        if not self.attention_ranks:
            return capacity
        value = torch.tensor(capacity, dtype=torch.int64)
        dist.broadcast(
            value, src=self.group.ranks[0], group=self.group._require()
        )
        return int(value.item())

    def agree(
        self, tokens: int, *, kind: int = 0, leaving: bool = False
    ) -> int:
        """Agree with the group on the next step's per-rank capacity.

        Every rank contributes the tokens its local graph or eager forward
        sends, or zero tokens when it has no forward, over
        the group's Gloo backend; every rank then computes the same result.
        Only source groups whose members all have the same capability ready
        may submit tokens. ``active`` reports whether this rank's pending
        forward was selected; otherwise it must retain its input and join
        with zero tokens. Ready capability ordinals take turns in cyclic order;
        ``self.kind`` identifies the selected capability after agreement. A rank
        with another capability joins with no tokens, then agrees again
        with its still-pending input. Returns zero when no rank has tokens,
        else the smallest capacity holding the selected senders. A rank shutting
        down agrees with ``leaving`` until ``released`` reports that every
        rank is leaving without tokens, so no rank stops agreeing while
        another still waits on it.

        Raises:
            RuntimeError: A rank sends more than the configured maximum.
        """
        local = torch.tensor([[tokens, int(leaving), kind]])
        dist.all_gather(
            list(self._records.split(1)), local, group=self.group._require()
        )
        records = self._records.tolist()
        eligible = [
            members
            for members in self._source_groups
            if all(records[rank][0] for rank in members)
            and len({records[rank][2] for rank in members}) == 1
        ]
        ready = sorted({records[members[0]][2] for members in eligible})
        self.active = False
        self.released = all(not count and left for count, left, _ in records)
        if not ready:
            return 0

        self.kind = next((tag for tag in ready if tag > self.kind), ready[0])
        selected = {
            rank
            for members in eligible
            if records[members[0]][2] == self.kind
            for rank in members
        }
        self.active = self.group.rank in selected
        most = max(records[rank][0] for rank in selected)

        capacity = next(
            (value for value in self.capacities if value >= most), None
        )
        if capacity is None:
            raise RuntimeError(
                f"expert step of {most} tokens exceeds the configured "
                f"{self.max_tokens} tokens per rank"
            )
        return capacity

    def begin(self, capacity: int) -> None:
        """Open a step carrying at most ``capacity`` tokens from each rank.

        Callers prepare their numerical contexts before the first step.
        The first expert call also waits for peers to reach that numerical
        boundary, so preceding attention compilation cannot consume a device
        protocol timeout. Subsequent steps require no preparation collective.
        """
        if not 0 < capacity <= self.max_tokens:
            raise ValueError(
                f"step capacity {capacity} is outside the exchange's "
                f"{self.max_tokens} tokens"
            )
        self.capacity = capacity
        self.invoked.clear()

    def end(self) -> None:
        """Close the open step."""
        self.capacity = 0
        self.invoked.clear()

    def enter(self, module: int) -> None:
        """Record that the open step reaches expert layer ``module``.

        Raises:
            RuntimeError: No step is open.
        """
        if not self.capacity:
            raise RuntimeError("an expert exchange runs inside an open step")
        if (
            self._elastic is not None or self.attention_ranks
        ) and not self._ready:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm the expert exchange before capture")
            # Host readiness belongs immediately before communication: source
            # ranks may compile attention before reaching their first expert.
            dist.all_reduce(
                torch.zeros((), dtype=torch.int32), group=self.group._require()
            )
            self._ready = True
        self.invoked.add(module)

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
        if self._elastic is not None:
            self._elastic.close()
            self._elastic = None
        if self.fused is not None and self.attention_ranks:
            self.fused.close()
            self.fused = None


class JoinGraphs:
    """Captured participation of a rank without work in expert steps.

    A rank of an expert group that has no forward of its own still takes
    part in every step another rank starts: at each expert-parallel layer of
    ``context`` it exchanges with no tokens of its own and runs its experts
    over the rows it receives (``ExecutionContext.join_expert_layers``).
    Launched eagerly, that is a host-issued run of kernels per layer, and
    every other rank waits for it at each layer's exchange; one captured
    graph per step capacity keeps a join at device speed, as SGLang's
    data-parallel ranks without work replay the decode CUDA graph in IDLE
    mode (``model_executor/runner/decode_cuda_graph_runner.py:1395-1450``).

    Construction is collective over ``exchange``'s group: every rank
    captures the same ``capacities`` in the same order, largest first, and
    warms each with one eager join step, which exchanges with every rank.
    With ``warm=False``, the caller has already warmed those same bindings
    and capacities; construction only captures and performs no exchange.
    The graphs allocate from ``pools`` (see ``CUDAGraph``) and read the
    bindings ``context`` has prepared, so the caller closes them before the
    context.
    """

    def __init__(
        self,
        context,
        exchange: ExpertExchange,
        capacities,
        *,
        pools=None,
        step=None,
        warm=True,
    ) -> None:
        from .cuda_graph import CUDAGraph

        self._graphs: dict[int, CUDAGraph] = {}
        try:
            for capacity in sorted(frozenset(capacities), reverse=True):

                def join(capacity=capacity):
                    if step is not None:
                        return step(capacity)
                    exchange.begin(capacity)
                    try:
                        context.join_expert_layers()
                    finally:
                        exchange.end()

                if warm:
                    with context.activate():
                        join()
                graph = CUDAGraph(context=context, pools=pools)
                self._graphs[capacity] = graph
                graph.capture(join)
        except BaseException:
            self.close()
            raise

    @property
    def capacities(self) -> frozenset[int]:
        """The step capacities a join replays at."""
        return frozenset(self._graphs)

    def replay(self, capacity: int) -> None:
        """Join the open step of ``capacity`` on the context's stream.

        Raises:
            RuntimeError: No join of ``capacity`` was captured.
        """
        graph = self._graphs.get(capacity)
        if graph is None:
            raise RuntimeError(
                f"no captured expert join serves step capacity {capacity}"
            )
        graph.replay()

    def close(self) -> None:
        """Release every join graph; the caller has drained their replays."""
        graphs, self._graphs = self._graphs, {}
        for graph in graphs.values():
            graph.close()
