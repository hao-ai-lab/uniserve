"""Token exchange of expert-parallel layers over NVLink all-to-all.

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

``ExpertExchange`` owns one ``MoeAlltoAll`` workspace for one worker and
expert group, shared by every expert layer and execution context of the
worker, which run their expert layers on one stream, one after another; the
workspace's dispatch/combine alternation serializes them. Its ranks must
also agree on
the order and the token capacity of every exchange, since ``MoeAlltoAll``
pairs the ranks' calls by position and lays out its receive buffers by the
capacity. Every forward that reaches expert layers is therefore one step:
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
    """One worker's all-to-all workspace for an expert group.

    Construction is collective over ``group``: every rank constructs its
    exchange at the same point of its startup, because the workspace maps
    peer memory. ``max_tokens`` bounds every step's per-rank capacity. The
    workspace stays mapped until process exit; FlashInfer caches it
    process-wide. ``transport`` selects the NVLink all-to-all (``alltoall``)
    or the fused MegaMoE kernel (``megamoe``, NVFP4 experts), which also
    needs the experts' gated width ``intermediate_size`` and gate
    ``activation``.
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
        transport: Literal["alltoall", "megamoe"] = "alltoall",
        intermediate_size: int | None = None,
        activation: str | None = None,
    ) -> None:
        if group.size < 2 or max_tokens < 1:
            raise ValueError(
                "an expert exchange spans two or more ranks and one token"
            )
        self.group = group
        self.max_tokens = max_tokens
        self.device = device
        self.transport = transport
        self.fused = None
        if transport == "megamoe":
            if intermediate_size is None or activation is None:
                raise ValueError(
                    "the MegaMoE transport needs the experts' width and gate"
                )
            from uniserve.runtime.backends.moe.megamoe import MegaMoEBuffer

            self.fused = MegaMoEBuffer(
                group,
                max_tokens=max_tokens,
                num_experts=num_experts,
                top_k=top_k,
                hidden=hidden_size,
                intermediate=intermediate_size,
                activation=activation,
                device=device,
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
        # Whether the last agreement found every rank leaving the group.
        self.released = False

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

    def agree(
        self, tokens: int, *, kind: int = 0, leaving: bool = False
    ) -> int:
        """Agree with the group on the next step's per-rank capacity.

        Every rank contributes the tokens its local graph or eager forward
        sends, or zero tokens when it has no forward, over
        the group's Gloo backend; every rank then computes the same result.
        Ready capability ordinals take turns in ascending cyclic order;
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
        ready = sorted({tag for count, _, tag in records if count})
        self.released = not ready and all(left for _, left, _ in records)
        if not ready:
            return 0

        self.kind = next((tag for tag in ready if tag > self.kind), ready[0])
        most = max(count for count, _, tag in records if tag == self.kind)

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
        """Open a step carrying at most ``capacity`` tokens from each rank."""
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
        capacity: row ``p * C + i`` is the ``i``-th token rank ``p`` sent
        here. Ids of experts this rank does not hold, and of unused rows,
        read ``invalid_expert``, which the local expert kernel skips. The
        views are workspace-backed and valid until ``combine``.

        ``hidden`` is a dense ``[T, H]`` tensor or, as the NVFP4 expert
        kernels read it, a calibrated NVFP4 ``QuantizedTensor`` with linear
        block scales. An encoded row travels as its packed E2M1 values
        (``H / 2`` bytes) and E4M3 block scales (``H / 16`` bytes), 9/32 of
        its BF16 bytes, and arrives as a ``QuantizedTensor`` of the same
        quantizer and layout; its tensor scale is the static calibration
        every rank shares, so it stays local. Every rank of a step must send
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

        ``partial`` is ``[P * C, H]``: row ``p * C + i`` holds this rank's
        experts' weighted sum for the ``i``-th token rank ``p`` sent. Each
        token's partial sums from every rank add in FP32 and round to BF16
        once.
        """
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
    ) -> None:
        from .cuda_graph import CUDAGraph

        self._graphs: dict[int, CUDAGraph] = {}
        try:
            for capacity in sorted(frozenset(capacities), reverse=True):

                def join(capacity=capacity):
                    exchange.begin(capacity)
                    try:
                        context.join_expert_layers()
                    finally:
                        exchange.end()

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
