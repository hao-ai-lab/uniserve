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

``ExpertExchange`` owns one ``MoeAlltoAll`` workspace for one worker and
expert group, shared by every expert layer and execution context of the
worker, which run their expert layers on one stream, one after another; the
workspace's dispatch/combine alternation serializes them. Its ranks must
also agree on
the order and the token capacity of every exchange, since ``MoeAlltoAll``
pairs the ranks' calls by position and lays out its receive buffers by the
capacity. Every forward that reaches expert layers is therefore one step:
``agree`` gathers each rank's step kind and token count over the group's
host backend and returns one capacity every rank computes alike, the
smallest capacity every participating kind's graphs serve that holds the
most tokens any rank sends (ranks pad to it, as SGLang's and vLLM's
data-parallel ranks pad to the largest rank); ``begin`` opens the step, each
expert layer exchanges at that capacity, and ``invoked`` records the layers
the forward reached, so the execution context can join the layers it
skipped with zero tokens before ``end`` closes the step. A rank without
work agrees with no tokens and joins every layer. This is the per-step
agreement vLLM makes with ``coordinate_batch_across_dp``
(``v1/worker/gpu/dp_utils.py:16-117``) and SGLang with
``prepare_mlp_sync_batch`` (``scheduler_components/dp_attn.py``).
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from uniserve.distributed import Communicator
from uniserve.quantization import QuantizedTensor, ScaleLayout

__all__ = ["ExpertExchange"]


class ExpertExchange:
    """One worker's all-to-all workspace for an expert group.

    Construction is collective over ``group``: every rank constructs its
    exchange at the same point of its startup, because the workspace maps
    peer memory. ``max_tokens`` bounds every step's per-rank capacity. The
    workspace stays mapped until process exit; FlashInfer caches it
    process-wide.
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
    ) -> None:
        from flashinfer.comm.comm_backend import TorchDistBackend
        from flashinfer.comm.mapping import Mapping
        from flashinfer.comm.mnnvl import MnnvlConfig
        from flashinfer.comm.trtllm_moe_alltoall import MoeAlltoAll

        if group.size < 2 or max_tokens < 1:
            raise ValueError(
                "an expert exchange spans two or more ranks and one token"
            )
        self.group = group
        self.max_tokens = max_tokens
        self.device = device

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
            self._alltoall = MoeAlltoAll(
                mapping,
                max_tokens,
                top_k,
                num_experts,
                hidden_size=hidden_size,
                mnnvl_config=MnnvlConfig(
                    comm_backend=TorchDistBackend(group._require())
                ),
            )

        # The host step state: the open step's per-rank capacity and the
        # expert layers it has exchanged at, by module identity.
        self.capacity = 0
        self.invoked: set[int] = set()
        # The capacities each registered step kind's graphs serve, in
        # registration order, which every rank follows alike.
        self._kinds: list[frozenset[int]] = []
        self._records = torch.zeros((group.size, 3), dtype=torch.int64)
        # Whether the last agreement found every rank leaving the group.
        self.released = False

    @property
    def kinds(self) -> tuple[frozenset[int], ...]:
        """The capacities each registered step kind serves, in order."""
        return tuple(self._kinds)

    def register(self, capacities: frozenset[int]) -> int:
        """Register a step kind whose graphs serve ``capacities``; return it.

        Every rank registers the same kinds in the same order, so a kind
        number names the same graphs on every rank. A capacity above the
        exchange's ``max_tokens`` is not served.
        """
        self._kinds.append(
            frozenset(value for value in capacities if value <= self.max_tokens)
        )
        return len(self._kinds) - 1

    def agree(
        self, kind: int | None, tokens: int, *, leaving: bool = False
    ) -> int:
        """Agree with the group on the next step's per-rank capacity.

        Every rank contributes its step kind and the tokens its forward
        sends, or ``kind=None`` and no tokens when it has no forward, over
        the group's Gloo backend; every rank then computes the same result.
        Returns zero when no rank has tokens, else the smallest capacity at
        least the most tokens any rank sends that every kind taking part
        serves; a rank without work serves any capacity. A rank shutting
        down agrees with ``leaving`` until ``released`` reports that every
        rank is leaving without tokens, so no rank stops agreeing while
        another still waits on it.

        Raises:
            RuntimeError: No capacity serves every kind taking part.
        """
        local = torch.tensor(
            [[-1 if kind is None else kind, tokens, int(leaving)]]
        )
        dist.all_gather(
            list(self._records.split(1)), local, group=self.group._require()
        )
        records = self._records.tolist()
        most = max(count for _, count, _ in records)
        self.released = not most and all(left for _, _, left in records)
        if not most:
            return 0

        served = [self._kinds[number] for number, count, _ in records if count]
        capacity = min(
            (
                value
                for value in frozenset().union(*served)
                if value >= most and all(value in kind for kind in served)
            ),
            default=None,
        )
        if capacity is None:
            raise RuntimeError(
                f"no expert step capacity serves {most} tokens for every "
                "step kind taking part"
            )
        return capacity

    def begin(self, capacity: int) -> None:
        """Open a step whose exchanges carry ``capacity`` tokens per rank."""
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
            ValueError: ``hidden`` exceeds the step capacity or is encoded
                otherwise.
        """
        if not self.capacity:
            raise RuntimeError("an expert exchange runs inside an open step")
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

        self.invoked.add(module)
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
