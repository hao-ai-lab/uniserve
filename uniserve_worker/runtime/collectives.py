"""Bounded, execution-owned peer reduction workspaces for local tensor parallelism."""

from __future__ import annotations

import socket
from ctypes import c_void_p
from typing import Any

import torch
import torch.distributed as dist

from ..nn.mesh import Communicator

# Small-message peer reductions use the established 8 MiB custom-all-reduce
# budget. Larger payloads retain NCCL's bandwidth-oriented algorithms.
_MAX_REDUCTION_BYTES = 8 * 1024 * 1024


def supports_peer_reduction(group: Communicator) -> bool:
    """Resolve a common CUDA peer capability before allocating storage.

    UUID checks preserve physical device identity when process-local visibility
    differs. Every rank makes the same selection; cross-host and inaccessible
    peer groups continue through their ordinary process-group provider.
    """

    if group.world_size not in (2, 4, 8, 16) or group.device.type != "cuda":
        return False
    process_group = group._require()
    if dist.get_backend(process_group) != "nccl":
        return False
    from flashinfer.utils import is_confidential_compute

    device = group.device.index
    properties = torch.cuda.get_device_properties(device)
    identity = (
        socket.gethostname(),
        device,
        str(properties.uuid),
        0 if is_confidential_compute() else properties.major,
    )
    identities: list[Any] = [None] * group.world_size
    dist.all_gather_object(identities, identity, group=process_group)
    accessible = True
    for host, peer_device, uuid, major in identities:
        if (
            host != identity[0]
            or major < 9
            or peer_device >= torch.cuda.device_count()
            or str(torch.cuda.get_device_properties(peer_device).uuid) != uuid
        ):
            accessible = False
        elif peer_device != device:
            accessible &= torch.cuda.can_device_access_peer(device, peer_device)
    accessible &= len({item[2] for item in identities}) == group.world_size
    capabilities: list[Any] = [None] * group.world_size
    dist.all_gather_object(capabilities, bool(accessible), group=process_group)
    return all(capabilities)


class PeerSumReduction:
    """Own one serialized full-device scope's graph-replayable sum workspace.

    BF16/FP16 contributions accumulate in FP32 before the output rounding. The
    native kernel stages each contribution in peer storage before overwriting
    its input, preserving Communicator's in-place output contract. The native
    launch geometry requires the physical device's complete SM domain.
    """

    def __init__(self, group: Communicator) -> None:
        from flashinfer.comm.allreduce import TRTLLMAllReduceFusionWorkspace, allreduce_fusion
        from flashinfer.comm.trtllm_ar import AllReduceFusionPattern

        self.device = group.device
        self._reduce = allreduce_fusion
        self._pattern = AllReduceFusionPattern.kAllReduce
        self._workspace: TRTLLMAllReduceFusionWorkspace | None = TRTLLMAllReduceFusionWorkspace(
            tp_size=group.world_size,
            tp_rank=dist.get_rank(group._require()),
            max_token_num=1024,
            hidden_dim=_MAX_REDUCTION_BYTES // (1024 * 2),
            dtype=torch.bfloat16,
            group=group._require(),
        )

    def try_reduce(self, value: torch.Tensor) -> bool:
        """Sum supported contiguous hidden rows; preserve other NCCL input domains."""

        if (
            value.device != self.device
            or value.dtype not in (torch.bfloat16, torch.float16)
            or value.ndim < 2
            or not value.is_contiguous()
            or value.numel() == 0
            or value.numel() * value.element_size() > _MAX_REDUCTION_BYTES
            or value.shape[-1] % 128
            or value.shape[-1] > 8192
        ):
            return False
        rows = value.view(-1, value.shape[-1])
        self._reduce(
            input=rows,
            output=rows,
            workspace=self._workspace,
            pattern=self._pattern,
            use_oneshot=True,
            fp32_acc=True,
            launch_with_pdl=False,
            trigger_completion_at_end=True,
        )
        return True

    def close(self) -> None:
        """Release peer mappings after all dependent graph executions retire."""

        if self._workspace is None:
            return
        from flashinfer.comm.trtllm_ar import (
            cudart,
            trtllm_destroy_ipc_workspace_for_all_reduce_fusion,
        )

        torch.cuda.synchronize(self.device)
        workspace = self._workspace
        self._workspace = None
        # The workspace owns a separate cudaMalloc control word and its factory
        # retains peer mappings in a registry. Retire both before dropping the
        # wrapper's references to the symmetric-memory handles.
        trtllm_destroy_ipc_workspace_for_all_reduce_fusion(workspace.ipc_handles)
        cudart.cudaFree(c_void_p(workspace.metadata["control_flag_ptr"]))
        workspace.destroy()
