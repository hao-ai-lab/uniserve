"""Bounded, execution-owned peer reduction workspaces for local tensor parallelism."""

from __future__ import annotations

import socket
from collections.abc import Iterable
from ctypes import addressof, c_void_p
from typing import Any

import torch
import torch.distributed as dist

from uniserve.distributed.mesh import Communicator
from uniserve.runtime.cuda import cuda_status, cuda_value, driver

# Small-message peer reductions use the established 8 MiB custom-all-reduce
# budget. Larger payloads retain NCCL's bandwidth-oriented algorithms.
_MAX_REDUCTION_BYTES = 8 * 1024 * 1024


def supports_peer_reduction(group: Communicator) -> bool:
    """Resolve a common CUDA peer capability before allocating storage.

    UUID checks preserve physical device identity when process-local visibility
    differs. Every rank makes the same selection; cross-host and inaccessible
    peer groups continue through their ordinary process-group provider.
    """

    if group.size not in (2, 4, 8, 16) or group.device.type != "cuda":
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
    identities: list[Any] = [None] * group.size
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
    accessible &= len({item[2] for item in identities}) == group.size

    capabilities: list[Any] = [None] * group.size
    dist.all_gather_object(capabilities, bool(accessible), group=process_group)
    return all(capabilities)


class PeerReduction:
    """Own one serialized full-device scope's graph-replayable sum workspace.

    BF16/FP16 contributions accumulate in FP32 before the output rounding. The
    native kernel stages each contribution in peer storage before overwriting
    its input, preserving Communicator's in-place output behavior. The native
    launch requires the physical device's complete SM domain.
    """

    def __init__(self, group: Communicator) -> None:
        from flashinfer.comm.allreduce import TRTLLMAllReduceFusionWorkspace, allreduce_fusion
        from flashinfer.comm.trtllm_ar import AllReduceFusionPattern

        self.device = group.device
        self._reduce = allreduce_fusion
        self._pattern = AllReduceFusionPattern.kAllReduce

        # max_token_num rows of hidden_dim BF16 elements fill exactly the
        # 8 MiB small-message budget above.
        self._workspace: TRTLLMAllReduceFusionWorkspace | None = TRTLLMAllReduceFusionWorkspace(
            tp_size=group.size,
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

        # The kernel consumes a [rows, hidden] matrix; leading axes collapse.
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


class _CollectiveWork:
    """A published transfer whose consumer supplies the final stream dependency."""

    def __init__(self, event: torch.cuda.Event, owner: NcclCommunicator):
        self.event, self.owner = event, owner

    def block_current_stream(self) -> None:
        stream = torch.cuda.current_stream(self.owner._stream.device)
        stream.wait_event(self.event)
        if stream == self.owner._stream and self.owner._pending is self:
            # Once the origin has joined, this event must not introduce an
            # external dependency into a later, independent graph capture.
            self.owner._pending = None


class NcclCommunicator:
    """Own ordered NCCL transport for a borrowed computation stream.

    Synchronous numerical calls execute on the computation stream. Streamed
    publications use an owned stream in the same CUDA/Green Context and join
    when their remote outputs are consumed. Rank ordering matches the process
    group; the model Communicator handles logical membership ordering.
    """

    def __init__(self, group, stream: torch.cuda.Stream) -> None:
        import nccl.bindings.nccl as nccl

        self._nccl = nccl
        self._stream = stream
        self._transfer = None
        self._raw_transfer = None
        self._pending = None
        self._comm = c_void_p()
        self._windows: dict[tuple[int, int], tuple[c_void_p, torch.Tensor]] = {}
        self._rank = dist.get_rank(group)
        self._size = dist.get_world_size(group)
        self._ranks = tuple(dist.get_process_group_ranks(group))

        identity = [bytes(nccl.get_unique_id()) if self._rank == 0 else None]
        dist.broadcast_object_list(identity, src=self._ranks[0], group=group)
        unique_id = identity[0]
        if not isinstance(unique_id, bytes):
            raise RuntimeError("NCCL initialization did not receive a unique identifier")

        try:
            with torch.cuda.device(stream.device):
                cu = driver()
                origin = cu.CUstream(stream.cuda_stream)
                green = cuda_value(cu.cuStreamGetGreenCtx(origin), "query communication context")

                config = nccl.Config()
                # The deployed Green Context driver cannot batch-copy mapped
                # peer windows. Keep NCCL's native CTA algorithms on that
                # partition's SMs; ordinary contexts can use copy engines.
                config.cta_policy = 0 if int(green) else 0x02  # DEFAULT or ZERO
                nccl.comm_init_rank_config(
                    addressof(self._comm), self._size, bytearray(unique_id), self._rank, config.ptr
                )

                # Reuse the computation's actual context, including its SM
                # partition. Communication owns a stream, not another resource
                # partition, and every transfer rejoins its numerical consumer.
                flags = int(cu.CUstream_flags.CU_STREAM_NON_BLOCKING)
                if int(green):
                    raw = cuda_value(
                        cu.cuGreenCtxStreamCreate(green, flags, stream.priority),
                        "create partitioned communication stream",
                    )
                else:
                    context = cuda_value(cu.cuStreamGetCtx(origin), "query stream context")
                    cuda_status(cu.cuCtxPushCurrent(context), "enter communication context")
                    try:
                        raw = cuda_value(
                            cu.cuStreamCreateWithPriority(flags, stream.priority),
                            "create communication stream",
                        )
                    finally:
                        cuda_status(cu.cuCtxPopCurrent(), "leave communication context")

                self._raw_transfer = raw
                self._transfer = torch.cuda.ExternalStream(int(raw), device=stream.device)
        except BaseException as error:
            # Partial construction still attempts every acquired resource's
            # release, recording cleanup failures on the original error.
            if self._raw_transfer is not None:
                try:
                    cuda_status(
                        driver().cuStreamDestroy(self._raw_transfer), "destroy communication stream"
                    )
                except BaseException as cleanup_error:
                    error.add_note(f"Communication stream cleanup failed: {cleanup_error!r}")
                self._raw_transfer = None
            if self._comm.value:
                try:
                    nccl.comm_abort(self._comm.value)
                except BaseException as cleanup_error:
                    error.add_note(f"NCCL initialization cleanup failed: {cleanup_error!r}")
                self._comm = c_void_p()
            raise

    def _arguments(
        self, value: torch.Tensor, output: torch.Tensor | None = None, *, asynchronous: bool = False
    ) -> tuple[int, int]:
        """Validate operands and return the communicator and stream for a launch."""

        if not self._comm.value:
            raise RuntimeError("computation collective is closed")
        if value.device != self._stream.device or not value.is_contiguous():
            raise ValueError("computation collectives require contiguous tensors on their device")
        if output is not None and (
            output.device != value.device
            or output.dtype != value.dtype
            or not output.is_contiguous()
        ):
            raise ValueError("collective output must match input dtype, device, and layout")

        if asynchronous:
            return self._comm.value, self._transfer.cuda_stream

        if self._pending is not None:
            # Synchronous collectives keep the same communicator order even
            # when a projection consumer invokes one before exhausting a gather.
            self._stream.wait_event(self._pending.event)
            self._pending = None
        return self._comm.value, self._stream.cuda_stream

    def _start(self, operation, *args) -> _CollectiveWork:
        """Launch on the transfer stream after the computation stream's inputs."""

        self._transfer.wait_stream(self._stream)
        completed = torch.cuda.Event()
        try:
            operation(*args)
        except BaseException:
            # Join the partial launch back so the computation stream never
            # overtakes a failed transfer's still-running kernel.
            completed.record(self._transfer)
            self._stream.wait_event(completed)
            raise

        completed.record(self._transfer)
        work = _CollectiveWork(completed, self)
        self._pending = work
        return work

    def start_all_gather(self, output: torch.Tensor, value: torch.Tensor) -> _CollectiveWork:
        if output.numel() != value.numel() * self._size:
            raise ValueError("collective gather output must hold every rank's contribution")
        return self._start(
            self._nccl.all_gather,
            value.data_ptr(),
            output.data_ptr(),
            value.numel(),
            self._dtype(value),
            *self._arguments(value, output, asynchronous=True),
        )

    def start_all_to_all(self, output: torch.Tensor, value: torch.Tensor) -> _CollectiveWork:
        if output.numel() != value.numel() or value.numel() % self._size:
            raise ValueError("asynchronous exchange requires equal peer payloads")
        return self._start(
            self._nccl.allto_all,
            value.data_ptr(),
            output.data_ptr(),
            value.numel() // self._size,
            self._dtype(value),
            *self._arguments(value, output, asynchronous=True),
        )

    def _dtype(self, value: torch.Tensor) -> int:
        types = self._nccl.DataType
        return {
            torch.bool: types.Uint8,
            torch.uint8: types.Uint8,
            torch.int8: types.Int8,
            torch.int32: types.Int32,
            torch.int64: types.Int64,
            torch.float16: types.Float16,
            torch.bfloat16: types.Bfloat16,
            torch.float32: types.Float32,
            torch.float64: types.Float64,
        }[value.dtype]

    def all_reduce(self, value: torch.Tensor, op: str = "sum") -> None:
        reduction = {
            "sum": self._nccl.RedOp.Sum,
            "max": self._nccl.RedOp.Max,
            "min": self._nccl.RedOp.Min,
        }[op]
        self._nccl.all_reduce(
            value.data_ptr(),
            value.data_ptr(),
            value.numel(),
            self._dtype(value),
            reduction,
            *self._arguments(value),
        )

    def all_gather(self, output: torch.Tensor, value: torch.Tensor) -> None:
        if output.numel() != value.numel() * self._size:
            raise ValueError("collective gather output must hold every rank's contribution")
        self._nccl.all_gather(
            value.data_ptr(),
            output.data_ptr(),
            value.numel(),
            self._dtype(value),
            *self._arguments(value, output),
        )

    def all_to_all(
        self,
        output: torch.Tensor,
        value: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
    ) -> None:
        self._arguments(value, output)
        if len(input_splits) != self._size or len(output_splits) != self._size:
            raise ValueError("collective exchange requires one split per rank")

        if len(set(input_splits + output_splits)) == 1:
            self._nccl.allto_all(
                value.data_ptr(),
                output.data_ptr(),
                value.numel() // self._size,
                self._dtype(value),
                *self._arguments(value, output),
            )
            return

        # Uneven exchanges decompose into one grouped send/recv pair per peer.
        send_rows, receive_rows = value.split(input_splits), output.split(output_splits)
        self._nccl.group_start()
        try:
            for peer, (send, receive) in enumerate(zip(send_rows, receive_rows, strict=True)):
                if send.numel():
                    self.send(send, self._ranks[peer])
                if receive.numel():
                    self.recv(receive, self._ranks[peer])
        finally:
            self._nccl.group_end()

    def gather(self, outputs: list[torch.Tensor] | None, value: torch.Tensor, root: int) -> None:
        self._arguments(value)
        self._ranks.index(root)
        if self._ranks[self._rank] == root:
            if outputs is None or len(outputs) != self._size:
                raise ValueError("collective gather requires one destination per rank")
            for output in outputs:
                self._arguments(value, output)
                if output.numel() != value.numel():
                    raise ValueError("gather destination must match the contribution size")

        self._nccl.group_start()
        try:
            self.send(value, root)
            if self._ranks[self._rank] == root:
                assert outputs is not None
                for peer, output in zip(self._ranks, outputs, strict=True):
                    self.recv(output, peer)
        finally:
            self._nccl.group_end()

    def broadcast(self, value: torch.Tensor, root: int) -> None:
        self._nccl.broadcast(
            value.data_ptr(),
            value.data_ptr(),
            value.numel(),
            self._dtype(value),
            self._ranks.index(root),
            *self._arguments(value),
        )

    def reduce_scatter(self, output: torch.Tensor, value: torch.Tensor) -> None:
        if value.numel() != output.numel() * self._size:
            raise ValueError("collective reduction requires one output-sized partition per rank")
        self._nccl.reduce_scatter(
            value.data_ptr(),
            output.data_ptr(),
            output.numel(),
            self._dtype(value),
            self._nccl.RedOp.Sum,
            *self._arguments(value, output),
        )

    def send(self, value: torch.Tensor, peer: int) -> None:
        self._nccl.send(
            value.data_ptr(),
            value.numel(),
            self._dtype(value),
            self._ranks.index(peer),
            *self._arguments(value),
        )

    def recv(self, value: torch.Tensor, peer: int) -> None:
        self._nccl.recv(
            value.data_ptr(),
            value.numel(),
            self._dtype(value),
            self._ranks.index(peer),
            *self._arguments(value),
        )

    def send_recv(self, output: torch.Tensor, value: torch.Tensor, dst: int, src: int) -> None:
        self._nccl.group_start()
        try:
            self.send(value.reshape(-1).view(torch.uint8), dst)
            self.recv(output.reshape(-1).view(torch.uint8), src)
        finally:
            self._nccl.group_end()

    def register_buffers(self, *buffers: torch.Tensor) -> None:
        """Register matching VMM allocations collectively before their first use.

        Every rank supplies the same buffer sequence and byte capacities. Both
        sides of a symmetric exchange must use registered allocations; mixing
        these with ordinary CUDA allocations is unsafe under graph replay.
        Registrations retain their backing until the communicator is closed.
        """

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("register communication buffers before CUDA graph capture")

        for value in buffers:
            self._arguments(value)
            key = (value.data_ptr(), value.numel() * value.element_size())
            if key in self._windows:
                continue
            window = c_void_p()
            self._nccl.comm_window_register(
                self._comm.value, key[0], key[1], addressof(window), 0x01
            )  # NCCL_WIN_COLL_SYMMETRIC
            self._windows[key] = window, value

    def close(self) -> None:
        """Destroy the communicator after its runner has retired all graph use."""

        if self._transfer is not None:
            self._transfer.synchronize()

        if self._comm.value:
            for window, _ in self._windows.values():
                self._nccl.comm_window_deregister(self._comm.value, window.value)
            self._windows.clear()
            communicator, self._comm = self._comm.value, c_void_p()
            self._nccl.comm_destroy(communicator)

        if self._raw_transfer is not None:
            cuda_status(
                driver().cuStreamDestroy(self._raw_transfer), "destroy communication stream"
            )
            self._raw_transfer = None
            self._transfer = self._pending = None


def allocate_stream_collectives(
    groups: Iterable[Communicator], stream: torch.cuda.Stream
) -> dict[str, NcclCommunicator]:
    """Allocate independent communication resources for one computation stream."""

    bindings = {}
    try:
        for communicator in groups:
            if communicator.size == 1:
                continue
            group = communicator._require()
            if dist.get_backend(group) == "nccl" and group.group_name not in bindings:
                bindings[group.group_name] = NcclCommunicator(group, stream)
    except BaseException as error:
        for binding in reversed(tuple(bindings.values())):
            try:
                binding.close()
            except BaseException as cleanup_error:
                error.add_note(f"collective binding cleanup failed: {cleanup_error!r}")
        raise
    return bindings


def allocate_peer_reductions(groups: Iterable[Communicator]) -> dict[Any, PeerReduction]:
    """Allocate collective scratch for one serialized full-device execution scope.

    The runner invokes this before variable memory pools are sized and owns
    the returned workspaces until all of its graph executables retire.
    """

    bindings: dict[Any, PeerReduction] = {}
    try:
        for group in groups:
            if group.size == 1:
                continue
            process_group = group._require()
            if process_group not in bindings and supports_peer_reduction(group):
                bindings[process_group] = PeerReduction(group)
        return bindings
    except Exception:
        for reduction in reversed(tuple(bindings.values())):
            reduction.close()
        raise
