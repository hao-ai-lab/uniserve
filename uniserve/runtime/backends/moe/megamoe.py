"""Fused expert-parallel NVFP4 experts: the CuTeDSL MegaMoE kernel.

One persistent kernel per expert layer and rank of an expert group
dispatches each token to the ranks holding its experts, runs the W4A4 FC1
GEMM, the gated activation on the dequantized FC1 products, the NVFP4
encoding of the gated product, the FC2 GEMM, and returns the partial sums
to the token's rank, over NVSHMEM symmetric memory. It is FlashInfer's
MegaMoE (``moe_ep/kernel_src/cutedsl_megamoe``), vendored in
``uniserve_kernels.megamoe`` with the gated activation extended to
tanh-approximated GELU (``up * gate * sigmoid(sqrt(8/pi) * (gate +
0.044715 * gate^3))`` on the dequantized gate, in the epilogue every kernel
configuration runs).

The numerical contract is the W4A4 ModelOpt one the CuTeDSL provider keeps:
hidden states encode to NVFP4 with the static calibrated scale ``s13`` of
``up_gate.input_quantizer`` (global scale ``1 / s13``) into the bytes the
other NVFP4 providers read (stored encodings as they are, BF16 rows with
FlashInfer's ``fp4_quantize``) and stage into the symmetric buffer, FC1 products
dequantize with ``tensor_scale13[e] * s13`` before the activation, the gated
product encodes with ``1 / s2`` of ``down.input_quantizer``, FC2 products
dequantize with ``tensor_scale2[e] * s2``, and each route's FP32 weight
multiplies after FC2 (``apply_topk_in_fc1=False``), with the explicit BF16
top-k combine (no in-kernel atomic reduce, no low-precision wire, no clamp).

Weights stay one resident copy: preparation places ``up_gate`` in
``RowOrder.INTERLEAVED_16`` (each 16-row gate block followed by its up
block, as the fused gated epilogue reads) and ``down`` linearly, both with
128x4-swizzled block scales, and the kernel reads K-major transposed views
of them. ``MegaMoEBuffer`` holds the worker's symmetric staging and compiled
kernel for the group; every expert layer on the worker's stream shares it.
"""

from __future__ import annotations

import torch

from uniserve.distributed import Communicator
from uniserve.quantization import RowOrder

from . import NVFP4Backend, NVFP4Operator

__all__ = ["Backend", "MegaMoEBuffer"]

# NVSHMEM initializes once per process, collectively over the expert group.
_NVSHMEM_GROUP: Communicator | None = None


def _initialize_nvshmem(group: Communicator, device: torch.device) -> None:
    """Join NVSHMEM with the expert group's ranks, once per process.

    Rank 0 creates the unique id and broadcasts it over the group's host
    backend; every rank then initializes NVSHMEM as its group rank.
    """
    global _NVSHMEM_GROUP
    if _NVSHMEM_GROUP is not None:
        if _NVSHMEM_GROUP != group:
            raise RuntimeError("NVSHMEM already serves another expert group")
        return

    import numpy as np
    import nvshmem.core
    import torch.distributed as dist

    try:
        from cuda.core import Device
    except ImportError:
        from cuda.core.experimental import Device

    index = device.index if device.index is not None else 0
    cuda_device = Device(index)
    cuda_device.set_current()
    uid = nvshmem.core.get_unique_id(empty=group.rank != 0)
    payload = torch.from_numpy(uid._data.view(np.uint8).copy())
    dist.broadcast(payload, src=group.ranks[0], group=group._require())
    uid._data[:] = payload.numpy().view(uid._data.dtype)
    nvshmem.core.init(
        device=cuda_device,
        uid=uid,
        rank=group.rank,
        nranks=group.size,
        initializer_method="uid",
    )
    _NVSHMEM_GROUP = group


class MegaMoEBuffer:
    """One worker's MegaMoE symmetric staging and kernel for an expert group.

    Construction is collective over ``group``: NVSHMEM initialization and
    the symmetric allocation run on every rank at the same point of its
    startup. ``max_tokens`` bounds the tokens one rank stages per call;
    ``intermediate`` is each expert's gated width ``I`` and ``activation``
    the experts' gate nonlinearity. The kernel compiles at the first call,
    which must precede any capture on every rank.
    """

    def __init__(
        self,
        group: Communicator,
        *,
        max_tokens: int,
        num_experts: int,
        top_k: int,
        hidden: int,
        intermediate: int,
        activation: str,
        device: torch.device,
    ) -> None:
        _initialize_nvshmem(group, device)
        from uniserve_kernels.megamoe import get_symm_buffer_for_mega_moe

        self.group = group
        self.max_tokens = max_tokens
        self.hidden = hidden
        # The kernel's FC1 width is the concatenated gate and up rows.
        self.symmetric = get_symm_buffer_for_mega_moe(
            num_experts,
            max_tokens,
            top_k,
            hidden,
            2 * intermediate,
            group.rank,
            group.size,
            apply_topk_in_fc1=False,
            in_kernel_fc2_reduce=False,
            combine_dtype="bf16",
            activation=activation,
        )
        # Captured launches bind both a stream and a local token extent;
        # all extents retain the same maximum-sized symmetric allocation.
        self._launches: dict[tuple[int, int, int], object] = {}

    def launch(self, weights, fc1, fc2, tokens: int):
        """Return the launch of one layer's weights on the current stream.

        ``weights`` names the layer; ``fc1`` and ``fc2`` are its kernel-ready
        ``(weight, block scale)`` pairs. ``tokens`` is the positive local
        extent staged for this invocation, bounded by ``max_tokens``.
        """
        from uniserve_kernels.megamoe import nvfp4_mega_launch_thunk

        stream = torch.cuda.current_stream().cuda_stream
        key = (weights, stream, tokens)
        thunk = self._launches.get(key)
        if thunk is None:
            thunk = nvfp4_mega_launch_thunk(
                fc1, fc2, self.symmetric, num_tokens=tokens
            )
            self._launches[key] = thunk
        return thunk


class _MegaMoE(NVFP4Operator):
    """One expert-parallel NVFP4 layer's fused MegaMoE call.

    Hidden states arrive in BF16 or already in the experts' input encoding;
    either way the call returns this rank's tokens' combined ``[T, H]``
    rows (the exchange binding forms their one-route ``Routes``).
    """

    def __init__(self, *, module, size, workspace, buffer: MegaMoEBuffer):
        super().__init__(module=module, size=size, workspace=workspace)
        self.buffer = buffer
        up_gate, down = module.up_gate.weight, module.down.weight
        experts = up_gate.shape[0]
        fields13, fields2 = up_gate.buffers(), down.buffers()
        fp4 = torch.float4_e2m1fn_x2

        # Packed E2M1 bytes [E, rows, K / 2] read through K-major transposed
        # [E, K / 2, rows] views; each expert's 128x4-swizzled scales are a
        # contiguous run of its rows' blocks.
        self._fc1 = (
            fields13["values"].transpose(1, 2).view(fp4),
            fields13["block_scale"]
            .view(torch.float8_e4m3fn)
            .reshape(experts, -1),
        )
        self._fc2 = (
            fields2["values"].transpose(1, 2).view(fp4),
            fields2["block_scale"]
            .view(torch.float8_e4m3fn)
            .reshape(experts, -1),
        )

        # FP32 scale arithmetic on the device, as the CuTeDSL provider:
        # inputs encode with 1 / s13, FC1 dequantizes with tensor_scale13 *
        # s13, the gated product encodes with 1 / s2 and FC2 dequantizes with
        # tensor_scale2 * s2.
        input13 = module.up_gate.input_quantizer.calibrated_scale
        input2 = module.down.input_quantizer.calibrated_scale
        self._fc1_alpha = (fields13["tensor_scale"] * input13).float()
        self._fc2_alpha = (fields2["tensor_scale"] * input2).float()
        self._fc1_norm = torch.full(
            (experts,), 1.0 / input2, dtype=torch.float32, device=up_gate.device
        )

    def __call__(self, hidden, topk_ids, topk_weights, *, combine=True):
        if not combine:
            raise ValueError(
                "MegaMoE returns combined rows; its exchange binding forms "
                "their routes"
            )
        self._validate(hidden, topk_ids, topk_weights)
        if hidden.shape[0] > self.buffer.max_tokens:
            raise ValueError("routed tokens exceed the MegaMoE staging")
        from uniserve_kernels.megamoe import note_staged_tokens

        # Stage this rank's rows in the experts' input encoding, as FlashInfer's
        # MegaMoE backend stages pre-quantized inputs (moe_ep/backends/mega/
        # kernel/sm100/nvfp4_nvfp4_bf16_cutedsl/backend.py:181-210): values and
        # linear block scales and their routes. The launch reads only the
        # staged extent. Rows stored in that encoding stage as they are; BF16
        # rows encode with the operator's fp4_quantize, so both give the bytes
        # the other NVFP4 providers read.
        symmetric = self.buffer.symmetric
        tokens = hidden.shape[0]
        fields = self.encode(hidden).buffers()
        staged_scales = symmetric.x_sf.view(torch.uint8)
        if staged_scales.shape[1] != fields["block_scale"].shape[1]:
            raise ValueError("the staged block scales pad the hidden width")
        symmetric.x.view(torch.uint8)[:tokens].copy_(fields["values"])
        staged_scales[:tokens].copy_(fields["block_scale"])
        symmetric.topk_idx[:tokens].copy_(topk_ids)
        symmetric.topk_weights[:tokens].copy_(topk_weights)
        # The frontend's zero-token call is a no-op, but an idle EP rank
        # must still serve its peers. One invalid row keeps the collective
        # active without contributing a route or a caller-visible output.
        active_tokens = max(1, tokens)
        if not tokens:
            symmetric.topk_idx[:1].fill_(-1)
        note_staged_tokens(symmetric.topk_idx, tokens)
        # The staging's per-expert scales are shared by every layer.
        symmetric.fc1_alpha.copy_(self._fc1_alpha)
        symmetric.fc2_alpha.copy_(self._fc2_alpha)
        symmetric.fc1_norm_const.copy_(self._fc1_norm)
        # A rank with no tokens still launches, serving the tokens the other
        # ranks route to its experts.
        self.buffer.launch(id(self), self._fc1, self._fc2, active_tokens)()
        return symmetric.output_activation[:tokens].clone()

    def close(self) -> None:
        super().close()
        self.buffer = None
        self._fc1 = self._fc2 = None
        self._fc1_alpha = self._fc2_alpha = self._fc1_norm = None


class Backend(NVFP4Backend):
    """Places NVFP4 experts for MegaMoE and prepares its fused calls."""

    name = "megamoe"
    operator_class = _MegaMoE
    up_gate_order = RowOrder.INTERLEAVED_16
    down_order = RowOrder.LINEAR

    def kernels_unsupported(self, device) -> str | None:
        if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
            return "the kernels are built for SM100 and SM103 devices"
        return None

    def unsupported(self, module) -> str | None:
        reason = super().unsupported(module)
        if reason is not None:
            return reason
        if module.expert_group.size < 2:
            return "MegaMoE serves expert-parallel layers"
        if module.intermediate_size % 16:
            return "the gated rows interleave in 16-row blocks"
        return None

    def prepare_fused(self, *, module, size, buffer: MegaMoEBuffer):
        """Place ``module``'s weights and bind its fused call to ``buffer``."""
        from . import _place

        _place(module.up_gate, self.up_gate_order)
        _place(module.down, self.down_order)
        return _MegaMoE(module=module, size=size, workspace={}, buffer=buffer)
