"""Prepared routed-expert operators borrowing stacked expert weights.

A provider evaluates one ``FusedMoE`` call site: routed tokens of up to the
prepared ``TextSize`` capacity through that module's resident expert
weights. Operators borrow the module's parameters and context-owned
workspace; host planning happens only in ``prepare`` so calls can be
captured in CUDA graphs. A provider whose kernel reads expert weights in
another physical row order places them in that order when it is prepared;
the logical weights stay unchanged and only one copy stays resident
(``NVFP4Backend``).
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module

# The portable provider is the submodule ``torch``; importing it binds that
# name on this package, so the library keeps a distinct name here.
import torch as torch_lib
from torch import nn

from uniserve.model.inputs import TextSize
from uniserve.quantization import QuantizedTensor, RowOrder, ScaleLayout
from uniserve.tensors import BufferConfig

# Native providers in automatic selection order. Their representations are
# disjoint: CuTeDSL serves NVFP4 experts, CUTLASS BF16 and FP16 experts.
# trtllm-gen also serves NVFP4 experts, by explicit selection. CuTeDSL leads
# for NVFP4 because routed-expert layers of 256 tokens or more, the calls
# that prefill chunks and 256-token canvases make, take 0.78-0.93 of
# trtllm-gen's median time on SM100 (trtllm-gen is anomalously slow at 4096
# tokens), while 1 to 64 tokens take 1.17-1.51 of it
# (artifacts/diffusion_gemma/stage0/moe/measurements-cutedsl-separate/).
_NATIVE = ("cutedsl", "cutlass")
_PROVIDERS = frozenset((*_NATIVE, "trtllm", "torch"))


class Operator:
    """One ``FusedMoE`` call site's prepared kernel and borrowed resources."""

    def __init__(self, *, module, size: TextSize, workspace):
        self.module = module
        self.size = size
        self.workspace = workspace
        self._closed = False

    def _validate(self, hidden, topk_ids, topk_weights):
        if self._closed:
            raise RuntimeError("expert operator is closed")
        if hidden.shape[0] > self.size.num_tokens:
            raise ValueError("routed tokens exceed the prepared token capacity")
        # Hidden states may arrive already in the encoding the experts read
        # them in (see ``FusedMoE``), with the linear block scales every
        # provider gathers token rows from.
        if isinstance(hidden, QuantizedTensor) and (
            hidden.quantizer != self.module.up_gate.input_quantizer
            or hidden.scale_layout is not ScaleLayout.LINEAR
        ):
            raise ValueError(
                "encoded hidden states must use the experts' input encoding "
                "with linear block scales"
            )

    def __call__(
        self,
        hidden: torch_lib.Tensor,
        topk_ids: torch_lib.Tensor,
        topk_weights: torch_lib.Tensor,
    ) -> torch_lib.Tensor:
        raise NotImplementedError

    def tactic_space(self) -> list[list]:
        """Independent tactic dimensions, each listing its default first.

        A provider whose calls take a measured tactic (see ``tactics``)
        describes its choices here; the others have none.
        """
        return []

    def close(self) -> None:
        self._closed = True
        self.module = None
        self.workspace = {}


class NVFP4Operator(Operator):
    """An operator whose kernels read NVFP4 hidden states.

    The kernels gather token rows of ``[T, H / 2]`` packed E2M1 values and
    linear ``[T, H / 16]`` E4M3 block scales encoded with the static
    calibrated scale of ``up_gate.input_quantizer``. Hidden states already
    in that encoding (as ``sandwich_rms_norm`` stores a normalization the
    experts read) are read as stored; BF16 hidden states are encoded with
    FlashInfer's ``fp4_quantize``, which stores the same bytes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from flashinfer import fp4_quantize

        self._quantize = fp4_quantize
        # The encoder's global scale is 1 / s for the static scale s.
        self._input_scale: torch_lib.Tensor | None = torch_lib.tensor(
            [self.module.up_gate.input_quantizer.calibrated_scale],
            dtype=torch_lib.float32,
            device=self.module.up_gate.weight.device,
        ).reciprocal()

    def _encoded(self, hidden):
        """Return the E2M1 values and E4M3 block scales of ``hidden``."""
        tokens, width = hidden.shape
        if isinstance(hidden, QuantizedTensor):
            fields = hidden.buffers()
            values, scales = fields["values"], fields["block_scale"]
        else:
            values, scales = self._quantize(
                hidden.contiguous(),
                global_scale=self._input_scale,
                is_sf_swizzled_layout=False,
            )
        return values, scales.view(torch_lib.float8_e4m3fn).reshape(
            tokens, width // 16
        )

    def close(self) -> None:
        super().close()
        self._input_scale = None


class Backend:
    """Factory for routed-expert operators and their workspace declarations."""

    name: str
    operator_class: type[Operator]

    def unsupported(self, module) -> str | None:
        """Return why this provider cannot evaluate ``module``, or ``None``."""
        return None

    def invalid_expert(self, module) -> int:
        """The routed expert id this provider's kernels skip.

        An expert-parallel exchange rewrites the ids of experts another rank
        holds, and of unused receive rows, to it. The grouped kernels here
        skip global ids outside the module's resident experts, so the
        global expert count serves.
        """
        return module.num_experts

    def workspace_buffers(
        self, *, module, size: TextSize
    ) -> Mapping[str, BufferConfig]:
        return {}

    def prepare(self, *, module, size: TextSize, workspace) -> Operator:
        return self.operator_class(
            module=module, size=size, workspace=workspace
        )


def _place(linear, order: RowOrder) -> None:
    """Store an ``ExpertLinear``'s stacked NVFP4 weight in ``order``.

    Block scales take the 128x4 swizzled layout the grouped kernels read.
    The rearranged encoding replaces the parameter, so one copy stays
    resident; the transient copy is one projection of one layer. The work
    completes before returning: later calls may run on another stream, and
    the released encoding must have no pending reads.
    """
    weight = linear.weight
    if (
        weight.row_order is order
        and weight.scale_layout is ScaleLayout.SWIZZLED_128X4
    ):
        return
    if torch_lib.cuda.is_current_stream_capturing():
        raise RuntimeError("expert weights must be placed before capture")
    placed = weight.repack(
        scale_layout=ScaleLayout.SWIZZLED_128X4, row_order=order
    )
    torch_lib.cuda.current_stream(weight.device).synchronize()
    linear.weight = nn.Parameter(placed, requires_grad=False)


class NVFP4Backend(Backend):
    """Grouped W4A4 NVFP4 kernels over expert weights in a physical order.

    The kernels read ``[E, rows, K]`` NVFP4 weights with one FP32 tensor
    scale per expert and 128x4-swizzled E4M3 block scales, and encode their
    activations with the static calibrated scales of ``up_gate`` and
    ``down``'s input quantizers. A subclass names the physical row orders
    its kernels read (``up_gate_order``, ``down_order``) and the devices
    they run on (``kernels_unsupported``). Preparation places loaded
    linear-order weights in those orders once; weights another provider
    placed in a different order are rejected, because a prepared operator
    of that provider may still borrow them.
    """

    up_gate_order: RowOrder
    down_order: RowOrder

    def kernels_unsupported(self, device: torch_lib.device) -> str | None:
        """Return why the kernels cannot run on CUDA ``device``, or ``None``.

        Covers the device architecture and the installed kernel library.
        """
        return None

    def unsupported(self, module) -> str | None:
        up_gate, down = module.up_gate.weight, module.down.weight
        if up_gate.device.type != "cuda":
            return "the kernels run on CUDA devices"
        reason = self.kernels_unsupported(up_gate.device)
        if reason is not None:
            return reason
        if not all(
            isinstance(weight, QuantizedTensor)
            and weight.quantizer.format == "nvfp4"
            for weight in (up_gate, down)
        ):
            return "only NVFP4 expert weights are served"
        if up_gate.dtype != torch_lib.bfloat16:
            return "NVFP4 experts compute in BF16"
        if any(
            quantizer is None
            or quantizer.format != "nvfp4"
            or quantizer.calibrated_scale is None
            for quantizer in (
                module.up_gate.input_quantizer,
                module.down.input_quantizer,
            )
        ):
            return "W4A4 activations require static calibrated NVFP4 scales"
        if any(
            weight.buffers()["tensor_scale"].shape != (up_gate.shape[0],)
            for weight in (up_gate, down)
        ):
            return "each expert requires its own tensor scale"

        # Every expert's scales must fill whole 128x4 swizzle tiles, so the
        # stacked swizzle equals the per-expert swizzle the kernels index:
        # rows (H and 2I) multiples of 128, and K / 16 (for K = H and I)
        # multiples of four.
        hidden, intermediate = module.hidden_size, down.shape[2]
        if hidden % 128:
            return f"hidden width {hidden} must be a multiple of 128"
        if intermediate % 64:
            return f"intermediate width {intermediate} must be a multiple of 64"
        if module.group.size > 1:
            return "tensor-parallel expert shards are not validated"

        for name, order in (
            ("up_gate", self.up_gate_order),
            ("down", self.down_order),
        ):
            stored = getattr(module, name).weight.row_order
            if stored is not RowOrder.LINEAR and stored is not order:
                return (
                    f"{name} weights are stored in the {stored.value} row "
                    f"order another expert provider placed; these kernels "
                    f"read {order.value}"
                )
        return None

    def prepare(self, *, module, size: TextSize, workspace) -> Operator:
        _place(module.up_gate, self.up_gate_order)
        _place(module.down, self.down_order)
        return super().prepare(module=module, size=size, workspace=workspace)


def resolve(
    backend: str | Backend, *, module, device: torch_lib.device
) -> Backend:
    """Resolve a provider name for ``module`` on ``device``.

    ``auto`` selects a native grouped-expert kernel on a GPU and the
    portable implementation on the CPU. A GPU representation that no native
    kernel covers raises here, at preparation, naming each native provider's
    reason: the portable loop is a reference, not a GPU serving path.
    """
    if isinstance(backend, Backend):
        return backend
    if backend == "auto":
        if device.type != "cuda":
            backend = "torch"
        else:
            reasons = []
            for name in _NATIVE:
                provider = import_module(f"{__name__}.{name}").Backend()
                reason = provider.unsupported(module)
                if reason is None:
                    return provider
                reasons.append(f"{name}: {reason}")
            raise ValueError(
                "no native expert kernel covers this FusedMoE representation "
                f"on {device} ({'; '.join(reasons)})"
            )
    if backend not in _PROVIDERS:
        raise ValueError(f"unknown expert backend {backend!r}")
    provider = import_module(f"{__name__}.{backend}").Backend()
    reason = provider.unsupported(module)
    if reason is not None:
        raise ValueError(
            f"expert backend {backend!r} does not support this "
            f"representation: {reason}"
        )
    return provider


def evaluate(module, hidden, topk_ids, topk_weights):
    """Run one standalone call through a provider prepared just for it.

    The call owns its workspace for its duration; execution contexts bind a
    reusable operator instead.
    """
    from uniserve.runtime.tensor_buffers import TensorBuffers

    provider = resolve("auto", module=module, device=hidden.device)
    size = TextSize(hidden.shape[0], 1)
    requirements = provider.workspace_buffers(module=module, size=size)
    with TensorBuffers.allocate(requirements, device=hidden.device) as buffers:
        operator = provider.prepare(
            module=module, size=size, workspace=buffers.view(requirements)
        )
        try:
            return operator(hidden, topk_ids, topk_weights)
        finally:
            operator.close()
