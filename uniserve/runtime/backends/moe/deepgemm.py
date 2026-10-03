"""Split MegaMoE execution over a source/expert union communicator.

The runtime owns symmetric memory and native launch preparation. Models keep
ordinary routed-expert calls; their input quantizers specify the numerical
representation, independently of rank placement and transport.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from uniserve.quantization import QuantizedTensor, RowOrder, ScaleLayout
from uniserve.runtime._peer_storage import allocate_collective_buffer
from uniserve.runtime.microbatches import yield_microbatch

from . import Operator


class Buffer:
    """Collective staging for one microbatch's ordered expert calls.

    Attention ranks precede expert ranks in ``group``. The first preparation
    allocates the encoding-specific symmetric layout on every rank. Calls
    alternate dispatch and combine before reusing that layout. The caller
    drains streams and closes bound operators before retiring the buffer.
    """

    def __init__(
        self,
        group,
        *,
        max_tokens,
        num_experts,
        top_k,
        hidden,
        intermediate,
        activation,
        attention_ranks,
        device,
    ):
        if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
            raise ValueError("split MegaMoE requires a Blackwell CUDA device")
        if not 0 < attention_ranks < group.size:
            raise ValueError(
                "split MegaMoE requires both attention and experts"
            )
        if num_experts % (group.size - attention_ranks):
            raise ValueError("experts must divide the expert ranks")
        if activation != "silu":
            raise ValueError("split MegaMoE requires SiLU experts")
        self.group, self.device = group, device
        self.attention_ranks = attention_ranks
        self.expert_ranks = group.size - attention_ranks
        self.source = group.rank < attention_ranks
        self.max_tokens, self.num_experts = max_tokens, num_experts
        self.top_k, self.hidden_size = top_k, hidden
        # Physical tile padding belongs to the provider, not the model's
        # mathematical dimensions. Padded channels have exactly zero weight.
        self.hidden = (hidden + 511) // 512 * 512
        self.intermediate = (intermediate + 511) // 512 * 512
        self.format = None
        self.storage = self.handle = None
        self.operators = {}
        self.step = None

    def prepare(self, module, size):
        """Bind a numerical layer and compile its launches without execution."""
        quantizer = module.up_gate.input_quantizer
        if quantizer is None or quantizer.format not in {"mxfp8", "nvfp4"}:
            raise ValueError(
                "split MegaMoE requires MXFP8 or NVFP4 activations"
            )
        if module.down.input_quantizer is None or (
            module.down.input_quantizer.format != quantizer.format
        ):
            raise ValueError(
                "both expert projections must use the same encoding"
            )
        if quantizer.format == "nvfp4" and (
            quantizer.calibrated_scale is None
            or module.down.input_quantizer.calibrated_scale is None
        ):
            raise ValueError("NVFP4 experts require calibrated input scales")
        if self.format is None:
            self._allocate(quantizer.format)
        elif self.format != quantizer.format:
            raise ValueError("one exchange must share one activation encoding")
        operator = _SplitOperator(
            module=module, size=size, workspace={}, buffer=self
        )
        self.operators[id(module)] = operator
        return operator

    def bind_microbatches(self, buffers):
        """Prepare one expert launch for all layers and microbatches.

        A persistent grid occupies the expert device. Launching independent
        grids on different streams can make peers choose different lanes and
        wait forever. One launch visits layer-major, then microbatch order on
        every expert rank, matching the attention calls' cooperative order.
        """
        if not 1 <= len(buffers) <= 4:
            raise ValueError("split MegaMoE serves one to four microbatches")
        if any(
            tuple(buffer.operators) != tuple(self.operators)
            or (buffer.group, buffer.format, buffer.max_tokens)
            != (self.group, self.format, self.max_tokens)
            for buffer in buffers
        ):
            raise ValueError("microbatches must share an expert layer sequence")
        if not self.source:
            step = _Step(buffers)
            for buffer in buffers:
                buffer.step = step

    def _allocate(self, format):
        import torch.distributed._symmetric_memory as symmetric
        from uniserve_kernels.deepgemm import load

        self.native = load()
        self.format = format
        alignment = self.native.get_token_alignment_for_mega_moe()
        self.capacity = (
            (self.max_tokens + alignment - 1) // alignment * alignment
        )
        byte_count, _, views = (
            self.native.get_symm_buffer_size_for_mega_moe_m2n(
                self.attention_ranks,
                self.expert_ranks,
                self.num_experts,
                self.capacity,
                self.top_k,
                self.hidden,
                self.intermediate,
                format == "nvfp4",
            )
        )
        self.storage = allocate_collective_buffer(
            (byte_count,), dtype=torch.int8, device=self.device
        )
        self.handle = symmetric.rendezvous(
            self.storage, group=self.group._require()
        )
        self.pointers = list(self.handle.buffer_ptrs)
        self.storage.zero_()
        # The next launch waits for its predecessor's cleanup phase. Seed
        # the initial completed phase before publishing this buffer to peers.
        self.storage[20:24].view(torch.int32).fill_(self.group.size)
        torch.cuda.current_stream(self.device).synchronize()
        dist.all_reduce(
            torch.zeros((), dtype=torch.int32), group=self.group._require()
        )
        self.x, self.scales, self.ids, self.weights, *_ = views(self.storage)

    def attention(
        self, output, hidden, ids, weights, *, phase, compile_only=False
    ):
        self.native.mega_moe_m2n_ag(
            output,
            hidden,
            ids,
            weights,
            None,
            True,
            self.storage,
            self.pointers,
            self.group.rank,
            self.attention_ranks,
            self.expert_ranks,
            self.capacity,
            self.num_experts,
            self.top_k,
            self.intermediate,
            0,
            phase,
            True,
            False,
            self.format == "nvfp4",
            self.top_k,
            self.num_experts,
            0.0,
            1.0,
            None,
            compile_only,
        )

    def expert(self, operator, *, lanes=(), compile_only=False):
        buffers = (*lanes, *((None,) * (3 - len(lanes))))
        self.native.mega_moe_m2n_eg(
            operator.descriptors,
            operator.pointers,
            operator.descriptors.numel() // (4 * 128),
            self.num_experts // self.expert_ranks,
            self.hidden,
            self.intermediate,
            operator.weight_bytes,
            self.format == "mxfp8",
            self.format == "nvfp4",
            None,
            operator.alpha,
            self.storage,
            *(None if buffer is None else buffer.storage for buffer in buffers),
            self.pointers,
            *(
                self.pointers if buffer is None else buffer.pointers
                for buffer in buffers
            ),
            self.group.rank,
            self.attention_ranks,
            self.expert_ranks,
            self.capacity,
            self.num_experts,
            self.top_k,
            self.max_tokens,
            None,
            None,
            None,
            None,
            True,
            0,
            0,
            None,
            compile_only,
        )

    def close(self):
        """Retire collectively after every caller drains its local readers."""
        if self.storage is None:
            return
        # A local stream drain does not retire a peer's remote memory reads.
        # Every owner reaches this host barrier after draining its own work.
        dist.all_reduce(
            torch.zeros((), dtype=torch.int32), group=self.group._require()
        )
        self.x = self.scales = self.ids = self.weights = None
        self.handle = self.storage = None
        self.step = None
        self.operators.clear()


class _Step:
    """Native descriptors for a complete layer/microbatch expert sequence."""

    def __init__(self, buffers):
        self.buffers = tuple(buffers)
        self.operators = tuple(buffers[0].operators.values())
        self.descriptors = torch.stack(
            [op.descriptors for op in self.operators]
        )
        self.pointers = torch.cat([op.pointers for op in self.operators])
        self.weight_bytes = self.operators[0].weight_bytes
        self.alpha = (
            None
            if self.operators[0].alpha is None
            else torch.stack([op.alpha for op in self.operators])
        )
        buffers[0].expert(self, lanes=buffers[1:], compile_only=True)

    def launch(self, operator):
        if operator is self.operators[0]:
            self.buffers[0].expert(self, lanes=self.buffers[1:])


def _weights(linear, order, rows, columns, *, halves=1):
    """Reorder encoded rows and place their scales in MN-major order."""
    weight = linear.weight
    if not isinstance(weight, QuantizedTensor):
        raise ValueError("split MegaMoE requires encoded expert weights")
    if weight.quantizer.format not in {"mxfp8", "nvfp4"}:
        raise ValueError("split MegaMoE requires MXFP8 or NVFP4 expert weights")
    if weight.shape[1:] != (rows, columns):
        original = weight.repack(
            row_order=RowOrder.LINEAR, scale_layout=ScaleLayout.LINEAR
        )
        experts, original_rows, _ = original.shape
        padded = weight.quantizer.empty(
            (experts, rows, columns), dtype=weight.dtype, device=weight.device
        )
        fields = dict(padded.buffers())
        for name, field in original.buffers().items():
            if name == "tensor_scale":
                fields[name] = field
                continue
            source = field.reshape(experts, original_rows, -1)
            target = fields[name].reshape(experts, rows, -1)
            target.zero_()
            # Up and gate occupy separate logical halves. Move each half
            # independently so padding cannot shift their pairing.
            source_rows, target_rows = original_rows // halves, rows // halves
            for half in range(halves):
                target[
                    :,
                    half * target_rows : half * target_rows + source_rows,
                    : source.shape[-1],
                ].copy_(
                    source[:, half * source_rows : (half + 1) * source_rows]
                )
        placed = weight.quantizer.from_tensors(
            fields, shape=(experts, rows, columns), dtype=weight.dtype
        ).repack(row_order=order)
    else:
        placed = weight.repack(row_order=order, scale_layout=ScaleLayout.LINEAR)
    if placed is not weight and placed.shape == weight.shape:
        linear.weight = nn.Parameter(placed, requires_grad=False)
    fields = placed.buffers()
    experts, rows, _ = placed.shape
    nvfp4 = placed.quantizer.format == "nvfp4"
    scales = fields["block_scale" if nvfp4 else "scale"].reshape(
        experts, rows, -1
    )
    words = scales.contiguous().view(torch.int32)
    # UTCCP's 128-row tile loads four groups of 32 scale rows in transposed
    # order. The TMA descriptor advances N contiguously between word columns.
    scales = (
        words.reshape(experts, rows // 128, 4, 32, -1)
        .transpose(2, 3)
        .reshape_as(words)
        .transpose(1, 2)
        .contiguous()
    )
    values = fields["values"]
    return values.view(torch.int8 if nvfp4 else torch.float8_e4m3fn), scales


class _SplitOperator(Operator):
    def __init__(self, *, buffer, **kwargs):
        super().__init__(**kwargs)
        self.buffer = buffer
        self.alpha = None
        module = self.module
        if buffer.source:
            shape = (buffer.max_tokens, buffer.hidden)
            hidden = torch.empty(
                shape, dtype=torch.bfloat16, device=buffer.device
            )
            # Provision the public activation encoder before a peer can enter
            # the persistent expert kernel's device waits.
            module.up_gate.input_quantizer.quantize(
                hidden[:, : buffer.hidden_size].zero_()
            )
            ids = torch.empty(
                (buffer.max_tokens, buffer.top_k),
                dtype=torch.int32,
                device=buffer.device,
            )
            weights = torch.empty_like(ids, dtype=torch.float32)
            for phase in (1, 2):
                buffer.attention(
                    hidden, hidden, ids, weights, phase=phase, compile_only=True
                )
            return

        fc1 = _weights(
            module.up_gate,
            RowOrder.INTERLEAVED_8,
            2 * buffer.intermediate,
            buffer.hidden,
            halves=2,
        )
        fc2 = _weights(
            module.down, RowOrder.LINEAR, buffer.hidden, buffer.intermediate
        )
        for projection in (module.up_gate, module.down):
            if projection.weight.quantizer.format != buffer.format:
                raise ValueError(
                    "expert weights and activations must share an encoding"
                )
        self.operands = (fc1, fc2)
        self.descriptors = buffer.native.make_m2n_weight_descs(
            *fc1,
            *fc2,
            module.up_gate.num_experts,
            buffer.hidden,
            buffer.intermediate,
            16 if buffer.format == "nvfp4" else 32,
        ).to(buffer.device)
        self.pointers = torch.tensor(
            [fc1[0].data_ptr()], dtype=torch.int64, device=buffer.device
        )
        self.weight_bytes = fc1[0].numel() * fc1[0].element_size()
        if buffer.format == "nvfp4":
            first = module.up_gate.weight.buffers()["tensor_scale"].float()
            second = module.down.weight.buffers()["tensor_scale"].float()
            input_scale = module.up_gate.input_quantizer.calibrated_scale
            intermediate_scale = module.down.input_quantizer.calibrated_scale
            experts = module.up_gate.num_experts
            self.alpha = torch.stack(
                (
                    first.reshape(-1).expand(experts) * input_scale,
                    torch.full(
                        (experts,), 1 / intermediate_scale, device=buffer.device
                    ),
                    second.reshape(-1).expand(experts) * intermediate_scale,
                ),
                dim=-1,
            ).contiguous()
        buffer.expert(self, compile_only=True)

    def __call__(self, hidden, topk_ids, topk_weights, *, combine=True):
        self._validate(hidden, topk_ids, topk_weights)
        buffer = self.buffer
        count = hidden.shape[0]
        if count > buffer.max_tokens:
            raise ValueError("routed tokens exceed the split MegaMoE capacity")
        if not buffer.source:
            if buffer.step is None:
                buffer.expert(self)
            else:
                buffer.step.launch(self)
            yield_microbatch()
            return hidden.new_empty((0, buffer.hidden_size))

        if count:
            encoded = (
                hidden
                if isinstance(hidden, QuantizedTensor)
                else self.module.up_gate.input_quantizer.quantize(hidden)
            )
            fields = encoded.buffers()
            values = fields["values"].view(torch.uint8)
            buffer.x[:count, : values.shape[-1]].view(torch.uint8).copy_(values)
            scales = fields[
                "block_scale" if buffer.format == "nvfp4" else "scale"
            ]
            columns = buffer.hidden_size // (
                16 if buffer.format == "nvfp4" else 32
            )
            buffer.scales[:count].view(torch.uint8)[:, :columns].copy_(
                scales.reshape(count, columns)
            )
            buffer.weights[:count].copy_(topk_weights)
        output = torch.empty(
            (count, buffer.hidden), dtype=torch.bfloat16, device=buffer.device
        )
        # external_quant consumes the encoded staging, so hidden's argument
        # only supplies the BF16 shape expected by the native launch contract.
        buffer.attention(output, output, topk_ids, topk_weights, phase=1)
        yield_microbatch()
        buffer.attention(output, output, topk_ids, topk_weights, phase=2)
        # Strip physical channel padding before exposing numerical rows.
        return output[:, : buffer.hidden_size].contiguous()

    def close(self):
        super().close()
        self.buffer = self.operands = self.descriptors = self.pointers = None
        self.alpha = None
