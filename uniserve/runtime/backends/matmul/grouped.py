"""SM100 grouped GEMM over borrowed logical projection matrices."""

from __future__ import annotations

from math import ceil

import torch
import triton
import triton.language as tl

from uniserve.quantization import QuantizedTensor, ScaleLayout
from uniserve.tensors import BufferConfig

from . import Backend as _Backend
from . import MergedOperator as _MergedOperator
from . import _concatenate


def _format(weight):
    return weight.quantizer.format if isinstance(weight, QuantizedTensor) else weight.dtype


def _groups(weights, branch_width):
    """Map channel groups to physical output and padded block-scale intervals.

    Each group is ``(branch, start_row, length, output_offset, scale_offset)``;
    with a branch width, channel groups interleave across branches in
    round-robin order so uneven branches contribute fewer groups.
    """

    widths = tuple(weight.shape[0] for weight in weights.values())
    if branch_width is not None and (
        type(branch_width) is not int
        or branch_width < 1
        or any(width % branch_width for width in widths)
    ):
        raise ValueError("branch width must be positive and divide every branch")

    columns = next(iter(weights.values())).shape[1]
    vector = 16 if _format(next(iter(weights.values()))) == "nvfp4" else 32
    scale_columns = ceil(columns / (4 * vector)) * 4

    groups = []
    output_offset = scale_offset = 0
    count = 1 if branch_width is None else max(widths) // branch_width
    for index in range(count):
        for branch, width in enumerate(widths):
            length = width if branch_width is None else branch_width
            start = index * length
            if start >= width:
                continue
            groups.append((branch, start, length, output_offset, scale_offset))
            # TMA output rows and each group's base address are 16-byte aligned.
            output_offset += ceil(length / 4) * 4
            scale_offset += ceil(length / 128) * 128 * scale_columns

    return tuple(groups), output_offset, scale_offset, scale_columns


def supports(weights, input_quantizer, branch_width):
    """Return whether the grouped SM100 kernel can serve these branch weights."""

    first = next(iter(weights.values()))
    format = _format(first)
    if not (
        first.is_cuda
        and torch.cuda.get_device_capability(first.device) == (10, 0)
        and format in {torch.float16, torch.bfloat16, "fp8", "nvfp4", "mxfp8"}
        and (
            input_quantizer is None
            if isinstance(format, torch.dtype)
            else input_quantizer is not None and input_quantizer.format == format
        )
        and all(
            _format(weight) == format
            and weight.device == first.device
            and weight.shape[1] == first.shape[1]
            and weight.shape[0] > 0
            for weight in weights.values()
        )
    ):
        return False

    values = tuple(
        weight.buffers()["values"] if isinstance(weight, QuantizedTensor) else weight
        for weight in weights.values()
    )

    # TMA's contiguous extent and every row/base address must be 16-byte
    # aligned. Providers retain their ordinary kernels for other layouts.
    if any(
        value.stride(1) != 1
        or value.shape[1] * value.element_size() % 16
        or value.stride(0) * value.element_size() % 16
        or value.data_ptr() % 16
        for value in values
    ) or any((branch_width or weight.shape[0]) % 4 for weight in weights.values()):
        return False

    # Adjacent dense/FP8 branches already form one borrowed matrix. Logical
    # channel groups do not require separate GEMMs when the complete product
    # can be returned as named branch views without moving parameter values.
    if format not in {"nvfp4", "mxfp8"}:
        if _concatenate(values, copy=False) is not None:
            return False
    return True


def workspace_buffers(weights, *, max_rows, branch_width):
    """Describe the context-owned descriptor and scale scratch for one grouping."""

    groups, width, scale_size, columns = _groups(weights, branch_width)
    device = next(iter(weights.values())).device
    format = _format(next(iter(weights.values())))
    block_scaled = format in {"nvfp4", "mxfp8"}
    processors = torch.cuda.get_device_properties(device).multi_processor_count

    # The grouped.initial_* tensors are minimal stand-in operands whose layouts
    # pin the compiled kernel's TMA descriptor ranks; real per-group addresses
    # are published through grouped.pointers at launch time.
    configs = {
        "grouped.output": BufferConfig((max_rows, width), torch.float32),
        "grouped.shapes": BufferConfig((len(groups), 4), torch.int32),
        "grouped.strides": BufferConfig((len(groups), 3, 2), torch.int32),
        "grouped.pointers": BufferConfig((len(groups), 3), torch.int64),
        "grouped.maps": BufferConfig((processors, 5 if block_scaled else 3, 16), torch.int64),
        "grouped.initial_a": BufferConfig(
            (32, 32, 1), format if isinstance(format, torch.dtype) else torch.uint8
        ),
        "grouped.initial_b": BufferConfig(
            (32, 32, 1), format if isinstance(format, torch.dtype) else torch.uint8
        ),
        "grouped.initial_c": BufferConfig((32, 32, 1), torch.float32),
    }
    if block_scaled:
        configs.update(
            {
                "grouped.input_scale": BufferConfig(
                    (ceil(max_rows / 128) * 128 * columns,), torch.uint8
                ),
                "grouped.weight_scale": BufferConfig((scale_size,), torch.uint8),
                "grouped.scale_pointers": BufferConfig((len(groups), 2), torch.int64),
                "grouped.initial_sfa": BufferConfig((32, 32, 1), torch.uint8),
                "grouped.initial_sfb": BufferConfig((32, 32, 1), torch.uint8),
            }
        )
    return configs


@triton.jit
def _group_index(head, branch: tl.constexpr, parts: tl.constexpr):
    """Number preceding channel groups, including uneven branch head counts."""

    group = tl.full((), 0, tl.int32)
    for index in tl.static_range(len(parts)):
        group += tl.minimum(head, parts[index][1])
        if index < branch:
            group += (head < parts[index][1]).to(tl.int32)
    return group


@triton.jit
def _pack_scale(
    source,
    output,
    rows: tl.constexpr,
    columns: tl.constexpr,
    source_stride: tl.constexpr,
    swizzled: tl.constexpr,
    branch: tl.constexpr,
    parts: tl.constexpr,
    group_stride: tl.constexpr,
    block: tl.constexpr,
):
    """Copy scale bytes into a group-local 128x4 atom without requantizing.

    A 128x4 scale atom is 512 consecutive bytes. ``offset`` decodes to an
    ``(atom, row, column)`` triple inside the [rows, padded_columns] scale
    grid; ``swizzled`` selects the accelerator's atom-internal byte order over
    the plain row-major source layout.
    """

    padded_columns: tl.constexpr = triton.cdiv(columns, 4) * 4
    size: tl.constexpr = triton.cdiv(rows, 128) * 128 * padded_columns
    head = tl.program_id(1)
    group = _group_index(head, branch, parts)
    output += parts[branch][3] + group * group_stride
    offset = tl.program_id(0) * block + tl.arange(0, block)
    atom = offset // 512
    row = atom // (padded_columns // 4) * 128 + offset % 512 // 16 + offset % 16 // 4 * 32
    column = atom % (padded_columns // 4) * 4 + offset % 4
    source_row = row + head * rows
    if swizzled:
        address = (
            (source_row // 128 * (padded_columns // 4) + column // 4) * 512
            + source_row % 32 * 16
            + source_row % 128 // 32 * 4
            + column % 4
        )
    else:
        address = source_row * source_stride + column
    value = tl.load(
        source + address, mask=(offset < size) & (row < rows) & (column < columns), other=0
    )
    tl.store(output + offset, value, mask=offset < size)


@triton.jit
def _descriptors(
    x,
    weights,
    output,
    input_scale,
    weight_scale,
    shapes,
    strides,
    pointers,
    scale_pointers,
    rows: tl.constexpr,
    columns: tl.constexpr,
    input_stride: tl.constexpr,
    weight_strides: tl.constexpr,
    output_stride: tl.constexpr,
    item_bits: tl.constexpr,
    parts: tl.constexpr,
    output_group_stride: tl.constexpr,
    scale_group_stride: tl.constexpr,
):
    """Publish live addresses on the execution stream, including during replay.

    Per group this writes one shapes record [rows, length, columns, 1], three
    (row, column) stride pairs for input/weight/output, and the base pointers
    of the input, this group's weight rows, and its output interval. Packed
    4-bit formats express strides in elements, hence the 8 // item_bits scale.
    """

    head = tl.program_id(0)
    for branch in tl.static_range(len(parts)):
        if tl.program_id(1) == branch and head < parts[branch][1]:
            group = _group_index(head, branch, parts)
            length: tl.constexpr = parts[branch][0]
            start = head * length
            output_offset = parts[branch][2] + group * output_group_stride
            scale_offset = parts[branch][3] + group * scale_group_stride

            tl.store(shapes + group * 4, rows)
            tl.store(shapes + group * 4 + 1, length)
            tl.store(shapes + group * 4 + 2, columns)
            tl.store(shapes + group * 4 + 3, 1)

            tl.store(strides + group * 6, input_stride * 8 // item_bits)
            tl.store(strides + group * 6 + 1, 1)
            tl.store(strides + group * 6 + 2, weight_strides[branch] * 8 // item_bits)
            tl.store(strides + group * 6 + 3, 1)
            tl.store(strides + group * 6 + 4, output_stride)
            tl.store(strides + group * 6 + 5, 1)

            tl.store(pointers + group * 3, x.to(tl.int64))
            tl.store(
                pointers + group * 3 + 1,
                (weights[branch] + start * weight_strides[branch]).to(tl.int64),
            )
            tl.store(pointers + group * 3 + 2, (output + output_offset).to(tl.int64))
            if scale_pointers is not None:
                tl.store(scale_pointers + group * 2, input_scale.to(tl.int64))
                tl.store(scale_pointers + group * 2 + 1, (weight_scale + scale_offset).to(tl.int64))


@triton.jit
def _output(
    source,
    target,
    input_scale,
    weight_scale,
    bias,
    rows: tl.constexpr,
    width: tl.constexpr,
    source_stride: tl.constexpr,
    target_row_stride: tl.constexpr,
    target_column_stride: tl.constexpr,
    input_scale_stride: tl.constexpr,
    weight_scale_stride: tl.constexpr,
    encoded: tl.constexpr,
    branch: tl.constexpr,
    parts: tl.constexpr,
    group_stride: tl.constexpr,
    block: tl.constexpr,
):
    """Scatter one branch's FP32 group results into its typed output with bias."""

    offset = tl.program_id(0) * block + tl.arange(0, block)
    row, column = offset // width, offset % width
    head = tl.program_id(1)
    group = _group_index(head, branch, parts)
    source += parts[branch][2] + group * group_stride
    value = tl.load(source + row * source_stride + column, mask=row < rows, other=0)
    column += head * width
    if input_scale is not None:
        scale_a = tl.load(input_scale + row * input_scale_stride, mask=row < rows, other=0)
        scale_b = tl.load(weight_scale + column * weight_scale_stride, mask=row < rows, other=0)
        value = value * (scale_a * scale_b)
    # Encoded GEMM rounds before adding bias, matching the standalone operator.
    if encoded:
        value = value.to(target.dtype.element_ty)
    if bias is not None:
        value = value + tl.load(bias + column, mask=row < rows, other=0).to(value.dtype)
    tl.store(
        target + row * target_row_stride + column * target_column_stride, value, mask=row < rows
    )


_compiled = {}


class GroupedOperator(_MergedOperator):
    """Borrow branch parameters and context-owned native descriptor workspace."""

    def __init__(self, operators, *, branch_width, workspace):
        self.operators = operators
        self.workspace = workspace
        self.weights = {name: operator.weight for name, operator in operators.items()}
        self.first = next(iter(operators.values()))
        self.groups, self.width, _, self.scale_columns = _groups(self.weights, branch_width)

        # parts[branch] is (group length, group count, output base, scale base)
        # as consumed by the Triton helper kernels above.
        if branch_width is None:
            self.parts = tuple(
                (length, 1, output, scale) for _, _, length, output, scale in self.groups
            )
            self.output_group_stride = self.scale_group_stride = 0
        else:
            self.parts = tuple(
                (branch_width, weight.shape[0] // branch_width, 0, 0)
                for weight in self.weights.values()
            )
            self.output_group_stride = ceil(branch_width / 4) * 4
            self.scale_group_stride = ceil(branch_width / 128) * 128 * self.scale_columns

        self.format = _format(self.first.weight)
        self.block_scaled = self.format in {"nvfp4", "mxfp8"}
        self.vector = 16 if self.format == "nvfp4" else 32
        self._initial = self._metadata = self._maps = self._kernel = None

    def _prepare_kernel(self):
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from flashinfer.data.cutlass.examples.python.CuTeDSL.blackwell.grouped_blockscaled_gemm import (
            Sm100GroupedBlockScaledGemmKernel,
        )
        from flashinfer.data.cutlass.examples.python.CuTeDSL.blackwell.grouped_gemm import (
            GroupedGemmKernel,
        )

        encoded = {
            torch.float16: cutlass.Float16,
            torch.bfloat16: cutlass.BFloat16,
            "fp8": cutlass.Float8E4M3FN,
            "nvfp4": cutlass.Float4E2M1FN,
            "mxfp8": cutlass.Float8E4M3FN,
        }[self.format]

        names = ("a", "b", "c")
        types = (encoded, encoded, cutlass.Float32)
        fields = ("shapes", "strides", "pointers")
        if self.block_scaled:
            scale = cutlass.Float8E4M3FN if self.format == "nvfp4" else cutlass.Float8E8M0FNU
            names += ("sfa", "sfb")
            types += (scale, scale)
            fields += ("scale_pointers",)

        initial = []
        for name, dtype in zip(names, types, strict=True):
            tensor = from_dlpack(self.workspace[f"grouped.initial_{name}"], assumed_align=16)
            tensor.element_type = dtype
            # Keeping these dimensions dynamic preserves the TMA rank when a
            # real group's scale atoms exceed the small initial descriptor.
            initial.append(tensor.mark_layout_dynamic(leading_dim=1))

        metadata = tuple(
            from_dlpack(self.workspace[f"grouped.{name}"], assumed_align=16) for name in fields
        )
        maps = from_dlpack(self.workspace["grouped.maps"], assumed_align=16)
        active = self.workspace["grouped.maps"].shape[0]
        stream = cuda.CUstream(torch.cuda.current_stream(self.first.weight.device).cuda_stream)

        key = (self.format, len(self.groups), active)
        if key not in _compiled:
            kernel = (
                Sm100GroupedBlockScaledGemmKernel(self.vector, (128, 128), (1, 1))
                if self.block_scaled
                else GroupedGemmKernel(cutlass.Float32, False, (128, 128), (1, 1))
            )
            _compiled[key] = cute.compile(
                kernel,
                *initial,
                len(self.groups),
                *metadata,
                active,
                maps,
                active,
                stream,
                options="--opt-level 2",
            )
        self._initial, self._metadata, self._maps = tuple(initial), metadata, maps
        self._kernel = _compiled[key]

    def __call__(self, x, biases, *, out):
        import cuda.bindings.driver as cuda

        if set(out) != set(self.operators) or set(biases) != set(self.operators):
            raise ValueError("merged projection branches must agree")
        for name, operator in self.operators.items():
            if (
                out[name].shape != (x.shape[0], operator.weight.shape[0])
                or out[name].device != x.device
                or out[name].dtype != operator.output_dtype
            ):
                raise ValueError("merged output disagrees with its branch representation")

        x = self.first._input(x, next(iter(out.values())))
        if _format(x) != self.format:
            raise ValueError("grouped GEMM requires matching operand formats")
        rows, columns = x.shape
        if rows == 0:
            return out

        if self._kernel is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("grouped GEMM must be warmed before capture")
            self._prepare_kernel()

        left = x.buffers() if isinstance(x, QuantizedTensor) else {"values": x}
        input_values = left["values"]
        if (
            input_values.stride(1) != 1
            or input_values.stride(0) * input_values.element_size() % 16
            or input_values.data_ptr() % 16
        ):
            # A contiguous view can still begin at an unaligned offset; clone
            # gives both such views and strided views an aligned allocation.
            left = {**left, "values": input_values.clone(memory_format=torch.contiguous_format)}
        field = "block_scale" if self.format == "nvfp4" else "scale"
        input_scale = left[field].view(torch.uint8) if self.block_scaled else None
        if self.block_scaled and x.scale_layout is ScaleLayout.LINEAR:
            destination = self.workspace["grouped.input_scale"]
            size = ceil(rows / 128) * 128 * self.scale_columns
            _pack_scale[(triton.cdiv(size, 512), 1)](
                input_scale,
                destination,
                rows,
                columns // self.vector,
                input_scale.stride(0),
                False,
                0,
                ((rows, 1, 0, 0),),
                0,
                512,
            )
            input_scale = destination

        values = tuple(
            (weight.buffers()["values"] if isinstance(weight, QuantizedTensor) else weight).view(
                torch.uint8
            )
            for weight in self.weights.values()
        )
        weights = tuple(self.weights.values())
        for branch, (length, count, _, _) in enumerate(self.parts if self.block_scaled else ()):
            source = weights[branch].buffers()[field].view(torch.uint8)
            size = ceil(length / 128) * 128 * self.scale_columns
            _pack_scale[(triton.cdiv(size, 512), count)](
                source,
                self.workspace["grouped.weight_scale"],
                length,
                columns // self.vector,
                source.stride(0),
                weights[branch].scale_layout is ScaleLayout.SWIZZLED_128X4,
                branch,
                self.parts,
                self.scale_group_stride,
                512,
            )

        result = self.workspace["grouped.output"]
        input_values = left["values"].view(torch.uint8)
        _descriptors[(max(part[1] for part in self.parts), len(self.parts))](
            input_values,
            values,
            result,
            input_scale,
            self.workspace.get("grouped.weight_scale"),
            self.workspace["grouped.shapes"],
            self.workspace["grouped.strides"],
            self.workspace["grouped.pointers"],
            self.workspace.get("grouped.scale_pointers"),
            rows,
            columns,
            input_values.stride(0),
            tuple(value.stride(0) for value in values),
            self.width,
            4 if self.format == "nvfp4" else left["values"].element_size() * 8,
            self.parts,
            self.output_group_stride,
            self.scale_group_stride,
            num_warps=1,
        )
        stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        self._kernel(*self._initial, *self._metadata, self._maps, stream)

        names = tuple(self.weights)
        for branch, (length, count, _, _) in enumerate(self.parts):
            name = names[branch]
            destination = out[name]
            bias = biases[name]
            scale_a = scale_b = None
            stride_a = stride_b = 0
            if self.format == "nvfp4":
                scale_a = left["tensor_scale"]
                scale_b = weights[branch].buffers()["tensor_scale"]
            elif self.format == "fp8":
                scale_a = left["scale"]
                scale_b = weights[branch].buffers()["scale"]
                if x.quantizer.axis == 0:
                    stride_a = scale_a.stride(0)
                if weights[branch].quantizer.axis == 0:
                    stride_b = scale_b.stride(0)
            _output[(triton.cdiv(rows * length, 256), count)](
                result,
                destination,
                scale_a,
                scale_b,
                bias,
                rows,
                length,
                self.width,
                destination.stride(0),
                destination.stride(1),
                stride_a,
                stride_b,
                isinstance(x, QuantizedTensor),
                branch,
                self.parts,
                self.output_group_stride,
                256,
            )
        return out


class Backend(_Backend):
    """CUDA providers share pointer-grouped preparation and workspace ownership."""

    def merged_workspace_buffers(
        self, weights, *, input_dtype, input_quantizer, max_rows, branch_width, output_dtype
    ):
        if not supports(weights, input_quantizer, branch_width):
            return super().merged_workspace_buffers(
                weights,
                input_dtype=input_dtype,
                input_quantizer=input_quantizer,
                max_rows=max_rows,
                branch_width=branch_width,
                output_dtype=output_dtype,
            )
        # Every group borrows the same encoded input. Only one branch needs
        # input storage; group-specific descriptors and output are context-owned.
        return {
            **self.workspace_buffers(
                next(iter(weights.values())),
                input_dtype=input_dtype,
                input_quantizer=input_quantizer,
                max_rows=max_rows,
                output_dtype=output_dtype,
            ),
            **workspace_buffers(weights, max_rows=max_rows, branch_width=branch_width),
        }

    def prepare_merged(
        self,
        weights,
        *,
        input_dtype,
        input_quantizer,
        max_rows,
        branch_width,
        output_dtype,
        workspace,
    ):
        if not supports(weights, input_quantizer, branch_width):
            return super().prepare_merged(
                weights,
                input_dtype=input_dtype,
                input_quantizer=input_quantizer,
                max_rows=max_rows,
                branch_width=branch_width,
                output_dtype=output_dtype,
                workspace=workspace,
            )
        operators = {
            name: self.prepare(
                weight,
                input_dtype=input_dtype,
                input_quantizer=input_quantizer,
                max_rows=max_rows,
                output_dtype=output_dtype,
                workspace=workspace,
            )
            for name, weight in weights.items()
        }

        return GroupedOperator(operators, branch_width=branch_width, workspace=workspace)
