"""Numerical weight representation, explicit assignments and load results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import torch
from torch import nn

from uniserve._slices import intersection, subtract, within
from uniserve._slices import shape as region_shape
from uniserve.nn.linear import (
    Linear,
    MergedColumnParallelLinear,
    VocabParallelEmbedding,
    VocabParallelHead,
    _coalesce,
)
from uniserve.nn.moe import ExpertLinear, FusedMoE
from uniserve.quantization import (
    QuantizationConfig,
    QuantizedTensor,
    Quantizer,
    ScaleLayout,
)

from . import checkpoint


@dataclass(frozen=True, slots=True)
class Config:
    """Choose parameter dtype and quantization by longest module-path prefix."""

    dtype: torch.dtype = torch.bfloat16
    dtypes: Mapping[str, torch.dtype] = field(default_factory=dict)
    quantization: Mapping[str, QuantizationConfig | None] = field(
        default_factory=dict
    )

    def __post_init__(self):
        object.__setattr__(self, "dtypes", MappingProxyType(dict(self.dtypes)))
        object.__setattr__(
            self, "quantization", MappingProxyType(dict(self.quantization))
        )
        if not self.dtype.is_floating_point or any(
            not dtype.is_floating_point for dtype in self.dtypes.values()
        ):
            raise ValueError("weight compute dtypes must be floating point")
        if any(
            value is not None and not isinstance(value, QuantizationConfig)
            for value in self.quantization.values()
        ):
            raise TypeError(
                "weight quantization requires QuantizationConfig values or None"
            )


@dataclass(frozen=True, slots=True)
class Assignment:
    """Copy one logical source rectangle into one Parameter rectangle.

    Source slices select mathematical checkpoint branches. The loader applies
    a target layer's channel partition when a complete logical matrix is given.
    Target slices address resident storage. preserve_dtype keeps source dtype.
    A target rectangle may carry extra leading axes of extent one, such as
    one expert of a stacked ``[E, rows, columns]`` parameter receiving that
    expert's ``[rows, columns]`` checkpoint matrix.
    """

    target: nn.Parameter
    source: checkpoint.Weight
    source_slice: tuple[slice, ...] | None = None
    target_slice: tuple[slice, ...] | None = None
    preserve_dtype: bool = False


@dataclass(frozen=True, slots=True)
class ModuleMapping:
    """Map one checkpoint source onto a resident numerical module.

    required/optional name module parameters. nonresident names intentionally
    unused checkpoint tensors, including entries belonging to absent PP layers.
    post_load may compute derived constants while the source reader is open.
    """

    module: nn.Module
    source: str
    map_weights: Callable[[checkpoint.Reader], tuple[Assignment, ...]]
    required: frozenset[str]
    optional: frozenset[str] = frozenset()
    nonresident: frozenset[str] = frozenset()
    post_load: Callable[[checkpoint.Reader], None] | None = None


@dataclass(frozen=True, slots=True)
class Report:
    """Per-mapping load outcome.

    Resident parameters and leftover source tensors. loaded names module
    parameters fully or partly assigned; skipped names intentionally unused
    nonresident sources; missing names required parameters with no
    assignment; unexpected names source tensors no assignment or derived
    constant consumed; incomplete maps a loaded parameter to the target
    regions no assignment covered.
    """

    loaded: frozenset[str]
    skipped: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    incomplete: Mapping[str, tuple[str, ...]]

    def __post_init__(self):
        object.__setattr__(
            self, "incomplete", MappingProxyType(dict(self.incomplete))
        )


def _choice(path, mapping, default):
    candidates = (
        (len(prefix), value)
        for prefix, value in mapping.items()
        if not prefix or path == prefix or path.startswith(prefix + ".")
    )
    return max(candidates, key=lambda pair: pair[0], default=(-1, default))[1]


def _full(shape):
    return tuple(slice(0, size) for size in shape)


def _placed(device) -> torch.device:
    """Name the device a tensor moved to ``device`` reports.

    An unindexed CUDA device means the current one, which moved tensors
    report with its index.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _source_names(weight):
    if isinstance(weight, checkpoint.FP8Weight):
        return _source_names(weight.values) | _source_names(weight.scale)
    if isinstance(weight, checkpoint.NVFP4Weight):
        return {
            weight.name,
            *_source_names(weight.values),
            *_source_names(weight.block_scale),
            *_source_names(weight.tensor_scale),
            *(
                ()
                if weight.activation_scale is None
                else _source_names(weight.activation_scale)
            ),
        }
    return {weight.name}


def _source_identity(weight):
    if isinstance(weight, checkpoint.FP8Weight):
        return (
            type(weight),
            _source_identity(weight.values),
            _source_identity(weight.scale),
            weight.axis,
            weight.dtype,
        )
    if isinstance(weight, checkpoint.NVFP4Weight):
        return (
            type(weight),
            _source_identity(weight.values),
            _source_identity(weight.block_scale),
            _source_identity(weight.tensor_scale),
            None
            if weight.activation_scale is None
            else _source_identity(weight.activation_scale),
            weight.dtype,
        )
    return id(weight)


def _fp8_fragments(shape, fragments, *, device, dtype):
    """Assemble source encodings.

    Without deriving new scales from local values.
    """
    if not shape:
        raise ValueError(
            "a scalar FP8 parameter cannot have multiple disjoint assignments"
        )

    values = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.ones(
        (shape[0], *((1,) * (len(shape) - 1))),
        dtype=torch.float32,
        device=device,
    )
    initialized = torch.zeros(shape[0], dtype=torch.bool, device=device)
    per_row = any(fragment.quantizer.axis == 0 for _, fragment in fragments)

    for target, fragment in fragments:
        fields = fragment.buffers()
        rows = target[0]
        incoming = (
            fields["scale"]
            .to(device)
            .expand(rows.stop - rows.start, *((1,) * (len(shape) - 1)))
        )
        selected = initialized[rows]
        if selected.any() and not torch.equal(
            scales[rows][selected], incoming[selected]
        ):
            raise ValueError(
                "FP8 source fragments disagree within the same scale domain"
            )
        values[target].copy_(fields["values"].to(device))
        scales[rows].copy_(incoming)
        initialized[rows] = True

    # A uniform scale across all rows collapses back to a per-tensor domain.
    if (
        not per_row
        and shape[0]
        and torch.equal(scales, scales[:1].expand_as(scales))
    ):
        quantizer = Quantizer("fp8")
        scales = scales[0].reshape(())
    else:
        quantizer = Quantizer("fp8", axis=0)
    return quantizer.from_tensors(
        {"values": values, "scale": scales}, shape=tuple(shape), dtype=dtype
    )


def expert_assignments(
    module: FusedMoE,
    *,
    up: tuple[checkpoint.Weight, int],
    gate: tuple[checkpoint.Weight, int],
    down: checkpoint.Weight,
    expert: int | None = None,
) -> tuple[Assignment, ...]:
    """Assign complete checkpoint expert projections to a ``FusedMoE``.

    ``up`` and ``gate`` name each ``[I, H]`` projection's checkpoint tensor
    and the row where it starts inside that tensor, so a fused ``gate_up``
    tensor serves both (gate at row 0, up at row ``I`` in the Transformers
    layout). ``down`` is the ``[H, I]`` projection. With ``expert=None`` the
    tensors are stacked ``[E, ...]`` over all experts; otherwise they are one
    expert's matrices. The module's tensor-parallel interval of ``I`` selects
    the resident rows: local up rows first, then local gate rows.
    """
    local = module.intermediate_slice
    width = local.stop - local.start
    hidden = module.hidden_size
    if expert is None:
        experts: tuple[slice, ...] = (slice(0, module.num_experts),)
        targets: tuple[slice, ...] = experts
    else:
        if not 0 <= expert < module.num_experts:
            raise ValueError("expert index is outside the stacked experts")
        experts = ()
        targets = (slice(expert, expert + 1),)

    result = []
    for (weight, offset), rows in (
        (up, slice(0, width)),
        (gate, slice(width, 2 * width)),
    ):
        result.append(
            Assignment(
                module.up_gate.weight,
                weight,
                source_slice=(
                    *experts,
                    slice(offset + local.start, offset + local.stop),
                    slice(0, hidden),
                ),
                target_slice=(*targets, rows, slice(0, hidden)),
            )
        )
    result.append(
        Assignment(
            module.down.weight,
            down,
            source_slice=(*experts, slice(0, hidden), local),
            target_slice=(*targets, slice(0, hidden), slice(0, width)),
        )
    )
    return tuple(result)


def owner_path(loader, key) -> str:
    """Name a parameter by its owning module for load errors."""
    owner, name = loader._owners[key]
    return f"{type(owner).__name__}.{name}"


def _nvfp4_fragments(shape, fragments, *, device, dtype):
    """Assemble NVFP4 encodings from checkpoint fragments without decoding.

    A matrix ``[rows, columns]`` keeps one tensor scale, which every fragment
    must share. A stacked expert tensor ``[E, rows, columns]`` keeps one
    tensor scale per expert, which every fragment of that expert must share.
    """
    if len(shape) == 2:
        stacked = _nvfp4_expert_fragments(
            (1, *shape),
            tuple(
                ((slice(0, 1), *target), fragment)
                for target, fragment in fragments
            ),
            device=device,
            dtype=dtype,
        )
        fields = stacked.buffers()
        return Quantizer("nvfp4").from_tensors(
            {
                "values": fields["values"].reshape(shape[0], -1),
                "block_scale": fields["block_scale"],
                "tensor_scale": fields["tensor_scale"].reshape(()),
            },
            shape=shape,
            dtype=dtype,
        )
    return _nvfp4_expert_fragments(shape, fragments, device=device, dtype=dtype)


def _nvfp4_expert_fragments(shape, fragments, *, device, dtype):
    """Assemble stacked expert NVFP4 encodings without re-encoding values.

    ``shape`` is the stacked ``[E, rows, columns]`` parameter. Each fragment
    covers whole K16 column blocks of a row interval of one or more experts
    and carries its checkpoint tensor scale; fragments of one expert must
    agree on it, since the expert keeps a single tensor-scale domain. Block
    scales keep their E4M3 bytes in a linear ``[E * rows, columns / 16]``
    layout.
    """
    experts, rows, columns = shape
    if columns % 16:
        raise ValueError("NVFP4 expert columns must form whole K16 blocks")
    values = torch.zeros(
        (experts, rows, columns // 2), dtype=torch.uint8, device=device
    )
    block_scale = torch.zeros(
        (experts, rows, columns // 16), dtype=torch.uint8, device=device
    )
    tensor_scale = torch.ones(experts, dtype=torch.float32, device=device)
    assigned = torch.zeros(experts, dtype=torch.bool, device=device)

    for target, fragment in fragments:
        expert_axis, row_axis, column_axis = target
        if column_axis.start % 16 or column_axis.stop % 16:
            raise ValueError("NVFP4 expert fragments must keep K16 blocks")
        count = expert_axis.stop - expert_axis.start
        height = row_axis.stop - row_axis.start
        fields = fragment.repack(scale_layout=ScaleLayout.LINEAR).buffers()
        packed = slice(column_axis.start // 2, column_axis.stop // 2)
        blocks = slice(column_axis.start // 16, column_axis.stop // 16)
        values[expert_axis, row_axis, packed].copy_(
            fields["values"].reshape(count, height, -1).to(device)
        )
        block_scale[expert_axis, row_axis, blocks].copy_(
            fields["block_scale"].reshape(count, height, -1).to(device)
        )

        # Every fragment of an expert must carry that expert's tensor scale.
        incoming = fields["tensor_scale"].to(device).reshape(-1).expand(count)
        seen = assigned[expert_axis]
        if seen.any() and not torch.equal(
            tensor_scale[expert_axis][seen], incoming[seen]
        ):
            raise ValueError(
                "NVFP4 fragments of one expert disagree on their tensor scale"
            )
        tensor_scale[expert_axis] = incoming
        assigned[expert_axis] = True

    return Quantizer("nvfp4").from_tensors(
        {
            "values": values,
            "block_scale": block_scale.reshape(experts * rows, columns // 16),
            "tensor_scale": tensor_scale,
        },
        shape=shape,
        dtype=dtype,
    )


class _Loader:
    """Own readers and materialization.

    While preserving shared parameter identity.
    """

    def __init__(
        self,
        model,
        sources,
        mappings,
        *,
        device,
        weights,
        io,
        devices,
        excluded=(),
        selected=None,
        known_paths=None,
    ):
        self.model = model
        self.sources = sources
        self.mappings = mappings
        self.excluded = excluded
        self.selected = selected
        self._known_paths = known_paths
        self.device = torch.device(device)
        self.weights = weights
        self.io = io
        self.devices = (
            {}
            if devices is None
            else {path: torch.device(value) for path, value in devices.items()}
        )
        self._stack = ExitStack()
        self._readers = {}

        # Parameter identity indexes: who shares storage, who declared it,
        # and the device/dtype/quantization it must materialize with.
        self._aliases = defaultdict(list)
        self._owners = {}
        self._settings = {}

        # Load progress: validated assignment rectangles per parameter,
        # per-mapping assignments in declaration order, and consumed sources.
        self._assignments = defaultdict(list)
        self._regions = {}
        self._mapping_assignments = []
        self._used = defaultdict(set)
        self._loaded = set()
        self._padding = {}

        self._index_model()

    def _index_model(self):
        # Precision/device declarations address the complete architecture.
        # Pipeline binding may remove those paths before materialization, but
        # misspelled paths must still fail against the original module tree.
        paths = self._known_paths
        if paths is None:
            paths = {
                path
                for path, _ in self.model.named_modules(remove_duplicate=False)
            }
        for mapping in (
            self.weights.dtypes,
            self.weights.quantization,
            self.devices,
        ):
            unknown = set(mapping).difference(paths)
            if unknown:
                raise ValueError(
                    f"weight settings name unknown module paths: "
                    f"{sorted(unknown)}"
                )

        for path, module in self.model.named_modules(remove_duplicate=False):
            dtype = _choice(path, self.weights.dtypes, self.weights.dtype)
            quantization = _choice(path, self.weights.quantization, None)
            explicit = any(
                not prefix or path == prefix or path.startswith(prefix + ".")
                for prefix in self.weights.quantization
            )
            device = _choice(path, self.devices, self.device)

            if isinstance(module, (Linear, ExpertLinear)):
                module.input_quantizer = (
                    None if quantization is None else quantization.activation
                )

            for name, parameter in module.named_parameters(
                recurse=False, remove_duplicate=False
            ):
                key = id(parameter)
                quantizer = (
                    quantization.weight
                    if isinstance(module, (Linear, ExpertLinear))
                    and name == "weight"
                    and quantization is not None
                    else None
                )
                settings = (
                    device,
                    dtype
                    if parameter.dtype.is_floating_point
                    else parameter.dtype,
                    quantizer,
                    explicit,
                )
                previous = self._settings.setdefault(key, settings)
                if previous != settings:
                    raise ValueError(
                        "shared parameter aliases have conflicting device, "
                        "dtype or quantization choices"
                    )
                self._aliases[key].append((module, name))
                self._owners.setdefault(key, (module, name))

                # Padded vocabulary rows hold no checkpoint values; record the
                # padding rectangle so reports and materialization ignore it.
                if isinstance(
                    module, (VocabParallelEmbedding, VocabParallelHead)
                ):
                    start = max(
                        0,
                        min(
                            parameter.shape[0],
                            module.vocab.size - module.vocab.local_slice.start,
                        ),
                    )
                    self._padding[key] = (
                        slice(start, parameter.shape[0]),
                        *(slice(0, width) for width in parameter.shape[1:]),
                    )

    def _compile(self, assignment):
        key = id(assignment.target)
        if key not in self._owners:
            raise ValueError(
                "checkpoint assignment target is not a registered model "
                "Parameter"
            )
        source = (
            _full(assignment.source.shape)
            if assignment.source_slice is None
            else assignment.source_slice
        )
        target = (
            _full(assignment.target.shape)
            if assignment.target_slice is None
            else assignment.target_slice
        )
        if not within(source, assignment.source.shape) or not within(
            target, tuple(assignment.target.shape)
        ):
            raise ValueError(
                "checkpoint assignment rectangle exceeds its tensor"
            )
        # A lower-rank source fills a target rectangle whose extra leading
        # axes have extent one.
        extra = len(target) - len(source)
        if extra < 0 or any(
            axis.stop - axis.start != 1 for axis in target[:extra]
        ):
            raise ValueError(
                "checkpoint assignment adds only leading unit target axes"
            )

        # A complete logical source narrows to the owner's bound partition;
        # explicit rectangles already address resident storage and pass through.
        owner, name = self._owners[key]
        if isinstance(owner, (VocabParallelEmbedding, VocabParallelHead)):
            if source[0].stop - source[
                0
            ].start == owner.vocab.size and target == _full(
                assignment.target.shape
            ):
                start = min(owner.vocab.size, owner.vocab.local_slice.start)
                stop = min(owner.vocab.size, owner.vocab.local_slice.stop)
                source = (
                    slice(source[0].start + start, source[0].start + stop),
                    *source[1:],
                )
                target = (slice(0, stop - start), *target[1:])
        elif isinstance(owner, Linear) and name in {"weight", "bias"}:
            logical = (
                (owner.out_features, owner.in_features)
                if name == "weight"
                else (owner.out_features,)
            )
            local = (
                owner._weight_slice
                if name == "weight"
                else owner._weight_slice[:1]
            )
            if region_shape(source) == logical and target == _full(
                assignment.target.shape
            ):
                source = tuple(
                    slice(base.start + part.start, base.start + part.stop)
                    for base, part in zip(source, local, strict=True)
                )

        if region_shape(source) != region_shape(target)[extra:]:
            raise ValueError(
                f"checkpoint assignment shape mismatch: "
                f"{region_shape(source)} to {region_shape(target)}"
            )
        self._regions[id(assignment)] = source, target

        # Re-deriving the identical rectangle (e.g. shared source branches) is
        # a no-op; a genuinely different overlapping one is ambiguous.
        for previous in self._assignments[key]:
            if (
                _source_identity(previous.source)
                == _source_identity(assignment.source)
                and self._regions[id(previous)] == (source, target)
                and previous.preserve_dtype == assignment.preserve_dtype
            ):
                return
            if intersection(self._regions[id(previous)][1], target) is not None:
                raise ValueError(
                    "overlapping checkpoint assignments must identify the "
                    "same source rectangle"
                )
        self._assignments[key].append(assignment)

    def _reports(self):
        used = self._used
        reports = []
        for module_mapping, assignments in zip(
            self.mappings, self._mapping_assignments, strict=True
        ):
            parameters = dict(
                module_mapping.module.named_parameters(remove_duplicate=False)
            )
            unknown = module_mapping.required.difference(parameters)
            if unknown:
                raise ValueError(
                    f"mapping required names are not model parameters: "
                    f"{sorted(unknown)}"
                )
            ids = {id(assignment.target) for assignment in assignments}
            loaded = frozenset(
                name
                for name, parameter in parameters.items()
                if id(parameter) in ids
            )
            # Coverage: subtract every assigned target rectangle (and known
            # padding) from each loaded parameter's full region.
            incomplete = {}
            for name in loaded:
                parameter = parameters[name]
                uncovered: tuple[tuple[slice, ...], ...] = (
                    _full(parameter.shape),
                )
                if id(parameter) in self._padding:
                    uncovered = subtract(
                        uncovered[0], self._padding[id(parameter)]
                    )
                for assignment in self._assignments[id(parameter)]:
                    _, target = self._regions[id(assignment)]
                    uncovered = tuple(
                        piece
                        for region in uncovered
                        for piece in subtract(region, target)
                    )
                if uncovered and parameter.numel():
                    incomplete[name] = tuple(
                        str(region) for region in uncovered
                    )

            reader = self._readers[module_mapping.source]
            reports.append(
                Report(
                    loaded,
                    tuple(
                        sorted(set(reader.names()) & module_mapping.nonresident)
                    ),
                    tuple(
                        sorted(
                            module_mapping.required.difference(
                                module_mapping.optional
                            ).difference(loaded)
                        )
                    ),
                    tuple(
                        sorted(
                            set(reader.names()).difference(
                                used[module_mapping.source]
                            )
                        )
                    ),
                    incomplete,
                )
            )
        return tuple(reports)

    def load(self) -> tuple[Report, ...]:
        if self.io.mode == "dummy" and not any(
            module_mapping.post_load is not None
            for module_mapping in self.mappings
        ):
            return self._dummy()
        if len({source.name for source in self.sources}) != len(self.sources):
            raise ValueError("checkpoint source names must be unique")
        by_name = {source.name: source for source in self.sources}
        for module_mapping in self.mappings:
            if module_mapping.source not in self._readers:
                self._readers[module_mapping.source] = (
                    self._stack.enter_context(
                        by_name[module_mapping.source].open(io=self.io)
                    )
                )
            reader = self._readers[module_mapping.source]
            assignments = module_mapping.map_weights(reader)
            for assignment in assignments:
                self._compile(assignment)
                self._used[module_mapping.source].update(
                    _source_names(assignment.source)
                )
            self._used[module_mapping.source].update(module_mapping.nonresident)
            self._mapping_assignments.append(assignments)

        # A selected capability can share a file with other declared mappings.
        # Account for their known source fields without materializing them or
        # opening additional files. Unknown tensors still fail completeness.
        for module_mapping in self.excluded:
            reader = self._readers.get(module_mapping.source)
            if reader is not None:
                for assignment in module_mapping.map_weights(reader):
                    self._used[module_mapping.source].update(
                        _source_names(assignment.source)
                    )
                self._used[module_mapping.source].update(
                    module_mapping.nonresident
                )
        reports = self._reports()
        for report in reports:
            if report.missing or report.incomplete:
                raise RuntimeError(
                    f"checkpoint load mismatch: missing={report.missing}, "
                    f"unexpected={report.unexpected}, "
                    f"incomplete={dict(report.incomplete)}"
                )

        for key, assignments in self._assignments.items():
            self._materialize(key, assignments)

        for module_mapping in self.mappings:
            if module_mapping.post_load is not None:
                module_mapping.post_load(self._readers[module_mapping.source])
        # Auxiliary source values used to derive constants need not be mapped
        # onto Parameters. Account for actual reads while the reader is open;
        # merely enumerating source metadata does not consume a tensor.
        for source, reader in self._readers.items():
            self._used[source].update(reader._consumed)
        reports = tuple(
            replace(
                report,
                unexpected=tuple(
                    sorted(
                        set(
                            self._readers[module_mapping.source].names()
                        ).difference(self._used[module_mapping.source])
                    )
                ),
            )
            for module_mapping, report in zip(
                self.mappings, reports, strict=True
            )
        )
        for report in reports:
            if report.unexpected:
                raise RuntimeError(
                    f"checkpoint load mismatch: unexpected={report.unexpected}"
                )

        self._buffers()
        self._fuse()
        return reports

    def _fuse(self):
        assigned: set[int] = set()
        for module in self.model.modules():
            if isinstance(module, MergedColumnParallelLinear):
                _coalesce(module.projections, assigned=assigned)

    def _materialize(self, key, assignments):
        device, dtype, quantizer, explicit = self._settings[key]
        parameter = assignments[0].target
        preserved = {
            assignment.source.dtype
            for assignment in assignments
            if assignment.preserve_dtype
        }
        if len(preserved) > 1 or (
            preserved
            and not all(assignment.preserve_dtype for assignment in assignments)
        ):
            raise ValueError(
                "one Parameter cannot have conflicting source-dtype "
                "requirements"
            )
        if preserved:
            dtype = preserved.pop()

        complete = (
            len(assignments) == 1
            and self._regions[id(assignments[0])][1] == _full(parameter.shape)
            and len(self._regions[id(assignments[0])][0]) == parameter.ndim
        )
        if complete:
            assignment = assignments[0]
            source, _ = self._regions[id(assignment)]
            value = assignment.source.read(source).to(
                device=device, dtype=dtype
            )
        else:
            fragments = []
            for assignment in assignments:
                source, target = self._regions[id(assignment)]
                fragments.append((target, assignment.source.read(source)))
            if all(
                isinstance(fragment, QuantizedTensor)
                and fragment.quantizer.format == "fp8"
                for _, fragment in fragments
            ):
                value = _fp8_fragments(
                    parameter.shape, fragments, device=device, dtype=dtype
                )
            elif (
                parameter.ndim in {2, 3}
                and quantizer is not None
                and quantizer.format == "nvfp4"
                and all(
                    isinstance(fragment, QuantizedTensor)
                    and fragment.quantizer.format == "nvfp4"
                    for _, fragment in fragments
                )
            ):
                value = _nvfp4_fragments(
                    tuple(parameter.shape),
                    fragments,
                    device=device,
                    dtype=dtype,
                )
            elif quantizer is not None and any(
                isinstance(fragment, QuantizedTensor)
                for _, fragment in fragments
            ):
                # Decoding encoded fragments to re-encode them would replace
                # the checkpoint's calibrated scales with derived ones.
                raise ValueError(
                    f"checkpoint fragments of {owner_path(self, key)} are "
                    "encoded differently from its configured "
                    f"{quantizer.format} representation"
                )
            else:
                value = torch.empty(
                    tuple(parameter.shape), device=device, dtype=dtype
                )
                for target, fragment in fragments:
                    if isinstance(fragment, QuantizedTensor):
                        fragment = fragment.dequantize(dtype=dtype)
                    value[target].copy_(
                        fragment.reshape(region_shape(target)).to(
                            device=device, dtype=dtype
                        )
                    )

        if key in self._padding and not isinstance(value, QuantizedTensor):
            value[self._padding[key]].zero_()

        if (
            quantizer is not None
            and isinstance(value, QuantizedTensor)
            and quantizer.format == value.quantizer.format
        ):
            # Serialized scales describe the checkpoint's complete statistical
            # domain. Execution must not derive a rank-local replacement from
            # already encoded values.
            pass
        elif quantizer is not None and isinstance(value, QuantizedTensor):
            raise ValueError(
                f"checkpoint tensor for {owner_path(self, key)} is "
                f"{value.quantizer.format}, not the configured "
                f"{quantizer.format} representation"
            )
        elif quantizer is not None:
            owner, _ = self._owners[key]
            if isinstance(owner, ExpertLinear):
                # Encoding stacked experts here would need per-expert scale
                # statistics across their tensor-parallel shards and a
                # calibrated activation scale; checkpoints supply both.
                raise ValueError(
                    f"{owner_path(self, key)} loads dense weights, but "
                    f"{quantizer.format} experts load only from a checkpoint "
                    "that stores them encoded with calibrated input scales"
                )
            value = quantizer.quantize(
                value, distribution=owner.weight_distribution
            )
        elif isinstance(value, QuantizedTensor) and explicit:
            value = value.dequantize(dtype=dtype)

        owner, _ = self._owners[key]
        if (
            isinstance(owner, ExpertLinear)
            and isinstance(value, QuantizedTensor)
            and value.quantizer.format == "nvfp4"
        ):
            # Expert block scales reside in the swizzled layout grouped
            # expert kernels read; expert row blocks are whole 128-row tiles.
            value = value.repack(scale_layout=ScaleLayout.SWIZZLED_128X4)

        # One resident Parameter object serves every module that shared the
        # original placeholder, preserving alias identity after the swap.
        resident = nn.Parameter(value, requires_grad=parameter.requires_grad)
        for owner, name in self._aliases[key]:
            owner._parameters[name] = resident
        self._loaded.add(key)

    def _buffers(self):
        active = (
            self.selected
            if self.selected is not None
            else {
                id(module)
                for module_mapping in self.mappings
                for module in module_mapping.module.modules()
            }
        )
        moved: dict[int, torch.Tensor] = {}
        buffers = [
            (path, module, name, buffer)
            for path, module in self.model.named_modules(remove_duplicate=False)
            for name, buffer in module.named_buffers(recurse=False)
        ]
        for path, module, name, buffer in buffers:
            if id(module) not in active:
                continue
            device = _placed(_choice(path, self.devices, self.device))
            if buffer.is_meta:
                raise RuntimeError(
                    f"derived buffer {path}.{name} was not materialized by "
                    "its numerical owner"
                )

            # Aliased buffers move once; every owner references the same tensor.
            key = id(buffer)
            if key in moved and moved[key].device != device:
                raise ValueError(
                    "shared buffer aliases have conflicting devices"
                )
            if key not in moved:
                moved[key] = buffer.to(device=device)
            module._buffers[name] = moved[key]

    def _dummy(self):
        import hashlib

        reports = []
        parameters = [
            tuple(
                (name, parameter)
                for name, parameter in module_mapping.module.named_parameters(
                    remove_duplicate=False
                )
                if name in module_mapping.required
                or name in module_mapping.optional
            )
            for module_mapping in self.mappings
        ]
        for module_mapping, parameter_items in zip(
            self.mappings, parameters, strict=True
        ):
            loaded = set()
            for name, parameter in parameter_items:
                key = id(parameter)
                if key not in self._loaded:
                    device, dtype, quantizer, _ = self._settings[key]

                    # Values are deterministic per parameter name so repeated
                    # dummy loads and shared aliases observe identical data.
                    seed = int.from_bytes(
                        hashlib.sha256(name.encode()).digest()[:8], "little"
                    )
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    value = torch.empty(
                        tuple(parameter.shape), dtype=dtype, device="cpu"
                    )
                    if dtype.is_floating_point:
                        value.normal_(0, 0.02, generator=generator)
                    else:
                        value.zero_()
                    value = value.to(device)
                    if key in self._padding:
                        value[self._padding[key]].zero_()
                    if quantizer is not None:
                        owner, _ = self._owners[key]
                        value = quantizer.quantize(
                            value, distribution=owner.weight_distribution
                        )

                    resident = nn.Parameter(
                        value, requires_grad=parameter.requires_grad
                    )
                    for owner, field in self._aliases[key]:
                        owner._parameters[field] = resident
                    self._loaded.add(key)
                loaded.add(name)
            reports.append(Report(frozenset(loaded), (), (), (), {}))

        self._buffers()
        self._fuse()
        return tuple(reports)

    def close(self) -> None:
        self._stack.close()
        self._readers.clear()
        self._assignments.clear()
