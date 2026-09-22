"""Independent numerical operator, workspace and stream execution ownership."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from math import prod
from types import MappingProxyType
from typing import Generic, TypeVar
from weakref import WeakKeyDictionary

import torch
from torch import nn

from uniserve.distributed import Communicator, DeviceMesh
from uniserve.model.inputs import TextSize
from uniserve.nn import _binding
from uniserve.nn.attention import Attention
from uniserve.nn.attention.inputs import DenseInput
from uniserve.nn.attention.vsa import BlockAttention
from uniserve.nn.linear import (
    ColumnParallelLinear,
    Linear,
    MergedColumnParallelLinear,
)
from uniserve.quantization import QuantizedTensor
from uniserve.runtime._communication import stream_collective_scope
from uniserve.tensors import BufferConfig

from .backends import attention as attention_backend
from .backends import matmul as matmul_backend
from .tensor_buffers import TensorBuffers

SizeT = TypeVar("SizeT")


def _capturing(device):
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _representation(module, inherited):
    for value in (
        *module.parameters(recurse=False),
        *module.buffers(recurse=False),
    ):
        if value.is_floating_point() and not value.is_meta:
            return value.device, value.dtype
    return inherited


#: Communicator bindings by the stream they were established on. A binding is
#: a collective bootstrap, so it is made once per stream and shared by every
#: context that runs on it.
_STREAM_COLLECTIVES: MutableMapping[object, dict] = WeakKeyDictionary()


def _stream_collectives(stream) -> dict:
    """Return the bindings established on one stream, creating the record.

    The record is keyed weakly by the stream, so the bindings live exactly as
    long as the stream that carries them and are released with it rather than
    with whichever context happened to establish them.
    """
    bindings = _STREAM_COLLECTIVES.get(stream)
    if bindings is None:
        bindings = {}
        _STREAM_COLLECTIVES[stream] = bindings
    return bindings


def close_stream_collectives(stream, *, aborted: bool = False) -> None:
    """Release every binding established on one stream.

    Retiring a communicator normally is collective, so ``aborted`` releases
    each binding on this rank alone instead. A caller unwinding from a failure
    passes it: the ranks a collective retirement would wait for are still
    serving, and waiting for them never returns.
    """
    for binding in _STREAM_COLLECTIVES.pop(stream, {}).values():
        if aborted:
            binding.abort()
        else:
            binding.close()


def _communicators(module, *, stage_local=False):
    """Discover borrowed communication interfaces.

    Discover borrowed communication interfaces in ordinary module attributes.
    ``stage_local`` excludes the groups that join distinct pipeline stages,
    for a module one stage holds alone: the stages that do not hold it never
    prepare it, so opening a binding there would wait for participants that
    never arrive.
    """
    groups = {}

    def visit(value):
        if isinstance(value, Communicator):
            groups[value] = value
        elif isinstance(value, DeviceMesh):
            # A bound mesh holds a group for every axis combination, including
            # ones joining distinct pipeline stages.
            groups.update(
                (group, group)
                for axes, group in value._groups.items()
                if not (stage_local and "pp" in axes)
            )
        elif isinstance(value, Mapping):
            for member in value.values():
                if isinstance(member, (Communicator, DeviceMesh)):
                    visit(member)
        elif isinstance(value, (tuple, list)):
            for member in value:
                if isinstance(member, (Communicator, DeviceMesh)):
                    visit(member)

    for child in module.modules():
        for value in vars(child).values():
            visit(value)
    return groups.values()


@dataclass(frozen=True)
class _ExchangeBuffers:
    tensors: Mapping[str, torch.Tensor]

    def view(self, name, shape, like, *, offset=0):
        backing = self.tensors[name]
        # Offsets and extents are byte addresses into the flat uint8 backing.
        start, size = (
            offset * like.element_size(),
            prod(shape) * like.element_size(),
        )
        if (
            start < 0
            or start + size > backing.numel()
            or backing.device != like.device
        ):
            raise ValueError("attention exchange exceeds its prepared capacity")
        return backing.narrow(0, start, size).view(like.dtype).view(shape)


class _MatmulBinding:
    """Specialize one numerical call site.

    Specialize one numerical call site while retaining all borrowed backing.
    """

    def __init__(self, module, backend, max_rows, allocate):
        self.module, self.backend, self.max_rows = module, backend, max_rows
        self.allocate = allocate
        self.operators = {}

    def _prepare(self, dtype, output_dtype, quantizer, rows):
        if self.max_rows is not None and rows > self.max_rows:
            raise ValueError("matmul exceeds the prepared token capacity")
        rows = max(rows, self.max_rows or 0)
        key = (dtype, output_dtype, quantizer)
        previous = self.operators.get(key)
        if previous is not None and previous[0] >= rows:
            return previous[1]

        if _capturing(self._weight().device):
            raise RuntimeError(
                "matmul shape and representation must be prepared before "
                "capture"
            )

        options = {
            "input_dtype": dtype,
            "input_quantizer": quantizer,
            "max_rows": rows,
            "output_dtype": output_dtype,
        }
        provider = matmul_backend.resolve(self.backend, self._weight())
        if isinstance(self.module, MergedColumnParallelLinear):
            weights = {
                name: child.weight
                for name, child in self.module.projections.items()
            }
            options["branch_width"] = self.module.branch_width
            requirements = provider.merged_workspace_buffers(weights, **options)
            workspace = self.allocate(requirements, self._weight().device)
            operator = provider.prepare_merged(
                weights, **options, workspace=workspace
            )
        else:
            weight = self.module.weight
            requirements = provider.workspace_buffers(weight, **options)
            workspace = self.allocate(requirements, weight.device)
            operator = provider.prepare(weight, **options, workspace=workspace)

        self.operators[key] = (rows, operator)
        return operator

    def _weight(self):
        if isinstance(self.module, MergedColumnParallelLinear):
            return next(iter(self.module.projections.values())).weight
        return self.module.weight

    def quantize(self, x, quantizer, distribution):
        """Encode a complete logical domain.

        Encode a complete logical domain in this context's activation
        storage.
        """
        operator = self._prepare(x.dtype, x.dtype, quantizer, x.shape[0])
        target = operator._input_storage(x)
        return quantizer.quantize(x, distribution=distribution, out=target)

    def __call__(self, x, bias, *, out):
        destination = (
            next(iter(out.values())) if isinstance(out, Mapping) else out
        )
        quantizer = x.quantizer if isinstance(x, QuantizedTensor) else None
        return self._prepare(x.dtype, destination.dtype, quantizer, x.shape[0])(
            x, bias, out=out
        )


class _AttentionBinding:
    """Specialize one attention call site.

    Specialize one attention call site's operators, plans, and bound
    metadata.
    """

    def __init__(
        self,
        module,
        backend,
        cache,
        size,
        device,
        dtype,
        allocate,
        context_transport,
    ):
        self.module, self.backend, self.cache, self.size = (
            module,
            backend,
            cache,
            size,
        )
        self.device, self.dtype, self.allocate = device, dtype, allocate
        self.operators = {}
        self._bound = set()
        self._bound_batches = {}
        self.batch = None
        self.context_transport = context_transport
        self.context_plans = {}
        self.cache_batches = {}

    def _context_plan(self, batch, dtype):
        from ._attention_context import _ContextPlan

        key = dtype, _ContextPlan.signature(batch)
        if key not in self.context_plans:
            if _capturing(self.device):
                raise RuntimeError(
                    "prepare context attention metadata before capture"
                )
            self.context_plans[key] = _ContextPlan(
                self.module,
                batch,
                cache=self.cache,
                transport=self.context_transport,
                allocate=self.allocate,
                dtype=dtype,
            )
        return self.context_plans[key]

    def prepare(self, dtype, size):
        if self.size is not None and (
            size.num_tokens > self.size.num_tokens
            or size.batch_size > self.size.batch_size
        ):
            raise ValueError(
                "attention exceeds the prepared token or batch capacity"
            )

        previous = self.operators.get(dtype)
        if previous is not None and (
            previous.size.num_tokens >= size.num_tokens
            and previous.size.batch_size >= size.batch_size
        ):
            return previous

        if _capturing(self.device):
            raise RuntimeError(
                "attention shape and dtype must be prepared before capture"
            )

        if self.size is not None:
            size = TextSize(
                max(size.num_tokens, self.size.num_tokens),
                max(size.batch_size, self.size.batch_size),
            )

        options = {
            "num_heads": self.module._local_heads,
            "num_kv_heads": self.module._local_kv_heads,
            "head_dim": self.module.head_dim,
            "dtype": dtype,
            "size": size,
            "cache": self.cache,
        }
        provider = attention_backend.resolve(self.backend, device=self.device)
        requirements = provider.workspace_buffers(**options)
        operator = provider.prepare(
            **options, workspace=self.allocate(requirements, self.device)
        )
        self.operators[dtype] = operator
        self._bound.discard(dtype)
        return operator

    def sequence_inputs(self, batch):
        """Resolve lengths needed to construct mathematical token partitions."""
        if (
            self.module._context is None
            and self.module._exchange.group.size == 1
        ):
            return batch
        from .backends.attention._sequences import host_lengths

        return host_lengths(
            batch, prepared=self.batch if _capturing(self.device) else None
        )

    def bind(self, batch):
        """Bind numerical metadata before graph capture.

        Bind numerical metadata for every prepared dtype before graph
        capture.
        """
        from .backends.attention._sequences import host_lengths

        batch = self.sequence_inputs(batch)
        self.batch = batch

        if not isinstance(batch, DenseInput):
            if batch.queries.num_tokens is not None:
                self.prepare(
                    self.dtype,
                    TextSize(
                        batch.queries.num_tokens, batch.queries.batch_size
                    ),
                )
            elif self.size is not None:
                self.prepare(self.dtype, self.size)

        for dtype, operator in self.operators.items():
            numerical = (
                batch
                if self.module._context is None
                else self._context_plan(batch, dtype).refresh(batch)
            )
            if operator.requires_host_lengths(numerical):
                numerical = host_lengths(numerical)
            operator.bind(numerical)
            self._bound_batches[dtype] = numerical

        self._bound.update(self.operators)

    def __call__(self, q, k, v, batch, *, scale, out):
        from .backends.attention._sequences import host_lengths

        batch = self.sequence_inputs(batch)
        plan = None
        storage = _binding.attention_storage.get().get(id(self.module))
        if self.module._context is not None:
            plan = self._context_plan(batch, q.dtype)
            # Plan construction may have grown exchange backing for a longer
            # key domain than the root's query capacity.
            storage = _binding.attention_storage.get().get(id(self.module))
            q, k, v, batch = plan.inputs(q, k, v, batch, storage=storage)

        size = (
            TextSize(
                q.shape[0] if q.ndim == 3 else q.shape[0] * q.shape[2],
                1 if q.ndim == 3 else q.shape[0],
            )
            if isinstance(batch, DenseInput)
            else TextSize(q.shape[0], batch.queries.batch_size)
        )
        operator = self.prepare(q.dtype, size)

        # Numerical metadata can be passed directly in eager code. Providers
        # requiring host planning are bound explicitly before CUDA capture.
        if _capturing(q.device):
            if q.dtype not in self._bound:
                raise RuntimeError(
                    "bind numerical attention metadata before capture"
                )
            if operator.requires_host_lengths(batch):
                batch = host_lengths(
                    batch, prepared=self._bound_batches[q.dtype]
                )
        else:
            # Eager callers may pass fresh lengths or mutate borrowed columns
            # without an explicit bind call. A previous plan for this dtype
            # does not describe the current batch.
            if operator.requires_host_lengths(batch):
                batch = host_lengths(batch)
            operator.bind(batch)
            self._bound.add(q.dtype)
            self._bound_batches[q.dtype] = batch

        result = operator(
            q,
            k,
            v,
            batch,
            scale=scale,
            out=out if plan is None else torch.empty_like(q),
        )
        return (
            result
            if plan is None
            else plan.restore(result, storage=storage, out=out)
        )

    def update_cache(self, k, v, *, indices):
        """Write current K/V into the bound prefix cache at physical indices."""
        if self.cache is None:
            raise RuntimeError("this attention layer has no bound prefix state")

        if self.module._context is not None:
            from uniserve.nn.attention import SequenceLengths, VarlenInput

            count = indices.numel()
            if count not in self.cache_batches:
                if _capturing(indices.device):
                    raise RuntimeError(
                        "prepare context cache writes before capture"
                    )
                lengths = SequenceLengths.from_lengths(
                    (count,), device=indices.device
                )
                self.cache_batches[count] = VarlenInput(
                    lengths, lengths, (False,)
                )

            plan = self._context_plan(self.cache_batches[count], k.dtype)
            storage = _binding.attention_storage.get().get(id(self.module))
            k, v = plan.exchange(k, v, storage=storage)

        self.cache.update(k, v, indices=indices)

    def close(self):
        for operator in self.operators.values():
            operator.close()
        self.operators.clear()
        self._bound_batches.clear()
        self.context_plans.clear()
        self.cache_batches.clear()
        self.batch = self.cache = None


class _VsaBinding:
    """Own one VSA call site's plans and packed input backing.

    Own one VSA call site's plans, mutable query maps, and packed input
    backing.
    """

    def __init__(self, backend, allocate, scratch, shared_buffers, exchange):
        self.backend, self.allocate, self.scratch = backend, allocate, scratch
        self._shared_buffers, self.exchange = shared_buffers, exchange
        self.operators, self._buffers = {}, {}

    def prepare(self, pattern, q):
        key = (q.device, q.dtype, q.shape[1], q.shape[2], pattern)
        if key not in self.operators:
            if _capturing(q.device):
                raise RuntimeError(
                    "prepare VSA numerical shapes before capture"
                )
            from .backends.attention import vsa

            provider = vsa.resolve(self.backend, device=q.device)
            options = {
                "num_heads": q.shape[1],
                "head_dim": q.shape[2],
                "dtype": q.dtype,
            }
            requirements = provider.workspace_buffers(pattern, **options)
            self.operators[key] = provider.prepare(
                pattern,
                **options,
                workspace=self.scratch(requirements, q.device),
            )

        return self.operators[key]

    def buffers(self, requirements, device):
        key = (device, tuple(requirements.items()))
        if key not in self._buffers:
            if _capturing(device):
                raise RuntimeError(
                    "prepare VSA projected input backing before capture"
                )
            self._buffers[key] = self._shared_buffers(
                {
                    name: BufferConfig(shape, dtype)
                    for name, (shape, dtype) in requirements.items()
                },
                device,
            )

        return self._buffers[key]

    def close(self):
        for operator in self.operators.values():
            operator.close()
        self.operators.clear()
        self._buffers.clear()


class _GatherPool:
    """Lend registered transport buffers until their readers are enqueued.

    Iterators may nest while upstream projections still have unread payloads.
    Each active borrower receives distinct backing; completed borrowers reuse
    storage on the context's serialized stream. Graphs retain every allocation.
    """

    def __init__(self, group, collective):
        self.group = group
        self.collective = collective
        self.buffers = []
        self.borrowed = set()
        self.symmetric = (
            torch.distributed.get_backend(group._require()) == "nccl"
        )

    @contextmanager
    def borrow(self, size, device, *, capacity=None):
        if device != self.group.device:
            raise ValueError(
                "projection exchange must use its communicator's device"
            )
        if capacity is not None and size > capacity:
            raise ValueError("projection exchange exceeds the bound workspace")
        amount = size if capacity is None else capacity

        buffer = next(
            (
                value
                for value in self.buffers
                if id(value) not in self.borrowed and value.numel() >= amount
            ),
            None,
        )
        if buffer is None:
            if _capturing(device):
                raise RuntimeError(
                    "prepare projection exchange backing before capture"
                )
            if self.symmetric:
                from ._peer_memory import allocate_collective_buffer

                buffer = allocate_collective_buffer(
                    (amount,), dtype=torch.uint8, device=device
                )
            else:
                buffer = torch.empty(amount, dtype=torch.uint8, device=device)
            if self.collective is not None:
                self.collective.register_buffers(buffer)
            self.buffers.append(buffer)

        self.borrowed.add(id(buffer))
        try:
            yield buffer
        finally:
            self.borrowed.remove(id(buffer))


class ExecutionContext(Generic[SizeT]):
    """Own one independent invocation domain for shared numerical modules.

    prepare declares capacity and materializes root capability resources.
    Eager calls specialize additional numerical dtype/shape combinations;
    every combination used by a graph must be exercised before capture.
    Callers retire graphs and asynchronous readers before preparing again or
    closing this owner. A new preparation replaces the previous capacity.
    Weights and externally supplied buffers remain caller-owned.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        stream=None,
        cache=None,
        attention="auto",
        vsa="auto",
        matmul="auto",
        stage_local=False,
    ):
        self.module, self.stream, self.cache = module, stream, cache
        # A module one pipeline stage holds alone: see _communicators.
        self._stage_local = stage_local
        self._attention_backend, self._vsa_backend, self._matmul_backend = (
            attention,
            vsa,
            matmul,
        )

        reference = next(
            (value for value in module.parameters() if not value.is_meta), None
        )
        if reference is None:
            reference = next(
                (value for value in module.buffers() if not value.is_meta), None
            )
        self._device = (
            reference.device
            if reference is not None
            else (
                cache.device
                if cache is not None
                else stream.device
                if stream is not None
                else torch.device("cpu")
            )
        )
        self._dtype = (
            reference.dtype if reference is not None else torch.float32
        )
        if stream is not None and stream.device != self._device:
            raise ValueError(
                "the execution stream must belong to the root module device"
            )

        self.constants = self.workspace = MappingProxyType({})
        self._allocations = []
        self._scratch = {}
        self._matmul_scratch = {}

        self._operators = {}
        self._merged = {}
        self._attention = {}
        self._vsa = {}
        self._vsa_backing = {}
        self._vsa_output = {}
        self._vsa_context = {}
        self._context_backing = {}
        self._vsa_transport = {}
        self._exchange = {}
        self._chunks = {}
        self._gather_pools = {}

        from ._transfers import _Transfers

        self._transfers = _Transfers(module, self._device)
        # Active scopes borrow this mapping. Populate it in place so an
        # already-entered context uses the same stream bindings during warmup
        # and capture, including when prepare() discovers more components.
        # Bindings belong to the stream, not to this context. A communicator
        # binding is `ncclCommInitRankConfig` behind a broadcast over the
        # group, so every member has to reach it, and preparation is not a
        # point where they all do: contexts are keyed by numerical size and a
        # rank's media unit count decides which sizes it prepares. One stream
        # serves every size of one computation, so binding there is reached
        # once, by every rank, when the computation is first warmed.
        self._collectives = (
            _stream_collectives(stream) if stream is not None else None
        )
        self._entered = None
        self._max_tokens = None
        self._closed = False

    def _allocate(self, requirements, device):
        allocation = TensorBuffers.allocate(requirements, device=device)
        self._allocations.append(allocation)
        return allocation.view(requirements)

    def _matmul_workspace(self, requirements, device):
        # GEMM work areas are consumed entirely within one operator call. The
        # provider copies its merged result into caller-owned outputs before
        # returning, so subsequent layers on this serialized context can reuse
        # the same backing. Plans and weights retain their own call-site
        # binding.
        key = (device, tuple(requirements.items()))
        if key not in self._matmul_scratch:
            self._matmul_scratch[key] = self._allocate(requirements, device)
        return self._matmul_scratch[key]

    def _attention_workspace(self, requirements, device):
        # Native attention scratch is consumed on this context's serialized
        # device stream. Plans and metadata remain specific to each call site.
        # Separate contexts never share these mutable work areas.
        shared = requirements.get("scratch")
        views = dict(
            self._allocate(
                {
                    name: item
                    for name, item in requirements.items()
                    if name != "scratch"
                },
                device,
            )
        )
        if shared is not None:
            key = (device, shared)
            if key not in self._scratch:
                self._scratch[key] = self._allocate(
                    {"scratch": shared}, device
                )["scratch"]
            views["scratch"] = self._scratch[key]
        return views

    def _vsa_buffers(self, slot, requirements, device):
        # Two projection slots permit one layer's output consumption to overlap
        # the next layer's input preparation. Each context owns its own pair.
        key = (slot, device, tuple(requirements.items()))
        if key not in self._vsa_backing:
            self._vsa_backing[key] = self._allocate(requirements, device)
        return self._vsa_backing[key]

    def _vsa_exchange(self, layer, rows, heads, head_dim, dtype):
        parallel = getattr(layer, "_parallel", None)
        if parallel is None:
            return None

        key = (
            parallel.ulysses_group,
            parallel.context_group,
            parallel.key_group,
            rows,
            heads,
            head_dim,
            dtype,
        )
        if key not in self._vsa_transport:
            if _capturing(self._device):
                raise RuntimeError(
                    "prepare VSA communication storage before capture"
                )
            from .attention_storage import (
                allocate_context_storage,
                allocate_output_storage,
            )

            outputs = allocate_output_storage(
                (parallel,),
                rows=rows,
                heads=heads,
                head_dim=head_dim,
                dtype=dtype,
            )
            self._allocations.extend(outputs.allocations)

            group = parallel.ulysses_group
            # Registration hands NCCL a window its zero-CTA all-to-all can
            # use, and only symmetric storage backs one. A layer exchanging
            # out of ordinary storage cannot be registered and takes the
            # ordinary collective; registering it fails the communicator
            # rather than degrading it. The send buffer is this rank's own
            # destination either way, since where peer storage exists its own
            # entry is that same memory.
            if (
                self._collectives is not None
                and group.size > 1
                and outputs.views[parallel].peers
            ):
                buffers = outputs.views[parallel]
                self._collectives[group._require().group_name].register_buffers(
                    buffers.local, buffers.receive
                )

            context = allocate_context_storage(
                (parallel,),
                rows=rows,
                heads=heads,
                head_dim=head_dim,
                dtype=dtype,
                block_size=64,
            )
            self._vsa_transport[key] = (
                outputs.views[parallel],
                context.get(parallel),
            )

        output, context = self._vsa_transport[key]
        self._vsa_output[parallel] = output
        if context is not None:
            self._vsa_context[parallel] = context

        return parallel

    def _open(self):
        if self._closed:
            raise RuntimeError("execution context is closed")

    def prepare(self, size, *, constants=None, workspace=None):
        """Replace this context's capacity and rebuild every call-site binding.

        ``size`` declares the maximum token and batch extents. ``constants``
        and ``workspace`` optionally supply caller-owned backing; when omitted,
        the required buffers are allocated and owned by this context.
        """
        self._open()
        self._release()

        try:
            self._prepare(size, constants=constants, workspace=workspace)
        except BaseException as error:
            try:
                self._release()
            except BaseException as cleanup:
                error.add_note(
                    f"execution preparation cleanup failed: {cleanup!r}"
                )
            raise

    def _prepare(self, size, *, constants, workspace):
        max_rows = size.num_tokens if isinstance(size, TextSize) else None
        self._max_tokens = max_rows

        if self._collectives is not None:
            from uniserve.runtime._collectives import (
                allocate_stream_collectives,
            )

            pending = (
                group
                for group in _communicators(
                    self.module, stage_local=self._stage_local
                )
                if group.size > 1
                and group._require().group_name not in self._collectives
            )
            self._collectives.update(
                allocate_stream_collectives(pending, self.stream)
            )

        with self.activate():
            for name, supplied in (
                ("constants", constants),
                ("workspace", workspace),
            ):
                query = getattr(
                    self.module,
                    "constant_buffers"
                    if name == "constants"
                    else "workspace_buffers",
                    None,
                )
                requirements = {} if query is None else query(size)
                views = (
                    self._allocate(requirements, self._device)
                    if supplied is None
                    else supplied.view(requirements)
                )
                setattr(self, name, views)

            prepare = getattr(self.module, "prepare_constants", None)
            if prepare is not None:
                prepare(size, out=self.constants)

            representations = {"": (self._device, self._dtype)}
            vsa_slot = 0
            for path, child in self.module.named_modules():
                inherited = representations[path.rpartition(".")[0]]
                device, dtype = _representation(child, inherited)
                representations[path] = device, dtype

                if isinstance(child, (Linear, MergedColumnParallelLinear)):
                    binding = _MatmulBinding(
                        child,
                        self._matmul_backend,
                        max_rows,
                        self._matmul_workspace,
                    )
                    if isinstance(child, MergedColumnParallelLinear):
                        self._merged[id(child)] = binding
                        key = (
                            tuple(
                                (name, id(branch.weight))
                                for name, branch in child.projections.items()
                            ),
                            child.branch_width,
                        )
                        self._merged[key] = binding
                        branches = tuple(child.projections.values())
                        quantizers = {
                            branch.input_quantizer for branch in branches
                        }
                        quantizer = branches[0].input_quantizer
                    else:
                        self._operators[id(child)] = binding
                        self._operators[id(child.weight)] = binding
                        quantizers, quantizer = (
                            {child.input_quantizer},
                            child.input_quantizer,
                        )
                    if max_rows is not None and len(quantizers) == 1:
                        binding._prepare(dtype, dtype, quantizer, max_rows)

                if isinstance(child, ColumnParallelLinear):
                    axes = child.input_distribution.shard_axes(0)
                    group = child.input_distribution.mesh.get_group(
                        getattr(child, "_gather_axes", axes)
                    )
                    if group.size > 1:
                        # Two complete gather slots bound every row/chunk shape.
                        # FP32 transport also covers encoded values plus scales.
                        amount = None
                        if max_rows is not None:
                            rows = (
                                (max_rows + group.size - 1)
                                // group.size
                                * group.size
                            )
                            amount = 2 * rows * child.weight.shape[1] * 4

                        if group not in self._gather_pools:
                            collective = (self._collectives or {}).get(
                                group._require().group_name
                            )
                            self._gather_pools[group] = _GatherPool(
                                group, collective
                            )

                        pool = self._gather_pools[group]
                        self._chunks[id(child)] = partial(
                            pool.borrow, capacity=amount
                        )
                        if amount:
                            with pool.borrow(amount, device):
                                pass

                if isinstance(child, BlockAttention):
                    self._vsa[id(child)] = _VsaBinding(
                        self._vsa_backend,
                        self._allocate,
                        self._attention_workspace,
                        partial(self._vsa_buffers, vsa_slot % 2),
                        self._vsa_exchange,
                    )
                    vsa_slot += 1

                if isinstance(child, Attention):
                    state = None
                    if child.cache_name is not None and self.cache is not None:
                        state = self.cache.state(child.cache_name)
                        device, dtype = state.key.device, state.key.dtype

                    binding = _AttentionBinding(
                        child,
                        self._attention_backend,
                        state,
                        size if isinstance(size, TextSize) else None,
                        device,
                        dtype,
                        self._attention_workspace,
                        self._context_transport,
                    )
                    self._attention[id(child)] = binding

                    if isinstance(size, TextSize):
                        binding.prepare(dtype, size)
                        self._prepare_exchange(
                            child, size.num_tokens, device, dtype
                        )

    def _prepare_exchange(self, layer, num_tokens, device, dtype):
        group = layer._exchange.group
        if group.size == 1:
            return

        # Round up to whole rank shards so transport padding has backing.
        rows = (num_tokens + group.size - 1) // group.size * group.size
        requirements = {
            f"{role}_{direction}": BufferConfig(
                (rows * heads * layer.head_dim * dtype.itemsize,), torch.uint8
            )
            for role, heads in (
                ("query", layer._local_heads),
                ("key", layer._local_kv_heads),
                ("value", layer._local_kv_heads),
                ("output", layer._local_heads),
            )
            for direction in ("send", "receive")
        }
        previous = self._exchange.get(id(layer))
        if previous is not None and all(
            previous.tensors[name].numel() >= config.shape[0]
            for name, config in requirements.items()
        ):
            return

        self._exchange[id(layer)] = _ExchangeBuffers(
            self._allocate(requirements, device)
        )

    def _context_transport(self, layer, rows, dtype):
        from .attention_storage import allocate_context_storage

        parallel = layer._context
        if self._max_tokens is not None:
            members = parallel.context_group.size * layer._exchange.group.size
            rows = max(
                rows,
                (self._max_tokens + members - 1)
                // members
                * layer._exchange.group.size,
            )
        self._prepare_exchange(
            layer,
            rows * parallel.context_group.size,
            parallel.context_group.device,
            dtype,
        )

        if parallel.context_group.size == 1 or rows == 0:
            return None

        key = (
            parallel.key_group,
            rows,
            layer._local_kv_heads,
            layer.head_dim,
            dtype,
        )
        if key not in self._context_backing:
            self._context_backing[key] = allocate_context_storage(
                (parallel,),
                rows=rows,
                heads=layer._local_kv_heads,
                head_dim=layer.head_dim,
                dtype=dtype,
                block_size=1,
            )[parallel]

        buffers = self._context_backing[key]
        self._vsa_context[parallel] = buffers
        return buffers

    def bind_attention(self, batch):
        """Bind current batch metadata before capture.

        Bind current batch metadata on every attention binding before
        capture.
        """
        self._open()
        with self.activate():
            for binding in self._attention.values():
                binding.bind(batch)

    @contextmanager
    def activate(self):
        """Enter this context's execution scopes.

        Enter this context's stream, collectives, transfers, and call-site
        bindings.
        """
        self._open()
        variables = (
            (_binding.matmul, self._operators),
            (_binding.merged_matmul, self._merged),
            (_binding.attention, self._attention),
            (_binding.vsa, self._vsa),
            (_binding.attention_storage, self._exchange),
            (_binding.linear_chunks, self._chunks),
        )

        with ExitStack() as scope:
            if self.stream is not None:
                scope.enter_context(torch.cuda.device(self.stream.device))
                scope.enter_context(torch.cuda.stream(self.stream))
            scope.enter_context(stream_collective_scope(self._collectives))
            if self._transfers is not None:
                scope.enter_context(self._transfers.activate())
            from uniserve.nn.attention._parallel import (
                context_scope,
                output_scope,
            )

            scope.enter_context(output_scope(self._vsa_output))
            scope.enter_context(context_scope(self._vsa_context))
            for variable, value in variables:
                token = variable.set(value)
                scope.callback(variable.reset, token)
            yield

    def _release(self):
        """Retire prepared resources after borrowed uses.

        Retire prepared resources after the caller has ended borrowed uses.
        """
        from .resources import close_resources

        try:
            close_resources(
                *(binding.close for binding in self._attention.values()),
                *(binding.close for binding in self._vsa.values()),
                *(
                    allocation.close
                    for allocation in reversed(self._allocations)
                ),
                *(
                    (self._transfers.reset,)
                    if self._transfers is not None
                    else ()
                ),
            )
        finally:
            for values in (
                self._operators,
                self._merged,
                self._attention,
                self._vsa,
                self._vsa_backing,
                self._vsa_output,
                self._vsa_context,
                self._context_backing,
                self._vsa_transport,
                self._exchange,
                self._chunks,
                self._gather_pools,
            ):
                values.clear()
            self._allocations.clear()
            self._scratch.clear()
            self._matmul_scratch.clear()
            # Bindings outlive this context: they belong to its stream, and
            # a later context on the same stream reuses them rather than
            # issuing a bootstrap its peers have no reason to join.
            self.constants = self.workspace = MappingProxyType({})
            self._max_tokens = None

    def close(self, *, aborted=False):
        """Release prepared resources and reject subsequent execution.

        Aborted close retains resources and aborts stream collectives without
        waiting; the owning process must exit before reclaiming them.
        """
        if self._closed:
            return
        self._closed = True
        if aborted:
            from .resources import retain_until_exit

            retain_until_exit(self)
            if self.stream is not None:
                close_stream_collectives(self.stream, aborted=True)
            return
        try:
            self._release()
        finally:
            self._collectives = None
            if self._transfers is not None:
                self._transfers.close()
                self._transfers = None
            self.module = self.cache = None

    def __enter__(self):
        if self._entered is not None:
            raise RuntimeError(
                "execution context ownership cannot be entered twice"
            )
        self._entered = self.activate()
        self._entered.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self._entered.__exit__(exc_type, exc, traceback)
        finally:
            self._entered = None
            try:
                self.close(aborted=exc is not None)
            except BaseException as error:
                if exc is None:
                    raise
                exc.add_note(f"execution context cleanup failed: {error!r}")
