"""Independent numerical operator, workspace and stream execution ownership."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from contextvars import ContextVar
from functools import partial
from types import MappingProxyType
from typing import Generic, TypeVar, cast

import torch
from torch import nn

from uniserve.distributed import communicators
from uniserve.distributed.mesh import Communicator
from uniserve.model.inputs import TextSize
from uniserve.nn import _binding
from uniserve.nn.attention import Attention
from uniserve.nn.attention._parallel import (
    AttentionBuffers,
    OutputBuffers,
    ParallelAttention,
)
from uniserve.nn.attention.vsa import BlockAttention
from uniserve.nn.linear import (
    ColumnParallelLinear,
    Linear,
    MergedColumnParallelLinear,
)
from uniserve.runtime._collectives import GatherPool
from uniserve.runtime.communication import stream_collective_scope
from uniserve.tensors import BufferConfig

from .bindings import capturing
from .bindings.attention import AttentionBinding, ExchangeBuffers
from .bindings.matmul import MatmulBinding
from .bindings.vsa import VsaBinding
from .resources import streams_idle
from .stream import CUDAStream
from .tensor_buffers import TensorBuffers

SizeT = TypeVar("SizeT")
ValueT = TypeVar("ValueT")


def _install(
    scope: ExitStack, variable: ContextVar[ValueT], value: ValueT
) -> None:
    """Bind ``variable`` to ``value`` until ``scope`` exits."""
    token = variable.set(value)
    scope.callback(variable.reset, token)


def _representation(module, inherited):
    for value in (
        *module.parameters(recurse=False),
        *module.buffers(recurse=False),
    ):
        if value.is_floating_point() and not value.is_meta:
            return value.device, value.dtype
    return inherited


class Scratch:
    """Role-keyed transient work areas for calls serialized on one stream.

    A role names storage whose contents one call writes before reading and
    no later call reads, so every call site, size and context borrowing a
    role from one ``Scratch`` can share one backing, provided their calls
    run one after another: contexts sharing a ``Scratch`` execute on the same
    stream (without one, the same device's current stream from one thread).
    A backing is allocated at the first request's shapes; a later request
    whose every extent fits borrows compact leading views of it, and a
    larger one allocates a new backing outside capture. Earlier backings stay
    alive for the views and graphs that address them, so callers evaluating
    several sizes prepare the largest first.
    """

    def __init__(self):
        # Backings by (role, device, requirement schema), largest first.
        self._backings: dict[tuple[object, ...], list[TensorBuffers]] = {}

    def view(
        self,
        role: object,
        requirements: Mapping[str, BufferConfig],
        device: torch.device,
    ) -> Mapping[str, torch.Tensor]:
        """Borrow views of ``role``'s backing sized to ``requirements``.

        Raises:
            RuntimeError: A request exceeds every backing while a graph is
                being captured.
        """
        key = (
            role,
            device,
            tuple(
                (name, config.dtype, config.host, len(config.shape))
                for name, config in requirements.items()
            ),
        )
        backings = self._backings.setdefault(key, [])
        for backing in backings:
            try:
                return backing.view(requirements)
            except ValueError:
                continue
        if capturing(device):
            raise RuntimeError(
                f"prepare {role!r} scratch for this size before capture"
            )
        backing = TensorBuffers.allocate(requirements, device=device)
        backings.insert(0, backing)
        return backing.view(requirements)

    def close(self) -> None:
        """Release every backing once its borrowing calls and graphs retire."""
        try:
            for backings in self._backings.values():
                for backing in backings:
                    backing.close()
        finally:
            self._backings.clear()


class ExecutionContext(Generic[SizeT]):
    """Own one independent invocation domain for shared numerical modules.

    prepare declares capacity and materializes root capability resources.
    Eager calls specialize additional numerical dtype/shape combinations;
    every combination used by a graph must be exercised before capture.
    Callers retire graphs and asynchronous readers before preparing again or
    closing this owner. A new preparation replaces the previous capacity.
    Weights and externally supplied buffers remain caller-owned.

    ``stream`` is a borrowed :class:`CUDAStream`. Its communication owner
    holds the communicators, registered windows and window storage every
    context on the stream shares; this context never retires them.
    ``scratch``, when given, is a borrowed :class:`Scratch` whose other
    borrowers run on the same stream; the caller closes it after every
    borrower retires. Without it the context owns a private one.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        stream: CUDAStream | None = None,
        cache=None,
        attention="auto",
        vsa="auto",
        matmul="auto",
        groups=None,
        scratch: Scratch | None = None,
    ):
        # Close releases the module; a closed context never executes again.
        self.module: nn.Module | None = module
        self.stream, self.cache = stream, cache
        self._groups = (
            communicators(module) if groups is None else tuple(groups)
        )
        self._attention_backend, self._vsa_backend, self._matmul_backend = (
            attention,
            vsa,
            matmul,
        )

        reference: torch.Tensor | None = next(
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

        self.constants: Mapping[str, torch.Tensor] = MappingProxyType({})
        self.workspace: Mapping[str, torch.Tensor] = self.constants
        self._allocations: list[TensorBuffers] = []
        # Transient work areas by role (see ``scratch``); a private owner is
        # released with this context's preparation, a borrowed one is not.
        self._owns_scratch = scratch is None
        self._scratch = Scratch() if scratch is None else scratch

        self._operators: dict[int, MatmulBinding] = {}
        self._merged: dict[int | _binding.MergedKey, MatmulBinding] = {}
        self._attention: dict[int, AttentionBinding] = {}
        self._vsa: dict[int, VsaBinding] = {}
        self._vsa_output: dict[ParallelAttention, OutputBuffers] = {}
        self._vsa_context: dict[ParallelAttention, AttentionBuffers] = {}
        self._context_backing: dict[tuple[object, ...], AttentionBuffers] = {}
        self._vsa_transport: dict[
            tuple[object, ...], tuple[OutputBuffers, AttentionBuffers | None]
        ] = {}
        self._exchange: dict[int, ExchangeBuffers] = {}
        self._chunks: dict[int, _binding.ChunkStorage] = {}
        self._gather_pools: dict[Communicator, GatherPool] = {}

        from ._transfers import _Transfers

        # Close releases the transfers after their final reset.
        self._transfers: _Transfers | None = _Transfers(module, self._device)
        # Streams created for graphs that borrow this context without an
        # execution stream; their replays read this context's backing.
        self._graph_streams: set[torch.cuda.Stream] = set()
        self._entered: AbstractContextManager[None] | None = None
        self._max_tokens: int | None = None
        self._closed = False

    def _allocate(self, requirements, device):
        allocation = TensorBuffers.allocate(requirements, device=device)
        self._allocations.append(allocation)
        return allocation.view(requirements)

    def scratch(self, role, requirements, device):
        """Borrow transient work areas shared by every call of ``role``.

        Every call site and every size this context evaluates on its
        serialized stream shares one backing per role, as do the other
        borrowers of a shared ``Scratch``; see ``Scratch.view``.
        """
        return self._scratch.view(role, requirements, device)

    def _matmul_workspace(self, requirements, device):
        # GEMM work areas are consumed entirely within one operator call. The
        # provider copies its merged result into caller-owned outputs before
        # returning, so subsequent layers on this serialized context can reuse
        # the same backing. Plans and weights retain their own call-site
        # binding.
        return self.scratch("matmul", requirements, device)

    def _attention_workspace(self, requirements, device):
        # Native attention scratch is consumed on this context's serialized
        # device stream. Plans and metadata remain specific to each call site.
        # Separate contexts share these mutable work areas only through one
        # ``Scratch``, whose borrowers run on the same stream.
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
            views["scratch"] = self.scratch(
                "attention", {"scratch": shared}, device
            )["scratch"]
        return views

    def _vsa_buffers(self, slot, requirements, device):
        # Two projection slots permit one layer's output consumption to overlap
        # the next layer's input preparation. Each context owns its own pair.
        return self.scratch(("vsa", slot), requirements, device)

    def _vsa_exchange(self, layer, rows, heads, head_dim, dtype):
        parallel = getattr(layer, "parallel", None)
        if parallel is None:
            return None

        # Exchange storage serves every query length up to its rows
        # (``ParallelAttention`` borrows compact leading views), so a
        # transport prepared for more rows serves this call too.
        key = next(
            (
                key
                for key in self._vsa_transport
                if key[:3]
                == (
                    parallel.ulysses_group,
                    parallel.context_group,
                    parallel.key_group,
                )
                and key[3] >= rows
                and key[4:] == (heads, head_dim, dtype)
            ),
            (
                parallel.ulysses_group,
                parallel.context_group,
                parallel.key_group,
                rows,
                heads,
                head_dim,
                dtype,
            ),
        )
        if key not in self._vsa_transport:
            if capturing(self._device):
                raise RuntimeError(
                    "prepare VSA communication storage before capture"
                )
            from .attention_storage import (
                allocate_context_storage,
                allocate_output_storage,
            )

            def allocate():
                return allocate_output_storage(
                    (parallel,),
                    rows=rows,
                    heads=heads,
                    head_dim=head_dim,
                    dtype=dtype,
                )

            group = parallel.ulysses_group
            # Registration hands NCCL a window its zero-CTA all-to-all can
            # use, and only symmetric storage backs one: a layer whose context
            # partition composes peer rows. Its storage belongs to the stream,
            # whose communicator retains the window; other storage belongs to
            # this context and takes the ordinary collective. The send buffer
            # is this rank's own destination either way, since where peer
            # storage exists its own entry is that same storage.
            if (
                self.stream is not None
                and group.size > 1
                and parallel.context_group.size > 1
                and group._require().group_name
                in self.stream.communication.communicators
            ):

                def registered():
                    outputs = allocate()
                    buffers = outputs.views[parallel]
                    return (
                        buffers,
                        outputs.allocations,
                        (buffers.local, buffers.receive),
                    )

                # Only registered() stores values under "vsa_output" keys.
                output = cast(
                    OutputBuffers,
                    self.stream.communication.windows(
                        ("vsa_output", key), group, registered
                    ),
                )
            else:
                outputs = allocate()
                self._allocations.extend(outputs.allocations)
                output = outputs.views[parallel]

            contexts = allocate_context_storage(
                (parallel,),
                rows=rows,
                heads=heads,
                head_dim=head_dim,
                dtype=dtype,
                block_size=64,
            )
            self._vsa_transport[key] = (output, contexts.get(parallel))

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
        module = self.module
        if module is None:
            raise RuntimeError("execution context is closed")
        max_rows = size.num_tokens if isinstance(size, TextSize) else None
        self._max_tokens = max_rows

        if self.stream is not None:
            # Communicators belong to the stream and are created once, by the
            # first preparation on it that uses each group. Every member rank
            # prepares the stream's computation, even when ranks prepare
            # different numbers of sizes, so each binding is reached by all.
            self.stream.communication.bind(self._groups)

        with self.activate():
            for name, supplied in (
                ("constants", constants),
                ("workspace", workspace),
            ):
                query = getattr(
                    module,
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

            prepare = getattr(module, "prepare_constants", None)
            if prepare is not None:
                prepare(size, out=self.constants)

            representations = {"": (self._device, self._dtype)}
            vsa_slot = 0
            for path, child in module.named_modules():
                inherited = representations[path.rpartition(".")[0]]
                device, dtype = _representation(child, inherited)
                representations[path] = device, dtype

                if isinstance(child, (Linear, MergedColumnParallelLinear)):
                    binding = MatmulBinding(
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
                        # ModuleDict does not carry its value type; merged
                        # construction registers only column projections.
                        branches = cast(
                            tuple[ColumnParallelLinear, ...],
                            tuple(child.projections.values()),
                        )
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
                        getattr(child, "gather_axes", axes)
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

                        # Registered transport buffers belong to the stream
                        # whose communicator holds their windows.
                        if self.stream is not None:
                            pool = self.stream.communication.gather_pool(group)
                        else:
                            if group not in self._gather_pools:
                                self._gather_pools[group] = GatherPool(
                                    group, None
                                )
                            pool = self._gather_pools[group]
                        self._chunks[id(child)] = partial(
                            pool.borrow, capacity=amount
                        )
                        if amount:
                            with pool.borrow(amount, device):
                                pass

                if isinstance(child, BlockAttention):
                    self._vsa[id(child)] = VsaBinding(
                        self._vsa_backend,
                        self.scratch,
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

                    attention_binding = AttentionBinding(
                        child,
                        self._attention_backend,
                        state,
                        size if isinstance(size, TextSize) else None,
                        device,
                        dtype,
                        self._attention_workspace,
                        self._context_transport,
                    )
                    self._attention[id(child)] = attention_binding

                    if isinstance(size, TextSize):
                        attention_binding.prepare(dtype, size)
                        self._prepare_exchange(
                            child, size.num_tokens, device, dtype
                        )

    def _prepare_exchange(self, layer, num_tokens, device, dtype):
        group = layer.exchange.group
        if group.size == 1:
            return

        # Round up to whole rank shards so transport padding has backing.
        rows = (num_tokens + group.size - 1) // group.size * group.size
        requirements = {
            f"{role}_{direction}": BufferConfig(
                (rows * heads * layer.head_dim * dtype.itemsize,), torch.uint8
            )
            for role, heads in (
                ("query", layer.local_heads),
                ("key", layer.local_kv_heads),
                ("value", layer.local_kv_heads),
                ("output", layer.local_heads),
            )
            for direction in ("send", "receive")
        }
        previous = self._exchange.get(id(layer))
        if previous is not None and all(
            previous.tensors[name].numel() >= config.shape[0]
            for name, config in requirements.items()
        ):
            return

        self._exchange[id(layer)] = ExchangeBuffers(
            self._allocate(requirements, device)
        )

    def _context_transport(self, layer, rows, dtype):
        from .attention_storage import allocate_context_storage

        parallel = layer.context_parallel
        if self._max_tokens is not None:
            members = parallel.context_group.size * layer.exchange.group.size
            rows = max(
                rows,
                (self._max_tokens + members - 1)
                // members
                * layer.exchange.group.size,
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
            layer.local_kv_heads,
            layer.head_dim,
            dtype,
        )
        if key not in self._context_backing:
            self._context_backing[key] = allocate_context_storage(
                (parallel,),
                rows=rows,
                heads=layer.local_kv_heads,
                head_dim=layer.head_dim,
                dtype=dtype,
                block_size=1,
            )[parallel]

        buffers = self._context_backing[key]
        self._vsa_context[parallel] = buffers
        return buffers

    def bind_attention(self, batch):
        """Plan one call's attention metadata on every attention layer.

        Required before graph capture. In eager execution the next call of
        each layer on ``batch`` uses these plans instead of planning again;
        bind again after changing lengths in place. Exact host lengths are
        read at most once for all layers.
        """
        from .backends.attention._sequences import host_lengths

        self._open()
        with self.activate():
            mirrored = batch
            if any(
                binding.reads_host_lengths(batch)
                for binding in self._attention.values()
            ):
                mirrored = host_lengths(batch)
            for binding in self._attention.values():
                binding.bind(mirrored, source=batch)

    @contextmanager
    def activate(self):
        """Enter this context's execution scopes.

        Enter this context's stream, collectives, transfers, and call-site
        bindings.
        """
        self._open()

        with ExitStack() as scope:
            collectives = None
            if self.stream is not None:
                scope.enter_context(torch.cuda.device(self.stream.device))
                scope.enter_context(torch.cuda.stream(self.stream.stream))
                collectives = self.stream.communication.communicators
            scope.enter_context(stream_collective_scope(collectives))
            if self._transfers is not None:
                scope.enter_context(self._transfers.activate())
            from uniserve.nn.attention._parallel import (
                context_scope,
                output_scope,
            )

            scope.enter_context(output_scope(self._vsa_output))
            scope.enter_context(context_scope(self._vsa_context))
            _install(scope, _binding.matmul, self._operators)
            _install(scope, _binding.merged_matmul, self._merged)
            _install(scope, _binding.attention, self._attention)
            _install(scope, _binding.vsa, self._vsa)
            _install(scope, _binding.attention_storage, self._exchange)
            _install(scope, _binding.linear_chunks, self._chunks)
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
                *((self._scratch.close,) if self._owns_scratch else ()),
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
            # Communicators and registered windows outlive this context: they
            # belong to its stream, and a later context on the same stream
            # reuses them rather than issuing a bootstrap its peers have no
            # reason to join.
            self.constants = self.workspace = MappingProxyType({})
            self._max_tokens = None

    def _idle(self) -> bool:
        """Report, without waiting, that no submitted work reads the backing.

        Covers the execution stream (the device's current stream without
        one), cross-device delivery streams, streams of graphs borrowing this
        context, and the stream's communication transfers.
        """
        if self._closed or self._device.type != "cuda":
            return True
        streams = [
            *(
                self._transfers.streams.values()
                if self._transfers is not None
                else ()
            ),
            *self._graph_streams,
        ]
        if self.stream is not None:
            streams.append(self.stream.stream)
            streams.extend(
                transfer
                for communicator in (
                    self.stream.communication.communicators.values()
                )
                if (transfer := communicator.transfer_stream) is not None
            )
        else:
            streams.append(torch.cuda.current_stream(self._device))
        return streams_idle(streams)

    def close(self, *, aborted=False):
        """Release prepared resources and reject subsequent execution.

        Aborted close retains every resource without waiting; the owning
        process must exit before reclaiming them. The borrowed stream keeps
        its communicators either way.
        """
        if self._closed:
            return
        self._closed = True
        if aborted:
            from .resources import retain_until_exit

            retain_until_exit(self)
            return
        try:
            self._release()
        finally:
            if self._transfers is not None:
                self._transfers.close()
                self._transfers = None
            self._graph_streams.clear()
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
            if self._entered is None:
                raise RuntimeError(
                    "execution context ownership was not entered"
                )
            self._entered.__exit__(exc_type, exc, traceback)
        finally:
            self._entered = None
            try:
                # An exception alone does not show that device work is
                # unfinished. Release normally when every stream that reads
                # this context's backing has completed; otherwise retain it,
                # since freeing storage an unfinished access reads is unsafe
                # and waiting can require a failed peer.
                self.close(aborted=exc is not None and not self._idle())
            except BaseException as error:
                if exc is None:
                    raise
                exc.add_note(f"execution context cleanup failed: {error!r}")
