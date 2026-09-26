"""Select native attention by representation and input.

On CUDA, automatic selection chooses only native kernels:

- the TensorRT-LLM (trtllm-gen) paged kernels and UniServe's prefix-block
  kernel on SM100, for causal and non-causal paged rows and for segmented
  prefix reads whose queries see their whole current block;
- FlashAttention-4 on SM90 and the SM100 family, for dense, variable-length,
  visible-endpoint and segmented inputs and for paged rows without a history
  window. Its SM100 head-dimension-256 kernel accepts neither per-sequence key
  lengths nor mask functions, so it serves only dense and variable-length
  inputs of that width;
- SGLang's FlashAttention-3 build on SM90.

A call no native kernel serves raises when its layer is prepared or bound,
naming the call's path, shape, dtype and mask semantics. FlashInfer's own
attention kernels, the FlashAttention-2 library and the portable torch
provider remain selectable by name but never stand in for a native kernel.
The one CUDA call class the portable provider serves is FP32 single-head
dense attention of head dimension 512, which no native kernel computes
(``_portable``). Off CUDA the portable provider serves every call.

The dispatching operator records the provider that served each input class
it met (``Operator.selections``), so startup can report the kernel behind
every call site.
"""

from dataclasses import replace
from importlib import import_module
from types import MappingProxyType

import torch

from uniserve.nn.attention.inputs import (
    BlockTable,
    DenseInput,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
    VarlenInput,
    VisibleInput,
)
from uniserve.quantization import QuantizedTensor
from uniserve.tensors import BufferConfig

from .. import record_kernel_choice
from . import Backend as _Backend
from . import CachePages
from . import Operator as _Operator
from ._sequences import causal_runs


def _available(module, names):
    try:
        library = import_module(module)
    except (ImportError, OSError):
        return False
    return all(callable(getattr(library, name, None)) for name in names)


def _portable(*, num_heads, num_kv_heads, head_dim, dtype, pages, window):
    """Whether the portable provider serves a CUDA layer's dense calls.

    No native CUDA kernel computes FP32 single-head dense attention of head
    dimension 512 without a prefix cache (``pages`` is None) or history
    window, the spatial self-attention of FP32 image autoencoders: the
    TensorRT-LLM and FlashAttention-4 kernels compute in half precision, and
    neither serves head dimension 512 on dense inputs. The portable torch
    provider evaluates its non-causal unmasked dense calls; every other call
    of such a layer is rejected like any call without a native kernel.
    """
    return (
        dtype == torch.float32
        and head_dim == 512
        and num_heads == num_kv_heads == 1
        and pages is None
        and window is None
    )


def _layer(*, num_heads, num_kv_heads, head_dim, dtype, cache, window):
    """Describe a layer's attention shape, dtypes and history bound."""
    if cache is None:
        storage = "without a prefix cache"
    elif isinstance(cache.key, QuantizedTensor):
        storage = (
            f"over a per-block FP8 cache of {cache.block_size}-token pages"
        )
    else:
        storage = (
            f"over a {cache.key.dtype} cache of {cache.block_size}-token pages"
        )
    history = (
        "the whole history"
        if window is None
        else f"a {window}-token history window"
    )
    return (
        f"{num_heads} query and {num_kv_heads} KV heads of dimension "
        f"{head_dim}, {dtype} queries {storage}, reading {history}"
    )


def _causality(flags):
    if all(flags):
        return "causal rows"
    if not any(flags):
        return "non-causal rows"
    return "mixed causal and non-causal rows"


def _input_class(batch):
    """Return an input's ``(path, mask semantics)``.

    Provider choice depends on these properties and the layer, never on
    lengths, so together they name every call one choice applies to.
    """
    if isinstance(batch, DenseInput):
        path = "dense attention"
        mask = "causal" if batch.causal else "non-causal"
        if batch.mask is not None:
            mask += " with an explicit mask"
    elif isinstance(batch, PagedInput):
        path, mask = "paged attention", _causality(batch.causal)
    elif isinstance(batch, VarlenInput):
        path, mask = "variable-length attention", _causality(batch.causal)
    elif isinstance(batch, SegmentedInput):
        path = "segmented prefix read"
        mask = (
            "every current key visible"
            if batch.fully_visible_current
            else "per-query current key endpoints"
        )
    else:
        path = "visible-endpoint attention"
        mask = (
            "every key visible"
            if batch.fully_visible
            else "per-query key endpoints"
        )
    table = getattr(batch, "block_table", None)
    if table is not None and table.start_page is not None:
        mask += " over block tables starting after retired pages"
    return path, mask


def _paged_calls(page_tokens):
    """Return one-row samples of every paged call a cache layer receives.

    They are a causal and a non-causal paged row and a segmented prefix read
    whose queries see every current key, over ``page_tokens``-token pages.
    Routing depends only on their kind, causality and page size.
    """
    lengths = SequenceLengths.from_lengths((1,), device="cpu")
    table = BlockTable(torch.ones((1, 1), dtype=torch.int32), page_tokens)
    return (
        PagedInput(lengths, lengths, table, None, (True,)),
        PagedInput(lengths, lengths, table, None, (False,)),
        SegmentedInput(
            lengths,
            lengths,
            table,
            None,
            torch.ones((1, 1), dtype=torch.int32),
            True,
        ),
    )


def _names(batch, *, head_dim, window, architecture):
    """Name the providers that may read ``batch``, most preferred first.

    ``head_dim``, ``window`` and ``architecture`` (the CUDA major compute
    capability, or None off CUDA) describe the layer and the device; a
    provider must also be available for the layer to serve the call. Off
    CUDA the portable provider serves every call.
    """
    if architecture is None:
        return ("torch",)

    if isinstance(batch, DenseInput):
        if batch.mask is not None:
            # No native kernel consumes an arbitrary dense mask.
            return ()
        # The portable provider is a candidate only of the FP32
        # single-head layer class (``_portable``), whose non-causal calls
        # it evaluates.
        portable = () if batch.causal else ("torch",)
        return ("sgl_kernel", "flash_attn_4", *portable)
    if isinstance(batch, VarlenInput):
        # Packed Q/K/V need no cache plan. Native varlen kernels consume
        # offsets directly without reserving paged split-KV intermediates.
        return ("sgl_kernel", "flash_attn_4")

    # The SM100 head-dimension-256 FA4 kernel accepts neither per-sequence
    # key lengths nor mask functions, which paged, visible and segmented
    # inputs require.
    fa4_indexed = head_dim != 256
    if isinstance(batch, (VisibleInput, SegmentedInput)):
        # The prefix-block kernel reads a segmented prefix when every
        # query sees all of its current keys.
        block = (
            ("prefix_block",)
            if isinstance(batch, SegmentedInput) and batch.fully_visible_current
            else ()
        )
        # FA4 captures device-visible endpoints directly.
        fa4 = (
            ("flash_attn_4",)
            if architecture in (9, 10, 11) and fa4_indexed
            else ()
        )
        return (*block, *fa4)
    if isinstance(batch, PagedInput):
        # Non-causal blocks read their own keys and a prefix window with
        # the prefix-block kernel. TensorRT-LLM context kernels bound a
        # history window only along the causal diagonal.
        block = ("prefix_block",) if not any(batch.causal) else ()
        trtllm = ("trtllm",) if window is None or all(batch.causal) else ()
        ordinary = (
            ("sgl_kernel",) if batch.block_table.block_size % 256 == 0 else ()
        )
        fa4 = (
            ("flash_attn_4",)
            if architecture not in (8, 12) and fa4_indexed
            else ()
        )
        return (*block, *trtllm, *ordinary, *fa4)
    return ()


class _Automatic(_Operator):
    """Dispatching operator selecting native providers.

    Dispatching operator selecting a native provider for each numerical
    input. A paged batch whose causal and non-causal rows prefer different
    providers is evaluated as its contiguous runs of equal causality, each
    by its own provider, after the batch's cache write commits once.
    ``architecture`` is the CUDA major compute capability, or ``None`` off
    CUDA, where the portable provider serves every input.
    """

    # Retired-page tables reach only providers that read them.
    reads_retired_tables = True

    def __init__(self, providers, requirements, architecture, **kwargs):
        super().__init__(**kwargs)
        self._providers = providers
        self._requirements = requirements
        self._architecture = architecture
        self._operators = {}
        # Provider name by input class ("path: mask semantics"), in the
        # order the classes were first met.
        self._selections = {}
        self._arguments = kwargs

    def _name(self, batch):
        """Return the first available provider reading ``batch``, or None."""
        # A table with retired window pages is read only by providers that
        # consume its start pages; no other provider may substitute for one.
        table = getattr(batch, "block_table", None)
        retired = table is not None and table.start_page is not None
        return next(
            (
                name
                for name in _names(
                    batch,
                    head_dim=self.head_dim,
                    window=self.window,
                    architecture=self._architecture,
                )
                if name in self._providers
                and (
                    not retired
                    or self._providers[name].operator_class.reads_retired_tables
                )
            ),
            None,
        )

    def _runs(self, batch):
        """Whether a mixed-causality paged batch splits into causal runs.

        The providers of a mixed batch's causal and non-causal rows depend
        only on causality, not on lengths, so the decision needs no host
        values. A batch whose rows all prefer one provider stays whole.
        """
        if not isinstance(batch, PagedInput) or len(set(batch.causal)) < 2:
            return False
        rows = batch.queries.batch_size
        causal, blocks = (
            self._name(replace(batch, causal=(flag,) * rows))
            for flag in (True, False)
        )
        return causal != blocks

    def _describe(self, batch):
        """Name a call's path, shape, dtype and mask semantics."""
        path, mask = _input_class(batch)
        if not isinstance(batch, DenseInput):
            tokens = batch.queries.num_tokens
            path += f" of {batch.queries.batch_size} rows" + (
                "" if tokens is None else f" and {tokens} query tokens"
            )
        layer = _layer(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            cache=self.cache,
            window=self.window,
        )
        return (
            f"{path} on sm{self._architecture} with {mask}: {layer}; "
            f"available providers: {', '.join(sorted(self._providers))}"
        )

    def _operator(self, batch):
        name = self._name(batch)
        if name is None:
            raise ValueError(
                f"no native attention kernel serves {self._describe(batch)}"
            )
        selection = ": ".join(_input_class(batch))
        if selection not in self._selections:
            self._selections[selection] = name
            record_kernel_choice()

        if name not in self._operators:
            # Workspace buffers are namespaced per provider; the shared
            # scratch grant keeps its plain name.
            arguments = dict(self._arguments)
            arguments["workspace"] = {
                key: self.workspace[
                    "scratch" if key == "scratch" else f"{name}.{key}"
                ]
                for key in self._requirements[name]
            }
            self._operators[name] = self._providers[name].prepare(**arguments)

        return self._operators[name]

    def selections(self):
        # A mixed-causality batch evaluated as runs records each run's class.
        return MappingProxyType(dict(self._selections))

    def bind(self, batch):
        if self._closed:
            raise RuntimeError("attention operator is closed")
        if not self._runs(batch):
            # The selected operator owns the same dimensions and capacity and
            # validates them once together with its native preparation
            # requirements.
            self._operator(batch).bind(batch)
            return

        # Runs are sliced by exact host lengths, which the base binding
        # requires here. Rows split only between TensorRT-LLM (causal runs)
        # and the prefix-block kernel (non-causal runs); neither keeps a
        # per-batch plan, so one provider may bind several runs of a batch.
        super().bind(batch)
        for _, _, run in causal_runs(batch):
            self._operator(run).bind(run)

    def requires_host_lengths(self, batch):
        if self._runs(batch):
            return True
        return self._operator(batch).requires_host_lengths(batch)

    def __call__(self, q, k, v, batch, *, scale, out):
        if not self._runs(batch):
            return self._operator(batch)(q, k, v, batch, scale=scale, out=out)

        self._validate(q, k, v, batch, out)
        if q.ndim != 3 or k.ndim != 3:
            raise ValueError(
                "mixed-causality attention requires packed query and current "
                "K/V rows"
            )
        # A later run may read rows an earlier run of the same sequence
        # writes, so the batch's complete write commits before any run; the
        # runs carry no write addresses.
        if batch.write_indices is not None:
            self.update_cache(k, v, indices=batch.write_indices)
        for rows, _, run in causal_runs(batch):
            if rows.start == rows.stop:
                continue
            # Current K/V share the query rows of their run.
            self._operator(run)(
                q[rows], k[rows], v[rows], run, scale=scale, out=out[rows]
            )
        return out

    def close(self):
        for operator in self._operators.values():
            operator.close()
        self._operators.clear()
        self._arguments.clear()
        super().close()


class Backend(_Backend):
    """Provider factory probing installed native libraries.

    ``flashinfer``, a configured FlashInfer provider, sizes the scratch grant
    of the TensorRT-LLM kernels that ship with FlashInfer; FlashInfer's own
    attention kernels are not candidates.
    """

    def __init__(self, device, *, flashinfer=None):
        self.device = device
        self._architecture = (
            torch.cuda.get_device_capability(device)[0]
            if device.type == "cuda"
            else None
        )
        # The portable provider serves every call off CUDA and, on CUDA, the
        # one call class no native kernel computes (``_portable``).
        self._portable = import_module(f"{__package__}.torch").Backend()
        self._native = {}
        if device.type != "cuda":
            self.name = f"auto:{device.type}:torch"
            return

        # Probe each native library's entry points; uninstalled or incomplete
        # builds never become selection candidates.
        if self._architecture == 9 and _available(
            "sgl_kernel.flash_attn",
            ("flash_attn_varlen_func", "flash_attn_with_kvcache"),
        ):
            self._native["sgl_kernel"] = import_module(
                f"{__package__}.sgl_kernel"
            ).Backend()
        if (
            self._architecture == 10
            and _available(
                "flashinfer.decode", ("trtllm_batch_decode_with_kv_cache",)
            )
            and _available(
                "flashinfer.prefill", ("trtllm_batch_context_with_kv_cache",)
            )
        ):
            trtllm = import_module(f"{__package__}.trtllm")
            self._native["trtllm"] = (
                trtllm.Backend()
                if flashinfer is None
                else trtllm.Backend(
                    workspace_size=flashinfer.config.workspace_size
                )
            )
        from uniserve_kernels.attention import prefix_block

        if prefix_block.available(device):
            self._native["prefix_block"] = import_module(
                f"{__package__}.prefix_block"
            ).Backend()
        from .flash_attn_4 import available

        if available():
            self._native["flash_attn_4"] = import_module(
                f"{__package__}.flash_attn_4"
            ).Backend()

        # Replacement compatibility includes the resolved provider policy.
        self.name = f"auto:sm{self._architecture}:" + ",".join(
            sorted(self._native)
        )

    def _providers(
        self, dtype, head_dim, pages, window, num_heads, num_kv_heads
    ):
        """Return the available providers that serve a layer's shape.

        ``pages`` describes the layer's paged prefix cache, or is None for
        a layer without one.
        """
        if self.device.type != "cuda" or _portable(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            pages=pages,
            window=window,
        ):
            return {"torch": self._portable}

        # Native kernels require half precision and unquantized cache state.
        if dtype not in {torch.float16, torch.bfloat16} or (
            pages is not None and pages.quantized
        ):
            return {}

        from .prefix_block import unsupported

        # Restrict candidates to each kernel's head-dimension, cache-block and
        # history-window constraints. The pinned SM100 TensorRT-LLM cubins
        # specialize head dimension 512 context kernels only for causal and
        # dense masks over 16-, 32- and 64-token pages; FlashAttention
        # kernels do not take UniServe's history windows.
        return {
            name: backend
            for name, backend in self._native.items()
            if (
                name != "trtllm"
                or head_dim in {64, 128, 256}
                or (
                    head_dim == 512
                    and window is None
                    and pages is not None
                    and pages.page_tokens in {16, 32, 64}
                )
            )
            and (
                name != "prefix_block"
                or unsupported(
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    dtype=dtype,
                    pages=pages,
                )
                is None
            )
            and (window is None or name not in {"sgl_kernel", "flash_attn_4"})
            and (
                name != "sgl_kernel"
                or (
                    head_dim <= 256
                    and head_dim % 8 == 0
                    and (pages is None or pages.page_tokens % 256 == 0)
                )
            )
            and (
                name != "flash_attn_4"
                or (
                    head_dim % 8 == 0
                    and (
                        head_dim <= 128
                        or head_dim == 256
                        or (self._architecture == 9 and head_dim <= 256)
                    )
                )
            )
        }

    def reads_pages(
        self, *, num_heads, num_kv_heads, head_dim, dtype, window, pages
    ):
        """Report whether a provider serves each paged call of a layer.

        A layer on ``pages`` receives causal and non-causal paged calls and
        segmented reads of its prefix whose queries see every current key;
        each must reach an available provider that serves the layer's shape,
        as the prepared operator selects one per call.
        """
        providers = self._providers(
            dtype, head_dim, pages, window, num_heads, num_kv_heads
        )
        return all(
            any(
                name in providers
                for name in _names(
                    call,
                    head_dim=head_dim,
                    window=window,
                    architecture=self._architecture,
                )
            )
            for call in _paged_calls(pages.page_tokens)
        )

    def _requirements(self, arguments):
        cache = arguments["cache"]
        providers = self._providers(
            arguments["dtype"],
            arguments["head_dim"],
            None if cache is None else CachePages.of(cache),
            arguments.get("window"),
            arguments["num_heads"],
            arguments["num_kv_heads"],
        )
        requirements = {
            name: backend.workspace_buffers(**arguments)
            for name, backend in providers.items()
        }
        return providers, requirements

    def workspace_buffers(self, **kwargs):
        _, requirements = self._requirements(kwargs)
        result: dict[str, BufferConfig] = {}

        # Providers share the scratch grant; other buffers are namespaced per
        # provider so independently prepared operators cannot collide.
        for name, buffers in requirements.items():
            for key, config in buffers.items():
                target = "scratch" if key == "scratch" else f"{name}.{key}"
                if target in result and result[target] != config:
                    raise ValueError(
                        "native attention scratch declarations must agree"
                    )
                result[target] = config
        return result

    def prepare(self, **kwargs):
        arguments = {
            key: value for key, value in kwargs.items() if key != "workspace"
        }
        providers, requirements = self._requirements(arguments)
        if not providers:
            layer = _layer(
                num_heads=arguments["num_heads"],
                num_kv_heads=arguments["num_kv_heads"],
                head_dim=arguments["head_dim"],
                dtype=arguments["dtype"],
                cache=arguments["cache"],
                window=arguments.get("window"),
            )
            raise ValueError(
                f"no native attention kernel on sm{self._architecture} "
                f"serves {layer}"
            )
        return _Automatic(providers, requirements, self._architecture, **kwargs)
