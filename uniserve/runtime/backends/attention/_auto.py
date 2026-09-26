"""Select native attention by representation and input.

Select native attention from the actual representation and numerical input.

History-windowed attention and head dimension 512 have complete native
coverage on SM100: TensorRT-LLM context kernels evaluate causal paged rows,
and the prefix-block kernel evaluates non-causal paged blocks and segmented
prefix reads. On CUDA, FlashInfer, the FlashAttention-2 library and the
portable torch provider therefore never stand in for these calls: a call no
native kernel serves raises when its layer is prepared or bound, naming the
call's path, shape, dtype and mask semantics.
"""

from dataclasses import replace
from importlib import import_module

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

from . import Backend as _Backend
from . import CachePages
from . import Operator as _Operator
from ._sequences import causal_runs

# Providers that substitute for missing native coverage. They serve only
# calls outside the natively covered classes (see ``_native_only``).
_FALLBACKS = frozenset({"flash_attn", "flashinfer", "torch"})


def _available(module, names):
    try:
        library = import_module(module)
    except (ImportError, OSError):
        return False
    return all(callable(getattr(library, name, None)) for name in names)


def _native_only(device, head_dim, window):
    """Whether only native kernels may serve a layer's calls on ``device``.

    Windowed history and head dimension 512 are the call classes whose
    native coverage is complete; the fallbacks remain candidates for the
    others.
    """
    return device.type == "cuda" and (window is not None or head_dim == 512)


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


def _causality(flags):
    if all(flags):
        return "causal rows"
    if not any(flags):
        return "non-causal rows"
    return "mixed causal and non-causal rows"


def _names(batch, *, head_dim, window, architecture, ndim=None):
    """Name the providers that may read ``batch``, most preferred first.

    ``head_dim``, ``window`` and ``architecture`` describe the layer and
    the device; a provider must also be available for the layer to serve
    the call.
    """
    if isinstance(batch, DenseInput) and batch.mask is not None:
        return ("torch",)
    if isinstance(batch, DenseInput) and ndim == 4:
        # Keep the explicit batch in one native invocation. FlashInfer's
        # single-prefill path otherwise serializes its individual samples.
        return (
            "sgl_kernel",
            "flash_attn",
            "flash_attn_4",
            "flashinfer",
            "torch",
        )
    if isinstance(batch, VarlenInput):
        # Packed Q/K/V need no cache plan. Native varlen kernels consume
        # offsets directly without reserving paged split-KV intermediates.
        return (
            "sgl_kernel",
            "flash_attn",
            "flash_attn_4",
            "flashinfer",
            "torch",
        )
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
        # Device-visible endpoints are captured directly by FA4. This
        # retains the packed-attention preference for native mask kernels.
        fa4 = (
            ("flash_attn_4",)
            if architecture in (9, 10, 11) and fa4_indexed
            else ()
        )
        return (*block, *fa4, "flashinfer", "torch")
    if isinstance(batch, PagedInput):
        block_size = batch.block_table.block_size
        ordinary = ("sgl_kernel", "flash_attn") if block_size % 256 == 0 else ()
        fa4 = (
            ("flash_attn_4",)
            if architecture not in (8, 12) and fa4_indexed
            else ()
        )
        # Non-causal blocks read their own keys and a prefix window with
        # the prefix-block kernel. TensorRT-LLM context kernels bound a
        # history window only along the causal diagonal.
        block = ("prefix_block",) if not any(batch.causal) else ()
        trtllm = ("trtllm",) if window is None or all(batch.causal) else ()
        return (
            *block,
            *trtllm,
            *ordinary[:1],
            "flashinfer",
            *ordinary[1:],
            *fa4,
            "torch",
        )
    return (
        "sgl_kernel",
        "flashinfer",
        "flash_attn",
        "flash_attn_4",
        "torch",
    )


class _Automatic(_Operator):
    """Dispatching operator selecting native providers.

    Dispatching operator selecting a native provider for each numerical
    input. A paged batch whose causal and non-causal rows prefer different
    providers is evaluated as its contiguous runs of equal causality, each
    by its own provider, after the batch's cache write commits once.
    """

    # Retired-page tables reach only providers that read them.
    reads_retired_tables = True

    def __init__(self, providers, requirements, architecture, **kwargs):
        super().__init__(**kwargs)
        self._providers = providers
        self._requirements = requirements
        self._architecture = architecture
        self._operators = {}
        self._arguments = kwargs

    def _name(self, batch, *, ndim=None):
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
                    ndim=ndim,
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
        if isinstance(batch, DenseInput):
            path = "dense attention"
            mask = "causal" if batch.causal else "non-causal"
            if batch.mask is not None:
                mask += " with an explicit mask"
        else:
            if isinstance(batch, PagedInput):
                path, mask = "paged attention", _causality(batch.causal)
            elif isinstance(batch, VarlenInput):
                path, mask = (
                    "variable-length attention",
                    _causality(batch.causal),
                )
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
            tokens = batch.queries.num_tokens
            path += f" of {batch.queries.batch_size} rows" + (
                "" if tokens is None else f" and {tokens} query tokens"
            )
        table = getattr(batch, "block_table", None)
        if table is not None and table.start_page is not None:
            mask += " over block tables starting after retired pages"
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

    def _operator(self, batch, *, ndim=None):
        name = self._name(batch, ndim=ndim)
        if name is None:
            raise ValueError(
                f"no available attention kernel serves {self._describe(batch)}"
            )

        if name not in self._operators:
            # Workspace buffers are namespaced per provider; the shared
            # FlashInfer scratch grant keeps its plain name for TensorRT-LLM.
            arguments = dict(self._arguments)
            arguments["workspace"] = {
                key: self.workspace[
                    "scratch" if key == "scratch" else f"{name}.{key}"
                ]
                for key in self._requirements[name]
            }
            self._operators[name] = self._providers[name].prepare(**arguments)

        return self._operators[name]

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
            return self._operator(batch, ndim=q.ndim)(
                q, k, v, batch, scale=scale, out=out
            )

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

    Provider factory probing the native libraries installed for this device.
    """

    def __init__(self, device, *, flashinfer=None):
        self.device = device
        self._architecture = (
            torch.cuda.get_device_capability(device)[0]
            if device.type == "cuda"
            else None
        )
        self._factories = {
            "torch": import_module(f"{__package__}.torch").Backend()
        }
        if device.type != "cuda":
            self.name = f"auto:{device.type}:torch"
            return

        # Probe each native library's entry points; uninstalled or incomplete
        # builds never become selection candidates.
        libraries = {
            "sgl_kernel": (
                "sgl_kernel.flash_attn",
                ("flash_attn_varlen_func", "flash_attn_with_kvcache"),
            ),
            "flashinfer": (
                "flashinfer",
                (
                    "BatchPrefillWithPagedKVCacheWrapper",
                    "BatchDecodeWithPagedKVCacheWrapper",
                ),
            ),
            "flash_attn": (
                "flash_attn",
                (
                    "flash_attn_func",
                    "flash_attn_varlen_func",
                    "flash_attn_with_kvcache",
                ),
            ),
        }
        for name, (module, functions) in libraries.items():
            if name == "sgl_kernel" and self._architecture != 9:
                continue
            if _available(module, functions):
                self._factories[name] = (
                    flashinfer
                    if name == "flashinfer" and flashinfer is not None
                    else import_module(f"{__package__}.{name}").Backend()
                )
        if self._architecture == 10 and "flashinfer" in self._factories:
            self._factories["trtllm"] = import_module(
                f"{__package__}.trtllm"
            ).Backend(
                workspace_size=self._factories[
                    "flashinfer"
                ].config.workspace_size
            )
        from uniserve_kernels.attention import prefix_block

        if prefix_block.available(device):
            self._factories["prefix_block"] = import_module(
                f"{__package__}.prefix_block"
            ).Backend()
        from .flash_attn_4 import available

        if available():
            self._factories["flash_attn_4"] = import_module(
                f"{__package__}.flash_attn_4"
            ).Backend()

        # Replacement compatibility includes the resolved provider policy.
        self.name = f"auto:sm{self._architecture}:" + ",".join(
            sorted(self._factories)
        )

    def _providers(
        self, dtype, head_dim, pages, window, num_heads, num_kv_heads
    ):
        """Return the available providers that serve a layer's shape.

        ``pages`` describes the layer's paged prefix cache, or is None for
        a layer without one.
        """
        from .prefix_block import unsupported

        native_only = _native_only(self.device, head_dim, window)

        # Native kernels require half precision and unquantized cache state.
        if dtype not in {torch.float16, torch.bfloat16} or (
            pages is not None and pages.quantized
        ):
            return {} if native_only else {"torch": self._factories["torch"]}

        # Restrict candidates to each kernel's head-dimension, cache-block and
        # history-window constraints. The pinned SM100 TensorRT-LLM cubins
        # specialize head dimension 512 context kernels only for causal and
        # dense masks over 16-, 32- and 64-token pages; FlashAttention
        # providers do not take UniServe's history windows.
        return {
            name: backend
            for name, backend in self._factories.items()
            if not (native_only and name in _FALLBACKS)
            and (name != "flashinfer" or head_dim in {64, 128, 256, 512})
            and (
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
            and (
                window is None
                or name not in {"sgl_kernel", "flash_attn", "flash_attn_4"}
            )
            and (
                name not in {"sgl_kernel", "flash_attn"}
                or (head_dim <= 256 and head_dim % 8 == 0)
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
            and (
                name not in {"sgl_kernel", "flash_attn"}
                or pages is None
                or pages.page_tokens % 256 == 0
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
