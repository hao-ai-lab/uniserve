"""Select native attention by representation and input.

Select native attention from the actual representation and numerical input.
"""

from importlib import import_module

import torch

from uniserve.nn.attention.inputs import (
    DenseInput,
    PagedInput,
    SegmentedInput,
    VarlenInput,
    VisibleInput,
)
from uniserve.quantization import QuantizedTensor

from . import Backend as _Backend
from . import Operator as _Operator


def _available(module, names):
    try:
        library = import_module(module)
    except (ImportError, OSError):
        return False
    return all(callable(getattr(library, name, None)) for name in names)


class _Automatic(_Operator):
    """Dispatching operator selecting native providers.

    Dispatching operator selecting a native provider for each numerical
    input.
    """

    def __init__(self, providers, requirements, architecture, **kwargs):
        super().__init__(**kwargs)
        self._providers = providers
        self._requirements = requirements
        self._architecture = architecture
        self._operators = {}
        self._arguments = kwargs

    def _names(self, batch, *, ndim=None):
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
        if isinstance(batch, (VisibleInput, SegmentedInput)):
            # Device-visible endpoints are captured directly by FA4. This
            # retains the packed-attention preference for native mask kernels.
            fa4 = ("flash_attn_4",) if self._architecture in (9, 10, 11) else ()
            return (*fa4, "flashinfer", "torch")
        if isinstance(batch, PagedInput):
            block_size = batch.block_table.block_size
            ordinary = (
                ("sgl_kernel", "flash_attn") if block_size % 256 == 0 else ()
            )
            fa4 = ("flash_attn_4",) if self._architecture not in (8, 12) else ()
            return (
                "trtllm",
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

    def _operator(self, batch, *, ndim=None):
        name = next(
            name
            for name in self._names(batch, ndim=ndim)
            if name in self._providers
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
        # The selected operator owns the same dimensions and capacity and
        # validates them once together with its native preparation requirements.
        self._operator(batch).bind(batch)

    def requires_host_lengths(self, batch):
        return self._operator(batch).requires_host_lengths(batch)

    def __call__(self, q, k, v, batch, *, scale, out):
        return self._operator(batch, ndim=q.ndim)(
            q, k, v, batch, scale=scale, out=out
        )

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
        from .flash_attn_4 import available

        if available():
            self._factories["flash_attn_4"] = import_module(
                f"{__package__}.flash_attn_4"
            ).Backend()

        # Replacement compatibility includes the resolved provider policy.
        self.name = f"auto:sm{self._architecture}:" + ",".join(
            sorted(self._factories)
        )

    def _providers(self, dtype, head_dim, cache):
        # Native kernels require half precision and unquantized cache state.
        if dtype not in {torch.float16, torch.bfloat16} or (
            cache is not None and isinstance(cache.key, QuantizedTensor)
        ):
            return {"torch": self._factories["torch"]}

        # Restrict candidates to each kernel's head-dimension and cache-block
        # constraints.
        return {
            name: backend
            for name, backend in self._factories.items()
            if (name != "flashinfer" or head_dim in {64, 128, 256, 512})
            and (name != "trtllm" or head_dim in {64, 128, 256})
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
                or cache is None
                or cache.block_size % 256 == 0
            )
        }

    def _requirements(self, arguments):
        providers = self._providers(
            arguments["dtype"], arguments["head_dim"], arguments["cache"]
        )
        requirements = {
            name: backend.workspace_buffers(**arguments)
            for name, backend in providers.items()
        }
        return providers, requirements

    def workspace_buffers(self, **kwargs):
        _, requirements = self._requirements(kwargs)
        result = {}

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
        return _Automatic(providers, requirements, self._architecture, **kwargs)
