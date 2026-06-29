"""Attention backend registry package.

Consumers look up concrete backend instances by name, while runtime capability
selection lives in ``uniserve_worker.ops``. Concrete backend classes are
deliberately not re-exported here; import the submodule directly if a test needs
a concrete type. Available backends register themselves as an import side effect.
"""
import logging

from . import registry
from . import torch_sdpa as _torch_sdpa  # noqa: F401  # registers the portable default backend
from .base import AttentionBackend, AttentionCapabilities
from .registry import (
    get_attention_backend,
    has_attention_backend,
    list_attention_backends,
    normalize_attention_backend_name,
    register_attention_backend,
)

_log = logging.getLogger("uniserve.attention")


def _load_optional_backend(module: str, attr: str):
    """Import an optional attention backend, returning None when its dependency
    is absent. A genuinely missing optional dependency (ImportError) is silent;
    any other failure means the backend module exists but is broken/misconfigured,
    so it is logged at warning level (with traceback) instead of being swallowed."""
    try:  # pragma: no cover - optional dependency unavailable/misconfigured.
        mod = __import__(f"{__name__}.{module}", fromlist=[attr])
        return getattr(mod, attr)
    except ImportError:  # pragma: no cover - optional dependency simply not installed.
        return None
    except Exception:  # pragma: no cover - real misconfiguration; surface it.
        _log.warning(
            "Optional attention backend %r failed to import; treating as unavailable.",
            module,
            exc_info=True,
        )
        return None


_OPTIONAL_BACKENDS = (
    ("flashinfer", "FlashInferAttentionBackend"),
    ("fa4_cute", "Fa4CuteAttentionBackend"),
    ("flash_attn", "FlashAttentionBackend"),
    ("sgl_kernel", "SglKernelAttentionBackend"),
)


def init_attention_backends() -> tuple[str, ...]:
    """Ensure every available attention backend is registered.

    Registration also runs at import time; this entry point is safe to call repeatedly.
    Returns sorted registered backend names.
    """
    for module, attr in _OPTIONAL_BACKENDS:
        _load_optional_backend(module, attr)
    return list_attention_backends()


# Register every available optional backend as an import side effect; the
# portable ``torch_sdpa`` default is registered by importing its module above.
init_attention_backends()
registry._snapshot_default_backends_for_testing()

__all__ = [
    "AttentionBackend",
    "AttentionCapabilities",
    "get_attention_backend",
    "has_attention_backend",
    "init_attention_backends",
    "list_attention_backends",
    "normalize_attention_backend_name",
    "register_attention_backend",
]
