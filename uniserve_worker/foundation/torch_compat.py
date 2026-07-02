"""Small torch runtime-introspection helpers shared across layers."""
from __future__ import annotations

try:  # torch is an optional import for CPU-only control-plane tests.
    import torch
except Exception:  # pragma: no cover - exercised only in torch-free envs.
    torch = None  # type: ignore[assignment]

__all__ = ["torch_is_compiling"]


def torch_is_compiling() -> bool:
    """True while torch.compile/dynamo is tracing.

    Hot paths use this to skip work that breaks or pollutes a trace (pinned
    allocations, wall-clock stats). Resolved via ``getattr`` so it degrades to
    ``False`` on torch builds without the compiler namespace.
    """
    if torch is None:
        return False
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    if callable(is_compiling):
        return bool(is_compiling())
    is_compiling = getattr(getattr(torch, "_dynamo", None), "is_compiling", None)
    return bool(is_compiling()) if callable(is_compiling) else False
