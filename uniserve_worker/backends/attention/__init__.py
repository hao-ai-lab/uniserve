"""Construction of worker-owned attention implementations."""

from __future__ import annotations

from collections.abc import Callable

from ...execution.forward_batch import AttentionBackend, AttentionSelection
from ...foundation.errors import capability_mismatch
from ...foundation.runtime_config import FlashInferTuningConfig
from .base import AttentionCapabilities

__all__ = [
    "AttentionBackend",
    "AttentionCapabilities",
    "ATTENTION_BACKENDS",
    "resolve_attention_selection",
]


ATTENTION_BACKENDS = (
    "auto",
    "trtllm_mha",
    "sgl_kernel",
    "flashinfer",
    "flash_attn",
    "fa4_cute",
    "torch_sdpa",
)


def _constructors(
    tuning: FlashInferTuningConfig,
) -> tuple[tuple[str, Callable[[], AttentionBackend]], ...]:
    from .fa4_cute import Fa4CuteAttentionBackend
    from .flash_attn import FlashAttentionBackend
    from .flashinfer import FlashInferAttentionBackend
    from .sgl_kernel import SglKernelAttentionBackend
    from .torch_sdpa import TorchSDPAAttentionBackend
    from .trtllm_mha import TRTLLMMHAAttentionBackend

    return (
        ("trtllm_mha", lambda: TRTLLMMHAAttentionBackend(tuning=tuning)),
        ("sgl_kernel", SglKernelAttentionBackend),
        ("flashinfer", lambda: FlashInferAttentionBackend(tuning=tuning)),
        ("flash_attn", FlashAttentionBackend),
        ("fa4_cute", Fa4CuteAttentionBackend),
        ("torch_sdpa", TorchSDPAAttentionBackend),
    )


def resolve_attention_selection(
    name: str,
    *,
    tuning: FlashInferTuningConfig,
    block_size: int,
) -> AttentionSelection:
    """Resolve one canonical deployment choice without process-global state."""

    requested = str(name)
    if requested not in ATTENTION_BACKENDS:
        raise capability_mismatch(
            f"unknown attention backend {requested!r}; expected one of {ATTENTION_BACKENDS!r}"
        )
    available: list[AttentionBackend] = []
    for candidate, construct in _constructors(tuning):
        if requested != "auto" and candidate != requested:
            continue
        backend = construct()
        capabilities = backend.capabilities()
        multiple = int(getattr(capabilities, "paged_block_size_multiple", 1) or 1)
        if bool(getattr(capabilities, "available", True)) and int(block_size) % multiple == 0:
            available.append(backend)
    if not available:
        raise capability_mismatch(f"attention backend {requested!r} is unavailable")
    identity = requested if requested != "auto" else "+".join(value.name for value in available)
    return AttentionSelection(identity=identity, providers=tuple(available))
