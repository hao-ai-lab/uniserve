"""CUDA Green Context wrapper for SM-partitioned concurrent streams.

Wraps the CUDA driver API (``cuGreenCtxCreate`` / ``cuDevSmResourceSplitByCount``
/ ``cuGreenCtxStreamCreate``) so a device's SMs can be split between two CUDA
streams, letting prefill and decode execute concurrently with hardware-enforced
compute isolation. The created streams are wrapped as ``torch.cuda.ExternalStream``
so the rest of the worker drives them as ordinary torch streams.

Capability requirements (CUDA Green Contexts):
- CUDA >= 12.4 for ``cuGreenCtxCreate``
- CUDA >= 12.5 for the native ``cuGreenCtxStreamCreate`` used here

Everything degrades gracefully: when the ``cuda.bindings`` driver module is
absent, the driver is too old, or partitioning fails, the helpers raise
:class:`GreenContextError` and the :class:`~uniserve_worker.runtime.stream_manager.StreamManager`
falls back to plain full-SM torch streams. The feature is gated by the manager on
``UNISERVE_GREEN_CONTEXTS`` and a capability probe.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "GreenContextError",
    "GreenContextStreams",
    "green_contexts_available",
    "get_sm_available",
    "create_greenctx_streams",
]

logger = logging.getLogger(__name__)

# CU_STREAM_NON_BLOCKING — required by cuGreenCtxStreamCreate so the green-context
# stream performs no implicit sync against the legacy default stream.
_CU_STREAM_NON_BLOCKING = 1


class GreenContextError(RuntimeError):
    """A green-context driver call failed or the capability is unavailable."""


def _driver() -> Any | None:
    """Return the CUDA driver bindings module, or ``None`` if unavailable."""
    try:
        from cuda.bindings import driver

        return driver
    except Exception:  # noqa: BLE001 - optional dependency / old cuda-python.
        try:
            from cuda import cuda as driver  # deprecated alias on old cuda-python

            return driver
        except Exception:  # noqa: BLE001
            return None


_REQUIRED_SYMBOLS = (
    "cuInit",
    "cuDeviceGet",
    "cuDeviceGetDevResource",
    "cuDevSmResourceSplitByCount",
    "cuDevResourceGenerateDesc",
    "cuGreenCtxCreate",
    "cuGreenCtxStreamCreate",
)


def green_contexts_available() -> bool:
    """Whether green-context SM partitioning can be used on this build."""
    driver = _driver()
    if driver is None:
        return False
    if not all(hasattr(driver, name) for name in _REQUIRED_SYMBOLS):
        return False
    return hasattr(torch.cuda, "ExternalStream")


def _unwrap(ret: Any, name: str) -> tuple:
    """Split a ``(CUresult, *outputs)`` driver return, raising on error."""
    if not isinstance(ret, tuple):
        ret = (ret,)
    err = ret[0]
    if int(err) != 0:
        raise GreenContextError(f"{name} failed: {err!r}")
    return ret[1:]


def _ensure_init(driver: Any) -> None:
    _unwrap(driver.cuInit(0), "cuInit")


def get_sm_available(device_index: int) -> int:
    """Total SM count on ``device_index`` via ``cuDeviceGetDevResource``."""
    driver = _driver()
    if driver is None:
        raise GreenContextError("cuda driver bindings unavailable")
    _ensure_init(driver)
    (dev,) = _unwrap(driver.cuDeviceGet(device_index), "cuDeviceGet")
    (resource,) = _unwrap(
        driver.cuDeviceGetDevResource(
            dev, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        ),
        "cuDeviceGetDevResource",
    )
    return int(resource.sm.smCount)


@dataclass
class GreenContextStreams:
    """A prefill/decode stream pair backed by disjoint SM partitions.

    Holds the underlying green-context + raw-stream handles alive for the pair's
    lifetime: dropping them would let the driver reclaim the contexts and
    invalidate the wrapped ``ExternalStream``s.
    """

    prefill: "torch.cuda.Stream"
    decode: "torch.cuda.Stream"
    prefill_sm: int
    decode_sm: int
    _keepalive: list[Any] = field(default_factory=list, repr=False)


def _split_off(driver: Any, resource: Any, count: int, label: str) -> tuple[Any, Any]:
    """Split ``count`` SMs off ``resource`` → (partition, remaining)."""
    result, _nb_groups, remaining = _unwrap(
        driver.cuDevSmResourceSplitByCount(1, resource, 0, int(count)),
        f"cuDevSmResourceSplitByCount[{label}]",
    )
    if not result:
        raise GreenContextError(f"SM split for {label} produced no partition")
    partition = result[0] if isinstance(result, (list, tuple)) else result
    return partition, remaining


def _sm_count(partition: Any) -> int:
    return int(partition.sm.smCount)


def _stream_for_partition(
    driver: Any, dev: Any, partition: Any, device_index: int, keepalive: list[Any]
) -> "torch.cuda.Stream":
    """Build a torch ExternalStream over a green context for one SM partition."""
    (desc,) = _unwrap(
        driver.cuDevResourceGenerateDesc([partition], 1), "cuDevResourceGenerateDesc"
    )
    (green_ctx,) = _unwrap(
        driver.cuGreenCtxCreate(
            desc, dev, driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
        ),
        "cuGreenCtxCreate",
    )
    (raw_stream,) = _unwrap(
        driver.cuGreenCtxStreamCreate(green_ctx, _CU_STREAM_NON_BLOCKING, 0),
        "cuGreenCtxStreamCreate",
    )
    # Keep the descriptor, context and raw stream referenced for the pair's life.
    keepalive.extend([desc, green_ctx, raw_stream])
    return torch.cuda.ExternalStream(int(raw_stream), device=device_index)


def create_greenctx_streams(
    prefill_sm: int, decode_sm: int, device_index: int
) -> GreenContextStreams:
    """Create a prefill/decode stream pair on disjoint SM partitions.

    ``prefill_sm`` SMs are split off the device for the prefill stream and
    ``decode_sm`` SMs from the remainder for the decode stream. Raises
    :class:`GreenContextError` on any driver failure so the caller can fall back.
    """
    driver = _driver()
    if driver is None:
        raise GreenContextError("cuda driver bindings unavailable")
    if prefill_sm <= 0 or decode_sm <= 0:
        raise GreenContextError(
            f"green-context partition needs positive SM counts, got "
            f"prefill={prefill_sm} decode={decode_sm}"
        )
    _ensure_init(driver)
    (dev,) = _unwrap(driver.cuDeviceGet(device_index), "cuDeviceGet")
    (resource,) = _unwrap(
        driver.cuDeviceGetDevResource(
            dev, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        ),
        "cuDeviceGetDevResource",
    )
    keepalive: list[Any] = []
    # Split the prefill partition off the device; the decode partition is the
    # remainder. Splitting the remainder a *second* time (rather than using it
    # directly) is rejected by the driver with CUDA_ERROR_INVALID_RESOURCE_
    # CONFIGURATION, so the remainder is consumed as-is. The divisions guarantee
    # prefill_sm + decode_sm == total, so the remainder already holds decode_sm.
    prefill_part, remaining = _split_off(driver, resource, prefill_sm, "prefill")
    decode_part = remaining
    if _sm_count(decode_part) > decode_sm + 1:
        # Caller asked for fewer decode SMs than the remainder: carve exactly
        # decode_sm off the remainder (leaving the rest idle).
        decode_part, _ = _split_off(driver, remaining, decode_sm, "decode")
    prefill_stream = _stream_for_partition(driver, dev, prefill_part, device_index, keepalive)
    decode_stream = _stream_for_partition(driver, dev, decode_part, device_index, keepalive)
    return GreenContextStreams(
        prefill=prefill_stream,
        decode=decode_stream,
        prefill_sm=_sm_count(prefill_part),
        decode_sm=_sm_count(decode_part),
        _keepalive=keepalive,
    )
