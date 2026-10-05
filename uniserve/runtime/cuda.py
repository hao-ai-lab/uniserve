"""CUDA driver results and numerical views of native streams."""

from __future__ import annotations

from typing import Any

import torch

from uniserve_worker._uniserve_ipc import CUDAStream as NativeStream


class CUDAError(RuntimeError):
    """A CUDA resource or call could not satisfy its requirements."""

    def __init__(self, message: str, *, code: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message


def driver() -> Any:
    """Import the CUDA driver bindings used by numerical backends."""
    try:
        from cuda.bindings import (
            driver,  # pyright: ignore[reportAttributeAccessIssue]
        )
    except (
        ImportError
    ) as error:  # pragma: no cover - CUDA configurations install cuda-python.
        raise CUDAError("CUDA driver calls require cuda-python") from error
    return driver


def cuda_status(result: tuple[Any, ...], call: str) -> None:
    """Raise CUDAError with the decoded driver failure, if any."""
    cu = driver()
    if result[0] == cu.CUresult.CUDA_SUCCESS:
        return

    name_result = cu.cuGetErrorName(result[0])
    name = (
        str(name_result[1])
        if name_result[0] == cu.CUresult.CUDA_SUCCESS
        else str(result[0])
    )
    message_result = cu.cuGetErrorString(result[0])
    message = (
        str(message_result[1])
        if message_result[0] == cu.CUresult.CUDA_SUCCESS
        else name
    )
    raise CUDAError(f"{call} failed: {name}: {message}", code=int(result[0]))


def cuda_value(result: tuple[Any, ...], call: str) -> Any:
    """Extract the sole value from a successful CUDA driver result."""
    cuda_status(result, call)
    return result[1]


def create_sibling_stream(
    stream: torch.cuda.Stream,
    owner: NativeStream | None = None,
) -> tuple[NativeStream, torch.cuda.ExternalStream]:
    """Own a stream in the origin's context and return its PyTorch view.

    Forking an owned stream retains its SM partition. A caller supplying only
    a PyTorch view retains the origin's context until the new stream closes.
    """
    native = (
        owner.fork()
        if owner is not None
        else NativeStream.sibling(stream.device.index, stream.cuda_stream)
    )
    return native, torch.cuda.ExternalStream(
        native.handle, device=stream.device
    )
