"""Numerical contexts for cooperative native microbatch execution.

Each context has independent CUDA input, scratch and expert buffers. Rust
owns persistent host threads and their cooperative turns. The same ordinary
model calls execute eagerly or under CUDA Graph capture; captured replay
does not run the host threads again.
"""

import torch

from uniserve_worker._uniserve_ipc import (
    Microbatches as Microbatches,
)
from uniserve_worker._uniserve_ipc import (
    yield_microbatch as yield_microbatch,
)


def _prepare(contexts):
    if any(context.stream is None for context in contexts):
        raise ValueError("microbatches require explicit CUDA streams")
    device = contexts[0].stream.device
    streams = [context.stream.stream for context in contexts]
    if any(stream.device != device for stream in streams) or len(
        {stream.cuda_stream for stream in streams}
    ) != len(streams):
        raise ValueError("microbatches require distinct streams on one GPU")

    exchanges = [
        context.experts for context in contexts if context.experts is not None
    ]
    if exchanges:
        exchanges[0].bind_microbatches(exchanges)
    return device


def _warm(context):
    with torch.cuda.device(context.stream.device), context.activate():
        torch.cuda.current_blas_handle()


@torch.inference_mode()
def _execute(context, call):
    with torch.cuda.device(context.stream.device), context.activate():
        return call()


def _fork(contexts, device):
    current = torch.cuda.current_stream(device)
    for context in contexts:
        context.stream.wait(current)
    return current


def _join(current, contexts):
    for context in contexts:
        current.wait_stream(context.stream.stream)
