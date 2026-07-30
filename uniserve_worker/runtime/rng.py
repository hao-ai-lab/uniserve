"""Semantic counter-based random coordinates for deterministic execution."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from uniserve_worker.batch import RequestKey

_U64 = 0xFFFFFFFFFFFFFFFF
_SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15


def sampling_draw_seed(session_seed: int, position: int) -> int:
    """Return the retry-stable seed for one sequence-position draw."""

    return _splitmix_coordinate(session_seed, position)


def semantic_sampling_seed(
    request_key: RequestKey,
    session_seed: int,
    semantic_token_index: int,
    *,
    processor_stage: int = 0,
    draw_index: int = 0,
) -> int:
    """Return the Philox seed for one exact semantic sampling coordinate."""

    value = int(session_seed) & _U64
    for coordinate in (
        int(request_key.authority_id),
        int(request_key.session_id),
        int(request_key.epoch),
        int(semantic_token_index),
        int(processor_stage),
        int(draw_index),
    ):
        value = _splitmix_coordinate(value, coordinate)
    return value


def flow_noise_seed(session_seed: int, op_id: int) -> int:
    """Return the retry-stable seed for one flow operation's initial noise."""

    return _splitmix_coordinate(session_seed, op_id)


def uniform_samples(
    shape: Sequence[int],
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Draw explicit uniform samples from a generator local to one semantic seed."""

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.rand(tuple(int(value) for value in shape), device=device, generator=generator)


def normal_noise(
    shape: Sequence[int],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Draw explicit normal noise from a generator local to one semantic seed."""

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(
        tuple(int(value) for value in shape),
        device=device,
        dtype=dtype,
        generator=generator,
    )


def _splitmix_coordinate(seed: int, coordinate: int) -> int:
    if int(coordinate) < 0:
        raise ValueError("random coordinate must not be negative")
    value = (int(seed) + (int(coordinate) + 1) * _SPLITMIX64_GAMMA) & _U64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _U64
    return (value ^ (value >> 31)) & _U64


__all__ = [
    "flow_noise_seed",
    "normal_noise",
    "sampling_draw_seed",
    "semantic_sampling_seed",
    "uniform_samples",
]
