"""Semantic counter-based random coordinates for deterministic execution.

The token sampler draws through the canonical Philox4x32-10 mapping defined
here and mirrored bit-for-bit by the Rust simulator in
``crates/foundation/core/src/philox.rs``. A draw is addressed only by its
request lineage and semantic coordinates, so the same coordinate yields the
same uniform in every language and is independent of batch order, execution
depth, accepted proposal length, completion order, and replay.
"""

from __future__ import annotations

import torch

_U64 = 0xFFFFFFFFFFFFFFFF
_U32 = 0xFFFFFFFF
_SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15

_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_KEY_BUMP_0 = 0x9E3779B9
_PHILOX_KEY_BUMP_1 = 0xBB67AE85

# Draw layout identifiers matching the IPC ``DrawLayout`` discriminants. They
# separate the proposal and target draw spaces so rejected proposals cannot
# shift target coordinates.
DRAW_LAYOUT_TARGET = 0
DRAW_LAYOUT_PROPOSAL = 1
DRAW_LAYOUT_FLOW_NOISE = 2


def sampling_key(
    request_seed: int,
    engine_id: int,
    request_id: int,
    request_epoch: int,
    draw_layout: int,
) -> int:
    """Return the 64-bit Philox key for one request lineage and draw space."""

    value = int(request_seed) & _U64
    for coordinate in (
        int(engine_id),
        int(request_id),
        int(request_epoch),
        int(draw_layout),
    ):
        value = _splitmix_coordinate(value, coordinate)
    return value


def philox4x32_10(
    counter: tuple[int, int, int, int],
    key: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return the ten-round Philox4x32 bijection of ``counter`` under ``key``."""

    c0, c1, c2, c3 = (int(word) & _U32 for word in counter)
    k0, k1 = (int(word) & _U32 for word in key)
    for rounds in range(10):
        if rounds > 0:
            k0 = (k0 + _PHILOX_KEY_BUMP_0) & _U32
            k1 = (k1 + _PHILOX_KEY_BUMP_1) & _U32
        p0 = _PHILOX_M0 * c0
        p1 = _PHILOX_M1 * c2
        hi0, lo0 = (p0 >> 32) & _U32, p0 & _U32
        hi1, lo1 = (p1 >> 32) & _U32, p1 & _U32
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0) & _U32, lo1, (hi0 ^ c3 ^ k1) & _U32, lo0
    return c0, c1, c2, c3


def sampling_uniform(
    key: int,
    semantic_token_index: int,
    processor_stage: int = 0,
    draw_index: int = 0,
) -> float:
    """Return one uniform draw in ``[0, 1)`` for a semantic coordinate."""

    index = int(semantic_token_index)
    counter = (
        index & _U32,
        (index >> 32) & _U32,
        int(processor_stage) & _U32,
        int(draw_index) & _U32,
    )
    words = philox4x32_10(counter, (int(key) & _U32, (int(key) >> 32) & _U32))
    return (words[0] >> 8) * (1.0 / 16_777_216.0)


def flow_noise_seed(request_seed: int, semantic_image_index: int) -> int:
    """Return the schedule-stable seed for one semantic image's initial noise."""

    return _splitmix_coordinate(request_seed, semantic_image_index)


def normal_noise(seeds: tuple[int, ...], outputs: tuple[torch.Tensor, ...]) -> None:
    """Fill complete normal draws in row order using the supplied representations.

    Each output begins with the logical row dimension. One local generator per
    seed draws every output row in tuple order before any packing or sharding.
    Callers choose the seed mapping, device, dtype and backing; this function
    neither allocates output storage nor changes the global random generator.
    """

    if not seeds or not outputs:
        raise ValueError("normal noise requires row seeds and output views")
    device = outputs[0].device
    for value in outputs:
        if (
            value.ndim < 2
            or value.shape[0] != len(seeds)
            or value.device != device
            or not value.is_floating_point()
            or not value.is_contiguous()
        ):
            raise ValueError("noise views require aligned rows and contiguous floating storage")
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(int(seed))
        for value in outputs:
            value[index].normal_(generator=generator)


def _splitmix_coordinate(seed: int, coordinate: int) -> int:
    """Mix a seed and logical coordinate into one deterministic unsigned 64-bit value."""

    if int(coordinate) < 0:
        raise ValueError("random coordinate must not be negative")
    value = (int(seed) + (int(coordinate) + 1) * _SPLITMIX64_GAMMA) & _U64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _U64
    return (value ^ (value >> 31)) & _U64


__all__ = [
    "DRAW_LAYOUT_FLOW_NOISE",
    "DRAW_LAYOUT_PROPOSAL",
    "DRAW_LAYOUT_TARGET",
    "flow_noise_seed",
    "normal_noise",
    "philox4x32_10",
    "sampling_key",
    "sampling_uniform",
]
