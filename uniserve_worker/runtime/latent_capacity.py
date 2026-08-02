"""Physical byte geometry for transactional latent generations."""

from __future__ import annotations


def latent_generation_bytes(
    max_latent_units: int,
    latent_channels: int,
    latent_patch_size: int,
) -> int:
    """Bytes in one aggregate BF16 latent generation."""

    units = int(max_latent_units)
    channels = int(latent_channels)
    patch_size = int(latent_patch_size)
    if units < 0 or channels < 1 or patch_size < 1:
        raise ValueError("latent generation geometry is invalid")
    return units * channels * patch_size**2 * 2


def latent_store_capacity_bytes(
    max_latent_units: int,
    latent_channels: int,
    latent_patch_size: int,
) -> int:
    """Bytes for committed inputs and atomically published successors."""

    return 2 * latent_generation_bytes(
        max_latent_units,
        latent_channels,
        latent_patch_size,
    )


__all__ = ["latent_generation_bytes", "latent_store_capacity_bytes"]
