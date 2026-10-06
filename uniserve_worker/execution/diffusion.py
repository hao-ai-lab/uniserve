"""Numerical inputs for image diffusion over borrowed KV prefixes.

The native executor owns latent-bank selection, prefix preparation and
request progress. These functions build schedules, seeded samples and
model rows; the numerical runner integrates the guided predictions.
"""

from __future__ import annotations

import torch

from uniserve.diffusion import Renorm
from uniserve.nn.rng import flow_noise_seed
from uniserve_worker.execution.diffusion_state import DiffusionState
from uniserve_worker.model_executor.diffusion_inputs import DiffusionRow
from uniserve_worker.model_executor.input_batch import TokenRow
from uniserve_worker.protocol.call import ForwardMode, ImageParams, MediaCall
from uniserve_worker.sampling.metadata import TokenSelection


def _to_device(
    value: torch.Tensor, device: torch.device | None
) -> torch.Tensor:
    """Borrow a local tensor or copy it to the device on the caller's stream."""
    if device is None or value.device == device:
        return value

    # Host consumers need a completed D2H result; device consumers retain the
    # stream dependency and can overlap the copy with independent work.
    return value.to(device, non_blocking=device.type != "cpu")


def image_state(builder, size, image: ImageParams) -> DiffusionState:
    """Open an admitted image's diffusion state from its sampling choices.

    The schedules are built on the host; the native executor reads each
    step's time before filling its device timestep view. A non-positive
    ``timestep_shift`` is passed as ``None``, leaving the shift to the
    denoiser's ``make_schedules``.
    """
    denoiser = builder.denoiser
    return DiffusionState.open(
        denoiser,
        size,
        steps=image.steps,
        shift=image.timestep_shift if image.timestep_shift > 0 else None,
        device="cpu",
        guidance=denoiser.make_guidance(
            text_scale=image.cfg_text_scale,
            image_scale=image.cfg_img_scale,
            interval=image.cfg_interval,
            renorm=Renorm(image.cfg_renorm_type),
            renorm_min=image.cfg_renorm_min,
        ),
    )


def initial_latent(
    builder, size, buffer, units: int, seed: int, image_index: int
):
    """Draw an image's seeded noise, leaving physical page padding zero."""
    buffer.zero_()
    builder.initialize(
        size, seed=flow_noise_seed(seed, image_index), out=buffer[:units]
    )


def prefix_row(tokens: tuple[int, ...], slot: int, visible: int) -> TokenRow:
    """Append a guidance prefix without computing token logits."""
    return TokenRow(
        forward_mode=ForwardMode.PREFILL,
        token_ids=torch.tensor(tokens, dtype=torch.long),
        positions=torch.arange(
            visible, visible + len(tokens), dtype=torch.long
        ),
        selection=TokenSelection.HIDDEN,
        request_pool_idx=slot,
        seq_len=visible,
        write_kv=True,
        causal=True,
    )


def flow_rows(
    builder,
    trajectory: DiffusionState,
    current: torch.Tensor,
    timestep: torch.Tensor,
    coordinates: tuple[tuple[int, int, int], ...],
    *,
    device: torch.device,
) -> tuple[DiffusionRow, ...]:
    """Build rows over executor-selected (slot, KV length, temporal position).

    Every branch borrows one sample and timestep. Position tensors are
    shared by temporal position within the request's fixed image size.
    """
    current = _to_device(current, device)
    timestep = _to_device(timestep, device).reshape(1)
    size = trajectory.size
    rows = []

    for slot, visible, temporal in coordinates:
        if temporal not in trajectory.positions:
            trajectory.positions[temporal] = builder.positions(
                size, temporal, device=device
            )

        rows.append(
            DiffusionRow(
                forward_mode=MediaCall.DENOISING,
                positions=trajectory.positions[temporal],
                timestep=timestep,
                latent=current,
                image_tokens=builder.sequence_length(size),
                image_height=size.height,
                image_width=size.width,
                request_pool_idx=slot,
                seq_len=visible,
                write_kv=False,
                causal=False,
            )
        )

    return tuple(rows)
