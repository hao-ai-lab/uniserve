"""Windowed media reconstruction over borrowed numerical tensors."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from uniserve.media import image
from uniserve.nn.vae import LatentDecoder
from uniserve.tensors import OutputLayout, TensorOutput


class VideoDecoder(nn.Module):
    """Decode ordered latent windows and describe their place in the native output.

    Subclasses define legal output frame slices, native output layout and
    ``unpack_latents``: the mathematical conversion from a complete packed
    latent to one decoder input. Decoder inputs and scratch remain borrowed.
    """

    def __init__(self, decoder: LatentDecoder, *, frame_size: image.Config):
        super().__init__()
        self.decoder, self.frame_size = decoder, frame_size

    def frame_slices(self, num_frames: int) -> tuple[slice, ...]:
        raise NotImplementedError

    def output_layout(self, num_frames: int) -> Mapping[str, OutputLayout]:
        raise NotImplementedError

    def unpack_latents(self, latent, frames, num_frames, *, constants, workspace):
        """Return the native latent window for one legal output frame slice."""
        raise NotImplementedError

    @torch.inference_mode()
    def decode(
        self,
        latents: tuple[torch.Tensor, ...],
        *,
        frames: tuple[slice, ...],
        num_frames: tuple[int, ...],
        constants: Mapping[str, torch.Tensor],
        workspace: Mapping[str, torch.Tensor],
    ) -> tuple[TensorOutput | None, ...]:
        if not latents or len(latents) != len(frames) or len(latents) != len(num_frames):
            raise ValueError("video latents, frame slices and durations must align")
        units = []
        for interval, count in zip(frames, num_frames, strict=True):
            legal = self.frame_slices(count)
            if interval not in legal:
                raise ValueError("video frame slice must select one complete reconstruction window")
            units.append(legal.index(interval))
        outputs = []
        for latent, interval, count, unit in zip(latents, frames, num_frames, units, strict=True):
            inputs = self.unpack_latents(
                latent, interval, count, constants=constants, workspace=workspace
            )
            decoded = self.decoder(inputs).unsqueeze(0)
            # A decoder may return borrowed workspace. Preserve earlier results
            # across later numerical calls within this batch.
            if len(latents) > 1:
                decoded = decoded.clone()
            layout = self.output_layout(count)["video"]
            outputs.append(
                TensorOutput(
                    decoded,
                    OutputLayout(
                        layout.shape,
                        layout.dtype,
                        (slice(unit, unit + 1), *layout.local_slice[1:]),
                        variable_axes=layout.variable_axes,
                        value_range=layout.value_range,
                    ),
                )
            )
        return tuple(outputs)


class AudioDecoder(nn.Module):
    """Decode a packed latent timeline and crop to the requested sample count.

    Subclasses define ``latent_frames`` and ``unpack_latents`` from their
    codec's compression and channel layout. The decoded tensor is sample-major.
    """

    def __init__(self, decoder: LatentDecoder, *, sample_rate: int):
        super().__init__()
        if type(sample_rate) is not int or sample_rate < 1:
            raise ValueError("audio sample rate must be a positive integer")
        self.decoder, self.sample_rate = decoder, sample_rate

    def latent_frames(self, num_samples: int) -> int:
        raise NotImplementedError

    def unpack_latents(self, latent, num_samples, *, workspace):
        """Return the codec's native channel and time representation."""
        raise NotImplementedError

    @torch.inference_mode()
    def decode(
        self,
        latents: tuple[torch.Tensor, ...],
        *,
        num_samples: tuple[int, ...],
        workspace: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        if not latents or len(latents) != len(num_samples):
            raise ValueError("audio latents and sample counts must align")
        if any(type(count) is not int or count < 1 for count in num_samples):
            raise ValueError("audio durations must contain a positive sample count")
        outputs = []
        for latent, count in zip(latents, num_samples, strict=True):
            inputs = self.unpack_latents(latent, count, workspace=workspace)
            decoded = self.decoder(inputs)
            if decoded.ndim != 2 or decoded.shape[0] < count:
                raise ValueError("decoded audio must cover the requested sample timeline")
            output = decoded[:count]
            outputs.append(output.clone() if len(latents) > 1 else output)
        return tuple(outputs)


class VideoPostprocessor(nn.Module):
    """Blend temporal overlaps, crop decoder padding and produce RGB24 frames.

    ``overlap_weights`` weights the current window, in the decoded precision.
    A subclass supplies ``reconstruction_slices`` for the body and successor
    overlap within each native NCTHW segment. A frame slice starting at zero
    resets the overlap; later slices consume the preceding window's state.
    Returned tensors borrow disjoint slices of ``workspace['rgb_frames']``.
    """

    def __init__(self, overlap_weights: torch.Tensor, *, frame_size: image.Config, frame_rate: int):
        super().__init__()
        if type(frame_rate) is not int or frame_rate < 1:
            raise ValueError("video frame rate must be a positive integer")
        if (
            overlap_weights.is_meta
            or overlap_weights.ndim != 1
            or overlap_weights.numel() < 1
            or not overlap_weights.is_floating_point()
            or not bool(torch.isfinite(overlap_weights).all())
            or not bool(((overlap_weights >= 0) & (overlap_weights <= 1)).all())
        ):
            raise ValueError("overlap weights must be a real finite vector in [0, 1]")
        self.register_buffer("overlap_weights", overlap_weights, persistent=False)
        self.frame_size, self.frame_rate = frame_size, frame_rate

    def reconstruction_slices(self, frames: slice, num_frames: int) -> tuple[slice, slice]:
        """Locate the body and successor overlap within a native decoded segment."""
        raise NotImplementedError

    @torch.inference_mode()
    def forward(
        self,
        segments: tuple[TensorOutput, ...],
        *,
        frames: tuple[slice, ...],
        num_frames: tuple[int, ...],
        state: Mapping[str, torch.Tensor],
        constants: Mapping[str, torch.Tensor],
        workspace: Mapping[str, torch.Tensor],
    ) -> tuple[TensorOutput, ...]:
        if not segments or len(segments) != len(frames) or len(segments) != len(num_frames):
            raise ValueError("video segments, frame slices and durations must align")
        overlap = state["video_overlap"]
        pixels = workspace["rgb_frames"]
        mean, std = constants["pixel_mean"], constants["pixel_std"]
        height, width = self.frame_size.height, self.frame_size.width
        extent = self.overlap_weights.numel()
        values, slices = [], []
        total_frames = 0
        for index, (segment, interval, count) in enumerate(
            zip(segments, frames, num_frames, strict=True)
        ):
            if (
                type(count) is not int
                or count < 1
                or interval.step not in (None, 1)
                or interval.start is None
                or interval.stop is None
                or not 0 <= interval.start < interval.stop <= count
            ):
                raise ValueError("video frame slices must lie within their output duration")
            body_slice, next_slice = self.reconstruction_slices(interval, count)
            value = segment.tensor
            if value.ndim == 6 and value.shape[0] == 1:
                value = value[0]
            if value.ndim != 5 or value.shape[:2] != (1, 3) or value.shape[-2:] != (height, width):
                raise ValueError(
                    "video segments must have native NCTHW shape and the configured raster"
                )
            body, successor = value[:, :, body_slice], value[:, :, next_slice]
            expected = body.shape[2] + (extent if interval.stop == count else 0)
            if (
                successor.shape[2] != extent
                or body.shape[2] < extent
                or expected != interval.stop - interval.start
            ):
                raise ValueError(
                    "video reconstruction slices do not cover the requested output frames"
                )
            if value.dtype != overlap.dtype or value.device != overlap.device:
                raise ValueError("video overlap must share decoded precision and device")
            if (
                index
                and interval.start != 0
                and (num_frames[index - 1] != count or frames[index - 1].stop != interval.start)
            ):
                raise ValueError(
                    "successive video windows must describe a contiguous ordered range"
                )
            values.append(value)
            slices.append((body_slice, next_slice))
            total_frames += expected
        if overlap.shape != (1, 3, extent, height, width):
            raise ValueError("video overlap state must contain the complete temporal overlap")
        if (
            pixels.ndim != 4
            or pixels.shape[1:] != (height, width, 3)
            or pixels.dtype != torch.uint8
            or pixels.shape[0] < total_frames
        ):
            raise ValueError("RGB workspace must cover the complete output frame range")
        if mean.shape != (1, 3, 1, 1, 1) or std.shape != mean.shape:
            raise ValueError("video normalization requires one mean and scale per channel")
        if mean.dtype != torch.float32 or std.dtype != torch.float32:
            raise ValueError("video normalization constants must use float32")
        if any(
            value.device != overlap.device for value in (pixels, mean, std, self.overlap_weights)
        ):
            raise ValueError("video views must share the decoded input device")

        weights = self.overlap_weights.to(overlap.dtype).view(1, 1, extent, 1, 1)
        cursor = 0
        outputs = []
        for value, (body_slice, next_slice), interval, count in zip(
            values, slices, frames, num_frames, strict=True
        ):
            body = value[:, :, body_slice]
            if interval.start:
                blended = overlap * (1 - weights) + body[:, :, :extent] * weights
                body = torch.cat((blended, body[:, :, extent:]), dim=2)
            successor = value[:, :, next_slice]
            if interval.stop == count:
                body = torch.cat((body, successor), dim=2)
            # Preserve decoded-precision blending before FP32 denormalization.
            rgb = (body.float() * std + mean).clamp_(0, 1)
            rgb = rgb[0].permute(1, 2, 3, 0).mul_(255).round_().to(torch.uint8)
            output = pixels[cursor : cursor + rgb.shape[0]]
            output.copy_(rgb)
            overlap.copy_(successor)
            cursor += rgb.shape[0]
            outputs.append(
                TensorOutput(
                    output,
                    OutputLayout(
                        (count, height, width, 3),
                        torch.uint8,
                        (interval, slice(0, height), slice(0, width), slice(0, 3)),
                        variable_axes=(0,),
                        value_range=(0, 255),
                    ),
                )
            )
        return tuple(outputs)
