"""Windowed media reconstruction over borrowed numerical tensors."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from uniserve.distributed import Communicator
from uniserve.media import image
from uniserve.nn.vae import LatentDecoder
from uniserve.tensors import BufferConfig, OutputLayout, TensorOutput


class VideoDecoder(nn.Module):
    """Decode ordered latent windows.

    Describe their place in the native output. Subclasses define legal output
    frame slices, native output layout and ``unpack_latents``: the
    mathematical conversion from a complete packed latent to one decoder
    input. Decoder inputs and scratch remain borrowed.
    """

    def __init__(self, decoder: LatentDecoder, *, frame_size: image.Config):
        super().__init__()
        self.decoder, self.frame_size = decoder, frame_size

    def frame_slices(self, num_frames: int) -> tuple[slice, ...]:
        raise NotImplementedError

    def output_layout(self, num_frames: int) -> Mapping[str, OutputLayout]:
        raise NotImplementedError

    def unpack_latents(
        self, latent, frames, num_frames, *, constants, workspace
    ):
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
        if (
            not latents
            or len(latents) != len(frames)
            or len(latents) != len(num_frames)
        ):
            raise ValueError(
                "video latents, frame slices and durations must align"
            )

        units = []
        for interval, count in zip(frames, num_frames, strict=True):
            legal = self.frame_slices(count)
            if interval not in legal:
                raise ValueError(
                    "video frame slice must select one complete "
                    "reconstruction window"
                )
            units.append(legal.index(interval))

        outputs = []
        for latent, interval, count, unit in zip(
            latents, frames, num_frames, units, strict=True
        ):
            inputs = self.unpack_latents(
                latent,
                interval,
                count,
                constants=constants,
                workspace=workspace,
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
    """Decode media units of a packed latent timeline into sample-major PCM.

    Subclasses define ``latent_frames``, ``latent_rate``, ``latent_halo`` and
    ``unpack_latents`` from their codec's compression and channel layout. A
    media unit is a contiguous span of latent frames; because the decoder is
    convolutional, decoding a unit together with ``latent_halo`` frames of
    context on each side and discarding that context reproduces the whole-track
    decode of those samples exactly. The halo is the decoder's receptive field,
    so a unit's result depends on no sample outside the context it was given.
    """

    def __init__(self, decoder: LatentDecoder, *, sample_rate: int):
        super().__init__()
        if type(sample_rate) is not int or sample_rate < 1:
            raise ValueError("audio sample rate must be a positive integer")
        self.decoder, self.sample_rate = decoder, sample_rate

    def latent_frames(self, num_samples: int) -> int:
        raise NotImplementedError

    def track_samples(self, num_frames: int, frame_rate: int) -> int:
        """Return the sample count of the audio track generated with a video.

        The track is the decode of the latent timeline the model generates
        for ``num_frames`` video frames at ``frame_rate``, so it spans whole
        latent frames and may differ slightly from the video duration; the
        muxer aligns the two.
        """
        raise NotImplementedError

    def output_layout(self, num_samples: int) -> Mapping[str, OutputLayout]:
        """Describe the decoded sample-major track of ``num_samples``."""
        raise NotImplementedError

    @property
    def latent_rate(self) -> int:
        """Output samples one latent frame produces."""
        raise NotImplementedError

    def latent_halo(self) -> int:
        """Latent frames of context one decoded sample depends on, per side."""
        raise NotImplementedError

    def unpack_latents(self, latent, num_samples, *, window, workspace):
        """Return the codec's native channel and time representation.

        ``window`` selects the contiguous latent frames to unpack, including
        the halo the caller added around the media unit.
        """
        raise NotImplementedError

    def unit_frames(self, num_samples: int, units: int) -> tuple[slice, ...]:
        """Partition the latent timeline into ``units`` contiguous media units.

        Units divide latent frames rather than samples so every boundary falls
        on a frame the decoder produces whole.
        """
        frames = self.latent_frames(num_samples)
        if type(units) is not int or not 1 <= units <= frames:
            raise ValueError(
                "audio media units must divide the latent timeline"
            )
        edges = [frames * index // units for index in range(units + 1)]
        return tuple(
            slice(edges[index], edges[index + 1]) for index in range(units)
        )

    def unit_samples(self, num_samples: int, units: int) -> tuple[slice, ...]:
        """Return each media unit's span of the output sample timeline."""
        rate = self.latent_rate
        return tuple(
            slice(window.start * rate, min(window.stop * rate, num_samples))
            for window in self.unit_frames(num_samples, units)
        )

    @torch.inference_mode()
    def decode(
        self,
        latents: tuple[torch.Tensor, ...],
        *,
        frames: tuple[slice, ...],
        num_samples: tuple[int, ...],
        workspace: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        if (
            not latents
            or len(latents) != len(num_samples)
            or len(latents) != len(frames)
        ):
            raise ValueError(
                "audio latents, media units and sample counts must align"
            )
        if any(type(count) is not int or count < 1 for count in num_samples):
            raise ValueError(
                "audio durations must contain a positive sample count"
            )

        rate, halo = self.latent_rate, self.latent_halo()
        outputs = []
        for latent, window, count in zip(
            latents, frames, num_samples, strict=True
        ):
            total = self.latent_frames(count)
            if (
                window.step not in (None, 1)
                or window.start is None
                or window.stop is None
                or not 0 <= window.start < window.stop <= total
            ):
                raise ValueError(
                    "audio media unit must lie within its latent timeline"
                )

            # The halo is clipped at the track's own boundaries, where the
            # whole-track decode has no context either.
            context = slice(
                max(0, window.start - halo), min(total, window.stop + halo)
            )
            inputs = self.unpack_latents(
                latent, count, window=context, workspace=workspace
            )
            decoded = self.decoder(inputs)
            produced = (context.stop - context.start) * rate
            if decoded.ndim != 2 or decoded.shape[0] < produced:
                raise ValueError(
                    "decoded audio must cover the requested sample timeline"
                )

            # Discard the halo, then crop the final unit to the exact duration.
            begin = (window.start - context.start) * rate
            end = begin + min(window.stop * rate, count) - window.start * rate
            outputs.append(decoded[begin:end].clone())
        return tuple(outputs)


class VideoPostprocessor(nn.Module):
    """Blend temporal overlaps, crop decoder padding and produce RGB24 frames.

    ``overlap_weights`` weights the current window, in the decoded precision.
    A subclass supplies ``reconstruction_slices`` for the body and successor
    overlap within each native NCTHW segment. A frame slice starting at zero
    resets the overlap; later slices consume the preceding window's state.
    Returned tensors borrow disjoint slices of ``workspace['rgb_frames']``.

    Media units of one round are reconstructed by different ranks, and a unit's
    leading frames blend with the overlap its predecessor decoded, so ``units``
    orders those ranks as the units they hold and carries that overlap between
    them. Alone, a rank carries its own overlap forward, which is the serial
    reconstruction.
    """

    overlap_weights: torch.Tensor

    def __init__(
        self,
        overlap_weights: torch.Tensor,
        *,
        frame_size: image.Config,
        frame_rate: int,
    ):
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
            raise ValueError(
                "overlap weights must be a real finite vector in [0, 1]"
            )
        self.register_buffer(
            "overlap_weights", overlap_weights, persistent=False
        )
        self.frame_size, self.frame_rate = frame_size, frame_rate
        # Rebound by the runtime to the ranks this component is placed on.
        self.units = Communicator()

    def reconstruction_slices(
        self, frames: slice, num_frames: int
    ) -> tuple[slice, slice]:
        """Locate the body and successor overlap.

        Within a native decoded segment.
        """
        raise NotImplementedError

    def state_buffers(self, num_frames: int) -> Mapping[str, BufferConfig]:
        """Describe the state carried between the rounds of one video.

        The state holds the decoded overlap a unit's successor blends with.
        """
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
        unit_count: int = 1,
    ) -> tuple[TensorOutput, ...]:
        """Reconstruct this rank's media units of one round.

        ``unit_count`` is how many members of ``units`` hold a media unit in
        this round, which is fewer than the whole ring when the track has fewer
        units left than the component has ranks.
        """
        if (
            not segments
            or len(segments) != len(frames)
            or len(segments) != len(num_frames)
        ):
            raise ValueError(
                "video segments, frame slices and durations must align"
            )
        if (
            type(unit_count) is not int
            or not 0 <= self.units.rank < unit_count <= self.units.size
        ):
            raise ValueError(
                "media unit ring position must lie within the round it serves"
            )
        retained = state["video_overlap"]
        overlap = retained
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
                raise ValueError(
                    "video frame slices must lie within their output duration"
                )
            body_slice, next_slice = self.reconstruction_slices(interval, count)
            value = segment.tensor
            if value.ndim == 6 and value.shape[0] == 1:
                value = value[0]
            if (
                value.ndim != 5
                or value.shape[:2] != (1, 3)
                or value.shape[-2:] != (height, width)
            ):
                raise ValueError(
                    "video segments must have native NCTHW shape and the "
                    "configured raster"
                )
            body, successor = value[:, :, body_slice], value[:, :, next_slice]
            expected = body.shape[2] + (extent if interval.stop == count else 0)
            if (
                successor.shape[2] != extent
                or body.shape[2] < extent
                or expected != interval.stop - interval.start
            ):
                raise ValueError(
                    "video reconstruction slices do not cover the requested "
                    "output frames"
                )
            if value.dtype != overlap.dtype or value.device != overlap.device:
                raise ValueError(
                    "video overlap must share decoded precision and device"
                )
            if (
                index
                and interval.start != 0
                and (
                    num_frames[index - 1] != count
                    or frames[index - 1].stop != interval.start
                )
            ):
                raise ValueError(
                    "successive video windows must describe a contiguous "
                    "ordered range"
                )
            values.append(value)
            slices.append((body_slice, next_slice))
            total_frames += expected

        if overlap.shape != (1, 3, extent, height, width):
            raise ValueError(
                "video overlap state must contain the complete temporal overlap"
            )
        # Every participant sends the tail of its last unit to the rank holding
        # the next one and receives its predecessor's, so one exchange carries
        # the whole round's overlaps. The rank holding the round's first unit
        # blends with what it retained from the previous round and keeps what
        # arrives here for the next one; the ring's wrap-around therefore
        # delivers one tail that the final round never consumes.
        incoming = workspace["overlap_exchange"]
        outgoing = values[-1][:, :, slices[-1][1]]
        if incoming.shape != outgoing.shape or incoming.dtype != outgoing.dtype:
            raise ValueError(
                "video overlap exchange must match the overlap it carries"
            )
        if unit_count == 1:
            incoming.copy_(outgoing)
        else:
            position = self.units.rank
            self.units.send_recv(
                outgoing,
                dst=(position + 1) % unit_count,
                src=(position - 1) % unit_count,
                out=incoming,
            )
        if self.units.rank:
            # Only the round's first unit reads a retained overlap; every other
            # rank blends with the one its predecessor just sent.
            overlap = incoming
        if (
            pixels.ndim != 4
            or pixels.shape[1:] != (height, width, 3)
            or pixels.dtype != torch.uint8
            or pixels.shape[0] < total_frames
        ):
            raise ValueError(
                "RGB workspace must cover the complete output frame range"
            )
        if mean.shape != (1, 3, 1, 1, 1) or std.shape != mean.shape:
            raise ValueError(
                "video normalization requires one mean and scale per channel"
            )
        if mean.dtype != torch.float32 or std.dtype != torch.float32:
            raise ValueError("video normalization constants must use float32")
        if any(
            value.device != overlap.device
            for value in (pixels, mean, std, self.overlap_weights)
        ):
            raise ValueError("video views must share the decoded input device")

        # Weights broadcast over the temporal axis of each NCTHW window.
        weights = self.overlap_weights.to(overlap.dtype).view(
            1, 1, extent, 1, 1
        )
        cursor = 0
        outputs = []
        for value, (body_slice, next_slice), interval, count in zip(
            values, slices, frames, num_frames, strict=True
        ):
            body = value[:, :, body_slice]
            if interval.start:
                blended = (
                    overlap * (1 - weights) + body[:, :, :extent] * weights
                )
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
                        (
                            interval,
                            slice(0, height),
                            slice(0, width),
                            slice(0, 3),
                        ),
                        variable_axes=(0,),
                        value_range=(0, 255),
                    ),
                )
            )
        if not self.units.rank:
            # The round's last tail belongs to the round that follows it.
            retained.copy_(incoming)
        return tuple(outputs)
