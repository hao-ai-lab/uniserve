"""Windowed media encoding and reconstruction over borrowed tensors."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from uniserve.distributed import Communicator
from uniserve.media import image, video
from uniserve.nn.vae import LatentDecoder, LatentEncoder, SpatialEncoder
from uniserve.tensors import BufferConfig, OutputLayout, TensorOutput


class VideoEncoder(nn.Module):
    """Encode units of RGB24 frames into latent rows.

    A unit is a contiguous run of a video's frames whose latents depend on no
    frame outside it. ``frame_slices`` lists a video's units in order and
    ``latent_slices`` the contiguous latent frames each one produces, so the
    units of one video may be encoded by different ranks, in any order, and
    their rows assemble the whole-video encoding exactly. A still frame (a
    one-frame video) may further be encoded in bands of its latent rows:
    ``row_bands`` partitions its latent raster into bands of whole row
    groups, each band's rows are a contiguous run of the frame's rows, and
    the bands' rows assemble the frame's encoding exactly. Banding needs a
    latent encoder that encodes a band on its own, such as one over a
    ``SpatialEncoder``.

    Subclasses define both temporal partitions; ``output_layout``, which
    describes a video's complete latent as frame-major rows along its leading
    axis, each latent frame owning the same number of consecutive rows;
    ``latent_size``, a frame's native latent raster; ``row_group``, the
    latent rows of one row group, a frame's rows being row-major over its
    row groups, each group owning the same number of consecutive rows;
    ``unpack_pixels``, the conversion of one unit's frames into the native
    encoder input; and ``pack_latents``, the conversion of that unit's native
    NCTHW latents into its output rows. An encoder whose posterior is sampled
    defines ``posterior_noise``: the draw spans the video's complete latent
    and each unit takes the share of its own latent frames and rows, which
    reproduces the draw of a whole-video encoding. Inputs remain borrowed;
    results are new tensors.
    """

    #: Latent rows of one row group, the unit of a still frame's bands.
    row_group: int = 1

    def __init__(self, encoder: LatentEncoder):
        super().__init__()
        self.encoder = encoder

    def frame_slices(self, num_frames: int) -> tuple[slice, ...]:
        """Partition a video of ``num_frames`` frames into its units."""
        raise NotImplementedError

    def latent_slices(self, num_frames: int) -> tuple[slice, ...]:
        """Return the latent frames each unit produces, in unit order."""
        raise NotImplementedError

    def output_layout(self, size: video.Config) -> Mapping[str, OutputLayout]:
        """Describe the complete latent rows of one video under ``video``."""
        raise NotImplementedError

    def latent_size(self, frame: image.Config) -> tuple[int, int]:
        """Return the native ``(height, width)`` latent raster of a frame.

        The height is a whole number of row groups.
        """
        raise NotImplementedError

    def row_bands(self, frame: image.Config, count: int) -> tuple[slice, ...]:
        """Partition a still frame's latent rows into ``count`` bands.

        The frame's ``g`` row groups are dealt in order as evenly as
        possible: band ``b`` holds groups ``b * g // count`` to
        ``(b + 1) * g // count``. Returns each band's latent rows.

        Raises:
            ValueError: ``count`` is not an integer from one to the frame's
                row groups.
        """
        height, _ = self.latent_size(frame)
        groups = height // self.row_group
        if type(count) is not int or not 1 <= count <= groups:
            raise ValueError(
                f"a frame of {groups} latent row groups splits into 1 to "
                f"{groups} bands, got {count}"
            )
        return tuple(
            slice(
                band * groups // count * self.row_group,
                (band + 1) * groups // count * self.row_group,
            )
            for band in range(count)
        )

    def still_tile(
        self, device: torch.device
    ) -> tuple[SpatialEncoder, torch.Tensor] | None:
        """Return the tiled encoder of still frames and one tile's input.

        A latent encoder over a ``SpatialEncoder`` encodes every still frame
        at least a tile high and wide in tiles of one shape: the native input
        of a tile-sized frame, which ``unpack_pixels`` returns with the
        frame's raster as its trailing axes. Returns that encoder and a zero
        tile of its native input on ``device``, or None when the latent
        encoder does not tile.
        """
        spatial = self.encoder.encoder
        if not isinstance(spatial, SpatialEncoder):
            return None
        frame = torch.zeros(
            (1, spatial.tile_height, spatial.tile_width, 3),
            dtype=torch.uint8,
            device=device,
        )
        return spatial, self.unpack_pixels(frame, slice(0, 1), 1)

    def posterior_noise(self, size: video.Config) -> torch.Tensor | None:
        """Return one video's complete NCTHW posterior draw, or None.

        None leaves the latent to the latent encoder's own posterior: its
        mean, a draw of its own, or no posterior at all.
        """
        return None

    def unpack_pixels(
        self, pixels: torch.Tensor, frames: slice, num_frames: int
    ) -> torch.Tensor:
        """Return the native encoder input of one unit's frames.

        ``pixels`` holds exactly the frames ``frames`` selects, as
        ``[frames, height, width, 3]`` uint8 RGB.
        """
        raise NotImplementedError

    def pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Return the output rows of one unit's native NCTHW latents."""
        raise NotImplementedError

    @torch.inference_mode()
    def encode(
        self,
        pixels: tuple[torch.Tensor, ...],
        *,
        frames: tuple[slice, ...],
        num_frames: tuple[int, ...],
        rows: tuple[slice | None, ...] | None = None,
    ) -> tuple[TensorOutput, ...]:
        """Encode units of RGB24 videos into their latent rows.

        ``pixels[i]`` holds exactly the frames that ``frames[i]`` selects from
        a video of ``num_frames[i]`` frames, as ``[frames, height, width, 3]``
        uint8, and ``frames[i]`` must be one of that video's units. ``rows[i]``
        is None for the unit's whole raster, or, for a still frame, a band of
        whole latent row groups, such as one of ``row_bands``. Each result
        holds its unit's rows at their place in the video's complete latent.

        Raises:
            ValueError: The inputs break the unit contract above, or the
                encoded rows disagree with the declared layout.
        """
        if rows is None:
            rows = (None,) * len(pixels)
        if (
            not pixels
            or len(pixels) != len(frames)
            or len(pixels) != len(num_frames)
            or len(pixels) != len(rows)
        ):
            raise ValueError(
                "video frames, unit slices, durations and bands must align"
            )

        units = []
        for value, interval, count, band in zip(
            pixels, frames, num_frames, rows, strict=True
        ):
            legal = self.frame_slices(count)
            if interval not in legal:
                raise ValueError(
                    "video frame slice must select one complete encoding unit"
                )
            if (
                value.ndim != 4
                or value.dtype != torch.uint8
                or value.shape[-1] != 3
                or value.shape[0] != interval.stop - interval.start
            ):
                raise ValueError(
                    "video units must be uint8 RGB frames "
                    "[frames, height, width, 3] covering their frame slice"
                )
            if band is not None:
                height, _ = self.latent_size(
                    image.Config(int(value.shape[1]), int(value.shape[2]))
                )
                group = self.row_group
                if (
                    count != 1
                    or band.step not in (None, 1)
                    or band.start is None
                    or band.stop is None
                    or not 0 <= band.start < band.stop <= height
                    or band.start % group
                    or band.stop % group
                ):
                    raise ValueError(
                        "a band must be whole latent row groups of a still "
                        "frame"
                    )
            units.append(legal.index(interval))

        # One posterior draw per video size serves every unit of this call
        # that belongs to a video of that size.
        draws: dict[video.Config, torch.Tensor | None] = {}
        outputs = []
        for value, interval, count, unit, band in zip(
            pixels, frames, num_frames, units, rows, strict=True
        ):
            size = video.Config(
                count, image.Config(int(value.shape[1]), int(value.shape[2]))
            )
            layout = self.output_layout(size)["video"]
            windows = self.latent_slices(count)
            window = windows[unit]
            extent = window.stop - window.start
            if layout.shape[0] % windows[-1].stop:
                raise ValueError(
                    "video latent rows must divide evenly among latent frames"
                )
            rows_per_frame = layout.shape[0] // windows[-1].stop

            # A band keeps a run of its frame's row groups, whose rows follow
            # each other within the frame.
            first, stop = (
                window.start * rows_per_frame,
                window.stop * rows_per_frame,
            )
            if band is not None:
                groups = self.latent_size(size.frame)[0] // self.row_group
                if rows_per_frame % groups:
                    raise ValueError(
                        "a frame's latent rows must divide evenly among its "
                        "row groups"
                    )
                rows_per_group = rows_per_frame // groups
                first = band.start // self.row_group * rows_per_group
                stop = band.stop // self.row_group * rows_per_group

            if size not in draws:
                draws[size] = self.posterior_noise(size)
            noise = draws[size]
            if noise is not None:
                noise = noise[:, :, window]
                if band is not None:
                    noise = noise[..., band, :]
                noise = noise.to(value.device)

            # A unit whose input was padded to the encoder's temporal extent
            # yields trailing latent frames beyond its own; the window keeps
            # the leading ones, so the posterior never samples the others.
            latents = self.encoder(
                self.unpack_pixels(value, interval, count),
                rows=band,
                window=(slice(None), slice(None), slice(0, extent)),
                noise=noise,
            )
            values = self.pack_latents(latents)
            if (
                values.shape != (stop - first, *layout.shape[1:])
                or values.dtype != layout.dtype
            ):
                raise ValueError(
                    "packed video latents must match the declared output rows"
                )
            outputs.append(
                TensorOutput(
                    values,
                    OutputLayout(
                        layout.shape,
                        layout.dtype,
                        (slice(first, stop), *layout.local_slice[1:]),
                        variable_axes=layout.variable_axes,
                        value_range=layout.value_range,
                    ),
                )
            )
        return tuple(outputs)


class AudioEncoder(nn.Module):
    """Encode sample-major PCM tracks into latent rows.

    Tracks are encoded whole rather than by media unit. Subclasses define
    ``latent_frames`` and ``latent_rate`` from their codec's compression;
    ``output_layout``; ``unpack_samples``, the conversion of one track's
    ``[samples, channels]`` PCM into the native encoder input; and
    ``pack_latents``, the conversion of the track's native latents into its
    output rows. Inputs carry ``sample_rate`` samples per second and remain
    borrowed; results are new tensors.
    """

    def __init__(self, encoder: LatentEncoder, *, sample_rate: int):
        super().__init__()
        if type(sample_rate) is not int or sample_rate < 1:
            raise ValueError("audio sample rate must be a positive integer")
        self.encoder, self.sample_rate = encoder, sample_rate

    def latent_frames(self, num_samples: int) -> int:
        """Return the latent frames that encode ``num_samples`` samples."""
        raise NotImplementedError

    @property
    def latent_rate(self) -> int:
        """Input samples one latent frame encodes."""
        raise NotImplementedError

    def output_layout(self, num_samples: int) -> Mapping[str, OutputLayout]:
        """Describe the latent rows of one track under ``audio``."""
        raise NotImplementedError

    def unpack_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """Return the native encoder input of one PCM track."""
        raise NotImplementedError

    def pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Return the output rows of one track's native latents."""
        raise NotImplementedError

    @torch.inference_mode()
    def encode(
        self, samples: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Encode ``[samples, channels]`` floating-point PCM tracks.

        Values lie in ``[-1, 1]``. Each result holds one track's complete
        latent rows as ``output_layout`` describes them.
        """
        if not samples:
            raise ValueError("audio encoding requires at least one track")
        outputs = []
        for track in samples:
            if (
                track.ndim != 2
                or track.shape[0] < 1
                or track.shape[1] < 1
                or not track.is_floating_point()
            ):
                raise ValueError(
                    "audio tracks must be nonempty floating-point PCM "
                    "[samples, channels]"
                )
            layout = self.output_layout(int(track.shape[0]))["audio"]
            rows = self.pack_latents(self.encoder(self.unpack_samples(track)))
            if rows.shape != layout.shape or rows.dtype != layout.dtype:
                raise ValueError(
                    "packed audio latents must match the declared output rows"
                )
            outputs.append(rows)
        return tuple(outputs)


class VideoDecoder(nn.Module):
    """Decode a video's packed latent one reconstruction window at a time.

    A video's size is its frame count and raster
    (``uniserve.media.video.Config``). Its native output is cut at the legal
    output frame slices of ``frame_slices``, and each slice is reconstructed
    from one window of the packed latent into one native segment. Subclasses
    define the slices, the native output layout, each slice's ``segment``,
    the decoder input of a segment's window (``window_input``) and
    ``unpack_latents``: the mathematical conversion from a complete packed
    latent to one window's decoder input.

    Only unpacking depends on the video's frame count. ``decode`` evaluates
    unpacked windows, whose computation depends on their segment alone, so a
    caller prepares decoding resources and graphs per segment rather than per
    video size, and unpacks each window outside them. Decoder inputs and
    scratch remain borrowed.
    """

    def __init__(self, decoder: LatentDecoder):
        super().__init__()
        self.decoder = decoder

    def frame_slices(self, num_frames: int) -> tuple[slice, ...]:
        raise NotImplementedError

    def output_layout(self, size: video.Config) -> Mapping[str, OutputLayout]:
        """Describe a video's native output: one segment per frame slice."""
        raise NotImplementedError

    def segment(self, size: video.Config, frames: slice) -> video.Config:
        """Return the native frames and raster one frame slice decodes to.

        Raises:
            ValueError: ``frames`` is not a legal output frame slice of
                ``size``.
        """
        raise NotImplementedError

    def window_input(self, segment: video.Config) -> BufferConfig:
        """Describe the decoder input of a window decoding to ``segment``."""
        raise NotImplementedError

    def unpack_latents(
        self,
        latent: torch.Tensor,
        frames: slice,
        size: video.Config,
        *,
        out: torch.Tensor,
    ) -> None:
        """Write the decoder input of one legal output frame slice.

        ``latent`` is the complete packed latent of a video of ``size`` and
        ``out`` the window's input, laid out as ``window_input`` describes
        for ``segment(size, frames)``. The conversion is ordered on the
        caller's current stream.

        Raises:
            ValueError: ``latent`` is not a complete packed latent of
                ``size``, or ``frames`` is not one of its legal slices.
        """
        raise NotImplementedError

    @torch.inference_mode()
    def decode(
        self,
        windows: tuple[torch.Tensor, ...],
        *,
        segments: tuple[video.Config, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Reconstruct unpacked windows into their native segments.

        Each window is laid out as ``window_input`` describes for its
        segment. Each result leads with the segment's unit axis of one,
        which is one row of ``output_layout`` (``place`` describes where).

        Raises:
            ValueError: The windows and segments do not align, or a window
                is not laid out for its segment.
        """
        if not windows or len(windows) != len(segments):
            raise ValueError("video windows and segments must align")
        for window, segment in zip(windows, segments, strict=True):
            config = self.window_input(segment)
            if (
                tuple(window.shape) != tuple(config.shape)
                or window.dtype != config.dtype
            ):
                raise ValueError(
                    "a video window must be laid out for its segment"
                )

        outputs = []
        for window in windows:
            decoded = self.decoder(window).unsqueeze(0)
            # A decoder may return borrowed workspace. Preserve earlier results
            # across later numerical calls within this batch.
            if len(windows) > 1:
                decoded = decoded.clone()
            outputs.append(decoded)
        return tuple(outputs)

    def place(
        self, decoded: torch.Tensor, frames: slice, size: video.Config
    ) -> TensorOutput:
        """Describe one decoded segment's place in the output of ``size``.

        ``decoded`` is ``decode``'s result for the window of ``frames``; the
        returned output names its row of ``output_layout(size)``.

        Raises:
            ValueError: ``frames`` is not a legal output frame slice of
                ``size``.
        """
        legal = self.frame_slices(size.num_frames)
        if frames not in legal:
            raise ValueError(
                "video frame slice must select one complete reconstruction "
                "window"
            )
        unit = legal.index(frames)
        layout = self.output_layout(size)["video"]
        return TensorOutput(
            decoded,
            OutputLayout(
                layout.shape,
                layout.dtype,
                (slice(unit, unit + 1), *layout.local_slice[1:]),
                variable_axes=layout.variable_axes,
                value_range=layout.value_range,
            ),
        )


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
    reconstruction. A video's size is its frame count and raster
    (``uniserve.media.video.Config``).
    """

    overlap_weights: torch.Tensor

    def __init__(self, overlap_weights: torch.Tensor, *, frame_rate: int):
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
        self.frame_rate = frame_rate
        # Rebound by the runtime to the ranks this component is placed on.
        self.units = Communicator()

    def reconstruction_slices(
        self, frames: slice, num_frames: int
    ) -> tuple[slice, slice]:
        """Locate the body and successor overlap.

        Within a native decoded segment.
        """
        raise NotImplementedError

    def state_buffers(self, size: video.Config) -> Mapping[str, BufferConfig]:
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
        sizes: tuple[video.Config, ...],
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
            or len(segments) != len(sizes)
            or len({size.frame for size in sizes}) != 1
        ):
            raise ValueError(
                "video segments, frame slices and sizes of one raster must "
                "align"
            )
        num_frames = tuple(size.num_frames for size in sizes)
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
        height, width = sizes[0].frame.height, sizes[0].frame.width
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
                    "requested raster"
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
