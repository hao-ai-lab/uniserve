"""Media unit encoding and artifact assembly scheduled on a host rank's lane.

A host rank encodes the media units it is handed and, when it is the muxer,
encodes the audio track and concatenates the encoded tracks in the mux
session's codec process. Every input is a host product borrowed in place from
the shared-memory segment its producer published. This module owns the
rank-side scheduling: which job runs for which call, on which borrow, and
what its result becomes.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..foundation.errors import invalid_descriptor
from ..protocol.identity import CallId, RequestKey
from ..protocol.output import MediaOutput, PosixShmArtifact
from .codec_process import (
    AvMuxConfig,
    EncodeAudioTrack,
    EncodeVideoUnit,
    MuxAppend,
    MuxFinalize,
    SessionKey,
    SharedSlice,
    require_media_codecs,
)

if TYPE_CHECKING:
    import torch

    from ..runtime.host_lane import HostLane, HostTask
    from ..transfer.tickets import HostBorrow

__all__ = [
    "AvMuxConfig",
    "MediaEncoder",
    "MediaMux",
    "encoded_unit_bytes",
    "require_media_codecs",
]

# A framed media unit carries its own length because the product row that holds
# it is sized for the largest unit a request can produce.
_LENGTH_BYTES = 8
# Container structure and the codec's parameter sets cost about a kilobyte and a
# half regardless of raster, so a row is never smaller than this.
_CONTAINER_FLOOR = 1 << 16


def encoded_unit_bytes(frames: int, height: int, width: int) -> int:
    """Bytes a framed encoded media unit occupies, including its length.

    The bound is a quarter of the planar raster the encoder consumes, which is
    about seven times the largest unit measured on real output at the serving
    encoder's settings. A small unit is dominated by the container and its
    parameter sets rather than by its raster, so the bound does not fall below
    the floor those cost. A unit that exceeds the bound fails by name.
    """
    raster = int(frames) * int(height) * int(width)
    return _LENGTH_BYTES + max(raster * 3 // 8, _CONTAINER_FLOOR)


def frame_encoded_unit(payload: bytes, destination: torch.Tensor) -> None:
    """Write one encoded media unit and its length into a product row."""
    import torch

    capacity = int(destination.numel()) - _LENGTH_BYTES
    if len(payload) > capacity:
        raise invalid_descriptor(
            f"encoded media unit of {len(payload)} bytes exceeds the "
            f"{capacity} bytes reserved for it"
        )
    header = np.frombuffer(
        np.uint64(len(payload)).tobytes(), dtype=np.uint8
    ).copy()
    destination[:_LENGTH_BYTES].copy_(torch.from_numpy(header))
    body = np.frombuffer(payload, dtype=np.uint8).copy()
    destination[_LENGTH_BYTES : _LENGTH_BYTES + len(payload)].copy_(
        torch.from_numpy(body)
    )


def read_encoded_unit(row: torch.Tensor) -> bytes:
    """Return the encoded media unit a product row carries."""
    values = row.numpy()
    length = int(
        np.frombuffer(values[:_LENGTH_BYTES].tobytes(), dtype=np.uint64)[0]
    )
    if length > len(values) - _LENGTH_BYTES:
        raise invalid_descriptor("encoded media unit names an invalid length")
    return values[_LENGTH_BYTES : _LENGTH_BYTES + length].tobytes()


def session_key(request_key: RequestKey) -> SessionKey:
    """Name the mux session of one request epoch."""
    return (
        int(request_key.engine_id),
        int(request_key.request_id),
        int(request_key.request_epoch),
    )


@dataclass(slots=True)
class MuxSession:
    """One request epoch's assembly bookkeeping on the muxer rank.

    The container itself lives in the codec process that owns the session.
    """

    config: AvMuxConfig
    #: The audio encode task, which finalization follows.
    audio_tail: concurrent.futures.Future[object] | None = None
    #: The last append task, which the next append and the finalization follow.
    mux_tail: concurrent.futures.Future[object] | None = None
    finalized: bool = False


class MediaEncoder:
    """One host rank's encoder for the media units it is handed."""

    def __init__(self, *, rank: int) -> None:
        self.rank = rank

    def unit(
        self,
        request_key: RequestKey,
        *,
        config: AvMuxConfig,
        unit_index: int,
        source: HostBorrow,
        reservation: HostTask,
        call_id: CallId,
    ) -> HostTask:
        """Schedule one media unit's encode from its borrowed RGB bytes.

        The bytes become this call's product when the job completes; the
        encoded length is not known until then. The borrow is released, and
        the producer's segment acknowledged, once the encoder has read it.
        """
        return reservation.configure(
            EncodeVideoUnit(
                config,
                SharedSlice(source.segment, source.offset, source.nbytes),
            ),
            dependencies=(),
            input_ready=None,
            input_completion=None,
            release=source.release,
            profile_name=(
                f"uniserve.host.encode request={_key_label(request_key)} "
                f"step={call_id.batch_id} "
                f"op={call_id.request_index} "
                f"kind=video unit={unit_index} rank={self.rank}"
            ),
        )


class MediaMux:
    """Request-indexed artifact assembly on the muxer rank."""

    def __init__(self, *, rank: int, lane: HostLane) -> None:
        self.rank = rank
        self._lane = lane
        self._sessions: dict[RequestKey, MuxSession] = {}

    def open(self, request_key: RequestKey, *, config: AvMuxConfig) -> None:
        """Create the request-owned assembly session under its settings."""
        if request_key in self._sessions:
            return
        self._sessions[request_key] = MuxSession(config)

    def config(self, request_key: RequestKey) -> AvMuxConfig:
        """Return the container settings this request assembles under."""
        session = self._sessions.get(request_key)
        if session is None:
            raise invalid_descriptor("media output has no active session")
        return session.config

    def audio(
        self,
        request_key: RequestKey,
        sources: tuple[HostBorrow, ...],
        reservation: HostTask,
        call_id: CallId,
    ) -> HostTask:
        """Schedule the audio track's encode from its borrowed PCM bytes.

        The borrows hold the raw PCM timeline in sample order, one per
        decoding rank's publication, which the job reads as stereo int16
        samples. The encoded track is the session's own state rather than a
        result, because only the assembled artifact is this request's output.
        """
        session = self._sessions.get(request_key)
        if session is None:
            raise invalid_descriptor("media output has no active session")
        if session.finalized or session.audio_tail is not None:
            raise invalid_descriptor("audio output is already written")

        key = session_key(request_key)

        def release() -> None:
            for source in sources:
                source.release()

        task = reservation.configure(
            EncodeAudioTrack(
                key,
                session.config,
                tuple(
                    SharedSlice(source.segment, source.offset, source.nbytes)
                    for source in sources
                ),
            ),
            dependencies=(),
            input_ready=None,
            input_completion=None,
            release=release,
            profile_name=(
                f"uniserve.host.encode request={_key_label(request_key)} "
                f"step={call_id.batch_id} "
                f"op={call_id.request_index} "
                f"kind=audio rank={self.rank}"
            ),
            session=key,
        )
        session.audio_tail = task.promise
        return task

    def append_units(
        self,
        request_key: RequestKey,
        units: tuple[bytes, ...],
        reservation: HostTask,
        call_id: CallId,
    ) -> HostTask:
        """Schedule the next media units into the request's container.

        Units of consecutive rounds are appended in order, so a round's task
        follows the previous round's on the host lane through its dependency.
        """
        session = self._sessions.get(request_key)
        if session is None or session.finalized:
            raise invalid_descriptor(
                "media assembly requires an open assembly session"
            )
        if not units:
            raise invalid_descriptor("media assembly received no media units")
        dependencies = () if session.mux_tail is None else (session.mux_tail,)

        key = session_key(request_key)
        task = reservation.configure(
            MuxAppend(key, session.config, tuple(units)),
            dependencies=dependencies,
            input_ready=None,
            input_completion=None,
            release=None,
            profile_name=(
                f"uniserve.host.mux request={_key_label(request_key)} "
                f"step={call_id.batch_id} "
                f"op={call_id.request_index} "
                f"kind=units rank={self.rank}"
            ),
            session=key,
        )
        session.mux_tail = task.promise
        return task

    def finalize_artifact(
        self,
        request_key: RequestKey,
        reservation: HostTask,
        call_id: CallId,
    ) -> HostTask:
        """Schedule the artifact's assembly after every unit and the audio.

        The engine schedules this call after the last encode round and the
        audio track have completed, so the task depends on the last append
        and on the audio encode rather than on any input of its own. The job
        publishes the artifact from the codec process; its result is the
        artifact handle.
        """
        session = self._sessions.get(request_key)
        if session is None or session.finalized:
            raise invalid_descriptor(
                "media finalization requires an open assembly session"
            )
        if session.audio_tail is None:
            raise invalid_descriptor(
                "artifact finalization precedes the audio track's encode"
            )
        dependencies = tuple(
            promise
            for promise in (session.mux_tail, session.audio_tail)
            if promise is not None
        )

        key = session_key(request_key)
        task = reservation.configure(
            MuxFinalize(key),
            dependencies=dependencies,
            input_ready=None,
            input_completion=None,
            release=None,
            profile_name=(
                f"uniserve.host.mux request={_key_label(request_key)} "
                f"step={call_id.batch_id} "
                f"op={call_id.request_index} "
                f"kind=artifact rank={self.rank}"
            ),
            session=key,
            ends_session=True,
            transform=_artifact_output,
        )
        session.finalized = True
        return task

    def drop(self, request_id: int) -> None:
        """Remove every assembly session owned by a request identifier."""
        for key in [
            key for key in self._sessions if key.request_id == int(request_id)
        ]:
            self._discard(key)

    def close(self) -> None:
        """Discard all active assembly sessions and reject new media work."""
        for key in list(self._sessions):
            self._discard(key)

    def _discard(self, request_key: RequestKey) -> None:
        """Forget a session here and in the codec process that holds it."""
        session = self._sessions.pop(request_key)
        if not session.finalized:
            self._lane.discard_session(session_key(request_key))


def _artifact_output(value: object) -> MediaOutput:
    """Wrap a finalized artifact's shared-memory name and size."""
    name, nbytes = value  # type: ignore[misc]
    return MediaOutput(
        handle=PosixShmArtifact(name=str(name)), bytes=int(nbytes)
    )


def _key_label(request_key: RequestKey) -> str:
    """Format a stable request key for media task profiling."""
    return (
        f"{request_key.engine_id}:{request_key.request_id}:"
        f"{request_key.request_epoch}"
    )
