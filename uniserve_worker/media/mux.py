"""Media unit encoding and artifact assembly scheduled on a host rank's lane.

A host rank encodes the media units it is handed and, when it is the muxer,
encodes the audio track and assembles the encoded tracks into the request's
container, each as a task on the rank's lane. A unit decoded on this host is
read in place from the shared-storage segment its producer published. This
module owns the rank-side scheduling: which task runs for which call, on
which input, and what its result becomes. Encoded units reach the muxer as
length-prefixed rows of the encode call's product (`frame_encoded_unit`,
`read_encoded_unit`).
"""

from __future__ import annotations

import mmap
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.media.container import (
    AvMuxConfig,
    AvMuxSession,
    encode_audio_track,
    encode_video_unit,
    encoded_video_bytes,
    require_media_codecs,
)
from uniserve_worker.media.storage import publish_media_bytes
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.protocol.output import MediaOutput, PosixShmArtifact

if TYPE_CHECKING:
    import torch

    from uniserve_worker.execution.host import HostTask
    from uniserve_worker.transport.shm import HostBorrow

__all__ = [
    "AvMuxConfig",
    "MediaEncoder",
    "MediaMux",
    "encoded_unit_bytes",
    "require_media_codecs",
]

# A framed media unit carries its own length because the product row that holds
# it is sized for the largest unit a request can produce. The prefix is a
# uint64 in the host's native byte order, written by `frame_encoded_unit` and
# read by `read_encoded_unit`.
_LENGTH_BYTES = 8


def encoded_unit_bytes(frames: int, height: int, width: int) -> int:
    """Storage bound for a serving codec unit, including its length prefix."""
    return _LENGTH_BYTES + encoded_video_bytes(frames, height, width)


def frame_encoded_unit(
    payload: bytes, destination: torch.Tensor
) -> torch.Tensor:
    """Write a framed unit and return its initialized prefix for publication.

    ``destination`` is a 1-D uint8 row sized by `encoded_unit_bytes`. Bytes
    past the returned view are left untouched; the caller
    (`uniserve_worker.execution.host_media`) publishes only the view's span
    as the row's region.

    Raises:
        WorkerError: When ``payload`` does not fit in the row after the
            length prefix.
    """
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
    return destination[: _LENGTH_BYTES + len(payload)]


def read_encoded_unit(row: torch.Tensor) -> bytes:
    """Return the encoded media unit a framed CPU uint8 row carries.

    Raises:
        WorkerError: When the length prefix exceeds the row's capacity.
    """
    values = row.numpy()
    length = int(
        np.frombuffer(values[:_LENGTH_BYTES].tobytes(), dtype=np.uint64)[0]
    )
    if length > len(values) - _LENGTH_BYTES:
        raise invalid_descriptor("encoded media unit names an invalid length")
    return values[_LENGTH_BYTES : _LENGTH_BYTES + length].tobytes()


def _pixels(source: HostBorrow | np.ndarray, config: AvMuxConfig) -> np.ndarray:
    """Return a media unit's RGB24 frames without copying borrowed bytes.

    A borrowed unit stays in its producer's shared-storage segment, mapped
    read-only for as long as the returned view lives.
    """
    raw = source if isinstance(source, np.ndarray) else _map(source)
    raster = config.height * config.width * 3
    if raw.size % raster != 0:
        raise invalid_descriptor("video capture has invalid RGB24 dimensions")
    return raw.reshape(-1, config.height, config.width, 3)


def _map(source: HostBorrow) -> np.ndarray:
    """Map a borrowed region of a shared-storage segment read-only.

    The returned array keeps the ``mmap`` object, and so the mapping, alive
    until the array and its views are collected; closing the descriptor
    right after mapping does not invalidate it.

    Raises:
        WorkerError: When the region is negative or extends past the
            segment's size.
        OSError: When the segment cannot be opened or mapped.
    """
    if source.offset < 0 or source.nbytes < 0:
        raise invalid_descriptor(
            "media unit lies outside its shared-storage segment"
        )
    name = source.segment.removeprefix("/")
    descriptor = os.open(f"/dev/shm/{name}", os.O_RDONLY)
    try:
        if source.offset + source.nbytes > os.fstat(descriptor).st_size:
            raise invalid_descriptor(
                "media unit lies outside its shared-storage segment"
            )
        # The mapping starts at offset zero and the array view below selects
        # the region, so ``source.offset`` need not be page-aligned.
        mapping = mmap.mmap(
            descriptor, source.offset + source.nbytes, prot=mmap.PROT_READ
        )
    finally:
        os.close(descriptor)
    return np.frombuffer(
        mapping, dtype=np.uint8, count=source.nbytes, offset=source.offset
    )


@dataclass(slots=True)
class MuxSession:
    """One request epoch's container and encoded audio on the muxer rank.

    Lane tasks hold ``lock`` while they touch the container. A request that
    ends early is marked discarded, and the container is closed only while
    ``lock`` is held, so never under a task that is using it.

    The discard runs on the executor thread and must not wait for a task, so
    it closes the container only if ``lock`` is free. Each task checks the
    mark again after releasing ``lock`` and closes a discarded container
    the same way. A side that finds ``lock`` held leaves the close to the
    holder, which checks the mark after its release, so the container is
    closed once the discard and every task have finished; more than one
    side may close it, which `AvMuxSession.close` permits.

    ``audio_scheduled`` and ``finalized`` are set by `MediaMux` when it
    schedules the corresponding task, before the task runs; ``audio`` is
    written by the audio task under ``lock``.
    """

    config: AvMuxConfig
    container: AvMuxSession
    lock: Lock = field(default_factory=Lock)
    audio: bytes | None = None
    audio_scheduled: bool = False
    finalized: bool = False
    discarded: bool = False

    @contextmanager
    def held(self) -> Iterator[AvMuxSession]:
        """Hold the container for one task, refusing a discarded session."""
        try:
            with self.lock:
                if self.discarded:
                    raise invalid_descriptor("media assembly was discarded")
                yield self.container
        finally:
            # A discard that ran while this task held the lock could not
            # close the container; checking after the release also covers a
            # discard that lands between the task's last use and the release.
            if self.discarded:
                self._close_if_free()

    def discard(self) -> None:
        """Close the container now, or after the task that holds it."""
        self.discarded = True
        self._close_if_free()

    def _close_if_free(self) -> None:
        """Close the container unless another side holds it and will close."""
        if self.lock.acquire(blocking=False):
            try:
                self.container.close()
            finally:
                self.lock.release()


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
        source: HostBorrow | np.ndarray,
        reservation: HostTask,
        call_id: CallId,
    ) -> HostTask:
        """Schedule one media unit's encode from its RGB bytes.

        A borrowed unit is read in place and its borrow released, which
        acknowledges the producer's segment, once the encoder has read it.
        The encoded bytes become this call's product when the task
        completes; their length is not known until then.
        """

        def encode() -> bytes:
            return encode_video_unit(config, _pixels(source, config))

        return reservation.configure(
            encode,
            release=None if isinstance(source, np.ndarray) else source.release,
            profile_name=(
                f"uniserve.host.encode request={_key_label(request_key)} "
                f"step={call_id.batch_id} "
                f"call={call_id.request_index} "
                f"kind=video unit={unit_index} rank={self.rank}"
            ),
        )


class MediaMux:
    """Request-indexed artifact assembly on the muxer rank.

    The engine orders a request's assembly: it schedules the next append only
    after the previous one completed and the final call only after the audio
    track and every unit are in, so the rank's single lane runs them in
    order without dependencies between its tasks.
    """

    def __init__(self, *, rank: int) -> None:
        self.rank = rank
        self._sessions: dict[RequestKey, MuxSession] = {}

    def open(self, request_key: RequestKey, *, config: AvMuxConfig) -> None:
        """Create the request-owned assembly session under its settings.

        Every audio and muxing call opens its session first
        (`uniserve_worker.execution.host_media.execute`), so opening a key
        that already has a session keeps that session and ignores ``config``.
        """
        if request_key in self._sessions:
            return
        self._sessions[request_key] = MuxSession(config, AvMuxSession(config))

    def config(self, request_key: RequestKey) -> AvMuxConfig:
        """Return the container settings this request assembles under."""
        return self._session(request_key).config

    def audio(
        self,
        request_key: RequestKey,
        pcm: np.ndarray,
        reservation: HostTask,
        call_id: CallId,
    ) -> HostTask:
        """Schedule the audio track's encode from its stereo int16 PCM.

        The encoded track is the session's own state rather than a result,
        because only the assembled artifact is this request's output.

        Raises:
            WorkerError: When the request has no session, its audio is
                already scheduled, or its artifact is already finalized.
        """
        session = self._session(request_key)
        if session.finalized or session.audio_scheduled:
            raise invalid_descriptor("audio output is already written")
        session.audio_scheduled = True

        def encode() -> None:
            with session.held():
                session.audio = encode_audio_track(
                    session.config, pcm.reshape(-1, 2)
                )

        return reservation.configure(
            encode,
            profile_name=(
                f"uniserve.host.encode request={_key_label(request_key)} "
                f"step={call_id.batch_id} "
                f"call={call_id.request_index} "
                f"kind=audio rank={self.rank}"
            ),
        )

    def append_units(
        self,
        request_key: RequestKey,
        units: tuple[bytes, ...],
        reservation: HostTask,
        call_id: CallId,
    ) -> HostTask:
        """Schedule the next media units into the request's container.

        Raises:
            WorkerError: When the request has no session, its artifact is
                already finalized, or ``units`` is empty.
        """
        session = self._session(request_key)
        if session.finalized:
            raise invalid_descriptor(
                "media assembly requires an open assembly session"
            )
        if not units:
            raise invalid_descriptor("media assembly received no media units")

        def append() -> None:
            with session.held() as container:
                container.append(tuple(units))

        return reservation.configure(
            append,
            profile_name=(
                f"uniserve.host.mux request={_key_label(request_key)} "
                f"step={call_id.batch_id} "
                f"call={call_id.request_index} "
                f"kind=units rank={self.rank}"
            ),
        )

    def finalize_artifact(
        self,
        request_key: RequestKey,
        reservation: HostTask,
        call_id: CallId,
    ) -> HostTask:
        """Schedule the artifact's assembly after every unit and the audio.

        The task muxes the audio track, publishes the MP4 to shared storage,
        and results in the artifact's handle. Scheduling checks only that the
        audio encode was scheduled; the task itself fails, for example, when
        the encoded audio is missing or `AvMuxSession.finalize` finds a unit
        missing.

        Raises:
            WorkerError: When the request has no session, its artifact is
                already finalized, or its audio encode was never scheduled.
        """
        session = self._session(request_key)
        if session.finalized:
            raise invalid_descriptor(
                "media finalization requires an open assembly session"
            )
        if not session.audio_scheduled:
            raise invalid_descriptor(
                "artifact finalization precedes the audio track's encode"
            )
        session.finalized = True

        def finalize() -> MediaOutput:
            with session.held() as container:
                if session.audio is None:
                    raise invalid_descriptor(
                        "artifact assembly has no encoded audio"
                    )
                payload = container.finalize(session.audio)
            return MediaOutput(
                handle=PosixShmArtifact(name=publish_media_bytes(payload)),
                bytes=len(payload),
            )

        return reservation.configure(
            finalize,
            profile_name=(
                f"uniserve.host.mux request={_key_label(request_key)} "
                f"step={call_id.batch_id} "
                f"call={call_id.request_index} "
                f"kind=artifact rank={self.rank}"
            ),
        )

    def drop(self, request_id: int) -> None:
        """Discard every assembly session whose key carries ``request_id``.

        Keys match on the identifier alone, whatever their epoch or engine.
        A dropped session's scheduled task that has not started fails in
        `MuxSession.held`; see `MuxSession.discard` for one in progress.
        """
        for key in [
            key for key in self._sessions if key.request_id == int(request_id)
        ]:
            self._sessions.pop(key).discard()

    def close(self) -> None:
        """Discard all active assembly sessions.

        The worker calls this when it closes or its startup fails. A later
        `open` still creates a new session.
        """
        for key in list(self._sessions):
            self._sessions.pop(key).discard()

    def _session(self, request_key: RequestKey) -> MuxSession:
        session = self._sessions.get(request_key)
        if session is None:
            raise invalid_descriptor("media output has no active session")
        return session


def _key_label(request_key: RequestKey) -> str:
    """Format a stable request key for media task profiling."""
    return (
        f"{request_key.engine_id}:{request_key.request_id}:"
        f"{request_key.request_epoch}"
    )
