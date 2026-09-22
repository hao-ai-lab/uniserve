"""Host media calls: media unit encoding, audio encoding and muxing.

These calls run on host ranks. A video encode consumes the RGB media units
its rank is handed from a decode round, borrowed in place from the
shared-memory segment the decoding rank on this host published, so a codec
process reads the producer's bytes directly; the encoded unit rows are this
rank's product, published when the encodes complete. An audio encode
consumes the request's PCM timeline, imported like any product because its
decoding ranks may be on other hosts, and staged in a segment of this rank's
own for the codec; a mux consumes the encoded unit rows.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from uniserve_worker.foundation.errors import (
    invalid_descriptor,
    unsupported_setup,
)
from uniserve_worker.media.mux import (
    MediaEncoder,
    MediaMux,
    frame_encoded_unit,
    read_encoded_unit,
)
from uniserve_worker.protocol.call import Call, CallStatus, MediaCall
from uniserve_worker.protocol.output import FinishFlags
from uniserve_worker.protocol.transfer import PosixShmTransfer
from uniserve_worker.runtime.host_lane import HostTask

from . import calls
from .batch_state import BatchState
from .output import PendingOutput

if TYPE_CHECKING:
    import torch

    from uniserve_worker.bootstrap.config import ComponentConfig
    from uniserve_worker.protocol.batch import TensorPublication
    from uniserve_worker.runtime.tensor_store import TensorStore
    from uniserve_worker.transfer.tickets import (
        HostBorrow,
        ShmTransport,
        Transport,
    )

    from .model_runner import ModelRunner

__all__ = ["BORROWED_INPUT_CALLS", "HOST_MEDIA_CALLS", "execute"]

#: The calls host ranks serve.
HOST_MEDIA_CALLS = frozenset(
    {MediaCall.VIDEO_ENCODING, MediaCall.AUDIO_ENCODING, MediaCall.MUXING}
)
#: The calls whose media inputs are read in place rather than imported.
BORROWED_INPUT_CALLS = frozenset({MediaCall.VIDEO_ENCODING})


def encoded_unit_positions(
    call: Call, *, state: BatchState, component: ComponentConfig, rank: int
) -> tuple[int, ...]:
    """Return the positions within the decode round this rank encodes.

    A round's units are dealt to the encoder's ranks in order, each taking
    ``units_per_rank`` consecutive positions, the same order the engine used
    to project the call onto its ranks.
    """
    from .video import decode_range

    params = decode_range(call, state=state)
    per_rank = max(1, int(component.units_per_rank))
    position = component.ranks.index(rank)
    first = position * per_rank
    return tuple(range(first, min(first + per_rank, int(params.max_units))))


def _input_publication(
    call: Call, state: BatchState, index: int = 0
) -> TensorPublication:
    """Return the publication carrying one of the call's inputs."""
    if len(call.inputs) <= index:
        raise invalid_descriptor("host media call lacks its input product")
    product = call.inputs[index]
    for payload in state.input_products:
        if payload.product == product:
            return payload
    raise invalid_descriptor("host media input has no published locations")


def _shm(transports: Mapping[str, Transport]) -> ShmTransport:
    transport = transports.get("shm")
    if transport is None:
        raise unsupported_setup(
            "host media inputs are borrowed over shared memory, which this "
            "rank does not bind"
        )
    return transport  # type: ignore[return-value]


def _locations(publication: TensorPublication) -> str:
    """Describe a publication's locations for an error naming them."""
    tensor = publication.value.tensor
    return "shape {} locations {}".format(
        tuple(tensor.shape),
        [
            (location.backend, tuple(location.offset), tuple(location.shape))
            for location in tensor.locations
        ],
    )


def _borrow(
    publication: TensorPublication,
    row: int,
    *,
    transports: Mapping[str, Transport],
) -> HostBorrow:
    """Borrow one leading-axis row of a product in place.

    The location holding the row must be a shared-memory segment on this
    host: a host product's other mechanisms carry copies, which an encoder
    reading in place has no use for.
    """
    shm = _shm(transports)
    tensor = publication.value.tensor
    for location in tensor.locations:
        if not isinstance(location.transport, PosixShmTransfer):
            continue
        start = location.offset[0] if location.offset else 0
        if start <= row < start + int(location.shape[0]):
            region = (
                slice(row, row + 1),
                *(slice(0, extent) for extent in tensor.shape[1:]),
            )
            return shm.borrow(location, region)
    raise invalid_descriptor(
        f"media unit {row} is not published over shared memory on this host: "
        + _locations(publication)
    )


def _stage_tensor(value: torch.Tensor) -> HostBorrow:
    """Copy a tensor into a segment of this rank's own for a codec process.

    Codecs read media bytes from a named shared-memory segment. An input
    imported from another host instead lives in this rank's tensor store, so
    it is copied to a local segment whose borrow unlinks it after the codec has
    read it.
    """
    import torch

    from uniserve_worker.foundation.shared_memory import (
        allocate_shared_memory,
    )
    from uniserve_worker.transfer.tickets import HostBorrow

    raw = value.detach().to("cpu").contiguous().view(torch.uint8).reshape(-1)
    nbytes = int(raw.numel())
    if nbytes < 1:
        raise invalid_descriptor("codec input is empty")
    segment = allocate_shared_memory(nbytes)
    try:
        # The mapping is dropped before the borrow is released, so closing
        # the segment finds no exported buffer.
        torch.frombuffer(segment.buf, dtype=torch.uint8).copy_(raw)
    except BaseException:
        segment.close()
        segment.unlink()
        raise

    def release() -> None:
        segment.close()
        segment.unlink()

    return HostBorrow(
        segment=segment.name, offset=0, nbytes=nbytes, _release=release
    )


def execute(
    call: Call,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    media_mux: MediaMux | None,
    publication_transports: Mapping[str, Transport],
    transports: Mapping[str, Transport],
    model_runner: ModelRunner,
) -> PendingOutput:
    """Schedule one host media call on the rank's lane."""
    from . import transfer
    from .video import mux_config

    request = state.pending_output(call.request_key.request_id)
    media = request.request.admission.diffusion
    if media is None:
        raise invalid_descriptor("host media call has no admitted media")
    reservations = request.completion_tasks
    if not reservations:
        raise RuntimeError("host media call has no reserved lane slots")

    tasks: tuple[HostTask, ...] = ()
    finish: Callable[[tuple[object, ...]], None] | None = None
    if call.kind is MediaCall.VIDEO_ENCODING:
        binding = model_runner.bindings[call.component]
        positions = encoded_unit_positions(
            call,
            state=state,
            component=binding.config,
            rank=model_runner.worker_config.rank,
        )
        if len(positions) != len(reservations):
            raise invalid_descriptor(
                "video encoding reserved a different number of lane slots "
                "than the media units it takes"
            )
        from .video import decode_range

        cursor = int(decode_range(call, state=state).cursor)
        config = mux_config(model_runner, media)
        publication = _input_publication(call, state)
        imported = call.inputs[0].buffer_id not in state.borrowed_inputs
        imported_read = None
        imported_units = None
        if imported:
            imported_read = tensor_store.consume(
                call.inputs[0],
                consumer_call_id=call.call_id,
                device=model_runner.call_devices(call)[0],
            )
            request.device_reads.append(imported_read)
            if imported_read.region is not None or imported_read.tensor is None:
                raise invalid_descriptor(
                    "video encoding requires the complete decoded unit round"
                )
            imported_units = imported_read.tensor
        encoder = MediaEncoder(rank=model_runner.worker_config.rank)
        scheduled: list[HostTask] = []
        borrows: list[HostBorrow] = []
        try:
            for position, reservation in zip(
                positions, reservations, strict=True
            ):
                # The round's product is indexed from its own first unit;
                # the unit's index in the track names its frame count.
                unit = cursor + position
                if imported_units is None:
                    borrow = _borrow(
                        publication, position, transports=transports
                    )
                else:
                    if position >= int(imported_units.shape[0]):
                        raise invalid_descriptor(
                            "video encoding position exceeds the imported "
                            "decoded unit round"
                        )
                    borrow = _stage_tensor(
                        imported_units[position : position + 1]
                    )
                borrows.append(borrow)
                frames = config.video_unit_frames[unit]
                expected = frames * config.height * config.width * 3
                if borrow.nbytes < expected:
                    raise invalid_descriptor(
                        "decoded media unit holds fewer bytes than its frames"
                    )
                borrow.nbytes = expected
                scheduled.append(
                    encoder.unit(
                        call.request_key,
                        config=config,
                        unit_index=unit,
                        source=borrow,
                        reservation=reservation,
                        call_id=call.call_id,
                    )
                )
        except BaseException:
            for borrow in borrows[len(scheduled) :]:
                borrow.release()
            raise
        tasks = tuple(scheduled)

        # The encoded rows are this call's product: reserved now at the
        # rank's positions of the round, filled and published once every
        # encode has completed.
        write, rows = transfer.reserved_unit_rows(
            call, state=state, count=len(positions)
        )
        tensor_store.defer_write(write)

        def publish(results: tuple[object, ...]) -> None:
            from .commit import _validate_completion_products

            for row, result in zip(rows.unbind(0), results, strict=True):
                if not isinstance(result, bytes):
                    raise RuntimeError("media unit encode produced no bytes")
                frame_encoded_unit(result, row)
            products = (
                transfer.publish_deferred_product(
                    call.outputs[0],
                    write,
                    rows,
                    tensor_store=tensor_store,
                    publication_transports=publication_transports,
                    consumers=call.consumer_slots,
                ),
            )
            _validate_completion_products(call, products)
            # Products were recorded when the call was committed,
            # before its encodes ran; these join them for the batch's result.
            request.products = products
            state.products = (*state.products, *products)

        finish = publish

    elif call.kind is MediaCall.AUDIO_ENCODING:
        if media_mux is None:
            raise unsupported_setup("audio encoding has no muxer resources")
        config = mux_config(model_runner, media)
        media_mux.open(call.request_key, config=config)
        if len(call.inputs) != 1:
            raise invalid_descriptor("audio encoding requires one PCM input")
        # The timeline's shards come from every audio decoding rank, on this
        # host or another, so it is imported like any product and read as
        # one complete tensor.
        read = tensor_store.consume(
            call.inputs[0],
            consumer_call_id=call.call_id,
            device=model_runner.call_devices(call)[0],
        )
        request.device_reads.append(read)
        if read.region is not None or read.tensor is None:
            raise invalid_descriptor(
                "audio encoding requires the complete PCM timeline"
            )
        track = _stage_tensor(read.tensor)
        try:
            tasks = (
                media_mux.audio(
                    call.request_key, track, reservations[0], call.call_id
                ),
            )
        except BaseException:
            track.release()
            raise

    elif call.kind is MediaCall.MUXING:
        if media_mux is None:
            raise unsupported_setup("artifact assembly has no muxer resources")
        media_mux.open(call.request_key, config=mux_config(model_runner, media))
        # A muxing call carries the encode rounds completed since the last
        # one, one product per round in media unit order, and the muxer
        # appends them to the request's container. The final call carries
        # none: every unit and the audio track are in, and it assembles the
        # artifact.
        units: list[bytes] = []
        for product in call.inputs:
            read = tensor_store.consume(
                product,
                consumer_call_id=call.call_id,
                device=model_runner.call_devices(call)[0],
            )
            request.device_reads.append(read)
            if read.region is not None or read.tensor is None:
                raise invalid_descriptor(
                    "artifact assembly requires every encoded media unit"
                )
            rows = read.tensor.to("cpu")
            units.extend(read_encoded_unit(row) for row in rows.unbind(0))
        if units:
            tasks = (
                media_mux.append_units(
                    call.request_key,
                    tuple(units),
                    reservations[0],
                    call.call_id,
                ),
            )
        else:
            tasks = (
                media_mux.finalize_artifact(
                    call.request_key, reservations[0], call.call_id
                ),
            )
    else:
        raise invalid_descriptor(f"unsupported host media call {call.kind!r}")

    request.status = CallStatus.OK
    # Host media calls consume products without advancing any trajectory.
    request.progress = calls.execution_runtime(request, None)
    request.finish_flags = FinishFlags()
    request.product_generations = calls.output_generations(call)
    request.completion_tasks = tasks
    request.finish = finish
    request.products = ()
    return request
