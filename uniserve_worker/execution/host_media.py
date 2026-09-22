"""Host media calls: media unit encoding, audio encoding and muxing.

These calls run on host ranks. A video encode consumes the RGB media units
its rank is handed from a decode round, borrowed in place from the
shared-storage segment the decoding rank on this host published, so a codec
process reads the producer's bytes directly; the encoded unit rows are this
rank's product, published when the encodes complete. An audio encode
consumes the request's PCM timeline, imported like any product because its
decoding ranks may be on other hosts, and staged in a segment of this rank's
own for the codec; a mux consumes the encoded unit rows.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from uniserve_worker.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.execution import calls
from uniserve_worker.execution.batch import BatchState
from uniserve_worker.execution.host import HostTask
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.media.mux import (
    MediaEncoder,
    MediaMux,
    frame_encoded_unit,
    read_encoded_unit,
)
from uniserve_worker.protocol.call import Call, CallStatus, MediaCall
from uniserve_worker.protocol.output import FinishFlags
from uniserve_worker.protocol.transfer import (
    ChannelTransfer,
    LocalTransfer,
    PosixShmTransfer,
    TensorTransfer,
)

if TYPE_CHECKING:
    import torch

    from uniserve_worker.config.deployment import ComponentConfig
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.protocol.batch import TensorPublication
    from uniserve_worker.storage.tensor_store import TensorStore
    from uniserve_worker.transport.interface import Transport
    from uniserve_worker.transport.shm import HostBorrow, ShmTransport

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
    from uniserve_worker.execution.media import decode_range

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
            "host media inputs are borrowed over shared storage, which this "
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

    The location holding the row must be a shared-storage segment on this
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
        f"media unit {row} is not published over shared storage on this host: "
        + _locations(publication)
    )


def read_encoded_units(
    tensor: TensorTransfer, *, transports: Mapping[str, Transport]
) -> tuple[bytes, ...]:
    """Read initialized encoded rows, retaining shared storage through the copy.

    Logical row capacity bounds encoding; physical locations carry only the
    framed bytes. Shared storage reaches this host and channel bytes reach
    other hosts. Never import the uninitialized remainder of a logical row.
    """
    import torch

    from uniserve_worker.transport.shared_storage import open_shared_storage

    if len(tensor.shape) != 2 or tensor.dtype != "uint8":
        raise invalid_descriptor("encoded units require byte rows")
    units: dict[int, bytes] = {}
    shm = transports.get("shm")
    for location in tensor.locations:
        handle = location.transport
        if isinstance(handle, PosixShmTransfer):
            if shm is None or location.source.node != shm.source.node:
                continue
        elif isinstance(handle, LocalTransfer):
            local = transports.get("local")
            if (
                local is None
                or location.source.address_space != local.source.address_space
            ):
                continue
        elif not isinstance(handle, ChannelTransfer):
            continue
        if location.offset[1] != 0:
            raise invalid_descriptor(
                "encoded unit location lacks its length prefix"
            )
        first = location.offset[0]
        indices = range(first, first + location.shape[0])
        if all(index in units for index in indices):
            continue
        if isinstance(handle, LocalTransfer):
            ticket = local.fetch(location, device=torch.device("cpu"))
            try:
                rows = ticket.result()
                for index, row in zip(indices, rows.unbind(0), strict=True):
                    units[index] = read_encoded_unit(row)
            finally:
                ticket.close()
            continue
        if isinstance(handle, PosixShmTransfer):
            borrow = shm.borrow(location)
            try:
                with open_shared_storage(
                    borrow.segment, borrow.offset + borrow.nbytes
                ) as mapping:
                    raw = bytearray(
                        mapping[borrow.offset : borrow.offset + borrow.nbytes]
                    )
            finally:
                borrow.release()
        else:
            raw = bytearray(handle.payload)
        rows = torch.frombuffer(raw, dtype=torch.uint8).reshape(location.shape)
        for index, row in zip(indices, rows.unbind(0), strict=True):
            units[index] = read_encoded_unit(row)
    if len(units) != tensor.shape[0]:
        raise invalid_descriptor(
            f"artifact assembly requires every encoded media unit: received "
            f"{sorted(units)} for {tensor.shape[0]} rows"
        )
    return tuple(units[index] for index in range(tensor.shape[0]))


def _stage_tensor(value: torch.Tensor) -> HostBorrow:
    """Copy a tensor into a segment of this rank's own for a codec process.

    Codecs read media bytes from a named shared-storage segment. An input
    imported from another host instead lives in this rank's tensor store, so
    it is copied to a local segment whose borrow unlinks it after the codec has
    read it.
    """
    import torch

    from uniserve_worker.transport.shared_storage import allocate_shared_storage
    from uniserve_worker.transport.shm import HostBorrow

    raw = value.detach().to("cpu").contiguous().view(torch.uint8).reshape(-1)
    nbytes = int(raw.numel())
    if nbytes < 1:
        raise invalid_descriptor("codec input is empty")
    segment = allocate_shared_storage(nbytes)
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
    model_runner: ModelExecutor,
) -> PendingOutput:
    """Schedule one host media call on the rank's lane."""
    from uniserve_worker.execution import transfer
    from uniserve_worker.execution.media import mux_config

    request = state.pending_output(call.request_key.request_id)
    media = request.request.admission.diffusion
    if media is None:
        raise invalid_descriptor("host media call has no admitted media")
    reservations = request.host.tasks
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
        from uniserve_worker.execution.media import decode_range

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
            from uniserve_worker.execution.commit import (
                _validate_completion_products,
            )

            regions = []
            for index, (row, result) in enumerate(
                zip(rows.unbind(0), results, strict=True)
            ):
                if not isinstance(result, bytes):
                    raise RuntimeError("media unit encode produced no bytes")
                framed = frame_encoded_unit(result, row)
                regions.append(
                    (slice(index, index + 1), slice(0, framed.numel()))
                )
            products = (
                transfer.publish_deferred_product(
                    call.outputs[0],
                    write,
                    rows,
                    tensor_store=tensor_store,
                    publication_transports=publication_transports,
                    consumers=call.consumer_slots,
                    regions=regions,
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
        for index, product in enumerate(call.inputs):
            if product.buffer_id in state.borrowed_inputs:
                publication = _input_publication(call, state, index)
                units.extend(
                    read_encoded_units(
                        publication.value.tensor,
                        transports=publication_transports,
                    )
                )
                continue

            # Wholly resident products arrive by identity, without wire
            # locations. The producer already committed their reserved rows
            # to this rank's tensor store; retain the read through consumption.
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
            units.extend(
                read_encoded_unit(row) for row in read.tensor.unbind(0)
            )
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
    request.host.tasks = tasks
    request.host.finish = finish
    request.products = ()
    return request
