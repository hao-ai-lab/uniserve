"""Host media calls: media unit encoding, audio encoding and muxing.

These calls run on host ranks, one codec task at a time per rank. A video
encode consumes the RGB media units its rank is handed from a decode round.
When the decoding rank on this host published the round to a shared-storage
segment, the units are borrowed in place, so the codec reads the producer's
bytes directly; otherwise (for example, a round decoded on another host) it
is imported and copied into a host array the task owns. The encoded unit
rows are this rank's product, published when the encodes complete. An audio
encode consumes the request's PCM timeline, imported like any product
because its decoding ranks may be on other hosts; a mux consumes the encoded
unit rows.

``execute`` only schedules the codec work: it configures the host tasks
reserved for the call during batch preparation and returns the call's
``PendingOutput``, which is not ready until those tasks finish.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np

from uniserve_worker._uniserve_ipc import BatchState
from uniserve_worker.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.execution.host import HostTask
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.media.mux import (
    MediaEncoder,
    MediaMux,
    frame_encoded_unit,
    read_encoded_unit,
)
from uniserve_worker.protocol.call import Call, MediaCall
from uniserve_worker.protocol.transfer import (
    ChannelTransfer,
    LocalTransfer,
    PosixShmTransfer,
    TensorTransfer,
)

if TYPE_CHECKING:
    import torch

    from uniserve_worker._uniserve_ipc import SharedRead
    from uniserve_worker.config.deployment import ComponentConfig
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.protocol.batch import TensorExport
    from uniserve_worker.storage.tensor_store import TensorStore
    from uniserve_worker.transport.interface import Transport
    from uniserve_worker.transport.shm import ShmTransport

__all__ = ["HOST_MEDIA_CALLS", "execute"]

#: The calls host ranks serve.
HOST_MEDIA_CALLS = frozenset(
    {MediaCall.VIDEO_ENCODING, MediaCall.AUDIO_ENCODING, MediaCall.MUXING}
)


def encoded_unit_positions(
    call: Call, *, state: BatchState, component: ComponentConfig, rank: int
) -> tuple[int, ...]:
    """Return the positions within the decode round this rank encodes.

    A round's units are dealt to the encoder's ranks in order, each taking
    ``units_per_rank`` consecutive positions, the same order the engine used
    to project the call onto its ranks. The result is empty for a rank whose
    first position is at or past the round's ``max_units``.

    Raises:
        ValueError: When ``rank`` is not one of the component's ranks.
    """
    from uniserve_worker.execution.media import decode_range

    params = decode_range(call, state=state)
    per_rank = max(1, int(component.units_per_rank))
    position = component.ranks.index(rank)
    first = position * per_rank
    return tuple(range(first, min(first + per_rank, int(params.max_units))))


def _input_export(
    call: Call, state: BatchState, index: int = 0
) -> TensorExport:
    """Return the export carrying one of the call's inputs."""
    if len(call.inputs) <= index:
        raise invalid_descriptor("host media call lacks its input product")
    product = call.inputs[index]
    for payload in state.batch.input_products:
        if payload.product == product:
            return payload
    raise invalid_descriptor("host media input has no published locations")


def _shm(transports: Mapping[str, Transport]) -> ShmTransport:
    """Return the rank's shared-storage transport, bound under ``"shm"``."""
    from uniserve_worker.transport.shm import ShmTransport

    transport = transports.get("shm")
    if not isinstance(transport, ShmTransport):
        raise unsupported_setup(
            "host media inputs are borrowed over shared storage, which this "
            "rank does not bind"
        )
    return transport


def _locations(export: TensorExport) -> str:
    """Describe an export's locations for an error naming them."""
    tensor = export.value.tensor
    return "shape {} locations {}".format(
        tuple(tensor.shape),
        [
            (location.backend, tuple(location.offset), tuple(location.shape))
            for location in tensor.locations
        ],
    )


def _borrow(
    export: TensorExport,
    row: int,
    *,
    transports: Mapping[str, Transport],
) -> SharedRead:
    """Borrow one leading-axis row of a product in place.

    The location holding the row must be a shared-storage segment on this
    host: a host product's other mechanisms carry copies, which an encoder
    reading in place has no use for. The borrow spans the row's full padded
    extent; the caller releases it, or hands it to the encode task that does.
    """
    shm = _shm(transports)
    tensor = export.value.tensor
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
        + _locations(export)
    )


def read_encoded_units(
    tensor: TensorTransfer, *, transports: Mapping[str, Transport]
) -> tuple[bytes, ...]:
    """Read initialized encoded rows, retaining shared storage through the copy.

    Logical row capacity bounds encoding; physical locations carry only the
    framed bytes. Shared storage reaches this host and channel bytes reach
    other hosts. Never import the uninitialized remainder of a logical row.

    Locations are tried in export order and skipped when their
    mechanism is not reachable from this rank (shared storage when this rank
    binds none or it lies on another node, a local transfer when this rank
    binds none or it lies in another address space, or any other mechanism
    than a channel) or when every row they carry was already read.

    Raises:
        WorkerError: When ``tensor`` is not 2-D ``uint8``, a location does
            not start at its row's length prefix, a row names an invalid
            length, or some row has no reachable location.
    """
    import torch

    from uniserve_worker.transport.shm import ShmTransport

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
        # Each location's mechanism was found reachable above.
        if isinstance(handle, LocalTransfer):
            assert local is not None
            ticket = local.fetch(location, device=torch.device("cpu"))
            try:
                # A product region is published as one tensor, so a borrowed
                # read yields that tensor rather than first-axis spans.
                rows = ticket.result()
                assert isinstance(rows, torch.Tensor)
                for index, row in zip(indices, rows.unbind(0), strict=True):
                    units[index] = read_encoded_unit(row)
            finally:
                ticket.close()
            continue
        if isinstance(handle, PosixShmTransfer):
            assert isinstance(shm, ShmTransport)
            borrow = shm.borrow(location)
            try:
                raw = bytearray(borrow)
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


def _host_array(value: torch.Tensor) -> np.ndarray:
    """Copy an imported tensor's bytes into a host array the rank owns.

    An input imported from another host lives in this rank's tensor store,
    whose storage the call's retirement returns; the codec runs later on the
    lane, so it reads a copy the task owns.
    """
    import torch

    return (
        value.detach().to("cpu").contiguous().view(torch.uint8).numpy().copy()
    )


def execute(
    call: Call,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    media_mux: MediaMux | None,
    export_transports: Mapping[str, Transport],
    transports: Mapping[str, Transport],
    model_runner: ModelExecutor,
) -> PendingOutput:
    """Schedule one host media call on the rank's lane.

    Configures the call's reserved ``HostTask`` slots (one per encoded unit
    position for a video encode, one otherwise) and stages them on the
    returned ``PendingOutput`` as ``host_tasks``. A video encode also stages
    ``host.finish``, which frames and publishes the encoded rows once every
    encode has completed. Audio encodes and unit appends leave their results
    in the request's mux session; the task of the final mux call, which
    carries no inputs, results in a ``MediaOutput`` holding the assembled
    artifact's handle.

    Raises:
        WorkerError: When, for example, the call's inputs, reservations or
            mux resources disagree with its kind.
        RuntimeError: When the call has no reserved lane slots, or
            ``HostTask.configure`` rejects one.
    """
    from uniserve_worker.execution import transfer
    from uniserve_worker.execution.media import mux_config

    request = state.pending_output(call.request_key.request_id)
    media = request.request.admission.diffusion
    if media is None:
        raise invalid_descriptor("host media call has no admitted media")
    reservations = request.host_tasks
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
        export = _input_export(call, state)
        # Batch preparation borrows the round only when it is published over
        # shared storage on this node; otherwise the tensor store holds it.
        imported = not state.inputs.is_borrowed(call.inputs[0].buffer_id)
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
        borrows: list[SharedRead] = []
        try:
            for position, reservation in zip(
                positions, reservations, strict=True
            ):
                # The round's product is indexed from its own first unit;
                # the unit's index in the track names its frame count. The
                # unit's row is padded to the round's longest unit.
                unit = cursor + position
                frames = config.video_unit_frames[unit]
                # RGB24: three bytes per pixel.
                expected = frames * config.height * config.width * 3
                source: SharedRead | np.ndarray
                if imported_units is None:
                    borrow = _borrow(export, position, transports=transports)
                    borrows.append(borrow)
                    if borrow.nbytes < expected:
                        raise invalid_descriptor(
                            "decoded media unit holds fewer bytes than its "
                            "frames"
                        )
                    # Narrow the borrow to this unit's own frames.
                    borrow.truncate(expected)
                    source = borrow
                else:
                    if position >= int(imported_units.shape[0]):
                        raise invalid_descriptor(
                            "video encoding position exceeds the imported "
                            "decoded unit round"
                        )
                    source = _host_array(imported_units[position])
                    if source.size < expected:
                        raise invalid_descriptor(
                            "decoded media unit holds fewer bytes than its "
                            "frames"
                        )
                    source = source[:expected]
                scheduled.append(
                    encoder.unit(
                        call.request_key,
                        config=config,
                        unit_index=unit,
                        source=source,
                        reservation=reservation,
                        call_id=call.call_id,
                    )
                )
        except BaseException:
            # A configured task owns its borrow; release only the rest.
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
                transfer.export_deferred_product(
                    call.outputs[0],
                    write,
                    rows,
                    tensor_store=tensor_store,
                    export_transports=export_transports,
                    consumers=call.consumer_slots,
                    regions=regions,
                ),
            )
            _validate_completion_products(call, products)
            # The batch recorded its products when the call committed, before
            # its encodes ran and with none from this call; append these rows
            # so the batch's result carries them.
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
        pcm = _host_array(read.tensor).view(np.int16)
        tasks = (
            media_mux.audio(
                call.request_key, pcm, reservations[0], call.call_id
            ),
        )

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
            if state.inputs.is_borrowed(product.buffer_id):
                export = _input_export(call, state, index)
                units.extend(
                    read_encoded_units(
                        export.value.tensor,
                        transports=export_transports,
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

    request.set_host_tasks(tasks, finish=finish)
    # The call has no product at commit; a video encode's ``finish`` sets
    # them once its encodes complete.
    request.products = ()
    return request
