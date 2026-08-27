"""Media-only worker root for the fixed MiniMax H3 serving topology."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Callable, cast

import torch

from ..batch import (
    Batch,
    BatchPartition,
    Close,
    Commit,
    CompletionReport,
    DecodeKind,
    DecodePlacement,
    DevicePoint,
    FinishFlags,
    FixedPoint,
    ForwardMode,
    LogicalLengths,
    ModelOutput,
    Operation,
    OpStatus,
    PartitionCompletion,
    ProductKind,
    RegistrationAck,
    Release,
    SamplingOwnership,
    TimingCounters,
    TokenSpan,
)
from ..capabilities import (
    LaneCapabilities,
    RankInfo,
    RequestKind,
    ResourceClass,
    WorkerCapabilities,
)
from ..foundation.errors import capability_mismatch, invalid_descriptor
from ..models.identity import architecture_identity
from ..models.minimax_h3 import MiniMaxH3Model
from ..models.minimax_h3.execution import (
    H3MuxCoordinator,
    H3OutputRing,
    H3OutputRingLease,
    require_h3_codecs,
)
from ..models.minimax_h3.state import H3StateSlot
from ..runtime.device_events import DeviceEventPool
from ..runtime.device_products import DeviceProductRead, DeviceProducts, DeviceProductWrite
from ..server.completion import DeferredDigest, PinnedOutputBuffer
from ..server.cpu_tasks import BoundedCpuTaskPool, CpuTaskReservation
from ..server.profiler import profile_range

__all__ = ["MediaWorker", "is_h3_checkpoint"]

_MEDIA_WORK = (
    ForwardMode.GEN_TRANSITION,
    ForwardMode.GEN_FLOW,
    ForwardMode.GEN_DECODE,
    ForwardMode.MATERIALIZE,
)


def is_h3_checkpoint(model_path: str) -> bool:
    """Identify the modular H3 composition root without loading its weights."""

    from pathlib import Path

    candidate = Path(model_path).expanduser()
    if candidate.is_dir():
        return (candidate / "modular_model_index.json").is_file() and (
            candidate / "transformer" / "config.json"
        ).is_file()
    normalized = model_path.strip().lower()
    return normalized in {
        "fastvideo/fastvideo-minimax-fasth3-preview-v0.2",
        "fastvideo-minimax-fasth3-preview-v0.2",
    }


class MediaWorker:
    """Execute ordinary media batches serially over one shared H3 scratch lane."""

    @classmethod
    def from_config(cls, config: object) -> "MediaWorker":
        from ..server.distributed import build_device_mesh
        from ..server.worker_kind import WorkerKind

        worker_kind = getattr(config, "worker_kind")
        if worker_kind not in (WorkerKind.FULL, WorkerKind.GEN):
            raise capability_mismatch("MiniMax H3 requires a full or generation worker role")
        placement = getattr(config, "placement")
        if int(placement.tp_size) != 4:
            raise capability_mismatch("MiniMax H3 requires one TP4/SP4 replica")
        mesh = build_device_mesh(
            tp_rank=placement.tp_rank,
            tp_size=placement.tp_size,
            device=placement.device,
            tower_devices=placement.tower_devices,
            tower_primary=0,
            tp_backend=placement.tp_backend,
            tp_init_method=placement.tp_init_method,
        )
        if mesh.local_device.type != "cuda" or torch.cuda.get_device_capability(
            mesh.local_device
        ) < (10, 0):
            raise capability_mismatch("MiniMax H3 requires an SM100-class CUDA device")
        model_config = getattr(config, "model")
        if model_config is None:
            raise capability_mismatch("MiniMax H3 requires a checkpoint")
        state_slots = 2
        unresolved_window = 2
        pipeline_depth = int(getattr(config, "ipc").pipeline_depth)
        required_depth = state_slots * (unresolved_window + 1)
        if pipeline_depth < required_depth:
            raise capability_mismatch(
                f"MiniMax H3 pipeline depth must be at least {required_depth}"
            )
        media_spool_value = getattr(config, "media_spool", None)
        if not media_spool_value:
            raise capability_mismatch("MiniMax H3 requires a shared media spool")
        media_spool = Path(str(media_spool_value)).expanduser()
        if not media_spool.is_absolute():
            raise capability_mismatch("MiniMax H3 media spool must be absolute")
        load = getattr(config, "load")
        model = MiniMaxH3Model.from_pretrained(
            model_config.path,
            mesh,
            state_slots=state_slots,
            cache_dir=load.download_dir,
        )
        return cls(
            model,
            pipeline_depth=pipeline_depth,
            unresolved_window=unresolved_window,
            max_batch_operations=min(
                state_slots, int(getattr(config, "resources").max_batch_operations)
            ),
            media_spool=media_spool,
        )

    def __init__(
        self,
        model: MiniMaxH3Model,
        *,
        pipeline_depth: int,
        unresolved_window: int,
        max_batch_operations: int,
        media_spool: Path,
    ) -> None:
        self.model = model
        self.mesh = model.mesh
        self.device = model.mesh.local_device
        self.media_spool = media_spool
        self.device_events = DeviceEventPool()
        product_capacity = max(1, int(pipeline_depth) * int(max_batch_operations))
        self.device_products = DeviceProducts(
            capacity=product_capacity,
            byte_capacity=max(1 << 20, product_capacity * 64),
            event_pool=self.device_events,
        )
        task_capacity = model.states.slot_count * (int(unresolved_window) + 1)
        self.cpu_tasks = BoundedCpuTaskPool(
            capacity=task_capacity,
            workers=min(4, task_capacity),
        )
        self.mux = H3MuxCoordinator()
        self.output_ring = (
            H3OutputRing(
                state_slots=model.states.slot_count,
                unresolved_window=int(unresolved_window),
            )
            if self.mesh.coord("sp") == 0
            else None
        )
        self.capabilities = self._build_capabilities(
            pipeline_depth=int(pipeline_depth),
            unresolved_window=int(unresolved_window),
            max_batch_operations=int(max_batch_operations),
        )

    def _build_capabilities(
        self,
        *,
        pipeline_depth: int,
        unresolved_window: int,
        max_batch_operations: int,
    ) -> WorkerCapabilities:
        layout = self.model.layout
        identity = architecture_identity(
            self.model.architecture,
            {
                "profile": "minimax_h3_t2va_v0.2",
                "height": 768,
                "width": 1344,
                "frames": 124,
                "evaluations": 4,
            },
        )
        sm_count = int(torch.cuda.get_device_properties(self.device).multi_processor_count)
        lane = LaneCapabilities(
            lane_id="h3",
            domains=(self.model_domain,),
            resolved_sm_count=max(1, sm_count),
            kv_capacity_tokens=None,
            latent_capacity_units=layout.persistent_units * self.model.states.slot_count,
            max_batch_operations=max_batch_operations,
            max_batch_tokens=max_batch_operations,
            max_inflight=pipeline_depth,
            graph_buckets=(),
            eager_max_batch_operations=max_batch_operations,
            eager_max_batch_tokens=max_batch_operations,
        )
        return WorkerCapabilities(
            block_size=0,
            num_blocks=0,
            num_layers=0,
            num_kv_heads=0,
            head_dim=0,
            supported_work=_MEDIA_WORK,
            latent_page_units=layout.persistent_units,
            num_latent_pages=self.model.states.slot_count + 1,
            latent_width=1,
            latent_dtype="float32",
            latent_downsample=1,
            max_vae_grid_tokens=int(layout.packed.video_indices.numel()),
            max_vit_grid_tokens=0,
            max_latent_feature_bytes=0,
            max_vision_feature_bytes=0,
            commit_marker_tokens=0,
            gen_rope_advance=1,
            max_cfg_branches=1,
            bytes_per_token=0,
            groups=(),
            kv_dtype="",
            model_dtype="bfloat16",
            attention_backend="h3_vsa_sm100",
            rank=RankInfo(
                tp_rank=self.mesh.coord("sp"),
                tp_size=self.mesh.size("sp"),
            ),
            pipeline_depth=pipeline_depth,
            encoder_cache_budget=0,
            supported_controls=(RequestKind.DROP_SESSION, RequestKind.RELEASE_PRODUCTS),
            max_batch_operations=max_batch_operations,
            max_batch_tokens=max_batch_operations,
            max_request_pool_size=self.model.states.slot_count,
            max_unresolved_window=unresolved_window,
            incremental_kv_publication=False,
            mixed_buckets=(),
            sampling_ownership=SamplingOwnership.DESIGNATED_RANK,
            resource_classes=(ResourceClass.IMAGE_LATENT,),
            model_identity=identity,
            weight_digest=self.model.checkpoint_digest,
            lanes=(lane,),
        )

    @property
    def model_domain(self):
        from ..batch import Domain

        return Domain.FLOW

    def prepare_execute(self, batch: Batch) -> None:
        return None

    def _slot(self, operation: Operation) -> H3StateSlot:
        for slot in self.model.states.slots:
            if slot.request_key == operation.request_key:
                return slot
        raise invalid_descriptor("H3 operation references a request without resident state")

    @staticmethod
    def _parent_semantic(operation: Operation, slot: H3StateSlot) -> object:
        point = operation.parent.point
        if isinstance(point, FixedPoint):
            if int(operation.parent.producer_op_id) != int(slot.producer_op_id):
                raise invalid_descriptor("H3 fixed parent producer does not match resident state")
            if str(point.semantic_digest) != str(slot.semantic_digest):
                raise invalid_descriptor("H3 fixed parent digest does not match resident state")
            return slot.semantic_digest
        if not isinstance(point, DevicePoint):
            raise invalid_descriptor("H3 operation carries an unknown parent point")
        if (
            int(operation.parent.producer_op_id) != int(slot.producer_op_id)
            or point.producer_plan_digest != slot.producer_plan_digest
        ):
            raise invalid_descriptor("H3 device parent does not match resident state")
        return slot.semantic_digest

    @staticmethod
    def _placement(partition: BatchPartition, operation: Operation):
        selected = [
            placement
            for placement in partition.latent_placements
            if placement.request_key == operation.request_key and placement.op_id == operation.op_id
        ]
        if len(selected) != 1:
            raise invalid_descriptor("H3 trajectory operation has no exact latent placement")
        return selected[0]

    @staticmethod
    def _decode_placement(partition: BatchPartition, operation: Operation) -> DecodePlacement:
        selected = [
            placement
            for placement in partition.decode_placements
            if placement.request_key == operation.request_key and placement.op_id == operation.op_id
        ]
        if len(selected) != 1:
            raise invalid_descriptor("H3 decode operation has no exact decode placement")
        return selected[0]

    def _capture_capacity(self, partition: BatchPartition) -> int:
        return len(partition.operations)

    def _reserve_tasks(
        self, partition: BatchPartition
    ) -> dict[tuple[object, int], tuple[CpuTaskReservation, H3OutputRingLease | None]]:
        if self.mesh.coord("sp") != 0:
            return {}
        output_ring = self.output_ring
        if output_ring is None:
            raise RuntimeError("rank zero has no H3 output ring")
        reservations: dict[
            tuple[object, int], tuple[CpuTaskReservation, H3OutputRingLease | None]
        ] = {}
        try:
            for operation in partition.operations:
                if operation.work in {
                    ForwardMode.GEN_DECODE,
                    ForwardMode.MATERIALIZE,
                }:
                    lease = None
                    if operation.work is ForwardMode.GEN_DECODE:
                        placement = self._decode_placement(partition, operation)
                        lease = output_ring.reserve(placement.kind.value)
                    try:
                        cpu = self.cpu_tasks.reserve()
                    except BaseException:
                        if lease is not None:
                            lease.release()
                        raise
                    reservations[(operation.request_key, operation.op_id)] = (
                        cpu,
                        lease,
                    )
        except BaseException:
            for reservation, lease in reservations.values():
                reservation.abandon()
                if lease is not None:
                    lease.release()
            raise
        return reservations

    def _consume_predicate(self, operation: Operation) -> DeviceProductRead | None:
        predicate = operation.predicate
        if predicate is None:
            return None
        if predicate.kind is not ProductKind.COMPLETION:
            raise invalid_descriptor("H3 successor predicate must be a completion product")
        producer_digest = (
            operation.parent.point.producer_plan_digest
            if isinstance(operation.parent.point, DevicePoint)
            else None
        )
        return self.device_products.consume(
            predicate,
            consumer_op_id=operation.op_id,
            producer_plan_digest=producer_digest,
            device=self.device,
        )

    def _bind_outputs(self, partition: BatchPartition) -> tuple[DeviceProductWrite, ...]:
        bindings = []
        for operation in partition.operations:
            if operation.work is ForwardMode.MATERIALIZE:
                if operation.outputs:
                    raise invalid_descriptor("H3 materialize must not publish device products")
                continue
            if (
                len(operation.outputs) != 1
                or operation.outputs[0].kind is not ProductKind.COMPLETION
            ):
                raise invalid_descriptor("H3 quantum must publish one completion predicate")
            bindings.append((operation.outputs[0], operation.plan_digest, self.device))
        return self.device_products.bind_outputs(tuple(bindings))

    def _execute_operation(
        self,
        operation: Operation,
        partition: BatchPartition,
        slot: H3StateSlot,
        buffer: PinnedOutputBuffer,
        reservation: CpuTaskReservation | None,
        ring_lease: H3OutputRingLease | None,
    ) -> tuple[object, ...]:
        variant = operation.work
        if variant is ForwardMode.GEN_TRANSITION:
            placement = self._placement(partition, operation)
            if placement.start_step != 0 or placement.step_count != 0:
                raise invalid_descriptor("H3 transition placement must carry zero denoise steps")
            return ()
        if variant is ForwardMode.GEN_FLOW:
            placement = self._placement(partition, operation)
            self.model.denoise(slot, placement.start_step, placement.step_count)
            return ()
        if variant is ForwardMode.GEN_DECODE:
            placement = self._decode_placement(partition, operation)
            if placement.kind is DecodeKind.VIDEO:
                rgb = self.model.decode_video(slot, placement)
                if self.mesh.coord("sp") == 0:
                    if rgb is None or reservation is None or ring_lease is None:
                        raise RuntimeError("rank zero lost its H3 video capture resources")
                    with profile_range(
                        f"uniserve.h3.decode_copy request={_request_label(operation)} "
                        f"op={operation.op_id} kind=video unit={placement.start_unit} "
                        f"rank={self.mesh.coord('sp')}"
                    ):
                        capture = buffer.capture_bytes_into(rgb, ring_lease.storage)
                    try:
                        return (
                            self.mux.video(
                                operation.request_key,
                                placement.start_unit,
                                capture,
                                reservation,
                                ring_lease,
                                operation.op_id,
                            ),
                        )
                    except BaseException:
                        ring_lease.defer_until_capture_ready(capture)
                        raise
                return ()
            pcm = self.model.decode_audio(slot, placement)
            if self.mesh.coord("sp") == 0:
                if pcm is None or reservation is None or ring_lease is None:
                    raise RuntimeError("rank zero lost its H3 audio capture resources")
                with profile_range(
                    f"uniserve.h3.decode_copy request={_request_label(operation)} "
                    f"op={operation.op_id} kind=audio rank={self.mesh.coord('sp')}"
                ):
                    capture = buffer.capture_bytes_into(pcm.view(torch.uint8), ring_lease.storage)
                try:
                    return (
                        self.mux.audio(
                            operation.request_key,
                            capture,
                            reservation,
                            ring_lease,
                            operation.op_id,
                        ),
                    )
                except BaseException:
                    ring_lease.defer_until_capture_ready(capture)
                    raise
            return ()
        if variant is ForwardMode.MATERIALIZE:
            if self.mesh.coord("sp") == 0:
                if reservation is None:
                    raise RuntimeError("rank zero lost its H3 materialize reservation")
                return (self.mux.materialize(operation.request_key, reservation, operation.op_id),)
            return ()
        raise invalid_descriptor(f"unsupported H3 work variant {variant.value!r}")

    def _execute_partition(self, step_id: int, partition: BatchPartition) -> PartitionCompletion:
        started = time.perf_counter_ns()
        reservations = self._reserve_tasks(partition)
        buffer = PinnedOutputBuffer(
            len(partition.operations),
            token_capacity=self._capture_capacity(partition),
            devices=(self.device,),
            event_pool=self.device_events,
        )
        writes: tuple[DeviceProductWrite, ...] = ()
        reads: list[DeviceProductRead] = []
        records: list[ModelOutput] = []
        write_by_operation: dict[int, DeviceProductWrite] = {}
        try:
            writes = self._bind_outputs(partition)
            write_by_operation = {int(write.reference.producer_op_id): write for write in writes}
            buffer.begin_device(self.device)
            for row, operation in enumerate(partition.operations):
                reservation_key = (operation.request_key, operation.op_id)
                reservation, ring_lease = reservations.get(reservation_key, (None, None))
                with profile_range(
                    f"uniserve.h3.quantum step={step_id} "
                    f"partition={partition.partition_id} "
                    f"request={_request_label(operation)} op={operation.op_id} "
                    f"work={operation.work.value} rank={self.mesh.coord('sp')}"
                ):
                    read = self._consume_predicate(operation)
                    if read is not None:
                        reads.append(read)
                    slot = self._slot(operation)
                    parent = self._parent_semantic(operation, slot)
                    tasks = self._execute_operation(
                        operation,
                        partition,
                        slot,
                        buffer,
                        reservation,
                        ring_lease,
                    )
                    if ring_lease is not None and tasks:
                        reservations[reservation_key] = (reservation, None)
                    write = write_by_operation.get(int(operation.op_id))
                    if write is not None:
                        self.device_products.publish_scalar_write(write, True)
                pending = DeferredDigest(
                    parent,
                    operation.plan_digest,
                    buffer,
                    row,
                    lambda: (_ for _ in ()).throw(
                        RuntimeError("H3 completion was unexpectedly predicated")
                    ),
                    status=OpStatus.OK,
                    selected_point=1,
                    completion_tasks=cast(tuple, tasks),
                )
                record = pending.bind_record(
                    ModelOutput(
                        request_key=operation.request_key,
                        op_id=operation.op_id,
                        completion_slot_generation=buffer.generation,
                        status=OpStatus.OK,
                        selected_point=1,
                        logical_lengths=LogicalLengths(),
                        token_span=TokenSpan(),
                        committed_tokens=(),
                        finish_flags=FinishFlags(),
                        product_generations=tuple(
                            int(output.generation) for output in operation.outputs
                        ),
                        semantic_digest=pending,
                        error_code=None,
                        timing_counters=TimingCounters(),
                    )
                )
                slot.semantic_digest = pending
                slot.producer_op_id = int(operation.op_id)
                slot.producer_plan_digest = operation.plan_digest
                records.append(record)
            buffer.seal()
            self.device_products.commit_writes(writes)
            if reads:
                after = tuple(write_by_operation.get(int(read.consumer_op_id)) for read in reads)
                if all(write is not None for write in after):
                    self.device_products.record_readers(
                        tuple(reads),
                        after_writes=cast(tuple[DeviceProductWrite, ...], after),
                    )
                else:
                    self.device_products.record_readers(tuple(reads), device=self.device)
            return PartitionCompletion(
                partition_id=partition.partition_id,
                completions=tuple(records),
                registration=RegistrationAck(visible=True),
                worker_exec_us=(time.perf_counter_ns() - started) // 1000,
            )
        except BaseException:
            for reservation, lease in reservations.values():
                reservation.abandon()
                if lease is not None:
                    lease.release()
            if writes:
                self.device_products.abandon_writes(writes)
            buffer.abandon()
            raise

    def _apply_admissions(self, batch: Batch) -> None:
        admissions = {admission.request_key: admission for admission in batch.admissions}
        transitions = {
            operation.request_key: operation
            for operation in batch.operations
            if operation.work is ForwardMode.GEN_TRANSITION
        }
        if set(admissions) != set(transitions):
            raise invalid_descriptor("H3 admissions must exactly match transition operations")
        for request_key, admission in admissions.items():
            media = admission.media
            if media is None:
                raise invalid_descriptor("H3 admission is missing its media request")
            output_path = Path(media.output_path).expanduser()
            try:
                output_parent = output_path.parent.resolve(strict=True)
            except OSError as error:
                raise invalid_descriptor("H3 output directory is unavailable") from error
            if output_parent != self.media_spool or output_path.suffix != ".mp4":
                raise invalid_descriptor("H3 output must be an MP4 in the configured media spool")
            output_path = output_parent / output_path.name
            slot = self.model.states.get(admission.request_pool_idx)
            if slot.active:
                raise invalid_descriptor("H3 admission targets an occupied request slot")
            operation = transitions[request_key]
            point = operation.parent.point
            if (
                operation.parent.producer_op_id != 0
                or not isinstance(point, FixedPoint)
                or point.semantic_digest != admission.digest
            ):
                raise invalid_descriptor("H3 transition does not name its admission root")
            self.model.prepare(slot, admission)
            slot.output_path = output_path
            if self.mesh.coord("sp") == 0:
                if slot.output_path is None:
                    raise RuntimeError("H3 admission lost its output path")
                self.mux.open(request_key, slot.output_path)

    def _apply_controls(self, batch: Batch) -> None:
        releases = tuple(
            (control.request_key, control.op_id)
            for control in batch.controls
            if isinstance(control, Release)
        )
        self.device_products.release_operations(releases)
        for control in batch.controls:
            if isinstance(control, Commit):
                raise invalid_descriptor("H3 batches do not accept commit controls")
            if isinstance(control, Close):
                self.drop_session(control.request_key.session_id)

    def execute(self, batch: Batch) -> CompletionReport:
        self._apply_admissions(batch)
        reports = tuple(
            self._execute_partition(batch.step_id, partition) for partition in batch.partitions
        )
        self._apply_controls(batch)
        return CompletionReport(step_id=batch.step_id, partitions=reports)

    def drop_session(self, session_id: int) -> None:
        self.device_products.drop_session(int(session_id))
        self.model.states.drop_session(int(session_id))
        if self.mesh.coord("sp") == 0:
            self.mux.drop(int(session_id))

    def copy_kv(self, _copies: tuple[object, ...]) -> None:
        raise capability_mismatch("MiniMax H3 has no KV cache")

    def release_products(self, handles: tuple[int, ...]) -> None:
        self.device_products.release_generations(tuple(int(value) for value in handles))

    def snapshot_session(self, _placement: object) -> object:
        raise capability_mismatch("MiniMax H3 does not support session snapshots")

    def restore_session(self, _reference: object, _placement: object) -> None:
        raise capability_mismatch("MiniMax H3 does not support session snapshots")

    def resource_pressure(self) -> list[dict[str, object]]:
        used = sum(slot.active for slot in self.model.states.slots)
        video_ring, audio_ring = self.output_ring.used if self.output_ring is not None else (0, 0)
        return [
            {
                "resource": ResourceClass.IMAGE_LATENT.value,
                "used": used * self.model.layout.persistent_units,
                "capacity": self.model.states.slot_count * self.model.layout.persistent_units,
                "video_ring_used": video_ring,
                "audio_ring_used": audio_ring,
            }
        ]

    def warmup(self) -> None:
        try:
            self.media_spool = self.media_spool.resolve(strict=True)
        except OSError as error:
            raise RuntimeError(f"media spool {self.media_spool} is unavailable") from error
        if not self.media_spool.is_dir():
            raise RuntimeError(f"media spool {self.media_spool} is not a directory")
        try:
            descriptor, probe = tempfile.mkstemp(
                dir=self.media_spool,
                prefix=f".uniserve-worker-{self.mesh.coord('sp')}-",
            )
            os.close(descriptor)
            Path(probe).unlink()
        except OSError as error:
            raise RuntimeError(f"media spool {self.media_spool} is not writable") from error
        if self.mesh.coord("sp") == 0:
            require_h3_codecs()
        self.model.warmup()

    def set_completion_wake(
        self,
        wake: Callable[[], None],
        wake_on_stream: Callable[[int], None],
    ) -> None:
        self.device_events.set_completion_wake(wake_on_stream)
        self.cpu_tasks.set_completion_wake(wake)

    def close(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.cpu_tasks.close()
        self.mux.close()
        self.device_products.close()
        self.device_events.close()


def _request_label(operation: Operation) -> str:
    key = operation.request_key
    return f"{key.authority_id}:{key.session_id}:{key.epoch}"
