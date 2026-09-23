"""Worker-owned execution of declared numerical capabilities."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict, defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch
from torch import nn

from uniserve.distributed import Communicator, DeviceMesh
from uniserve.model import (
    AudioDecoder,
    CausalLM,
    Denoiser,
    ImageDenoiser,
    PatchEncoder,
    TextEncoder,
    TextSize,
    VideoDecoder,
    VideoPostprocessor,
)
from uniserve.nn.vae import PatchAutoencoder
from uniserve.processing import ImageProcessor
from uniserve.profiling import profile_range
from uniserve.runtime import (
    CUDAStream,
    ExecutionContext,
    partition_streams,
)
from uniserve.runtime.backends.attention import resolve as attention_backend
from uniserve.runtime.backends.attention.flashinfer import Backend as FlashInfer
from uniserve.runtime.cuda_graph import CUDAGraphError
from uniserve.runtime.device import canonical_device
from uniserve.runtime.resources import close_resources
from uniserve_worker.bootstrap.components import (
    VIDEO_ENCODER_COMPONENT,
    bind_components,
    call_kinds,
    describe_components,
)
from uniserve_worker.bootstrap.inputs import (
    capability,
    image_builder,
    media_builder,
)
from uniserve_worker.bootstrap.outputs import resolve_outputs
from uniserve_worker.config.deployment import ComponentConfig
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.errors import (
    ComputeError,
    InputError,
    ResourceError,
    WorkerError,
    WorkerErrorCode,
    classify,
    invalid_descriptor,
)
from uniserve_worker.model_executor.component_binding import (
    ComponentBinding,
)
from uniserve_worker.model_executor.cuda_graph import (
    input_signature,
)
from uniserve_worker.model_executor.diffusion_inputs import DiffusionRow
from uniserve_worker.model_executor.diffusion_runner import TrajectoryRunner
from uniserve_worker.model_executor.graph_inputs import (
    PrefillShape,
    select_flow_captures,
    select_prefill_captures,
)
from uniserve_worker.model_executor.graph_storage import GraphStorage
from uniserve_worker.model_executor.image_inputs import DecodeRow, VisionRow
from uniserve_worker.model_executor.input_batch import (
    InputRow,
    TokenRow,
)
from uniserve_worker.model_executor.input_buffers import (
    DiffusionBuffers,
    TokenBuffers,
    buffered_kinds,
    input_buffer_config,
)
from uniserve_worker.model_executor.model_runner import ModelRunner, runner_type
from uniserve_worker.model_executor.output import ExecutionOutput
from uniserve_worker.model_executor.resources import (
    media_state_buffers,
    output_layouts,
)
from uniserve_worker.profiling import record_component
from uniserve_worker.protocol.call import (
    Call,
    ForwardMode,
    MediaCall,
)
from uniserve_worker.protocol.identity import CallId
from uniserve_worker.protocol.output import ForwardStats
from uniserve_worker.sampling.metadata import TokenSelection

if TYPE_CHECKING:
    from uniserve_worker.storage.block_tables import BlockTables
    from uniserve_worker.storage.decode_state import DecodeState
    from uniserve_worker.storage.kv_cache import KVCacheManager

logger = logging.getLogger(__name__)


def _observations(name, started, path):
    """Build single-call forward statistics for one module, in microseconds."""
    elapsed = (time.perf_counter_ns() - started) // 1000
    return ForwardStats(
        mode_counts={name: 1},
        mode_tokens={name: 1},
        mode_us={name: elapsed},
        component_us={"forward": elapsed},
        cuda_graph_runtime_mode_counts={path: 1},
        cuda_graph_captures=int(path == "graph_capture"),
        cuda_graph_replays=int(path == "graph_replay"),
    )


class ModelExecutor:
    """Dispatch capability runners and own their execution lane streams.

    Each numerical call uses its declared capability. Shared Parameters remain
    in the loaded module tree; independent call/lane contexts own mutable plans,
    workspaces, communication resources and captured input/output storage.
    """

    def __init__(
        self,
        model: nn.Module,
        worker_config: WorkerConfig,
        *,
        bindings=None,
        entry_points=None,
        attention=None,
        image_processor=None,
        flow_prompt=None,
        max_inflight=1,
    ):
        self.model, self.worker_config = model, worker_config
        self._event_slots = max_inflight + 1
        self.attention = attention_backend(
            attention or worker_config.attention_backend or "auto",
            device=canonical_device(worker_config.device),
            flashinfer=FlashInfer(worker_config.flashinfer),
        )
        self.processor, self.flow_prompt = image_processor, flow_prompt
        self.image_builder = image_builder(model)
        self.media_builder = media_builder(model, worker_config)
        self.text = capability(model, CausalLM)

        # Capability discovery belongs to initialization. Request execution uses
        # these borrowed modules without walking the loaded parameter tree.
        self.video_decoder = capability(model, VideoDecoder)
        self.audio_decoder = capability(model, AudioDecoder)
        self.video_postprocessor = capability(model, VideoPostprocessor)
        self.outputs = resolve_outputs(model, worker_config)
        self._declarations = describe_components(model, entries=entry_points)

        # Without supplied bindings, every declared component runs standalone
        # on this rank; only video decoders distribute over temporal units.
        if bindings is None:
            device, rank = (
                canonical_device(worker_config.device),
                worker_config.rank,
            )
            group = Communicator((rank,), 0, device=device)
            bindings = {}
            for name, calls in self._declarations.items():
                temporal = any(
                    isinstance(call.module, VideoDecoder) for call in calls
                )
                config = ComponentConfig(
                    (rank,), distribution="temporal_units" if temporal else None
                )
                mesh = DeviceMesh(
                    ranks=(rank,),
                    rank=rank,
                    shape=tuple(
                        size for _, size in config.parallel_config.dimensions
                    ),
                    axes=tuple(
                        axis for axis, _ in config.parallel_config.dimensions
                    ),
                )
                bindings[name] = ComponentBinding(
                    name, config, group, mesh, device
                )
        self.bindings = MappingProxyType(dict(bindings))
        bind_components(model, self.bindings, declarations=self._declarations)
        # A host rank holds components with no numerical method: it warms
        # nothing up and captures nothing, and its model is a description.
        self.numerical = any(
            binding.calls for binding in self.bindings.values()
        )
        self.state_buffers = media_state_buffers(
            self.bindings, self.media_builder
        )

        self.entries = {}
        self._forward_calls = {}
        self._module_calls = {}
        self._call_kinds = {}
        self._runner_types = {}
        self._calls_by_kind = defaultdict(list)
        self._module_entries: OrderedDict[tuple, ModelRunner] = OrderedDict()
        self._module_streams = {}

        self._lane_streams = []
        self._preparation_stream = None

        self.graph_storage = GraphStorage()
        self.decode_shapes, self.prefill_shapes = {}, {}
        self.prefill_row_sizes = ()
        self.flow_captures, self.flow_cfg_branches = (), ()
        self.decode_context_blocks = 0
        self.decode_predicates = None
        self.kv_cache = None
        self.denoising = None

        self.uses_lanes = False
        self._startup_complete = self._closed = False

        try:
            for name, binding in self.bindings.items():
                for call in binding.calls:
                    if not call_kinds((call,)):
                        continue
                    key = (name, call.path, call.entry_point.method)
                    self._module_calls[key] = (binding, call)
                    self._call_kinds[id(call)] = call_kinds((call,))
                    self._runner_types[id(call)] = runner_type(call.module)
                    for kind in call_kinds((call,)):
                        self._calls_by_kind[name, kind].append(call)

                    if self.media_builder is not None and isinstance(
                        call.module, Denoiser
                    ):
                        self.denoising = TrajectoryRunner(
                            call.module,
                            device=binding.device,
                            stream=self.module_stream(
                                name, method=call.entry_point.method
                            ),
                            capture=worker_config.graph_policy != "off",
                            graph_storage=self.graph_storage,
                            groups=call.groups,
                            capacity=worker_config.max_request_pool_size,
                            # Resident requests may carry different numerical
                            # sizes, and every declared video shape holds the
                            # context its captured ladders borrow.
                            shapes=max(
                                worker_config.max_request_pool_size,
                                len(worker_config.video_graph_shapes),
                            ),
                            attention=self.attention,
                            additional_devices=self._capture_devices(
                                binding.device
                            ),
                        )
            # A rank that denoises draws each request's seeded CPU noise on
            # its own thread, off the service thread that launches device
            # work, as soon as the request is admitted.
            self.noise_draws = (
                ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="worker-noise"
                )
                if self.denoising is not None
                else None
            )
        except BaseException as error:
            try:
                self.close(aborted=True)
            except BaseException as cleanup:
                error.add_note(
                    f"execution resource cleanup failed: {cleanup!r}"
                )
            raise

    def _capture_devices(self, device):
        """List CUDA devices besides ``device`` that graphs may allocate on."""
        return tuple(
            value
            for value in dict.fromkeys(
                canonical_device(value)
                for value in (
                    self.worker_config.device,
                    self.worker_config.generation_device,
                )
                if value is not None
            )
            if value.type == "cuda" and value != device
        )

    def component(self, kind, *, capability_type=None):
        """Return the single module that provides a computation kind on this.

        rank.
        """
        calls = [
            call
            for _, call in self._module_calls.values()
            if kind in self._call_kinds[id(call)]
            and (
                capability_type is None
                or isinstance(call.module, capability_type)
            )
        ]
        if len(calls) != 1:
            raise InputError(
                f"computation {kind.value} requires one local capability"
            )
        return calls[0].module

    def _module_call(self, name, method=None):
        """Return the uniquely identified ``(binding.

        call)`` pair for a component.
        """
        if self._closed:
            raise RuntimeError("model runner is closed")
        entries = [
            (binding, call)
            for (entry, _, entry_method), (
                binding,
                call,
            ) in self._module_calls.items()
            if entry == name and (method is None or entry_method == method)
        ]
        if len(entries) != 1:
            raise InputError(
                f"component {name!r} requires an unambiguous numerical method"
            )
        return entries[0]

    def prepare_module(self, name, size, *, method=None):
        """Prepare one exact numerical size before dependent media calls.

        Prepared contexts are kept in an LRU per (entry, path, method); when
        the residency bound is reached, the oldest context and the graphs that
        borrow it are retired.
        """
        binding, call = self._module_call(name, method)
        key = (name, call.path, call.entry_point.method, input_signature(size))
        if key not in self._module_entries:
            resident = tuple(
                value for value in self._module_entries if value[:3] == key[:3]
            )
            if len(resident) >= self.worker_config.max_request_pool_size:
                self._retire_module(resident[0])

            stream = self.module_stream(name, method=call.entry_point.method)
            context = ExecutionContext(
                call.module,
                attention=self.attention,
                stream=stream,
                # A method declared for one pipeline stage runs on that stage
                # alone, so its preparation opens no binding over a group the
                # other stages never reach.
                groups=call.groups,
            )
            entry = self._runner_types[id(call)](
                name,
                call,
                binding.device,
                (),
                None,
                context,
                storage=self.graph_storage,
                devices=(binding.device,)
                if stream is not None
                and self.worker_config.graph_policy != "off"
                and not isinstance(call.module, VideoPostprocessor)
                else (),
            )
            try:
                if stream is not None:
                    stream.wait(torch.cuda.current_stream(binding.device))
                with self.graph_storage.allocate(entry):
                    context.prepare(size)
                self.graph_storage.check()
            except BaseException:
                entry.close()
                raise

            self._module_entries[key] = entry

        self._module_entries.move_to_end(key)
        return self._module_entries[key]

    def module_stream(self, name, *, method=None):
        """Bind one numerical entry to a stream in its existing resource.

        grant.
        """
        binding, call = self._module_call(name, method)
        if binding.device.type != "cuda":
            return None

        self._initialize_streams(event_slots=self._event_slots)
        key = (name, call.path, call.entry_point.method)
        if key not in self._module_streams:
            # A standalone entry forks the lane stream that covers its call
            # kinds, or creates a full-device stream when lanes are off.
            entry_kinds = self._call_kinds[id(call)]
            parents = tuple(
                stream
                for lane, stream in self._lane_streams
                if stream.device == binding.device
                and (lane is None or entry_kinds.intersection(lane.call_kinds))
            )
            if len(parents) > 1:
                raise InputError(
                    "one numerical entry requires an unambiguous execution "
                    "partition"
                )
            if parents:
                owner = parents[0].fork()
            elif self.worker_config.lanes:
                raise InputError(
                    "numerical entry has no initialized execution partition"
                )
            else:
                owner = CUDAStream(
                    device=binding.device,
                    stream=torch.cuda.Stream(device=binding.device),
                    sm_count=torch.cuda.get_device_properties(
                        binding.device
                    ).multi_processor_count,
                )
            # Weights and runtime backing are initialized before their first
            # borrowed use. Later calls depend on tensor fences, not other
            # entries.
            try:
                owner.wait(torch.cuda.current_stream(binding.device))
            except BaseException:
                owner.close()
                raise
            self._module_streams[key] = owner
        return self._module_streams[key]

    def call_stream(self, call):
        """Select a standalone capability's stream.

        staged batches bind their own lanes.
        """
        if (call.component, call.kind) in self._forward_calls:
            return None
        if (
            call.kind is MediaCall.LATENT_PREPARATION
            and (call.component, MediaCall.DENOISING) in self._forward_calls
        ):
            return None
        calls = self._calls_by_kind.get((call.component, call.kind), ())
        if not calls:
            return None

        if call.kind is MediaCall.LATENT_PREPARATION:
            # Preparation composes the denoiser's initialization with optional
            # conditioning encoders. The denoiser owns this call's stream;
            # its encoder calls retain their ordinary input/output dependencies.
            denoisers = tuple(
                candidate
                for candidate in calls
                if isinstance(candidate.module, Denoiser)
            )
            calls = denoisers or calls
        elif call.kind is MediaCall.VIDEO_DECODING:
            calls = tuple(
                candidate
                for candidate in calls
                if isinstance(candidate.module, VideoDecoder)
            )

        if len(calls) != 1:
            raise InputError("call requires one bound numerical capability")
        owner = self.module_stream(
            call.component, method=calls[0].entry_point.method
        )
        return None if owner is None else owner.stream

    def _initialize_streams(self, *, event_slots):
        """Realize execution grants for both staged and standalone.

        capabilities.
        """
        if self._lane_streams:
            return

        config, device = (
            self.worker_config,
            canonical_device(self.worker_config.device),
        )
        self.uses_lanes = bool(config.lanes)
        if config.lanes:
            # Lanes partition one device's SMs, so multi-device capture is
            # incompatible with a lane configuration.
            if self._capture_devices(device):
                raise ValueError(
                    "Green Context lanes require one physical device"
                )
            slots = tuple(
                int(lane.max_inflight or (event_slots - 1)) + 1
                for lane in config.lanes
            )
            streams = partition_streams(
                device,
                tuple(lane.sm_budget for lane in config.lanes),
                event_slots=slots,
            )
            self._lane_streams.extend(zip(config.lanes, streams, strict=True))
        else:
            for target in (device, *self._capture_devices(device)):
                if target.type == "cuda":
                    self._lane_streams.append(
                        (
                            None,
                            CUDAStream(
                                device=target,
                                stream=torch.cuda.Stream(device=target),
                                sm_count=torch.cuda.get_device_properties(
                                    target
                                ).multi_processor_count,
                                event_slots=event_slots,
                            ),
                        )
                    )

    def _retire_module(self, key):
        """Drain a producer before releasing the graphs that borrow its.

        resources.
        """
        entry = self._module_entries.pop(key)
        if entry.context.stream is not None:
            entry.context.stream.synchronize()
        entry.close()

    @property
    def encoder_kinds(self):
        return frozenset(
            self._encoder_kind(call.module)
            for _, call in self._module_calls.values()
            if call.entry_point.method == "encode"
        )

    @staticmethod
    def _encoder_kind(module):
        if isinstance(module, TextEncoder):
            return "text"
        if isinstance(module, PatchEncoder):
            return "vision"
        if isinstance(module, PatchAutoencoder):
            return "latent"
        return "conditioning"

    def run_encoder(self, kind, *values):
        """Run the rank's encoder of the given kind ("text", "vision", "latent".

        "conditioning").
        """
        found = [
            (name, call)
            for (name, _, _), (_, call) in self._module_calls.items()
            if call.entry_point.method == "encode"
            and self._encoder_kind(call.module) == kind
        ]
        if len(found) != 1:
            raise InputError(f"rank does not participate in {kind} encoding")
        name, call = found[0]

        # Text encoders take flat token sequences; other encoders keep their
        # batched input shapes, which also serve as the preparation size.
        inputs = (
            tuple(value.reshape(-1) for value in values)
            if kind == "text"
            else values
        )
        size = (
            TextSize(sum(value.numel() for value in inputs), len(inputs))
            if kind == "text"
            else tuple(value.shape for value in inputs)
        )
        return self.run_module(name, inputs, method="encode", size=size)

    @torch.inference_mode()
    def run_module(self, name, *args, method=None, size=None, **kwargs):
        """Resolve a bound capability and invoke its numerical runner."""
        binding, call = self._module_call(name, method)
        runner = self.prepare_module(name, size, method=call.entry_point.method)
        with profile_range(
            f"uniserve.model.module rank={self.worker_config.rank} work={name}"
        ):
            try:
                return runner.execute_model(*args, **kwargs)
            except CUDAGraphError:
                self._retire_module(
                    (
                        name,
                        call.path,
                        call.entry_point.method,
                        input_signature(size),
                    )
                )
                raise

    def run_denoising(self, trajectory, index):
        """Run one denoising step for a resident trajectory and time it."""
        if self.denoising is None:
            raise InputError("rank does not own denoising computation")
        started = time.perf_counter_ns()
        with profile_range(
            f"uniserve.model.denoise rank={self.worker_config.rank} "
            "work=denoiser"
        ):
            values, path = self.denoising.step(trajectory, index)
        return replace(
            ModelRunner.result(values),
            stats=_observations("denoiser", started, path),
        )

    def output_layout(
        self, entry, output_index, media, decode, num_prompt_tokens
    ):
        """Resolve the published layout of one call output on this rank.

        Returns None for outputs this rank does not publish: non-output ranks,
        degenerate denoiser slices, and empty temporal-unit shares of a
        distributed decoder.
        """
        binding = self.bindings.get(entry)
        if (
            binding is not None
            and binding.process_group.global_rank not in binding.output_ranks
        ):
            return None

        frames = None if media is None else media.num_frames
        results = tuple(
            (call.module, name, layout)
            for call in self._declarations[entry]
            for name, layout in output_layouts(
                self.worker_config,
                call,
                builder=self.media_builder,
                clock=self.video_postprocessor,
                frames=frames,
                prompt_tokens=num_prompt_tokens,
            ).items()
        )
        if entry == VIDEO_ENCODER_COMPONENT:
            # The video encoder owns no numerical method; its product is the
            # encoded rows of the media units it is handed.
            from uniserve_worker.model_executor.resources import (
                encoded_units_layout,
            )

            count = (
                self.media_builder.maximum.num_frames
                if frames is None
                else frames
            )
            results = (
                (
                    None,
                    "encoded_units",
                    encoded_units_layout(self.video_decoder, count),
                ),
            )
        module, name, layout = results[output_index]
        if isinstance(module, Denoiser) and any(
            axis.stop == axis.start for axis in layout.local_slice
        ):
            return None
        if binding is None or binding.config.distribution is None:
            return layout

        # Protocol decoder outputs contain the units in this scheduled call;
        # numerical decoder layouts describe their complete temporal timeline.
        # A video layout leads with its unit axis, so the total is read from
        # it; an audio layout leads with samples, so the placement states how
        # many media units its ranks reconstruct together.
        audio = isinstance(module, AudioDecoder)
        total = (
            len(binding.config.ranks) * max(1, binding.config.units_per_rank)
            if audio
            else layout.shape[0]
        )
        units = total if decode is None else decode.max_units
        position = binding.config.ranks.index(binding.process_group.global_rank)
        start = position * binding.config.units_per_rank
        count = min(binding.config.units_per_rank, units - start)
        if count < 1:
            return None

        if audio:
            # An audio media unit is a span of the sample timeline, so this
            # rank publishes the samples of the units it reconstructs.
            spans = module.unit_samples(layout.shape[0], units)
            return replace(
                layout,
                local_slice=(
                    slice(spans[start].start, spans[start + count - 1].stop),
                    *layout.local_slice[1:],
                ),
            )

        return replace(
            layout,
            shape=(units, *layout.shape[1:]),
            local_slice=(slice(start, start + count), *layout.local_slice[1:]),
        )

    def configure_inputs(
        self,
        *,
        input_config,
        kv_cache,
        latent_pool,
        decode_predicates,
        max_calls,
        request_slots,
        max_tokens,
        latent_capacity_units,
        decode_context_blocks,
        max_inflight,
    ):
        """Bind staged input resources and graph budgets for every capability.

        Creates one capability runner per (entry, path, lane) covering a
        computation kind, with its input buffers, execution context, decode /
        prefill capture shapes, and private CUDA graph storage pools. Callable
        exactly once.
        """
        from uniserve_worker.config.execution import (
            DEFAULT_PREFILL_GRAPH_ROW_BUCKETS,
        )

        if self.entries:
            raise RuntimeError("input execution resources are already bound")

        self.kv_cache, self.decode_predicates = kv_cache, decode_predicates
        self.decode_context_blocks = decode_context_blocks
        self.prefill_row_sizes = DEFAULT_PREFILL_GRAPH_ROW_BUCKETS
        config = self.worker_config
        self._initialize_streams(event_slots=max_inflight + 1)

        max_rows = min(max_calls, request_slots)
        decode_sizes = tuple(
            value
            for value in config.decode_graph_batch_sizes
            if 0 < value <= max_rows and value < kv_cache.info.num_blocks
        )
        prefill_capacity = min(
            max_tokens,
            config.max_sequence_tokens,
            (kv_cache.info.num_blocks - 1) * config.block_size,
        )
        prefill_sizes = tuple(
            value
            for value in config.prefill_graph_token_sizes
            if 0 < value <= prefill_capacity
        )
        if self.image_builder is not None:
            from uniserve.media import image

            self.flow_cfg_branches = (1, 2, 3)
            self.flow_captures = select_flow_captures(
                config.flow_graph_shapes,
                config.flow_graph_batch_sizes,
                self.flow_cfg_branches,
                max_calls=max_rows,
                max_tokens=input_config.max_tokens,
                per_image_capacity=latent_capacity_units,
                latent_capacity=latent_pool.capacity_units,
                physical_tokens=lambda height, width: (
                    self.image_builder.sequence_length(
                        image.Config(height, width)
                    )
                ),
                image_tokens=lambda height, width: (
                    self.image_builder.denoiser.latent_shape(
                        "image", image.Config(height, width)
                    )[0]
                ),
            )

        staged = buffered_kinds(diffusion=self.image_builder is not None)

        for (name, path, method), (
            placement,
            call,
        ) in self._module_calls.items():
            entry_kinds = self._call_kinds[id(call)] & staged
            if not entry_kinds:
                continue

            target = (
                canonical_device(config.generation_device or config.device)
                if entry_kinds
                & {MediaCall.LATENT_ENCODING, MediaCall.IMAGE_DECODING}
                else placement.device
            )
            streams = [
                (lane, stream)
                for lane, stream in self._lane_streams
                if stream.device == target
            ]
            for lane, stream in streams or ((None, None),):
                kinds = (
                    entry_kinds
                    if lane is None
                    else entry_kinds.intersection(lane.call_kinds)
                )
                if not kinds:
                    continue

                rows = (
                    max_rows
                    if lane is None
                    else min(max_rows, lane.max_batch_calls or max_rows)
                )
                tokens = (
                    input_config.max_tokens
                    if lane is None
                    else min(
                        input_config.max_tokens,
                        lane.max_batch_tokens or input_config.max_tokens,
                    )
                )
                decode = (
                    tuple(value for value in decode_sizes if value <= rows)
                    if ForwardMode.DECODE in kinds
                    else ()
                )
                prefill = (
                    select_prefill_captures(
                        prefill_sizes,
                        self.prefill_row_sizes,
                        max_rows=rows,
                        max_tokens=tokens,
                        visual=self.image_builder is not None,
                    )
                    if ForwardMode.PREFILL in kinds
                    else ()
                )
                # A prefill bucket includes a padding sequence beyond admitted
                # requests. It consumes staging, but no scheduler request slot.
                fields = (
                    replace(
                        input_config,
                        max_rows=max(
                            input_config.max_rows,
                            *(shape.row_bucket for shape in prefill),
                        ),
                    )
                    if prefill
                    else input_config
                )
                inputs = context = entry = None
                try:
                    if stream is not None:
                        stream.wait(torch.cuda.current_stream(target))
                    with (
                        nullcontext()
                        if stream is None
                        else torch.cuda.stream(stream.stream)
                    ):
                        buffer_type, buffer_config = input_buffer_config(
                            next(iter(kinds)), fields
                        )
                        inputs = buffer_type(
                            config=buffer_config,
                            device=target,
                            max_inflight=max_inflight,
                            **(
                                {"image_builder": self.image_builder}
                                if buffer_type
                                in (TokenBuffers, DiffusionBuffers)
                                else {}
                            ),
                        )
                        context = ExecutionContext(
                            call.module,
                            cache=kv_cache.cache
                            if isinstance(
                                call.module, (CausalLM, ImageDenoiser)
                            )
                            else None,
                            attention=self.attention,
                            stream=stream,
                            groups=call.groups,
                        )
                        # Text staging counts canonical tokens. Spatial codecs
                        # and vision towers expand those into different query
                        # domains, so their layer plans use the actual numerical
                        # shapes encountered during preparation/eager execution.
                        size = (
                            TextSize(fields.max_tokens, fields.max_rows)
                            if isinstance(
                                call.module, (CausalLM, ImageDenoiser)
                            )
                            else None
                        )
                        entry = self._runner_types[id(call)](
                            name,
                            call,
                            target,
                            kinds,
                            stream,
                            context,
                            inputs,
                            storage=self.graph_storage,
                            prefill_graph=config.prefill_cuda_graph,
                            cache=kv_cache.cache,
                            predicates=decode_predicates,
                            rank=config.rank,
                            devices=(target, *self._capture_devices(target))
                            if config.graph_policy != "off"
                            and target.type == "cuda"
                            else (),
                        )
                        with self.graph_storage.allocate(entry):
                            context.prepare(size)
                        self.graph_storage.check()
                except BaseException as error:
                    try:
                        close_resources(
                            *(
                                owner.close
                                for owner in (
                                    (entry,)
                                    if entry is not None
                                    else (context, inputs)
                                )
                                if owner is not None
                            )
                        )
                    except BaseException as cleanup:
                        error.add_note(
                            f"input resource cleanup failed: {cleanup!r}"
                        )
                    raise

                self.entries[
                    (name, path, None if lane is None else lane.lane_id)
                ] = entry
                self.decode_shapes[entry], self.prefill_shapes[entry] = (
                    decode,
                    prefill,
                )

                entry.decode_shapes, entry.prefill_shapes = decode, prefill
                entry.decode_context_blocks = decode_context_blocks

                for kind in kinds:
                    key = (name, kind)
                    if key in self._forward_calls:
                        raise ValueError(
                            f"computation {key} has multiple lane bindings"
                        )
                    self._forward_calls[key] = entry

    def call_devices(self, call):
        """Return ``(compute, staged, output)`` devices for one call.

        Media-generation kinds run on the generation device when one is
        configured. The staged device is the bound execution entry's device,
        or the compute device when the call has no staged binding.
        """
        binding = self.bindings.get(call.component)
        source = (
            canonical_device(self.worker_config.device)
            if binding is None
            else binding.device
        )
        if self.image_builder is not None and call.kind in {
            MediaCall.LATENT_PREPARATION,
            MediaCall.DENOISING,
            MediaCall.IMAGE_DECODING,
        }:
            source = canonical_device(
                self.worker_config.generation_device
                or self.worker_config.device
            )

        entry = self._forward_calls.get((call.component, call.kind))
        return source, source if entry is None else entry.device, source

    def warmup(self, storage):
        """Run media warmup passes, then drain every owned stream."""
        if self.media_builder is not None:
            from uniserve_worker.execution.media import (
                capture_denoising,
                warmup_conditioning,
                warmup_decoders,
                warmup_denoising,
                warmup_postprocess,
            )

            warmup_denoising(self, storage)
            warmup_conditioning(self)
            warmup_decoders(self)
            warmup_postprocess(self, storage)
            # Capture last so the declared shapes hold the prepared contexts
            # the captured ladders borrow.
            capture_denoising(self, storage)
        self.synchronize()

    @torch.inference_mode()
    def capture(self, *, tokenizer, latents):
        """Capture every entry's configured prefill, decode, and flow graphs."""
        from uniserve_worker.model_executor.startup import (
            prepare_decode,
            prepare_prefill,
        )

        for phase in ("prefill", "decode", "flow"):
            for entry in self.entries.values():
                forward = entry.batch_forward
                if (
                    phase == "prefill"
                    and ForwardMode.PREFILL in entry.call_kinds
                ):
                    shapes = (
                        self.prefill_shapes[entry]
                        if self.worker_config.graph_policy != "off"
                        and self.worker_config.prefill_cuda_graph
                        else (PrefillShape(1, 1, 1),)
                    )
                    prepare_prefill(
                        self, entry, entry.input_buffers, forward, shapes
                    )
                elif (
                    phase == "decode" and ForwardMode.DECODE in entry.call_kinds
                ):
                    prepare_decode(self, entry, entry.input_buffers, forward)
                elif (
                    phase == "flow" and MediaCall.DENOISING in entry.call_kinds
                ):
                    from uniserve_worker.model_executor.startup import (
                        prepare_flow,
                    )

                    prepare_flow(self, entry, latents, tokenizer)
        self.synchronize()

    def complete_startup(self):
        """Seal startup: check captured-graph storage budgets and stream.

        grants.
        """
        self.graph_storage.check()

        for _, stream in self._lane_streams:
            stream.verify()
        self._startup_complete = True
        for entry in self.entries.values():
            entry._startup_complete = True

    def synchronize(self):
        """Host-synchronize every owned stream and each active device's current.

        stream.
        """
        streams = [self._preparation_stream]
        streams.extend(stream.stream for _, stream in self._lane_streams)
        streams.extend(
            stream.stream for stream in self._module_streams.values()
        )

        devices = {
            canonical_device(self.worker_config.device),
            *self._capture_devices(canonical_device(self.worker_config.device)),
        }
        streams.extend(
            torch.cuda.current_stream(device)
            for device in devices
            if device.type == "cuda"
        )

        close_resources(
            *(
                stream.synchronize
                for stream in dict.fromkeys(
                    stream for stream in streams if stream is not None
                )
            )
        )

    def close_graphs(self):
        """The caller drains all borrowed output readers before this.

        call.
        """
        actions = [
            entry.close_graphs
            for entry in (
                *self.entries.values(),
                *self._module_entries.values(),
            )
        ]
        if self.denoising is not None:
            actions.append(self.denoising.close)
        close_resources(*actions)

    def close(self, *, aborted: bool = False):
        """Release every owned context, buffer, graph.

        and stream exactly once.

        ``aborted`` releases after a failure on this rank. Retiring a
        communicator is collective and synchronizing waits on the device, and
        neither returns when the peers are still serving or the device already
        holds stuck work, so an aborted release does neither.
        """
        if self._closed:
            return
        self._closed = True

        # A draw writes request storage, so it completes before its owners
        # release anything; an aborted release leaves it running until exit.
        noise_draws = getattr(self, "noise_draws", None)
        if noise_draws is not None:
            noise_draws.shutdown(wait=not aborted, cancel_futures=True)

        if aborted:
            from uniserve.runtime.resources import retain_until_exit

            # Each stream owner retains its communicators, windows and native
            # stream without waiting for peers or the device.
            retain_until_exit(self)
            close_resources(
                *(
                    partial(owner.close, aborted=True)
                    for owner in (
                        *self._module_streams.values(),
                        *(owner for _, owner in self._lane_streams),
                    )
                )
            )
            return

        actions = [self.synchronize]
        actions.append(self.close_graphs)
        actions.extend(entry.close for entry in self._module_entries.values())
        for entry in self.entries.values():
            actions.append(entry.close)

        # Each stream retires the communicators every context on it shared,
        # then its native resources. Streams close in reverse creation order
        # so forks retire before the lane streams they borrow from, on every
        # rank in the same order.
        actions.extend(
            stream.close
            for stream in reversed(tuple(self._module_streams.values()))
        )
        actions.extend(
            stream.close for _, stream in reversed(self._lane_streams)
        )

        try:
            close_resources(*actions)
        finally:
            self.entries.clear()
            self._module_entries.clear()
            self.graph_storage.close()
            self._module_streams.clear()
            self._lane_streams.clear()

    def prepare_text_tokens(self, tokens: tuple[int, ...]) -> torch.Tensor:
        """Prepare the text encoder's input using its own reusable backing."""
        found = [
            (name, call)
            for (name, _, method), (_, call) in self._module_calls.items()
            if method == "encode" and isinstance(call.module, TextEncoder)
        ]
        if len(found) != 1:
            raise InputError("rank has no unambiguous text encoder")
        name, call = found[0]
        runner = self.prepare_module(
            name, TextSize(len(tokens), 1), method="encode"
        )
        return runner.prepare_tokens(
            tokens, capacity=self.worker_config.max_sequence_tokens
        )

    @contextmanager
    def preparing_inputs(
        self, transfers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    ) -> Iterator[None]:
        """Overlap reserved input copies with computation.

        then join device order.

        Sources and destinations must remain reserved through this call's
        physical completion. The final stream wait also covers failed compute,
        so completion and request retirement cannot overtake the input copies.
        """
        stream = self._preparation_stream
        if (
            self._closed
            or canonical_device(self.worker_config.device).type != "cuda"
        ):
            raise RuntimeError(
                "input preparation requires a live CUDA execution owner"
            )
        if stream is None:
            stream = torch.cuda.Stream(device=self.worker_config.device)
            self._preparation_stream = stream

        try:
            with torch.cuda.stream(stream):
                for destination, source in transfers:
                    if (
                        destination.shape != source.shape
                        or destination.dtype != source.dtype
                    ):
                        raise ValueError(
                            "prepared input must match destination shape and "
                            "dtype"
                        )
                    destination.copy_(source, non_blocking=True)
            yield
        finally:
            torch.cuda.current_stream(self.worker_config.device).wait_stream(
                stream
            )

    def image_processor(self) -> ImageProcessor:
        """Require image preprocessing settings for the active call."""
        value = self.processor
        if not isinstance(value, ImageProcessor):
            raise invalid_descriptor("call requires model image processing")
        return value

    def forward(
        self,
        tasks: tuple[tuple[InputRow, Call], ...],
        *,
        cache: KVCacheManager | None,
        tables: BlockTables | None,
        states: DecodeState | None,
    ) -> Iterator[tuple[tuple[int, ...], ExecutionOutput | BaseException]]:
        """Execute actual compatible rows.

        yielding results at their original indexes.

        A failed model call identifies every participating row. The caller owns
        batches and decides which dependent calls to suppress.
        Fatal failures propagate immediately because later device work is
        unsafe.
        """
        # Rows sharing an entry, forward mode, device, and media shape form
        # one homogeneous numerical call.
        grouped: dict[tuple[object, ...], list[int]] = defaultdict(list)
        bindings: dict[int, ModelRunner] = {}
        for index, (task, call) in enumerate(tasks):
            entry = self._forward_calls.get((call.component, task.forward_mode))
            if entry is None:
                yield (
                    (index,),
                    invalid_descriptor(
                        f"execution has no {call.kind.value!r} binding "
                        f"for {call.component!r}"
                    ),
                )
                continue

            target = entry.device
            bindings[index] = entry
            shape: tuple[int, ...] = ()
            if isinstance(task, VisionRow):
                shape = tuple(int(value) for value in task.encode_pixels.shape)
            elif isinstance(task, (DiffusionRow, DecodeRow)):
                shape = (task.image_height, task.image_width)
            grouped[(entry, task.forward_mode, str(target), shape)].append(
                index
            )

        groups = list(grouped.values())
        for group_index, indexes in enumerate(groups):
            rows = tuple(tasks[index][0] for index in indexes)
            try:
                forward_started = time.perf_counter_ns()
                result = self.run_forward_group(
                    rows,
                    calls=tuple(tasks[index][1] for index in indexes),
                    cache=cache,
                    tables=tables,
                    states=states,
                )
                if result.request_pool_indices is None:
                    raise RuntimeError(
                        "forward output has no request slot views"
                    )

                # Join the lane stream's output fence so later control-stream
                # work observes this group's writes in order.
                if result.output_event is not None:
                    torch.cuda.current_stream(
                        bindings[indexes[0]].device
                    ).wait_event(result.output_event)

                if all(row.forward_mode is ForwardMode.DECODE for row in rows):
                    if result.stats is None:
                        raise RuntimeError("text forward lost its statistics")
                    components = dict(result.stats.component_us)
                    record_component(
                        components, "text_model_forward", forward_started
                    )
                    result = replace(
                        result,
                        stats=replace(result.stats, component_us=components),
                    )

                # A later call may replay the same graph or reuse its staging.
                # Preserve these numerical values until their consumer runs.
                if any(
                    bindings[later[0]] is bindings[indexes[0]]
                    for later in groups[group_index + 1 :]
                ):
                    result = result.clone()
            except BaseException as error:
                if classify(error).fatal:
                    raise
                yield tuple(indexes), error
            else:
                yield tuple(indexes), result

    def run_forward_group(
        self,
        rows: tuple[InputRow, ...],
        *,
        calls: tuple[Call, ...],
        cache: KVCacheManager | None,
        tables: BlockTables | None,
        states: DecodeState | None,
    ) -> ExecutionOutput:
        """Stage forward rows, choose eager or CUDA graph execution.

        invoke the model, and validate outputs.
        """
        if self._closed:
            raise RuntimeError("model runner is closed")
        tasks = rows
        if not tasks:
            raise ValueError("model runner received an empty call")

        started = time.perf_counter_ns()
        graph_eligible = all(
            isinstance(task.forward_mode, ForwardMode)
            or task.forward_mode is MediaCall.DENOISING
            for task in tasks
        )
        modes = frozenset(task.forward_mode for task in tasks)
        if len(modes) != 1:
            raise ValueError(
                "one numerical call requires homogeneous call kinds"
            )
        forward_mode = next(iter(modes))

        call_keys = tuple(
            (
                call.request_key.engine_id,
                call.request_key.request_id,
                call.request_key.request_epoch,
                call.call_id,
            )
            for call in calls
        )
        entry = self._forward_calls.get((calls[0].component, forward_mode))
        if entry is None:
            raise InputError(
                f"model runner has no {calls[0].kind.value!r} binding "
                f"for {calls[0].component!r}",
                phase="input_staging",
                route=forward_mode.value,
                calls=call_keys,
            )

        target = entry.device
        lane_runtime = entry.cuda_stream
        buffers = entry.input_buffers
        assert buffers is not None

        # Staging copies and the forward itself belong to one stage on the
        # timeline, so the range opens before input staging.
        with profile_range(
            f"uniserve.model.forward rank={self.worker_config.rank} "
            f"work={calls[0].component}.{forward_mode.value}"
        ):
            try:
                if lane_runtime is not None:
                    lane_runtime.wait(torch.cuda.current_stream(target))
                stream_context = (
                    nullcontext()
                    if lane_runtime is None
                    else torch.cuda.stream(lane_runtime.stream)
                )
                with stream_context:
                    batch = entry.prepare_inputs(
                        tasks,
                        forward_mode=forward_mode,
                        cache=cache,
                        tables=tables,
                        states=states,
                    )
                    request_pool_indices = batch.request_pool_indices
            except Exception as error:
                output_event = (
                    None if lane_runtime is None else lane_runtime.record()
                )
                if output_event is not None:
                    torch.cuda.current_stream(target).wait_event(output_event)
                raise _input_failure(error, forward_mode, call_keys) from error

            output_event = None
            try:
                # Only a uniform single-token last-logits decode may borrow the
                # graph's output storage; mixed rows receive owned values.
                with torch.inference_mode():
                    output = entry.run_batch(
                        batch,
                        entry.batch_forward,
                        eligible=graph_eligible,
                        borrow_output=all(
                            isinstance(task, TokenRow)
                            and task.query_tokens == 1
                            and task.selection is TokenSelection.LAST_LOGITS
                            for task in tasks
                        ),
                    )

                output.validate_for(batch)
                _validate_outputs(output.values, tasks, target)

                output_event = (
                    None if lane_runtime is None else lane_runtime.record()
                )
                duration_us = (time.perf_counter_ns() - started) // 1000
                if output.stats is None:
                    raise RuntimeError(
                        "entry forward lost its execution statistics"
                    )
                stats = replace(
                    output.stats,
                    mode_counts={forward_mode.value: 1},
                    mode_tokens={forward_mode.value: len(tasks)},
                    mode_us={forward_mode.value: duration_us},
                    component_us={"forward": duration_us},
                )
                return replace(
                    output,
                    request_pool_indices=request_pool_indices,
                    output_event=output_event,
                    stats=stats,
                )
            except Exception as error:
                # A failed model can leave kernels on a lane stream. Its caller
                # retires storage behind the current stream's output fence, so
                # join every submitted lane access before reporting the failure.
                if output_event is None:
                    output_event = (
                        None if lane_runtime is None else lane_runtime.record()
                    )
                if output_event is not None:
                    torch.cuda.current_stream(target).wait_event(output_event)
                raise _execution_failure(
                    error, forward_mode, call_keys
                ) from error


def _validate_outputs(
    values: tuple[torch.Tensor, ...],
    tasks: tuple[InputRow, ...],
    device: torch.device,
) -> None:
    """Validate one model output tensor per task on the expected device."""
    for value, task in zip(values, tasks, strict=True):
        if value.device != device:
            raise ValueError(
                f"model output is on {value.device}, expected {device}"
            )
        if not value.is_floating_point():
            raise ValueError("raw neural outputs must use a floating dtype")

        if isinstance(task, TokenRow) and value.ndim < 2:
            raise ValueError(
                "token output must retain token and feature dimensions"
            )

        if isinstance(task, DiffusionRow) and value.shape != task.latent.shape:
            raise ValueError("flow prediction shape does not match its latent")

        if isinstance(task, VisionRow) and value.numel() == 0:
            raise ValueError("encoder output must not be empty")

        # A decode (as opposed to denoise) latent task returns an image whose
        # trailing dimensions are [height, width].
        if isinstance(task, DecodeRow):
            if value.ndim < 2 or tuple(value.shape[-2:]) != (
                task.image_height,
                task.image_width,
            ):
                raise ValueError(
                    "decoded tensor does not match the requested image shape"
                )


def _input_failure(
    error: BaseException,
    forward_mode: ForwardMode | MediaCall,
    calls: tuple[tuple[int, int, int, CallId], ...],
) -> InputError:
    """Classify invalid model inputs with their phase and call.

    identities.
    """
    if isinstance(error, InputError):
        return error
    return InputError(
        str(error) or type(error).__name__,
        phase="input_staging",
        route=forward_mode.value,
        calls=calls,
    )


def _execution_failure(
    error: BaseException,
    forward_mode: ForwardMode | MediaCall,
    calls: tuple[tuple[int, int, int, CallId], ...],
) -> WorkerError:
    """Classify a model failure and attach the active phase and call.

    identities.
    """
    if isinstance(error, WorkerError):
        return error
    classified = classify(error)
    if isinstance(error, CUDAGraphError) or classified.code in {
        WorkerErrorCode.RESOURCE_ERROR,
        WorkerErrorCode.FATAL_WORKER_FAILURE,
    }:
        return ResourceError(
            str(error) or type(error).__name__,
            phase="graph_or_device",
            route=forward_mode.value,
            calls=calls,
            fatal=classified.fatal,
        )
    return ComputeError(
        str(error) or type(error).__name__,
        phase="neural_execution",
        route=forward_mode.value,
        calls=calls,
    )
