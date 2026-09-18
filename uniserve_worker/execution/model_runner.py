"""Worker-owned execution of declared numerical capabilities."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Mapping
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
    ImageDecoder,
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
    CUDAGraph,
    CUDAStream,
    ExecutionContext,
    partition_streams,
)
from uniserve.runtime.backends.attention import resolve as attention_backend
from uniserve.runtime.backends.attention.flashinfer import Backend as FlashInfer
from uniserve.runtime.cuda_graph import CUDAGraphError
from uniserve.runtime.device import canonical_device, fill_cpu_ints
from uniserve.runtime.resources import close_resources
from uniserve.tensors import OutputLayout, TensorOutput
from uniserve_worker.config import WorkerConfig
from uniserve_worker.foundation.errors import (
    ComputeError,
    InputError,
    ResourceError,
    WorkerError,
    WorkerErrorCode,
    classify,
    invalid_descriptor,
)
from uniserve_worker.protocol.call import (
    Call,
    ForwardMode,
    ImageParams,
    PipelineStage,
)
from uniserve_worker.protocol.identity import CallId
from uniserve_worker.protocol.output import ForwardStats
from uniserve_worker.runtime.results import resolve_outputs
from uniserve_worker.runtime.staging_buffers import StagingBuffers

from ..bootstrap.components import (
    bind_components,
    call_kinds,
    describe_components,
)
from ..bootstrap.config import ComponentConfig
from ..bootstrap.inputs import capability, image_builder, media_builder
from ..profiling import record_component
from .batch import ExecutionOutput, InputBatch
from .denoising_runner import DenoisingRunner
from .graph_inputs import (
    BatchGraph,
    PrefillShape,
    clone_inputs,
    copy_inputs,
    input_signature,
    pad_text,
    private_pool_bytes,
    select_flow_captures,
    select_prefill_captures,
    text_shape,
    widen_prefix,
)
from .input_buffers import InputBuffers
from .model_entry import ModelEntry, capture_required
from .resources import media_state_buffers, output_layouts
from .rows import ForwardRow
from .sampling import TokenSelection
from .text import TextCall

if TYPE_CHECKING:
    from ..runtime.block_tables import BlockTables
    from ..runtime.cache_manager import CacheManager
    from ..runtime.decode_state import DecodeState

logger = logging.getLogger(__name__)


def capture_image_parameters(cfg_branches, *, steps, height, width):
    """Build fixed image-generation params whose CFG scales match the branch.

    count.

    One branch runs unconditioned, two add text guidance, and three add text
    and image guidance; each count maps to its (text, image) scale pair.
    """
    text, image = {1: (1.0, 1.0), 2: (4.0, 1.0), 3: (4.0, 2.0)}[cfg_branches]
    return ImageParams(
        steps=steps,
        cfg_text_scale=text,
        cfg_img_scale=image,
        height=height,
        width=width,
        seed=0,
    )


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


class ModelRunner:
    """Own contexts, input storage, stream bindings and graph residency.

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
        attention=None,
        image_processor=None,
        flow_prompt=None,
    ):
        self.model, self.worker_config = model, worker_config
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
        self._declarations = describe_components(model)

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
                bindings[name] = ModelEntry(name, config, group, mesh, device)
        self.bindings = MappingProxyType(dict(bindings))
        bind_components(model, self.bindings)
        self.state_buffers = media_state_buffers(
            model, self.bindings, worker_config
        )

        self.entries = {}
        self._forward_entries = {}
        self._module_entries = {}
        self._module_contexts = OrderedDict()
        self._module_pools = {}
        self._module_graphs = {}
        self._module_streams = {}
        self._text_calls = {}

        self._lane_streams = []
        self._capture_stream = None
        self._preparation_stream = None
        self._text_staging = self._text_tokens = None

        self.batch_graphs = {}
        self.graph_pools = {}
        self.graph_memory_budgets = {}
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
                    if (
                        isinstance(call.module, CausalLM)
                        and call.entry.method == "forward"
                    ):
                        self._text_calls[id(call.module)] = TextCall(
                            call.module
                        )
                    elif (
                        isinstance(call.module, TextEncoder)
                        and call.entry.method == "encode"
                    ):
                        device = binding.device
                        self._text_tokens = torch.empty(
                            worker_config.max_sequence_tokens,
                            dtype=torch.int64,
                            device=device,
                        )
                        self._text_staging = StagingBuffers(
                            tuple(self._text_tokens.shape),
                            dtype=torch.int64,
                            depth=2,
                            device=device,
                        )

                    if not call_kinds((call,)):
                        continue
                    key = (name, call.path, call.entry.method)
                    self._module_entries[key] = (binding, call)

                    if self.media_builder is not None and isinstance(
                        call.module, Denoiser
                    ):
                        self.denoising = DenoisingRunner(
                            call.module,
                            device=binding.device,
                            capture_stream=self.capture_stream(),
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
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note(
                    f"execution resource cleanup failed: {cleanup!r}"
                )
            raise

    def capture_stream(self):
        """Return the lazily created graph-capture stream.

        or None when graphs are disabled.
        """
        device = canonical_device(self.worker_config.device)
        if self.worker_config.graph_policy == "off" or device.type != "cuda":
            return None
        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream(device=device)
        return self._capture_stream

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
            for _, call in self._module_entries.values()
            if kind in call_kinds((call,))
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

        call)`` pair for an entry.
        """
        entries = [
            (binding, call)
            for (entry, _, entry_method), (
                binding,
                call,
            ) in self._module_entries.items()
            if entry == name and (method is None or entry_method == method)
        ]
        if len(entries) != 1:
            raise InputError(
                f"entry {name!r} requires an unambiguous numerical method"
            )
        return entries[0]

    def prepare_module(self, name, size, *, method=None):
        """Prepare one exact numerical size before dependent media calls.

        Prepared contexts are kept in an LRU per (entry, path, method); when
        the residency bound is reached, the oldest context and the graphs that
        borrow it are retired.
        """
        binding, call = self._module_call(name, method)
        key = (name, call.path, call.entry.method, input_signature(size))
        if key not in self._module_contexts:
            resident = tuple(
                value for value in self._module_contexts if value[:3] == key[:3]
            )
            if len(resident) >= self.worker_config.max_request_pool_size:
                self._retire_module(resident[0])

            stream = self.module_stream(name, method=call.entry.method)
            context = ExecutionContext(
                call.module, attention=self.attention, stream=stream
            )
            try:
                if stream is not None:
                    stream.wait_stream(
                        torch.cuda.current_stream(binding.device)
                    )
                context.prepare(size)
            except BaseException:
                context.close()
                raise

            self._module_contexts[key] = context
            if stream is not None:
                with torch.cuda.device(binding.device):
                    self._module_pools[id(context)] = {
                        binding.device: torch.cuda.MemPool()
                    }

        self._module_contexts.move_to_end(key)
        return self._module_contexts[key]

    def module_stream(self, name, *, method=None):
        """Bind one numerical entry to a stream in its existing resource.

        grant.
        """
        binding, call = self._module_call(name, method)
        if binding.device.type != "cuda":
            return None

        self._initialize_streams(event_slots=2)
        key = (name, call.path, call.entry.method)
        if key not in self._module_streams:
            # A standalone entry forks the lane stream that covers its call
            # kinds, or creates a full-device stream when lanes are off.
            entry_kinds = call_kinds((call,))
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
        return self._module_streams[key].stream

    def call_stream(self, call):
        """Select a standalone capability's stream.

        staged batches bind their own lanes.
        """
        if (call.entry, call.kind) in self._forward_entries:
            return None
        if (
            call.kind is PipelineStage.LATENT_PREPARATION
            and (call.entry, PipelineStage.DENOISING) in self._forward_entries
        ):
            return None
        calls = tuple(
            call
            for (name, _, _), (_, call) in self._module_entries.items()
            if name == call.entry and call.kind in call_kinds((call,))
        )
        if not calls:
            return None

        if call.kind is PipelineStage.LATENT_PREPARATION:
            # Preparation composes the denoiser's initialization with optional
            # conditioning encoders. The denoiser owns this call's stream;
            # its encoder calls retain their ordinary input/output dependencies.
            calls = tuple(
                call for call in calls if isinstance(call.module, Denoiser)
            )

        if len(calls) != 1:
            raise InputError("call requires one bound numerical capability")
        return self.module_stream(call.entry, method=calls[0].entry.method)

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
        context = self._module_contexts.pop(key)
        if context.stream is not None:
            context.stream.synchronize()

        for graph_key in tuple(self._module_graphs):
            if graph_key[0] == id(context):
                self._module_graphs.pop(graph_key)[0].close()

        context.close()
        self._module_pools.pop(id(context), None)

    @property
    def encoder_kinds(self):
        return frozenset(
            self._encoder_kind(call.module)
            for _, call in self._module_entries.values()
            if call.entry.method == "encode"
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
            for (name, _, _), (_, call) in self._module_entries.items()
            if call.entry.method == "encode"
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
        """Run an actual capability method with caller-supplied numerical.

        arguments.

        The call runs eager or through a per-signature CUDA graph; either way
        its result is cloned so the caller owns values independent of the
        context's staging storage.
        """
        # The range's ``work`` field lets timeline tooling attribute this
        # entry's kernels and copies to the serving stage that issued them.
        with profile_range(
            f"uniserve.model.module rank={self.worker_config.rank} work={name}"
        ):
            binding, call = self._module_call(name, method)
            context = self.prepare_module(name, size, method=call.entry.method)
            started, path = time.perf_counter_ns(), "eager"

            stream = context.stream
            if stream is not None:
                stream.wait_stream(torch.cuda.current_stream(binding.device))

            # These views belong to this prepared context. Graph input staging
            # owns only changing numerical arguments, never a duplicate of the
            # workspace.
            resources = {}
            if isinstance(call.module, (VideoDecoder, VideoPostprocessor)):
                resources["constants"] = context.constants
                resources["workspace"] = context.workspace
            elif isinstance(call.module, AudioDecoder):
                resources["workspace"] = context.workspace

            values = (args, kwargs)
            key = (id(context), input_signature(values))
            graph = self._module_graphs.get(key)
            can_capture = (
                stream is not None
                and self.worker_config.graph_policy != "off"
                and not isinstance(call.module, VideoPostprocessor)
            )
            with context.activate():
                if can_capture:
                    missing = capture_required(
                        graph is None, call.groups, binding.device
                    )
                    if missing:
                        if graph is not None:
                            graph[0].close()
                        # One eager warmup run on the static inputs precedes
                        # capture.
                        static = clone_inputs(values)
                        call.forward(*static[0], **static[1], **resources)
                        executable = CUDAGraph(
                            context=context,
                            pools=self._module_pools[id(context)],
                        )
                        try:
                            executable.capture(
                                lambda: call.forward(
                                    *static[0], **static[1], **resources
                                )
                            )
                        except BaseException:
                            executable.close()
                            raise
                        graph = (executable, static)
                        self._module_graphs[key] = graph
                        path = "graph_capture"
                    else:
                        path = "graph_replay"
                    copy_inputs(graph[1], values)
                    result = graph[0].replay()
                else:
                    result = call.forward(*args, **kwargs, **resources)
                output = self._result(result).clone()

            if stream is not None:
                torch.cuda.current_stream(binding.device).wait_stream(stream)
            return replace(output, stats=_observations(name, started, path))

    @staticmethod
    def _result(result):
        """Normalize a module call result into tensors plus optional layouts."""
        if isinstance(result, torch.Tensor):
            result = (result,)
        if isinstance(result, Mapping):
            result = tuple(
                value for values in result.values() for value in values
            )

        values, layouts = [], []
        for value in result:
            if isinstance(value, TensorOutput):
                values.append(value.tensor)
                layouts.append(value.layout)
            elif isinstance(value, torch.Tensor):
                values.append(value)
                layouts.append(None)
            else:
                raise ComputeError(
                    "participating numerical call did not return a tensor"
                )
        return ExecutionOutput(tuple(values), layouts=tuple(layouts))

    def run_denoising(self, inputs, schedules, *, state, slot, input_key):
        """Run one denoising step for a resident trajectory and time it."""
        if self.denoising is None:
            raise InputError("rank does not own denoising computation")
        started = time.perf_counter_ns()
        with profile_range(
            f"uniserve.model.denoise rank={self.worker_config.rank} "
            "work=denoiser"
        ):
            values, path = self.denoising.step(
                inputs, schedules, state=state, slot=slot, input_key=input_key
            )
        return replace(
            self._result(values), stats=_observations("denoiser", started, path)
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

        results = tuple(
            (call, name, layout)
            for call in self._declarations[entry]
            for name, layout in output_layouts(
                self.model,
                self.worker_config,
                call,
                frames=None if media is None else media.num_frames,
                prompt_tokens=num_prompt_tokens,
            ).items()
        )
        call, name, layout = results[output_index]
        if isinstance(call.module, Denoiser) and any(
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
        audio = isinstance(call.module, AudioDecoder)
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
            spans = call.module.unit_samples(layout.shape[0], units)
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
        variants,
        max_inflight,
    ):
        """Bind staged input resources and graph budgets for every capability.

        Creates one ModelEntry per (entry, path, lane) covering a staged
        computation kind, with its input buffers, execution context, decode /
        prefill capture shapes, and private CUDA graph memory pools. Callable
        exactly once.
        """
        from ..bootstrap.capacity import device_total_bytes
        from ..config import (
            DEFAULT_PREFILL_GRAPH_ROW_BUCKETS,
            graph_memory_budget_bytes,
        )

        if self.batch_graphs:
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

        staged = {
            ForwardMode.PREFILL,
            ForwardMode.DECODE,
            ForwardMode.VERIFY,
            PipelineStage.VISION_ENCODING,
            PipelineStage.LATENT_ENCODING,
            PipelineStage.IMAGE_DECODING,
        }
        if self.image_builder is not None:
            staged.add(PipelineStage.DENOISING)

        for (name, path, method), (
            placement,
            call,
        ) in self._module_entries.items():
            staged_kinds = call_kinds((call,)) & staged
            if not staged_kinds:
                continue

            target = (
                canonical_device(config.generation_device or config.device)
                if staged_kinds
                & {PipelineStage.LATENT_ENCODING, PipelineStage.IMAGE_DECODING}
                else placement.device
            )
            streams = [
                (lane, stream)
                for lane, stream in self._lane_streams
                if stream.device == target
            ]
            for lane, stream in streams or ((None, None),):
                kinds = (
                    staged_kinds
                    if lane is None
                    else staged_kinds.intersection(lane.call_kinds)
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
                entry = ModelEntry(
                    name,
                    placement.config,
                    placement.process_group,
                    placement.mesh,
                    target,
                    calls=(call,),
                    call_kinds=tuple(kinds),
                    cuda_stream=stream,
                )
                try:
                    if stream is not None:
                        stream.wait(torch.cuda.current_stream(target))
                    with (
                        nullcontext()
                        if stream is None
                        else torch.cuda.stream(stream.stream)
                    ):
                        entry.input_buffers = InputBuffers(
                            config=fields,
                            device=target,
                            image_builder=self.image_builder,
                            max_inflight=max_inflight,
                        )
                        entry.context = ExecutionContext(
                            call.module,
                            cache=kv_cache.cache
                            if isinstance(
                                call.module, (CausalLM, ImageDenoiser)
                            )
                            else None,
                            attention=self.attention,
                            stream=None if stream is None else stream.stream,
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
                        entry.context.prepare(size)
                except BaseException as error:
                    try:
                        close_resources(
                            *(
                                owner.close
                                for owner in (
                                    entry.context,
                                    entry.input_buffers,
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
                self.batch_graphs[entry] = {}
                self.decode_shapes[entry], self.prefill_shapes[entry] = (
                    decode,
                    prefill,
                )

                if config.graph_policy != "off" and target.type == "cuda":
                    pools = {}
                    for graph_device in (
                        target,
                        *self._capture_devices(target),
                    ):
                        with torch.cuda.device(graph_device):
                            pools[graph_device] = torch.cuda.MemPool()
                        self.graph_memory_budgets[graph_device] = (
                            graph_memory_budget_bytes(
                                device_total_bytes(graph_device)
                            )
                        )
                    self.graph_pools[entry] = pools

                for kind in kinds:
                    key = (name, kind)
                    if key in self._forward_entries:
                        raise ValueError(
                            f"computation {key} has multiple lane bindings"
                        )
                    self._forward_entries[key] = entry

    def batch_forward(self, entry, batch, *, padded=False):
        """Dispatch one staged batch to the numerical call of its capability.

        kind.
        """
        call = entry.calls[0]
        module, inputs = call.module, batch.inputs

        if isinstance(module, CausalLM):
            text = self._text_calls[id(module)]
            return (
                text.last_logits(inputs)
                if padded
                and batch.token_selections[0] is TokenSelection.LAST_LOGITS
                else text(inputs, batch.token_selections)
            )

        if isinstance(module, ImageDenoiser):
            result = module(
                inputs,
                state={},
                constants=entry.context.constants,
                workspace=entry.context.workspace,
            )["image"]

            # Predictions come from the last pipeline stage; other stages
            # broadcast placeholder storage that the broadcast overwrites, so
            # every rank returns the same per-image values.
            pipeline = module.mesh.get_group(
                "pp" if "pp" in module.mesh.axes else ()
            )
            values = []
            for prediction, size in zip(result, inputs.sizes, strict=True):
                value = (
                    prediction.tensor
                    if prediction is not None
                    else torch.empty(
                        module.latent_shape("image", size),
                        dtype=module.prediction_dtype,
                        device=entry.device,
                    )
                )
                pipeline.broadcast(value, src=pipeline.size - 1)
                values.append(value)
            return ExecutionOutput(tuple(values))

        if isinstance(module, PatchEncoder):
            return ExecutionOutput(module.encode(inputs))

        if isinstance(module, PatchAutoencoder):
            return ExecutionOutput(
                tuple(module.encode(torch.stack(inputs.images)).unbind(0))
            )

        if isinstance(module, ImageDecoder):
            values = module.decode(inputs.latents, sizes=inputs.sizes)
            return ExecutionOutput(
                values,
                layouts=tuple(
                    OutputLayout(
                        tuple(value.shape),
                        value.dtype,
                        tuple(slice(0, extent) for extent in value.shape),
                        value_range=module.decoder.value_range,
                    )
                    for value in values
                ),
            )

        raise InputError("staged entry has no supported numerical capability")

    def select_graph_shape(self, entry, batch, *, eligible):
        """Pick a CUDA graph bucket for a staged batch, or None to stay eager.

        Returns ``(graph key, padded/exact execution batch, is text bucket)``.
        Text batches are padded into a configured bucket keyed by shape and
        input signature; other capabilities replay only their exact signature.
        """
        if not eligible or entry not in self.graph_pools:
            return None

        text = isinstance(entry.calls[0].module, CausalLM)
        decode = (
            text
            and batch.forward_mode is ForwardMode.DECODE
            and batch.inputs.attention.queries.host == (1,) * batch.row_count
        )
        if not decode and not self.worker_config.prefill_cuda_graph:
            return None

        if text:
            shape = text_shape(
                batch,
                decode_sizes=self.decode_shapes[entry],
                prefill_shapes=self.prefill_shapes[entry],
                context_blocks=self.decode_context_blocks,
            )
            if shape is None:
                return None

            if shape[-1] and batch.decode_force_finish is None:
                # Direct token inputs and device continuations use the same
                # numerical decode graph. Only the latter consumes its greedy
                # continuation result; the former supplies an inert finish mask.
                finish = entry.input_buffers.decode_force_finish[
                    : batch.row_count
                ]
                finish.zero_()
                batch = replace(batch, decode_force_finish=finish)

            padded = pad_text(batch, *shape)
            inputs = padded.inputs
            key = (
                "text",
                shape,
                inputs.input_ids.dtype,
                inputs.positions.ndim,
                inputs.embeddings is not None,
                batch.decode_force_finish is not None,
                inputs.attention.causal[0],
                padded.token_selections[0],
            )
            return key, padded, True

        execution = widen_prefix(batch, entry.input_buffers.max_blocks_per_row)
        return ("exact", input_signature(execution)), execution, False

    @torch.inference_mode()
    def eager_batch(self, entry, batch, forward):
        """Run a staged batch eagerly inside its entry's execution context."""
        with entry.context.activate():
            attention = getattr(batch.inputs, "attention", None)
            if attention is not None:
                entry.context.bind_attention(attention)
            return forward(batch)

    @torch.inference_mode()
    def capture_batch(self, entry, batch, forward):
        """Capture a staged batch's graph on the entry stream.

        fenced on both sides.
        """
        context = entry.context
        current = (
            torch.cuda.current_stream(entry.device)
            if entry.device.type == "cuda"
            else None
        )
        if context.stream is not None:
            context.stream.wait_stream(current)
        try:
            with context.activate():
                self._capture_batch(entry, batch, forward)
        finally:
            if context.stream is not None:
                current.wait_stream(context.stream)

    def _capture_batch(self, entry, batch, forward):
        if self._startup_complete:
            raise CUDAGraphError("batch capture is outside startup preparation")

        selected = self.select_graph_shape(entry, batch, eligible=True)
        if selected is None:
            self.eager_batch(entry, batch, forward)
            return

        key, execution, padded = selected
        if key in self.batch_graphs[entry]:
            return

        invoke = (
            partial(self.batch_forward, entry, padded=True)
            if padded
            else forward
        )
        # Text buckets use the entry's stable staging addresses, ordered on
        # its execution stream. Exact calls can include borrowed request
        # latents; own those inputs independently of their pool-slot lifetime.
        static = execution if padded else clone_inputs(execution)
        graph = BatchGraph.capture(
            entry.context,
            static,
            invoke,
            pools=self.graph_pools[entry],
            cache=self.kv_cache.cache,
            predicates=self.decode_predicates,
        )
        self.batch_graphs[entry][key] = graph

    @torch.inference_mode()
    def run_batch(
        self, entry, batch, forward, *, eligible, borrow_output=False
    ):
        with entry.context.activate():
            return self._run_batch(
                entry,
                batch,
                forward,
                eligible=eligible,
                borrow_output=borrow_output,
            )

    def _run_batch(
        self, entry, batch, forward, *, eligible, borrow_output=False
    ):
        selected = self.select_graph_shape(entry, batch, eligible=eligible)
        if selected is None:
            return replace(
                self.eager_batch(entry, batch, forward),
                stats=ForwardStats(cuda_graph_runtime_mode_counts={"eager": 1}),
            )

        key, execution, bucketed = selected
        captured = False
        if key not in self.batch_graphs[entry]:
            # Only configured text buckets may capture at run time; an exact
            # signature without a resident graph simply runs eager.
            if not bucketed:
                return replace(
                    self.eager_batch(entry, batch, forward),
                    stats=ForwardStats(
                        cuda_graph_runtime_mode_counts={"eager": 1}
                    ),
                )
            if self._startup_complete:
                raise CUDAGraphError(
                    f"configured graph bucket is not resident: {key!r}"
                )
            self.capture_batch(entry, batch, forward)
            captured = True

        result = self.batch_graphs[entry][key].replay(
            execution, rows=batch.row_count, borrow=borrow_output
        )
        if batch.decode_force_finish is None:
            result = replace(result, greedy=None)

        return replace(
            result,
            stats=ForwardStats(
                cuda_graph_runtime_mode_counts={
                    "graph_capture" if captured else "graph_replay": 1
                },
                cuda_graph_captures=int(captured),
                cuda_graph_replays=int(not captured),
                cuda_graph_unpadded_tokens=batch.row_count,
                cuda_graph_padded_tokens=execution.row_count - batch.row_count,
            ),
        )

    def call_devices(self, call):
        """Return ``(compute, staged, output)`` devices for one call.

        Media-generation kinds run on the generation device when one is
        configured. The staged device is the bound execution entry's device,
        or the compute device when the call has no staged binding.
        """
        binding = self.bindings.get(call.entry)
        source = (
            canonical_device(self.worker_config.device)
            if binding is None
            else binding.device
        )
        if self.image_builder is not None and call.kind in {
            PipelineStage.LATENT_PREPARATION,
            PipelineStage.DENOISING,
            PipelineStage.IMAGE_DECODING,
        }:
            source = canonical_device(
                self.worker_config.generation_device
                or self.worker_config.device
            )

        entry = self._forward_entries.get((call.entry, call.kind))
        return source, source if entry is None else entry.device, source

    def warmup(self, storage):
        """Run media warmup passes, then drain every owned stream."""
        if self.media_builder is not None:
            from .video import (
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
        from .startup import prepare_decode, prepare_prefill

        for phase in ("prefill", "decode", "flow"):
            for entry in self.batch_graphs:
                forward = partial(self.batch_forward, entry)
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
                    phase == "flow"
                    and PipelineStage.DENOISING in entry.call_kinds
                ):
                    from .flow import prepare_flow

                    prepare_flow(self, entry, latents, tokenizer)
        self.synchronize()

    def complete_startup(self):
        """Seal startup: check captured-graph memory budgets and stream.

        grants.
        """
        for device, budget in self.graph_memory_budgets.items():
            pools = {
                tuple(values[device].id)
                for values in self.graph_pools.values()
                if device in values
            }
            if private_pool_bytes(device, pools) > budget:
                raise CUDAGraphError(
                    "captured graph residency exceeds its device budget"
                )

        for _, stream in self._lane_streams:
            stream.verify()
        self._startup_complete = True

    def synchronize(self):
        """Host-synchronize every owned stream and each active device's current.

        stream.
        """
        streams = [self._capture_stream, self._preparation_stream]
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
            graph.graph.close
            for graphs in self.batch_graphs.values()
            for graph in graphs.values()
        ]
        actions.extend(graph.close for graph, _ in self._module_graphs.values())
        if self.denoising is not None:
            actions.append(self.denoising.close)
        try:
            close_resources(*actions)
        finally:
            for graphs in self.batch_graphs.values():
                graphs.clear()
            self._module_graphs.clear()

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

        actions = [] if aborted else [self.synchronize]
        actions.append(self.close_graphs)
        actions.extend(
            context.close for context in self._module_contexts.values()
        )
        for entry in self.batch_graphs:
            actions.extend((entry.context.close, entry.input_buffers.close))
        if self._text_staging is not None:
            actions.append(self._text_staging.close)

        # A stream's communicator bindings are released with it, because they
        # are shared by every context that ran on it rather than owned by
        # whichever one established them.
        from uniserve.runtime.execution import close_stream_collectives

        actions.extend(
            partial(close_stream_collectives, owner.stream, aborted=aborted)
            for owner in self._module_streams.values()
        )
        # Streams close in reverse creation order so forks retire before
        # the lane streams they borrow from.
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
            self.batch_graphs.clear()
            self.graph_pools.clear()
            self._module_contexts.clear()
            self._module_pools.clear()
            self._module_streams.clear()
            self._lane_streams.clear()
            self._text_tokens = self._text_staging = None

    def stage_text_tokens(self, tokens: tuple[int, ...]) -> torch.Tensor:
        """Stage one text encoder input on the caller's stream without a world.

        broadcast.

        The input is borrowed until the next staging call. Host sources remain
        retained through the copy fence independently of subsequent model work.
        """
        staging, storage = self._text_staging, self._text_tokens
        if staging is None or storage is None:
            raise InputError("rank has no text encoder input storage")
        if not 1 <= len(tokens) <= storage.numel():
            raise InputError("text encoder input exceeds its token capacity")

        index, host = staging.acquire()
        fill_cpu_ints(host, tokens)
        target = storage[: len(tokens)]
        target.copy_(host[: len(tokens)], non_blocking=storage.is_cuda)
        staging.record_copy(index)
        return target.view(1, -1)

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
        tasks: tuple[tuple[ForwardRow, Call], ...],
        *,
        cache: CacheManager | None,
        tables: BlockTables | None,
        states: DecodeState | None,
    ) -> Iterator[tuple[tuple[int, ...], ExecutionOutput | BaseException]]:
        """Execute actual compatible rows.

        yielding results at their original indexes.

        A failed model call identifies every participating row. The caller owns
        completion groups and decides which dependent calls to suppress.
        Fatal failures propagate immediately because later device work is
        unsafe.
        """
        # Rows sharing an entry, forward mode, device, and media shape form
        # one homogeneous numerical call.
        grouped: dict[tuple[object, ...], list[int]] = defaultdict(list)
        bindings: dict[int, ModelEntry] = {}
        for index, (task, call) in enumerate(tasks):
            entry = self._forward_entries.get((call.entry, task.forward_mode))
            if entry is None:
                yield (
                    (index,),
                    invalid_descriptor(
                        f"execution has no {call.kind.value!r} binding "
                        f"for {call.entry!r}"
                    ),
                )
                continue

            target = entry.device
            bindings[index] = entry
            shape: tuple[int, ...] = ()
            if task.encode_pixels is not None:
                shape = tuple(int(value) for value in task.encode_pixels.shape)
            elif task.latent is not None:
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
        rows: tuple[ForwardRow, ...],
        *,
        calls: tuple[Call, ...],
        cache: CacheManager | None,
        tables: BlockTables | None,
        states: DecodeState | None,
    ) -> ExecutionOutput:
        """Stage forward rows, choose eager or CUDA graph execution.

        invoke the model, and validate outputs.
        """
        tasks = rows
        if not tasks:
            raise ValueError("model runner received an empty call")

        started = time.perf_counter_ns()
        graph_eligible = all(
            isinstance(task.forward_mode, ForwardMode)
            or task.forward_mode is PipelineStage.DENOISING
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
        entry = self._forward_entries.get((calls[0].entry, forward_mode))
        if entry is None:
            raise InputError(
                f"model runner has no {calls[0].kind.value!r} binding "
                f"for {calls[0].entry!r}",
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
            f"work={calls[0].entry}.{forward_mode.value}"
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
                    batch = buffers.stage(
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

            def invoke(value: InputBatch) -> ExecutionOutput:
                return self.batch_forward(entry, value)

            output_event = None
            try:
                # Only a uniform single-token last-logits decode may borrow the
                # graph's output storage; mixed rows receive owned values.
                with torch.inference_mode():
                    output = self.run_batch(
                        entry,
                        batch,
                        invoke,
                        eligible=graph_eligible,
                        borrow_output=all(
                            (
                                task.request_indexed_decode
                                or task.token_ids is not None
                            )
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
    tasks: tuple[ForwardRow, ...],
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

        if (
            task.request_indexed_decode
            or task.token_ids is not None
            or task.token_embeddings is not None
        ) and value.ndim < 2:
            raise ValueError(
                "token output must retain token and feature dimensions"
            )

        if (
            task.latent is not None
            and task.image_tokens > 0
            and value.shape != task.latent.shape
        ):
            raise ValueError("flow prediction shape does not match its latent")

        if task.encode_pixels is not None and value.numel() == 0:
            raise ValueError("encoder output must not be empty")

        # A decode (as opposed to denoise) latent task returns an image whose
        # trailing dimensions are [height, width].
        if task.latent is not None and task.image_tokens == 0:
            if value.ndim < 2 or tuple(value.shape[-2:]) != (
                task.image_height,
                task.image_width,
            ):
                raise ValueError(
                    "decoded tensor does not match the requested image shape"
                )


def _input_failure(
    error: BaseException,
    forward_mode: ForwardMode | PipelineStage,
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
    forward_mode: ForwardMode | PipelineStage,
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
