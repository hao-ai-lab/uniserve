"""Worker-owned execution of declared numerical capabilities.

``ModelExecutor`` binds a loaded model's components to this rank, discovers
their capabilities, and owns everything that executes them: CUDA streams
(execution lanes and their forks), input staging, and prepared execution
contexts and captured graphs, whose allocations its ``GraphStorage``
accounts against per-device budgets. It keeps three kinds of runner:

- staged entries, created once by ``configure_inputs`` per (component, path,
  lane) with fixed input buffers, which ``forward`` drives with homogeneous
  batches of token rows and image-path rows (vision and latent encoding,
  image denoising and decoding);
- module entries, prepared by ``prepare_module`` per exact numerical size
  and invoked through ``run_module`` for standalone calls such as encoders,
  decoders and the video post-processor;
- diffusion runners, one per layout of a standalone video denoiser, which
  ``run_denoising`` steps (see ``uniserve_worker.execution.media``).

The worker constructs one executor per rank and drives it through startup
(``bind_diffusion_storage`` on a denoising rank, ``configure_inputs`` on a
rank that owns a KV cache, ``warmup`` and ``capture`` on a rank with
numerical methods, then ``complete_startup``), serves calls, and at shutdown
releases it with ``close_graphs`` and ``close`` (an aborted shutdown calls
only ``close``).
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from uniserve.distributed import Communicator, DeviceMesh
from uniserve.model import (
    AudioDecoder,
    AudioEncoder,
    CausalLM,
    Denoiser,
    ImageDenoiser,
    MultimodalEncoder,
    PatchEncoder,
    TextConditioner,
    TextEncoder,
    TextSize,
    VideoDecoder,
    VideoEncoder,
    VideoPostprocessor,
    VisionInput,
)
from uniserve.nn.vae import PatchAutoencoder
from uniserve.processing import ImageProcessor
from uniserve.profiling import profile_range
from uniserve.runtime import (
    CUDAStream,
    ExecutionContext,
    Scratch,
    partition_streams,
)
from uniserve.runtime.backends.attention import resolve as attention_backend
from uniserve.runtime.backends.attention.flashinfer import Backend as FlashInfer
from uniserve.runtime.cuda_graph import CUDAGraphError
from uniserve.runtime.device import canonical_device, process_device_bytes
from uniserve.runtime.resources import close_resources
from uniserve.tensors import OutputLayout
from uniserve_worker.bootstrap.components import (
    VIDEO_CODEC_COMPONENT,
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
from uniserve_worker.config.execution import LaneConfig, WorkerConfig
from uniserve_worker.errors import (
    ComputeError,
    InputError,
    ResourceError,
    WorkerError,
    WorkerErrorCode,
    classify,
    invalid_descriptor,
)
from uniserve_worker.execution.conditions import (
    CONDITION_PRODUCTS,
    condition_encoder,
    condition_layout,
    library_conditions,
)
from uniserve_worker.model_executor.component_binding import (
    ComponentBinding,
)
from uniserve_worker.model_executor.cuda_graph import (
    input_signature,
)
from uniserve_worker.model_executor.diffusion_inputs import DiffusionRow
from uniserve_worker.model_executor.diffusion_runner import DiffusionRunner
from uniserve_worker.model_executor.graph_inputs import (
    DiffusionShape,
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
    CallKind,
    ForwardMode,
    MediaCall,
)
from uniserve_worker.protocol.identity import CallId
from uniserve_worker.protocol.output import ForwardStats
from uniserve_worker.sampling.metadata import TokenSelection

if TYPE_CHECKING:
    from uniserve_worker.model_executor.media_inputs import MediaBuilder
    from uniserve_worker.storage.block_tables import BlockTables
    from uniserve_worker.storage.decode_state import DecodeState
    from uniserve_worker.storage.kv_cache import KVCacheManager
    from uniserve_worker.storage.latent_pool import LatentPool

logger = logging.getLogger(__name__)


def _observations(name, started, path):
    """Build single-call forward statistics for one module, in microseconds.

    A module invocation takes opaque numerical arguments with no query-token
    notion, so it reports its call and time under ``name`` and no tokens.
    """
    elapsed = (time.perf_counter_ns() - started) // 1000
    return ForwardStats(
        mode_counts={name: 1},
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

    Args:
        model: The loaded model whose components this rank executes.
        worker_config: The rank's execution configuration.
        bindings: Component bindings by entry name. ``None`` binds every
            declared component to this rank alone, a video decoder's as a
            one-rank temporal-unit distribution.
        entry_points: IPC entry declarations; ``None`` uses the
            ``entry_points`` of the module that defines the model's class.
        attention: Attention backend name; ``None`` uses the configured
            backend, or ``"auto"``.
        image_processor: Image preprocessing settings, required by calls
            that go through ``image_processor``.
        flow_prompt: The model's flow prompt, from which image denoising
            calls, warmup requests and startup graph capture resolve their
            branch prefixes.
        max_inflight: The worker's queue depth. It sizes the streams' event
            rings when ``module_stream`` initializes them before
            ``configure_inputs`` does; a lane's own ``max_inflight``
            overrides it.
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

        self.entries: dict[tuple[str, str, str | None], ModelRunner] = {}
        self._forward_calls: dict[tuple[str, CallKind], ModelRunner] = {}
        self._module_calls = {}
        self._call_kinds = {}
        self._runner_types = {}
        self._calls_by_kind = defaultdict(list)
        self._module_entries: OrderedDict[tuple, ModelRunner] = OrderedDict()
        # Keys of the module entries prepared during startup, which stay
        # resident.
        self._startup_modules: set[tuple] = set()
        self._module_streams: dict[tuple[str, str, str], CUDAStream] = {}
        # Transient work areas the startup contexts of one (entry, path,
        # method) borrow on that entry's stream; see ``prepare_module``.
        self._module_scratch: dict[tuple[str, str, str], Scratch] = {}

        self._lane_streams: list[tuple[LaneConfig | None, CUDAStream]] = []
        self._preparation_stream: torch.cuda.Stream | None = None

        self.graph_storage = GraphStorage()
        self.decode_shapes: dict[ModelRunner, tuple[int, ...]] = {}
        self.prefill_shapes: dict[ModelRunner, tuple[PrefillShape, ...]] = {}
        self.prefill_row_sizes: tuple[int, ...] = ()
        self.flow_captures: tuple[DiffusionShape, ...] = ()
        self.flow_cfg_branches: tuple[int, ...] = ()
        self.table_widths: tuple[int, ...] = ()
        self.decode_predicates = None
        self.kv_cache = None
        # A standalone denoiser's component, binding and call, the request
        # bank and latent pool its ladders gather through, and the one
        # runner that serves every layout the media builder admits.
        self._denoiser = None
        self.diffusion_bank: Mapping[str, torch.Tensor] = {}
        self.latent_pool: LatentPool | None = None
        self._diffusion: DiffusionRunner | None = None
        # Layouts the runner prepared while serving, least recently used
        # first (``diffusion_layout``).
        self._serving_layouts: OrderedDict[object, None] = OrderedDict()

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
                        self._denoiser = name, binding, call
            # A rank that denoises draws each request's seeded CPU noise on
            # its own thread, off the service thread that launches device
            # work, as soon as the request is admitted.
            self.noise_draws = (
                ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="worker-noise"
                )
                if self._denoiser is not None
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
        """Return the single module that provides ``kind`` on this rank.

        ``capability_type`` further restricts the candidates to instances of
        that class. Raises ``InputError`` unless exactly one module matches.
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

    def _module_call(self, name, method=None, path=None):
        """Return the unique ``(binding, call)`` pair of component ``name``.

        ``method`` and ``path`` select among the component's numerical
        methods, ``path`` naming the module of a component that exposes one
        method name on several modules. Raises ``RuntimeError`` once the
        executor is closed and ``InputError`` unless exactly one call matches.
        """
        if self._closed:
            raise RuntimeError("model runner is closed")
        entries = [
            (binding, call)
            for (entry, entry_path, entry_method), (
                binding,
                call,
            ) in self._module_calls.items()
            if entry == name
            and (method is None or entry_method == method)
            and (path is None or entry_path == path)
        ]
        if len(entries) != 1:
            raise InputError(
                f"component {name!r} requires an unambiguous numerical method"
            )
        return entries[0]

    def prepare_module(self, name, size, *, method=None, path=None):
        """Prepare one exact numerical size before dependent media calls.

        Prepared contexts are keyed by the input signature of ``size`` per
        (entry, path, method). Contexts prepared during startup stay resident
        for the worker's lifetime and capture into one graph pool per
        (entry, path, method), whose calls run one at a time on their shared
        stream and borrow one ``Scratch``. A caller warming several sizes of
        one entry prepares all of them before its first capture, since a
        later persistent allocation could land in a block an earlier graph
        reuses, and evaluates the largest first, since call sites bind their
        scratch at their first eager call and a later, larger size would
        grow it. Contexts prepared while serving never capture and are kept
        in an LRU whose bound is the request pool size; reaching it retires
        the least recently used one. Returns the prepared runner.
        """
        binding, call = self._module_call(name, method, path)
        key = (name, call.path, call.entry_point.method, input_signature(size))
        if key not in self._module_entries:
            resident = tuple(
                value
                for value in self._module_entries
                if value[:3] == key[:3] and value not in self._startup_modules
            )
            if len(resident) >= self.worker_config.max_request_pool_size:
                self._retire_module(resident[0])

            stream = self.module_stream(
                name, method=call.entry_point.method, path=call.path
            )
            # The worker holds a module's preparation size as an opaque value.
            context: ExecutionContext[object] = ExecutionContext(
                call.module,
                attention=self.attention,
                stream=stream,
                # A method declared for one pipeline stage runs on that stage
                # alone, so its preparation opens no binding over a group the
                # other stages never reach.
                groups=call.groups,
                # Every context of one entry runs on its one stream, so its
                # startup contexts borrow one set of transient work areas. A
                # shared set keeps each backing it grows, so a context
                # prepared while serving owns its own and retires it with
                # the context.
                scratch=None
                if self._startup_complete
                else self._module_scratch.setdefault(key[:3], Scratch()),
                # Serving inputs carry their host sequence lengths; a call
                # that lacks one fails rather than copying it from the device.
                derive_host_lengths=False,
            )
            # An entry given graph devices captures a graph the first time it
            # executes each input signature during startup
            # (``ModelRunner.execute_model``); a video post-processor, and
            # any entry prepared while serving, is given none and always runs
            # eagerly. Startup entries of one call share its graph pool.
            share = next(
                (
                    self._module_entries[value]
                    for value in self._startup_modules
                    if value[:3] == key[:3]
                ),
                None,
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
                and not self._startup_complete
                else (),
                share=share,
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

            # An entry prepared while serving never captures.
            entry._startup_complete = self._startup_complete
            if not self._startup_complete:
                self._startup_modules.add(key)
            self._module_entries[key] = entry

        self._module_entries.move_to_end(key)
        return self._module_entries[key]

    def module_stream(self, name, *, method=None, path=None):
        """Bind one numerical entry to a stream in its existing resource grant.

        ``method`` and ``path`` select the entry as ``_module_call`` does.
        Returns ``None`` for an entry on a non-CUDA device. Each (entry, path,
        method) gets one stream on first use, kept until ``close``. Raises
        ``InputError`` when more than one lane stream on the entry's device
        covers its call kinds, or when lanes are configured and none does;
        errors of ``_module_call`` and ``_initialize_streams`` propagate.
        """
        binding, call = self._module_call(name, method, path)
        if binding.device.type != "cuda":
            return None

        self._initialize_streams(event_slots=self._event_slots)
        key = (name, call.path, call.entry_point.method)
        if key not in self._module_streams:
            # A standalone entry forks the stream on its device that covers
            # its call kinds: a lane stream, or without lanes the device's
            # full-device stream, which covers every kind. Without lanes, an
            # entry on a device that has no full-device stream gets a stream
            # of its own.
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

    def diffusion_entry(self, call) -> DiffusionRunner:
        """Return the batched runner that predicts a KV-conditioned call.

        A denoising call whose denoiser attends to KV prefixes is predicted
        in its component's batched entry; that runner also integrates the
        call's guided predictions.
        """
        entry = self._forward_calls.get((call.component, MediaCall.DENOISING))
        if not isinstance(entry, DiffusionRunner):
            raise invalid_descriptor(
                "denoising call has no batched diffusion runner"
            )
        return entry

    def call_stream(self, call):
        """Select the stream a standalone capability's batch runs on.

        Returns ``None`` for a call a staged entry serves, since staged
        batches bind their own lanes; for latent preparation of a component
        whose denoising is staged; for a call with no bound capability; and
        for a capability on a non-CUDA device. Raises ``InputError`` when the
        component's capabilities for the call do not narrow to exactly one,
        and propagates the errors of ``module_stream``.
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
        elif call.kind is MediaCall.LATENT_ENCODING:
            # A condition encoding round runs the encoder of the latents it
            # publishes: visual rounds the video encoder, the audio round
            # the audio encoder.
            encoder = condition_encoder(call, self.outputs)
            if encoder is not None:
                calls = tuple(
                    candidate
                    for candidate in calls
                    if isinstance(candidate.module, encoder)
                )

        if len(calls) != 1:
            raise InputError("call requires one bound numerical capability")
        owner = self.module_stream(
            call.component,
            method=calls[0].entry_point.method,
            path=calls[0].path,
        )
        return None if owner is None else owner.stream

    def _initialize_streams(self, *, event_slots):
        """Realize execution grants for both staged and standalone capabilities.

        Once streams exist, later calls keep them, whatever their
        ``event_slots``. With lanes configured it partitions the worker
        device's SMs into one stream per lane, and raises ``ValueError`` if
        graphs may also allocate on another device; without lanes it creates
        one full-device stream per CUDA device the worker computes or
        captures on.
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
        """Retire the module entry under ``key`` and its captured graphs.

        The entry's stream is drained first, so no queued work still uses
        what the entry releases.
        """
        self._startup_modules.discard(key)
        entry = self._module_entries.pop(key)
        if entry.context.stream is not None:
            entry.context.stream.synchronize()
        entry.close()

    @property
    def encoder_kinds(self):
        """Encoder kinds, as ``run_encoder`` names them, bound on this rank."""
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
        if isinstance(module, VideoEncoder):
            return "video_condition"
        if isinstance(module, AudioEncoder):
            return "audio_condition"
        return "conditioning"

    def run_encoder(self, kind, *values, **options):
        """Run this rank's encoder of ``kind``.

        ``kind`` is ``"text"``, ``"vision"``, ``"latent"`` or
        ``"conditioning"`` (see ``encoder_kinds``); ``options`` are the
        encoder's keyword inputs, such as a ``TextConditioner``'s
        ``lengths``. Raises ``InputError`` unless exactly one bound encoder
        has that kind.
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
        return self.run_module(
            name, inputs, method="encode", path=call.path, size=size, **options
        )

    def encode_vision(self, inputs: VisionInput) -> ExecutionOutput:
        """Run this rank's vision encoder on packed patch samples.

        Returns one ``[tokens, features]`` tensor per sample of ``inputs``.
        The encoder's context does not depend on the samples' grids, so one
        context serves every call. Raises ``InputError`` unless exactly one
        bound vision encoder exists.
        """
        found = [
            (name, call)
            for (name, _, method), (_, call) in self._module_calls.items()
            if method == "encode" and isinstance(call.module, PatchEncoder)
        ]
        if len(found) != 1:
            raise InputError("rank does not participate in vision encoding")
        name, call = found[0]
        return self.run_module(
            name, inputs, method="encode", path=call.path, size=None
        )

    def _text_capacity(self, num_tokens: int) -> int:
        """Rows a text of ``num_tokens`` tokens is padded to before encoding.

        A worker with a media builder encodes every text in the smallest
        text capacity of its denoiser layouts that holds it, the rows
        startup prepared; any other worker encodes exact lengths.
        """
        builder = self.media_builder
        if builder is None:
            return num_tokens
        return min(
            capacity
            for capacity in builder.text_capacities
            if capacity >= num_tokens
        )

    def encode_text(
        self,
        token_ids,
        *,
        visual: torch.Tensor | None = None,
        image_grids: tuple[tuple[int, int, int], ...] = (),
        video_grids: tuple[tuple[int, int, int], ...] = (),
    ) -> ExecutionOutput:
        """Encode one prompt's tokens at its text capacity.

        The tokens are followed by padding up to ``_text_capacity``. The text
        encoder attends causally, so the padding never reaches the prompt's
        rows, whose features are returned. A prompt holding vision blocks
        passes their features as ``visual``, one row per placeholder in
        prompt order, with the blocks' image and video grids, from which the
        encoder derives the prompt's rotary coordinates; padding is text and
        continues them.
        """
        count = len(token_ids)
        capacity = self._text_capacity(count)
        # Token 0 fills the padding; any vocabulary token serves.
        padded = (*token_ids, *(0,) * (capacity - count))
        tokens = self.prepare_text_tokens(padded)
        options = {}
        if visual is not None:
            encoder = capability(self.model, MultimodalEncoder)
            positions = encoder.positions(
                padded, image_grids=image_grids, video_grids=video_grids
            )
            options = {
                "positions": (positions.to(tokens.device, non_blocking=True),),
                "visual": (visual,),
            }
        result = self.run_encoder("text", tokens, **options)
        return replace(
            result, values=tuple(value[:count] for value in result.values)
        )

    def encode_conditioning(self, features: torch.Tensor) -> ExecutionOutput:
        """Refine one prompt's ``[tokens, width]`` text features.

        A ``TextConditioner`` refines them in the prompt's text capacity,
        with the padding rows masked out, and returns the prompt's rows; any
        other conditioning encoder refines the exact rows.
        """
        found = [
            call.module
            for (_, _, method), (_, call) in self._module_calls.items()
            if method == "encode"
            and self._encoder_kind(call.module) == "conditioning"
        ]
        count = features.shape[0]
        capacity = self._text_capacity(count)
        if not isinstance(found[0] if found else None, TextConditioner):
            return self.run_encoder("conditioning", features)
        padded = features.new_zeros((capacity, *features.shape[1:]))
        padded[:count].copy_(features)
        lengths = torch.full(
            (1,), count, dtype=torch.int32, device=features.device
        )
        result = self.run_encoder("conditioning", padded, lengths=lengths)
        return replace(
            result, values=tuple(value[:count] for value in result.values)
        )

    @torch.inference_mode()
    def prepare_text_capacities(self) -> None:
        """Prepare the text path at every text capacity before serving.

        A placeholder prompt filling each capacity, the largest first, runs
        through this rank's text encoder and conditioning encoder, which
        prepares their contexts and, while graphs may be captured, their
        graphs. Every admitted prompt is padded to one of these capacities,
        so serving finds its text path prepared. Collective across each
        encoder's ranks.
        """
        builder = self.media_builder
        kinds = self.encoder_kinds
        if builder is None or not kinds & {"text", "conditioning"}:
            return
        # The conditioning encoder refines the text encoder's features.
        text = capability(self.model, TextEncoder)
        # Each kind's component and module path: a component may expose
        # ``encode`` on several modules, as a conditioner does on its vision
        # tower.
        encoders = {
            self._encoder_kind(call.module): (name, call.path)
            for (name, _, method), (_, call) in self._module_calls.items()
            if method == "encode"
        }
        capacities = tuple(reversed(builder.text_capacities))
        dtype = getattr(torch, self.worker_config.model_dtype)

        def features(capacity):
            layout = text.output_layout(capacity, dtype)["conditioning"]
            return torch.zeros(
                layout.shape,
                dtype=layout.dtype,
                device=self.bindings[encoders["conditioning"][0]].device,
            )

        # Every capacity's context precedes the first capture into each
        # encoder's shared graph pool (``prepare_module``).
        for capacity in capacities:
            if "text" in kinds:
                name, path = encoders["text"]
                self.prepare_module(
                    name, TextSize(capacity, 1), method="encode", path=path
                )
            if "conditioning" in kinds and text is not None:
                name, path = encoders["conditioning"]
                self.prepare_module(
                    name,
                    (features(capacity).shape,),
                    method="encode",
                    path=path,
                )
        for capacity in capacities:
            if "text" in kinds:
                self.encode_text((0,) * capacity)
            if "conditioning" in kinds and text is not None:
                self.encode_conditioning(features(capacity))

    @torch.inference_mode()
    def run_module(
        self, name, *args, method=None, path=None, size=None, **kwargs
    ):
        """Resolve a bound capability and invoke its numerical runner."""
        binding, call = self._module_call(name, method, path)
        runner = self.prepare_module(
            name, size, method=call.entry_point.method, path=call.path
        )
        with profile_range(
            f"uniserve.model.module rank={self.worker_config.rank} work={name}"
        ):
            try:
                return runner.execute_model(*args, **kwargs)
            except CUDAGraphError:
                # Retire the failed context and its graphs; the next call at
                # this size prepares a fresh one.
                self._retire_module(
                    (
                        name,
                        call.path,
                        call.entry_point.method,
                        input_signature(size),
                    )
                )
                raise

    @property
    def denoises(self) -> bool:
        """Whether this rank advances a standalone denoiser's ladders."""
        return self._denoiser is not None

    def bind_diffusion_storage(
        self, bank: Mapping[str, torch.Tensor], pool: LatentPool
    ) -> None:
        """Borrow the storage the standalone denoiser's ladders gather.

        The request bank holds each request's state and the latent pool its
        samples. The runner borrows both when it is built, so they are bound
        before the first layout is prepared.
        """
        if self._diffusion is not None:
            raise RuntimeError(
                "bind request storage before preparing diffusion runners"
            )
        self.diffusion_bank = dict(bank)
        self.latent_pool = pool

    @property
    def diffusion(self) -> DiffusionRunner:
        """Return the standalone denoiser's runner, built on first use.

        One runner serves every layout of ``media_builder.layouts``: its
        context is prepared for the largest, whose storage the others share,
        and ``prepare_layouts`` prepares them. It captures when the graph
        policy and the component's stream allow.
        """
        if self._denoiser is None or self.latent_pool is None:
            raise InputError("rank does not own denoising computation")
        if self._diffusion is None:
            builder = cast("MediaBuilder", self.media_builder)
            name, binding, call = self._denoiser
            stream = self.module_stream(
                name, method=call.entry_point.method, path=call.path
            )
            captures = (
                self.worker_config.graph_policy != "off" and stream is not None
            )
            self._diffusion = DiffusionRunner.for_layouts(
                name,
                call,
                builder.maximum_layout,
                device=binding.device,
                stream=stream,
                storage=self.graph_storage,
                devices=(
                    binding.device,
                    *self._capture_devices(binding.device),
                )
                if captures
                else (),
                bank=self.diffusion_bank if captures else None,
                slots=self.worker_config.max_request_pool_size,
                pool=self.latent_pool,
                pages=builder.sample_pages.pages,
                attention=self.attention,
            )
        return self._diffusion

    def prepare_layouts(self) -> tuple:
        """Prepare the media builder's capacity layouts, largest first.

        The runner first prepares ``maximum_layout``, which bounds every
        layout's workspace, then every layout of ``layouts``. Collective
        across the denoiser's ranks, which prepare the same layouts in the
        same order. Returns ``layouts``, the layouts startup warms and
        captures.
        """
        builder = cast("MediaBuilder", self.media_builder)
        runner = self.diffusion
        layouts = builder.layouts()
        for layout in (builder.maximum_layout, *layouts):
            runner.prepare(layout, pages=builder.layout_pages(layout))
        self.graph_storage.check()
        return layouts

    def diffusion_layout(self, layout):
        """Return a prepared layout, preparing it on first use.

        A layout startup did not prepare, such as the own layout of a
        request with conditions, is prepared while serving and its steps run
        eagerly. At most ``max_request_pool_size`` such
        layouts stay prepared; reaching the bound retires the least recently
        used, which a later request prepares again. Collective across the
        denoiser's ranks, which run a request's calls in the same order.

        Raises:
            ValueError: The layout's workspace does not fit
                ``maximum_layout``'s, or its samples exceed the runner's.
        """
        builder = cast("MediaBuilder", self.media_builder)
        runner, serving = self.diffusion, self._serving_layouts
        if layout in serving:
            serving.move_to_end(layout)
            return runner.layout(layout)
        if layout in runner.layouts:
            return runner.layout(layout)
        if len(serving) >= self.worker_config.max_request_pool_size:
            retired, _ = serving.popitem(last=False)
            runner.retire(retired)
        entry = runner.prepare(layout, pages=builder.layout_pages(layout))
        serving[layout] = None
        return entry

    def run_denoising(self, ladder, index, bank):
        """Run one denoising step of a bound ladder and time it.

        ``bank`` holds the request's committed samples; see
        ``DiffusionRunner.step``.
        """
        runner = self.diffusion
        started = time.perf_counter_ns()
        with profile_range(
            f"uniserve.model.denoise rank={self.worker_config.rank} "
            "work=denoiser"
        ):
            values, path = runner.step(ladder, index, bank)
        return replace(
            ModelRunner.result(values),
            stats=_observations("denoiser", started, path),
        )

    def output_layout(
        self,
        entry,
        output_index,
        media,
        decode,
        num_prompt_tokens,
        conditions=None,
    ):
        """Resolve the published layout of one call output on this rank.

        ``media`` supplies the request's frame count (the builder's maximum
        when ``None``), ``decode`` the media units of a scheduled decode round
        (every unit when ``None``), ``num_prompt_tokens`` the prompt length,
        and ``conditions`` the request's ``VideoAdmission``, which sizes its
        condition products and places the denoiser's rows in the request's
        own layout. A distributed component's layout is narrowed to the units
        this rank publishes.

        Returns None for outputs this rank does not publish: non-output ranks,
        degenerate denoiser slices, and empty temporal-unit shares of a
        distributed component.
        """
        binding = self.bindings.get(entry)
        if (
            binding is not None
            and binding.process_group.global_rank not in binding.output_ranks
        ):
            return None

        declared = self.outputs.get(entry, ())
        if output_index < len(declared) and (
            declared[output_index].name in CONDITION_PRODUCTS
        ):
            if conditions is None:
                raise InputError("a condition product needs its conditions")
            return condition_layout(
                declared[output_index], conditions, decode, binding
            )

        frames = None if media is None else media.num_frames
        canvas = None if media is None else media.canvas
        results: tuple[tuple[nn.Module | None, str, OutputLayout], ...] = tuple(
            (call.module, name, layout)
            for call in self._declarations[entry]
            for name, layout in output_layouts(
                self.worker_config,
                call,
                builder=self.media_builder,
                clock=self.video_postprocessor,
                frames=frames,
                canvas=canvas,
                prompt_tokens=num_prompt_tokens,
                conditions=()
                if conditions is None
                else library_conditions(conditions),
            ).items()
        )
        if entry == VIDEO_CODEC_COMPONENT:
            # The video codec owns no numerical method; its product is the
            # encoded rows of the media units it is handed. A row carries its
            # own length, so every request uses the row that bounds the
            # largest unit at every admitted canvas, as declared.
            from uniserve_worker.model_executor.resources import (
                bounding_layout,
                encoded_units_layout,
            )

            sizes = self.media_builder.video_sizes()
            results = (
                (
                    None,
                    "encoded_units",
                    bounding_layout(
                        tuple(
                            encoded_units_layout(self.video_decoder, size)
                            for size in sizes
                        )
                    ),
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
        # The product holds only the round's units, so the rank's run is
        # counted from the round's first unit.
        run = binding.media_units(0, units)
        if not run:
            return None

        if audio:
            # An audio media unit is a span of the sample timeline, so this
            # rank publishes the samples of the units it reconstructs.
            spans = module.unit_samples(layout.shape[0], units)
            return replace(
                layout,
                local_slice=(
                    slice(spans[run.start].start, spans[run.stop - 1].stop),
                    *layout.local_slice[1:],
                ),
            )

        return replace(
            layout,
            shape=(units, *layout.shape[1:]),
            local_slice=(slice(run.start, run.stop), *layout.local_slice[1:]),
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
        table_widths,
        max_inflight,
    ):
        """Bind staged input resources and graph budgets for every capability.

        Creates one capability runner per (entry, path, lane) covering a
        computation kind, with its input buffers, execution context, decode /
        prefill capture shapes, and private CUDA graph storage pools. Callable
        once: a second call raises ``RuntimeError`` when the first bound any
        entry. Raises ``ValueError`` when two staged entries of one component
        cover the same computation kind, and as ``_initialize_streams`` does.
        """
        from uniserve_worker.config.execution import (
            DEFAULT_PREFILL_GRAPH_ROW_BUCKETS,
        )

        if self.entries:
            raise RuntimeError("input execution resources are already bound")

        self.kv_cache, self.decode_predicates = kv_cache, decode_predicates
        self.table_widths = tuple(table_widths)
        self.prefill_row_sizes = DEFAULT_PREFILL_GRAPH_ROW_BUCKETS
        config = self.worker_config
        self._initialize_streams(event_slots=max_inflight + 1)

        # A decode row holds at least one page of every cache group, and one
        # prefill row at most the tokens the pool's units cover in every
        # group.
        max_rows = min(max_calls, request_slots)
        decode_sizes = tuple(
            value
            for value in config.decode_graph_batch_sizes
            if 0 < value <= max_rows
            and value * kv_cache.row_units < kv_cache.info.num_units
        )
        prefill_capacity = min(
            max_tokens,
            config.max_sequence_tokens,
            kv_cache.token_capacity,
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

            # An entry covering latent encoding or image decoding is staged on
            # the generation device, or the worker device when none is
            # configured; any other entry on its component's placement device.
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
            # One runner per stream on the target device whose lane covers
            # some of the entry's kinds (a full-device stream covers all);
            # with no stream there, as on a CPU device, one runner without.
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
                        # Staging supplies every host sequence length and
                        # start page, so attention planning never copies
                        # them from the device while serving.
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
                            derive_host_lengths=False,
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
                entry.table_widths = self.table_widths

                for kind in kinds:
                    key = (name, kind)
                    if key in self._forward_calls:
                        raise ValueError(
                            f"computation {key} has multiple lane bindings"
                        )
                    self._forward_calls[key] = entry

    def call_devices(self, call):
        """Return ``(compute, staged, output)`` devices for one call.

        The compute and output device is the component's placement device, or
        the worker device for an unbound component. With an image builder,
        latent preparation, denoising and image decoding compute and output on
        the generation device, or the worker device when none is configured. The
        staged device is the bound execution entry's device, or the compute
        device when the call has no staged binding.
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
                prepare_denoising,
                warmup_decoders,
                warmup_postprocess,
            )

            prepare_denoising(self, storage)
            self.prepare_text_capacities()
            warmup_decoders(self)
            warmup_postprocess(self, storage)
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
        """Seal startup: check captured-graph storage budgets and stream grants.

        Afterwards no runner captures: the staged entries capture no
        further batch graphs, module entries, including ones prepared later,
        evaluate signatures without a resident graph eagerly, and the
        denoiser runner refuses capture (see ``ModelRunner``), so the graph
        storage is sealed (``GraphStorage.seal``). Raises ``CUDAGraphError``
        when graph residency exceeds its budget, and ``CUDAError`` when a
        lane stream has lost its SM partition.
        """
        self.graph_storage.check()
        # Resident storage by device and, for graph pools, by runner kind,
        # for sizing. Scratch and eager warm calls allocate outside the pools,
        # and graph executables, communicators and loaded modules outside the
        # caching allocator, so the process's whole device footprint is
        # reported alongside the allocator's reservation and the pools.
        for device, pooled in sorted(
            self.graph_storage.pool_bytes().items(), key=str
        ):
            logger.info(
                "device storage on %s: %.2f GiB held by this process at "
                "startup, %.2f GiB of it reserved by the caching allocator "
                "and %.2f GiB of that in graph pools",
                device,
                process_device_bytes(device) / 2**30,
                torch.cuda.memory_reserved(device) / 2**30,
                pooled / 2**30,
            )
        totals: dict = {}
        for (owner, device), used in self.graph_storage.owner_bytes().items():
            method = getattr(getattr(owner, "call", None), "entry_point", None)
            label = f"{getattr(owner, 'name', type(owner).__name__)}" + (
                f".{method.method}" if method is not None else ""
            )
            totals[device, label] = totals.get((device, label), 0) + used
        for (device, label), used in sorted(
            totals.items(), key=lambda item: (str(item[0][0]), item[0][1])
        ):
            logger.info(
                "graph storage on %s: %s holds %.2f GiB at startup",
                device,
                label,
                used / 2**30,
            )

        for _, stream in self._lane_streams:
            stream.verify()
        self.graph_storage.seal()
        self._startup_complete = True
        for entry in (
            *self.entries.values(),
            *self._module_entries.values(),
            *(() if self._diffusion is None else (self._diffusion,)),
        ):
            entry._startup_complete = True

    def synchronize(self):
        """Host-synchronize every owned stream and each device's current stream.

        The devices are the worker device and the devices graphs may allocate
        on. Every stream is attempted even if one fails; the first failure is
        raised.
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
        """Destroy every runner's captured graphs, keeping their contexts.

        The caller drains all borrowed output readers first. A non-aborted
        worker close calls this before releasing the KV cache, latent pool and
        product backing the graphs reference.
        """
        actions = [
            entry.close_graphs
            for entry in (
                *self.entries.values(),
                *self._module_entries.values(),
                *(() if self._diffusion is None else (self._diffusion,)),
            )
        ]
        close_resources(*actions)

    def close(self, *, aborted: bool = False):
        """Release every owned context, buffer, graph and stream exactly once.

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
        # A constructor failure closes before ``noise_draws`` is assigned.
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
        actions.extend(
            scratch.close for scratch in self._module_scratch.values()
        )
        if self._diffusion is not None:
            actions.append(self._diffusion.close)
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
            self._module_scratch.clear()
            self._diffusion = None
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
            name, TextSize(len(tokens), 1), method="encode", path=call.path
        )
        # The token buffer holds the longest prompt at its text capacity.
        return runner.prepare_tokens(
            tokens,
            capacity=self._text_capacity(
                self.worker_config.max_sequence_tokens
            ),
        )

    @contextmanager
    def preparing_inputs(
        self, transfers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    ) -> Iterator[None]:
        """Overlap reserved input copies with computation, then join order.

        Each ``(destination, source)`` pair is copied on the executor's
        preparation stream while the body runs on the current stream. The
        copies are not ordered after work already queued on the current
        stream, so sources must be ready and destinations free on entry.
        Sources and destinations must remain reserved through this call's
        physical completion. The final stream wait also covers failed compute,
        so completion and request retirement cannot overtake the input copies.
        Raises ``RuntimeError`` without a live CUDA executor and ``ValueError``
        when a pair differs in shape or dtype.
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
        """Execute rows as homogeneous numerical calls, yielding by index.

        Yields ``(indexes, outcome)`` pairs, where ``indexes`` are positions in
        ``tasks`` and ``outcome`` is the group's ``ExecutionOutput`` or the
        error that failed it. A row with no staged entry yields its own
        ``invalid_descriptor`` error before any group runs; groups then run in
        the order their first row appears.

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
        """Stage one homogeneous group of rows, run it, and validate outputs.

        The staged entry chooses eager or CUDA graph execution. Returns the
        output with its request slot indices, its lane output fence, and
        statistics for this one call. Raises ``RuntimeError`` once closed,
        ``ValueError`` for an empty or mixed-kind group, ``InputError`` when
        the group has no staged entry or staging fails, and the error
        ``_execution_failure`` classifies when execution or validation fails.
        """
        if self._closed:
            raise RuntimeError("model runner is closed")
        tasks = rows
        if not tasks:
            raise ValueError("model runner received an empty call")

        # Only token forward modes and denoising are eligible for CUDA graphs;
        # other staged kinds run eagerly.
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
                # Join staging copies already submitted to the lane before
                # reporting, as for execution failures below.
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
                # The mode's token counter counts the query tokens the group
                # computed; encoder and decoder groups have none and report
                # only their call and time.
                tokens = batch.query_tokens
                stats = replace(
                    output.stats,
                    mode_counts={forward_mode.value: 1},
                    mode_tokens={}
                    if tokens is None
                    else {forward_mode.value: tokens},
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
    """Classify invalid model inputs with their phase and call identities.

    An ``InputError`` is returned unchanged.
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
    """Classify a model failure and attach the active phase and call identities.

    A ``WorkerError`` is returned unchanged. A CUDA graph failure, or an error
    ``classify`` maps to a resource or fatal worker failure, becomes a
    ``ResourceError`` that keeps the classified fatality; anything else becomes
    a ``ComputeError``.
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
