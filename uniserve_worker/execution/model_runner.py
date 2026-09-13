"""Direct staged execution of concrete model phases."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterator, Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import torch

from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.batch import ExecutionOutput, InputBatch
from uniserve_worker.execution.graph_inputs import (
    DiffusionShape,
    GraphExecutionError,
)
from uniserve_worker.foundation.errors import (
    ComputeError,
    InputError,
    ResourceError,
    WorkerError,
    WorkerErrorCode,
    classify,
    invalid_descriptor,
)
from uniserve_worker.foundation.resources import close_resources
from uniserve_worker.modeling.batch import DecodeBatch, DiffusionBatch, EncodeBatch
from uniserve_worker.modeling.components import Call
from uniserve_worker.modeling.decoder import DecodeKind, DecoderMixin
from uniserve_worker.modeling.diffusion import DiffusionMixin
from uniserve_worker.modeling.encoder import EncodeKind, EncoderMixin
from uniserve_worker.modeling.geometry import MediaShape, TensorOutputLayout, TextShape
from uniserve_worker.modeling.image_diffusion import ImageDiffusion
from uniserve_worker.modeling.inputs import ImageProcessor, PatchTransform
from uniserve_worker.modeling.model import Model
from uniserve_worker.modeling.tensors import TensorViews, TokenSelection
from uniserve_worker.modeling.text import TextMixin
from uniserve_worker.nn.attention_storage import attention_exchange_scope
from uniserve_worker.nn.collective import collective_scope
from uniserve_worker.nn.diffusion.schedule import DiffusionSchedule
from uniserve_worker.nn.vae.decoder import LatentDecoder, decoder_scope
from uniserve_worker.protocol.batch import (
    COMPUTATIONS,
    Computation,
    ComputationId,
    DecodeRange,
    DiffusionSamplingParams,
    ForwardMode,
    ForwardStats,
    ImageParams,
    PipelineStage,
    ScheduledRequest,
    StaticDim,
    TensorSpec,
)
from uniserve_worker.runtime.device import canonical_device, fill_cpu_ints
from uniserve_worker.runtime.results import resolve_outputs
from uniserve_worker.runtime.staging_buffers import StagingBuffers
from uniserve_worker.runtime.tensor_store import device_product_storage
from uniserve_worker.runtime.tensors import TensorResources, media_calls, resolve_resources

from ..backends.attention.context import attention_scope
from ..bootstrap.components import call_operations, supported_operations
from ..nn.mesh import Communicator, DeviceMesh
from ..nn.parallel import ComponentConfig
from ..profiling import record_component
from ..runtime.tensor_buffers import TensorBuffers
from ..transfer.layout import TensorRegion
from .cuda_graph import CudaGraph, capture_pools
from .diffusion_runner import DiffusionRunner
from .input_buffers import InputBuffers, InputGeometry
from .model_entry import ModelEntry, TensorOutput, capture_required, tensor_signature

if TYPE_CHECKING:
    from uniserve_worker.backends.attention.selection import AttentionSelection

    from ..runtime.block_tables import BlockTables
    from ..runtime.decode_state import DecodeState
    from ..runtime.kv_cache import KVCache
from uniserve_worker.modeling.tensors import AttentionMode

from ..config import LaneConfig
from ..nn.collective import stream_collective_scope
from ..nn.sparse_attention import SparseAttention
from .cuda_stream import CudaStream, create_partitioned_streams
from .graph_inputs import (
    _GRAPH_BINDINGS,
    PrefillShape,
    _attention_inputs,
    _batch_tensors,
    _copy_tensors,
    _cuda_batch,
    _decode_geometry,
    _decode_signature,
    _exact_signature,
    _graph_batch,
    _graph_provider,
    _GraphMiss,
    _greedy_decode,
    _live_attention,
    _normalize_exact_batch,
    _pad_decode_batch,
    _pad_prefill_batch,
    _prefill_geometry,
    _prefill_signature,
    _private_pool_bytes,
    _release_call,
    _trim_greedy,
    _trim_output,
)
from .rows import ForwardRow
from .sampling import SamplerOutput

logger = logging.getLogger(__name__)


def _validate_products(
    name: str, schemas: tuple[TensorSpec, ...], values: tuple[torch.Tensor, ...]
) -> None:
    """Check numerical results against their loaded product geometry."""

    if len(values) != len(schemas):
        raise ComputeError(f"entry {name!r} returned the wrong number of tensor results")
    for value, schema in zip(values, schemas, strict=True):
        dtype, _ = device_product_storage(schema.dtype)
        if str(value.dtype).removeprefix("torch.") != dtype:
            raise ComputeError(f"entry {name!r} changed tensor result {schema.name!r} dtype")
        dims = schema.shape_bound.dims
        if value.ndim != len(dims) or any(
            extent != dim.extent if isinstance(dim, StaticDim) else not 0 < extent <= dim.bound
            for extent, dim in zip(value.shape, dims, strict=True)
        ):
            raise ComputeError(f"entry {name!r} exceeded tensor result {schema.name!r} shape")


def _invoke(
    ids: torch.Tensor,
    positions: torch.Tensor,
    batch: InputBatch,
    *,
    text: TextMixin | None,
    diffusion_model: DiffusionMixin | None,
    encoder: EncoderMixin | None,
    decoder: DecoderMixin | None,
    pipeline: Communicator | None,
) -> ExecutionOutput:
    """Invoke one homogeneous numerical capability and wrap its result for execution."""

    from ..modeling.batch import DecodeBatch, DiffusionBatch, EncodeBatch, TextBatch
    from ..modeling.geometry import MediaShape

    if isinstance(batch.forward_mode, ForwardMode):
        if text is None:
            raise InputError("text execution requires the text computational capability")
        numerical = TextBatch(
            ids,
            positions,
            batch.attention,
            batch.token_selections,
            batch.input_embeddings,
            batch.embedding_mask,
        )
        hidden_needs = text.tensor_specs(Call.TEXT, TextShape(ids.numel(), numerical.row_count))
        hidden = text.forward(numerical, constants={}, scratch={})
        hidden_needs.outputs["hidden_states"].validate(hidden, state={}, scratch={})
        text_result = text.compute_logits(hidden, numerical)
        text_result.validate(
            tuple(
                text.tensor_specs(Call.TEXT, TextShape(count, selection=selection))
                for count, selection in zip(
                    numerical.attention.query_lens_cpu, numerical.selections, strict=True
                )
            )
        )
        return ExecutionOutput(text_result.values, text_result.vocabularies)
    if batch.forward_mode is PipelineStage.DENOISING:
        if diffusion_model is None:
            raise InputError("diffusion execution requires the diffusion computational capability")
        diffusion = DiffusionBatch(
            latents={"image": batch.flow_latents},
            shapes=tuple(
                MediaShape(height, width)
                for height, width in zip(batch.flow_heights, batch.flow_widths, strict=True)
            ),
            timesteps={"image": batch.flow_timesteps},
            positions=batch.flow_positions,
            conditioning={"image": batch.flow_conditioning},
            sequence_lengths=batch.flow_image_tokens,
            attention=batch.attention,
        )
        requirements = tuple(
            diffusion_model.tensor_specs(Call.DIFFUSION, shape) for shape in diffusion.shapes
        )
        result = diffusion_model.forward_diffusion(diffusion, state={}, constants={}, scratch={})
        result.validate(requirements, state={}, scratch={})
        values = list(result.values["image"])
        if pipeline is not None and pipeline.world_size > 1:
            # All mathematical PP participants retain solver state. Only the
            # final stage evaluates the head; execution distributes its result.
            for index, value in enumerate(values):
                if value is None:
                    value = torch.empty_like(
                        diffusion.latents["image"][index],
                        dtype=requirements[index].outputs["image"].dtype,
                    )
                    values[index] = value
                pipeline.broadcast(value, src=pipeline.world_size - 1)
        if any(value is None for value in values):
            raise RuntimeError("diffusion result is missing its output-stage value")
        return ExecutionOutput(cast(tuple[torch.Tensor, ...], tuple(values)))
    if batch.forward_mode in {PipelineStage.VISION_ENCODING, PipelineStage.LATENT_ENCODING}:
        if encoder is None:
            raise InputError("encoding requires the encoder computational capability")
        kind: EncodeKind = (
            "vision" if batch.forward_mode is PipelineStage.VISION_ENCODING else "latent"
        )
        encoding = EncodeBatch(batch.encode_pixels, batch.encode_grids, batch.encode_grid_shapes)
        processor = encoder.image_processor
        transform = None if processor is None else processor.vit
        shapes = []
        for index, pixels in enumerate(encoding.values):
            if kind == "vision" and isinstance(transform, PatchTransform):
                grid = encoding.grid_shapes[index] if encoding.grid_shapes else None
                if grid is None:
                    raise InputError("patch encoding requires host-known grid geometry")
                height, width = (extent * transform.patch_size for extent in grid)
            else:
                height, width = pixels.shape[-2:]
            shapes.append(MediaShape(height, width, dtype=pixels.dtype))
        call = Call.ENCODE_VISION if kind == "vision" else Call.ENCODE_LATENT
        requirements = tuple(encoder.tensor_specs(call, shape) for shape in shapes)
        result = encoder.encode(
            kind,
            encoding,
            constants={},
            scratch={},
        )
        result.validate(requirements, state={}, scratch={})
        name = "features" if kind == "vision" else "latents"
    elif batch.forward_mode is PipelineStage.IMAGE_DECODING:
        if decoder is None:
            raise InputError("decoding requires the decoder computational capability")
        decoding = DecodeBatch(
            batch.decode_latents,
            tuple(
                MediaShape(height, width, dtype=latent.dtype)
                for height, width, latent in zip(
                    batch.decode_heights, batch.decode_widths, batch.decode_latents, strict=True
                )
            ),
        )
        result = decoder.decode(
            "image",
            decoding,
            constants={},
            scratch={},
        )
        result.validate(
            tuple(decoder.tensor_specs(Call.DECODE_IMAGE, shape) for shape in decoding.shapes),
            state={},
            scratch={},
        )
        name = "image"
    else:
        raise TypeError(f"unsupported model phase {batch.forward_mode.value!r}")
    rows = result.values[name]
    if any(value is None for value in rows):
        raise RuntimeError("numerical result is missing its output-stage value")
    return ExecutionOutput(cast(tuple[torch.Tensor, ...], rows), layouts=result.layouts[name])


def _encode(
    model: EncoderMixin, kind: EncodeKind, *values: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Invoke a tensor-only encoder through its standard numerical batch."""

    output = model.encode(kind, EncodeBatch(values), constants={}, scratch={})
    rows = output.values["conditioning"]
    if any(value is None for value in rows):
        raise RuntimeError("the bound encoder partition did not produce its conditioning")
    return cast(tuple[torch.Tensor, ...], rows)


def capture_image_parameters(
    cfg_branches: int,
    *,
    steps: int,
    height: int,
    width: int,
) -> ImageParams:
    """Build deterministic bounded image-generation parameters for startup warmup."""

    scales = {
        1: (1.0, 1.0),
        2: (4.0, 1.0),
        3: (4.0, 2.0),
    }
    try:
        text_scale, image_scale = scales[int(cfg_branches)]
    except KeyError as error:
        raise invalid_descriptor(
            "flow CFG branch geometry exceeds the concrete branch set"
        ) from error
    return ImageParams(
        steps=int(steps),
        cfg_text_scale=text_scale,
        cfg_img_scale=image_scale,
        height=int(height),
        width=int(width),
        seed=0,
    )


class ModelRunner:
    """Own model bindings, numerical input storage, and CUDA graph execution."""

    def __init__(
        self,
        model: Model,
        worker_config: WorkerConfig,
        *,
        bindings: Mapping[str, ModelEntry] | None = None,
        attention: AttentionSelection | None = None,
        schedule: DiffusionSchedule | None = None,
    ) -> None:
        """Bind loaded compute modules to one public execution resource owner."""

        from ..backends.attention import resolve_attention_selection
        from ..nn.attention import RadixAttention, bind_dense_attention_modules

        self.attention = attention or resolve_attention_selection(
            worker_config.attention_backend or "auto",
            tuning=worker_config.flashinfer,
            block_size=worker_config.block_size,
        )
        bind_dense_attention_modules(model, self.attention)
        self.model = model
        self.output_schema = resolve_outputs(model)
        self.bindings = MappingProxyType(dict(bindings or {}))
        self.schedule = schedule
        self.worker_config = worker_config
        self.uses_lanes = False
        self.flow_captures: tuple[DiffusionShape, ...] = ()
        self.flow_cfg_branches: tuple[int, ...] = ()
        self._startup_complete = False
        self._forward_entries: dict[tuple[str, Computation], ModelEntry] = {}
        self.batch_graphs: dict[
            ModelEntry, dict[Hashable, CudaGraph[tuple[ExecutionOutput, SamplerOutput | None]]]
        ] = {}
        self.graph_streams: dict[ModelEntry, torch.cuda.Stream] = {}
        self.graph_pools: dict[ModelEntry, Any] = {}
        self.graph_device_pools: dict[ModelEntry, dict[torch.device, torch.cuda.MemPool]] = {}
        self.graph_memory_budgets: dict[torch.device, int] = {}
        self.decode_shapes: dict[ModelEntry, tuple[int, ...]] = {}
        self.prefill_shapes: dict[ModelEntry, tuple[PrefillShape, ...]] = {}
        self.prefill_row_sizes: tuple[int, ...] = ()
        self.decode_context_blocks = 0
        self.decode_predicates: torch.Tensor | None = None
        self.kv_cache: KVCache
        self._streams: list[CudaStream] = []
        self.entries: dict[tuple[str, str, str], ModelEntry] = {}
        self.diffusion: DiffusionRunner | None = None
        self._capture_stream: torch.cuda.Stream | None = None
        self._text_staging: StagingBuffers | None = None
        self._text_tokens: torch.Tensor | None = None
        self._encoders: dict[EncodeKind, str] = {}
        self._decoders: dict[DecodeKind, tuple[str, LatentDecoder]] = {}
        self._preparation_stream: torch.cuda.Stream | None = None
        self._sum_reductions = {}
        self._attention_exchange_storage = {}
        self.scratch: TensorBuffers | None = None
        self.tensor_resources = TensorResources({}, {})
        self._closed = False
        try:
            from ..runtime.collectives import allocate_peer_reductions

            meshes = (entry.mesh for entry in self.bindings.values() if entry.mesh is not None)
            self._sum_reductions = allocate_peer_reductions(mesh.get_group("tp") for mesh in meshes)
            exchange_modules = tuple(
                module
                for module in model.modules()
                if isinstance(module, RadixAttention)
                and module.exchange.ulysses_group.world_size > 1
            )
            if exchange_modules:
                from ..bootstrap.capacity import packed_input_geometry
                from ..runtime.attention_storage import allocate_attention_exchange_storage

                geometry = packed_input_geometry(model, worker_config)
                dtype = getattr(torch, worker_config.model_dtype.removeprefix("torch."))
                for lane in worker_config.lanes or (None,):
                    lane_id = None if lane is None else lane.lane_id
                    self._attention_exchange_storage[lane_id] = allocate_attention_exchange_storage(
                        exchange_modules,
                        max_tokens=geometry.max_tokens,
                        dtype=dtype,
                    )

            self._bind_tensors()
            self._bind_encoders()
            self._bind_decoders()
            self._bind_video()
            if self.diffusion is None and model.generation is not None:
                self.diffusion = DiffusionRunner(
                    generation=model.generation,
                    device=canonical_device(
                        worker_config.generation_device or worker_config.device
                    ),
                    capture_stream=None,
                    groups=(),
                    capacity=worker_config.max_request_pool_size,
                )
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note(f"Resource cleanup also failed: {cleanup_error!r}")
            raise

    def _bind_tensors(self) -> None:
        """Resolve local request state and allocate the caller's shared scratch."""

        self.tensor_resources = resolve_resources(
            self.model, media_calls(self.model, self.bindings)
        )
        if self.tensor_resources.scratch:
            self.scratch = TensorBuffers.allocate(
                self.tensor_resources.scratch, self.worker_config.device
            )

    def capture_stream(self) -> torch.cuda.Stream | None:
        """Borrow the module capture stream when CUDA graph execution is enabled."""

        device = canonical_device(self.worker_config.device)
        if self.worker_config.graph_policy == "off" or device.type != "cuda":
            return None
        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream(device=device)
        return self._capture_stream

    def _bind_encoders(self) -> None:
        """Bind encoder participation from the numerical kind and component mesh."""

        model = self.model
        if not isinstance(model, EncoderMixin):
            return
        kinds: dict[Call, EncodeKind] = {
            Call.ENCODE_TEXT: "text",
            Call.ENCODE_CONDITIONING: "conditioning",
        }
        for component, binding in self.bindings.items():
            if binding.mesh is None:
                continue
            mesh = binding.mesh
            for call in binding.local_calls:
                kind = kinds.get(call.call)
                if kind is None:
                    continue
                if kind not in model.encoder_kinds:
                    raise ValueError(
                        f"component {component!r} declares unsupported {kind} encoding"
                    )
                if kind in self._encoders:
                    raise ValueError(f"{kind} encoding has multiple local component bindings")
                name = "text_encoder" if kind == "text" else "conditioner"
                groups = tuple(mesh.get_group(axis) for axis in call.groups if mesh.size(axis) > 1)
                entry = self.bind_module(
                    name,
                    partial(_encode, model, kind),
                    groups=groups,
                    placement=component,
                )
                if kind == "text":
                    entry.output_schema = binding.output_schema
                self._encoders[kind] = name
                if kind == "text":
                    self._text_staging = StagingBuffers(
                        model.text_max_tokens, dtype=torch.int64, depth=2, device=binding.device
                    )
                    self._text_tokens = torch.empty(
                        model.text_max_tokens, dtype=torch.int64, device=binding.device
                    )

    def _bind_decoders(self) -> None:
        """Bind declared decoder capabilities independently of model classification."""

        model = self.model
        kinds: dict[Call, DecodeKind] = {
            Call.DECODE_IMAGE: "image",
            Call.DECODE_VIDEO: "video",
            Call.DECODE_AUDIO: "audio",
        }
        for component, binding in self.bindings.items():
            for call in binding.local_calls:
                kind = kinds.get(call.call)
                if kind is None:
                    continue
                if not isinstance(model, DecoderMixin) or kind not in model.decoder_kinds:
                    raise ValueError(
                        f"component {component!r} declares unsupported {kind} decoding"
                    )
                mesh = binding.mesh
                if mesh is None or kind == "image":
                    continue
                module = model.video_decoder if kind == "video" else model.audio_decoder
                if not isinstance(module, LatentDecoder):
                    raise ValueError(
                        f"resident component {component!r} has no native {kind} decoder"
                    )
                if kind in self._decoders:
                    raise ValueError(f"{kind} decoding has multiple local component bindings")
                groups = tuple(mesh.get_group(axis) for axis in call.groups if mesh.size(axis) > 1)
                fixed = None
                if all(extent is not None for extent in module.latent_shape):
                    # A fully specified native shape supports one fixed capture,
                    # independent of the enclosing request's window or duration.
                    fixed = (
                        torch.zeros(
                            cast(tuple[int, ...], module.latent_shape),
                            dtype=torch.float32,
                            device=binding.device,
                        ),
                    )
                self.bind_module(component, module, inputs=fixed, groups=groups)
                self._decoders[kind] = (component, module)

    def _bind_video(self) -> None:
        """Bind resident media computations using declared calls and physical entries."""

        from ..modeling.video import VideoMixin

        model = self.model
        if not isinstance(model, VideoMixin) or not isinstance(model, DiffusionMixin):
            return
        for component, binding in self.bindings.items():
            mesh = binding.mesh
            if mesh is None:
                continue
            for call in binding.local_calls:
                groups = tuple(mesh.get_group(axis) for axis in call.groups if mesh.size(axis) > 1)
                if call.call is Call.DIFFUSION:
                    if self.diffusion is not None:
                        raise ValueError("video diffusion requires one resident denoiser binding")
                    from ..modeling.geometry import MediaShape
                    from ..nn.parallel_attention import AttentionBuffers, ParallelAttention
                    from ..nn.sparse_attention import TILE
                    from ..runtime.attention_storage import (
                        allocate_context_storage,
                        allocate_output_storage,
                    )

                    layers = tuple(
                        layer for layer in model.modules() if isinstance(layer, ParallelAttention)
                    )
                    contexts: Mapping[ParallelAttention, AttentionBuffers] = {}
                    output_storage = None
                    if layers:
                        maximum = model.output_capacity
                        needs = model.tensor_specs(
                            Call.DIFFUSION,
                            MediaShape(
                                maximum.height,
                                maximum.width,
                                frames=maximum.frame_count,
                                prompt_tokens=model.text_max_tokens,
                            ),
                        )
                        output = needs.scratch["attention_output"]
                        rows, heads, width = output.shape
                        contexts = allocate_context_storage(
                            layers,
                            rows=rows,
                            heads=heads,
                            head_dim=width,
                            dtype=output.dtype,
                            block_size=TILE,
                        )
                        output_storage = allocate_output_storage(
                            layers,
                            rows=rows,
                            heads=heads,
                            head_dim=width,
                            dtype=output.dtype,
                        )
                    self.diffusion = DiffusionRunner(
                        model,
                        device=binding.device,
                        additional_devices=self._capture_devices(binding.device),
                        capture_stream=self.capture_stream(),
                        groups=groups,
                        capacity=self.worker_config.max_request_pool_size,
                        context_buffers=contexts,
                        output_storage=output_storage,
                        sparse_layers=tuple(
                            layer for layer in model.modules() if isinstance(layer, SparseAttention)
                        ),
                    )

    def bind_module(
        self,
        name: str,
        forward: Callable[..., TensorOutput],
        *,
        inputs: tuple[torch.Tensor, ...] | None = None,
        groups: tuple[Communicator, ...] = (),
        placement: str | None = None,
        outputs: tuple[TensorSpec, ...] | None = None,
    ) -> ModelEntry:
        """Bind a local computation and optional caller-owned publication contract.

        Explicit output descriptors serve standalone Python computations. Model
        capabilities use the output requirements resolved during construction.
        """

        device = canonical_device(self.worker_config.device)
        key = (name, str(device), "default")
        if key in self.entries:
            raise ValueError(f"numerical entry {name!r} is already bound")
        module = self._entry(placement or name, device)
        if placement is not None:
            module = replace(module, name=name)
        module.forward = forward
        module.fixed_inputs = inputs
        module.output_schema = self.output_schema.get(name, ()) if outputs is None else outputs
        # Groups describe numerical participation, including rank-local calls
        # with no collectives, independently of the enclosing component mesh.
        module.groups = groups
        self.entries[key] = module
        return module

    def _entry(self, name: str, device: torch.device) -> ModelEntry:
        """Resolve checkpoint placement, or a standalone local model binding."""

        if self.bindings:
            entry = self.bindings.get(name)
            if entry is None or not entry.owns:
                raise ValueError(f"rank does not own computation entry {name!r}")
            if entry.device == device:
                return entry
            # A model may place generation submodules on another local device.
            # The physical binding shares rank geometry, not mutable GPU storage.
            return ModelEntry(
                name, entry.config, entry.process_group, entry.mesh, device, calls=entry.calls
            )
        rank = self.worker_config.rank
        group = Communicator((rank,), rank, device=device)
        config = ComponentConfig((rank,))
        mesh = DeviceMesh((rank,), rank, config.parallel_config, device)
        return ModelEntry(name, config, group, mesh, device)

    def warmup(self, storage: tuple[TensorBuffers, ...]) -> None:
        """Prepare model/provider geometry through its numerical execution owners."""

        started = time.perf_counter()
        logger.info("starting numerical eager warmup device=%s", self.worker_config.device)
        with torch.inference_mode(), collective_scope(self._sum_reductions):
            from ..modeling.video import VideoMixin
            from .video import warmup_denoising, warmup_postprocess

            if isinstance(self.model, VideoMixin):
                warmup_denoising(self.model, self, storage)
                shape = self.model.output_capacity
                geometry = MediaShape(shape.height, shape.width, frames=shape.frame_count)
                for component, binding in self.bindings.items():
                    if binding.owns and any(
                        call.call is Call.DECODE_AUDIO for call in binding.calls
                    ):
                        native = self.model.tensor_specs(Call.DECODE_AUDIO, geometry).scratch[
                            "audio_latents"
                        ]
                        inputs = torch.zeros(
                            native.shape, dtype=native.dtype, device=binding.device
                        )
                        self.warmup_module(component, inputs)
                warmup_postprocess(self.model, self, storage)
        self.synchronize()
        logger.info("completed numerical eager warmup seconds=%.3f", time.perf_counter() - started)

    def batch_forward(self, entry: ModelEntry, batch: InputBatch) -> ExecutionOutput:
        """Use the same numerical and communication binding for startup and serving."""

        lane = entry.cuda_stream
        inputs = entry.input_buffers
        assert inputs is not None
        ids = inputs.input_ids[:0] if batch.input_ids is None else batch.input_ids
        positions = inputs.positions[0, :0] if batch.positions is None else batch.positions
        reductions = self._sum_reductions if lane is None or lane.full_device else {}
        storage = self._attention_exchange_storage.get(None if lane is None else lane.name)
        exchanges = {} if storage is None else storage.views
        with (
            collective_scope(reductions),
            attention_exchange_scope(exchanges),
            attention_scope(batch.binding, capture=batch.cuda_graph_capture),
        ):
            assert entry.forward is not None
            output = entry.forward(ids, positions, batch)
            if not isinstance(output, ExecutionOutput):
                raise TypeError("batch entry must return ExecutionOutput")
            return output

    @torch.inference_mode()
    def capture(self, *, tokenizer, latents) -> None:
        """Prepare configured entry inputs in dependency order before runtime warmup."""

        from functools import partial

        from .graph_inputs import PrefillShape
        from .runners.decode import prepare_decode
        from .runners.prefill import prepare_prefill

        processor = self.model.image_processor
        transform = None if processor is None else processor.vit
        patch_size = int(transform.patch_size) if isinstance(transform, PatchTransform) else None

        for phase in ("prefill", "decode", "flow"):
            started = time.perf_counter()
            captures_before = sum(len(graphs) for graphs in self.batch_graphs.values())
            logger.info("starting entry preparation phase=%s", phase)
            for entry in self.batch_graphs:
                lane = entry.cuda_stream
                assert entry.input_buffers is not None
                if entry.device != canonical_device(self.worker_config.device):
                    continue
                context = nullcontext() if lane is None else torch.cuda.stream(lane.stream)
                if lane is not None:
                    lane.wait(torch.cuda.current_stream(lane.device))
                with context:
                    forward = partial(self.batch_forward, entry)
                    if phase == "prefill" and ForwardMode.PREFILL in entry.computations:
                        shapes = (
                            self.prefill_shapes[entry]
                            if (self.worker_config.graph_policy != "off")
                            and self.worker_config.prefill_cuda_graph
                            else (PrefillShape(1, 1, 1),)
                        )
                        prepare_prefill(
                            self,
                            entry,
                            entry.input_buffers,
                            forward,
                            shapes,
                            packed=self.model.text_attention_mode is AttentionMode.PACKED,
                        )
                    elif phase == "decode" and ForwardMode.DECODE in entry.computations:
                        prepare_decode(
                            self,
                            entry,
                            entry.input_buffers,
                            forward,
                            packed=self.model.text_attention_mode is AttentionMode.PACKED,
                        )
                    elif (
                        phase == "flow"
                        and PipelineStage.DENOISING in entry.computations
                        and self.model.generation is not None
                        and latents is not None
                    ):
                        assert self.diffusion is not None
                        if (
                            (self.worker_config.graph_policy != "off")
                            and self.worker_config.prefill_cuda_graph
                            and self.flow_captures
                        ):
                            self.diffusion.prepare_flow(
                                self,
                                entry,
                                latents,
                                tokenizer,
                                patch_size,
                                self.flow_captures,
                                capture=True,
                            )
                        else:
                            import math

                            generation = self.model.generation
                            capacity = min(
                                generation.max_latent_tokens,
                                generation.max_vae_grid_tokens,
                                latents.capacity_units,
                            )
                            side = max(1, math.isqrt(capacity)) * generation.latent_downsample
                            representative = tuple(
                                next(
                                    (
                                        shape
                                        for shape in reversed(self.flow_captures)
                                        if shape.cfg_branches == branches
                                    ),
                                    DiffusionShape(1, side, side, branches),
                                )
                                for branches in self.flow_cfg_branches
                            )
                            self.diffusion.prepare_flow(
                                self,
                                entry,
                                latents,
                                tokenizer,
                                patch_size,
                                representative,
                                capture=False,
                            )
            logger.info(
                "completed entry preparation phase=%s captured_shapes=%d seconds=%.3f",
                phase,
                sum(len(graphs) for graphs in self.batch_graphs.values()) - captures_before,
                time.perf_counter() - started,
            )
        self.synchronize()

    def stage_text_tokens(self, tokens: tuple[int, ...]) -> torch.Tensor:
        """Stage one text encoder input on the caller's stream without a world broadcast.

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

    def run_entry(self, name: str, *inputs: torch.Tensor) -> ExecutionOutput:
        """Execute an entry and validate its declared logical Tensor results."""

        entry = self.entries.get(
            (name, str(canonical_device(self.worker_config.device)), "default")
        )
        schemas = None if entry is None else entry.output_schema
        if not schemas:
            raise InputError(f"model does not declare computation entry {name!r}")

        return self._run_module(name, inputs, partial(_validate_products, name, schemas))

    def run_module(self, name: str, *inputs: object) -> ExecutionOutput:
        """Execute bound numerical modules over Tensor views and typed metadata.

        Numerical returns may be rank-local intermediate values. Logical product
        publication belongs to the caller and uses the entry's result declaration.
        Captured inputs remain Tensors with stable storage; eager calls may also
        receive mathematical metadata, explicit views and solver parameters.
        """

        return self._run_module(name, inputs)

    def run_decoder(
        self,
        kind: DecodeKind,
        batch: DecodeBatch,
        *,
        constants: TensorViews,
        scratch: TensorViews,
    ) -> ExecutionOutput:
        """Run the standard decoder call with native graphs bound by this owner.

        Packing and logical output trimming remain numerical model operations.
        The native layer uses the same fixed inputs, capture policy, and graph
        lifetime as its standalone entry. No runtime object enters the model.
        """

        binding = self._decoders.get(kind)
        model = self.model
        if binding is None or not isinstance(model, DecoderMixin):
            raise InputError(f"rank does not participate in {kind} decoding")
        name, layer = binding
        entry = self.entries[(name, str(canonical_device(self.worker_config.device)), "default")]
        path = "eager"
        layouts: tuple[TensorOutputLayout | None, ...] = ()

        def native(value: torch.Tensor) -> torch.Tensor:
            nonlocal path
            # Capture or replay the ordinary numerical layer, without recursively
            # dispatching the same binding during eager execution or capture.
            with decoder_scope(None):
                output, observed = self._call_module(entry, (value,))
            if path != "graph_capture":
                path = observed
            if not isinstance(output, torch.Tensor):
                raise ComputeError("native decoder must return one tensor")
            return output

        def compute() -> tuple[TensorOutput, str]:
            nonlocal layouts
            with decoder_scope({layer: native}):
                output = model.decode(kind, batch, constants=constants, scratch=scratch)
            rows = output.values[kind]
            if any(value is None for value in rows):
                raise ComputeError("decoder partition did not produce its numerical output")
            layouts = output.layouts[kind]
            return cast(tuple[torch.Tensor, ...], rows), path

        result = self._observe_module(name, compute)
        return replace(result, layouts=layouts)

    @property
    def encoder_kinds(self) -> frozenset[EncodeKind]:
        """Return encoder kinds that participate on this execution owner."""

        return frozenset(self._encoders)

    def run_encoder(self, kind: EncodeKind, *values: torch.Tensor) -> ExecutionOutput:
        """Execute the encoder assigned to this partition's declared numerical call."""

        name = self._encoders.get(kind)
        if name is None:
            raise InputError(f"rank does not participate in {kind} encoding")
        entry = self.entries[(name, str(canonical_device(self.worker_config.device)), "default")]
        validate = (
            partial(_validate_products, name, entry.output_schema) if entry.output_schema else None
        )
        return self._run_module(name, values, validate)

    def output_layout(
        self,
        entry: str,
        output_index: int,
        media: DiffusionSamplingParams | None,
        decode: DecodeRange | None,
        num_prompt_tokens: int,
    ) -> TensorOutputLayout | None:
        """Combine numerical result geometry with physical output participation.

        Temporal distribution assigns consecutive units along the declared
        result's first dimension. Numerical models describe that complete range;
        they do not select units using physical process ranks.
        """

        binding = self.bindings.get(entry)
        if binding is not None and binding.process_group.rank not in binding.output_ranks:
            return None
        layout = self.model.output_layout(
            entry,
            output_index,
            frames=None if media is None else media.num_frames,
            units=None if decode is None else decode.max_units,
            prompt_tokens=num_prompt_tokens,
        )
        if layout is None or binding is None or binding.config.distribution is None:
            return layout
        if layout.shape is None or layout.region is not None:
            raise ValueError("temporal distribution requires complete numerical output geometry")
        position = binding.config.ranks.index(binding.process_group.rank)
        start = position * binding.config.units_per_rank
        count = min(binding.config.units_per_rank, layout.shape[0] - start)
        if count < 1:
            return None
        return replace(
            layout,
            region=TensorRegion(
                (start, *((0,) * (len(layout.shape) - 1))),
                (count, *layout.shape[1:]),
            ),
        )

    @torch.inference_mode()
    def _run_module(
        self,
        name: str,
        inputs: tuple[object, ...],
        validate: Callable[[tuple[torch.Tensor, ...]], None] | None = None,
    ) -> ExecutionOutput:
        """Own eager/captured selection and observations for numerical calls."""

        module = self.entries.get(
            (name, str(canonical_device(self.worker_config.device)), "default")
        )
        if module is None:
            raise InputError(f"rank does not own computation entry {name!r}")
        if any(not isinstance(value, torch.Tensor) for value in inputs):
            raise InputError("module arguments must be tensors")
        tensors = cast(tuple[torch.Tensor, ...], inputs)
        return self._observe_module(name, lambda: self._call_module(module, tensors), validate)

    @torch.inference_mode()
    def warmup_module(self, name: str, *inputs: torch.Tensor) -> None:
        """Initialize numerical providers using this entry's actual callable."""

        entry = self.entries[(name, str(canonical_device(self.worker_config.device)), "default")]
        values = inputs or entry.fixed_inputs
        if values is None:
            raise ValueError("module warmup requires representative inputs")
        assert entry.forward is not None
        entry.forward(*values)

    def _capture_devices(self, device: torch.device) -> tuple[torch.device, ...]:
        """Return additional local GPUs whose branches can join this capture."""

        configured = (self.worker_config.device, self.worker_config.generation_device)
        devices = dict.fromkeys(
            canonical_device(value) for value in configured if value is not None
        )
        return tuple(value for value in devices if value.type == "cuda" and value != device)

    def _capture_module(self, entry: ModelEntry, inputs: tuple[torch.Tensor, ...]) -> None:
        stream = self.capture_stream()
        if stream is None:
            raise GraphExecutionError("module capture requires CUDA graph execution")
        graph = CudaGraph[TensorOutput](
            device=entry.device,
            stream=stream,
            device_pools=capture_pools(self._capture_devices(entry.device)),
        )
        graph.capture(
            partial(cast(Callable[..., TensorOutput], entry.forward), *inputs), keepalive=inputs
        )
        entry.graph = graph
        graph.inputs = inputs
        entry.signature = tensor_signature(inputs)

    def _call_module(
        self, entry: ModelEntry, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[TensorOutput, str]:
        if self._closed:
            raise GraphExecutionError("model runner is closed")
        if any(value.device != entry.device for value in inputs):
            raise GraphExecutionError("module input device changed")
        key = tensor_signature(inputs)
        if entry.fixed_inputs is not None:
            fixed_key = tensor_signature(entry.fixed_inputs)
            if tuple(value[:2] for value in key) != tuple(value[:2] for value in fixed_key):
                raise GraphExecutionError("CUDA graph input geometry changed")
            # Strided sources are copied into the fixed destination layout.
            key = fixed_key
        if self.capture_stream() is None:
            assert entry.forward is not None
            return cast(TensorOutput, entry.forward(*inputs)), "eager"
        missing = (
            False
            if entry.fixed_inputs is not None
            else capture_required(
                entry.signature != key or entry.graph is None, entry.groups, entry.device
            )
        )
        if missing:
            torch.cuda.current_stream(entry.device).synchronize()
            if entry.graph is not None:
                entry.graph.close()
                entry.graph = None
            entry.signature = None
            stable = tuple(
                torch.empty_strided(
                    value.shape, value.stride(), dtype=value.dtype, device=value.device
                )
                for value in inputs
            )
            for source, target in zip(inputs, stable, strict=True):
                target.copy_(source)
            self._capture_module(entry, stable)
        if entry.graph is None:
            raise GraphExecutionError("module graph is not captured")
        stable_inputs = cast(tuple[torch.Tensor, ...], entry.graph.inputs)
        for source, target in zip(inputs, stable_inputs, strict=True):
            target.copy_(source)
        return entry.graph.replay(), "graph_capture" if missing else "graph_replay"

    @staticmethod
    def _close_entry(entry: ModelEntry) -> None:
        if entry.graph is not None:
            entry.graph.close()
            entry.graph = None
        entry.fixed_inputs = None
        entry.signature = None

    def run_denoising(
        self,
        batch: DiffusionBatch,
        count: int,
        schedule: DiffusionSchedule,
        *,
        state: TensorViews,
        constants: TensorViews,
        scratch: TensorViews,
        slot: Hashable,
        geometry: Hashable,
    ) -> ExecutionOutput:
        diffusion = self.diffusion
        if diffusion is None:
            raise InputError("rank does not own denoising computation")
        if count != 1:
            raise InputError("denoising calls evaluate exactly one scheduled step")
        return self._observe_module(
            "denoiser",
            lambda: diffusion.step(
                batch,
                schedule,
                state=state,
                constants=constants,
                scratch=scratch,
                slot=slot,
                geometry=geometry,
            ),
        )

    @torch.inference_mode()
    def _observe_module(
        self,
        name: str,
        run: Callable[[], tuple[TensorOutput, str]],
        validate: Callable[[tuple[torch.Tensor, ...]], None] | None = None,
    ) -> ExecutionOutput:
        started = time.perf_counter_ns()
        with collective_scope(self._sum_reductions):
            output, path = run()
        if path == "graph_capture":
            logger.info(
                "completed first-use capture entry=%s device=%s seconds=%.3f",
                name,
                self.worker_config.device,
                (time.perf_counter_ns() - started) / 1e9,
            )
        values = (output,) if isinstance(output, torch.Tensor) else output
        if not isinstance(values, tuple) or any(
            not isinstance(value, torch.Tensor) for value in values
        ):
            raise ComputeError(f"entry {name!r} returned a non-tensor result")
        if any(value.device != canonical_device(self.worker_config.device) for value in values):
            raise ComputeError(f"entry {name!r} returned a tensor on another device")
        if validate is not None:
            validate(values)
        elapsed = (time.perf_counter_ns() - started) // 1000
        return ExecutionOutput(
            values,
            stats=ForwardStats(
                mode_counts={name: 1},
                mode_tokens={name: 1},
                mode_us={name: elapsed},
                component_us={"forward": elapsed},
                cuda_graph_runtime_mode_counts={path: 1},
                cuda_graph_captures=int(path == "graph_capture"),
                cuda_graph_replays=int(path == "graph_replay"),
            ),
        )

    def configure_inputs(
        self,
        *,
        geometry: InputGeometry,
        kv_cache,
        latent_pool,
        decode_predicates: torch.Tensor,
        max_operations: int,
        request_slots: int,
        max_tokens: int,
        latent_capacity_units: int,
        decode_context_blocks: int,
        variants,
        max_inflight: int,
    ) -> None:
        """Resolve each lane's physical catalog from its bound capacities."""

        from ..bootstrap.capacity import device_total_bytes
        from ..config import DEFAULT_PREFILL_GRAPH_ROW_BUCKETS, graph_memory_budget_bytes
        from ..nn.diffusion.cfg import build_flow_cfg_plan
        from .graph_inputs import (
            PrefillShape,
            select_flow_captures,
            select_prefill_captures,
        )

        worker_config = self.worker_config
        packed_model = self.model
        flow = packed_model.generation
        max_rows = min(
            max_operations,
            request_slots,
        )

        # Derive fixed staging and graph catalogs from the intersection of model,
        # lane, cache, latent, and scheduler capacities.
        decode_lane = next(
            (lane for lane in worker_config.lanes if ForwardMode.DECODE in lane.computations),
            None,
        )
        prefill_lane = next(
            (lane for lane in worker_config.lanes if ForwardMode.PREFILL in lane.computations),
            None,
        )
        flow_lane = next(
            (lane for lane in worker_config.lanes if PipelineStage.DENOISING in lane.computations),
            None,
        )
        decode_max_operations = min(
            max_rows,
            max_rows if decode_lane is None else int(decode_lane.max_batch_operations or max_rows),
        )
        prefill_max_tokens = min(
            int(max_tokens),
            (
                int(max_tokens)
                if prefill_lane is None
                else int(prefill_lane.max_batch_tokens or max_tokens)
            ),
        )
        flow_max_operations = min(
            max_rows,
            max_rows if flow_lane is None else int(flow_lane.max_batch_operations or max_rows),
        )
        decode_graph_batch_sizes = tuple(
            value
            for value in worker_config.decode_graph_batch_sizes
            if 0 < int(value) <= decode_max_operations and int(value) < int(kv_cache.num_pages)
        )
        prefill_capacity = min(
            int(max_tokens),
            int(packed_model.text_max_tokens),
            max(0, int(kv_cache.num_pages) - 1) * int(worker_config.block_size),
        )
        prefill_graph_token_sizes = tuple(
            value
            for value in worker_config.prefill_graph_token_sizes
            if 0 < int(value) <= min(prefill_capacity, prefill_max_tokens)
        )
        prefill_graph_row_sizes = DEFAULT_PREFILL_GRAPH_ROW_BUCKETS
        flow_cfg_branches: tuple[int, ...] = ()
        if flow is not None:
            branch_counts: list[int] = []
            for cfg_branches in range(1, int(flow.max_cfg_branches) + 1):
                image = capture_image_parameters(
                    cfg_branches,
                    steps=1,
                    height=16,
                    width=16,
                )
                guide = build_flow_cfg_plan(
                    cfg_text_scale=float(image.cfg_text_scale),
                    cfg_img_scale=float(image.cfg_img_scale),
                    recipe=flow.cfg_recipe,
                    renorm=image.cfg_renorm_type,
                    renorm_min=float(image.cfg_renorm_min),
                    use_cfg=True,
                )
                if len(guide.branches) != cfg_branches:
                    raise invalid_descriptor(
                        "flow CFG startup parameters do not realize their branch geometry"
                    )
                branch_counts.append(cfg_branches)
            flow_cfg_branches = tuple(branch_counts)
        flow_graph_buckets = (
            select_flow_captures(
                worker_config.flow_graph_shapes,
                worker_config.flow_graph_batch_sizes,
                flow_cfg_branches,
                max_operations=flow_max_operations,
                max_tokens=(geometry.max_tokens),
                per_image_capacity=int(latent_capacity_units),
                latent_capacity=int(latent_pool.capacity_units),
                physical_tokens=flow.sequence_length,
                image_tokens=flow.image_tokens,
            )
            if flow is not None and latent_pool is not None
            else ()
        )
        self.flow_cfg_branches = flow_cfg_branches
        self.kv_cache = kv_cache
        self.decode_predicates = decode_predicates
        self.decode_context_blocks = decode_context_blocks
        self.prefill_row_sizes = prefill_graph_row_sizes

        def configure_entry(entry: ModelEntry) -> None:
            device = entry.device
            binding = entry.cuda_stream
            lane = None if binding is None else binding.config
            computations = entry.computations
            owns_model_compute = device == canonical_device(worker_config.device)
            max_rows_for_entry = (
                max_rows if lane is None else int(lane.max_batch_operations or max_rows)
            )
            max_tokens_for_entry = (
                prefill_capacity if lane is None else int(lane.max_batch_tokens or prefill_capacity)
            )
            self.decode_shapes[entry] = (
                tuple(value for value in decode_graph_batch_sizes if value <= max_rows_for_entry)
                if owns_model_compute and ForwardMode.DECODE in computations
                else ()
            )
            prefill_tokens = (
                tuple(value for value in prefill_graph_token_sizes if value <= max_tokens_for_entry)
                if owns_model_compute and ForwardMode.PREFILL in computations
                else ()
            )
            self.prefill_shapes[entry] = (
                tuple(PrefillShape(value, 1, 1) for value in prefill_tokens)
                if packed_model.text_attention_mode is AttentionMode.PACKED
                else select_prefill_captures(
                    prefill_tokens,
                    prefill_graph_row_sizes,
                    max_rows=max_rows_for_entry,
                    max_tokens=max_tokens_for_entry,
                )
            )
            self.graph_memory_budgets[device] = graph_memory_budget_bytes(
                device_total_bytes(device)
            )
            if worker_config.graph_policy != "off" and device.type == "cuda":
                self.graph_streams[entry] = (
                    binding.stream
                    if binding is not None and binding.context is not None
                    else torch.cuda.Stream(device=device)
                )
                self.graph_pools[entry] = torch.cuda.graph_pool_handle()
                self.graph_device_pools[entry] = capture_pools(self._capture_devices(device))
                for additional in self.graph_device_pools[entry]:
                    self.graph_memory_budgets[additional] = graph_memory_budget_bytes(
                        device_total_bytes(additional)
                    )

        self._bind_inputs(
            geometry=geometry,
            lanes=worker_config.lanes,
            max_inflight=max_inflight,
            configure_entry=configure_entry,
            flow_captures=flow_graph_buckets,
        )

    def _bind_inputs(
        self,
        *,
        geometry: InputGeometry,
        configure_entry: Callable[[ModelEntry], None],
        lanes: tuple[LaneConfig, ...] = (),
        max_inflight: int = 1,
        flow_captures: tuple[DiffusionShape, ...] = (),
    ) -> None:
        """Construct device-and-lane runtimes around one pure execution model."""

        self.flow_captures = flow_captures
        worker_config = self.worker_config
        canonical = tuple(
            dict.fromkeys(
                (
                    str(torch.device(worker_config.device)),
                    str(torch.device(worker_config.generation_device or worker_config.device)),
                )
            )
        )
        if self.batch_graphs:
            raise RuntimeError("entry execution resources are already bound")
        self.uses_lanes = bool(lanes)

        def make_buffer(device: str) -> InputBuffers:
            """Allocate fixed-capacity staging storage for one execution device."""

            return InputBuffers(
                geometry=geometry,
                device=device,
                max_inflight=max_inflight,
            )

        if self.bindings:
            placements = tuple(self.bindings.values())
        else:
            # Standalone callers supply no physical entries; every declared
            # numerical component participates on the caller's local device.
            local = []
            for component in self.model.components(self.model.config):
                placement = self._entry(component.name, torch.device(canonical[0]))
                placement.calls = component.calls
                local.append(placement)
            placements = tuple(local)
        batch_calls = {Call.TEXT, Call.ENCODE_VISION, Call.ENCODE_LATENT, Call.DECODE_IMAGE}
        if self.model.generation is not None:
            batch_calls.add(Call.DIFFUSION)
        participants = tuple(
            (placement, calls)
            for placement in placements
            if (calls := tuple(call for call in placement.local_calls if call.call in batch_calls))
        )

        startup = ExitStack()
        bindings: list[tuple[torch.device, CudaStream | None, int | None]] = []
        binding: CudaStream | None
        entries: list[ModelEntry] = []
        available = supported_operations(self.model)
        try:
            if lanes:
                if len(canonical) != 1:
                    raise ValueError("Green Context lanes require one physical CUDA device")
                missing = available.difference(kind for lane in lanes for kind in lane.computations)
                if missing:
                    names = ", ".join(sorted(kind.value for kind in missing))
                    raise ValueError(f"lane configuration has no binding for computations: {names}")
                device = torch.device(canonical[0])
                streams = create_partitioned_streams(lanes, device, event_slots=max_inflight + 1)
                for binding in streams:
                    startup.callback(binding.close)
                    binding.verify()
                    bindings.append((device, binding, int(binding.context)))
            else:
                for device_name in canonical:
                    device = torch.device(device_name)
                    binding = (
                        CudaStream(
                            device=device,
                            stream=torch.cuda.current_stream(device),
                            sm_count=int(
                                torch.cuda.get_device_properties(device).multi_processor_count
                            ),
                            event_slots=max_inflight + 1,
                        )
                        if device.type == "cuda"
                        else None
                    )
                    if binding is not None:
                        startup.callback(binding.close)
                    bindings.append((device, binding, None))

            for placement, calls in participants:
                for device, binding, expected_context in bindings:
                    context = (
                        nullcontext() if binding is None else torch.cuda.stream(binding.stream)
                    )
                    with context:
                        inputs = make_buffer(str(device))
                    placement_entry = self._entry(placement.name, device)
                    # Each component and stream owns its addresses. Numerical stage
                    # participation comes from the component declaration, not its name.
                    entry = ModelEntry(
                        placement_entry.name,
                        placement_entry.config,
                        placement_entry.process_group,
                        placement_entry.mesh,
                        placement_entry.device,
                        calls=calls,
                    )
                    model = self.model
                    entry.forward = partial(
                        _invoke,
                        text=model if isinstance(model, TextMixin) else None,
                        diffusion_model=model if isinstance(model, DiffusionMixin) else None,
                        encoder=model if isinstance(model, EncoderMixin) else None,
                        decoder=model if isinstance(model, DecoderMixin) else None,
                        pipeline=None if entry.mesh is None else entry.mesh.get_group("pp"),
                    )
                    entry.output_schema = self.output_schema.get(entry.name, ())
                    configured = (
                        COMPUTATIONS
                        if binding is None or binding.config is None
                        else binding.config.computations
                    )
                    operations = call_operations(calls)
                    entry.computations = tuple(kind for kind in configured if kind in operations)
                    entry.input_buffers = inputs
                    entry.cuda_stream = binding
                    self.batch_graphs[entry] = {}
                    startup.callback(self._close_batch_entry, entry)
                    configure_entry(entry)
                    if expected_context is not None:
                        from ..runtime.collectives import allocate_stream_collectives

                        assert binding is not None
                        entry.collectives = allocate_stream_collectives(
                            entry.groups, binding.stream
                        )
                    entries.append(entry)
        except BaseException as error:
            try:
                self.synchronize()
                for _device, binding, _context in bindings:
                    if binding is not None:
                        binding.stream.synchronize()
            except BaseException as cleanup_error:
                error.add_note(f"Resource synchronization also failed: {cleanup_error!r}")
            try:
                startup.close()
            except BaseException as cleanup_error:
                error.add_note(f"Resource cleanup also failed: {cleanup_error!r}")
            raise
        self._streams.extend(
            binding for _device, binding, _context in bindings if binding is not None
        )
        for entry in entries:
            key = (
                entry.name,
                str(entry.device),
                "default" if entry.cuda_stream is None else entry.cuda_stream.name or "default",
            )
            self.entries[key] = entry
            for kind in entry.computations:
                if not isinstance(kind, ForwardMode) and kind not in {
                    PipelineStage.VISION_ENCODING,
                    PipelineStage.LATENT_ENCODING,
                    PipelineStage.DENOISING,
                    PipelineStage.IMAGE_DECODING,
                }:
                    continue
                # VAE/image reconstruction inputs live with generation modules;
                # token, vision, and learned denoising calls use the model binding.
                device = canonical_device(
                    worker_config.generation_device or worker_config.device
                    if kind in {PipelineStage.LATENT_ENCODING, PipelineStage.IMAGE_DECODING}
                    else worker_config.device
                )
                if entry.device == device:
                    self._forward_entries[(entry.name, kind)] = entry
        startup.pop_all()

    @torch.inference_mode()
    def prepare_fixed_modules(self) -> None:
        """Warm up and capture fixed module inputs during worker startup."""

        for module in self.entries.values():
            name = module.name
            if module.fixed_inputs is None:
                continue
            started = time.perf_counter()
            with collective_scope(self._sum_reductions):
                if self.capture_stream() is None:
                    self.warmup_module(name)
                else:
                    self._capture_module(module, module.fixed_inputs)
            logger.info(
                "prepared fixed module entry=%s mode=%s seconds=%.3f",
                name,
                "eager" if module.graph is None else "capture",
                time.perf_counter() - started,
            )

    def complete_startup(self) -> None:
        """Freeze attention bindings and model state after warmup completes."""

        for device, budget in self.graph_memory_budgets.items():
            pools = {
                tuple(pool) for entry, pool in self.graph_pools.items() if entry.device == device
            }
            pools.update(
                tuple(bindings[device].id)
                for bindings in self.graph_device_pools.values()
                if device in bindings
            )
            if _private_pool_bytes(device, pools) > budget:
                raise GraphExecutionError("captured graph residency exceeds its device budget")
        for entry in self.batch_graphs:
            if entry.cuda_stream is not None:
                entry.cuda_stream.verify()
        signature = tuple(
            (
                None if entry.cuda_stream is None else entry.cuda_stream.name,
                0 if entry.cuda_stream is None else entry.cuda_stream.sm_count,
                tuple(kind.value for kind in entry.computations),
                self.decode_shapes[entry],
                self.prefill_shapes[entry],
                tuple(sorted(repr(key) for key in graphs)),
            )
            for entry, graphs in self.batch_graphs.items()
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            logger.info("verifying tensor-parallel execution lane agreement")
            gathered: list[object] = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, signature)
            if any(value != signature for value in gathered):
                raise GraphExecutionError("tensor-parallel execution lanes disagree")
            logger.info("verified tensor-parallel execution lane agreement")

        self._startup_complete = True

    def close_graphs(self) -> None:
        """Release drained executables before backing, keeping streams alive for allocator cleanup.

        The caller has stopped execution and drained output consumers. Runtime
        pinned buffers can still enqueue allocator events while being freed, so
        their producer streams remain owned until close finishes teardown.
        """

        actions: list[Callable[[], object]] = [
            partial(self._close_entry, entry) for entry in self.entries.values()
        ]
        if self.diffusion is not None:
            actions.append(self.diffusion.close)
        actions.extend(
            graph.close for graphs in self.batch_graphs.values() for graph in graphs.values()
        )
        try:
            close_resources(*actions)
        finally:
            for graphs in self.batch_graphs.values():
                graphs.clear()

    def close(self) -> None:
        """Drain computation, then release graphs and staging before physical lanes."""

        if self._closed:
            return
        self._closed = True
        actions: list[Callable[[], object]] = [self.synchronize, self.close_graphs]
        if self._text_staging is not None:
            actions.append(self._text_staging.close)
        actions.extend(
            partial(self._close_batch_entry, entry) for entry in reversed(tuple(self.batch_graphs))
        )
        actions.extend(lane.close for lane in reversed(self._streams))
        actions.extend(
            reduction.close for reduction in reversed(tuple(self._sum_reductions.values()))
        )
        try:
            close_resources(*actions)
        finally:
            self.entries.clear()
            self._decoders.clear()
            self.diffusion = None
            self._capture_stream = None
            self._preparation_stream = None
            self._text_staging = None
            self._text_tokens = None
            self.batch_graphs.clear()
            self.graph_pools.clear()
            self.graph_device_pools.clear()
            self.graph_streams.clear()
            self.decode_shapes.clear()
            self.prefill_shapes.clear()
            self._sum_reductions.clear()
            self.scratch = None
            self._attention_exchange_storage.clear()
            self._forward_entries.clear()
            self._streams.clear()

    def synchronize(self) -> None:
        """Drain each compute and preparation stream before releasing resident resources."""

        actions: list[Callable[[], object]] = []
        if canonical_device(self.worker_config.device).type == "cuda":
            actions.append(torch.cuda.current_stream(self.worker_config.device).synchronize)
        if self._preparation_stream is not None:
            actions.append(self._preparation_stream.synchronize)
        if self._capture_stream is not None:
            actions.append(self._capture_stream.synchronize)
        actions.extend(stream.synchronize for stream in self.graph_streams.values())
        actions.extend(lane.stream.synchronize for lane in self._streams if lane.stream is not None)
        close_resources(*actions)

    @contextmanager
    def preparing_inputs(
        self, transfers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    ) -> Iterator[None]:
        """Overlap reserved input copies with computation, then join device order.

        Sources and destinations must remain reserved through this operation's
        physical completion. The final stream wait also covers failed compute,
        so completion and request retirement cannot overtake the input copies.
        """

        stream = self._preparation_stream
        if self._closed or canonical_device(self.worker_config.device).type != "cuda":
            raise RuntimeError("input preparation requires a live CUDA execution owner")
        if stream is None:
            stream = torch.cuda.Stream(device=self.worker_config.device)
            self._preparation_stream = stream
        try:
            with torch.cuda.stream(stream):
                for destination, source in transfers:
                    if destination.shape != source.shape or destination.dtype != source.dtype:
                        raise ValueError("prepared input must match destination shape and dtype")
                    destination.copy_(source, non_blocking=True)
            yield
        finally:
            torch.cuda.current_stream(self.worker_config.device).wait_stream(stream)

    def generation(self) -> ImageDiffusion:
        """Require the model's diffusion generation contract for the active operation."""

        value = self.model.generation
        if not isinstance(value, ImageDiffusion):
            raise invalid_descriptor("operation requires model generation behavior")
        return value

    def image_processor(self) -> ImageProcessor:
        """Require the model's image preprocessing contract for the active operation."""

        value = self.model.image_processor
        if not isinstance(value, ImageProcessor):
            raise invalid_descriptor("operation requires model image processing")
        return value

    def operation_devices(
        self, operation: ScheduledRequest
    ) -> tuple[torch.device, torch.device, torch.device]:
        """Resolve input consumption, numerical computation, and result publication.

        Forward inputs and outputs use the concrete numerical entry. Image
        trajectories consume and publish on the diffusion storage device, which
        may differ from the learned prediction's device. Module/media entries
        inherit their loaded placement. Completion storage must cover all three.
        """

        placement = self.bindings.get(operation.entry)
        device = (
            canonical_device(self.worker_config.device) if placement is None else placement.device
        )
        entry = self._forward_entries.get((operation.entry, operation.kind))
        if self.model.generation is not None and operation.kind in {
            PipelineStage.LATENT_PREPARATION,
            PipelineStage.DENOISING,
            PipelineStage.IMAGE_DECODING,
        }:
            if self.diffusion is None:
                raise invalid_descriptor("image operation has no bound diffusion storage device")
            device = self.diffusion.device
        compute = device if entry is None else entry.device
        return device, compute, device

    def _close_batch_entry(self, entry: ModelEntry) -> None:
        actions = [graph.close for graph in self.batch_graphs.get(entry, {}).values()]
        if entry.collectives is not None:
            actions.extend(binding.close for binding in entry.collectives.values())
        if entry.input_buffers is not None:
            actions.append(entry.input_buffers.close)
        close_resources(*actions)
        self.batch_graphs.pop(entry, None)
        self.graph_pools.pop(entry, None)
        self.graph_device_pools.pop(entry, None)
        self.graph_streams.pop(entry, None)
        entry.collectives = None
        entry.input_buffers = None

    def select_graph_shape(
        self,
        entry: ModelEntry,
        batch: InputBatch,
        *,
        eligible: bool,
    ) -> tuple[tuple[object, ...], InputBatch, int, bool] | None:
        """Select physical geometry while preserving each path's eager policy."""

        rows = batch.row_count
        if not eligible or self.worker_config.graph_policy == "off" or not _cuda_batch(batch):
            return None
        if batch.attention.attention_mode is AttentionMode.PAGED_VARLEN and (
            not self.worker_config.prefill_cuda_graph
            or any(
                selection is not TokenSelection.LAST_LOGITS for selection in batch.token_selections
            )
        ):
            return None
        if (
            batch.attention.attention_mode is AttentionMode.PACKED
            and not self.worker_config.prefill_cuda_graph
        ):
            return None
        if batch.attention.attention_mode is AttentionMode.PACKED and self.kv_cache.is_quantized:
            return None
        try:
            _graph_provider(
                self.attention,
                batch.attention.attention_mode,
                head_dim=self.model.cache_geometry.head_dim,
                block_size=self.worker_config.block_size,
                device=batch.request_pool_indices.device,
            )
        except _GraphMiss:
            return None

        decode = _decode_geometry(
            batch,
            self.decode_shapes[entry],
            self.worker_config.block_size,
            self.decode_context_blocks,
        )
        prefill = (
            None
            if decode is not None
            else _prefill_geometry(
                batch,
                tuple(shape.token_bucket for shape in self.prefill_shapes[entry])
                if self.model.text_attention_mode is not AttentionMode.PACKED
                else (),
                self.worker_config.block_size,
                self.prefill_row_sizes,
                self.decode_context_blocks,
            )
        )
        if decode is not None:
            execution = _pad_decode_batch(batch, *decode, self.worker_config.block_size)
            state_key = _decode_signature(batch, *decode)
            signature = state_key
            padded_rows = decode[0]
            startup_resident = True
        elif prefill is not None:
            execution = _pad_prefill_batch(batch, *prefill, self.worker_config.block_size)
            state_key = _prefill_signature(batch, *prefill, self.worker_config.block_size)
            signature = state_key
            padded_rows = prefill[0].row_bucket
            startup_resident = True
        else:
            execution = _normalize_exact_batch(
                batch,
                context_blocks=cast(InputBuffers, entry.input_buffers).max_blocks_per_row,
                block_size=self.worker_config.block_size,
            )
            signature = _exact_signature(execution)
            state_key = ("exact", signature)
            padded_rows = rows
            startup_resident = False

        return state_key, execution, padded_rows, startup_resident

    @torch.inference_mode()
    def capture_batch(
        self,
        entry: ModelEntry,
        batch: InputBatch,
        forward: Callable[[InputBatch], ExecutionOutput],
    ) -> None:
        """Capture one physical shape while restoring its mutable numerical inputs."""

        if self._startup_complete:
            raise GraphExecutionError("entry capture is outside startup preparation")
        selected = self.select_graph_shape(entry, batch, eligible=True)
        if selected is None:
            self.eager_batch(entry, batch, forward)
            return
        state_key, execution, _, bucketed = selected
        assert self.graph_streams[entry] is not None
        if state_key in self.batch_graphs[entry]:
            return
        if self.graph_memory_budgets[entry.device] == 0:
            raise GraphExecutionError("configured CUDA graph residency has no memory budget")
        static = _graph_batch(execution, next(_GRAPH_BINDINGS), own_inputs=not bucketed)
        releases: tuple[Callable[[], None], ...] = ()
        stream = None if entry.cuda_stream is None else entry.cuda_stream.stream
        context = nullcontext() if stream is None else torch.cuda.stream(stream)
        with context, stream_collective_scope(entry.collectives):
            restore = self._capture_restore(static)
            try:
                releases = self._prepare_graph_attention(entry, static, execution, capture=True)

                def compute() -> tuple[ExecutionOutput, SamplerOutput | None]:
                    output = forward(static)
                    return output, _greedy_decode(static, output, self.decode_predicates)

                graph = CudaGraph[tuple[ExecutionOutput, SamplerOutput | None]](
                    device=entry.device,
                    stream=self.graph_streams[entry],
                    pool=self.graph_pools[entry],
                    device_pools=self.graph_device_pools[entry],
                    expected_context=(
                        None if entry.cuda_stream is None else entry.cuda_stream.context
                    ),
                )
                graph.capture(compute, keepalive=(static,), restore=restore)
                self.batch_graphs[entry][state_key] = graph
                graph.inputs = static
                graph.releases = releases
                releases = ()
                graph.input_leaves = () if bucketed else tuple(_batch_tensors(static))
                graph.attention_leaves = tuple(_attention_inputs(static))
            except BaseException as error:
                if stream is not None:
                    stream.synchronize()
                else:
                    torch.cuda.current_stream(entry.device).synchronize()
                self._discard_batch_graph(entry, state_key)
                for release in reversed(releases):
                    release()
                error.add_note(f"entry capture device={entry.device} shape={state_key!r}")
                raise
            finally:
                restore()
                torch.cuda.current_stream(entry.device).synchronize()

    def _capture_restore(self, batch: InputBatch) -> Callable[[], None]:
        """Retain the bounded KV write set and graph-greedy mutable input."""

        tensors: list[tuple[torch.Tensor, torch.Tensor]] = []
        finish = batch.decode_force_finish
        if finish is not None:
            tensors.append((finish, finish.clone()))
        pages = torch.unique(batch.attention.out_cache_loc // self.worker_config.block_size)
        pages = pages[pages != 0].long()
        cache = self.kv_cache
        saved = (cache.k.index_select(1, pages), cache.v.index_select(1, pages))

        def restore() -> None:
            for tensor, snapshot in tensors:
                tensor.copy_(snapshot)
            cache.k.index_copy_(1, pages, saved[0])
            cache.v.index_copy_(1, pages, saved[1])

        return restore

    def run_batch(
        self,
        entry: ModelEntry,
        batch: InputBatch,
        forward: Callable[[InputBatch], ExecutionOutput],
        *,
        eligible: bool,
        borrow_output: bool = False,
    ) -> ExecutionOutput:
        """Stage metadata and replay a resident key, or use the established eager path."""

        rows = batch.row_count
        selected = self.select_graph_shape(entry, batch, eligible=eligible)
        if selected is None:
            return replace(
                self.eager_batch(entry, batch, forward),
                stats=ForwardStats(cuda_graph_runtime_mode_counts={"eager": 1}),
            )
        state_key, execution, padded_rows, bucketed = selected
        captured = False
        if state_key not in self.batch_graphs[entry]:
            if not bucketed:
                return replace(
                    self.eager_batch(entry, execution, forward),
                    stats=ForwardStats(cuda_graph_runtime_mode_counts={"eager": 1}),
                )
            if self._startup_complete:
                raise GraphExecutionError("configured CUDA graph bucket is not resident")
            # Direct execution may precede explicit startup. Materialize the
            # configured bucket on its binding, preserving the same full policy.
            self.capture_batch(entry, batch, forward)
            captured = True
        assert self.graph_streams[entry] is not None
        graph = self.batch_graphs[entry].get(state_key)
        if graph is None:
            raise GraphExecutionError("entry graph has no input owner")
        output, greedy = self._replay_batch(entry, graph, execution, rows)
        output = (
            _trim_output(output, rows)
            if borrow_output
            else self._publish_batch_output(entry, output, rows)
        )
        return replace(
            output,
            stats=ForwardStats(
                cuda_graph_runtime_mode_counts={"graph_capture" if captured else "graph_replay": 1},
                cuda_graph_captures=int(captured),
                cuda_graph_replays=int(not captured),
                cuda_graph_unpadded_tokens=rows,
                cuda_graph_padded_tokens=padded_rows - rows,
            ),
            greedy=greedy,
        )

    def _discard_batch_graph(self, entry: ModelEntry, key: Hashable) -> None:
        graph = self.batch_graphs[entry].pop(key, None)
        if graph is not None:
            graph.close()
        if not self.batch_graphs[entry] and entry in self.graph_pools:
            # CUDA retires a pool with its last graph; retained output allocations
            # must not be reused under the retired pool identity.
            self.graph_pools[entry] = torch.cuda.graph_pool_handle()
            self.graph_device_pools[entry] = capture_pools(self._capture_devices(entry.device))

    def eager_batch(
        self,
        entry: ModelEntry,
        batch: InputBatch,
        forward: Callable[[InputBatch], ExecutionOutput],
    ) -> ExecutionOutput:
        """Execute a numerical batch on the lane's eager path."""

        stream = None if entry.cuda_stream is None else entry.cuda_stream.stream
        context = nullcontext() if stream is None else torch.cuda.stream(stream)
        with context, stream_collective_scope(entry.collectives):
            return forward(batch)

    def _replay_batch(
        self,
        entry: ModelEntry,
        graph: CudaGraph[tuple[ExecutionOutput, SamplerOutput | None]],
        execution: InputBatch,
        rows: int,
    ) -> tuple[ExecutionOutput, SamplerOutput | None]:
        stream = None if entry.cuda_stream is None else entry.cuda_stream.stream
        context = nullcontext() if stream is None else torch.cuda.stream(stream)
        with context, stream_collective_scope(entry.collectives):
            if graph.input_leaves:
                _copy_tensors(graph.input_leaves, tuple(_batch_tensors(execution)), "forward")
            else:
                _copy_tensors(
                    graph.attention_leaves, tuple(_attention_inputs(execution)), "attention"
                )
            self._prepare_graph_attention(
                entry, cast(InputBatch, graph.inputs), execution, capture=False
            )
            assert self.graph_streams[entry] is not None
            output, greedy = graph.replay()
            # Trimming packs completion columns with a device copy. It must
            # follow replay on its producer stream, before the output fence.
            return output, _trim_greedy(greedy, rows)

    def _publish_batch_output(
        self, entry: ModelEntry, output: ExecutionOutput, rows: int
    ) -> ExecutionOutput:
        """Publish live rows before another graph reuses capture storage."""

        stream = None if entry.cuda_stream is None else entry.cuda_stream.stream
        consumer = None if stream is None else torch.cuda.current_stream(stream.device)
        context = nullcontext() if stream is None else torch.cuda.stream(stream)
        with context, stream_collective_scope(entry.collectives):
            published = _trim_output(output, rows).clone()
        if consumer is not None and consumer != stream:
            for value in published.values:
                value.record_stream(consumer)
        return published

    def _prepare_graph_attention(
        self,
        entry: ModelEntry,
        static_batch: InputBatch,
        live_batch: InputBatch,
        *,
        capture: bool,
    ) -> tuple[Callable[[], None], ...]:
        """Bind static attention wrappers to live page metadata for capture or replay."""

        # Dense and entry attention carry no persistent backend plan. Paged
        # modes first copy live table views into the static graph batch.
        static = static_batch.attention
        live = live_batch.attention
        if static.attention_mode is not live.attention_mode:
            raise _GraphMiss("attention form changed for a graph bucket")
        if static.attention_mode in {AttentionMode.DENSE, AttentionMode.PACKED}:
            return ()
        if static.attention_mode not in {AttentionMode.PAGED_DECODE, AttentionMode.PAGED_VARLEN}:
            return ()
        prepared = _live_attention(static, live)
        key_cache, _value_cache = self.kv_cache.layer_cache(0, static.group_id)
        backend = _graph_provider(
            self.attention,
            static.attention_mode,
            head_dim=self.model.cache_geometry.head_dim,
            block_size=self.worker_config.block_size,
            device=key_cache.device,
        )
        q_dtype = key_cache.dtype
        kv_dtype = key_cache.dtype
        releases: list[Callable[[], None]] = []

        if capture:
            release_name = (
                "release_paged_decode_graph_binding"
                if static.attention_mode is AttentionMode.PAGED_DECODE
                else "release_paged_prefill_graph_wrapper"
            )
            release = getattr(backend, release_name, None)
            if callable(release):
                releases.append(_release_call(release, static_batch.binding))
        try:
            # Decode wrappers are keyed by graph binding and can be replanned for
            # each live table while retaining fixed tensor addresses.
            if static.attention_mode is AttentionMode.PAGED_DECODE:
                prepare = getattr(backend, "prepare_paged_decode_cuda_graph", None)
                if callable(prepare):
                    prepare(
                        static_batch.binding,
                        prepared,
                        batch_size=int(cast(torch.Tensor, static.block_table).shape[0]),
                        max_indices=max(1, int(cast(torch.Tensor, static.block_table).numel())),
                        num_q_heads=int(self.model.cache_geometry.num_attention_heads),
                        num_kv_heads=int(self.model.cache_geometry.num_kv_heads),
                        head_dim=int(self.model.cache_geometry.head_dim),
                        page_size=self.worker_config.block_size,
                        q_dtype=q_dtype,
                        kv_dtype=kv_dtype,
                    )
                return tuple(releases)

            # Prefill capture owns a graph-bound wrapper until graph eviction;
            # replay updates only its caller-owned metadata buffers.
            if static.attention_mode is not AttentionMode.PAGED_VARLEN:
                return ()
            bind = getattr(backend, "bind_paged_prefill_graph_wrapper", None)
            prepare = getattr(backend, "prepare_paged_prefill_cuda_graph", None)
            if callable(bind) and callable(prepare):
                if capture:
                    bind(
                        static_batch.binding,
                        static,
                        device=cast(torch.Tensor, static.block_table).device,
                    )
                prepare(
                    static_batch.binding,
                    prepared,
                    num_q_heads=int(self.model.cache_geometry.num_attention_heads),
                    num_kv_heads=int(self.model.cache_geometry.num_kv_heads),
                    head_dim=int(self.model.cache_geometry.head_dim),
                    page_size=self.worker_config.block_size,
                    q_dtype=q_dtype,
                    kv_dtype=kv_dtype,
                    causal=static.causal,
                )
            return tuple(releases)
        except BaseException as error:
            for release in reversed(releases):
                try:
                    release()
                except BaseException as cleanup:
                    error.add_note(f"attention binding cleanup failed: {cleanup!r}")
            raise

    def forward(
        self,
        tasks: tuple[tuple[ForwardRow, ScheduledRequest], ...],
        *,
        cache: KVCache | None,
        tables: BlockTables | None,
        states: DecodeState | None,
    ) -> Iterator[tuple[tuple[int, ...], ExecutionOutput | BaseException]]:
        """Execute actual compatible rows, yielding results at their original indexes.

        A failed model call identifies every participating row. The caller owns
        completion groups and decides which dependent operations to suppress.
        Fatal failures propagate immediately because later device work is unsafe.
        """

        grouped: dict[tuple[object, ...], list[int]] = defaultdict(list)
        bindings: dict[int, ModelEntry] = {}
        for index, (task, operation) in enumerate(tasks):
            entry = self._forward_entries.get((operation.entry, operation.kind))
            if entry is None:
                yield (
                    (index,),
                    invalid_descriptor(
                        f"execution has no {operation.kind.value!r} binding for {operation.entry!r}"
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
            grouped[(entry, task.forward_mode, str(target), shape)].append(index)

        groups = list(grouped.values())
        for group_index, indexes in enumerate(groups):
            rows = tuple(tasks[index][0] for index in indexes)
            try:
                forward_started = time.perf_counter_ns()
                result = self.run_forward_group(
                    rows,
                    operations=tuple(tasks[index][1] for index in indexes),
                    cache=cache,
                    tables=tables,
                    states=states,
                )
                if result.request_pool_indices is None:
                    raise RuntimeError("forward output has no request slot views")
                if result.output_event is not None:
                    torch.cuda.current_stream(bindings[indexes[0]].device).wait_event(
                        result.output_event
                    )
                if all(row.forward_mode is ForwardMode.DECODE for row in rows):
                    if result.stats is None:
                        raise RuntimeError("text forward lost its statistics")
                    components = dict(result.stats.component_us)
                    record_component(components, "text_model_forward", forward_started)
                    result = replace(result, stats=replace(result.stats, component_us=components))
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
        operations: tuple[ScheduledRequest, ...],
        cache: KVCache | None,
        tables: BlockTables | None,
        states: DecodeState | None,
    ) -> ExecutionOutput:
        """Stage forward rows, choose eager or CUDA graph execution, invoke the model, and validate outputs."""

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
            raise ValueError("one numerical call requires homogeneous computations")
        forward_mode = next(iter(modes))
        operation_keys = tuple(
            (
                operation.request_key.engine_id,
                operation.request_key.request_id,
                operation.request_key.request_epoch,
                operation.op_id,
            )
            for operation in operations
        )
        entry = self._forward_entries.get((operations[0].entry, operations[0].kind))
        if entry is None:
            raise InputError(
                f"model runner has no {operations[0].kind.value!r} binding for {operations[0].entry!r}",
                phase="input_staging",
                route=forward_mode.value,
                operations=operation_keys,
            )
        target = entry.device
        lane_runtime = entry.cuda_stream
        buffers = entry.input_buffers
        assert buffers is not None
        try:
            if lane_runtime is not None:
                lane_runtime.wait(torch.cuda.current_stream(target))
            stream_context = (
                nullcontext() if lane_runtime is None else torch.cuda.stream(lane_runtime.stream)
            )
            with stream_context:
                batch = buffers.stage(
                    tasks,
                    forward_mode=forward_mode,
                    cache=cache,
                    tables=tables,
                    states=states,
                    packed=self.model.text_attention_mode is AttentionMode.PACKED,
                    binding=_binding_identity(operations),
                )
                request_pool_indices = batch.request_pool_indices
        except Exception as error:
            output_event = None if lane_runtime is None else lane_runtime.record()
            if output_event is not None:
                torch.cuda.current_stream(target).wait_event(output_event)
            raise _input_failure(error, forward_mode, operation_keys) from error

        def invoke(value: InputBatch) -> ExecutionOutput:
            return self.batch_forward(entry, value)

        output_event = None
        try:
            with torch.inference_mode():
                output = self.run_batch(
                    entry,
                    batch,
                    invoke,
                    eligible=graph_eligible,
                    borrow_output=all(
                        (task.request_indexed_decode or task.token_ids is not None)
                        and task.query_tokens == 1
                        and task.selection is TokenSelection.LAST_LOGITS
                        for task in tasks
                    ),
                )
            output.validate_for(batch)
            _validate_outputs(output.values, tasks, target)
            output_event = None if lane_runtime is None else lane_runtime.record()
            duration_us = (time.perf_counter_ns() - started) // 1000
            if output.stats is None:
                raise RuntimeError("entry forward lost its execution statistics")
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
            # retires storage behind the current stream's output fence, so join
            # every submitted lane access before reporting the failure.
            if output_event is None:
                output_event = None if lane_runtime is None else lane_runtime.record()
            if output_event is not None:
                torch.cuda.current_stream(target).wait_event(output_event)
            raise _execution_failure(error, forward_mode, operation_keys) from error


def _validate_outputs(
    values: tuple[torch.Tensor, ...],
    tasks: tuple[ForwardRow, ...],
    device: torch.device,
) -> None:
    """Validate one model output tensor per task on the expected device."""

    for value, task in zip(values, tasks, strict=True):
        if value.device != device:
            raise ValueError(f"model output is on {value.device}, expected {device}")
        if not value.is_floating_point():
            raise ValueError("raw neural outputs must use a floating dtype")
        if (
            task.request_indexed_decode
            or task.token_ids is not None
            or task.token_embeddings is not None
        ) and value.ndim < 2:
            raise ValueError("token output must retain token and feature dimensions")
        if task.latent is not None and task.image_tokens > 0 and value.shape != task.latent.shape:
            raise ValueError("flow prediction shape does not match its latent")
        if task.encode_pixels is not None and value.numel() == 0:
            raise ValueError("encoder output must not be empty")
        if task.latent is not None and task.image_tokens == 0:
            if value.ndim < 2 or tuple(value.shape[-2:]) != (
                task.image_height,
                task.image_width,
            ):
                raise ValueError("decoded tensor does not match declared image geometry")


def _input_failure(
    error: BaseException,
    forward_mode: ForwardMode | PipelineStage,
    operations: tuple[tuple[int, int, int, ComputationId], ...],
) -> InputError:
    """Classify invalid model inputs with their phase and operation identities."""

    if isinstance(error, InputError):
        return error
    return InputError(
        str(error) or type(error).__name__,
        phase="input_staging",
        route=forward_mode.value,
        operations=operations,
    )


def _execution_failure(
    error: BaseException,
    forward_mode: ForwardMode | PipelineStage,
    operations: tuple[tuple[int, int, int, ComputationId], ...],
) -> WorkerError:
    """Classify a model failure and attach the active phase and operation identities."""

    if isinstance(error, WorkerError):
        return error
    classified = classify(error)
    if isinstance(error, GraphExecutionError) or classified.code in {
        WorkerErrorCode.RESOURCE_ERROR,
        WorkerErrorCode.FATAL_WORKER_FAILURE,
    }:
        return ResourceError(
            str(error) or type(error).__name__,
            phase="graph_or_device",
            route=forward_mode.value,
            operations=operations,
            retryable=classified.retryable,
            fatal=classified.fatal,
        )
    return ComputeError(
        str(error) or type(error).__name__,
        phase="neural_execution",
        route=forward_mode.value,
        operations=operations,
    )


__all__ = ["ModelRunner"]


def _binding_identity(operations: tuple[ScheduledRequest, ...]) -> int:
    """Return a shared attention binding when every forward row agrees."""

    hasher = hashlib.blake2b(digest_size=8)
    for operation in operations:
        hasher.update(int(operation.request_key.request_id).to_bytes(8, "little"))
        hasher.update(int(operation.request_key.request_epoch).to_bytes(8, "little"))
        hasher.update(operation.request_key.engine_id.to_bytes(8, "little"))
        hasher.update(operation.op_id.batch_id.to_bytes(8, "little"))
        hasher.update(operation.op_id.request_index.to_bytes(4, "little"))
    return int.from_bytes(hasher.digest(), "little")
