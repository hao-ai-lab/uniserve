"""Numerical parameters and allocation views for native execution batches.

Rust owns Batch validation, request commands and resource relationships.
Python constructors supply immutable parameters; numerical code reads views
of the same native batch used by direct submission and the rank service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from uniserve import sampling
from uniserve.media import image
from uniserve_worker._uniserve_ipc import Batch as Batch
from uniserve_worker._uniserve_ipc import CanvasSampling as CanvasSampling
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol import call, identity, tensor, transfer
from uniserve_worker.protocol.validation import (
    _map,
    _nonnegative,
    _str,
    _uint,
    _uints,
)
from uniserve_worker.protocol.video import VideoAdmission


@dataclass(frozen=True, slots=True)
class Start:
    """Registers a new request and its immutable execution parameters."""

    request: NewRequest

    @property
    def request_key(self) -> identity.RequestKey:
        """Expose the request generation registered by this start command."""
        return self.request.request_key


@dataclass(frozen=True, slots=True)
class Finish:
    """Closes one request epoch, keeping the buffers it names as retained.

    The worker retires the request's state and storage and revokes its
    unretained exports once their readers finish. Each buffer in
    ``retained_buffers`` stays readable until a `Free` command names it.
    Native batch validation requires each retained buffer to belong to
    ``request_key`` and appear once.
    """

    request_key: identity.RequestKey
    retained_buffers: tuple[identity.BufferId, ...] = ()


@dataclass(frozen=True, slots=True)
class Free:
    """Releases one scheduler-owned persistent buffer."""

    buffer: identity.BufferId

    @property
    def request_key(self) -> identity.RequestKey:
        """Expose the request generation owning the freed persistent buffer."""
        return self.buffer.owner


BatchCommand: TypeAlias = Start | Finish | Free


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """Sampling policy and token controls for autoregressive execution.

    Serialized under the ``ar`` key of a `NewRequest` mapping.
    """

    sampling: sampling.SamplingParams = field(
        default_factory=sampling.SamplingParams
    )
    # Tokenized negative prompt, used as negative-conditioning input.
    negative_token_ids: tuple[int, ...] = ()
    # Stop tokens; must be strictly ascending (canonical form, no duplicates).
    finish_token_ids: tuple[int, ...] = ()
    # Prompt tokens already computed at admission; the engine sets it from a
    # prefix-cache hit, and `RequestPool.start` begins the request with this
    # many tokens visible and computed.
    initial_position: int = 0
    # Block-diffusion sampling of a request generating in canvases.
    canvas: CanvasSampling | None = None

    def __post_init__(self) -> None:
        """Require a nonnegative initial position and canonical stop tokens.

        `SamplingParams` checks its own invariants when constructed.
        """
        _nonnegative(self.initial_position, "autoregressive initial position")
        if any(
            left >= right
            for left, right in zip(
                self.finish_token_ids,
                self.finish_token_ids[1:],
                strict=False,
            )
        ):
            raise invalid_descriptor(
                "autoregressive finish token ids are not canonical"
            )

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "autoregressive parameters"
    ) -> GenerationParams:
        """Parse sampling policy, token controls, and initial position.

        Absent fields take their defaults.
        """
        data = _map(value, where)
        return cls(
            sampling=sampling.SamplingParams.from_mapping(
                _map(data.get("sampling", {}), f"{where}.sampling")
            ),
            negative_token_ids=_uints(
                data.get("negative_token_ids", ()),
                f"{where}.negative_token_ids",
            ),
            finish_token_ids=_uints(
                data.get("finish_token_ids", ()), f"{where}.finish_token_ids"
            ),
            initial_position=_uint(
                data.get("initial_position", 0), f"{where}.initial_position"
            ),
            canvas=None
            if data.get("canvas") is None
            else CanvasSampling.from_mapping(data.get("canvas")),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize autoregressive sampling and token controls."""
        return {
            "sampling": self.sampling.to_mapping(),
            "negative_token_ids": list(self.negative_token_ids),
            "finish_token_ids": list(self.finish_token_ids),
            "initial_position": self.initial_position,
            "canvas": None if self.canvas is None else self.canvas.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class DiffusionParams:
    """Effective diffusion bounds and seed resolved by model preprocessing.

    Mirrors the Rust `DiffusionSamplingParams`.
    """

    # Output frames after model-specific alignment.
    num_frames: int
    # Video media units the decoder reconstructs, independent of rank count.
    video_units: int
    # Denoising steps in the trajectory.
    num_inference_steps: int
    # Deterministic noise seed; nonnegative.
    seed: int
    # Output raster in pixels: the request's canvas.
    width: int
    height: int

    def __post_init__(self) -> None:
        """Require positive work bounds and a nonnegative deterministic seed."""
        for name in (
            "num_frames",
            "video_units",
            "num_inference_steps",
            "width",
            "height",
        ):
            if getattr(self, name) < 1:
                raise invalid_descriptor(f"diffusion {name} must be positive")
        _nonnegative(self.seed, "diffusion seed")

    @property
    def canvas(self) -> image.Config:
        """The request's output raster."""
        return image.Config(self.height, self.width)

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "diffusion sampling"
    ) -> DiffusionParams:
        """Parse the core's effective diffusion parameters."""
        data = _map(value, where)
        return cls(
            num_frames=_uint(data.get("num_frames"), f"{where}.num_frames"),
            video_units=_uint(data.get("video_units"), f"{where}.video_units"),
            num_inference_steps=_uint(
                data.get("num_inference_steps"), f"{where}.num_inference_steps"
            ),
            seed=_uint(data.get("seed"), f"{where}.seed"),
            width=_uint(data.get("width"), f"{where}.width"),
            height=_uint(data.get("height"), f"{where}.height"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize the shared diffusion parameter definition."""
        return {
            "num_frames": self.num_frames,
            "video_units": self.video_units,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class NewRequest:
    """Binds one request key to its slot and execution-family parameters.

    The families are autoregressive (``generation``), image (``image``), and
    diffusion (``diffusion``). An image-generating request carries both
    ``generation`` and ``image``.
    """

    request_key: identity.RequestKey
    # Scheduler-assigned request-pool slot; slot 0 is reserved and rejected.
    request_pool_idx: int
    # At least one family parameter set must be present. ``generation`` is
    # keyed ``ar`` on the wire.
    generation: GenerationParams | None = None
    image: call.ImageParams | None = None
    diffusion: DiffusionParams | None = None
    # Prompt tokens; required (non-empty) for diffusion requests.
    prompt_token_ids: tuple[int, ...] = ()
    # Number of input images the request carries. Model preprocessing may
    # bound each input image by its share of a pixel budget the images
    # share (``PatchTransform.pixel_bound``).
    input_images: int = 0
    # A video request's task, presentation tags and conditions; present
    # exactly with ``diffusion``.
    video: VideoAdmission | None = None

    def __post_init__(self) -> None:
        """Require a positive slot and at least one family parameter set.

        Diffusion requests also need non-empty prompt tokens, each with its
        presentation tag. Each family's own parameters are validated by its
        record.
        """
        if self.diffusion is not None and not self.prompt_token_ids:
            raise invalid_descriptor(
                "diffusion prompt tokens must not be empty"
            )
        if (self.video is None) != (self.diffusion is None):
            raise invalid_descriptor(
                "a video admission carries both its sampling and its inputs"
            )
        if self.video is not None and len(self.video.text_tags) != len(
            self.prompt_token_ids
        ):
            raise invalid_descriptor(
                "every video prompt token requires one tag"
            )
        if self.request_pool_idx < 1:
            raise invalid_descriptor("request-pool index must be positive")
        if (
            self.generation is None
            and self.image is None
            and self.diffusion is None
        ):
            raise invalid_descriptor(
                "request start must declare one runtime-family parameter set"
            )

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "admission"
    ) -> NewRequest:
        """Parse a request's identity, slot, prompt, and family parameters.

        A family key that is absent or null leaves that family unset.
        """
        data = _map(value, where)
        image_data = data.get("image")
        admission = cls(
            request_key=identity.RequestKey.from_mapping(
                data.get("request_key"), f"{where}.request_key"
            ),
            request_pool_idx=_uint(
                data.get("request_pool_idx"), f"{where}.request_pool_idx"
            ),
            prompt_token_ids=_uints(
                data.get("prompt_token_ids", ()), f"{where}.prompt_token_ids"
            ),
            input_images=_uint(
                data.get("input_images", 0), f"{where}.input_images"
            ),
            generation=(
                None
                if data.get("ar") is None
                else GenerationParams.from_mapping(data["ar"], f"{where}.ar")
            ),
            image=(
                None
                if image_data is None
                else call.ImageParams.from_mapping(
                    image_data,
                    f"{where}.image",
                )
            ),
            diffusion=(
                None
                if data.get("diffusion") is None
                else DiffusionParams.from_mapping(
                    data["diffusion"], f"{where}.diffusion"
                )
            ),
            video=(
                None
                if data.get("video") is None
                else VideoAdmission.from_mapping(
                    data["video"], f"{where}.video"
                )
            ),
        )
        return admission

    def to_mapping(self) -> dict[str, object]:
        """Serialize request identity, slot, and family parameters."""
        return {
            "request_key": self.request_key.to_mapping(),
            "request_pool_idx": self.request_pool_idx,
            "prompt_token_ids": list(self.prompt_token_ids),
            "input_images": self.input_images,
            "ar": None
            if self.generation is None
            else self.generation.to_mapping(),
            "image": None if self.image is None else self.image.to_mapping(),
            "diffusion": None
            if self.diffusion is None
            else self.diffusion.to_mapping(),
            "video": None if self.video is None else self.video.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class BlockTable:
    """The complete unit table of one request slot and KV cache group.

    ``unit_ids`` lists the units of logical pages ``start_page..``,
    page-major: the group's ``units_per_page`` units of ``start_page`` come
    first. Earlier pages are retired window pages that no call reads.
    ``allocated_tokens`` is the absolute token extent the table covers; the
    worker checks it against the group's page shape.
    """

    # Request-pool slot owning the table; slot 0 is rejected.
    request_pool_idx: int
    # KV cache group addressed by the table.
    group_id: int
    # First logical page the units cover.
    start_page: int
    # Physical unit identifiers; unique, unit 0 rejected.
    unit_ids: tuple[int, ...]
    # Absolute token extent the table covers.
    allocated_tokens: int

    def to_mapping(self) -> dict[str, object]:
        """Serialize an installed request-and-group KV unit table."""
        return {
            "request_pool_idx": self.request_pool_idx,
            "group_id": self.group_id,
            "start_page": self.start_page,
            "unit_ids": list(self.unit_ids),
            "allocated_tokens": self.allocated_tokens,
        }


@dataclass(frozen=True, slots=True)
class CacheUnitAllocation:
    """Physical KV units newly assigned to a request slot and KV group.

    The units are a subset of the batch's `BlockTable` for the same slot and
    group, as checked by native batch validation.
    The worker resets every listed unit before a call uses it.
    """

    # Request-pool slot receiving the units; slot 0 is rejected.
    request_pool_idx: int
    group_id: int
    # Newly assigned physical units; non-empty, unique, unit 0 rejected.
    unit_ids: tuple[int, ...]

    def to_mapping(self) -> dict[str, object]:
        """Serialize newly assigned KV units for lane registration."""
        return {
            "request_pool_idx": self.request_pool_idx,
            "group_id": self.group_id,
            "unit_ids": list(self.unit_ids),
        }


@dataclass(frozen=True, slots=True)
class LatentParams:
    """Solver-step range and optional paged storage of one trajectory.

    For a paged trajectory, `LatentPool` records the pages, units, and raster
    when the trajectory's slot is initialized and checks later calls against
    them.
    """

    request_key: identity.RequestKey
    call_id: identity.CallId
    # `LatentPool` pages in logical order; unique, page 0 rejected. Empty
    # exactly when ``latent_units`` is zero, which means the trajectory lives
    # in request-owned tensors instead of the pool.
    page_table: tuple[int, ...]
    # Latent rows stored in ``page_table``; a `LatentPool` page holds
    # ``page_units`` rows.
    latent_units: int
    # Output height and width in pixels.
    height: int
    width: int
    # The call runs solver steps [start_step, start_step + step_count).
    start_step: int
    step_count: int

    def to_mapping(self) -> dict[str, object]:
        """Serialize the trajectory's pages, raster, and step range."""
        return {
            "request_key": self.request_key.to_mapping(),
            "call_id": self.call_id.to_mapping(),
            "page_table": list(self.page_table),
            "latent_units": self.latent_units,
            "height": self.height,
            "width": self.width,
            "start_step": self.start_step,
            "step_count": self.step_count,
        }


class MediaTrack(StrEnum):
    """Independent media stream addressed by a bounded call."""

    VIDEO = "video"
    AUDIO = "audio"


@dataclass(frozen=True, slots=True)
class DecodeRange:
    """Selects a bounded range of media units for one media call.

    The Rust `Batch::validate` requires one for each video or audio decoding
    or encoding call in a batch.
    """

    request_key: identity.RequestKey
    call_id: identity.CallId
    # First media unit assigned to the call.
    cursor: int
    # Maximum media units the call may process; positive.
    max_units: int

    def to_mapping(self) -> dict[str, object]:
        """Serialize the call's cursor and unit bound."""
        return {
            "request_key": self.request_key.to_mapping(),
            "call_id": self.call_id.to_mapping(),
            "cursor": self.cursor,
            "max_units": self.max_units,
        }


@dataclass(frozen=True, slots=True)
class BufferAllocation:
    """Scheduler-chosen byte span for one persistent cross-call buffer."""

    buffer: identity.BufferId
    # Half-open byte span [offset, offset + bytes) in a device's
    # persistent-buffer arena; the end must fit in u64. `BufferPool` binds
    # the buffer at ``offset`` unless it places buffers compactly.
    offset: int
    bytes: int

    def to_mapping(self) -> dict[str, object]:
        """Serialize persistent-buffer identity, offset, and byte extent."""
        return {
            "buffer": self.buffer.to_mapping(),
            "offset": self.offset,
            "bytes": self.bytes,
        }


@dataclass(frozen=True, slots=True)
class TensorExport:
    """A tensor result and the locations from which consumers can read it.

    Batches carry these as host-supplied ``input_products``, and a rank
    reports its exported results in `BatchOutput`.
    """

    product: tensor.TensorRef
    value: transfer.TransferValue

    def encoded_size_bound(self) -> int:
        """Return the estimated encoded size of the tensor's locators.

        Raises:
            WorkerError: From `invalid_descriptor` when the estimate exceeds
                ``transfer.MAX_TRANSFER_HANDLE_BYTES`` or a locator names an
                unknown transport.
        """
        size = transfer._tensor_transfers_size((self.value.tensor,))
        if size > transfer.MAX_TRANSFER_HANDLE_BYTES:
            raise invalid_descriptor("transfer metadata exceeds its byte bound")
        return size

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "tensor export"
    ) -> TensorExport:
        """Parse a tensor identity and its tagged transfer value.

        The value's kind is ``encoder``, ``device_product``, or ``latent``.

        Raises:
            WorkerError: From `invalid_descriptor` on an unknown kind or an
                invalid field.
        """
        data = _map(value, where)
        product = tensor.TensorRef.from_mapping(
            data.get("product"), f"{where}.product"
        )
        transfer_data = _map(data.get("value"), f"{where}.value")
        kind = _str(transfer_data.get("kind"), f"{where}.value.kind")
        payload = _map(transfer_data.get("value"), f"{where}.value.value")

        # Height/width default to zero for device products and are required
        # for every other transfer kind.
        height = _uint(
            payload.get("height", 0)
            if kind == "device_product"
            else payload.get("height"),
            f"{where}.value.height",
        )
        width = _uint(
            payload.get("width", 0)
            if kind == "device_product"
            else payload.get("width"),
            f"{where}.value.width",
        )
        tensor_value = transfer.TensorTransfer.from_mapping(
            payload.get("tensor"), f"{where}.value.tensor"
        )
        if kind == "encoder":
            typed: transfer.TransferValue = transfer.EncoderTransferValue(
                height=height,
                width=width,
                payload_kind=_str(
                    payload.get("payload_kind"), f"{where}.value.payload_kind"
                ),
                tensor=tensor_value,
            )
        elif kind == "device_product":
            typed = transfer.DeviceProductTransferValue(
                height=height,
                width=width,
                value_range=_str(
                    payload.get("value_range", ""), f"{where}.value.value_range"
                ),
                tensor=tensor_value,
            )
        elif kind == "latent":
            typed = transfer.LatentTransferValue(
                height=height,
                width=width,
                latent_units=_uint(
                    payload.get("latent_units"), f"{where}.value.latent_units"
                ),
                step=_uint(payload.get("step", 0), f"{where}.value.step"),
                tensor=tensor_value,
            )
        else:
            raise invalid_descriptor(f"{where}.value.kind is invalid")
        return cls(product=product, value=typed)

    def to_mapping(self) -> dict[str, object]:
        """Encode the tensor identity and its kind-tagged transfer value."""
        typed = self.value
        value: dict[str, object] = {
            "height": typed.height,
            "width": typed.width,
            "tensor": typed.tensor.to_mapping(),
        }
        if isinstance(typed, transfer.EncoderTransferValue):
            kind = "encoder"
            value["payload_kind"] = typed.payload_kind
        elif isinstance(typed, transfer.DeviceProductTransferValue):
            kind = "device_product"
            value["value_range"] = typed.value_range
        else:
            kind = "latent"
            value["latent_units"] = typed.latent_units
            value["step"] = typed.step
        return {
            "product": self.product.to_mapping(),
            "value": {"kind": kind, "value": value},
        }
