"""Image decoding and staging for model-owned processing policy.

The model declares its image policy as a ``uniserve.processing``
``ImageProcessor``; this module applies it for the worker. An inline base64
payload (``prepare_image``) or an already decoded RGB tensor
(``prepare_tensor_image``) is resized to the model's canvas (a patch tower
keeps the source size) and then to the selected tower's input size with the
declared resampling, normalized, packed into patches in the declared layout
for a ``PatchTransform`` tower, and staged on the target device. The input
rows for vision encoding and image decoding live here as well.
"""

from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision
from torchvision.transforms.v2 import functional as tensor_vision

from uniserve.nn.functional import patchify
from uniserve.processing import (
    ImageProcessor,
    PatchTransform,
    StrideResize,
    TowerTransform,
)
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.model_executor.input_batch import InputRow
from uniserve_worker.protocol.call import MediaCall

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# The Transformers image processors' rescale factor, 0.00392156862745098;
# ``"unit"`` normalization multiplies 8-bit values by it in float32.
_UNIT_SCALE = 1 / 255


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """Normalized pixels, patch coordinates, and model canvas dimensions.

    For a ``PatchTransform`` tower, ``pixels`` is
    [patches, channels * patch * patch] with patches in raster order and
    each row in the transform's ``patch_layout``, and ``grid`` is a [1, 2]
    int64 tensor holding ``grid_shape`` (rows, columns of patches). For a
    ``TowerTransform`` tower, ``pixels`` is [channels, height, width] and
    both grid fields are None. ``height`` and ``width`` are the canvas
    dimensions, not the tower's input size.
    """

    pixels: torch.Tensor
    grid: torch.Tensor | None
    grid_shape: tuple[int, int] | None
    height: int
    width: int


def _image_plan(
    processor: ImageProcessor,
    kind: MediaCall,
    height: int,
    width: int,
    input_images: int | None,
) -> tuple[PatchTransform | TowerTransform, tuple[int, int], tuple[int, int]]:
    """Resolve canvas and tower dimensions independently of pixel storage.

    Vision encoding uses the processor's ``vit`` transform and every other
    call kind its ``vae`` transform. A patch tower keeps the source size as
    its canvas, and its input size is ``PatchTransform.resized_size`` for
    ``input_images``. For a ``TowerTransform``, whether ``vit`` or ``vae``,
    the canvas is the source resized by the ``vae`` stride policy when the
    processor declares a ``vae`` transform, and the tower input applies the
    selected transform's stride policy to that canvas.

    Returns:
        The selected transform, the canvas (height, width) and the tower's
        input (height, width).

    Raises:
        WorkerError: An ``invalid_descriptor`` error when the model declares
            no transform for ``kind``, or, for a patch tower, when its resize
            policy rejects the dimensions.
    """
    transform = (
        processor.vit if kind is MediaCall.VISION_ENCODING else processor.vae
    )
    if transform is None:
        raise invalid_descriptor(
            f"model declares no {kind.value} image transform"
        )
    if isinstance(transform, PatchTransform):
        try:
            tower = transform.resized_size(height, width, input_images)
        except ValueError as error:
            raise invalid_descriptor(str(error)) from error
        return transform, (height, width), tower

    canvas = (
        (height, width)
        if processor.vae is None
        else _stride_image_shape(height, width, processor.vae.resize)
    )
    return transform, canvas, _stride_image_shape(*canvas, transform.resize)


def _prepared_pixels(processor, transform, pixels, canvas, device):
    """Pack the numerical tower input and attach its canvas coordinates."""
    grid = grid_shape = None
    if isinstance(transform, PatchTransform):
        # [C, H, W] -> one flattened row per patch, in raster order over the
        # patch grid; the declared layout orders the values within a row.
        patch = transform.patch_size
        channels, height, width = pixels.shape
        grid_shape = (height // patch, width // patch)
        if transform.patch_layout == "channels_last":
            pixels = patchify(pixels, patch_size=patch)
        else:
            # [C, H, W] -> [gh, gw, C, patch, patch] -> rows.
            pixels = (
                pixels.reshape(
                    channels, grid_shape[0], patch, grid_shape[1], patch
                )
                .permute(1, 3, 0, 2, 4)
                .reshape(
                    grid_shape[0] * grid_shape[1], channels * patch * patch
                )
            )
        grid = torch.tensor([grid_shape], dtype=torch.long, device=device)
    return PreparedImage(
        _stage(pixels, processor, device), grid, grid_shape, *canvas
    )


def prepare_image(
    processor: ImageProcessor,
    kind: MediaCall,
    encoded: str,
    *,
    device: torch.device,
    input_images: int,
) -> PreparedImage:
    """Decode a request input image and apply the model's transforms.

    ``input_images`` is the number of input images in the image's request;
    a patch tower whose images share a pixel budget bounds each by its share
    (see ``PatchTransform.pixel_bound``).
    """
    image = _decode_rgb(encoded, processor.alpha)
    transform, canvas, tower = _image_plan(
        processor, kind, image.height, image.width, input_images
    )
    if (
        isinstance(transform, PatchTransform)
        and transform.resampling == "torchvision"
    ):
        pixels = _normalize(
            _resample_bytes(image, tower).numpy(), transform.normalization
        )
        return _prepared_pixels(processor, transform, pixels, canvas, device)

    if not isinstance(transform, PatchTransform):
        image = vision.resize(
            image,
            canvas,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    image = vision.resize(
        image, tower, interpolation=InterpolationMode.BICUBIC, antialias=True
    )
    pixels = _normalize(
        np.asarray(image).transpose(2, 0, 1), transform.normalization
    )
    return _prepared_pixels(processor, transform, pixels, canvas, device)


def _resample_bytes(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    """Resize an RGB image as a ``uint8`` CHW tensor with torchvision.

    Returns the ``[3, height, width]`` ``uint8`` image at ``size``, following
    the Transformers torchvision image-processor path call for call:
    ``pil_to_tensor`` yields a channels-last view of the decoded bytes, and
    the antialiased bicubic filter runs on it only when the size changes,
    rounding back to 8 bits. torchvision picks its interpolation kernel by
    host architecture and memory format, so the same calls on the host
    reproduce the reference pixels independently of the staging device.
    """
    pixels = tensor_vision.pil_to_tensor(image)
    if tuple(pixels.shape[1:]) == size:
        return pixels
    return tensor_vision.resize(
        pixels,
        list(size),
        interpolation=tensor_vision.InterpolationMode.BICUBIC,
        antialias=True,
    )


def prepare_tensor_image(
    processor: ImageProcessor,
    kind: MediaCall,
    image: torch.Tensor,
    *,
    device: torch.device,
    signed_unit: bool,
) -> PreparedImage:
    """Apply the same canvas and tower policy to an already decoded RGB view.

    ``image`` is [3, height, width] or [1, 3, height, width] with values in
    [0, 1], or in [-1, 1] when ``signed_unit``; values are clamped to [0, 1]
    before the transforms. The view is a generated image rather than a
    request input, so a patch tower applies its single-image pixel bound.
    Continuous values resize with torch's antialiased bicubic filter
    whatever resampling the transform declares for 8-bit images.
    """
    value = image.detach().to(dtype=torch.float32)
    if value.ndim == 4:
        if int(value.shape[0]) != 1:
            raise invalid_descriptor(
                "generated image staging accepts one image"
            )
        value = value[0]
    if value.ndim != 3 or int(value.shape[0]) != 3:
        raise invalid_descriptor(
            "generated image tensor must have shape [3, height, width]"
        )
    if signed_unit:
        value = (value + 1.0) * 0.5
    value = value.clamp(0.0, 1.0)
    transform, canvas, tower = _image_plan(
        processor, kind, int(value.shape[1]), int(value.shape[2]), None
    )
    if not isinstance(transform, PatchTransform):
        value = _resize_tensor(value, *canvas)
    value = _resize_tensor(value, *tower)
    pixels = _normalize_tensor(value, transform.normalization)
    return _prepared_pixels(processor, transform, pixels, canvas, device)


def _decode_rgb(encoded: str, alpha: str) -> Image.Image:
    """Decode a base64 image payload and convert it to RGB.

    ``alpha`` is the processor's ``ImageProcessor.alpha`` policy for images
    with transparency.
    """
    if not isinstance(encoded, str) or not encoded:
        raise invalid_descriptor(
            "inline image payload must be non-empty base64"
        )
    try:
        raw = base64.b64decode(encoded, validate=True)
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (binascii.Error, OSError, ValueError) as error:
        raise invalid_descriptor(
            "inline image payload is not a valid encoded image"
        ) from error

    transparent = (
        image.mode == "RGBA" or image.info.get("transparency") is not None
    )
    if alpha == "white" and transparent:
        rgba = image.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, (255, 255, 255))
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        return rgb
    return image.convert("RGB")


def patch_grid_shape(
    transform: PatchTransform,
    source_height: int,
    source_width: int,
    input_images: int | None,
) -> tuple[int, int]:
    """Return the host-known patch grid for the declared image dimensions.

    ``input_images`` is the number of input images in the request of a
    request input image, or None for a generated image.

    Raises:
        WorkerError: An ``invalid_descriptor`` error when the transform's
            resize policy rejects the dimensions.
    """
    try:
        return transform.grid_shape(source_height, source_width, input_images)
    except ValueError as error:
        raise invalid_descriptor(str(error)) from error


def _stride_image_shape(
    height: int, width: int, processor: StrideResize
) -> tuple[int, int]:
    """Resolve both image dimensions at the configured spatial stride.

    Returns (height, width), while ``_stride_shape`` takes and returns
    (width, height).
    """
    scale = min(int(processor.max_size) / max(width, height), 1.0)
    scale = max(scale, int(processor.min_size) / min(width, height))
    new_width, new_height = _stride_shape(
        width, height, scale, int(processor.stride)
    )
    if new_width * new_height > int(processor.max_pixels):
        scale = int(processor.max_pixels) / (new_width * new_height)
        new_width, new_height = _stride_shape(
            new_width, new_height, scale, int(processor.stride)
        )
    if max(new_width, new_height) > int(processor.max_size):
        scale = int(processor.max_size) / max(new_width, new_height)
        new_width, new_height = _stride_shape(
            new_width, new_height, scale, int(processor.stride)
        )
    return new_height, new_width


def _stride_shape(
    width: int, height: int, scale: float, stride: int
) -> tuple[int, int]:
    """Round scaled dimensions to positive multiples of the required stride."""

    def align(value: float) -> int:
        """Round a scaled edge to the nearest positive model stride."""
        return max(stride, round(value / stride) * stride)

    return align(width * scale), align(height * scale)


def _normalize(values: np.ndarray, name: str) -> torch.Tensor:
    """Normalize ``uint8`` CHW RGB bytes into contiguous FP32 model inputs."""
    # PIL decoding and resizing produce host bytes. Keep the pointwise FP32
    # transform in one owned CHW array instead of dispatching each pass through
    # the process-wide tensor thread pool on the serving thread.
    pixels = values.astype(np.float32, order="C")
    if name == "unit":
        # One float32 multiplication by the float32 factor, bit for bit the
        # Transformers ``rescale`` of a uint8 tensor.
        pixels *= np.float32(_UNIT_SCALE)
        return torch.from_numpy(pixels)

    pixels /= np.float32(255.0)
    if name == "signed_unit":
        pixels -= np.float32(0.5)
        pixels /= np.float32(0.5)
    elif name == "imagenet":
        pixels -= np.asarray(_IMAGENET_MEAN, dtype=np.float32)[:, None, None]
        pixels /= np.asarray(_IMAGENET_STD, dtype=np.float32)[:, None, None]
    else:
        raise invalid_descriptor(f"unknown image normalization {name!r}")
    return torch.from_numpy(pixels)


def _normalize_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Apply the named normalization policy to an image tensor in [0, 1]."""
    if name == "unit":
        return tensor
    if name == "signed_unit":
        return (tensor - 0.5) / 0.5
    if name == "imagenet":
        mean = tensor.new_tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std = tensor.new_tensor(_IMAGENET_STD).view(3, 1, 1)
        return (tensor - mean) / std
    raise invalid_descriptor(f"unknown image normalization {name!r}")


def _resize_tensor(
    value: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Resize a batched tensor image with bicubic interpolation."""
    return F.interpolate(
        value.unsqueeze(0),
        size=(int(height), int(width)),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[0]


def _stage(
    value: torch.Tensor, processor: ImageProcessor, device: torch.device
) -> torch.Tensor:
    """Convert preprocessing output to the staging dtype and device."""
    dtype = processor.staging_dtype
    if processor.staging_dtype is not None and not isinstance(
        dtype, torch.dtype
    ):
        raise invalid_descriptor(
            f"unknown image staging dtype {processor.staging_dtype!r}"
        )
    return value.to(device=device, dtype=dtype, non_blocking=True)


__all__ = [
    "PreparedImage",
    "patch_grid_shape",
    "prepare_image",
    "prepare_tensor_image",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class VisionRow(InputRow):
    """Prepared pixels and optional patch-grid coordinates."""

    encode_pixels: torch.Tensor
    encode_grid: torch.Tensor | None = None
    encode_grid_shape: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodeRow(InputRow):
    """Latent sample and the image dimensions requested from its decoder."""

    latent: torch.Tensor
    image_height: int
    image_width: int
