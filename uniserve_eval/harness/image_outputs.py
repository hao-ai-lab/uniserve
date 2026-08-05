"""Decode and describe generated images returned by OpenAI-compatible APIs.

The harness persists the exact response bytes separately from request JSON. This
module is the single validation boundary for image-generation responses: base64
must be strict, a declared data-URL MIME type must agree with the decoded image,
and Pillow must be able to verify and fully decode the image.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal, Sequence

from PIL import Image, UnidentifiedImageError


class ImageOutputError(ValueError):
    """A generated image failed protocol or decoding validation."""

    def __init__(self, classifier: str) -> None:
        super().__init__(classifier)
        self.classifier = classifier


@dataclass(frozen=True)
class DecodedImage:
    """Validated image bytes plus compact, reproducible metadata."""

    data: bytes
    sha256: str
    byte_size: int
    mime: str
    width: int
    height: int
    sample_filename: str

    def metadata_dict(self) -> dict[str, int | str]:
        return {
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "mime": self.mime,
            "width": self.width,
            "height": self.height,
            "sample_filename": self.sample_filename,
        }


ImageOutputMode = Literal["none", "optional", "required"]


@dataclass(frozen=True)
class ImageOutputContract:
    """Image-output obligations declared by one emitted request."""

    mode: ImageOutputMode
    count: int | None
    count_is_cap: bool
    width: int | None
    height: int | None


_FORMAT_EXTENSIONS = {
    "PNG": "png",
    "JPEG": "jpg",
    "WEBP": "webp",
    "GIF": "gif",
    "BMP": "bmp",
    "TIFF": "tiff",
}
_MIME_ALIASES = {
    "image/jpg": "image/jpeg",
    "image/x-png": "image/png",
}


def decode_openai_image_part(part: dict[str, Any]) -> DecodedImage:
    """Decode one ``b64_json`` or chat ``image_url`` response part."""
    encoded: Any = part.get("b64_json")
    declared_mime: str | None = None
    if not isinstance(encoded, str) or not encoded:
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if not isinstance(image_url, str) or not image_url:
            raise ImageOutputError("protocol_missing_image_payload")
        declared_mime, encoded = _parse_image_data_url(image_url)
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError) as error:
        raise ImageOutputError("protocol_invalid_image_base64") from error
    if not payload:
        raise ImageOutputError("protocol_empty_image_payload")
    return inspect_image_bytes(payload, declared_mime=declared_mime)


def decode_openai_image_parts(parts: Sequence[dict[str, Any]]) -> list[DecodedImage]:
    """Decode all response parts without silently dropping malformed entries."""
    return [decode_openai_image_part(part) for part in parts]


def inspect_image_bytes(data: bytes, *, declared_mime: str | None = None) -> DecodedImage:
    """Verify exact bytes and return their content-addressed description."""
    try:
        with Image.open(BytesIO(data)) as image:
            image_format = image.format
            width, height = image.size
            image.verify()
        with Image.open(BytesIO(data)) as image:
            image.load()
    except (
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as error:
        raise ImageOutputError("protocol_undecodable_image") from error
    if not isinstance(image_format, str) or width <= 0 or height <= 0:
        raise ImageOutputError("protocol_undecodable_image")
    actual_mime = Image.MIME.get(image_format)
    extension = _FORMAT_EXTENSIONS.get(image_format)
    if not isinstance(actual_mime, str) or extension is None:
        raise ImageOutputError("protocol_unsupported_image_format")
    actual_mime = _normalize_mime(actual_mime)
    if declared_mime is not None and _normalize_mime(declared_mime) != actual_mime:
        raise ImageOutputError("protocol_image_mime_mismatch")
    digest = hashlib.sha256(data).hexdigest()
    return DecodedImage(
        data=data,
        sha256=digest,
        byte_size=len(data),
        mime=actual_mime,
        width=width,
        height=height,
        sample_filename=f"{digest}.{extension}",
    )


def image_output_contract(
    payload: dict[str, Any], *, request_kind: str, task: str
) -> ImageOutputContract:
    """Extract generated-image obligations without inspecting input-image properties."""
    modalities = payload.get("modalities")
    image_enabled = request_kind == "images_generations" or (
        isinstance(modalities, list) and "image" in modalities
    )
    if not image_enabled:
        return ImageOutputContract("none", None, False, None, None)
    mode: ImageOutputMode = "optional" if task == "interleave" else "required"
    image_config = payload.get("image_config")
    image = image_config if isinstance(image_config, dict) else {}
    count = _positive_int(
        image.get(
            "num_images",
            payload.get("n", payload.get("num_outputs_per_prompt")),
        )
    )
    width = _positive_int(image.get("width", payload.get("width")))
    height = _positive_int(image.get("height", payload.get("height")))
    size = payload.get("size")
    if (width is None or height is None) and isinstance(size, str):
        parsed = _parse_size(size)
        if parsed is not None:
            width = width or parsed[0]
            height = height or parsed[1]
    return ImageOutputContract(
        mode,
        count,
        request_kind == "openai_chat",
        width,
        height,
    )


def image_output_mismatch(
    images: Sequence[DecodedImage], contract: ImageOutputContract
) -> str | None:
    """Return the first declared-work mismatch, if any."""
    if contract.mode == "required" and not images:
        return "protocol_missing_decoded_image"
    if (
        contract.count is not None
        and contract.count_is_cap
        and len(images) > contract.count
    ):
        return "protocol_image_count_mismatch"
    if (
        contract.count is not None
        and not contract.count_is_cap
        and len(images) != contract.count
    ):
        return "protocol_image_count_mismatch"
    if contract.width is not None and any(
        image.width != contract.width for image in images
    ):
        return "protocol_image_dimensions_mismatch"
    if contract.height is not None and any(
        image.height != contract.height for image in images
    ):
        return "protocol_image_dimensions_mismatch"
    return None


def _parse_image_data_url(value: str) -> tuple[str, str]:
    if not value.startswith("data:") or "," not in value:
        raise ImageOutputError("protocol_image_payload_not_embedded")
    header, encoded = value[5:].split(",", maxsplit=1)
    segments = header.split(";")
    mime = segments[0].lower()
    if not mime.startswith("image/") or "base64" not in segments[1:]:
        raise ImageOutputError("protocol_invalid_image_data_url")
    if not encoded:
        raise ImageOutputError("protocol_empty_image_payload")
    return mime, encoded


def _normalize_mime(value: str) -> str:
    mime = value.strip().lower()
    return _MIME_ALIASES.get(mime, mime)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _parse_size(value: str) -> tuple[int, int] | None:
    pieces = value.lower().split("x", maxsplit=1)
    if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
        return None
    width, height = (int(piece) for piece in pieces)
    return (width, height) if width > 0 and height > 0 else None
