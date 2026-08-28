"""Decode and validate generated images returned by public APIs."""

from __future__ import annotations

import base64
import binascii
import hashlib
from io import BytesIO
from typing import Any, Sequence

from PIL import Image, UnidentifiedImageError

from ..types import DecodedImage


class ImageOutputError(ValueError):
    def __init__(self, classifier: str) -> None:
        super().__init__(classifier)
        self.classifier = classifier


_FORMAT_EXTENSIONS = {
    "PNG": "png",
    "JPEG": "jpg",
    "WEBP": "webp",
    "GIF": "gif",
    "BMP": "bmp",
    "TIFF": "tiff",
}
_MIME_ALIASES = {"image/jpg": "image/jpeg", "image/x-png": "image/png"}


def decode_openai_image_part(part: dict[str, Any]) -> DecodedImage:
    encoded: Any = part.get("b64_json")
    declared_mime: str | None = None
    if not isinstance(encoded, str) or not encoded:
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if not isinstance(image_url, str) or not image_url:
            raise ImageOutputError("response_missing_image_payload")
        declared_mime, encoded = _parse_image_data_url(image_url)
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError) as error:
        raise ImageOutputError("response_invalid_image_base64") from error
    if not payload:
        raise ImageOutputError("response_empty_image_payload")
    return inspect_image_bytes(payload, declared_mime=declared_mime)


def decode_openai_image_parts(parts: Sequence[dict[str, Any]]) -> list[DecodedImage]:
    return [decode_openai_image_part(part) for part in parts]


def inspect_image_bytes(data: bytes, *, declared_mime: str | None = None) -> DecodedImage:
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
        raise ImageOutputError("response_undecodable_image") from error
    if not isinstance(image_format, str) or width <= 0 or height <= 0:
        raise ImageOutputError("response_undecodable_image")
    actual_mime = Image.MIME.get(image_format)
    extension = _FORMAT_EXTENSIONS.get(image_format)
    if not isinstance(actual_mime, str) or extension is None:
        raise ImageOutputError("response_unsupported_image_format")
    actual_mime = _normalize_mime(actual_mime)
    if declared_mime is not None and _normalize_mime(declared_mime) != actual_mime:
        raise ImageOutputError("response_image_mime_mismatch")
    checksum = hashlib.sha256(data).hexdigest()
    return DecodedImage(
        data=data,
        sha256=checksum,
        byte_size=len(data),
        mime=actual_mime,
        width=width,
        height=height,
        sample_filename=f"{checksum}.{extension}",
    )


def _parse_image_data_url(value: str) -> tuple[str, str]:
    if not value.startswith("data:") or "," not in value:
        raise ImageOutputError("response_image_payload_not_embedded")
    header, encoded = value[5:].split(",", maxsplit=1)
    segments = header.split(";")
    mime = segments[0].lower()
    if not mime.startswith("image/") or "base64" not in segments[1:]:
        raise ImageOutputError("response_invalid_image_data_url")
    if not encoded:
        raise ImageOutputError("response_empty_image_payload")
    return mime, encoded


def _normalize_mime(value: str) -> str:
    mime = value.strip().lower()
    return _MIME_ALIASES.get(mime, mime)
