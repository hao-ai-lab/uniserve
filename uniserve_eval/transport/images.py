"""Decodes and validates generated images returned by public APIs.

Only embedded images are accepted: a ``b64_json`` field or a base64 ``data:``
URL. Validation failures raise ``ImageOutputError``;
``uniserve_eval.transport.client`` records its ``classifier`` as the request's
failure classifier. ``uniserve_eval.artifacts`` also calls
``inspect_image_bytes`` to re-verify image samples before writing them.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from collections.abc import Sequence
from io import BytesIO
from typing import Any

from PIL import Image, UnidentifiedImageError

from ..types import DecodedImage


class ImageOutputError(ValueError):
    """Carries a stable classifier for invalid generated image output.

    Attributes:
        classifier: Stable ``response_*`` failure label, also the message.
    """

    def __init__(self, classifier: str) -> None:
        """Initialize the error with its external classifier."""
        super().__init__(classifier)
        self.classifier = classifier


# Pillow format names mapped to sample file extensions. The mapping is also the
# allowlist of supported formats: an image Pillow decodes in any other format
# raises `response_unsupported_image_format`.
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
    """Decode one embedded OpenAI-compatible image response part.

    A non-empty ``b64_json`` string takes precedence. Otherwise
    ``image_url`` is read either as a string or as ``{"url": ...}`` and must
    be a base64 ``data:`` URL whose declared MIME type then has to match the
    decoded format.

    Raises:
        ImageOutputError: ``response_missing_image_payload`` when neither
            field holds a non-empty string,
            ``response_image_payload_not_embedded`` or
            ``response_invalid_image_data_url`` for a URL that is not an
            image base64 data URL, ``response_invalid_image_base64`` or
            ``response_empty_image_payload`` for invalid or empty base64
            data, or any classifier raised by ``inspect_image_bytes``.
    """
    encoded: Any = part.get("b64_json")
    declared_mime: str | None = None
    if not isinstance(encoded, str) or not encoded:
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if not isinstance(image_url, str) or not image_url:
            raise ImageOutputError("response_missing_image_payload")
        declared_mime, encoded = _parse_image_data_url(image_url)
    # `validate=True` rejects characters outside the base64 alphabet instead
    # of silently discarding them.
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError) as error:
        raise ImageOutputError("response_invalid_image_base64") from error
    if not payload:
        raise ImageOutputError("response_empty_image_payload")
    return inspect_image_bytes(payload, declared_mime=declared_mime)


def decode_openai_image_parts(
    parts: Sequence[dict[str, Any]],
) -> list[DecodedImage]:
    """Decode an ordered collection of embedded image response parts."""
    return [decode_openai_image_part(part) for part in parts]


def inspect_image_bytes(
    data: bytes, *, declared_mime: str | None = None
) -> DecodedImage:
    """Decode image bytes fully and return verified content metadata.

    Args:
        data: Encoded image file bytes.
        declared_mime: MIME type the producer claimed, if any; it must match
            the decoded format after alias normalization.

    Returns:
        A ``DecodedImage`` whose ``sample_filename`` is content-addressed by
        the SHA-256 of ``data``.

    Raises:
        ImageOutputError: ``response_undecodable_image``,
            ``response_unsupported_image_format``, or
            ``response_image_mime_mismatch``.
    """
    # Pillow requires reopening an image after `verify()`, so the structural
    # check and the full pixel decode use separate opens.
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
    if (
        declared_mime is not None
        and _normalize_mime(declared_mime) != actual_mime
    ):
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
    """Split a base64 image data URL into MIME type and encoded payload."""
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
    """Normalize supported image MIME aliases."""
    mime = value.strip().lower()
    return _MIME_ALIASES.get(mime, mime)
