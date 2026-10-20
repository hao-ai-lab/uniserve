"""Exposes the video request transport and MP4 inspection."""

from .client import send_request
from .video import VideoOutputError, inspect_video_bytes

__all__ = [
    "VideoOutputError",
    "inspect_video_bytes",
    "send_request",
]
