"""UniServe overlay for selected FlashAttention CUTE modules.

The overlay shadows a small set of FA4 Python sources while delegating every
other ``flash_attn.cute`` import back to the installed ``flash-attn-4`` package.
"""
from __future__ import annotations

from importlib import metadata
from pathlib import Path


def _append_provider_cute_path() -> None:
    try:
        dist = metadata.distribution("flash-attn-4")
    except metadata.PackageNotFoundError:
        return
    for entry in dist.files or ():
        if entry.as_posix() == "flash_attn/cute/interface.py":
            provider_cute = Path(dist.locate_file(entry)).parent
            provider_path = str(provider_cute)
            if provider_path not in __path__:
                __path__.append(provider_path)
            return


_append_provider_cute_path()
