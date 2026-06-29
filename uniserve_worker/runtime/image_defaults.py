"""Default generated-image dimensions for worker fallback paths.

These MUST mirror the authoritative ``ImageParams::default()`` in
``crates/foundation/core/src/lib.rs`` (height=512, width=512). The Rust
scheduler always populates ``ImageParams.height``/``width`` (non-Optional
``u32``) before sending the op to the worker, so on the live path the fallback
below never fires. Keeping these values aligned avoids a divergent second
source of truth: if dims ever fail to round-trip (older op shape, a path that
omits the field), the worker synthesizes the same 512x512 latent the host
budgeted for rather than a capacity-busting 2048x1152 one
(2048x1152 at latent_downsample=16 = 9216 latent tokens > the worker's default
max_latent_size of 4096).
"""
from __future__ import annotations

__all__ = [
    'SENSENOVA_DEFAULT_WIDTH',
    'SENSENOVA_DEFAULT_HEIGHT',
    'image_width',
    'image_height',
]

# Mirror crates/foundation/core/src/lib.rs ImageParams::default().
SENSENOVA_DEFAULT_WIDTH = 512
SENSENOVA_DEFAULT_HEIGHT = 512


def image_width(params: dict | None) -> int:
    value = (params or {}).get("width")
    return int(value if value is not None else SENSENOVA_DEFAULT_WIDTH)


def image_height(params: dict | None) -> int:
    value = (params or {}).get("height")
    return int(value if value is not None else SENSENOVA_DEFAULT_HEIGHT)
