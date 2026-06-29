"""Native worker IPC bridge.

`_uniserve_ipc` is the PyO3 extension built from the `uniserve-ipc-py` crate and
installed as a submodule of this package. The worker reaches the Rust
shared-memory transport through `Server`.
"""
from __future__ import annotations

try:
    from .. import _uniserve_ipc as _native
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "_uniserve_ipc is not installed; reinstall the package (pip install -e .) to build it."
    ) from exc

Server = _native.Server


class EndpointBusyError(Exception):
    """The IPC server endpoint is already bound (a startup race).

    Raised at this IPC seam instead of leaving callers to sniff the message
    text; the serve loop catches this type to back off and retry rather than
    treating it as a fatal transport failure.
    """


__all__ = ["Server", "EndpointBusyError"]
