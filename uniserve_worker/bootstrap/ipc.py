"""Native worker IPC endpoint construction.

`_uniserve_ipc` is the PyO3 extension built from the `uniserve-ipc-py` crate.
``WorkerIpcEndpoint`` is the worker-side request/response endpoint.
"""

from __future__ import annotations

try:
    from .._uniserve_ipc import Server as WorkerIpcEndpoint
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "_uniserve_ipc is not installed; reinstall the package (pip install -e .) to build it."
    ) from exc


class EndpointBusyError(Exception):
    """Typed startup error for an IPC server endpoint that is already bound."""


__all__ = ["EndpointBusyError", "WorkerIpcEndpoint"]
