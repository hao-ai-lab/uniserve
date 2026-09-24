"""Process launch for one configured worker rank.

A rank opens its IPC channel endpoint, reports that endpoint to the head over a
one-shot TCP connection to the registration address, and only then builds the
``Worker`` (model loading, process groups, resource allocation) and runs its
blocking serve loop. The endpoint mechanism is the one the launch descriptor's
``channel_transport`` names: shared storage for a rank on the head's host, a
TCP socket for a rank elsewhere.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket

from uniserve_worker.config.deployment import WorkerProcessArgs

logger = logging.getLogger(__name__)

# A rank reports as soon as its endpoint exists, well before any weights load,
# so this only has to cover the head accepting an already-bound connection.
REGISTRATION_TIMEOUT_SECONDS = 60

# A socket endpoint binds every interface so a rank on another host can be
# reached; the port is chosen by the system and reported once it exists.
SOCKET_BIND_INTERFACE = "0.0.0.0"
SOCKET_CHANNEL = "tcp"

try:
    from uniserve_worker._uniserve_ipc import Server as WorkerIpcEndpoint
    from uniserve_worker._uniserve_ipc import service_name
except (
    ImportError
) as exc:  # pragma: no cover - depends on the installed native extension.
    raise ImportError(
        "_uniserve_ipc is not installed; reinstall the package "
        "(pip install -e .) to build it."
    ) from exc


def endpoint_name(config: WorkerProcessArgs) -> str:
    """Name this rank's channel endpoint.

    The placement decides the mechanism and the rank names the endpoint. A
    shared-storage endpoint is named by a service distinct across ranks and
    across successive launches of one rank; a socket endpoint is named by the
    address it binds, so the rank offers the interface to bind on and reports
    the address that binding produced.
    """
    if config.ipc.channel_transport == SOCKET_CHANNEL:
        return SOCKET_BIND_INTERFACE
    return service_name(
        f"{os.getpid()}_{config.execution.rank}_{secrets.token_hex(8)}"
    )


def register_endpoint(config: WorkerProcessArgs, endpoint: str) -> None:
    """Report this rank's bound endpoint to the head's registration address.

    One JSON line carries the report (``worker_id``, ``rank``, ``transport``
    and ``endpoint``); the connection carries nothing else and closes after
    the write. ``endpoint`` is what the channel actually bound.

    A socket endpoint binds every interface, so its bound address names no
    interface the head can dial. The report therefore replaces the host part
    with this connection's local address, which is an address of this host
    that the head routes to, and keeps the bound port.

    A connection failure, including a connect or write that exceeds
    ``REGISTRATION_TIMEOUT_SECONDS``, raises ``OSError``.
    """
    host, _, port = config.ipc.registration_address.rpartition(":")
    with socket.create_connection(
        (host, int(port)), timeout=REGISTRATION_TIMEOUT_SECONDS
    ) as connection:
        if config.ipc.channel_transport == SOCKET_CHANNEL:
            _, bound_port = endpoint.rsplit(":", 1)
            endpoint = f"{connection.getsockname()[0]}:{bound_port}"
        report = {
            "worker_id": config.worker_id,
            "rank": int(config.execution.rank),
            "transport": config.ipc.channel_transport,
            "endpoint": endpoint,
        }
        connection.sendall(json.dumps(report).encode("utf-8") + b"\n")


def run_worker(config: WorkerProcessArgs) -> None:
    """Own the IPC endpoint around model construction and the blocking run.

    Returns when ``Worker.run`` returns. Every exit, including an exception
    from registration, construction or the run, closes the endpoint, and a
    constructed worker is closed before it.
    """
    from uniserve_worker.worker import Worker

    endpoint_service = endpoint_name(config)
    with WorkerIpcEndpoint(
        endpoint_service,
        max_payload=config.ipc.max_payload_bytes,
        max_inflight=config.ipc.queue_depth,
        transport=config.ipc.channel_transport,
    ) as endpoint:
        # The endpoint exists before it is named to anyone, so the head can
        # bind its side as soon as it reads the report. A socket reports the
        # address it bound rather than the interface it was given.
        register_endpoint(config, endpoint.endpoint(endpoint_service))
        with Worker.from_config(config) as worker:
            worker.bind(endpoint)
            logger.info(
                "worker IPC endpoint bound",
                extra={
                    "service": endpoint_service,
                    "max_payload_bytes": config.ipc.max_payload_bytes,
                    "max_inflight": config.ipc.queue_depth,
                    "supported_calls": sorted(
                        value.value for value in config.supported_calls
                    ),
                },
            )
            worker.run()


__all__ = ["WorkerIpcEndpoint", "run_worker"]
