"""Process launch for one configured worker."""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket

from .config import WorkerProcessArgs

logger = logging.getLogger(__name__)

# A rank reports as soon as its endpoint exists, well before any weights load,
# so this only has to cover the head accepting an already-bound connection.
REGISTRATION_TIMEOUT_SECONDS = 60

# A socket endpoint binds every interface so a rank on another host can be
# reached; the port is chosen by the system and reported once it exists.
SOCKET_BIND_INTERFACE = "0.0.0.0"
SOCKET_CHANNEL = "tcp"

try:
    from .._uniserve_ipc import Server as WorkerIpcEndpoint
    from .._uniserve_ipc import service_name
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
    shared-memory endpoint is named by a service distinct across ranks and
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

    One JSON line carries the report; the connection carries nothing else and
    closes once the head has read it. The endpoint is what the rank actually
    bound, which for a socket is the address rather than the interface it was
    given.

    A socket endpoint accepts on every interface, so what it bound names none
    of them and the head cannot dial it. This connection is the answer: its
    local end is an address of this host that the head, at the other end,
    routes to, so the report names the rank by that address and the port it
    bound.
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
    """Own the IPC endpoint around model construction and the blocking run."""
    from ..worker import Worker

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
