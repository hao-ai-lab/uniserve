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

    The rank owns the name because only the rank knows which mechanism it can
    offer; a rank on the head's host offers a shared-memory service. The name
    is distinct across ranks and across successive launches of one rank.
    """
    return service_name(
        f"{os.getpid()}_{config.execution.rank}_{secrets.token_hex(8)}"
    )


def register_endpoint(config: WorkerProcessArgs, endpoint: str) -> None:
    """Report this rank's bound endpoint to the head's registration address.

    One JSON line carries the report; the connection carries nothing else and
    closes once the head has read it.
    """
    host, _, port = config.ipc.registration_address.rpartition(":")
    report = {
        "worker_id": config.worker_id,
        "rank": int(config.execution.rank),
        "transport": "iceoryx2",
        "endpoint": endpoint,
    }
    with socket.create_connection(
        (host, int(port)), timeout=REGISTRATION_TIMEOUT_SECONDS
    ) as connection:
        connection.sendall(json.dumps(report).encode("utf-8") + b"\n")


def run_worker(config: WorkerProcessArgs) -> None:
    """Own the IPC endpoint around model construction and the blocking run."""
    from ..worker import Worker

    endpoint_service = endpoint_name(config)
    with WorkerIpcEndpoint(
        endpoint_service,
        max_payload=config.ipc.max_payload_bytes,
        max_inflight=config.ipc.queue_depth,
    ) as endpoint:
        # The endpoint exists before it is named to anyone, so the head can
        # bind its side as soon as it reads the report.
        register_endpoint(config, endpoint_service)
        with Worker.from_config(config) as worker:
            worker.bind(endpoint)
            logger.info(
                "worker IPC endpoint bound",
                extra={
                    "service": endpoint_service,
                    "max_payload_bytes": config.ipc.max_payload_bytes,
                    "max_inflight": config.ipc.queue_depth,
                    "supported_ops": sorted(
                        value.value for value in config.supported_ops
                    ),
                },
            )
            worker.run()


__all__ = ["WorkerIpcEndpoint", "run_worker"]
