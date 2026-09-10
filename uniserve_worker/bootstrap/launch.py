"""Process launch for one configured worker."""

from __future__ import annotations

import logging

from .config import WorkerProcessArgs

logger = logging.getLogger(__name__)


def run_worker(config: WorkerProcessArgs) -> None:
    """Own the IPC endpoint around model construction and the worker's blocking run."""

    from ..worker import Worker
    from .ipc import WorkerIpcEndpoint

    with (
        WorkerIpcEndpoint(
            config.ipc.service_name,
            max_payload=config.ipc.max_payload_bytes,
            max_inflight=config.ipc.max_inflight,
        ) as endpoint,
        Worker.from_config(config) as worker,
    ):
        worker.bind(endpoint)
        logger.info(
            "worker IPC endpoint bound",
            extra={
                "service": config.ipc.service_name,
                "max_payload_bytes": config.ipc.max_payload_bytes,
                "max_inflight": config.ipc.max_inflight,
                "supported_ops": sorted(value.value for value in config.supported_ops),
            },
        )
        worker.run()


__all__ = ["run_worker"]
