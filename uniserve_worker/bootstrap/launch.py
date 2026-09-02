"""Process launch for one configured worker."""

from __future__ import annotations

import logging

from .config import WorkerProcessArgs

logger = logging.getLogger(__name__)


def run_worker(config: WorkerProcessArgs) -> None:
    """Open the IPC endpoint, create the worker, and serve until shutdown."""

    from ..process import WorkerProcess
    from .ipc import WorkerIpcEndpoint
    from ..worker import Worker

    endpoint = WorkerIpcEndpoint(
        config.ipc.service_name,
        max_payload=config.ipc.max_payload_bytes,
        max_inflight=config.ipc.max_inflight,
    )
    logger.info(
        "worker IPC endpoint opened",
        extra={
            "service": config.ipc.service_name,
            "max_payload_bytes": config.ipc.max_payload_bytes,
            "max_inflight": config.ipc.max_inflight,
            "worker_role": config.worker_role.value,
        },
    )
    worker = Worker.from_config(config)
    # Request admission begins only after the configured first-use kernel work succeeds
    # and the worker opens a clean serving collective epoch.
    worker.warmup()
    WorkerProcess(worker, endpoint).serve()


__all__ = ["run_worker"]
