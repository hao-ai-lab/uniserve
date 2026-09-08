"""Process launch for one configured worker."""

from __future__ import annotations

import logging

from .config import WorkerProcessArgs

logger = logging.getLogger(__name__)


def run_worker(config: WorkerProcessArgs) -> None:
    """Open the IPC endpoint, create the worker, and serve until shutdown."""

    from ..process import WorkerProcess
    from ..worker import Worker
    from .ipc import WorkerIpcEndpoint

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
            "supported_ops": sorted(value.value for value in config.supported_ops),
        },
    )
    worker = Worker.from_config(config)
    worker.warmup()
    WorkerProcess(worker, endpoint).serve()


__all__ = ["run_worker"]
