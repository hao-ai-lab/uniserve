"""Process launch for one configured worker."""

from __future__ import annotations

import logging

from .config import WorkerLaunchConfig

logger = logging.getLogger(__name__)


def run_worker(config: WorkerLaunchConfig) -> None:
    """Open the IPC endpoint, create the worker, and serve until shutdown."""

    from ..server.app import WorkerServer
    from ..server.ipc import WorkerIpcEndpoint
    from ..worker import MediaWorker, Worker
    from ..worker.media_worker import is_h3_checkpoint

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
            "worker_kind": config.worker_kind.value,
        },
    )
    model_path = "" if config.model is None else config.model.path
    worker = (
        MediaWorker.from_config(config)
        if model_path and is_h3_checkpoint(model_path)
        else Worker.from_config(config)
    )
    # Admission begins only after the configured first-use kernel work succeeds
    # and the worker opens a clean serving collective epoch.
    worker.warmup()
    WorkerServer(worker, endpoint).serve()


__all__ = ["run_worker"]
