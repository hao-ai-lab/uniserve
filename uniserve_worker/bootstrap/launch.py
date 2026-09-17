"""Process launch for one configured worker."""

from __future__ import annotations

import logging

from .config import WorkerProcessArgs

logger = logging.getLogger(__name__)

try:
    from .._uniserve_ipc import Server as WorkerIpcEndpoint
except (
    ImportError
) as exc:  # pragma: no cover - depends on the installed native extension.
    raise ImportError(
        "_uniserve_ipc is not installed; reinstall the package "
        "(pip install -e .) to build it."
    ) from exc


def run_worker(config: WorkerProcessArgs) -> None:
    """Own the IPC endpoint around model construction and the blocking run."""
    from ..worker import Worker

    with (
        WorkerIpcEndpoint(
            config.ipc.service_name,
            max_payload=config.ipc.max_payload_bytes,
            max_inflight=config.ipc.queue_depth,
        ) as endpoint,
        Worker.from_config(config) as worker,
    ):
        worker.bind(endpoint)
        logger.info(
            "worker IPC endpoint bound",
            extra={
                "service": config.ipc.service_name,
                "max_payload_bytes": config.ipc.max_payload_bytes,
                "max_inflight": config.ipc.queue_depth,
                "supported_ops": sorted(
                    value.value for value in config.supported_ops
                ),
            },
        )
        worker.run()


__all__ = ["WorkerIpcEndpoint", "run_worker"]
