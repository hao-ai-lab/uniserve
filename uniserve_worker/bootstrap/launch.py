"""Native endpoint registration and lifecycle for one configured rank."""

from uniserve_worker._uniserve_ipc import Server as WorkerIpcEndpoint
from uniserve_worker._uniserve_ipc import (
    endpoint_name,
    register_endpoint,
    run_worker,
)

__all__ = [
    "WorkerIpcEndpoint",
    "endpoint_name",
    "register_endpoint",
    "run_worker",
]
