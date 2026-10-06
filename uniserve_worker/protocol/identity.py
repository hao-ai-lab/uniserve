"""Native request epochs, scheduler call ordinals and buffer generations."""

from typing import TypeAlias

from uniserve_worker._uniserve_ipc import BufferId as BufferId
from uniserve_worker._uniserve_ipc import CallId as CallId
from uniserve_worker._uniserve_ipc import RequestKey as RequestKey

CallIdentity: TypeAlias = tuple[RequestKey, CallId]
