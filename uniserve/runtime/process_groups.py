"""Native process-world and component-group ownership.

``initialize_process_groups`` selects the local device and joins the worker
world or shared expert union. ``ProcessGroups.bind`` binds numerical meshes
in collective order. Close the owner after its communication users retire;
aborted owners retain backend groups until the process exits.
"""

from uniserve_worker._uniserve_ipc import (
    ProcessGroups,
    Rendezvous,
    initialize_process_groups,
)

__all__ = ["ProcessGroups", "Rendezvous", "initialize_process_groups"]
