"""A rank whose teardown completes only once every rank of its group joins.

The rank serves the stub `Worker` until the head closes it, then waits at a
barrier over its process world before its worker releases anything, as the
finalization of a NCCL communicator waits for every member. A rank that
passes the barrier writes ``<rank>.closed`` to the directory
``UNISERVE_TEST_TEARDOWN_DIR`` names.
"""

import os
from pathlib import Path

import torch.distributed as dist

from uniserve_worker.bootstrap.cli import parse_worker_args
from uniserve_worker.bootstrap.launch import (
    WorkerIpcEndpoint,
    endpoint_name,
    register_endpoint,
)
from uniserve_worker.worker import Worker


def main() -> None:
    directory = Path(os.environ["UNISERVE_TEST_TEARDOWN_DIR"])
    config = parse_worker_args()
    name = endpoint_name(config)
    with WorkerIpcEndpoint(
        name,
        max_payload=config.ipc.max_payload_bytes,
        max_inflight=config.ipc.queue_depth,
        transport=config.ipc.channel_transport,
    ) as endpoint:
        register_endpoint(config, endpoint.endpoint(name))
        with Worker.from_config(config) as worker:
            worker.bind(endpoint)
            worker.run()

            # `run` returns once this rank has acknowledged Close. The world
            # group stays alive until the worker closes.
            dist.barrier()
            (directory / f"{config.execution.rank}.closed").write_text("")


if __name__ == "__main__":
    main()
