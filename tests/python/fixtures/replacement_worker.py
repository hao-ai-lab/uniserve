"""An external rank whose replacement advertises a different capability."""

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from uniserve_worker.bootstrap.cli import parse_worker_args
from uniserve_worker.bootstrap.launch import (
    WorkerIpcEndpoint,
    endpoint_name,
    register_endpoint,
)
from uniserve_worker.worker import Worker


def run(directory: Path) -> None:
    # WorkerGroup invokes its Python executable with -m and the worker module.
    sys.argv = [sys.argv[0], *sys.argv[3:]]
    config = parse_worker_args()
    (directory / f"{config.execution.rank}.pid").write_text(str(os.getpid()))
    name = endpoint_name(config)
    with WorkerIpcEndpoint(
        name,
        max_payload=config.ipc.max_payload_bytes,
        max_inflight=config.ipc.queue_depth,
        transport=config.ipc.channel_transport,
    ) as endpoint:
        register_endpoint(config, endpoint.endpoint(name))

        class Rank(Worker):
            @property
            def info(self):
                value = super().info
                change = directory / "replacement.json"
                return (
                    replace(value, **json.loads(change.read_text()))
                    if change.exists()
                    else value
                )

        with Rank.from_config(config) as worker:
            worker.bind(endpoint)
            worker.run()
