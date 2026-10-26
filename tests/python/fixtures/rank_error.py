"""A real CPU Worker that answers its first batch with an execution error.

Every later batch runs normally. After answering the first batch, each rank
writes ``<UNISERVE_TEST_ANSWERED>-<rank>`` so a test can submit its next
batch only once that error is waiting to be drained.
"""

import os
from pathlib import Path

from tests.python.fixtures.depth_one import finalized_report
from uniserve_worker.bootstrap.cli import parse_worker_args
from uniserve_worker.bootstrap.launch import (
    WorkerIpcEndpoint,
    endpoint_name,
    register_endpoint,
)
from uniserve_worker.errors import resource_error
from uniserve_worker.protocol.messages import error_response
from uniserve_worker.worker import Worker


def main():
    config = parse_worker_args()
    service = endpoint_name(config)
    answered = False
    with WorkerIpcEndpoint(
        service,
        max_inflight=config.ipc.queue_depth,
    ) as endpoint:
        register_endpoint(config, service)
        with Worker.from_config(config) as worker:
            worker.warmup()
            while True:
                request = endpoint.recv()
                kind = request["kind"]
                message_id = request["message_id"]
                if kind == "info":
                    endpoint.respond(
                        {
                            "kind": "info",
                            "message_id": message_id,
                            "info": worker.info.to_mapping(),
                        }
                    )
                    continue
                if kind == "close":
                    endpoint.respond({"kind": "ok", "message_id": message_id})
                    break
                if kind != "submit":
                    raise ValueError(f"unsupported worker request {kind}")
                if not answered:
                    # A non-fatal execution error of the batch, as a request
                    # running out of device memory reports it.
                    endpoint.respond(
                        error_response(
                            resource_error("injected execution error"),
                            request,
                        )
                    )
                    answered = True
                    Path(
                        f"{os.environ['UNISERVE_TEST_ANSWERED']}"
                        f"-{config.execution.rank}"
                    ).touch()
                    continue
                report = finalized_report(
                    worker, worker.submit(request["batch"])
                ).to_mapping()
                endpoint.respond(
                    {
                        "kind": "result",
                        "message_id": message_id,
                        "result": report,
                    }
                )


if __name__ == "__main__":
    main()
