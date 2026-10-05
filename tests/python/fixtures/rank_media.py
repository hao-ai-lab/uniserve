"""A real CPU Worker whose results carry test-selected media payloads."""

import os
from pathlib import Path

from tests.python.fixtures.depth_one import finalized_report
from uniserve_worker._uniserve_ipc import publish_media_bytes
from uniserve_worker.bootstrap.cli import parse_worker_args
from uniserve_worker.bootstrap.launch import (
    WorkerIpcEndpoint,
    endpoint_name,
    register_endpoint,
)
from uniserve_worker.protocol.output import MediaOutput, PosixShmArtifact
from uniserve_worker.worker import Worker


def main():
    config = parse_worker_args()
    service = endpoint_name(config)
    with WorkerIpcEndpoint(
        service,
        max_payload=config.ipc.max_payload_bytes,
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
                if kind == "submit":
                    report = finalized_report(
                        worker, worker.submit(request["batch"])
                    ).to_mapping()
                    media_case = os.environ.get("UNISERVE_TEST_MEDIA_RESPONSE")
                    media_rank = 1 if media_case == "rank-output" else 0
                    if (
                        media_case
                        and config.execution.rank == media_rank
                        and report["completions"]
                    ):
                        completion = report["completions"][0]
                        payload = b"generated media content"
                        name = publish_media_bytes(payload)
                        Path(os.environ["UNISERVE_TEST_MEDIA_NAME"]).write_text(
                            name
                        )
                        completion["media_output"] = MediaOutput(
                            handle=PosixShmArtifact(name),
                            bytes=len(payload)
                            + (1 if media_case == "short-storage" else 0),
                        ).to_mapping()
                        if media_case == "unknown-call":
                            completion["call_id"] = {
                                **completion["call_id"],
                                "request_index": 1000,
                            }
                    endpoint.respond(
                        {
                            "kind": "result",
                            "message_id": message_id,
                            "result": report,
                        }
                    )
                else:
                    raise ValueError(f"unsupported worker request {kind}")


if __name__ == "__main__":
    main()
