"""Replicated deployments: profile declaration and client-side routing."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from uniserve_eval.config import load_config
from uniserve_eval.pipeline.run import run_point
from uniserve_eval.types import (
    VIDEOS_SYNC,
    BenchmarkPoint,
    LoadConfig,
    MetricDefinition,
    TaskName,
)

pytestmark = pytest.mark.unit


def _profile(tmp_path, servers: str):
    path = tmp_path / "profile.toml"
    path.write_text(f'root = "."\n{servers}')
    return load_config(path)


def test_replicas_inherit_the_shared_host_and_environment(tmp_path):
    config = _profile(
        tmp_path,
        """
[servers.engine]
host = "127.0.0.2"
environment = { SHARED = "1", DEVICES = "all" }

[[servers.engine.replicas]]
command = ["serve", "--port", "9001"]
port = 9001
environment = { DEVICES = "0,1" }

[[servers.engine.replicas]]
command = ["serve", "--port", "9002"]
port = 9002
environment = { DEVICES = "2,3" }
""",
    )

    server = config.servers["engine"]
    assert server.base_urls == (
        "http://127.0.0.2:9001",
        "http://127.0.0.2:9002",
    )
    assert [instance.environment for instance in server.instances] == [
        {"SHARED": "1", "DEVICES": "0,1"},
        {"SHARED": "1", "DEVICES": "2,3"},
    ]
    assert [instance.command[-1] for instance in server.instances] == [
        "9001",
        "9002",
    ]


def test_a_replica_may_listen_on_another_host(tmp_path):
    config = _profile(
        tmp_path,
        """
[servers.engine]

[[servers.engine.replicas]]
command = ["serve"]
port = 9001

[[servers.engine.replicas]]
command = ["ssh", "peer", "serve"]
host = "peer"
port = 9001
""",
    )

    assert config.servers["engine"].base_urls == (
        "http://127.0.0.1:9001",
        "http://peer:9001",
    )


@pytest.mark.parametrize(
    "ports, message",
    [((9001, 9001), "distinct ports"), ((9001,), "two or more")],
)
def test_replica_declarations_reject_ambiguous_deployments(
    tmp_path, ports, message
):
    replicas = "".join(
        f'[[servers.engine.replicas]]\ncommand = ["serve"]\nport = {port}\n'
        for port in ports
    )
    with pytest.raises(ValueError, match=message):
        _profile(tmp_path, f"[servers.engine]\n{replicas}")


def test_requests_go_to_the_replica_with_the_fewest_in_flight(tmp_path, media):
    """An idle deployment rotates; a busy replica is skipped for an idle one."""

    def replica(delay_for_slow: float):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                request = json.loads(
                    self.rfile.read(int(self.headers["content-length"]))
                )
                if request["prompt"] == "slow":
                    time.sleep(delay_for_slow)
                self.send_response(200)
                self.send_header("content-type", "video/mp4")
                self.send_header("content-length", str(len(media)))
                self.end_headers()
                self.wfile.write(media)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    servers = [replica(1.0), replica(0.0)]
    urls = [f"http://127.0.0.1:{server.server_port}" for server, _ in servers]
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"id": prompt, "prompt": prompt, "seconds": 4}) + "\n"
            for prompt in ("slow", "fast-1", "fast-2")
        )
    )
    point = BenchmarkPoint(
        name="replicas",
        server="engine",
        task=TaskName.VIDEO,
        model="h3",
        dataset="jsonl",
        dataset_path=str(manifest),
        endpoint=VIDEOS_SYNC,
        metrics=(MetricDefinition(("videos_per_second",), "higher"),),
        load=LoadConfig(num_prompts=3, max_concurrency=2, warmup_requests=0),
    )
    try:
        result = asyncio.run(run_point(urls, point, tmp_path / "result"))
    finally:
        for server, thread in servers:
            server.shutdown()
            server.server_close()
            thread.join()

    records = {
        record["request_id"]: record
        for record in map(
            json.loads,
            (result.output_dir / "requests.jsonl").read_text().splitlines(),
        )
    }
    # The idle deployment rotates through its replicas; the third request
    # arrives while the first replica still serves the slow one.
    assert records["slow"]["base_url"] == urls[0]
    assert records["fast-1"]["base_url"] == urls[1]
    assert records["fast-2"]["base_url"] == urls[1]
    assert all(record["success"] for record in records.values())
    assert result.summary["base_urls"] == urls
