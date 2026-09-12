"""Endpoint lifecycle behavior against an isolated HTTP service (no GPUs)."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

HELPER = Path(__file__).with_name("h3-endpoint.py")


class EndpointTests(unittest.TestCase):
    def test_publish_exact_model_and_remove_only_own_record(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.headers.get("Authorization") != "Bearer test-token":
                    self.send_error(401)
                    return
                self.send_response(200)
                self.end_headers()
                payload = {"data": [{"id": "minimax-h3-base"}]} if self.path == "/v1/models" else {}
                self.wfile.write(json.dumps(payload).encode())

            def log_message(self, *args):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            endpoint = root / "h3-base-rack3.json"
            key = root / "key"
            key.write_text("test-token\n")
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            env = {
                **os.environ,
                "ENTRY": "base",
                "ENDPOINT_FILE": str(endpoint),
                "SLURM_JOB_ID": "123",
                "KEY_FILE": str(key),
                "PORT": str(server.server_port),
                "ADVERTISE_HOST": "127.0.0.1",
                "EXPIRES_UTC": "2030-01-01T00:00:00Z",
                "RUN_ID": "test-run",
            }
            try:
                subprocess.run([sys.executable, HELPER, "publish"], env=env, check=True, timeout=10)
                record = json.loads(endpoint.read_text())
                self.assertEqual(record["models"], ["minimax-h3-base"])
                self.assertEqual(record["kind"], "uniserve")
                self.assertEqual(record["api_key_file"], str(key))
                self.assertEqual(record["health_path"], "/health")
                self.assertEqual(record["metrics_path"], "/metrics")
                self.assertEqual(record["slurm_job_id"], "123")
                self.assertEqual(record["run_id"], "test-run")
                subprocess.run(
                    [sys.executable, HELPER, "remove"], env={**env, "SLURM_JOB_ID": "456"}, check=True
                )
                self.assertEqual(json.loads(endpoint.read_text()), record)
                subprocess.run([sys.executable, HELPER, "remove"], env=env, check=True)
                self.assertFalse(endpoint.exists())
            finally:
                server.shutdown()
                thread.join()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
