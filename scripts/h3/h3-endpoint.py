#!/usr/bin/env python3
"""Validate the selected contract and publish only a ready, single-model endpoint."""

import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


def main():
    entry = os.environ["ENTRY"]
    model = f"minimax-h3-{entry}"
    path = Path(os.environ["ENDPOINT_FILE"])
    job = os.environ["SLURM_JOB_ID"]
    action = sys.argv[1]
    if action == "validate":
        from uniserve_worker.bootstrap.inspect_model import inspect_model

        contract = inspect_model(os.environ["CHECKPOINT_ROOT"])["contract"]
        expected = {
            "base": ("base", 49, "dense"),
            "8step": ("fasth3", 8, "vsa"),
            "ref": ("ref", 49, "dense"),
        }[entry]
        actual = (contract["variant"], contract["denoise_steps"], contract["attention"])
        if actual != expected:
            raise ValueError(f"ENTRY {entry} requires {expected}, checkpoint resolves {actual}")
        if entry == "8step" and (
            contract["attention_backend"] != "VIDEO_SPARSE_ATTN_H3"
            or contract["ladder"] != [999, 874, 749, 624, 500, 375, 250, 125]
            or contract["sigma_shifts"] != [10.0, 3.0]
            or contract["sparsity"] != 0.8
        ):
            raise ValueError("8step requires the VSA-H3 sparsity-0.8 shift-10 release recipe")
        print(json.dumps(contract, sort_keys=True))
        return
    if action == "remove":
        try:
            record = json.loads(path.read_text())
            if record["slurm_job_id"] == job:
                path.unlink()
        except FileNotFoundError:
            pass
        return
    if action != "publish":
        raise ValueError("expected validate, publish, or remove")

    key = Path(os.environ["KEY_FILE"]).read_text().strip()
    port = int(os.environ["PORT"])
    headers = {"Authorization": f"Bearer {key}"}
    # The parent owns server lifetime; this observer never restarts or stops it.
    while True:
        try:
            for route in ("/health", "/v1/models"):
                request = urllib.request.Request(f"http://127.0.0.1:{port}{route}", headers=headers)
                with urllib.request.urlopen(request, timeout=5) as response:
                    body = response.read()
                if route == "/v1/models":
                    models = [item["id"] for item in json.loads(body)["data"]]
                    if models != [model]:
                        raise ValueError(f"Expected sole served model {model}, got {models}")
            break
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    record = {
        "id": path.stem,
        "kind": "uniserve",
        "name": f"UniServe H3 {entry} rack 3",
        "base_url": f"http://{os.environ['ADVERTISE_HOST']}:{port}",
        "health_path": "/health",
        "metrics_path": "/metrics",
        "api_key_file": os.environ["KEY_FILE"],
        "slurm_job_id": job,
        "expires_utc": os.environ["EXPIRES_UTC"],
        "models": models,
    }
    if os.environ.get("RUN_ID"):
        record["run_id"] = os.environ["RUN_ID"]
    temporary = path.with_suffix(f".{job}.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)
    print(f"Ready: {path}", flush=True)


if __name__ == "__main__":
    main()
