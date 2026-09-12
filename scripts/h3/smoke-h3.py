#!/usr/bin/env python3
"""Exercise a published H3 endpoint once and save its text-only audiovisual MP4."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.request

DEPLOY = Path("/mnt/lustre/vlm-wlsaidhi/src/uniserve-deploy")
OUTPUT = Path("/mnt/lustre/vlm-wlsaidhi/fastvideo/eval/uniserve-smoke")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", required=True, choices=("base", "8step"))
    args = parser.parse_args()
    endpoint = json.loads((DEPLOY / "endpoints" / f"h3-{args.entry}-rack3.json").read_text())
    model = f"minimax-h3-{args.entry}"
    if endpoint["models"] != [model]:
        raise ValueError("Endpoint must advertise exactly the requested model")
    if datetime.fromisoformat(endpoint["expires_utc"].replace("Z", "+00:00")) <= datetime.now(timezone.utc):
        raise ValueError("Endpoint allocation has expired")
    key = Path(endpoint["api_key_file"]).read_text().strip()
    headers = {"Authorization": f"Bearer {key}"}
    origin = endpoint["base_url"].rstrip("/")
    timings = {}
    for route in ("/v1/models", endpoint["health_path"]):
        start = time.perf_counter()
        request = urllib.request.Request(origin + route, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
        timings[route] = time.perf_counter() - start
        if route == "/v1/models" and [item["id"] for item in json.loads(body)["data"]] != [model]:
            raise ValueError("Server does not expose the requested single-model contract")

    # H3 rounds 5 seconds at 24 FPS up to its 17n+5 frame lattice: 124 frames.
    payload = {
        "model": model,
        "prompt": "A calm sea at sunrise, with gentle waves and natural ocean sound.",
        "seconds": 5,
        "seed": 0,
    }
    if args.entry == "base":
        payload["steps"] = 50
    directory = OUTPUT / args.entry
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = directory / f"smoke-{stamp}.mp4"
    temporary = output.with_suffix(".part")
    request = urllib.request.Request(
        origin + "/v1/videos/sync",
        data=json.dumps(payload).encode(),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    # Exactly one generation attempt; do not retry expensive requests.
    with urllib.request.urlopen(request, timeout=36000) as response:
        if response.headers.get_content_type() != "video/mp4":
            raise ValueError("Generation did not return video/mp4")
        with temporary.open("xb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
    if temporary.stat().st_size == 0:
        raise ValueError("Generation returned an empty MP4")
    temporary.rename(output)
    timings["generation_seconds"] = time.perf_counter() - start
    receipt = {"entry": args.entry, "request": payload, "output": str(output), "timings": timings}
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
