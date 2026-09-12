#!/usr/bin/env python3
"""Send one text-only and one inline-image request to a published H3 reference endpoint."""

import argparse
import base64
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEPLOY = Path("/mnt/lustre/vlm-wlsaidhi/src/uniserve-deploy")
OUTPUT = Path("/mnt/lustre/vlm-wlsaidhi/fastvideo/eval/uniserve-smoke/ref")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image", type=Path, help="PNG/JPEG; dimensions multiples of 32, at most 4096"
    )
    args = parser.parse_args()
    if not 0 < args.image.stat().st_size <= 32 * 1024 * 1024:
        parser.error("image must fit the 32 MiB source bound")
    encoded = base64.b64encode(args.image.read_bytes()).decode("ascii")
    endpoint = json.loads((DEPLOY / "endpoints/h3-ref-rack3.json").read_text())
    model = "minimax-h3-ref"
    if endpoint["models"] != [model]:
        raise ValueError("Endpoint must advertise exactly minimax-h3-ref")
    if datetime.fromisoformat(endpoint["expires_utc"].replace("Z", "+00:00")) <= datetime.now(
        timezone.utc
    ):
        raise ValueError("Endpoint allocation has expired")
    key = Path(endpoint["api_key_file"]).read_text().strip()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    origin = endpoint["base_url"].rstrip("/")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    for case in ("text", "image"):
        payload = {
            "model": model,
            "prompt": "A calm sea at sunrise, with gentle waves and natural ocean sound.",
            "seconds": 5,
            "steps": 50,
            "seed": 0,
        }
        if case == "image":
            payload["references"] = [
                {
                    "type": "image",
                    "task": "reference",
                    "role": "reference",
                    "source": {"type": "base64", "value": encoded},
                }
            ]
        request = urllib.request.Request(
            origin + "/v1/videos/sync",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        output = OUTPUT / f"{case}-{stamp}.mp4"
        temporary = output.with_suffix(".part")
        # Serial requests, exactly one attempt each; no expensive generation retries.
        with urllib.request.urlopen(request, timeout=36000) as response:
            if response.headers.get_content_type() != "video/mp4":
                raise ValueError("Generation did not return video/mp4")
            with temporary.open("xb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        if temporary.stat().st_size == 0:
            raise ValueError("Generation returned an empty MP4")
        temporary.rename(output)
        print(output)


if __name__ == "__main__":
    main()
