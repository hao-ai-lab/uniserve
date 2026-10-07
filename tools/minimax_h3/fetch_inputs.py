"""Fetch the MiniMax-H3 reference inputs and record their provenance.

The official MiniMax-H3 request scripts
(``scripts/readme/reproducible-768p-{t2va,fl2va,ref2va}-request.sh`` in the
``MiniMaxAI/MiniMax-H3`` checkpoint) carry the complete H3-Context-IR prompts
and the CDN media of the three reproducible 768p cases. This script extracts
their JSON request bodies, downloads every condition URI, a fixed set of
additional official media published in the model repository (used by the
reference workloads that the official scripts do not cover) and the official
768p results, and writes into ``--output``:

* ``media/<name>``: every downloaded input;
* ``official_outputs/<name>.mp4``: the official results of the three cases;
* ``official_requests.json``: the three request bodies verbatim, each
  condition annotated with the local file it was saved to;
* ``prompts.json``: the official prompts by task;
* ``manifest.json``: per file the source URL, byte size, sha256 and the
  ``ffprobe -show_format -show_streams`` facts of the pinned FFmpeg build.

Downloads are idempotent: a file whose sha256 already matches the manifest
entry is kept. Hugging Face media are addressed by commit so the bytes are
reproducible.
"""

import argparse
import hashlib
import json
import re
import subprocess
import urllib.request
from pathlib import Path

CHECKPOINT = Path("/workspace/models/MiniMax-H3")
FFPROBE = Path("/workspace/tools/ffmpeg-8.1.2/bin/ffprobe")
HF_REPO = "https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve"

# The revision the local checkpoint copy was downloaded at (the official
# results live there) and the revision the OmniRef contract pins (the
# additional reference media were published there and removed afterwards).
MAIN_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
PINNED_REVISION = "9bfb6693f2cf6de171db46d1aa586f67d773a1da"

OFFICIAL_CASES = ("t2va", "fl2va", "ref2va")

# Additional official media for the workloads the three cases do not cover:
# a second keyframe, image references, a second video reference.
REPOSITORY_MEDIA = {
    "hf_reference_image_1.png": "assets/reference-image-1.png",
    "hf_reference_image_2.png": "assets/reference-image-2.png",
    "hf_fl2va_clay_fox_reference.png": "assets/fl2va-clay-fox-reference.png",
    "hf_character_action_reference.png": (
        "assets/character-action-reference.png"
    ),
    "hf_action_reference.mov": "assets/action-reference.mov",
    "hf_character_replacement_action_reference.mp4": (
        "assets/character-replacement-action-reference.mp4"
    ),
    "hf_robot_arm_red_cube.mp4": "assets/robot-arm-red-cube.mp4",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(
        url, headers={"User-Agent": "uniserve-minimax-h3-inputs"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        partial.write_bytes(response.read())
    partial.rename(path)


def _probe(path: Path, ffprobe: Path) -> dict:
    """Return the ffprobe format and stream facts of one media file."""
    result = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    facts = json.loads(result.stdout)
    # The file name is the manifest key already; the absolute path would
    # make the manifest machine-specific.
    facts.get("format", {}).pop("filename", None)
    return facts


def _official_request(case: str) -> dict:
    """Extract the JSON request body of one official request script."""
    script = (
        CHECKPOINT
        / "scripts"
        / "readme"
        / f"reproducible-768p-{case}-request.sh"
    ).read_text()
    match = re.search(r"<<'JSON'\n(.*?)\nJSON\n", script, re.DOTALL)
    if match is None:
        raise ValueError(f"no JSON request body in the {case} script")
    return json.loads(match.group(1))


def _media_name(case: str, index: int, condition: dict) -> str:
    suffix = Path(urllib.request.url2pathname(condition["uri"])).suffix
    return f"official_{case}_{index}_{condition['type']}{suffix.lower()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, default=FFPROBE)
    args = parser.parse_args()

    output = args.output
    sources = {}
    requests = {}

    # 1. The official cases: request bodies and their condition media.
    for case in OFFICIAL_CASES:
        body = _official_request(case)
        for index, condition in enumerate(body["conditions"]):
            name = _media_name(case, index, condition)
            condition["local_file"] = f"media/{name}"
            sources[f"media/{name}"] = condition["uri"]
        requests[case] = {
            "script": (
                f"scripts/readme/reproducible-768p-{case}-request.sh"
                f"@{MAIN_REVISION}"
            ),
            "body": body,
        }

    # 2. Additional official media and the official results.
    for name, path in REPOSITORY_MEDIA.items():
        sources[f"media/{name}"] = f"{HF_REPO}/{PINNED_REVISION}/{path}"
    for case in OFFICIAL_CASES:
        sources[f"official_outputs/{case}.mp4"] = (
            f"{HF_REPO}/{MAIN_REVISION}/assets/{case}.mp4"
        )

    previous = {}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())["files"]

    files = {}
    for relative, url in sorted(sources.items()):
        path = output / relative
        known = previous.get(relative, {}).get("sha256")
        if not (path.exists() and known and _sha256(path) == known):
            _download(url, path)
        files[relative] = {
            "url": url,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "ffprobe": _probe(path, args.ffprobe),
        }
        print(f"{relative}: {files[relative]['sha256']}")

    ffprobe_version = subprocess.run(
        [str(args.ffprobe), "-version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    manifest = {
        "checkpoint_revision": MAIN_REVISION,
        "repository_media_revision": PINNED_REVISION,
        "ffprobe": ffprobe_version,
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "official_requests.json").write_text(
        json.dumps(requests, indent=2, ensure_ascii=False) + "\n"
    )
    prompts = {case: requests[case]["body"]["prompt"] for case in requests}
    (output / "prompts.json").write_text(
        json.dumps(prompts, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
