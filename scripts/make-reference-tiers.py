#!/usr/bin/env python3
"""Build the reference tiers of issue #130 from public MiniMax-H3 media.

usage: make-reference-tiers.py <out dir> [--ffmpeg FFMPEG]

Writes <out dir>/media/ (the five source files, downloaded and checked by
SHA-256), <out dir>/media130/ (the tier media, derived from them with FFmpeg)
and <out dir>/tiers.jsonl, one request row per tier, duration and aspect
ratio. A server serving the rows needs ``--media-directory <out dir>``; each
row's ``conditions`` name files relative to it.

The tiers follow issue #130's sizes:

- T1: one 2048x3648 image.
- T2: three 2048x3648 images and a stereo 5 s voice.
- T3: three images, a silent 5 s video and two stereo 5 s voices.
- T4: a silent video as long as the target.
- T1L: T1's image as 3648x2048, the same rows at the target's aspect.

Images are scaled to cover the size and centre-cropped; the voices are two
5 s spans of one track; the 5 s video is the source's first 5 s, and the
target-length videos are cut from a forward-and-backward loop of it. Every
row is 16:9 at 5, 10 and 14.375 s (345 frames), and T3 and T4 also run at
9:16, 1:1 and 21:9 at 14.375 s. The prompt, 1,584 tokens, names two
speakers, a setting and the motion reference over eight shots.
"""

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/9bfb6693f2cf6de171db46d1aa586f67d773a1da/assets"
CDN = "https://cdn.hailuoai.com/prod/hailuo_demo/testsets"
SOURCES = {
    "hf_character_action_reference.png": (
        f"{HF}/character-action-reference.png",
        "01a6c6f29e1ce9d414276e6e2eca06b171e5a68dd54a86074a4ab77bb9cb8883",
    ),
    "hf_fashion_glasses_reference_3.png": (
        f"{HF}/fashion-glasses-reference-3.png",
        "7b1491e5a227f226fa1395e71ac89a84358a435d974acb9f9065b9ded0982cbe",
    ),
    "official_fl2va_0_image.png": (
        f"{CDN}/H3_AA_I2VA/gallery/sr_v17_variants_seed42_43_20260724/inputs/"
        "4a3a90bf9100_KDmcbkhzYo5sjjxr9FqcVmWVnzb.png",
        "352e959e08b886406d3766cee25ebe7bbf679e6d53eeffbd1c607b8dc93e2068",
    ),
    "official_ref2va_0_video.mp4": (
        f"{CDN}/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/"
        "297573323635_00_%E8%A7%86%E9%A2%911_YnyRbxEwio_video_20260525_163755_1927e9d3.mp4",
        "d6edc20ac82aa980c9cbd0b05b371f60d12dc8d018492e571b42de0086819c41",
    ),
    "official_ref2va_1_audio.mp3": (
        f"{CDN}/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/"
        "f463d523c5ce_01_%E9%9F%B3%E9%A2%911_RSLcbpzJPo_6%E6%9C%885%E6%97%A5(1).mp3",
        "2a48d447b343b38a2c504d5281a0c97c688db0ba87931ed9282c598bd6752454",
    ),
}
IMAGES = (
    "hf_character_action_reference.png",
    "hf_fashion_glasses_reference_3.png",
    "official_fl2va_0_image.png",
)
SECONDS = (5, 10, 14.375)

LABELS = (
    "subject_definitions:\n<Subject 1> is the person in <Picture "
    "1>.\n<Subject 2> is the person in <Picture 2>.\n<Subject 3> is the "
    "setting in <Picture 3>.\n<Video 1> provides the motion and camera "
    "framing.\n<Audio 1> is the voice-timbre reference for <Subject 1>; "
    "<Audio 2> for <Subject 2>.\n\n"
)
SHOT = (
    "[Shot {n}] At 00:{s:02d}.000, a {kind} shot. <Subject 1> and <Subject "
    "2> walk through <Subject 3> at golden hour, the camera tracking "
    "slowly from left to right while warm light falls across the gravel "
    "path, the low stone walls, the trimmed hedges and the lanterns strung "
    "between the trees. <Subject 1> turns toward <Subject 2> and says, in "
    "the voice of <Audio 1>, <d>[English] We should have come here years "
    "ago, before the city swallowed every quiet place we knew.</d> "
    "<Subject 2> answers in the voice of <Audio 2>, <d>[English] Then let "
    "us stay until the lanterns burn out.</d> Leaves move in a light "
    "breeze, birds cross the sky, and distant bells ring twice as the shot "
    "holds on their faces.\n"
)
KINDS = (
    "wide",
    "medium",
    "close-up",
    "over-the-shoulder",
    "tracking",
    "low-angle",
    "high-angle",
    "two-shot",
)
PROMPT = (
    LABELS
    + "summary:\n[reference generation + audio reference] "
    + "A conversation at golden hour.\n\n"
    + "detailed_description:\n"
    + "".join(SHOT.format(n=i + 1, s=i, kind=KINDS[i]) for i in range(8))
)


def fetch(media: Path) -> None:
    media.mkdir(parents=True, exist_ok=True)
    for name, (url, sha256) in SOURCES.items():
        path = media / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=300) as response:
                path.write_bytes(response.read())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sha256:
            raise SystemExit(f"{name}: SHA-256 {digest}, expected {sha256}")


def derive(media: Path, out: Path, ffmpeg: str) -> None:
    out.mkdir(parents=True, exist_ok=True)

    def run(*args: str) -> None:
        subprocess.run(
            [ffmpeg, "-nostdin", "-loglevel", "error", "-y", *args], check=True
        )

    for index, name in enumerate(IMAGES, start=1):
        for width, height in ((2048, 3648), (3648, 2048)):
            run(
                "-i",
                str(media / name),
                "-vf",
                f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
                "-frames:v",
                "1",
                str(out / f"img{index}_{width}x{height}.png"),
            )
    voice = str(media / "official_ref2va_1_audio.mp3")
    for index, start in ((1, "0"), (2, "3.5")):
        run(
            "-i",
            voice,
            "-ss",
            start,
            "-t",
            "5",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(out / f"voice{index}_5s.wav"),
        )
    video = str(media / "official_ref2va_0_video.mp4")
    encode = ("-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p")
    run("-i", video, "-t", "5", *encode, str(out / "video_5s.mp4"))
    loop = out / "loop_24s.mp4"
    run(
        "-i",
        video,
        "-an",
        "-filter_complex",
        "[0:v]split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1[p];[p]split[p1][p2];[p1][p2]concat=n=2:v=1[o]",
        "-map",
        "[o]",
        *encode[1:],
        str(loop),
    )
    for seconds in (10, 14.375):
        run(
            "-i",
            str(loop),
            "-t",
            str(seconds),
            *encode,
            str(out / f"video_{seconds}s.mp4"),
        )
    loop.unlink()


def rows() -> list[dict]:
    m = "media130/"
    image = [f"{m}img{i}_2048x3648.png" for i in (1, 2, 3)]

    def condition(kind: str, path: str) -> dict:
        return {"type": kind, "media": path, "role": "reference"}

    def video(seconds: float) -> dict:
        name = {
            5: "video_5s.mp4",
            10: "video_10s.mp4",
            14.375: "video_14.375s.mp4",
        }[seconds]
        return condition("video", m + name)

    voices = [condition("audio", f"{m}voice{i}_5s.wav") for i in (1, 2)]
    tiers = {
        "T1": lambda s: [condition("image", image[0])],
        "T1L": lambda s: [condition("image", f"{m}img1_3648x2048.png")],
        "T2": lambda s: [condition("image", p) for p in image] + voices[:1],
        "T3": lambda s: (
            [condition("image", p) for p in image] + [video(5)] + voices
        ),
        "T4": lambda s: [video(s)],
    }
    out = []

    def add(tier: str, seconds: float, aspect: str) -> None:
        out.append(
            {
                "id": f"{tier}-{seconds}s-{aspect.replace(':', 'x')}",
                "tier": tier,
                "prompt": PROMPT,
                "seconds": seconds,
                "seed": 0,
                "aspect": aspect,
                "conditions": tiers[tier](seconds),
            }
        )

    for seconds in SECONDS:
        for tier in ("T1", "T2", "T3", "T4"):
            add(tier, seconds, "16:9")
    add("T1L", 5, "16:9")
    for aspect in ("9:16", "1:1", "21:9"):
        for tier in ("T3", "T4"):
            add(tier, 14.375, aspect)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    fetch(args.out / "media")
    derive(args.out / "media", args.out / "media130", args.ffmpeg)
    tiers = rows()
    (args.out / "tiers.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in tiers)
    )
    print(f"{len(tiers)} rows in {args.out / 'tiers.jsonl'}")


if __name__ == "__main__":
    main()
