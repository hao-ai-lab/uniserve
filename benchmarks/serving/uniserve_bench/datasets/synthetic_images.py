"""Deterministic synthetic images + local image directories for i2t.

``synthetic-images`` draws seeded geometric scenes with PIL so an i2t
comparison reproduces from a clean checkout with zero downloads; ``image-dir``
loads real photos from a local directory when representativeness matters more
than hermeticity. Both emit rows shaped for the i2t task:
``{"id", "prompt", "input_image_b64", "width", "height"}``.
"""
from __future__ import annotations

import base64
import io
import random
from pathlib import Path
from typing import Any

_PALETTE = [
    (178, 34, 34),
    (34, 139, 34),
    (65, 105, 225),
    (255, 215, 0),
    (138, 43, 226),
    (255, 140, 0),
    (0, 139, 139),
    (199, 21, 133),
]


def _synthetic_png_b64(index: int, seed: int, width: int, height: int) -> str:
    from PIL import Image, ImageDraw

    rng = random.Random(seed * 1_000_003 + index)
    background = tuple(rng.randint(140, 235) for _ in range(3))
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    for _ in range(rng.randint(4, 9)):
        color = _PALETTE[rng.randrange(len(_PALETTE))]
        x0 = rng.randint(0, width - 60)
        y0 = rng.randint(0, height - 60)
        x1 = x0 + rng.randint(40, max(41, width // 3))
        y1 = y0 + rng.randint(40, max(41, height // 3))
        shape = rng.choice(("rectangle", "ellipse", "triangle"))
        if shape == "rectangle":
            draw.rectangle([x0, y0, x1, y1], fill=color)
        elif shape == "ellipse":
            draw.ellipse([x0, y0, x1, y1], fill=color)
        else:
            draw.polygon([(x0, y1), ((x0 + x1) // 2, y0), (x1, y1)], fill=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def load_synthetic_images(
    num_prompts: int,
    *,
    seed: int,
    question: str,
    width: int = 1024,
    height: int = 768,
) -> list[dict[str, Any]]:
    return [
        {
            "id": f"synthetic-{index}",
            "prompt": question,
            "input_image_b64": _synthetic_png_b64(index, seed, width, height),
            "width": width,
            "height": height,
        }
        for index in range(num_prompts)
    ]


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_image_dir(
    path: str,
    num_prompts: int,
    *,
    seed: int,
    question: str,
) -> list[dict[str, Any]]:
    directory = Path(path)
    if not directory.is_dir():
        raise ValueError(f"dataset 'image-dir' requires a directory, got {path!r}")
    files = sorted(
        file for file in directory.iterdir() if file.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not files:
        raise ValueError(f"no images found under {path!r}")
    rng = random.Random(seed)
    if len(files) > num_prompts:
        files = rng.sample(files, num_prompts)
    rows = []
    for index, file in enumerate(files):
        rows.append(
            {
                "id": f"image-dir-{index}-{file.stem}",
                "prompt": question,
                "input_image_b64": base64.b64encode(file.read_bytes()).decode(),
            }
        )
    return rows
