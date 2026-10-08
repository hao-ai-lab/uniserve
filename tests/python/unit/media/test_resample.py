"""The banded LANCZOS resize returns Pillow's own resize, byte for byte.

Random RGB noise overshoots the LANCZOS lobes at nearly every pixel, so the
comparisons cover the fixed-point sums' clipping at both ends as well as
every band split of the sizes below.
"""

import numpy as np
import pytest
from PIL import Image

from uniserve_worker.media.resample import lanczos

pytestmark = pytest.mark.unit

# ((input width, height), (output width, height)).
SIZES = [
    # Upscales at the reference images' ratios, integer or not.
    ((1440, 1440), (2048, 2048)),
    ((2560, 1440), (3648, 2048)),
    ((1474, 900), (3808, 2048)),
    ((1920, 1080), (3648, 2048)),
    # Downscales, the kernel widening with the ratio.
    ((4000, 3000), (2731, 2048)),
    ((1999, 1001), (640, 333)),
    ((1000, 10), (7, 3)),
    # One axis keeps its length, so Pillow runs only the other pass.
    ((100, 80), (100, 33)),
    ((64, 50), (17, 50)),
    ((700, 650), (700, 1999)),
    ((1321, 723), (2800, 723)),
    # Tiny and odd sizes, where the kernel is clipped at both edges.
    ((1, 1), (5, 3)),
    ((2, 2), (1, 1)),
    ((3, 1), (1, 7)),
    ((37, 5), (3, 41)),
    ((641, 479), (1283, 957)),
    # More than 100 times taller than wide and shortened: Pillow resamples
    # the columns first.
    ((2, 300), (3, 150)),
    ((3, 1000), (1, 9)),
    ((5, 2000), (700, 1500)),
    # The image's own size.
    ((50, 40), (50, 40)),
]


def _noise(width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def _assert_pillow_resize(image: Image.Image, size: tuple[int, int]):
    expected = np.asarray(image.resize(size, Image.Resampling.LANCZOS))
    resized = lanczos(image, size)
    assert resized.dtype == np.uint8
    np.testing.assert_array_equal(resized, expected)


@pytest.mark.parametrize(("source", "size"), SIZES)
def test_resize_equals_pillow(source, size):
    _assert_pillow_resize(_noise(*source, seed=sum(source + size)), size)


def test_resize_equals_pillow_at_random_sizes():
    # Each axis independently upscales, downscales or keeps its length,
    # across lengths that split into one to several bands.
    rng = np.random.default_rng(0)
    for case in range(48):
        source = tuple(int(n) for n in rng.integers(1, 1500, 2))
        size = tuple(
            length if keep else int(rng.integers(1, 2500))
            for length, keep in zip(source, rng.random(2) < 0.2, strict=True)
        )
        _assert_pillow_resize(_noise(*source, seed=case), size)
