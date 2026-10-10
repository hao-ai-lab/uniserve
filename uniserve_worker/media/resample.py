"""Pillow's LANCZOS resize of an RGB image, run as bands on several threads.

Pillow resizes in two separable passes (``ImagingResampleInner`` in
``Resample.c``): a horizontal pass resamples every row on its own into an
8-bit intermediate, then a vertical pass resamples every column of that
intermediate on its own. A pass's coefficient table depends only on the
input and output lengths of the axis it resamples, its fixed-point sums
only on one row's (or column's) pixels, and a pass that keeps its axis's
length is skipped.

``lanczos`` runs each pass as Pillow resizes of bands: row bands for the
horizontal pass, each resized to the output width at its own height, and
column bands for the vertical pass, each resized to the output height at its
own width. Every band computes the whole image's coefficient table and sums
the same pixels with it, so the joined bands are byte-identical to
``Image.resize(size, LANCZOS)``. Pillow's one-step resize runs its
horizontal pass only over the rows its vertical pass reads, which with the
whole image as the box are all of them. ``Image.resize`` shortens an image
more than 100 times taller than wide column-first, and so does ``lanczos``.

Pillow releases the interpreter lock while a pass runs but holds it while
it computes a coefficient table, about 16 ns per output sample and tap, and
while ``np.asarray`` copies a result out of it. Every band recomputes its
pass's whole table, so the band count trades that serialized work against
the parallel passes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from PIL import Image

#: The fewest lines a band holds: rows of a horizontal-pass band, columns of
#: a vertical-pass band. Per line, a band's pass costs about 1/18 of the
#: coefficient table it recomputes, so a band of 320 lines spends about 5%
#: of its time on the table, while the lock is held.
BAND = 320

#: Threads shared by the process's resizes; concurrent resizes queue their
#: bands here. The interpreter lock, held for every band's coefficient table
#: and result copy, bounds the useful threads well below a host's cores.
_POOL = ThreadPoolExecutor(max_workers=32, thread_name_prefix="lanczos")


def _bands(lines: int) -> list[tuple[int, int]]:
    """Split ``lines`` into near-equal ``[start, stop)`` bands.

    Each band holds at least ``BAND`` lines, unless ``lines`` is fewer, when
    one band holds them all.
    """
    count = max(1, lines // BAND)
    edges = [lines * index // count for index in range(count + 1)]
    return list(zip(edges[:-1], edges[1:], strict=True))


def _paste(
    joined: Image.Image,
    horizontal: bool,
    start: int,
    stop: int,
    band: Image.Image,
) -> None:
    # Pillow copies a paste's rows with the interpreter lock released, and
    # concurrent bands write disjoint rows or columns of ``joined``.
    joined.paste(band, (0, start) if horizontal else (start, 0))


def _store(
    pixels: np.ndarray,
    horizontal: bool,
    start: int,
    stop: int,
    band: Image.Image,
) -> None:
    # Pillow exports a band's pixels with the interpreter lock held; NumPy
    # copies them into the band's disjoint rows or columns of the result.
    if horizontal:
        pixels[start:stop] = np.asarray(band)
    else:
        pixels[:, start:stop] = np.asarray(band)


def _resample(source: Image.Image, horizontal: bool, length: int, store):
    """Run one pass of ``source`` as bands on the shared threads.

    A horizontal pass resamples the rows to ``length`` columns in row
    bands; a vertical pass resamples the columns to ``length`` rows in
    column bands. ``store(start, stop, band)`` receives each resized band,
    rows or columns ``[start, stop)`` of the pass's output, concurrently
    from the pool's threads. Returns once every band is stored.
    """

    def run(start: int, stop: int) -> None:
        if horizontal:
            box = (0, start, source.width, stop)
            size = (length, stop - start)
        else:
            box = (start, 0, stop, source.height)
            size = (stop - start, length)
        band = source.crop(box).resize(size, Image.Resampling.LANCZOS)
        store(start, stop, band)

    lines = source.height if horizontal else source.width
    tasks = [_POOL.submit(run, start, stop) for start, stop in _bands(lines)]
    for task in tasks:
        task.result()


def lanczos(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Resize an RGB image exactly as ``image.resize(size, LANCZOS)`` does.

    ``size`` is ``(width, height)``. The image is loaded here, once, before
    the bands crop it concurrently; the caller's thread then waits for the
    bands on the shared threads.

    Returns:
        ``[height, width, 3]`` uint8 RGB, the pixels of Pillow's result.

    Raises:
        ValueError: The image is not RGB, or ``size`` is not positive.
    """
    if image.mode != "RGB":
        raise ValueError(f"a LANCZOS band resize takes RGB, got {image.mode}")
    width, height = size
    if width < 1 or height < 1:
        raise ValueError("a resized image needs a positive width and height")
    image.load()

    # The passes that change their axis, in Pillow's order: True for the
    # horizontal pass, False for the vertical one.
    passes = [
        horizontal
        for horizontal, changed in (
            (True, width != image.width),
            (False, height != image.height),
        )
        if changed
    ]
    if image.height > 100 * image.width and height < image.height:
        passes.reverse()

    pixels = np.empty((height, width, 3), dtype=np.uint8)
    if not passes:
        pixels[...] = np.asarray(image)
        return pixels

    # The first of two passes joins its bands into the image the second
    # splits; the last pass's bands land in the result.
    source = image
    for horizontal in passes[:-1]:
        joined = Image.new(
            "RGB",
            (width, source.height) if horizontal else (source.width, height),
        )
        _resample(
            source,
            horizontal,
            width if horizontal else height,
            partial(_paste, joined, horizontal),
        )
        source = joined
    horizontal = passes[-1]
    _resample(
        source,
        horizontal,
        width if horizontal else height,
        partial(_store, pixels, horizontal),
    )
    return pixels


__all__ = ["BAND", "lanczos"]
