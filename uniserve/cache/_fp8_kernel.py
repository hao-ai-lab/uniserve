"""Native block-local FP8 rescaling with no host address synchronization."""

import triton
import triton.language as tl


@triton.jit
def rescale_blocks(
    values,
    old_scales,
    new_scales,
    initialized,
    width: tl.constexpr,
    tile: tl.constexpr,
    compute_dtype: tl.constexpr,
):
    # values: [blocks, width] flattened FP8 codes; scales and flags: [blocks].
    block = tl.program_id(0)
    old = tl.load(old_scales + block)
    new = tl.load(new_scales + block)
    active = tl.load(initialized + block) & (new > old)
    # Untouched and non-growing blocks perform only the metadata reads. A
    # growing block retains its old scale until every encoded element is read.
    if active:
        for start in range(0, width, tile):
            indices = start + tl.arange(0, tile)
            value = tl.load(values + block * width + indices, indices < width, other=0.0).to(
                tl.float32
            )
            decoded = (value * old).to(compute_dtype).to(tl.float32)
            encoded = tl.maximum(-448.0, tl.minimum(448.0, decoded / new))
            tl.store(values + block * width + indices, encoded, indices < width)
