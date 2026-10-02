"""Generate the MiniMax-H3 multi-region sparse layout vectors from FastVideo.

The vectors in ``minimax_h3_regions.json`` are consumed by
``tests/python/unit/models/test_h3_regions.py``. Every expected value comes
from FastVideo's FastH3 OmniRef inference path (branch
``feat/fasth3-omniref-pdd-inference``), run on the latent geometry of each
case on the CPU:

* the packed ``ref2va`` sequence ``[text | references | target audio |
  target video]`` from ``build_ref2va_packed_sequence``: every row's rotary
  coordinates, its token tag and its timestep under
  ``build_row_timesteps`` with one distinct timestep per group;
* the ``p2_multi_region`` tiles from the denoising stage's
  ``_h3_vsa_ref2va_segments`` and ``MiniMaxH3VSAMetadataBuilder.build`` at
  the checkpoint's tile size, sparsity and reference keep rate: each tile's
  valid size, the tile slot of every packed row, the dense prefix tiles and
  each video region's tile span;
* the block mask ``_build_region_block_mask`` selects from seeded random
  pooled scores.

Each case states its conditions as the pixels and samples UniServe's
``Condition`` carries; the generator converts them to the latent geometry
FastVideo's prepared references hold (``17 * n + 5`` frames to
``5 * n + 2`` latent frames, a 16x spatial compression, one audio latent per
800 samples).

Run with the FastVideo environment's interpreter from the repository root:

    /workspace/envs/minimax_h3/fastvideo/bin/python \
        tests/python/fixtures/generate_minimax_h3_regions.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from fastvideo.attention.backends.video_sparse_attn_h3 import (
    MiniMaxH3VSAMetadataBuilder,
    _build_block_mask,
)
from fastvideo.pipelines.basic.minimax_h3.packing import (
    audio_latent_num_frames,
    build_ref2va_packed_sequence,
    build_row_timesteps,
    video_latent_num_frames,
)
from fastvideo.pipelines.basic.minimax_h3.reference import (
    MiniMaxH3PreparedReference,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising import (
    _h3_vsa_ref2va_segments,
)

OUTPUT = Path(__file__).with_name("minimax_h3_regions.json")
PATCH = (1, 2, 2)
# The FastH3 OmniRef contract (fastvideo_inference.json).
TILE, SPARSITY, REFERENCE_KEEP = 128, 0.9, 0.1
# One timestep per row group, so each row's timestep names its group.
GROUP_TIMESTEPS = {
    "video": 0.125,
    "audio": 0.25,
    "visual_condition": 0.5,
    "audio_condition": 0.75,
}
HEADS = 2

CASES = (
    {
        # A video with its soundtrack, an image and an audio reference; the
        # prompt holds a vision span.
        "name": "video_image_audio",
        "frames": 22,
        "height": 256,
        "width": 448,
        "text_tokens": 150,
        "vision_spans": [[10, 74]],
        "conditions": [
            {"frames": 22, "height": 128, "width": 256, "samples": 48000},
            {"frames": 1, "height": 192, "width": 320, "samples": 0},
            {"frames": 0, "height": 0, "width": 0, "samples": 32000},
        ],
        "seed": 11,
    },
    {
        # Two silent videos of different rasters around an image.
        "name": "two_videos",
        "frames": 39,
        "height": 192,
        "width": 192,
        "text_tokens": 300,
        "vision_spans": [],
        "conditions": [
            {"frames": 39, "height": 128, "width": 128, "samples": 0},
            {"frames": 1, "height": 64, "width": 512, "samples": 0},
            {"frames": 22, "height": 96, "width": 160, "samples": 0},
        ],
        "seed": 12,
    },
    {
        # One image reference: the generated video is the only region.
        "name": "image_only",
        "frames": 22,
        "height": 128,
        "width": 224,
        "text_tokens": 129,
        "vision_spans": [[0, 128]],
        "conditions": [
            {"frames": 1, "height": 128, "width": 224, "samples": 0},
        ],
        "seed": 13,
    },
)


def _reference(condition: dict) -> MiniMaxH3PreparedReference:
    audio = math.ceil(condition["samples"] / 800)
    if not condition["frames"]:
        return MiniMaxH3PreparedReference(
            media_type="audio", has_audio=True, num_audio_latents=audio
        )
    frames = condition["frames"]
    return MiniMaxH3PreparedReference(
        media_type="image" if frames == 1 else "video",
        has_audio=bool(audio),
        num_latent_frames=1 if frames == 1 else video_latent_num_frames(frames),
        latent_height=condition["height"] // 16,
        latent_width=condition["width"] // 16,
        num_audio_latents=audio,
    )


def _case(case: dict) -> dict:
    tags = torch.ones(case["text_tokens"], dtype=torch.long)
    for start, stop in case["vision_spans"]:
        tags[start:stop] = 0
    references = [_reference(condition) for condition in case["conditions"]]
    layout = build_ref2va_packed_sequence(
        tags,
        references,
        video_latent_num_frames(case["frames"]),
        case["height"] // 16,
        case["width"] // 16,
        audio_latent_num_frames(case["frames"]),
        PATCH,
    )
    unique, inverse = build_row_timesteps(
        layout,
        video_timestep=GROUP_TIMESTEPS["video"],
        audio_timestep=GROUP_TIMESTEPS["audio"],
        condition_video_timestep=GROUP_TIMESTEPS["visual_condition"],
        condition_audio_timestep=GROUP_TIMESTEPS["audio_condition"],
    )
    prefix, videos, offsets = _h3_vsa_ref2va_segments(layout, PATCH)
    metadata = MiniMaxH3VSAMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=None,
        patch_size=PATCH,
        VSA_sparsity=SPARSITY,
        prefix_segments=prefix,
        device=torch.device("cpu"),
        tile_size=TILE,
        video_segments=videos,
        video_offsets=offsets,
        ref_keep_rate=REFERENCE_KEEP,
    )
    tiles = metadata.variable_block_sizes.numel()
    generator = torch.Generator().manual_seed(case["seed"])
    scores = torch.randn((1, HEADS, tiles, tiles), generator=generator)
    mask = _build_block_mask(
        scores,
        metadata.num_prefix_tiles,
        metadata.num_video_tiles,
        metadata.VSA_sparsity,
        metadata.exempt,
        metadata.video_tile_spans,
        metadata.span_sparsities,
    )[0]
    return {
        "name": case["name"],
        "frames": case["frames"],
        "height": case["height"],
        "width": case["width"],
        "text_tokens": case["text_tokens"],
        "vision_spans": case["vision_spans"],
        "conditions": case["conditions"],
        "position_ids": layout.position_ids.tolist(),
        "token_tags": layout.token_tags.tolist(),
        "row_timesteps": unique[inverse].tolist(),
        "valid_sizes": metadata.variable_block_sizes.tolist(),
        "row_slots": metadata.untile_combined_index.tolist(),
        "prefix_tiles": metadata.num_prefix_tiles,
        "region_spans": [list(span) for span in metadata.video_tile_spans],
        "scores": scores[0].tolist(),
        "mask": [
            [row.nonzero().flatten().tolist() for row in head] for head in mask
        ],
    }


def main() -> None:
    document = {
        "tile": TILE,
        "sparsity": SPARSITY,
        "reference_keep": REFERENCE_KEEP,
        "group_timesteps": GROUP_TIMESTEPS,
        "cases": [_case(case) for case in CASES],
    }
    OUTPUT.write_text(json.dumps(document) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
