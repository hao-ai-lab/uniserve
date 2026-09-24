"""Builds fixed-length MiniMax H3 text-to-video-with-audio prompts.

The prompt length is a workload dimension (``VideoConfig.prompt_tokens``)
alongside the video duration. This adapter synthesizes one prompt whose token
count under the model's tokenizer equals the target exactly, and every
example repeats that prompt with the same seed. Server profiles that pass
``--video-graph-shapes`` name shapes as seconds by prompt tokens; requests
outside a captured shape's layout denoise without graphs.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..types import Example
from .base import Dataset

_PROMPT = (
    "integrated_multimodal_description: [Shot 1] A 9-second 16:9 widescreen "
    "educational documentary tutorial in clean paper-textured motion-graphics "
    "design, following one illustrated paper-craft instructor at a neatly "
    "gridded tabletop. The instructor places a coral square of paper on the "
    "grid, aligns its corners, and folds it diagonally into a sharp crease; "
    'a Japanese title card reading "折り紙ランタン" appears at the upper left, '
    'while a thin animated guide line and the label "谷折り" appear beside '
    "the crease. The overhead camera makes a measured push-in as the "
    "instructor smooths the fold, with small geometric accents tracking the "
    "paper edges. [Shot 2] At 00:04.500, the same instructor unfolds the "
    "paper, rotates it a quarter turn, and presses the intersecting creases "
    "into a compact lantern shape; the camera cuts to a close three-quarter "
    "tabletop view and makes a short lateral track to reveal the dimensional "
    "form. Animated arrows trace the final fold, and the Japanese completion "
    'card "完成" settles along the lower edge as the instructor places the '
    "lantern in a small cardboard tray. overall_soundscape: Close ASMR foley "
    "records the paper's dry flex and crisp crease, fingertip taps on the "
    "matte work surface, and a soft sleeve rustle, with a quiet studio room "
    "tone underneath. A small cardboard tray clicks when the instructor sets "
    "the finished lantern down, while gentle breathing remains audible. "
    "non_diegetic_music: N/A"
)
# Per-example generation seed; the video task sends it in place of the
# point's load seed.
_SEED = 1000


class MiniMaxH3Dataset(Dataset):
    """Synthesizes deterministic prompts at an exact tokenizer length."""

    name: ClassVar[str] = "minimax-h3"
    requires_path = False
    requires_tokenizer = True

    def load(self, tokenizer: Any | None = None) -> list[Example]:
        """Construct repeated examples whose decoded prompt has the target length.

        Token counts exclude special tokens.

        Raises:
            ValueError: If no tokenizer is supplied, the tokenizer encodes the
                base or filler text to no tokens, or the decoded prompt does
                not re-encode to exactly ``prompt_tokens`` tokens.
        """  # noqa: E501
        if tokenizer is None:
            raise ValueError(
                "MiniMax H3 benchmark prompt synthesis requires its tokenizer"
            )
        target = int(self.point.video.prompt_tokens)

        # Truncate the base description or pad it with whole or partial
        # copies of the filler sentence until the id count reaches the target.
        base_ids = list(tokenizer.encode(_PROMPT, add_special_tokens=False))
        filler_ids = list(
            tokenizer.encode(
                " A coherent continuation preserves the scene, motion, "
                "lighting, and sound.",
                add_special_tokens=False,
            )
        )
        if not base_ids or not filler_ids:
            raise ValueError("MiniMax H3 tokenizer produced no prompt tokens")
        token_ids = base_ids[:target]
        while len(token_ids) < target:
            token_ids.extend(filler_ids[: target - len(token_ids)])

        # Decoding and re-encoding need not preserve the token count, so the
        # decoded text is measured again and rejected on a mismatch.
        prompt = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        measured = tokenizer.encode(prompt, add_special_tokens=False)
        if len(measured) != target:
            raise ValueError(
                f"MiniMax H3 synthesized prompt measured {len(measured)} "
                f"tokens, expected {target}"
            )

        return [
            Example(
                id=f"minimax-h3-{index:04d}",
                prompt=prompt,
                prompt_len=target,
                seed=_SEED,
                seconds=float(self.point.video.seconds),
            )
            for index in range(self.point.load.num_prompts)
        ]


__all__ = ["MiniMaxH3Dataset"]
