"""Fixed MiniMax H3 T2VA qualification prompts."""

from __future__ import annotations

from typing import Any, ClassVar

from ..types import Example
from .base import Dataset

_PROMPT = (
    "integrated_multimodal_description: A red panda walks along a mossy path through "
    "a sunlit bamboo forest while the camera tracks smoothly beside it. "
    "overall_soundscape: Soft footsteps, rustling bamboo leaves, and distant birds "
    "blend into a calm natural ambience."
)
_SEED = 1000


class MiniMaxH3Dataset(Dataset):
    name: ClassVar[str] = "minimax-h3"
    requires_path = False
    requires_tokenizer = False

    def load(self, tokenizer: Any | None) -> list[Example]:
        del tokenizer
        return [
            Example(id=f"minimax-h3-{index:04d}", prompt=_PROMPT, seed=_SEED)
            for index in range(self.point.load.num_prompts)
        ]


__all__ = ["MiniMaxH3Dataset"]
