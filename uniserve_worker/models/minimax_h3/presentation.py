"""Shape-only Qwen presentation planning shared by entry owners."""

from dataclasses import dataclass

from ...execution.batch import MediaGeometry
from .packing import TEXT_TAG, VIDEO_TAG


@dataclass(frozen=True, slots=True)
class H3MediaGeometry(MediaGeometry):
    """Worker-local geometry; admitted prompt tokens remain unmodified."""

    reference_shape: tuple[int, int] | None = None
    presentation_tags: tuple[int, ...] = ()


def image_presentation_tags(
    processor, shape: tuple[int, int], prompt_tokens: int
) -> tuple[int, ...]:
    """Predict the processor's merged image span without decoding or allocating pixels."""

    image_processor = processor.image_processor
    patches = image_processor.get_number_of_image_patches(*shape, images_kwargs={})
    rows = patches // image_processor.merge_size**2
    label = processor.tokenizer("<Picture 1>: ", add_special_tokens=False)["input_ids"]
    return (TEXT_TAG,) * len(label) + (VIDEO_TAG,) * (rows + 2) + (TEXT_TAG,) * prompt_tokens
