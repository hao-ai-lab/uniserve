"""Native module selection and request-input capacity at startup.

Models declare numerical methods through ordinary module composition. The
worker selects the placed denoiser and provisions frame, text and condition
capacity; numerical builders prepare tensors within those bounds.
"""

from uniserve_worker._uniserve_ipc import (
    condition_capacity as condition_capacity,
)
from uniserve_worker._uniserve_ipc import (
    executed_video_tasks as executed_video_tasks,
)
from uniserve_worker._uniserve_ipc import image_input_builder as image_builder
from uniserve_worker._uniserve_ipc import media_input_builder as media_builder
from uniserve_worker._uniserve_ipc import model_capability as capability
from uniserve_worker._uniserve_ipc import video_denoiser as video_denoiser

__all__ = [
    "capability",
    "condition_capacity",
    "executed_video_tasks",
    "image_builder",
    "media_builder",
    "video_denoiser",
]
