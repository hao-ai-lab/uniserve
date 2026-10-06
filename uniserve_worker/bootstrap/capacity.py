"""Native worker capacity planning and model-specific feature dimensions."""

from torch import nn

from uniserve.model import PatchEncoder
from uniserve.processing import ImageProcessor, PatchTransform
from uniserve_worker._uniserve_ipc import (
    ArenaCapacity as ArenaCapacity,
)
from uniserve_worker._uniserve_ipc import (
    KVCapacity as KVCapacity,
)
from uniserve_worker._uniserve_ipc import (
    check_startup_storage as check_startup_storage,
)
from uniserve_worker._uniserve_ipc import (
    derive_runtime_kv_capacity as derive_runtime_kv_capacity,
)
from uniserve_worker._uniserve_ipc import (
    device_total_bytes as device_total_bytes,
)
from uniserve_worker._uniserve_ipc import (
    graph_table_widths as graph_table_widths,
)
from uniserve_worker._uniserve_ipc import (
    input_buffer_config as input_buffer_config,
)
from uniserve_worker._uniserve_ipc import (
    local_product_storage_bytes as local_product_storage_bytes,
)
from uniserve_worker._uniserve_ipc import (
    resolve_request_capacity as resolve_request_capacity,
)
from uniserve_worker._uniserve_ipc import (
    tensor_slot_capacity as tensor_slot_capacity,
)
from uniserve_worker.bootstrap.inputs import capability


def vision_tokens(model: nn.Module, processor: ImageProcessor | None) -> int:
    """Bound feature rows from the actual image transform and encoder stride.

    Returns zero when the model has no ``PatchEncoder``, or there is no
    processor or it has no ViT transform.
    """
    encoder = capability(model, PatchEncoder)
    if encoder is None or processor is None or processor.vit is None:
        return 0
    transform = processor.vit
    pixels = (
        transform.pixel_bound()
        if isinstance(transform, PatchTransform)
        else min(transform.resize.max_pixels, transform.resize.max_size**2)
    )
    return pixels // (encoder.patch_size * encoder.downsample) ** 2
