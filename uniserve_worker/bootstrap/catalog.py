"""The worker's model catalog: explicit architecture-to-constructor bootstrap data.

This composition root is the only startup location that binds the concrete
model classes to their stable architecture identifiers. Request execution
receives only the ready model resolved through the catalog; the architecture
identifier plays no role past loading.
"""

from __future__ import annotations

from ..models.bagel import BagelForUnifiedGeneration
from ..models.catalog import Catalog
from ..models.qwen3 import Qwen3ForCausalLM
from ..models.sensenova.model import SenseNovaU1ForUnifiedGeneration

__all__ = ["MODEL_CATALOG"]

MODEL_CATALOG = Catalog(
    (
        Qwen3ForCausalLM,
        BagelForUnifiedGeneration,
        SenseNovaU1ForUnifiedGeneration,
    )
)
