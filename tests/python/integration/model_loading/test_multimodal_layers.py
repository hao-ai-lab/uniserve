"""Conformance for shared multimodal helpers."""

from __future__ import annotations

import pytest

from uniserve_worker.models.bagel import BagelForUnifiedGeneration
from uniserve_worker.processors import get_processor_for_model
from uniserve_worker.processors.bagel import BagelImageProcessor
from uniserve_worker.worker.model import ModelWorker

pytestmark = pytest.mark.integration


def test_processor_registry_auto_discovers_bagel_processor():
    assert isinstance(get_processor_for_model(BagelForUnifiedGeneration), BagelImageProcessor)


def test_model_worker_attaches_discovered_multimodal_processor():
    worker = ModelWorker(BagelForUnifiedGeneration(config={}))
    assert isinstance(
        worker.model_runner.multimodal_processor,
        BagelImageProcessor,
    )
