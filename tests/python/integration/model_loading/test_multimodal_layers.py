"""Conformance for shared multimodal helpers."""
from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from uniserve_worker.models.bagel import BagelForUnifiedGeneration
from uniserve_worker.processors import get_processor_for_model
from uniserve_worker.processors.bagel import BagelImageProcessor
from uniserve_worker.server.runner_driver import RunnerDriver

pytestmark = pytest.mark.integration


def test_bagel_processor_decodes_transparent_images_on_white_background():
    image = Image.new("RGBA", (1, 1), (255, 0, 0, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    decoded = BagelImageProcessor.decode_image_b64(base64.b64encode(buf.getvalue()).decode())
    assert decoded.mode == "RGB"
    assert decoded.getpixel((0, 0)) == (255, 255, 255)


def test_processor_registry_auto_discovers_bagel_processor():
    assert isinstance(get_processor_for_model(BagelForUnifiedGeneration), BagelImageProcessor)


def test_runner_driver_attaches_discovered_multimodal_processor():
    driver = RunnerDriver(BagelForUnifiedGeneration(config={}))
    assert isinstance(driver.runner.multimodal_processor, BagelImageProcessor)
