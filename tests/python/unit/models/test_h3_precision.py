"""H3 precision choices reject unsupported per-component formats."""

import pytest

from uniserve_models.minimax_h3 import weight_config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "choices",
    [
        {"attention": "mxfp8"},
        {"mlp": "fp16"},
        {"text_encoder": "mxfp8"},
        {"video_vae": "fp8"},
    ],
)
def test_unsupported_component_formats(choices):
    with pytest.raises(ValueError, match=next(iter(choices))):
        weight_config(**choices)
