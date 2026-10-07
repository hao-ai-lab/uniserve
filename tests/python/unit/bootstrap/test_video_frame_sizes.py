"""A deployment's ``--video-frame-sizes`` selects the H3 output rasters."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from uniserve.media import image
from uniserve_models.minimax_h3 import Config, output
from uniserve_models.minimax_h3.packing import FRAME_SIZES
from uniserve_worker.bootstrap.model_loader import _serve_frame_sizes
from uniserve_worker.config.execution import _parse_image_shapes
from uniserve_worker.errors import WorkerError

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class Source:
    model: object


def test_frame_sizes_parse_as_height_by_width():
    parse = lambda raw: _parse_image_shapes(  # noqa: E731
        raw, default=(), name="video frame sizes"
    )
    assert parse(None) == ()
    assert parse("768x1344, 1344x768") == ((768, 1344), (1344, 768))
    for raw in ("768", "768x1344,768x1344", "0x768"):
        with pytest.raises(ValueError, match="video frame sizes"):
            parse(raw)


def test_output_rasters_must_be_distinct_training_buckets():
    rasters = tuple(
        image.Config(height, width) for height, width in FRAME_SIZES
    )
    assert len(rasters) == 12
    assert output.Config(frame_sizes=rasters).frame_sizes == rasters
    with pytest.raises(ValueError, match="training buckets"):
        output.Config(frame_sizes=(image.Config(1024, 1024),))
    with pytest.raises(ValueError, match="distinct"):
        output.Config(frame_sizes=(rasters[0], rasters[0]))


def test_deployment_rasters_replace_the_checkpoint_rasters():
    source = Source(Config())
    assert (
        _serve_frame_sizes(source, SimpleNamespace(video_frame_sizes=()))
        is source
    )

    served = _serve_frame_sizes(
        source, SimpleNamespace(video_frame_sizes=((416, 992), (768, 768)))
    )
    assert served.model.output.frame_sizes == (
        image.Config(416, 992),
        image.Config(768, 768),
    )
    with pytest.raises(WorkerError):
        _serve_frame_sizes(
            source, SimpleNamespace(video_frame_sizes=((32, 32),))
        )
    with pytest.raises(WorkerError):
        _serve_frame_sizes(
            Source(SimpleNamespace()),
            SimpleNamespace(video_frame_sizes=((768, 768),)),
        )
