from __future__ import annotations

from pathlib import Path

import pytest

from uniserve_eval.profiles import load_config

pytestmark = pytest.mark.unit


def test_decode_runtime_suite_resolves_to_four_explicit_points() -> None:
    config = load_config()
    points = config.selected_points("decode-runtime")
    assert tuple(point.name for point in points) == (
        "qwen-uniserve-sharegpt-r16",
        "sensenova-uniserve-i2t-c32",
        "sensenova-uniserve-t2i-c32",
        "sensenova-uniserve-interleave-c4",
    )
    assert tuple((metric.name, metric.direction) for metric in points[-1].metrics) == (
        ("mean_ttft_ms", "lower"),
        ("mean_tpot_ms", "lower"),
        ("images.image_latency_ms.mean", "lower"),
    )
    assert config.suites["decode-runtime"].max_regression is None


def test_toml_rejects_an_unknown_benchmark_field(tmp_path: Path) -> None:
    config = tmp_path / "profiles.toml"
    config.write_text(
        """
[servers.local]
port = 8000
command = ["server"]

[benchmarks.point]
server = "local"
task = "text"
model = "model"
unexpected = true

[benchmarks.point.metrics]
output_throughput = "higher"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected"):
        load_config(config)
