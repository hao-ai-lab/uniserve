from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

import uniserve_eval.harness.datasets.mixed_image_text as mixed_dataset
from uniserve_eval.harness.metrics import summarize_mixed
from uniserve_eval.harness.metrics.common import RequestRecord
from uniserve_eval.harness.spec import BenchmarkSpec, TaskName
from uniserve_eval.harness.tasks.mixed import MixedTask

pytestmark = pytest.mark.unit


def _spec(image_dir: Path) -> BenchmarkSpec:
    return BenchmarkSpec(
        task=TaskName.MIXED,
        model="BAGEL",
        dataset_path=str(image_dir),
        num_prompts=8,
        warmup_requests=4,
        workload_mix={"t2i": 2, "i2t": 6},
        warmup_mix={"t2i": 1, "i2t": 3},
        max_tokens=256,
        width=1024,
        height=1024,
        steps=50,
        denoise_updates=50,
        max_images=1,
    )


def test_mixed_dataset_interleaves_disjoint_measured_and_warmup_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        mixed_dataset,
        "load_mjhq",
        lambda _path, count, **_kwargs: [
            {"id": f"t{index}", "task": "t2i", "prompt": "draw"} for index in range(count)
        ],
    )
    monkeypatch.setattr(
        mixed_dataset,
        "load_image_dir",
        lambda _path, count, **_kwargs: [
            {"id": f"i{index}", "prompt": "describe", "input_image_b64": "QUJD"}
            for index in range(count)
        ],
    )

    rows = mixed_dataset.load_mixed_image_text(_spec(tmp_path))

    assert [row["task"] for row in rows.measured] == [
        "t2i",
        "i2t",
        "i2t",
        "i2t",
        "t2i",
        "i2t",
        "i2t",
        "i2t",
    ]
    assert [row["task"] for row in rows.warmup] == ["t2i", "i2t", "i2t", "i2t"]
    assert {row["id"] for row in rows.measured}.isdisjoint(row["id"] for row in rows.warmup)
    assert all(row["max_tokens"] == 256 for row in rows.measured if row["task"] == "i2t")


@pytest.mark.parametrize(
    ("workload_mix", "warmup_mix", "measured_block", "warmup_block"),
    [
        (
            {"t2i": 8, "i2t": 24},
            {"t2i": 1, "i2t": 3},
            ["t2i", "i2t", "i2t", "i2t"],
            ["t2i", "i2t", "i2t", "i2t"],
        ),
        (
            {"t2i": 16, "i2t": 16},
            {"t2i": 2, "i2t": 2},
            ["t2i", "i2t"],
            ["t2i", "i2t"],
        ),
        (
            {"t2i": 24, "i2t": 8},
            {"t2i": 3, "i2t": 1},
            ["t2i", "t2i", "t2i", "i2t"],
            ["t2i", "t2i", "t2i", "i2t"],
        ),
    ],
)
def test_mixed_matrix_ratios_use_the_declared_proportional_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workload_mix: dict[str, int],
    warmup_mix: dict[str, int],
    measured_block: list[str],
    warmup_block: list[str],
) -> None:
    monkeypatch.setattr(
        mixed_dataset,
        "load_mjhq",
        lambda _path, count, **_kwargs: [
            {"id": f"t{index}", "task": "t2i", "prompt": "draw"} for index in range(count)
        ],
    )
    monkeypatch.setattr(
        mixed_dataset,
        "load_image_dir",
        lambda _path, count, **_kwargs: [
            {"id": f"i{index}", "prompt": "describe", "input_image_b64": "QUJD"}
            for index in range(count)
        ],
    )
    spec = dataclasses.replace(
        _spec(tmp_path),
        num_prompts=32,
        workload_mix=workload_mix,
        warmup_mix=warmup_mix,
    )

    rows = mixed_dataset.load_mixed_image_text(spec)

    assert [row["task"] for row in rows.measured] == measured_block * (32 // len(measured_block))
    assert [row["task"] for row in rows.warmup] == warmup_block * (4 // len(warmup_block))


def test_mixed_task_routes_each_row_through_its_semantic_adapter(tmp_path: Path) -> None:
    task = MixedTask(_spec(tmp_path))

    image = task.build_request({"task": "t2i", "prompt": "draw"})
    text = task.build_request({"task": "i2t", "prompt": "describe", "input_image_b64": "QUJD"})

    assert image.semantic_task == "t2i"
    assert image.kind == "openai_chat_json"
    assert image.payload["modalities"] == ["image"]
    assert text.semantic_task == "i2t"
    assert text.kind == "openai_chat"
    assert text.payload["modalities"] == ["text"]
    assert text.payload["max_completion_tokens"] == 256


def test_mixed_metrics_keep_task_units_separate_and_report_client_overlap() -> None:
    records = [
        RequestRecord(
            request_id="image",
            task="t2i",
            success=True,
            start_time=10.0,
            latency=4.0,
            images=1,
        ),
        RequestRecord(
            request_id="text",
            task="i2t",
            success=True,
            start_time=11.0,
            latency=2.0,
            output_len=256,
            prompt_len=32,
            token_timing_available=True,
            ttft=0.5,
        ),
    ]

    metrics = summarize_mixed(records, dur_s=4.0)

    assert metrics["mixed_request_throughput"] == pytest.approx(0.5)
    assert metrics["t2i"]["images_per_minute"] == pytest.approx(15.0)
    assert metrics["i2t"]["output_throughput"] == pytest.approx(64.0)
    assert metrics["client_cross_task_overlap"] == {
        "duration_s": pytest.approx(2.0),
        "timed_region_fraction": pytest.approx(0.5),
        "common_active_time_fraction": pytest.approx(1.0),
    }


def test_mixed_spec_rejects_a_task_mix_that_changes_the_declared_count(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="workload_mix counts must sum to 8"):
        BenchmarkSpec(
            task=TaskName.MIXED,
            model="BAGEL",
            dataset_path=str(tmp_path),
            num_prompts=8,
            warmup_requests=4,
            workload_mix={"t2i": 1, "i2t": 6},
            warmup_mix={"t2i": 1, "i2t": 3},
        )
