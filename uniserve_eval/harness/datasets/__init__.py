"""Dataset loaders for the UniServe serving benchmark.

Each task is backed by a real dataset (ShareGPT / MJHQ-30K / PIE-Bench / UEval),
selected by ``spec.dataset``. ``trace`` loads an arbitrary JSONL via
``--dataset-path`` for ad-hoc workloads.
"""
from __future__ import annotations

from typing import Any

from ..spec import BenchmarkSpec, TaskName
from .mjhq import load_mjhq
from .pie_bench import load_pie_bench
from .sharegpt import load_sharegpt
from .synthetic_images import load_image_dir, load_synthetic_images
from .trace import trace_items
from .ueval import load_ueval

__all__ = [
    "load_benchmark_inputs",
    "load_dataset_rows",
    "load_image_dir",
    "load_mjhq",
    "load_pie_bench",
    "load_sharegpt",
    "load_synthetic_images",
    "load_ueval",
    "trace_items",
]

_PIE_BENCH_ALIASES = {"pie-bench", "pie_bench", "piebench", "pie"}
_SYNTHETIC_IMAGE_ALIASES = {"synthetic-images", "synthetic_images", "synthetic"}


def load_benchmark_inputs(spec: BenchmarkSpec) -> tuple[list[dict[str, Any]], Any | None]:
    tokenizer: Any | None = None
    if spec.task == TaskName.TEXT:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            spec.tokenizer or spec.model,
            trust_remote_code=True,
        )
    rows = load_dataset_rows(spec, tokenizer=tokenizer)
    if len(rows) > spec.num_prompts:
        rows = rows[: spec.num_prompts]
    if len(rows) != spec.num_prompts:
        raise ValueError(
            f"dataset resolved {len(rows)} rows; benchmark contract requires "
            f"exactly {spec.num_prompts}"
        )
    return rows, tokenizer


def load_dataset_rows(spec: BenchmarkSpec, *, tokenizer: Any | None = None) -> list[dict[str, Any]]:
    """Resolve ``spec`` to a concrete list of request rows."""
    dataset = (spec.dataset or "").lower()
    path = spec.dataset_path or None

    if dataset == "trace":
        if not path:
            raise ValueError("dataset 'trace' requires --dataset-path")
        return trace_items(path)[: spec.num_prompts]

    if dataset == "sharegpt" or spec.task == TaskName.TEXT:
        if tokenizer is None:
            raise ValueError("ShareGPT requires a tokenizer (pass --tokenizer or --model)")
        return load_sharegpt(
            path or "",
            spec.num_prompts,
            tokenizer,
            fixed_output_len=spec.sharegpt_output_len,
            context_len=spec.sharegpt_context_len,
            seed=spec.seed,
        )

    if dataset == "mjhq" or spec.task == TaskName.T2I:
        return load_mjhq(path, spec.num_prompts, seed=spec.seed)

    if dataset in _SYNTHETIC_IMAGE_ALIASES or (
        not dataset and spec.task == TaskName.I2T
    ):
        return load_synthetic_images(
            spec.num_prompts, seed=spec.seed, question=spec.i2t_question
        )

    if dataset == "image-dir":
        if not path:
            raise ValueError("dataset 'image-dir' requires --dataset-path")
        return load_image_dir(
            path, spec.num_prompts, seed=spec.seed, question=spec.i2t_question
        )

    if dataset in _PIE_BENCH_ALIASES or spec.task == TaskName.I2I:
        return load_pie_bench(path, spec.num_prompts, seed=spec.seed)

    if dataset == "ueval" or spec.task == TaskName.DEFAULT:
        return load_ueval(path, spec.num_prompts, seed=spec.seed)

    raise ValueError(f"unknown dataset {spec.dataset!r} for task {spec.task.value!r}")
