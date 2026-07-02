"""Side-by-side comparison of measured perf points (``compare`` subcommand).

Reads each workload's ``summary.json`` from its artifact directory and emits a
markdown table (rows = metrics, columns = workloads) plus a ratio column per
candidate against the first workload (the baseline). Higher-is-better metrics
are reported as candidate/baseline; latency-like metrics as baseline/candidate,
so >1.0 always reads "candidate is better". The table prints to stdout and is
written with its raw data under ``<artifact_root>/comparisons/<name>/`` as
``compare.md`` + ``compare.json``.

Runs standalone (``uniserve-eval compare <workload> <workload> [...]``) and
automatically after suites that declare ``compare`` groups.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from .profiles import artifact_root, load_config, workload_dir

# (display label, dotted path into the summary, higher-is-better)
_FAMILY_METRICS: dict[str, list[tuple[str, str, bool]]] = {
    "image": [
        ("image_latency_ms.mean", "metrics.image_latency_ms.mean", False),
        ("image_latency_ms.p50", "metrics.image_latency_ms.p50", False),
        ("images_per_minute", "metrics.images_per_minute", True),
    ],
    "stream": [
        ("output_throughput", "metrics.output_throughput", True),
        ("mean_ttft_ms", "metrics.mean_ttft_ms", False),
        ("mean_itl_ms", "metrics.mean_itl_ms", False),
        ("mean_e2e_latency_ms", "metrics.mean_e2e_latency_ms", False),
    ],
}

_COMMON_METRICS: list[tuple[str, str, bool]] = [
    ("request_count", "request_count", True),
    ("ok_count", "ok_count", True),
    ("gpu_memory.peak_single_gpu_mib", "gpu_memory.peak_single_gpu_mib", False),
]


def _lookup(summary: dict[str, Any], dotted: str) -> float | None:
    node: Any = summary
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, (int, float)) else None


def _load_summaries(config: dict[str, Any], names: list[str]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for name in names:
        path = workload_dir(config, name) / "summary.json"
        if not path.exists():
            raise SystemExit(
                f"no summary for workload {name!r} at {path}; "
                f"run the point first: uniserve-eval perf {name}"
            )
        summaries[name] = json.loads(path.read_text(encoding="utf-8"))
    return summaries


def _fmt_value(value: float | None) -> str:
    if value is None:
        return "-"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def _fmt_ratio(ratio: float | None) -> str:
    return "-" if ratio is None else f"{ratio:.2f}x"


def compare_workloads(
    config: dict[str, Any], names: list[str], comparison_name: str | None = None
) -> dict[str, Any]:
    """Build, print, and persist the comparison. Returns the raw comparison data."""
    if len(names) < 2:
        raise SystemExit("compare needs at least two workloads")
    summaries = _load_summaries(config, names)
    families = {name: summaries[name].get("metric_family") for name in names}
    if len(set(families.values())) != 1:
        raise SystemExit(f"refusing to compare mixed metric families: {families}")
    family = str(families[names[0]])
    if family not in _FAMILY_METRICS:
        raise SystemExit(f"unknown metric family {family!r}")

    metrics = _FAMILY_METRICS[family] + [
        row
        for row in _COMMON_METRICS
        if any(_lookup(summaries[name], row[1]) is not None for name in names)
    ]
    baseline = names[0]
    candidates = names[1:]

    rows: list[dict[str, Any]] = []
    for label, dotted, higher_is_better in metrics:
        values = {name: _lookup(summaries[name], dotted) for name in names}
        ratios: dict[str, float | None] = {}
        base = values[baseline]
        for candidate in candidates:
            cand = values[candidate]
            if base is None or cand is None:
                ratios[candidate] = None
            elif higher_is_better:
                ratios[candidate] = cand / base if base else None
            else:
                ratios[candidate] = base / cand if cand else None
        rows.append(
            {
                "metric": label,
                "higher_is_better": higher_is_better,
                "values": values,
                "ratios": ratios,
            }
        )

    header = ["metric", f"{baseline} (baseline)"]
    for candidate in candidates:
        header.extend([candidate, f"{candidate} (x vs baseline)"])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [str(row["metric"]), _fmt_value(row["values"][baseline])]
        for candidate in candidates:
            cells.extend([_fmt_value(row["values"][candidate]), _fmt_ratio(row["ratios"][candidate])])
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)

    name = comparison_name or "_vs_".join(names)
    data = {
        "name": name,
        "metric_family": family,
        "baseline": baseline,
        "workloads": names,
        "rows": rows,
        "ratio_semantics": "candidate/baseline for higher-is-better metrics, baseline/candidate for latency metrics; >1.0 means the candidate is better",
    }
    out_dir = artifact_root(config) / "comparisons" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare.md").write_text(f"# {name}\n\n{table}\n", encoding="utf-8")
    (out_dir / "compare.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(table)
    print(f"comparison written: {out_dir}")
    return data


def compare(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    compare_workloads(config, list(args.workloads), getattr(args, "name", None))
