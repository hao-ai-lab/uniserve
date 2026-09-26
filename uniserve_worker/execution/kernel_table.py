"""The startup table of the kernel serving every prepared call site.

Startup logs one line, ``uniserve-kernel-table <JSON>``, once warmup and
graph capture have resolved every selection the served call kinds make. The
JSON object has these fields:

- ``rank`` and ``device``: the worker rank and its compute device;
- ``runners``: one entry per runner (a bound computation and its call
  kinds), with ``runner``, ``device`` and ``call_sites``. Each call site
  groups the layers whose records are identical apart from their module
  path: ``layers`` lists the paths, with runs of layer indices compressed as
  ``{0-63}``, and the remaining fields are the ``ExecutionContext.kernels``
  record: ``op`` (``attention``, ``vsa``, ``moe`` or ``matmul``), the
  representation, the resolved ``provider`` and, for automatic attention
  selection, ``inputs``, the provider that served each input class;
- ``portable``: every call site a CUDA runner serves with the portable torch
  provider, listed again so that no such call site is silent.

Harnesses capture the line from the worker log as their artifact.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from uniserve.runtime.device import canonical_device

KERNEL_TABLE_TAG = "uniserve-kernel-table"


def _ranges(indices: list[int]) -> str:
    """Compress sorted integers into ``a-b`` runs joined by commas."""
    runs: list[tuple[int, int]] = []
    for index in indices:
        if runs and index == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((index, index))
    return ",".join(
        str(start) if start == stop else f"{start}-{stop}"
        for start, stop in runs
    )


def layer_paths(paths: Iterable[str]) -> list[str]:
    """Compress module paths that differ only in one numeric component.

    ``model.layers.3.attn`` and ``model.layers.4.attn`` become
    ``model.layers.{3-4}.attn``. Paths whose numeric components differ in
    more than one position are listed individually.
    """
    templates: dict[tuple[str, ...], list[list[str]]] = {}
    for path in paths:
        parts = path.split(".")
        key = tuple("*" if part.isdigit() else part for part in parts)
        templates.setdefault(key, []).append(parts)

    result: list[str] = []
    for key, members in templates.items():
        varying = [
            position
            for position, part in enumerate(key)
            if part == "*" and len({member[position] for member in members}) > 1
        ]
        if len(varying) > 1:
            result.extend(".".join(member) for member in members)
            continue
        parts = list(members[0])
        if varying:
            position = varying[0]
            indices = sorted({int(member[position]) for member in members})
            parts[position] = "{" + _ranges(indices) + "}"
        result.append(".".join(parts))
    return result


def _label(runner: Any) -> str:
    """Name a runner by its component, entry method and call kinds."""
    label = str(runner.name)
    entry = getattr(runner.call, "entry_point", None)
    if entry is not None:
        label += f".{entry.method}"
    kinds = tuple(str(kind) for kind in getattr(runner, "call_kinds", ()))
    if kinds:
        label += f"[{','.join(kinds)}]"
    return label


def _portable(record: dict[str, Any]) -> list[str]:
    """Return what a record serves with the portable torch provider."""
    if record["provider"] == "torch":
        return ["every input"]
    return [
        name
        for name, provider in record.get("inputs", {}).items()
        if provider == "torch"
    ]


def kernel_table(runners: Iterable[Any], *, rank: int, device: str) -> dict:
    """Group the kernel records of ``runners`` for the startup table.

    ``runners`` are the worker's prepared runners, each with ``name``,
    ``call``, ``device`` and an ``ExecutionContext`` as ``context``. Runners
    with the same label (one computation prepared at several sizes) merge.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for runner in runners:
        label = _label(runner)
        entry = grouped.setdefault(
            label,
            {"device": str(runner.device), "sites": {}},
        )
        for record in runner.context.kernels():
            record = dict(record)
            path = record.pop("path")
            key = json.dumps(record, sort_keys=True)
            entry["sites"].setdefault(key, (record, []))[1].append(path)

    table: dict[str, Any] = {
        "rank": rank,
        "device": device,
        "runners": [],
        "portable": [],
    }
    for label, entry in grouped.items():
        sites = []
        for record, paths in entry["sites"].values():
            layers = layer_paths(dict.fromkeys(paths))
            sites.append({"layers": layers, **record})
            served = _portable(record)
            if served and canonical_device(entry["device"]).type == "cuda":
                table["portable"].append(
                    {
                        "runner": label,
                        "op": record["op"],
                        "layers": layers,
                        "inputs": served,
                    }
                )
        table["runners"].append(
            {"runner": label, "device": entry["device"], "call_sites": sites}
        )
    return table


def format_kernel_table(table: dict) -> str:
    """Render the table as its single tagged log line."""
    return f"{KERNEL_TABLE_TAG} {json.dumps(table, separators=(',', ':'))}"
