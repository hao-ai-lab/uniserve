"""The startup table of the kernel serving every prepared call site.

Startup logs one line, ``uniserve-kernel-table <JSON>``, once warmup and
graph capture have resolved the selections they exercise. A call site that
prepares at its first call and a call kind that startup does not exercise
choose their kernels while serving; the worker then logs the complete table
again after the call that made the choice. The last line is therefore the
complete record. The JSON object has these fields:

- ``stage``: ``startup`` or ``serving``;
- ``rank`` and ``device``: the worker rank and its compute device;
- ``runners``: one entry per runner (a bound computation and its call
  kinds), with ``runner``, ``device`` and ``call_sites``. Each call site
  groups the layers whose records are identical apart from their module
  path: ``layers`` lists the paths, with runs of layer indices compressed as
  ``{0-63}``, and the remaining fields are the runner's kernel record
  (``ModelRunner.kernels``): ``op`` (``attention``, ``vsa``, ``moe`` or
  ``matmul`` from ``ExecutionContext.kernels``, or ``product`` for the
  canvas sampler's self-conditioning product), the representation, the
  resolved ``provider`` and, for automatic attention selection, ``inputs``,
  the provider that served each input class. An attention call site not
  called yet has ``dtype`` and ``provider`` None;
- ``portable``: every call site a CUDA runner serves with the portable torch
  provider, listed again so that no such call site is silent.

Harnesses capture the line from the worker log as their artifact.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any

import torch

from uniserve.runtime.device import canonical_device, process_device_bytes

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
    # Call kinds form a set; sorting keeps one runner's label identical
    # across processes and runs.
    kinds = sorted(str(kind) for kind in getattr(runner, "call_kinds", ()))
    if kinds:
        label += f"[{','.join(kinds)}]"
    return label


def _portable(record: dict[str, Any]) -> list[str]:
    """Return what a record serves with the portable torch provider."""
    if record["provider"] is None:
        return []
    if record["provider"] == "torch":
        return ["every input"]
    return [
        name
        for name, provider in record.get("inputs", {}).items()
        if provider == "torch"
    ]


class KernelRecords:
    """Accumulate the kernel records of a worker's runners.

    A call site's records are those its runners report the last time they
    are gathered: a record that grows (an automatic selection meeting a new
    input class, a first call preparing a site) replaces its earlier form.
    A computation retired while serving keeps the records it last reported.
    """

    def __init__(self) -> None:
        # Runner label -> (device, {path: {record key: record}}).
        self._runners: dict[str, tuple[str, dict[str, dict[str, dict]]]] = {}

    def add(self, runners: Iterable[Any]) -> bool:
        """Gather the current records of ``runners``; report any change.

        ``runners`` are the worker's runners, each with ``name``, ``call``,
        ``device`` and ``kernels()`` (``ModelRunner.kernels``). Runners with
        the same label (one computation prepared at several sizes) merge.
        """
        current: dict[str, tuple[str, dict[str, dict[str, dict]]]] = {}
        for runner in runners:
            _, paths = current.setdefault(
                _label(runner), (str(runner.device), {})
            )
            for record in runner.kernels():
                record = dict(record)
                site = paths.setdefault(record.pop("path"), {})
                site[json.dumps(record, sort_keys=True)] = record

        changed = False
        for label, (runner_device, paths) in current.items():
            _, stored = self._runners.setdefault(label, (runner_device, {}))
            for path, records in paths.items():
                if stored.get(path) != records:
                    stored[path] = records
                    changed = True
        return changed

    def table(self, *, stage: str, rank: int, device: str) -> dict:
        """Return the logged table of every gathered record.

        ``stage`` is ``startup`` or ``serving``; ``rank`` and ``device`` name
        the worker rank and its compute device.
        """
        table: dict[str, Any] = {
            "stage": stage,
            "rank": rank,
            "device": device,
            "runners": [],
            "portable": [],
        }
        for label, (runner_device, paths) in self._runners.items():
            # Group the paths of identical records into one call site. A
            # site prepared at several sizes may report a prepared and a
            # not yet called form; only the prepared one is kept.
            sites: dict[str, tuple[dict, list[str]]] = {}
            for path, records in paths.items():
                prepared = any(
                    record["op"] == "attention" and record["provider"]
                    for record in records.values()
                )
                for key, record in records.items():
                    unprepared = (
                        record["op"] == "attention"
                        and record["provider"] is None
                    )
                    if not (prepared and unprepared):
                        sites.setdefault(key, (record, []))[1].append(path)

            call_sites = []
            for record, site_paths in sites.values():
                layers = layer_paths(site_paths)
                call_sites.append({"layers": layers, **record})
                served = _portable(record)
                if served and canonical_device(runner_device).type == "cuda":
                    table["portable"].append(
                        {
                            "runner": label,
                            "op": record["op"],
                            "layers": layers,
                            "inputs": served,
                        }
                    )
            table["runners"].append(
                {
                    "runner": label,
                    "device": runner_device,
                    "call_sites": call_sites,
                }
            )
        return table


def format_kernel_table(table: dict) -> str:
    """Render the table as its single tagged log line."""
    return f"{KERNEL_TABLE_TAG} {json.dumps(table, separators=(',', ':'))}"


def log_startup_memory(storage) -> None:
    """Report process, allocator, and graph pool memory after warmup."""
    logger = logging.getLogger(__name__)
    # Resident storage by device and, for graph pools, by runner kind,
    # for sizing. Scratch and eager warm calls allocate outside the pools,
    # and graph executables, communicators and loaded modules outside the
    # caching allocator, so the process's whole device footprint is
    # reported alongside the allocator's reservation and the pools.
    for device, pooled in sorted(storage.pool_bytes().items(), key=str):
        logger.info(
            "device storage on %s: %.2f GiB held by this process at "
            "startup, %.2f GiB of it reserved by the caching allocator "
            "and %.2f GiB of that in graph pools",
            device,
            process_device_bytes(device) / 2**30,
            torch.cuda.memory_reserved(device) / 2**30,
            pooled / 2**30,
        )
    totals: dict = {}
    for (owner, device), used in storage.owner_bytes().items():
        label = owner.name
        totals[device, label] = totals.get((device, label), 0) + used
    for (device, label), used in sorted(
        totals.items(), key=lambda item: (str(item[0][0]), item[0][1])
    ):
        logger.info(
            "graph storage on %s: %s holds %.2f GiB at startup",
            device,
            label,
            used / 2**30,
        )
