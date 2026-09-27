"""Captures a server's Prometheus counters across a measured window.

`run_point` scrapes `GET /metrics` just before the measured window opens and
again after it closes, persists both expositions verbatim, and summarizes
how much every counter advanced in between. Counter samples are those whose
names end in `_total`, or a histogram's `_sum` and `_count`; the difference
of a gauge between two instants carries no window meaning and is left out.
UniServe's execution-domain times (`uniserve:scheduler_domain_time_us`, by
domain and phase) and worker forward times are such counters, so their
deltas attribute the window's server time; other servers' counters are kept
under their own names.
"""

from __future__ import annotations

import math
from typing import Any

import httpx

# Sample-name suffixes whose difference across a window is meaningful.
_COUNTER_SUFFIXES = ("_total", "_sum", "_count")


async def scrape_metrics(
    client: httpx.AsyncClient, base_url: str
) -> str | None:
    """Return the server's `/metrics` exposition text, if it serves one.

    A non-200 status or any request error yields `None`: a server without
    metrics (or with them disabled) still completes its benchmark point.
    """
    try:
        response = await client.get(
            base_url.rstrip("/") + "/metrics", timeout=15.0
        )
    except Exception:
        return None
    if response.status_code != 200:
        return None
    return response.text


def parse_samples(text: str) -> dict[str, float]:
    """Parse exposition sample lines into a `name{labels}` -> value map.

    Comment and metadata lines (`#`) and lines without a numeric value are
    skipped. The label set is kept verbatim as the server rendered it, so
    two scrapes of one server key the same series identically. A trailing
    timestamp, when present, is ignored.
    """
    samples: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Label values may contain spaces, so the series key ends at the
        # closing brace when there is a label set.
        if "{" in line:
            close = line.rfind("}")
            if close < 0:
                continue
            key, rest = line[: close + 1], line[close + 1 :]
        else:
            key, _, rest = line.partition(" ")
        fields = rest.split()
        if not fields:
            continue
        try:
            value = float(fields[0])
        except ValueError:
            continue
        samples[key] = value
    return samples


def counter_deltas(before: str, after: str) -> dict[str, float]:
    """Return how far each counter series advanced between two scrapes.

    A series absent from the first scrape started at zero. Series that did
    not advance are omitted, which keeps the summary to the work the window
    performed; so is any series whose delta is not finite.
    """
    start = parse_samples(before)
    deltas: dict[str, float] = {}
    for key, value in parse_samples(after).items():
        name = key.split("{", 1)[0]
        if not name.endswith(_COUNTER_SUFFIXES):
            continue
        delta = value - start.get(key, 0.0)
        if delta != 0.0 and math.isfinite(delta):
            deltas[key] = delta
    return dict(sorted(deltas.items()))


def window_summary(before: str | None, after: str | None) -> dict[str, Any]:
    """Summarize the measured window's counter deltas for a run summary.

    `available` is false when either scrape failed, and the deltas are then
    empty.
    """
    if before is None or after is None:
        return {"available": False, "counter_deltas": {}}
    return {"available": True, "counter_deltas": counter_deltas(before, after)}
