r"""Measured grouped-expert kernel tactics per routed token count.

FlashInfer's grouped expert kernels run each GEMM with a tactic: its tile,
cluster and scheduling configuration. Without one they run a fixed default
(CUTLASS: the first 128x128x128 configuration of each GEMM; CuTeDSL:
128-row routing tiles and 128x128 MMA tiles) whatever the expert shape and
token count. The shipped table ``tactics.json`` holds, per device model,
FlashInfer version, provider and expert shape, the tactic measured best
at each measured token count. A prepared operator (see :class:`Tactics`)
runs a call of ``T`` routed tokens with the entry of the largest measured
count not above ``T``, and with the kernel's default below the smallest
count or when no entry matches its key.

A provider describes its tactic space as independent dimensions (for
CUTLASS the GEMM1 and GEMM2 configurations, for CuTeDSL one dimension of
complete tactics); the first option of each dimension is the kernel's
default. The measurement starts from the defaults and sweeps one dimension
at a time with the others held at their best, as FlashInfer's autotuner
tunes the CUTLASS GEMMs one after the other. It times every candidate
under each routing distribution of ``ROUTINGS`` and keeps the one whose
slowest routing, relative to the default under that routing, is fastest,
so a table entry is never slower than the default under any of them. It
admits a candidate only when its outputs equal the default tactic's bit
for bit and repeat identically in a replayed graph, so the table changes
device time, not results. Run it on an otherwise idle device of the model
the entries are for::

    python -m uniserve.runtime.backends.moe.tactics --provider cutlass \
        --format bfloat16 --experts 128 --hidden 2816 --intermediate 704 \
        --top-k 8 --activation gelu_tanh [--tokens 1 2 4 ...] [--output FILE]

Entries of other keys in the file stay. Weights and hidden states are
random; the kernels' speed depends on the values only through power draw,
and on the routing through how many rows each expert receives.
"""

from __future__ import annotations

import argparse
import datetime
import json
import statistics
from bisect import bisect_right
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import torch
from torch import nn

from uniserve.quantization import QuantizedTensor, Quantizer

# Tactics measured per device model, FlashInfer version, provider and expert
# shape; this module's measurement writes the file.
TACTIC_TABLE = Path(__file__).with_name("tactics.json")
KEY_FIELDS = (
    "device",
    "flashinfer",
    "provider",
    "format",
    "experts",
    "hidden",
    "intermediate",
    "top_k",
    "activation",
)
# Routing distributions every admitted tactic must serve, by the size of
# the expert pool each token draws its distinct experts from: ``uniform``
# draws from every expert, as tokens of varied text route; ``shared`` from
# ``2 * top_k`` experts, as canvas positions that still hold the same mask
# token route alike. The best tactic of one can be slower than the default
# under the other.
ROUTINGS = {
    "uniform": lambda args: args.experts,
    "shared": lambda args: min(args.experts, 2 * args.top_k),
}
DEFAULT_TOKENS = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048,
    3072, 4096, 6144, 8192, 12288, 16384,
)  # fmt: skip


def table_key(provider: str, module) -> tuple:
    """The table key of ``provider`` evaluating the ``FusedMoE`` ``module``.

    (device name, FlashInfer version, provider, weight format, resident
    experts, hidden width, resident intermediate width, top-k, activation).
    The weight format is the encoding (``nvfp4``) or the dense dtype name.
    The experts are those resident on this rank, the problem the grouped
    kernel runs: the table measures every expert resident with routes over
    all of them, so an expert-parallel layer's local share (a fraction of
    the experts, an expert offset, and received rows that mostly skip) has
    no entry and runs the kernel's default tactic.
    """
    import flashinfer

    weight = module.up_gate.weight
    encoding = (
        weight.quantizer.format
        if isinstance(weight, QuantizedTensor)
        else str(weight.dtype).removeprefix("torch.")
    )
    return (
        torch.cuda.get_device_name(weight.device),
        flashinfer.__version__,
        provider,
        encoding,
        weight.shape[0],
        module.hidden_size,
        module.down.weight.shape[2],
        module.top_k,
        module.activation,
    )


@cache
def _shipped() -> dict[tuple, tuple[tuple[int, list], ...]]:
    """Every shipped key's ``(tokens, tactic)`` entries, by token count."""
    table: dict[tuple, list[tuple[int, list]]] = {}
    for entry in json.loads(TACTIC_TABLE.read_text())["entries"]:
        key = tuple(entry[field] for field in KEY_FIELDS)
        table.setdefault(key, []).append((entry["tokens"], entry["tactic"]))
    return {
        key: tuple(sorted(rows, key=lambda row: row[0]))
        for key, rows in table.items()
    }


class Tactics:
    """The shipped tactics of one prepared call site, by routed token count.

    Built at preparation, on the host; :meth:`select` is a host lookup, so a
    captured call records the tactic of its token count.
    """

    def __init__(self, provider: str, module):
        rows = _shipped().get(table_key(provider, module), ())
        self._counts = [tokens for tokens, _ in rows]
        self._tactics = [tactic for _, tactic in rows]

    def select(self, tokens: int) -> list | None:
        """The tactic measured at the largest count not above ``tokens``.

        ``None`` when no measured count is that small or no entry matches:
        the kernel then runs its default.
        """
        index = bisect_right(self._counts, tokens) - 1
        return None if index < 0 else self._tactics[index]


def _experts(args, device: torch.device, generator: torch.Generator):
    """A ``FusedMoE`` of the requested shape with random resident weights.

    NVFP4 weights hold uniformly random E2M1 codes, E4M3 block scales in
    [0.5, 2] and per-expert tensor scales, with static activation scales
    that keep every product finite.
    """
    from uniserve.nn.moe import FusedMoE

    dtype = (
        torch.bfloat16
        if args.format == "nvfp4"
        else getattr(torch, args.format)
    )
    module = FusedMoE(
        args.experts,
        args.hidden,
        args.intermediate,
        top_k=args.top_k,
        activation=args.activation,
        device=device,
        dtype=dtype,
    )
    if args.format != "nvfp4":
        for linear in (module.up_gate, module.down):
            linear.weight.normal_(0.0, 0.02, generator=generator)
        return module

    for linear, rows, width, input_scale in (
        (module.up_gate, 2 * args.intermediate, args.hidden, 4.0 / 2688),
        (module.down, args.hidden, args.intermediate, 64.0 / 2688),
    ):
        values = torch.randint(
            0,
            256,
            (args.experts, rows, width // 2),
            generator=generator,
            device=device,
            dtype=torch.uint8,
        )
        # E4M3 bytes 0x30..0x40 encode 0.5 .. 2.
        block_scale = torch.randint(
            0x30,
            0x41,
            (args.experts * rows, width // 16),
            generator=generator,
            device=device,
            dtype=torch.uint8,
        )
        weight = Quantizer("nvfp4").from_tensors(
            {
                "values": values,
                "block_scale": block_scale,
                "tensor_scale": torch.full(
                    (args.experts,), 1.0 / 64, device=device
                ),
            },
            shape=(args.experts, rows, width),
            dtype=torch.bfloat16,
        )
        linear.weight = nn.Parameter(weight, requires_grad=False)
        # ModelOpt's calibrated scale amax / (6 * 448) for amax 4 and 64.
        linear.input_quantizer = Quantizer(
            "nvfp4", calibrated_scale=input_scale
        )
    return module


def _routes(tokens: int, pool: int, args, device, generator):
    """Random hidden states routed to distinct experts drawn from ``pool``.

    Each token's ``top_k`` experts are distinct and uniformly drawn from the
    first ``pool`` experts, with positive route weights summing to one.
    """
    dtype = (
        torch.bfloat16
        if args.format == "nvfp4"
        else getattr(torch, args.format)
    )
    hidden = torch.randn(
        tokens, args.hidden, generator=generator, device=device
    ).to(dtype)
    ids = (
        torch.rand(tokens, pool, generator=generator, device=device)
        .topk(args.top_k, dim=-1)
        .indices.to(torch.int32)
    )
    weights = torch.rand(tokens, args.top_k, generator=generator, device=device)
    weights = (weights + 0.25) / (weights + 0.25).sum(-1, keepdim=True)
    return hidden, ids, weights


def _time(operator, inputs, tactic, *, calls: int, rounds: int):
    """Median device time per call and the replayed output of ``tactic``.

    ``calls`` calls form one captured graph; each of ``rounds`` replays is
    timed with CUDA events. Returns ``(milliseconds, output)``.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        operator(*inputs, tactic=tactic)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            output = operator(*inputs, tactic=tactic)
    graph.replay()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) / calls)
    output = output.clone()
    del graph
    return statistics.median(samples), output


def _evaluate(operator, routings, tactic, expected, args):
    """Median per routing of ``tactic``, or why it is not admitted.

    Returns ``({routing: milliseconds}, None)``, or ``(None, reason)`` when
    the tactic raises (``failed``) or an eager or replayed output differs
    from ``expected`` (``mismatched``). With ``expected`` ``None`` the
    outputs are returned in place of the check, for the default tactic.
    """
    timings, outputs = {}, {}
    for name, inputs in routings.items():
        try:
            milliseconds, replayed = _time(
                operator, inputs, tactic, calls=args.calls, rounds=args.rounds
            )
            eager = operator(*inputs, tactic=tactic)
        except (RuntimeError, ValueError):
            return None, "failed"
        if expected is not None and not (
            torch.equal(replayed, expected[name])
            and torch.equal(eager, expected[name])
        ):
            return None, "mismatched"
        timings[name], outputs[name] = milliseconds, eager.clone()
    return timings, outputs if expected is None else None


def measure(provider, module, tokens: int, args, generator) -> dict:
    """The table entry of one token count.

    Every candidate runs under each of ``ROUTINGS``; the entry holds the
    admitted tactic whose slowest routing, relative to the default's time
    under that routing, is fastest, so no routing slows down. It records
    the medians of the tactic and of the default per routing, and how many
    candidates ran, failed and mismatched the default's outputs.
    """
    from uniserve.model.inputs import TextSize
    from uniserve.runtime.tensor_buffers import TensorBuffers

    device = module.up_gate.weight.device
    routings = {
        name: _routes(tokens, pool(args), args, device, generator)
        for name, pool in ROUTINGS.items()
    }
    size = TextSize(tokens, 1)
    requirements = provider.workspace_buffers(module=module, size=size)
    counts = {"candidates": 0, "failed": 0, "mismatched": 0}
    with TensorBuffers.allocate(requirements, device=device) as buffers:
        operator = provider.prepare(
            module=module, size=size, workspace=buffers.view(requirements)
        )
        try:
            dimensions = operator.tactic_space()
            if not dimensions:
                raise ValueError(f"{provider.name} calls take no tactic")
            best = [options[0] for options in dimensions]
            default, expected = _evaluate(operator, routings, best, None, args)
            if default is None:
                raise RuntimeError("the default tactic does not run")
            best_ms, best_ratio = default, 1.0
            for index, options in enumerate(dimensions):
                for option in options[1:]:
                    candidate = [*best[:index], option, *best[index + 1 :]]
                    counts["candidates"] += 1
                    timings, reason = _evaluate(
                        operator, routings, candidate, expected, args
                    )
                    if timings is None:
                        counts[reason] += 1
                        continue
                    ratio = max(
                        timings[name] / default[name] for name in timings
                    )
                    if ratio < best_ratio:
                        best, best_ms, best_ratio = candidate, timings, ratio
        finally:
            operator.close()
    return {
        "tokens": tokens,
        "tactic": best,
        "median_ms": best_ms,
        "default_median_ms": default,
        **counts,
    }


def main(argv: Sequence[str] | None = None) -> None:
    from uniserve.runtime.backends import moe

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--provider", required=True)
    parser.add_argument("--format", required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--intermediate", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--activation", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output", type=Path, default=TACTIC_TABLE)
    args = parser.parse_args(argv)

    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(20260927)
    with torch.inference_mode():
        module = _experts(args, device, generator)
        provider = moe.resolve(args.provider, module=module, device=device)
        key = dict(
            zip(KEY_FIELDS, table_key(provider.name, module), strict=True)
        )
        measured = []
        for tokens in sorted(args.tokens):
            entry = measure(provider, module, tokens, args, generator)
            print(json.dumps(entry), flush=True)
            measured.append(entry)

    stamp = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    table: dict[str, Any] = (
        json.loads(args.output.read_text())
        if args.output.exists()
        else {
            "description": (
                "Measured grouped-expert tactic per device model, "
                "FlashInfer version, provider, weight format and expert "
                "shape at each routed token count: among tactics whose "
                "outputs equal the default's bit for bit, the one whose "
                "slowest routing distribution, relative to the default "
                "under it, is fastest. A call uses the entry of the largest "
                "count not above its own. Written by "
                "uniserve.runtime.backends.moe.tactics"
            ),
            "entries": [],
        }
    )
    table["entries"] = [
        entry
        for entry in table["entries"]
        if tuple(entry[field] for field in KEY_FIELDS) != tuple(key.values())
    ] + [
        {**key, **entry, "measured_utc": stamp, "rounds": args.rounds}
        for entry in measured
    ]
    args.output.write_text(json.dumps(table, indent=1) + "\n")


if __name__ == "__main__":
    main()
