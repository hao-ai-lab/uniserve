"""Analyze existing profiler captures without starting or changing a worker."""

import argparse
import json
import sqlite3
from dataclasses import asdict

from . import calls, nsys
from .rules import RULES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    trace = commands.add_parser("trace", help="Inspect Nsight Systems SQLite")
    trace.add_argument("capture")
    trace.add_argument("--scope", default="uniserve.", help="NVTX name prefix")
    trace.add_argument("--rule", action="append", choices=RULES)
    trace.add_argument("--json", action="store_true")

    call_graph = commands.add_parser(
        "calls", help="Inspect a cProfile call graph"
    )
    call_graph.add_argument("capture")
    call_graph.add_argument(
        "--native",
        action="append",
        required=True,
        help="Native function name substring (repeat to select more functions)",
    )
    call_graph.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        if args.command == "trace":
            report = nsys.analyze(args.capture, scope_prefix=args.scope)
            if args.rule:
                report.hints = [h for h in report.hints if h.rule in args.rule]
            if args.json:
                print(json.dumps(asdict(report), indent=2))
                return

            print(
                "Coverage: "
                + ", ".join(
                    f"{name}={count}" for name, count in report.coverage.items()
                )
            )
            for hint in report.hints:
                print(f"\n[{hint.rule}] {hint.observation}")
                print(f"  {hint.suggestion}")
                for event in hint.evidence:
                    duration_us = (event.end_ns - event.start_ns) / 1_000
                    print(
                        f"  {event.start_ns}..{event.end_ns} ns "
                        f"({duration_us:.3f} us) tid={event.global_tid} "
                        f"batch={event.scope.batch_id} "
                        f"correlation={event.correlation_id} {event.name}"
                    )
                    print(f"    scope: {event.scope.name}")
                    for copy in event.copies:
                        print(
                            f"    GPU {copy.start_ns}..{copy.end_ns} ns: "
                            f"{copy.direction} {copy.bytes} B "
                            f"device={copy.device} stream={copy.stream} "
                            f"{copy.source_memory}->{copy.destination_memory}"
                        )
                    if not event.stack:
                        print("    stack: not captured")
                    for frame in event.stack:
                        print(f"    {frame}")
            notes = report.notes
        else:
            graph = calls.analyze(args.capture, native_names=tuple(args.native))
            if args.json:
                print(json.dumps(asdict(graph), indent=2))
                return

            print(f"Recorded native functions: {len(graph.native_functions)}")
            for edge in graph.hints:
                print(
                    f"\n[{edge.direction}] {edge.calls} calls, "
                    f"{edge.total_seconds:.6f} s inclusive"
                    f"\n  {edge.caller}\n  -> {edge.callee}"
                )
            notes = list(graph.notes)

        print()
        for note in notes:
            print(note)
    except (OSError, sqlite3.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
