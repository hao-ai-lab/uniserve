"""Nsight export correlation and scope filtering through the public reader."""

import sqlite3
from contextlib import closing

import pytest

from uniserve.sanitizer.nsys import analyze

pytestmark = pytest.mark.unit


def test_scope_filter_keeps_batch_context_and_counts_each_api_once(tmp_path):
    filename = tmp_path / "trace.sqlite"
    with closing(sqlite3.connect(filename)) as db:
        # A small export fragment with two threads sharing a correlation ID.
        # Nsight emits a separate backtrace activity for the first API call.
        db.executescript("""
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            INSERT INTO StringIds VALUES
                (1, 'cudaEventSynchronize_v3020'),
                (2, 'cudaEventSynchronize'),
                (3, 'cuEventSynchronize');

            CREATE TABLE ENUM_NSYS_EVENT_CLASS (id INTEGER, name TEXT);
            INSERT INTO ENUM_NSYS_EVENT_CLASS VALUES
                (0, 'TRACE_PROCESS_EVENT_CUDA_RUNTIME'),
                (1, 'TRACE_PROCESS_EVENT_CUDA_DRIVER'),
                (67, 'TRACE_PROCESS_EVENT_CUDABACKTRACE');

            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER, end INTEGER, eventClass INTEGER,
                globalTid INTEGER, correlationId INTEGER, nameId INTEGER,
                returnValue INTEGER, callchainId INTEGER
            );
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
                (125, 130, 0, 16777217, 17, 1, 0, NULL),
                (124, 131, 67, 16777217, 17, 2, 0, 5),
                (125, 130, 1, 16777218, 17, 3, 0, NULL);

            CREATE TABLE NVTX_EVENTS (
                start INTEGER, end INTEGER, eventType INTEGER,
                rangeId INTEGER, category INTEGER, color INTEGER, text TEXT,
                globalTid INTEGER, endGlobalTid INTEGER, textId INTEGER,
                domainId INTEGER, uint64Value INTEGER
            );
            INSERT INTO NVTX_EVENTS
                (start, end, eventType, text, globalTid, uint64Value) VALUES
                (90, 210, 59, 'uniserve.worker.execute', 16777217, 7),
                (100, 200, 59, 'model.forward', 16777217, NULL),
                (100, 200, 59, 'model.forward', 16777218, NULL);
        """)

    report = analyze(filename, scope_prefix="model.")
    waits = [hint for hint in report.hints if hint.rule == "host-sync"]
    assert len(waits) == 2
    observed = {
        hint.evidence[0].global_tid: hint.evidence[0].scope.batch_id
        for hint in waits
    }
    assert observed == {16777217: 7, 16777218: None}
    assert report.coverage["scoped_calls"] == 2
    assert any("No GPU copy activities" in note for note in report.notes)

    missing = analyze(filename, scope_prefix="uncaptured.")
    assert not missing.hints
    assert any("No closed NVTX ranges match" in note for note in missing.notes)
