"""Standard profiler call edges remain evidence, not inferred batch counts."""

import cProfile

import pytest

from uniserve.sanitizer.calls import analyze

pytestmark = pytest.mark.unit


def test_native_callbacks_and_python_calls_have_separate_directions(tmp_path):
    def key(value):
        return -value

    def workload():
        for _ in range(2):
            assert sorted([1, 2, 3], key=key) == [3, 2, 1]

    filename = str(tmp_path / "calls.pstats")
    profiler = cProfile.Profile()
    profiler.runcall(workload)
    profiler.dump_stats(filename)
    report = analyze(filename, native_names=("sorted",))

    edges = {hint.direction: hint for hint in report.hints}
    assert edges["python-to-native"].calls == 2
    assert edges["native-to-python"].calls == 6
    assert edges["native-to-python"].callee.endswith("(key)")
    assert any("no timeline" in note for note in report.notes)

    missing = analyze(filename, native_names=("not_captured",))
    assert not missing.hints
    assert any("coverage is unknown" in note for note in missing.notes)
