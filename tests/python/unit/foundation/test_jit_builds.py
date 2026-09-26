"""Native JIT extension builds keyed by their content.

``uniserve_kernels.jit.load`` must give every caller the build of its own
source files: two checkouts whose sources differ under one extension name
each load their own module in the same cache, and identical sources in
different directories share one build instead of recompiling.
"""

from __future__ import annotations

import pytest

from uniserve_kernels import jit

pytestmark = [pytest.mark.unit, pytest.mark.slow]

_SOURCE = """
#include <torch/extension.h>

#include "value.h"

int64_t value() { return VALUE; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("value", &value);
}
"""


def _checkout(root, value):
    """Write one checkout's source and header; the header sets the value."""
    root.mkdir()
    (root / "value.cpp").write_text(_SOURCE)
    (root / "value.h").write_text(f"#define VALUE {value}\n")
    return root / "value.cpp", root / "value.h"


def _load(source, header):
    return jit.load(
        "uniserve_jit_probe", [source], headers=[header], cxx_flags=["-O0"]
    )


def test_checkouts_with_different_files_load_their_own_builds(
    tmp_path, monkeypatch
):
    cache = tmp_path / "cache"
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(cache))
    first = _load(*_checkout(tmp_path / "first", 1))
    second = _load(*_checkout(tmp_path / "second", 2))
    third = _load(*_checkout(tmp_path / "third", 1))

    assert (first.value(), second.value(), third.value()) == (1, 2, 1)
    # The third checkout's files equal the first's, so it adds no build.
    assert len([entry for entry in cache.iterdir() if entry.is_dir()]) == 2
