"""A process killed while building an extension never blocks a later load."""

import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# A CPython extension with no torch headers compiles in about a second.
_SOURCE = textwrap.dedent("""
    #include <Python.h>

    #define CONCAT_IMPL(a, b) a##b
    #define CONCAT(a, b) CONCAT_IMPL(a, b)

    static PyObject* answer(PyObject*, PyObject*) {
      return PyLong_FromLong(42);
    }

    static PyMethodDef methods[] = {
        {"answer", answer, METH_NOARGS, nullptr},
        {nullptr, nullptr, 0, nullptr},
    };

    static PyModuleDef definition = {
        PyModuleDef_HEAD_INIT, "probe", nullptr, -1, methods,
    };

    PyMODINIT_FUNC CONCAT(PyInit_, TORCH_EXTENSION_NAME)() {
      return PyModule_Create(&definition);
    }
""")

_LOAD = textwrap.dedent("""
    import sys
    from pathlib import Path
    from uniserve_kernels import jit

    module = jit.load("probe", [Path(sys.argv[1]) / "probe.cpp"])
    print(module.answer())
""")


def _stalling_compiler(path: Path, started: Path) -> None:
    """Write a compiler that answers toolchain probes and never compiles.

    Torch probes the compiler's version before building; those calls reach
    the real compiler. The compile itself writes its process id to
    ``started`` and sleeps, so the loading process is inside its build when
    the test kills it.
    """
    compiler = shutil.which("c++")
    assert compiler is not None, "the build needs a host C++ compiler"
    path.write_text(
        textwrap.dedent(f"""\
            #!/bin/sh
            for argument in "$@"; do
              if [ "$argument" = "-c" ]; then
                echo $$ > '{started}.partial'
                mv '{started}.partial' '{started}'
                exec sleep 600
              fi
            done
            exec '{compiler}' "$@"
        """)
    )
    path.chmod(0o755)


def test_load_completes_after_a_builder_is_killed_mid_build(tmp_path):
    sources = tmp_path / "csrc"
    sources.mkdir()
    (sources / "probe.cpp").write_text(_SOURCE)
    started = tmp_path / "compile-started"
    compiler = tmp_path / "stalling-c++"
    _stalling_compiler(compiler, started)
    environment = {
        **os.environ,
        "TORCH_EXTENSIONS_DIR": str(tmp_path / "extensions"),
    }
    command = [sys.executable, "-c", _LOAD, str(sources)]

    # The first loader leads its own process group, which its build tool
    # joins; the build tool runs the stalled compile in a group of its own.
    log = tmp_path / "builder.log"
    with log.open("wb") as output:
        builder = subprocess.Popen(
            command,
            env={**environment, "CXX": str(compiler)},
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 120
        while not started.exists():
            if builder.poll() is not None:
                pytest.fail(
                    f"loader exited before compiling:\n{log.read_text()}"
                )
            if time.monotonic() > deadline:
                pytest.fail("loader never reached its compile step")
            time.sleep(0.1)

        # SIGKILL runs no cleanup in the loader; its compiler processes
        # outlive it, as they do when a worker is torn down mid-build.
        builder.kill()
        builder.wait()

        try:
            loaded = subprocess.run(
                command,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            pytest.fail("load stayed blocked after its builder was killed")
    finally:
        builder.kill()
        builder.wait()
        leftovers = [(os.killpg, builder.pid)]
        if started.exists():
            leftovers.append((os.kill, int(started.read_text())))
        for stop, target in leftovers:
            try:
                stop(target, signal.SIGKILL)
            except ProcessLookupError:
                pass

    assert loaded.returncode == 0, loaded.stdout + loaded.stderr
    assert loaded.stdout.strip() == "42"


def test_loaded_extensions_follow_headers_and_build_flags(
    tmp_path, monkeypatch
):
    from uniserve_kernels import jit

    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "extensions"))
    sources = tmp_path / "sources"
    left, right = sources / "left" / "value.h", sources / "right" / "value.h"
    left.parent.mkdir(parents=True)
    right.parent.mkdir(parents=True)
    left.write_text("#define LEFT 20\n")
    right.write_text("#define RIGHT 21\n")
    source = sources / "probe.cpp"
    source.write_text(
        '#include "left/value.h"\n#include "right/value.h"\n'
        + _SOURCE.replace(
            "PyLong_FromLong(42)", "PyLong_FromLong(LEFT + RIGHT + OFFSET)"
        )
    )

    def load(offset):
        return jit.load(
            "header_probe",
            [source],
            headers=[left, right],
            source_root=sources,
            include_dirs=["."],
            cxx_flags=["-O2", f"-DOFFSET={offset}"],
        )

    first = load(1)
    assert first.answer() == 42
    left.write_text("#define LEFT 30\n")
    assert load(1).answer() == 52
    assert load(2).answer() == 53
    assert first.answer() == 42
