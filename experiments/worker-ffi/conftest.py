"""Build artifacts are supplied explicitly by the experiment invocation."""

import pytest
from bindings import load_library


def pytest_addoption(parser):
    parser.addoption("--ffi-library", required=True)
    parser.addoption("--ffi-request", required=True)
    parser.addoption("--ffi-kv-request", required=True)


@pytest.fixture(scope="session", autouse=True)
def library(request):
    load_library(request.config.getoption("--ffi-library"))


@pytest.fixture(scope="session")
def wire_request(request):
    with open(request.config.getoption("--ffi-request"), "rb") as source:
        return source.read()


@pytest.fixture(scope="session")
def kv_request(request):
    from bindings import WorkerRequest

    with open(request.config.getoption("--ffi-kv-request"), "rb") as source:
        return WorkerRequest(source.read())
