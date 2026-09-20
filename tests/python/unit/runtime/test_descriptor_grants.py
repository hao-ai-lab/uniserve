"""A publication's descriptor reaches a consumer process that asks for it.

A file descriptor names an open file of the process that opened it, so the
bytes of one carry no meaning anywhere else. A device that exports descriptors
rather than fabric handles therefore has to hand the descriptor itself to its
readers, which is what these cover: a registered publication is granted, an
unregistered or withdrawn one is refused, and an endpoint that serves nothing
is refused rather than hung on.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import uuid

import pytest

from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.transfer.descriptor_grants import DescriptorGrants, fetch

pytestmark = [pytest.mark.unit]


def _identity(descriptor: int) -> tuple[int, int]:
    """Name the file a descriptor is open on, independently of its number."""
    status = os.fstat(descriptor)
    return status.st_dev, status.st_ino


def _fetch_identity(endpoint: str, publication: str, channel) -> None:
    try:
        descriptor = fetch(endpoint, publication)
    except Exception as error:  # reported to the parent as a value
        channel.send(("error", type(error).__name__))
        return
    try:
        channel.send(("granted", _identity(descriptor)))
    finally:
        os.close(descriptor)


def _in_child(endpoint: str, publication: str) -> tuple[str, object]:
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_fetch_identity, args=(endpoint, publication, child)
    )
    process.start()
    try:
        assert parent.poll(60), "the consumer process never answered"
        return parent.recv()
    finally:
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join(10)


def test_a_registered_descriptor_opens_the_same_file_in_a_consumer() -> None:
    endpoint = f"uniserve-publications-{uuid.uuid4().hex}"
    grants = DescriptorGrants(endpoint)
    publication = uuid.uuid4().hex
    with tempfile.TemporaryFile() as source:
        grants.register(publication, source.fileno())
        try:
            kind, value = _in_child(endpoint, publication)
        finally:
            grants.close()
        assert kind == "granted"
        # The number the consumer received is its own; the file is the same.
        assert value == _identity(source.fileno())


def test_a_withdrawn_publication_is_refused() -> None:
    endpoint = f"uniserve-publications-{uuid.uuid4().hex}"
    grants = DescriptorGrants(endpoint)
    publication = uuid.uuid4().hex
    with tempfile.TemporaryFile() as source:
        grants.register(publication, source.fileno())
        grants.release(publication)
        try:
            kind, value = _in_child(endpoint, publication)
        finally:
            grants.close()
    assert kind == "error" and value == WorkerError.__name__


def test_an_endpoint_that_serves_nothing_is_refused() -> None:
    kind, value = _in_child(
        f"uniserve-publications-{uuid.uuid4().hex}", uuid.uuid4().hex
    )
    assert kind == "error" and value == WorkerError.__name__
