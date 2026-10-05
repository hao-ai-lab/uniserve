"""A publication's descriptor reaches a consumer process that asks for it.

A file descriptor names an open file of the process that opened it, so the
bytes of one carry no meaning anywhere else. A device that exports descriptors
rather than fabric handles therefore has to hand the descriptor itself to its
readers, which is what these cover: a registered publication is granted, an
unregistered or withdrawn one is refused, an endpoint that serves nothing is
refused rather than hung on, and a rank's readers are served when they all ask
at once, which is what a batch of products actually produces.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from uniserve_worker.errors import WorkerError
from uniserve_worker.transport.descriptor_grants import DescriptorGrants, fetch

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


def test_registration_and_receiver_own_their_descriptors() -> None:
    endpoint = f"uniserve-publications-{uuid.uuid4().hex}"
    grants = DescriptorGrants(endpoint)
    publication = uuid.uuid4().hex
    try:
        with tempfile.TemporaryFile() as source:
            source.write(b"allocation")
            source.flush()
            grants.register(publication, source.fileno())

        # Registration retains the open allocation after its caller closes.
        received = fetch(endpoint, publication)
        try:
            grants.release(publication)
            assert os.pread(received, 10, 0) == b"allocation"
            assert not os.get_inheritable(received)
            with pytest.raises(WorkerError):
                fetch(endpoint, publication)
        finally:
            os.close(received)
    finally:
        grants.close()


@pytest.mark.timeout(10)
@pytest.mark.parametrize("release", ("close", "drop"))
def test_retirement_releases_an_idle_endpoint(release) -> None:
    endpoint = f"uniserve-publications-{uuid.uuid4().hex}"
    owners = [DescriptorGrants(endpoint)]

    def retire():
        if release == "close":
            owners[0].close()
        else:
            owners.clear()

    with ThreadPoolExecutor(max_workers=1) as thread:
        thread.submit(retire).result(timeout=5)

    replacement = DescriptorGrants(endpoint)
    try:
        with tempfile.TemporaryFile() as source:
            publication = uuid.uuid4().hex
            replacement.register(publication, source.fileno())
            received = fetch(endpoint, publication)
            try:
                assert _identity(received) == _identity(source.fileno())
            finally:
                os.close(received)

            if release == "close":
                with pytest.raises(OSError, match="closed"):
                    owners[0].register(publication, source.fileno())
    finally:
        replacement.close()


def _fetch_many(endpoint: str, publications: list[str], channel) -> None:
    """Ask for one publication and report what file it names."""
    results = []
    for publication in publications:
        try:
            descriptor = fetch(endpoint, publication)
        except Exception as error:
            results.append(("error", type(error).__name__))
            continue
        try:
            results.append(("granted", _identity(descriptor)))
        finally:
            os.close(descriptor)
    channel.send(results)


def test_concurrent_consumers_are_all_served() -> None:
    endpoint = f"uniserve-publications-{uuid.uuid4().hex}"
    grants = DescriptorGrants(endpoint)
    context = mp.get_context("spawn")
    readers = 8
    sources = [tempfile.TemporaryFile() for _ in range(4)]
    publications = [uuid.uuid4().hex for _ in sources]
    expected = []
    try:
        for publication, source in zip(publications, sources, strict=True):
            grants.register(publication, source.fileno())
            expected.append(_identity(source.fileno()))

        # Every reader asks for every publication at the same time, which is
        # what a rank reading a batch of products from several producers does.
        pipes = [context.Pipe() for _ in range(readers)]
        processes = [
            context.Process(
                target=_fetch_many, args=(endpoint, publications, child)
            )
            for _parent, child in pipes
        ]
        for process in processes:
            process.start()
        try:
            for parent, _child in pipes:
                assert parent.poll(120), "a consumer process never answered"
                assert parent.recv() == [
                    ("granted", identity) for identity in expected
                ]
        finally:
            for process in processes:
                process.join(30)
                if process.is_alive():
                    process.terminate()
                    process.join(10)
    finally:
        grants.close()
        for source in sources:
            source.close()
