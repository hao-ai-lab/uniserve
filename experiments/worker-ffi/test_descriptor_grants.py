"""FFI containers retain a live native descriptor service."""

import os
import tempfile
import uuid

import pytest
import tvm_ffi
from bindings import DescriptorGrants, fetch_descriptor


def test_container_retains_grants_until_final_release():
    endpoint = f"uniserve-ffi-{uuid.uuid4().hex}"
    export = uuid.uuid4().hex
    retained = tvm_ffi.Array([DescriptorGrants(endpoint)])
    try:
        with tempfile.TemporaryFile() as source:
            source.write(b"allocation")
            source.flush()
            retained[0].register(export, source.fileno())

        received = fetch_descriptor(endpoint, export)
        try:
            retained[0].release(export)
            with pytest.raises(RuntimeError, match="no live descriptor"):
                fetch_descriptor(endpoint, export)
            assert os.pread(received, 10, 0) == b"allocation"
            assert not os.get_inheritable(received)
        finally:
            os.close(received)
    finally:
        # Release the container's final reference, which joins the native
        # service. The endpoint can be bound again immediately afterwards.
        del retained

    replacement = DescriptorGrants(endpoint)
    replacement.close()
