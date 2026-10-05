"""Final FFI ownership drains GPU work while Python can make progress."""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event

import pytest
import torch
import tvm_ffi
from bindings import (
    CUDAStream,
    Executor,
    HostBuffers,
    KVImporter,
    TransferPool,
    VmmPool,
    WorkerRequest,
    load_library,
)
from uniserve_kernels.peer_storage import copy_host_device, empty

from tests.python.fixtures.cuda_stream import blocked_stream


def _retire(library, resource, owner, kv_request):
    load_library(library)
    value = KVImporter(1, 1, 0) if resource == "kv_import" else None
    output = torch.zeros(32, dtype=torch.int64, device="cuda:0")
    host = torch.full((32,), 17, pin_memory=True)
    backing = (
        empty((1 << 20,), dtype=torch.uint8, device=torch.device("cuda:0"))
        if resource == "vmm_pool"
        else None
    )

    # Load the numerical kernels before holding the device behind a gate.
    output.fill_(0)
    entered = Event()

    def drop(retained):
        entered.set()
        retained.clear()

    with ThreadPoolExecutor(max_workers=1) as threads:
        with blocked_stream("cuda:0") as stream:
            with torch.cuda.stream(stream), tvm_ffi.use_torch_stream(stream):
                if resource == "buffers":
                    value = HostBuffers([host], 0)
                    slot, source = value.acquire()
                    output.copy_(torch.from_dlpack(source), non_blocking=True)
                    value.record_copy(slot)
                    del source
                elif resource == "executor":
                    value = Executor(
                        tvm_ffi.convert_func(
                            lambda tensor: tensor.fill_(17),
                            tensor_cls=torch.Tensor,
                        ),
                        1,
                    )
                    value.submit(1, output)
                elif resource == "vmm_pool":
                    value = VmmPool(backing)
                    value.reserve(64)
                    output.fill_(17)
                elif resource == "transfer":
                    value = TransferPool(
                        output.numel() * output.element_size(), 1, 1
                    )

                    def copy(handle):
                        copy_host_device(
                            output,
                            host,
                            torch.cuda.ExternalStream(handle, device=0),
                        )

                    value.submit(
                        tvm_ffi.convert_func(copy),
                        output,
                        output.numel() * output.element_size(),
                    )
                elif resource == "kv_import":
                    copied = Event()

                    def copy(_workspace, handle):
                        numerical = torch.cuda.ExternalStream(handle, device=0)
                        numerical.wait_stream(stream)
                        with torch.cuda.stream(numerical):
                            output.fill_(17)
                        copied.set()

                    value.reserve(
                        WorkerRequest(kv_request),
                        0,
                        1,
                        tvm_ffi.convert_func(lambda _workspace, _stream: None),
                        tvm_ffi.convert_func(copy),
                    )
                    assert copied.wait(5)
                else:
                    value = CUDAStream(0, stream.cuda_stream, 2)
                    output.fill_(17)

            if owner == "array":
                retained = [tvm_ffi.Array([value])]
            elif owner == "any":
                retained = [tvm_ffi.CAny(value)]
            else:
                retained = [value]
            del value

            pending = threads.submit(drop, retained)
            assert entered.wait(5)
            with pytest.raises(TimeoutError):
                pending.result(timeout=0.1)

        # Exiting blocked_stream needs the GIL to release the device gate.
        pending.result(timeout=5)

    assert output.cpu().tolist() == [17] * 32


@pytest.mark.parametrize(
    "resource",
    ("buffers", "executor", "stream", "vmm_pool", "transfer", "kv_import"),
)
@pytest.mark.parametrize("owner", ("object", "array", "any"))
def test_final_owner_drains_without_blocking_python(
    request, resource, owner, kv_request
):
    # A GIL deadlock prevents in-process timeouts from running. The parent
    # process must remain independent so a regression cannot hang pytest.
    process = multiprocessing.get_context("spawn").Process(
        target=_retire,
        args=(
            request.config.getoption("--ffi-library"),
            resource,
            owner,
            kv_request.encode(),
        ),
    )
    process.start()
    try:
        process.join(timeout=30)
        assert not process.is_alive(), "FFI resource disposal blocked Python"
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        process.close()
