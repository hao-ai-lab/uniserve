"""Final FFI ownership drains GPU work while Python can make progress."""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event

import pytest
import torch
import tvm_ffi
from bindings import CUDAStream, Executor, HostBuffers, load_library

from tests.python.fixtures.cuda_stream import blocked_stream


def _retire(library, resource, owner):
    load_library(library)
    output = torch.zeros(32, dtype=torch.int64, device="cuda:0")
    host = torch.full((32,), 17, pin_memory=True)

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


@pytest.mark.parametrize("resource", ("buffers", "executor", "stream"))
@pytest.mark.parametrize("owner", ("object", "array", "any"))
def test_final_owner_drains_without_blocking_python(request, resource, owner):
    # A GIL deadlock prevents in-process timeouts from running. The parent
    # process must remain independent so a regression cannot hang pytest.
    process = multiprocessing.get_context("spawn").Process(
        target=_retire,
        args=(request.config.getoption("--ffi-library"), resource, owner),
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
