"""Capture GIL ownership while draining copies submitted by CUDA Graphs."""

import argparse
import threading
import time

import torch
import tvm_ffi
from bindings import HostBuffers, load_library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffi-library", required=True)
    args = parser.parse_args()
    load_library(args.ffi_library)

    stream = torch.cuda.Stream(device=0)
    torch.cuda._sleep(1)
    torch.cuda.synchronize()

    for owner in ("close", "object", "array", "any"):
        buffers = HostBuffers([torch.full((32,), 17, pin_memory=True)], 0)
        slot, tensor = buffers.acquire()
        host = torch.from_dlpack(tensor)
        output = torch.empty_like(host, device="cuda:0")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            # Keep retirement pending long enough to observe GIL contention.
            torch.cuda._sleep(100_000_000)
            output.copy_(host, non_blocking=True)
        del tensor, host

        graph.replay()
        torch.cuda.synchronize()
        ready = threading.Event()

        def contender():
            ready.set()
            time.sleep(0.01)

        thread = threading.Thread(target=contender)
        thread.start()
        ready.wait()

        with torch.cuda.stream(stream), tvm_ffi.use_torch_stream(stream):
            graph.replay()
            buffers.record_copy(slot)

        if owner == "array":
            retained = tvm_ffi.Array([buffers])
            del buffers
        elif owner == "any":
            retained = tvm_ffi.CAny(buffers)
            del buffers

        with torch.cuda.nvtx.range("uniserve.gil.retirement." + owner):
            if owner == "close":
                buffers.close()
                del buffers
            elif owner in ("array", "any"):
                del retained
            else:
                del buffers

        thread.join()
        assert output.cpu().tolist() == [17] * 32
        del graph


if __name__ == "__main__":
    main()
