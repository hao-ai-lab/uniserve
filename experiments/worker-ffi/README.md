# Worker TVM-FFI experiment

This standalone library evaluates TVM-FFI as the worker's sole Rust/Python interface. It has no PyO3 dependency and does not load the production worker extension. It uses the existing Rust worker codec, the repository's installed PyTorch and TVM-FFI packages, and ordinary PyTorch numerical callbacks.

`Executor` admits a bounded number of asynchronous numerical batches. `Batch` owns its DLPack input and result references, cancellation state, and native CUDA completion event. A cancelled batch cannot expose a result or return its execution capacity before physical completion. Python callback failures preserve their exception type and retain the input through device completion. Closing drains outstanding work before releasing the callable. Dropping the last batch owner early also drains its device work; unknown physical completion retains the tensor references instead of returning their memory for reuse.

Public objects and methods are registered in `src/lib.rs`; Python classes contain no lifecycle implementation. Registration uses TVM-FFI's stable C API because the Rust derive resolves existing types and the Rust sys crate does not currently declare the registration entry points. `Any` and reflection remain in this module. `src/execution.rs` uses native state, owning tensor references, and a numerical `Function`; it never reads Python attributes, events, or futures. `WorkerRequest` retains the existing native IPC enum and serializes through the shared codec without constructing Python request records.

The numerical callback runs on the submitting thread. It must run one homogeneous batch on CPU or on the selected CUDA stream and device. For CUDA, enter both `torch.cuda.stream(stream)` and `tvm_ffi.use_torch_stream(stream)` around submission and result consumption. The callback receives ordinary `torch.Tensor` arguments through `convert_func(..., tensor_cls=torch.Tensor)`. A result borrowed on another CUDA stream first waits on its producer event. Tensor references retain allocations; event completion determines when execution capacity returns.

Internal execution ownership uses ordinary Rust `Arc` and `Mutex`; it does not depend on the reflected object hierarchy. The upstream Rust tensor and function handles are not generally `Send`/`Sync`. This experiment keeps their Rust thread restrictions and exercises callbacks on Python caller threads, with synchronized native state. It does not establish that arbitrary FFI tensors or callbacks can be moved into Rust task executors. A production numerical lane should create and use its callable on its owning thread; completion notifications can carry native request identifiers without transporting interpreter-owned callbacks.

The Rust dependency is pinned to upstream commit `df463f9bd269867fc7f4d578d62d80cc2a11d004`. The published `0.1.0-alpha.0` export macro lacks the unmangled symbol required for a loadable module; the pinned upstream macro exports it and contains panic-to-error handling. The Python environment remains at the repository-locked `apache-tvm-ffi==0.1.12` and `torch==2.13.0+cu130`.

From the repository root:

```bash
export PATH="$PWD/.venv/bin:$PATH"
cargo build --locked --manifest-path experiments/worker-ffi/Cargo.toml --target-dir artifacts/rust-worker/tvm-ffi/target
cargo run --locked --manifest-path experiments/worker-ffi/Cargo.toml --target-dir artifacts/rust-worker/tvm-ffi/target --example wire_fixture > artifacts/rust-worker/tvm-ffi/request.bin
.venv/bin/python -m pytest -q experiments/worker-ffi --ffi-library "$PWD/artifacts/rust-worker/tvm-ffi/target/debug/libuniserve_worker_ffi.so" --ffi-request "$PWD/artifacts/rust-worker/tvm-ffi/request.bin"
```

The suite requires CUDA. It exercises CPU and GPU module execution, strided shared tensor storage, callback failure after device submission, cross-stream consumption, cancellation, occupied capacity, concurrent Python callers, shutdown, and a native request-retirement IPC message. Warmup precedes checks that require outstanding GPU work, so first-use allocation or kernel loading does not turn them into synchronous calls. Generated libraries, messages, and logs belong under `artifacts/`.

On the locked Python environment and a GB200, all eight behavioral cases passed. Rust formatting, Clippy with warnings denied, and Python Ruff checks passed. The dynamic library's dependencies include `libtvm_ffi.so` and system libraries, with no Python linkage. These results establish the exercised language and lifetime mechanisms, not full worker integration or performance.

This is a feasibility experiment, not a serving implementation or throughput result. It does not yet replace production request admission, KV/latent pools, transfers, the IPC service loop, graphs, distributed collectives, or model loading. Adopting TVM-FFI for the worker requires moving those owners to native state and replacing their consumers together; forwarding their current Python attribute and Future operations through callbacks would preserve the interpreter dependency this interface is intended to remove. End-to-end model, graph, cancellation, and AFD validation remains necessary for that replacement.

References: [object model](https://tvm.apache.org/ffi/concepts/object_and_class.html), [Python callbacks](https://tvm.apache.org/ffi/reference/python/generated/tvm_ffi.convert_func.html), [Rust support](https://github.com/apache/tvm-ffi/tree/df463f9bd269867fc7f4d578d62d80cc2a11d004/rust).
