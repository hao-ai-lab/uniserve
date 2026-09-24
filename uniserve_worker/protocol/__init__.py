"""Typed records and validation for the scheduler-worker wire protocol.

The records mirror the Rust `uniserve_worker_ipc` types that the PyO3
transport (`crates/worker-ipc-py`) converts to and from Python. Submodules
are imported directly; this package re-exports nothing.

- `identity`: request, call, and buffer identities.
- `tensor`: bounded logical tensor descriptions.
- `transfer`: physical locations of tensors published between workers.
- `call`: the per-request call descriptors in a batch.
- `batch`: the submitted `Batch`, its lifecycle commands, and its
  allocation records.
- `construction`: assembly of batches the Rust decoder already validated.
- `output`: per-batch completion records.
- `messages`: request decoding and response envelopes.
- `worker_info`: startup information the worker reports to the scheduler.
- `validation`: primitive decoders shared by the ``from_mapping`` parsers.
"""
