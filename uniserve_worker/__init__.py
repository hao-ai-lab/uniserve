"""Request execution, serving configuration and worker resource ownership.

One worker process serves one rank of a Rust engine WorkerGroup.
``bootstrap`` turns the rank's launch descriptor into a loaded model and a
sized ``Worker``; ``service`` receives the engine's requests over the native
``_uniserve_ipc`` extension; ``execution`` and ``model_executor`` run calls on
the model; ``storage`` owns request slots, caches and tensor products; and
``transport`` moves published products between ranks.
"""
