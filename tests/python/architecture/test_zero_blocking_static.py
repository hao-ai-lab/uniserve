"""Static zero-blocking gate for the worker request path.

`specs/decode-runtime.md` (Zero-blocking execution) requires that between the
first request submission and the terminal public commit, worker request-path
code uses query-only progress: no accelerator synchronize, live device scalar
extraction, pageable progress copy, synchronous transport, sleep poll, or
blocking future wait. This gate roots at the worker package and rejects every
forbidden-synchronization call site unless it is classified as belonging to a
non-steady-state phase (startup, administrative snapshot export/restore, public
delivery of an already-ready host product, a query-only readiness ticket, a
host-known plan tensor, the reference/simulation backend), or the single
controller progress yield tracked for elimination.

The allowlist is exhaustive: a new forbidden call in production code fails the
gate until it is either removed or classified, and a stale classification whose
call site no longer exists fails until it is deleted. The classification prefix
records why each site is outside the steady-state device-observation contract.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.architecture

PACKAGE = "uniserve_worker"
PACKAGE_ROOT = Path(__file__).resolve().parents[3] / PACKAGE

# Scalar-extraction / device→host / blocking primitives detected on any receiver.
_SCALAR_ATTRS = {"item", "tolist", "numpy"}

# Permitted classification prefixes. Every allowlisted site states one; the
# steady-state device-observation contract permits only these phases plus the
# single controller yield, which is tracked for elimination in specs/tasks.md.
_PERMITTED_PREFIXES = (
    "startup:",
    "admin:",
    "delivery:",
    "ticket:",
    "infra:",
    "host-plan:",
    "reference-backend:",
    "sim-stub:",
    "controller-yield:",
)

# (module, enclosing qualname, pattern) -> classification.
ALLOWLIST: dict[tuple[str, str, str], str] = {
    # Startup: weight loading and CUDA-graph warmup, before the steady interval.
    ("uniserve_worker/loader/weight_set.py", "module_weight_digest", ".numpy()"): (
        "startup: weight-set digest over host weights during load"
    ),
    ("uniserve_worker/loader/weight_set.py", "module_weight_digest", ".cpu()"): (
        "startup: weight-set digest over host weights during load"
    ),
    ("uniserve_worker/worker/model.py", "ModelWorker._execute_warmup", "time.sleep()"): (
        "startup: cuda-graph warmup readiness spin"
    ),
    ("uniserve_worker/worker/model.py", "ModelWorker._warmup_sequence", ".synchronize()"): (
        "startup: cuda-graph warmup barrier"
    ),
    # Admin: snapshot export/restore, outside the steady interval.
    ("uniserve_worker/runtime/kv_store.py", "KvStore._snapshot_pages.take", ".cpu()"): (
        "admin: KV snapshot page export"
    ),
    ("uniserve_worker/runtime/kv_store.py", "KvStore._snapshot_pages", ".cpu()"): (
        "admin: KV snapshot page export"
    ),
    ("uniserve_worker/runtime/latent_store.py", "LatentStore.snapshot_records", ".cpu()"): (
        "admin: latent snapshot export"
    ),
    ("uniserve_worker/runtime/product_store.py", "_snapshot_payload", ".cpu()"): (
        "admin: device-product snapshot export"
    ),
    ("uniserve_worker/runtime/snapshot_store.py", "SnapshotProvider._encode.tensor", ".cpu()"): (
        "admin: snapshot tensor export"
    ),
    ("uniserve_worker/runtime/transfer.py", "ShmTransport._complete_publications", ".numpy()"): (
        "admin: background publication drain thread (event-gated)"
    ),
    ("uniserve_worker/runtime/transfer.py", "ShmTransport.publish", '.to("cpu")'): (
        "admin: synchronous publish; the steady path is publish_async"
    ),
    ("uniserve_worker/runtime/transfer.py", "ShmTransport.publish", ".numpy()"): (
        "admin: synchronous publish; the steady path is publish_async"
    ),
    # Delivery: encoding/reading an already-ready host product for the client.
    (
        "uniserve_worker/runtime/image_utils.py",
        "uint8_image_to_png_base64_bytes",
        ".numpy()",
    ): "delivery: PNG-encode a query-ready CPU uint8 image",
    ("uniserve_worker/runtime/completion_store.py", "CompletionByteCapture.numpy", ".numpy()"): (
        "delivery: pinned-host completion view after its copy event is ready"
    ),
    ("uniserve_worker/runtime/completion_store.py", "CompletionLease.read_tokens", ".tolist()"): (
        "delivery: pinned-host tokens after the completion copy event is ready"
    ),
    ("uniserve_worker/execution/executor.py", "_CompletionImagePayload.finalize", ".result()"): (
        "delivery: image future, ready-gated with timeout=0"
    ),
    # Ticket: query-only transfer/completion readiness, never a blocking wait.
    ("uniserve_worker/execution/executor.py", "_PreparedTransferInput.tensors", ".result()"): (
        "ticket: prepared-transfer result read only after ready()"
    ),
    ("uniserve_worker/runtime/transfer.py", "_FutureTransferTicket.result", ".result()"): (
        "ticket: transfer ticket result read only after ready()"
    ),
    ("uniserve_worker/runtime/transfer.py", "_ShmReadTicket.result", ".result()"): (
        "ticket: shared-memory read ticket result read only after ready()"
    ),
    # Infra: non-CUDA source branch; the CUDA path is a non-blocking copy.
    ("uniserve_worker/runtime/completion_store.py", "CompletionLease.capture", '.to("cpu")'): (
        "infra: non-CUDA source branch; the CUDA path is a non_blocking pinned copy"
    ),
    ("uniserve_worker/runtime/completion_store.py", "CompletionLease.capture_bytes", '.to("cpu")'): (
        "infra: non-CUDA source branch; the CUDA path is a non_blocking pinned copy"
    ),
    ("uniserve_worker/execution/executor.py", "_capture_sample_span", ".tolist()"): (
        "infra: non-CUDA metadata branch; the CUDA path captures through the arena"
    ),
    ("uniserve_worker/execution/executor.py", "_sample_logprob_details", ".tolist()"): (
        "infra: non-CUDA logprob branch; the CUDA path captures through the arena"
    ),
    # Host-plan: attention plan tensors built on the host from known lengths.
    (
        "uniserve_worker/backends/attention/flashinfer_plan.py",
        "_fast_decode_plan_host_tensors",
        ".cpu()",
    ): "host-plan: defensive host indptr fallback; production callers pass a host indptr",
    (
        "uniserve_worker/backends/attention/flashinfer_plan.py",
        "_cpu_int32_tensor",
        '.to("cpu")',
    ): "host-plan: ensures a host int32 plan tensor; passthrough when already host",
    ("uniserve_worker/backends/attention/flashinfer_plan.py", "_indptr_last", ".item()"): (
        "host-plan: last element of a host-known cpu indptr"
    ),
    # Reference backend: the torch-SDPA fallback/stub attention path, not the
    # configured device route (flashinfer / fa4-cute).
    ("uniserve_worker/backends/attention/torch_sdpa.py", "_integer_values", ".tolist()"): (
        "reference-backend: torch-SDPA fallback/stub; configured routes use device attention"
    ),
    ("uniserve_worker/backends/attention/torch_sdpa.py", "_integer_values", '.to("cpu")'): (
        "reference-backend: torch-SDPA fallback/stub; configured routes use device attention"
    ),
    # Simulation stub worker: a deterministic model over wire CPU tensors.
    ("uniserve_worker/server/stub.py", "_token_ids", ".tolist()"): (
        "sim-stub: simulation worker deterministic next-token map"
    ),
    # Controller yield: the one steady-state CPU yield between query-only
    # readiness polls while device work is in flight. It waits on no device or
    # transport value; specs/tasks.md tracks replacing it with a completion
    # host-callback signal so the controller becomes poll-free.
    ("uniserve_worker/server/process.py", "WorkerServeLoop.run", "time.sleep()"): (
        "controller-yield: bounded CPU yield between query-only readiness polls"
    ),
}


def _module_name(path: Path) -> str:
    return str(path.relative_to(PACKAGE_ROOT.parent))


class _SyncVisitor(ast.NodeVisitor):
    """Collect (module, enclosing qualname, pattern) for every forbidden call."""

    def __init__(self, module: str) -> None:
        self.module = module
        self._scope: list[str] = []
        self.sites: list[tuple[str, str, str]] = []

    def _enter(self, node: ast.AST) -> None:
        self._scope.append(node.name)  # type: ignore[attr-defined]
        self.generic_visit(node)
        self._scope.pop()

    visit_FunctionDef = _enter
    visit_AsyncFunctionDef = _enter
    visit_ClassDef = _enter

    def visit_Call(self, node: ast.Call) -> None:
        pattern = _pattern(node)
        if pattern is not None:
            qualname = ".".join(self._scope) if self._scope else "<module>"
            self.sites.append((self.module, qualname, pattern))
        self.generic_visit(node)


def _pattern(node: ast.Call) -> str | None:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    attr = func.attr
    if attr in _SCALAR_ATTRS:
        return f".{attr}()"
    if attr == "cpu":
        return ".cpu()"
    if attr == "synchronize":
        return ".synchronize()"
    if attr == "result":
        return ".result()"
    if attr == "sleep" and isinstance(func.value, ast.Name) and func.value.id == "time":
        return "time.sleep()"
    if attr == "to":
        moves_to_cpu = any(
            isinstance(arg, ast.Constant) and arg.value == "cpu" for arg in node.args
        ) or any(
            keyword.arg == "device"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "cpu"
            for keyword in node.keywords
        )
        if moves_to_cpu:
            return '.to("cpu")'
    return None


def _detected_sites() -> set[tuple[str, str, str]]:
    sites: set[tuple[str, str, str]] = set()
    for path in PACKAGE_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        module = _module_name(path)
        visitor = _SyncVisitor(module)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        sites.update(visitor.sites)
    return sites


def test_every_forbidden_sync_site_is_classified() -> None:
    detected = _detected_sites()
    unclassified = sorted(site for site in detected if site not in ALLOWLIST)
    assert not unclassified, (
        "unclassified forbidden-synchronization call site(s) on the worker path; "
        "remove them or add a steady-state classification to ALLOWLIST:\n"
        + "\n".join(f"  {module}: {func}  {pattern}" for module, func, pattern in unclassified)
    )


def test_no_stale_classification_remains() -> None:
    detected = _detected_sites()
    stale = sorted(site for site in ALLOWLIST if site not in detected)
    assert not stale, (
        "classified site(s) no longer present; delete the stale ALLOWLIST entry:\n"
        + "\n".join(f"  {module}: {func}  {pattern}" for module, func, pattern in stale)
    )


def test_every_classification_states_a_permitted_phase() -> None:
    mislabeled = sorted(
        site for site, reason in ALLOWLIST.items() if not reason.startswith(_PERMITTED_PREFIXES)
    )
    assert not mislabeled, (
        "classification must begin with a permitted phase prefix "
        f"{_PERMITTED_PREFIXES}:\n"
        + "\n".join(f"  {module}: {func}  {pattern}" for module, func, pattern in mislabeled)
    )


def test_only_the_controller_yield_is_a_steady_state_site() -> None:
    # Every other classified site is proven to run in startup, administration,
    # delivery of a ready host product, a query-only ticket, host-known plan
    # construction, or a non-configured reference/sim backend. The controller
    # yield is the sole steady-state entry and is singular.
    steady = sorted(
        site for site, reason in ALLOWLIST.items() if reason.startswith("controller-yield:")
    )
    assert steady == [
        ("uniserve_worker/server/process.py", "WorkerServeLoop.run", "time.sleep()"),
    ]
