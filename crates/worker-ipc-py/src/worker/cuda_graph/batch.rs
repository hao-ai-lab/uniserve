//! Homogeneous batch capture, attention rebinding and live output selection.

use pyo3::prelude::*;

use super::CUDAGraphRunner;
use crate::worker::execution_context::ExecutionContext;
use crate::worker::model_inputs::InputBatch;
use crate::worker::model_results::ExecutionOutput;

/// Capture one numerical batch and its greedy continuations on shared backing.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (context, batch, call, *, pools=None, cache=None, predicates=None, warmup=None))]
pub(in crate::worker) fn capture_batch(
    py: Python<'_>,
    context: Py<ExecutionContext>,
    batch: Py<InputBatch>,
    call: Py<PyAny>,
    pools: Option<&Bound<'_, PyAny>>,
    cache: Option<Py<PyAny>>,
    predicates: Option<Py<PyAny>>,
    warmup: Option<Py<PyAny>>,
) -> PyResult<CUDAGraphRunner> {
    let partial = py.import("functools")?.getattr("partial")?;
    let compute = partial.call1((backend(py)?.getattr("batch_output")?, call, predicates))?;
    let warmup = warmup
        .map(|warmup| partial.call1((warmup, &compute)).map(Bound::unbind))
        .transpose()?;
    capture_hidden(py, context, batch, compute.unbind(), pools, cache, warmup)
}

/// Capture a prefill backbone or cache-only call, restoring request state.
#[pyfunction]
#[pyo3(signature = (context, batch, call, *, pools=None, cache=None, warmup=None))]
pub(in crate::worker) fn capture_hidden(
    py: Python<'_>,
    context: Py<ExecutionContext>,
    batch: Py<InputBatch>,
    call: Py<PyAny>,
    pools: Option<&Bound<'_, PyAny>>,
    cache: Option<Py<PyAny>>,
    warmup: Option<Py<PyAny>>,
) -> PyResult<CUDAGraphRunner> {
    // Warmup and capture may write KV rows and completion controls. Preserve
    // them before either call, then restore after both through the same owner.
    let restore = ExecutionContext::with_active(context.bind(py), || {
        let inputs = batch.borrow(py).inputs.bind(py).clone();
        if let Some(attention) = inputs
            .getattr_opt("attention")?
            .filter(|value| !value.is_none())
        {
            ExecutionContext::bind_attention(context.bind(py), &attention, false)?;
        }
        Ok(backend(py)?
            .call_method1("restore_writes", (&batch, cache))?
            .unbind())
    })?;
    CUDAGraphRunner::capture(
        py,
        context,
        batch.into_any(),
        call,
        pools,
        Some(restore),
        true,
        warmup,
    )
}

/// Replay live inputs and drop padding rows. Borrowed results are overwritten
/// by the next replay; other callers receive independent tensor copies.
#[pyfunction]
#[pyo3(signature = (graph, batch, *, rows=None, borrow=false))]
pub(in crate::worker) fn replay_batch(
    py: Python<'_>,
    graph: PyRef<'_, CUDAGraphRunner>,
    batch: Py<InputBatch>,
    rows: Option<usize>,
    borrow: bool,
) -> PyResult<Py<ExecutionOutput>> {
    let count = rows.unwrap_or_else(|| batch.borrow(py).row_count);
    let replayed = replay(&graph, batch.bind(py))?;
    let (output, greedy): (Py<ExecutionOutput>, Py<PyAny>) = replayed.extract(py)?;
    let mut result = output.borrow(py).rows(py, 0, count);
    result.greedy = backend(py)?
        .call_method1("trim_greedy", (greedy, count))?
        .extract()?;
    let result = Py::new(py, result)?;
    if borrow {
        Ok(result)
    } else {
        result.borrow(py).wait(py)?;
        ExecutionOutput::copy(result.bind(py))
    }
}

/// Replay a prefill graph and borrow its hidden output until the next replay.
#[pyfunction]
pub(in crate::worker) fn replay_hidden(
    graph: PyRef<'_, CUDAGraphRunner>,
    batch: &Bound<'_, InputBatch>,
) -> PyResult<Py<PyAny>> {
    replay(&graph, batch)
}

fn replay(graph: &CUDAGraphRunner, batch: &Bound<'_, InputBatch>) -> PyResult<Py<PyAny>> {
    let py = batch.py();
    let executable = graph.executable.borrow(py);
    let context = executable.context(py)?;
    ExecutionContext::with_active(context.bind(py), || {
        let fixed = graph.inputs.borrow(py);
        fixed.copy(batch.as_any())?;
        let inputs = batch.borrow().inputs.bind(py).clone();
        if let Some(attention) = inputs
            .getattr_opt("attention")?
            .filter(|value| !value.is_none())
        {
            let static_batch = fixed.value.bind(py).cast::<InputBatch>()?.borrow();
            let static_attention = static_batch.inputs.bind(py).getattr("attention")?;
            let attention = super::attention::bind(&static_attention, &attention)?;
            ExecutionContext::bind_attention(context.bind(py), &attention, true)?;
        }
        executable.replay(py)
    })
}

fn backend(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("uniserve_worker.model_executor.graph_inputs")
}

pub(in crate::worker) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(capture_batch, module)?)?;
    module.add_function(wrap_pyfunction!(capture_hidden, module)?)?;
    module.add_function(wrap_pyfunction!(replay_batch, module)?)?;
    module.add_function(wrap_pyfunction!(replay_hidden, module)?)?;
    Ok(())
}
