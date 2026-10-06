//! Startup capture, eager preparation and standalone module execution.

use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker_ipc::ForwardStats as NativeStats;

use crate::stats::ForwardStats;
use crate::worker::cuda_graph::{CUDAGraphError, CUDAGraphRunner};
use crate::worker::execution::{Execution, on_stream};
use crate::worker::host::with_context;
use crate::worker::model_results::ExecutionOutput;

use super::dispatch::{exchange, select};
use super::{context, execution};

/// Evaluate prepared tensors in the caller's active numerical context.
pub(super) fn run_eager(
    runner: &Bound<'_, PyAny>,
    batch: &Bound<'_, PyAny>,
    forward: &Bound<'_, PyAny>,
) -> PyResult<Py<ExecutionOutput>> {
    let py = runner.py();
    let owner = execution(runner)?;
    if owner.borrow().image_capacity > 0 {
        return Execution::encode_images(&owner, runner, &batch.getattr("inputs")?);
    }
    let numerical = context(runner)?;
    if let Some(attention) = batch.getattr("inputs")?.getattr_opt("attention")?
        && !attention.is_none()
    {
        numerical.call_method1("bind_attention", (attention,))?;
    }

    let exchange = exchange(runner)?;
    let Some(exchange) = exchange.filter(|value| value.borrow().capacity() == 0) else {
        return Ok(forward.call1((batch,))?.extract()?);
    };

    let capacity = numerical.getattr("experts")?.getattr("max_tokens")?;
    exchange.borrow().warmup(py, capacity.extract()?)?;
    execution(runner)?
        .borrow()
        .begin_expert_step(py, capacity.extract()?)?;
    let result: PyResult<Py<ExecutionOutput>> = (|| {
        let joined = py
            .import("uniserve_worker.model_executor.model_runner")?
            .call_method1("joining_experts", (forward, &numerical))?;
        Ok(execution(runner)?
            .borrow()
            .warm_experts(py, &joined, batch)?
            .extract(py)?)
    })();
    let ended = execution(runner)?.borrow().end_expert_step(py);
    let result = result?;
    ended?;
    Ok(result)
}

pub(super) fn capture(
    runner: &Bound<'_, PyAny>,
    batch: &Bound<'_, PyAny>,
    forward: &Bound<'_, PyAny>,
) -> PyResult<()> {
    on_stream(&context(runner)?, &runner.getattr("device")?, || {
        capture_batch(runner, batch, forward)
    })
}

fn capture_batch(
    runner: &Bound<'_, PyAny>,
    batch: &Bound<'_, PyAny>,
    forward: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = runner.py();
    if execution(runner)?.borrow().sealed {
        return Err(CUDAGraphError::new_err(
            "batch capture is outside startup preparation",
        ));
    }
    let Some(selected) = select(runner, batch, true)? else {
        run_eager(runner, batch, forward)?;
        return Ok(());
    };
    let key = selected.key.bind(py);
    let buckets = execution(runner)?.borrow().buckets.bind(py).clone();
    if buckets.contains(key)? {
        return Ok(());
    }

    let backend = py.import("uniserve_worker.model_executor.cuda_graph")?;
    let storage = execution(runner)?.getattr("storage")?;
    let inputs = with_context(
        &storage.call_method1("allocate", (execution(runner)?,))?,
        || {
            // Configured buckets already own fixed addresses. Exact numerical
            // signatures may borrow a request's latents, so give them backing.
            if selected.bucketed {
                Ok(selected.inputs.bind(py).clone())
            } else {
                backend.call_method1("clone_inputs", (selected.inputs.bind(py),))
            }
        },
    )?;
    let exchange = if exchange(runner)?.is_some() {
        Some(context(runner)?.getattr("experts")?)
    } else {
        None
    };
    let capacities = if let Some(exchange) = &exchange {
        let tokens: usize = runner.call_method1("graph_tokens", (key,))?.extract()?;
        let values: Vec<usize> = exchange.getattr("capacities")?.extract()?;
        let values = values
            .into_iter()
            .rev()
            .filter(|&value| value >= tokens)
            .map(Some)
            .collect::<Vec<_>>();
        if values.is_empty() {
            return Err(CUDAGraphError::new_err(format!(
                "local graph of {tokens} tokens exceeds the expert exchange capacity"
            )));
        }
        values
    } else {
        vec![None]
    };
    let call = if selected.bucketed {
        let kwargs = PyDict::new(py);
        kwargs.set_item("padded", true)?;
        py.import("functools")?
            .getattr("partial")?
            .call((runner.getattr("batch_forward")?,), Some(&kwargs))?
    } else {
        forward.clone()
    };

    let bucket = backend.getattr("GraphBucket")?.call0()?;
    for capacity in capacities {
        let captured: PyResult<()> = (|| {
            if let (Some(exchange), Some(capacity)) = (&exchange, capacity) {
                exchange.call_method1("warmup", (capacity,))?;
                execution(runner)?
                    .borrow()
                    .begin_expert_step(py, capacity)?;
            }
            let captured: PyResult<()> = (|| {
                let graph = runner.call_method1("capture_graph", (key, &inputs, &call))?;
                bucket.set_item(capacity, graph)?;
                // Ending the step clears the invoked layer set. Retain it
                // before that reset because replay runs no host layer hooks.
                if let Some(exchange) = &exchange {
                    bucket.setattr("expert_layers", exchange.getattr("invoked")?)?;
                }
                Ok(())
            })();
            let ended = if exchange.is_some() {
                execution(runner)?.borrow().end_expert_step(py)
            } else {
                Ok(())
            };
            captured?;
            ended?;
            storage.call_method0("check")?;
            Ok(())
        })();
        if let Err(error) = captured {
            let _ = error.value(py).call_method1(
                "add_note",
                (format!(
                    "capturing {} bucket {}, expert capacity {capacity:?}",
                    runner.getattr("name")?,
                    key.repr()?
                ),),
            );
            let closed = bucket.call_method0("close");
            if let Err(cleanup) = closed {
                let _ = error.value(py).call_method1(
                    "add_note",
                    (format!("Graph cleanup also failed: {cleanup}"),),
                );
            }
            return Err(error);
        }
    }
    buckets.set_item(key, bucket)
}

pub(super) fn run_module(
    runner: &Bound<'_, PyAny>,
    args: &Bound<'_, PyTuple>,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<Py<ExecutionOutput>> {
    let py = runner.py();
    let started = Instant::now();
    let mut path = "eager";
    let output = on_stream(&context(runner)?, &runner.getattr("device")?, || {
        let backend = py.import("uniserve_worker.model_executor.cuda_graph")?;
        let values = (args, kwargs);
        let key = backend.call_method1("input_signature", (values,))?;
        let buckets = execution(runner)?.borrow().buckets.bind(py).clone();
        let bucket = buckets.get_item(&key)?;
        let graph = bucket
            .as_ref()
            .map(|bucket| bucket.get_item(py.None()))
            .transpose()?;
        let resources = runner.call_method0("resources")?.cast_into::<PyDict>()?;
        let numerical = py
            .import("uniserve_worker.model_executor.model_runner")?
            .getattr("_module_forward")?;
        let forward = runner.getattr("call")?.getattr("forward")?;
        let pools = execution(runner)?.borrow().pools.bind(py).clone();
        let result = if pools.is_truthy()?
            && (graph.is_some() || !execution(runner)?.borrow().sealed)
        {
            let graph = match graph {
                Some(graph) => {
                    path = "graph_replay";
                    graph
                }
                None => {
                    let storage = execution(runner)?.getattr("storage")?;
                    let inputs = with_context(
                        &storage.call_method1("allocate", (execution(runner)?,))?,
                        || backend.call_method1("clone_inputs", (values,)),
                    )?;
                    let capture_kwargs = PyDict::new(py);
                    capture_kwargs.set_item("pools", &pools)?;
                    let call = py
                        .import("functools")?
                        .getattr("partial")?
                        .call1((&numerical, &forward, &resources))?;
                    let graph = backend.getattr("CUDAGraphRunner")?.call_method(
                        "capture",
                        (context(runner)?, &inputs, call),
                        Some(&capture_kwargs),
                    )?;
                    if let Err(error) = storage.call_method0("check") {
                        graph.call_method0("close")?;
                        return Err(error);
                    }
                    let variants = PyDict::new(py);
                    variants.set_item(py.None(), &graph)?;
                    buckets.set_item(&key, backend.getattr("GraphBucket")?.call1((variants,))?)?;
                    path = "graph_capture";
                    graph
                }
            };
            graph
                .cast::<CUDAGraphRunner>()?
                .borrow()
                .replay(py, Some(values.into_pyobject(py)?.into_any().unbind()))?
                .into_bound(py)
        } else {
            numerical.call1((&forward, &resources, values))?
        };
        let output = runner
            .call_method1("result", (result,))?
            .cast_into::<ExecutionOutput>()?;
        ExecutionOutput::copy(&output)
    })?;
    let name: String = runner.getattr("name")?.extract()?;
    record_stats(py, &output, &name, path, started)?;
    Ok(output)
}

pub(super) fn record_stats(
    py: Python<'_>,
    output: &Py<ExecutionOutput>,
    name: &str,
    path: &str,
    started: Instant,
) -> PyResult<()> {
    let elapsed = started.elapsed().as_micros() as u64;
    let name = name.to_owned();
    output.borrow_mut(py).stats = Some(Py::new(
        py,
        ForwardStats::from(NativeStats {
            mode_counts: [(name.clone(), 1)].into(),
            mode_us: [(name, elapsed)].into(),
            component_us: [("forward".into(), elapsed)].into(),
            cuda_graph_runtime_mode_counts: [(path.into(), 1)].into(),
            cuda_graph_captures: u64::from(path == "graph_capture"),
            cuda_graph_replays: u64::from(path == "graph_replay"),
            ..NativeStats::default()
        }),
    )?);
    Ok(())
}
