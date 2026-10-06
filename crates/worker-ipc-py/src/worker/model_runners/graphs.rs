//! Startup capture, eager preparation and standalone module execution.

use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker_ipc::ForwardStats as NativeStats;

use crate::stats::ForwardStats;
use crate::worker::host::with_context;
use crate::worker::model_results::ExecutionOutput;

use super::dispatch::{exchange, graph_error, select};

/// Evaluate prepared tensors in the caller's active numerical context.
pub(super) fn run_eager(
    runner: &Bound<'_, PyAny>,
    batch: &Bound<'_, PyAny>,
    forward: &Bound<'_, PyAny>,
) -> PyResult<Py<ExecutionOutput>> {
    let py = runner.py();
    let context = runner.getattr("context")?;
    if let Some(attention) = batch.getattr("inputs")?.getattr_opt("attention")?
        && !attention.is_none()
    {
        context.call_method1("bind_attention", (attention,))?;
    }

    let exchange = exchange(runner)?;
    let Some(exchange) = exchange.filter(|value| value.borrow().capacity() == 0) else {
        return Ok(forward.call1((batch,))?.extract()?);
    };

    let capacity = context.getattr("experts")?.getattr("max_tokens")?;
    exchange.borrow().warmup(py, capacity.extract()?)?;
    runner.call_method1("begin_expert_step", (&capacity,))?;
    let result: PyResult<Py<ExecutionOutput>> = (|| {
        let joined = py
            .import("uniserve_worker.model_executor.model_runner")?
            .call_method1("joining_experts", (forward, &context))?;
        Ok(runner
            .call_method1("warm_experts", (joined, batch))?
            .extract()?)
    })();
    let ended = runner.call_method0("end_expert_step");
    let result = result?;
    ended?;
    Ok(result)
}

/// Order a numerical context between the caller's stream accesses, including
/// exceptional exits. The context itself restores PyTorch's current stream.
fn on_stream<T>(runner: &Bound<'_, PyAny>, call: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    let py = runner.py();
    let context = runner.getattr("context")?;
    let stream = context.getattr("stream")?;
    let cuda = py.import("torch.cuda")?;
    let current = || cuda.call_method1("current_stream", (runner.getattr("device")?,));
    if !stream.is_none() {
        stream.call_method1("wait", (current()?,))?;
    }
    let result = with_context(&context.call_method0("activate")?, call);
    let joined = if stream.is_none() {
        Ok(())
    } else {
        current()?
            .call_method1("wait_stream", (stream.getattr("stream")?,))
            .map(drop)
    };
    let result = result?;
    joined?;
    Ok(result)
}

pub(super) fn capture(
    runner: &Bound<'_, PyAny>,
    batch: &Bound<'_, PyAny>,
    forward: &Bound<'_, PyAny>,
) -> PyResult<()> {
    on_stream(runner, || capture_batch(runner, batch, forward))
}

fn capture_batch(
    runner: &Bound<'_, PyAny>,
    batch: &Bound<'_, PyAny>,
    forward: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = runner.py();
    if runner.getattr("_startup_complete")?.is_truthy()? {
        return Err(graph_error(
            py,
            "batch capture is outside startup preparation".into(),
        ));
    }
    let Some(selected) = select(runner, batch, true)? else {
        run_eager(runner, batch, forward)?;
        return Ok(());
    };
    let key = selected.key.bind(py);
    let buckets = runner.getattr("buckets")?.cast_into::<PyDict>()?;
    if buckets.contains(key)? {
        return Ok(());
    }

    let backend = py.import("uniserve_worker.model_executor.cuda_graph")?;
    let storage = runner.getattr("graph_storage")?;
    let inputs = with_context(&storage.call_method1("allocate", (runner,))?, || {
        // Configured buckets already own fixed addresses. Exact numerical
        // signatures may borrow a request's latents, so give them backing.
        if selected.bucketed {
            Ok(selected.inputs.bind(py).clone())
        } else {
            backend.call_method1("clone_inputs", (selected.inputs.bind(py),))
        }
    })?;
    let exchange = if exchange(runner)?.is_some() {
        Some(runner.getattr("context")?.getattr("experts")?)
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
            return Err(graph_error(
                py,
                format!("local graph of {tokens} tokens exceeds the expert exchange capacity"),
            ));
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
            if let Some(exchange) = &exchange {
                exchange.call_method1("warmup", (capacity,))?;
                runner.call_method1("begin_expert_step", (capacity,))?;
            }
            let captured: PyResult<()> = (|| {
                let graph = runner.call_method1("capture_graph", (key, &inputs, &call))?;
                bucket.getattr("graphs")?.set_item(capacity, graph)?;
                // Ending the step clears the invoked layer set. Retain it
                // before that reset because replay runs no host layer hooks.
                if let Some(exchange) = &exchange {
                    bucket.setattr("expert_layers", exchange.getattr("invoked")?)?;
                }
                Ok(())
            })();
            let ended = if exchange.is_some() {
                runner.call_method0("end_expert_step").map(drop)
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
    let output = on_stream(runner, || {
        let backend = py.import("uniserve_worker.model_executor.cuda_graph")?;
        let values = (args, kwargs);
        let key = backend.call_method1("input_signature", (values,))?;
        let buckets = runner.getattr("buckets")?.cast_into::<PyDict>()?;
        let bucket = buckets.get_item(&key)?;
        let graph = bucket
            .as_ref()
            .map(|bucket| bucket.getattr("graphs")?.get_item(py.None()))
            .transpose()?;
        let resources = runner.call_method0("resources")?.cast_into::<PyDict>()?;
        let numerical = py
            .import("uniserve_worker.model_executor.model_runner")?
            .getattr("_module_forward")?;
        let forward = runner.getattr("call")?.getattr("forward")?;
        let pools = runner.getattr("pools")?;
        let result = if pools.is_truthy()?
            && (graph.is_some() || !runner.getattr("_startup_complete")?.is_truthy()?)
        {
            let graph = match graph {
                Some(graph) => {
                    path = "graph_replay";
                    graph
                }
                None => {
                    let storage = runner.getattr("graph_storage")?;
                    let inputs =
                        with_context(&storage.call_method1("allocate", (runner,))?, || {
                            backend.call_method1("clone_inputs", (values,))
                        })?;
                    let capture_kwargs = PyDict::new(py);
                    capture_kwargs.set_item("pools", &pools)?;
                    let call = py
                        .import("functools")?
                        .getattr("partial")?
                        .call1((&numerical, &forward, &resources))?;
                    let graph = backend.getattr("CUDAGraphRunner")?.call_method(
                        "capture",
                        (runner.getattr("context")?, &inputs, call),
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
            graph.call_method1("replay", (values,))?
        } else {
            numerical.call1((&forward, &resources, values))?
        };
        let output = runner
            .call_method1("result", (result,))?
            .cast_into::<ExecutionOutput>()?;
        ExecutionOutput::copy(&output)
    })?;
    let elapsed = started.elapsed().as_micros() as u64;
    let name: String = runner.getattr("name")?.extract()?;
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
    Ok(output)
}
