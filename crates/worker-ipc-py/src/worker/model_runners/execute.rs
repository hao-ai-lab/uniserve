//! Numerical invocation order, stream dependencies and result observations.
//!
//! Python supplies row packing and numerical forwards. A failed invocation
//! joins any submitted input copies or model kernels before its caller can
//! retire the borrowed request storage.

use std::time::Instant;

use pyo3::exceptions::{PyException, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker_ipc::{CallKind, ForwardMode, MediaCall};

use crate::calls::Call;
use crate::ids::CallId;
use crate::stats::ForwardStats;

use super::super::events::CUDAEvent;
use super::super::host::with_context;
use super::super::model_results::ExecutionOutput;
use super::super::stream::CUDAStream;

#[allow(clippy::too_many_arguments)]
pub(super) fn run_batch<'py>(
    py: Python<'py>,
    owner: &Bound<'py, PyAny>,
    runner: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyTuple>,
    calls: &Bound<'py, PyTuple>,
    cache: &Bound<'py, PyAny>,
    tables: &Bound<'py, PyAny>,
    states: &Bound<'py, PyAny>,
    join: bool,
) -> PyResult<Bound<'py, ExecutionOutput>> {
    super::modules::ensure_open(owner)?;

    let started = Instant::now();
    let mode = rows.get_item(0)?.getattr("forward_mode")?;
    let kind: CallKind = pythonize::depythonize(&mode)?;
    let eligible = matches!(
        kind,
        CallKind::Forward(_) | CallKind::Media(MediaCall::Denoising)
    );
    let device = runner.getattr("device")?;
    let runtime = runner.getattr("cuda_stream")?;
    let stream = if runtime.is_none() {
        None
    } else {
        Some(runtime.getattr("_native")?.cast_into::<CUDAStream>()?)
    };
    let cuda = py.import("torch.cuda")?;
    let backend = py.import("uniserve_worker.execution.model_executor")?;
    let component = calls
        .get_item(0)?
        .extract::<PyRef<'_, Call>>()?
        .inner
        .component
        .clone();
    let rank: usize = crate::worker::config::native(&owner.getattr("worker_config")?)?.rank;
    let range = py.import("uniserve.profiling")?.call_method1(
        "profile_range",
        (format!(
            "uniserve.model.forward rank={rank} work={component}.{}",
            kind.as_str()
        ),),
    )?;

    with_context(&range, || {
        let prepared = (|| {
            let scope = match &stream {
                Some(stream) => {
                    let current = cuda.call_method1("current_stream", (&device,))?;
                    stream
                        .borrow()
                        .wait(py, current.getattr("cuda_stream")?.extract()?)?;
                    cuda.call_method1("stream", (runtime.getattr("stream")?,))?
                }
                None => py.import("contextlib")?.call_method0("nullcontext")?,
            };
            with_context(&scope, || {
                let count = runner.getattr("peers")?.len()?;
                let partitions = (0..count).map(|index| {
                    // Whole rows give concurrent microbatches disjoint KV writes.
                    let first = index * rows.len() / count;
                    let last = (index + 1) * rows.len() / count;
                    rows.get_slice(first, last)
                });
                let kwargs = PyDict::new(py);
                kwargs.set_item("cache", cache)?;
                kwargs.set_item("tables", tables)?;
                kwargs.set_item("states", states)?;
                backend
                    .call_method(
                        "_prepare_inputs",
                        (runner, rows, PyTuple::new(py, partitions)?),
                        Some(&kwargs),
                    )?
                    .extract::<(Bound<'_, PyTuple>, Bound<'_, PyAny>, bool)>()
            })
        })();
        let (batches, slots, borrow) = match prepared {
            Ok(value) => value,
            Err(error) => {
                join_failure(py, stream.as_ref(), &cuda, &device, None)?;
                return Err(classify(py, error, "_input_failure", &mode, calls));
            }
        };

        let mut completed = None;
        let executed = (|| {
            let (output, tokens) = super::dispatch::run(runner, &batches, eligible, borrow)?;
            backend.call_method1(
                "_validate_outputs",
                (output.borrow(py).values.bind(py), rows, &device),
            )?;

            completed = record(py, stream.as_ref(), &cuda, &device)?;
            let elapsed = started.elapsed().as_micros() as u64;
            let mut output = output.borrow(py).clone_ref(py);
            let mut stats = output
                .stats
                .as_ref()
                .ok_or_else(|| {
                    PyRuntimeError::new_err("model forward lost its execution statistics")
                })?
                .borrow(py)
                .inner
                .clone();
            stats.mode_counts = [(kind.as_str().into(), 1)].into();
            stats.mode_us = [(kind.as_str().into(), elapsed)].into();
            stats.mode_tokens = tokens
                .map(|value| (kind.as_str().into(), value))
                .into_iter()
                .collect();
            stats.component_us = [("forward".into(), elapsed)].into();

            if join {
                if let Some(event) = &completed {
                    let current = cuda.call_method1("current_stream", (&device,))?;
                    event.borrow(py).wait(py, Some(&current))?;
                }
                if kind == CallKind::Forward(ForwardMode::Decode) {
                    stats.component_us.insert(
                        "text_model_forward".into(),
                        started.elapsed().as_micros() as u64,
                    );
                }
            }
            output.request_pool_indices = Some(slots.unbind());
            output.output_event = completed.as_ref().map(|event| event.clone_ref(py));
            output.stats = Some(Py::new(py, ForwardStats::from(stats))?);
            owner.call_method0("_report_new_kernels")?;
            Py::new(py, output).map(|output| output.into_bound(py))
        })();
        match executed {
            Ok(output) => Ok(output),
            Err(error) => {
                join_failure(py, stream.as_ref(), &cuda, &device, completed)?;
                Err(classify(py, error, "_execution_failure", &mode, calls))
            }
        }
    })
}

fn record<'py>(
    py: Python<'py>,
    stream: Option<&Bound<'py, CUDAStream>>,
    cuda: &Bound<'py, PyModule>,
    device: &Bound<'py, PyAny>,
) -> PyResult<Option<Py<CUDAEvent>>> {
    let Some(stream) = stream else {
        return Ok(None);
    };
    let current = cuda.call_method1("current_stream", (device,))?;
    stream
        .borrow()
        .record(py, current.getattr("cuda_stream")?.extract()?)?
        .map(|event| Py::new(py, event))
        .transpose()
}

fn join_failure<'py>(
    py: Python<'py>,
    stream: Option<&Bound<'py, CUDAStream>>,
    cuda: &Bound<'py, PyModule>,
    device: &Bound<'py, PyAny>,
    event: Option<Py<CUDAEvent>>,
) -> PyResult<()> {
    let event = match event {
        Some(event) => Some(event),
        None => record(py, stream, cuda, device)?,
    };
    if let Some(event) = event {
        let current = cuda.call_method1("current_stream", (device,))?;
        event.borrow(py).wait(py, Some(&current))?;
    }
    Ok(())
}

fn classify(
    py: Python<'_>,
    error: PyErr,
    operation: &str,
    mode: &Bound<'_, PyAny>,
    calls: &Bound<'_, PyTuple>,
) -> PyErr {
    if !error.is_instance_of::<PyException>(py) {
        return error;
    }

    let classified = (|| {
        // Request coordinates are only needed on failure, never for a successful
        // numerical invocation. Read the native calls without Python field walks.
        let keys = calls
            .iter()
            .map(|call| {
                let call = call.extract::<PyRef<'_, Call>>()?;
                let request = call.inner.request_key;
                Ok((
                    request.engine_id,
                    request.request_id.0,
                    request.request_epoch,
                    Py::new(
                        py,
                        CallId {
                            inner: call.inner.call_id,
                        },
                    )?,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        py.import("uniserve_worker.execution.model_executor")?
            .call_method1(operation, (error.value(py), mode, PyTuple::new(py, keys)?))
    })();
    match classified {
        Ok(value) if value.is(error.value(py)) => error,
        Ok(value) => {
            let classified = PyErr::from_value(value);
            classified.set_cause(py, Some(error));
            classified
        }
        Err(classification) => classification,
    }
}
