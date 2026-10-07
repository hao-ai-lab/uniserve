//! Numerical invocation order, stream dependencies and result observations.
//!
//! Rust prepares native rows; Python supplies numerical tensor operations. A failed invocation
//! joins any submitted input copies or model kernels before its caller can
//! retire the borrowed request storage.

use std::time::Instant;

use super::ModelExecutor;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::sync::Arc;
use uniserve_worker::TokenSelection;
use uniserve_worker_ipc::{CallKind, ForwardMode, MediaCall};

use crate::calls::Call;
use crate::stats::ForwardStats;

use super::super::events::CUDAEvent;
use super::super::host::with_context;
use super::super::model_results::ExecutionOutput;
use super::super::stream::CUDAStream;
use crate::worker::error::model_failure;
use crate::worker::input_buffers::InputBuffers;
use crate::worker::model_inputs::{InputBatch, InputRow, TokenRow};

#[allow(clippy::too_many_arguments)]
pub(super) fn run_batch<'py>(
    py: Python<'py>,
    owner: &Bound<'py, ModelExecutor>,
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
    let kind = rows.get_item(0)?.cast::<InputRow>()?.borrow().kind;
    let eligible = matches!(
        kind,
        CallKind::Forward(_) | CallKind::Media(MediaCall::Denoising)
    );
    let device = runner.getattr("device")?;
    let runtime = runner.getattr("cuda_stream")?;
    let stream = if runtime.is_none() {
        None
    } else {
        Some(runtime.cast::<CUDAStream>()?.clone())
    };
    let cuda = py.import("torch.cuda")?;
    let component = calls
        .get_item(0)?
        .extract::<PyRef<'_, Call>>()?
        .inner
        .component
        .clone();
    let rank: usize = Arc::clone(&owner.borrow().config).rank;
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
                        .wait_for(py, current.getattr("cuda_stream")?.extract()?)?;
                    cuda.call_method1("stream", (runtime.getattr("stream")?,))?
                }
                None => py.import("contextlib")?.call_method0("nullcontext")?,
            };
            with_context(&scope, || {
                prepare_inputs(runner, rows, kind, cache, tables, states)
            })
        })();
        let (batches, slots, borrow) = match prepared {
            Ok(value) => value,
            Err(error) => {
                join_failure(py, stream.as_ref(), &cuda, &device, None)?;
                return Err(model_failure(py, error, true, kind, calls));
            }
        };

        let mut completed = None;
        let executed = (|| {
            let (output, tokens) = super::dispatch::run(runner, &batches, eligible, borrow)?;
            output.borrow(py).validate(rows, &device)?;

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
            ModelExecutor::report_new_kernels(owner)?;
            Py::new(py, output).map(|output| output.into_bound(py))
        })();
        match executed {
            Ok(output) => Ok(output),
            Err(error) => {
                join_failure(py, stream.as_ref(), &cuda, &device, completed)?;
                Err(model_failure(py, error, false, kind, calls))
            }
        }
    })
}

/// Partition whole request rows into independent numerical buffers. Empty peers
/// retain their position so every AFD microbatch keeps the collective sequence.
fn prepare_inputs<'py>(
    runner: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyTuple>,
    kind: CallKind,
    cache: &Bound<'py, PyAny>,
    tables: &Bound<'py, PyAny>,
    states: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyAny>, bool)> {
    let py = runner.py();
    let peers = runner.getattr("peers")?.cast_into::<PyTuple>()?;
    if !runner
        .getattr("execution")?
        .getattr("microbatches")?
        .is_none()
    {
        runner
            .getattr("input_buffers")?
            .cast::<InputBuffers>()?
            .borrow()
            .validate(rows)?;
    }
    let mut batches = Vec::with_capacity(peers.len());
    let mut slots = Vec::new();
    for (index, peer) in peers.iter().enumerate() {
        let first = index * rows.len() / peers.len();
        let last = (index + 1) * rows.len() / peers.len();
        if first == last {
            batches.push(py.None());
            continue;
        }
        let buffers = peer.getattr("input_buffers")?.cast_into::<InputBuffers>()?;
        let batch = InputBuffers::prepare(
            &buffers,
            &rows.get_slice(first, last),
            kind,
            None,
            (!cache.is_none()).then_some(cache),
            (!tables.is_none()).then_some(tables),
            (!states.is_none()).then_some(states),
        )?;
        let batch = batch.cast_bound::<InputBatch>(py)?;
        slots.push(batch.borrow().request_pool_indices.clone_ref(py));
        batches.push(batch.clone().into_any().unbind());
    }
    let slots = if peers.len() == 1 {
        slots
            .pop()
            .ok_or_else(|| PyRuntimeError::new_err("numerical call has no input rows"))?
            .into_bound(py)
    } else {
        py.import("torch")?
            .call_method1("cat", (PyTuple::new(py, slots)?,))?
    };

    let mut borrow = true;
    for row in rows {
        let Ok(row) = row.cast::<TokenRow>() else {
            borrow = false;
            break;
        };
        let row = row.borrow();
        if row.query_tokens(py)? != 1 || row.selection != Some(TokenSelection::LastLogits) {
            borrow = false;
            break;
        }
    }
    Ok((PyTuple::new(py, batches)?, slots, borrow))
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
        .record_for(py, current.getattr("cuda_stream")?.extract()?)?
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
