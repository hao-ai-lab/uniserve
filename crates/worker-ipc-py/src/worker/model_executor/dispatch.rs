//! Prepared model calls, resident graph selection and expert microbatches.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyTuple};
use uniserve_worker_ipc::ForwardStats as NativeStats;

use crate::stats::ForwardStats;
use crate::worker::cuda_graph::CUDAGraphError;
use crate::worker::execution::GraphBucket;
use crate::worker::expert_exchange::ExpertExchange;
use crate::worker::host::with_context;
use crate::worker::microbatches::Microbatches;
use crate::worker::model_inputs::InputBatch;
use crate::worker::model_results::ExecutionOutput;

use super::{context, execution, graphs};

/// The numerical backend supplies a key and its padded or exact input view.
pub(super) struct Selection {
    pub key: Py<PyAny>,
    pub inputs: Py<PyAny>,
    pub bucketed: bool,
}

pub(super) fn select(
    runner: &Bound<'_, PyAny>,
    batch: &Bound<'_, PyAny>,
    eligible: bool,
) -> PyResult<Option<Selection>> {
    let kwargs = PyDict::new(runner.py());
    kwargs.set_item("eligible", eligible)?;
    let result = runner.call_method("select_graph_shape", (batch,), Some(&kwargs))?;
    if result.is_none() {
        return Ok(None);
    }

    let (key, inputs, bucketed) = result.extract()?;
    Ok(Some(Selection {
        key,
        inputs,
        bucketed,
    }))
}

pub(super) fn exchange<'py>(
    runner: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, ExpertExchange>>> {
    if execution(runner)?.borrow().expert_order.is_none() {
        return Ok(None);
    }
    let exchange = context(runner)?.getattr("experts")?;
    if exchange.is_none() {
        Ok(None)
    } else {
        Ok(Some(exchange.getattr("_control")?.cast_into()?))
    }
}

pub(super) fn profile<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import("uniserve.profiling")?
        .call_method1("profile_range", (name,))
}

fn extent(
    runner: &Bound<'_, PyAny>,
    batch: &Bound<'_, PyAny>,
    selected: Option<&Selection>,
) -> PyResult<usize> {
    let live: usize = runner.call_method1("expert_tokens", (batch,))?.extract()?;
    let padded = selected
        .map(|shape| {
            runner
                .call_method1("graph_tokens", (shape.key.bind(runner.py()),))?
                .extract()
        })
        .transpose()?
        .unwrap_or(0);
    Ok(live.max(padded))
}

/// Join other source groups without disturbing this runner's prepared inputs.
fn agree(
    runner: &Bound<'_, PyAny>,
    exchange: &Bound<'_, ExpertExchange>,
    tokens: usize,
) -> PyResult<usize> {
    let py = runner.py();
    let kind = execution(runner)?.borrow().expert_order.unwrap_or(0);
    loop {
        let capacity = with_context(&profile(py, "uniserve.expert.agree")?, || {
            exchange.borrow().agree(py, tokens, kind, false)
        })?;
        if capacity == 0 {
            continue;
        }
        if exchange.borrow().active() {
            return Ok(capacity);
        }

        with_context(
            &profile(
                py,
                &format!("uniserve.expert.step tokens=0 capacity={capacity}"),
            )?,
            || {
                super::experts::join(execution(runner)?.as_unbound(), py, capacity)?;
                Ok(())
            },
        )?;
    }
}

/// All paths return the same native result, with graph observations attached.
fn forward(
    runner: &Bound<'_, PyAny>,
    batch: &Bound<'_, PyAny>,
    call: &Bound<'_, PyAny>,
    selected: Option<&Selection>,
    borrow: bool,
) -> PyResult<Py<ExecutionOutput>> {
    let py = runner.py();
    let buckets = execution(runner)?.borrow().buckets.bind(py).clone();
    let mut captured = false;
    let selected = match selected {
        Some(shape) if buckets.contains(shape.key.bind(py))? => Some(shape),
        Some(shape) if shape.bucketed => {
            if execution(runner)?.borrow().sealed {
                return Err(CUDAGraphError::new_err(format!(
                    "configured graph bucket is not resident: {}",
                    shape.key.bind(py).repr()?
                )));
            }
            graphs::capture(runner, batch, call)?;
            captured = true;
            Some(shape)
        }
        _ => None,
    };

    let Some(shape) = selected else {
        let result = graphs::run_eager(runner, batch, call)?;
        // Numerical backends may replay their own subgraphs, such as packed
        // vision slots. Keep those observations instead of labelling them eager.
        if result.borrow(py).stats.is_none() {
            let mut output = result.borrow(py).clone_ref(py);
            output.stats = Some(Py::new(
                py,
                ForwardStats::from(NativeStats {
                    cuda_graph_runtime_mode_counts: [("eager".into(), 1)].into(),
                    ..NativeStats::default()
                }),
            )?);
            return Py::new(py, output);
        }
        return Ok(result);
    };

    if let Some(exchange) = exchange(runner)? {
        let bucket = buckets
            .as_any()
            .get_item(shape.key.bind(py))?
            .cast_into::<GraphBucket>()?;
        exchange
            .borrow()
            .record_layers(bucket.borrow().layers.clone());
    }
    let live = batch
        .cast::<InputBatch>()?
        .borrow()
        .query_tokens
        .ok_or_else(|| PyValueError::new_err("graph input has no token extent"))?
        as u64;
    let padded = shape
        .inputs
        .bind(py)
        .cast::<InputBatch>()?
        .borrow()
        .query_tokens
        .ok_or_else(|| PyValueError::new_err("graph input has no token extent"))?
        as u64;
    let kwargs = PyDict::new(py);
    kwargs.set_item("borrow", borrow)?;
    let result = runner
        .call_method(
            "replay_graph",
            (shape.key.bind(py), shape.inputs.bind(py), batch),
            Some(&kwargs),
        )?
        .cast_into::<ExecutionOutput>()?;
    let mut result = result.borrow().clone_ref(py);
    result.stats = Some(Py::new(
        py,
        ForwardStats::from(NativeStats {
            cuda_graph_runtime_mode_counts: [(
                if captured {
                    "graph_capture"
                } else {
                    "graph_replay"
                }
                .into(),
                1,
            )]
            .into(),
            cuda_graph_captures: u64::from(captured),
            cuda_graph_replays: u64::from(!captured),
            cuda_graph_unpadded_tokens: live,
            cuda_graph_padded_tokens: padded - live,
            ..NativeStats::default()
        }),
    )?);
    Py::new(py, result)
}

fn single(
    runner: &Bound<'_, PyAny>,
    batch: &Bound<'_, PyAny>,
    eligible: bool,
    borrow: bool,
) -> PyResult<Py<ExecutionOutput>> {
    let py = runner.py();
    let context = context(runner)?;
    with_context(&context.call_method0("activate")?, || {
        let selected = select(runner, batch, eligible)?;
        let call = runner.getattr("batch_forward")?;
        let Some(exchange) = exchange(runner)? else {
            return forward(runner, batch, &call, selected.as_ref(), borrow);
        };

        let tokens = extent(runner, batch, selected.as_ref())?;
        let capacity = agree(runner, &exchange, tokens)?;
        exchange.borrow().begin(capacity)?;
        let result = with_context(
            &profile(
                py,
                &format!(
                    "uniserve.expert.step tokens={} capacity={capacity} local_tokens={tokens}",
                    batch
                        .cast::<InputBatch>()?
                        .borrow()
                        .query_tokens
                        .map_or_else(|| "None".to_owned(), |count| count.to_string())
                ),
            )?,
            || {
                let result = forward(runner, batch, &call, selected.as_ref(), borrow)?;
                context.call_method0("join_expert_layers")?;
                Ok(result)
            },
        );
        exchange.borrow().end();
        result
    })
}

fn microbatches(
    runner: &Bound<'_, PyAny>,
    batches: &Bound<'_, PyTuple>,
    eligible: bool,
    borrow: bool,
    rotation: &Bound<'_, Microbatches>,
) -> PyResult<Py<ExecutionOutput>> {
    let py = runner.py();
    let peers = runner.getattr("peers")?.cast_into::<PyTuple>()?;
    if peers.len() != batches.len() {
        return Err(PyValueError::new_err(
            "microbatch inputs must match prepared peers",
        ));
    }
    with_context(&context(runner)?.call_method0("activate")?, || {
        let mut selected = Vec::with_capacity(peers.len());
        let mut tokens = 0;
        for (peer, batch) in peers.iter().zip(batches.iter()) {
            let shape = if batch.is_none() {
                None
            } else {
                select(&peer, &batch, eligible)?
            };
            if !batch.is_none() {
                tokens = tokens.max(extent(&peer, &batch, shape.as_ref())?);
            }
            selected.push(shape);
        }
        let exchange = exchange(runner)?
            .ok_or_else(|| PyValueError::new_err("microbatches require expert execution"))?;
        let capacity = agree(runner, &exchange, tokens)?;
        execution(runner)?
            .borrow()
            .begin_expert_step(py, capacity)?;
        let result = (|| {
            let mut calls = Vec::with_capacity(peers.len());
            for ((peer, batch), shape) in peers.iter().zip(batches.iter()).zip(selected) {
                let peer = peer.unbind();
                let batch = batch.unbind();
                calls.push(PyCFunction::new_closure(
                    py,
                    None,
                    None,
                    move |args, _kwargs| -> PyResult<Py<PyAny>> {
                        let py = args.py();
                        let peer = peer.bind(py);
                        let batch = batch.bind(py);
                        if batch.is_none() {
                            let joins = execution(peer)?
                                .borrow()
                                .microbatch_joins
                                .as_ref()
                                .map(|joins| joins.clone_ref(py));
                            if let Some(joins) = joins {
                                joins.borrow(py).replay(py, capacity)?;
                                return Ok(py.None());
                            }
                        }
                        let result = if batch.is_none() {
                            py.None()
                        } else {
                            let result = forward(
                                peer,
                                batch,
                                &peer.getattr("batch_forward")?,
                                shape.as_ref(),
                                borrow,
                            )?;
                            result.borrow(py).validate_for(py, batch)?;
                            result.into_any()
                        };
                        context(peer)?.call_method0("join_expert_layers")?;
                        Ok(result)
                    },
                )?);
            }
            // The shared rotation joins all streams and retires host calls,
            // including failures, before any result is combined or reused.
            let results = rotation.borrow().__call__(
                py,
                calls
                    .into_iter()
                    .map(|call| call.unbind().into_any())
                    .collect(),
            )?;
            let mut populated = Vec::new();
            for result in results {
                if !result.is_none(py) {
                    populated.push(result);
                }
            }
            ExecutionOutput::combine(py, PyTuple::new(py, populated)?.as_any())
        })();
        let ended = execution(runner)?.borrow().end_expert_step(py);
        let result = result?;
        ended?;
        Ok(result)
    })
}

pub(super) fn run(
    runner: &Bound<'_, PyAny>,
    batches: &Bound<'_, PyTuple>,
    eligible: bool,
    borrow: bool,
) -> PyResult<(Py<ExecutionOutput>, Option<u64>)> {
    let py = runner.py();
    with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
        let rotation = execution(runner)?
            .borrow()
            .microbatches
            .as_ref()
            .map(|rotation| rotation.clone_ref(py));
        let result = if let Some(rotation) = rotation {
            microbatches(runner, batches, eligible, borrow, rotation.bind(py))?
        } else {
            let batch = batches.get_item(0)?;
            let result = single(runner, &batch, eligible, borrow)?;
            result.borrow(py).validate_for(py, &batch)?;
            result
        };
        let mut tokens = Some(0);
        for batch in batches {
            if !batch.is_none() {
                let count = batch
                    .cast::<InputBatch>()?
                    .borrow()
                    .query_tokens
                    .map(|count| count as u64);
                tokens = tokens.zip(count).map(|(total, count)| total + count);
            }
        }
        Ok((result, tokens))
    })
}
