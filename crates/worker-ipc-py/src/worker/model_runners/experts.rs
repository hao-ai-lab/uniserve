//! Expert execution discovery, idle participation and shared join graphs.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use uniserve_worker_ipc::{CallKind, ForwardMode};

use super::{ModelRunners, context, execution};
use crate::worker::execution::{Execution, JoinGraphs, close_all};
use crate::worker::expert_exchange::ExpertExchange;
use crate::worker::host::with_context;
use crate::worker::stream::CUDAStream;

pub(super) fn configure(
    slf: &Bound<'_, ModelRunners>,
    owner: &Bound<'_, PyAny>,
    max_tokens: usize,
) -> PyResult<()> {
    let py = slf.py();
    let config = owner.getattr("worker_config")?;
    let config_native = crate::worker::config::native(&config)?;
    let count: usize = config_native.expert_microbatches;
    let attention_ranks: usize = owner.getattr("attention_ranks")?.extract()?;
    let fused = py.import("uniserve.nn.moe")?.getattr("FusedMoE")?;
    let mut layers = Vec::new();
    if owner.getattr("expert_weights")?.is_none() {
        for module in owner
            .getattr("model")?
            .call_method0("modules")?
            .try_iter()?
        {
            let module = module?;
            if module.is_instance(&fused)?
                && (attention_ranks != 0
                    || module
                        .getattr("expert_group")?
                        .getattr("size")?
                        .extract::<usize>()?
                        > 1)
            {
                layers.push(module);
            }
        }
    }
    let Some(first) = layers.first() else {
        if count != 1 {
            return Err(PyValueError::new_err(
                "microbatch execution requires routed experts",
            ));
        }
        return Ok(());
    };
    let group = if attention_ranks != 0 {
        owner.getattr("expert_group")?
    } else {
        first.getattr("expert_group")?
    };
    for layer in &layers[1..] {
        let same_group = attention_ranks != 0 || group.eq(layer.getattr("expert_group")?)?;
        let same_shape =
            ["num_experts", "top_k", "hidden_size"]
                .iter()
                .try_fold(true, |same, field| {
                    Ok::<_, PyErr>(same && layer.getattr(*field)?.eq(first.getattr(*field)?)?)
                })?;
        if !same_group || !same_shape {
            return Err(PyValueError::new_err(
                "expert-parallel layers share one group, expert count, top-k and hidden size",
            ));
        }
    }
    let streams = std::sync::Arc::clone(&slf.borrow().streams);
    if streams.lane_count() > 1 {
        return Err(PyValueError::new_err(
            "an expert-parallel worker runs its forwards on one lane",
        ));
    }

    let mut max_tokens = max_tokens;
    if attention_ranks != 0 {
        // Expert-only ranks borrow the source's capacity, including graph
        // padding and multimodal rows. All microbatches use this same bound.
        let torch = py.import("torch")?;
        let options = PyDict::new(py);
        options.set_item("dtype", torch.getattr("int64")?)?;
        options.set_item("device", "cpu")?;
        let capacity = torch.call_method("tensor", (max_tokens,), Some(&options))?;
        let options = PyDict::new(py);
        options.set_item("src", group.getattr("ranks")?.get_item(0)?)?;
        options.set_item("group", group.call_method0("_require")?)?;
        py.import("torch.distributed")?
            .call_method("broadcast", (&capacity,), Some(&options))?;
        max_tokens = capacity.call_method0("item")?.extract()?;
    }
    let options = PyDict::new(py);
    if attention_ranks != 0 && config_native.role != "experts" {
        let first_rank = group.getattr("rank")?.extract::<usize>()? - config_native.rank;
        let width: usize = config_native.world_size;
        let ranks: Vec<usize> = group.getattr("ranks")?.extract()?;
        options.set_item(
            "source_group",
            PyTuple::new(py, &ranks[first_rank..first_rank + width])?,
        )?;
    }
    options.set_item("max_tokens", max_tokens)?;
    for field in [
        "top_k",
        "num_experts",
        "hidden_size",
        "intermediate_size",
        "activation",
    ] {
        options.set_item(field, first.getattr(field)?)?;
    }
    options.set_item(
        "device",
        py.import("uniserve.runtime.device")?
            .call_method1("canonical_device", (&config_native.device,))?,
    )?;
    options.set_item("attention_ranks", attention_ranks)?;
    options.set_item("transport", &config_native.expert_exchange)?;
    let exchange_type = py
        .import("uniserve.runtime.expert_exchange")?
        .getattr("ExpertExchange")?;
    for index in 0..count {
        let exchange = exchange_type.call((&group,), Some(&options))?;
        // Retain every accepted exchange before another allocation can fail.
        slf.borrow_mut().exchanges.push(exchange.unbind());
        if index != 0 {
            streams.fork_microbatch(py)?;
        }
    }
    Ok(())
}

pub(super) fn configure_worker(
    slf: &Bound<'_, ModelRunners>,
    owner: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = slf.py();
    let config = owner.getattr("worker_config")?;
    let config_native = crate::worker::config::native(&config)?;
    let streams = std::sync::Arc::clone(&slf.borrow().streams);
    streams.initialize(owner, None)?;
    configure(slf, owner, config_native.max_batch_tokens)?;
    let exchanges = slf
        .borrow()
        .exchanges
        .iter()
        .map(|exchange| exchange.clone_ref(py))
        .collect::<Vec<_>>();
    if exchanges.is_empty() {
        return Err(PyValueError::new_err(
            "expert workers require an expert exchange",
        ));
    }

    let capture = config_native.graph_policy != "off";
    let storage = owner.getattr("graph_storage")?;
    let model = owner.getattr("model")?;
    let runtime = py.import("uniserve.runtime")?;
    let size = py.import("uniserve.model")?.getattr("TextSize")?;
    for (stream, exchange) in streams.expert_streams(py)?.iter().zip(exchanges) {
        let options = PyDict::new(py);
        options.set_item("stream", &stream)?;
        options.set_item("experts", exchange.bind(py))?;
        let context = runtime
            .getattr("ExecutionContext")?
            .call((&model,), Some(&options))?;
        let options = PyDict::new(py);
        options.set_item("storage", &storage)?;
        options.set_item(
            "devices",
            PyTuple::new(
                py,
                if capture {
                    vec![stream.getattr("device")?]
                } else {
                    vec![]
                },
            )?,
        )?;
        let execution = py
            .get_type::<Execution>()
            .call(("experts", &context), Some(&options))?
            .cast_into::<Execution>()?;
        slf.borrow_mut()
            .expert_executions
            .push(execution.clone().unbind());
        with_context(&storage.call_method1("allocate", (&execution,))?, || {
            context.call_method1(
                "prepare",
                (size.call1((exchange.bind(py).getattr("max_tokens")?, 1))?,),
            )
        })?;
    }
    let peers = slf
        .borrow()
        .expert_executions
        .iter()
        .map(|peer| peer.clone_ref(py))
        .collect::<Vec<_>>();
    if peers.len() > 1 {
        Execution::bind_microbatches(py, peers)?;
    }
    Ok(())
}

/// Discover one primary runner per numerical call; microbatch peers share its
/// call order. Rank plans are compared once before warmup enters collectives.
pub(super) fn bind(slf: &Bound<'_, ModelRunners>, owner: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = slf.py();
    let Some(exchange) = slf
        .borrow()
        .exchanges
        .first()
        .map(|exchange| exchange.clone_ref(py))
    else {
        return Ok(());
    };
    let exchange = exchange.bind(py);
    let runners = slf
        .borrow()
        .inner
        .iter()
        .map(|runner| runner.clone_ref(py))
        .collect::<Vec<_>>();
    let mut selected = Vec::new();
    let mut calls = Vec::new();
    for runner in runners {
        let runner = runner.bind(py);
        let peers = runner.getattr("peers")?.cast_into::<PyTuple>()?;
        if !peers.get_item(0)?.is(runner) || !context(runner)?.getattr("experts")?.is(exchange) {
            continue;
        }
        let eligible = runner.getattr("call_kinds")?.try_iter()?.try_fold(
            false,
            |found, kind| -> PyResult<bool> {
                let kind: CallKind = pythonize::depythonize(&kind?)?;
                Ok(found
                    || matches!(
                        kind,
                        CallKind::Forward(ForwardMode::Prefill | ForwardMode::TokenDenoising)
                    ))
            },
        )?;
        if eligible {
            let peers = peers
                .iter()
                .map(|peer| execution(&peer).map(Bound::unbind))
                .collect::<PyResult<Vec<_>>>()?;
            for peer in &peers {
                peer.borrow_mut(py).expert_order = Some(selected.len() as i64);
            }
            calls.push(runner.call_method0("capture_plan")?);
            selected.push(peers);
        }
    }
    slf.borrow_mut().expert_runners = selected;

    let plan = (
        exchange.getattr("capacities")?,
        crate::worker::config::native(&owner.getattr("worker_config")?)?.expert_microbatches,
        PyTuple::new(py, calls)?,
    )
        .into_pyobject(py)?;
    let group = exchange.getattr("group")?;
    let ranks: usize = group.getattr("size")?.extract()?;
    let plans = PyList::new(py, (0..ranks).map(|_| py.None()))?;
    let options = PyDict::new(py);
    options.set_item("group", group.call_method0("_require")?)?;
    py.import("torch.distributed")?.call_method(
        "all_gather_object",
        (&plans, plan),
        Some(&options),
    )?;
    let sources: usize = exchange.getattr("attention_ranks")?.extract()?;
    let first = plans.get_item(0)?;
    for (index, plan) in plans.iter().enumerate() {
        let same_call = (sources != 0 && index >= sources) || plan.eq(&first)?;
        let same_capacity =
            plan.get_item(0)?.eq(first.get_item(0)?)? && plan.get_item(1)?.eq(first.get_item(1)?)?;
        if !same_call || !same_capacity {
            return Err(PyRuntimeError::new_err(format!(
                "expert-parallel ranks have different capture plans: {}",
                plans.repr()?
            )));
        }
    }
    Ok(())
}

fn first(slf: &Bound<'_, ModelRunners>) -> Option<Py<Execution>> {
    let owner = slf.borrow();
    owner
        .expert_executions
        .first()
        .or_else(|| owner.expert_runners.first().and_then(|peers| peers.first()))
        .map(|execution| execution.clone_ref(slf.py()))
}

pub(super) fn capture(slf: &Bound<'_, ModelRunners>) -> PyResult<()> {
    let py = slf.py();
    let Some(first) = first(slf) else {
        return Ok(());
    };
    let (context, pools) = {
        let first = first.borrow(py);
        (first.context.clone_ref(py), first.pools.clone_ref(py))
    };
    if pools.bind(py).is_empty() {
        return Ok(());
    }
    let exchange = context.bind(py).getattr("experts")?;
    let capacities = exchange.getattr("capacities")?;
    let options = PyDict::new(py);
    options.set_item("pools", pools.bind(py))?;
    options.set_item("step", first.bind(py).getattr("join_expert_step")?)?;
    let joins = py
        .get_type::<JoinGraphs>()
        .call((context.bind(py), &exchange, &capacities), Some(&options))?
        .cast_into::<JoinGraphs>()?
        .unbind();
    first.borrow_mut(py).joins = Some(joins.clone_ref(py));
    let runners = slf
        .borrow()
        .expert_runners
        .iter()
        .map(|peers| {
            peers
                .iter()
                .map(|peer| peer.clone_ref(py))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for peer in runners.iter().flatten() {
        peer.borrow_mut(py).joins = Some(joins.clone_ref(py));
    }

    let Some(runner) = runners
        .first()
        .filter(|_| first.borrow(py).microbatches.is_some())
    else {
        return Ok(());
    };
    // The complete join warmed every peer. Capture each empty microbatch in
    // its own pool so it can overlap populated peers without sharing scratch.
    for (index, peer) in runner.iter().enumerate() {
        let (context, pools) = {
            let peer = peer.borrow(py);
            (peer.context.clone_ref(py), peer.pools.clone_ref(py))
        };
        let options = PyDict::new(py);
        options.set_item("pools", pools.bind(py))?;
        options.set_item("warm", false)?;
        let joins = py
            .get_type::<JoinGraphs>()
            .call(
                (
                    context.bind(py),
                    context.bind(py).getattr("experts")?,
                    &capacities,
                ),
                Some(&options),
            )?
            .cast_into::<JoinGraphs>()?
            .unbind();
        for runner in &runners {
            runner[index].borrow_mut(py).microbatch_joins = Some(joins.clone_ref(py));
        }
    }
    Ok(())
}

pub(super) fn join(execution: &Py<Execution>, py: Python<'_>, capacity: usize) -> PyResult<()> {
    let joins = execution
        .borrow(py)
        .joins
        .as_ref()
        .map(|joins| joins.clone_ref(py));
    match joins {
        Some(joins) => joins.borrow(py).replay(py, capacity),
        None => execution.borrow(py).join_expert_step(py, capacity),
    }
}

pub(super) fn idle(slf: &Bound<'_, ModelRunners>, leaving: bool) -> PyResult<bool> {
    let py = slf.py();
    let Some(execution) = first(slf) else {
        return Ok(false);
    };
    let context = execution.borrow(py).context.clone_ref(py);
    let exchange = context
        .bind(py)
        .getattr("experts")?
        .getattr("_control")?
        .cast_into::<ExpertExchange>()?;
    let capacity = with_context(
        &super::dispatch::profile(py, "uniserve.expert.agree")?,
        || exchange.borrow().agree(py, 0, 0, leaving),
    )?;
    if capacity == 0 {
        return Ok(false);
    }
    let stream = context.bind(py).getattr("stream")?;
    let native = if stream.is_none() {
        None
    } else {
        Some(stream.getattr("_native")?.cast_into::<CUDAStream>()?)
    };
    let current = || {
        py.import("torch.cuda")?
            .call_method1("current_stream", (stream.getattr("device")?,))
    };
    if let Some(native) = &native {
        native
            .borrow()
            .wait(py, current()?.getattr("cuda_stream")?.extract()?)?;
    }
    let result = with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
        with_context(
            &super::dispatch::profile(
                py,
                &format!("uniserve.expert.step tokens=0 capacity={capacity}"),
            )?,
            || join(&execution, py, capacity),
        )
    });
    let joined = (|| {
        if let Some(native) = native {
            let current = current()?;
            if let Some(event) = native
                .borrow()
                .record(py, current.getattr("cuda_stream")?.extract()?)?
            {
                event.wait(py, Some(&current))?;
            }
        }
        Ok(())
    })();
    close_all(py, [result, joined])?;
    Ok(true)
}
