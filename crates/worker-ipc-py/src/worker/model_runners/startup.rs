//! Startup ordering for numerical preparation and expert participation.

use std::collections::HashSet;

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use uniserve_worker_ipc::{CallKind, ForwardMode, MediaCall};

use super::{ModelRunners, execution, experts, resources};
use crate::worker::expert_exchange::ExpertExchange;
use crate::worker::host::with_context;

pub(super) fn prepare(
    slf: &Bound<'_, ModelRunners>,
    owner: &Bound<'_, PyAny>,
    tokenizer: &Bound<'_, PyAny>,
    latents: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = slf.py();
    with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
        experts::bind(slf, owner)?;
        let (exchange, expert) = {
            let runners = slf.borrow();
            (
                runners.exchanges.first().map(|value| value.clone_ref(py)),
                runners
                    .expert_executions
                    .first()
                    .map(|value| value.clone_ref(py)),
            )
        };
        let control = exchange
            .as_ref()
            .map(|exchange| -> PyResult<_> {
                Ok(exchange
                    .bind(py)
                    .getattr("_control")?
                    .cast_into::<ExpertExchange>()?)
            })
            .transpose()?;
        if let (Some(expert), Some(control)) = (expert, &control) {
            // Model ranks announce actual eager calls, including warmup for
            // captures. Expert-only ranks execute the matching layer sequence.
            loop {
                let capacity = control.borrow().warmup(py, 0)?;
                if capacity == 0 {
                    break;
                }
                expert.borrow(py).join_expert_step(py, capacity)?;
            }
        } else {
            let runners = slf
                .borrow()
                .inner
                .iter()
                .map(|runner| runner.clone_ref(py))
                .collect::<Vec<_>>();
            let mut prepared = Vec::new();
            for runner in runners {
                let kinds = runner
                    .bind(py)
                    .getattr("call_kinds")?
                    .try_iter()?
                    .map(|kind| Ok(pythonize::depythonize(&kind?)?))
                    .collect::<PyResult<HashSet<CallKind>>>()?;
                prepared.push((runner, kinds));
            }
            let backend = py.import("uniserve_worker.model_executor.startup")?;
            for kind in [
                CallKind::Forward(ForwardMode::Prefill),
                CallKind::Forward(ForwardMode::Decode),
                CallKind::Forward(ForwardMode::TokenDenoising),
                CallKind::Media(MediaCall::Denoising),
            ] {
                for (runner, kinds) in &prepared {
                    if !kinds.contains(&kind) {
                        continue;
                    }
                    let runner = runner.bind(py);
                    match kind {
                        CallKind::Forward(ForwardMode::Prefill) => {
                            let mut shapes = runner.getattr("shapes")?.getattr("prefill")?;
                            if !shapes.is_truthy()? {
                                // One causal token prepares an eager runner
                                // whose policy has no prefill graph buckets.
                                let shape = py
                                    .import("uniserve_worker.model_executor.graph_inputs")?
                                    .getattr("PrefillShape")?
                                    .call1((1, 1, 1))?;
                                shapes = PyTuple::new(py, [shape])?.into_any();
                            }
                            backend.call_method1(
                                "prepare_prefill",
                                (
                                    owner,
                                    runner,
                                    runner.getattr("input_buffers")?,
                                    runner.getattr("batch_forward")?,
                                    shapes,
                                ),
                            )?;
                        }
                        CallKind::Forward(ForwardMode::Decode) => {
                            backend.call_method1(
                                "prepare_decode",
                                (
                                    owner,
                                    runner,
                                    runner.getattr("input_buffers")?,
                                    runner.getattr("batch_forward")?,
                                ),
                            )?;
                        }
                        CallKind::Forward(ForwardMode::TokenDenoising) => {
                            backend.call_method1("prepare_canvas", (owner, runner))?;
                        }
                        CallKind::Media(MediaCall::Denoising) => {
                            backend.call_method1(
                                "prepare_flow",
                                (owner, runner, latents, tokenizer),
                            )?;
                        }
                        _ => unreachable!("fixed startup call kinds"),
                    }
                }
            }
            backend.call_method1("prepare_images", (owner, latents))?;
            if let Some(control) = control
                && owner.getattr("attention_ranks")?.is_truthy()?
            {
                control.borrow().warmup(py, 0)?;
            }
        }
        experts::capture(slf)?;
        resources::synchronize(slf, owner)
    })
}

pub(super) fn seal(slf: &Bound<'_, ModelRunners>, owner: &Bound<'_, PyAny>) -> PyResult<()> {
    owner.getattr("graph_storage")?.call_method0("seal")?;
    slf.borrow_mut().sealed = true;
    for runner in ModelRunners::all(slf, owner)? {
        execution(&runner)?.borrow_mut().sealed = true;
    }
    for expert in &slf.borrow().expert_executions {
        expert.borrow_mut(slf.py()).sealed = true;
    }
    Ok(())
}
