//! Drain readers, retire graphs and contexts, then close shared streams.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

use super::ModelExecutor;
use super::streams::cuda_index;
use crate::worker::execution::close_all;

pub(super) fn synchronize(slf: &Bound<'_, ModelExecutor>) -> PyResult<()> {
    let py = slf.py();
    let streams = slf.borrow().streams.owned(py);
    let mut results: Vec<_> = streams
        .iter()
        .map(|stream| stream.bind(py).call_method0("synchronize").map(drop))
        .collect();

    // Numerical initialization may use each device's current stream outside
    // a runner. Explicit shutdown drains it before freeing model backing.
    let device = py.import("uniserve.runtime.device")?.call_method1(
        "canonical_device",
        (&Arc::clone(&slf.borrow().config).device,),
    )?;
    let mut devices = vec![device.clone()];
    devices.extend(super::streams::capture_devices(slf, &device)?);
    let cuda = py.import("torch.cuda")?;
    for device in devices {
        if cuda_index(&device)?.is_some() {
            results.push(
                cuda.call_method1("current_stream", (device,))?
                    .call_method0("synchronize")
                    .map(drop),
            );
        }
    }
    close_all(py, results)
}

pub(super) fn close_graphs(slf: &Bound<'_, ModelExecutor>) -> PyResult<()> {
    let py = slf.py();
    let mut results: Vec<_> = ModelExecutor::all(slf)?
        .iter()
        .map(|runner| runner.call_method0("close_graphs").map(drop))
        .collect();
    let experts: Vec<_> = slf
        .borrow()
        .expert_executions
        .iter()
        .map(|execution| execution.clone_ref(py))
        .collect();
    results.extend(
        experts
            .iter()
            .map(|execution| execution.call_method0(py, "close_graphs").map(drop)),
    );
    close_all(py, results)
}

pub(super) fn close(slf: &Bound<'_, ModelExecutor>, aborted: bool) -> PyResult<()> {
    let py = slf.py();
    {
        let mut runners = slf.borrow_mut();
        if runners.closed {
            return Ok(());
        }
        runners.closed = true;
    }

    // A host draw writes request storage. Stop admission before draining it;
    // an aborted worker keeps its backing until immediate process exit.
    let draws = slf
        .borrow()
        .noise_draws
        .as_ref()
        .map(|draws| draws.clone_ref(py));
    if let Some(draws) = draws {
        draws.get().abort();
        if !aborted {
            draws.get().close(py)?;
        }
    }

    let streams = slf.borrow().streams.owned(py);
    if aborted {
        py.import("uniserve.runtime.resources")?
            .call_method1("retain_until_exit", (slf,))?;
        let options = PyDict::new(py);
        options.set_item("aborted", true)?;
        return close_all(
            py,
            streams.iter().map(|stream| {
                stream
                    .bind(py)
                    .call_method("close", (), Some(&options))
                    .map(drop)
            }),
        );
    }

    let result = {
        let mut results = vec![
            synchronize(slf),
            close_graphs(slf),
            ModelExecutor::close_modules(slf),
        ];
        let diffusion = slf
            .borrow()
            .diffusion
            .as_ref()
            .map(|value| value.clone_ref(py));
        if let Some(diffusion) = diffusion {
            results.push(diffusion.call_method0(py, "close").map(drop));
        }
        let runners = slf
            .borrow()
            .inner
            .iter()
            .map(|runner| runner.clone_ref(py))
            .collect::<Vec<_>>();
        results.extend(
            runners
                .iter()
                .map(|runner| runner.call_method0(py, "close").map(drop)),
        );
        let weights = slf.borrow().expert_weights.bind(slf.py()).clone();
        if !weights.is_none() {
            results.push(weights.call_method0("close").map(drop));
        }
        let experts = {
            let runners = slf.borrow();
            runners
                .expert_executions
                .iter()
                .map(|execution| execution.clone_ref(py).into_any())
                .chain(
                    runners
                        .exchanges
                        .iter()
                        .map(|exchange| exchange.clone_ref(py)),
                )
                .collect::<Vec<_>>()
        };
        results.extend(
            experts
                .iter()
                .map(|expert| expert.call_method0(py, "close").map(drop)),
        );
        results.extend(
            streams
                .iter()
                .map(|stream| stream.bind(py).call_method0("close").map(drop)),
        );
        close_all(py, results)
    };

    slf.borrow_mut().clear();
    let storage = slf.borrow().graph_storage(py)?;
    crate::worker::graph_storage::GraphStorage::close(storage.borrow_mut());
    result
}
