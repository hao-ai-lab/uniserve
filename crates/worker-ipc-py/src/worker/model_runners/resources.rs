//! Drain readers, retire graphs and contexts, then close shared streams.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::ModelRunners;
use super::streams::cuda_index;
use crate::worker::execution::close_all;

pub(super) fn synchronize(slf: &Bound<'_, ModelRunners>, owner: &Bound<'_, PyAny>) -> PyResult<()> {
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
        (owner.getattr("worker_config")?.getattr("device")?,),
    )?;
    let mut devices = vec![device.clone()];
    devices.extend(
        owner
            .call_method1("_capture_devices", (&device,))?
            .try_iter()?
            .collect::<PyResult<Vec<_>>>()?,
    );
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

pub(super) fn close_graphs(
    slf: &Bound<'_, ModelRunners>,
    owner: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = slf.py();
    let mut runners = owner
        .getattr("entries")?
        .call_method0("values")?
        .try_iter()?
        .collect::<PyResult<Vec<_>>>()?;
    runners.extend(slf.borrow().prepared(py)?.iter());
    let diffusion = owner.getattr("_diffusion")?;
    if !diffusion.is_none() {
        runners.push(diffusion);
    }
    let joins = owner.getattr("_expert_joins")?;
    owner.setattr("_expert_joins", py.None())?;
    let microbatches = owner.getattr("_microbatch_joins")?;
    owner.setattr("_microbatch_joins", PyList::empty(py))?;

    let mut results: Vec<_> = runners
        .iter()
        .map(|runner| runner.call_method0("close_graphs").map(drop))
        .collect();
    if !joins.is_none() {
        results.push(joins.call_method0("close").map(drop));
    }
    for joins in microbatches.try_iter()? {
        results.push(
            joins
                .and_then(|joins| joins.call_method0("close"))
                .map(drop),
        );
    }
    close_all(py, results)
}

pub(super) fn close(
    slf: &Bound<'_, ModelRunners>,
    owner: &Bound<'_, PyAny>,
    aborted: bool,
) -> PyResult<()> {
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
    if let Ok(draws) = owner.getattr("noise_draws")
        && !draws.is_none()
    {
        draws.call_method0("abort")?;
        if !aborted {
            draws.call_method0("close")?;
        }
    }

    let streams = slf.borrow().streams.owned(py);
    if aborted {
        py.import("uniserve.runtime.resources")?
            .call_method1("retain_until_exit", (owner,))?;
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

    let result = (|| {
        let mut results = vec![
            synchronize(slf, owner),
            close_graphs(slf, owner),
            ModelRunners::close_modules(slf),
        ];
        let diffusion = owner.getattr("_diffusion")?;
        if !diffusion.is_none() {
            results.push(diffusion.call_method0("close").map(drop));
        }
        for runner in owner
            .getattr("entries")?
            .call_method0("values")?
            .try_iter()?
        {
            results.push(
                runner
                    .and_then(|runner| runner.call_method0("close"))
                    .map(drop),
            );
        }
        let weights = owner.getattr("expert_weights")?;
        if !weights.is_none() {
            results.push(weights.call_method0("close").map(drop));
        }
        for field in ["_expert_executions", "_expert_exchanges"] {
            for resource in owner.getattr(field)?.try_iter()? {
                results.push(
                    resource
                        .and_then(|resource| resource.call_method0("close"))
                        .map(drop),
                );
            }
        }
        results.extend(
            streams
                .iter()
                .map(|stream| stream.bind(py).call_method0("close").map(drop)),
        );
        close_all(py, results)
    })();

    slf.borrow_mut().clear();
    owner.getattr("entries")?.call_method0("clear")?;
    owner.setattr("_diffusion", py.None())?;
    owner.setattr("_expert_exchanges", PyTuple::empty(py))?;
    owner.setattr("_expert_executions", PyTuple::empty(py))?;
    close_all(
        py,
        [
            result,
            owner
                .getattr("graph_storage")?
                .call_method0("close")
                .map(drop),
        ],
    )
}
