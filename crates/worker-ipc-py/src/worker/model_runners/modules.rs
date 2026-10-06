//! Bound numerical modules, prepared contexts and their resource lifetime.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker_ipc::CallKind;

use super::{ModelRunners, context};
use crate::worker::host::with_context;

/// A declared method and the contexts that borrow its stream and scratch.
/// Numerical sizes remain opaque keys at the Python interface. Startup
/// contexts stay resident; serving contexts are ordered by their last use.
pub(super) struct Module {
    pub(super) name: String,
    pub(super) path: String,
    pub(super) method: String,
    pub(super) kinds: HashSet<CallKind>,
    pub(super) encoder: Option<String>,
    pub(super) denoiser: bool,
    pub(super) video_decoder: bool,
    pub(super) binding: Py<PyAny>,
    pub(super) call: Py<PyAny>,
    pub(super) runner_type: Py<PyAny>,
    pub(super) resident: Py<PyDict>,
    pub(super) serving: Py<PyDict>,
    pub(super) stream: Mutex<Option<Py<PyAny>>>,
    pub(super) streams: Arc<super::streams::Streams>,
    pub(super) scratch: Mutex<Option<Py<PyAny>>>,
}

impl Module {
    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.binding)?;
        visit.call(&self.call)?;
        visit.call(&self.runner_type)?;
        visit.call(&self.resident)?;
        visit.call(&self.serving)?;
        if let Some(stream) = self
            .stream
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            visit.call(stream)?;
        }
        if let Some(scratch) = self
            .scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            visit.call(scratch)?;
        }
        Ok(())
    }

    pub(super) fn stream<'py>(&self, owner: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = owner.py();
        let device = self.binding.bind(py).getattr("device")?;
        if device.getattr("type")?.extract::<String>()? != "cuda" {
            return Ok(py.None().into_bound(py));
        }
        if let Some(stream) = self
            .stream
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            return Ok(stream.bind(py).clone());
        }
        let stream = self.streams.module(owner, &device, &self.kinds)?;
        *self
            .stream
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(stream.clone().unbind());
        Ok(stream)
    }

    pub(super) fn prepare<'py>(
        &self,
        owner: &Bound<'py, PyAny>,
        size: &Bound<'py, PyAny>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = owner.py();
        if let Some(runner) = self.resident.bind(py).get_item(key)? {
            return Ok(runner);
        }
        let serving = self.serving.bind(py);
        if let Some(runner) = serving.get_item(key)? {
            // Python's dictionary preserves insertion order. Move a hit to
            // the end without rebuilding keys or a second ownership table.
            serving.del_item(key)?;
            serving.set_item(key, &runner)?;
            return Ok(runner);
        }
        let capacity: usize = owner
            .getattr("worker_config")?
            .getattr("max_request_pool_size")?
            .extract()?;
        if serving.len() >= capacity
            && let Some((key, runner)) = serving.iter().next()
        {
            serving.del_item(key)?;
            retire(&runner)?;
        }

        let startup = !owner.getattr("_startup_complete")?.is_truthy()?;
        let call = self.call.bind(py);
        let device = self.binding.bind(py).getattr("device")?;
        let stream = self.stream(owner)?;
        let runtime = py.import("uniserve.runtime")?;
        let options = PyDict::new(py);
        options.set_item("attention", owner.getattr("attention")?)?;
        options.set_item("stream", &stream)?;
        options.set_item("groups", call.getattr("groups")?)?;
        let scratch = if startup {
            let existing = self
                .scratch
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_ref()
                .map(|value| value.clone_ref(py));
            match existing {
                Some(scratch) => scratch.into_bound(py),
                None => {
                    let scratch = runtime.getattr("Scratch")?.call0()?;
                    *self
                        .scratch
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) =
                        Some(scratch.clone().unbind());
                    scratch
                }
            }
        } else {
            py.None().into_bound(py)
        };
        options.set_item("scratch", scratch)?;
        options.set_item("derive_host_lengths", false)?;
        let context = runtime
            .getattr("ExecutionContext")?
            .call((call.getattr("module")?,), Some(&options))?;

        let storage = owner.getattr("graph_storage")?;
        let options = PyDict::new(py);
        options.set_item("storage", &storage)?;
        let captures = startup
            && !stream.is_none()
            && owner
                .getattr("worker_config")?
                .getattr("graph_policy")?
                .extract::<String>()?
                != "off"
            && !call
                .getattr("module")?
                .is_instance(&py.import("uniserve.model")?.getattr("VideoPostprocessor")?)?;
        options.set_item("devices", PyTuple::new(py, captures.then_some(&device))?)?;
        let share = self
            .resident
            .bind(py)
            .iter()
            .next()
            .map(|(_, runner)| runner);
        options.set_item("share", share)?;
        let runner = self.runner_type.bind(py).call(
            (
                &self.name,
                call,
                &device,
                PyTuple::empty(py),
                py.None(),
                &context,
            ),
            Some(&options),
        )?;
        let prepared = (|| -> PyResult<()> {
            if !stream.is_none() {
                let current = py
                    .import("torch.cuda")?
                    .call_method1("current_stream", (&device,))?;
                stream.call_method1("wait", (current,))?;
            }
            with_context(
                &storage.call_method1("allocate", (runner.getattr("execution")?,))?,
                || context.call_method1("prepare", (size,)).map(drop),
            )?;
            storage.call_method0("check")?;
            Ok(())
        })();
        if let Err(error) = prepared {
            runner.call_method0("close")?;
            return Err(error);
        }
        runner.setattr("_startup_complete", !startup)?;
        if startup {
            self.resident.bind(py)
        } else {
            serving
        }
        .set_item(key, &runner)?;
        Ok(runner)
    }

    pub(super) fn remove(&self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = key.py();
        for cache in [&self.resident, &self.serving] {
            if let Some(runner) = cache.bind(py).get_item(key)? {
                cache.bind(py).del_item(key)?;
                return retire(&runner);
            }
        }
        Ok(())
    }
}

fn retire(runner: &Bound<'_, PyAny>) -> PyResult<()> {
    let stream = context(runner)?.getattr("stream")?;
    if !stream.is_none() {
        stream.call_method0("synchronize")?;
    }
    runner.call_method0("close").map(drop)
}

pub(super) fn ensure_open(owner: &Bound<'_, PyAny>) -> PyResult<()> {
    if owner
        .getattr("batch_runners")?
        .cast_into::<ModelRunners>()?
        .borrow()
        .closed
    {
        return Err(PyRuntimeError::new_err("model runner is closed"));
    }
    Ok(())
}

impl ModelRunners {
    pub(super) fn module(
        &self,
        py: Python<'_>,
        name: &str,
        method: Option<&str>,
        path: Option<&str>,
    ) -> PyResult<Arc<Module>> {
        let mut matches = self.modules.values().filter(|module| {
            module.name == name
                && method.is_none_or(|method| module.method == method)
                && path.is_none_or(|path| module.path == path)
        });
        let first = matches.next();
        match (first, matches.next()) {
            (Some(module), None) => Ok(Arc::clone(module)),
            _ => Err(input_error(
                py,
                format!("component {name:?} requires an unambiguous numerical method"),
            )),
        }
    }
}

/// Numerical library callers distinguish invalid inputs from worker failures.
pub(super) fn input_error(py: Python<'_>, message: impl Into<String>) -> PyErr {
    match py
        .import("uniserve_worker.errors")
        .and_then(|module| module.getattr("InputError"))
        .and_then(|class| class.call1((message.into(),)))
    {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}
