//! Numerical runner bindings and batched dispatch.

mod dispatch;
mod execute;
mod forward;
mod graphs;
mod modules;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyTuple};
use uniserve_worker::ModelRunners as NativeModelRunners;
use uniserve_worker_ipc::{CallKind, MediaCall};

use crate::calls::Call;

use super::error::native_error;
use super::execution::Execution;
use super::host::with_context;
use super::model_results::ExecutionOutput;
use modules::{Module, ensure_open, input_error};

#[pyclass(module = "uniserve_worker._uniserve_ipc")]
#[derive(Default)]
pub(crate) struct ModelRunners {
    inner: NativeModelRunners<Py<PyAny>>,
    modules: IndexMap<(String, String, String), Arc<modules::Module>>,
    outputs: HashMap<String, Vec<String>>,
    // Creation order matters when closing communicators shared by ranks.
    streams: Arc<Mutex<Vec<Py<PyAny>>>>,
}

#[pymethods]
impl ModelRunners {
    /// Register one locally bound numerical method during model discovery.
    #[pyo3(signature = (name, binding, call, kinds, outputs, *, encoder=None))]
    #[allow(clippy::too_many_arguments)]
    fn register(
        &mut self,
        py: Python<'_>,
        name: String,
        binding: Py<PyAny>,
        call: Py<PyAny>,
        kinds: &Bound<'_, PyAny>,
        outputs: &Bound<'_, PyTuple>,
        encoder: Option<String>,
    ) -> PyResult<()> {
        let path: String = call.bind(py).getattr("path")?.extract()?;
        let method: String = call
            .bind(py)
            .getattr("entry_point")?
            .getattr("method")?
            .extract()?;
        let runner_type = py
            .import("uniserve_worker.model_executor.model_runner")?
            .call_method1("runner_type", (call.bind(py).getattr("module")?,))?
            .unbind();
        let kinds = kinds
            .try_iter()?
            .map(|kind| Ok(pythonize::depythonize(&kind?)?))
            .collect::<PyResult<_>>()?;
        let module = call.bind(py).getattr("module")?;
        let types = py.import("uniserve.model")?;
        let denoiser = module.is_instance(&types.getattr("Denoiser")?)?;
        let video_decoder = module.is_instance(&types.getattr("VideoDecoder")?)?;
        if !self.outputs.contains_key(&name) {
            let names = outputs
                .iter()
                .map(|output| output.getattr("name")?.extract())
                .collect::<PyResult<_>>()?;
            self.outputs.insert(name.clone(), names);
        }
        self.modules.insert(
            (name.clone(), path.clone(), method.clone()),
            Arc::new(Module {
                name,
                path,
                method,
                kinds,
                encoder,
                denoiser,
                video_decoder,
                binding,
                call,
                runner_type,
                resident: PyDict::new(py).unbind(),
                serving: PyDict::new(py).unbind(),
                stream: Mutex::new(None),
                scratch: Mutex::new(None),
                streams: Arc::clone(&self.streams),
            }),
        );
        Ok(())
    }

    /// Numerical declarations in binding order, for startup input construction.
    fn calls<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            self.modules
                .values()
                .map(|module| (&module.name, module.binding.bind(py), module.call.bind(py))),
        )
    }

    #[pyo3(signature = (kind, *, capability_type=None))]
    fn component(
        &self,
        py: Python<'_>,
        kind: &Bound<'_, PyAny>,
        capability_type: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let kind: CallKind = pythonize::depythonize(kind)?;
        let mut found = Vec::new();
        for module in self
            .modules
            .values()
            .filter(|module| module.kinds.contains(&kind))
        {
            let value = module.call.bind(py).getattr("module")?;
            if capability_type
                .map(|kind| value.is_instance(kind))
                .transpose()?
                .unwrap_or(true)
            {
                found.push(value);
            }
        }
        if found.len() != 1 {
            return Err(input_error(
                py,
                format!(
                    "computation {} requires one local capability",
                    kind.as_str()
                ),
            ));
        }
        Ok(found.remove(0).unbind())
    }

    fn encoder<'py>(&self, py: Python<'py>, kind: &str) -> PyResult<Bound<'py, PyTuple>> {
        let mut found = self
            .modules
            .values()
            .filter(|module| module.encoder.as_deref() == Some(kind));
        let first = found.next();
        match (first, found.next()) {
            (Some(module), None) => {
                Ok((module.name.as_str(), module.call.bind(py)).into_pyobject(py)?)
            }
            _ => Err(input_error(
                py,
                format!("rank does not participate in {kind} encoding"),
            )),
        }
    }

    #[getter]
    fn encoder_kinds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        PyFrozenSet::new(
            py,
            self.modules
                .values()
                .filter_map(|module| module.encoder.as_deref()),
        )
    }

    /// Select the module's execution stream from the native call and outputs.
    /// Buffered calls use their batch runner's stream instead.
    fn call_stream(
        slf: &Bound<'_, Self>,
        owner: &Bound<'_, PyAny>,
        call: &Call,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let call = &call.inner;
        let (mut modules, condition) = {
            let runners = slf.borrow();
            if runners.inner.get(&call.component, call.code).is_some()
                || (call.code == CallKind::Media(MediaCall::LatentPreparation)
                    && runners
                        .inner
                        .get(&call.component, CallKind::Media(MediaCall::Denoising))
                        .is_some())
            {
                return Ok(py.None());
            }
            let modules: Vec<_> = runners
                .modules
                .values()
                .filter(|module| module.name == call.component && module.kinds.contains(&call.code))
                .cloned()
                .collect();
            let names = runners.outputs.get(&call.component);
            let has = |name: &str| {
                call.outputs.iter().any(|output| {
                    names
                        .and_then(|names| names.get(output.output_index as usize))
                        .is_some_and(|value| value == name)
                })
            };
            let condition = if call.code != CallKind::Media(MediaCall::LatentEncoding) {
                None
            } else if has("condition_video_latents") {
                Some("video_condition")
            } else if has("condition_audio_latents") {
                Some("audio_condition")
            } else {
                None
            };
            (modules, condition)
        };
        if modules.is_empty() {
            return Ok(py.None());
        }
        match call.code {
            // Initialization and conditioning share a call, but the denoiser
            // owns its stream. Encoders keep their ordinary tensor dependencies.
            CallKind::Media(MediaCall::LatentPreparation)
                if modules.iter().any(|module| module.denoiser) =>
            {
                modules.retain(|module| module.denoiser);
            }
            CallKind::Media(MediaCall::VideoDecoding) => {
                modules.retain(|module| module.video_decoder)
            }
            CallKind::Media(MediaCall::LatentEncoding) if condition.is_some() => {
                modules.retain(|module| module.encoder.as_deref() == condition);
            }
            _ => {}
        }
        if modules.len() != 1 {
            return Err(input_error(
                py,
                "call requires one bound numerical capability",
            ));
        }
        ensure_open(owner)?;
        let stream = modules[0].stream(owner)?;
        Ok(if stream.is_none() {
            stream
        } else {
            stream.getattr("stream")?
        }
        .unbind())
    }

    #[pyo3(signature = (owner, name, size, *, method=None, path=None))]
    fn prepare_module(
        slf: &Bound<'_, Self>,
        owner: &Bound<'_, PyAny>,
        name: &str,
        size: &Bound<'_, PyAny>,
        method: Option<&str>,
        path: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        ensure_open(owner)?;
        let module = slf.borrow().module(slf.py(), name, method, path)?;
        let key = slf
            .py()
            .import("uniserve_worker.model_executor.cuda_graph")?
            .call_method1("input_signature", (size,))?;
        module.prepare(owner, size, &key).map(Bound::unbind)
    }

    #[pyo3(signature = (owner, name, *, method=None, path=None))]
    fn module_stream(
        slf: &Bound<'_, Self>,
        owner: &Bound<'_, PyAny>,
        name: &str,
        method: Option<&str>,
        path: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        ensure_open(owner)?;
        let module = slf.borrow().module(slf.py(), name, method, path)?;
        module.stream(owner).map(Bound::unbind)
    }

    /// Prepare and invoke one numerical method, retiring a failed graph's
    /// context before propagating its error. Results own their tensor storage.
    #[pyo3(signature = (owner, name, args, kwargs, *, size=None, method=None, path=None))]
    #[allow(clippy::too_many_arguments)]
    fn run_module(
        slf: &Bound<'_, Self>,
        owner: &Bound<'_, PyAny>,
        name: &str,
        args: &Bound<'_, PyTuple>,
        kwargs: &Bound<'_, PyDict>,
        size: Option<&Bound<'_, PyAny>>,
        method: Option<&str>,
        path: Option<&str>,
    ) -> PyResult<Py<ExecutionOutput>> {
        let py = slf.py();
        ensure_open(owner)?;
        let module = slf.borrow().module(py, name, method, path)?;
        let none = py.None().into_bound(py);
        let size = size.unwrap_or(&none);
        let key = py
            .import("uniserve_worker.model_executor.cuda_graph")?
            .call_method1("input_signature", (size,))?;
        with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
            let runner = module.prepare(owner, size, &key)?;
            let rank: usize = owner.getattr("worker_config")?.getattr("rank")?.extract()?;
            let result = with_context(
                &dispatch::profile(
                    py,
                    &format!("uniserve.model.module rank={rank} work={name}"),
                )?,
                || graphs::run_module(&runner, args, kwargs),
            );
            if let Err(error) = &result {
                let graph_error = py
                    .import("uniserve.runtime.cuda_graph")?
                    .getattr("CUDAGraphError")?;
                if error.value(py).is_instance(&graph_error)? {
                    module.remove(&key)?;
                }
            }
            let result = result?;
            owner.call_method0("_report_new_kernels")?;
            Ok(result)
        })
    }

    #[getter]
    fn prepared<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let values = self.modules.values().flat_map(|module| {
            module
                .resident
                .bind(py)
                .iter()
                .chain(module.serving.bind(py).iter())
                .map(|(_, runner)| runner)
        });
        PyTuple::new(py, values.collect::<Vec<_>>())
    }

    #[getter]
    fn module_streams<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let streams: Vec<_> = self
            .streams
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .map(|stream| stream.clone_ref(py))
            .collect();
        // Python allocation can run GC, whose visitor takes the same lock.
        PyTuple::new(py, streams)
    }

    /// The caller drains streams and output readers before releasing contexts.
    /// Every close is attempted; shared scratch outlives the contexts using it.
    fn close_modules(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let modules: Vec<_> = slf.borrow().modules.values().cloned().collect();
        let mut actions = Vec::new();
        for module in &modules {
            for cache in [&module.resident, &module.serving] {
                let cache = cache.bind(py);
                for (_, runner) in cache.iter() {
                    actions.push(runner.getattr("close")?);
                }
                cache.clear();
            }
        }
        for module in &modules {
            let scratch = module
                .scratch
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_ref()
                .map(|scratch| scratch.clone_ref(py));
            if let Some(scratch) = scratch {
                actions.push(scratch.bind(py).getattr("close")?);
            }
        }
        py.import("uniserve.runtime.resources")?
            .getattr("close_resources")?
            .call1(PyTuple::new(py, actions)?)?;
        Ok(())
    }

    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Evaluate a prepared numerical batch in its context, including experts.
    /// The caller orders input and output accesses with the runner's stream.
    fn run_eager(
        &self,
        py: Python<'_>,
        runner: &Bound<'_, PyAny>,
        batch: &Bound<'_, PyAny>,
        forward: &Bound<'_, PyAny>,
    ) -> PyResult<Py<ExecutionOutput>> {
        super::host::with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
            super::host::with_context(&context(runner)?.call_method0("activate")?, || {
                graphs::run_eager(runner, batch, forward)
            })
        })
    }

    /// Capture every expert-capacity variant of a selected startup bucket.
    /// Fails after startup is sealed; joins the caller's stream on every exit.
    fn capture(
        &self,
        py: Python<'_>,
        runner: &Bound<'_, PyAny>,
        batch: &Bound<'_, PyAny>,
        forward: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        super::host::with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
            graphs::capture(runner, batch, forward)
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (owner, tasks, *, cache, tables, states))]
    pub(super) fn forward(
        &self,
        py: Python<'_>,
        owner: Py<PyAny>,
        tasks: Py<PyTuple>,
        cache: Py<PyAny>,
        tables: Py<PyAny>,
        states: Py<PyAny>,
    ) -> PyResult<forward::ModelBatches> {
        forward::prepare(py, &self.inner, owner, tasks, cache, tables, states)
    }

    /// Execute prepared numerical rows with native stream and result ordering.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (owner, runner, rows, *, calls, cache, tables, states))]
    fn run_batch<'py>(
        &self,
        py: Python<'py>,
        owner: &Bound<'py, PyAny>,
        runner: &Bound<'py, PyAny>,
        rows: &Bound<'py, PyTuple>,
        calls: &Bound<'py, PyTuple>,
        cache: &Bound<'py, PyAny>,
        tables: &Bound<'py, PyAny>,
        states: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, ExecutionOutput>> {
        execute::run_batch(py, owner, runner, rows, calls, cache, tables, states, false)
    }

    fn clear(&mut self) {
        self.inner.clear();
        self.modules.clear();
        self.outputs.clear();
        let streams = std::mem::take(
            &mut *self
                .streams
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        drop(streams);
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for runner in self.inner.iter() {
            visit.call(runner)?;
        }
        for module in self.modules.values() {
            // An active numerical call holds its own native reference. Its
            // Python resources remain roots until that call returns.
            if Arc::strong_count(module) == 1 {
                module.traverse(&visit)?;
            }
        }
        for stream in self
            .streams
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
        {
            visit.call(stream)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.clear();
    }

    /// Bind whole-row numerical peers to their shared native execution group.
    fn bind_microbatches(&self, py: Python<'_>, peers: &Bound<'_, PyTuple>) -> PyResult<()> {
        let executions = peers
            .iter()
            .map(|peer| execution(&peer).map(Bound::unbind))
            .collect::<PyResult<Vec<_>>>()?;
        Execution::bind_microbatches(py, executions)?;
        for peer in peers {
            peer.setattr("peers", peers)?;
        }
        Ok(())
    }

    fn bind(
        &mut self,
        py: Python<'_>,
        component: &str,
        kinds: &Bound<'_, PyAny>,
        runner: Py<PyAny>,
    ) -> PyResult<()> {
        let kinds = kinds
            .try_iter()?
            .map(|kind| pythonize::depythonize(&kind?).map_err(Into::into))
            .collect::<PyResult<Vec<CallKind>>>()?;
        self.inner
            .bind(component, &kinds, runner)
            .map_err(|error| native_error(py, error))
    }

    fn get(
        &self,
        py: Python<'_>,
        component: &str,
        kind: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        Ok(self
            .inner
            .get(component, pythonize::depythonize(kind)?)
            .map(|runner| runner.clone_ref(py)))
    }

    fn first(&self, py: Python<'_>, kind: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
        Ok(self
            .inner
            .first(pythonize::depythonize(kind)?)
            .map(|runner| runner.clone_ref(py)))
    }
}

pub(super) fn execution<'py>(runner: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Execution>> {
    Ok(runner.getattr("execution")?.cast_into()?)
}

pub(super) fn context<'py>(runner: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    Ok(execution(runner)?
        .borrow()
        .context
        .bind(runner.py())
        .clone())
}
