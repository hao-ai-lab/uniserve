//! Numerical bindings and workspace lifetime shared by library and worker calls.

mod activation;
mod attention;
mod preparation;
mod workspace;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet, PyString, PyTuple, PyType};

use super::execution::close_all;
use super::tensor_buffers::{Scratch, TensorBuffers};
use activation::Activation;

/// Bind ordinary numerical modules to independent operator plans and backing.
/// Callers retire captured graphs and readers before preparation or closure.
/// Streams, cache, experts, weights and an explicitly supplied scratch owner
/// are borrowed; the context releases only its own bindings and allocations.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct ExecutionContext {
    state: Option<State>,
    entered: Option<Py<Activation>>,
}

struct State {
    module: Py<PyAny>,
    stream: Py<PyAny>,
    cache: Py<PyAny>,
    experts: Py<PyAny>,
    weights: Py<PyAny>,
    groups: Py<PyAny>,
    device: Py<PyAny>,
    dtype: Py<PyAny>,
    attention_backend: Py<PyAny>,
    vsa_backend: Py<PyAny>,
    matmul_backend: Py<PyAny>,
    moe_backend: Py<PyAny>,
    derive_host_lengths: bool,
    constants: Py<PyAny>,
    workspace: Py<PyAny>,
    allocations: Vec<Py<TensorBuffers>>,
    scratch: Py<Scratch>,
    owns_scratch: bool,
    operators: Py<PyDict>,
    merged: Py<PyDict>,
    attention: Py<PyDict>,
    vsa: Py<PyDict>,
    vsa_operators: Py<PyDict>,
    moe: Py<PyDict>,
    vsa_output: Py<PyDict>,
    vsa_context: Py<PyDict>,
    context_backing: Py<PyDict>,
    vsa_transport: Py<PyDict>,
    exchange: Py<PyDict>,
    chunks: Py<PyDict>,
    gather_pools: Py<PyDict>,
    transfers: Py<PyAny>,
    graph_streams: Py<PySet>,
    max_tokens: Option<usize>,
}

#[pymethods]
impl ExecutionContext {
    #[new]
    #[pyo3(signature = (module, *, stream=None, cache=None,
        attention=auto_backend(),
        vsa=auto_backend(),
        matmul=auto_backend(),
        moe=auto_backend(),
        groups=None, scratch=None, derive_host_lengths=true, experts=None, weights=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        module: Py<PyAny>,
        stream: Option<Py<PyAny>>,
        cache: Option<Py<PyAny>>,
        attention: Py<PyAny>,
        vsa: Py<PyAny>,
        matmul: Py<PyAny>,
        moe: Py<PyAny>,
        groups: Option<Py<PyAny>>,
        scratch: Option<Py<Scratch>>,
        derive_host_lengths: bool,
        experts: Option<Py<PyAny>>,
        weights: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let stream = stream.unwrap_or_else(|| py.None());
        let cache = cache.unwrap_or_else(|| py.None());
        let groups = match groups {
            Some(groups) => py.get_type::<PyTuple>().call1((groups,))?.unbind(),
            None => py
                .import("uniserve.distributed")?
                .call_method1("communicators", (&module,))?
                .unbind(),
        };
        let (device, dtype): (Py<PyAny>, Py<PyAny>) = py
            .import("uniserve.runtime.execution")?
            .call_method1("_representation", (&module, &cache, &stream))?
            .extract()?;
        if !stream.is_none(py) && !stream.bind(py).getattr("device")?.eq(device.bind(py))? {
            return Err(PyValueError::new_err(
                "the execution stream must belong to the root module device",
            ));
        }

        let owns_scratch = scratch.is_none();
        let scratch = match scratch {
            Some(scratch) => scratch,
            None => Py::new(py, Scratch::new(py))?,
        };
        let transfers = py
            .import("uniserve.runtime._transfers")?
            .call_method1("_Transfers", (&module, &device))?
            .unbind();
        let empty = empty_views(py)?;
        Ok(Self {
            state: Some(State {
                module,
                stream,
                cache,
                experts: experts.unwrap_or_else(|| py.None()),
                weights: weights.unwrap_or_else(|| py.None()),
                groups,
                device,
                dtype,
                attention_backend: attention,
                vsa_backend: vsa,
                matmul_backend: matmul,
                moe_backend: moe,
                derive_host_lengths,
                constants: empty.clone_ref(py),
                workspace: empty,
                allocations: Vec::new(),
                scratch,
                owns_scratch,
                operators: PyDict::new(py).unbind(),
                merged: PyDict::new(py).unbind(),
                attention: PyDict::new(py).unbind(),
                vsa: PyDict::new(py).unbind(),
                vsa_operators: PyDict::new(py).unbind(),
                moe: PyDict::new(py).unbind(),
                vsa_output: PyDict::new(py).unbind(),
                vsa_context: PyDict::new(py).unbind(),
                context_backing: PyDict::new(py).unbind(),
                vsa_transport: PyDict::new(py).unbind(),
                exchange: PyDict::new(py).unbind(),
                chunks: PyDict::new(py).unbind(),
                gather_pools: PyDict::new(py).unbind(),
                transfers,
                graph_streams: PySet::empty(py)?.unbind(),
                max_tokens: None,
            }),
            entered: None,
        })
    }

    #[classmethod]
    fn __class_getitem__<'py>(
        cls: &Bound<'py, PyType>,
        item: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        cls.py()
            .import("types")?
            .call_method1("GenericAlias", (cls, item))
    }

    #[getter]
    fn module(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.state.as_ref().map(|state| state.module.clone_ref(py))
    }

    #[getter]
    pub(super) fn stream(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.state.as_ref().map(|state| state.stream.clone_ref(py))
    }

    #[getter]
    fn cache(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.state.as_ref().map(|state| state.cache.clone_ref(py))
    }

    #[getter]
    fn experts(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.state.as_ref().map(|state| state.experts.clone_ref(py))
    }

    #[getter]
    fn weights(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.state.as_ref().map(|state| state.weights.clone_ref(py))
    }

    #[getter]
    fn constants(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.state.as_ref().map_or_else(
            || empty_views(py),
            |state| Ok(state.constants.clone_ref(py)),
        )
    }

    #[getter]
    fn workspace(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.state.as_ref().map_or_else(
            || empty_views(py),
            |state| Ok(state.workspace.clone_ref(py)),
        )
    }

    #[getter(_device)]
    pub(super) fn device(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.open()?.device.clone_ref(py))
    }

    /// Bind only layers reading this batch's cache tables. Missing host lengths
    /// are derived once when enabled; serving supplies those mirrors itself.
    /// Replay checks one reader per table if launches consume device metadata.
    #[pyo3(signature = (batch, *, replay=false))]
    fn bind_attention(
        slf: &Bound<'_, Self>,
        batch: &Bound<'_, PyAny>,
        replay: bool,
    ) -> PyResult<()> {
        attention::bind(slf, batch, replay)
    }

    /// Complete the open expert step at layers skipped by this rank's forward.
    fn join_expert_layers(slf: &Bound<'_, Self>) -> PyResult<()> {
        attention::join(slf)
    }

    /// Report selected kernels by module path after numerical warmup. Closed
    /// contexts have no prepared call sites and return an empty report.
    fn kernels(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        preparation::kernels(slf)
    }

    /// Borrow one role's transient work area on this serialized stream.
    fn scratch(
        &self,
        py: Python<'_>,
        role: &Bound<'_, PyAny>,
        requirements: &Bound<'_, PyAny>,
        device: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.open()?
            .scratch
            .borrow(py)
            .view(py, role, requirements, device)
    }

    fn _matmul_workspace(
        &self,
        py: Python<'_>,
        requirements: &Bound<'_, PyAny>,
        device: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.scratch(
            py,
            &PyString::new(py, "matmul").into_any(),
            requirements,
            device,
        )
    }

    fn _moe_workspace(
        &self,
        py: Python<'_>,
        requirements: &Bound<'_, PyAny>,
        device: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.scratch(
            py,
            &PyString::new(py, "moe").into_any(),
            requirements,
            device,
        )
    }

    fn _vsa_buffers(
        &self,
        py: Python<'_>,
        slot: usize,
        requirements: &Bound<'_, PyAny>,
        device: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.scratch(
            py,
            &("vsa", slot).into_pyobject(py)?.into_any(),
            requirements,
            device,
        )
    }

    fn _attention_workspace(
        slf: &Bound<'_, Self>,
        requirements: &Bound<'_, PyAny>,
        device: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        workspace::attention(slf, requirements, device)
    }

    fn _vsa_exchange(
        slf: &Bound<'_, Self>,
        layer: &Bound<'_, PyAny>,
        rows: usize,
        heads: usize,
        head_dim: usize,
        dtype: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        workspace::vsa_exchange(slf, layer, rows, heads, head_dim, dtype)
    }

    fn _context_transport(
        slf: &Bound<'_, Self>,
        layer: &Bound<'_, PyAny>,
        rows: usize,
        dtype: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        workspace::context_transport(slf, layer, rows, dtype)
    }

    /// Replace capacity after all previous readers and graphs have retired.
    /// Supplied buffers remain borrowed and are never closed by this owner.
    #[pyo3(signature = (size, *, constants=None, workspace=None))]
    fn prepare(
        slf: &Bound<'_, Self>,
        size: &Bound<'_, PyAny>,
        constants: Option<&Bound<'_, TensorBuffers>>,
        workspace: Option<&Bound<'_, TensorBuffers>>,
    ) -> PyResult<()> {
        slf.borrow().open()?;
        release(slf)?;
        if let Err(error) = preparation::prepare(slf, size, constants, workspace) {
            if let Err(cleanup) = release(slf) {
                let _ = error.value(slf.py()).call_method1(
                    "add_note",
                    (format!("execution preparation cleanup failed: {cleanup}"),),
                );
            }
            return Err(error);
        }
        Ok(())
    }

    /// Enter independent numerical bindings, transfers and the borrowed stream.
    fn activate(slf: &Bound<'_, Self>) -> Activation {
        Activation::new(slf.clone().unbind())
    }

    /// Release owned resources after their readers finish. An aborted close
    /// retains all backing without waiting for a possibly failed peer.
    #[pyo3(signature = (*, aborted=false))]
    #[allow(clippy::mem_forget)] // Outstanding CUDA readers keep their backing.
    fn close(slf: &Bound<'_, Self>, aborted: bool) -> PyResult<()> {
        if slf.borrow().state.is_none() {
            return Ok(());
        }
        if aborted {
            std::mem::forget(slf.borrow_mut().state.take());
            return Ok(());
        }

        let released = release(slf);
        let state = slf.borrow_mut().state.take();
        let closed = match state {
            Some(state) => state.transfers.call_method0(slf.py(), "close").map(drop),
            None => Ok(()),
        };
        close_all(slf.py(), [released, closed])
    }

    fn __enter__(slf: &Bound<'_, Self>) -> PyResult<Py<Self>> {
        if slf.borrow().entered.is_some() {
            return Err(PyRuntimeError::new_err(
                "execution context ownership cannot be entered twice",
            ));
        }
        let scope = Py::new(slf.py(), Self::activate(slf))?;
        scope.borrow_mut(slf.py()).enter(slf.py())?;
        slf.borrow_mut().entered = Some(scope);
        Ok(slf.clone().unbind())
    }

    fn __exit__(
        slf: &Bound<'_, Self>,
        kind: &Bound<'_, PyAny>,
        error: &Bound<'_, PyAny>,
        traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let py = slf.py();
        let entered = slf.borrow_mut().entered.take();
        let exited = entered
            .ok_or_else(|| PyRuntimeError::new_err("execution context ownership was not entered"))
            .and_then(|scope| scope.borrow_mut(py).exit(py, kind, error, traceback));
        let aborted = !error.is_none() && !slf.borrow().idle(py)?;
        if let Err(cleanup) = Self::close(slf, aborted) {
            if error.is_none() {
                return Err(cleanup);
            }
            let _ = error.call_method1(
                "add_note",
                (format!("execution context cleanup failed: {cleanup}"),),
            );
        }
        exited
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.entered)?;
        if let Some(state) = &self.state {
            for value in [
                &state.module,
                &state.stream,
                &state.cache,
                &state.experts,
                &state.weights,
                &state.groups,
                &state.device,
                &state.dtype,
                &state.attention_backend,
                &state.vsa_backend,
                &state.matmul_backend,
                &state.moe_backend,
                &state.constants,
                &state.workspace,
                &state.transfers,
            ] {
                visit.call(value)?;
            }
            for values in state.bindings() {
                visit.call(values)?;
            }
            for allocation in &state.allocations {
                visit.call(allocation)?;
            }
            visit.call(&state.scratch)?;
            visit.call(&state.graph_streams)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.entered = None;
        self.state = None;
    }
}

impl ExecutionContext {
    pub(super) fn idle(&self, py: Python<'_>) -> PyResult<bool> {
        match &self.state {
            Some(state) => state.idle(py),
            None => Ok(true),
        }
    }

    pub(super) fn graph_streams(&self, py: Python<'_>) -> PyResult<Py<PySet>> {
        Ok(self.open()?.graph_streams.clone_ref(py))
    }

    /// Enter native ownership directly when both caller and context are Rust.
    pub(super) fn with_active<T>(
        owner: &Bound<'_, Self>,
        call: impl FnOnce() -> PyResult<T>,
    ) -> PyResult<T> {
        let py = owner.py();
        let mut scope = Self::activate(owner);
        scope.enter(py)?;
        let result = call();
        match &result {
            Ok(_) => scope.exit(
                py,
                py.None().bind(py),
                py.None().bind(py),
                py.None().bind(py),
            )?,
            Err(error) => scope.exit(
                py,
                error.get_type(py).as_any(),
                error.value(py).as_any(),
                error.traceback(py).into_pyobject(py)?.as_any(),
            )?,
        }
        result
    }

    fn open(&self) -> PyResult<&State> {
        self.state
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("execution context is closed"))
    }

    fn open_mut(&mut self) -> PyResult<&mut State> {
        self.state
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("execution context is closed"))
    }
}

impl State {
    fn bindings(&self) -> [&Py<PyDict>; 13] {
        [
            &self.operators,
            &self.merged,
            &self.attention,
            &self.vsa,
            &self.vsa_operators,
            &self.moe,
            &self.vsa_output,
            &self.vsa_context,
            &self.context_backing,
            &self.vsa_transport,
            &self.exchange,
            &self.chunks,
            &self.gather_pools,
        ]
    }

    fn idle(&self, py: Python<'_>) -> PyResult<bool> {
        if self.device.bind(py).getattr("type")?.extract::<String>()? != "cuda" {
            return Ok(true);
        }
        let mut streams = self
            .transfers
            .bind(py)
            .getattr("streams")?
            .cast::<PyDict>()?
            .values()
            .iter()
            .map(Bound::unbind)
            .collect::<Vec<_>>();
        streams.extend(self.graph_streams.bind(py).iter().map(Bound::unbind));
        let stream = self.stream.bind(py);
        if stream.is_none() {
            streams.push(
                py.import("torch.cuda")?
                    .call_method1("current_stream", (&self.device,))?
                    .unbind(),
            );
        } else {
            streams.push(stream.getattr("stream")?.unbind());
            for communicator in stream
                .getattr("communication")?
                .getattr("communicators")?
                .call_method0("values")?
                .try_iter()?
            {
                let transfer = communicator?.getattr("transfer_stream")?;
                if !transfer.is_none() {
                    streams.push(transfer.unbind());
                }
            }
        }
        for stream in streams {
            match stream
                .call_method0(py, "query")
                .and_then(|value| value.extract(py))
            {
                Ok(true) => {}
                Ok(false) => return Ok(false),
                Err(error) if error.is_instance_of::<PyRuntimeError>(py) => return Ok(false),
                Err(error) => return Err(error),
            }
        }
        Ok(true)
    }
}

fn release(owner: &Bound<'_, ExecutionContext>) -> PyResult<()> {
    let py = owner.py();
    // Detach release actions from the owner before invoking numerical cleanup;
    // callbacks may borrow the context while closing a prepared operator.
    let actions = {
        let context = owner.borrow();
        let state = context.open()?;
        let mut actions = Vec::new();
        for bindings in [&state.attention, &state.vsa] {
            for (_, binding) in bindings.bind(py).iter() {
                actions.push(binding.getattr("close")?.unbind());
            }
        }
        for (_, pair) in state.vsa_operators.bind(py).iter() {
            actions.push(pair.get_item(1)?.getattr("close")?.unbind());
        }
        for (_, binding) in state.moe.bind(py).iter() {
            actions.push(binding.getattr("close")?.unbind());
        }
        for allocation in state.allocations.iter().rev() {
            actions.push(allocation.getattr(py, "close")?);
        }
        if state.owns_scratch {
            actions.push(state.scratch.getattr(py, "close")?);
        }
        actions.push(state.transfers.getattr(py, "reset")?);
        actions
    };
    let released = close_all(
        py,
        actions.into_iter().map(|action| action.call0(py).map(drop)),
    );
    let empty = empty_views(py)?;
    let mut context = owner.borrow_mut();
    let state = context.open_mut()?;
    for bindings in state.bindings() {
        bindings.bind(py).clear();
    }
    state.allocations.clear();
    state.constants = empty.clone_ref(py);
    state.workspace = empty;
    state.max_tokens = None;
    released
}

fn empty_views(py: Python<'_>) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("types")?
        .call_method1("MappingProxyType", (PyDict::new(py),))?
        .unbind())
}

fn auto_backend() -> Py<PyAny> {
    Python::attach(|py| PyString::new(py, "auto").into_any().unbind())
}
