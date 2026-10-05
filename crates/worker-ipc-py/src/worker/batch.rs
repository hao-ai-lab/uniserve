//! Batch-owned inputs, output rows and borrowed numerical views.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use uniserve_worker_ipc::{Batch, CallStatus};

use super::error::{invalid, native_error};
use super::inputs::BatchInputs;
use super::kv_import::KVImporter;
use super::latent::LatentPool;
use super::output::OutputBuffer;
use super::pending::PendingOutput;
use super::request::RequestPool;
use super::storage::TensorStore;

/// Resources retained until the executor delivers or abandons one submission.
/// Numerical callbacks borrow this object; they do not advance the executor.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct BatchState {
    plan: Arc<Batch>,
    request_indexes: HashMap<u64, usize>,
    pub(super) outputs: Vec<Py<PendingOutput>>,
    pub(super) buffer: Option<Py<OutputBuffer>>,
    predicate_values: Option<Py<PyDict>>,

    #[pyo3(get)]
    pub(super) batch: Py<PyAny>,
    #[pyo3(get)]
    pub(super) inputs: Py<BatchInputs>,

    // Each predicate capture names a call, an (offset, word count) span and
    // its output row. Deferred transfers fill the remaining rows before sealing.
    #[pyo3(get, set)]
    predicate_entries: Py<PyList>,
    #[pyo3(get, set)]
    predicate_transfers: Py<PyTuple>,

    #[pyo3(get, set)]
    input_products: Py<PyTuple>,
    #[pyo3(get, set)]
    kv_inputs: Py<PyTuple>,

    #[pyo3(get, set)]
    stream: Option<Py<PyAny>>,
    // Python perf_counter_ns at reservation, shared with numerical timers.
    #[pyo3(get)]
    pub(super) started_ns: u64,

    #[pyo3(get)]
    pub(super) forward_stats: Py<PyList>,
    #[pyo3(get)]
    pub(super) component_us: Py<PyDict>,
    #[pyo3(get)]
    pub(super) forward_indices: Py<PyDict>,

    #[pyo3(get, set)]
    pub(super) products: Py<PyTuple>,
}

impl BatchState {
    pub(super) fn new(py: Python<'_>, batch: Py<PyAny>, plan: Arc<Batch>) -> PyResult<Self> {
        let request_indexes = plan
            .calls
            .iter()
            .enumerate()
            .map(|(index, call)| (call.request_key.request_id.0, index))
            .collect();

        Ok(Self {
            plan,
            request_indexes,
            outputs: Vec::new(),
            buffer: None,
            predicate_values: None,
            batch,
            inputs: Py::new(py, BatchInputs::new())?,
            predicate_entries: PyList::empty(py).unbind(),
            predicate_transfers: PyTuple::empty(py).unbind(),
            input_products: PyTuple::empty(py).unbind(),
            kv_inputs: PyTuple::empty(py).unbind(),
            stream: None,
            started_ns: 0,
            forward_stats: PyList::empty(py).unbind(),
            component_us: PyDict::new(py).unbind(),
            forward_indices: PyDict::new(py).unbind(),
            products: PyTuple::empty(py).unbind(),
        })
    }

    pub(super) fn close(
        slf: &Bound<'_, Self>,
        tensors: &TensorStore,
        latents: Option<&Bound<'_, LatentPool>>,
        imports: Option<&Bound<'_, KVImporter>>,
    ) -> PyResult<()> {
        let py = slf.py();
        let (inputs, outputs) = {
            let this = slf.borrow();
            let outputs = this
                .outputs
                .iter()
                .map(|output| output.clone_ref(py))
                .collect::<Vec<_>>();
            (this.inputs.clone_ref(py), outputs)
        };

        // Completion observers can reenter numerical consumers. Release the
        // batch borrow before closing, and attempt every owner after failures.
        let mut failure = BatchInputs::close(inputs.bind(py), tensors, latents, imports).err();
        for output in outputs {
            if let Err(error) = PendingOutput::abandon(output.bind(py)) {
                if let Some(first) = &failure {
                    let _ = first.value(py).call_method1(
                        "add_note",
                        (format!("batch output cleanup failed: {error}"),),
                    );
                } else {
                    failure = Some(error);
                }
            }
        }
        failure.map_or(Ok(()), Err)
    }
}

#[pymethods]
impl BatchState {
    #[getter]
    fn batch_id(&self) -> u64 {
        self.plan.batch_id
    }

    #[getter]
    pub(super) fn route(&self) -> Option<&'static str> {
        self.plan.calls.first().map(|call| call.code.as_str())
    }

    /// Enter the numerical stream, or retain the caller's current stream.
    pub(super) fn scope(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let stream = slf
            .borrow()
            .stream
            .as_ref()
            .map(|stream| stream.clone_ref(py));
        match stream {
            Some(stream) => py.import("torch.cuda")?.call_method1("stream", (stream,)),
            None => py.import("contextlib")?.call_method0("nullcontext"),
        }
        .map(Bound::unbind)
    }

    #[getter]
    pub(super) fn output_buffer(&self, py: Python<'_>) -> PyResult<Py<OutputBuffer>> {
        self.buffer
            .as_ref()
            .map(|buffer| buffer.clone_ref(py))
            .ok_or_else(|| PyRuntimeError::new_err("batch has no reserved output buffer"))
    }

    /// Bind all output rows before exposing any to numerical execution.
    fn bind_outputs(
        slf: &Bound<'_, Self>,
        requests: &Bound<'_, RequestPool>,
        buffer: Py<OutputBuffer>,
        started_ns: u64,
        predicated: HashSet<u64>,
    ) -> PyResult<()> {
        let py = slf.py();
        let (plan, batch) = {
            let this = slf.borrow();
            (Arc::clone(&this.plan), this.batch.clone_ref(py))
        };
        let bindings = {
            let requests = requests.borrow();
            let rows = plan
                .calls
                .iter()
                .map(|call| {
                    let request = requests
                        .pool
                        .get(call.request_key.request_id.0)
                        .map_err(|error| native_error(py, error))?;
                    Ok((call.request_key, call.call_id, request.slot()))
                })
                .collect::<PyResult<Vec<_>>>()?;
            requests
                .pool
                .bind_calls(&rows)
                .map_err(|error| native_error(py, error))?
                .iter()
                .map(|request| requests.get(py, request.key().request_id.0))
                .collect::<PyResult<Vec<_>>>()?
        };

        let calls = batch.bind(py).getattr("calls")?;
        let mut outputs = Vec::with_capacity(plan.calls.len());
        for (index, (call, request)) in plan.calls.iter().zip(bindings).enumerate() {
            let mut output = PendingOutput::for_call(
                py,
                calls.get_item(index)?.unbind(),
                call,
                request,
                buffer.clone_ref(py),
                index,
            )?;
            if predicated.contains(&call.request_key.request_id.0) {
                output.status = CallStatus::Predicated;
            }
            outputs.push(Py::new(py, output)?);
        }

        let mut this = slf.borrow_mut();
        this.outputs = outputs;
        this.buffer = Some(buffer);
        this.started_ns = started_ns;
        Ok(())
    }

    /// Borrow the complete set of output rows in scheduler call order.
    fn pending_outputs(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        if self.outputs.len() != self.plan.calls.len() {
            return Err(PyRuntimeError::new_err(
                "batch has no reserved pending outputs",
            ));
        }
        PyTuple::new(py, &self.outputs).map(Bound::unbind)
    }

    fn pending_output(&self, py: Python<'_>, request_id: u64) -> PyResult<Py<PendingOutput>> {
        let index = self
            .request_indexes
            .get(&request_id)
            .ok_or_else(|| invalid(py, format!("batch has no request {request_id}")))?;
        self.outputs
            .get(*index)
            .map(|output| output.clone_ref(py))
            .ok_or_else(|| PyRuntimeError::new_err("request has no reserved pending output"))
    }

    /// Read completed U8 predicates once. Device-gated predicates remain tensors.
    fn predicate_values(slf: &Bound<'_, Self>) -> PyResult<Py<PyDict>> {
        let py = slf.py();
        let (inputs, entries) = {
            let this = slf.borrow();
            if let Some(values) = &this.predicate_values {
                return Ok(values.clone_ref(py));
            }
            (
                this.inputs.clone_ref(py),
                this.predicate_entries.clone_ref(py),
            )
        };
        let Some(buffer) = inputs.borrow(py).predicate(py) else {
            return Ok(PyDict::new(py).unbind());
        };
        if !buffer.get().ready(py)? {
            return Err(PyRuntimeError::new_err(
                "prepared predicates were observed before readiness",
            ));
        }

        let values = PyDict::new(py);
        let read = (|| -> PyResult<()> {
            let mut entries: Vec<(Py<PyAny>, (usize, usize), usize)> = entries.extract(py)?;
            entries.sort_by_key(|entry| entry.2);
            for (identity, (offset, count), row) in entries {
                let value = buffer.get().with_readback(py, |buffer| {
                    match buffer.read_tokens(offset, count)? {
                        [0] => Ok(false),
                        [1] => Ok(true),
                        _ => Err(uniserve_worker::Error::Invalid(
                            "call predicate is not a canonical boolean".into(),
                        )),
                    }
                })?;
                values.set_item(identity, value)?;
                buffer.get().observe(py, row)?;
            }
            Ok(())
        })();
        if let Err(error) = read {
            OutputBuffer::abandon(buffer.bind(py))?;
            return Err(error);
        }

        slf.borrow_mut().predicate_values = Some(values.clone().unbind());
        inputs.borrow_mut(py).set_predicate(None);
        Ok(values.unbind())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.batch)?;
        visit.call(&self.inputs)?;
        visit.call(&self.buffer)?;
        for output in &self.outputs {
            visit.call(output)?;
        }
        visit.call(&self.predicate_entries)?;
        visit.call(&self.predicate_transfers)?;
        visit.call(&self.predicate_values)?;
        visit.call(&self.input_products)?;
        visit.call(&self.kv_inputs)?;
        visit.call(&self.stream)?;
        visit.call(&self.forward_stats)?;
        visit.call(&self.component_us)?;
        visit.call(&self.forward_indices)?;
        visit.call(&self.products)
    }

    fn __clear__(&mut self, py: Python<'_>) {
        self.batch = py.None();
        self.outputs.clear();
        self.buffer = None;
        self.predicate_values = None;
        self.stream = None;
        self.predicate_entries = PyList::empty(py).unbind();
        self.predicate_transfers = PyTuple::empty(py).unbind();
        self.input_products = PyTuple::empty(py).unbind();
        self.kv_inputs = PyTuple::empty(py).unbind();
        self.forward_stats = PyList::empty(py).unbind();
        self.component_us = PyDict::new(py).unbind();
        self.forward_indices = PyDict::new(py).unbind();
        self.products = PyTuple::empty(py).unbind();
    }
}
