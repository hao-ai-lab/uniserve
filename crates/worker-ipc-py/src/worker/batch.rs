//! Batch-owned inputs, output rows and borrowed numerical views.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use uniserve_worker_ipc::{Batch, CallStatus, DType};

use super::block_tables::BlockTables;
use super::completion::CompletionRef;
use super::error::{invalid, native_error};
use super::inputs::BatchInputs;
use super::kv_cache::KVCacheManager;
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
    stream: Option<Py<PyAny>>,
    // Python perf_counter_ns at reservation, shared with numerical timers.
    #[pyo3(get)]
    pub(super) started_ns: u64,

    #[pyo3(get)]
    pub(super) forward_stats: Py<PyList>,
    #[pyo3(get)]
    pub(super) component_us: Py<PyDict>,
    forward_indices: Vec<Vec<usize>>,

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

        let mut forward_indices = vec![Vec::new(); plan.calls.len()];
        for (row, &call) in plan.forward.call_indices.iter().enumerate() {
            forward_indices[call as usize].push(row);
        }

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
            stream: None,
            started_ns: 0,
            forward_stats: PyList::empty(py).unbind(),
            component_us: PyDict::new(py).unbind(),
            forward_indices,
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

    /// Return numerical forward rows in their scheduler-supplied order.
    fn forward_rows<'py>(&self, py: Python<'py>, request_id: u64) -> PyResult<Bound<'py, PyTuple>> {
        let rows = self
            .request_indexes
            .get(&request_id)
            .map_or(&[][..], |&index| self.forward_indices[index].as_slice());
        PyTuple::new(py, rows)
    }

    /// Install active calls' KV assignments and retain their physical access.
    /// Callbacks only copy numerical tables and reset recycled cache units.
    fn bind_cache(
        &self,
        py: Python<'_>,
        cache: &Bound<'_, KVCacheManager>,
        tables: &Bound<'_, BlockTables>,
        copy: &Bound<'_, PyAny>,
        recycle: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let started = Instant::now();
        let forward = &self.plan.forward;
        let mut active = Vec::new();
        let mut slots = HashSet::new();
        for (index, output) in self.outputs.iter().enumerate() {
            let output = output.borrow(py);
            let pending = output.lock(py)?;
            if pending.output.status == CallStatus::Predicated {
                continue;
            }

            let slot = output.request.borrow(py).request.slot() as u32;
            active.push((index, slot, pending.progress.kv_visible_len));
            slots.insert(slot);
            slots.extend(
                self.forward_indices[index]
                    .iter()
                    .map(|&row| forward.request_pool_indices[row]),
            );
        }

        // Alternative-prefix rows need their own assignments even though they
        // share a call's progress. Numerical copying precedes host-table commit.
        let assignments = self
            .plan
            .block_tables
            .iter()
            .filter(|table| slots.contains(&table.request_pool_idx))
            .cloned()
            .collect::<Vec<_>>();
        tables.borrow_mut().install(py, &assignments, copy)?;

        // Import copies reset their own destinations. All other new units are
        // recycled together before this batch can submit model writes.
        let mut initialized = HashSet::new();
        for write in self.inputs.borrow(py).cache_imports(py) {
            let write = write.get();
            initialized.extend(
                write
                    .initialized_units
                    .iter()
                    .map(|&unit| (write.request_pool_idx() as u32, unit)),
            );
        }

        let mut recycled = Vec::new();
        let mut seen = HashSet::new();
        for allocation in &self.plan.new_cache_units {
            if !slots.contains(&allocation.request_pool_idx) {
                continue;
            }
            for unit in &allocation.unit_ids {
                if initialized.contains(&(allocation.request_pool_idx, unit.0)) {
                    continue;
                }
                if !seen.insert(unit.0) {
                    return Err(invalid(py, "KV allocation repeats a physical unit"));
                }
                recycled.push(unit.0);
            }
        }

        if !recycled.is_empty() {
            let unit_tokens = tables
                .borrow()
                .tables
                .groups()
                .iter()
                .map(|group| group.page_tokens)
                .max()
                .unwrap_or(0);
            let spans = recycled
                .iter()
                .map(|&unit| (unit, 0, unit_tokens))
                .collect::<Vec<_>>();
            cache
                .borrow_mut()
                .inner
                .require_reusable(&spans)
                .map_err(|error| native_error(py, error))?;
            recycle.call1((PyTuple::new(py, recycled)?,))?;
        }

        let completion = self.output_buffer(py)?.borrow(py).completion(py)?;
        let mut tables = tables.borrow_mut();
        let tables = &mut tables.tables;
        let mut cache = cache.borrow_mut();
        for (index, main_slot, visible) in active {
            let call = &self.plan.calls[index];
            let rows = &self.forward_indices[index];
            if let Some(&row) = rows
                .iter()
                .find(|&&row| forward.request_pool_indices[row] == main_slot)
            {
                let declared = u64::from(forward.seq_lens[row] - forward.query_lens[row]);
                let relayed = call
                    .predicate
                    .as_ref()
                    .is_some_and(|value| value.dtype == DType::I64);
                // A queued token relay carries a capacity bound. Its device
                // predicate determines the actual visible length at execution.
                if declared < visible || (!relayed && declared != visible) {
                    return Err(invalid(
                        py,
                        "forward row sequence length disagrees with execution state",
                    ));
                }
            }

            let mut spans = Vec::new();
            for &row in rows {
                let slot = forward.request_pool_indices[row];
                if slot != main_slot {
                    if forward.seq_lens[row] - forward.query_lens[row]
                        > tables.allocated_length(slot)
                    {
                        return Err(invalid(
                            py,
                            "forward row exceeds alternative-prefix capacity",
                        ));
                    }
                    tables.retain_prefix(call.request_key, slot);
                }

                // Read-only rows retain the prefix; writers also retain query
                // tokens. A windowed group starts at its first resident page.
                let length = forward.seq_lens[row]
                    - if forward.write_kv[row] {
                        0
                    } else {
                        forward.query_lens[row]
                    };
                for group in 0..tables.groups().len() as u32 {
                    let table = tables
                        .table(slot, group)
                        .map_err(|error| native_error(py, error))?;
                    let start = u64::from(table.start_page) * u64::from(table.shape.page_tokens);
                    if u64::from(length) > start {
                        spans.extend(
                            table
                                .spans(start, u64::from(length) - start)
                                .map_err(|error| native_error(py, error))?,
                        );
                    }
                }
            }
            cache.inner.retain_execution(
                call.request_key,
                &spans,
                CompletionRef::new(py, completion.clone_ref(py)),
            );
        }

        let components = self.component_us.bind(py);
        let previous = components
            .get_item("bc_tables")?
            .map_or(Ok(0), |value| value.extract::<u64>())?;
        components.set_item("bc_tables", previous + started.elapsed().as_micros() as u64)?;
        Ok(())
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
            let output = PendingOutput::for_call(
                py,
                calls.get_item(index)?.unbind(),
                call,
                request,
                buffer.clone_ref(py),
                index,
            )?;
            if predicated.contains(&call.request_key.request_id.0) {
                output.lock(py)?.output.status = CallStatus::Predicated;
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
        visit.call(&self.stream)?;
        visit.call(&self.forward_stats)?;
        visit.call(&self.component_us)?;
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
        self.forward_stats = PyList::empty(py).unbind();
        self.component_us = PyDict::new(py).unbind();
        self.products = PyTuple::empty(py).unbind();
    }
}
