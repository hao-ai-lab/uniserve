//! Fixed host collective buffers for native expert-step coordination.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pyo3::buffer::PyBuffer;
use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFrozenSet, PyTuple};
use uniserve_worker::{Error, ExpertExchange as NativeExchange};

struct Collectives {
    local: PyBuffer<i64>,
    records: PyBuffer<i64>,
    gather: Py<PyAny>,
    prepare: Option<Py<PyAny>>,
    broadcast: Option<Py<PyAny>>,
}

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct ExpertExchange {
    inner: Mutex<NativeExchange>,
    collectives: Mutex<Option<Arc<Collectives>>>,
    coordinating: AtomicBool,
}

#[pymethods]
#[allow(clippy::expect_used)] // Buffer exports keep their validated shape until close.
impl ExpertExchange {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (ranks, rank, memberships, *, attention_ranks, max_tokens, fused, local, records, gather, prepare, broadcast))]
    fn new(
        py: Python<'_>,
        ranks: Vec<usize>,
        rank: usize,
        memberships: Vec<Vec<usize>>,
        attention_ranks: usize,
        max_tokens: usize,
        fused: bool,
        local: &Bound<'_, PyAny>,
        records: &Bound<'_, PyAny>,
        gather: Py<PyAny>,
        prepare: Option<Py<PyAny>>,
        broadcast: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let inner = NativeExchange::new(
            &ranks,
            rank,
            &memberships,
            attention_ranks,
            max_tokens,
            fused,
            prepare.is_some(),
        )
        .map_err(error)?;

        // Exporting these CPU arrays retains their allocations and prevents
        // resizing. All-gather completes before Rust reads the received rows.
        let local = PyBuffer::<i64>::get(local)?;
        let records = PyBuffer::<i64>::get(records)?;
        if local.as_mut_slice(py).is_none()
            || local.item_count() != 3
            || records.as_slice(py).is_none()
            || records.item_count() != ranks.len() * 3
        {
            return Err(PyValueError::new_err(
                "expert coordination needs contiguous int64 host rows of three values",
            ));
        }

        Ok(Self {
            inner: Mutex::new(inner),
            collectives: Mutex::new(Some(Arc::new(Collectives {
                local,
                records,
                gather,
                prepare,
                broadcast,
            }))),
            coordinating: AtomicBool::new(false),
        })
    }

    #[pyo3(signature = (tokens, *, kind = 0, leaving = false))]
    fn agree(&self, py: Python<'_>, tokens: usize, kind: i64, leaving: bool) -> PyResult<usize> {
        let tokens = count(tokens)?;
        self.collective(|buffers| {
            let local = buffers.local.as_mut_slice(py).expect("fixed host row");
            for (field, value) in local.iter().zip([tokens, i64::from(leaving), kind]) {
                field.set(value);
            }

            buffers.gather.call0(py)?;

            let records = buffers.records.as_slice(py).expect("fixed host records");
            self.state()
                .select(|rank| {
                    let row = &records[rank * 3..rank * 3 + 3];
                    (row[0].get() as usize, row[1].get() != 0, row[2].get())
                })
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))
        })
    }

    fn warmup(&self, py: Python<'_>, capacity: usize) -> PyResult<usize> {
        self.collective(|buffers| {
            let Some(broadcast) = &buffers.broadcast else {
                return Ok(capacity);
            };
            let local = buffers.local.as_mut_slice(py).expect("fixed host row");
            local[0].set(count(capacity)?);
            broadcast.call0(py)?;
            Ok(local[0].get() as usize)
        })
    }

    fn begin(&self, capacity: usize) -> PyResult<()> {
        self.state().begin(capacity).map_err(error)
    }

    fn end(&self) {
        self.state().end();
    }

    fn enter(&self, py: Python<'_>, module: usize) -> PyResult<()> {
        let prepare = self.state().needs_warmup().map_err(error)?;
        if prepare {
            self.collective(|buffers| {
                if let Some(prepare) = &buffers.prepare {
                    prepare.call0(py)?;
                }
                Ok(())
            })?;
        }

        self.state().enter(module).map_err(error)
    }

    fn pending_layers(&self, modules: Vec<usize>) -> PyResult<Vec<usize>> {
        self.state().pending_layers(modules).map_err(error)
    }

    fn reset_layers(&self) {
        self.state().reset_layers();
    }

    fn record_layers(&self, modules: &Bound<'_, PyAny>) -> PyResult<()> {
        let modules = modules
            .try_iter()?
            .map(|module| module?.extract::<usize>())
            .collect::<PyResult<Vec<_>>>()?;
        self.state().record_layers(modules);
        Ok(())
    }

    #[getter]
    fn invoked<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        let modules = self.state().invoked().collect::<Vec<_>>();
        PyFrozenSet::new(py, modules)
    }

    #[getter]
    fn capacities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let capacities = self.state().capacities().to_vec();
        PyTuple::new(py, capacities)
    }

    #[getter]
    fn capacity(&self) -> usize {
        self.state().capacity()
    }

    #[getter]
    fn kind(&self) -> i64 {
        self.state().kind()
    }

    #[getter]
    fn active(&self) -> bool {
        self.state().active()
    }

    #[getter]
    fn released(&self) -> bool {
        self.state().released()
    }

    fn close(&self) -> PyResult<()> {
        if self.coordinating.load(Ordering::Acquire) {
            return Err(PyRuntimeError::new_err(
                "an expert collective is still running",
            ));
        }

        let retired = self
            .collectives
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        drop(retired);
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        let collectives = self
            .collectives
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(buffers) = &*collectives {
            // tp_traverse runs attached. Inspect existing owner pointers
            // only: neither obj() nor visit.call() calls Python code.
            let py = unsafe { Python::assume_attached() };
            if let Some(owner) = buffers.local.obj(py) {
                visit.call(owner.as_unbound())?;
            }
            if let Some(owner) = buffers.records.obj(py) {
                visit.call(owner.as_unbound())?;
            }
            visit.call(&buffers.gather)?;
            visit.call(&buffers.prepare)?;
            visit.call(&buffers.broadcast)?;
        }
        Ok(())
    }

    fn __clear__(&self, py: Python<'_>) {
        if let Err(error) = self.close() {
            error.write_unraisable(py, None);
        }
    }
}

impl ExpertExchange {
    fn state(&self) -> MutexGuard<'_, NativeExchange> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn collective<R>(&self, run: impl FnOnce(&Collectives) -> PyResult<R>) -> PyResult<R> {
        let buffers = self
            .collectives
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("expert exchange is closed"))?;
        if self.coordinating.swap(true, Ordering::AcqRel) {
            return Err(PyRuntimeError::new_err(
                "an expert collective is already running",
            ));
        }

        // The backend can release the GIL while Gloo writes these buffers.
        // Retain them and reject another writer, without holding owner locks
        // or a mutable Python borrow across the collective callback.
        let result = run(&buffers);
        self.coordinating.store(false, Ordering::Release);
        result
    }
}

fn count(value: usize) -> PyResult<i64> {
    i64::try_from(value).map_err(|_| PyOverflowError::new_err("expert token count exceeds int64"))
}

fn error(error: Error) -> PyErr {
    match error {
        Error::Invalid(message) => PyValueError::new_err(message),
        error => PyRuntimeError::new_err(error.to_string()),
    }
}
