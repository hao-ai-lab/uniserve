//! Reusable gather buffers for nested projection consumers on one stream.

use std::collections::BTreeSet;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct GatherPool {
    group: Py<PyAny>,
    buffers: Vec<Py<PyAny>>,
    borrowed: BTreeSet<usize>,
}

#[pymethods]
impl GatherPool {
    #[new]
    pub(super) fn new(group: Py<PyAny>) -> Self {
        Self {
            group,
            buffers: Vec::new(),
            borrowed: BTreeSet::new(),
        }
    }

    /// Borrow distinct storage until a projection consumer has enqueued its
    /// reads. Closed borrows return capacity; graphs retain every allocation.
    #[pyo3(signature = (size, device, *, capacity=None))]
    fn borrow(
        slf: &Bound<'_, Self>,
        size: usize,
        device: Py<PyAny>,
        capacity: Option<usize>,
    ) -> GatherBuffer {
        GatherBuffer {
            pool: slf.clone().unbind(),
            size,
            capacity,
            device,
            slot: None,
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.group)?;
        for buffer in &self.buffers {
            visit.call(buffer)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.buffers.clear();
    }
}

#[pyclass]
struct GatherBuffer {
    pool: Py<GatherPool>,
    size: usize,
    capacity: Option<usize>,
    device: Py<PyAny>,
    slot: Option<usize>,
}

#[pymethods]
impl GatherBuffer {
    fn __enter__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut pool = self.pool.borrow_mut(py);
        if !self
            .device
            .bind(py)
            .eq(pool.group.bind(py).getattr("device")?)?
        {
            return Err(PyValueError::new_err(
                "projection exchange must use its communicator's device",
            ));
        }
        if self.capacity.is_some_and(|capacity| self.size > capacity) {
            return Err(PyValueError::new_err(
                "projection exchange exceeds the bound workspace",
            ));
        }
        let amount = self.capacity.unwrap_or(self.size);
        let mut selected = None;
        for (index, buffer) in pool.buffers.iter().enumerate() {
            if !pool.borrowed.contains(&index)
                && buffer.call_method0(py, "numel")?.extract::<usize>(py)? >= amount
            {
                selected = Some(index);
                break;
            }
        }
        let slot = match selected {
            Some(slot) => slot,
            None => {
                if super::capturing(self.device.bind(py))? {
                    return Err(PyRuntimeError::new_err(
                        "prepare projection exchange backing before capture",
                    ));
                }
                // Ordinary allocations use NCCL transport buffers. Symmetric
                // registration is reserved for explicitly owned peer windows.
                let torch = py.import("torch")?;
                let options = PyDict::new(py);
                options.set_item("dtype", torch.getattr("uint8")?)?;
                options.set_item("device", &self.device)?;
                let buffer = torch.call_method("empty", (amount,), Some(&options))?;
                let slot = pool.buffers.len();
                pool.buffers.push(buffer.unbind());
                slot
            }
        };
        pool.borrowed.insert(slot);
        self.slot = Some(slot);
        Ok(pool.buffers[slot].clone_ref(py))
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        _kind: &Bound<'_, PyAny>,
        _error: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) {
        if let Some(slot) = self.slot.take() {
            self.pool.borrow_mut(py).borrowed.remove(&slot);
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.pool)?;
        visit.call(&self.device)
    }
}
