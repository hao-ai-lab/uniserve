//! Persistent tensor banks indexed by scheduler-assigned request slots.

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::error::invalid;
use super::tensor_buffers::{TensorBuffers, mapping};

/// Device fields share a leading slot axis for graph-time indexing. Host
/// fields have one allocation per slot and can be filled independently.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct RequestSlots {
    #[pyo3(get)]
    capacity: usize,
    #[pyo3(get)]
    pub(super) bank: Py<PyDict>,
    pub(super) tensor_slots: Vec<Py<TensorBuffers>>,
    host_slots: Vec<Py<TensorBuffers>>,
    backing: Option<Py<TensorBuffers>>,
    closed: bool,
}

#[pymethods]
impl RequestSlots {
    #[new]
    #[pyo3(signature = (capacity, *, state_buffers, device))]
    pub(super) fn new(
        py: Python<'_>,
        capacity: isize,
        state_buffers: Option<&Bound<'_, PyAny>>,
        device: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        if capacity < 1 {
            return Err(PyValueError::new_err(
                "request storage capacity must be positive",
            ));
        }
        let capacity = capacity as usize;
        let mut storage = Self {
            capacity,
            bank: PyDict::new(py).unbind(),
            tensor_slots: Vec::new(),
            host_slots: Vec::new(),
            backing: None,
            closed: false,
        };
        let Some(fields) = state_buffers else {
            return Ok(storage);
        };
        let fields = mapping(fields)?;
        if fields.is_empty() {
            return Ok(storage);
        }

        let device = py.import("torch")?.call_method1("device", (device,))?;
        let pin_storage = device.getattr("type")?.extract::<String>()? == "cuda";
        let config_type = py.import("uniserve.tensors")?.getattr("BufferConfig")?;
        let device_fields = PyDict::new(py);
        let host_fields = PyDict::new(py);
        for (name, config) in fields.iter() {
            if config.getattr("host")?.is_truthy()? {
                host_fields.set_item(name, config)?;
                continue;
            }

            let shape: Vec<usize> = config.getattr("shape")?.extract()?;
            let bounds: Option<Vec<usize>> = config.getattr("capacity_shape")?.extract()?;
            let bounds = bounds.as_ref().unwrap_or(&shape);
            let prepend_slot = |shape: &[usize]| {
                let mut dimensions = Vec::with_capacity(shape.len() + 1);
                dimensions.push(capacity);
                dimensions.extend_from_slice(shape);
                PyTuple::new(py, dimensions)
            };
            device_fields.set_item(
                name,
                config_type.call1((
                    prepend_slot(&shape)?,
                    config.getattr("dtype")?,
                    prepend_slot(bounds)?,
                ))?,
            )?;
        }

        let backing = TensorBuffers::allocate(py, &device_fields, &device, false, None)?;
        for name in device_fields.keys() {
            storage
                .bank
                .bind(py)
                .set_item(&name, backing.backing(py, name.extract()?)?)?;
        }
        for index in 0..capacity {
            let host = TensorBuffers::allocate(py, &host_fields, &device, pin_storage, None)?;
            let tensors = PyDict::new(py);
            for (name, bank) in storage.bank.bind(py).iter() {
                tensors.set_item(name, bank.get_item(index)?)?;
            }
            for name in host_fields.keys() {
                tensors.set_item(&name, host.backing(py, name.extract()?)?)?;
            }
            storage.host_slots.push(Py::new(py, host)?);
            storage
                .tensor_slots
                .push(Py::new(py, TensorBuffers::from_tensors(py, &tensors)?)?);
        }
        storage.backing = Some(Py::new(py, backing)?);
        Ok(storage)
    }

    #[getter]
    fn tensor_slots<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.tensor_slots)
    }

    /// Borrow one-based slot views until the request's execution lease retires.
    pub(super) fn tensors(
        &self,
        py: Python<'_>,
        request_pool_idx: isize,
    ) -> PyResult<Py<TensorBuffers>> {
        if self.closed {
            return Err(PyRuntimeError::new_err("request storage is closed"));
        }
        if request_pool_idx < 1 || request_pool_idx as usize > self.capacity {
            return Err(invalid(py, "request storage slot exceeds capacity"));
        }
        self.tensor_slots
            .get(request_pool_idx as usize - 1)
            .map(|slot| slot.clone_ref(py))
            .ok_or_else(|| invalid(py, "request has no declared persistent tensor storage"))
    }

    /// Release views before backing, after all execution consumers have drained.
    pub(super) fn close(&mut self, py: Python<'_>) {
        for slot in self.tensor_slots.drain(..).chain(self.host_slots.drain(..)) {
            slot.borrow_mut(py).close(py);
        }
        if let Some(backing) = self.backing.take() {
            backing.borrow_mut(py).close(py);
        }
        self.bank.bind(py).clear();
        self.closed = true;
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.bank)?;
        visit.call(&self.backing)?;
        for slot in self.tensor_slots.iter().chain(&self.host_slots) {
            visit.call(slot)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        Python::attach(|py| self.close(py));
    }
}
