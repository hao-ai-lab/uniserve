//! Tensor backing and compact views shared by numerical execution owners.

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PyTuple};

/// Own tensor references and symmetric mappings until their readers retire.
/// Numerical views borrow these allocations and can outlive this wrapper.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TensorBuffers {
    tensors: Py<PyDict>,
    views: Py<PyDict>,
    peers: Py<PyDict>,
    symmetric: Vec<Py<PyAny>>,
    closed: bool,
}

#[pymethods]
impl TensorBuffers {
    #[new]
    fn new(py: Python<'_>) -> Self {
        Self {
            tensors: PyDict::new(py).unbind(),
            views: PyDict::new(py).unbind(),
            peers: PyDict::new(py).unbind(),
            symmetric: Vec::new(),
            closed: false,
        }
    }

    /// Retain contiguous external tensors without changing their values.
    #[staticmethod]
    fn from_tensors(py: Python<'_>, tensors: &Bound<'_, PyAny>) -> PyResult<Self> {
        let tensors = mapping(tensors)?.copy()?;
        for (name, tensor) in tensors.iter() {
            if name.extract::<String>().is_err()
                || !name.is_truthy()?
                || !tensor.call_method0("is_contiguous")?.is_truthy()?
            {
                return Err(PyValueError::new_err(
                    "tensor backing requires nonempty names and contiguous storage",
                ));
            }
        }
        let mut buffers = Self::new(py);
        buffers.tensors = tensors.unbind();
        Ok(buffers)
    }

    /// Allocate named capacities; initialization belongs to numerical callers.
    #[staticmethod]
    #[pyo3(signature = (configs, *, device, pin_storage=false, symmetric=None))]
    pub(super) fn allocate(
        py: Python<'_>,
        configs: &Bound<'_, PyAny>,
        device: &Bound<'_, PyAny>,
        pin_storage: bool,
        symmetric: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let configs = mapping(configs)?;
        let symmetric = symmetric
            .map(mapping)
            .transpose()?
            .unwrap_or_else(|| PyDict::new(py));
        let torch = py.import("torch")?;
        let device = torch.call_method1("device", (device,))?;
        for (name, group) in symmetric.iter() {
            let config = configs.get_item(&name)?.ok_or_else(|| {
                PyValueError::new_err("symmetric allocations must name declared buffer fields")
            })?;
            if config.getattr("host")?.is_truthy()? || !group.getattr("device")?.eq(&device)? {
                return Err(PyValueError::new_err(
                    "symmetric storage requires the group's assigned device",
                ));
            }
        }

        let mut buffers = Self::new(py);
        for (name, config) in configs.iter() {
            let shape = config.getattr("capacity_shape")?;
            let shape = if shape.is_none() {
                config.getattr("shape")?
            } else {
                shape
            };
            let options = PyDict::new(py);
            options.set_item("dtype", config.getattr("dtype")?)?;
            let tensor = if let Some(group) = symmetric.get_item(&name)? {
                let allocation = py.import("uniserve.runtime._peer_storage")?.call_method(
                    "allocate_symmetric_storage",
                    (group, shape),
                    Some(&options),
                )?;
                let tensor = allocation.getattr("local")?;
                buffers
                    .peers
                    .bind(py)
                    .set_item(&name, allocation.getattr("peers")?)?;
                buffers.symmetric.push(allocation.unbind());
                tensor
            } else {
                let host = config.getattr("host")?.is_truthy()?;
                options.set_item(
                    "device",
                    if host {
                        "cpu".into_pyobject(py)?.into_any()
                    } else {
                        device.clone()
                    },
                )?;
                options.set_item("pin_memory", host && pin_storage)?;
                torch.call_method("empty", (shape,), Some(&options))?
            };
            buffers.tensors.bind(py).set_item(name, tensor)?;
        }
        Ok(buffers)
    }

    /// Borrow a field's entire capacity without changing its layout.
    fn backing(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        self.open()?;
        self.tensors
            .bind(py)
            .get_item(name)?
            .map(Bound::unbind)
            .ok_or_else(|| PyValueError::new_err(format!("tensor {name:?} has no backing")))
    }

    /// Borrow compact leading elements, not a strided rectangular crop.
    /// Cached views preserve both their tensor addresses and their values.
    pub(super) fn view(&self, py: Python<'_>, configs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.open()?;
        let configs = mapping(configs)?;
        let key = PyTuple::new(py, configs.iter())?;
        if let Some(views) = self.views.bind(py).get_item(&key)? {
            return Ok(views.unbind());
        }

        let views = PyDict::new(py);
        for (name, config) in configs.iter() {
            let tensor = self.tensors.bind(py).get_item(&name)?.ok_or_else(|| {
                PyValueError::new_err(format!(
                    "tensor {name} has no backing with the declared dtype"
                ))
            })?;
            if !tensor.getattr("dtype")?.eq(config.getattr("dtype")?)? {
                return Err(PyValueError::new_err(format!(
                    "tensor {name} has no backing with the declared dtype"
                )));
            }
            if config.getattr("host")?.is_truthy()?
                && tensor
                    .getattr("device")?
                    .getattr("type")?
                    .extract::<String>()?
                    != "cpu"
            {
                return Err(PyValueError::new_err(format!(
                    "tensor {name} requires host representation"
                )));
            }
            let shape: Vec<usize> = config.getattr("shape")?.extract()?;
            let capacity: Vec<usize> = tensor.getattr("shape")?.extract()?;
            if shape.len() != capacity.len()
                || shape
                    .iter()
                    .zip(&capacity)
                    .any(|(size, limit)| size > limit)
            {
                return Err(PyValueError::new_err(format!(
                    "tensor {name} exceeds resident capacity"
                )));
            }
            let elements = shape.iter().product::<usize>();
            let tensor = tensor
                .call_method1("reshape", (-1,))?
                .get_item(PySlice::new(py, 0, elements as isize, 1))?
                .call_method1("view", (PyTuple::new(py, &shape)?,))?;
            views.set_item(name, tensor)?;
        }
        let views = py
            .import("types")?
            .call_method1("MappingProxyType", (views,))?;
        self.views.bind(py).set_item(key, &views)?;
        Ok(views.unbind())
    }

    /// Peer views retain the allocation communicator's rank order.
    fn peers(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        self.open()?;
        Ok(self.peers.bind(py).as_any().get_item(name)?.unbind())
    }

    pub(super) fn close(&mut self, py: Python<'_>) {
        self.views.bind(py).clear();
        self.peers.bind(py).clear();
        self.tensors.bind(py).clear();
        self.symmetric.clear();
        self.closed = true;
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        slf.open()?;
        Ok(slf)
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        _kind: &Bound<'_, PyAny>,
        _error: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) {
        self.close(py);
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.tensors)?;
        visit.call(&self.views)?;
        visit.call(&self.peers)?;
        for owner in &self.symmetric {
            visit.call(owner)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        Python::attach(|py| self.close(py));
    }
}

impl TensorBuffers {
    fn open(&self) -> PyResult<()> {
        if self.closed {
            Err(PyRuntimeError::new_err("tensor buffers are closed"))
        } else {
            Ok(())
        }
    }
}

/// Share transient work areas between contexts serialized on one stream.
/// Larger backings remain available to earlier graphs when capacity grows.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct Scratch {
    backings: Py<PyDict>,
}

#[pymethods]
impl Scratch {
    #[new]
    pub(super) fn new(py: Python<'_>) -> Self {
        Self {
            backings: PyDict::new(py).unbind(),
        }
    }

    pub(super) fn view(
        &self,
        py: Python<'_>,
        role: &Bound<'_, PyAny>,
        requirements: &Bound<'_, PyAny>,
        device: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let configs = mapping(requirements)?;
        let schema = configs
            .iter()
            .map(|(name, config)| {
                Ok((
                    name,
                    config.getattr("dtype")?,
                    config.getattr("host")?,
                    config.getattr("shape")?.len()?,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let key = (role, device, PyTuple::new(py, schema)?);
        let backings = match self.backings.bind(py).get_item(&key)? {
            Some(backings) => backings.cast_into::<PyList>()?,
            None => {
                let backings = PyList::empty(py);
                self.backings.bind(py).set_item(&key, &backings)?;
                backings
            }
        };
        for backing in &backings {
            match backing.cast::<TensorBuffers>()?.borrow().view(py, &configs) {
                Ok(view) => return Ok(view),
                Err(error) if error.is_instance_of::<PyValueError>(py) => {}
                Err(error) => return Err(error),
            }
        }
        if py
            .import("uniserve.runtime.bindings")?
            .call_method1("capturing", (device,))?
            .is_truthy()?
        {
            return Err(PyRuntimeError::new_err(format!(
                "prepare {role:?} scratch for this size before capture"
            )));
        }

        let backing = Py::new(
            py,
            TensorBuffers::allocate(py, &configs, device, false, None)?,
        )?;
        let view = backing.borrow(py).view(py, &configs)?;
        backings.insert(0, backing)?;
        Ok(view)
    }

    pub(super) fn close(&self, py: Python<'_>) -> PyResult<()> {
        for (_, backings) in self.backings.bind(py).iter() {
            for backing in backings.cast::<PyList>()? {
                backing.cast::<TensorBuffers>()?.borrow_mut().close(py);
            }
        }
        self.backings.bind(py).clear();
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.backings)
    }

    fn __clear__(&mut self) {
        Python::attach(|py| self.backings.bind(py).clear());
    }
}

pub(super) fn mapping<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    match value.cast::<PyDict>() {
        Ok(value) => Ok(value.clone()),
        Err(_) => value
            .py()
            .get_type::<PyDict>()
            .call1((value,))?
            .cast_into()
            .map_err(Into::into),
    }
}
