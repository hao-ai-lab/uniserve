//! PyTorch allocator handles retained by native graph storage.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use uniserve_worker::{GraphPool, GraphStorage as NativeGraphStorage};

// PyTorch snapshot rows: device ordinal, allocator pool handle, reserved bytes.
type PoolSegment = (i32, (u64, u64), u64);

/// Shared byte budgets and pool ownership across numerical executions.
/// Allocation scopes use PyTorch's allocator; replay does no accounting.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct GraphStorage {
    inner: NativeGraphStorage<Py<PyAny>, Py<PyAny>>,
}

#[pymethods]
impl GraphStorage {
    #[new]
    #[pyo3(signature = (*, budgets=None))]
    fn new(py: Python<'_>, budgets: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut limits = BTreeMap::new();
        if let Some(budgets) = budgets {
            for (device, amount) in budgets.iter() {
                let device = cuda_index(py, &device)?.ok_or_else(|| {
                    PyValueError::new_err("graph storage budgets require CUDA devices")
                })?;
                limits.insert(device, nonnegative(amount.extract()?)?);
            }
        }

        Ok(Self {
            inner: NativeGraphStorage::new(limits),
        })
    }

    /// Reserve private pools, or borrow an existing owner's matching devices.
    /// Shared owners must serialize replay and allocate persistent inputs
    /// before their first capture. Non-CUDA devices have no graph pool.
    #[pyo3(signature = (owner, devices, *, share=None))]
    fn reserve<'py>(
        &mut self,
        py: Python<'py>,
        owner: &Bound<'py, PyAny>,
        devices: &Bound<'py, PyAny>,
        share: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        // Keeping the owner alive makes its address a stable local key.
        let key = owner.as_ptr() as usize;
        let shared = self
            .inner
            .prepare(key, share.map(|owner| owner.as_ptr() as usize))
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let result = PyDict::new(py);
        let mut seen = HashSet::new();
        let mut pools = Vec::new();
        let mut defaults = Vec::new();

        for device in devices.try_iter()? {
            let Some(index) = cuda_index(py, &device?)? else {
                continue;
            };
            if !seen.insert(index) {
                continue;
            }
            let device = torch_device(py, index)?;
            let pool = match shared.iter().find(|pool| pool.device == index) {
                Some(pool) => Arc::clone(pool),
                None => {
                    let (value, id, total): (Py<PyAny>, (u64, u64), i64) = backend(py)?
                        .call_method1("_new_pool", (&device,))?
                        .extract()?;
                    defaults.push((index, graph_storage_budget_bytes(total)));
                    Arc::new(GraphPool {
                        device: index,
                        id,
                        value,
                    })
                }
            };
            result.set_item(device, pool.value.bind(py))?;
            pools.push(pool);
        }

        // Allocation can fail on any device. Publish the owner only once all
        // pools and the returned numerical handles have been constructed.
        self.inner
            .commit(key, owner.clone().unbind(), pools, defaults);
        Ok(result)
    }

    /// Enter the owner's allocator pools for persistent preparation inputs.
    fn allocate(&self, py: Python<'_>, owner: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let pools = self
            .inner
            .pools(owner.as_ptr() as usize)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let handles = pools
            .iter()
            .map(|pool| Ok((torch_device(py, pool.device)?, pool.value.clone_ref(py))))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(backend(py)?
            .call_method1("_allocate_scope", (handles,))?
            .unbind())
    }

    /// Raise CUDAGraphError if any preparation or capture exceeds its budget.
    pub(super) fn check(&self, py: Python<'_>) -> PyResult<()> {
        if let Err(error) = self.inner.check(&self.residency(py)?) {
            let exception = py
                .import("uniserve.runtime.cuda_graph")?
                .getattr("CUDAGraphError")?
                .call1((error.to_string(),))?;
            return Err(PyErr::from_value(exception));
        }
        Ok(())
    }

    /// Bind a startup grant, including process growth outside allocator pools.
    fn set_budget(
        &mut self,
        py: Python<'_>,
        device: &Bound<'_, PyAny>,
        amount: i64,
    ) -> PyResult<()> {
        let amount = nonnegative(amount)?;
        let device = cuda_index(py, device)?
            .ok_or_else(|| PyValueError::new_err("graph storage budgets require CUDA devices"))?;
        let pooled = self.inner.pool_bytes(&self.segments(py)?);
        let process = backend(py)?
            .call_method1("_process_bytes", (device,))?
            .extract()?;
        self.inner.set_budget(
            device,
            amount,
            process,
            pooled.get(&device).copied().unwrap_or(0),
        );
        self.check(py)
    }

    /// After startup, charge only pools; serving growth uses the worker grant.
    pub(super) fn seal(&mut self) {
        self.inner.seal();
    }

    fn resident_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        for (device, used) in self.residency(py)? {
            result.set_item(torch_device(py, device)?, used)?;
        }
        Ok(result)
    }

    /// Reserved pool bytes include reusable graph workspace.
    fn pool_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        for (device, used) in self.inner.pool_bytes(&self.segments(py)?) {
            result.set_item(torch_device(py, device)?, used)?;
        }
        Ok(result)
    }

    /// Attribute a shared pool once, to its first remaining owner.
    fn owner_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        for ((owner, device), used) in self.inner.owner_bytes(&self.segments(py)?) {
            result.set_item(
                (self.inner.owner(owner).bind(py), torch_device(py, device)?),
                used,
            )?;
        }
        Ok(result)
    }

    /// Release an owner after its graphs and borrowed views have retired.
    fn release(mut slf: PyRefMut<'_, Self>, owner: &Bound<'_, PyAny>) {
        let retired = slf.inner.release(owner.as_ptr() as usize);
        drop(slf);
        drop(retired);
    }

    pub(super) fn close(mut slf: PyRefMut<'_, Self>) {
        let retired = slf.inner.close();
        // Finalizers may inspect storage or release another resource. They
        // run after native state is updated and its exclusive borrow ends.
        drop(slf);
        drop(retired);
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        let mut seen = HashSet::new();
        for owner in self.inner.owners() {
            visit.call(&owner.owner)?;
            for pool in &owner.pools {
                // Shared Arcs own one Python reference, not one per owner.
                if seen.insert(Arc::as_ptr(pool)) {
                    visit.call(&pool.value)?;
                }
            }
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.inner.close();
    }
}

impl GraphStorage {
    fn segments(&self, py: Python<'_>) -> PyResult<Vec<PoolSegment>> {
        if !self.inner.has_pools() {
            return Ok(Vec::new());
        }
        backend(py)?.call_method0("_snapshot")?.extract()
    }

    fn residency(&self, py: Python<'_>) -> PyResult<BTreeMap<i32, i128>> {
        let pools = if self.inner.needs_pool_sizes() {
            self.inner.pool_bytes(&self.segments(py)?)
        } else {
            BTreeMap::new()
        };
        let mut process = BTreeMap::new();
        for device in self.inner.bound_devices() {
            let bytes = backend(py)?
                .call_method1("_process_bytes", (device,))?
                .extract()?;
            process.insert(device, bytes);
        }
        Ok(self.inner.resident_bytes(&pools, |device| process[&device]))
    }
}

#[pyfunction]
pub(super) fn graph_storage_budget_bytes(total_device_bytes: i64) -> u64 {
    uniserve_worker::graph_storage_budget_bytes(total_device_bytes)
}

fn backend(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("uniserve_worker.model_executor.graph_storage")
}

fn torch_device(py: Python<'_>, index: i32) -> PyResult<Bound<'_, PyAny>> {
    py.import("torch")?
        .getattr("device")?
        .call1(("cuda", index))
}

fn cuda_index(py: Python<'_>, device: &Bound<'_, PyAny>) -> PyResult<Option<i32>> {
    let device = py
        .import("uniserve.runtime.device")?
        .call_method1("canonical_device", (device,))?;
    if device.getattr("type")?.extract::<String>()? == "cuda" {
        Ok(Some(device.getattr("index")?.extract()?))
    } else {
        Ok(None)
    }
}

fn nonnegative(amount: i64) -> PyResult<u64> {
    u64::try_from(amount)
        .map_err(|_| PyValueError::new_err("graph storage budgets must be nonnegative"))
}
