//! Native startup planning over numerical model and storage descriptions.

pub(in crate::worker) mod inputs;
pub(in crate::worker) mod pools;
pub(in crate::worker) mod report;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker::capacity as native;

use super::error::unsupported;

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<report::WorkerLayout>()?;
    module.add_function(wrap_pyfunction!(report::build_worker_layout, module)?)?;
    module.add_class::<KVCapacity>()?;
    module.add_class::<ArenaCapacity>()?;
    module.add_function(wrap_pyfunction!(tensor_slot_capacity, module)?)?;
    module.add_function(wrap_pyfunction!(device_total_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(derive_runtime_kv_capacity, module)?)?;
    module.add_function(wrap_pyfunction!(check_startup_storage, module)?)?;
    module.add_function(wrap_pyfunction!(inputs::resolve_page_size, module)?)?;
    module.add_function(wrap_pyfunction!(inputs::input_buffer_config, module)?)?;
    module.add_function(wrap_pyfunction!(inputs::table_widths, module)?)?;
    module.add_function(wrap_pyfunction!(inputs::resident_width, module)?)?;
    module.add_function(wrap_pyfunction!(inputs::graph_table_widths, module)?)?;
    module.add_function(wrap_pyfunction!(
        pools::local_product_storage_bytes,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(pools::resolve_request_capacity, module)?)?;
    module.add_function(wrap_pyfunction!(pools::loaded_worker_config, module)?)?;
    Ok(())
}

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct KVCapacity {
    pub(super) inner: native::KVCapacity,
}

#[pymethods]
impl KVCapacity {
    #[getter]
    fn num_units(&self) -> u64 {
        self.inner.num_units
    }

    #[getter]
    fn unit_bytes(&self) -> u64 {
        self.inner.unit_bytes
    }
}

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct ArenaCapacity {
    pub(super) inner: native::ArenaCapacity,
}

#[pymethods]
impl ArenaCapacity {
    #[getter]
    fn latent_pool_bytes(&self) -> u64 {
        self.inner.latent_pool_bytes
    }

    #[getter]
    fn tensor_store(&self) -> usize {
        self.inner.tensor_store
    }

    #[getter]
    fn device_product_bytes(&self) -> u64 {
        self.inner.device_product_bytes
    }

    #[getter]
    fn transfer_bytes(&self) -> u64 {
        self.inner.transfer_bytes
    }

    #[getter]
    fn transfer_tickets(&self) -> usize {
        self.inner.transfer_tickets
    }

    #[getter]
    fn host_lane_inflight(&self) -> usize {
        self.inner.host_lane_inflight
    }
}

/// Numerical dtype and physical page dimensions used to allocate one latent pool.
pub(crate) struct LatentPoolPlan {
    pub(in crate::worker) request_pool_size: usize,
    pub(in crate::worker) num_pages: usize,
    pub(in crate::worker) page_units: usize,
    pub(in crate::worker) latent_width: usize,
    pub(in crate::worker) dtype: Py<PyAny>,
    pub(in crate::worker) with_workspace: bool,
}

impl LatentPoolPlan {
    pub(super) fn capacity_bytes(&self, py: Python<'_>) -> PyResult<u64> {
        native::latent_pool_bytes(
            self.request_pool_size,
            self.num_pages,
            self.page_units,
            self.latent_width,
            self.dtype.bind(py).getattr("itemsize")?.extract()?,
            self.with_workspace,
        )
        .map_err(value_error)
    }
}

fn value_error(error: uniserve_worker::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn capability<'py>(model: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    let py = model.py();
    py.import("uniserve_worker.bootstrap.inputs")?.call_method1(
        "capability",
        (model, py.import("uniserve.model")?.getattr(name)?),
    )
}

fn schema_bytes(schema: &Bound<'_, PyAny>, device_only: bool) -> PyResult<u64> {
    let mut bytes = 0;
    for field in schema.call_method0("values")?.try_iter()? {
        let field = field?;
        if !device_only || !field.getattr("host")?.is_truthy()? {
            bytes += field.getattr("nbytes")?.extract::<u64>()?;
        }
    }
    Ok(bytes)
}

fn devices<'py>(config: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let native = crate::worker::config::native(config)?;
    let devices = native
        .devices()
        .map(|device| Ok(device.into_pyobject(config.py())?.into_any()))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(devices)
}

fn is_cuda(device: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(device
        .py()
        .import("torch")?
        .getattr("device")?
        .call1((device,))?
        .getattr("type")?
        .extract::<String>()?
        == "cuda")
}

/// All ranks test the same candidates because a smaller request count can
/// increase the output horizon. Reducing local maxima would miss that case.
#[pyfunction]
#[pyo3(signature = (requirements, group, *, minimum, available_bytes))]
fn tensor_slot_capacity(
    requirements: Vec<u64>,
    group: &Bound<'_, PyAny>,
    minimum: usize,
    available_bytes: u64,
) -> PyResult<usize> {
    if minimum == 0 || requirements.is_empty() {
        return Err(PyValueError::new_err(
            "request tensor capacity requires valid slot bounds",
        ));
    }
    let available = available_bytes;
    let py = group.py();
    let torch = py.import("torch")?;
    let options = PyDict::new(py);
    options.set_item("dtype", torch.getattr("int32")?)?;
    options.set_item("device", group.getattr("device")?)?;
    let fits: Vec<_> = requirements
        .iter()
        .map(|&bytes| i32::from(bytes <= available))
        .collect();
    let tensor = torch.call_method("tensor", (fits,), Some(&options))?;
    let options = PyDict::new(py);
    options.set_item("op", "min")?;
    group.call_method("all_reduce", (&tensor,), Some(&options))?;
    let fits: Vec<i32> = tensor
        .call_method0("cpu")?
        .call_method0("tolist")?
        .extract()?;

    fits.iter().rposition(|&fits| fits != 0).map(|offset| minimum + offset).ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "insufficient device storage for a common request tensor slot count: candidate range {minimum}..{}, local requirements {requirements:?}, {available} bytes available",
            minimum + requirements.len() - 1,
        ))
    })
}

#[pyfunction]
pub(in crate::worker) fn device_total_bytes(device: &Bound<'_, PyAny>) -> PyResult<u64> {
    if !is_cuda(device)? {
        return Ok(0);
    }

    device
        .py()
        .import("torch.cuda")?
        .call_method1("mem_get_info", (device,))?
        .get_item(1)?
        .extract()
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (*, pages, kv_token_capacity, unit_bytes, device=None, available_bytes=None, floor=1, default_units=None, resident_copies=1, co_resident_units=0))]
fn derive_runtime_kv_capacity(
    pages: Vec<(u32, u32)>,
    kv_token_capacity: Option<u64>,
    unit_bytes: u64,
    device: Option<&Bound<'_, PyAny>>,
    available_bytes: Option<u64>,
    floor: u64,
    default_units: Option<u64>,
    resident_copies: u64,
    co_resident_units: u64,
) -> PyResult<KVCapacity> {
    if kv_token_capacity.is_none()
        && available_bytes.is_none()
        && device.map(is_cuda).transpose()?.unwrap_or(false)
    {
        return Err(PyValueError::new_err(
            "automatic CUDA KV sizing requires a host storage grant",
        ));
    }

    Ok(KVCapacity {
        inner: native::KVCapacity::derive(
            &pages,
            kv_token_capacity,
            unit_bytes,
            available_bytes,
            floor,
            default_units.unwrap_or(native::DEFAULT_NUM_UNITS),
            resident_copies,
            co_resident_units,
        )
        .map_err(value_error)?,
    })
}

#[pyfunction]
pub(in crate::worker) fn check_startup_storage(
    worker_config: &Bound<'_, PyAny>,
    product_capacity_bytes: u64,
    tensor_store: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let worker_config_native = crate::worker::config::native(worker_config)?;
    let py = worker_config.py();
    let device = py.import("uniserve.runtime.device")?;
    let fraction: f64 = worker_config_native.kv_storage_fraction;
    let devices = devices(worker_config)?;
    let product_bytes = product_capacity_bytes / devices.len() as u64;

    for target in devices {
        if !is_cuda(&target)? {
            continue;
        }
        let budget = device.call_method1("device_storage_budget", (&target, fraction))?;
        let available: u64 = budget.get_item(0)?.extract()?;
        let free: u64 = budget.get_item(1)?.extract()?;
        let grant = (device_total_bytes(&target)? as f64 * fraction) as u64;
        let resident = tensor_store
            .call_method1("resident_bytes", (&target,))?
            .extract()?;
        let remaining = product_bytes.saturating_sub(resident);
        let process: u64 = device
            .call_method1("process_device_bytes", (&target,))?
            .extract()?;

        if remaining > available || process + remaining > grant {
            return Err(unsupported(
                py,
                format!(
                    "initialized runtime on {target} exceeds its static storage grant: it requires {} bytes ({process} process-resident bytes and {remaining} reserved product bytes) of a {grant}-byte grant, with {free} device-free bytes",
                    process + remaining,
                ),
            ));
        }
    }
    Ok(())
}

/// All members allocate the same logical unit ids, bounded by the smallest grant.
fn minimum_capacity(group: Option<&Bound<'_, PyAny>>, units: u64) -> PyResult<u64> {
    let Some(group) = group else {
        return Ok(units);
    };
    if group.getattr("size")?.extract::<usize>()? <= 1 {
        return Ok(units);
    }
    let py = group.py();
    let torch = py.import("torch")?;
    let options = PyDict::new(py);
    options.set_item("dtype", torch.getattr("int64")?)?;
    options.set_item("device", group.getattr("device")?)?;
    let tensor = torch.call_method("tensor", (units,), Some(&options))?;
    let options = PyDict::new(py);
    options.set_item("op", "min")?;
    group.call_method("all_reduce", (&tensor,), Some(&options))?;
    tensor.call_method0("item")?.extract()
}
