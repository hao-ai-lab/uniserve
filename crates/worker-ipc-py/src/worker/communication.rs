//! Stream-owned communicators, registered windows and projection transport.

mod gather;
mod nccl;

pub(super) use gather::GatherPool;
pub(super) use nccl::NcclCommunicator;

use std::sync::{Arc, Mutex};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use uniserve_worker::CUDAStream as NativeStream;

use super::execution::close_all;

/// Groups, windows and their backing outlive every context borrowing this
/// stream. Binding and normal retirement follow the same rank order.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct StreamCommunication {
    stream: Py<PyAny>,
    owner: Arc<Mutex<NativeStream>>,
    communicators: Py<PyDict>,
    pools: Py<PyDict>,
    windows: Py<PyDict>,
    closed: bool,
}

impl StreamCommunication {
    pub(super) fn new(py: Python<'_>, stream: Py<PyAny>, owner: Arc<Mutex<NativeStream>>) -> Self {
        Self {
            stream,
            owner,
            communicators: PyDict::new(py).unbind(),
            pools: PyDict::new(py).unbind(),
            windows: PyDict::new(py).unbind(),
            closed: false,
        }
    }

    fn open(&self) -> PyResult<()> {
        if self.closed {
            Err(PyRuntimeError::new_err("stream communication is closed"))
        } else {
            Ok(())
        }
    }

    pub(super) fn has_communicators(&self, py: Python<'_>) -> bool {
        !self.communicators.bind(py).is_empty()
    }
}

#[pymethods]
impl StreamCommunication {
    #[getter]
    fn communicators(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // A live view also exposes groups bound after a numerical scope began.
        Ok(py
            .import("types")?
            .call_method1("MappingProxyType", (&self.communicators,))?
            .unbind())
    }

    fn bind(slf: &Bound<'_, Self>, groups: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let (stream, owner, bindings) = {
            let this = slf.borrow();
            this.open()?;
            (
                this.stream.clone_ref(py),
                Arc::clone(&this.owner),
                this.communicators.clone_ref(py),
            )
        };
        let created = PyDict::new(py);
        let result: PyResult<()> = (|| {
            let dist = py.import("torch.distributed")?;
            for communicator in groups.try_iter()? {
                let communicator = communicator?;
                if communicator.getattr("size")?.extract::<usize>()? == 1 {
                    continue;
                }
                let group = communicator.call_method0("_require")?;
                let name = group.getattr("group_name")?;
                if bindings.bind(py).contains(&name)? || created.contains(&name)? {
                    continue;
                }
                let backends: String = dist
                    .call_method1("get_backend_config", (&group,))?
                    .extract()?;
                if backends.split(',').any(|backend| backend == "cuda:nccl") {
                    let binding = Py::new(
                        py,
                        NcclCommunicator::new(py, &group, stream.bind(py), &owner)?,
                    )?;
                    created.set_item(name, binding)?;
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            for binding in created.values().iter().rev() {
                if let Err(cleanup) = binding.cast::<NcclCommunicator>()?.get().close(py) {
                    let _ = error.value(py).call_method1(
                        "add_note",
                        (format!("collective binding cleanup failed: {cleanup}"),),
                    );
                }
            }
            return Err(error);
        }
        bindings.bind(py).update(created.as_mapping())
    }

    fn gather_pool(&self, py: Python<'_>, group: Py<PyAny>) -> PyResult<Py<GatherPool>> {
        self.open()?;
        if let Some(pool) = self.pools.bind(py).get_item(&group)? {
            return Ok(pool.extract()?);
        }
        let pool = Py::new(py, GatherPool::new(group.clone_ref(py)))?;
        self.pools.bind(py).set_item(group, &pool)?;
        Ok(pool)
    }

    /// Allocate and register once per numerical transport layout. The stream
    /// retains window backing across context replacement and graph reuse.
    fn windows(
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        group: &Bound<'_, PyAny>,
        allocate: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let (stream, bindings, windows) = {
            let this = slf.borrow();
            this.open()?;
            (
                this.stream.clone_ref(py),
                this.communicators.clone_ref(py),
                this.windows.clone_ref(py),
            )
        };
        if let Some(retained) = windows.bind(py).get_item(key)? {
            return Ok(retained.get_item(0)?.unbind());
        }
        if capturing(&stream.bind(py).getattr("device")?)? {
            return Err(PyRuntimeError::new_err(
                "prepare registered communication storage before capture",
            ));
        }
        let name = group.call_method0("_require")?.getattr("group_name")?;
        let communicator = bindings.bind(py).get_item(name)?.ok_or_else(|| {
            PyRuntimeError::new_err("bind a group's communicator before registering windows")
        })?;
        let (value, allocations, tensors): (Py<PyAny>, Py<PyAny>, Vec<Bound<'_, PyAny>>) =
            allocate.call0()?.extract()?;
        if let Err(error) = communicator
            .cast::<NcclCommunicator>()?
            .get()
            .register_buffers(py, tensors)
        {
            let releases = allocations
                .bind(py)
                .try_iter()?
                .map(|allocation| allocation?.call_method0("close").map(drop));
            close_all(py, std::iter::once(Err(error)).chain(releases))?;
        }
        windows.bind(py).set_item(key, (&value, allocations))?;
        Ok(value)
    }

    /// Normal close is collective and follows graph retirement. Aborted close
    /// never waits for peers and keeps all window allocations until exit.
    #[pyo3(signature = (*, aborted=false))]
    #[allow(clippy::mem_forget)]
    pub(super) fn close(slf: &Bound<'_, Self>, aborted: bool) -> PyResult<()> {
        let py = slf.py();
        let (bindings, pools, windows) = {
            let mut this = slf.borrow_mut();
            if this.closed {
                return Ok(());
            }
            this.closed = true;
            let bindings = this.communicators.clone_ref(py);
            let pools = this.pools.clone_ref(py);
            let windows = this.windows.clone_ref(py);
            (bindings, pools, windows)
        };
        if aborted {
            for (_, communicator) in bindings.bind(py).iter() {
                communicator.cast::<NcclCommunicator>()?.get().abort(py)?;
            }
            std::mem::forget((bindings, pools, windows));
            return Ok(());
        }
        let communicators = bindings
            .bind(py)
            .values()
            .iter()
            .map(|value| value.cast::<NcclCommunicator>()?.get().close(py))
            .collect::<Vec<_>>();
        if let Err(error) = close_all(py, communicators) {
            // A failed deregistration cannot release the VMM mappings still
            // addressed by the communicator or captured peer operations.
            std::mem::forget((bindings, pools, windows));
            return Err(error);
        }
        let mut releases = Vec::new();
        for (_, retained) in windows.bind(py).iter() {
            for allocation in retained.get_item(1)?.try_iter()? {
                releases.push(allocation?.call_method0("close").map(drop));
            }
        }
        bindings.bind(py).clear();
        pools.bind(py).clear();
        windows.bind(py).clear();
        close_all(py, releases)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.stream)?;
        visit.call(&self.communicators)?;
        visit.call(&self.pools)?;
        visit.call(&self.windows)
    }
}

fn capturing(device: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(device.getattr("type")?.extract::<String>()? == "cuda"
        && device
            .py()
            .import("torch.cuda")?
            .call_method0("is_current_stream_capturing")?
            .is_truthy()?)
}
