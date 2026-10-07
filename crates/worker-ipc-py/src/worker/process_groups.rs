//! Process-world ownership and component communication domains.

mod components;
mod initialize;
pub(super) use components::initialize_components;

use std::collections::{BTreeSet, HashMap};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::execution::close_all;

pub(super) use initialize::initialize_process_groups;

/// Rank zero takes ownership of an optional inherited listening socket.
#[pyclass(
    frozen,
    eq,
    hash,
    get_all,
    skip_from_py_object,
    module = "uniserve_worker._uniserve_ipc"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Rendezvous {
    host: String,
    port: u16,
    listen_fd: Option<i32>,
}

#[pymethods]
impl Rendezvous {
    #[new]
    #[pyo3(signature = (host, port, listen_fd=None))]
    pub(super) fn new(host: String, port: u16, listen_fd: Option<i32>) -> Self {
        Self {
            host,
            port,
            listen_fd,
        }
    }
}

/// Own created groups, borrow an existing world and preserve logical rank order.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct ProcessGroups {
    #[pyo3(get)]
    rank: usize,
    #[pyo3(get)]
    world_size: usize,
    #[pyo3(get)]
    device: Py<PyAny>,
    #[pyo3(get)]
    backend: String,
    #[pyo3(get)]
    experts: Py<PyAny>,
    instance: Py<PyAny>,
    groups: Vec<Py<PyAny>>,
}

#[pymethods]
impl ProcessGroups {
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        slf: &Bound<'_, Self>,
        _kind: &Bound<'_, PyAny>,
        error: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let aborted = !error.is_none() && !slf.borrow().groups.is_empty();
        match Self::close(slf, aborted) {
            Err(cleanup) if !error.is_none() => {
                let _ = error.call_method1(
                    "add_note",
                    (format!("Resource cleanup also failed: {cleanup}"),),
                );
                Ok(())
            }
            result => result,
        }
    }

    #[getter]
    fn process_group(&self, py: Python<'_>) -> Py<PyAny> {
        self.instance.clone_ref(py)
    }

    /// Create fibers in the same order on every participating process. Separate
    /// bindings retain separate ordering domains even when membership matches.
    #[pyo3(signature = (mesh, *, device, axes=None))]
    fn bind(
        &mut self,
        py: Python<'_>,
        mesh: &Bound<'_, PyAny>,
        device: &Bound<'_, PyAny>,
        axes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let ranks: Vec<usize> = mesh.getattr("ranks")?.extract()?;
        if mesh.getattr("rank")?.extract::<usize>()? != self.rank
            || ranks.iter().any(|&rank| rank >= self.world_size)
        {
            return Err(PyValueError::new_err(
                "mesh ranks disagree with the process world",
            ));
        }
        let device = py.import("torch")?.call_method1("device", (device,))?;
        let axes: BTreeSet<Vec<String>> = match axes {
            Some(axes) => axes
                .try_iter()?
                .map(|axis| axis?.extract())
                .collect::<PyResult<_>>()?,
            None => mesh
                .getattr("axes")?
                .extract::<Vec<String>>()?
                .into_iter()
                .map(|axis| vec![axis])
                .collect(),
        };
        let physical: Vec<usize> = self.instance.bind(py).getattr("ranks")?.extract()?;
        let local_sync = self.experts.bind(py).getattr("size")?.extract::<usize>()? > 1;
        let dist = py.import("torch.distributed")?;
        let groups = PyDict::new(py);
        let mut handles = HashMap::<Vec<usize>, Py<PyAny>>::new();
        for selection in axes {
            let selection = PyTuple::new(py, selection)?;
            let fibers: Vec<Vec<usize>> = mesh.call_method1("members", (&selection,))?.extract()?;
            for members in fibers {
                let mut ordered = members.clone();
                ordered.sort_unstable();
                let mapped: Vec<usize> = members.iter().map(|&rank| physical[rank]).collect();
                if members.len() > 1 && !handles.contains_key(&ordered) {
                    if !dist.call_method0("is_initialized")?.is_truthy()? {
                        return Err(PyRuntimeError::new_err(
                            "multi-rank mesh binding requires an initialized process world",
                        ));
                    }
                    let handle =
                        initialize::subgroup(py, &self.backend, &device, &mapped, local_sync)?;
                    if members.contains(&self.rank) {
                        self.groups.push(handle.clone_ref(py));
                    }
                    handles.insert(ordered.clone(), handle);
                }
                if let Some(rank) = members.iter().position(|&rank| rank == self.rank) {
                    let axes: Vec<String> = selection.extract()?;
                    let name = if axes.is_empty() {
                        "local".to_owned()
                    } else {
                        axes.join(".")
                    };
                    let handle = handles.get(&ordered).map(|handle| handle.bind(py));
                    groups.set_item(
                        &selection,
                        communicator(py, &mapped, rank, &name, &device, handle)?,
                    )?;
                }
            }
        }

        // Mesh topology remains a numerical value. Bind its borrowed handles
        // only after all fibers have been created in collective order.
        let options = PyDict::new(py);
        for name in ["ranks", "shape", "axes", "rank"] {
            options.set_item(name, mesh.getattr(name)?)?;
        }
        let result = py
            .import("uniserve.distributed.mesh")?
            .getattr("DeviceMesh")?
            .call((), Some(&options))?;
        let set = py.get_type::<pyo3::types::PyAny>().getattr("__setattr__")?;
        set.call1((&result, "_device", device))?;
        set.call1((
            &result,
            "_groups",
            py.import("types")?
                .call_method1("MappingProxyType", (groups,))?,
        ))?;
        Ok(result.unbind())
    }

    /// Retire subgroups before their parent world, after numerical users stop.
    /// An aborted process retains groups so teardown cannot wait on live peers.
    #[pyo3(signature = (*, aborted=false))]
    pub(super) fn close(slf: &Bound<'_, Self>, aborted: bool) -> PyResult<()> {
        let py = slf.py();
        if aborted {
            py.import("uniserve.runtime.resources")?
                .call_method1("retain_until_exit", (slf,))?;
            return Ok(());
        }
        let (groups, device) = {
            let mut owner = slf.borrow_mut();
            (
                std::mem::take(&mut owner.groups),
                owner.device.clone_ref(py),
            )
        };
        if groups.is_empty() {
            return Ok(());
        }
        let synchronized = if device.bind(py).getattr("type")?.extract::<&str>()? == "cuda" {
            py.import("torch.cuda")?
                .call_method1("synchronize", (&device,))
                .map(drop)
        } else {
            Ok(())
        };
        let dist = py.import("torch.distributed")?;
        close_all(
            py,
            std::iter::once(synchronized).chain(groups.into_iter().rev().map(|group| {
                dist.call_method1("destroy_process_group", (group,))
                    .map(drop)
            })),
        )
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.device)?;
        visit.call(&self.experts)?;
        visit.call(&self.instance)?;
        for group in &self.groups {
            visit.call(group)?;
        }
        Ok(())
    }
}

fn communicator(
    py: Python<'_>,
    ranks: &[usize],
    rank: usize,
    name: &str,
    device: &Bound<'_, PyAny>,
    group: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("uniserve.distributed.mesh")?
        .call_method1(
            "Communicator",
            (PyTuple::new(py, ranks)?, rank, name, device, group),
        )?
        .unbind())
}
