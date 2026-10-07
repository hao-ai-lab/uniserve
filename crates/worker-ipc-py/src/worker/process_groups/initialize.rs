//! Join physical worlds and configure the PyTorch communication backend.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::{ProcessGroups, Rendezvous, communicator};

/// Initialize a worker world or one replica within the shared expert union.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (*, rank, local_rank, world_size, device, backend=None, init_method=None, rendezvous=None, experts=None))]
pub(in crate::worker) fn initialize_process_groups(
    py: Python<'_>,
    rank: i64,
    local_rank: i64,
    world_size: i64,
    device: &Bound<'_, PyAny>,
    backend: Option<String>,
    init_method: Option<String>,
    rendezvous: Option<PyRef<'_, Rendezvous>>,
    experts: Option<(i64, i64, Py<Rendezvous>)>,
) -> PyResult<Py<ProcessGroups>> {
    if world_size < 1 || rank < 0 || rank >= world_size || local_rank < 0 {
        return Err(PyValueError::new_err(
            "launch rank must satisfy 0 <= rank < positive world_size",
        ));
    }
    let dist = py.import("torch.distributed")?;
    let initialized = dist.call_method0("is_initialized")?.is_truthy()?;
    if let Some((expert_rank, expert_size, _)) = &experts {
        if initialized {
            return Err(PyValueError::new_err(
                "joining an expert union requires an uninitialized process world",
            ));
        }
        if rendezvous.is_some() || init_method.is_some() {
            return Err(PyValueError::new_err(
                "the expert union owns the process-world rendezvous",
            ));
        }
        let start = expert_rank - rank;
        if start < 0 || start + world_size > *expert_size {
            return Err(PyValueError::new_err(
                "worker ranks must fit inside the expert union",
            ));
        }
    }
    if init_method.is_some() && rendezvous.is_some() {
        return Err(PyValueError::new_err(
            "init_method and rendezvous are exclusive",
        ));
    }
    if rank != 0
        && rendezvous
            .as_ref()
            .is_some_and(|store| store.listen_fd.is_some())
    {
        return Err(PyValueError::new_err(
            "only rank 0 serves the rendezvous store",
        ));
    }

    let device = local_device(py, device, local_rank)?;
    let cuda = device.getattr("type")?.extract::<&str>()? == "cuda";
    let backend = backend
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| if cuda { "nccl" } else { "gloo" }.to_owned());
    let rank = rank as usize;
    let world_size = world_size as usize;
    let mut groups = Vec::new();
    let mut expert_group = py
        .import("uniserve.distributed.mesh")?
        .call_method0("Communicator")?
        .unbind();
    let instance;

    if let Some((expert_rank, expert_size, rendezvous)) = experts {
        connect_at_startup(py)?;
        let options = PyDict::new(py);
        options.set_item("backend", "cpu:gloo,cuda:nccl")?;
        options.set_item(
            "store",
            store(
                py,
                &rendezvous.borrow(py),
                expert_rank as usize,
                expert_size as usize,
                "nccl",
            )?,
        )?;
        options.set_item("rank", expert_rank)?;
        options.set_item("world_size", expert_size)?;
        dist.call_method("init_process_group", (), Some(&options))?;
        let world = dist.getattr("group")?.getattr("WORLD")?;
        groups.push(world.clone().unbind());
        if cuda {
            // Leave WORLD's device_id unbound: ncclCommSplit would force every
            // union rank to participate in each independent replica's binding.
            let options = PyDict::new(py);
            options.set_item("device", &device)?;
            let zero = py
                .import("torch")?
                .call_method("zeros", ((),), Some(&options))?;
            dist.call_method1("all_reduce", (zero,))?;
        }
        expert_group = communicator(
            py,
            &(0..expert_size as usize).collect::<Vec<_>>(),
            expert_rank as usize,
            "experts",
            &device,
            Some(&world),
        )?;
        let start = expert_rank as usize - rank;
        let members: Vec<usize> = (start..start + world_size).collect();
        let group = subgroup(py, &backend, &device, &members, true)?;
        instance = communicator(
            py,
            &members,
            rank,
            "instance",
            &device,
            Some(group.bind(py)),
        )?;
        groups.push(group);
    } else {
        if initialized {
            if dist.call_method0("get_rank")?.extract::<usize>()? != rank
                || dist.call_method0("get_world_size")?.extract::<usize>()? != world_size
            {
                return Err(PyValueError::new_err(
                    "existing process world disagrees with supplied rank/world_size",
                ));
            }
            if dist.call_method0("get_backend")?.extract::<String>()? != backend {
                return Err(PyValueError::new_err(
                    "existing process world disagrees with supplied backend",
                ));
            }
        } else if world_size > 1 {
            let options = options(py, &backend, &device)?;
            options.set_item("rank", rank)?;
            options.set_item("world_size", world_size)?;
            if let Some(rendezvous) = rendezvous {
                options.set_item("store", store(py, &rendezvous, rank, world_size, &backend)?)?;
            } else {
                let method = match init_method.filter(|method| !method.is_empty()) {
                    Some(method) => method,
                    None => {
                        let port = std::env::var("MASTER_PORT")
                            .ok()
                            .filter(|port| !port.is_empty())
                            .ok_or_else(|| {
                                PyValueError::new_err(
                                    "MASTER_PORT or an explicit init_method is required",
                                )
                            })?;
                        let host =
                            std::env::var("MASTER_ADDR").unwrap_or_else(|_| "127.0.0.1".to_owned());
                        format!("tcp://{host}:{port}")
                    }
                };
                options.set_item("init_method", method)?;
            }
            dist.call_method("init_process_group", (), Some(&options))?;
            groups.push(dist.getattr("group")?.getattr("WORLD")?.unbind());
        }
        let world = if initialized || world_size > 1 {
            Some(dist.getattr("group")?.getattr("WORLD")?)
        } else {
            None
        };
        instance = communicator(
            py,
            &(0..world_size).collect::<Vec<_>>(),
            rank,
            "instance",
            &device,
            world.as_ref(),
        )?;
    }

    Py::new(
        py,
        ProcessGroups {
            rank,
            world_size,
            device: device.unbind(),
            backend,
            experts: expert_group,
            instance,
            groups,
        },
    )
}

pub(super) fn subgroup(
    py: Python<'_>,
    backend: &str,
    device: &Bound<'_, PyAny>,
    members: &[usize],
    local_sync: bool,
) -> PyResult<Py<PyAny>> {
    let mut ordered = members.to_vec();
    ordered.sort_unstable();
    let options = options(py, backend, device)?;
    options.set_item("ranks", PyList::new(py, ordered)?)?;
    options.set_item("use_local_synchronization", local_sync)?;
    Ok(py
        .import("torch.distributed")?
        .call_method("new_group", (), Some(&options))?
        .unbind())
}

fn local_device<'py>(
    py: Python<'py>,
    device: &Bound<'py, PyAny>,
    local_rank: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let torch = py.import("torch")?;
    let device = torch.call_method1("device", (device,))?;
    if device.getattr("type")?.extract::<&str>()? != "cuda" {
        return Ok(device);
    }
    let index = device
        .getattr("index")?
        .extract::<Option<i64>>()?
        .unwrap_or(local_rank);
    let device = torch.call_method1("device", ("cuda", index))?;
    let cuda = py.import("torch.cuda")?;
    let count: i64 = cuda.call_method0("device_count")?.extract()?;
    if !cuda.call_method0("is_available")?.is_truthy()? || index >= count {
        return Err(PyValueError::new_err(format!(
            "cuda device {device} is outside the {count} visible CUDA device(s)"
        )));
    }
    cuda.call_method1("set_device", (&device,))?;
    Ok(device)
}

fn options<'py>(
    py: Python<'py>,
    backend: &str,
    device: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let options = PyDict::new(py);
    options.set_item("backend", backend)?;
    if backend == "nccl" {
        connect_at_startup(py)?;
        let nccl = py
            .import("torch.distributed")?
            .getattr("ProcessGroupNCCL")?;
        let pg = nccl.call_method0("Options")?;
        pg.setattr("use_pg_for_symm_mem_rendezvous", true)?;
        // NCCL chooses an eligible algorithm for each operation and topology;
        // zero-CTA is a preference, not a claim about a logical mesh axis.
        pg.getattr("config")?
            .setattr("cta_policy", nccl.getattr("NCCL_CTA_POLICY_ZERO")?)?;
        options.set_item("pg_options", pg)?;
    }
    if device.getattr("type")?.extract::<&str>()? == "cuda"
        && (backend == "nccl" || backend.split(',').any(|value| value == "cuda:nccl"))
    {
        options.set_item("device_id", device)?;
    }
    Ok(options)
}

fn connect_at_startup(py: Python<'_>) -> PyResult<()> {
    // Deferred peer handshakes would allocate buffers after storage is sealed.
    // Preserve an explicit process setting, and initialize before the first NCCL group.
    py.import("os")?
        .getattr("environ")?
        .call_method1("setdefault", ("NCCL_RUNTIME_CONNECT", "0"))?;
    Ok(())
}

fn store(
    py: Python<'_>,
    rendezvous: &Rendezvous,
    rank: usize,
    size: usize,
    backend: &str,
) -> PyResult<Py<PyAny>> {
    let constants = py.import("torch.distributed.constants")?;
    let timeout = if backend == "nccl" {
        constants.getattr("default_pg_nccl_timeout")?
    } else {
        py.None().into_bound(py)
    };
    let timeout = if timeout.is_none() {
        constants.getattr("default_pg_timeout")?
    } else {
        timeout
    };
    if let Some(fd) = rendezvous.listen_fd {
        py.import("os")?
            .call_method1("set_inheritable", (fd, false))?;
    }
    let options = PyDict::new(py);
    options.set_item("is_master", rank == 0)?;
    options.set_item("timeout", timeout)?;
    options.set_item("master_listen_fd", rendezvous.listen_fd)?;
    Ok(py
        .import("torch.distributed")?
        .call_method(
            "TCPStore",
            (&rendezvous.host, rendezvous.port, size),
            Some(&options),
        )?
        .unbind())
}
