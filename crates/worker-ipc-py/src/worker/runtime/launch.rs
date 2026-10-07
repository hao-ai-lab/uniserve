//! Bind and register one rank before loading weights; retire it collectively.

use std::io::{self, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

use pyo3::exceptions::{PyImportError, PyValueError};
use pyo3::prelude::*;
use uniserve_worker_ipc::{RankReport, SOCKET_CHANNEL};

use super::Worker;
use crate::PyServer;
use crate::worker::config;
use crate::worker::host::with_context;

// Registration precedes weight loading. This bounds connection establishment
// and the single report write, not model startup.
const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(60);

#[pyfunction]
pub(in crate::worker) fn endpoint_name(config: &Bound<'_, PyAny>) -> PyResult<String> {
    let transport: String = config
        .getattr("ipc")?
        .getattr("channel_transport")?
        .extract()?;
    if transport == SOCKET_CHANNEL {
        return Ok("0.0.0.0".into());
    }
    let execution = config::native(&config.getattr("execution")?)?;
    Ok(uniserve_worker_ipc::service_name(&format!(
        "{}_{}_{}",
        std::process::id(),
        execution.rank,
        uuid::Uuid::new_v4().simple()
    )))
}

/// Report the bound channel once. TCP reports the interface used to reach the
/// head, since a wildcard bind address cannot be dialed by another host.
#[pyfunction]
pub(in crate::worker) fn register_endpoint(
    config: &Bound<'_, PyAny>,
    endpoint: String,
) -> PyResult<()> {
    let ipc = config.getattr("ipc")?;
    let address: String = ipc.getattr("registration_address")?.extract()?;
    let mut report = RankReport {
        worker_id: config.getattr("worker_id")?.extract()?,
        rank: config::native(&config.getattr("execution")?)?.rank as u32,
        transport: ipc.getattr("channel_transport")?.extract()?,
        endpoint,
    };
    config.py().detach(move || -> PyResult<()> {
        let mut failure = io::Error::new(
            io::ErrorKind::AddrNotAvailable,
            "registration address resolves to no endpoints",
        );
        let mut connection = None;
        for address in address.to_socket_addrs()? {
            match TcpStream::connect_timeout(&address, REGISTRATION_TIMEOUT) {
                Ok(stream) => {
                    connection = Some(stream);
                    break;
                }
                Err(error) => failure = error,
            }
        }
        let mut connection = connection.ok_or(failure)?;
        connection.set_write_timeout(Some(REGISTRATION_TIMEOUT))?;
        if report.transport == SOCKET_CHANNEL {
            let (_, port) = report
                .endpoint
                .rsplit_once(':')
                .ok_or_else(|| PyValueError::new_err("TCP endpoint must name a port"))?;
            let port = port.parse().map_err(|error| {
                PyValueError::new_err(format!("invalid endpoint port: {error}"))
            })?;
            report.endpoint = SocketAddr::new(connection.local_addr()?.ip(), port).to_string();
        }
        let mut line = serde_json::to_vec(&report)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        line.push(b'\n');
        connection.write_all(&line)?;
        Ok(())
    })
}

/// Own endpoint and worker scopes through startup, serving and failure. A
/// constructed worker retires its readers before the endpoint is released.
#[pyfunction]
pub(in crate::worker) fn run_worker(config: &Bound<'_, PyAny>) -> PyResult<()> {
    if cfg!(debug_assertions) {
        return Err(PyImportError::new_err(
            "_uniserve_ipc is a debug build; install a release build (pip install -e .)",
        ));
    }
    let py = config.py();
    let ipc = config.getattr("ipc")?;
    let service = endpoint_name(config)?;
    let payload: usize = ipc.getattr("max_payload_bytes")?.extract()?;
    let depth: usize = ipc.getattr("queue_depth")?.extract()?;
    let transport: String = ipc.getattr("channel_transport")?.extract()?;
    let endpoint = Bound::new(py, PyServer::new(&service, payload, depth, &transport)?)?;
    with_context(endpoint.as_any(), || {
        register_endpoint(config, endpoint.borrow().endpoint(&service)?)?;
        let worker =
            Worker::from_config(&py.get_type::<Worker>(), config)?.cast_into::<Worker>()?;
        with_context(worker.as_any(), || {
            Worker::bind(worker.clone(), Some(endpoint.clone()))?;
            py.import("logging")?
                .call_method1("getLogger", ("uniserve_worker.bootstrap.launch",))?
                .call_method1("info", ("worker IPC endpoint bound",))?;
            Worker::warmup(&worker)?;
            leave_shutdown_to_head(py)?;
            Worker::run(&worker)
        })
    })
}

fn leave_shutdown_to_head(py: Python<'_>) -> PyResult<()> {
    // Only a serving rank can receive collective close from the head. Before
    // warmup, normal signal handling still terminates a failed startup. Python
    // installs a handled signal so exec children regain the default disposition.
    let signal = py.import("signal")?;
    let handler = wrap_pyfunction!(shutdown_signal, py)?;
    for name in ["SIGINT", "SIGTERM"] {
        signal.call_method1("signal", (signal.getattr(name)?, &handler))?;
    }
    Ok(())
}

#[pyfunction]
fn shutdown_signal(py: Python<'_>, signum: i32, _frame: &Bound<'_, PyAny>) -> PyResult<()> {
    let name = py
        .import("signal")?
        .call_method1("Signals", (signum,))?
        .getattr("name")?;
    py.import("logging")?
        .call_method1("getLogger", ("uniserve_worker.bootstrap.launch",))?
        .call_method1(
            "info",
            (
                "ignoring %s; the head shuts this rank down with its worker group",
                name,
            ),
        )?;
    Ok(())
}

pub(in crate::worker) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(endpoint_name, module)?)?;
    module.add_function(wrap_pyfunction!(register_endpoint, module)?)?;
    module.add_function(wrap_pyfunction!(run_worker, module)?)?;
    Ok(())
}
