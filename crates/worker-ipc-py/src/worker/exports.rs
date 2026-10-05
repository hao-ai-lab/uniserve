//! Transport registrations retained by tensor, KV and latent storage.

use std::collections::HashSet;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker_ipc::{BufferId, RequestKey};

use super::error::{resource, unsupported};
use super::protocol::buffer_id;
use super::vmm_pool::PoolExhaustedError;
use crate::convert;

/// Export a tensor or ordered first-axis spans through the configured
/// mechanisms that reach its consumers. Storage owners retain each backend's
/// retirement signal before the next export starts. A failure revokes all
/// accepted locations; their physical accesses still retire asynchronously.
#[pyfunction]
#[pyo3(signature = (transports, source, *, retain, offset=None, consumers=Vec::new(), host=false))]
pub(super) fn export_tensor<'py>(
    transports: &Bound<'py, PyAny>,
    source: &Bound<'py, PyAny>,
    retain: &Bound<'py, PyAny>,
    offset: Option<&Bound<'py, PyAny>>,
    consumers: Vec<u32>,
    host: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let py = transports.py();
    if !transports.is_truthy()? {
        return Err(unsupported(
            py,
            "tensor export requires a configured transport",
        ));
    }

    let first = if source.is_instance_of::<PyTuple>() {
        source.get_item(0)?
    } else {
        source.clone()
    };
    let device = first.getattr("is_cuda")?.extract::<bool>()? && !host;
    let names: &[&str] = if device && transports.contains("cuda_vmm")? {
        &["local", "cuda_vmm"]
    } else {
        &["local", "shm", "channel"]
    };
    let consumers = PyTuple::new(py, consumers)?;
    let mut selected = Vec::new();
    for &name in names {
        if transports.contains(name)? {
            let transport = transports.get_item(name)?;
            if transport
                .call_method1("serves", (&consumers,))?
                .is_truthy()?
            {
                selected.push(transport);
            }
        }
    }

    let kwargs = PyDict::new(py);
    kwargs.set_item("offset", offset)?;
    kwargs.set_item("consumers", &consumers)?;
    let mut locations = Vec::new();
    let result = (|| -> PyResult<()> {
        for transport in selected {
            match transport.call_method("export", (source,), Some(&kwargs)) {
                Ok(location) => retain_location(&transport, location, retain, &mut locations)?,
                Err(error) if error.is_instance_of::<PoolExhaustedError>(py) => {
                    // A device payload that cannot fit its VMM allocation
                    // travels over the host mechanisms reaching its readers.
                    // Local was already selected before VMM, so never repeat it.
                    for name in ["shm", "channel"] {
                        if !transports.contains(name)? {
                            continue;
                        }
                        let fallback = transports.get_item(name)?;
                        if !fallback
                            .call_method1("serves", (&consumers,))?
                            .is_truthy()?
                        {
                            continue;
                        }
                        let location = fallback.call_method("export", (source,), Some(&kwargs))?;
                        retain_location(&fallback, location, retain, &mut locations)?;
                    }
                    if locations.is_empty() {
                        let failure = resource(
                            py,
                            "device product does not fit its VMM pool and the rank binds no other mechanism that exports it",
                        );
                        failure.set_cause(py, Some(error));
                        return Err(failure);
                    }
                }
                Err(error) => return Err(error),
            }
        }
        Ok(())
    })();

    if let Err(error) = result {
        for (transport, location) in &locations {
            transport.call_method1("release", (location,))?;
        }
        return Err(error);
    }
    PyTuple::new(py, locations.into_iter().map(|(_, location)| location))
}

fn retain_location<'py>(
    transport: &Bound<'py, PyAny>,
    location: Bound<'py, PyAny>,
    retain: &Bound<'py, PyAny>,
    locations: &mut Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
) -> PyResult<()> {
    // Include this location in failure cleanup even when its retirement
    // lookup or the storage owner's retention callback raises an error.
    locations.push((transport.clone(), location.clone()));
    let retirement = transport.call_method1("retirement", (location,))?;
    retain.call1((retirement,))?;
    Ok(())
}

pub(super) fn validate_exports(
    resident: &Bound<'_, PyDict>,
    candidates: &Bound<'_, PyDict>,
) -> PyResult<()> {
    for (buffer, locations) in candidates {
        if let Some(existing) = resident.get_item(buffer)?
            && !existing.eq(locations)?
        {
            return Err(PyRuntimeError::new_err(
                "buffer already has different export locations",
            ));
        }
    }
    Ok(())
}

/// Select local registrations; imported remote locations never enter a
/// storage owner's export map. Explicit Free takes precedence over retention.
pub(super) fn select(
    exports: &Bound<'_, PyDict>,
    buffers: &HashSet<BufferId>,
    requests: &HashSet<RequestKey>,
    retained: &HashSet<BufferId>,
) -> PyResult<Vec<Py<PyAny>>> {
    let mut selected = Vec::new();
    for (key, _) in exports {
        let buffer = buffer_id(&key)?;
        if buffers.contains(&buffer)
            || (requests.contains(&buffer.owner) && !retained.contains(&buffer))
        {
            selected.push(key.unbind());
        }
    }
    Ok(selected)
}

/// Revocation rejects new readers. The storage owner still retains its
/// backing until the transport reports physical retirement.
pub(super) fn release(exports: &Bound<'_, PyDict>, buffers: &[Py<PyAny>]) -> PyResult<()> {
    for buffer in buffers {
        if let Some(locations) = exports.get_item(buffer)? {
            for location in locations.try_iter()? {
                let (transport, locator): (Bound<'_, PyAny>, Bound<'_, PyAny>) =
                    location?.extract()?;
                transport.call_method1("release", (locator,))?;
            }
        }
    }
    Ok(())
}

pub(super) fn release_buffers(
    exports: &Bound<'_, PyDict>,
    buffers: &HashSet<BufferId>,
) -> PyResult<()> {
    let keys = buffers
        .iter()
        .map(|buffer| convert::buffer_id_to_py(exports.py(), buffer).map(Bound::unbind))
        .collect::<PyResult<Vec<_>>>()?;
    release(exports, &keys)
}

#[pyfunction]
pub(super) fn release_exports(
    exports: &Bound<'_, PyDict>,
    buffers: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let buffers = buffers
        .try_iter()?
        .map(|value| value.map(Bound::unbind))
        .collect::<PyResult<Vec<_>>>()?;
    release(exports, &buffers)
}

pub(super) fn forget(exports: &Bound<'_, PyDict>, buffers: &[Py<PyAny>]) -> PyResult<()> {
    for buffer in buffers {
        if exports.contains(buffer)? {
            exports.del_item(buffer)?;
        }
    }
    Ok(())
}
