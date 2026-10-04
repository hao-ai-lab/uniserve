//! Transport registrations retained by tensor, KV and latent storage.

use std::collections::HashSet;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use uniserve_worker_ipc::{BufferId, RequestKey};

use super::protocol::buffer_id;
use crate::convert;

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
